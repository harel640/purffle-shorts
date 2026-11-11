"""Pluggable LLMs: any OpenAI-compatible endpoint plus native Anthropic Claude, with a fallback chain.

Pick one with LLM_PROVIDER (or --provider) and LLM_MODEL (or --model). With LLM_PROVIDER=auto the
first provider whose API key is present is used. LLM_FALLBACKS=anthropic,gemini adds backups that
are tried in order when the primary fails.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .utils import PermanentError, check_cancel, http, raise_for_status, redact, with_retries

log = logging.getLogger("purffle")

# A per-minute rate limit (Groq TPM/RPM) clears within a minute, so wait and retry rather than give up.
# A daily quota will not clear in a minute, so it is excluded and falls through to the next provider.
GROQ_RATE_LIMIT_WAIT = 60.0
GROQ_RATE_LIMIT_RETRIES = 3


# Groq's free tier allows 8,000 tokens a minute for this model, and refuses outright (HTTP 413 "Request too
# large") a request it sizes above that. Its open-weight reasoning models (openai/gpt-oss-*) spend thousands of
# hidden reasoning tokens per reply - 2,926 for one script - which count against the same limit and against
# max_tokens ("max completion tokens reached before generating a valid document"). ``effort="low"`` asks for
# short reasoning; it is used for checking tasks, and for the one retry after a 413 when little room is left.
GROQ_TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "8000"))


def _fitting_reply_tokens(system: str, user: str, limit: int = 0) -> int:
    """The largest max_tokens with which a Groq request is sure to be accepted: Groq sizes a request whose
    max_tokens fits as prompt + max_tokens. The prompt is estimated generously (3.4 characters a token; the
    story writer's 11,226 characters were 3,085 tokens)."""
    prompt = int((len(system) + len(user)) / 3.4) + 150
    return (limit or GROQ_TPM_LIMIT) - prompt - 200


def _is_too_large(error: Exception) -> bool:
    t = str(error).lower()
    return "413" in t or "request too large" in t


def _is_per_minute_rate_limit(error: Exception) -> bool:
    """True for a per-minute token/request rate limit (worth a 60s wait), not a daily/quota cap."""
    t = str(error).lower()
    if "429" not in t and "rate limit" not in t and "too many requests" not in t:
        return False
    if any(k in t for k in ("per day", "per-day", " rpd", " tpd", "daily", "exceeded your current quota",
                            "quota exceeded", "insufficient_quota", "billing")):
        return False
    return any(k in t for k in ("per minute", "per-minute", "tpm", "rpm", "tokens per minute",
                                "requests per minute", "rate limit reached"))


class LLMError(RuntimeError):
    pass


# Circuit breaker: Gemini's free tier answers 503/429 (or hangs) for hours at a time, and every call used to
# spend its timeout and retries on it before falling back to Groq - the 19:00 runs spent 20+ minutes in the
# script stages and hit the 25-minute watchdog. After CIRCUIT_AFTER consecutive transient failures a provider
# is skipped for the rest of the run (still tried as a last resort if every other provider fails).
CIRCUIT_PROVIDERS = {"gemini"}
CIRCUIT_AFTER = 2
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "30"))     # per call, seconds (was 180)
_TRANSIENT_RE = re.compile(r"(?i)(http (429|5\d\d)|timed out|timeout|connection (aborted|reset|error)|"
                           r"max retries|connectionerror|temporarily|overloaded|unavailable|high demand)")


def is_transient_failure(error: BaseException) -> bool:
    """A 503/429/timeout-style failure (the provider is busy or down), not a bad request or a bad reply."""
    return bool(_TRANSIENT_RE.search(f"{type(error).__name__} {error}"))


class ImageUnusable(LLMError):
    """The image itself can't be sent to a vision model (undecodable, or the backend says its format is
    unsupported) - a problem with THIS image, not an outage: the caller skips it and tries another."""


VISION_MAX_SIDE = 1024
# A backend's HTTP 400 that is about the picture (not the key, quota or model).
_IMAGE_ERROR_RE = re.compile(r"(?i)(unsupported[^.]{0,30}(image|format)|could not decode|corrupt|"
                             r"invalid image|not a valid image|image format|cannot identify)")


def vision_jpeg(image_path) -> bytes:
    """``image_path`` as a plain baseline JPEG (RGB, long side <= VISION_MAX_SIDE) for a vision model.
    Commons photos can be MPO (multi-picture) or have odd colour modes, which vision backends reject
    ("Unsupported input image format 'MPO'"); this also shrinks a 3 MB photo to ~100 KB so the call is fast.
    Raises ImageUnusable when the file can't be decoded."""
    import io

    try:
        from PIL import Image, ImageOps
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            im.draft("RGB", (VISION_MAX_SIDE * 2, VISION_MAX_SIDE * 2))
            im = im.convert("RGB")
            im.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:  # noqa: BLE001 - any decode failure means this image can't be used
        raise ImageUnusable(f"cannot decode {image_path}: {e}") from e


class LLMConfigError(LLMError):
    """Missing key, unknown provider, etc. — stop instead of retrying every video."""


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    base_url: str
    key_envs: tuple[str, ...]
    default_model: str
    kind: str = "openai"          # openai (chat/completions compatible) | anthropic
    json_mode: bool = True        # supports response_format={"type":"json_object"}
    local: bool = False
    schema_mode: bool = False     # supports strict response_format={"type":"json_schema"} (exact keys and types)


PRESETS: dict[str, Preset] = {p.name: p for p in [
    Preset("openai", "OpenAI (GPT)", "https://api.openai.com/v1", ("OPENAI_API_KEY",), "gpt-4o-mini",
           schema_mode=True),
    Preset("anthropic", "Anthropic (Claude)", "", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
           "claude-opus-5-5", kind="anthropic"),
    Preset("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
           ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "gemini-2.5-flash", schema_mode=True),
    Preset("groq", "Groq (Llama, fast)", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",),
           "llama-3.3-70b-versatile"),
    Preset("openrouter", "OpenRouter (hundreds of models)", "https://openrouter.ai/api/v1",
           ("OPENROUTER_API_KEY",), "openai/gpt-4o-mini", schema_mode=True),
    Preset("deepseek", "DeepSeek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "deepseek-chat"),
    Preset("mistral", "Mistral", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",), "mistral-small-latest"),
    Preset("together", "Together AI", "https://api.together.xyz/v1", ("TOGETHER_API_KEY",),
           "meta-llama/Llama-3.3-70B-Instruct-Turbo", json_mode=False),
    Preset("xai", "xAI (Grok)", "https://api.x.ai/v1", ("XAI_API_KEY",), "grok-3-mini"),
    Preset("ollama", "Ollama (local, free)", "http://localhost:11434/v1", (), "llama3.1", local=True,
           schema_mode=True),
    Preset("lmstudio", "LM Studio (local, free)", "http://localhost:1234/v1", (), "local-model",
           json_mode=False, local=True, schema_mode=True),
    Preset("custom", "Any OpenAI-compatible URL (LLM_BASE_URL)", "", ("LLM_API_KEY",), ""),
]}

AUTO_ORDER = ["openai", "anthropic", "gemini", "groq", "openrouter", "deepseek", "mistral", "together", "xai"]

# Claude models whose requests get server-side refusal fallbacks ("default" routing) by default.
_CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# Claude Opus 5.5 thinks at "medium" effort unless asked; scripts and reviews are worth "high".
_CLAUDE_DEFAULT_EFFORT = {"claude-opus-5-5": "high"}


def _key_for(preset: Preset, settings: Settings) -> str:
    if preset.name == "custom":
        return settings.llm_api_key
    for env in preset.key_envs:
        if os.getenv(env):
            return os.environ[env]
    return settings.llm_api_key if preset.name == settings.llm_provider else ""


def _ollama_up(base_url: str) -> bool:
    try:
        root = base_url.rsplit("/v1", 1)[0]
        return http().get(root + "/api/tags", timeout=1.5).ok
    except Exception:
        return False


def ollama_url() -> str:
    host = os.getenv("OLLAMA_HOST", "").rstrip("/") or PRESETS["ollama"].base_url
    if not host.startswith("http"):
        host = "http://" + host
    return host if host.endswith("/v1") else host + "/v1"


def ollama_models(base_url: str) -> list[str] | None:
    """Models pulled into Ollama, or None when Ollama is not reachable."""
    try:
        r = http().get(base_url.rsplit("/v1", 1)[0] + "/api/tags", timeout=1.5)
        return [m.get("name", "") for m in r.json().get("models", [])] if r.ok else None
    except Exception:
        return None


def ollama_has(installed: list[str], model: str) -> bool:
    return model in installed or (":" not in model and f"{model}:latest" in installed)


def available_providers(settings: Settings) -> list[str]:
    """Providers that have credentials configured (in auto order)."""
    return [n for n in AUTO_ORDER if _key_for(PRESETS[n], settings)]


def resolve_provider_name(settings: Settings) -> str:
    name = settings.llm_provider or "auto"
    if name != "auto":
        if name not in PRESETS:
            raise LLMConfigError(f"Unknown LLM_PROVIDER '{name}'. Choose one of: {', '.join(PRESETS)}")
        return name
    if settings.llm_base_url:
        return "custom"
    found = available_providers(settings)
    if found:
        return found[0]
    if _ollama_up(ollama_url()):
        return "ollama"
    raise LLMConfigError(
        "No LLM is configured. Set one API key in .env (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
        "GROQ_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY, ...), run Ollama locally, "
        "or try the no-key demo:  python -m purffle_shorts demo"
    )


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply that may include code fences or chatter."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.S).strip()
    try:
        val = json.loads(t)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        chunk = t[start:end + 1]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            # Trailing commas are the most common model mistake.
            return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
    raise LLMError(f"Model did not return JSON: {text[:200]!r}")


class Provider:
    name: str = ""
    model: str = ""

    def complete(self, system: str, user: str, *, schema: dict | None = None,
                 temperature: float = 0.9, max_tokens: int = 4000) -> str:
        raise NotImplementedError

    def takes_effort(self) -> bool:
        """True if complete() accepts ``effort`` (a reasoning-effort hint: low | medium | high)."""
        return False

    def __str__(self) -> str:
        return f"{self.name}:{self.model}"


class OpenAICompatible(Provider):
    def __init__(self, preset: Preset, model: str, api_key: str, base_url: str):
        self.name, self.preset, self.model = preset.name, preset, model
        self.api_key, self.base_url = api_key, base_url.rstrip("/")
        self._json_mode = preset.json_mode
        self._schema_mode = preset.schema_mode and os.getenv("LLM_JSON_SCHEMA", "true").lower() not in ("0", "false")

    def _is_reasoning_model(self) -> bool:
        return bool(re.match(r"^(o\d|gpt-5)", self.model.split("/")[-1]))

    def takes_effort(self) -> bool:
        return self.name == "groq" and "gpt-oss" in self.model

    def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000, effort=None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if effort and self.takes_effort():
            payload["reasoning_effort"] = effort
        if self._is_reasoning_model():
            payload["max_completion_tokens"] = max(max_tokens, 8000)  # room for hidden reasoning tokens
        else:
            payload["temperature"] = temperature
            payload["max_tokens"] = max_tokens
        if schema and self._schema_mode:
            # Strict structured output: the reply is guaranteed to have exactly the keys and types asked for.
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        elif schema is not None and self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/Chamanrajragu/purffle-shorts"
            headers["X-Title"] = "PurffleShorts"
        try:
            # A local model on a laptop CPU can take minutes for one script; cloud APIs answer far sooner.
            timeout = 600 if self.preset.local else (GEMINI_TIMEOUT if self.name == "gemini" else 180)
            r = http().post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=timeout)
            raise_for_status(r, f"{self}")
        except PermanentError as e:
            fmt = payload.get("response_format", {}).get("type")
            text = str(e).lower()
            named = any(k in text for k in ("response_format", "json_schema", "schema", "structured"))
            # Strict schemas are newer: a plain "bad request" from an endpoint in that mode is worth one retry
            # without it (an auth error would fail the same way and is reported as such).
            generic = fmt == "json_schema" and ("http 400" in text or "http 422" in text)
            if fmt and (named or generic):
                # The endpoint (or this model) can't do it: step down to JSON mode, then to prompt + parser.
                if fmt == "json_schema":
                    self._schema_mode = False
                    log.info("%s: no strict JSON schema support; using JSON mode", self)
                else:
                    self._json_mode = False
                return self.complete(system, user, schema=schema, temperature=temperature, max_tokens=max_tokens,
                                     effort=effort)
            raise
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self}: unexpected response {str(data)[:200]}") from e
        if not content:
            raise LLMError(f"{self}: empty reply (finish_reason={data['choices'][0].get('finish_reason')})")
        return content


class AnthropicProvider(Provider):
    """Native Claude via the official Anthropic SDK (structured JSON output + refusal fallbacks)."""

    def __init__(self, model: str, api_key: str, effort: str = ""):
        try:
            import anthropic
        except ImportError as e:
            raise LLMConfigError("The Anthropic provider needs the SDK:  pip install anthropic") from e
        self.name, self.model, self.effort = "anthropic", model, effort
        self._anthropic = anthropic
        # With no explicit key the SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, system, user, *, schema=None, temperature=0.9, max_tokens=4000) -> str:
        a = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,  # thinking is on by default on current Claude models; leave room
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict[str, Any] = {}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        effort = self.effort or _CLAUDE_DEFAULT_EFFORT.get(self.model, "")
        if effort:
            output_config["effort"] = effort
        if output_config:
            kwargs["output_config"] = output_config
        # Sampling params (temperature) are rejected by current Claude models, so none are sent.
        try:
            if self.model in _CLAUDE_FALLBACK_MODELS:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except (a.BadRequestError, a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError) as e:
            raise PermanentError(f"{self}: {e}") from e
        except (a.RateLimitError, a.APIStatusError, a.APIConnectionError) as e:
            raise LLMError(f"{self}: {e}") from e
        if resp.stop_reason == "refusal":
            raise PermanentError(f"{self}: request declined by the model")
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"{self}: the reply was cut off at the token limit")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text.strip():
            raise LLMError(f"{self}: empty reply (stop_reason={resp.stop_reason})")
        return text


AUTO_SELECT_PROVIDERS = {"gemini", "groq"}


def _auto_model(name: str, preset: Preset, settings: Settings, *, exclude=frozenset(),
                force_refresh: bool = False) -> str:
    """Newest suitable model for an auto-select provider, or '' to fall back to the preset
    default. Runs only when a key is present, so the no-key paths make no network call."""
    key = _key_for(preset, settings)
    if not key:
        return ""
    cache_dir = os.path.join(settings.data_dir or "data", "cache")
    try:
        from . import model_select
        if name == "gemini":
            model = model_select.select_gemini_model(key, cache_dir, exclude=set(exclude),
                                                      force_refresh=force_refresh)
        elif name == "groq":
            model = model_select.select_groq_model(key, cache_dir, exclude=set(exclude),
                                                    force_refresh=force_refresh)
        else:
            return ""
        if model:
            log.info("Auto-selected %s model: %s", name, model)
            return model
        log.warning("Could not auto-select a %s model; using default %s", name, preset.default_model)
    except Exception as e:  # never fatal: fall back to the default; self-heal handles a dead default
        log.warning("Auto-select for %s failed (%s); using default %s", name, redact(e), preset.default_model)
    return ""


def build_provider(name: str, settings: Settings, model: str = "") -> Provider:
    preset = PRESETS[name]
    chosen = model
    if not model and name in AUTO_SELECT_PROVIDERS:
        model = _auto_model(name, preset, settings)
    model = model or preset.default_model
    key = _key_for(preset, settings)
    if preset.kind == "anthropic":
        return AnthropicProvider(model, key, settings.anthropic_effort)
    base_url = preset.base_url
    if name == "custom":
        base_url = settings.llm_base_url
        if not base_url or not model:
            raise LLMConfigError("LLM_PROVIDER=custom needs LLM_BASE_URL and LLM_MODEL")
    elif name == "ollama":
        base_url = ollama_url()
        installed = None if chosen else ollama_models(base_url)
        if installed and not ollama_has(installed, model):
            # No model chosen and the default is not pulled: use one that is, instead of failing every video.
            log.info("Ollama has no '%s'; using installed model '%s' (set LLM_MODEL to choose)", model, installed[0])
            model = installed[0]
    if settings.llm_base_url and name == settings.llm_provider:
        base_url = settings.llm_base_url
    if not key and not preset.local and name != "custom":
        raise LLMConfigError(f"{preset.label}: set {' or '.join(preset.key_envs)} in .env")
    return OpenAICompatible(preset, model, key, base_url)


_MODEL_GONE_RE = re.compile(
    r"(?i)(http 404|not found|no longer available|does not exist|unknown model|model .*\bnot\b)")


def _is_model_gone_error(error: Exception) -> bool:
    """True when the error means the requested model no longer exists (so re-selecting helps)."""
    return bool(_MODEL_GONE_RE.search(str(error)))


class LLM:
    """Primary provider + ordered fallbacks. ``complete_json`` returns a parsed dict."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_used = ""
        primary = resolve_provider_name(settings)
        names = [primary] + [n for n in settings.llm_fallbacks if n != primary]
        # Automatic backup: if a Groq key is set and Groq isn't already in the chain, append it.
        if "groq" not in names and _key_for(PRESETS["groq"], settings):
            names.append("groq")
            log.info("Groq backup enabled (GROQ_API_KEY set)")
        elif "groq" not in names:
            log.debug("No Groq backup; set GROQ_API_KEY to add an automatic fallback")
        self.providers: list[Provider] = []
        for n in names:
            if n not in PRESETS:
                log.warning("Ignoring unknown LLM fallback '%s'", n)
                continue
            try:
                self.providers.append(build_provider(n, settings, settings.llm_model if n == primary else ""))
            except LLMError as e:
                if n == primary:
                    raise
                log.warning("Skipping fallback %s: %s", n, e)

    @property
    def label(self) -> str:
        return str(self.providers[0])

    def complete_json(self, system: str, user: str, schema: dict | None = None,
                      max_tokens: int = 4000, temperature: float | None = None,
                      effort: str | None = None) -> dict[str, Any]:
        """``temperature`` overrides LLM_TEMPERATURE for this call (low for grading, where repeatability matters).
        ``effort`` ("low") asks a reasoning model for short reasoning, where the provider supports it."""
        errors: list[str] = []
        shrunk: set[int] = set()           # providers already retried once after "request too large"
        temp = self.settings.llm_temperature if temperature is None else temperature
        # Providers whose circuit is open go last: they are tried only if every other provider fails.
        providers = ([p for p in self.providers if not self.circuit_open(p.name)]
                     + [p for p in self.providers if self.circuit_open(p.name)])
        rl_waits: dict[int, int] = {}      # per-provider count of 60s waits already spent on a per-minute limit
        i = 0
        while i < len(providers):
            p = providers[i]

            takes_effort = getattr(p, "takes_effort", lambda: False)()

            # After a "request too large": a reply budget that is sure to fit. The model keeps its normal
            # reasoning when that leaves it room (a script written with low effort came out mostly invented);
            # only a tight budget asks for short reasoning.
            fit = min(max_tokens, _fitting_reply_tokens(system, user)) if i in shrunk else max_tokens
            small_effort = "low" if i in shrunk and fit < 3800 else effort

            def _call(p=p, fit=fit, small_effort=small_effort, takes_effort=takes_effort):
                t0 = time.time()
                extra = {"effort": small_effort} if small_effort and takes_effort else {}
                try:
                    text = p.complete(system, user, schema=schema if schema is not None else {},
                                      max_tokens=fit, temperature=temp, **extra)
                except Exception as e:
                    self.note_failure(p.name, e)
                    raise
                self.note_success(p.name)
                log.info("LLM %s answered in %.1fs", p, time.time() - t0)
                return extract_json(text)
            try:
                result = with_retries(_call, attempts=2, base_delay=3, label=f"LLM {p}")
                self.last_used = str(p)
                if p is not self.providers[0]:
                    log.info("Used fallback LLM %s", p)
                return result
            except Exception as e:
                # Self-heal: the chosen model is gone (404 / "no longer available"). Pick the next
                # newest model for this provider and retry it once before moving to the next provider.
                healed = self._reselect_on_model_gone(p, e)
                if healed is not None:
                    log.info("%s model '%s' unavailable; retrying with '%s'", p.name, p.model, healed.model)
                    providers[i] = healed
                    continue
                # Groq sized the request above its per-minute limit: ask once more for a short reply with
                # short reasoning, which it sizes as prompt + max_tokens.
                if (p.name == "groq" and _is_too_large(e) and i not in shrunk
                        and _fitting_reply_tokens(system, user) >= 1500):
                    shrunk.add(i)
                    log.warning("Groq refused the request as too large; retrying once with a %d-token reply",
                                min(max_tokens, _fitting_reply_tokens(system, user)))
                    continue
                # The Groq backup gets per-minute rate-limited on the free tier: that clears within a minute,
                # so wait 60s and retry (up to 3 times) instead of skipping the video.
                if p.name == "groq" and _is_per_minute_rate_limit(e) and rl_waits.get(i, 0) < GROQ_RATE_LIMIT_RETRIES:
                    rl_waits[i] = rl_waits.get(i, 0) + 1
                    log.warning("Groq hit a per-minute rate limit; waiting %.0fs then retrying (%d/%d)",
                                GROQ_RATE_LIMIT_WAIT, rl_waits[i], GROQ_RATE_LIMIT_RETRIES)
                    check_cancel()
                    time.sleep(GROQ_RATE_LIMIT_WAIT)
                    continue
                errors.append(f"{p}: {e}")
                log.warning("LLM %s failed: %s", p, e)
                i += 1
        raise LLMError("All LLM providers failed:\n  " + "\n  ".join(errors))

    # -- circuit breaker (see CIRCUIT_PROVIDERS) ------------------------------------------------------------
    def circuit_open(self, name: str) -> bool:
        return name in self.__dict__.setdefault("_cb_open", set())

    def note_failure(self, name: str, error: BaseException) -> None:
        """Count a transient failure; the CIRCUIT_AFTER-th in a row opens the circuit for the rest of the run."""
        if name not in CIRCUIT_PROVIDERS or not is_transient_failure(error):
            return
        fails = self.__dict__.setdefault("_cb_fail", {})
        fails[name] = fails.get(name, 0) + 1
        if fails[name] >= CIRCUIT_AFTER and not self.circuit_open(name):
            self._cb_open.add(name)
            log.warning("%s circuit open: %d consecutive 503/429/timeouts (last: %s); skipping it for the rest "
                        "of this run", name.title(), fails[name], redact(str(error))[:120])

    def note_success(self, name: str) -> None:
        self.__dict__.setdefault("_cb_fail", {})[name] = 0

    @staticmethod
    def _yes_no(answer: str) -> bool | None:
        a = (answer or "").strip().lower()
        if a.startswith(("yes", "y ")) or a in ("y", "true") or re.search(r"\byes\b", a):
            return True
        if a.startswith(("no", "n ")) or a in ("n", "false") or re.search(r"\bno\b", a):
            return False
        return None

    def _vision_question(self, subject: str) -> str:
        return (f'Does the main subject of this image clearly show {subject}? It must be {subject}, not a beach, '
                f'a street, an animal, a person, a sign, or an unrelated place. Answer only "yes" or "no".')

    def vision_available(self) -> bool:
        """True when a vision backend is configured (so a None result means 'the call failed', not 'no model')."""
        if any(os.getenv(e) for e in ("GEMINI_API_KEY", "GOOGLE_API_KEY")):
            return True
        return bool(self.settings.cloudflare_account_id and self.settings.cloudflare_api_token)

    def _gemini_vision_model(self) -> tuple[str, str] | None:
        """(key, model) for Gemini vision — the account's current model, not a hardcoded (and now 404'd) one."""
        key = next((os.environ[e] for e in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if os.getenv(e)), "")
        if not key:
            return None
        model = (self.settings.vision_model or "").strip()
        if not model:
            try:
                from . import model_select
                cache_dir = os.path.join(self.settings.data_dir or "data", "cache")
                model = model_select.select_gemini_model(key, cache_dir) or ""
            except Exception:
                model = ""
        return key, (model or "gemini-3.8-flash")

    def image_shows(self, image_path, subject: str) -> bool | None:
        """Ask a free vision model whether ``image_path`` clearly shows ``subject``. Tries Gemini (its current
        model) then Cloudflare Workers AI (LLaVA, using the image-gen credentials already configured). Returns
        True/False, or None when no vision model answered (down, rate-limited, unusable reply). The picture is
        re-encoded to a plain JPEG first. Raises ImageUnusable when the image itself is the problem (it can't
        be decoded, or every backend that looked at it refused its format), so the caller can skip just that
        image rather than treat vision as down."""
        import base64

        raw = vision_jpeg(image_path)                       # ImageUnusable propagates
        mime = "image/jpeg"
        question = self._vision_question(subject)
        image_error: Exception | None = None

        gem = self._gemini_vision_model() if not self.circuit_open("gemini") else None
        if gem:
            key, model = gem
            uri = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
            payload = {"model": model, "temperature": 0, "max_tokens": 4,
                       "messages": [{"role": "user", "content": [
                           {"type": "text", "text": question},
                           {"type": "image_url", "image_url": {"url": uri}}]}]}
            try:
                r = http().post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                                json=payload, timeout=GEMINI_TIMEOUT,
                                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                raise_for_status(r, "Gemini vision")
                v = self._yes_no(r.json()["choices"][0]["message"]["content"])
                self.note_success("gemini")
                if v is not None:
                    return v
            except Exception as e:
                log.debug("Gemini vision failed: %s", redact(e))
                self.note_failure("gemini", e)
                if isinstance(e, PermanentError) and _IMAGE_ERROR_RE.search(str(e)):
                    image_error = e

        acct, tok = self.settings.cloudflare_account_id, self.settings.cloudflare_api_token
        if acct and tok:
            url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/llava-hf/llava-1.5-7b-hf"
            try:
                r = http().post(url, json={"image": list(raw), "prompt": question, "max_tokens": 8},
                                headers={"Authorization": f"Bearer {tok}"}, timeout=90)
                raise_for_status(r, "Cloudflare vision")
                v = self._yes_no(str((r.json().get("result") or {}).get("description", "")))
                if v is not None:
                    return v
            except Exception as e:
                log.debug("Cloudflare vision failed: %s", redact(e))
                if isinstance(e, PermanentError) and _IMAGE_ERROR_RE.search(str(e)):
                    image_error = e
        if image_error is not None:
            raise ImageUnusable(f"vision backend refused the image: {redact(image_error)}")
        return None

    def _reselect_on_model_gone(self, provider: Provider, error: Exception) -> Provider | None:
        """If ``error`` means the model is gone, rebuild the provider on the next newest model
        (refreshing the list) and return it, else None. Each model name is tried at most once."""
        if provider.name not in AUTO_SELECT_PROVIDERS or not _is_model_gone_error(error):
            return None
        tried = self.__dict__.setdefault("_tried", {}).setdefault(provider.name, set())
        tried.add(provider.model)
        new_model = _auto_model(provider.name, PRESETS[provider.name], self.settings,
                                exclude=tried, force_refresh=True)
        if not new_model or new_model in tried:
            return None
        try:
            return build_provider(provider.name, self.settings, new_model)
        except LLMError:
            return None
