"""Self-maintaining model selection.

Providers whose model names change over time (Gemini and Groq) are not hard-pinned.
At startup the newest *stable* model that fits the job is chosen from the provider's own
list-models API and cached for 24 hours, so model ids never have to be edited by hand.
An explicit LLM_MODEL always wins (see llm.build_provider).

Network/parse failures here are never fatal: callers fall back to the preset default and,
if that default is gone, the self-heal path in llm.py refreshes and tries the next model.
See CLAUDE.md.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from .utils import http, raise_for_status

log = logging.getLogger("purffle")

CACHE_TTL = 24 * 3600  # re-pick a model at most once a day

# Substrings that mark a model as not a stable, generally-available release.
_UNSTABLE = ("preview", "exp", "experimental", "-rc", "beta", "nightly", "-tuning", "latest-")


def is_stable(model_id: str) -> bool:
    low = model_id.lower()
    return not any(tok in low for tok in _UNSTABLE)


def version_key(model_id: str) -> tuple[int, ...]:
    """Sort key from the version numbers in a model id: higher = newer.

    'gemini-3.8-flash' -> (3, 8); 'gemini-2.5-flash' -> (2, 5); a model with no number
    (e.g. a bare '-latest' alias) sorts lowest so a concretely-versioned release wins.
    """
    nums = re.findall(r"\d+", model_id)
    return tuple(int(n) for n in nums) if nums else (-1,)


# --------------------------------------------------------------------------- cache
def _cache_file(cache_dir: str, provider: str) -> Path:
    return Path(cache_dir) / "models" / f"{provider}.json"


def read_cache(cache_dir: str, provider: str, *, exclude: set[str]) -> str:
    try:
        d = json.loads(_cache_file(cache_dir, provider).read_text("utf-8"))
    except (OSError, ValueError):
        return ""
    model = str(d.get("model", ""))
    fresh = (time.time() - float(d.get("ts", 0))) < CACHE_TTL
    return model if (model and model not in exclude and fresh) else ""


def write_cache(cache_dir: str, provider: str, model: str) -> None:
    try:
        f = _cache_file(cache_dir, provider)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"model": model, "ts": time.time()}), "utf-8")
    except OSError as e:  # a read-only cache dir must not break model selection
        log.debug("Could not cache %s model: %s", provider, e)


# --------------------------------------------------------------------------- Gemini
def list_gemini_models(api_key: str) -> list[dict]:
    r = http().get("https://generativelanguage.googleapis.com/v1beta/models",
                   params={"key": api_key, "pageSize": 1000}, timeout=30)
    raise_for_status(r, "Gemini ListModels")
    return r.json().get("models", [])


# Flash models that do not write text (wrong output modality): skip them in auto-select,
# even though they report generateContent.
_GEMINI_NON_TEXT = ("tts", "image", "audio", "embedding", "aqa")


def choose_gemini_model(api_key: str, *, exclude: set[str] = frozenset()) -> str:
    """Newest stable text 'flash' model that supports generateContent, or '' if none.

    A tie on version prefers the plainer (shorter) name, so 'gemini-3.8-flash' wins over a
    specialised variant such as 'gemini-3.8-flash-tts'.
    """
    cands: list[tuple[tuple[int, ...], int, str]] = []
    for m in list_gemini_models(api_key):
        name = (m.get("name") or "").split("/")[-1]  # "models/gemini-3.8-flash" -> "gemini-3.8-flash"
        low = name.lower()
        if not name or name in exclude:
            continue
        methods = m.get("supportedGenerationMethods") or m.get("supportedActions") or []
        if "generateContent" not in methods:
            continue
        if "flash" not in low or not is_stable(name):
            continue
        if any(tok in low for tok in _GEMINI_NON_TEXT):
            continue
        cands.append((version_key(name), -len(name), name))
    cands.sort(reverse=True)
    return cands[0][2] if cands else ""


def select_gemini_model(api_key: str, cache_dir: str, *, exclude: set[str] = frozenset(),
                        force_refresh: bool = False) -> str:
    """Cached newest-stable Gemini flash model. Returns '' if the list can't be fetched."""
    if not force_refresh and not exclude:
        cached = read_cache(cache_dir, "gemini", exclude=set(exclude))
        if cached:
            return cached
    model = choose_gemini_model(api_key, exclude=set(exclude))
    if model and not exclude:
        write_cache(cache_dir, "gemini", model)
    return model


# --------------------------------------------------------------------------- Groq
# Groq hosts more than chat models; skip the ones that cannot write a script.
_GROQ_SKIP = ("whisper", "tts", "guard", "embed", "prompt-guard", "-vision", "allam", "distil", "moderation")


def list_groq_models(api_key: str) -> list[dict]:
    r = http().get("https://api.groq.com/openai/v1/models",
                   headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    raise_for_status(r, "Groq ListModels")
    return r.json().get("data", [])


def choose_groq_model(api_key: str, *, exclude: set[str] = frozenset()) -> str:
    """Newest active, stable Groq chat model, or ''. 'Newest' = highest release date
    (the API's ``created`` timestamp), then version number."""
    cands: list[tuple[tuple[float, tuple[int, ...]], str]] = []
    for m in list_groq_models(api_key):
        mid = m.get("id") or ""
        low = mid.lower()
        if not mid or mid in exclude or m.get("active") is False:
            continue
        if any(tok in low for tok in _GROQ_SKIP) or not is_stable(mid):
            continue
        cands.append(((float(m.get("created") or 0), version_key(mid)), mid))
    cands.sort(reverse=True)
    return cands[0][1] if cands else ""


def select_groq_model(api_key: str, cache_dir: str, *, exclude: set[str] = frozenset(),
                      force_refresh: bool = False) -> str:
    """Cached newest-stable Groq chat model. Returns '' if the list can't be fetched."""
    if not force_refresh and not exclude:
        cached = read_cache(cache_dir, "groq", exclude=set(exclude))
        if cached:
            return cached
    model = choose_groq_model(api_key, exclude=set(exclude))
    if model and not exclude:
        write_cache(cache_dir, "groq", model)
    return model
