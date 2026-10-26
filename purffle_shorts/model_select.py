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


def choose_gemini_model(api_key: str, *, exclude: set[str] = frozenset()) -> str:
    """Newest stable 'flash' model that supports generateContent, or '' if none."""
    cands: list[tuple[tuple[int, ...], str]] = []
    for m in list_gemini_models(api_key):
        name = (m.get("name") or "").split("/")[-1]  # "models/gemini-3.8-flash" -> "gemini-3.8-flash"
        if not name or name in exclude:
            continue
        methods = m.get("supportedGenerationMethods") or m.get("supportedActions") or []
        if "generateContent" not in methods:
            continue
        if "flash" not in name.lower() or not is_stable(name):
            continue
        cands.append((version_key(name), name))
    cands.sort(reverse=True)
    return cands[0][1] if cands else ""


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
