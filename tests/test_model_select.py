"""Tests for self-maintaining model selection, self-heal and the Groq backup (no network)."""

import pytest

from purffle_shorts import llm as L
from purffle_shorts import model_select as M
from purffle_shorts.config import Settings
from purffle_shorts.utils import PermanentError
import purffle_shorts.utils as U


# --------------------------------------------------------------- pure helpers
def test_version_key_orders_newest_highest():
    assert M.version_key("gemini-3.8-flash") > M.version_key("gemini-2.5-flash")
    assert M.version_key("gemini-2.10-flash") > M.version_key("gemini-2.5-flash")
    assert M.version_key("gemini-flash-latest") == (-1,)


def test_is_stable():
    assert M.is_stable("gemini-3.8-flash")
    assert not M.is_stable("gemini-3.8-flash-preview-09-2026")
    assert not M.is_stable("gemini-2.0-flash-exp")


# --------------------------------------------------------------- Gemini selection
GEMINI_LIST = [
    {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.8-flash-preview-09", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.8-pro", "supportedGenerationMethods": ["generateContent"]},   # not flash
    {"name": "models/gemini-4.0-flash", "supportedGenerationMethods": ["embedContent"]},      # no generateContent
]


def test_choose_gemini_picks_newest_stable_flash(monkeypatch):
    monkeypatch.setattr(M, "list_gemini_models", lambda key: GEMINI_LIST)
    assert M.choose_gemini_model("k") == "gemini-3.8-flash"


def test_choose_gemini_excludes_tried(monkeypatch):
    monkeypatch.setattr(M, "list_gemini_models", lambda key: GEMINI_LIST)
    assert M.choose_gemini_model("k", exclude={"gemini-3.8-flash"}) == "gemini-2.5-flash"


def test_select_gemini_caches_then_refreshes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(M, "list_gemini_models", lambda key: calls.append(1) or GEMINI_LIST)
    cache = str(tmp_path)
    assert M.select_gemini_model("k", cache) == "gemini-3.8-flash"
    assert M.select_gemini_model("k", cache) == "gemini-3.8-flash"
    assert len(calls) == 1  # second call served from the 24h cache
    assert M.select_gemini_model("k", cache, force_refresh=True) == "gemini-3.8-flash"
    assert len(calls) == 2  # force_refresh bypasses the cache


def test_cache_entry_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "list_gemini_models", lambda key: GEMINI_LIST)
    cache = str(tmp_path)
    M.select_gemini_model("k", cache)
    monkeypatch.setattr(M, "CACHE_TTL", -1)
    assert M.read_cache(cache, "gemini", exclude=set()) == ""  # now stale


# --------------------------------------------------------------- Groq selection
GROQ_LIST = [
    {"id": "llama-3.1-8b-instant", "active": True, "created": 100},
    {"id": "llama-3.3-70b-versatile", "active": True, "created": 200},
    {"id": "whisper-large-v3", "active": True, "created": 999},                 # not a chat model
    {"id": "meta-llama/llama-guard-4-12b", "active": True, "created": 999},      # guard
    {"id": "llama-4-scout-17b", "active": False, "created": 999},                # inactive
]


def test_choose_groq_picks_newest_active_chat(monkeypatch):
    monkeypatch.setattr(M, "list_groq_models", lambda key: GROQ_LIST)
    assert M.choose_groq_model("k") == "llama-3.3-70b-versatile"


# --------------------------------------------------------------- build_provider
def test_build_provider_auto_selects_gemini(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(M, "list_gemini_models", lambda key: GEMINI_LIST)
    p = L.build_provider("gemini", Settings(data_dir=str(tmp_path)))
    assert p.model == "gemini-3.8-flash"


def test_explicit_model_overrides_auto_select(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    called = []
    monkeypatch.setattr(M, "list_gemini_models", lambda key: called.append(1) or GEMINI_LIST)
    p = L.build_provider("gemini", Settings(data_dir=str(tmp_path)), "gemini-2.5-flash")
    assert p.model == "gemini-2.5-flash"
    assert called == []  # a pinned LLM_MODEL means no ListModels call


def test_no_key_means_no_network(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(M, "list_gemini_models", lambda key: called.append(1) or GEMINI_LIST)
    with pytest.raises(L.LLMConfigError):
        L.build_provider("gemini", Settings(data_dir=str(tmp_path)))
    assert called == []  # nothing is fetched without a key


# --------------------------------------------------------------- self-heal
class _Gone(L.Provider):
    name = "gemini"

    def __init__(self, model):
        self.model = model

    def complete(self, *a, **k):
        raise PermanentError(f"gemini:{self.model}: HTTP 404 — no longer available")


class _Good(L.Provider):
    name = "gemini"

    def __init__(self, model):
        self.model = model

    def complete(self, *a, **k):
        return '{"ok": true}'


def test_self_heal_switches_to_next_model(monkeypatch):
    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [_Gone("old")]
    monkeypatch.setattr(L, "_auto_model", lambda *a, **k: "new")
    monkeypatch.setattr(L, "build_provider", lambda name, s, model: _Good(model))
    assert chain.complete_json("s", "u") == {"ok": True}
    assert chain.last_used == "gemini:new"


def test_no_self_heal_on_transient_error(monkeypatch):
    monkeypatch.setattr(U.time, "sleep", lambda *_: None)  # don't wait out the retry backoff

    class Busy(L.Provider):
        name, model = "gemini", "m"

        def complete(self, *a, **k):
            raise RuntimeError("HTTP 503 — high demand")

    chain = object.__new__(L.LLM)
    chain.settings, chain.providers = Settings(), [Busy()]
    reselect = []
    monkeypatch.setattr(L, "_auto_model", lambda *a, **k: reselect.append(1) or "new")
    with pytest.raises(L.LLMError, match="All LLM providers failed"):
        chain.complete_json("s", "u")
    assert reselect == []  # a 503 is not a "model gone" error


# --------------------------------------------------------------- Groq backup
def _fake_build(name, s, model=""):
    return type("P", (L.Provider,), {"name": name, "model": "m"})()


def test_groq_backup_added_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setattr(L, "build_provider", _fake_build)
    chain = L.LLM(Settings())
    assert [p.name for p in chain.providers] == ["gemini", "groq"]


def test_no_groq_backup_without_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr(L, "build_provider", _fake_build)
    chain = L.LLM(Settings())
    assert [p.name for p in chain.providers] == ["gemini"]
