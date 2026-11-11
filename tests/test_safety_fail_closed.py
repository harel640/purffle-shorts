"""P2: the travel-advisory check fails CLOSED.

Before this, safety.check() returned "safe" whenever the LLM could not be asked where a topic was, and four
videos were uploaded after an unverified check. The LLM call itself failed so often because Groq's gpt-oss
model spends 110-355 hidden reasoning tokens on a 60-token answer, and the call allowed only 250.
"""

import json
from types import SimpleNamespace

import pytest

from purffle_shorts import llm as L
from purffle_shorts import pipeline as P
from purffle_shorts import safety
from purffle_shorts import topics as T
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.script import Scene, Script
from purffle_shorts.utils import PermanentError

FEED = {"japan": 1, "india": 2, "burma": 4, "bhutan": 1, "philippines": 2, "vietnam": 1}


@pytest.fixture(autouse=True)
def _feed(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)


class LocLLM:
    """Answers the location question with ``reply`` (a dict, or an exception to raise)."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete_json(self, system, user, schema=None, max_tokens=0, temperature=None, **k):
        self.calls.append(max_tokens)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class NoLLM:
    def complete_json(self, *a, **k):
        pytest.fail("a listed place must be checked without asking the LLM")


DOWN = RuntimeError("All LLM providers failed:\n  groq: HTTP 400 Failed to validate JSON")


# ------------------------------------------------------------------ safety.check
def test_llm_down_is_unsafe_and_transient_not_safe():
    v = safety.check("Fort Pilar, Zamboanga City, Philippines", LocLLM(DOWN))
    assert v.unsafe and v.transient and "could not be verified" in v.reason


def test_an_unusable_reply_is_unsafe_too():
    v = safety.check("Hampi, India", LocLLM(["not", "a", "dict"]))
    assert v.unsafe and v.transient


def test_no_country_passes_only_when_a_place_is_not_required():
    llm = LocLLM({"country": "", "aliases": [], "region": ""})
    assert not safety.check("hidden gems in Asia", llm).unsafe          # a theme, not a destination
    v = safety.check("Somewhere, Nowhere", llm, require_place=True)      # a resolved place must have a country
    assert v.unsafe and not v.transient


def test_the_location_call_has_room_for_hidden_reasoning():
    llm = LocLLM({"country": "Japan", "aliases": [], "region": "Gifu"})
    assert not safety.check("Shirakawa-go, Japan", llm).unsafe
    assert llm.calls == [safety.IDENTIFY_MAX_TOKENS] and safety.IDENTIFY_MAX_TOKENS >= 1000   # was 250


def test_feed_down_is_transient(monkeypatch):
    def boom():
        raise RuntimeError("feed down")
    monkeypatch.setattr(safety, "advisories", boom)
    v = safety.verdict_for(["japan"])
    assert v.unsafe and v.transient


def test_unsafe_topic_carries_transient():
    e = safety.unsafe_topic(safety.Verdict(True, "x", transient=True))
    assert isinstance(e, safety.UnsafeTopic) and e.transient and str(e) == "x"
    assert not safety.UnsafeTopic("y").transient


def test_same_country_knows_aliases():
    assert safety.same_country("Myanmar", "Burma") and safety.same_country("India", "india")
    assert not safety.same_country("Japan", "China")


# ------------------------------------------------------------------ the pipeline's advisory step
def _studio(tmp_path, llm):
    st = P.Studio(Settings(output_dir=str(tmp_path / "o"), data_dir=str(tmp_path / "d"), upload=False,
                           visual_sources=["wikimedia"], safety_filter=True))
    st._llm = llm
    return st


def _listed(place, country, region=""):
    return T.Topic(T.list_subject(place, region, country), "list", key=f"list:2026-11-09:{place}|{country}".lower(),
                   place=place, country=country, region=region)


@pytest.mark.parametrize("place, country, region, unsafe", [
    ("Shirakawa-go", "Japan", "Gifu", False),
    ("Taktsang Monastery", "Bhutan", "Paro", False),
    ("Mrauk U", "Myanmar", "Rakhine", True),          # Level 4, filed as Burma
    ("Srinagar", "India", "Jammu and Kashmir", True), # a named Level 4 area
    ("Hampi", "India", "", True),                     # a country with Level 3-4 areas, region unknown
    ("Wat Phu", "Laos", "Champasak", True),           # whole country blocked until curated
])
def test_a_listed_place_is_decided_from_the_list_without_the_llm(tmp_path, place, country, region, unsafe):
    st = _studio(tmp_path, NoLLM())
    assert st._advisory(_listed(place, country, region)).unsafe is unsafe


def test_a_listed_place_that_resolves_to_another_country_is_skipped(tmp_path):
    st = _studio(tmp_path, NoLLM())
    pick = _listed("Shirakawa-go", "Japan", "Gifu")
    v = st._advisory(pick, place="Shirakawa", country="China")
    assert v.unsafe and "list vetted" in v.reason and not v.transient
    assert not st._advisory(pick, place="Shirakawa-go", country="Japan").unsafe
    assert not st._advisory(pick, place="Shirakawa-go", country="").unsafe       # no country: the list's stands


def test_other_topics_ask_the_llm_and_fail_closed(tmp_path):
    pick = T.Topic("hidden gems in Asia", "niche")
    v = _studio(tmp_path, LocLLM(DOWN))._advisory(pick)                          # draft stage, LLM down
    assert v.unsafe and v.transient
    v = _studio(tmp_path, LocLLM({"country": "", "aliases": [], "region": ""}))._advisory(
        pick, place="Some Ruins", country="")
    assert v.unsafe                                                               # produce stage: must be a place


# ------------------------------------------------------------------ an outage does not use up a listed place
def _story():
    return Script(topic="Shirakawa-go", title="t", hook_text="h", description="d", style="history",
                  scenes=[Scene("Shirakawa-go has steep thatched roofs.", "q")])


@pytest.mark.parametrize("transient, released", [(True, True), (False, False)])
def test_draft_stage_skip_releases_the_claim_only_when_transient(monkeypatch, tmp_path, transient, released):
    st = _studio(tmp_path, LocLLM(DOWN))
    pick = _listed("Shirakawa-go", "Japan", "Gifu")
    st.history.mark_topic(pick.key)
    monkeypatch.setattr(P, "pick_topic", lambda *a, **k: pick)
    monkeypatch.setattr(st, "_advisory", lambda *a, **k: safety.Verdict(True, "x", transient=transient))
    with pytest.raises(safety.UnsafeTopic):
        st.draft()
    assert st.history.topic_used(pick.key) is not released


@pytest.mark.parametrize("transient, released", [(True, True), (False, False)])
def test_produce_stage_skip_releases_the_claim_only_when_transient(monkeypatch, tmp_path, transient, released):
    st = _studio(tmp_path, LocLLM(DOWN))
    pick = _listed("Shirakawa-go", "Japan", "Gifu")
    st.history.mark_topic(pick.key)
    monkeypatch.setattr(st, "draft", lambda *a, **k: (_story(), pick, "w"))

    def produce(*a, **k):
        raise safety.UnsafeTopic("the location could not be verified", transient=transient)
    monkeypatch.setattr(st, "produce", produce)
    r = st.make(upload=False)
    assert r.status == "skipped" and st.history.topic_used(pick.key) is not released


def test_release_topic(tmp_path):
    h = History(tmp_path / "h.db")
    h.mark_topic("list:x|y")
    h.release_topic("LIST:X|Y")
    assert not h.topic_used("list:x|y")


def test_the_list_hands_its_region_to_the_topic(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps({"generated": "2099-01-01", "places": [
        {"rank": 1, "place": "Taktsang Monastery", "region": "Paro", "country": "Bhutan",
         "wikipedia_title": "Paro Taktsang", "lat": 27.49, "lon": 89.36}]}), encoding="utf-8")
    import datetime as dt
    data = json.loads(p.read_text(encoding="utf-8"))
    data["generated"] = dt.datetime.now(dt.timezone.utc).isoformat()
    p.write_text(json.dumps(data), encoding="utf-8")
    t = T.from_list(Settings(topic_list_file=str(p)), History(tmp_path / "h.db"))
    assert t.region == "Paro" and t.country == "Bhutan"


# ------------------------------------------------------------------ the real cause of the HTTP 400
class _Prov:
    def __init__(self, script):
        self.name, self.model, self.script, self.seen = "groq", "openai/gpt-oss-120b", list(script), []

    def __str__(self):
        return "groq:gpt-oss"

    def takes_effort(self):
        return True

    def complete(self, system, user, **k):
        self.seen.append(k["max_tokens"])
        r = self.script.pop(0) if self.script else '{"ok": true}'
        if isinstance(r, Exception):
            raise r
        return r


def _client(monkeypatch, *providers):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L, "with_retries", lambda fn, attempts=2, **k: fn())
    obj = L.LLM.__new__(L.LLM)
    obj.settings, obj.last_used, obj.providers = Settings(), "", list(providers)
    return obj


TRUNCATED = [
    'groq: HTTP 400 — {"error":{"message":"Failed to validate JSON. Please adjust your prompt.",'
    '"code":"json_validate_failed","failed_generation":""}}',
    'groq: HTTP 400 — {"error":{"message":"Failed to generate JSON.",'
    '"failed_generation":"max completion tokens reached before generating a valid document"}}',
]


@pytest.mark.parametrize("message", TRUNCATED)
def test_a_reply_cut_off_by_max_tokens_is_retried_once_with_more_room(monkeypatch, message):
    p = _Prov([PermanentError(message)])               # what raise_for_status turns Groq's HTTP 400 into
    assert _client(monkeypatch, p).complete_json("s", "u", max_tokens=1500) == {"ok": True}
    assert p.seen[0] == 1500 and p.seen[1] >= 4000
    twice = _Prov([PermanentError(message), PermanentError(message)])
    with pytest.raises(L.LLMError):
        _client(monkeypatch, twice).complete_json("s", "u", max_tokens=1500)
    assert len(twice.seen) == 2                                        # once, not forever


def test_bad_json_that_was_generated_is_not_mistaken_for_a_cut_off():
    e = RuntimeError('groq: HTTP 400 {"code":"json_validate_failed","failed_generation":"{\\"country\\": 1}"}')
    assert not L._is_truncated(e)


def test_hidden_reasoning_models_always_get_room(monkeypatch):
    sent = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json)
            return SimpleNamespace(status_code=200, text="", json=lambda: {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(L, "http", lambda: Session())
    oss = L.OpenAICompatible(L.PRESETS["groq"], "openai/gpt-oss-120b", "k", "https://x")
    oss.complete("s", "u", max_tokens=250)
    assert sent[-1]["max_tokens"] == L.MIN_REASONING_REPLY_TOKENS
    oss.complete("s", "u", max_tokens=6000)
    assert sent[-1]["max_tokens"] == 6000                              # larger budgets are left alone
    llama = L.OpenAICompatible(L.PRESETS["groq"], "llama-3.3-70b-versatile", "k", "https://x")
    llama.complete("s", "u", max_tokens=250)
    assert sent[-1]["max_tokens"] == 250                               # models without hidden reasoning: unchanged
