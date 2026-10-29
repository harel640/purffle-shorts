"""Tests for the Maker guardrails: travel-safety filter and the fact-check pass (no network)."""

from purffle_shorts import factcheck, safety
from purffle_shorts.script import Scene, Script


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete_json(self, system, user, **k):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# --------------------------------------------------------------- safety filter
FEED = {"burma": 4, "india": 2, "saudi arabia": 3, "thailand": 1}


def test_safety_blocks_myanmar(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Explore Bagan, Myanmar", FakeLLM({"country": "Myanmar", "aliases": ["Burma"]}))
    assert v.unsafe and v.level == 4 and "burma" in v.country.lower()


def test_safety_blocks_reconsider_level3(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Riyadh desert trip", FakeLLM({"country": "Saudi Arabia", "aliases": []}))
    assert v.unsafe and v.level == 3


def test_safety_allows_india(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Hampi, India", FakeLLM({"country": "India", "aliases": []}))
    assert not v.unsafe and v.level == 2


def test_safety_ignores_non_country(monkeypatch):
    called = []
    monkeypatch.setattr(safety, "advisories", lambda: called.append(1) or FEED)
    v = safety.check("productivity tips", FakeLLM({"country": "", "aliases": []}))
    assert not v.unsafe and called == []  # no feed lookup for non-destinations


def test_safety_fails_closed_when_feed_down(monkeypatch):
    def boom():
        raise RuntimeError("feed down")
    monkeypatch.setattr(safety, "advisories", boom)
    v = safety.check("Hampi, India", FakeLLM({"country": "India", "aliases": []}))
    assert v.unsafe and "could not be verified" in v.reason  # fail CLOSED on outage


def test_safety_fails_closed_when_country_missing(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Visit Testlandia", FakeLLM({"country": "Testlandia", "aliases": []}))
    assert v.unsafe and "could not verify" in v.reason  # unknown country -> skip


# --------------------------------------------------------------- fact check
def _script():
    return Script(
        topic="Bagan", title="85% of temples are hidden",
        hook_text="85% of Bagan's 2,300 temples are never seen",
        scenes=[
            Scene(narration="Bagan has exactly 2,300 temples.", search_query="bagan", emphasis=["2,300"]),
            Scene(narration="Most are peaceful at dawn.", search_query="dawn", emphasis=[]),
        ],
        description="",
    )


def test_factcheck_rewrites_unsupported_stats():
    reply = {
        "title": "Most temples are hidden",
        "hook": "Most of Bagan's temples are never seen",
        "lines": [
            {"id": 0, "text": "Bagan has thousands of temples."},
            {"id": 1, "text": "Most are peaceful at dawn."},
        ],
        "flagged": ["removed 2,300 and 85%"],
    }
    out = factcheck.scrub(FakeLLM(reply), _script(), settings=None)
    assert "thousands" in out.scenes[0].narration and "2,300" not in out.scenes[0].narration
    assert out.scenes[0].emphasis == []          # "2,300" pruned — no longer in the line
    assert "85%" not in out.title and "85%" not in out.hook_text


def test_factcheck_fails_open_on_error():
    sc = _script()
    before = sc.scenes[0].narration
    out = factcheck.scrub(FakeLLM(RuntimeError("boom")), sc, settings=None)
    assert out.scenes[0].narration == before     # unchanged on LLM error
