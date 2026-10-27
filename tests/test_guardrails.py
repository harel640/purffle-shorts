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
def test_safety_blocks_high_advisory(monkeypatch):
    monkeypatch.setattr(safety, "advisory_score", lambda iso2: (4.2, "US State Dept"))
    v = safety.check("Visit Bagan, Myanmar", FakeLLM({"country": "Myanmar", "iso2": "MM"}))
    assert v.unsafe and "Myanmar" in v.reason


def test_safety_allows_low_advisory(monkeypatch):
    monkeypatch.setattr(safety, "advisory_score", lambda iso2: (1.8, "src"))
    v = safety.check("Hampi, India", FakeLLM({"country": "India", "iso2": "IN"}))
    assert not v.unsafe


def test_safety_ignores_non_country(monkeypatch):
    called = []
    monkeypatch.setattr(safety, "advisory_score", lambda iso2: called.append(1) or (5.0, "x"))
    v = safety.check("productivity tips", FakeLLM({"country": "", "iso2": ""}))
    assert not v.unsafe and called == []  # no advisory lookup for non-destinations


def test_safety_fails_open_on_lookup_error(monkeypatch):
    def boom(iso2):
        raise RuntimeError("network down")
    monkeypatch.setattr(safety, "advisory_score", boom)
    v = safety.check("Somewhere, Testland", FakeLLM({"country": "Testland", "iso2": "TL"}))
    assert not v.unsafe  # fail-open so an outage never halts the channel


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
