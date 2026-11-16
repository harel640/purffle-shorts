"""Week-2 analytics: curiosity headlines (no years, no 'PLACE: fact'), every headline premise checked, one event
instead of a list of years, and parallax that does not run the render out of memory."""

import concurrent.futures

import pytest

from purffle_shorts import factcheck as F
from purffle_shorts import render as R
from purffle_shorts.config import Settings
from purffle_shorts.media import MediaItem
from purffle_shorts.script import Scene, Script, ensure_one_event, narration_years


def _script(*lines, title="t", hook="h"):
    return Script(topic="t", title=title, hook_text=hook, description="d", style="history",
                  scenes=[Scene(x, "q") for x in lines])


SAN_AGUSTIN = _script("Why was one church the only building left standing?",
                      "Construction began in 1586.", "The church was completed in 1607.",
                      "The stone edifice survived earthquakes in 1645, 1699, 1754, 1796, 1825, and 1852.",
                      "On June 3, 1863, Manila experienced its strongest quake.",
                      "Which resilient place would you protect?")


# ------------------------------------------------------------------ one event, not a list of years
def test_the_years_of_a_narration_are_found():
    assert narration_years(SAN_AGUSTIN) == ["1586", "1607", "1645", "1699", "1754", "1796", "1825", "1852", "1863"]


class _LLM:
    def __init__(self, scenes):
        self.scenes, self.calls = scenes, 0

    def complete_json(self, system, user, schema=None, **k):
        self.calls += 1
        assert "AT MOST TWO years" in user and "ONE central event" in user
        return {"scenes": [{"narration": n, "search_query": "q"} for n in self.scenes]}


def test_a_list_of_years_is_rewritten_once_around_one_event():
    llm = _LLM(["Why was one church the only building left standing?", "Its builders laid the first stones.",
                "Earthquake after earthquake shook Manila, and the church stood.",
                "On June 3, 1863, Manila experienced its strongest quake.", "The church came through it undamaged.",
                "Which resilient place would you protect?"])
    out = ensure_one_event(llm, SAN_AGUSTIN)
    assert llm.calls == 1 and narration_years(out) == ["1863"]
    assert "Earthquake after earthquake" in out.narration


def test_a_rewrite_that_still_lists_years_keeps_the_original():
    llm = _LLM(["In 1586, 1607 and 1645 it was built and shaken."] * 6)
    assert ensure_one_event(llm, SAN_AGUSTIN) is SAN_AGUSTIN


def test_two_years_or_fewer_need_no_rewrite():
    llm = _LLM([])
    s = _script("How did a city vanish in months?", "In 1565 an army sacked it.", "By 1600 it was empty.")
    assert ensure_one_event(llm, s) is s and llm.calls == 0


def test_the_writer_is_told_one_event_and_never_a_birth():
    from purffle_shorts.script import STORY_SHAPE_RULE
    assert "at most TWO years" in STORY_SHAPE_RULE and "never a list of years" in STORY_SHAPE_RULE
    assert "never a birth, a founding date or an administrative fact" in STORY_SHAPE_RULE


# ------------------------------------------------------------------ headline questions are checked by their premise
def _row(item, said, quote=""):
    return F._judge(item, {"verdict": said, "quote": quote}, F._norm("a villager's roof survived the heavy snow"),
                    "", frozenset())


def test_headline_items_carry_their_role():
    items = F.claim_items(_script("How did one roof survive massive snow?", "Snow fell for weeks.",
                                  title="How did one roof survive?", hook="Buried in snow?"))
    roles = {it["text"]: it.get("role") for it in items}
    assert roles["How did one roof survive?"] == "TITLE" and roles["Buried in snow?"] == "HOOK"
    assert roles["How did one roof survive massive snow?"] == "OPENING" and roles["Snow fell for weeks."] is None


@pytest.mark.parametrize("role", ["TITLE", "HOOK", "OPENING"])
def test_a_headline_question_cannot_pass_as_not_factual(role):
    item = {"id": 0, "kind": "title" if role == "TITLE" else "hook" if role == "HOOK" else "narration",
            "text": "How did one villager's roof survive massive snow?", "role": role}
    row = _row(item, "not_factual")
    assert row["verdict"] == "unsupported" and row["reason"] == F._HEADLINE_QUESTION


def test_the_closing_viewer_question_is_still_not_factual():
    item = {"id": 9, "kind": "narration", "text": "Which village would you visit?", "scene": 5, "pos": 0}
    assert _row(item, "not_factual")["verdict"] == "not_factual"


def test_the_checker_is_told_to_judge_a_headline_question_by_its_premise():
    assert "judged by what it takes for granted" in F._CLAIMS_RULES and "never not_factual" in F._CLAIMS_RULES


# ------------------------------------------------------------------ parallax runs at most 3 segments at a time
def test_parallax_renders_at_most_three_segments_at_once(monkeypatch, tmp_path):
    seen = {}
    real = concurrent.futures.ThreadPoolExecutor

    def pool(max_workers=None, **k):
        seen.setdefault("workers", max_workers)
        return real(max_workers=max_workers, **k)
    monkeypatch.setattr(R.concurrent.futures, "ThreadPoolExecutor", pool)
    monkeypatch.setattr(R, "prepare_segment", lambda m, L, i, s, w: tmp_path / f"s{i}.mp4")
    monkeypatch.setattr(R, "compose", lambda *a, **k: tmp_path / "out.mp4")
    monkeypatch.setattr(R.os, "cpu_count", lambda: 24)
    media = [MediaItem("wikimedia", str(i), "image", extra={"depth": "d.png"}) for i in range(12)]
    R.render_video(Settings(), media, [4.0] * 12, tmp_path / "v.wav", 48.0, None, tmp_path / "o.mp4", tmp_path)
    assert seen["workers"] == R.PARALLAX_WORKERS == 3
    seen.clear()
    plain = [MediaItem("wikimedia", str(i), "image") for i in range(12)]
    R.render_video(Settings(), plain, [4.0] * 12, tmp_path / "v.wav", 48.0, None, tmp_path / "o.mp4", tmp_path)
    assert seen["workers"] > 3
