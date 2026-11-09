"""Week-1 analytics rules: country pacing, concrete-stakes hooks, the hard 45-60s gate, no recent tragedies,
and period-only AI scenes."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from purffle_shorts import media as M
from purffle_shorts import pipeline as P
from purffle_shorts import topics as T
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.media import SkipTopic
from purffle_shorts.script import (Scene, Script, ensure_concrete_hook, recent_violence_reason,
                                   vague_hook_reason)


def _ago(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


# ------------------------------------------------------------------ 2. country pacing
def test_one_video_per_country_per_3_days_and_two_per_week(tmp_path):
    h = History(tmp_path / "h.db")
    assert h.country_quota_reason("India", 3, 2) == ""
    h.add_video(status="uploaded", title="a", country="India", created_at=_ago(1))
    assert "last 3 days" in h.country_quota_reason("India", 3, 2)
    assert h.country_quota_reason("Japan", 3, 2) == ""                      # other countries unaffected
    h2 = History(tmp_path / "h2.db")
    h2.add_video(status="uploaded", title="a", country="India", created_at=_ago(4))
    assert h2.country_quota_reason("India", 3, 2) == ""                     # 4 days ago: one this week is fine
    h2.add_video(status="uploaded", title="b", resolved_place="Hampi, India", created_at=_ago(6))  # legacy row
    assert "2 India videos already this week" in h2.country_quota_reason("India", 3, 2)
    h2.add_video(status="uploaded", title="c", country="India", created_at=_ago(9))
    assert h2.country_quota_reason("india", 3, 3) == ""                     # 9 days ago is outside the week
    assert h.country_quota_reason("India", 0, 0) == ""                      # off by default


def _list(tmp_path, places):
    p = tmp_path / "list.json"
    p.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "places": places}),
                 encoding="utf-8")
    return Settings(topic_list_file=str(p), place_country_gap_days=3, place_country_week_max=2)


def test_the_list_skips_a_country_made_too_recently_without_claiming_it(tmp_path):
    s = _list(tmp_path, [{"rank": 1, "place": "Hampi", "country": "India", "wikipedia_title": "Hampi"},
                         {"rank": 2, "place": "Shirakawa-go", "country": "Japan", "wikipedia_title": "Shirakawa-go"}])
    h = History(tmp_path / "h.db")
    h.add_video(status="uploaded", title="x", country="India", created_at=_ago(1))
    assert T.from_list(s, h).place == "Shirakawa-go"                        # India waits
    keys = [r["key"] for r in h._all("SELECT key FROM topics_used")]
    assert not any("hampi" in k for k in keys)                              # Hampi stays on the list


# ------------------------------------------------------------------ 3. concrete stakes
@pytest.mark.parametrize("text", [
    "What secret kept the sky-bridge hidden?", "Spiti's secret 10th-century shrine",
    "Why does Gokarna's Shiva temple stand untouched for centuries", "Echoes today", "Vanished overnight?",
    "Why is a Sri Lankan plain called Horton?"])
def test_vague_hooks_are_caught(text):
    assert vague_hook_reason(text)


@pytest.mark.parametrize("text", [
    "One of history's richest cities vanished in months", "A 1,600-year-old temple survived three invasions",
    "The village that survived 400 winters", "How did 300 masons carve this in 12 years?"])
def test_concrete_hooks_pass(text):
    assert vague_hook_reason(text) == ""


def _story(title="What secret kept the stone city hidden?", hook="A HIDDEN MYSTERY",
           opening="What secret did this city hide?", style="history"):
    return Script(topic="Hampi", title=title, hook_text=hook, description="d", style=style,
                  scenes=[Scene(opening, "Hampi"), Scene("In 1565 an army sacked the capital in months.", "Hampi")])


class _HookLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete_json(self, system, user, schema=None, **k):
        self.calls += 1
        assert "CONCRETE STAKES" in user and "1565" in user                 # rewritten from the script's facts
        return self.reply


def test_a_vague_hook_is_rewritten_once_with_concrete_stakes():
    llm = _HookLLM({"title": "One of history's richest cities fell in months", "hook_text": "SACKED IN 1565",
                    "opening": "In 1565, one of the richest cities on Earth was sacked in months."})
    s = ensure_concrete_hook(llm, _story(), place_title=True)
    assert llm.calls == 1 and s.title.startswith("One of history's richest")
    assert s.hook_text == "SACKED IN 1565" and "1565" in s.scenes[0].narration


def test_a_rewrite_that_is_still_vague_keeps_the_original_and_logs(caplog):
    llm = _HookLLM({"title": "The hidden mystery", "hook_text": "SECRET CITY", "opening": "What secret?"})
    with caplog.at_level("WARNING", logger="purffle"):
        s = ensure_concrete_hook(llm, _story())
    assert llm.calls == 1 and s.title == "What secret kept the stone city hidden?"
    assert any("still vague" in r.getMessage() for r in caplog.records)


def test_concrete_hooks_and_non_story_formats_are_left_alone():
    llm = _HookLLM({})
    concrete = _story(title="One of the richest cities fell in 1565", hook="SACKED IN 1565",
                      opening="In 1565 one of the richest cities on Earth fell in months.")
    assert ensure_concrete_hook(llm, concrete) is concrete and llm.calls == 0
    ensure_concrete_hook(llm, _story(style="facts"))
    assert llm.calls == 0


# ------------------------------------------------------------------ 5. no recent tragedies
@pytest.mark.parametrize("line", [
    "In 2015 an earthquake killed nearly 9,000 people across the valley.",
    "Protesters were shot here in 1989.", "Riots broke out in the town in 2002.",
    "The 1971 war left the fort in ruins.", "Recently, militants attacked the shrine."])
def test_stories_of_recent_deaths_protests_or_political_violence_are_skipped(line):
    s = _story(title="One of the richest cities fell in 1565", opening=line)
    assert recent_violence_reason(s)


@pytest.mark.parametrize("line", [
    "In 1565 an army sacked the capital in months.", "The siege of 1303 lasted eight months.",
    "In 1992 UNESCO protected the valley.", "Today pilgrims climb the 600 steps at dawn."])
def test_old_history_and_peaceful_modern_facts_are_fine(line):
    assert recent_violence_reason(_story(title="One of the richest cities fell", opening=line)) == ""


# ------------------------------------------------------------------ 4. hard 45-60 s gate on the real voice-over
def _studio(tmp_path, durations, fit):
    st = P.Studio.__new__(P.Studio)
    st.s, st._llm = Settings(), object()
    spoken = iter(durations)
    st._speak = lambda script, work, s: SimpleNamespace(duration=next(spoken))
    return st


@pytest.mark.parametrize("first, after, ok", [
    (52.0, None, True),          # inside the window: no rewrite
    (40.0, 50.0, True),          # too short -> rewritten once -> fits
    (70.0, 55.0, True),          # too long  -> trimmed once  -> fits
    (40.0, 42.0, False),         # still short after the one rewrite -> skip
    (70.0, 66.0, False),         # still long -> skip
])
def test_length_gate(monkeypatch, tmp_path, first, after, ok):
    calls = []

    def fit(llm, script, seconds, lo, hi):
        calls.append(seconds)
        return script

    monkeypatch.setattr(P, "fit_story_length", fit)
    st = _studio(tmp_path, [after] if after is not None else [], fit)
    script = _story()
    total = round(first + P.END_PAD, 3)
    if ok:
        _s, _sp, out = st._length_gate(script, SimpleNamespace(duration=first), total, tmp_path, tmp_path, st.s)
        assert 45 <= out <= 60
    else:
        with pytest.raises(SkipTopic, match="outside 45-60s after one rewrite"):
            st._length_gate(script, SimpleNamespace(duration=first), total, tmp_path, tmp_path, st.s)
    assert len(calls) == (0 if 45 <= total <= 60 else 1)                    # at most one rewrite


def test_length_gate_skips_when_the_rewrite_itself_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "fit_story_length", lambda *a, **k: None)
    st = _studio(tmp_path, [], None)
    with pytest.raises(SkipTopic):
        st._length_gate(_story(), SimpleNamespace(duration=38.0), 38.6, tmp_path, tmp_path, st.s)


# ------------------------------------------------------------------ 6. period-only AI scenes
@pytest.mark.parametrize("prompt, kept, gone", [
    ("Monks carving frescoes by oil lamp, holding a scroll, in a mud-brick hall", "frescoes", "scroll"),
    ("Builders in cotton dhotis raising granite pillars beside a royal banner", "granite pillars", "banner"),
    ("Traders unloading spice sacks from wooden boats at the river ghat", "wooden boats", None),
])
def test_ai_prompt_keeps_only_period_appropriate_things(prompt, kept, gone):
    out = M.period_prompt(prompt)
    assert kept in out and (gone is None or gone not in out.lower())


@pytest.mark.parametrize("prompt", ["A UNESCO team inspects the pillars with maps",
                                    "A governor signing documents with a flag behind him",
                                    "A villager rides a bicycle past modern concrete houses"])
def test_a_scene_that_is_about_modern_things_is_not_drawn(prompt):
    assert M.period_prompt(prompt) == ""


def test_the_image_provider_never_sees_a_map_or_flag(monkeypatch, tmp_path):
    from PIL import Image
    seen = []

    def fake_poll(prompt, dest, width, height, key=""):
        seen.append(prompt)
        Image.new("RGB", (64, 64)).save(dest)
        return dest

    monkeypatch.setattr(M, "pollinations_image", fake_poll)
    v = M.Visuals(Settings(ai_scene_source="pollinations"))
    v._historical_image("Masons raising a stone gateway, a surveyor holding a map", tmp_path, 0, arch="stone fort")
    assert seen and "map" not in seen[0].lower() and "Masons raising a stone gateway" in seen[0]
    assert "hand-made things of that time" in seen[0]                       # period items named positively
    assert v._historical_image("A clerk reading a newspaper", tmp_path, 1, arch="stone fort") is None
    assert len(seen) == 1                                                   # nothing drawable -> real photo
