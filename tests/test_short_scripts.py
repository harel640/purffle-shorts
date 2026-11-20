"""Nov 20: the claim check cut a story to 35.8 s and the video was skipped only AFTER it was voiced (edge-tts, up
to 3 x 60 s). Now the length is checked before the voice: one rewrite with more of the SAME event from the
article, a second claim check, and a skip (the place given back ONCE) only if it is still clearly short."""

import pytest

from purffle_shorts import pipeline as P
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.media import SkipTopic
from purffle_shorts.script import HISTORY_RANGE, STORY_VOICE_WPS, Scene, Script


def _script(words: int) -> Script:
    return Script(topic="t", title="How did it fall?", hook_text="h", description="d", style="history",
                  scenes=[Scene(" ".join(["word"] * words) + ".", "q")])


def _maker(monkeypatch, expanded_words=None, claim_words=None, tmp_path=None):
    """A Studio whose length rewrite returns a script of ``expanded_words`` and whose claim check keeps
    ``claim_words`` of them."""
    st = P.Studio.__new__(P.Studio)
    st._llm = object()
    st.s = Settings()
    calls = {"fit": 0, "claim": 0}

    def fit(llm, script, seconds, lo, hi, source=""):
        calls["fit"] += 1
        calls["seconds"], calls["source"] = seconds, source
        return _script(expanded_words) if expanded_words else None

    monkeypatch.setattr(P, "fit_story_length", fit)
    monkeypatch.setattr(P.factcheck, "ground", lambda script, source, s: script)
    monkeypatch.setattr(P.factcheck, "story_source", lambda text, n: text)

    def claim(self, script, pick, **k):
        calls["claim"] += 1
        calls["kinds"] = k.get("kinds")
        return _script(claim_words) if claim_words else script

    monkeypatch.setattr(P.Studio, "_claim_check", claim)
    return st, calls


class Pick:
    source_full, source_text, claim_reports = "ARTICLE", "", []


def test_the_estimate_is_words_at_the_narrator_pace():
    assert P.Studio.estimate_seconds(_script(115)) == pytest.approx(115 / STORY_VOICE_WPS + P.END_PAD)


def test_a_long_enough_script_is_left_alone(monkeypatch):
    st, calls = _maker(monkeypatch)
    s = _script(120)
    assert st._length_before_voice(s, Pick(), None) is s and calls["fit"] == 0


def test_a_short_script_is_rewritten_once_checked_again_and_then_voiced(monkeypatch):
    st, calls = _maker(monkeypatch, expanded_words=125, claim_words=120)
    out = st._length_before_voice(_script(60), Pick(), None)
    assert calls["fit"] == 1 and calls["claim"] == 1 and calls["kinds"] == ("narration",)
    assert calls["source"] == "ARTICLE"                                   # only what the article states
    assert len(out.narration.split()) == 120


def test_still_clearly_short_after_the_rewrite_is_a_short_skip(monkeypatch):
    st, _ = _maker(monkeypatch, expanded_words=70, claim_words=60)
    with pytest.raises(SkipTopic, match="under 45 s after one rewrite") as e:
        st._length_before_voice(_script(60), Pick(), None)
    assert e.value.short and P.failure_kind(e.value) == "short"


def test_a_script_just_under_the_window_goes_on_to_the_measured_gate(monkeypatch):
    lo = HISTORY_RANGE[0]
    words = int((lo - 2 - P.END_PAD) * STORY_VOICE_WPS)               # an estimate 2 s under: within the margin
    st, _ = _maker(monkeypatch, expanded_words=None)
    assert st._length_before_voice(_script(words), Pick(), None).narration.count("word") == words


def test_a_failed_rewrite_still_skips_a_script_that_is_far_too_short(monkeypatch):
    st, calls = _maker(monkeypatch, expanded_words=None)
    with pytest.raises(SkipTopic) as e:
        st._length_before_voice(_script(60), Pick(), None)
    assert e.value.short and calls["claim"] == 0


# ------------------------------------------------------------------ the place is given back once
class _Pick:
    key, source, place, subject = "list:2026-11-16:gwalior fort|india", "list", "Gwalior Fort", "Gwalior Fort"


def _short():
    return SkipTopic("script would run about 36s, under 45 s after one rewrite", short=True)


def test_the_first_short_skip_gives_the_place_back_and_the_second_uses_it_up(tmp_path):
    st = P.Studio.__new__(P.Studio)
    st.history = History(tmp_path / "h.db")
    pick = _Pick()
    st.history.claim_topic(pick.key)
    st._settle_claim(pick, _short())
    st._settle_claim(pick, _short())          # draft() and make() both settle the same skip: still given back once
    assert not st.history.topic_used(pick.key)
    st.history.claim_topic(pick.key)          # a later attempt takes it again ... and is short again
    st._settle_claim(pick, _short())
    st._settle_claim(pick, _short())
    assert st.history.topic_used(pick.key)                                # now it stays used
    assert P.failure_kind(_short()) == "short" and P.failure_kind(SkipTopic("x")) == "rule"


def test_a_short_skip_of_another_week_is_a_fresh_start(tmp_path):
    st = P.Studio.__new__(P.Studio)
    st.history = History(tmp_path / "h.db")
    for key in ("list:2026-11-16:a|x", "list:2026-11-23:a|x"):
        pick = type("P", (), {"key": key, "source": "list", "place": "A", "subject": "A"})()
        st.history.claim_topic(key)
        st._settle_claim(pick, _short())
        assert not st.history.topic_used(key)


def test_the_measured_gate_marks_a_short_video_but_not_a_long_one():
    assert SkipTopic("x", short=True).short and not SkipTopic("x").short
