"""Nov 21 19:00: three places skipped. Kellie's Castle for a natural death read as violence; Bujang Valley and
Fort Zeelandia because the vision check rejected real photos (event subjects such as "Koxinga siege Fort
Zeelandia", and a question too strict for obscure ruins) - and each skip used its place up."""

import pytest

from purffle_shorts import pipeline as P
from purffle_shorts.history import History
from purffle_shorts.llm import LLM
from purffle_shorts.media import SkipTopic, event_scene, visible_subject
from purffle_shorts.script import Scene, Script, natural_death, recent_violence_reason


def _script(*lines):
    return Script(topic="t", title="t", hook_text="h", description="d", style="history",
                  scenes=[Scene(x, "q") for x in lines])


# ------------------------------------------------------------------ 1. natural deaths are not violence
KELLIE = "Kellie‑Smith died of pneumonia during a short trip to Lisbon in 1926."


@pytest.mark.parametrize("sentence", [
    KELLIE, "The sultan died of old age in 1931.", "She died peacefully in 1950.",
    "He died of natural causes in 1904.", "The owner died after a long illness in 1912."])
def test_a_natural_death_after_1900_is_not_violence(sentence):
    assert natural_death(sentence) and recent_violence_reason(_script("The castle was never finished.", sentence)) == ""


@pytest.mark.parametrize("sentence", [
    "He was killed in the 1926 riot.", "Soldiers executed the rebels in 1942.",
    "Thousands died of cholera in 1920.", "He died in the war in 1944.", "Protesters died in 1989.",
    "The 1934 earthquake killed hundreds.", "He died when troops opened fire in 1960."])
def test_violence_war_executions_and_disasters_after_1900_still_block(sentence):
    assert not natural_death(sentence)
    assert recent_violence_reason(_script(sentence)) != ""


# ------------------------------------------------------------------ 2. event subjects are checked as the place
@pytest.mark.parametrize("query", ["Koxinga siege Fort Zeelandia", "Fort Zeelandia water shortage",
                                   "Dutch arrival Tainan", "battle at the fort walls", "Mongol invasion"])
def test_an_event_scene_is_checked_against_the_place_itself(query):
    assert event_scene(query) and visible_subject(query, "Fort Zeelandia", []) == "Fort Zeelandia"


def test_a_plain_scene_keeps_its_own_subject():
    assert not event_scene("Fort Zeelandia walls")
    assert visible_subject("worship hall interior", "Fort Zeelandia", []) == "worship hall interior"


def test_pre_1900_event_scenes_become_ai_scenes_within_the_caps():
    qs = ["Fort Zeelandia gate", "Koxinga siege Fort Zeelandia", "Fort Zeelandia water shortage",
          "Fort Zeelandia museum today", "Dutch surrender Tainan", "Fort Zeelandia walls"]
    scenes = [Scene("x", q) for q in qs]
    modern = [False, False, False, True, False, False]
    assert P.ai_scene_flags(scenes, modern, "history") == [False, True, True, False, True, False]
    assert P.ai_scene_flags(scenes, modern, "place") == [False] * 6           # present-day format: photos only
    assert P.ai_scene_flags(scenes, [True] * 6, "history") == [False] * 6     # a modern event is a photo


# ------------------------------------------------------------------ 3. photo skips give the place back once
def test_a_photo_skip_gives_the_place_back_once(tmp_path):
    st = P.Studio.__new__(P.Studio)
    st.history = History(tmp_path / "h.db")
    pick = type("Pick", (), {"key": "list:2026-11-16:bujang valley|malaysia", "source": "list",
                             "place": "Bujang Valley", "subject": "Bujang Valley"})()
    skip = SkipTopic("no vision-verified photo left for scene 8 of Bujang Valley", photos=True)
    assert P.failure_kind(skip) == "photos"
    st.history.claim_topic(pick.key)
    st._settle_claim(pick, skip)
    st._settle_claim(pick, skip)                       # settled twice for one skip: still given back once
    assert not st.history.topic_used(pick.key)
    st.history.claim_topic(pick.key)
    st._settle_claim(pick, skip)
    assert st.history.topic_used(pick.key)             # the second photo skip uses it up


def test_photo_and_short_skips_are_counted_apart(tmp_path):
    st = P.Studio.__new__(P.Studio)
    st.history = History(tmp_path / "h.db")
    pick = type("Pick", (), {"key": "list:w:x|y", "source": "list", "place": "X", "subject": "X"})()
    for skip in (SkipTopic("short", short=True), SkipTopic("photos", photos=True)):
        st.history.claim_topic(pick.key)
        st._settle_claim(pick, skip)
        assert not st.history.topic_used(pick.key)


def test_the_pool_exhaustion_skip_is_a_photo_skip():
    import inspect

    from purffle_shorts import media
    src = inspect.getsource(media.Visuals.for_place_scenes)
    assert "photos=True" in src.split("no vision-verified photo left for scene")[1][:400]


# ------------------------------------------------------------------ 4. the calibrated question
def test_the_vision_question_is_the_calibrated_plausibility_question():
    q = LLM._vision_question(None, "Bujang Valley")
    assert q.startswith("Could this plausibly be a real photograph of Bujang Valley")
    assert "old black-and-white" in q and "ruin" in q and "engraving" in q
