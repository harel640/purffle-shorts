import json
from pathlib import Path

import pytest

from purffle_shorts.config import Settings
from purffle_shorts.llm import extract_json
from purffle_shorts.script import (
    SCRIPT_SCHEMA,
    STYLES,
    build_prompt,
    effective_seconds,
    load_script,
    normalize,
    offline_script,
    scene_plan,
)


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here it is:\n{"a": 1}\nHope that helps.',
    '{"a": 1,}',
])
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def test_normalize_cleans_everything():
    data = {
        "topic": "Octopus", "title": '"Octopus Secrets" #shorts 🐙', "hook_text": "3 HEARTS?!",
        "scenes": [
            {"narration": "[Hook] Narrator: Octopuses have three hearts! 🐙 #facts", "search_query": "octopus",
             "image_prompt": "an octopus"},
            {"narration": "", "search_query": "x", "image_prompt": "x"},
            "Plain string scene works too.",
        ],
        "description": "Wow. #octopus", "hashtags": ["octopus", "#Ocean Life", "#shorts", "#octopus"],
        "tags": ["a" * 70] + [f"tag{i}" for i in range(200)], "category": "nonsense",
    }
    s = normalize(data, "octopus", "facts", "en")
    assert s.scenes[0].narration == "Octopuses have three hearts!"
    assert len(s.scenes) == 2
    assert s.title == "Octopus Secrets"
    assert s.hashtags[0] == "#shorts" and "#OceanLife" in s.hashtags
    assert len([h for h in s.hashtags if h.lower() == "#octopus"]) == 1
    assert sum(len(t) + 1 for t in s.tags) <= 500
    assert s.category == "education" and s.category_id == "27"
    assert "#" not in s.description


def test_normalize_merges_too_many_scenes():
    data = {"scenes": [{"narration": f"Sentence {i}.", "search_query": "q", "image_prompt": "p"} for i in range(20)]}
    s = normalize(data, "x", "facts", "en")
    assert len(s.scenes) == 12
    assert s.narration.count("Sentence") == 20


def test_offline_script_is_complete():
    s = offline_script("", Settings())
    assert s.title and len(s.scenes) >= 4 and s.hashtags[0] == "#shorts"


def test_prompt_mentions_language_and_length():
    system, user = build_prompt("space", "facts", Settings(language="es", target_seconds=30), ["Old title"])
    assert "Spanish" in user and "78 words" in user and "Old title" in user
    assert "JSON" in system
    assert set(SCRIPT_SCHEMA["required"]) >= {"title", "scenes", "hashtags"}


def test_history_style_is_story_driven_and_longer():
    assert "history" in STYLES
    # The story format targets ~50s (45-60s window) even when the global target is tuned shorter.
    words, n = scene_plan("history", 40)
    assert words >= int(45 * 2.6) and 6 <= n <= 12
    assert effective_seconds("history", 40) == 50 and effective_seconds("history", 90) == 90
    assert effective_seconds("facts", 40) == 40


def test_history_prompt_asks_for_mystery_and_hybrid_visuals():
    _system, user = build_prompt("Hampi", "history", Settings(), [])
    low = user.lower()
    assert "mystery question" in low and "turning point" in low   # story structure
    assert "historical" in low and "today" in low                 # hybrid visual routing explained
    assert "call to action" not in low                            # calm payoff, no CTA push
    assert "historical" in SCRIPT_SCHEMA["properties"]["scenes"]["items"]["required"]


def test_history_enforces_length_and_strips_cta():
    from purffle_shorts.script import lint_script
    # A "like/share/follow" CTA on the payoff is stripped, keeping the reflective line.
    data = {"scenes": [
        {"narration": "Why did the city fall?", "search_query": "x", "image_prompt": "p", "historical": False},
        {"narration": "It thrived for centuries. Like, share, and follow for more.", "search_query": "x",
         "image_prompt": "p", "historical": True},
    ]}
    s = normalize(data, "City", "history", "en")
    last = s.scenes[-1].narration.lower()
    assert "follow" not in last and "like" not in last and "share" not in last and "thrived" in last
    # A CTA joined to the payoff by a semicolon (not a new sentence) is also stripped.
    data2 = {"scenes": [
        {"narration": "Why did it vanish?", "search_query": "x", "image_prompt": "p"},
        {"narration": "The empire glittered and fell; subscribe for more hidden histories.",
         "search_query": "x", "image_prompt": "p", "historical": True}]}
    s2 = normalize(data2, "City", "history", "en")
    assert "subscribe" not in s2.scenes[-1].narration.lower() and "glittered and fell" in s2.scenes[-1].narration
    # A too-short history script is flagged so the Script Doctor expands it into the 45-60s range.
    short = normalize({"scenes": [{"narration": "A short tale about a city.", "search_query": "x",
                                   "image_prompt": "p"}]}, "City", "history", "en")
    assert any("45-60" in f or "expand" in f for f in lint_script(short, Settings()))


def test_title_is_capped_under_60_chars():
    long_q = "Did a Portuguese explorer witness a city comparable to the great capitals of its time?"
    s = normalize({"title": long_q, "scenes": [{"narration": long_q, "search_query": "hampi",
                                                 "image_prompt": "p"}]}, "Hampi", "history", "en")
    assert len(s.title) <= 60


def test_history_scriptwriting_gets_more_token_headroom():
    from purffle_shorts.script import write_script
    seen = {}

    class LLM:
        def complete_json(self, system, user, schema=None, max_tokens=4000, temperature=None):
            seen["mt"] = max_tokens
            return {"topic": "Hampi", "title": "Why?", "hook_text": "",
                    "scenes": [{"narration": "Why did it fall?", "search_query": "hampi", "image_prompt": "p",
                                "historical": False}],
                    "description": "d", "hashtags": [], "tags": [], "category": "education"}

    write_script(LLM(), "Hampi", Settings(), style="history")
    assert seen["mt"] == 12000                       # a long, grounded story needs room to finish the JSON
    write_script(LLM(), "octopus", Settings(), style="facts")
    assert seen["mt"] == 4000


def test_normalize_keeps_historical_flag():
    data = {"scenes": [
        {"narration": "Why was this city abandoned?", "search_query": "Hampi", "image_prompt": "p",
         "historical": False},
        {"narration": "A king once ruled here.", "search_query": "Hampi", "image_prompt": "a king on a throne",
         "historical": True},
    ]}
    s = normalize(data, "Hampi", "history", "en")
    assert s.scenes[0].historical is False and s.scenes[1].historical is True


def test_load_script_cleans_like_an_ai_script(tmp_path):
    f = tmp_path / "mine.json"
    f.write_text(json.dumps({"title": "My Own Script #viral", "style": "not-a-style", "hashtags": ["cats"],
                             "scenes": [{"narration": "Cats sleep a lot.", "search_query": "sleeping cat"}]}))
    sc = load_script(f, Settings())
    assert sc.title == "My Own Script" and sc.style == "facts"
    assert sc.hashtags[:2] == ["#shorts", "#cats"] and sc.topic == "My Own Script #viral"
    assert sc.scenes[0].image_prompt == "sleeping cat"


def test_load_script_rejects_non_objects(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_script(f, Settings())


@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "examples").glob("*.json")))
def test_shipped_examples_load(path):
    sc = load_script(path, Settings())
    assert len(sc.scenes) >= 4 and sc.hook_text and all(s.image_prompt for s in sc.scenes)


def test_narration_cleanup_turns_dashes_into_clean_pauses():
    from purffle_shorts.utils import clean_narration
    assert clean_narration("packed with copper — not iron.") == "packed with copper, not iron."
    assert clean_narration("blue–green , really ?") == "blue, green, really?"
    assert clean_narration("It ends here —.") == "It ends here."
    assert clean_narration("[Hook] Wow! #facts 🐙") == "Wow!"
