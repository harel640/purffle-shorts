"""Nov 17: four of six attempts since the curiosity-headline rules were skipped because the title or hook could
not be supported, and each used up its place. Now: several candidates are checked, a safe headline is built from
the best-supported sentence, a headline alone never skips a video, and a headline-only failure gives the place
back. The claim check itself is unchanged: every headline that is used has been judged supported."""

import json

import pytest

from purffle_shorts import factcheck as F
from purffle_shorts import pipeline as P
from purffle_shorts.media import SkipTopic
from purffle_shorts.script import Scene, Script

SOURCE = ("Nisa was an ancient settlement of the Parthian Empire. An earthquake in the first decade BC destroyed "
          "the city. Mithridates I renamed the fortress Mithradatkert. Soldiers guarded its royal treasury.")
QUAKE = "An earthquake in the first decade BC destroyed the city"
GUARD = "Soldiers guarded its royal treasury"


class FakeLLM:
    """The checker supports an item when a key of ``table`` is in its text; ``replies`` are the answers to the
    headline calls, in order."""

    def __init__(self, table, replies):
        self.table, self.replies = table, list(replies)
        self.headline_prompts, self.batches = [], []

    def complete_json(self, system, user, schema=None, **k):
        if schema is F.HEADLINE_SCHEMA:
            self.headline_prompts.append(user)
            return self.replies.pop(0) if self.replies else {"titles": [], "hooks": [], "openings": []}
        items = json.loads(user.rsplit("ITEMS:\n", 1)[1])
        self.batches.append(items)
        out = []
        for it in items:
            quote = next((q for key, q in self.table.items() if key in it["text"]), "")
            out.append({"id": it["id"], "verdict": "supported" if quote else "unsupported", "quote": quote,
                        "unsupported_part": "" if quote else it["text"]})
        return {"items": out}


def _script(title="How did a market vanish underground?", hook="Market swallowed whole?"):
    return Script(topic="Nisa", title=title, hook_text=hook, description="", style="history",
                  scenes=[Scene("An earthquake in the first decade BC destroyed the city.", "q"),
                          Scene("Soldiers guarded its royal treasury.", "q")])


TABLE = {"earthquake in the first decade BC destroyed": QUAKE, "Soldiers guarded": GUARD}


def _opts(titles=(), hooks=()):
    return {"titles": list(titles), "hooks": list(hooks), "openings": []}


def test_four_candidates_are_checked_together_and_the_first_supported_rule_passing_one_wins():
    table = {**TABLE, "What destroyed the Parthian city": QUAKE, "Who guarded the royal treasury": GUARD,
             "Soldiers guarded": GUARD, "earthquake destroyed": QUAKE, "Who guarded": GUARD}
    llm = FakeLLM(table, [_opts(
        titles=["Did a flood bury Nisa?", "What destroyed the Parthian city?",
                "Who guarded the royal treasury?", "A tale of Nisa"],
        hooks=["Flood or fire?", "An earthquake destroyed the city", "Who guarded the gold?", "Gone"])])
    s, report = F.verify_claims(llm, _script(), SOURCE, place_words="Nisa")
    assert s.title == "What destroyed the Parthian city?"           # candidate 2: the first supported one
    # a dramatic STATEMENT is a headline too, and the event outranks the question (week-3 analytics)
    assert s.hook_text == "An earthquake destroyed the city"
    assert len(llm.headline_prompts) == 1
    assert "4 DIFFERENT titles" in llm.headline_prompts[0] and "4 DIFFERENT on-screen hooks" in llm.headline_prompts[0]
    assert len(llm.batches[1]) == 8                                  # all 8 candidates judged in ONE batch
    used = [r for r in report["items"] if r.get("rewritten")]
    assert {r["kind"] for r in used} == {"title", "hook"} and all(r["verdict"] == "supported" for r in used)


def test_a_writer_headline_that_is_supported_needs_no_candidates():
    table = {**TABLE, "How did a market": QUAKE, "Market swallowed": QUAKE}
    llm = FakeLLM(table, [])
    s, _ = F.verify_claims(llm, _script(), SOURCE, place_words="Nisa")
    assert s.title == "How did a market vanish underground?" and llm.headline_prompts == []


def test_when_no_candidate_is_supported_a_safe_headline_is_built_on_the_best_sentence():
    table = {**TABLE, "What did soldiers guard": GUARD}
    llm = FakeLLM(table, [_opts(titles=["Did a flood bury Nisa?"] * 4, hooks=["Flood or fire?"] * 4),
                          _opts(titles=["What did soldiers guard at Nisa?"], hooks=["What did soldiers guard?"])])
    s, _ = F.verify_claims(llm, _script(), SOURCE, place_words="Nisa")
    assert s.title == "What did soldiers guard at Nisa?" and s.hook_text == "What did soldiers guard?"
    assert len(llm.headline_prompts) == 2
    safe = llm.headline_prompts[1]
    assert "THIS ONE QUOTE" in safe and safe.split("QUOTES from the source:")[1].count("- ") == 1   # one quote


def test_if_even_the_safe_headline_fails_the_title_is_the_best_supported_sentence_and_no_skip():
    llm = FakeLLM(TABLE, [_opts(titles=["Did a flood bury Nisa?"]), _opts(titles=["Did fire bury Nisa?"])])
    s, report = F.verify_claims(llm, _script(), SOURCE, place_words="Nisa")
    assert s.title == "An earthquake in the first decade BC destroyed the city"   # supported narration only
    assert s.hook_text == ""                                          # no on-screen hook rather than an unsupported one
    assert len(s.scenes) == 2


def test_the_last_resort_title_has_no_year():
    table = {"earthquake in 1950 destroyed": "An earthquake in 1950 destroyed the city"}
    s = Script(topic="Nisa", title="How did a market vanish?", hook_text="", description="", style="history",
               scenes=[Scene("An earthquake in 1950 destroyed the city.", "q")])
    out, _ = F.verify_claims(FakeLLM(table, []), s, "An earthquake in 1950 destroyed the city.", place_words="Nisa")
    assert out.title == "An earthquake destroyed the city"


def test_unsupported_narration_still_skips_the_story():
    with pytest.raises(F.NothingSupported, match="no sentence of the narration") as e:
        F.verify_claims(FakeLLM({}, []), _script(), SOURCE, place_words="Nisa")
    assert not e.value.headline and P.failure_kind(SkipTopic("x")) == "rule"


# ------------------------------------------------------------------ a headline-only failure gives the place back
def test_a_headline_only_failure_gives_the_place_back():
    cause = F.NothingSupported("the title could not be supported by the source", headline=True)
    try:
        raise SkipTopic(f"story not supported by its source: {cause}") from cause
    except SkipTopic as e:
        wrapped = e
    assert P.headline_only(wrapped) and P.failure_kind(wrapped) == "headline"
    assert P.failure_kind(F.NothingSupported("no sentence of the narration is supported")) == "rule"

    class History:
        released, finished = [], []

        def topic_used(self, key):
            return True

        def release_topic(self, key):
            self.released.append(key)

        def finish_topic(self, key):
            self.finished.append(key)

    class Pick:
        key, source, place, subject = "list:nisa", "list", "Nisa", "Nisa"

    maker = P.Studio.__new__(P.Studio)
    maker.history = History()
    assert maker._end_run(Pick(), wrapped) == "headline"
    assert maker.history.released == ["list:nisa"] and maker.history.finished == []
