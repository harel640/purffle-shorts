"""Claim-level fact check (factcheck.verify_claims): every sentence of a story, its title and its hook must be
supported by a passage of the source article. The checker LLM proposes verdicts; the code verifies the checker.

The cases are the ones that reached the channel: an invented storm night at Kampong Ayer, a chariot "carved
from a single block" at Hampi, and rewrites (hook, length) that ran after the old check and were never looked at.
"""

import json
from types import SimpleNamespace

import pytest

from purffle_shorts import factcheck
from purffle_shorts import pipeline as P
from purffle_shorts import topics as T
from purffle_shorts.config import Settings
from purffle_shorts.media import SkipTopic
from purffle_shorts.script import Scene, Script

# Excerpt of https://en.wikipedia.org/wiki/Kampong_Ayer (CC BY-SA 4.0), as fetched on 2026-11-11.
KAMPONG_AYER = (
    "Kampung Ayer is a prominent traditional settlement in Bandar Seri Begawan, the capital of Brunei. It "
    "comprises neighbourhoods of traditional houses, schools and mosques built on stilts above the Brunei River "
    "near the capital's city centre. It has an area of about 10 square kilometres (3.9 sq mi); the total "
    "population was 10,250 in 2016. It has been historically nicknamed 'Venice of the East'.\n\n"
    "History\n"
    "Kampong Ayer is believed to have been inhabited for several centuries. There are several historical "
    "records, particularly foreign sources, which reported the existence of 'water settlements' on the Brunei "
    "River. The most well known is arguably the account by Antonio Pigafetta, an Italian explorer, on his visit "
    "to Brunei as part of the Magellan expedition in 1521.\n\n"
    "Entirely built in salt water... It contains twenty-five thousand hearths (families). The houses are all "
    "constructed of wood and built up from the ground on tall pillars.\n"
    "There is a possibility that the stilt settlement might not have always been where it is today, that "
    "Kampong Ayer might have undergone relocation throughout history. Olivier van Noort, a Dutch, on his stay "
    "in Brunei from December 1600 to January 1601, describing the houses (of the nobles) as being \"made of "
    "wood, and built on such light piles that when there is a storm or some other untoward event these houses "
    "can be removed from one side of the river to the other.\"\n")

STILTS = "built on stilts above the Brunei River"
NOORT = "built on such light piles that when there is a storm or some other untoward event these houses can be removed"

# The four sentences that were uploaded (video __KWOS053gE). None of them is in the article.
INVENTED = [
    "In the early 17th century, a Dutch explorer reported seeing thousands of stilt homes being affected by storms.",
    "Those poles were not fixed; when monsoon winds rose, villagers could slide the entire structure to safer water.",
    "One night a sudden storm slammed the river, and many homes were lifted, drifting downstream like wooden rafts.",
    "Even now a whole village can be shifted by the tide.",
]
TRUE_NOORT = ("In 1600 the Dutch visitor Olivier van Noort wrote that the houses stood on piles so light that "
              "in a storm they could be moved across the river.")


class Checker:
    """A fake checker LLM. ``table`` maps a piece of an item's text to (verdict, quote); anything else is
    'unsupported'. ``headline`` is the reply to the title/hook rewrite. ``fail`` makes every call raise."""

    def __init__(self, table=None, headline=None, fail=False, skip_ids=()):
        self.table, self.headline, self.fail, self.skip_ids = table or {}, headline, fail, set(skip_ids)
        self.prompts, self.rewrites = [], 0

    def complete_json(self, system, user, schema=None, max_tokens=0, temperature=None, effort=None):
        if self.fail:
            raise RuntimeError("All LLM providers failed")
        assert effort == "low" and max_tokens <= 3000          # must fit Groq's 8,000 tokens a minute
        if schema is factcheck.HEADLINE_SCHEMA:
            self.rewrites += 1
            self.prompts.append(user)
            return self.headline
        assert schema is factcheck.CLAIMS_SCHEMA and temperature == 0.0
        self.prompts.append(user)
        out = []
        for it in json.loads(user.rsplit("ITEMS:\n", 1)[1]):
            if it["id"] in self.skip_ids:
                continue
            verdict, quote = next((v for k, v in self.table.items() if k in it["text"]), ("unsupported", ""))
            out.append({"id": it["id"], "verdict": verdict, "quote": quote,
                        "unsupported_part": it["text"] if verdict == "unsupported" else ""})
        return {"items": out}


def _script(sentences, title="Houses Built To Be Moved", hook="HOUSES THAT MOVE",
            description="Kampong Ayer, Brunei: a village on stilts above the river.", topic="Kampong Ayer"):
    return Script(topic=topic, title=title, hook_text=hook, description=description, style="history",
                  scenes=[Scene(s, "q") for s in sentences])


BASE = {  # what an honest checker says about the true parts of the Kampong Ayer story
    "Houses Built To Be Moved": ("supported", NOORT),
    "HOUSES THAT MOVE": ("supported", NOORT),
    "a village on stilts above the river": ("supported", STILTS),
    "stands on stilts": ("supported", STILTS),
    "Olivier van Noort wrote": ("supported", NOORT),
    "Would you live": ("not_factual", ""),
}


# ------------------------------------------------------------------ the uploaded fabrications
def test_the_invented_storm_is_dropped_and_the_real_remark_survives():
    table = dict(BASE)
    # a careless checker even "supports" the storm night with a quote it made up - the code must not believe it
    table["One night a sudden storm"] = ("supported", "one night a sudden storm lifted many homes downstream")
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River.", INVENTED[0], INVENTED[1],
                  TRUE_NOORT + " " + INVENTED[2], INVENTED[3] + " Would you live on the water?"])
    out, report = factcheck.verify_claims(Checker(table), sc, KAMPONG_AYER, place_words="Kampong Ayer, Brunei")
    told = out.narration
    for lie in INVENTED:
        assert lie not in told
    assert TRUE_NOORT in told and "stands on stilts" in told and told.endswith("Would you live on the water?")
    assert len(out.scenes) == 3                                   # two scenes held nothing but inventions
    assert report["counts"] == {"supported": 5, "unsupported": 4, "not_factual": 1}
    storm = next(r for r in report["items"] if "One night" in r["text"])
    assert storm["llm_verdict"] == "supported" and storm["verdict"] == "unsupported"
    assert "quote is not in the source" in storm["reason"]
    assert out.title == "Houses Built To Be Moved"                # supported: untouched


def test_single_block_is_dropped_unless_the_source_says_it():
    src = ("The Vitthala temple complex in Hampi contains the Stone Chariot, a shrine built in the form of a "
           "temple chariot. It stands in the courtyard of the temple and is dedicated to Garuda.")
    line = "Masons carved the chariot from a single block of granite."
    table = {"Stone Chariot stands": ("supported", "a shrine built in the form of a temple chariot"),
             "single block": ("supported", "carved from a single block of granite"),     # not in this source
             "Stone Chariot": ("supported", "contains the Stone Chariot"),
             "STONE CHARIOT": ("supported", "contains the Stone Chariot")}
    sc = _script(["The Stone Chariot stands in the temple courtyard.", line], title="Stone Chariot",
                 hook="STONE CHARIOT", description="", topic="Hampi")
    out, _ = factcheck.verify_claims(Checker(table), sc, src, place_words="Hampi, India")
    assert "single block" not in out.narration and len(out.scenes) == 1
    # the same sentence is kept when the article really says so
    sc = _script(["The Stone Chariot stands in the temple courtyard.", line], title="Stone Chariot",
                 hook="STONE CHARIOT", description="", topic="Hampi")
    out, _ = factcheck.verify_claims(Checker(table), sc, src + " It was carved from a single block of granite.",
                                     place_words="Hampi, India")
    assert "single block" in out.narration


# ------------------------------------------------------------------ the checker is not trusted
@pytest.mark.parametrize("sentence, said, quote, why", [
    ("Storms moved the houses across the river.", "supported", "storms moved the houses every year", "quote is not in"),
    ("The houses stood on light piles.", "supported", "light piles", "quote is not in"),            # too short to prove
    ("A storm carried the houses out to sea.", "supported",                                         # half invented
     "when there is a storm the houses were carried out to sea and lost", "quote is not in"),
    ("In 1750 the houses stood on light piles.", "supported", NOORT, "number not in the source: 1750"),
    ("The houses seen by Marco Polo stood on light piles.", "supported", NOORT, "name not in the source"),
    ("In 1750 everything changed.", "not_factual", "", "number not in the source"),
    ("The Portuguese fleet arrived at dawn.", "not_factual", "", "states names or numbers without a quote"),
    ("Would Marco Polo have recognised it?", "not_factual", "", "name not in the source"),
])
def test_a_verdict_the_source_does_not_back_is_overruled(sentence, said, quote, why):
    table = dict(BASE)
    table[sentence[:25]] = (said, quote)
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River.", sentence])
    out, report = factcheck.verify_claims(Checker(table), sc, KAMPONG_AYER, place_words="Kampong Ayer")
    assert sentence not in out.narration
    row = next(r for r in report["items"] if r["text"] == sentence)
    assert row["verdict"] == "unsupported" and why in row["reason"]


@pytest.mark.parametrize("sentence, said, quote", [
    (TRUE_NOORT, "supported", NOORT),                                             # a paraphrase with a real quote
    ("An Italian explorer, Pigafetta, described it in 1521.", "supported",
     "the account by Antonio Pigafetta, an Italian explorer, on his visit to Brunei"),
    ("The village stands on stilts.", "supported", "Built on stilts ... above the Brunei River"),  # case, ellipsis
    ("Records speak of water settlements on the river.", "supported",              # one word miscopied
     "which described the existence of 'water settlements' on the Brunei River"),
    ("Would Pigafetta recognise it today?", "not_factual", ""),                   # a question naming a real name
    ("Here is what happened next.", "not_factual", ""),
])
def test_what_the_source_backs_is_kept(sentence, said, quote):
    table = dict(BASE)
    table[sentence[:25]] = (said, quote)
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River.", sentence])
    out, report = factcheck.verify_claims(Checker(table), sc, KAMPONG_AYER, place_words="Kampong Ayer")
    assert sentence in out.narration
    row = next(r for r in report["items"] if r["text"] == sentence)
    assert row["verdict"] == said
    if said == "supported":
        assert 0 <= row["overlap"] <= 1                                           # logged, not enforced


def test_an_item_the_checker_skipped_counts_as_unsupported():
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River.", TRUE_NOORT])
    llm = Checker(dict(BASE), skip_ids={4})                    # ids: title 0, hook 1, description 2, then sentences
    out, report = factcheck.verify_claims(llm, sc, KAMPONG_AYER, place_words="Kampong Ayer")
    assert TRUE_NOORT not in out.narration
    assert next(r for r in report["items"] if r["id"] == 4)["reason"] == "the checker gave no verdict for it"


# ------------------------------------------------------------------ title, hook, description
def test_an_unsupported_title_is_rewritten_from_the_quotes_and_checked_again():
    table = dict(BASE)
    table["Houses built to move in a storm"] = ("supported", NOORT)
    llm = Checker(table, headline={"title": "Houses built to move in a storm", "hook_text": "HOUSES THAT MOVE"})
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River.", TRUE_NOORT],
                 title="A storm swept a village away")
    out, report = factcheck.verify_claims(llm, sc, KAMPONG_AYER, place_words="Kampong Ayer", title_len=40)
    assert out.title == "Houses built to move in a storm" and out.hook_text == "HOUSES THAT MOVE"
    assert llm.rewrites == 1
    assert NOORT in llm.prompts[1] and "storm swept" in llm.prompts[1]          # rewritten from the quotes only
    assert any(r.get("rewritten") and r["kind"] == "title" and r["verdict"] == "supported" for r in report["items"])


def test_a_title_that_cannot_be_supported_skips_the_story_or_is_kept_for_a_manual_topic():
    llm = Checker(dict(BASE), headline={"title": "A storm destroyed it all", "hook_text": "HOUSES THAT MOVE"})

    def story():
        return _script(["Kampong Ayer stands on stilts above the Brunei River.", TRUE_NOORT],
                       title="A storm swept a village away")
    with pytest.raises(factcheck.NothingSupported, match="title could not be supported"):
        factcheck.verify_claims(llm, story(), KAMPONG_AYER, place_words="Kampong Ayer")
    out, _ = factcheck.verify_claims(llm, story(), KAMPONG_AYER, place_words="Kampong Ayer", strict=False)
    assert out.title == "A storm swept a village away" and TRUE_NOORT in out.narration


def test_an_unsupported_description_is_cleared_and_a_story_with_nothing_left_is_refused():
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River."],
                 description="A storm once carried the whole village out to sea.")
    out, _ = factcheck.verify_claims(Checker(dict(BASE)), sc, KAMPONG_AYER, place_words="Kampong Ayer")
    assert out.description == ""                     # the pipeline then writes "The story of <place>, <country>."
    # the real checker once waved "Explore how the village survived a storm" through as "not factual"
    table = dict(BASE)
    table["survived a storm"] = ("not_factual", "")
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River."],
                 description="Explore how the village survived a storm that moved it.")
    out, report = factcheck.verify_claims(Checker(table), sc, KAMPONG_AYER, place_words="Kampong Ayer")
    assert out.description == "" and report["counts"]["unsupported"] == 1
    with pytest.raises(factcheck.NothingSupported, match="no sentence of the narration"):
        factcheck.verify_claims(Checker(dict(BASE)), _script(INVENTED), KAMPONG_AYER, place_words="Kampong Ayer")


# ------------------------------------------------------------------ what the checker is shown
def test_a_long_article_keeps_its_lead_and_the_passages_the_script_talks_about():
    filler = "\n".join(f"Paragraph {i} is about fishing boats, markets and the weather in the bay." * 6
                       for i in range(60))
    article = ("Lead paragraph about the water village.\n" + filler
               + "\nOlivier van Noort described houses on light piles that can be removed in a storm.\n" + filler)
    assert len(article) > 30000
    shown = factcheck.select_source(article, "Van Noort saw houses on light piles moved in a storm", budget=6000)
    assert len(shown) <= 6000 and shown.startswith("Lead paragraph") and "Olivier van Noort" in shown
    assert factcheck.select_source("short article", "anything") == "short article"


def test_foreign_scripts_are_taken_out_and_a_too_large_request_is_retried_smaller():
    article = ("Paro Taktsang (Dzongkha: \u0f66\u0fa4\u0f71\u0f0b\u0f62\u0f7c\u0f0b\u0f66\u0f9f\u0f42\u0f0b"
               "\u0f5a\u0f44\u0f0b; \u864e\u7a74\u5bfa) is a sacred Vajrayana site built on a cliff in Bhutan. "
               "The caf\u00e9 below serves tea.")
    clean = factcheck.clean_source(article)
    assert clean == ("Paro Taktsang (Dzongkha: ; ) is a sacred Vajrayana site built on a cliff in Bhutan. "
                     "The caf\u00e9 below serves tea.")                          # Latin accents stay
    assert factcheck._cost("\u0f66\u0fa4") == 8 and factcheck._cost("abc") == 3

    class Fussy(Checker):                       # refuses the first, full-size request like Groq's HTTP 413
        sizes = []

        def complete_json(self, system, user, schema=None, **k):
            self.sizes.append(len(user))
            if len(self.sizes) == 1:
                raise RuntimeError("groq: HTTP 413 Request too large for model, please reduce your message size")
            return super().complete_json(system, user, schema, **k)

    long = KAMPONG_AYER + "\n" + "\n".join(f"Paragraph {i} about boats, markets and weather. " * 8 for i in range(40))
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River.", TRUE_NOORT])
    llm = Fussy(dict(BASE))
    out, report = factcheck.verify_claims(llm, sc, long, place_words="Kampong Ayer")
    assert len(llm.sizes) == 2 and llm.sizes[1] < llm.sizes[0] * 0.75           # asked again with less article
    assert TRUE_NOORT in out.narration and report["shown_chars"] < len(long) / 2


def test_a_sentence_an_earlier_check_accepted_is_not_judged_again():
    first = "Kampong Ayer stands on stilts above the Brunei River."
    sc = _script([first, TRUE_NOORT])
    llm = Checker(dict(BASE))
    _out, report = factcheck.verify_claims(llm, sc, KAMPONG_AYER, place_words="Kampong Ayer")
    known = factcheck.accepted([report])
    assert first in known and TRUE_NOORT in known

    class Changed(Checker):                 # this time the checker would refuse everything it is asked about
        def complete_json(self, system, user, schema=None, **k):
            asked = [it["text"] for it in json.loads(user.rsplit("ITEMS:\n", 1)[1])]
            assert first not in asked and TRUE_NOORT not in asked          # the verified ones are not sent
            return super().complete_json(system, user, schema, **k)

    sc2 = _script([first, TRUE_NOORT + " " + INVENTED[2]])
    out, report2 = factcheck.verify_claims(Changed({}), sc2, KAMPONG_AYER, kinds=("narration",), known=known)
    assert first in out.narration and TRUE_NOORT in out.narration and INVENTED[2] not in out.narration
    assert sum(bool(r.get("carried")) for r in report2["items"]) == 2
    # nothing new at all -> no call
    silent = Checker(fail=True)
    out, _ = factcheck.verify_claims(silent, _script([first, TRUE_NOORT]), KAMPONG_AYER, kinds=("narration",),
                                     known=known)
    assert out.narration == first + " " + TRUE_NOORT


def test_only_the_narration_is_checked_when_asked():
    llm = Checker(dict(BASE))
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River."], title="A storm swept a village away")
    out, report = factcheck.verify_claims(llm, sc, KAMPONG_AYER, kinds=("narration",))
    assert out.title == "A storm swept a village away" and {r["kind"] for r in report["items"]} == {"narration"}


# ------------------------------------------------------------------ fail closed
def test_checker_down_or_no_source_is_unavailable_not_a_pass():
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River."])
    with pytest.raises(factcheck.FactCheckUnavailable):
        factcheck.verify_claims(Checker(fail=True), sc, KAMPONG_AYER)
    with pytest.raises(factcheck.FactCheckUnavailable):
        factcheck.verify_claims(Checker(dict(BASE)), sc, "")

    class Empty:
        def complete_json(self, *a, **k):
            return {"items": []}
    with pytest.raises(factcheck.FactCheckUnavailable):
        factcheck.verify_claims(Empty(), sc, KAMPONG_AYER)


def _studio(llm, tmp_path, **over):
    st = P.Studio.__new__(P.Studio)
    st.s, st._llm = Settings(output_dir=str(tmp_path), data_dir=str(tmp_path), **over), llm
    return st


def test_pipeline_skips_a_list_topic_when_the_checker_is_down_but_not_a_manual_one(tmp_path):
    st = _studio(Checker(fail=True), tmp_path)
    sc = _script(["Kampong Ayer stands on stilts above the Brunei River."])
    listed = T.Topic("Kampong Ayer, Brunei", "list", source_full=KAMPONG_AYER, place="Kampong Ayer")
    with pytest.raises(SkipTopic, match="fact check unavailable"):
        st._claim_check(sc, listed)
    manual = T.Topic("Kampong Ayer", "manual", source_text=KAMPONG_AYER)
    assert st._claim_check(sc, manual) is sc and manual.claim_reports == []      # continues, with a warning
    st2 = _studio(Checker(dict(BASE)), tmp_path)
    with pytest.raises(SkipTopic, match="not supported by its source"):
        st2._claim_check(_script(INVENTED), listed)


# ------------------------------------------------------------------ the rewrites are checked too
def test_the_length_rewrite_gets_the_source_and_its_additions_are_checked(monkeypatch, tmp_path):
    seen = {}

    def fit(llm, script, seconds, lo, hi, source=""):
        seen["source"] = source
        # the "expansion" adds the storm night
        return _script(["Kampong Ayer stands on stilts above the Brunei River.", TRUE_NOORT, INVENTED[2],
                        "Would you live on the water?"])

    monkeypatch.setattr(P, "fit_story_length", fit)
    st = _studio(Checker(dict(BASE)), tmp_path)
    st._speak = lambda script, work, s: SimpleNamespace(duration=50.0)
    pick = T.Topic("Kampong Ayer, Brunei", "list", source_full=KAMPONG_AYER, source_text=KAMPONG_AYER[:300],
                   place="Kampong Ayer", source_url="https://en.wikipedia.org/wiki/Kampong_Ayer")
    short = _script(["Kampong Ayer stands on stilts above the Brunei River."])
    out, _speech, total = st._length_gate(short, SimpleNamespace(duration=20.0), 20.6, tmp_path, tmp_path, st.s, pick)
    assert "Olivier van Noort" in seen["source"]                 # the rewrite may only draw on the article
    assert INVENTED[2] not in out.narration and TRUE_NOORT in out.narration and 45 <= total <= 60
    saved = json.loads((tmp_path / "claims.json").read_text(encoding="utf-8"))
    assert saved["source_url"].endswith("Kampong_Ayer")
    kinds = {r["kind"] for r in saved["checks"][-1]["items"]}
    assert kinds == {"narration"}                                # title and hook did not change: not re-judged
    assert json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))["scenes"][1]["narration"] == TRUE_NOORT


def test_length_rewrite_prompt_takes_facts_from_the_source_not_from_imagination():
    from purffle_shorts.script import fit_story_length
    seen = {}

    class LLM:
        def complete_json(self, system, user, schema=None, max_tokens=0, temperature=None):
            seen["user"] = user
            return {"scenes": [{"narration": "Kampong Ayer stands on stilts.", "search_query": "q"}]}

    sc = _script(["Kampong Ayer stands on stilts above the Brunei River."])
    fit_story_length(LLM(), sc, seconds=4.0, source=KAMPONG_AYER)
    assert "that the SOURCE below states" in seen["user"] and "light piles" in seen["user"]
    assert "Keep EVERY existing sentence exactly as it is" in seen["user"]       # verified sentences stay
    assert "vivid" not in seen["user"]
    fit_story_length(LLM(), sc, seconds=4.0)                     # no article: nothing new may be added at all
    assert "adding no new facts" in seen["user"] and "vivid" not in seen["user"]
    words = len(sc.narration.split())
    fit_story_length(LLM(), sc, seconds=4.0, source=KAMPONG_AYER)
    assert f"BETWEEN {int(words / 4.0 * 52)} and {int(words / 4.0 * 58)} words" in seen["user"]   # aims high
    fit_story_length(LLM(), sc, seconds=90.0, source=KAMPONG_AYER)               # too long: cut, don't reword
    assert "DELETING whole sentences" in seen["user"] and "Do NOT reword" in seen["user"]


def test_the_writer_and_the_script_doctor_are_told_not_to_add_what_the_source_lacks():
    from purffle_shorts.script import STAKES_RULE, build_prompt, review_prompt
    _system, user = build_prompt("Kampong Ayer", "history", Settings(), [], context=KAMPONG_AYER, place_title=True)
    assert "checked against that background" in user and "is DELETED" in user
    assert "Do NOT add weather" in user and "TAKEN FROM THE BACKGROUND" in user
    assert "TAKEN FROM THE BACKGROUND" in STAKES_RULE and "never make" in STAKES_RULE
    story = _script(["Kampong Ayer stands on stilts above the Brunei River."])
    assert "This is a TRUE story" in review_prompt(story, Settings(), rewrite=True)[1]
    quiz = _script(["Which animal has three hearts?"])
    quiz.style = "facts"
    assert "This is a TRUE story" not in review_prompt(quiz, Settings(), rewrite=True)[1]     # other formats: as before


def test_draft_checks_claims_last_so_the_hook_rewrite_is_covered(monkeypatch, tmp_path):
    """draft(): writer -> number grounding -> hook rewrite -> claim check. The hook rewrite here injects the
    storm night as the new opening line; the claim check, running after it, removes it."""
    pick = T.Topic("Kampong Ayer, Brunei", "list", wiki_title="Kampong Ayer", place="Kampong Ayer",
                   country="Brunei")
    monkeypatch.setattr(P, "pick_topic", lambda *a, **k: pick)

    def ground(p, lang="en"):
        p.source_full = KAMPONG_AYER
        return KAMPONG_AYER[:400], "https://en.wikipedia.org/wiki/Kampong_Ayer"

    monkeypatch.setattr(P, "grounding_article", ground)
    monkeypatch.setattr(P, "write_script", lambda *a, **k: _script(
        ["Kampong Ayer stands on stilts above the Brunei River.", TRUE_NOORT, "Would you live on the water?"]))
    order = []

    def hook(llm, script, place_title=False):
        order.append("hook")
        script.scenes[0].narration = INVENTED[2]
        return script

    monkeypatch.setattr(P, "ensure_concrete_hook", hook)
    real = factcheck.verify_claims
    monkeypatch.setattr(factcheck, "verify_claims", lambda *a, **k: (order.append("claims"), real(*a, **k))[1])
    monkeypatch.setattr(factcheck, "scrub", lambda *a, **k: pytest.fail("the source-less pass must not run"))
    modes = []
    monkeypatch.setattr(P, "review_script", lambda llm, script, s: (modes.append(s.script_review), script)[1])
    st = P.Studio(Settings(output_dir=str(tmp_path / "o"), data_dir=str(tmp_path / "d"), upload=False,
                           visual_sources=["wikimedia"], script_review="rewrite", safety_filter=False,
                           seo_keywords=False, style="auto"))
    st._llm = Checker(dict(BASE))
    st._llm.last_used, st._llm.label = "fake", "fake"
    script, got, _writer = st.draft()
    assert order == ["hook", "claims"]
    assert modes == ["score"]                 # the Script Doctor scores a checked story, it does not rewrite it
    assert INVENTED[2] not in script.narration and TRUE_NOORT in script.narration
    assert got.claim_reports and got.claim_reports[0]["counts"]["unsupported"] == 1
    assert got.source_text == KAMPONG_AYER[:400]                 # the writer still reads the short version


def test_a_place_channel_never_makes_a_story_with_no_article(monkeypatch, tmp_path):
    pick = T.Topic("Somewhere, Nowhere", "niche")
    monkeypatch.setattr(P, "pick_topic", lambda *a, **k: pick)
    monkeypatch.setattr(P, "grounding_article", lambda p, lang="en": ("", ""))
    monkeypatch.setattr(P, "write_script", lambda *a, **k: _script(["Somewhere is old."], topic="Somewhere"))
    st = P.Studio(Settings(output_dir=str(tmp_path / "o"), data_dir=str(tmp_path / "d"), upload=False,
                           visual_sources=["wikimedia"], script_review="off", safety_filter=False,
                           seo_keywords=False, style="auto"))
    st._llm = Checker(dict(BASE))
    with pytest.raises(SkipTopic, match="no source article"):
        st.draft()


def test_grounding_keeps_more_of_the_article_for_the_checker_than_for_the_writer(monkeypatch):
    long = "Hampi was a capital. " * 600                                          # 12,600 chars
    seen = {}

    def page(title, lang="en", max_chars=5000):
        seen["max"] = max_chars
        return {"title": title, "url": "u", "text": long[:max_chars], "lat": None, "lon": None}

    monkeypatch.setattr(T, "article_page", page)
    pick = T.Topic("Hampi, India", "list", wiki_title="Hampi")
    text, _ = P.grounding_article(pick)
    assert seen["max"] == factcheck.SOURCE_FETCH_CHARS
    assert len(text) == P.WRITER_SOURCE_CHARS and len(pick.source_full) == len(long)


# ------------------------------------------------------------------ fitting Groq's 8,000 tokens a minute
class _Prov:
    """A provider double that records how it was called. ``script``: replies/exceptions, one per call."""

    def __init__(self, name, script, effort=True):
        self.name, self.model, self.script, self.seen, self._effort = name, "openai/gpt-oss-120b", list(script), [], effort

    def __str__(self):
        return f"{self.name}:m"

    def takes_effort(self):
        return self._effort

    def complete(self, system, user, **k):
        self.seen.append(k)
        r = self.script.pop(0) if self.script else '{"ok": true}'
        if isinstance(r, Exception):
            raise r
        return r


def _client(*providers, monkeypatch):
    from purffle_shorts import llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L, "with_retries", lambda fn, attempts=2, **k: fn())
    obj = L.LLM.__new__(L.LLM)
    obj.settings, obj.last_used, obj.providers = Settings(), "", list(providers)
    return obj


def test_reasoning_effort_reaches_only_a_provider_that_takes_it(monkeypatch):
    groq = _Prov("groq", [])
    assert _client(groq, monkeypatch=monkeypatch).complete_json("s", "u", effort="low") == {"ok": True}
    assert groq.seen[0]["effort"] == "low"
    other = _Prov("gemini", [], effort=False)
    _client(other, monkeypatch=monkeypatch).complete_json("s", "u", effort="low")
    assert "effort" not in other.seen[0]                         # providers without the option are not handed it
    plain = _Prov("groq", [])
    _client(plain, monkeypatch=monkeypatch).complete_json("s", "u")
    assert "effort" not in plain.seen[0]                         # and nothing changes for callers that don't ask


def test_a_request_groq_calls_too_large_is_retried_once_small_with_low_effort(monkeypatch):
    too_large = RuntimeError("groq:m: HTTP 413 Request too large for model on tokens per minute (TPM): "
                             "Limit 8000, Requested 8569")
    groq = _Prov("groq", [too_large])
    assert _client(groq, monkeypatch=monkeypatch).complete_json("s", "u", max_tokens=12000) == {"ok": True}
    assert len(groq.seen) == 2
    assert groq.seen[0]["max_tokens"] == 12000 and "effort" not in groq.seen[0]
    assert groq.seen[1]["max_tokens"] == 4000 and groq.seen[1]["effort"] == "low"
    twice = _Prov("groq", [too_large, too_large])                # still too large: no endless loop
    from purffle_shorts.llm import LLMError
    with pytest.raises(LLMError):
        _client(twice, monkeypatch=monkeypatch).complete_json("s", "u", max_tokens=12000)
    assert len(twice.seen) == 2


def test_groq_payload_carries_reasoning_effort_only_for_gpt_oss(monkeypatch):
    from purffle_shorts import llm as L
    sent = []

    class Resp:
        status_code, text = 200, ""

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json)
            return Resp()

    monkeypatch.setattr(L, "http", lambda: Session())
    groq = L.PRESETS["groq"]
    oss = L.OpenAICompatible(groq, "openai/gpt-oss-120b", "k", "https://x")
    oss.complete("s", "u", effort="low")
    assert sent[-1]["reasoning_effort"] == "low" and oss.takes_effort()
    llama = L.OpenAICompatible(groq, "llama-3.3-70b-versatile", "k", "https://x")
    llama.complete("s", "u", effort="low")
    assert "reasoning_effort" not in sent[-1] and not llama.takes_effort()
