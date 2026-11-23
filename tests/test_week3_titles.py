"""Week-3 analytics: the best title was a concrete dramatic event stated outright ("A 7.2 magnitude earthquake
struck Bohol": 401 views, 69% stayed); the worst were dull fallbacks ("UNESCO declared the ruins a World Heritage
Site": 59, "Gurjari Mahal still stands": 50); vague subjects sat mid-pack."""

import pytest

from purffle_shorts import factcheck as F
from purffle_shorts.script import DESIGNATION_RE, headline_score, vague_hook_reason


@pytest.mark.parametrize("title", [
    "A 7.2 magnitude earthquake struck Bohol", "Spanish flu struck the workmen building a castle",
    "Kellie-Smith died on a short trip to Lisbon", "Why did a king betray his own brother?",
    "How did one villager's roof survive massive snow?", "Soldiers burned the city in a single night"])
def test_a_title_may_be_a_statement_about_a_dramatic_event_or_a_named_person(title):
    assert vague_hook_reason(title, "title") == ""


@pytest.mark.parametrize("title", [
    "UNESCO declared the ruins a World Heritage Site", "Gurjari Mahal still stands",
    "The fort was built by a sultan", "The temple was founded in the 9th century",
    "The ruins are a heritage site today", "Why does the fort still stand?"])
def test_designations_still_stands_and_founding_facts_are_never_headlines(title):
    assert DESIGNATION_RE.search(title) and vague_hook_reason(title, "title") != ""


def test_a_dramatic_event_or_a_named_person_outranks_a_generic_subject():
    quake = headline_score("A 7.2 magnitude earthquake struck Bohol")
    person = headline_score("Kellie-Smith died on a short trip to Lisbon")
    generic = [headline_score(t) for t in ("Why did a monarch spend a fortune on a father's memory?",
                                           "Why did a jungle castle stop rising?")]
    dull = [headline_score(t) for t in ("UNESCO declared the ruins a World Heritage Site", "Gurjari Mahal still stands")]
    assert quake > person > max(generic) > max(dull)


def _row(text, quote="q"):
    return {"kind": "narration", "verdict": "supported", "quote": quote, "text": text, "overlap": 0.9}


def test_the_fallback_title_is_the_most_dramatic_supported_sentence_never_a_designation():
    rows = [_row("UNESCO declared the ruins a World Heritage Site."), _row("Gurjari Mahal still stands."),
            _row("The palace was founded by a sultan."),
            _row("Raiders sacked the palace and carried off its treasury.")]
    assert F._best_row(rows)["text"] == "Raiders sacked the palace and carried off its treasury."
    assert F._best_row(rows, fit=60)["text"].startswith("Raiders sacked")


def test_with_only_designations_supported_there_is_no_fallback_sentence():
    assert F._best_row([_row("UNESCO declared the ruins a World Heritage Site."),
                        _row("Gurjari Mahal still stands.")]) is None


def test_the_headline_prompt_asks_for_events_and_named_people_not_generic_nouns():
    r = F._HEADLINE_RULES
    assert "QUESTION or a STATEMENT" in r and "DRAMATIC EVENT" in r and "NAMED PERSON" in r
    assert "UNESCO" in r and "still stands" in r and "monarch" in r
