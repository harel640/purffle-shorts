"""Nov 22 09:00: Kellie's Castle was skipped as "recent violence" because the hook "Dream dies, ruin lives" was
dated by the article sentence "Kellie-Smith died at the age of 56 of pneumonia ... in 1926" - the natural-death
exemption looked only at the script's own words."""

import pytest

from purffle_shorts.script import Scene, Script, recent_violence_reason

# A short excerpt of the article's facts (Kellie's Castle, Wikipedia), the parts that name deaths and dates.
ARTICLE = (
    "Kellie's Castle is an unfinished castle located in Batu Gajah, Perak, Malaysia. The unfinished, ruined "
    "mansion was built by the Scottish planter William Kellie-Smith.\n\n"
    "Construction on a mansion called Kellas House started in 1910. With the birth of a son in 1915, Kellie-Smith "
    "began planning for a huge castle.\n"
    "In 1918, a virulent strain of Spanish flu struck the workmen. When they approached Kellie-Smith to build a "
    "temple nearby, he readily agreed.\n"
    "Kellie-Smith died at the age of 56 of pneumonia during a short trip to Lisbon, Portugal, in 1926. His wife "
    "was devastated and decided to move back to Scotland; construction on the castle was never completed.")


def _script(title, hook, *lines):
    return Script(topic="Kellie's Castle", title=title, hook_text=hook, description="d", style="history",
                  scenes=[Scene(x, "q") for x in lines])


def test_the_kellies_castle_hook_and_opening_pass():
    s = _script("Did a Scottish planter's dream die with his wife's leaving?", "Dream dies, ruin lives",
                "By 1926 the Scottish planter died, leaving his massive unfinished castle to crumble.",
                "Kellie-Smith had planned a tower, a tunnel and Malaya's first elevator.",
                "His wife went home to Scotland, and the work never resumed.")
    assert recent_violence_reason(s, ARTICLE) == ""


def test_a_death_the_script_names_by_the_person_is_matched_to_the_article():
    s = _script("t", "h", "Kellie-Smith died on a short trip to Lisbon in 1926.")
    assert recent_violence_reason(s, ARTICLE) == ""


@pytest.mark.parametrize("article_line", [
    "Kellie-Smith was killed in a riot in Lisbon in 1926.",
    "Kellie-Smith died in 1926 when troops opened fire on the crowd.",
    "Thousands died of the Spanish flu in Perak in 1918.",
])
def test_a_violent_or_mass_death_in_the_article_still_blocks(article_line):
    article = ARTICLE.replace(
        "Kellie-Smith died at the age of 56 of pneumonia during a short trip to Lisbon, Portugal, in 1926.",
        article_line)
    s = _script("t", "Dream dies, ruin lives", "By 1926 the Scottish planter died in Lisbon.")
    assert recent_violence_reason(s, article) != ""


def test_an_undated_plain_death_with_no_death_in_the_article_still_blocks():
    article = "Kellie's Castle is an unfinished castle in Perak. Construction started in 1910."
    assert recent_violence_reason(_script("t", "h", "Soldiers came, and many were dead."), article) != ""


def test_a_killing_in_the_script_is_never_cleared_by_a_natural_death_in_the_article():
    s = _script("t", "h", "In 1926 the planter was killed in Lisbon.")
    assert recent_violence_reason(s, ARTICLE) != ""
