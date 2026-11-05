"""The modern-attraction rules also run on the place a script RESOLVED to, not just the topic that was picked."""

import pytest

from purffle_shorts import pipeline as P
from purffle_shorts import topics as T
from purffle_shorts.topics import Topic

WULINGYUAN = Topic("Wulingyuan, Hunan Province, China", "list", wiki_title="Wulingyuan", place="Wulingyuan",
                   country="China")
NICHE = Topic("underrated places to visit in Asia", "niche")

ARTICLES = {
    "zhangjiajie glass bridge": ("Zhangjiajie Glass Bridge", "Zhangjiajie Glass footpath is a skywalk bridge in "
                                 "Hunan. The bridge, opened to the public on 20 August 2016, is 430 m long."),
    "gardens by the bay": ("Gardens by the Bay", "Gardens by the Bay is a nature park spanning 101 hectares. The "
                           "Gardens opened in 2012."),
    "tianqiaoshengkong": ("Wulingyuan", "Wulingyuan is a scenic area in Hunan with quartzite sandstone pillars."),
    "marina bay sands": ("Marina Bay Sands", "Marina Bay Sands is an integrated resort fronting Marina Bay."),
    "sunway lagoon": ("Sunway Lagoon", "Sunway Lagoon is a amusement park in Petaling Jaya, Malaysia."),
    "tabo monastery": ("Tabo Monastery", "Tabo Monastery is a Tibetan Buddhist monastery in Spiti, founded in 996 CE."),
}


@pytest.fixture(autouse=True)
def _wikipedia(monkeypatch):
    def article(query, lang="en", max_chars=5000):
        for k, (title, text) in ARTICLES.items():
            if query.lower().startswith(k):
                return text, "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        return "", ""
    monkeypatch.setattr(T, "article", article)


@pytest.mark.parametrize("place, country", [
    ("Zhangjiajie Glass Bridge", "China"),            # a 2016 skywalk: blocked by its name alone
    ("Singapore Zoo", "Singapore"), ("Genting Highlands Resort", "Malaysia"),
])
def test_attractions_the_script_resolved_to_are_blocked_by_name(place, country):
    assert P.resolved_place_reason(WULINGYUAN, place, country)       # even though the topic (Wulingyuan) was fine
    assert P.resolved_place_reason(NICHE, place, country)


def test_blocked_by_its_own_article_when_the_name_gives_nothing_away():
    reason = P.resolved_place_reason(NICHE, "Gardens by the Bay", "Singapore")
    assert "2012" in reason and "after 1950" in reason
    assert "resort" in P.resolved_place_reason(NICHE, "Marina Bay Sands", "Singapore")     # defined as a resort
    assert "amusement park" in P.resolved_place_reason(WULINGYUAN, "Sunway Lagoon", "Malaysia")


def test_the_topics_own_place_and_natural_places_stay_allowed():
    assert P.resolved_place_reason(WULINGYUAN, "Wulingyuan", "China") == ""                   # the listed place
    assert P.resolved_place_reason(WULINGYUAN, "Tianqiaoshengkong", "China") == ""            # a natural arch in it
    assert P.resolved_place_reason(NICHE, "Tabo Monastery", "India") == ""
    assert P.resolved_place_reason(NICHE, "Horton Plains National Park", "Sri Lanka") == ""   # no article found


def test_an_article_about_something_else_is_ignored(monkeypatch):
    # A search that lands on another place's article (it once turned Shirakawa-go into Emperor Go-Shirakawa) must
    # not block - or clear - a place by what that other article says.
    monkeypatch.setattr(T, "article", lambda q, lang="en", max_chars=5000: (
        "The Mall of Asia is a shopping mall opened in 2006.", "https://en.wikipedia.org/wiki/Mall_of_Asia"))
    assert P.resolved_place_reason(NICHE, "Pulau Ubin", "Singapore") == ""


def test_a_place_the_user_asked_for_is_never_blocked():
    for source in ("manual", "url", "document", "script", "clip"):
        assert P.resolved_place_reason(Topic("Zhangjiajie Glass Bridge", source), "Zhangjiajie Glass Bridge",
                                       "China") == ""


def test_glass_bridge_and_skywalk_names_are_attractions():
    for name in ("Zhangjiajie Glass Bridge, China", "Dongjiawan glass skywalk", "Zhangjiajie Glass footpath"):
        assert T.modern_attraction_reason(name, ""), name
    assert T.modern_attraction_reason("Wulingyuan, China", "") == ""
