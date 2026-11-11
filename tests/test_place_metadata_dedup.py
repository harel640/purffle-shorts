"""Place Shorts: exact-title grounding, title/description format, page/coordinate de-dup, foreign photos."""

import json
from datetime import datetime, timezone

import pytest

from purffle_shorts import media as M
from purffle_shorts import topics as T
from purffle_shorts.config import Settings
from purffle_shorts.history import History, same_place
from purffle_shorts.media import MediaItem, SkipTopic
from purffle_shorts.pipeline import final_description, final_title, grounding_article, place_title
from purffle_shorts.script import Scene, Script


class _R:
    def __init__(self, data):
        self.status_code, self._d, self.text = 200, data, ""

    def json(self):
        return self._d


# ------------------------------------------------------------------ 1. exact Wikipedia title for list places
def test_article_page_looks_up_the_exact_title_never_searches(monkeypatch):
    seen = {}

    class S:
        def get(self, url, params=None, timeout=None):
            seen.update(params)
            return _R({"query": {"pages": [{"title": "Ban Rak Thai", "extract": "Ban Rak Thai is a village.",
                                            "coordinates": [{"lat": 19.6, "lon": 97.9}]}]}})

    monkeypatch.setattr(T, "http", lambda: S())
    page = T.article_page("Ban Rak Thai")
    assert seen["titles"] == "Ban Rak Thai" and "generator" not in seen and "gsrsearch" not in seen
    assert page["title"] == "Ban Rak Thai" and (page["lat"], page["lon"]) == (19.6, 97.9)


def test_list_place_is_grounded_in_its_own_article_or_skipped(monkeypatch):
    pick = T.Topic("Ban Rak Thai, Mae Hong Son, Thailand", "list", wiki_title="Ban Rak Thai")
    monkeypatch.setattr(T, "article", lambda *a, **k: pytest.fail("a listed place must not be searched for"))
    monkeypatch.setattr(T, "article_page", lambda title, lang="en", **k: {
        "title": "Ban Rak Thai", "url": "https://en.wikipedia.org/wiki/Ban_Rak_Thai",
        "text": "Ban Rak Thai is a Yunnanese village.", "lat": 19.6, "lon": 97.9})
    text, url = grounding_article(pick)
    assert "Yunnanese village" in text and url.endswith("/Ban_Rak_Thai") and pick.lat == 19.6
    monkeypatch.setattr(T, "article_page", lambda title, lang="en", **k: None)
    with pytest.raises(SkipTopic, match="not found"):           # missing: skip, never fall back to a search
        grounding_article(T.Topic("x", "list", wiki_title="Nope"))


def test_other_topics_still_use_the_search(monkeypatch):
    monkeypatch.setattr(T, "article", lambda q, lang="en", **k: ("text about " + q, "https://en.wikipedia.org/wiki/Q"))
    assert grounding_article(T.Topic("Hampi", "niche"))[0] == "text about Hampi"


def test_list_entry_carries_its_page_and_coordinates(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "places": [
        {"rank": 1, "place": "Shey Gompa", "region": "Dolpa", "country": "Nepal", "wikipedia_title": "Shey Gompa",
         "lat": 29.352, "lon": 82.825}]}), encoding="utf-8")
    t = T.from_list(Settings(topic_list_file=str(p)), History(tmp_path / "h.db"))
    assert (t.wiki_title, t.place, t.country, t.lat, t.lon) == ("Shey Gompa", "Shey Gompa", "Nepal", 29.352, 82.825)


# ------------------------------------------------------------------ 2. title: PLACE: hook #shorts, under 60
def _script(title="The Richest City That Vanished", hook="LOST CITY OF KINGS", desc="", tags=(), hashtags=()):
    return Script(topic="t", title=title, hook_text=hook, scenes=[Scene("n", "q")], description=desc,
                  tags=list(tags), hashtags=list(hashtags), style="history")


@pytest.mark.parametrize("hook, place, expect", [
    ("The Richest City That Vanished", "Hampi", "HAMPI: The Richest City That Vanished #shorts"),
    ("Hampi: the richest city that vanished", "Hampi", "HAMPI: The richest city that vanished #shorts"),
    ("A cave big enough to fly a jet through", "Gunung Mulu National Park",
     "GUNUNG MULU: A cave big enough to fly a jet through #shorts"),
    ("A village frozen in snow and time", "Shirakawa-go", "SHIRAKAWA-GO: A village frozen in snow and time #shorts"),
])
def test_title_leads_with_the_place_then_the_hook(hook, place, expect):
    assert place_title(hook, place) == expect and len(expect) < 60


def test_title_never_reaches_60_and_a_long_hook_gives_way_to_the_short_one():
    long = "Why does this ancient Shiva temple on the shore stand untouched for sixteen centuries"
    t = final_title(_script(title=long, hook="ANCIENT STONE DEFIES TIME"), Settings(), place="Gokarna")
    assert t == "GOKARNA: Ancient Stone Defies Time #shorts" and len(t) < 60
    for place in ("Mahabaleshwar Temple", "Ban Gioc–Detian Falls", "Khangchendzonga National Park"):
        assert len(final_title(_script(title=long, hook=""), Settings(), place=place)) < 60


def test_non_place_videos_keep_the_old_title():
    assert final_title(_script(title="Octopuses Are Aliens"), Settings()) == "Octopuses Are Aliens #shorts"


# ------------------------------------------------------------------ 3. description: summary, keywords, 4 hashtags
def test_description_is_summary_then_keywords_then_four_hashtags():
    sc = _script(desc="From the Kadamba kings to today, this shore temple has outlasted empires. Come see it.",
                 tags=["gokarna karnataka temple", "shorts", "Mahabaleshwar Temple", "#viralshorts", "viral travel",
                       "Kadamba dynasty", "shortsfeed", "ancient Indian temples"],
                 hashtags=["#viralshorts", "#fyp", "#Gokarna"])
    d = final_description(sc, set(), Settings(), extra=["Source: https://en.wikipedia.org/wiki/Gokarna"],
                          place="Gokarna", country="India")
    summary, keywords, hashtags, source = d.split("\n\n")
    assert summary == "Gokarna, India: From the Kadamba kings to today, this shore temple has outlasted empires."
    kw = keywords.removeprefix("Keywords: ").split(", ")
    assert 3 <= len(kw) <= 5 and not any("viral" in k or "shorts" in k for k in kw)
    assert hashtags == "#shorts #Gokarna #India #travel"
    assert "viralshorts" not in d and "#fyp" not in d and source.startswith("Source:")


def test_description_names_place_and_country_in_its_first_sentence():
    sc = _script(desc="A Gokarna shore temple that outlasted empires.")
    assert final_description(sc, set(), Settings(), place="Gokarna", country="India").split("\n\n")[0] == \
        "A Gokarna shore temple that outlasted empires, in India."
    one = final_description(_script(desc=""), set(), Settings(), place="Singapore Botanic Gardens",
                            country="Singapore").split("\n\n")
    assert one[0].startswith("The story of Singapore Botanic Gardens") and one[2].count("#") in (3, 4)


# ------------------------------------------------------------------ 4. de-dup by page/coordinates, not name
SHEY_GOMPA = {"wiki_page": "Shey Gompa", "lat": 29.352, "lon": 82.825, "place": "Shey Gompa", "country": "Nepal"}


def test_same_wikipedia_page_under_another_name_is_the_same_place():
    made = {"resolved_place": "Crystal Monastery, Nepal", "wiki_page": "Shey_Gompa", "lat": None, "lon": None}
    assert "same Wikipedia page" in same_place(SHEY_GOMPA, made)


def test_a_shared_word_is_not_the_same_place():
    shey_palace = {"resolved_place": "Shey Palace, India", "wiki_page": "Shey Palace", "lat": 34.07, "lon": 77.64}
    assert same_place(SHEY_GOMPA, shey_palace) == ""


def test_two_names_for_one_spot_match_by_coordinates():
    vittala = {"resolved_place": "Vittala Temple, Hampi", "wiki_page": "Vittala Temple", "lat": 15.342, "lon": 76.475}
    assert "km apart" in same_place({"wiki_page": "Hampi", "lat": 15.335, "lon": 76.462}, vittala)


def test_old_rows_without_page_or_coordinates_fall_back_to_the_name(tmp_path):
    h = History(tmp_path / "h.db")
    h.add_video(status="uploaded", title="x", resolved_place="Vittala Temple, Hampi, India")       # legacy row
    assert h.place_recently_made(place="Hampi", country="India", wiki_page="Hampi", lat=15.3, lon=76.4)
    h2 = History(tmp_path / "h2.db")
    h2.add_video(status="uploaded", title="x", resolved_place="Shey Palace, India", wiki_page="Shey Palace",
                 lat=34.07, lon=77.64)
    assert h2.place_recently_made(**{k: v for k, v in SHEY_GOMPA.items()}) == ""
    h2.add_video(status="uploaded", title="y", resolved_place="Crystal Monastery, Nepal", wiki_page="Shey Gompa")
    assert "Shey Gompa" in h2.place_recently_made(**SHEY_GOMPA)


def test_the_list_skips_a_place_already_made_under_another_name(tmp_path):
    h = History(tmp_path / "h.db")
    h.add_video(status="uploaded", title="x", resolved_place="Crystal Monastery, Nepal", wiki_page="Shey Gompa",
                lat=29.352, lon=82.825)
    p = tmp_path / "list.json"
    p.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "places": [
        {"rank": 1, "place": "Shey Gompa", "country": "Nepal", "wikipedia_title": "Shey Gompa",
         "lat": 29.352, "lon": 82.825},
        {"rank": 2, "place": "Shey Palace", "country": "India", "wikipedia_title": "Shey Palace",
         "lat": 34.07, "lon": 77.64}]}), encoding="utf-8")
    assert T.from_list(Settings(topic_list_file=str(p)), h).place == "Shey Palace"   # not blocked by "Shey"


# ------------------------------------------------------------------ 5. photos of the same name elsewhere
def _cpage(pid, title, cats="", coords=None, gps=None):
    meta = {"LicenseShortName": {"value": "CC BY-SA 4.0"}, "Artist": {"value": "A"},
            "ObjectName": {"value": title[5:]}, "Categories": {"value": cats}}
    if gps:
        meta.update(GPSLatitude={"value": str(gps[0])}, GPSLongitude={"value": str(gps[1])})
    pg = {"pageid": pid, "title": title, "index": pid,
          "imageinfo": [{"url": f"https://c/{pid}.jpg", "width": 1600, "height": 1200,
                         "descriptionurl": f"https://commons/{pid}", "extmetadata": meta}]}
    if coords:
        pg["coordinates"] = [{"lat": coords[0], "lon": coords[1]}]
    return pg


def test_commons_items_keep_where_the_photo_was_taken():
    assert M._wiki_item(_cpage(1, "File:Shey Gompa.jpg", coords=(29.35, 82.83)), 0, 800).extra["lat"] == 29.35
    assert M._wiki_item(_cpage(2, "File:Shey Palace.jpg", gps=(34.07, 77.64)), 0, 800).extra["lon"] == 77.64
    assert "lat" not in M._wiki_item(_cpage(3, "File:Shey.jpg"), 0, 800).extra


def test_photo_of_the_same_name_in_another_place_or_country_is_rejected():
    far = M._wiki_item(_cpage(1, "File:Shey Palace Ladakh.jpg", coords=(34.07, 77.64)), 0, 800)
    near = M._wiki_item(_cpage(2, "File:Shey Gompa courtyard.jpg", coords=(29.36, 82.83)), 0, 800)
    india = M._wiki_item(_cpage(3, "File:Shey monastery.jpg", cats="Monasteries in Ladakh|Buddhism in India"), 0, 800)
    nepal = M._wiki_item(_cpage(4, "File:Shey 2.jpg", cats="Dolpa District|Monasteries in Nepal"), 0, 800)
    border = M._wiki_item(_cpage(5, "File:Shey 3.jpg", cats="Nepal-India border"), 0, 800)
    assert "km from the place" in M._foreign_photo(far, 29.352, 82.825, "Nepal")
    assert M._foreign_photo(near, 29.352, 82.825, "Nepal") == ""
    assert "India" in M._foreign_photo(india, 29.352, 82.825, "Nepal")
    assert M._foreign_photo(nepal, 29.352, 82.825, "Nepal") == ""
    assert M._foreign_photo(border, 29.352, 82.825, "Nepal") == ""        # names its own country too
    burma = M._wiki_item(_cpage(6, "File:Bagan.jpg", cats="Temples in Burma"), 0, 800)
    assert M._foreign_photo(burma, None, None, "Myanmar") == ""            # Burma is Myanmar


def test_place_pool_drops_the_foreign_namesakes(monkeypatch):
    v = M.Visuals(Settings())
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: [])
    same = [M._wiki_item(_cpage(i, f"File:Shey Gompa {i}.jpg", coords=(29.35, 82.83)), 0, 800) for i in range(1, 4)]
    other = [M._wiki_item(_cpage(9, "File:Shey Gompa Ladakh copy.jpg", coords=(34.07, 77.64)), 0, 800)]
    monkeypatch.setattr(v, "_wiki_search", lambda q: same + other)
    pool = v._place_pool("Shey Gompa", "Nepal", 29.352, 82.825, 3)
    assert {i.id for i in pool} == {"1", "2", "3"}
