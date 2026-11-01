"""Two rejected videos (a modern zoo/attraction, and a small place drawn with the wrong architecture and
reused photos) drove these: the modern-attraction topic filter, history-vs-place style gating, Commons
geosearch within ~5 km, architecture derived from the article/region, and the no-3-in-a-row image rule."""

import pytest

from purffle_shorts import media as M
from purffle_shorts import script as SC
from purffle_shorts import topics as T
from purffle_shorts.config import Settings
from purffle_shorts.media import MediaItem, SkipTopic, Visuals


class _R:
    def __init__(self, data):
        self.status_code, self._d, self.text = 200, data, ""

    def json(self):
        return self._d


# ---------------------------------------------------------------- 1. modern-attraction filter
def test_modern_attraction_flagged_by_name():
    for name in ["Singapore River Wonders zoo", "Grand Plaza Mall", "Sunway Lagoon theme park",
                 "Atlantis resort", "The Oberoi hotel", "SeaWorld aquarium"]:
        assert T.modern_attraction_reason(name), name


def test_modern_attraction_flagged_by_article_and_year():
    assert T.modern_attraction_reason("River Wonders",
                                      "River Wonders is a river-themed wildlife park that opened in 2013.")
    r = T.modern_attraction_reason("Riverside Dam", "The dam was completed in 1975 and opened in 1980.")
    assert r and "1975" in r                                  # only recorded dates are after 1950


def test_hidden_natural_or_historic_place_is_not_flagged():
    assert T.modern_attraction_reason(
        "Pulga", "Pulga is a remote Himalayan village in the Parvati Valley known for its deodar forests.") == ""
    assert T.modern_attraction_reason(
        "Hampi", "Hampi was the capital of the Vijayanagara Empire, founded in the 14th century.") == ""


# ---------------------------------------------------------------- 2. real-history gate (history vs place)
def test_has_real_history_detects_pre_1900():
    assert T.has_real_history("The temple was built in 1150 by the Hoysala dynasty.")
    assert T.has_real_history("Settled in the 9th century, it flourished in antiquity.")
    assert T.has_real_history("The site dates to 300 BCE.")


def test_has_real_history_false_for_modern_or_natural_places():
    assert not T.has_real_history("Pulga is a village at 2100 metres with pine forests and a waterfall.")
    assert not T.has_real_history("The park opened in 2013 and draws two million visitors a year.")
    assert not T.has_real_history("")


# ---------------------------------------------------------------- 3. geosearch + place-accurate pool
def _cpage(pid, title, lic="CC BY-SA 4.0", w=1600, h=1200, cats="Pulga"):
    return {"pageid": pid, "title": title, "index": pid,
            "imageinfo": [{"url": f"https://c/{pid}.jpg", "width": w, "height": h,
                           "descriptionurl": f"https://commons/{title}",
                           "extmetadata": {"LicenseShortName": {"value": lic}, "Artist": {"value": "<a>A</a>"},
                                           "ObjectName": {"value": title[5:]}, "Categories": {"value": cats}}}]}


def test_geosearch_images_sends_coords_and_filters_license(monkeypatch):
    data = {"query": {"pages": [_cpage(1, "File:Pulga village.jpg"),
                                _cpage(2, "File:NC shot.jpg", "CC BY-NC 4.0")]}}  # NC -> dropped
    seen = {}

    class _Sess:
        def get(self, url, params=None, headers=None, timeout=None):
            seen.update(params)
            return _R(data)

    monkeypatch.setattr(M, "http", lambda: _Sess())
    items = M.geosearch_images(32.0, 77.3, 5000)
    assert [i.id for i in items] == ["1"]
    assert seen["generator"] == "geosearch" and seen["ggscoord"] == "32.0|77.3" and seen["ggsradius"] == 5000


def _wimg(key_id, text):
    return MediaItem("wikimedia", key_id, "image", url=f"https://c/{key_id}.jpg", width=1600, height=1200,
                     author=f"Author {key_id}", license="CC BY-SA 4.0", license_url="https://cc/by-sa",
                     page_url=f"https://commons/{key_id}", text=text, extra={"wiki_title": text})


def test_place_pool_takes_nearby_photos_and_title_matches_only(monkeypatch):
    v = Visuals(Settings())
    near = [_wimg(f"g{i}", f"random hillside {i}") for i in range(3)]        # within 5 km: accepted as-is
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(near))
    named = [_wimg("n1", "Pulga village path"), _wimg("n2", "Pulga deodar forest"),
             _wimg("x1", "Kasol market street"), _wimg("x2", "unrelated waterfall")]  # x* don't name Pulga
    monkeypatch.setattr(v, "_wiki_search", lambda q: list(named))
    pool = {i.key for i in v._place_pool("Pulga", "India", 32.0, 77.3, 5)}
    assert {"wikimedia:g0", "wikimedia:n1", "wikimedia:n2"} <= pool
    assert "wikimedia:x1" not in pool and "wikimedia:x2" not in pool      # title must name the place


def test_place_pool_skips_topic_when_too_few_place_accurate(monkeypatch):
    v = Visuals(Settings())
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: [])
    monkeypatch.setattr(v, "_wiki_search", lambda q: [_wimg("x", "somewhere else entirely")])
    with pytest.raises(SkipTopic):
        v._place_pool("Pulga", "India", None, None, 5)      # no coords, no title match -> empty -> skip


# ---------------------------------------------------------------- 4. architecture from article/region
def test_architecture_derives_from_article_region_not_country_alone():
    himal = M._architecture_hint("Pulga", "India",
                                 "Pulga is a village in Himachal Pradesh built in the kath-kuni style.")
    assert "kath-kuni" in himal.lower() and "no domes" in himal
    # Without the article, the country fallback would wrongly draw a Himalayan village as a South Indian temple:
    assert "gopuram" in M._architecture_hint("Pulga", "India").lower()


# ---------------------------------------------------------------- 5. no image for >2 scenes in a row
def test_no_image_repeats_more_than_twice_in_a_row(monkeypatch, tmp_path):
    s = Settings(visual_sources=["wikimedia"])               # no stock-video fallback in the test
    v = Visuals(s)
    pool = [_wimg("a", "Pulga ridge"), _wimg("b", "Pulga river")]   # only two photos, five scenes
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool))
    monkeypatch.setattr(v, "_wiki_search", lambda q: [])
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"s{i}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    jobs = [("Pulga", "", False)] * 5
    keys = [m.key for m in v.for_place_scenes(jobs, "Pulga", "India", tmp_path / "m",
                                              min_images=2, lat=32.0, lon=77.3)]
    assert len(keys) == 5
    assert all(not (keys[i] == keys[i + 1] == keys[i + 2]) for i in range(len(keys) - 2)), keys


# ---------------------------------------------------------------- the present-day "place" story style
def test_place_style_is_a_present_day_story_with_no_ai_and_no_cta():
    assert "place" in SC.STYLES and "place" in SC.STORY_STYLES and "place" in SC.CALM_STYLES
    _, user = SC.build_prompt("Pulga village", "place", Settings(), avoid=[], context="")
    assert "ALWAYS false" in user                            # every scene is present-day
    assert "no AI image is generated" in user
    assert "Do NOT ask viewers to like" in user              # calm payoff, no call to action
    assert "165 and 235 words" in user                       # 60-90 seconds like history


def test_place_style_uses_story_length_and_scene_count():
    assert SC.effective_seconds("place", 40) == SC.HISTORY_MIN_SECONDS
    assert SC.scene_plan("place", 75)[1] >= 6


def test_place_style_strips_a_trailing_cta_and_keeps_all_scenes_present_day():
    data = {"topic": "Pulga", "title": "What hides in this valley?", "hook_text": "A hidden valley",
            "scenes": [{"narration": "What hides in this quiet Himalayan valley?",
                        "search_query": "Pulga", "image_prompt": "Pulga", "historical": False},
                       {"narration": "It stays green and calm. Follow for more hidden places!",
                        "search_query": "Pulga valley", "image_prompt": "x", "historical": False}],
            "description": "d", "hashtags": ["#a"], "tags": ["t"], "category": "education"}
    sc = SC.normalize(data, "Pulga", "place", "en")
    assert not SC._CTA.search(sc.scenes[-1].narration)       # the "Follow for more" CTA is stripped
    assert all(not scene.historical for scene in sc.scenes)
