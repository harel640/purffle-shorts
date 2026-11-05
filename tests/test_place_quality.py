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


@pytest.mark.parametrize("subject, article", [
    # A date of something AT a historic site, not the site opening (Koh Ker: a 10th-century temple city).
    ("Koh Ker, Cambodia", "Koh Ker is a remote archaeological site in northern Cambodia, capital of the Khmer "
     "empire in the 10th century. Travellers can stay in Koh Ker Jungle Lodge Homestay, a sustainable tourism "
     "project built in the village of Koh Ker in 2009. A visitor centre opened in 2015."),
    ("Ta Prohm, Cambodia", "Ta Prohm is a temple built in the 12th century. Restoration work completed in 2013 "
     "stabilised the galleries."),
    # A town that HAS resorts is not a resort (Kep); nor is a park that hosts one, or an atoll with resort islands.
    ("Kep, Cambodia", "Kep is a seaside resort town in southern Cambodia, founded as a French colonial retreat "
     "in 1908. It has several resorts and hotels."),
    ("Ulu Temburong National Park, Brunei", "Ulu Temburong National Park is the first national park to be "
     "established in Brunei, protected since 1991. It hosts the Ulu Ulu Resort."),
    ("Baa Atoll, Maldives", "Baa Atoll is a natural atoll. The atoll hosts several resort islands."),
    ("Nam Et-Phou Louey, Laos", "Nam Et-Phou Louey is a protected area in northern Laos. Visitors enjoy wildlife "
     "viewing from the observation tower (under construction) nearby."),
    # A natural area's post-1950 date is when it was protected; a commune's is an administrative change.
    ("Nino Konis Santana National Park, Timor-Leste", "Nino Konis Santana National Park is Timor-Leste's first "
     "national park. The park, established on 15 August 2007, covers 1,236 square kilometres."),
    ("Koshi Tappu, Nepal", "Koshi Tappu Wildlife Reserve is a protected area in the Terai. It was established in "
     "1976 and designated as a Ramsar site in December 1987."),
    ("Pu Luong, Vietnam", "Pu Luong is a commune of Thanh Hoa province, Vietnam. The commune was established on "
     "16 June 2025 on the basis of merging the former communes."),
])
def test_modern_attraction_rule_ignores_facilities_restorations_designations(subject, article):
    assert T.modern_attraction_reason(subject, article) == "", subject


@pytest.mark.parametrize("subject, article, kind", [
    ("Nusa Dua, Indonesia", "Nusa Dua is a beach resort on the Bukit peninsula.", "resort"),
    ("Marina Bay Sands, Singapore", "Marina Bay Sands is an integrated resort fronting Marina Bay.", "resort"),
    ("Genting Highlands, Malaysia", "Genting Highlands is a hill resort in Pahang.", "resort"),
    ("Siam Paragon, Thailand", "Siam Paragon is a shopping mall in Bangkok.", "shopping mall"),
    # Its own opening counts, even when other words in the sentence are "facilities" or the region is historic.
    ("Zhangjiajie Glass Bridge, China", "Zhangjiajie Glass Bridge is a skywalk bridge above Wulingyuan, a region "
     "settled since the Ming dynasty. The bridge, opened to the public on 20 August 2016, is 430 m long.", "glass bridge"),
    ("Zhangjiajie Footbridge, China", "Zhangjiajie Footbridge is a footbridge above Wulingyuan, a region settled "
     "since the Ming dynasty. The bridge, opened to the public on 20 August 2016, is 430 m long.", "2016"),
    ("Heritage Museum, Malaysia", "The Heritage Museum is a museum of the region's 15th-century sultanate. The "
     "museum opened in 2005.", "2005"),
])
def test_modern_attraction_rule_still_blocks_what_is_one(subject, article, kind):
    assert kind in T.modern_attraction_reason(subject, article), subject


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


def test_commons_busy_error_body_is_retried_not_read_as_no_images(monkeypatch):
    # Commons answers 200 {"error": {"info": "Search is currently too busy"}} when overloaded. That used to
    # come back as 0 photos, so a well-photographed place was skipped for "too few images".
    import purffle_shorts.utils as U
    monkeypatch.setattr(U.time, "sleep", lambda s: None)
    busy = _R({"error": {"code": "search-error", "info": "Search is currently too busy. Please try again later."}})
    replies = [busy, _R({"query": {"pages": [_cpage(1, "File:Pulga village.jpg")]}})]

    class _Sess:
        def get(self, url, params=None, headers=None, timeout=None):
            return replies.pop(0)

    monkeypatch.setattr(M, "http", lambda: _Sess())
    assert [i.id for i in M.geosearch_images(32.0, 77.3)] == ["1"]          # retried, then found
    replies[:] = [busy] * 3
    with pytest.raises(RuntimeError, match="too busy"):                       # still busy: an error, not []
        M.geosearch_images(32.0, 77.3)


def _wimg(key_id, text, w=1600, h=1200):
    return MediaItem("wikimedia", key_id, "image", url=f"https://c/{key_id}.jpg", width=w, height=h,
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


def test_wiki_item_skips_stereoscopic_3d_images():
    # A side-by-side 3D / stereoscopic / anaglyph Commons file renders as a split, doubled frame with a seam
    # down the middle — reject it by title keyword like we already reject maps, signs and butchery images.
    for title in ("File:Gokarna AdiGokarna DeityL BackWall 3D.jpg", "File:Temple stereoscopic pair.jpg",
                  "File:Shrine anaglyph red cyan.jpg", "File:Mahabaleshwar gopura SBS.jpg"):
        assert M._wiki_item(_cpage(1, title), 0, 800) is None, title
    # A normal flat photo of the same landmark is still kept.
    assert M._wiki_item(_cpage(2, "File:Mahabaleshwar Temple Gokarna.jpg"), 0, 800) is not None


# ---------------------------------------------------------------- 4. architecture from article/region
def test_architecture_derives_from_article_text_never_the_country():
    himal = M.look_from_text("Pulga", "Pulga is a village in Himachal Pradesh built in the kath-kuni style.")
    assert "kath-kuni" in himal["architecture"].lower() and " no " not in himal["architecture"].lower()
    assert "pattu" in himal["clothing"].lower()
    # The country alone says nothing about the style: no default (it once drew a Himalayan village with gopurams).
    assert M.look_from_text("Pulga", "") == {}
    assert M.look_from_text("Pulga", "Pulga is a village in India.") == {}


@pytest.mark.parametrize("place, article, arch_words, clothing_words", [
    ("Shey Gompa", "The Shey Gompa is a Tibetan Buddhist monastery in the Dolpo region of Nepal.",
     ["whitewashed", "red", "flat roofs"], ["maroon robes"]),
    ("Chennakeshava Temple", "The Chennakeshava Temple at Belur is a 12th-century Hoysala temple in Karnataka.",
     ["star-shaped", "soapstone"], ["dhoti"]),
    ("Shirakawa-go", "Shirakawa-go is known for its gassho-zukuri farmhouses.", ["thatched"], ["kimono"]),
])
def test_look_matches_the_articles_own_style(place, article, arch_words, clothing_words):
    look = M.look_from_text(place, article)
    assert all(w in look["architecture"].lower() for w in arch_words), look
    assert all(w in look["clothing"].lower() for w in clothing_words), look


def test_look_uses_the_llm_reading_the_article_when_no_rule_matches():
    article = "Old Kandahar Port is a fishing harbour ringed by coral-stone warehouses with teak shutters."

    class LLM:
        calls = 0

        def complete_json(self, system, user, *, schema=None, **k):
            LLM.calls += 1
            assert schema is M.LOOK_SCHEMA and article in user           # it reads THIS place's article
            return {"architecture": "coral-stone warehouses with teak shutters", "clothing": "sailors in lungis"}

    v = Visuals(Settings(), llm=LLM())
    look = v.place_look("Old Port", "Maldives", article)
    assert "coral-stone" in look["architecture"] and look["clothing"] == "sailors in lungis"
    v.place_look("Old Port", "Maldives", article)
    assert LLM.calls == 1                                                # cached per place + article
    assert Visuals(Settings()).place_look("Somewhere", "India", "A village in India.") == {}   # no basis, no LLM


def test_no_look_means_no_ai_image_and_the_prompt_carries_the_clothing(monkeypatch, tmp_path):
    from PIL import Image
    seen = {}

    def fake_poll(prompt, dest, width, height, key=""):
        seen["p"] = prompt
        Image.new("RGB", (1024, 1024), (40, 40, 40)).save(dest)
        return dest

    monkeypatch.setattr(M, "pollinations_image", fake_poll)
    v = Visuals(Settings(ai_scene_source="pollinations"))
    assert v._historical_image("monks at prayer", tmp_path, 0, arch="") is None     # nothing place-true to draw
    look = M.look_from_text("Shey Gompa", "a Tibetan Buddhist monastery in Dolpo")
    v._historical_image("monks at prayer", tmp_path, 1, arch=look["architecture"], clothing=look["clothing"])
    assert seen["p"].startswith("Tibetan Buddhist monastery") and "maroon robes" in seen["p"]


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
    assert "open question to the viewer" in user and "Never ask them to like" in user  # open question, no CTA
    assert "115 and 155 words" in user                       # 45-60 seconds like history


def test_place_style_uses_story_length_and_scene_count():
    assert SC.effective_seconds("place", 40) == SC.HISTORY_MIN_SECONDS
    assert SC.scene_plan("place", 75)[1] >= 6


def test_story_length_targets_45_to_60_seconds_and_a_tight_middle():
    assert SC.HISTORY_RANGE == (45, 60)
    assert SC.effective_seconds("history", 40) == 50 and SC.effective_seconds("place", 40) == 50
    _, user = SC.build_prompt("Hampi", "history", Settings(), avoid=[], context="x")
    assert "45-60 second" in user and "115 and 155" in user
    assert "no lists of rulers" in user


def test_story_ending_keeps_an_open_question_but_strips_platform_cta():
    data = {"topic": "Gokarna", "title": "Why does Gokarna hide a secret?", "hook_text": "",
            "scenes": [{"narration": "A temple stands by the sea.", "search_query": "gokarna",
                        "image_prompt": "gokarna", "historical": False},
                       {"narration": "Gokarna still guards its secret. Which temple would you visit? "
                                     "Like and subscribe for more!",
                        "search_query": "gokarna temple", "image_prompt": "x", "historical": False}],
            "description": "d", "hashtags": ["#a"], "tags": ["t"], "category": "education"}
    last = SC.normalize(data, "Gokarna", "history", "en").scenes[-1].narration.lower()
    assert "would you visit" in last                        # the open viewer question survives
    assert "subscribe" not in last and "like and" not in last   # the platform CTA is stripped


class _Sess:
    def __init__(self, data):
        self.data = data

    def get(self, url, params=None, headers=None, timeout=None):
        return _R(self.data)


def test_wikimedia_skips_signs_notices_and_boards(monkeypatch):
    data = {"query": {"pages": [
        _cpage(1, "File:Temple gopuram tower.jpg"),
        _cpage(2, "File:Information board at the temple.jpg"),     # board -> skip
        _cpage(3, "File:Warning sign near the cliff.jpg"),         # warning + sign -> skip
        _cpage(4, "File:Notice board.jpg"),                        # notice + board -> skip
        _cpage(5, "File:Signpost to the gate.jpg"),                # signpost -> skip
        _cpage(6, "File:Temple courtyard at dawn.jpg"),
    ]}}
    monkeypatch.setattr(M, "http", lambda: _Sess(data))
    assert [i.id for i in M.wikimedia_images("temple")] == ["1", "6"]


def test_historical_image_never_generates_a_deity(monkeypatch, tmp_path):
    v = Visuals(Settings(ai_scene_source="pollinations"))
    calls = {"n": 0}
    monkeypatch.setattr(M, "pollinations_image", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    assert v._historical_image("Lord Shiva blesses the king", tmp_path, 0, arch="x") is None
    assert v._historical_image("Ganesha appears as a cowherd", tmp_path, 1, arch="x") is None
    assert v._historical_image("Ravana carries the lingam", tmp_path, 2, arch="x") is None
    assert calls["n"] == 0                                         # the generator was never even called
    assert v.ai_used is False


def test_historical_image_prompt_has_no_negatives(monkeypatch, tmp_path):
    from PIL import Image
    seen = {}

    def fake_poll(prompt, dest, width, height, key=""):
        seen["p"] = prompt
        Image.new("RGB", (1024, 1024), (40, 40, 40)).save(dest)
        return dest

    monkeypatch.setattr(M, "pollinations_image", fake_poll)
    v = Visuals(Settings(ai_scene_source="pollinations"))
    v._historical_image("stonemasons raising a temple wall", tmp_path, 0,
                        arch="South Indian gopuram temple towers")
    p = seen["p"].lower()
    assert " no " not in p and "letterbox" not in p and "without" not in p    # positive description only
    assert "full-bleed vertical" in p


def test_wikimedia_skips_food_dishes_and_restaurants(monkeypatch):
    data = {"query": {"pages": [
        _cpage(1, "File:Temple gopuram.jpg"),
        _cpage(2, "File:Masala dosa dish.jpg"),            # dish -> skip
        _cpage(3, "File:Thali meal at a restaurant.jpg"),  # meal + restaurant + thali -> skip
        _cpage(4, "File:People eating lunch.jpg"),          # eating + lunch -> skip
        _cpage(5, "File:Street food stall.jpg"),            # food -> skip
        _cpage(6, "File:Temple tank at dawn.jpg"),
    ]}}
    monkeypatch.setattr(M, "http", lambda: _Sess(data))
    assert [i.id for i in M.wikimedia_images("temple")] == ["1", "6"]


def test_opening_scene_is_the_strongest_landmark_photo_not_generic(monkeypatch, tmp_path):
    v = Visuals(Settings(visual_sources=["wikimedia"]))
    pool = [
        _wimg("g1", "grass field and trees Gokarna"),               # generic -> never the opener
        _wimg("t1", "Mahabaleshwar Temple Gokarna", w=1600, h=2000),  # the landmark, hi-res portrait
        _wimg("b1", "Gokarna beach"),
    ]
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool))
    monkeypatch.setattr(v, "_wiki_search", lambda q: [])
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"s{i}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    jobs = [("temple", "", False), ("beach", "", False), ("temple", "", False),
            ("sea", "", False), ("temple", "", False)]
    out = v.for_place_scenes(jobs, "Gokarna", "India", tmp_path / "m", min_images=3, lat=14.5, lon=74.3,
                             title="Why does Gokarna's ancient Mahabaleshwar temple hide a secret?")
    assert out[0].id == "t1"       # opener is the hi-res temple photo, not the grass/trees shot


class _FakeVisionLLM:
    """Stands in for the LLM vision check. ``verdicts`` maps an image key (or 'subject:...') to yes/no/None."""
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = 0

    def image_shows(self, path, subject):
        self.calls += 1
        for key, v in self.verdicts.items():
            if key.startswith("subject:") and key[8:] in subject:
                return v
        return self.verdicts.get(str(path), self.verdicts.get("*"))


def _real_jpeg(path, fmt="JPEG", size=(64, 48)):
    from PIL import Image
    im = Image.new("RGB", size, (120, 90, 60))
    if fmt == "MPO":                                   # a two-frame multi-picture file, as many cameras write
        im.save(path, format="MPO", save_all=True, append_images=[Image.new("RGB", size, (10, 10, 10))])
    else:
        im.save(path, format=fmt)
    return path


def test_image_shows_parses_yes_no(monkeypatch, tmp_path):
    from purffle_shorts.llm import LLM
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    img = _real_jpeg(tmp_path / "x.jpg")
    answers = iter(["Yes.", "no", "maybe"])

    class _Resp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": next(answers)}}]}

    class _Sess:
        def post(self, url, json=None, headers=None, timeout=None):
            return _Resp()

    monkeypatch.setattr("purffle_shorts.llm.http", lambda: _Sess())
    llm = LLM.__new__(LLM)                       # bypass provider wiring; we only test image_shows
    llm.settings = Settings(vision_model="gemini-x")   # fixed model -> no network model lookup
    assert llm.image_shows(img, "a temple") is True
    assert llm.image_shows(img, "a temple") is False
    assert llm.image_shows(img, "a temple") is None    # unusable answer -> Cloudflare absent -> None


def test_image_shows_falls_back_to_cloudflare_when_gemini_absent(monkeypatch, tmp_path):
    from purffle_shorts.llm import LLM
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    img = _real_jpeg(tmp_path / "x.jpg")

    class _Resp:
        status_code = 200
        def json(self):
            return {"result": {"description": "No, this is a beach."}}

    monkeypatch.setattr("purffle_shorts.llm.http", lambda: type("S", (), {
        "post": lambda self, url, json=None, headers=None, timeout=None: _Resp()})())
    llm = LLM.__new__(LLM)
    llm.settings = Settings(cloudflare_account_id="acct", cloudflare_api_token="tok")
    assert llm.image_shows(img, "a temple") is False    # Cloudflare LLaVA answer parsed


def test_verify_subject_rejects_and_caches(monkeypatch, tmp_path):
    v = Visuals(Settings(), llm=_FakeVisionLLM({"*": False}))
    item = _wimg("k1", "Gokarna temple")
    item.path = tmp_path / "k1.jpg"
    item.path.write_bytes(b"x")
    assert v.verify_subject(item, "a temple") is False
    v.llm.verdicts["*"] = True                   # change the answer; cache must stick to the first verdict
    assert v.verify_subject(item, "a temple") is False
    assert v.llm.calls == 1                       # second call served from cache


def test_good_wiki_skips_a_photo_the_pixels_dont_show(monkeypatch, tmp_path):
    # Vision says the dirt-road photo (t1) is NOT a temple; the next candidate (t2) is accepted.
    verdicts = {}
    v = Visuals(Settings(visual_sources=["wikimedia"]),
                llm=_FakeVisionLLM(verdicts))
    pool = [_wimg("t1", "Gokarna Mahabaleshwar temple", w=2000, h=1500),   # title says temple, pixels don't
            _wimg("t2", "Gokarna Mahabaleshwar temple", w=1600, h=1200)]
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"{item.id}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)

    def shows(path, subject):
        v.llm.calls += 1
        return "t2" in str(path)                  # only t2 really shows the temple

    v.llm.image_shows = shows
    got = v._good_wiki(pool, "temple", M._terms("Gokarna temple"), tmp_path / "m", 0, subject="a temple")
    assert got is not None and got.id == "t2"


def test_landmark_opener_excludes_animal_and_people_titles(monkeypatch, tmp_path):
    # No vision available (llm=None): the opener still refuses a monkey/tourist photo for the landmark.
    v = Visuals(Settings(visual_sources=["wikimedia"]), llm=None)
    pool = [_wimg("m1", "Monkey at Gokarna temple", w=2000, h=1500),   # animal main subject -> never the opener
            _wimg("t1", "Gokarna Mahabaleshwar temple tower", w=1600, h=1200)]
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool))
    monkeypatch.setattr(v, "_wiki_search", lambda q: [])
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"{item.id}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    jobs = [("Mahabaleshwar temple", "", False), ("temple", "", False), ("temple", "", False)]
    out = v.for_place_scenes(jobs, "Gokarna", "India", tmp_path / "m", min_images=2, lat=14.5, lon=74.3,
                             title="Why does Gokarna's Mahabaleshwar temple hide a secret?")
    assert out[0].id == "t1"                      # the monkey photo is never the opener


def test_vision_available_detects_a_backend(monkeypatch):
    from purffle_shorts.llm import LLM
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    llm = LLM.__new__(LLM)
    llm.settings = Settings()
    assert llm.vision_available() is False
    llm.settings = Settings(cloudflare_account_id="a", cloudflare_api_token="t")
    assert llm.vision_available() is True
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    llm.settings = Settings()
    assert llm.vision_available() is True


def test_fail_closed_skips_video_when_vision_unavailable(monkeypatch, tmp_path):
    class _LLM:                                   # a None/empty answer must be treated as unavailable, not a pass
        def image_shows(self, path, subject):     # (no vision_available() -> still fails closed on None)
            return None

    v = Visuals(Settings(visual_sources=["wikimedia"]), llm=_LLM())
    pool = [_wimg(f"p{i}", f"Gokarna Mahabaleshwar temple {i}") for i in range(5)]
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool))
    monkeypatch.setattr(v, "_wiki_search", lambda q: [])
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"{item.id}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    with pytest.raises(SkipTopic, match="vision unavailable"):
        v.for_place_scenes([("temple", "", False)] * 3, "Gokarna", "India", tmp_path / "m",
                           min_images=5, lat=14.5, lon=74.3, title="Why does Gokarna temple hide a secret?")


def test_verify_subject_none_is_unavailable_only_yes_is_a_pass():
    def mk(answer):
        class _L:
            def image_shows(self, p, s):
                return answer
        it = _wimg("k", "Gokarna temple")
        it.path = "k.jpg"                          # truthy so the check runs
        return Visuals(Settings(), llm=_L()), it

    v, it = mk(None)                           # blank: never a pass; one image alone isn't an outage
    assert v.verify_subject(it, "a temple") == M.NO_ANSWER and v._vision_unavailable is False
    v, it = mk("")
    assert v.verify_subject(it, "a temple") == M.NO_ANSWER and v._vision_unavailable is False
    v, it = mk(True)
    assert v.verify_subject(it, "a temple") is True and v._vision_unavailable is False  # only yes passes
    v, it = mk(False)
    assert v.verify_subject(it, "a temple") is False and v._vision_unavailable is False  # no rejects, not unavailable


def test_neuron_budget_reserves_allocation_for_vision(tmp_path):
    from purffle_shorts.history import History
    from purffle_shorts.media import NeuronBudget
    b = NeuronBudget(History(tmp_path / "h.db"), daily=1000, reserve=400)   # 600 spendable on image gen
    assert b.can_spend(500) is True
    b.record(500)
    assert b.can_spend(500) is False              # 500 + 500 > 600
    assert b.can_spend(100) is True               # 500 + 100 == 600, still allowed


def test_ai_image_falls_back_to_photo_when_reserve_hit(monkeypatch, tmp_path):
    from purffle_shorts.history import History
    from purffle_shorts.media import NeuronBudget
    b = NeuronBudget(History(tmp_path / "h.db"), daily=1000, reserve=1000)  # nothing left for gen
    v = Visuals(Settings(ai_scene_source="cloudflare", cloudflare_account_id="a", cloudflare_api_token="t"),
                budget=b)
    called = {"n": 0}
    monkeypatch.setattr(M, "cloudflare_image", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    assert v._historical_image("stonemasons raising a temple wall", tmp_path, 0, arch="x") is None
    assert called["n"] == 0                        # generation never ran; the scene falls back to a real photo


def test_ai_scene_style_is_cinematic_painterly_not_cartoon():
    style = Settings().ai_scene_style.lower()
    assert "painterly" in style and "cinematic" in style
    assert "cartoon" not in style and "anime" not in style


def test_plan_ai_scenes_first_last_real_cap_5_and_40pct_real():
    from purffle_shorts.pipeline import plan_ai_scenes
    out = plan_ai_scenes([True] * 9)                      # all marked historical
    assert out[0] is False and out[-1] is False           # first & last always real
    assert sum(out) <= 5                                  # at most 5 AI
    assert (len(out) - sum(out)) >= 4                     # >= ceil(0.4*9) real photos
    small = plan_ai_scenes([True, True, True])
    assert small[0] is False and small[-1] is False and sum(small) <= 1
    assert plan_ai_scenes([False] * 6) == [False] * 6    # nothing forced on


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
