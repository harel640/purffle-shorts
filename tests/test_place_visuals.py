"""Place-locked visuals: Wikimedia Commons sourcing, license filtering, the >=5-or-skip gate,
per-image credits and the place-named stock-video fallback (no network)."""

import pytest

from purffle_shorts import media as M
from purffle_shorts.media import MediaItem, SkipTopic, Visuals
from purffle_shorts.pipeline import image_credits_block


class R:
    def __init__(self, data):
        self.status_code, self._d, self.text = 200, data, ""

    def json(self):
        return self._d


class S:
    def __init__(self, data):
        self.data = data

    def get(self, url, params=None, headers=None, timeout=None):
        return R(self.data)


# --------------------------------------------------------------- license + author parsing
def test_license_filter_allows_free_rejects_restricted():
    for ok in ["CC BY-SA 4.0", "CC BY 2.0", "CC0", "Public domain", "PD-old"]:
        assert M._wiki_license_ok(ok)[0], ok
    for bad in ["CC BY-NC 4.0", "CC BY-ND 3.0", "CC BY-NC-SA 4.0", "All rights reserved", "", "Fair use"]:
        assert not M._wiki_license_ok(bad)[0], bad


def test_author_strips_html():
    assert M._wiki_author('<a href="/wiki/User:Jane">Jane Doe</a>') == "Jane Doe"
    assert M._wiki_author("") == "Unknown author"


# --------------------------------------------------------------- Wikimedia parse + filtering
def _page(pid, title, lic, w, h, artist="<span>Anon</span>"):
    return {"pageid": pid, "title": title, "index": pid,
            "imageinfo": [{"url": f"https://commons/{pid}.jpg", "width": w, "height": h,
                           "descriptionurl": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                           "extmetadata": {"LicenseShortName": {"value": lic}, "Artist": {"value": artist},
                                           "LicenseUrl": {"value": "https://creativecommons.org/x"},
                                           "ObjectName": {"value": title[5:]}, "Categories": {"value": "Mrauk U"}}}]}


def test_wikimedia_images_filters_license_and_size(monkeypatch):
    data = {"query": {"pages": [
        _page(1, "File:Shittaung Pagoda.jpg", "CC BY-SA 4.0", 2000, 1333, '<a href="x">Jane Doe</a>'),
        _page(2, "File:NC shot.jpg", "CC BY-NC 4.0", 2000, 1333),      # restricted -> dropped
        _page(3, "File:Tiny.jpg", "Public domain", 400, 300),          # too small -> dropped
        _page(4, "File:Kaladan River.jpg", "CC BY 2.0", 1600, 1200),
    ]}}
    monkeypatch.setattr(M, "http", lambda: S(data))
    items = M.wikimedia_images("Mrauk U Myanmar")
    assert [i.id for i in items] == ["1", "4"]
    it = items[0]
    assert it.source == "wikimedia" and it.kind == "image" and it.author == "Jane Doe"
    assert it.license == "CC BY-SA 4.0" and it.page_url.endswith("Shittaung_Pagoda.jpg")
    assert "shittaung" in it.text.lower()


# --------------------------------------------------------------- credits block
def test_image_credits_block_formats_and_dedups():
    creds = [
        {"title": "Shittaung Pagoda", "author": "Jane Doe", "license": "CC BY-SA 4.0",
         "page_url": "https://commons/p1", "license_url": "https://cc/by-sa"},
        {"title": "Shittaung Pagoda", "author": "Jane Doe", "license": "CC BY-SA 4.0",
         "page_url": "https://commons/p1", "license_url": "https://cc/by-sa"},   # duplicate -> once
        {"title": "Kaladan River", "author": "John Roe", "license": "CC BY 2.0",
         "page_url": "https://commons/p2", "license_url": "https://cc/by"},
    ]
    block = image_credits_block(creds)
    assert block.count("Shittaung Pagoda") == 1 and "Kaladan River" in block
    assert "Jane Doe (CC BY-SA 4.0)" in block and "https://commons/p1" in block
    assert image_credits_block([]) == ""


# --------------------------------------------------------------- place-locked assignment + gate
def _wiki_item(pid, text):
    return MediaItem("wikimedia", str(pid), "image", url=f"https://c/{pid}.jpg", width=1600, height=1200,
                     author=f"Author {pid}", license="CC BY-SA 4.0", license_url="https://cc/by-sa",
                     page_url=f"https://commons/{pid}", text=text, extra={"wiki_title": text})


def _visuals(settings, monkeypatch, pool):
    v = Visuals(settings)
    monkeypatch.setattr(v, "_wiki_search", lambda q: list(pool))
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"s{i}.jpg") or item))
    return v


def test_wikimedia_skips_maps_and_butchery(monkeypatch):
    data = {"query": {"pages": [
        _page(1, "File:Shittaung Pagoda.jpg", "CC BY-SA 4.0", 1600, 1200),
        _page(2, "File:Map of Mrauk U district.jpg", "CC BY-SA 4.0", 1600, 1200),      # map -> skip
        _page(3, "File:Butchered goat carcass at market.jpg", "CC BY 2.0", 1600, 1200),  # butchery -> skip
        _page(4, "File:Kaladan River boats.jpg", "CC BY 2.0", 1600, 1200),
    ]}}
    monkeypatch.setattr(M, "http", lambda: S(data))
    assert [i.id for i in M.wikimedia_images("Mrauk U")] == ["1", "4"]


def test_wiki_rank_penalizes_closeup_people():
    terms = M._terms("Mrauk U Shittaung Pagoda temple")
    person = MediaItem("wikimedia", "1", "image", width=1200, height=1600, text="Portrait of a monk, Mrauk U")
    temple = MediaItem("wikimedia", "2", "image", width=1200, height=1600, text="Shittaung Pagoda, Mrauk U temple")
    assert M._wiki_rank(temple, "temple", terms) > M._wiki_rank(person, "temple", terms)


def test_low_quality_image_flags_blank_not_detailed(tmp_path):
    from PIL import Image
    blank = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), (128, 130, 129)).save(blank)
    detailed = tmp_path / "detailed.png"
    img = Image.new("RGB", (200, 200))
    img.putdata([(0, 0, 0) if y < 100 else (255, 255, 255) for y in range(200) for x in range(200)])
    img.save(detailed)
    assert M._low_quality_image(blank) is True
    assert M._low_quality_image(detailed) is False


def test_place_scenes_skips_near_blank_image(settings, monkeypatch, tmp_path):
    pool = [_wiki_item(i, f"Mrauk U landmark {i}") for i in range(6)]
    v = _visuals(settings, monkeypatch, pool)
    calls = {"n": 0}

    def fake_lowq(_path):
        calls["n"] += 1
        return calls["n"] == 1          # only the first fetched image is near-blank

    monkeypatch.setattr(M, "_low_quality_image", fake_lowq)
    out = v.for_place_scenes([("landmark", "", 3.0)], "Mrauk U", "Myanmar", tmp_path / "m", min_images=5)
    assert len(out) == 1 and out[0].source == "wikimedia" and calls["n"] >= 2   # rejected one, took the next


def test_wiki_rank_prefers_landmark_portrait_highres_over_generic():
    terms = M._terms("Shittaung Pagoda Mrauk U temple sunrise")
    landmark = MediaItem("wikimedia", "1", "image", width=1500, height=2000,          # portrait, hi-res, named
                         text="Shittaung Pagoda Mrauk U", rank=5)
    generic = MediaItem("wikimedia", "2", "image", width=1600, height=1200,           # landscape, generic
                        text="green field grass and sky", rank=0)
    assert M._wiki_rank(landmark, "Shittaung Pagoda", terms) > M._wiki_rank(generic, "Shittaung Pagoda", terms)


def test_place_scenes_skips_when_too_few_images(settings, monkeypatch, tmp_path):
    pool = [_wiki_item(i, f"Mrauk U temple {i}") for i in range(3)]        # only 3 < 5
    v = _visuals(settings, monkeypatch, pool)
    jobs = [("temple", "", 3.0), ("river", "", 3.0)]
    with pytest.raises(SkipTopic):
        v.for_place_scenes(jobs, "Mrauk U", "Myanmar", tmp_path / "m", min_images=5)


def test_place_scenes_assigns_and_credits(settings, monkeypatch, tmp_path):
    pool = [_wiki_item(i, f"Mrauk U {name}") for i, name in
            enumerate(["Shittaung Pagoda", "Kaladan River", "Mahamuni", "bamboo boat", "sunrise temple", "skyline"])]
    v = _visuals(settings, monkeypatch, pool)
    jobs = [("Shittaung Pagoda", "", 3.0), ("Kaladan River", "", 3.0), ("bamboo boat", "", 3.0)]
    out = v.for_place_scenes(jobs, "Mrauk U", "Myanmar", tmp_path / "m", min_images=5)
    assert len(out) == 3 and all(m.source == "wikimedia" and m.path for m in out)
    assert out[0].text.lower().find("shittaung") >= 0          # best match goes to the matching scene
    assert "Wikimedia Commons" in v.credits
    assert len(v.image_credits) == 3 and all(c["author"] and c["license"] for c in v.image_credits)


# --------------------------------------------------------------- stock video fallback is place-gated
def test_stock_video_fallback_requires_place_in_metadata(settings, monkeypatch, tmp_path):
    settings.visual_sources = ["pixabay"]
    v = Visuals(settings)
    named = MediaItem("pixabay", "10", "video", url="u", width=1080, height=1920, text="mrauk u temple myanmar")
    other = MediaItem("pixabay", "11", "video", url="u", width=1080, height=1920, text="random city street")
    monkeypatch.setattr(v, "_search", lambda source, q: [other, named])
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / "v.mp4") or item))
    got = v._place_stock_video("temple", "Mrauk U", "Myanmar", tmp_path, 0)
    assert got is not None and got.id == "10"                  # only the place-named clip is allowed


def test_stock_video_fallback_none_when_no_place_match(settings, monkeypatch, tmp_path):
    settings.visual_sources = ["pixabay"]
    v = Visuals(settings)
    other = MediaItem("pixabay", "11", "video", url="u", width=1080, height=1920, text="random city street")
    monkeypatch.setattr(v, "_search", lambda source, q: [other])
    assert v._place_stock_video("temple", "Mrauk U", "Myanmar", tmp_path, 0) is None
