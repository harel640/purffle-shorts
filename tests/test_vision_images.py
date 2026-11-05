"""Vision check on real-world photos: re-encoding to plain JPEG, and a per-image failure vs a real outage.

Commons photos can be MPO (multi-picture JPEG): Cloudflare's vision model answers HTTP 400 "Unsupported input
image format 'MPO'" for them, and Gemini is flaky, so every image got no answer and the video was skipped as
"vision unavailable" - twice on 2026-11-05."""

import base64
import io

import pytest
from PIL import Image

from purffle_shorts import media as M
from purffle_shorts.config import Settings
from purffle_shorts.llm import LLM, ImageUnusable, vision_jpeg
from purffle_shorts.media import NO_ANSWER, MediaItem, Visuals


def _img(path, fmt="JPEG", size=(64, 48), mode="RGB"):
    im = Image.new(mode, size, (120, 90, 60) if mode == "RGB" else 128)
    if fmt == "MPO":
        im.save(path, format="MPO", save_all=True, append_images=[Image.new("RGB", size, (9, 9, 9))])
    else:
        im.save(path, format=fmt)
    return path


# ------------------------------------------------------------------ 1. re-encode to a plain JPEG
def test_an_mpo_photo_is_re_encoded_to_a_plain_jpeg(tmp_path):
    src = _img(tmp_path / "p.jpg", "MPO", size=(2592, 1944))
    assert Image.open(src).format == "MPO"                       # the file really is an MPO
    data = vision_jpeg(src)
    out = Image.open(io.BytesIO(data))
    assert out.format == "JPEG" and max(out.size) <= 1024 and out.mode == "RGB"   # plain, small
    assert len(data) < src.stat().st_size


@pytest.mark.parametrize("fmt, mode", [("PNG", "RGB"), ("PNG", "L"), ("WEBP", "RGB"), ("GIF", "L"), ("JPEG", "L")])
def test_other_formats_and_colour_modes_become_rgb_jpeg(tmp_path, fmt, mode):
    out = Image.open(io.BytesIO(vision_jpeg(_img(tmp_path / "x.img", fmt, mode=mode))))
    assert out.format == "JPEG" and out.mode == "RGB"


def test_an_undecodable_file_is_unusable_not_an_outage(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    with pytest.raises(ImageUnusable):
        vision_jpeg(bad)
    with pytest.raises(ImageUnusable):
        vision_jpeg(tmp_path / "missing.jpg")


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._b, self.text = status, body or {}, text

    def json(self):
        return self._b


def _llm(monkeypatch, handler, gemini=True):
    if gemini:
        monkeypatch.setenv("GEMINI_API_KEY", "k")
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("purffle_shorts.llm.http", lambda: type("S", (), {"post": handler})())
    llm = LLM.__new__(LLM)
    llm.settings = Settings(vision_model="gemini-x", cloudflare_account_id="a", cloudflare_api_token="t")
    return llm


def test_both_backends_receive_the_re_encoded_jpeg_not_the_mpo(monkeypatch, tmp_path):
    src = _img(tmp_path / "p.jpg", "MPO", size=(2592, 1944))
    seen = {}

    def post(self, url, json=None, headers=None, timeout=None):
        if "generativelanguage" in url:
            seen["gemini"] = base64.b64decode(json["messages"][0]["content"][1]["image_url"]["url"].split(",")[1])
            return _Resp(503, text="overloaded")
        seen["cf"] = bytes(json["image"])
        return _Resp(200, {"result": {"description": "Yes"}})

    assert _llm(monkeypatch, post).image_shows(src, "a highland plain") is True
    for name in ("gemini", "cf"):
        assert Image.open(io.BytesIO(seen[name])).format == "JPEG", name


# ------------------------------------------------------------------ 2. per-image error vs no backend answering
def test_a_backend_refusing_the_image_format_is_an_image_problem(monkeypatch, tmp_path):
    src = _img(tmp_path / "p.jpg")

    def post(self, url, json=None, headers=None, timeout=None):
        return _Resp(400, text='{"errors":[{"message":"AiError: Unsupported input image format \'MPO\'"}]}')

    with pytest.raises(ImageUnusable):
        _llm(monkeypatch, post, gemini=False).image_shows(src, "a temple")


def test_an_outage_is_none_not_an_image_problem(monkeypatch, tmp_path):
    src = _img(tmp_path / "p.jpg")
    llm = _llm(monkeypatch, lambda self, url, json=None, headers=None, timeout=None: _Resp(503, text="busy"))
    assert llm.image_shows(src, "a temple") is None                  # 503 on both -> no answer (not unusable)
    quota = _llm(monkeypatch, lambda self, url, json=None, headers=None, timeout=None: _Resp(429, text="quota"))
    assert quota.image_shows(src, "a temple") is None


class _L:
    """image_shows scripted per file name: True/False/None, or an exception."""
    def __init__(self, script):
        self.script, self.calls = script, []

    def image_shows(self, path, subject):
        self.calls.append(path)
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        r = self.script[name]
        if isinstance(r, Exception):
            raise r
        return r


def _item(name):
    it = MediaItem("wikimedia", name, "image", width=1600, height=1200, text=f"Horton Plains {name}",
                   extra={"wiki_title": name})
    it.path = f"/x/{name}"
    return it


def test_an_unreadable_image_is_skipped_and_vision_stays_available():
    v = Visuals(Settings(), llm=_L({"a": ImageUnusable("MPO"), "b": True}))
    assert v.verify_subject(_item("a"), "a plain") is False           # skip this image...
    assert v._vision_unavailable is False and v._vision_streak == 0   # ...it is not an outage
    assert v.verify_subject(_item("b"), "a plain") is True


def test_one_image_without_an_answer_is_skipped_not_fatal():
    v = Visuals(Settings(), llm=_L({"a": None, "b": True}))
    assert v.verify_subject(_item("a"), "a plain") == NO_ANSWER and v._vision_unavailable is False
    assert v.verify_subject(_item("b"), "a plain") is True and v._vision_streak == 0


def test_vision_is_down_only_after_several_images_and_none_answered():
    v = Visuals(Settings(), llm=_L({n: None for n in "abcdef"}))
    for n in "ab":
        v.verify_subject(_item(n), "x")
    assert v._vision_unavailable is False                              # two silent images: not yet
    v.verify_subject(_item("c"), "x")
    assert v._vision_unavailable is True and v._vision_answered == 0   # third, with none ever answered


def test_a_backend_that_dies_mid_video_is_caught_by_the_streak():
    script = {"a": True, **{n: None for n in "bcdefg"}}
    v = Visuals(Settings(), llm=_L(script))
    v.verify_subject(_item("a"), "x")
    for n in "bcde":
        v.verify_subject(_item(n), "x")
    assert v._vision_unavailable is False                              # 4 silent in a row after an answer: fine
    v.verify_subject(_item("f"), "x")
    v.verify_subject(_item("g"), "x")
    assert v._vision_unavailable is True                               # 6th in a row: it has gone down


# ------------------------------------------------------------------ the scene loops use the above
def _pool(monkeypatch, v, n=6):
    pool = [M.MediaItem("wikimedia", f"p{i}", "image", url=f"https://c/{i}.jpg", width=1600, height=1200,
                        author="A", license="CC BY-SA 4.0", license_url="u", page_url=f"u{i}",
                        text=f"Horton Plains National Park {i}", extra={"wiki_title": f"HP {i}"})
            for i in range(1, n + 1)]
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"{item.id}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    return pool


def test_the_scene_picker_moves_past_unreadable_and_silent_images(monkeypatch, tmp_path):
    llm = _L({"p1": ImageUnusable("MPO"), "p2": None, "p3": True, "p4": True, "p5": True, "p6": True})
    v = Visuals(Settings(), llm=llm)
    got = v._good_wiki(_pool(monkeypatch, v), "plain", M._terms("Horton Plains"), tmp_path, 0, subject="a plain")
    assert got is not None and got.id in ("p3", "p4", "p5", "p6")      # never the unreadable/silent ones
    assert v._vision_unavailable is False


def test_an_unverified_image_is_never_used_when_vision_has_gone_quiet(monkeypatch, tmp_path):
    v = Visuals(Settings(), llm=_L({f"p{i}": None for i in range(1, 7)}))
    got = v._good_wiki(_pool(monkeypatch, v), "plain", M._terms("Horton Plains"), tmp_path, 0, subject="a plain")
    assert got is None and v._vision_unavailable is True               # nothing unverified slips through


def test_a_video_is_skipped_only_when_no_image_could_be_verified(monkeypatch, tmp_path):
    # Mixed: some images unreadable, some silent, some answered -> the video proceeds.
    mixed = {"p1": ImageUnusable("MPO"), "p2": ImageUnusable("MPO"), "p3": None, "p4": True, "p5": True,
             "p6": True}
    v = Visuals(Settings(visual_sources=["wikimedia"]), llm=_L(mixed))
    pool = _pool(monkeypatch, v)
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool))
    monkeypatch.setattr(v, "_wiki_search", lambda q: [])
    out = v.for_place_scenes([("Horton Plains", "", False)] * 3, "Horton Plains", "Sri Lanka", tmp_path / "m",
                             min_images=3, lat=6.8, lon=80.8)
    assert len(out) == 3 and all(m.source == "wikimedia" for m in out)
    # Nothing answers for any image -> SkipTopic("vision unavailable").
    v2 = Visuals(Settings(visual_sources=["wikimedia"]), llm=_L({f"p{i}": None for i in range(1, 7)}))
    pool2 = _pool(monkeypatch, v2)
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool2))
    monkeypatch.setattr(v2, "_wiki_search", lambda q: [])
    with pytest.raises(M.SkipTopic, match="vision unavailable"):
        v2.for_place_scenes([("Horton Plains", "", False)] * 3, "Horton Plains", "Sri Lanka", tmp_path / "m2",
                            min_images=3, lat=6.8, lon=80.8)
