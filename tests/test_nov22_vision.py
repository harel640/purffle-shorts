"""Nov 22 09:00: no photo could be checked. Cloudflare's daily 10,000 neurons were used up (vision calls were
never counted in the budget), and a Gemini TEXT timeout had switched Gemini VISION off too. Vision calls now
count, a "daily allocation used up" reply switches Cloudflare off after one call, the circuits are separate,
and a local CLIP check takes over when no cloud backend answers."""

import json

import pytest
from PIL import Image

from purffle_shorts import clip_local
from purffle_shorts import llm as L
from purffle_shorts import media as M
from purffle_shorts.config import Settings


class _Hist:
    def __init__(self):
        self.meta = {}

    def get_meta(self, k):
        return self.meta.get(k)

    def set_meta(self, k, v):
        self.meta[k] = v


def _img(tmp_path, size=(1600, 1200), name="x.jpg"):
    p = tmp_path / name
    Image.new("RGB", size, (120, 110, 90)).save(p)
    return p


class _Reply:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)
        self.headers = {}

    def json(self):
        return self._body


def _llm(monkeypatch, replies, budget=None):
    sent = []

    class Sess:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json)
            return replies.pop(0) if replies else _Reply(200, {"result": {"response": "yes"}})

    monkeypatch.setattr(L, "http", lambda: Sess())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm = L.LLM.__new__(L.LLM)
    llm.settings = Settings(cloudflare_account_id="a", cloudflare_api_token="t")
    llm.neurons = budget
    return llm, sent


# ------------------------------------------------------------------ 2. every vision call is counted
def test_neurons_come_from_the_replys_token_usage():
    body = {"result": {"usage": {"prompt_tokens": 1000, "completion_tokens": 2}}}
    assert L.cf_vision_neurons(b"", body) == pytest.approx(1000 * 31876 / 1e6 + 2 * 50488 / 1e6)


def test_neurons_are_estimated_from_the_picture_without_usage(tmp_path):
    small = L.vision_jpeg(_img(tmp_path), L.CF_VISION_MAX_SIDE)
    with Image.open(__import__("io").BytesIO(small)) as im:
        assert max(im.size) == 640                       # the picture sent to Cloudflare is capped at 640 px
    est = L.cf_vision_neurons(small)
    assert 10 < est < 40                                 # ~23 x 18 tokens + the question: about 20 neurons


def test_a_vision_call_is_recorded_in_the_budget(monkeypatch, tmp_path):
    b = M.NeuronBudget(_Hist(), 10000, 3500)
    llm, sent = _llm(monkeypatch, [_Reply(200, {"result": {"response": "Yes", "usage": {
        "prompt_tokens": 600, "completion_tokens": 1}}})], b)
    assert llm.image_shows(_img(tmp_path), "Kellie's Castle") is True
    assert b.calls == {"vision": 1} and b.run["vision"] == pytest.approx(600 * 31876 / 1e6 + 50488 / 1e6)
    assert b.spent_today() == pytest.approx(b.run["vision"], abs=0.1)
    assert "Cloudflare neurons this run: 19 - vision 19 (1 call); today (UTC): 19 of 10000" in b.summary()


def test_image_generation_keeps_the_reserve_but_vision_may_use_it():
    h = _Hist()
    b = M.NeuronBudget(h, 10000, 3500)
    h.meta[b.key] = "6400"
    assert not b.can_spend(720) and b.can_spend(720, "vision")
    h.meta[b.key] = "9990"
    assert not b.can_spend(20, "vision")


def test_no_cloudflare_call_once_todays_budget_is_used_up(monkeypatch, tmp_path):
    h = _Hist()
    b = M.NeuronBudget(h, 10000, 3500)
    h.meta[b.key] = "9999"
    llm, sent = _llm(monkeypatch, [], b)
    assert llm.image_shows(_img(tmp_path), "a fort") is None and sent == []
    assert llm.circuit_open(L.CF_VISION)


def test_a_used_up_daily_allocation_stops_cloudflare_after_one_call(monkeypatch, tmp_path):
    used_up = {"errors": [{"message": "AiError: AiError: you have used up your daily free allocation of 10,000 "
                                      "neurons, please upgrade", "code": 4006}], "success": False}
    b = M.NeuronBudget(_Hist(), 10000, 3500)
    llm, sent = _llm(monkeypatch, [_Reply(429, used_up), _Reply(429, used_up)], b)
    assert llm.image_shows(_img(tmp_path), "a fort") is None
    assert llm.image_shows(_img(tmp_path, name="y.jpg"), "a fort") is None
    assert len(sent) == 1 and b.exhausted and "allocation is used up" in b.summary()


def test_cloudflare_vision_failures_do_not_touch_the_text_circuit(monkeypatch, tmp_path):
    llm, _ = _llm(monkeypatch, [_Reply(503, {"errors": [{"message": "busy"}]})] * 2)
    llm.image_shows(_img(tmp_path), "a fort")
    llm.image_shows(_img(tmp_path, name="y.jpg"), "a fort")
    assert llm.circuit_open(L.CF_VISION) and not llm.circuit_open("gemini")


# ------------------------------------------------------------------ 4. the local CLIP fallback
class _NoCloud:
    def image_shows(self, path, subject):
        return None                                      # no cloud backend answered


def _visuals():
    return M.Visuals(Settings(), set(), llm=_NoCloud())


def _item(tmp_path, place_accurate=True):
    it = M.MediaItem("wikimedia", "f1", "image", path=_img(tmp_path, (400, 300)), text="Kellie's Castle")
    if place_accurate:
        it.extra["place_accurate"] = "geotagged within 5 km"
    return it


def test_the_local_check_answers_when_no_cloud_backend_does(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(clip_local, "available", lambda: True)
    monkeypatch.setattr(clip_local, "photo_of_a_place",
                        lambda p, exterior=False: (seen.append(exterior) or True, "CLIP: a castle (place 0.97)"))
    v = _visuals()
    assert v.verify_subject(_item(tmp_path), "Kellie's Castle, seen from outside") is True
    assert seen == [True] and not v._vision_unavailable and v._vision_answered == 1


def test_the_local_check_rejects_a_graphic(monkeypatch, tmp_path):
    monkeypatch.setattr(clip_local, "available", lambda: True)
    monkeypatch.setattr(clip_local, "photo_of_a_place", lambda p, exterior=False: (False, "CLIP: a map (0.05)"))
    assert _visuals().verify_subject(_item(tmp_path), "Kellie's Castle") is False


def test_the_local_check_never_vouches_for_a_photo_that_is_not_place_accurate(monkeypatch, tmp_path):
    monkeypatch.setattr(clip_local, "available", lambda: True)
    monkeypatch.setattr(clip_local, "photo_of_a_place", lambda p, exterior=False: pytest.fail("not asked"))
    assert _visuals().verify_subject(_item(tmp_path, place_accurate=False), "Kellie's Castle") is False


def test_without_the_model_vision_is_still_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(clip_local, "available", lambda: False)
    v = _visuals()
    for i in range(M.VISION_GIVE_UP_FRESH):
        it = _item(tmp_path)
        it.id = f"f{i}"
        assert v.verify_subject(it, "Kellie's Castle") == M.NO_ANSWER
    assert v._vision_unavailable


def test_the_label_embeddings_match_the_pinned_model_and_the_label_list():
    data = json.loads(clip_local.LABELS_FILE.read_text(encoding="utf-8"))
    assert data["model"] == f"{clip_local.CLIP_REPO}@{clip_local.CLIP_REV}"
    assert data["text_model_sha256"] == clip_local.CLIP_TEXT_SHA256
    want = [(g, x) for g, xs in clip_local.LABELS.items() for x in xs]
    assert [(r["group"], r["label"]) for r in data["labels"]] == want
    assert all(len(r["embed"]) == 512 for r in data["labels"])


def test_the_dockerfile_pins_the_same_clip_model():
    from pathlib import Path
    df = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
    assert f"ARG CLIP_REV={clip_local.CLIP_REV}" in df and clip_local.CLIP_VISION_SHA256 in df


@pytest.mark.skipif(not clip_local.available(), reason="CLIP model not installed (it is in the Docker image)")
def test_clip_tells_a_photo_of_a_place_from_a_flat_graphic(tmp_path):
    flat = tmp_path / "flat.png"
    Image.new("RGB", (300, 300), (200, 30, 30)).save(flat)
    ok, _ = clip_local.photo_of_a_place(flat)
    assert ok is False
