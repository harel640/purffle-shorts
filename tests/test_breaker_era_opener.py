"""Gemini circuit breaker, era-aware AI look, no gore in AI prompts, exterior opener and the statue cap."""

import pytest

from purffle_shorts import llm as L
from purffle_shorts import media as M
from purffle_shorts.config import Settings
from purffle_shorts.media import MediaItem, Visuals


# ------------------------------------------------------------------ 1. circuit breaker
class _P:
    """A provider double: ``script`` is a list of replies/exceptions, consumed one per call."""
    def __init__(self, name, script):
        self.name, self.model, self.script, self.calls = name, "m", list(script), 0

    def __str__(self):
        return f"{self.name}:m"

    def complete(self, system, user, **k):
        self.calls += 1
        r = self.script.pop(0) if self.script else '{"ok": true}'
        if isinstance(r, Exception):
            raise r
        return r


def _llm(gemini, groq):
    obj = L.LLM.__new__(L.LLM)
    obj.settings, obj.last_used, obj.providers = Settings(), "", [gemini, groq]
    return obj


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L, "with_retries", lambda fn, attempts=2, **k: _retry(fn, attempts))


def _retry(fn, attempts):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise


def test_gemini_is_skipped_for_the_rest_of_the_run_after_two_consecutive_503s(caplog):
    busy = RuntimeError("gemini:m: HTTP 503 — high demand")
    gem, groq = _P("gemini", [busy] * 20), _P("groq", [])
    llm = _llm(gem, groq)
    with caplog.at_level("WARNING", logger="purffle"):
        assert llm.complete_json("s", "u") == {"ok": True}      # call 1: gemini fails twice, groq answers
        first = gem.calls
        for _ in range(3):
            assert llm.complete_json("s", "u") == {"ok": True}
    assert first == 2 and gem.calls == 2                         # never asked again once the circuit is open
    assert groq.calls == 4 and llm.circuit_open("gemini")
    opened = [r for r in caplog.records if "Gemini circuit open" in r.getMessage()]
    assert len(opened) == 1                                      # logged exactly once


def test_timeouts_and_429s_count_too_but_a_bad_request_does_not():
    for err in (RuntimeError("Read timed out. (read timeout=30)"), RuntimeError("HTTP 429 quota")):
        gem, groq = _P("gemini", [err] * 4), _P("groq", [])
        llm = _llm(gem, groq)
        llm.complete_json("s", "u")
        assert llm.circuit_open("gemini")
    gem, groq = _P("gemini", [RuntimeError("HTTP 400 bad schema")] * 6), _P("groq", [])
    llm = _llm(gem, groq)
    llm.complete_json("s", "u")
    llm.complete_json("s", "u")
    assert not llm.circuit_open("gemini")                        # a 400 is the request's fault, not an outage


def test_a_success_resets_the_count():
    busy = RuntimeError("HTTP 503")
    gem = _P("gemini", [busy, '{"ok": true}', busy, '{"ok": true}'])     # fail, ok, fail, ok: never 2 in a row
    llm = _llm(gem, _P("groq", []))
    for _ in range(2):
        assert llm.complete_json("s", "u") == {"ok": True}
    assert not llm.circuit_open("gemini") and gem.calls == 4


def test_an_open_circuit_is_still_the_last_resort():
    busy = RuntimeError("HTTP 503")
    gem = _P("gemini", [busy, busy, '{"from": "gemini"}'])
    groq = _P("groq", [RuntimeError("groq down"), RuntimeError("groq down")])
    llm = _llm(gem, groq)
    llm.note_failure("gemini", busy)
    llm.note_failure("gemini", busy)                             # circuit open before the call
    gem.script = ['{"from": "gemini"}']
    assert llm.complete_json("s", "u") == {"from": "gemini"}     # groq failed, so open gemini is tried after all
    assert groq.calls == 2


def test_gemini_calls_use_a_30s_timeout():
    assert L.GEMINI_TIMEOUT == 30.0


def test_vision_skips_gemini_once_its_circuit_is_open(monkeypatch, tmp_path):
    from PIL import Image
    img = tmp_path / "x.jpg"
    Image.new("RGB", (32, 32)).save(img)
    urls = []

    class Sess:
        def post(self, url, json=None, headers=None, timeout=None):
            urls.append((url, timeout))
            class R:
                status_code = 200
                text = ""
                def json(self_):
                    return {"result": {"description": "Yes"}}
            return R()

    monkeypatch.setattr(L, "http", lambda: Sess())
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    llm = L.LLM.__new__(L.LLM)
    llm.settings = Settings(vision_model="gemini-x", cloudflare_account_id="a", cloudflare_api_token="t")
    llm.note_failure("gemini", RuntimeError("HTTP 503"))
    llm.note_failure("gemini", RuntimeError("HTTP 503"))
    assert llm.image_shows(img, "a temple") is True
    assert all("generativelanguage" not in u for u, _ in urls)   # straight to Cloudflare


# ------------------------------------------------------------------ 3. nothing gory in an AI prompt
@pytest.mark.parametrize("bad", ["skeletons lying near a grave", "blood on the floor", "a corpse in the road",
                                 "burial sites with ancient bones", "a funeral pyre", "bodies on the battlefield",
                                 "wounded soldiers", "a cemetery"])
def test_gore_and_funerary_clauses_are_dropped_from_an_ai_prompt(bad):
    out = M.period_prompt(f"Villagers carving stone tools, {bad}, under a rock overhang")
    assert out == "Villagers carving stone tools, under a rock overhang"


def test_a_scene_about_burials_alone_is_not_drawn():
    assert M.period_prompt("Burial sites with skeletons and bones") == ""


# ------------------------------------------------------------------ 2. era-aware look
ARTICLE = ("Dambulla cave temple is a World Heritage Site. The first temple was built in the 1st century BCE by "
           "King Valagamba. It was inscribed by UNESCO in 1991 and restored in 2010.")


def test_built_year_comes_from_what_the_article_says_was_built():
    assert M.built_year(ARTICLE) == -100                          # 1st c. BCE, earliest edge; UNESCO/restoration ignored
    assert M.built_year("The fort was founded in 1565 and rebuilt in 1700.") == 1565
    assert M.built_year("A village in a valley.") is None


@pytest.mark.parametrize("text, expect", [
    ("Centuries earlier, prehistoric peoples inhabited the cliffs", True),
    ("Long before the temple, hunter-gatherers sheltered here", True),
    ("In the 3rd century BCE monks first lived in these caves", True),
    ("Valagamba fled and hid in the caves in 89 BCE", False),         # inside the century the temple was built
    ("In the 18th century Kandyan kings added murals", False),
    ("Pilgrims climb the steps today", False)])
def test_scenes_set_before_the_place_existed(text, expect):
    assert M.scene_precedes_place(text, -100) is expect


def _visuals(monkeypatch, n=8, shows=lambda s: True):
    class _L:
        def image_shows(self, path, subject):
            return shows(subject)

    v = Visuals(Settings(visual_sources=["wikimedia"], ai_scene_source="pollinations"), llm=_L())
    pool = [MediaItem("wikimedia", f"p{i}", "image", url=f"https://c/{i}.jpg", width=1600, height=1200,
                      author="A", license="CC BY-SA 4.0", license_url="u", page_url=f"u{i}",
                      text=f"Dambulla cave temple {i}", extra={"wiki_title": f"D{i}"}) for i in range(1, n + 1)]
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"{item.id}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool))
    monkeypatch.setattr(v, "_wiki_search", lambda q: [])
    return v, pool


TIBETAN = "Tabo is a Tibetan Buddhist monastery in Spiti, built in 996 CE."


def test_a_scene_before_the_place_gets_a_neutral_look_and_later_ones_the_places_look(monkeypatch, tmp_path):
    v, _ = _visuals(monkeypatch)
    seen = {}

    def fake_ai(prompt, dest_dir, idx, arch="", clothing=""):
        seen[idx] = (arch, clothing)
        return MediaItem("cloudflare", f"ai{idx}", "image", path=dest_dir / f"ai{idx}.jpg", width=1080,
                         height=1920, query=prompt)

    monkeypatch.setattr(v, "_historical_image", fake_ai)
    jobs = [("Tabo", "", False), ("cliffs", "people in a valley", True), ("hall", "monks at prayer", True),
            ("Tabo", "", False)]
    texts = ["Tabo monastery stands in Spiti.", "Prehistoric peoples lived on these cliffs.",
             "In 996 CE monks built the great hall.", "Today pilgrims visit."]
    v.for_place_scenes(jobs, "Tabo Monastery", "India", tmp_path / "m", min_images=3, lat=32.2, lon=78.0,
                       article=TIBETAN, title="Tabo", scene_texts=texts)
    assert seen[1] == (M.NEUTRAL_LOOK["architecture"], M.NEUTRAL_LOOK["clothing"])    # before it existed
    assert "Tibetan" in seen[2][0] and "maroon" in seen[2][1]                          # after: its own look
    assert "stupa" not in seen[1][0].lower() and "shelters" in seen[1][0]


# ------------------------------------------------------------------ 4. exterior opener, statue cap
def _cand(i, title, w=1600, h=1200):
    return MediaItem("wikimedia", f"c{i}", "image", url=f"https://c/{i}.jpg", width=w, height=h, author="A",
                     license="CC BY-SA 4.0", license_url="u", page_url=f"u{i}", text=title,
                     extra={"wiki_title": title})


def test_the_opener_prefers_the_exterior_over_an_interior_or_a_statue(monkeypatch, tmp_path):
    asked = []
    v = Visuals(Settings(visual_sources=["wikimedia"]), llm=type("L", (), {
        "image_shows": lambda self, p, s: asked.append(s) or True})())
    pool = [_cand(1, "Dambulla cave temple Buddha statue", 2400, 1800),          # big, but a statue
            _cand(2, "Dambulla cave temple interior mural", 2400, 1800),
            _cand(3, "Dambulla cave temple exterior view", 1600, 1200)]
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"{item.id}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    got = v._open_landmark_image(pool, M._terms("Dambulla cave temple"), "Dambulla Cave Temple", "Sri Lanka",
                                 tmp_path, subject="the exterior of Dambulla Cave Temple")
    assert got.id == "c3" and asked == ["the exterior of Dambulla Cave Temple"]


def test_the_opener_is_vision_checked_as_the_exterior_of_the_place(monkeypatch, tmp_path):
    asked = []
    v, _ = _visuals(monkeypatch, shows=lambda s: asked.append(s) or True)
    v.for_place_scenes([("Dambulla", "", False)] * 3, "Dambulla Cave Temple", "Sri Lanka", tmp_path / "m",
                       min_images=3, lat=7.8, lon=80.6)
    assert asked[0] == "the exterior of Dambulla Cave Temple"


def test_an_opener_that_is_not_the_exterior_is_rejected_and_the_next_tried(monkeypatch, tmp_path):
    v, _ = _visuals(monkeypatch, shows=lambda s: not s.startswith("the exterior") or False)
    # nothing ever shows "the exterior": the opener can't be verified, so the topic is skipped, not faked
    with pytest.raises(M.SkipTopic, match="no vision-verified photo"):
        v.for_place_scenes([("Dambulla", "", False)] * 3, "Dambulla Cave Temple", "Sri Lanka", tmp_path / "m",
                           min_images=3, lat=7.8, lon=80.6)


def test_at_most_two_scenes_show_a_statue(monkeypatch, tmp_path):
    v = Visuals(Settings(visual_sources=["wikimedia"]), llm=type("L", (), {
        "image_shows": lambda self, p, s: True})())
    pool = ([_cand(i, f"Dambulla cave temple Buddha statue {i}") for i in range(1, 6)]
            + [_cand(i, f"Dambulla cave temple view {i}") for i in range(6, 12)])
    monkeypatch.setattr(v, "_fetch", lambda item, d, i: (setattr(item, "path", d / f"{item.id}.jpg") or item))
    monkeypatch.setattr(M, "_low_quality_image", lambda p: False)
    monkeypatch.setattr(M, "geosearch_images", lambda *a, **k: list(pool))
    monkeypatch.setattr(v, "_wiki_search", lambda q: [])
    jobs = [("Dambulla exterior", "", False)] + [("Buddha statue", "", False)] * 6 + [("Dambulla view", "", False)]
    out = v.for_place_scenes(jobs, "Dambulla Cave Temple", "Sri Lanka", tmp_path / "m", min_images=3,
                             lat=7.8, lon=80.6)
    assert sum(1 for m in out if M._STATUE_RE.search(m.text)) <= M.MAX_STATUE_SCENES
    assert len(out) == len(jobs)


def test_statue_and_exterior_patterns():
    for s in ("Buddha statue", "Reclining Buddha", "Standing idol", "stone sculpture", "bodhisattva figure"):
        assert M._STATUE_RE.search(s), s
    for s in ("Dambulla cave temple exterior", "Golden Temple entrance view", "aerial view"):
        assert M._EXTERIOR_RE.search(s) and not M._STATUE_RE.search(s), s
    assert M._INTERIOR_RE.search("Cave 2 interior") and M._INTERIOR_RE.search("ceiling frescoes")
