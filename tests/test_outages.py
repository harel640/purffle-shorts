"""P6: outages do not burn list places; the voice-over is retried, then voiced locally (Kokoro).

A run that ends because something it needed was down (every LLM, every voice, vision, the fact checker, the
network) gives its listed place back; a run that ends because the place broke a rule uses it up. A claim left
pending by a run that was killed frees itself after CLAIM_STALE_HOURS."""

import json
import logging
import sqlite3
import wave
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests

from purffle_shorts import factcheck, safety, tts
from purffle_shorts import pipeline as P
from purffle_shorts import topics as T
from purffle_shorts.config import Settings
from purffle_shorts.history import History
from purffle_shorts.llm import LLMConfigError, LLMError
from purffle_shorts.media import SkipTopic
from purffle_shorts.script import Scene, Script
from purffle_shorts.utils import PermanentError


# ------------------------------------------------------------------ 1. two kinds of failure
@pytest.mark.parametrize("e", [
    tts.VoiceUnavailable("All voice engines failed: edge-tts exceeded 60s"),
    LLMError("All LLM providers failed"),
    SkipTopic("vision unavailable", transient=True),
    SkipTopic("fact check unavailable", transient=True),
    safety.UnsafeTopic("advisory feed unavailable", transient=True),
    requests.ConnectionError("connection reset"),
    TimeoutError("read timed out"),
])
def test_outages_are_transient(e):
    assert P.failure_kind(e) == "transient"


def test_a_network_error_behind_another_error_is_transient():
    try:
        try:
            raise requests.Timeout("wikipedia")
        except requests.Timeout as inner:
            raise RuntimeError("could not build the story") from inner
    except RuntimeError as e:
        assert P.failure_kind(e) == "transient"


@pytest.mark.parametrize("e", [
    SkipTopic("sensitive region (Xinjiang) — skipped by policy"),
    SkipTopic("video would run 38.0s, outside 45-60s after one rewrite"),
    SkipTopic("story not supported by its source"),
    safety.UnsafeTopic("Burma: US State Dept Level 4"),
    factcheck.NothingSupported("no sentence is supported"),
])
def test_rule_skips_are_rules(e):
    assert P.failure_kind(e) == "rule"


def test_other_errors_and_config_errors_are_errors():
    assert P.failure_kind(ValueError("bug")) == "error"
    assert P.failure_kind(LLMConfigError("no API key")) == "error"


# ------------------------------------------------------------------ claims through a real list pick
def _list(tmp_path, *places):
    f = tmp_path / "topic-list.json"
    f.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(),
                             "places": [{"rank": i + 1, "place": p, "region": "Gansu", "country": "China"}
                                        for i, p in enumerate(places)]}), encoding="utf-8")
    return f


def _studio(tmp_path, *places):
    s = Settings(output_dir=str(tmp_path / "o"), data_dir=str(tmp_path / "d"), upload=False, safety_filter=False,
                 topic_list_file=str(_list(tmp_path, *places)), visual_sources=["wikimedia"])
    return P.Studio(s)


def _key(place):
    stamp = datetime.now(timezone.utc).date().isoformat()
    return f"list:{stamp}:{place}|china".lower()


def _drafted(st):
    """draft() as the real one starts: the list pick is claimed (pending)."""
    def draft(*a, **k):
        pick = T.from_list(st.s, st.history)
        st._pick = pick
        return Script(topic=pick.subject, title="t", hook_text="h", description="d",
                      scenes=[Scene("x", "q")]), pick, "test"
    return draft


@pytest.mark.parametrize("error, released", [
    (tts.VoiceUnavailable("All voice engines failed"), True),
    (LLMError("All LLM providers failed"), True),
    (SkipTopic("vision unavailable", transient=True), True),
    (SkipTopic("fact check unavailable: x", transient=True), True),
    (requests.ConnectionError("reset"), True),
    (SkipTopic("sensitive region (Tibet)"), False),
    (SkipTopic("not a hidden natural/historic place: looks like a zoo"), False),
    (SkipTopic("place 'Hampi' was already made in the last 90 days"), False),
    (SkipTopic("topic centred on recent violence"), False),
    (SkipTopic("video would run 38.0s, outside 45-60s"), False),
    (SkipTopic("story not supported by its source"), False),
    (safety.UnsafeTopic("Level 4"), False),
])
def test_the_claim_is_given_back_only_after_an_outage(tmp_path, monkeypatch, caplog, error, released):
    st = _studio(tmp_path, "Mogao Caves", "Yungang Grottoes")
    monkeypatch.setattr(st, "draft", _drafted(st))

    def produce(*a, **k):
        raise error
    monkeypatch.setattr(st, "produce", produce)
    caplog.set_level(logging.INFO, logger="purffle")
    r = st.make()
    assert not r.ok
    assert st.history.topic_used(_key("Mogao Caves")) is not released
    assert not st.history.topic_used(_key("Yungang Grottoes"))         # no second place was tried
    ended = [x.getMessage() for x in caplog.records if x.getMessage().startswith("Run ended by a")]
    assert len(ended) == 1 and ("TRANSIENT" if released else "RULE") in ended[0]
    if not released:                                                   # used for good, not left pending
        row = sqlite3.connect(st.history.path).execute(
            "SELECT pending FROM topics_used WHERE key = ?", (_key("Mogao Caves"),)).fetchone()
        assert row == (None,)


def test_a_wikipedia_outage_in_the_draft_stage_gives_the_place_back(tmp_path, monkeypatch):
    st = _studio(tmp_path, "Mogao Caves")

    def down(*a, **k):
        raise requests.ConnectionError("wikipedia down")
    pick =T.Topic("Mogao Caves, Gansu, China", "list", key=_key("Mogao Caves"), wiki_title="Mogao Caves",
                   place="Mogao Caves", country="China")
    st.history.claim_topic(pick.key)
    monkeypatch.setattr(P, "pick_topic", lambda *a, **k: pick)
    monkeypatch.setattr(T, "article_page", lambda *a, **k: down())
    st._llm = object()
    r = st.make()
    assert not r.ok and "could not be reached" in r.error
    assert not st.history.topic_used(pick.key)


def test_a_missing_article_uses_the_place_up(tmp_path, monkeypatch):
    st = _studio(tmp_path, "Mogao Caves")
    pick = T.Topic("Nowhere, Gansu, China", "list", key=_key("Nowhere"), wiki_title="Nowhere",
                   place="Nowhere", country="China")
    st.history.claim_topic(pick.key)
    monkeypatch.setattr(P, "pick_topic", lambda *a, **k: pick)
    monkeypatch.setattr(T, "article_page", lambda *a, **k: None)
    st._llm = object()
    r = st.make()
    assert not r.ok and "not found" in r.error
    assert st.history.topic_used(pick.key)


def test_a_made_video_uses_the_place_up(tmp_path, monkeypatch):
    st = _studio(tmp_path, "Mogao Caves")
    monkeypatch.setattr(st, "draft", _drafted(st))
    monkeypatch.setattr(st, "produce", lambda *a, **k: P.Result(True, status="rendered"))
    assert st.make().ok
    row = sqlite3.connect(st.history.path).execute(
        "SELECT pending FROM topics_used WHERE key = ?", (_key("Mogao Caves"),)).fetchone()
    assert row == (None,)


# ------------------------------------------------------------------ 2. a stale claim is reused
def _age(h: History, key: str, hours: float):
    when = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    with sqlite3.connect(h.path) as c:
        c.execute("UPDATE topics_used SET used_at = ? WHERE key = ?", (when, key))


def test_a_pending_claim_older_than_two_hours_with_no_video_is_free_again(tmp_path):
    h = History(tmp_path / "h.db")
    h.claim_topic("list:x:a|china")
    assert h.topic_used("list:x:a|china")                     # a run is making it right now
    _age(h, "list:x:a|china", 3)
    assert not h.topic_used("list:x:a|china")                 # that run was killed: free again


def test_a_stale_claim_with_a_video_row_stays_used(tmp_path):
    h = History(tmp_path / "h.db")
    h.claim_topic("list:x:a|china")
    h.add_video(status="rendered", title="t", topic_key="list:x:a|china")
    _age(h, "list:x:a|china", 3)
    assert h.topic_used("list:x:a|china")


def test_finished_and_old_style_claims_never_expire(tmp_path):
    h = History(tmp_path / "h.db")
    h.claim_topic("list:x:a|china")
    h.finish_topic("list:x:a|china")
    h.mark_topic("niche:b")
    for k in ("list:x:a|china", "niche:b"):
        _age(h, k, 500)
        assert h.topic_used(k)


def test_from_list_takes_a_stale_claim_again(tmp_path):
    st = _studio(tmp_path, "Mogao Caves", "Yungang Grottoes")
    first = T.from_list(st.s, st.history)
    _age(st.history, first.key, 3)
    again = T.from_list(st.s, st.history)
    assert again.place == "Mogao Caves"


def test_an_old_database_gets_the_new_columns(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE topics_used (key TEXT PRIMARY KEY, used_at TEXT NOT NULL)")
        c.execute("INSERT INTO topics_used VALUES ('list:old:a|china', '2026-10-01T00:00:00+00:00')")
    h = History(db)
    assert h.topic_used("list:old:a|china")                   # an old claim stays used


# ------------------------------------------------------------------ 3. the voice-over is retried, then voiced locally
def _speech(engine="edge", voice="v"):
    return SimpleNamespace(engine=engine, voice=voice, duration=50.0, words=[])


def _voice_studio(tmp_path, monkeypatch, outcomes, **kw):
    """Studio whose tts.synthesize follows ``outcomes`` per engine: a list of "fail"/"ok" consumed in order."""
    monkeypatch.setattr(P, "VOICE_RETRY_WAIT", 0)
    calls = []

    def synth(text, out, cfg):
        calls.append(cfg.tts_engine)
        todo = outcomes.get(cfg.tts_engine, ["fail"])
        step = todo.pop(0) if todo else "fail"
        if step == "missing":
            raise PermanentError("not installed")
        if step == "fail":
            raise TimeoutError(f"{cfg.tts_engine} exceeded 60s")
        return _speech(cfg.tts_engine)
    monkeypatch.setattr(tts, "synthesize", synth)
    st = P.Studio(Settings(output_dir=str(tmp_path / "o"), data_dir=str(tmp_path / "d"), **kw))
    script = Script(topic="t", title="t", hook_text="h", description="d", scenes=[Scene("Hello there.", "q")])
    return st, script, calls


def test_tts_fails_twice_then_succeeds_and_the_video_continues(tmp_path, monkeypatch, caplog):
    st, script, calls = _voice_studio(tmp_path, monkeypatch, {"edge": ["fail", "fail", "ok"]})
    caplog.set_level(logging.INFO, logger="purffle")
    sp = st._speak(script, tmp_path, st.s)
    assert sp.engine == "edge" and calls == ["edge"] * 3
    assert any(r.getMessage().startswith("Voice used: edge") for r in caplog.records)


def test_edge_fails_three_times_and_kokoro_voices_it(tmp_path, monkeypatch, caplog):
    st, script, calls = _voice_studio(tmp_path, monkeypatch, {"edge": ["fail"] * 3, "kokoro": ["ok"]})
    caplog.set_level(logging.INFO, logger="purffle")
    sp = st._speak(script, tmp_path, st.s)
    assert sp.engine == "kokoro" and calls == ["edge", "edge", "edge", "kokoro"]
    used = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Voice used:")]
    assert used == ["Voice used: kokoro (v) - fallback, edge failed"]


def test_every_voice_fails_and_the_run_ends_as_an_outage(tmp_path, monkeypatch):
    st, script, calls = _voice_studio(tmp_path, monkeypatch, {"edge": ["fail"] * 3, "kokoro": ["missing"]})
    with pytest.raises(tts.VoiceUnavailable) as e:
        st._speak(script, tmp_path, st.s)
    assert calls == ["edge", "edge", "edge", "kokoro"] and P.failure_kind(e.value) == "transient"


def test_tts_fails_three_times_the_claim_is_released_and_no_second_place_is_tried(tmp_path, monkeypatch):
    st = _studio(tmp_path, "Mogao Caves", "Yungang Grottoes")
    monkeypatch.setattr(st, "draft", _drafted(st))
    monkeypatch.setattr(P, "VOICE_RETRY_WAIT", 0)
    tries = []

    def synth(text, out, cfg):
        tries.append(cfg.tts_engine)
        raise TimeoutError("edge-tts exceeded 60s")
    monkeypatch.setattr(tts, "synthesize", synth)
    st.s.tts_fallback = []

    def produce(script, pick, *a, **k):
        st._speak(script, tmp_path, st.s)
    monkeypatch.setattr(st, "produce", produce)
    r = st.make()
    assert not r.ok and "All voice engines failed" in r.error and tries == ["edge"] * 3
    assert not st.history.topic_used(_key("Mogao Caves"))       # given back
    assert not st.history.topic_used(_key("Yungang Grottoes"))  # never reached


def test_kokoro_forced_is_used_first(tmp_path, monkeypatch):
    st, script, calls = _voice_studio(tmp_path, monkeypatch, {"kokoro": ["ok"]}, tts_engine="kokoro")
    assert st._speak(script, tmp_path, st.s).engine == "kokoro" and calls == ["kokoro"]


def test_tts_fallback_setting(monkeypatch):
    monkeypatch.delenv("TTS_FALLBACK", raising=False)
    assert Settings.from_env().tts_fallback == ["kokoro"]
    monkeypatch.setenv("TTS_FALLBACK", "none")
    assert Settings.from_env().tts_fallback == []


# ------------------------------------------------------------------ the local Kokoro voice (kokoro_local)
def test_espeak_ipa_is_mapped_to_kokoro_phonemes():
    from purffle_shorts import kokoro_local as K
    # espeak-ng 1.52 `--ipa --tie=^ -v en-us`, "Why? joy, eight, smoke, chair"
    assert K.espeak_to_kokoro("wˈa^ɪ") == "wˈI"
    assert K.espeak_to_kokoro("d^ʒˈɔ^ɪ") == "ʤˈY"
    assert K.espeak_to_kokoro("ˈe^ɪt") == "ˈAt"
    assert K.espeak_to_kokoro("smˈo^ʊk") == "smˈOk"
    assert K.espeak_to_kokoro("t^ʃˈɛɹ") == "ʧˈɛɹ"
    assert K.espeak_to_kokoro("ɐ sˈiːld") == "ə sˈild"          # length marks dropped, ɐ -> ə


def test_punctuation_is_kept_for_the_pauses():
    from purffle_shorts import kokoro_local as K
    fake = {"In 1900 he followed smoke": "ɪn nˈa^ɪntiːn", "to a wall": "tʊ ɐ wˈɔːl", "Why": "wˈa^ɪ"}
    ps = K.phonemize("In 1900 he followed smoke, to a wall. Why?", ipa=lambda x: fake[x])
    assert ps == "ɪn nˈIntin, tʊ ə wˈɔl. wˈI?"
    vocab = {c: i for i, c in enumerate(";:,.!?—… ɪnˈItəwɔl")}
    assert all(isinstance(i, int) for i in K.tokens(ps, vocab)) and len(K.tokens("x" * 600, {"x": 1})) == 510


def test_only_known_voice_names_are_used(monkeypatch):
    from purffle_shorts import kokoro_local as K
    monkeypatch.delenv("KOKORO_VOICE", raising=False)
    assert K.voice_name() == "am_michael" and K.voice_name("bm_george") == "bm_george"
    assert K.voice_name("../../etc/passwd") == "am_michael"


def test_kokoro_voices_each_sentence_and_times_the_words_by_sentence(monkeypatch, tmp_path):
    import numpy as np

    from purffle_shorts import kokoro_local as K
    lengths = {"In 1900 Wang found a cave.": 1.0, "Behind the wall lay scrolls!": 2.0, "Why?": 0.5}
    said = []
    sr = K.SAMPLE_RATE

    def speak_all(sentences, voice, speed, work):
        said.append((list(sentences), voice, speed))
        # as Kokoro returns them: ~0.4 s of silence before each sentence and ~0.6 s after it
        return [np.concatenate([np.zeros(int(0.4 * sr)), np.full(int(lengths[s] * sr), 0.1),
                                np.zeros(int(0.6 * sr))]).astype(np.float32) for s in sentences]
    monkeypatch.setattr(K, "speak_all", speak_all)
    monkeypatch.setattr(K, "available", lambda voice="": True)
    monkeypatch.delenv("KOKORO_SPEED", raising=False)
    path, words, voice = tts._kokoro("In 1900 Wang found a cave. Behind the wall lay scrolls! Why?", tmp_path,
                                     Settings())
    assert said == [(list(lengths), "am_michael", tts.KOKORO_SPEED)]
    assert voice == "kokoro am_michael"
    with wave.open(str(path)) as w:
        total = w.getnframes() / w.getframerate()
    gap, pad = tts.KOKORO_SENTENCE_GAP, K.LEAD_KEEP + K.TAIL_KEEP      # the padding is trimmed to a breath
    assert abs(total - (3.5 + 3 * pad + 2 * gap)) < 0.02
    starts = {w.text: w.start for w in words}
    behind = 1.0 + pad + gap
    assert starts["In"] < 0.2 and abs(starts["Behind"] - behind) < 0.2 and starts["Why?"] >= 3.0 + 2 * gap
    assert [w.start for w in words] == sorted(w.start for w in words)


def test_the_local_kokoro_is_english_only(monkeypatch, tmp_path):
    from purffle_shorts import kokoro_local as K
    monkeypatch.setattr(K, "available", lambda voice="": True)
    with pytest.raises(PermanentError, match="English only"):
        tts._kokoro("Hola.", tmp_path, Settings(language="es"))


def test_without_the_model_kokoro_needs_its_package(monkeypatch, tmp_path):
    from purffle_shorts import kokoro_local as K
    monkeypatch.setattr(K, "available", lambda voice="": False)
    import sys
    monkeypatch.setitem(sys.modules, "kokoro", None)          # the optional package is not installed
    with pytest.raises(PermanentError, match="KOKORO_DIR"):
        tts._kokoro("Hello.", tmp_path, Settings())


def test_the_voice_cache_does_not_reuse_kokoro_audio_made_at_another_pace(monkeypatch, tmp_path):
    s = Settings(data_dir=str(tmp_path), tts_engine="kokoro")
    monkeypatch.delenv("KOKORO_SPEED", raising=False)
    a = tts._cache_dir("Hello there.", s, "kokoro")
    monkeypatch.setenv("KOKORO_SPEED", "0.9")
    b = tts._cache_dir("Hello there.", s, "kokoro")
    monkeypatch.setattr(tts, "KOKORO_AUDIO_VERSION", "old")
    monkeypatch.delenv("KOKORO_SPEED")
    c = tts._cache_dir("Hello there.", s, "kokoro")
    assert len({a, b, c}) == 3
