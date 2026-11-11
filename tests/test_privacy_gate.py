"""P5: a missing YT_PRIVACY must never turn into a public video, and the log says what YouTube really applied."""

import logging

import pytest

from purffle_shorts import pipeline as P
from purffle_shorts import youtube as Y
from purffle_shorts.config import Settings


def settings(tmp_path, **kw):
    kw.setdefault("visual_sources", ["wikimedia"])
    return Settings(output_dir=str(tmp_path / "o"), data_dir=str(tmp_path / "d"), **kw)


def test_unset_privacy_is_the_default_and_means_refuse(monkeypatch):
    monkeypatch.delenv("YT_PRIVACY", raising=False)
    s = Settings.from_env()
    assert s.privacy == ""
    with pytest.raises(Y.PrivacyNotSet, match="YT_PRIVACY is not set"):
        Y.require_privacy(s)


def test_blank_env_value_is_unset(monkeypatch):
    monkeypatch.setenv("YT_PRIVACY", "   ")
    with pytest.raises(Y.PrivacyNotSet):
        Y.require_privacy(Settings.from_env())


@pytest.mark.parametrize("value", ["private", "Unlisted", "PUBLIC"])
def test_valid_values_pass(monkeypatch, value):
    monkeypatch.setenv("YT_PRIVACY", value)
    assert Y.require_privacy(Settings.from_env()) == value.lower()


def test_invalid_value_is_refused():
    with pytest.raises(Y.PrivacyNotSet, match="not valid"):
        Y.require_privacy(Settings(privacy="friends"))


def test_privacy_error_is_a_clean_cli_error():
    assert issubclass(Y.PrivacyNotSet, Y.NotAuthorized)     # the CLI prints one line, the pipeline re-raises


def test_upload_refuses_before_touching_the_network(monkeypatch, tmp_path):
    monkeypatch.setattr(Y, "service", lambda *a, **k: pytest.fail("must not reach the YouTube API"))
    with pytest.raises(Y.PrivacyNotSet):
        Y.upload(Settings(privacy=""), tmp_path / "v.mp4", {})


def test_make_refuses_before_claiming_or_rendering(tmp_path):
    st = P.Studio(settings(tmp_path, upload=True, privacy=""))
    with pytest.raises(Y.PrivacyNotSet):
        st.make(topic="x")
    with pytest.raises(Y.PrivacyNotSet):
        st.autopilot(once=True)


def test_no_upload_runs_do_not_need_privacy(tmp_path, monkeypatch):
    st = P.Studio(settings(tmp_path, upload=False, privacy=""))
    monkeypatch.setattr(Y, "require_privacy", lambda s: pytest.fail("not needed without an upload"))
    try:
        st.make(topic="x", upload=False)
    except Exception as e:
        assert not isinstance(e, Y.PrivacyNotSet)


def _rec(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    return {"title": "T", "description": "d", "tags": "[]", "video_path": str(f)}


@pytest.mark.parametrize("returned, requested, warns", [
    ("private", "private", False),
    ("public", "private", True),           # YouTube applied something else: say so loudly
])
def test_the_log_carries_the_privacy_youtube_returned(tmp_path, monkeypatch, caplog, returned, requested, warns):
    st = P.Studio(settings(tmp_path, privacy=requested))
    monkeypatch.setattr(Y, "upload", lambda s, v, b: {"id": "AAA111", "status": {"privacyStatus": returned}})
    monkeypatch.setattr(st.history, "update_video", lambda *a, **k: None)
    caplog.set_level(logging.INFO)
    status = st._to_youtube(1, _rec(tmp_path), {}, tmp_path, None, None)
    assert status == "uploaded"
    assert f"Uploaded ({returned}): https://youtube.com/shorts/AAA111" in caplog.text
    assert ("ALERT: YouTube reports privacy" in caplog.text) is warns


def test_missing_status_in_the_response_is_logged_as_unknown(tmp_path, monkeypatch, caplog):
    st = P.Studio(settings(tmp_path, privacy="private"))
    monkeypatch.setattr(Y, "upload", lambda s, v, b: {"id": "BBB"})
    monkeypatch.setattr(st.history, "update_video", lambda *a, **k: None)
    caplog.set_level(logging.INFO)
    st._to_youtube(1, _rec(tmp_path), {}, tmp_path, None, None)
    assert "Uploaded (unknown): https://youtube.com/shorts/BBB" in caplog.text
