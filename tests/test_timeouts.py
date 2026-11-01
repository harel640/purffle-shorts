"""The evening Maker run once froze for hours inside edge-tts on a half-open websocket, at 0% CPU
with no error and no data. These cover the hard wall-clock ceiling that now bounds any single call,
the edge-tts guard+retry built on it, and the (connect, read) timeout on streamed downloads."""

import sys
import time
import types

import pytest

from purffle_shorts import tts, utils
from purffle_shorts.config import Settings


# ------------------------------------------------------------------ run_with_timeout
def test_run_with_timeout_returns_value():
    assert utils.run_with_timeout(lambda: 7, 1.0) == 7


def test_run_with_timeout_propagates_the_callables_exception():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        utils.run_with_timeout(boom, 1.0)


def test_run_with_timeout_aborts_a_hang_without_waiting_it_out():
    def hang():
        time.sleep(10)
        return "never"

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        utils.run_with_timeout(hang, 0.2, "sleeper")
    assert time.monotonic() - t0 < 2.0  # gave up near the ceiling, did not block for the full 10s


# ------------------------------------------------------------------ edge-tts guard
def _install_fake_edge(monkeypatch, behaviours):
    """Replace the edge_tts module. ``behaviours`` is one entry per Communicate() call: 'hang' sleeps
    past the ceiling, 'ok' streams a little audio and one word boundary."""
    state = {"n": 0, "kwargs": []}
    mod = types.ModuleType("edge_tts")

    class Communicate:
        def __init__(self, text, voice, **kw):
            self._i = state["n"]
            state["n"] += 1
            state["kwargs"].append(kw)

        def stream_sync(self):
            if behaviours[min(self._i, len(behaviours) - 1)] == "hang":
                time.sleep(3)  # >> the shrunk ceiling the test sets
            yield {"type": "audio", "data": b"\xff\xfb\x90\x00"}
            yield {"type": "WordBoundary", "text": "hi", "offset": 0, "duration": 5_000_000}

    mod.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", mod)
    return state


def test_edge_passes_explicit_connect_and_receive_timeouts(monkeypatch, tmp_path):
    state = _install_fake_edge(monkeypatch, ["ok"])
    path, words, _ = tts._edge("hello world", tmp_path, Settings())
    assert path.exists() and path.stat().st_size > 0 and words[0].text == "hi"
    assert state["kwargs"][0]["connect_timeout"] == tts.EDGE_CONNECT_TIMEOUT
    assert state["kwargs"][0]["receive_timeout"] == tts.EDGE_RECEIVE_TIMEOUT


def test_edge_abandons_a_wedged_attempt_and_the_retry_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "EDGE_SYNTH_TIMEOUT", 0.3)
    state = _install_fake_edge(monkeypatch, ["hang", "ok"])  # 1st attempt wedges, retry works
    path, words, _ = tts._edge("hello", tmp_path, Settings())
    assert state["n"] == 2                       # it really retried after giving up on the hang
    assert path.exists() and path.stat().st_size > 0 and words[0].text == "hi"


def test_edge_is_bounded_even_when_every_attempt_hangs(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "EDGE_SYNTH_TIMEOUT", 0.2)
    _install_fake_edge(monkeypatch, ["hang", "hang"])
    t0 = time.monotonic()
    with pytest.raises(Exception):               # TimeoutError bubbles after the retry also hangs
        tts._edge("hello", tmp_path, Settings())
    assert time.monotonic() - t0 < 10            # bounded in seconds, not hours


# ------------------------------------------------------------------ streamed download timeout
def test_download_uses_a_connect_and_read_timeout(monkeypatch, tmp_path):
    seen = {}

    class _Resp:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, chunk_size=0):
            yield b"data"

    class _Sess:
        def get(self, url, stream=False, timeout=None, headers=None):
            seen["timeout"] = timeout
            return _Resp()

    monkeypatch.setattr(utils, "http", lambda: _Sess())
    dest = tmp_path / "f.bin"
    utils.download("http://example/x", dest)
    assert isinstance(seen["timeout"], tuple) and len(seen["timeout"]) == 2  # (connect, read)
    assert dest.read_bytes() == b"data"
