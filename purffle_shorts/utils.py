"""Small shared helpers: retries, HTTP, slugs, text cleanup."""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import requests

log = logging.getLogger("purffle")

T = TypeVar("T")

USER_AGENT = "PurffleShorts/3.0 (+https://github.com/Chamanrajragu/purffle-shorts)"


class PermanentError(Exception):
    """An error that retrying will not fix (bad key, 4xx, invalid input)."""


class Cancelled(Exception):
    """The user stopped the job (Studio "Cancel"): raised at the next stage boundary or ffmpeg step."""


# Set by the Studio to stop the running job. Jobs run one at a time, so one flag for the process is enough.
CANCEL = threading.Event()


def check_cancel() -> None:
    if CANCEL.is_set():
        raise Cancelled("cancelled")


def emphasis_key(word: str) -> str:
    """How a caption word is compared for highlighting: lower case, without the punctuation around it."""
    return re.sub(r"^\W+|\W+$", "", word.lower(), flags=re.UNICODE)


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 2.0,
    label: str = "operation",
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``fn`` with exponential backoff + jitter. PermanentError is never retried."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except PermanentError:
            raise
        except retry_on as e:
            if i == attempts:
                log.error("%s failed after %d attempts: %s", label, attempts, redact(e))
                raise
            wait = base_delay * (2 ** (i - 1)) + random.uniform(0, 0.5)
            log.warning("%s attempt %d/%d failed: %s — retrying in %.1fs", label, i, attempts, redact(e), wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_with_timeout(fn: Callable[[], T], seconds: float, label: str = "operation") -> T:
    """Run ``fn`` in a background thread and give up on it after ``seconds``.

    A network library that ignores its own timeouts can still wedge a run forever: edge-tts has
    stalled on a half-open websocket at 0% CPU with no error and no data, outliving its own
    connect/receive timeouts. This puts a hard ceiling on any single call so the step fails fast and
    can be retried. The worker is a daemon thread, so an abandoned (still-hung) call never blocks the
    process from exiting. Raises TimeoutError on overrun (retryable); otherwise returns ``fn``'s
    result or re-raises whatever it raised."""
    box: dict[str, object] = {}

    def _worker() -> None:
        try:
            box["ok"] = fn()
        except BaseException as e:  # noqa: BLE001 — carried across the thread boundary and re-raised below
            box["err"] = e

    t = threading.Thread(target=_worker, name=f"timeout:{label}", daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"{label} exceeded {seconds:.0f}s")
    if "err" in box:
        raise box["err"]  # type: ignore[misc]
    return box["ok"]  # type: ignore[return-value]


_session: requests.Session | None = None


def http() -> requests.Session:
    """A shared requests session with a descriptive User-Agent."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _session = s
    return _session


def raise_for_status(resp: requests.Response, what: str) -> None:
    """Raise PermanentError for 4xx (except 408/429) and a retryable error otherwise."""
    if resp.status_code < 400:
        return
    body = resp.text[:300].replace("\n", " ")
    msg = f"{what}: HTTP {resp.status_code} — {body}"
    if 400 <= resp.status_code < 500 and resp.status_code not in (408, 409, 429):
        raise PermanentError(msg)
    raise RuntimeError(msg)


def download(url: str, dest: Path, *, headers: dict | None = None,
             timeout: tuple[int, int] | int = (10, 60),
             max_bytes: int = 400 * 1024 * 1024) -> Path:
    """Stream ``url`` to ``dest`` atomically (via a .part file), with retries.

    ``timeout`` is a (connect, read) pair so a dead host fails the connect in seconds and a server
    that opens the socket then goes silent fails the read, instead of hanging the job. A bare int
    sets both phases to the same value (what ``requests`` already did)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _do() -> Path:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with http().get(url, stream=True, timeout=timeout, headers=headers or {}) as r:
            raise_for_status(r, f"download {url[:80]}")
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        size += len(chunk)
                        if size > max_bytes:
                            raise PermanentError(f"download too large (> {max_bytes >> 20} MB): {url[:80]}")
                        f.write(chunk)
        if size == 0:
            raise RuntimeError(f"empty download: {url[:80]}")
        os.replace(tmp, dest)
        return dest

    return with_retries(_do, attempts=3, label=f"download {dest.name}")


_SECRET_RE = re.compile(r"(?i)\b((?:api_?)?key|token|access_token)=[^&\s'\"]+")
# Some services echo your IP in error bodies (e.g. "Queue full for IP: ..."); logs get pasted into issues.
_IP_RE = re.compile(r"(?i)(?<![\w:.])(?:(?:[0-9a-f]{1,4}:){4,7}[0-9a-f]{1,4}|(?:\d{1,3}\.){3}\d{1,3})(?![\w:.])")


# Webhook URLs carry their secret in the path.
_HOOK_RE = re.compile(r"(?i)(/api/webhooks/\d+/|hooks\.slack\.com/services/)[\w/-]+")


def redact(text: object) -> str:
    """Hide API keys that some services (e.g. Pixabay) put in URLs, webhook tokens and IP addresses
    before they reach a log."""
    t = _HOOK_RE.sub(r"\1***", _SECRET_RE.sub(r"\1=***", str(text)))
    return _IP_RE.sub("<ip>", t)


def slugify(text: str, max_len: int = 48) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:max_len].rstrip("-")) or "short"


_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D\U00002190-\U000021FF\U00002B00-\U00002BFF]+"
)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def clean_narration(text: str) -> str:
    """Remove things a TTS voice should never read aloud: stage directions, labels, emoji, hashtags, markdown."""
    t = strip_emoji(text)
    t = re.sub(r"\[[^\]]*\]", " ", t)                      # [Hook], [Music]
    t = re.sub(r"\((?:pause|beat|music|sfx|laughs?)[^)]*\)", " ", t, flags=re.I)
    t = re.sub(r"^\s*(?:narrator|voice ?over|vo|host|scene \d+|hook|cta)\s*:\s*", "", t, flags=re.I | re.M)
    t = re.sub(r"(?<!\w)#\w+", " ", t)                     # hashtags
    t = re.sub(r"[*_`~>|]+", " ", t)                        # markdown
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s*[—–]\s*", ", ", t)   # dashes read better as pauses ("copper — not iron" -> "copper, not iron")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)  # no space before punctuation
    t = re.sub(r",\s*([,.!?;:])", r"\1", t)  # "blue, ." -> "blue."
    return t


def truncate_words(text: str, limit: int) -> str:
    """Trim to ``limit`` characters on a word boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:-")
    return cut or text[:limit]
