"""Kokoro-82M on the CPU through onnxruntime - the local voice used when edge-tts is down.

No PyTorch, and nothing GPL is imported:

* the model is the ONNX export of hexgrad/Kokoro-82M (Apache-2.0) from onnx-community/Kokoro-82M-v1.0-ONNX
  (Apache-2.0), one voice file from the same repo, and the phoneme vocabulary from hexgrad/Kokoro-82M's
  config.json. The Dockerfile downloads them from pinned commits and checks their SHA-256.
* text -> phonemes: the `espeak-ng` program (GPL-3.0, already in the image, run as a separate process) gives
  IPA, which is mapped to Kokoro's phoneme set with the table misaki - Kokoro's own G2P (Apache-2.0,
  hexgrad/misaki, misaki/espeak.py) - uses for the same purpose.

The narration is spoken one sentence at a time (the caller joins the sentences with a short pause), so each
sentence's start is exact for the captions.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

KOKORO_DIR = "/opt/kokoro"
DEFAULT_VOICE = "am_michael"       # American male, calm and warm; graded C+ (the best US male) in VOICES.md
SAMPLE_RATE = 24000
MAX_TOKENS = 510                   # the model's context is 512, with a pad token at each end

# misaki/espeak.py (hexgrad/misaki, Apache-2.0): espeak-ng IPA -> the phonemes Kokoro v1.0 was trained on.
_E2M = sorted({
    'ʔˌn̩': 'ʔn', 'ʔn̩': 'ʔn',
    'a^ɪ': 'I', 'a^ʊ': 'W',
    'd^ʒ': 'ʤ',
    'e^ɪ': 'A', 'e': 'A',
    't^ʃ': 'ʧ',
    'ɔ^ɪ': 'Y',
    'ə^l': 'ᵊl',
    'ʲo': 'jo', 'ʲə': 'jə', 'ʲ': '',
    'ɚ': 'əɹ',
    'r': 'ɹ',
    'x': 'k', 'ç': 'k',
    'ɐ': 'ə',
    'ɬ': 'l',
    '̃': '',
}.items(), key=lambda kv: -len(kv[0]))
_PUNCT = ",;:.!?—…"


class KokoroUnavailable(RuntimeError):
    pass


def model_dir() -> Path:
    return Path(os.getenv("KOKORO_DIR") or KOKORO_DIR)


def voice_name(requested: str = "") -> str:
    v = (requested or os.getenv("KOKORO_VOICE") or DEFAULT_VOICE).strip()
    return v if re.fullmatch(r"[a-z]{2}_[a-z]+", v) else DEFAULT_VOICE


def available(voice: str = "") -> bool:
    d = model_dir()
    return bool(shutil.which("espeak-ng")) and all(
        (d / f).exists() for f in ("model.onnx", "config.json", f"voices/{voice_name(voice)}.bin"))


def espeak_to_kokoro(ps: str) -> str:
    """One clause of espeak-ng IPA (ties written as '^') in Kokoro's phoneme set (American English)."""
    for old, new in _E2M:
        ps = ps.replace(old, new)
    ps = re.sub(r"(\S)̩", r"ᵊ\1", ps).replace(chr(809), "")
    ps = ps.replace("o^ʊ", "O").replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ").replace("ɪə", "iə").replace("ː", "")
    ps = ps.replace("o", "ɔ")                                   # for espeak < 1.52
    ps = ps.replace("ɾ", "T").replace("ʔ", "t")
    return ps.replace("^", "")


def _espeak(text: str) -> str:
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "--tie=^", "-v", "en-us", text], capture_output=True,
                       text=True, timeout=30, check=True)
    return " ".join(x.strip() for x in r.stdout.splitlines() if x.strip())


def phonemize(sentence: str, ipa=None) -> str:
    """A sentence in Kokoro phonemes, its punctuation kept (it shapes the pauses and the intonation)."""
    ipa = ipa or _espeak
    out: list[str] = []
    for part in re.split(rf"([{re.escape(_PUNCT)}]+)", sentence):
        if not part.strip():
            continue
        if re.fullmatch(rf"[{re.escape(_PUNCT)}]+", part.strip()):
            p = part.strip()
            if out:
                out[-1] += p
            continue
        ps = espeak_to_kokoro(ipa(part.strip()))
        if ps:
            out.append(ps)
    return " ".join(out)


class _Model:
    _lock = threading.Lock()
    _cache: dict = {}

    @classmethod
    def get(cls, voice: str):
        with cls._lock:
            key = (str(model_dir()), voice)
            if key not in cls._cache:
                import numpy as np
                import onnxruntime as ort
                d = model_dir()
                vocab = json.loads((d / "config.json").read_text(encoding="utf-8"))["vocab"]
                sess = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
                styles = np.fromfile(d / "voices" / f"{voice}.bin", dtype=np.float32).reshape(-1, 1, 256)
                cls._cache[key] = (sess, vocab, styles)
            return cls._cache[key]


def tokens(ps: str, vocab: dict) -> list[int]:
    return [vocab[c] for c in ps if c in vocab][:MAX_TOKENS]


def speak(sentence: str, voice: str = "", speed: float = 1.0):
    """float32 mono samples at SAMPLE_RATE for one sentence."""
    import numpy as np
    v = voice_name(voice)
    if not available(v):
        raise KokoroUnavailable(f"Kokoro model or voice {v!r} not found in {model_dir()}, or espeak-ng missing")
    sess, vocab, styles = _Model.get(v)
    ids = tokens(phonemize(sentence), vocab)
    if not ids:
        return np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
    feeds = {"input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
             "style": styles[min(len(ids), len(styles) - 1)],
             "speed": np.array([speed], dtype=np.float32)}
    names = {i.name for i in sess.get_inputs()}
    feeds = {k: v for k, v in feeds.items() if k in names}
    return np.asarray(sess.run(None, feeds)[0], dtype=np.float32).reshape(-1)
