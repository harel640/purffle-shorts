"""Text-to-speech engines. Every engine returns audio plus per-word timings (native when the engine
provides them, otherwise from an aligner or an estimate) so captions land exactly on the voice.

  edge        Microsoft neural voices — free, no key, 320+ voices in 75 languages, native word timings (default)
  openai      OpenAI gpt-4o-mini-tts / tts-1-hd (OPENAI_API_KEY)
  elevenlabs  ElevenLabs premium voices, native timings (ELEVENLABS_API_KEY)
  kokoro      Kokoro-82M open-weight model, runs locally (pip install kokoro soundfile)
  coqui       Coqui TTS models, runs locally (pip install coqui-tts)
  system      The OS voice (macOS `say`, Linux espeak-ng, Windows SAPI) — offline
  silent      No voice; for layout previews and tests
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg
from .config import Settings
from .timing import Word, anchored_timings, estimate_timings, tokenize, transfer_timings
from .utils import PermanentError, http, raise_for_status, run_with_timeout, with_retries

log = logging.getLogger("purffle")

# Hand-picked, natural-sounding default voices per language (edge-tts). Override with TTS_VOICE.
EDGE_VOICES = {
    "en": "en-US-AndrewMultilingualNeural", "es": "es-MX-JorgeNeural", "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural", "it": "it-IT-DiegoNeural", "pt": "pt-BR-AntonioNeural",
    "hi": "hi-IN-MadhurNeural", "ta": "ta-IN-ValluvarNeural", "te": "te-IN-MohanNeural",
    "bn": "bn-IN-BashkarNeural", "mr": "mr-IN-ManoharNeural", "ur": "ur-PK-AsadNeural",
    "ar": "ar-SA-HamedNeural", "ru": "ru-RU-DmitryNeural", "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural", "zh": "zh-CN-YunxiNeural", "id": "id-ID-ArdiNeural",
    "tr": "tr-TR-AhmetNeural", "nl": "nl-NL-MaartenNeural", "pl": "pl-PL-MarekNeural",
    "vi": "vi-VN-NamMinhNeural", "th": "th-TH-NiwatNeural", "fil": "fil-PH-AngeloNeural",
    "uk": "uk-UA-OstapNeural", "sv": "sv-SE-MattiasNeural",
}
# Second speaker for dialogue and chat formats (a contrasting voice). Override with TTS_VOICE_B.
EDGE_VOICES_B = {
    "en": "en-US-AvaMultilingualNeural", "es": "es-MX-DaliaNeural", "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural", "it": "it-IT-ElsaNeural", "pt": "pt-BR-FranciscaNeural",
    "hi": "hi-IN-SwaraNeural", "ta": "ta-IN-PallaviNeural", "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural", "mr": "mr-IN-AarohiNeural", "ur": "ur-PK-UzmaNeural",
    "ar": "ar-SA-ZariyahNeural", "ru": "ru-RU-SvetlanaNeural", "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural", "zh": "zh-CN-XiaoxiaoNeural", "id": "id-ID-GadisNeural",
    "tr": "tr-TR-EmelNeural", "nl": "nl-NL-ColetteNeural", "pl": "pl-PL-ZofiaNeural",
    "vi": "vi-VN-HoaiMyNeural", "th": "th-TH-PremwadeeNeural", "fil": "fil-PH-BlessicaNeural",
    "uk": "uk-UA-PolinaNeural", "sv": "sv-SE-SofieNeural",
}
SECOND_VOICE = {"openai": "nova", "elevenlabs": "21m00Tcm4TlvDq8ikWAM", "kokoro": "af_heart"}
LINE_GAP = 0.18  # silence between two speakers' lines

KOKORO_LANG = {"en": ("a", "am_michael"), "es": ("e", "em_alex"), "fr": ("f", "ff_siwis"),
               "hi": ("h", "hm_omega"), "it": ("i", "im_nicola"), "ja": ("j", "jm_kumo"),
               "pt": ("p", "pm_alex"), "zh": ("z", "zm_yunxi")}


@dataclass
class Speech:
    audio: Path
    words: list[Word]
    duration: float
    engine: str
    voice: str
    timing: str  # native | aligned | anchored (pauses pinned) | estimated
    line_starts: list[float] | None = None  # multi-voice: when each line starts


def _pick_voice(settings: Settings, default: str) -> str:
    voices = [v.strip() for v in settings.tts_voice.split(",") if v.strip()]
    return random.choice(voices) if voices else default


def _rate_multiplier(rate: str) -> float:
    try:
        return 1.0 + float(rate.strip().rstrip("%")) / 100.0
    except ValueError:
        return 1.0


# edge-tts talks to Microsoft over a websocket. Its own connect/receive timeouts have failed to fire on
# a half-open socket (the evening run froze here for hours), so each attempt also runs under a hard
# wall-clock ceiling and we retry once. Keep the ceiling well under the launcher's docker-run watchdog.
EDGE_CONNECT_TIMEOUT = 15   # seconds to open the websocket
EDGE_RECEIVE_TIMEOUT = 30   # seconds to wait for each audio / word-boundary message
EDGE_SYNTH_TIMEOUT = 60     # hard ceiling on one whole synthesis attempt (a 150-word story takes ~40 s)
EDGE_ATTEMPTS = 1           # per call: the voice-over as a whole is retried (Studio._speak, VOICE_ATTEMPTS)


class VoiceUnavailable(RuntimeError):
    """Every voice engine failed: an outage, not a problem with the topic (a later run tries it again)."""
    transient = True


# Piper: a local voice that needs no network, the fallback when edge-tts is down. It runs as its own program
# (the `piper` command of the piper-tts package, GPL-3.0 - never imported). The model path comes from
# PIPER_VOICE (the Docker image ships en_US "norman", trained on public-domain LibriVox readings).
PIPER_DEFAULT_VOICE = "/opt/piper-voices/en_US-norman-medium.onnx"
PIPER_SENTENCE_GAP = 0.32   # seconds of silence between sentences (Piper speaks each one separately)
PIPER_TIMEOUT = 180
# Piper's own pace is ~3.2 words/s - quick for documentary narration and well above the ~2.4 w/s the 45-60 s
# story scripts are written for. 1.2 = 20% longer phonemes (~2.6 w/s with the sentence pauses).
# PIPER_LENGTH_SCALE overrides it; TTS_RATE is edge-tts's and does not apply.
PIPER_LENGTH_SCALE = 1.2


# ------------------------------------------------------------------------------------------ engines
def _edge(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    try:
        import edge_tts
    except ImportError as e:
        raise PermanentError("edge-tts is not installed:  pip install edge-tts") from e
    voice = _pick_voice(settings, EDGE_VOICES.get(settings.language, EDGE_VOICES["en"]))
    path = out / "voice.mp3"

    def _do():
        words: list[Word] = []
        # Each attempt writes its own temp file: a timed-out attempt's thread may still be writing, so
        # the retry must not share the handle. Only a fully streamed file is moved into place.
        tmp = out / f"voice.{threading.get_ident()}.part.mp3"
        try:
            comm = edge_tts.Communicate(text, voice, rate=settings.tts_rate, boundary="WordBoundary",
                                        connect_timeout=EDGE_CONNECT_TIMEOUT, receive_timeout=EDGE_RECEIVE_TIMEOUT)
            with open(tmp, "wb") as f:
                for chunk in comm.stream_sync():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        start = chunk["offset"] / 1e7
                        words.append(Word(chunk["text"], start, start + chunk["duration"] / 1e7))
            if tmp.stat().st_size == 0:
                raise RuntimeError("edge-tts returned no audio")
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return words

    words = with_retries(lambda: run_with_timeout(_do, EDGE_SYNTH_TIMEOUT, f"edge-tts {voice}"),
                         attempts=EDGE_ATTEMPTS, label=f"edge-tts {voice}")
    return path, words, voice


def _piper_sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", " ".join((text or "").split())) if x.strip()]


def _piper(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    """Voice ``text`` with Piper, one sentence at a time (one `piper` call; it writes a WAV per input line),
    joined with a short pause. Each sentence's start is then exact, and the words inside it are timed in
    proportion - so captions stay in step without a word aligner."""
    if settings.language.split("-")[0] != "en":
        raise PermanentError(f"the Piper voice is English only (language {settings.language})")
    exe = shutil.which("piper")
    model = Path(settings.tts_voice if settings.tts_voice.endswith(".onnx") else
                 os.getenv("PIPER_VOICE") or PIPER_DEFAULT_VOICE)
    if not exe or not model.exists():
        raise PermanentError(f"Piper is not available (piper command: {exe or 'missing'}, voice: {model})")
    sentences = _piper_sentences(text)
    if not sentences:
        raise PermanentError("nothing to say")
    parts_dir = out / "piper"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    cmd = [exe, "-m", str(model), "-d", str(parts_dir),
           "--length-scale", f"{float(os.getenv('PIPER_LENGTH_SCALE') or PIPER_LENGTH_SCALE):.3f}"]
    try:
        subprocess.run(cmd, input="\n".join(sentences) + "\n", text=True, capture_output=True, check=True,
                       timeout=PIPER_TIMEOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"piper failed: {(e.stderr or '').strip()[-300:]}") from e
    wavs = sorted(parts_dir.glob("*.wav"))      # named by a timestamp: sorted = spoken order
    if len(wavs) != len(sentences):
        raise RuntimeError(f"piper wrote {len(wavs)} file(s) for {len(sentences)} sentence(s)")
    path = out / "voice.wav"
    words: list[Word] = []
    t = 0.0
    with wave.open(str(path), "wb") as dst:
        for i, (wav, sentence) in enumerate(zip(wavs, sentences)):
            with wave.open(str(wav), "rb") as src:
                if i == 0:
                    dst.setnchannels(src.getnchannels())
                    dst.setsampwidth(src.getsampwidth())
                    dst.setframerate(src.getframerate())
                rate, frames = src.getframerate(), src.readframes(src.getnframes())
            dur = len(frames) / (rate * src.getsampwidth() * src.getnchannels())
            words += [Word(w.text, round(w.start + t, 3), round(w.end + t, 3))
                      for w in estimate_timings(tokenize(sentence, settings.language), dur, lead=0.05, tail=0.1)]
            dst.writeframes(frames)
            t += dur
            if i + 1 < len(wavs):
                gap = int(PIPER_SENTENCE_GAP * rate)
                dst.writeframes(b"\x00" * gap * src.getsampwidth() * src.getnchannels())
                t += gap / rate
    shutil.rmtree(parts_dir, ignore_errors=True)
    return path, words, model.stem


def _openai(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    if not settings.openai_api_key:
        raise PermanentError("TTS_ENGINE=openai needs OPENAI_API_KEY")
    model = settings.tts_model or "gpt-4o-mini-tts"
    voice = _pick_voice(settings, "onyx")
    body = {"model": model, "voice": voice, "input": text, "response_format": "mp3"}
    if model.startswith("gpt-4o"):
        body["instructions"] = settings.tts_instructions
    else:
        body["speed"] = max(0.25, min(4.0, _rate_multiplier(settings.tts_rate)))
    path = out / "voice.mp3"

    def _do():
        r = http().post("https://api.openai.com/v1/audio/speech", json=body, timeout=180,
                        headers={"Authorization": f"Bearer {settings.openai_api_key}"})
        raise_for_status(r, "OpenAI TTS")
        path.write_bytes(r.content)

    with_retries(_do, attempts=3, label="OpenAI TTS")
    return path, [], voice


def _elevenlabs(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    if not settings.elevenlabs_api_key:
        raise PermanentError("TTS_ENGINE=elevenlabs needs ELEVENLABS_API_KEY")
    voice = _pick_voice(settings, "pNInz6obpgDQGcFmaJgB")  # "Adam" premade voice; use any voice id
    model = settings.tts_model or "eleven_multilingual_v2"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128"
    path = out / "voice.mp3"

    def _do():
        r = http().post(url, timeout=180, headers={"xi-api-key": settings.elevenlabs_api_key},
                        json={"text": text, "model_id": model,
                              "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.3}})
        raise_for_status(r, "ElevenLabs TTS")
        return r.json()

    data = with_retries(_do, attempts=3, label="ElevenLabs TTS")
    path.write_bytes(base64.b64decode(data["audio_base64"]))
    al = data.get("alignment") or data.get("normalized_alignment") or {}
    words: list[Word] = []
    cur, cs, prev_e = "", 0.0, 0.0
    for ch, s, e in zip(al.get("characters", []), al.get("character_start_times_seconds", []),
                        al.get("character_end_times_seconds", [])):
        if ch.isspace():
            if cur:
                words.append(Word(cur, cs, prev_e))
            cur = ""
            continue
        if not cur:
            cs = s
        cur += ch
        prev_e = e
    if cur:
        words.append(Word(cur, cs, prev_e))
    return path, words, voice


_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def _kokoro(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    try:
        import numpy as np
        from kokoro import KPipeline
    except ImportError as e:
        raise PermanentError("TTS_ENGINE=kokoro needs:  pip install kokoro soundfile") from e
    lang_code, default_voice = KOKORO_LANG.get(settings.language, KOKORO_LANG["en"])
    voice = _pick_voice(settings, default_voice)
    with _model_lock:
        pipe = _model_cache.get(f"kokoro-{lang_code}")
        if pipe is None:
            pipe = _model_cache[f"kokoro-{lang_code}"] = KPipeline(lang_code=lang_code)
        parts = [np.asarray(audio, dtype=np.float32)
                 for _, _, audio in pipe(text, voice=voice, speed=_rate_multiplier(settings.tts_rate))]
    if not parts:
        raise RuntimeError("kokoro produced no audio")
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2")
    path = out / "voice.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    return path, [], voice


def _coqui(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    try:
        from TTS.api import TTS
    except ImportError as e:
        raise PermanentError("TTS_ENGINE=coqui needs:  pip install coqui-tts") from e
    model = settings.tts_model or "tts_models/en/ljspeech/vits"
    path = out / "voice.wav"
    with _model_lock:
        tts = _model_cache.get(model)
        if tts is None:
            tts = _model_cache[model] = TTS(model_name=model)
        tts.tts_to_file(text=text, file_path=str(path))
    return path, [], model


def _system(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    txt = out / "voice.txt"
    txt.write_text(text, encoding="utf-8")
    voice = settings.tts_voice or ""
    if sys.platform == "darwin":
        raw = out / "voice.aiff"
        cmd = ["say", "-r", str(int(185 * _rate_multiplier(settings.tts_rate))), "-o", str(raw), "-f", str(txt)]
        if voice:
            cmd[1:1] = ["-v", voice]
    elif sys.platform.startswith("win"):
        raw = out / "voice.wav"
        ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SetOutputToWaveFile('{raw}'); $s.Speak([IO.File]::ReadAllText('{txt}')); $s.Dispose()")
        cmd = ["powershell", "-NoProfile", "-Command", ps]
    else:
        raw = out / "voice.wav"
        exe = "espeak-ng"
        cmd = [exe, "-v", voice or settings.language, "-s", str(int(165 * _rate_multiplier(settings.tts_rate))),
               "-w", str(raw), "-f", str(txt)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except FileNotFoundError as e:
        raise PermanentError(f"System voice not available ({cmd[0]} missing). On Linux: apt install espeak-ng") from e
    return raw, [], voice or "system-default"


def _silent(text: str, out: Path, settings: Settings) -> tuple[Path, list[Word], str]:
    seconds = max(2.0, len(tokenize(text)) / 2.6)
    path = out / "voice.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{seconds:.2f}", str(path)],
               label="silent-voice")
    return path, [], "silent"


_ENGINE_FUNCS = {"edge": _edge, "openai": _openai, "elevenlabs": _elevenlabs, "kokoro": _kokoro,
                 "coqui": _coqui, "piper": _piper, "system": _system, "silent": _silent}
NETWORK_ENGINES = {"edge", "openai", "elevenlabs"}       # may fail for a while and then work again


# ------------------------------------------------------------------------------------------ aligners
def _cuda() -> bool:
    try:
        import ctranslate2  # installed with faster-whisper
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def whisper_model(settings: Settings | None = None) -> str:
    """WHISPER_MODEL, or for "auto" the most accurate model that is still quick on this machine."""
    name = (settings.whisper_model if settings else "") or "auto"
    if name != "auto":
        return name
    return "large-v3-turbo" if _cuda() else "small"


def _align_faster_whisper(audio: Path, language: str, model_name: str = "small", *, prompt: str = "",
                          vad: bool = False) -> list[Word]:
    """Word timestamps from faster-whisper. ``prompt`` (the known script) helps it spell names and numbers the
    way they are written; ``vad`` skips silence and music in long recordings and stops it from inventing text."""
    from faster_whisper import WhisperModel  # optional dependency
    cuda = _cuda()
    with _model_lock:
        key = f"whisper:{model_name}"
        model = _model_cache.get(key)
        if model is None:
            model = _model_cache[key] = WhisperModel(model_name, device="cuda" if cuda else "cpu",
                                                     compute_type="float16" if cuda else "int8")
        kwargs = {"language": language or None, "word_timestamps": True}
        if prompt:
            kwargs["initial_prompt"] = prompt[:800]
        if vad:
            kwargs.update(vad_filter=True, condition_on_previous_text=False)
        segments, _ = model.transcribe(_pcm16k(audio), **kwargs)
        return [Word(w.word.strip(), w.start, w.end) for seg in segments for w in (seg.words or [])]


def _pcm16k(audio: Path):
    """The audio as the 16 kHz mono float samples Whisper works on, decoded by our ffmpeg. faster-whisper's
    own decoder (PyAV) breaks whenever the two packages' versions drift apart (faster-whisper 1.2.1 with
    PyAV 19 fails with "unexpected keyword argument 'metadata_errors'")."""
    import subprocess

    import numpy as np  # installed with faster-whisper

    from . import ffmpeg
    proc = subprocess.run([ffmpeg.ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(audio), "-vn", "-ac", "1",
                           "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"could not decode {Path(audio).name}: {proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _align_openai(audio: Path, language: str, api_key: str) -> list[Word]:
    with open(audio, "rb") as f:
        r = http().post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": "whisper-1", "response_format": "verbose_json", "language": language,
                  "timestamp_granularities[]": "word"},
            files={"file": (audio.name, f)}, timeout=180)
    raise_for_status(r, "OpenAI alignment")
    return [Word(w["word"], float(w["start"]), float(w["end"])) for w in r.json().get("words", [])]


def _align(audio: Path, settings: Settings, text: str = "") -> tuple[list[Word], str]:
    mode = settings.align
    if mode == "none":
        return [], "estimated"
    lang = settings.language.split("-")[0]
    if mode in ("auto", "faster-whisper"):
        try:
            return _align_faster_whisper(audio, lang, whisper_model(settings), prompt=text), "aligned"
        except ImportError:
            if mode == "faster-whisper":
                log.warning("ALIGN=faster-whisper but it isn't installed (pip install faster-whisper)")
        except Exception as e:
            log.warning("faster-whisper alignment failed: %s", e)
    if mode == "openai" and settings.openai_api_key:
        try:
            return _align_openai(audio, lang, settings.openai_api_key), "aligned"
        except Exception as e:
            log.warning("OpenAI alignment failed: %s", e)
    return [], "estimated"


# ------------------------------------------------------------------------------------------ voice cache
CACHED_ENGINES = {"edge", "openai", "elevenlabs", "kokoro", "coqui", "piper"}
CACHE_LIMIT_MB = 400


def _cache_dir(text: str, settings: Settings, engine: str) -> Path | None:
    """Where this exact line, in this exact voice, is kept. None when it can't be reused (a rotating voice
    list picks a different voice each time, and the silent/system voices cost nothing to redo)."""
    if not settings.voice_cache or engine not in CACHED_ENGINES or "," in settings.tts_voice:
        return None
    parts = [engine, settings.tts_voice, settings.language, settings.tts_rate, settings.tts_model,
             settings.tts_instructions if engine == "openai" else "", settings.align, text]
    key = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return settings.data_path / "cache" / "voice" / key[:2] / key


def _cache_load(d: Path | None) -> Speech | None:
    if d is None or not (d / "speech.json").exists():
        return None
    try:
        meta = json.loads((d / "speech.json").read_text(encoding="utf-8"))
        audio = d / meta["audio"]
        if not audio.exists():
            return None
        os.utime(d / "speech.json")  # most recently used: kept longest
        return Speech(audio=audio, words=[Word(*w) for w in meta["words"]], duration=float(meta["duration"]),
                      engine=meta["engine"], voice=meta["voice"], timing=meta["timing"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _cache_store(d: Path | None, sp: Speech) -> None:
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        name = "audio" + sp.audio.suffix
        shutil.copyfile(sp.audio, d / name)
        meta = {"audio": name, "words": [[w.text, w.start, w.end] for w in sp.words], "duration": sp.duration,
                "engine": sp.engine, "voice": sp.voice, "timing": sp.timing}
        (d / "speech.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if random.random() < 0.1:
            _cache_prune(d.parent.parent)
    except OSError as e:
        log.debug("voice cache write failed: %s", e)


def _cache_prune(root: Path, limit_mb: int = CACHE_LIMIT_MB) -> None:
    """Drop the least recently used lines once the cache is over its size limit."""
    entries = []
    for meta in root.glob("*/*/speech.json"):
        size = sum(f.stat().st_size for f in meta.parent.iterdir() if f.is_file())
        entries.append((meta.stat().st_mtime, size, meta.parent))
    total = sum(e[1] for e in entries)
    for _, size, d in sorted(entries):
        if total <= limit_mb * 1024 * 1024:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size


def synthesize(text: str, out_dir: Path, settings: Settings) -> Speech:
    """Speak ``text`` and return audio + one timed Word per script token. A line spoken before with the same
    voice comes from the voice cache, so re-rendering an edited script only voices what changed."""
    engine = settings.tts_engine if settings.tts_engine in _ENGINE_FUNCS else "edge"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_dir(text, settings, engine)
    hit = _cache_load(cache)
    if hit:
        log.info("Voice: %s/%s, %.1fs (from the voice cache)", hit.engine, hit.voice, hit.duration)
        return hit
    audio, native, voice = _ENGINE_FUNCS[engine](text, out_dir, settings)
    dur = ffmpeg.duration(audio)
    if dur <= 0:
        raise RuntimeError(f"{engine} produced unreadable audio: {audio}")
    tokens = tokenize(text, settings.language)
    if native:
        # Piper's words are timed per sentence (exact sentence starts, proportional inside)
        words, timing = transfer_timings(tokens, native, dur), "sentence" if engine == "piper" else "native"
    elif engine == "silent":
        words, timing = estimate_timings(tokens, dur), "estimated"
    else:
        aligned, timing = _align(audio, settings, text)
        if aligned:
            words = transfer_timings(tokens, aligned, dur)
        else:  # no aligner: pin phrase ends to the pauses in the audio, estimate only in between
            words, pinned = anchored_timings(tokens, dur, ffmpeg.silences(audio))
            timing = "anchored" if pinned else "estimated"
    log.info("Voice: %s/%s, %.1fs, %d words (%s timing)", engine, voice, dur, len(words), timing)
    speech = Speech(audio=audio, words=words, duration=dur, engine=engine, voice=voice, timing=timing)
    _cache_store(cache, speech)
    return speech


def second_voice(settings: Settings) -> str:
    if settings.tts_voice_b:
        return settings.tts_voice_b
    engine = settings.tts_engine
    if engine == "edge":
        return EDGE_VOICES_B.get(settings.language.split("-")[0], EDGE_VOICES_B["en"])
    return SECOND_VOICE.get(engine, "")


def synthesize_lines(lines: list[tuple[str, str]], out_dir: Path, settings: Settings) -> Speech:
    """Voice a conversation: each (speaker, text) line with that speaker's voice, joined into one track.

    Word timings stay exact because every line is timed on its own and shifted by where it lands."""
    import concurrent.futures
    from dataclasses import replace
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_a = _pick_voice(settings, "")  # blank = engine default for speaker A
    voice_b = second_voice(settings)
    # One voice per speaker for the whole video (a comma list in TTS_VOICE must not change mid-talk).
    cfg = {"A": replace(settings, tts_voice=voice_a), "B": replace(settings, tts_voice=voice_b or voice_a)}

    def one(i: int, speaker: str, text: str) -> Speech:
        return synthesize(text, out_dir / f"line{i:02d}", cfg[speaker])

    workers = 4 if settings.tts_engine in ("edge", "openai", "elevenlabs") else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(lambda a: one(*a), [(i, sp, t) for i, (sp, t) in enumerate(lines)]))

    # Normalise every line to the same PCM format and join them with a short pause in between.
    gap = out_dir / "gap.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{LINE_GAP:.3f}", str(gap)], label="gap")
    listing, words, starts, t = [], [], [], 0.0
    for i, ((speaker, _), sp) in enumerate(zip(lines, parts)):
        wav = out_dir / f"line{i:02d}.wav"
        # Keep only the spoken part (plus a breath): engines pad each clip with silence, and between two
        # speakers that padding adds up to a second-long dead gap.
        cut0 = max(0.0, sp.words[0].start - 0.05) if sp.words else 0.0
        cut1 = min(sp.duration, sp.words[-1].end + 0.15) if sp.words else sp.duration
        ffmpeg.run(["-i", str(sp.audio), "-af", f"atrim=start={cut0:.3f}:end={cut1:.3f},asetpts=PTS-STARTPTS",
                    "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], label="line")
        dur = ffmpeg.duration(wav)
        starts.append(t)
        words += [Word(w.text, round(w.start - cut0 + t, 3), round(w.end - cut0 + t, 3), speaker) for w in sp.words]
        listing.append(f"file '{wav.name}'")
        t += dur
        if i + 1 < len(parts):
            listing.append(f"file '{gap.name}'")
            t += LINE_GAP
    (out_dir / "lines.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
    audio = out_dir / "voice.wav"
    ffmpeg.run(["-f", "concat", "-safe", "0", "-i", "lines.txt", "-c:a", "pcm_s16le", str(audio.name)],
               label="join voices", cwd=out_dir)
    dur = ffmpeg.duration(audio)
    timing = min((p.timing for p in parts), key=["estimated", "anchored", "sentence", "aligned", "native"].index)
    voices = "/".join(dict.fromkeys(p.voice for p in parts))
    log.info("Voices: %s, %.1fs, %d lines, %d words (%s timing)", voices, dur, len(lines), len(words), timing)
    return Speech(audio=audio, words=words, duration=dur, engine=parts[0].engine, voice=voices, timing=timing,
                  line_starts=starts)


def list_edge_voices(language: str = "") -> list[dict]:
    import asyncio

    import edge_tts
    voices = asyncio.run(edge_tts.list_voices())
    if language:
        voices = [v for v in voices if v["Locale"].lower().startswith(language.lower())]
    return sorted(voices, key=lambda v: v["ShortName"])
