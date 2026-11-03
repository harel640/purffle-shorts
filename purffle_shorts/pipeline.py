"""The production line: topic -> script -> review -> voice -> footage -> captions -> render -> upload.

``Studio.draft`` stops after the script (so it can be read and edited first); ``Studio.make`` runs the
whole line, optionally followed by translated copies that reuse the same footage."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import analytics, crosspost, ffmpeg, notify, seo, sfx, topics, tts, youtube
from .config import Settings
# Place de-duplication lives in history.py (no heavy imports) so other tools can apply the same rule.
from .history import PLACE_REUSE_DAYS, History, now_iso, place_recently_used  # noqa: F401 (re-exported)
from .llm import LLM, LLMConfigError
from . import factcheck, safety
from .media import MediaItem, NeuronBudget, SkipTopic, Visuals, background_item, resolve_place
from .overlays import EMOJI_CREDIT, ChatMessage, build_overlays
from .render import pick_cover, render_video, split_sizes
from .script import (
    AUTO_STYLES,
    CARD_STYLES,
    DEFAULT_CAST,
    LANGUAGES,
    MULTI_SPEAKER,
    STORY_STYLES,
    STYLES,
    Script,
    load_script,
    localize_metadata,
    review_script,
    translate_script,
    write_script,
)
from .timing import SceneTiming, caption_chunks, scene_timeline, split_shots, srt, tokenize
from .topics import Topic, pick_topic
from .utils import Cancelled, check_cancel, redact, slugify, truncate_words

log = logging.getLogger("purffle")

END_PAD = 0.6          # seconds of breathing room after the last word
AI_SOURCES = {"openai-images", "ai", "dalle", "gpt-image", "pollinations", "cloudflare", "gemini", "nano-banana"}
REUSABLE_SOURCES = {"generated", "local", "background"} | AI_SOURCES
# Added to the description (and the YouTube synthetic-content disclosure is set) whenever a video uses AI images.
AI_DISCLOSURE = "Some scenes are AI illustrations."

# Rough share of the work done when each stage starts (for progress bars before any timing is learned).
STAGES = {"topic": 2, "script": 8, "review": 20, "voice": 32, "footage": 45, "captions": 62, "render": 72,
          "check": 92, "upload": 95, "translate": 97, "done": 100}
ORDER = list(STAGES)
# Seconds each stage takes for a 40-second video until this machine's own timings are known.
DEFAULT_SECONDS = {"topic": 1.0, "script": 15.0, "review": 25.0, "voice": 6.0, "footage": 12.0, "captions": 3.0,
                   "render": 20.0, "check": 1.0, "upload": 30.0, "translate": 70.0}

ProgressFn = Callable[[str, int, str], None]


class StageClock:
    """How long each stage takes on this machine: a moving average per stage, scaled to the video's length
    and kept in the history database, so progress bars and ETAs follow real time instead of a fixed table."""

    KEY = "stage_seconds"

    def __init__(self, history: History, target_seconds: int):
        self.history = history
        self.scale = max(0.25, target_seconds / 40)
        try:
            self.avg: dict[str, float] = json.loads(history.get_meta(self.KEY) or "{}")
        except ValueError:
            self.avg = {}

    @property
    def learned(self) -> bool:
        return {"voice", "render"} <= set(self.avg)

    def expected(self, stage: str) -> float:
        if stage == "done":
            return 0.0
        return self.avg.get(stage, DEFAULT_SECONDS.get(stage, 2.0)) * self.scale

    def record(self, stage: str, seconds: float) -> None:
        if stage not in DEFAULT_SECONDS or seconds <= 0:
            return
        raw = seconds / self.scale
        self.avg[stage] = round(raw if stage not in self.avg else 0.7 * self.avg[stage] + 0.3 * raw, 2)
        try:
            self.history.set_meta(self.KEY, json.dumps(self.avg, sort_keys=True))
        except Exception:  # timing statistics are a nicety
            log.debug("could not save stage timings", exc_info=True)


@dataclass
class Result:
    ok: bool
    title: str = ""
    folder: Path | None = None
    video: Path | None = None
    status: str = ""
    youtube_id: str | None = None
    publish_at: str | None = None
    seconds: float = 0.0
    error: str = ""
    record_id: int | None = None
    score: int | None = None
    language: str = ""
    variants: list[Result] = field(default_factory=list)
    reason: str = ""          # clips: why the AI picked this moment
    posts: dict = field(default_factory=dict)  # platform -> {status, url, id, error} beyond YouTube


def quota_day_start() -> str:
    """YouTube's daily quota resets at midnight Pacific time."""
    from zoneinfo import ZoneInfo
    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    start = pt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).isoformat(timespec="seconds")


def seconds_until_quota_reset() -> float:
    from zoneinfo import ZoneInfo
    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    nxt = (pt.replace(hour=0, minute=0, second=0, microsecond=0)).timestamp() + 86400
    return max(60.0, nxt - pt.timestamp() + 120)


def _future(iso: str | None) -> datetime | None:
    """A stored publish time that is still ahead, so a retry posts the other networks at the same slot."""
    if not iso:
        return None
    try:
        t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return t if t > datetime.now(timezone.utc) else None


def plan_ai_scenes(historical: list[bool]) -> list[bool]:
    """Which scenes get an AI illustration vs a real photo. The first and last scene are always real photos,
    at most 5 scenes are AI, and at least ~40% of scenes are real. Extra AI scenes become real photos, dropped
    from the end inward so the earliest story beats keep their illustration."""
    hist = list(historical)
    n = len(hist)
    if n:
        hist[0] = hist[-1] = False
    allowed = min(5, n - (2 * n + 4) // 5)   # (2n+4)//5 == ceil(0.4 n): the real-photo floor
    for i in range(n - 1, -1, -1):
        if sum(hist) <= allowed:
            break
        hist[i] = False
    return hist


def final_title(script: Script, settings: Settings | None = None, text: str | None = None) -> str:
    t = script.title if text is None else text
    if settings is not None and not settings.vertical:
        return truncate_words(t, 100)  # a 16:9 video is not a Short
    # Keep the base title under 60 even after a fact-check rewrite (which runs after the writer's own cap),
    # before the " #shorts" tag is appended.
    t = truncate_words(t, 60)
    return f"{t} #shorts" if len(t) + 8 <= 100 else truncate_words(t, 100)


def image_credits_block(creds: list[dict]) -> str:
    """Attribution block for Wikimedia images, as CC-BY / CC-BY-SA require (author + license + links)."""
    seen: set[str] = set()
    lines: list[str] = []
    for c in creds or []:
        key = c.get("page_url") or c.get("title") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        who, lic = c.get("author") or "Unknown author", c.get("license") or ""
        tail = " ".join(x for x in (c.get("page_url", ""), c.get("license_url", "")) if x)
        lines.append(f"- {c.get('title') or 'Image'} - {who} ({lic}) {tail}".strip())
    if not lines:
        return ""
    return "Image credits (Wikimedia Commons, used under their licenses):\n" + "\n".join(lines)


def final_description(script: Script, credits: set[str], settings: Settings, text: str | None = None,
                      extra: list[str] | None = None) -> str:
    body = (script.description if text is None else text).strip()
    parts = [body] if body else []
    tags = script.hashtags if settings.vertical else [h for h in script.hashtags if h.lower() != "#shorts"]
    parts.append(" ".join(tags))
    if settings.credit_footage and credits:
        parts.append("Footage: " + ", ".join(sorted(credits)))
    parts.extend(extra or [])
    return "\n\n".join(p for p in parts if p)


def _unique_folder(root: Path, name: str) -> Path:
    folder = root / name
    n = 2
    while folder.exists():
        folder = root / f"{name}-{n}"
        n += 1
    return folder


def _media_for(shots, media_map: dict[tuple[int, int], MediaItem]) -> list[MediaItem]:
    """Reuse the original video's footage for a translated copy. Its scenes may merge or split into shots
    differently, so each shot takes the closest (scene, shot) of the original."""
    keys = sorted(media_map)
    out = []
    for st in shots:
        same = [k for k in keys if k[0] == st.index]
        if same:
            k = max([k for k in same if k[1] <= st.part], default=same[0])
        else:
            k = max([k for k in keys if k[0] <= st.index], default=keys[0])
        out.append(media_map[k])
    return out


class Studio:
    def __init__(self, settings: Settings, progress: ProgressFn | None = None):
        self.s = settings
        self.history = History(settings.data_path / "history.db")
        self._llm: LLM | None = None
        self.progress = progress
        self._stats_synced = 0.0
        self.clock = StageClock(self.history, settings.target_seconds)
        self._learn = False            # only whole videos teach the clock (clips use the stages differently)
        self._quiet = False            # translated copies run as one "translate" stage
        self._plan: list[str] = []     # the stages this job will go through
        self._stage, self._stage_t0, self._frac, self._msg = "", 0.0, 0.0, ""
        self._t_start, self._pct = 0.0, 0

    # ------------------------------------------------------------------ progress, ETA, cancel
    def _begin(self, plan: list[str]) -> None:
        self._plan, self._learn = plan, True
        self._stage, self._t_start, self._pct = "", time.time(), 0

    def _step(self, stage: str, message: str = "") -> None:
        check_cancel()
        if self._quiet:
            return
        now = time.time()
        # Demo runs write instantly, so they must not teach the clock how long the AI stages take.
        writing = self._stage in ("topic", "script", "review", "translate")
        if self._learn and self._stage and self._stage != stage and not (self.s.offline and writing):
            self.clock.record(self._stage, now - self._stage_t0)
        self._stage, self._stage_t0, self._frac, self._msg = stage, now, 0.0, message
        self._report()
        if stage == "done":  # the job is over: the next draft or make starts its own plan
            self._learn, self._plan = False, []

    def _sub(self, frac: float, message: str = "") -> None:
        """Progress inside the current stage (0..1), e.g. render frames written."""
        if self._quiet:
            return
        self._frac = max(self._frac, min(1.0, frac))
        if message:
            self._msg = message
        self._report()

    def eta(self) -> float | None:
        """Seconds until the job is done, from this machine's stage timings, or None when unknown."""
        if not self._stage or self._stage == "done" or not self._plan:
            return None
        cur = self._stage
        spent = time.time() - self._stage_t0
        exp = self.clock.expected(cur)
        left = exp * (1 - self._frac) if self._frac > 0.02 else max(0.15 * exp, exp - spent)
        after = self._plan[self._plan.index(cur) + 1:] if cur in self._plan else []
        return round(left + sum(self.clock.expected(st) for st in after), 1)

    def _report(self) -> None:
        stage = self._stage
        if stage == "done":
            pct = 100
        elif self.clock.learned and self._plan and (eta := self.eta()) is not None:
            spent = time.time() - self._t_start
            pct = int(99 * spent / max(0.1, spent + eta))
        else:
            nxt = next((STAGES[x] for x in ORDER[ORDER.index(stage) + 1:]), 100) if stage in STAGES else 0
            pct = int(STAGES.get(stage, 0) + self._frac * (nxt - STAGES.get(stage, 0)))
        self._pct = max(self._pct, pct)
        if self.progress:
            try:
                self.progress(stage, self._pct, self._msg)
            except Exception:  # a progress display must never break a video
                log.debug("progress callback failed", exc_info=True)

    @property
    def llm(self) -> LLM | None:
        if self.s.offline:
            return None
        if self._llm is None:
            self._llm = LLM(self.s)
            log.info("LLM: %s%s", self._llm.label,
                     f" (fallbacks: {', '.join(str(p) for p in self._llm.providers[1:])})"
                     if len(self._llm.providers) > 1 else "")
        return self._llm

    # ------------------------------------------------------------------ voice with fallbacks
    def _speak(self, script: Script, work: Path, s: Settings) -> tts.Speech:
        engines = [s.tts_engine]
        if s.tts_engine != "edge":
            engines.append("edge")
        if s.offline:
            engines += ["system", "silent"]
        lines = [(sc.speaker, sc.narration) for sc in script.scenes] if (
            script.multi_speaker or script.style in MULTI_SPEAKER) else None
        last: Exception | None = None
        for eng in dict.fromkeys(engines):
            try:
                cfg = s if eng == s.tts_engine else replace(s, tts_engine=eng, tts_voice="", tts_voice_b="")
                if lines:
                    return tts.synthesize_lines(lines, work / "voice", cfg)
                return tts.synthesize(script.narration, work, cfg)
            except Exception as e:
                last = e
                log.warning("Voice engine '%s' failed: %s", eng, e)
        raise RuntimeError(f"All voice engines failed: {last}")

    # ------------------------------------------------------------------ script only
    def draft(self, topic: str | None = None, *, source: str | None = None, style: str | None = None,
              url: str | None = None, document: str | Path | None = None) -> tuple[Script, Topic, str]:
        """Pick a topic, write the script and let the Script Doctor improve it. Nothing is rendered."""
        s = self.s
        standalone = not self._plan  # a draft on its own (Studio "Draft"), not the first half of make()
        if standalone:
            self._begin(["topic", "script"] + (["review"] if not s.offline and s.script_review != "off" else [])
                        + ["done"])
        self._step("topic")
        if url:
            pick = topics.from_url(url)
        elif document:
            pick = topics.from_document(document)
        else:
            pick = pick_topic(s, self.history, explicit=topic, source=source)
        if topic and (url or document):
            pick.subject = f"{topic} (source: {pick.subject})"
        log.info("Topic [%s]: %s", pick.source, pick.subject)
        try:
            hit = safety.sensitive_region(pick.subject)   # fail fast before writing the script
            if hit:
                raise SkipTopic(f"sensitive region ({hit}) — skipped by policy")
            if self.llm is not None and not s.offline and s.safety_filter:
                verdict = safety.check(pick.subject, self.llm)
                if verdict.unsafe:
                    raise safety.UnsafeTopic(verdict.reason)
            self._step("script", pick.subject)
            insights = analytics.insights(self.history) if s.learn_from_stats else ""
            style = style or pick.style or None
            place_locked_channel = "wikimedia" in [x.lower() for x in s.visual_sources]
            user_chose_topic = pick.source in ("manual", "url", "document", "script", "clip")
            # Ground the story in the place's real Wikipedia article FIRST: it decides history-vs-place style,
            # gates out modern attractions, and every factual claim the writer makes must come from it.
            if (self.llm is not None and not s.offline and not pick.source_text
                    and (style == "history" or place_locked_channel)):
                text, url = topics.article(pick.subject, s.language.split("-")[0])
                if text:
                    pick.source_text, pick.source_url = text, url
                    pick.context = (pick.context + "\n\n" + text).strip() if pick.context else text
                    log.info("Grounded in Wikipedia: %s (%d chars)", url, len(text))
            # Hidden natural + historic places only: skip zoos, theme parks, malls, resorts, hotels and anything
            # opened after 1950 (recognised by name or by the article's own description).
            if place_locked_channel and not user_chose_topic:
                reason = topics.modern_attraction_reason(pick.subject, pick.source_text)
                if reason:
                    raise SkipTopic(f"not a hidden natural/historic place: {reason}")
            # Place-locked channels are story-driven: the AI-illustrated "history" format only when the article
            # records real pre-1900 history; otherwise a present-day "place" story with real photos only.
            if not style and s.style not in STYLES and place_locked_channel:
                style = "history" if topics.has_real_history(pick.source_text) else "place"
                log.info("Style: %s (%s pre-1900 history in the article)",
                         style, "found" if style == "history" else "no")
            if (not style and s.learn_from_stats and s.style not in STYLES
                    and pick.source not in ("trending", "wikipedia", "reddit", "rss", "url", "document")):
                style = analytics.choose_style(self.history, sorted(set(AUTO_STYLES)))
                if style:
                    log.info("Format from the channel's results: %s", style)
            if s.seo_keywords and self.llm is not None and not s.offline and not pick.keywords:
                pick.keywords = seo.search_phrases(pick.subject, s.language, s.trends_geo)
                if pick.keywords:
                    log.info("YouTube searches: %s", "; ".join(pick.keywords[:4]))
            script = write_script(self.llm, pick.subject, s, source=pick.source, avoid=self.history.recent_titles(),
                                  context=pick.context, style=style, insights=insights,
                                  keywords=seo.keywords_block(pick.keywords))
            if self.llm is not None and s.script_review != "off":
                self._step("review", "Script Doctor")
                script = review_script(self.llm, script, s)
            if self.llm is not None and not s.offline and s.fact_check:
                script = factcheck.scrub(self.llm, script, s)
                script = factcheck.ground(script, pick.source_text, s)   # drop any invented specifics
        except BaseException:
            if pick.idea_id:
                self.history.release_idea(pick.idea_id)
            raise
        if standalone:
            self._step("done", script.title)
        used = (self.llm.last_used or self.llm.label) if self.llm else "offline"
        if self.llm is not None:
            log.info("Video LLM: %s", used)
        return script, pick, used

    # ------------------------------------------------------------------ one video
    def make(self, topic: str | None = None, *, source: str | None = None, upload: bool | None = None,
             style: str | None = None, script_file: str | Path | None = None, script: Script | None = None,
             url: str | None = None, document: str | Path | None = None,
             languages: list[str] | None = None) -> Result:
        t0 = time.time()
        s = self.s
        upload = s.upload if upload is None else upload
        languages = s.also_languages if languages is None else languages
        state: dict = {}
        pick: Topic | None = None
        writes = script is None and not script_file
        reviews = writes and not s.offline and s.script_review != "off"
        self._begin((["topic", "script"] if writes else []) + (["review"] if reviews else [])
                    + ["voice", "footage", "captions", "render", "check"] + (["upload"] if upload else [])
                    + ["translate"] * len([x for x in dict.fromkeys(languages or []) if x != s.language]) + ["done"])
        try:
            if script is not None or script_file:
                if script is None:
                    script = load_script(script_file, s)
                pick, writer = Topic(script.topic, "script"), ("script file" if script_file else "editor")
                log.info("Script from %s (%d scenes): %s", script_file or "editor", len(script.scenes), script.title)
                if script_file and self.llm is not None and not s.offline:
                    # Re-verify a reused script before producing (no re-writing): safety + fact-check still apply.
                    if s.safety_filter:
                        verdict = safety.check(script.topic or script.title, self.llm)
                        if verdict.unsafe:
                            raise safety.UnsafeTopic(verdict.reason)
                    if s.fact_check:
                        script = factcheck.scrub(self.llm, script, s)
            else:
                script, pick, writer = self.draft(topic, source=source, style=style, url=url, document=document)
            return self.produce(script, pick, writer, upload=upload, t0=t0, languages=languages, state=state)
        except (youtube.NotAuthorized, LLMConfigError, KeyboardInterrupt):
            raise  # setup problems stop the run instead of failing every video the same way
        except Cancelled:
            if pick is not None and pick.idea_id and not state.get("record_id"):
                self.history.release_idea(pick.idea_id)
            if state.get("folder") and not state.get("record_id"):
                shutil.rmtree(state["folder"], ignore_errors=True)  # nothing usable was made
            log.info("Stopped: the job was cancelled")
            raise
        except safety.UnsafeTopic as e:
            log.warning("Skipped (travel advisory): %s", redact(e))
            return Result(False, folder=state.get("folder"), status="skipped", error=str(e),
                          seconds=time.time() - t0)
        except SkipTopic as e:
            log.warning("Skipped (visuals): %s", redact(e))
            return Result(False, folder=state.get("folder"), status="skipped", error=str(e),
                          seconds=time.time() - t0)
        except Exception as e:
            if pick is not None and pick.idea_id and not state.get("record_id"):
                self.history.release_idea(pick.idea_id)
            log.error("Video failed: %s", redact(e), exc_info=log.isEnabledFor(logging.DEBUG))
            notify.send(s, "failed", title=getattr(script, "title", "") or (pick.subject if pick else ""),
                        detail=redact(e))
            return Result(False, folder=state.get("folder"), status="failed", error=redact(e),
                          seconds=time.time() - t0)

    def produce(self, script: Script, pick: Topic, writer: str, *, upload: bool, t0: float | None = None,
                languages: list[str] | None = None, media_map: dict[tuple[int, int], MediaItem] | None = None,
                credits: set[str] | None = None, parent_id: int | None = None, settings: Settings | None = None,
                state: dict | None = None) -> Result:
        t0 = t0 or time.time()
        state = state if state is not None else {}
        s = settings or self.s
        if script.language and script.language != s.language:
            s = replace(s, language=script.language, tts_voice="", tts_voice_b="")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{slugify(script.title)}" + (f"-{s.language}" if parent_id else "")
        folder = _unique_folder(s.out_path, name).resolve()  # absolute: history outlives cwd
        state["folder"] = folder
        work = folder / "work"
        work.mkdir(parents=True, exist_ok=True)
        (folder / "script.json").write_text(json.dumps(script.to_dict(), indent=2, ensure_ascii=False),
                                            encoding="utf-8")
        try:
            # Place-locked visuals (Wikimedia): resolve the real place and, before spending any time,
            # run the travel-advisory check on that concrete place (not just the generic topic).
            place = country = ""
            image_credits: list[dict] = []
            place_locked = (media_map is None and self.llm is not None and not s.offline
                            and "wikimedia" in [x.lower() for x in s.visual_sources])
            if place_locked:
                place, country = resolve_place(self.llm, script.title, script.topic,
                                               " ".join(sc.narration for sc in script.scenes[:3]))
                if not place:
                    # Fail closed: without the place we can't guarantee place-accurate, attributed visuals.
                    raise SkipTopic("could not identify the place for place-accurate visuals")
                log.info("Place: %s", ", ".join(x for x in (place, country) if x))
                if s.safety_filter:
                    v = safety.check(", ".join(x for x in (place, country) if x), self.llm)
                    if v.unsafe:
                        raise safety.UnsafeTopic(v.reason)
                # Don't let the autopilot repeat a place made recently (matched by resolved place, not title).
                # A user who explicitly asks for a topic (source "manual"/url/document) is never blocked.
                if pick.source not in ("manual", "script", "clip", "url", "document"):
                    if place_recently_used(place, country, self.history.recent_places(PLACE_REUSE_DAYS)):
                        raise SkipTopic(f"place '{place}' was already used in the last {PLACE_REUSE_DAYS} days")
            # Sensitive-region policy skip (separate from travel advisories): by the RESOLVED place's
            # region/province ONLY — not by words that merely appear in the narration or article. A village
            # that mentions a nearby border (e.g. Chitkul near Tibet) is not in that region and is not blocked;
            # a place that resolves to Lhasa or Kashgar is.
            hit = safety.sensitive_region(place, country)
            if hit:
                raise SkipTopic(f"sensitive region ({hit}) — skipped by policy")
            self._step("voice", f"{s.tts_engine} voice")
            speech = self._speak(script, work, s)
            total = round(speech.duration + END_PAD, 3)
            chat = script.style == "chat"
            timeline = scene_timeline([sc.narration for sc in script.scenes], speech.words, total, s.language,
                                      starts=speech.line_starts, min_len=4.5 if chat else 1.2)
            # A chat story's picture is mostly the phone screen: more cuts behind it would only cost time.
            shots = split_shots(timeline, speech.words, 0 if chat else s.max_shot_seconds)
            joins = [a.index == b.index for a, b in zip(shots, shots[1:])]
            T = s.transition_seconds if s.transition != "none" else 0.0
            background = background_item(s, total)
            split = background is not None and s.background_layout == "split"
            if background is not None:
                log.info("Background (%s): %s from %.0fs", s.background_layout, background.id,
                         background.extra.get("start", 0))

            self._step("footage", f"{len(timeline)} scenes, {len(shots)} shots")
            if background is not None and not split:
                # The whole Short plays over one continuous stretch of the background video.
                shots, joins, media, credits = [SceneTiming(0, 0.0, total)], [], [background], set()
            elif media_map:
                media = _media_for(shots, media_map)
                credits = set(credits or ())
            else:
                # split layout: the footage fills the top half, so search for the shape of that half
                budget = (NeuronBudget(self.history, s.cloudflare_daily_neurons, s.cloudflare_vision_reserve)
                          if s.cloudflare_account_id and s.cloudflare_api_token else None)
                visuals = Visuals(replace(s, resolution=split_sizes(s)[0]) if split else s,
                                  self.history.used_media(), llm=self.llm, budget=budget)
                if place_locked and place:
                    # Hybrid, one image per scene: a real Wikimedia photo of the place today, or an AI
                    # illustration for scenes set in the past. All shots of a scene share its image (a new
                    # camera move each), so a historical scene is generated once, not per shot. AI past scenes
                    # are only for the "history" format; the "place" format is present-day, real photos only.
                    # AI past scenes only for the "history" style, and kept grounded: first & last scene real,
                    # at least ~40% of scenes real, at most 5 AI scenes (see plan_ai_scenes).
                    hist = plan_ai_scenes([bool(sc.historical) and script.style == "history"
                                           for sc in script.scenes])
                    scene_jobs = [(sc.search_query, sc.image_prompt, h) for sc, h in zip(script.scenes, hist)]
                    if script.scenes:
                        log.info("Scenes: %d total, %d AI, %d real photos (first & last real)",
                                 len(hist), sum(hist), len(hist) - sum(hist))
                    coords = topics.place_coordinates(", ".join(x for x in (place, country) if x) or place,
                                                      s.language.split("-")[0])
                    if coords:
                        log.info("Coordinates: %.4f, %.4f (Commons images within 5 km)", *coords)
                    per_scene = visuals.for_place_scenes(
                        scene_jobs, place, country, work / "media", progress=self._sub,
                        min_images=s.min_place_images, lat=coords[0] if coords else None,
                        lon=coords[1] if coords else None, article=pick.source_text, title=script.title)
                    media = [per_scene[st.index] for st in shots]
                else:
                    # AI images cost money or are rate-limited: the extra shots of a scene reuse its picture with
                    # a new camera move instead of asking for another one.
                    ai_first = bool(s.visual_sources) and s.visual_sources[0] in AI_SOURCES
                    wanted = [st for st in shots if st.part == 0 or not ai_first]
                    jobs = [(script.scenes[st.index].search_query, script.scenes[st.index].image_prompt,
                             st.duration + T) for st in wanted]
                    scene_media = visuals.for_scenes(jobs, script.topic, work / "media", progress=self._sub)
                    got = dict(zip(((st.index, st.part) for st in wanted), scene_media))
                    media = [got.get((st.index, st.part)) or got[(st.index, 0)] for st in shots]
                credits = visuals.credits
                image_credits = visuals.image_credits
            by_scene = {(st.index, st.part): m for st, m in zip(shots, media)}

            self._step("captions")
            chunks = caption_chunks(speech.words, s.caption_max_words, total=total)
            labels = ({"A": script.speaker_name("A"), "B": script.speaker_name("B")}
                      if script.style == "dialogue" and script.multi_speaker else None)
            messages = ([ChatMessage(sc.speaker, sc.narration, st) for sc, st in zip(script.scenes, speech.line_starts)]
                        if chat and speech.line_starts else None)
            card, card_until, on_screen = None, 0.0, chunks
            if script.style in CARD_STYLES:
                # The post card shows the title while it is read; captions start with the story.
                n0 = len(tokenize(script.scenes[0].narration, s.language))
                if 0 < n0 < len(speech.words):
                    card_until = min(speech.words[n0 - 1].end + 0.35, total * 0.6)
                    community, user = (script.cast + DEFAULT_CAST["reddit"][len(script.cast):])[:2]
                    card = (community, user, script.scenes[0].narration)
                    on_screen = caption_chunks(speech.words[n0:], s.caption_max_words, total=total)
            emoji = []  # each scene's emoji pops up as the scene starts (not over a chat or a post card)
            for k, (sc, tm) in enumerate(zip(script.scenes, timeline)):
                if sc.emoji and not chat and not (card and k == 0):
                    emoji.append((tm.start + 0.05, min(tm.end, tm.start + 1.65), sc.emoji))
            # Story videos (history/place) end on a calm payoff, so no "Follow for more!" end-card is burned in.
            ov = replace(s, end_cta="") if script.style in STORY_STYLES else s
            plan = build_overlays(ov, on_screen, total, script.hook_text, work, labels=labels, chat=messages,
                                  contact=script.speaker_name("A"), card=card, card_until=card_until,
                                  emphasis={k for sc in script.scenes for k in sc.emphasis}, emoji=emoji)

            self._step("render", f"{s.width}x{s.height}")
            video = folder / ("short.mp4" if s.vertical else "video.mp4")
            effects = sfx.events_for(scene_starts=[st.start for st, j in zip(shots[1:], joins) if not j],
                                     message_starts=speech.line_starts if messages else None,
                                     hook=bool(plan.hook_until or plan.card_until))
            render_video(s, media, [st.duration for st in shots], speech.audio, total, plan, video, work, joins,
                         progress=self._sub, effects=effects, background=background if split else None)

            self._step("check")
            info = ffmpeg.probe(video)
            if (info["width"], info["height"]) != s.resolution or abs(info["duration"] - total) > 0.5:
                raise RuntimeError(f"Render check failed: got {info['width']}x{info['height']} "
                                   f"{info['duration']:.2f}s, expected {s.width}x{s.height} {total:.2f}s")
            cover_at = pick_cover(video, folder / "cover.jpg", work, total=total, avoid=[st.start for st in shots[1:]],
                                  until=card_until or plan.hook_until or min(2.8, total * 0.4))
            (folder / "captions.srt").write_text(srt(chunks), encoding="utf-8")

            # YouTube's synthetic-content disclosure and the description line go on only when AI images were used.
            used_ai = s.synthetic_media and any(m is not None and m.source in AI_SOURCES for m in media)
            title = final_title(script, s)
            credit_block = image_credits_block(image_credits) if s.credit_footage else ""
            extra = ([credit_block] if credit_block else []) + ([EMOJI_CREDIT] if plan.emoji and s.credit_footage else [])
            if used_ai:
                extra.append(AI_DISCLOSURE)
            if pick.source_url:
                extra.append(f"Source: {pick.source_url}")
            description = final_description(script, credits, s, extra=extra)
            localizations = {}
            wanted = [x for x in s.localize_metadata if x != s.language]
            if wanted and self.llm is not None and not s.offline:
                try:
                    for lang, v in localize_metadata(self.llm, script, wanted).items():
                        localizations[lang] = {"title": final_title(script, s, v["title"]),
                                               "description": final_description(script, credits, s,
                                                                                v["description"], extra)}
                    log.info("Title and description localised: %s", ", ".join(localizations) or "none")
                except Cancelled:
                    raise
                except Exception as e:  # a nicety: never fail the video over it
                    log.warning("Localising the title and description failed: %s", redact(e))
            meta = {
                "title": title, "description": description, "tags": script.tags,
                "category_id": script.category_id, "language": s.language,
                "topic": script.topic, "source": pick.source, "style": script.style,
                "duration": round(info["duration"], 2), "resolution": f"{s.width}x{s.height}",
                "llm": writer, "score": script.score, "rubric": script.rubric, "review": script.review,
                "search_keywords": pick.keywords, "localizations": localizations, "cover_at": round(cover_at, 2),
                "cast": script.cast,
                "voice": f"{speech.engine}:{speech.voice}", "timing": speech.timing,
                "visuals": [{"source": m.source, "id": m.id, "kind": m.kind, "query": m.query}
                            for m in media + ([background] if split else [])],
                "synthetic_media": used_ai,
                "source_url": pick.source_url,
                "parent_id": parent_id, "created_at": now_iso(),
            }
            (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            rid = self.history.add_video(
                subject=pick.subject, source=pick.source, topic=script.topic, style=script.style,
                language=s.language, title=title, description=description, tags=script.tags,
                folder=str(folder), video_path=str(video), duration=info["duration"],
                llm=meta["llm"], voice=meta["voice"], status="rendered", score=script.score, parent_id=parent_id,
                resolved_place=(", ".join(x for x in (place, country) if x) or None))
            state["record_id"] = rid
            if not media_map:
                self.history.mark_media([m.key for m in media if m.source not in REUSABLE_SOURCES])
            if pick.key and not parent_id:
                self.history.mark_topic(pick.key)
            if pick.idea_id and not parent_id:
                self.history.link_idea(pick.idea_id, rid)

            took = time.time() - t0
            log.info("Rendered %s (%.1fs video) in %.0fs -> %s", title, info["duration"], took, video)
            result = Result(True, title, folder, video, "rendered", seconds=took, record_id=rid,
                            score=script.score, language=s.language)
            if upload:
                self._step("upload")
                self.upload_record(rid, result)
            else:
                notify.send(s, "rendered", title=title, detail=str(video))

            for lang in [x for x in dict.fromkeys(languages or []) if x != s.language]:
                self._step("translate", lang)
                result.variants.append(self._translation(script, pick, writer, lang, s, by_scene, credits,
                                                         rid, upload))
            self._step("done", title)
            return result
        finally:
            if not os.getenv("KEEP_WORK"):
                shutil.rmtree(work, ignore_errors=True)

    def _translation(self, script: Script, pick: Topic, writer: str, lang: str, s: Settings,
                     by_scene: dict[tuple[int, int], MediaItem], credits: set[str], parent_id: int,
                     upload: bool) -> Result:
        t0 = time.time()
        try:
            if self.llm is None:
                raise RuntimeError("translations need an LLM (not available in demo mode)")
            translated = translate_script(self.llm, script, lang)
            translated.score, translated.review, translated.rubric = script.score, script.review, script.rubric
            log.info("Translated to %s: %s", lang, translated.title)
            self._quiet = True  # the copy's own stages are one "translate" step for the progress bar
            try:
                return self.produce(translated, pick, writer, upload=upload, t0=t0, media_map=by_scene,
                                    credits=credits, parent_id=parent_id,
                                    settings=replace(s, language=lang, tts_voice="", tts_voice_b=""))
            finally:
                self._quiet = False
        except (youtube.NotAuthorized, Cancelled):
            raise
        except Exception as e:  # including LLMConfigError: the original video is already made
            log.error("Translation to %s failed: %s", lang, redact(e))
            return Result(False, status="failed", error=redact(e), language=lang, seconds=time.time() - t0)

    # ------------------------------------------------------------------ uploading
    @property
    def _refused_on(self) -> str:
        """The quota day on which YouTube itself refused an upload. Kept in the history database, so every
        Studio job, a restarted autopilot and the MCP server all know it."""
        return self.history.get_meta("youtube_refused_on") or ""

    @_refused_on.setter
    def _refused_on(self, day: str) -> None:
        self.history.set_meta("youtube_refused_on", day or None)

    def quota_left(self) -> int:
        day = quota_day_start()
        if self._refused_on == day:  # YouTube said no today: nothing more goes up until the reset
            return 0
        return self.s.daily_upload_limit - self.history.uploads_since(day)

    def upload_record(self, rid: int, result: Result | None = None) -> str:
        """Publish a rendered video to every PUBLISH_TO destination it hasn't reached yet: YouTube (uploaded,
        or scheduled into the next free slot) and the other networks (see crosspost). Calling it again only
        retries what didn't go out. Returns the video's new status."""
        rec = self.history.get(rid)
        if not rec or not rec.get("video_path") or not Path(rec["video_path"]).exists():
            log.error("Video #%s has no file to upload", rid)
            return "missing"
        to = list(dict.fromkeys(self.s.publish_to or ["youtube"]))
        folder = Path(rec["folder"])
        meta_file = folder / "metadata.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        posted = dict(meta.get("published") or {})
        on_youtube = "youtube" in to and not rec.get("youtube_id")
        others = [p for p in to if p != "youtube" and (posted.get(p) or {}).get("status") not in crosspost.DONE]
        if not on_youtube and not others:
            log.info("Video #%d is already on %s", rid, ", ".join(to))
            return rec["status"]
        if self.quota_left() <= 0:
            log.warning("Daily upload limit (%d) reached — video #%d queued for tomorrow", self.s.daily_upload_limit, rid)
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="daily upload limit reached")
            return "queued"
        status = rec["status"]
        when = _future(rec.get("publish_at"))
        if on_youtube:
            when = youtube.next_publish_slot(self.s, self.history.scheduled_times())
            status = self._to_youtube(rid, rec, meta, folder, when, result)
            if status not in ("uploaded", "scheduled"):
                when = None
        elif "youtube" not in to and when is None and not posted:
            when = youtube.next_publish_slot(self.s, self.history.scheduled_times())  # the slot is for the networks
        if others:
            results = crosspost.publish(self.s, Path(rec["video_path"]), meta, others, when)
            posted.update({p: r.to_dict() for p, r in results.items()})
            meta["published"] = posted
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            out = [r for r in results.values() if r.status in crosspost.DONE]
            failed = [r for r in results.values() if r.status not in crosspost.DONE]
            for r in out:
                notify.send(self.s, "uploaded", title=f"{rec['title']} on {r.platform}", url=r.url, detail=r.status)
            for r in failed:
                notify.send(self.s, "failed", title=f"{rec['title']} on {r.platform}", detail=r.error)
            if "youtube" not in to:  # without YouTube, the video's status is where the networks stand
                ok = bool(out) or any((posted.get(p) or {}).get("status") in crosspost.DONE for p in to)
                status = ("scheduled" if when and out and all(r.status == "scheduled" for r in out) else
                          "uploaded") if ok else "failed"
                self.history.update_video(
                    rid, status=status, uploaded_at=now_iso() if ok else rec.get("uploaded_at"),
                    publish_at=youtube.to_rfc3339(when) if when and ok else rec.get("publish_at"),
                    error=None if not failed else "; ".join(f"{r.platform}: {r.error}" for r in failed)[:500])
            if result:
                result.posts = {p: posted[p] for p in others}
                if "youtube" not in to:
                    result.status = status
        # The MP4 can go once every destination has it (KEEP_VIDEOS=false); otherwise a retry needs it.
        youtube_done = "youtube" not in to or status in ("uploaded", "scheduled")
        everywhere = youtube_done and all((posted.get(p) or {}).get("status") in crosspost.DONE
                                          for p in to if p != "youtube")
        if everywhere and not self.s.keep_videos:
            Path(rec["video_path"]).unlink(missing_ok=True)
            self.history.update_video(rid, video_path=None)
        return status

    def _to_youtube(self, rid: int, rec: dict, meta: dict, folder: Path, publish, result: Result | None) -> str:
        body = youtube.build_body(self.s, rec["title"], rec["description"] or "",
                                  json.loads(rec["tags"] or "[]"), meta.get("category_id", "27"),
                                  meta.get("language", self.s.language), publish,
                                  localizations=meta.get("localizations"),
                                  synthetic=meta.get("synthetic_media"))
        try:
            resp = youtube.upload(self.s, Path(rec["video_path"]), body)
        except youtube.QuotaExceeded:
            log.warning("YouTube refused the upload (quota or upload limit) — video #%d queued until midnight Pacific",
                        rid)
            self._refused_on = quota_day_start()
            self.history.update_video(rid, status="queued")
            if result:
                result.status = "queued"
            notify.send(self.s, "queued", title=rec["title"], detail="YouTube's upload limit reached")
            return "queued"
        except youtube.NotAuthorized:
            raise
        except Exception as e:
            log.error("Upload of #%d failed: %s", rid, e)
            self.history.update_video(rid, status="failed", error=str(e)[:500])
            if result:
                result.status, result.error = "upload-failed", str(e)
            notify.send(self.s, "failed", title=rec["title"], detail=f"upload failed: {redact(e)}")
            return "failed"
        vid = resp.get("id")
        if vid and self.s.upload_captions:
            lang = meta.get("language", self.s.language)
            youtube.upload_captions(self.s, vid, folder / "captions.srt", lang,
                                    LANGUAGES.get(lang.split("-")[0], lang))
        status = "scheduled" if publish else "uploaded"
        publish_iso = youtube.to_rfc3339(publish) if publish else None
        self.history.update_video(rid, status=status, youtube_id=vid, uploaded_at=now_iso(), publish_at=publish_iso,
                                  error=None)
        when = f", goes public {publish.strftime('%a %d %b %H:%M %Z')}" if publish else ""
        log.info("%s: https://youtube.com/shorts/%s%s", status.capitalize(), vid, when)
        notify.send(self.s, status, title=rec["title"], url=f"https://youtube.com/shorts/{vid}",
                    detail=f"public {publish.strftime('%a %d %b %H:%M %Z')}" if publish else "")
        if result:
            result.status, result.youtube_id, result.publish_at = status, vid, publish_iso
        return status

    def flush_queue(self, include_rendered: bool = False) -> int:
        statuses = ("queued", "rendered") if include_rendered else ("queued",)
        done = 0
        for rec in self.history.pending_uploads(statuses):
            if self.quota_left() <= 0:
                break
            if self.upload_record(rec["id"]) in ("uploaded", "scheduled"):
                done += 1
        return done

    def _resolve_place_for_backfill(self, row: dict) -> tuple[str, str]:
        """Best resolved place for an existing row, and how it was found: the video's saved Wikipedia
        source_url, else the LLM resolving from its title/topic/script, else a simple heuristic."""
        folder = row.get("folder")
        title = (row.get("title") or "").strip()
        topic = (row.get("topic") or "").strip()
        narration, meta = "", {}
        if folder:
            try:
                meta = json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            try:
                sj = json.loads((Path(folder) / "script.json").read_text(encoding="utf-8"))
                narration = " ".join(str(s.get("narration", "")) for s in (sj.get("scenes") or [])[:3])
                topic = topic or str(sj.get("topic", ""))
                title = title or str(sj.get("title", ""))
            except Exception:
                pass
        # 1. The Wikipedia article the video was grounded in IS its resolved place.
        m = re.search(r"/wiki/([^?#]+)", str(meta.get("source_url") or ""))
        if m:
            from urllib.parse import unquote
            return unquote(m.group(1)).replace("_", " ").strip(), "source_url"
        # 2. Resolve from the title/topic, exactly as production does. When the LLM runs and finds no single
        #    place, trust that (don't let the heuristic invent one from a non-place title like "Ostrich Eyes:").
        if self.llm is not None:
            try:
                place, country = resolve_place(self.llm, title, topic, narration)
                if place:
                    return ", ".join(x for x in (place, country) if x), "llm"
                return "", "unresolved"
            except Exception as e:
                log.warning("Backfill resolve failed for #%s: %s", row.get("id"), redact(e))
                # fall through to the heuristic only when the LLM could not be consulted
        # 3. Heuristic fallback (LLM offline or errored): a "Place, Country" topic, or the title's lead phrase.
        head = topic.split(":")[0].strip()
        if "," in head and not head.lower().endswith("in asia"):
            return head, "topic"
        lead = title.split(":")[0].strip()
        if ":" in title and lead and lead[0].isupper() and len(lead.split()) <= 4:
            return lead, "title"
        return "", "unresolved"

    def backfill_places(self) -> list[dict]:
        """Fill resolved_place for every history row that lacks it, so the place de-dup covers everything
        already made. Returns a record per row touched: {id, title, place, via}."""
        out: list[dict] = []
        for row in self.history.videos_for_place_backfill():
            if (row.get("resolved_place") or "").strip():
                continue
            place, via = self._resolve_place_for_backfill(row)
            if place:
                self.history.update_video(row["id"], resolved_place=place)
            out.append({"id": row["id"], "title": row.get("title") or "", "place": place or None, "via": via})
        return out

    def delete(self, rid: int) -> bool:
        """Remove a video from the library and delete its folder (never touches YouTube)."""
        rec = self.history.get(rid)
        if not rec:
            return False
        folder = Path(rec["folder"]) if rec.get("folder") else None
        if folder and folder.is_dir() and folder.resolve().is_relative_to(self.s.out_path.resolve()):
            shutil.rmtree(folder, ignore_errors=True)
        self.history.delete_video(rid)
        return True

    # ------------------------------------------------------------------ autopilot
    def autopilot(self, *, count: int | None = None, once: bool = False, upload: bool | None = None,
                  topic: str | None = None, source: str | None = None) -> int:
        s = self.s
        upload = s.upload if upload is None else upload
        produced, batch, empty_batches = 0, 0, 0
        mode = "upload" if upload else "no-upload"
        _ = self.llm  # fail fast on LLM configuration problems
        log.info("Autopilot started (%s, batch %d, workers %d%s)", mode, s.batch_size, s.workers,
                 f", target {count}" if count else "")
        try:
            while True:
                batch += 1
                if s.learn_from_stats and time.time() - self._stats_synced > 12 * 3600:
                    analytics.sync_stats(s, self.history)
                    self._stats_synced = time.time()
                if upload:
                    self.flush_queue()
                    if self.quota_left() <= 0 and self.history.pending_uploads():
                        wait = seconds_until_quota_reset()
                        log.info("Upload limit reached and videos are queued; sleeping %.1fh until quota resets",
                                 wait / 3600)
                        time.sleep(wait)
                        continue
                n = s.batch_size if count is None else min(s.batch_size, count - produced)
                workers = min(s.workers, n)
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    # Parallel videos each get their own Studio: progress, timing and cancel state are per video.
                    makers = [self] if workers == 1 else [Studio(s) for _ in range(n)]
                    futs = [ex.submit(makers[i % len(makers)].make, topic, source=source, upload=upload)
                            for i in range(n)]
                    results = [f.result() for f in futs]
                ok = sum(r.ok for r in results)
                produced += ok
                log.info("Batch %d: %d/%d succeeded (total %d)", batch, ok, n, produced)
                empty_batches = empty_batches + 1 if ok == 0 else 0
                if once or (count is not None and produced >= count):
                    break
                if empty_batches >= 3:
                    log.error("Three batches in a row produced nothing — stopping. Check the log above.")
                    break
                if s.batch_delay:
                    log.info("Next batch in %ds", s.batch_delay)
                    time.sleep(s.batch_delay)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
        return produced
