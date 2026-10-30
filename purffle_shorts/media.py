"""Visual sourcing: per-scene stock footage and photos (Pexels, Pixabay), your own media folder,
and AI-generated images (OpenAI gpt-image-1 / DALL·E, Pollinations). Clips are never reused across
videos, portrait footage is preferred, and a scene always gets *something* (animated gradient)."""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import random
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import quote

from .config import Settings
from .utils import PermanentError, download, http, raise_for_status, redact, with_retries

log = logging.getLogger("purffle")

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STOP_WORDS = {"the", "a", "an", "of", "and", "in", "on", "with", "for", "to", "at", "by", "from", "closeup"}


class SkipTopic(Exception):
    """Raised when a topic can't be made credibly, e.g. too few freely-licensed images of the place.
    Handled like a travel-advisory skip: the autopilot moves on to another topic."""


@dataclass
class MediaItem:
    source: str
    id: str
    kind: str                     # video | image | generated
    url: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    credit: str = ""
    path: Path | None = None
    query: str = ""
    extra: dict = field(default_factory=dict)
    rank: int = 0                 # position in the search results (the API's own relevance order)
    text: str = ""                # what the provider says is in it: page slug, alt text, tags
    author: str = ""              # Wikimedia: the file's author/creator, for attribution
    license: str = ""             # Wikimedia: short license name, e.g. "CC BY-SA 4.0"
    license_url: str = ""         # Wikimedia: link to the license deed
    page_url: str = ""            # Wikimedia: the file's Commons description page (the work's URL)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def portrait(self) -> bool:
        return self.height > self.width


# ------------------------------------------------------------------------------------------ background video
def background_item(settings: Settings, seconds: float) -> MediaItem | None:
    """One continuous stretch of BACKGROUND_VIDEO (a file, or a random video from a folder) long enough for
    the whole Short, starting at a random point so consecutive videos don't all open on the same frames."""
    from . import ffmpeg
    if not settings.background_video:
        return None
    root = Path(settings.background_video).expanduser()
    if root.is_dir():
        files = sorted(f for f in root.rglob("*") if f.suffix.lower() in VIDEO_EXT and not f.name.startswith("."))
    else:
        files = [root] if root.is_file() else []
    if not files:
        log.warning("BACKGROUND_VIDEO=%s has no video files; using footage instead", settings.background_video)
        return None
    f = random.choice(files)
    info = ffmpeg.probe(f)
    if not info["has_video"]:
        log.warning("%s has no picture; using footage instead", f)
        return None
    dur = info["duration"]
    start = random.uniform(0.0, dur - seconds - 0.5) if dur > seconds + 1.0 else 0.0
    return MediaItem("background", f.name, "video", width=info["width"], height=info["height"], duration=dur,
                     path=f, query="background", extra={"start": round(start, 2)})


# ------------------------------------------------------------------------------------------ Pexels
def _pexels_pick_file(files: list[dict], prefer_4k: bool) -> dict | None:
    files = [f for f in files if f.get("link") and f.get("width") and f.get("height")
             and f.get("file_type", "video/mp4") == "video/mp4"]
    if not files:
        return None
    cap = 4096 if prefer_4k else 2160

    def short(f):
        return min(f["width"], f["height"])
    good = [f for f in files if 1080 <= short(f) <= cap]
    if good:
        return min(good, key=short) if not prefer_4k else max(good, key=short)
    return max(files, key=short)


def pexels_videos(query: str, key: str, prefer_4k: bool = False, orientation: str = "portrait") -> list[MediaItem]:
    params = {"query": query, "per_page": 15, "size": "medium"}
    if orientation:
        params["orientation"] = orientation
    r = http().get("https://api.pexels.com/videos/search", params=params, headers={"Authorization": key}, timeout=30)
    raise_for_status(r, "Pexels video search")
    items = []
    for rank, v in enumerate(r.json().get("videos", [])):
        f = _pexels_pick_file(v.get("video_files", []), prefer_4k)
        if not f:
            continue
        tags = " ".join(str(t) for t in v.get("tags") or [])
        items.append(MediaItem("pexels", str(v["id"]), "video", f["link"], f["width"], f["height"],
                               float(v.get("duration") or 0), (v.get("user") or {}).get("name", ""), query=query,
                               rank=rank, text=f"{_slug_words(v.get('url', ''))} {tags}".strip()))
    return items


def pexels_photos(query: str, key: str, orientation: str = "portrait") -> list[MediaItem]:
    params = {"query": query, "per_page": 15}
    if orientation:
        params["orientation"] = orientation
    r = http().get("https://api.pexels.com/v1/search", params=params, headers={"Authorization": key}, timeout=30)
    raise_for_status(r, "Pexels photo search")
    items = []
    for rank, p in enumerate(r.json().get("photos", [])):
        src = p.get("src") or {}
        url = src.get("original")
        if not url:
            continue
        url += ("&" if "?" in url else "?") + "auto=compress&cs=tinysrgb&h=2400"
        items.append(MediaItem("pexels-photo", str(p["id"]), "image", url, p.get("width", 0), p.get("height", 0),
                               credit=p.get("photographer", ""), query=query, rank=rank,
                               text=f"{p.get('alt') or ''} {_slug_words(p.get('url', ''))}".strip()))
    return items


# ------------------------------------------------------------------------------------------ Pixabay
def pixabay_videos(query: str, key: str, prefer_4k: bool = False) -> list[MediaItem]:
    r = http().get("https://pixabay.com/api/videos/", timeout=30, params={
        "key": key, "q": query[:100], "per_page": 20, "safesearch": "true"})
    raise_for_status(r, "Pixabay video search")
    order = ["large", "medium", "small"] if prefer_4k else ["medium", "large", "small"]
    items = []
    for rank, h in enumerate(r.json().get("hits", [])):
        vids = h.get("videos") or {}
        f = next((vids[q] for q in order if (vids.get(q) or {}).get("url")), None)
        if not f:
            continue
        items.append(MediaItem("pixabay", str(h["id"]), "video", f["url"], f.get("width", 0), f.get("height", 0),
                               float(h.get("duration") or 0), h.get("user", ""), query=query, rank=rank,
                               text=str(h.get("tags") or "")))
    return items


def pixabay_photos(query: str, key: str, orientation: str = "portrait") -> list[MediaItem]:
    pix = {"portrait": "vertical", "landscape": "horizontal"}.get(orientation, "all")
    r = http().get("https://pixabay.com/api/", timeout=30, params={
        "key": key, "q": query[:100], "image_type": "photo", "orientation": pix,
        "per_page": 20, "safesearch": "true"})
    raise_for_status(r, "Pixabay photo search")
    return [MediaItem("pixabay-photo", str(h["id"]), "image", h["largeImageURL"], h.get("imageWidth", 0),
                      h.get("imageHeight", 0), credit=h.get("user", ""), query=query, rank=rank,
                      text=str(h.get("tags") or ""))
            for rank, h in enumerate(r.json().get("hits", [])) if h.get("largeImageURL")]


# ------------------------------------------------------------------------------------------ Wikimedia Commons
_WIKI_API = "https://commons.wikimedia.org/w/api.php"
_WIKI_UA = "purffle-shorts/1.0 (Asia Unseen; place-accurate visuals)"
# Licenses we are NOT allowed to use (anything with NC, ND, or no free license).
_WIKI_DENY = ("non-commercial", "noncommercial", "-nc", " nc ", "no deriv", "noderiv", "-nd", " nd ",
              "fair use", "all rights reserved", "unknown", "no license")
_TAG_RE = re.compile(r"<[^>]+>")


def _wiki_license_ok(short: str) -> tuple[bool, str]:
    """(allowed, clean name) for a Commons LicenseShortName. Allow public domain / CC0 / CC-BY / CC-BY-SA only."""
    short = (short or "").strip()
    if not short:
        return False, ""
    s = f" {short.lower()} "
    if any(d in s for d in _WIKI_DENY):
        return False, ""
    flat = s.replace(" ", "")
    if "cc0" in flat or "publicdomain" in flat or "pdm" in flat or flat.strip("-").startswith("pd"):
        return True, short or "Public domain"
    if "cc-by-sa" in flat or "ccbysa" in flat or "by-sa" in flat:
        return True, short
    if "cc-by" in flat or "ccby" in flat or " by " in s:
        return True, short
    return False, ""


def _wiki_author(artist_html: str) -> str:
    txt = _TAG_RE.sub(" ", artist_html or "").replace("&amp;", "&")
    txt = re.sub(r"\s+", " ", txt).strip(" ,;")
    return txt[:120] or "Unknown author"


def _wiki_title(title: str) -> str:
    t = re.sub(r"^File:", "", title or "")
    t = re.sub(r"\.(jpe?g|png|webp|tiff?|gif)$", "", t, flags=re.I)
    return re.sub(r"[_]+", " ", t).strip() or "Wikimedia image"


def wikimedia_images(query: str, limit: int = 20, min_px: int = 800) -> list[MediaItem]:
    """Freely-licensed (public-domain / CC-BY / CC-BY-SA) photos of ``query`` from Wikimedia Commons.
    Author and license are captured on each item so the video can credit them as the licenses require."""
    r = http().get(_WIKI_API, timeout=30, headers={"User-Agent": _WIKI_UA}, params={
        "action": "query", "format": "json", "formatversion": 2,
        "generator": "search", "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
        "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 1600})
    raise_for_status(r, "Wikimedia Commons search")
    pages = (r.json().get("query") or {}).get("pages") or []
    items: list[MediaItem] = []
    for rank, pg in enumerate(sorted(pages, key=lambda p: p.get("index", 999))):
        info = (pg.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        ok, lic = _wiki_license_ok((meta.get("LicenseShortName") or {}).get("value", ""))
        if not ok:
            continue
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        if min(w, h) < min_px:
            continue
        url = info.get("url") or info.get("thumburl")
        if not url:
            continue
        title = pg.get("title", "")
        page_url = info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}"
        author = _wiki_author((meta.get("Artist") or {}).get("value", ""))
        lic_url = (meta.get("LicenseUrl") or {}).get("value", "")
        obj = _TAG_RE.sub(" ", (meta.get("ObjectName") or {}).get("value", "") or "")
        cats = (meta.get("Categories") or {}).get("value", "")
        items.append(MediaItem("wikimedia", str(pg.get("pageid") or title), "image", url=url, width=w, height=h,
                               credit=author, author=author, license=lic, license_url=lic_url, page_url=page_url,
                               query=query, rank=rank, text=f"{title} {obj} {cats}".strip(),
                               extra={"wiki_title": _wiki_title(title)}))
    return items


def resolve_place(llm, title: str, topic: str, sample: str = "") -> tuple[str, str]:
    """Best-effort (place, country) the video is actually about — for place-accurate visuals and the
    per-place travel-advisory check. Returns ("", "") when there is no single real place."""
    if llm is None:
        return "", ""
    system = "You extract the single real-world place a travel video is about. Reply only with strict JSON."
    user = ('Return STRICT JSON {"place": "<most specific real place/landmark/city, or empty>", '
            '"country": "<country, or empty>"}. If there is no single real place, use empty strings.\n'
            f'Title: {title}\nTopic: {topic}\nFirst lines: {sample[:600]}')
    try:
        d = llm.complete_json(system, user, max_tokens=120, temperature=0.0)
    except Exception as e:
        log.warning("Place resolve failed (%s)", redact(e))
        return "", ""
    if not isinstance(d, dict):
        return "", ""
    return str(d.get("place", "")).strip(), str(d.get("country", "")).strip()


# ------------------------------------------------------------------------------------------ AI images
def openai_image(prompt: str, dest: Path, key: str, model: str, orientation: str = "portrait") -> Path:
    body = {"model": model, "prompt": prompt, "n": 1}
    if model.startswith("dall-e"):
        size = {"portrait": "1024x1792", "landscape": "1792x1024"}.get(orientation, "1024x1024")
        body.update(size=size, response_format="b64_json", quality="hd" if model == "dall-e-3" else "standard")
    else:
        size = {"portrait": "1024x1536", "landscape": "1536x1024"}.get(orientation, "1024x1024")
        body.update(size=size, quality="medium")

    def _do():
        r = http().post("https://api.openai.com/v1/images/generations", json=body, timeout=240,
                        headers={"Authorization": f"Bearer {key}"})
        raise_for_status(r, "OpenAI image")
        d = r.json()["data"][0]
        if d.get("b64_json"):
            dest.write_bytes(base64.b64decode(d["b64_json"]))
        else:
            download(d["url"], dest)
        return dest

    return with_retries(_do, attempts=2, label="OpenAI image")


_POLLINATIONS_LOCK = threading.Lock()


def pollinations_image(prompt: str, dest: Path, width: int, height: int, key: str = "") -> Path:
    url = (f"https://image.pollinations.ai/prompt/{quote(prompt[:900])}"
           f"?width={width}&height={height}&nologo=true&model=flux&seed={random.randint(1, 10**9)}")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    # One request at a time: parallel scenes would all be rate-limited. Since September 2026 anonymous
    # use is limited to roughly one image every few minutes (HTTP 402 "payment required" otherwise).
    with _POLLINATIONS_LOCK:
        return download(url, dest, headers=headers, timeout=180, max_bytes=30 * 1024 * 1024)


# ------------------------------------------------------------------------------------------ selection
def simplify_query(query: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", query.lower()) if w not in STOP_WORDS]
    return " ".join(words[:2])


def _slug_words(url: str) -> str:
    """'https://www.pexels.com/video/octopus-swimming-in-the-sea-2169880/' -> 'octopus swimming in the sea'."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return " ".join(w for w in re.split(r"[-_]+", slug) if w and not w.isdigit())


def _stem(word: str) -> str:
    """A tiny English stemmer, enough to match 'octopuses'/'octopus', 'swimming'/'swims', 'cities'/'city',
    'diving'/'dives' between a scene's query and a clip's description."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "shes", "ches", "xes", "zes", "uses")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "ls":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[A-Za-z]+", text.lower()) if w not in STOP_WORDS and len(w) > 1}


def relevance(item: MediaItem, query: str) -> float:
    """0..1: how much of the scene's search query the clip's own description covers. Clips without a
    description (local files, AI images) count as neutral."""
    want = _terms(query)
    if not want or not item.text:
        return 0.5
    have = _terms(item.text)
    return len(want & have) / len(want)


def _shape(item: MediaItem) -> str:
    if item.height > item.width:
        return "portrait"
    return "landscape" if item.width > item.height else "square"


def _score(item: MediaItem, need: float, orientation: str = "portrait", query: str = "") -> float:
    """Higher is better. Relevance to the scene matters most, then the search engine's own ranking, then
    whether the clip fits the frame and lasts long enough."""
    s = 4.0 * relevance(item, query or item.query)
    s += 2.0 / (1.0 + 0.25 * item.rank)
    if _shape(item) == orientation or (orientation == "square" and item.width and item.height):
        s += 2.5
    if item.kind == "video":
        s += 2 if item.duration >= need else (1 if item.duration >= need * 0.6 else 0)
    if min(item.width, item.height) >= 1080:
        s += 1
    return s + random.uniform(0, 0.6)  # a little variety between near-identical candidates


class Visuals:
    def __init__(self, settings: Settings, used_keys: set[str] | None = None):
        self.s = settings
        self.used = set(used_keys or ())
        self.lock = threading.Lock()
        self._search_cache: dict[tuple[str, str], list[MediaItem]] = {}
        self.sources = [x for x in settings.visual_sources if x]
        if settings.offline:
            self.sources = ["local"]
        self.credits: set[str] = set()
        self.image_credits: list[dict] = []          # per Wikimedia image: title, author, license, urls
        self._wiki_cache: dict[str, list[MediaItem]] = {}

    # -- search per source (cached per query) --
    def _search(self, source: str, query: str) -> list[MediaItem]:
        ck = (source, query.lower())
        if ck in self._search_cache:
            return self._search_cache[ck]
        s = self.s
        items: list[MediaItem] = []
        try:
            o = s.orientation
            if source == "pexels" and s.pexels_api_key:
                items = pexels_videos(query, s.pexels_api_key, s.prefer_4k, orientation=o)
                if len(items) < 3:
                    items += pexels_videos(query, s.pexels_api_key, s.prefer_4k, orientation="")
                if not items:
                    items = pexels_photos(query, s.pexels_api_key, orientation=o)
            elif source == "pexels-photos" and s.pexels_api_key:
                items = pexels_photos(query, s.pexels_api_key, orientation=o)
            elif source == "pixabay" and s.pixabay_api_key:
                items = (pixabay_videos(query, s.pixabay_api_key, s.prefer_4k)
                         or pixabay_photos(query, s.pixabay_api_key, orientation=o))
            elif source == "pixabay-photos" and s.pixabay_api_key:
                items = pixabay_photos(query, s.pixabay_api_key, orientation=o)
            elif source == "wikimedia":
                items = self._wiki_search(query)
            elif source == "local":
                items = self._local(query)
            seen: set[str] = set()
            items = [i for i in items if not (i.key in seen or seen.add(i.key))]
        except PermanentError as e:
            log.warning("%s search failed permanently (%s); disabling it for this video", source, redact(e))
            self.sources = [x for x in self.sources if x != source]
        except Exception as e:
            log.warning("%s search for %r failed: %s", source, query, redact(e))
        self._search_cache[ck] = items
        return items

    def _local(self, query: str) -> list[MediaItem]:
        root = Path(self.s.media_dir)
        if not root.is_dir():
            return []
        words = set(simplify_query(query).split()) | set(query.lower().split())
        # background videos (media/backgrounds, BACKGROUND_VIDEO) play under stories, never as scene footage
        skip = [(root / "backgrounds").resolve()]
        if self.s.background_video:
            skip.append(Path(self.s.background_video).expanduser().resolve())
        items = []
        for p in root.rglob("*"):
            ext = p.suffix.lower()
            if ext not in VIDEO_EXT | IMAGE_EXT:
                continue
            rp = p.resolve()
            if any(rp == d or d in rp.parents for d in skip):
                continue
            stem = set(re.findall(r"[a-z]+", p.stem.lower()))
            item = MediaItem("local", str(p), "video" if ext in VIDEO_EXT else "image", path=p, query=query)
            item.extra["overlap"] = len(words & stem)
            items.append(item)
        items.sort(key=lambda i: i.extra["overlap"], reverse=True)
        return items

    def _claim(self, candidates: list[MediaItem], need: float, query: str = "") -> MediaItem | None:
        with self.lock:
            fresh = [c for c in candidates if c.key not in self.used]
            if not fresh:
                return None
            if fresh[0].source == "local":
                best = fresh[0] if fresh[0].extra.get("overlap") else random.choice(fresh)
            else:
                best = max(fresh, key=lambda c: _score(c, need, self.s.orientation, query))
            self.used.add(best.key)
            return best

    def _fetch(self, item: MediaItem, dest_dir: Path, idx: int) -> MediaItem:
        if item.source == "local":
            return item
        ext = ".mp4" if item.kind == "video" else ".jpg"
        item.path = download(item.url, dest_dir / f"scene{idx:02d}_{item.source}_{item.id}{ext}")
        return item

    def _ai(self, source: str, prompt: str, dest_dir: Path, idx: int) -> MediaItem | None:
        s = self.s
        full = f"{prompt}. {s.visual_style}. No text, no letters, no watermark."
        dest = dest_dir / f"scene{idx:02d}_ai.png"
        try:
            if source in ("openai-images", "ai", "dalle", "gpt-image") and s.openai_api_key:
                openai_image(full, dest, s.openai_api_key, s.image_model, s.orientation)
            elif source == "pollinations":
                pollinations_image(full, dest.with_suffix(".jpg"), s.width, s.height, s.pollinations_api_key)
                dest = dest.with_suffix(".jpg")
            else:
                return None
        except PermanentError as e:
            # A refused key or an exhausted free tier won't recover within this video: stop asking.
            with self.lock:
                if source in self.sources:
                    self.sources = [x for x in self.sources if x != source]
                    hint = (" Anonymous Pollinations use is now heavily rate-limited; set POLLINATIONS_API_KEY "
                            "or use Pexels/Pixabay (free keys)." if source == "pollinations" and " 402" in str(e)
                            else "")
                    log.warning("AI images from %s refused (%s); skipping it for this video.%s",
                                source, redact(e)[:160], hint)
            return None
        except Exception as e:
            log.warning("AI image (%s) for scene %d failed: %s", source, idx + 1, redact(e))
            return None
        log.info("Scene %d: %s image", idx + 1, source)
        return MediaItem(source, dest.name, "image", path=dest, width=s.width, height=s.height, query=prompt)

    # -- place-locked sourcing: every clip is a freely-licensed Wikimedia photo of the named place --
    def _wiki_search(self, query: str) -> list[MediaItem]:
        q = (query or "").strip()
        if not q:
            return []
        if q.lower() in self._wiki_cache:
            return self._wiki_cache[q.lower()]
        try:
            items = wikimedia_images(q)
        except Exception as e:
            log.warning("Wikimedia search for %r failed: %s", q, redact(e))
            items = []
        self._wiki_cache[q.lower()] = items
        return items

    def _claim_wiki(self, cands: list[MediaItem], query: str) -> MediaItem | None:
        with self.lock:
            fresh, seen = [], set()
            for c in cands:
                if c.key in self.used or c.key in seen:
                    continue
                seen.add(c.key)
                fresh.append(c)
            if not fresh:
                return None
            best = max(fresh, key=lambda c: relevance(c, query) * 3 + 1.0 / (1 + 0.25 * c.rank))
            self.used.add(best.key)
            return best

    def _place_stock_video(self, query: str, place: str, country: str, dest_dir: Path, idx: int) -> MediaItem | None:
        """Stock VIDEO fallback — allowed only when the clip's own metadata names the place (or country)."""
        tokens = {t for t in re.findall(r"[a-z]+", f"{place} {country}".lower()) if len(t) > 2}
        if not tokens:
            return None
        for source in [x for x in self.sources if x in ("pexels", "pixabay")]:
            for q in dict.fromkeys([f"{query} {place}".strip(), f"{place} {country}".strip(), place]):
                for it in self._search(source, q):
                    if it.kind != "video":
                        continue
                    have = set(re.findall(r"[a-z]+", f"{it.text} {it.query}".lower()))
                    if not (tokens & have):
                        continue
                    with self.lock:
                        if it.key in self.used:
                            continue
                        self.used.add(it.key)
                    try:
                        self._fetch(it, dest_dir, idx)
                    except Exception as e:
                        log.warning("Stock video download failed for %s: %s", it.key, redact(e))
                        continue
                    return it
        return None

    def _record_credit(self, item: MediaItem) -> None:
        if item.source == "wikimedia":
            self.credits.add("Wikimedia Commons")
            self.image_credits.append({"title": item.extra.get("wiki_title") or item.id, "author": item.author,
                                       "license": item.license, "license_url": item.license_url,
                                       "page_url": item.page_url})
        elif item.source.startswith("pexels"):
            self.credits.add("Pexels")
        elif item.source.startswith("pixabay"):
            self.credits.add("Pixabay")

    def for_place_scenes(self, jobs: list[tuple[str, str, float]], place: str, country: str, dest_dir: Path,
                         progress=None, min_images: int = 5) -> list[MediaItem]:
        """One Wikimedia image per scene, all of the named place. Raises SkipTopic if the place has fewer than
        ``min_images`` freely-licensed images. Stock video is used only as a place-named fallback."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        pc = ", ".join(x for x in (place, country) if x)
        pool, seen = [], set()
        for q in dict.fromkeys([x for x in (pc, place, f"{place} temple", f"{place} landscape") if x]):
            for it in self._wiki_search(q):
                if it.key not in seen:
                    seen.add(it.key)
                    pool.append(it)
        log.info("Wikimedia: %d freely-licensed image(s) for %s", len(pool), pc or place)
        if len(pool) < min_images:
            raise SkipTopic(f"only {len(pool)} freely-licensed Wikimedia image(s) for '{pc or place}'; "
                            f"need {min_images}")
        results: list[MediaItem] = []
        for i, (q, _prompt, _need) in enumerate(jobs):
            item = self._claim_wiki(self._wiki_search(f"{q} {place}") + pool, q)
            if item is not None:
                try:
                    self._fetch(item, dest_dir, i)
                except Exception as e:
                    log.warning("Wikimedia download failed (%s)", redact(e))
                    item = None
            if item is None:                                   # no fresh place image -> place-named stock video
                item = self._place_stock_video(q, place, country, dest_dir, i)
            if item is None:                                   # reuse the best place photo, with its own camera move
                item = replace(max(pool, key=lambda c: relevance(c, q)))
                try:
                    self._fetch(item, dest_dir, i)
                except Exception as e:
                    log.warning("Wikimedia reuse download failed (%s)", redact(e))
                    item = MediaItem("generated", f"gradient-{i}", "generated", query=q)
            self._record_credit(item)
            log.info("Scene %d: %s %s (%r)", i + 1, item.source, item.kind, q)
            results.append(item)
            if progress:
                progress((i + 1) / len(jobs))
        return results

    def for_scene(self, idx: int, query: str, image_prompt: str, topic: str, need: float, dest_dir: Path) -> MediaItem:
        queries = []
        for q in (query, simplify_query(query), simplify_query(topic), topic):
            q = (q or "").strip()
            if q and q.lower() not in [x.lower() for x in queries]:
                queries.append(q)
        for source in list(self.sources):
            if source in ("openai-images", "ai", "dalle", "gpt-image", "pollinations"):
                item = self._ai(source, image_prompt or query, dest_dir, idx)
                if item:
                    return item
                continue
            for q in queries:
                cand = self._search(source, q)
                item = self._claim(cand, need, query)  # always judged against the scene's own query
                if not item:
                    continue
                try:
                    self._fetch(item, dest_dir, idx)
                except Exception as e:
                    log.warning("Download failed for %s: %s", item.key, redact(e))
                    continue
                self._record_credit(item)   # always attribute (Wikimedia author+license, or Pexels/Pixabay)
                log.info("Scene %d: %s %s (%r)", idx + 1, item.source, item.kind, q)
                return item
        log.info("Scene %d: no footage found for %r — using an animated background", idx + 1, query)
        return MediaItem("generated", f"gradient-{idx}", "generated", query=query)

    def for_scenes(self, jobs: list[tuple[str, str, float]], topic: str, dest_dir: Path,
                   progress=None) -> list[MediaItem]:
        """jobs: [(search_query, image_prompt, seconds_needed)] -> one MediaItem per scene. ``progress`` gets
        the fraction of scenes that have their footage."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        workers = 2 if any(s in ("openai-images", "ai", "pollinations") for s in self.sources) else 6
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self.for_scene, i, q, p, topic, need, dest_dir) for i, (q, p, need) in enumerate(jobs)]
            for done, _ in enumerate(concurrent.futures.as_completed(futs), 1):
                if progress:
                    progress(done / len(futs))
            return [f.result() for f in futs]
