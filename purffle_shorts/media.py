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
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from . import safety
from .config import Settings
from .history import distance_km
from .llm import ImageUnusable
from .utils import PermanentError, download, http, raise_for_status, redact, with_retries

log = logging.getLogger("purffle")

# Rough Cloudflare Workers AI neuron cost of one FLUX.1-schnell image (used only to keep a daily reserve for
# the vision check; the exact figure doesn't matter, only that gen stops before it eats the reserve).
CF_FLUX_NEURONS = 720


class NeuronBudget:
    """Tracks Cloudflare Workers AI neurons spent on image generation today (persisted in history meta) and
    keeps a reserve so the vision check always has allocation. Image generation stops at the reserve."""

    def __init__(self, history, daily: int, reserve: int):
        self.history, self.daily, self.reserve = history, daily, reserve
        self.key = f"cf_neurons:{datetime.now(timezone.utc):%Y-%m-%d}"

    def _spent(self) -> int:
        try:
            return int(self.history.get_meta(self.key) or 0)
        except (ValueError, TypeError):
            return 0

    def can_spend(self, cost: int) -> bool:
        return self._spent() + cost <= max(0, self.daily - self.reserve)

    def record(self, cost: int) -> None:
        try:
            self.history.set_meta(self.key, str(self._spent() + cost))
        except Exception:  # accounting is best-effort; never fail a render over it
            pass

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STOP_WORDS = {"the", "a", "an", "of", "and", "in", "on", "with", "for", "to", "at", "by", "from", "closeup"}


NO_ANSWER = "no-answer"      # verify_subject: a vision check was expected but no backend answered for this image
VISION_GIVE_UP_FRESH = 3     # consecutive images with no answer, none ever answered  -> vision is down
VISION_GIVE_UP_STREAK = 6    # consecutive images with no answer, even after some answered -> it went down


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


# Titles/categories that mark a file we never want as footage: maps & diagrams, signs/notices/text boards
# (they read as on-screen text), food/dishes/restaurants/people-eating, or butchery/carcasses.
_SKIP_IMAGE_RE = re.compile(
    r"(?<![a-z])(maps?|satellite|topographic|locator|diagram|schematic|floor ?plan|site ?plan|"
    r"sketch ?map|street ?map|road ?map|cadastral|"
    r"signs?|signboards?|signposts?|noticeboards?|notices?|warnings?|billboards?|boards?|plaques?|"
    r"food|foods|dish|dishes|meals?|cuisine|restaurants?|cafes?|dining|buffets?|thali|curry|curries|"
    r"breakfast|lunch|dinner|snacks?|eating|eaten|feast|"
    r"butcher\w*|carcass\w*|slaughter\w*|abattoir|flayed|animal corpse|dead animal)(?![a-z])", re.I)
# Titles that suggest a close-up identifiable person — preferred against, not skipped outright.
_PEOPLE_RE = re.compile(r"(?<![a-z])(portraits?|selfies?|close[ -]?ups?|headshots?|posing|mugshot)(?![a-z])", re.I)
# Titles whose MAIN subject is an animal or a person — never used for a landmark scene when the vision check
# can't run (fix for a "temple" file that is really a monkey or a tourist).
_ANIMAL_PEOPLE_RE = re.compile(
    r"(?<![a-z])(monkeys?|langurs?|macaques?|dogs?|cats?|cows?|bulls?|goats?|sheep|birds?|peacocks?|"
    r"elephants?|horses?|donkeys?|monkey|animals?|wildlife|"
    r"portraits?|selfies?|men|women|man|woman|child|children|boys?|girls?|people|crowd|crowds|"
    r"tourists?|pilgrims?|sadhus?|priests?|family|families|group photo)(?![a-z])", re.I)

# Stereoscopic / 3D source images (side-by-side pairs, anaglyphs) render as a split or doubled frame with a
# seam down the middle, not a flat photo — skip them by title/description keyword like we skip maps and signs.
_STEREO_RE = re.compile(r"(?<![a-z0-9])(3[ _-]?d|stereo\w*|anaglyph\w*|sbs)(?![a-z])", re.I)


def _wiki_item(pg: dict, rank: int, min_px: int, query: str = "") -> MediaItem | None:
    """Turn one Commons page (from search or geosearch) into a licensed MediaItem, or None when it is
    non-free, too small, has no URL, or is a map/diagram/butchery image we never want as footage."""
    info = (pg.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata") or {}
    ok, lic = _wiki_license_ok((meta.get("LicenseShortName") or {}).get("value", ""))
    if not ok:
        return None
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    if min(w, h) < min_px:
        return None
    url = info.get("url") or info.get("thumburl")
    if not url:
        return None
    title = pg.get("title", "")
    page_url = info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}"
    author = _wiki_author((meta.get("Artist") or {}).get("value", ""))
    lic_url = (meta.get("LicenseUrl") or {}).get("value", "")
    obj = _TAG_RE.sub(" ", (meta.get("ObjectName") or {}).get("value", "") or "")
    cats = (meta.get("Categories") or {}).get("value", "")
    blob = f"{title} {obj} {cats}"
    if _SKIP_IMAGE_RE.search(blob):                             # a map/diagram or butchery/carcass image
        return None
    if _STEREO_RE.search(blob):                                 # a stereoscopic/3D/anaglyph pair — renders split
        return None
    extra: dict = {"wiki_title": _wiki_title(title)}
    lat, lon = _image_coords(pg, meta)
    if lat is not None:
        extra.update(lat=lat, lon=lon)
    return MediaItem("wikimedia", str(pg.get("pageid") or title), "image", url=url, width=w, height=h,
                     credit=author, author=author, license=lic, license_url=lic_url, page_url=page_url,
                     query=query, rank=rank, text=f"{title} {obj} {cats}".strip(), extra=extra)


def _image_coords(pg: dict, meta: dict) -> tuple[float | None, float | None]:
    """Where a Commons photo was taken: its page coordinates, else the GPS fields of its metadata."""
    co = (pg.get("coordinates") or [{}])[0]
    try:
        if "lat" in co and "lon" in co:
            return float(co["lat"]), float(co["lon"])
        la, lo = (meta.get("GPSLatitude") or {}).get("value"), (meta.get("GPSLongitude") or {}).get("value")
        if la not in (None, "") and lo not in (None, ""):
            return float(la), float(lo)
    except (TypeError, ValueError):
        pass
    return None, None


FOREIGN_KM = 50          # a name-matched photo taken farther than this from the place shows somewhere else
_COUNTRY_WORDS = ("india", "nepal", "bhutan", "china", "pakistan", "bangladesh", "sri lanka", "maldives", "myanmar",
                  "burma", "thailand", "laos", "cambodia", "vietnam", "malaysia", "singapore", "indonesia",
                  "philippines", "brunei", "timor-leste", "east timor", "japan", "south korea", "north korea",
                  "korea", "taiwan", "mongolia", "kazakhstan", "kyrgyzstan", "tajikistan", "turkmenistan",
                  "uzbekistan", "afghanistan", "iran")


def _foreign_photo(item: MediaItem, lat: float | None, lon: float | None, country: str) -> str:
    """Why a photo whose title names the place actually shows another place, or "". By its coordinates when
    it has them (taken more than FOREIGN_KM away: "Shey Palace", Ladakh is not "Shey Gompa", Nepal); else by
    its title/categories naming a different country and not the place's own."""
    ilat, ilon = item.extra.get("lat"), item.extra.get("lon")
    if lat is not None and lon is not None and ilat is not None and ilon is not None:
        d = distance_km(lat, lon, ilat, ilon)
        return f"taken {d:.0f} km from the place" if d > FOREIGN_KM else ""
    if not country:
        return ""
    own = safety.country_names(country)
    text = (item.text or "").lower()
    named = [c for c in _COUNTRY_WORDS if re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", text)]
    mine = [c for c in named if any(c in o or o in c for o in own)]
    return f"its description places it in {named[0].title()}, not {country}" if named and not mine else ""


def _commons_query(params: dict, what: str) -> dict:
    """One Commons API query. Commons answers HTTP 200 with an ``error`` body when it is overloaded ("Search
    is currently too busy") — treated as a retryable failure, never as "no images" (which would wrongly skip
    a place for having too few photos)."""
    def call() -> dict:
        r = http().get(_WIKI_API, timeout=30, headers={"User-Agent": _WIKI_UA}, params=params)
        raise_for_status(r, what)
        data = r.json()
        err = data.get("error") if isinstance(data, dict) else None
        if err:
            raise RuntimeError(f"{what}: {err.get('info') or err.get('code') or err}")
        return data
    return with_retries(call, attempts=3, base_delay=3, label=what)


def wikimedia_images(query: str, limit: int = 20, min_px: int = 800) -> list[MediaItem]:
    """Freely-licensed (public-domain / CC-BY / CC-BY-SA) photos of ``query`` from Wikimedia Commons.
    Author and license are captured on each item so the video can credit them as the licenses require.
    Maps/diagrams and butchery/carcass images are dropped outright."""
    data = _commons_query({
        "action": "query", "format": "json", "formatversion": 2,
        "generator": "search", "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
        "gsrlimit": limit, "prop": "imageinfo|coordinates", "iiprop": "url|size|extmetadata", "iiurlwidth": 1600,
        "coprimary": "all", "colimit": "max"},
        "Wikimedia Commons search")
    pages = (data.get("query") or {}).get("pages") or []
    items: list[MediaItem] = []
    for rank, pg in enumerate(sorted(pages, key=lambda p: p.get("index", 999))):
        it = _wiki_item(pg, rank, min_px, query)
        if it is not None:
            items.append(it)
    return items


def geosearch_images(lat: float, lon: float, radius_m: int = 5000, limit: int = 40,
                     min_px: int = 800) -> list[MediaItem]:
    """Freely-licensed Commons photos geotagged within ``radius_m`` of (lat, lon). For a small or obscure
    place, a name search returns little or the wrong place; geosearch finds the photos actually taken there."""
    data = _commons_query({
        "action": "query", "format": "json", "formatversion": 2,
        "generator": "geosearch", "ggscoord": f"{lat}|{lon}", "ggsradius": int(max(10, min(10000, radius_m))),
        "ggsnamespace": 6, "ggslimit": limit,
        "prop": "imageinfo|coordinates", "iiprop": "url|size|extmetadata", "iiurlwidth": 1600,
        "coprimary": "all", "colimit": "max"}, "Wikimedia geosearch")
    pages = (data.get("query") or {}).get("pages") or []
    items: list[MediaItem] = []
    for rank, pg in enumerate(sorted(pages, key=lambda p: p.get("index", 999))):
        it = _wiki_item(pg, rank, min_px, query="")
        if it is not None:
            items.append(it)
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
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"place": {"type": "string"}, "country": {"type": "string"}},
              "required": ["place", "country"]}
    try:
        d = llm.complete_json(system, user, schema=schema, max_tokens=400, temperature=0.0)
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


def cloudflare_image(prompt: str, dest: Path, account_id: str, token: str, steps: int = 6) -> Path:
    """Cloudflare Workers AI running FLUX.1 [schnell] (Apache-2.0 model, commercial use allowed). The free
    tier gives a daily allocation of 'neurons' — enough for dozens of images a day at no cost."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"

    def _do():
        r = http().post(url, json={"prompt": prompt[:2048], "steps": max(1, min(8, steps))}, timeout=120,
                        headers={"Authorization": f"Bearer {token}"})
        raise_for_status(r, "Cloudflare Workers AI")
        b64 = ((r.json().get("result") or {}).get("image")) or ""
        if not b64:
            raise PermanentError(f"Cloudflare Workers AI returned no image: {str(r.text)[:160]}")
        dest.write_bytes(base64.b64decode(b64))
        return dest

    return with_retries(_do, attempts=2, label="Cloudflare image")


# Period discipline for every historical AI illustration. Positive detail only — FLUX draws whatever is named.
# The clothing and architecture come from the place's own article (place_look), never a generic default.
_HISTORICAL_PERIOD = "period-accurate historical scene, lit by oil lamps and daylight"

# Scenes that must NOT be AI-generated: a deity, god or religious figure. We illustrate these with a real photo
# of the temple/place instead and let the narration tell the myth as "legend says".
_DEITY_RE = re.compile(
    r"(?<![a-z])(deity|deities|god|gods|goddess(?:es)?|divine|lord\s+\w+|idol|avatar|incarnation|"
    r"shiva|vishnu|brahma|ganesha|ganesh|parvati|durga|kali|lakshmi|hanuman|krishna|rama|lingam|atmalinga|"
    r"buddha|bodhisattva|avalokiteshvara|guanyin|allah|prophet|jesus|christ|saint|angel|demon|rakshasa|"
    r"ravana|asura|naga)(?![a-z])", re.I)

# How a past scene of a place should LOOK — its architecture and its people's clothing — keyed by architectural
# and cultural terms in the place's own Wikipedia text (never by country: "India" alone says nothing about a
# Himalayan monastery vs a Hoysala temple). First match wins, so the more specific styles come first. Positive
# descriptions only: FLUX draws what is named, so the named roof/material is what keeps the wrong one out.
_LOOK_BY_KEYWORD: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"tibetan|gompa|gonpa|lamasery|vajrayana|nyingma|kagyu|gelug|sakya|drukpa|\bbon\b|dolpo|"
                r"ladakh|zanskar|upper mustang|spiti", re.I),
     "Tibetan Buddhist monastery architecture: whitewashed stone and mud-brick walls with dark red upper bands, "
     "flat roofs edged with prayer flags, small trapezoid windows framed in black, on a bare high-altitude slope",
     "monks in maroon robes with saffron sashes; villagers in thick wool chuba robes and felt boots"),
    (re.compile(r"hoysala|belur|halebid|halebeedu|somanathapura", re.I),
     "Hoysala temple architecture: star-shaped platforms and walls of grey-green soapstone, dense carved friezes "
     "of elephants, dancers and scrollwork, lathe-turned pillars",
     "people in draped cotton dhotis and saris with gold jewellery, 12th-century South Indian dress"),
    (re.compile(r"\bdzong\b", re.I),
     "Bhutanese dzong architecture: fortress-monasteries with sloping whitewashed walls, a red band under the "
     "roof, carved timber galleries and tiered shingle roofs",
     "men in knee-length gho robes and women in ankle-length kira, monks in maroon robes"),
    (re.compile(r"kath[\s-]?kuni|kinnaur|\bkullu\b|himachal|parvati valley", re.I),
     "Himalayan kath-kuni architecture: stacked timber-and-stone houses with carved wooden balconies and pitched "
     "slate roofs, pine forests and snow peaks behind",
     "villagers in hand-woven wool pattu shawls and round Himachali caps"),
    (re.compile(r"\bnewar|malla dynasty|kathmandu valley|durbar square|pagoda-style", re.I),
     "Newar architecture: brick temples with tiered sloping timber roofs on carved wooden struts, latticed windows",
     "men in daura-suruwal tunics and topi caps, women in black haku patasi saris with red borders"),
    (re.compile(r"gassho|minka|thatched farmhouse", re.I),
     "Japanese gassho-zukuri farmhouses: steep thatched roofs like praying hands over timber frames, rice paddies",
     "farmers in indigo cotton work kimono and straw hats"),
    (re.compile(r"\bshinto|\bshrine\b.*\bjapan|\btorii|\bedo period|\bsamurai|\bjapanese temple", re.I),
     "Japanese architecture: wooden temples and pagodas under sweeping curved tiled roofs",
     "people in kimono and wooden geta sandals"),
    (re.compile(r"\bhanok\b|joseon", re.I),
     "Korean hanok architecture: timber houses with gently curved tiled roofs and paper-screen doors",
     "people in white hanbok robes, men in black gat hats"),
    (re.compile(r"huizhou|hui-style|\bhui\b architecture|horse-head wall", re.I),
     "Huizhou architecture: whitewashed walls with stepped horse-head gables and black-tiled roofs by a stream",
     "villagers in plain cotton hanfu jackets and conical bamboo hats"),
    (re.compile(r"ming dynasty|qing dynasty|song dynasty|tang dynasty|siheyuan|courtyard house", re.I),
     "Chinese architecture: timber halls under upturned glazed-tile eaves around stone courtyards",
     "people in layered hanfu robes of cotton and silk"),
    (re.compile(r"\bkhmer\b|angkor|\bprasat\b|\bprang\b", re.I),
     "Khmer temple architecture: corn-cob sandstone prang towers, carved lintels and laterite galleries",
     "people in sampot wraps of cotton and silk, palm-leaf hats"),
    (re.compile(r"gopuram|gopura|dravidian|vijayanagara|chola|pallava|pandya|\bmandapa", re.I),
     "South Indian stone temple architecture: tall tiered gopuram gateway towers and pillared mandapa halls of "
     "carved granite",
     "people in draped cotton dhotis and saris with flowers in the hair"),
    (re.compile(r"\bdagoba|anuradhapura|polonnaruwa|sinhal", re.I),
     "Sinhalese architecture: white hemispherical dagoba stupas and carved stone pillars and moonstones",
     "people in white cotton sarongs and wrapped cloths"),
    (re.compile(r"\bzedi\b|\bpaya\b|pagan kingdom|\bbagan\b|mrauk|arakan", re.I),
     "Burmese temple architecture: brick and stone temples with bell-shaped stupas and tapering spires",
     "people in longyi wraps and gaungbaung headcloths"),
    (re.compile(r"\bstepwell|step well|\bbaori\b|\bbaoli\b|\bvav\b", re.I),
     "Indian stepwell architecture: symmetrical carved stone steps descending into an underground water tank",
     "women in bright cotton ghagra skirts and odhni veils carrying brass water pots"),
    (re.compile(r"\bhaveli", re.I),
     "Rajasthani haveli architecture: carved sandstone townhouses with projecting jharokha balcony windows",
     "people in bright turbans and ghagra-choli"),
    (re.compile(r"\bminaret|\bmosque|\bmadrasa|\bcaravanserai|silk road|timurid", re.I),
     "Central Asian Islamic architecture: tiled pointed iwans, slender minarets and glazed blue tilework",
     "merchants in long striped chapan robes and embroidered skullcaps"),
    (re.compile(r"stilt house|longhouse|bamboo house|thatched", re.I),
     "timber-and-bamboo stilt houses raised on posts with pitched thatched roofs",
     "villagers in hand-woven cotton wraps"),
    (re.compile(r"rock[\s-]?cut|cave temple", re.I),
     "rock-cut cave architecture: pillared halls and shrines carved into living stone cliffs",
     "monks and pilgrims in simple draped cotton robes"),
]
_LOOK_LEAD_CHARS = 1500          # rules read the article's opening: what the place IS, not a passing mention
LOOK_SCHEMA = {"type": "object", "additionalProperties": False,
               "properties": {"architecture": {"type": "string"}, "clothing": {"type": "string"}},
               "required": ["architecture", "clothing"]}


def look_from_text(place: str, article: str) -> dict:
    """{architecture, clothing} from the curated rules, matched against the place's name and the opening of
    its own article — or {} when nothing in the text identifies its style (there is NO country default)."""
    blob = f"{place}\n{(article or '')[:_LOOK_LEAD_CHARS]}"
    for rx, arch, clothing in _LOOK_BY_KEYWORD:
        if rx.search(blob):
            return {"architecture": arch, "clothing": clothing}
    return {}


def look_from_llm(llm, place: str, country: str, article: str) -> dict:
    """{architecture, clothing} an LLM reads out of the article itself, or {} (no basis in the text, or no
    LLM). Positive concrete details only, as FLUX draws whatever a prompt names."""
    if llm is None or not (article or "").strip():
        return {}
    system = "You turn a Wikipedia article into precise visual notes for a historical illustrator. Strict JSON."
    user = (f"Place: {place}{', ' + country if country else ''}\n"
            "From the article below ONLY, describe for an illustrator (1) `architecture`: this place's buildings "
            "in the period its history covers - materials, walls, roofs, colours, distinctive forms; (2) "
            "`clothing`: what its people wore then. Concrete positive details only (say what IS there, never "
            "'no ...'). Use what the article states or directly implies (e.g. 'a Tibetan Buddhist monastery' "
            "implies whitewashed walls with red bands, flat roofs, monks in maroon robes). Never use a country's "
            "typical style unless the article itself points to it. If the article gives no basis, return empty "
            "strings. At most 40 words each.\n\nArticle:\n" + article[:3500])
    try:
        d = llm.complete_json(system, user, schema=LOOK_SCHEMA, max_tokens=600, temperature=0.0)
    except Exception as e:
        log.warning("Reading the place's look from its article failed (%s)", redact(e))
        return {}
    arch = str((d or {}).get("architecture", "")).strip() if isinstance(d, dict) else ""
    clothing = str((d or {}).get("clothing", "")).strip() if isinstance(d, dict) else ""
    return {"architecture": arch, "clothing": clothing} if arch else {}


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


# Nature/filler words that, on their own, mean a weak "empty field / plain wall / lone tree" shot.
_GENERIC_WORDS = {"field", "fields", "grass", "tree", "trees", "wall", "sky", "road", "roads", "meadow",
                  "lawn", "bush", "bushes", "fence", "cloud", "clouds", "street", "path", "forest", "garden"}

# Generic words dropped when deciding whether an image's title actually NAMES the place.
_PLACE_STOP = {"the", "of", "and", "town", "city", "village", "temple", "temples", "fort", "ruins", "ruin",
               "valley", "river", "lake", "mount", "hill", "national", "park", "old", "ancient", "india",
               "view", "near", "road", "street"}


def _place_name_tokens(place: str, country: str) -> set[str]:
    """Distinctive (non-generic) tokens of a place name, used to accept only Commons images whose own title
    names the place — so a keyword search can't slip in a photo of somewhere else."""
    return {_stem(w) for w in re.findall(r"[a-z]+", f"{place} {country}".lower())
            if len(w) > 2 and w not in _PLACE_STOP}


def _names_place(item: MediaItem, place_tokens: set[str]) -> bool:
    if not place_tokens:
        return False
    have = {_stem(w) for w in re.findall(r"[a-z]+", (item.text or "").lower())}
    return bool(place_tokens & have)


def _wiki_rank(item: MediaItem, query: str, script_terms: set[str], avoid: set[str] | None = None) -> float:
    """Rank a Wikimedia candidate for a scene: relevance to the narration line first, then whether its
    title names a landmark the script talks about, its resolution, and a portrait/square shape (so a 9:16
    crop keeps more). A generic shot that names nothing from the script is pushed down, an extreme panorama
    is avoided (it crops badly), and a subject already used in this video is preferred against."""
    text_terms = _terms(item.text)
    s = 4.0 * relevance(item, query)
    s += min(len(text_terms & script_terms), 3) * 1.2            # title names script landmarks/place
    res = min(item.width, item.height)
    s += 1.2 if res >= 1600 else (0.8 if res >= 1000 else (0.3 if res >= 800 else 0.0))
    if item.height >= item.width:                                # portrait or square -> less cropping
        s += 1.5
    lo, hi = min(item.width, item.height), max(item.width, item.height)
    if lo and hi / lo > 2.0:                                     # panorama wider/taller than 2:1 -> crops badly
        s -= 3.0
    if (text_terms & _GENERIC_WORDS) and not (text_terms & script_terms):
        s -= 2.5                                                 # an empty field / plain wall of nowhere named
    if _PEOPLE_RE.search(item.text):
        s -= 2.0                                                 # prefer against a close-up identifiable person
    if avoid:                                                    # prefer a different subject from earlier scenes
        s -= 1.5 * min(len((text_terms - script_terms) & avoid), 3)
    return s + 0.6 / (1.0 + 0.25 * item.rank)                    # small tiebreak by the API's own order


def _low_quality_image(path: Path, min_std: float = 16.0) -> bool:
    """True for a near-blank / low-contrast image (a blank wall, flat sky, fog): the standard deviation of
    its luminance is tiny. Needs the downloaded file; returns False if it can't be read (don't over-skip)."""
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as im:
            g = im.convert("L")
            g.thumbnail((96, 96))
            return ImageStat.Stat(g).stddev[0] < min_std
    except Exception:
        return False


class Visuals:
    def __init__(self, settings: Settings, used_keys: set[str] | None = None, llm=None, budget=None):
        self.s = settings
        self.llm = llm                                # vision model for the photo-content check (may be None)
        self.budget = budget                          # Cloudflare neuron budget for AI image gen (may be None)
        self._vision_cache: dict[tuple[str, str], bool | None] = {}
        self._vision_unavailable = False              # set when the check is configured but no model answered
        self._vision_answered = 0                     # images a backend gave a yes/no for
        self._vision_streak = 0                       # consecutive images no backend answered for
        self._look_cache: dict = {}                   # (place, country, article) -> {architecture, clothing}
        self.used = set(used_keys or ())
        self.lock = threading.Lock()
        self._search_cache: dict[tuple[str, str], list[MediaItem]] = {}
        self.sources = [x for x in settings.visual_sources if x]
        if settings.offline:
            self.sources = ["local"]
        self.credits: set[str] = set()
        self.image_credits: list[dict] = []          # per Wikimedia image: title, author, license, urls
        self.ai_used = False                          # True once a scene used an AI illustration (disclosure)
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

    def place_look(self, place: str, country: str, article: str) -> dict:
        """How this place's past should look ({architecture, clothing}), from its own Wikipedia article: the
        curated style rules first, then an LLM reading the article; {} if the text gives no basis. Cached."""
        key = (place.lower(), country.lower(), hash(article or ""))
        if key not in self._look_cache:
            look = look_from_text(place, article) or look_from_llm(self.llm, place, country, article)
            if look:
                log.info("Look from the article: %s | %s", look["architecture"][:80], look["clothing"][:60])
            self._look_cache[key] = look
        return self._look_cache[key]

    def verify_subject(self, item: MediaItem, subject: str):
        """Ask the vision model whether the downloaded image actually shows ``subject`` (Commons titles lie).
        Cached per (image, subject). Returns True (shows it), False (doesn't - or the image itself can't be
        read by a vision model: skip it), NO_ANSWER (a check was expected but no backend answered for this
        image: don't use it unverified, try another), or None when no check is expected (disabled, offline,
        or no image). Only an explicit "yes" is a pass. One image failing is not an outage: vision counts as
        UNAVAILABLE (and the video is skipped, fail closed) only when no backend has answered for the first
        VISION_GIVE_UP_FRESH images, or for VISION_GIVE_UP_STREAK images in a row."""
        if not subject or not getattr(self.s, "vision_check", True) or self.llm is None or not item.path:
            return None                                       # check not expected -> accept via ranking
        ck = (item.key, subject.lower())
        if ck in self._vision_cache:
            cached = self._vision_cache[ck]
            return NO_ANSWER if cached is None else cached
        try:
            verdict = self.llm.image_shows(item.path, subject)
        except ImageUnusable as e:                            # THIS image can't be checked: skip it, not an outage
            log.info("Vision: skipping %s - %s", item.extra.get("wiki_title") or item.key, redact(e))
            self._vision_cache[ck] = False
            return False
        except Exception as e:
            log.debug("vision check errored (%s)", redact(e))
            verdict = None
        if verdict is True or verdict is False:               # only an explicit yes/no counts as an answer
            self._vision_cache[ck] = verdict
            self._vision_answered += 1
            self._vision_streak = 0
            if verdict is False:
                log.info("Vision: %s does not show %r — rejected", item.extra.get("wiki_title") or item.key,
                         subject)
            return verdict
        # None, empty, or any non-boolean answer: no backend answered for this image.
        self._vision_cache[ck] = None
        self._vision_streak += 1
        log.info("Vision: no backend answered for %s (%d in a row, %d answered so far)",
                 item.extra.get("wiki_title") or item.key, self._vision_streak, self._vision_answered)
        if (self._vision_answered == 0 and self._vision_streak >= VISION_GIVE_UP_FRESH)                 or self._vision_streak >= VISION_GIVE_UP_STREAK:
            self._vision_unavailable = True
        return NO_ANSWER

    def _claim_wiki(self, cands: list[MediaItem], query: str, script_terms: set[str],
                    avoid: set[str] | None = None) -> MediaItem | None:
        with self.lock:
            fresh, seen = [], set()
            for c in cands:
                if c.key in self.used or c.key in seen:
                    continue
                seen.add(c.key)
                fresh.append(c)
            if not fresh:
                return None
            best = max(fresh, key=lambda c: _wiki_rank(c, query, script_terms, avoid))
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

    def _good_wiki(self, cands: list[MediaItem], query: str, script_terms: set[str], dest_dir: Path,
                   idx: int, tries: int = 4, avoid: set[str] | None = None, subject: str = "",
                   landmark: bool = False) -> MediaItem | None:
        """Claim the best fresh candidate and download it, skipping to the next if it is near-blank, or if a
        vision model says it doesn't actually show ``subject``. When vision can't run, a landmark scene also
        refuses a photo whose title's main subject is an animal or a person. Returns a fetched item or None."""
        for _ in range(tries):
            item = self._claim_wiki(cands, query, script_terms, avoid)
            if item is None:
                return None
            try:
                self._fetch(item, dest_dir, idx)
            except Exception as e:
                log.warning("Wikimedia download failed (%s)", redact(e))
                continue
            if _low_quality_image(item.path):
                log.info("Scene %d: skipping near-blank/low-contrast image %s", idx + 1, item.key)
                continue
            verdict = self.verify_subject(item, subject)
            if verdict is False or verdict == NO_ANSWER:
                if self._vision_unavailable:
                    return None                                  # vision is down: the scene check raises SkipTopic
                continue                                         # not the subject, or unverified -> next candidate
            if verdict is None and landmark and _ANIMAL_PEOPLE_RE.search(item.text):
                continue                                         # no vision: don't risk an animal/person on a landmark
            return item
        return None

    def _historical_image(self, prompt: str, dest_dir: Path, idx: int, arch: str = "",
                          clothing: str = "") -> MediaItem | None:
        """A past scene the camera can't reach: generate one AI illustration in the configured consistent
        cinematic style, period-accurate, in the architecture ``arch`` and clothing ``clothing`` read from the
        place's own article. Without an architecture there is nothing place-true to draw, so None (a real
        photo instead) - never a generic or country-default look. Free providers only by default
        (pollinations/cloudflare = FLUX, Apache-2.0); paid providers (openai/gemini) must be chosen explicitly.
        Returns None to fall back to a real photo."""
        s = self.s
        source = (getattr(s, "ai_scene_source", "") or "").strip().lower()
        if source in ("", "none", "off") or not (arch or "").strip():
            return None
        # Never generate images of a deity or religious figure: fall back to a real photo of the temple/place
        # and let the narration tell the myth as "legend says".
        if _DEITY_RE.search(prompt or ""):
            log.info("Scene %d: a deity/religious figure — using a real photo, not an AI image", idx + 1)
            return None
        style = s.ai_scene_style or s.visual_style
        # Architecture leads (it anchors the look), then the people's clothing. Positive description only — FLUX
        # draws whatever is named, so we ask for a full-bleed vertical frame rather than listing negatives.
        people = f"{clothing.strip().rstrip('.')}. " if (clothing or "").strip() else ""
        full = (f"{arch.strip().rstrip('.')}. {people}{prompt}. {style}. {_HISTORICAL_PERIOD}. "
                "A full-bleed vertical 9:16 cinematic composition filling the entire frame edge to edge.")
        dest = dest_dir / f"scene{idx:02d}_ai.jpg"
        try:
            if source == "pollinations":
                pollinations_image(full, dest, s.width, s.height, s.pollinations_api_key)
                src = "pollinations"
            elif source in ("cloudflare", "workers-ai", "flux") and s.cloudflare_account_id and s.cloudflare_api_token:
                if self.budget is not None and not self.budget.can_spend(CF_FLUX_NEURONS):
                    log.info("Scene %d: Cloudflare neuron reserve reached — using a real photo, not an AI image",
                             idx + 1)
                    return None
                cloudflare_image(full, dest, s.cloudflare_account_id, s.cloudflare_api_token)
                if self.budget is not None:
                    self.budget.record(CF_FLUX_NEURONS)
                src = "cloudflare"
            elif source in ("openai", "openai-images", "gpt-image", "dalle") and s.openai_api_key:
                dest = dest.with_suffix(".png")
                openai_image(full, dest, s.openai_api_key, s.image_model, s.orientation)
                src = "openai-images"
            elif source in ("gemini", "nano-banana", "imagen"):
                log.warning("AI_SCENE_SOURCE=%s is a paid, billing-enabled service and is not enabled; "
                            "using a real photo for scene %d.", source, idx + 1)
                return None
            else:
                log.warning("AI_SCENE_SOURCE=%r is not usable (missing key?); using a real photo for scene %d.",
                            source, idx + 1)
                return None
        except Exception as e:
            log.warning("Historical AI image (%s) for scene %d failed (%s); using a real photo instead.",
                        source, idx + 1, redact(e))
            return None
        if not dest.exists() or dest.stat().st_size < 1024:
            return None
        # Use the image's REAL dimensions (providers may return a square): the render cover-crops by the true
        # aspect, so the frame is filled instead of letterboxed.
        w, h = s.width, s.height
        try:
            from PIL import Image
            with Image.open(dest) as im:
                w, h = im.size
        except Exception:
            pass
        self.ai_used = True
        log.info("Scene %d: %s AI illustration (historical)", idx + 1, src)
        return MediaItem(src, dest.name, "image", width=w, height=h, path=dest, query=prompt)

    def _place_pool(self, place: str, country: str, lat: float | None, lon: float | None,
                    min_images: int) -> list[MediaItem]:
        """Freely-licensed, PLACE-ACCURATE Commons photos: everything geotagged within ~5 km of the place,
        plus name-search hits whose own title names the place. A keyword search alone drifts to look-alike
        places, so both gates are place-specific. Raises SkipTopic below ``min_images``."""
        pc = ", ".join(x for x in (place, country) if x)
        place_tokens = _place_name_tokens(place, country)
        pool, seen = [], set()
        near = 0
        if lat is not None and lon is not None:                # within ~5 km of the place's coordinates
            try:
                for it in geosearch_images(lat, lon, 5000):
                    if it.key not in seen:
                        seen.add(it.key)
                        pool.append(it)
                        near += 1
            except Exception as e:
                log.warning("Wikimedia geosearch failed: %s", redact(e))
        for q in dict.fromkeys([x for x in (pc, place, f"{place} {country}") if x]):   # name search, title must match
            for it in self._wiki_search(q):
                if it.key not in seen and _names_place(it, place_tokens):
                    seen.add(it.key)
                    why = _foreign_photo(it, lat, lon, country)
                    if why:                                    # same name, another place (Shey Palace != Shey Gompa)
                        log.info("Wikimedia: skipped %r (%s)", it.extra.get("wiki_title") or it.id, why)
                        continue
                    pool.append(it)
        log.info("Wikimedia: %d place-accurate image(s) for %s (%d within 5 km%s)", len(pool), pc or place,
                 near, "" if lat is not None else "; no coordinates, titles only")
        if len(pool) < min_images:
            raise SkipTopic(f"only {len(pool)} place-accurate Wikimedia image(s) for '{pc or place}'; "
                            f"need {min_images}")
        return pool

    def _open_landmark_image(self, pool: list[MediaItem], landmark_terms: set[str], place: str, country: str,
                             dest_dir: Path, subject: str = "") -> MediaItem | None:
        """Scene 1: the strongest real photo of the main landmark named in the title — best title-match and
        resolution, never a generic ground/tree/sky shot, never an animal/person, and (when a vision model is
        available) one the pixels confirm shows the landmark. Returns None if the pool has nothing suitable."""
        place_terms = _terms(place) | _terms(country)

        def rank(c: MediaItem) -> float:
            tt = _terms(c.text)
            if (tt & _GENERIC_WORDS) and not (tt & landmark_terms):
                return -1.0                                    # a 'field / trees / sky' shot is never the opener
            if _ANIMAL_PEOPLE_RE.search(c.text):
                return -1.0                                    # a monkey / tourist is never the landmark opener
            named = len(tt & landmark_terms)
            if not named:
                return 0.0                                     # doesn't name the landmark -> not a valid opener
            res = min(c.width, c.height)
            s = 3.0 * named + 2.0 * relevance(c, " ".join(landmark_terms | place_terms))
            s += 1.5 if res >= 1600 else (1.0 if res >= 1000 else 0.3)
            if c.height >= c.width:                            # portrait/square crops to 9:16 with less loss
                s += 0.8
            return s

        for _ in range(6):                                     # try the best few, skipping near-blank/mismatched
            with self.lock:
                cands = [c for c in pool if c.key not in self.used and rank(c) > 0]
                if not cands:
                    return None
                best = max(cands, key=rank)
                self.used.add(best.key)
            item = replace(best)
            try:
                self._fetch(item, dest_dir, 0)
            except Exception as e:
                log.warning("Opening landmark download failed (%s)", redact(e))
                continue
            if _low_quality_image(item.path):
                continue
            verdict = self.verify_subject(item, subject)
            if verdict is False or verdict == NO_ANSWER:       # not the landmark, or unverified -> next
                if self._vision_unavailable:
                    return None
                continue
            return item
        return None

    def for_place_scenes(self, jobs: list[tuple[str, str, bool]], place: str, country: str, dest_dir: Path,
                         progress=None, min_images: int = 5, lat: float | None = None, lon: float | None = None,
                         article: str = "", title: str = "") -> list[MediaItem]:
        """One image per scene of the named place: a freely-licensed Wikimedia photo for scenes about the place
        TODAY, and a consistent-style AI illustration for scenes marked historical (the past can't be
        photographed). Only place-accurate photos are used (geosearch within ~5 km, or a title that names the
        place); raises SkipTopic if there are fewer than ``min_images``. Scene 1 is the strongest real photo of
        the landmark named in ``title`` (never a generic ground/tree shot). No image is shown for more than two
        scenes in a row. Stock video is a place-named fallback."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        pool = self._place_pool(place, country, lat, lon, min_images)
        # Landmarks/place the script talks about: used to prefer images whose titles name them.
        script_terms = _terms(place) | _terms(country)
        for q, _p, _h in jobs:
            script_terms |= _terms(q)
        opener_q = jobs[0][0] if jobs else ""                  # the first scene's own query names the landmark too
        landmark_terms = ((_terms(title) | _terms(opener_q)) - _terms(place) - _terms(country)) or script_terms
        # The subject the opener and the last scene must actually show (checked in pixels, not the title).
        landmark_subject = " ".join(
            dict.fromkeys(list(landmark_terms)[:4] + [w for w in [place] if w])).strip() or place
        # Draw the past in THIS place's architecture and clothing, read from its own article (no country default;
        # with nothing to go on, its past scenes become real photos).
        look = self.place_look(place, country, article) if any(h for _q, _p, h in jobs) else {}
        arch, clothing = look.get("architecture", ""), look.get("clothing", "")
        if any(h for _q, _p, h in jobs) and not arch:
            log.info("No architecture described in the article for %s; past scenes use real photos", place)
        results: list[MediaItem] = []
        chosen: set[str] = set()                               # distinctive subjects already shown this video
        prev_key, consec = None, 0
        last = len(jobs) - 1
        for i, (q, prompt, historical) in enumerate(jobs):
            is_landmark = i == 0 or i == last                  # opener & closer must show the main landmark
            subject = landmark_subject if is_landmark else (q or place)
            item = (self._historical_image(prompt or q, dest_dir, i, arch=arch, clothing=clothing)
                    if historical and arch else None)
            if item is None and i == 0:                        # opener: the strongest photo of the title landmark
                item = self._open_landmark_image(pool, landmark_terms, place, country, dest_dir, landmark_subject)
            if item is None:                                   # place-today scene, or AI unavailable/failed
                item = self._good_wiki(pool, q, script_terms, dest_dir, i, avoid=chosen,
                                       subject=subject, landmark=is_landmark)
            if item is None:                                   # no fresh place image -> place-named stock video
                item = self._place_stock_video(q, place, country, dest_dir, i)
            if item is None:                                   # last resort: an unused place photo (never a repeat)
                item = self._reuse_place_image(pool, q, script_terms, chosen, dest_dir, i, exclude=None)
            # Never show the same image for more than two scenes in a row: swap in a different pool photo.
            if item is not None and item.key == prev_key and consec >= 2:
                alt = self._reuse_place_image(pool, q, script_terms, chosen, dest_dir, i, exclude=item.key)
                if alt is not None:
                    item = alt
            if self._vision_unavailable:
                # The vision check is configured but no model answered: fail closed rather than ship a photo we
                # could not confirm shows its subject. The launcher logs this as "SKIPPED vision unavailable".
                raise SkipTopic("vision unavailable — cannot verify a real photo shows its subject")
            if item is None:
                item = MediaItem("generated", f"gradient-{i}", "generated", query=q)
            consec = consec + 1 if item.key == prev_key else 1
            prev_key = item.key
            if item.source == "wikimedia":                     # remember its subject so later scenes differ
                chosen |= (_terms(item.text) - script_terms)
            self._record_credit(item)
            log.info("Scene %d: %s %s (%r)", i + 1, item.source, item.kind, q)
            results.append(item)
            if progress:
                progress((i + 1) / len(jobs))
        return results

    def _reuse_place_image(self, pool: list[MediaItem], query: str, script_terms: set[str], chosen: set[str],
                           dest_dir: Path, idx: int, exclude: str | None) -> MediaItem | None:
        """Pick a pool photo for this scene, preferring one not used yet and never ``exclude`` (the image shown
        on the two previous scenes). Downloads it; returns None only if the pool is empty of alternatives."""
        with self.lock:
            cands = [c for c in pool if c.key != exclude]
            unused = [c for c in cands if c.key not in self.used]
            pick_from = unused or cands
            if not pick_from:
                return None
            best = max(pick_from, key=lambda c: _wiki_rank(c, query, script_terms, chosen))
            self.used.add(best.key)
        item = replace(best)
        try:
            self._fetch(item, dest_dir, idx)
        except Exception as e:
            log.warning("Wikimedia reuse download failed (%s)", redact(e))
            return None
        return item

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
