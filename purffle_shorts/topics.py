"""Where video ideas come from. Mix sources with TOPIC_SOURCES, optionally weighted:
TOPIC_SOURCES=niche:4,trending:1,wikipedia:1

  niche      the AI picks a fresh, specific topic inside one of your NICHES (default)
  trending   today's Google Trends searches for TRENDS_GEO (tragedies and elections are skipped)
  wikipedia  "On this day" historical events
  reddit     r/todayilearned top posts of the day
  file       your own list: one topic per line in topics.txt (each used once)
  list       a ranked JSON list of places (TOPIC_LIST_FILE), e.g. written weekly by a scout tool; when
             set it is picked from first, and skipped when missing, stale or used up
  rss        the newest unused item from your RSS_FEEDS (blogs, news sites, podcasts)
  search     questions people type into YouTube's search box about one of your NICHES

Ideas saved by the planner (`plan` command or the Studio's Ideas page) are always used first.
A single video can also be made from a web page (--url) or a local text/markdown file (--from-file).
"""

from __future__ import annotations

import html
import json
import logging
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from .config import Settings
from .history import PLACE_REUSE_DAYS, History, place_recently_used
from .utils import http, raise_for_status

log = logging.getLogger("purffle")

SENSITIVE = re.compile(
    r"\b(dies|died|dead|death|killed|kills|murder\w*|shooting|shot|crash\w*|funeral|obituary|suicide|attack\w*|"
    r"wars?|bomb\w*|hostage|victims?|arrested|charged|rape|abuse|massacre|genocide|terror\w*|assassinat\w*|coup|"
    r"elect\w*|ballot|campaign|president|prime minister|parliament)\b", re.I)


@dataclass
class Topic:
    subject: str
    source: str
    key: str = ""
    context: str = ""
    style: str = ""
    idea_id: int | None = None
    keywords: list[str] = field(default_factory=list)  # real YouTube searches handed to the script writer
    source_url: str = ""     # the grounding article's URL (shown in the description, saved in metadata)
    source_text: str = ""    # the grounding article's plain text (facts must come from here, not be invented)


def article(query: str, lang: str = "en", max_chars: int = 5000) -> tuple[str, str]:
    """Plain text and URL of the best-matching Wikipedia article for ``query`` (free API, no key). Used to
    ground a story in real facts. Returns ("", "") on any failure, so grounding is best-effort."""
    try:
        r = http().get(f"https://{lang}.wikipedia.org/w/api.php", timeout=20, params={
            "action": "query", "format": "json", "redirects": 1,
            "generator": "search", "gsrsearch": query, "gsrlimit": 1, "gsrnamespace": 0,
            "prop": "extracts", "explaintext": 1, "exsectionformat": "plain"})
        raise_for_status(r, "Wikipedia article")
        pages = (r.json().get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), None) if isinstance(pages, dict) else (pages[0] if pages else None)
        if not page or not page.get("extract"):
            return "", ""
        title = page.get("title", "")
        text = re.sub(r"\n{3,}", "\n\n", str(page["extract"]).strip())[:max_chars]
        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return text, url
    except Exception as e:
        log.warning("Wikipedia grounding for %r failed: %s", query, e)
        return "", ""


def place_coordinates(query: str, lang: str = "en") -> tuple[float, float] | None:
    """(lat, lon) of the best-matching Wikipedia article for ``query`` (free API, no key), so Commons images
    can be found by geography rather than by name alone. Returns None on any failure."""
    try:
        r = http().get(f"https://{lang}.wikipedia.org/w/api.php", timeout=20, params={
            "action": "query", "format": "json", "redirects": 1,
            "generator": "search", "gsrsearch": query, "gsrlimit": 1, "gsrnamespace": 0,
            "prop": "coordinates", "colimit": 1, "coprimary": "all"})   # some articles tag theirs non-primary
        raise_for_status(r, "Wikipedia coordinates")
        pages = (r.json().get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), None) if isinstance(pages, dict) else (pages[0] if pages else None)
        co = ((page or {}).get("coordinates") or [{}])[0]
        if "lat" in co and "lon" in co:
            return float(co["lat"]), float(co["lon"])
    except Exception as e:
        log.warning("Wikipedia coordinates for %r failed: %s", query, e)
    return None


# This channel is hidden natural + historic places only. Skip modern attractions: zoos, theme parks, malls,
# resorts, hotels and the like — recognised by name OR by the article's own description.
_ATTRACTION_RE = re.compile(
    r"\b(zoo|safari park|wildlife park|theme park|amusement park|water ?park|aqua ?park|fun ?fair|"
    r"aquarium|oceanarium|marine park|bird park|dolphinarium|"
    r"shopping (?:mall|centre|center|complex)|outlet (?:mall|village)|retail park|"
    r"resort|hotel|casino|golf course|holiday park|caravan park|"
    r"stadium|convention (?:centre|center)|observation (?:deck|tower|wheel)|ferris wheel)\b", re.I)
_MALL_RE = re.compile(r"(?<![a-z])malls?(?![a-z])", re.I)
# "opened/founded/built in YYYY" — used to reject attractions whose only recorded date is after 1950.
_OPEN_YEAR_RE = re.compile(
    r"\b(?:open(?:ed|ing)?|inaugurat\w*|establish\w*|founded|built|constructed|completed|commissioned)\b"
    r"[^.]{0,40}?\b(1\d{3}|20\d{2})\b", re.I)


def modern_attraction_reason(subject: str, article: str = "") -> str:
    """Why ``subject`` is a modern attraction rather than a hidden natural/historic place — or "" if it is
    fine. Matches zoos / theme parks / malls / resorts / hotels (by name or by the article's description), and
    any place whose only recorded opening or founding year is after 1950."""
    blob = f"{subject}\n{article}"
    m = _ATTRACTION_RE.search(blob) or _MALL_RE.search(blob)
    if m:
        return f"looks like a {m.group(0).strip().lower()}"
    years = [int(y) for y in _OPEN_YEAR_RE.findall(article)]
    if years and min(years) > 1950:
        return f"opened {min(years)} (after 1950)"
    return ""


_OLD_CENTURY_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)[- ]century\b", re.I)
_BCE_RE = re.compile(r"\b(?:BCE?|B\.C\.?E?\.?)\b")
_OLD_ERA_RE = re.compile(
    r"\b(ancient|medieval|mediaeval|prehistoric|dynasty|dynasties|empire|kingdom|sultanate|caliphate|"
    r"antiquity|Iron Age|Bronze Age|Neolithic|Palaeolithic|Paleolithic|centuries[- ]old|millennia[- ]old)\b",
    re.I)
# A dated event in the past: "in 1565", "founded c. 850", "AD 1200" — a real year, so false positives like
# "1,200 metres" (comma) or a bare measurement don't count as history.
_OLD_DATE_RE = re.compile(r"\b(?:in|since|by|around|about|circa|c\.?|AD)\s+(\d{3,4})\b", re.I)


def has_real_history(article: str) -> bool:
    """True when the Wikipedia text records real events before ~1900 — a BCE date, an early century, a dated
    pre-1900 event, or an unambiguous historical era. This gates the AI-illustrated ``history`` format; places
    without such a record get the present-day ``place`` format (real photos only)."""
    if not article:
        return False
    if _BCE_RE.search(article) or _OLD_ERA_RE.search(article):
        return True
    if any(int(y) < 1900 for y in _OLD_DATE_RE.findall(article)):
        return True
    return any(int(c) <= 19 for c in _OLD_CENTURY_RE.findall(article))


def history_depth(article: str) -> int:
    """How much real pre-1900 history the text records: the number of DISTINCT markers — BCE dates, early
    centuries, dated pre-1900 events and historical eras. 0 means has_real_history() is False."""
    if not article:
        return 0
    marks = {m.group(0).lower().replace(".", "") for m in _BCE_RE.finditer(article)}
    marks |= {m.lower() for m in _OLD_ERA_RE.findall(article)}
    marks |= {f"y{y}" for y in _OLD_DATE_RE.findall(article) if int(y) < 1900}
    marks |= {f"c{c}" for c in _OLD_CENTURY_RE.findall(article) if int(c) <= 19}
    return len(marks)


def _weighted_sources(settings: Settings) -> list[tuple[str, float]]:
    out = []
    for item in settings.topic_sources or ["niche"]:
        name, _, w = item.partition(":")
        try:
            weight = float(w) if w else 1.0
        except ValueError:
            weight = 1.0
        out.append((name.strip().lower(), max(0.0, weight)))
    return out or [("niche", 1.0)]


def from_niche(settings: Settings) -> Topic:
    niche = random.choice(settings.niches or ["interesting facts"])
    return Topic(niche, "niche", key="")


def from_file(settings: Settings, history: History) -> Topic | None:
    p = Path(settings.topics_file)
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = f"file:{line}"
        if not history.topic_used(key):
            return Topic(line, "file", key=key)
    log.info("All topics in %s have been used", p)
    return None


def list_subject(place: str, region: str = "", country: str = "") -> str:
    """The topic subject for a listed place — "Place, Region, Country" — leaving out any part the name
    already contains (e.g. "Gokarna, Karnataka, India"; "Singapore Botanic Gardens, Singapore")."""
    parts = [place.strip()]
    for x in (region.strip(), country.strip()):
        if x and x.lower() not in " ".join(parts).lower():
            parts.append(x)
    return ", ".join(p for p in parts if p)


def from_list(settings: Settings, history: History) -> Topic | None:
    """The next place from an external ranked list (TOPIC_LIST_FILE, e.g. written weekly by a scout tool):
    JSON {"generated": "<ISO date>", "places": [{"place", "region", "country"}, ...]} in rank order.

    An entry is claimed when picked, so a place that later gets skipped doesn't block the list; places made
    within PLACE_REUSE_DAYS are passed over. Returns None — and the normal topic sources take over — when no
    list is configured, the file is missing or unreadable, older than TOPIC_LIST_MAX_AGE_DAYS, or used up."""
    if not settings.topic_list_file:
        return None
    p = Path(settings.topic_list_file)
    if not p.exists():
        log.info("Topic list %s not found; using the normal topic sources", p)
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(str(data.get("generated") or "").replace("Z", "+00:00"))
    except (OSError, ValueError, AttributeError) as e:
        log.warning("Topic list %s is unreadable (%s); using the normal topic sources", p, e)
        return None
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - generated).days
    if age > settings.topic_list_max_age_days:
        log.warning("Topic list %s is stale (%d days old, max %d); using the normal topic sources",
                    p, age, settings.topic_list_max_age_days)
        return None
    recent = history.recent_places(PLACE_REUSE_DAYS)
    stamp = generated.date().isoformat()
    for e in data.get("places") or []:
        if not isinstance(e, dict):
            continue
        place, region, country = (str(e.get(k) or "").strip() for k in ("place", "region", "country"))
        if not place:
            continue
        key = f"list:{stamp}:{place}|{country}".lower()
        if history.topic_used(key) or place_recently_used(place, country, recent):
            continue
        history.mark_topic(key)    # claimed now: if this video is skipped, the next run takes the next place
        subject = list_subject(place, region, country)
        log.info("Topic list: #%s %s", e.get("rank", "?"), subject)
        return Topic(subject, "list", key=key)
    log.info("Every place in topic list %s is used; using the normal topic sources", p)
    return None


def from_trends(settings: Settings, history: History) -> Topic | None:
    r = http().get("https://trends.google.com/trending/rss", params={"geo": settings.trends_geo}, timeout=20)
    raise_for_status(r, "Google Trends RSS")
    root = ET.fromstring(r.content)
    ns = {"ht": "https://trends.google.com/trending/rss"}
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        news = [((n.findtext("ht:news_item_title", namespaces=ns) or "") + " — " +
                 (n.findtext("ht:news_item_snippet", namespaces=ns) or "")).strip(" —")
                for n in item.findall("ht:news_item", ns)]
        blob = " ".join([title, *news])
        key = f"trend:{title}"
        if not title or history.topic_used(key) or SENSITIVE.search(blob):
            continue
        return Topic(title, "trending", key=key, context="\n".join(news[:3]))
    return None


def from_search(settings: Settings, history: History) -> Topic | None:
    from . import seo
    niches = list(settings.niches or ["interesting facts"])
    random.shuffle(niches)
    for niche in niches[:3]:
        qs = seo.question_topics(niche, lambda p: history.topic_used(f"search:{p}"), settings.language,
                                 settings.trends_geo)
        if qs:
            listed = "\n".join(f"- {q}" for q in qs)
            return Topic(niche, "search", key=f"search:{qs[0]}", keywords=qs,
                         context=f"Questions people really type into YouTube about {niche}:\n{listed}\n"
                                 "Answer the ONE that best fits the niche and makes the best Short (skip any that "
                                 "is about something else, such as a game or a show that shares a word). Put "
                                 "its exact words in the title.")
    return None


def from_wikipedia(settings: Settings, history: History) -> Topic | None:
    today = datetime.now()
    lang = settings.language.split("-")[0]
    url = f"https://{lang}.wikipedia.org/api/rest_v1/feed/onthisday/events/{today:%m}/{today:%d}"
    r = http().get(url, timeout=20)
    raise_for_status(r, "Wikipedia On This Day")
    events = r.json().get("events", [])
    random.shuffle(events)
    for ev in events:
        text, year = ev.get("text", ""), ev.get("year")
        key = f"wiki:{year}:{text[:80]}"
        if not text or history.topic_used(key) or SENSITIVE.search(text):
            continue
        pages = ev.get("pages") or []
        extract = next((p.get("extract", "") for p in pages if p.get("extract")), "")
        return Topic(f"On this day in {year}: {text}", "wikipedia", key=key, context=extract[:1500])
    return None


def from_reddit(settings: Settings, history: History) -> Topic | None:
    # The JSON API now rejects anonymous clients; the public Atom feed still works.
    r = http().get("https://www.reddit.com/r/todayilearned/top/.rss", params={"t": "day"}, timeout=20)
    raise_for_status(r, "Reddit TIL")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in ET.fromstring(r.content).findall("a:entry", ns):
        raw = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        title = re.sub(r"^TIL\s*(that\s*)?", "", raw, flags=re.I).strip(" :-")
        key = f"reddit:{entry.findtext('a:id', default=raw, namespaces=ns)}"
        if not title or history.topic_used(key) or SENSITIVE.search(title):
            continue
        return Topic(title, "reddit", key=key, context=raw)
    return None


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def from_rss(settings: Settings, history: History) -> Topic | None:
    feeds = list(settings.rss_feeds)
    random.shuffle(feeds)
    for url in feeds:
        try:
            r = http().get(url, timeout=20)
            raise_for_status(r, f"RSS {url[:60]}")
            root = ET.fromstring(r.content)
        except Exception as e:
            log.warning("RSS feed %s failed: %s", url, e)
            continue
        atom = "{http://www.w3.org/2005/Atom}"
        entries = list(root.iter("item")) or list(root.iter(f"{atom}entry"))
        for e in entries:
            title = _strip_html(e.findtext("title") or e.findtext(f"{atom}title") or "")
            summary = _strip_html(e.findtext("description") or e.findtext(f"{atom}summary")
                                  or e.findtext(f"{atom}content") or "")
            link = (e.findtext("link") or "").strip()
            if not link:
                el = e.find(f"{atom}link")
                link = el.get("href", "") if el is not None else ""
            key = f"rss:{link or title}"
            if not title or history.topic_used(key) or SENSITIVE.search(f"{title} {summary}"):
                continue
            return Topic(title, "rss", key=key, context=summary[:2500])
    return None


class _TextExtractor(HTMLParser):
    """Readable text from an article page: headings, paragraphs and list items, minus navigation."""

    SKIP = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "button"}
    KEEP = {"p", "h1", "h2", "h3", "li", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.meta_title = ""
        self._skip = 0
        self._keep = 0
        self._in_title = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.KEEP:
            self._keep += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = dict(attrs)
            if a.get("property") == "og:title" and a.get("content"):
                self.meta_title = a["content"]

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.KEEP and self._keep:
            self._keep -= 1
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            self._buf = []
            if len(text) > 30 or tag.startswith("h"):
                self.parts.append(text)
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._keep and not self._skip:
            self._buf.append(data)


def extract_article(page: str) -> tuple[str, str]:
    """(title, text) from an HTML page."""
    p = _TextExtractor()
    p.feed(page)
    title = (p.meta_title or p.title).strip()
    title = re.split(r"\s+[|–—-]\s+", title)[0].strip() if title else ""
    return title, "\n".join(dict.fromkeys(t for t in p.parts if t))


_REDDIT_POST = re.compile(r"^https?://(?:www\.|old\.|new\.)?reddit\.com/r/\w+/comments/\w+", re.I)


def from_reddit_post(url: str) -> Topic:
    """A Reddit post as the source. Reddit refuses anonymous JSON and renders pages with JavaScript, but every
    post has an Atom feed whose first entry is the post itself. The topic asks for the Reddit story format."""
    m = _REDDIT_POST.match(url)
    if not m:
        raise ValueError(f"not a Reddit post link: {url}")
    r = http().get(m.group(0) + "/.rss", timeout=30)
    if r.status_code == 429:
        raise RuntimeError("Reddit is rate-limiting anonymous requests right now. Try again in a few minutes, "
                           "or save the post's text to a file and use --from-file.")
    raise_for_status(r, "fetch the Reddit post")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(r.content).find("a:entry", ns)
    if entry is None:
        raise ValueError(f"{url}: Reddit returned no post")
    title = _strip_html(entry.findtext("a:title", default="", namespaces=ns))
    body = _strip_html(entry.findtext("a:content", default="", namespaces=ns))
    body = re.sub(r"\s*submitted by\s+/u/\S+.*$", "", body, flags=re.I | re.S).strip()
    if not title:
        raise ValueError(f"{url}: couldn't read that Reddit post")
    return Topic(title, "url", key=f"url:{m.group(0)}", context=f"{title}\n\n{body}"[:6000], style="reddit")


def from_url(url: str) -> Topic:
    """Turn a web page (article, blog post, docs page, Wikipedia, a Reddit post) into a topic with the page
    as context."""
    if not re.match(r"^https?://", url):
        raise ValueError(f"not a web address: {url}")
    if _REDDIT_POST.match(url):
        return from_reddit_post(url)
    r = http().get(url, timeout=30, headers={"Accept": "text/html,application/xhtml+xml"})
    raise_for_status(r, f"fetch {url[:60]}")
    title, text = extract_article(r.text)
    if len(text) < 200:
        raise ValueError(f"{url}: couldn't find readable article text on that page")
    return Topic(title or url, "url", key=f"url:{url}", context=text[:6000])


def from_document(path: str | Path) -> Topic:
    """A local .txt / .md file (notes, a blog draft, a transcript) as the source of a video."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() in (".html", ".htm"):
        title, text = extract_article(text)
    else:
        first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
        title = first[:90]
    if len(text.strip()) < 40:
        raise ValueError(f"{p}: not enough text to make a video from")
    return Topic(title or p.stem, "document", key=f"doc:{p.name}", context=text[:6000])


def pick_topic(settings: Settings, history: History, explicit: str | None = None,
               source: str | None = None) -> Topic:
    if explicit:
        return Topic(explicit, "manual")
    if settings.offline:
        return Topic("octopus superpowers", "manual")
    if source in (None, "queue"):
        idea = history.claim_idea()
        if idea:
            angle = (idea.get("notes") or "").strip()
            subject = f'{idea["subject"]} (angle: {angle})' if angle else idea["subject"]
            return Topic(subject, "idea", key=f'idea:{idea["id"]}', style=idea.get("style") or "",
                         idea_id=idea["id"])
        if source == "queue":
            log.info("The idea queue is empty; using a niche topic")
            return from_niche(settings)
    if source is None:
        try:
            listed = from_list(settings, history)   # a ranked topic list, when configured, comes next
        except Exception as e:
            log.warning("Topic list failed (%s); using the normal topic sources", e)
            listed = None
        if listed:
            return listed
    choices = [(source, 1.0)] if source else _weighted_sources(settings)
    names, weights = zip(*choices)
    name = random.choices(names, weights=weights, k=1)[0] if sum(weights) > 0 else "niche"
    fetch = {"file": from_file, "list": from_list, "trending": from_trends, "wikipedia": from_wikipedia, "reddit": from_reddit,
             "rss": from_rss, "search": from_search}
    if name in fetch:
        try:
            topic = fetch[name](settings, history)
            if topic:
                return topic
            log.info("Topic source '%s' had nothing new; falling back to a niche topic", name)
        except Exception as e:
            log.warning("Topic source '%s' failed (%s); falling back to a niche topic", name, e)
    elif name != "niche":
        log.warning("Unknown topic source '%s'; using niche", name)
    return from_niche(settings)
