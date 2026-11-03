"""Travel-safety guardrail.

Before a script is written, skip any destination under an active "Reconsider Travel"
(Level 3) or "Do Not Travel" (Level 4) advisory, read from the official US Department of
State travel-advisories RSS feed. The country is identified from the topic by the LLM.

Fail-CLOSED: once a topic is identified as a specific destination, it is skipped unless the
advisory can be read and is Level 1-2. A feed outage or a country missing from the feed
therefore skips that destination (the autopilot picks another) rather than risking an unsafe
one. Topics that are not about a specific country are never blocked.
"""

from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime

from .utils import http, raise_for_status, redact

log = logging.getLogger("purffle")

FEED_URL = "https://travel.state.gov/_res/rss/TAsTWs.xml"
BLOCK_LEVEL = 3  # Level 3 (Reconsider Travel) and Level 4 (Do Not Travel) are skipped
_ITEM_RE = re.compile(r"<title>([^<]+?)\s*-\s*Level\s*(\d)\s*:", re.I)
_FEED_TTL = 6 * 3600

# The State Department lists a few countries under a different name than common usage.
_ALIASES = {
    "myanmar": ["burma"], "burma": ["myanmar"],
    "czechia": ["czech republic"], "czech republic": ["czechia"],
    "turkey": ["turkiye", "türkiye"], "türkiye": ["turkey"], "turkiye": ["turkey"],
    "ivory coast": ["cote d'ivoire"], "cote d'ivoire": ["ivory coast"],
    "south korea": ["korea, republic of"], "north korea": ["korea, democratic people's republic of"],
    "russia": ["russian federation"], "uae": ["united arab emirates"], "usa": ["united states"],
    "kyrgyzstan": ["kyrgyz republic"], "kyrgyz republic": ["kyrgyzstan"],   # feed: "The Kyrgyz Republic"
}

# Countries whose overall advisory is Level 1-2 but that contain specific Level 3-4 areas the State
# Department calls out by name. Curated from the per-country advisories; review periodically. Each entry
# is (region keyword, level). A place is blocked when its resolved region matches one of these; and a
# country listed here is failed CLOSED when the region can't be determined.
REGIONAL_WARNINGS: dict[str, list[tuple[str, int]]] = {
    "india": [("jammu", 4), ("kashmir", 4), ("manipur", 4), ("india-pakistan border", 4),
              ("chhattisgarh", 3), ("jharkhand", 3), ("bihar", 3), ("odisha", 3),
              ("arunachal", 3), ("nagaland", 3), ("mizoram", 3), ("assam", 3), ("maoist", 3)],
    "mexico": [("colima", 4), ("guerrero", 4), ("michoacan", 4), ("michoacán", 4), ("sinaloa", 4),
               ("tamaulipas", 4), ("zacatecas", 4)],
    "philippines": [("sulu", 4), ("mindanao", 4), ("marawi", 4)],
    "egypt": [("sinai", 4)],
    "colombia": [("arauca", 4), ("cauca", 4), ("norte de santander", 4)],
}
# The date REGIONAL_WARNINGS was last checked against the per-country advisories. advisory_drift() flags a
# listed country whose advisory was reissued after this, so the list is re-read rather than silently aging.
REGIONAL_WARNINGS_REVIEWED = "2026-10-31"

# Level 1-2 countries whose advisory names Level 3-4 areas that are NOT curated into REGIONAL_WARNINGS yet:
# blocked as a WHOLE country (fail closed) until someone reads the advisory and lists its areas above. Each
# entry: the advisory page and why it isn't curated. Remove the entry once its areas are in REGIONAL_WARNINGS.
UNCURATED_COUNTRIES: dict[str, str] = {
    # 2026-11-03: the feed flags higher-risk areas, but every advisory page below shows a "verify you are
    # human" check, so its named areas could not be read. Advisory issue dates are from the feed.
    "thailand": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
                "destination.tha.html (issued 2026-07-07)",
    "indonesia": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
                 "destination.idn.html (issued 2026-07-07)",
    "laos": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
            "destination.lao.html (issued 2026-07-28)",
    "cambodia": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
                "destination.khm.html (issued 2026-09-15)",
    "nepal": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
             "destination.npl.html (issued 2026-09-04)",
    "bangladesh": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
                  "destination.bgd.html (issued 2026-07-30)",
    "tajikistan": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
                  "destination.tjk.html (issued 2026-09-04)",
    # Same check on its page; listed once the Kyrgyz Republic alias let the feed match it (before that every
    # Kyrgyz place failed closed as "no advisory found").
    "kyrgyzstan": "https://travel.state.gov/content/tsg_aem/us/en/home/international-travel/travel-advisories/"
                  "destination.kgz.html (issued 2026-02-10)",
}

# Regions skipped for editorial/policy reasons, SEPARATE from travel advisories (no feed, no network).
# Extend by adding a region name mapped to the keywords (province, major places, common identifiers)
# that mark it.
SENSITIVE_REGIONS: dict[str, list[str]] = {
    "Xinjiang": ["xinjiang", "uyghur", "uighur", "uygur", "kashgar", "kashi", "urumqi", "ürümqi",
                 "turpan", "turfan", "hotan", "khotan", "aksu", "karamay", "korla", "yarkand",
                 "kuqa", "taklamakan", "tarim basin"],
    "Tibet": ["tibet", "tibetan", "xizang", "lhasa", "shigatse", "xigaze", "gyantse", "nyingchi",
              "nagqu", "qamdo", "chamdo", "ngari", "potala", "mount kailash", "mt kailash", "kailash",
              "everest base camp tibet", "tibetan plateau"],
}


def sensitive_region(*texts: str) -> str | None:
    """The sensitive region a topic/place falls in, or None. Keyword match only — separate from the
    travel-advisory feed — so it can be checked during topic selection without any network call."""
    blob = " ".join(t for t in texts if t).lower()
    if not blob.strip():
        return None
    for region, keywords in SENSITIVE_REGIONS.items():
        if any(re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", blob) for k in keywords):
            return region
    return None


_cache: dict = {"ts": 0.0, "data": None}


class UnsafeTopic(Exception):
    """Raised when a destination is under, or cannot be cleared of, a travel advisory."""


@dataclass
class Verdict:
    unsafe: bool
    reason: str = ""
    country: str = ""
    level: int = 0


def _feed_text() -> str:
    """The raw advisory feed (cached 6h, shared by advisories() and advisory_entries())."""
    now = time.time()
    if _cache.get("text") is not None and now - _cache.get("text_ts", 0.0) < _FEED_TTL:
        return _cache["text"]
    r = http().get(FEED_URL, timeout=25)
    raise_for_status(r, "US State Dept advisories")
    _cache.update(text=r.text, text_ts=now)
    return r.text


def advisories() -> dict[str, int]:
    """{country_name_lower: level 1-4} from the official US State Dept feed (cached 6h)."""
    now = time.time()
    if _cache["data"] is not None and now - _cache["ts"] < _FEED_TTL:
        return _cache["data"]
    data = {name.strip().lower(): int(lvl) for name, lvl in _ITEM_RE.findall(_feed_text())}
    if not data:
        raise RuntimeError("advisory feed parsed to 0 entries")
    _cache.update(ts=now, data=data)
    return data


@dataclass
class Advisory:
    """One country's entry in the feed. The feed names the overall level and summarises the advisory; the
    named higher-risk areas themselves are only on the per-country page, so ``areas`` records whether the
    summary says such areas exist."""
    country: str      # lower-cased, e.g. "india", "burma" (a "<name> Travel Advisory" title is normalised)
    level: int
    issued: str       # ISO date the advisory was last reissued, or ""
    areas: bool       # the summary says some areas carry a higher risk than the country as a whole


_FEED_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.S | re.I)
_FEED_TITLE_RE = re.compile(r"<title>\s*(?:<!\[CDATA\[)?\s*(.+?)\s*-\s*Level\s*(\d)\b", re.S | re.I)
_FEED_DATE_RE = re.compile(r"<pubDate>[^<]*?(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})", re.I)
_FEED_DESC_RE = re.compile(r"<description>(.*?)</description>", re.S | re.I)
_AREAS_RE = re.compile(r"some areas (?:have|carry) (?:an )?(?:increased|higher) risk|do not travel to|"
                       r"reconsider travel to", re.I)


def advisory_entries(text: str | None = None) -> list[Advisory]:
    """Every country in the official feed with its level, reissue date and whether it has higher-risk areas."""
    text = _feed_text() if text is None else text
    out: list[Advisory] = []
    for item in _FEED_ITEM_RE.findall(text):
        t = _FEED_TITLE_RE.search(item)
        if not t:
            continue
        name = re.sub(r"\s+travel advisory$", "", html.unescape(t.group(1)).strip(), flags=re.I).lower()
        name = re.sub(r"^the\s+", "", name)            # "The Kyrgyz Republic" -> "kyrgyz republic"
        issued = ""
        d = _FEED_DATE_RE.search(item)
        if d:
            try:
                issued = datetime.strptime(" ".join(d.groups()), "%d %b %Y").date().isoformat()
            except ValueError:
                issued = ""
        desc = _FEED_DESC_RE.search(item)
        summary = html.unescape(re.sub(r"<[^>]+>", " ", desc.group(1))) if desc else ""
        out.append(Advisory(name, int(t.group(2)), issued, bool(_AREAS_RE.search(summary))))
    return out


def country_names(country: str, aliases: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Lower-cased ``country`` plus its aliases and the State Department's own name for it (Burma for
    Myanmar, ...), de-duplicated in order — the names an advisory may be filed under."""
    low = [n.strip().lower() for n in [country, *aliases] if str(n).strip()]
    for n in list(low):
        low += _ALIASES.get(n, [])
    out: list[str] = []
    for n in low:
        if n and n not in out:
            out.append(n)
    return out


def advisory_drift(entries: list[Advisory], countries: list[str] | None = None,
                   reviewed: str = REGIONAL_WARNINGS_REVIEWED) -> list[tuple[str, str]]:
    """Where REGIONAL_WARNINGS no longer matches the live advisories. Each (country, reason) means a person
    should re-read that country's advisory and update the list:
      * a listed country is missing from the feed, is now Level 3-4 overall (a regional entry only makes
        sense for a Level 1-2 country), or had its advisory reissued after the list was ``reviewed``;
      * an unlisted Level 1-2 country whose advisory says some areas carry a higher risk — it has Level 3-4
        areas the list doesn't name (or it is in UNCURATED_COUNTRIES: blocked whole until they are listed).
    ``countries`` limits both checks to the countries a channel covers."""
    def find(names: list[str]) -> Advisory | None:
        for e in entries:
            if e.country in names:
                return e
        return None

    out: list[tuple[str, str]] = []
    listed: set[str] = set()
    scope = {n for c in countries for n in country_names(c)} if countries else None
    uncurated = {n for c in UNCURATED_COUNTRIES for n in country_names(c)}
    for c in REGIONAL_WARNINGS:
        names = country_names(c)
        listed.update(names)
        if scope is not None and not set(names) & scope:
            continue
        e = find(names)
        if e is None:
            out.append((c, "listed in REGIONAL_WARNINGS but missing from the advisory feed"))
        elif e.level >= BLOCK_LEVEL:
            out.append((c, f"now Level {e.level} overall, so the whole country is blocked and its regional "
                           f"entry is out of date"))
        elif e.issued and e.issued > reviewed:
            out.append((c, f"advisory reissued {e.issued}, after REGIONAL_WARNINGS was reviewed ({reviewed})"))
    for e in entries:
        if e.level >= BLOCK_LEVEL or not e.areas:
            continue
        names = set(country_names(e.country))
        if names & listed or (scope is not None and not names & scope):
            continue
        if names & uncurated:
            out.append((e.country, "whole country blocked until the higher-risk areas named in its advisory are "
                                   "added to REGIONAL_WARNINGS"))
        else:
            out.append((e.country, f"Level {e.level} overall but the advisory names higher-risk areas that are "
                                   f"not in REGIONAL_WARNINGS"))
    return out


_IDENTIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"country": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}},
                   "region": {"type": "string"}},
    "required": ["country", "aliases", "region"],
}


def _identify(topic: str, llm) -> tuple[list[str], str]:
    """(lower-cased country names incl. aliases, lower-cased region/state) the topic is about.
    Returns ([], "") when the topic is not about one specific real place."""
    system = "You identify where a travel topic is. Reply only with strict JSON."
    user = ('Return STRICT JSON {"country": "<name or empty>", "aliases": ["other names, including the '
            'official US State Department name if different, e.g. Burma for Myanmar"], '
            '"region": "<the state/province/region within the country the place is in, or empty>"}. '
            'If the topic is not about one specific real country/place, return an empty country.\nTopic: ' + topic)
    try:
        data = llm.complete_json(system, user, schema=_IDENTIFY_SCHEMA, max_tokens=250, temperature=0.0)
    except Exception as e:
        log.warning("Safety: could not determine location for %r (%s)", topic, redact(e))
        return [], ""
    if not isinstance(data, dict) or not str(data.get("country", "")).strip():
        return [], ""
    names = country_names(str(data.get("country", "")), [str(a) for a in (data.get("aliases") or [])])
    return names, str(data.get("region", "")).strip().lower()


def _regional_verdict(names: list[str], region: str) -> Verdict | None:
    """For a country with named Level 3-4 areas: block when the region is one of them, fail CLOSED when the
    region is unknown, and return None (country-level result stands) when the region is a safe one."""
    country = next((n for n in names if n in REGIONAL_WARNINGS), None)
    if country is None:
        return None
    if not region:
        return Verdict(True, f"{country.title()}: the region could not be determined, and this country has "
                             f"Level 3-4 areas — skipping to be safe", country)
    for kw, lvl in REGIONAL_WARNINGS[country]:
        if kw in region or region in kw:
            kind = "Do Not Travel" if lvl >= 4 else "Reconsider Travel"
            return Verdict(True, f"{country.title()} ({region.title()}): US State Dept Level {lvl} ({kind})",
                           country, lvl)
    return None


def _match_level(names: list[str], feed: dict[str, int]) -> tuple[int | None, str]:
    for n in names:                       # exact name match first
        if n in feed:
            return feed[n], n
    for n in names:                       # then a contains-match either way
        for country, lvl in feed.items():
            if n in country or country in n:
                return lvl, country
    return None, ""


def check(topic: str, llm) -> Verdict:
    names, region = _identify(topic, llm)
    if not names:
        return Verdict(False)  # not a specific destination -> nothing to check
    return verdict_for(names, region)


def verdict_for(names: list[str], region: str = "") -> Verdict:
    """The advisory decision for a place already located: ``names`` from country_names(), ``region`` its
    state/province. Same fail-CLOSED rules as check(), with no LLM call — so a topic list can be vetted
    ahead of time exactly as the Maker will vet each video."""
    names = [n.strip().lower() for n in names if n and n.strip()]
    region = (region or "").strip().lower()
    if not names:
        return Verdict(False)
    uncurated = next((n for n in names if n in UNCURATED_COUNTRIES), None)
    if uncurated:
        return Verdict(True, f"{uncurated.title()}: its advisory names Level 3-4 areas that are not curated yet "
                             f"- whole country blocked until they are", uncurated)
    try:
        feed = advisories()
    except Exception as e:
        log.warning("Safety: advisory feed unavailable (%s); failing CLOSED, skipping %r", redact(e), names[0])
        return Verdict(True, "travel advisory could not be verified (feed unavailable)", names[0])
    level, matched = _match_level(names, feed)
    if level is None:
        log.warning("Safety: no official advisory found for %r; failing CLOSED, skipping", names[0])
        return Verdict(True, f"no official advisory found for '{names[0]}' (could not verify)", names[0])
    if level >= BLOCK_LEVEL:
        kind = "Do Not Travel" if level >= 4 else "Reconsider Travel"
        return Verdict(True, f"{matched.title()}: US State Dept Level {level} ({kind})", matched, level)
    # The country as a whole is Level 1-2, but it may contain Level 3-4 areas (e.g. India: Jammu & Kashmir).
    regional = _regional_verdict(names, region)
    if regional is not None:
        if regional.unsafe:
            log.warning("Safety: %s", regional.reason)
        return regional
    log.info("Safety: %s is Level %d — ok", matched.title(), level)
    return Verdict(False, country=matched, level=level)
