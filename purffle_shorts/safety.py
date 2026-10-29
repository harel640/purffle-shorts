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

import logging
import re
import time
from dataclasses import dataclass

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
}

_cache: dict = {"ts": 0.0, "data": None}


class UnsafeTopic(Exception):
    """Raised when a destination is under, or cannot be cleared of, a travel advisory."""


@dataclass
class Verdict:
    unsafe: bool
    reason: str = ""
    country: str = ""
    level: int = 0


def advisories() -> dict[str, int]:
    """{country_name_lower: level 1-4} from the official US State Dept feed (cached 6h)."""
    now = time.time()
    if _cache["data"] is not None and now - _cache["ts"] < _FEED_TTL:
        return _cache["data"]
    r = http().get(FEED_URL, timeout=25)
    raise_for_status(r, "US State Dept advisories")
    data = {name.strip().lower(): int(lvl) for name, lvl in _ITEM_RE.findall(r.text)}
    if not data:
        raise RuntimeError("advisory feed parsed to 0 entries")
    _cache.update(ts=now, data=data)
    return data


def _candidates(topic: str, llm) -> list[str]:
    """Lower-cased country names (primary + aliases) the topic is about, or [] if not a place."""
    system = "You identify the country a travel topic is about. Reply only with strict JSON."
    user = ('Return STRICT JSON {"country": "<name or empty>", "aliases": ["other names, including the '
            'official US State Department name if different, e.g. Burma for Myanmar"]}. '
            'If the topic is not about one specific real country/place, return an empty country.\nTopic: ' + topic)
    try:
        data = llm.complete_json(system, user, max_tokens=200, temperature=0.0)
    except Exception as e:
        log.warning("Safety: could not determine country for %r (%s)", topic, redact(e))
        return []
    if not isinstance(data, dict) or not str(data.get("country", "")).strip():
        return []
    names = [str(data.get("country", ""))] + [str(a) for a in (data.get("aliases") or [])]
    low = [n.strip().lower() for n in names if str(n).strip()]
    for n in list(low):
        low += _ALIASES.get(n, [])
    out: list[str] = []
    for n in low:
        if n and n not in out:
            out.append(n)
    return out


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
    names = _candidates(topic, llm)
    if not names:
        return Verdict(False)  # not a specific destination -> nothing to check
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
    log.info("Safety: %s is Level %d — ok", matched.title(), level)
    return Verdict(False, country=matched, level=level)
