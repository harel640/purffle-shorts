"""Travel-safety guardrail.

Before a script is written, skip any destination under an active "Do Not Travel" or
"Reconsider Travel" advisory. The country is identified from the topic by the LLM, then its
advisory level is read from a public source (travel-advisory.info, which aggregates official
advisories including the US State Department) on a 0-5 scale.

Fail-open: if the country can't be determined or the lookup fails, the topic is allowed (with
a warning) so a transient outage never halts the channel. Non-travel topics are never blocked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .utils import http, raise_for_status, redact

log = logging.getLogger("purffle")

# travel-advisory.info score: 0-2.5 normal, 2.5-3.5 increased caution, >=3.5 ~ reconsider/do-not-travel.
UNSAFE_SCORE = 3.5


class UnsafeTopic(Exception):
    """Raised when a destination is under a do-not-travel / reconsider-travel advisory."""


@dataclass
class Verdict:
    unsafe: bool
    reason: str = ""
    country: str = ""
    score: float = 0.0


def _country_of(topic: str, llm) -> tuple[str, str]:
    """(country, ISO-3166 alpha-2) for the topic, or ('','') if it is not about a specific country."""
    system = "You identify the country a travel topic is about. Reply only with strict JSON."
    user = ('Return STRICT JSON {"country": "<name or empty>", "iso2": "<2-letter ISO code or empty>"}. '
            'If the topic is not about one specific real country/place, return empty strings.\nTopic: ' + topic)
    try:
        data = llm.complete_json(system, user, max_tokens=200, temperature=0.0)
    except Exception as e:
        log.warning("Safety: could not determine country for %r (%s)", topic, redact(e))
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    return str(data.get("country", "")).strip(), str(data.get("iso2", "")).strip().upper()


def advisory_score(iso2: str) -> tuple[float, str]:
    """(score 0-5, source) from travel-advisory.info; raises on network/HTTP error."""
    r = http().get("https://www.travel-advisory.info/api", params={"countrycode": iso2}, timeout=20)
    raise_for_status(r, "travel advisory")
    entry = (r.json().get("data", {}) or {}).get(iso2, {}) or {}
    adv = entry.get("advisory", {}) or {}
    return float(adv.get("score", 0) or 0), str(adv.get("source", "travel-advisory.info"))


def check(topic: str, llm) -> Verdict:
    """Decide whether a destination topic should be skipped. Fail-open on any uncertainty."""
    country, iso2 = _country_of(topic, llm)
    if len(iso2) != 2 or not iso2.isalpha():
        return Verdict(False)  # not a specific country -> nothing to check
    try:
        score, source = advisory_score(iso2)
    except Exception as e:
        log.warning("Safety: advisory lookup failed for %s (%s); allowing", iso2, redact(e))
        return Verdict(False)
    if score >= UNSAFE_SCORE:
        return Verdict(True, f"{country}: advisory score {score:.1f}/5 "
                             f"(reconsider/do-not-travel; source {source})", country, score)
    log.info("Safety: %s advisory score %.1f/5 — ok", country or iso2, score)
    return Verdict(False, country=country, score=score)
