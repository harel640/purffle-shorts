"""Travel-safety guardrail.

Before a script is written, skip any destination under an active "Reconsider Travel"
(Level 3) or "Do Not Travel" (Level 4) advisory, read from the official US Department of
State travel-advisories RSS feed. A place from the Scout's list carries its country and region,
so it is decided without an LLM (verdict_for). Any other topic is located by the LLM (check).

Fail-CLOSED, always: a destination is skipped unless its advisory can be read and is Level 1-2.
A feed outage, a country missing from the feed, an unknown region in a country with Level 3-4
areas, and an LLM that cannot be asked where the topic is all skip it (the autopilot picks
another) rather than risk an unsafe one. Outages are marked ``transient`` so a listed place is
tried again later instead of being used up. Only a topic the LLM itself says is not about one
specific place is let through - and never at the produce stage, where the place must be known.
"""

from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime

from .advisory_area_data import BORDER_LINES, GROUPS
from .advisory_areas import Area, as_area
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

def _area(country: str, name: str, level: int, keywords: tuple[str, ...], **kw) -> Area:
    """An advisory area with its reference towns and border lines from advisory_area_data."""
    return Area(name, level, keywords, tuple(tuple(g) for g in GROUPS.get((country, name), [])),
                lines=tuple((tuple(line), buf) for line, buf in BORDER_LINES.get((country, name), [])), **kw)


# Countries whose overall advisory is Level 1-2 but that contain specific Level 3-4 areas the State
# Department calls out by name. Curated from the per-country advisories; review periodically. A place is
# blocked when its region or name names one of these areas (the area's own name, its provinces, districts or
# well-known towns) OR its coordinates fall in or near it (advisory_areas.py: reference-town hulls and border
# lines). A country listed here is failed CLOSED when neither its region nor its coordinates are known.
# The levels are as curated on REGIONAL_WARNINGS_REVIEWED; 2026-11-11 added only the geography and the names.
REGIONAL_WARNINGS: dict[str, list[Area]] = {
    "india": [
        _area("india", "Jammu and Kashmir", 4,
              ('jammu and kashmir', 'jammu & kashmir', 'jammu', 'kashmir', 'srinagar', 'gulmarg', 'pahalgam',
               'sonamarg', 'anantnag', 'baramulla', 'kupwara', 'gurez', 'bandipora', 'ganderbal', 'kishtwar',
               'bhaderwah', 'udhampur', 'rajouri', 'poonch', 'kathua', 'akhnoor', 'vaishno devi', 'katra',
               'amarnath', 'shopian', 'kulgam', 'pulwama', 'budgam', 'banihal', 'kokernag', 'verinag',
               'dal lake', 'wular', 'kargil', 'drass', 'dras', 'uri', 'tangdhar', 'keran', 'reasi', 'ramban',
               'doda', 'patnitop'), margin_km=12),
        _area("india", "Manipur", 4,
              ('manipur', 'imphal', 'loktak', 'moirang', 'ukhrul', 'churachandpur', 'thoubal', 'kakching',
               'senapati', 'tamenglong', 'kangpokpi', 'jiribam', 'moreh', 'keibul lamjao', 'tengnoupal',
               'kamjong', 'noney', 'pherzawl'), margin_km=15),
        _area("india", "India-Pakistan border", 4,
              ('india-pakistan border', 'indo-pak border', 'pakistan border', 'wagah', 'attari',
               'hussainiwala', 'munabao', 'suchetgarh', 'dera baba nanak', 'khem karan', 'tanot', 'longewala',
               'sir creek', 'hindumalkot', 'hindumalkote', 'sulemanki')),
        _area("india", "Chhattisgarh", 3,
              ('chhattisgarh', 'raipur', 'bastar', 'jagdalpur', 'dantewada', 'sukma', 'kanker', 'kondagaon',
               'abujmarh', 'chitrakote', 'tirathgarh', 'korba', 'ambikapur', 'mainpat', 'kawardha',
               'bhoramdeo', 'dhamtari', 'durg', 'bhilai', 'rajnandgaon', 'mahasamund', 'gariaband', 'jashpur',
               'raigarh', 'surguja', 'dongargarh', 'sirpur mahasamund', 'barnawapara', 'kanger valley')),
        _area("india", "Jharkhand", 3,
              ('jharkhand', 'ranchi', 'jamshedpur', 'dhanbad', 'bokaro', 'deoghar', 'dumka', 'hazaribagh',
               'netarhat', 'betla', 'palamu', 'daltonganj', 'medininagar', 'giridih', 'parasnath', 'shikharji',
               'sahibganj', 'rajmahal', 'chaibasa', 'saranda', 'simdega', 'gumla', 'lohardaga', 'latehar',
               'koderma', 'jamtara', 'pakur', 'godda', 'khunti', 'seraikela', 'maluti', 'dassam', 'hundru',
               'jonha', 'baidyanath')),
        _area("india", "Bihar", 3,
              ('bihar', 'patna', 'gaya', 'bodh gaya', 'bodhgaya', 'nalanda', 'rajgir', 'vaishali', 'pawapuri',
               'vikramshila', 'vikramashila', 'bhagalpur', 'muzaffarpur', 'darbhanga', 'purnia', 'kishanganj',
               'katihar', 'araria', 'supaul', 'madhubani', 'sitamarhi', 'motihari', 'bettiah', 'champaran',
               'siwan', 'chhapra', 'buxar', 'sasaram', 'kaimur', 'nawada', 'jamui', 'munger', 'begusarai',
               'samastipur', 'hajipur', 'arrah', 'jehanabad', 'barabar', 'valmiki nagar',
               'valmiki tiger reserve', 'raxaul', 'kesaria', 'lauriya', 'bhabua')),
        _area("india", "Odisha", 3,
              ('odisha', 'orissa', 'bhubaneswar', 'cuttack', 'puri', 'konark', 'konarak', 'chilika', 'chilka',
               'berhampur', 'sambalpur', 'rourkela', 'balasore', 'baleswar', 'baripada', 'similipal',
               'simlipal', 'keonjhar', 'koraput', 'malkangiri', 'rayagada', 'kandhamal', 'phulbani',
               'daringbadi', 'bhitarkanika', 'hirakud', 'jeypore', 'bhawanipatna', 'kalahandi', 'nabarangpur',
               'angul', 'dhenkanal', 'jajpur', 'kendrapara', 'paradip', 'paradeep', 'talcher', 'sundargarh',
               'jharsuguda', 'bargarh', 'balangir', 'bolangir', 'nuapada', 'khandagiri', 'lalitgiri', 'dhauli',
               'raghurajpur', 'satkosia', 'gahirmatha', 'chandipur', 'gopalpur-on-sea')),
        _area("india", "Arunachal Pradesh", 3,
              ('arunachal', 'itanagar', 'tawang', 'bomdila', 'dirang', 'ziro', 'seppa', 'daporijo', 'pasighat',
               'yingkiong', 'tezu', 'anini', 'roing', 'changlang', 'khonsa', 'namdapha', 'mechuka', 'menchuka',
               'tuting', 'parshuram kund', 'sela pass', 'bum la', 'bhalukpong', 'namsai', 'miao', 'mayodia',
               'dambuk')),
        _area("india", "Nagaland", 3,
              ('nagaland', 'kohima', 'dimapur', 'mokokchung', 'tuensang', 'wokha', 'zunheboto', 'phek',
               'kiphire', 'longleng', 'peren', 'noklak', 'dzukou', 'dzüko', 'dzuko', 'khonoma', 'kisama',
               'hornbill festival', 'pfutsero', 'touphema', 'longwa')),
        _area("india", "Mizoram", 3,
              ('mizoram', 'aizawl', 'lunglei', 'champhai', 'kolasib', 'mamit', 'serchhip', 'saiha', 'siaha',
               'lawngtlai', 'phawngpui', 'palak dil', 'dampa', 'reiek', 'vantawng', 'tam dil', 'khawzawl')),
        _area("india", "Assam", 3,
              ('assam', 'guwahati', 'kaziranga', 'majuli', 'manas', 'sivasagar', 'sibsagar', 'jorhat',
               'dibrugarh', 'tinsukia', 'tezpur', 'nagaon', 'haflong', 'dima hasao', 'karbi anglong', 'diphu',
               'silchar', 'karimganj', 'hailakandi', 'barpeta', 'kokrajhar', 'dhubri', 'goalpara',
               'bongaigaon', 'nalbari', 'golaghat', 'north lakhimpur', 'dhemaji', 'sualkuchi', 'hajo',
               'pobitora', 'dibru-saikhowa', 'digboi', 'umananda', 'kamakhya', 'sadiya', 'morigaon', 'mayong',
               'hojai', 'udalguri', 'chirang', 'baksa', 'biswanath', 'charaideo', 'nameri', 'orang',
               'barak valley', 'brahmaputra valley')),
        # Not a place: a Maoist/Naxalite area is named by districts above, or by these words.
        Area("Maoist (Naxalite) areas", 3, ("maoist", "naxalite", "naxal", "red corridor")),
    ],
    "mexico": [
        _area("mexico", "Colima", 4,
              ('colima', 'manzanillo', 'tecoman', 'tecomán', 'revillagigedo', 'socorro island')),
        _area("mexico", "Guerrero", 4,
              ('guerrero', 'acapulco', 'taxco', 'zihuatanejo', 'ixtapa', 'chilpancingo', 'iguala')),
        _area("mexico", "Michoacán", 4,
              ('michoacan', 'michoacán', 'morelia', 'uruapan', 'patzcuaro', 'pátzcuaro', 'apatzingan',
               'apatzingán', 'zitacuaro', 'zitácuaro', 'paricutin', 'paricutín', 'janitzio',
               'monarch butterfly biosphere')),
        _area("mexico", "Sinaloa", 4,
              ('sinaloa', 'culiacan', 'culiacán', 'mazatlan', 'mazatlán', 'los mochis', 'cosala', 'cosalá',
               'topolobampo')),
        _area("mexico", "Tamaulipas", 4,
              ('tamaulipas', 'reynosa', 'matamoros', 'nuevo laredo', 'tampico', 'ciudad victoria',
               'ciudad madero')),
        _area("mexico", "Zacatecas", 4,
              ('zacatecas', 'fresnillo', 'sombrerete', 'jerez de garcía salinas', 'jerez de garcia salinas')),
    ],
    "philippines": [
        _area("philippines", "Sulu Archipelago", 4,
              ('sulu', 'jolo', 'tawi-tawi', 'tawi tawi', 'basilan', 'bongao', 'siasi', 'sitangkai',
               'pangutaran', 'lamitan', 'turtle islands', 'mapun', 'sibutu', 'simunul'), margin_km=25),
        _area("philippines", "Mindanao", 4,
              ('mindanao', 'zamboanga', 'davao', 'cotabato', 'cagayan de oro', 'iligan', 'lanao', 'bukidnon',
               'misamis', 'agusan', 'surigao', 'siargao', 'dinagat', 'camiguin', 'sarangani', 'general santos',
               'sultan kudarat', 'maguindanao', 'butuan', 'dipolog', 'dapitan', 'pagadian', 'ozamiz', 'ozamis',
               'oroquieta', 'koronadal', 'kidapawan', 'tagum', 'bislig', 'tandag', 'lake sebu', 'mount apo',
               'mt. apo', 'island garden city of samal', 'soccsksargen', 'caraga', 'bangsamoro', 'barmm',
               'hinatuan', 'malaybalay', 'gingoog', 'cabadbaran', 'digos', 'mati city', 'nabunturan',
               'kitanglad', 'hamiguitan', 'tinuy-an', 'fort pilar')),
        _area("philippines", "Marawi", 4,
              ('marawi', 'lake lanao', 'lanao del sur'), margin_km=20),
    ],
    "egypt": [
        _area("egypt", "Sinai", 4,
              ('sinai', 'sharm el sheikh', 'sharm el-sheikh', 'sharm', 'dahab', 'nuweiba', 'taba',
               'saint catherine', 'st catherine', 'st. catherine', 'el tor', 'arish', 'rafah', 'ras mohammed',
               'ras muhammad', 'sheikh zuweid', 'nekhel', 'ras sudr', 'ras sedr', 'blue hole'), margin_km=10),
    ],
    "colombia": [
        _area("colombia", "Arauca", 4,
              ('arauca', 'saravena', 'arauquita')),
        _area("colombia", "Cauca", 4,
              ('cauca', 'popayan', 'popayán', 'tierradentro', 'guapi', 'timbiqui', 'timbiquí',
               'santander de quilichao'), exclude=('valle del cauca', 'cauca river', 'río cauca', 'rio cauca')),
        _area("colombia", "Norte de Santander", 4,
              ('norte de santander', 'cucuta', 'cúcuta', 'ocaña', 'ocana', 'catatumbo', 'tibu', 'tibú')),
    ],
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
    """Raised when a destination is under, or cannot be cleared of, a travel advisory. ``transient``: it could
    not be cleared because something was down (LLM, advisory feed) - worth trying again later."""

    def __init__(self, reason: str = "", transient: bool = False):
        super().__init__(reason)
        self.transient = transient


class LocationUnavailable(RuntimeError):
    """The LLM could not be asked where a topic is (every provider failed). Not the same as 'not a place'."""


@dataclass
class Verdict:
    unsafe: bool
    reason: str = ""
    country: str = ""
    level: int = 0
    transient: bool = False     # unsafe only because something was down: may pass on a later run


def unsafe_topic(v: Verdict) -> UnsafeTopic:
    return UnsafeTopic(v.reason, transient=v.transient)


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


# The reply is ~60 tokens of JSON, but Groq's gpt-oss models first spend 110-355 hidden reasoning tokens on
# it, and those count against max_tokens. At the old 250 the call failed with HTTP 400 ("Failed to validate
# JSON", nothing generated) whenever the reasoning ran long - "Fort Pilar, Zamboanga City, Philippines" failed
# 2 of 3 times - and check() then let the topic through. The client also sets a floor for such models.
IDENTIFY_MAX_TOKENS = 1500


def _identify(topic: str, llm) -> tuple[list[str], str]:
    """(lower-cased country names incl. aliases, lower-cased region/state) the topic is about.
    Returns ([], "") when the LLM says the topic is not about one specific real place, and raises
    LocationUnavailable when the LLM could not be asked (or its reply was unusable)."""
    system = "You identify where a travel topic is. Reply only with strict JSON."
    user = ('Return STRICT JSON {"country": "<name or empty>", "aliases": ["other names, including the '
            'official US State Department name if different, e.g. Burma for Myanmar"], '
            '"region": "<the state/province/region within the country the place is in, or empty>"}. '
            'If the topic is not about one specific real country/place, return an empty country.\nTopic: ' + topic)
    try:
        data = llm.complete_json(system, user, schema=_IDENTIFY_SCHEMA, max_tokens=IDENTIFY_MAX_TOKENS,
                                 temperature=0.0)
    except Exception as e:
        log.warning("Safety: could not determine location for %r (%s); failing CLOSED", topic, redact(e))
        raise LocationUnavailable(redact(e)) from e
    if not isinstance(data, dict):
        log.warning("Safety: unusable location reply for %r; failing CLOSED", topic)
        raise LocationUnavailable("unusable reply")
    if not str(data.get("country", "")).strip():
        return [], ""
    names = country_names(str(data.get("country", "")), [str(a) for a in (data.get("aliases") or [])])
    return names, str(data.get("region", "")).strip().lower()


def _kind(level: int) -> str:
    return "Do Not Travel" if level >= 4 else "Reconsider Travel"


def _regional_verdict(names: list[str], region: str, place: str = "", lat: float | None = None,
                      lon: float | None = None) -> Verdict | None:
    """For a country with named Level 3-4 areas: block when the place's region or name names one of them, or
    its coordinates are in or near one; fail CLOSED when neither the region nor the coordinates are known;
    otherwise None (the country-level result stands)."""
    country = next((n for n in names if n in REGIONAL_WARNINGS), None)
    if country is None:
        return None
    areas = [as_area(e) for e in REGIONAL_WARNINGS[country]]
    for a in areas:
        word = a.named_in(region, place)
        if word:
            return Verdict(True, f"{country.title()} ({a.name}, named as {word!r}): US State Dept Level "
                                 f"{a.level} ({_kind(a.level)})", country, a.level)
    located = lat is not None and lon is not None
    if located:
        for a in areas:
            if a.has_geometry and a.contains(lat, lon):
                return Verdict(True, f"{country.title()} ({a.name}: {lat:.2f}, {lon:.2f} is in or near it): US "
                                     f"State Dept Level {a.level} ({_kind(a.level)})", country, a.level)
    if not region and not located:
        return Verdict(True, f"{country.title()}: neither the region nor the coordinates are known, and this "
                             f"country has Level 3-4 areas — skipping to be safe", country)
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


def check(topic: str, llm, *, require_place: bool = False, lat: float | None = None,
          lon: float | None = None) -> Verdict:
    """The advisory decision for a topic located by the LLM. Fails CLOSED when the LLM can't be asked
    (transient). ``require_place``: the topic must be a real place (e.g. the resolved place of a video) - the
    LLM finding no country is then a reason to skip, not a pass. ``lat``/``lon``, when known, are checked
    against the warned areas too, and the topic text itself against their names."""
    try:
        names, region = _identify(topic, llm)
    except LocationUnavailable as e:
        return Verdict(True, f"the location could not be verified ({str(e)[:120]})", transient=True)
    if not names:
        if require_place:
            return Verdict(True, f"could not tell which country {topic!r} is in")
        return Verdict(False)  # the LLM says it is not one specific destination: there is no advisory to read
    return verdict_for(names, region, place=topic, lat=lat, lon=lon)


def same_country(a: str, b: str) -> bool:
    """True when two country names are the same country (aliases included: Myanmar = Burma)."""
    return bool(set(country_names(a)) & set(country_names(b)))


def verdict_for(names: list[str], region: str = "", *, place: str = "", lat: float | None = None,
                lon: float | None = None) -> Verdict:
    """The advisory decision for a place already located: ``names`` from country_names(), ``region`` its
    state/province, ``place`` its name (matched against the warned areas' names too) and, when known, its
    coordinates. Same fail-CLOSED rules as check(), with no LLM call — so a topic list can be vetted ahead of
    time exactly as the Maker will vet each video."""
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
        return Verdict(True, "travel advisory could not be verified (feed unavailable)", names[0], transient=True)
    level, matched = _match_level(names, feed)
    if level is None:
        log.warning("Safety: no official advisory found for %r; failing CLOSED, skipping", names[0])
        return Verdict(True, f"no official advisory found for '{names[0]}' (could not verify)", names[0])
    if level >= BLOCK_LEVEL:
        kind = "Do Not Travel" if level >= 4 else "Reconsider Travel"
        return Verdict(True, f"{matched.title()}: US State Dept Level {level} ({kind})", matched, level)
    # The country as a whole is Level 1-2, but it may contain Level 3-4 areas (e.g. India: Jammu & Kashmir).
    regional = _regional_verdict(names, region, place, lat, lon)
    if regional is not None:
        if regional.unsafe:
            log.warning("Safety: %s", regional.reason)
        return regional
    log.info("Safety: %s is Level %d — ok", matched.title(), level)
    return Verdict(False, country=matched, level=level)
