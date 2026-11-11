"""Where the travel advisories' named Level 3-4 areas are.

A State Department advisory names areas ("Mindanao", "Jammu and Kashmir", "within 10 km of the India-Pakistan
border"), but a place reaches the safety check with whatever region name an LLM or a topic list happened to
give it - "Zamboanga City", "Zamboanga Peninsula" - and none of those contains the word "Mindanao". Matching
the area's own name let Fort Pilar in Zamboanga City (Mindanao) through as "Philippines, Level 2, ok".

So each area is described two ways, and a place is in it if EITHER says so (fail closed):

* ``keywords`` - the area's name and its provinces, districts and best-known towns, matched as whole words
  against the place's region and name. Only names that are unambiguous within the country: "Bishnupur" is
  also the West Bengal temple town and "Aurangabad" also the gateway to Ajanta, so they are left to the
  coordinates.
* geometry - ``hulls``: groups of reference towns (each group's convex hull, widened by ``margin_km``; a
  concave area such as Assam is several groups so a hull never swallows a neighbour like Meghalaya), and
  ``lines``: polylines with a buffer, for "within 10 km of the border".

The reference towns are real places with their English Wikipedia coordinates (advisory_area_data.py, built
2026-11-11), so every area can be audited town by town. Margins are generous on purpose: wrongly skipping a
safe place costs one topic; wrongly allowing an unsafe one is what this module exists to prevent.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property

Point = tuple[float, float]           # (lat, lon)
_KM_PER_DEG_LAT = 110.57


def _xy(p: Point, origin: Point) -> tuple[float, float]:
    """Kilometres east/north of ``origin`` (equirectangular; accurate to a few % over a few hundred km)."""
    return ((p[1] - origin[1]) * 111.32 * math.cos(math.radians(origin[0])), (p[0] - origin[0]) * _KM_PER_DEG_LAT)


def _hull(points: list[Point]) -> list[Point]:
    """Convex hull (Andrew's monotone chain) in lat/lon, counter-clockwise, computed in a local projection."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts
    o = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    xy = {p: _xy(p, o) for p in pts}
    pts.sort(key=lambda p: xy[p])

    def cross(a, b, c):
        (ax, ay), (bx, by), (cx, cy) = xy[a], xy[b], xy[c]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_hull_km(p: Point, hull: list[Point]) -> float:
    """0 inside the convex polygon ``hull`` (counter-clockwise), else the distance to its nearest edge."""
    if not hull:
        return math.inf
    v = [_xy(q, p) for q in hull]            # the place itself is the origin
    if len(v) == 1:
        return math.hypot(*v[0])
    if len(v) == 2:
        return _seg_dist((0.0, 0.0), v[0], v[1])
    inside = all((b[0] - a[0]) * (0 - a[1]) - (b[1] - a[1]) * (0 - a[0]) >= 0 for a, b in zip(v, v[1:] + v[:1]))
    if inside:
        return 0.0
    return min(_seg_dist((0.0, 0.0), a, b) for a, b in zip(v, v[1:] + v[:1]))


def distance_to_line_km(p: Point, line: list[Point]) -> float:
    v = [_xy(q, p) for q in line]
    if len(v) == 1:
        return math.hypot(*v[0])
    return min(_seg_dist((0.0, 0.0), a, b) for a, b in zip(v, v[1:]))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", (text or "").lower())).strip()


@dataclass(frozen=True)
class Area:
    """One named Level 3-4 area of a country's advisory."""
    name: str
    level: int
    keywords: tuple[str, ...] = ()
    groups: tuple[tuple[Point, ...], ...] = ()          # reference towns; each group -> one convex hull
    margin_km: float = 20.0
    lines: tuple[tuple[tuple[Point, ...], float], ...] = ()   # (polyline, buffer km)
    exclude: tuple[str, ...] = ()                      # phrases removed before keyword matching

    @cached_property
    def hulls(self) -> list[list[Point]]:
        return [_hull(list(g)) for g in self.groups if g]

    @cached_property
    def _patterns(self) -> list[tuple[str, re.Pattern]]:
        return [(k, re.compile(rf"(?<![\w-]){re.escape(_norm_text(k))}(?![\w-])")) for k in self.keywords if k]

    def named_in(self, *texts: str) -> str:
        """The keyword one of ``texts`` names this area by, or ""."""
        blob = " | ".join(_norm_text(t) for t in texts if t)
        for phrase in self.exclude:
            blob = blob.replace(_norm_text(phrase), " ")
        for k, pat in self._patterns:
            if pat.search(blob):
                return k
        return ""

    def contains(self, lat: float, lon: float) -> bool:
        p = (float(lat), float(lon))
        if any(distance_to_hull_km(p, h) <= self.margin_km for h in self.hulls):
            return True
        return any(distance_to_line_km(p, list(line)) <= buf for line, buf in self.lines)

    @property
    def has_geometry(self) -> bool:
        return bool(self.groups or self.lines)


def as_area(entry) -> Area:
    """REGIONAL_WARNINGS entries are Area objects; a bare (keyword, level) pair (older code, tests) is an area
    known by that one name."""
    if isinstance(entry, Area):
        return entry
    kw, level = entry
    return Area(kw, int(level), (kw,))
