"""SQLite history: every video made, what was uploaded or scheduled, which stock clips and topics
were used (so nothing repeats), today's upload count (YouTube quota), the idea queue filled by the
planner, and view counts for the videos that are live."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

PLACE_REUSE_DAYS = 90    # don't let the autopilot pick a place already made within this many days

# Generic words stripped when comparing place names, so "Hampi", "Hampi, India" and "Vittala Temple, Hampi"
# all reduce to the same distinctive token ("hampi").
_PLACE_GENERIC = {"temple", "temples", "city", "town", "fort", "ruins", "ruin", "old", "ancient", "great",
                  "palace", "site", "complex", "valley", "river", "lake", "mount", "hill", "cave", "caves",
                  "monastery", "stupa", "pagoda", "the", "of", "and", "at", "national", "park"}


def _place_tokens(text: str, extra_stop: frozenset[str] = frozenset()) -> set[str]:
    stop = _PLACE_GENERIC | extra_stop
    return {t for t in re.findall(r"[a-z]+", (text or "").lower()) if len(t) > 2 and t not in stop}


SAME_PLACE_KM = 3.0      # two places this close are one place for de-duplication (Hampi / Vittala Temple)


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _page_key(title: str | None) -> str:
    return re.sub(r"\s+", " ", (title or "").replace("_", " ")).strip().casefold()


def same_place(a: dict, b: dict) -> str:
    """Why places ``a`` and ``b`` ({wiki_page, lat, lon, place, country}) are the same place, or "".
    By the Wikipedia page (one article, whatever it is called - "Crystal Monastery" is "Shey Gompa") or by
    coordinates within SAME_PLACE_KM. Names are compared only when neither page nor coordinates can be: two
    different places that share a word ("Shey Palace", Ladakh vs "Shey Gompa", Nepal) are NOT the same."""
    pa, pb = _page_key(a.get("wiki_page")), _page_key(b.get("wiki_page"))
    if pa and pb and pa == pb:
        return f"same Wikipedia page ({a.get('wiki_page') or b.get('wiki_page')})"
    coords = all(isinstance(x, (int, float)) for x in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon")))
    if coords:
        d = distance_km(a["lat"], a["lon"], b["lat"], b["lon"])
        if d <= SAME_PLACE_KM:
            return f"{d:.1f} km apart"
    if (pa and pb) or coords:
        return ""                  # comparable by page/coordinates, and they differ: different places
    name_b = b.get("resolved_place") or ", ".join(x for x in (b.get("place"), b.get("country")) if x)
    if name_b and place_recently_used(a.get("place") or "", a.get("country") or "", [name_b]):
        return f"same name as {name_b!r}"
    return ""


def place_recently_used(place: str, country: str, recent_places: list[str]) -> bool:
    """True when ``place`` shares a distinctive name token with any recently-made place, so a place is matched
    by what it resolves to rather than by the exact title (e.g. 'Hampi' == 'Vittala Temple, Hampi')."""
    key = _place_tokens(place, frozenset(re.findall(r"[a-z]+", (country or "").lower())))
    if not key:
        return False
    return any(key & _place_tokens(rp) for rp in recent_places)

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    subject TEXT, source TEXT, topic TEXT, style TEXT, language TEXT,
    title TEXT, description TEXT, tags TEXT,
    folder TEXT, video_path TEXT, duration REAL,
    llm TEXT, voice TEXT,
    status TEXT NOT NULL,          -- rendered | queued | uploaded | scheduled | failed
    youtube_id TEXT, publish_at TEXT, uploaded_at TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS media_used (key TEXT PRIMARY KEY, used_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS topics_used (key TEXT PRIMARY KEY, used_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    subject TEXT NOT NULL,
    style TEXT, notes TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | used
    used_at TEXT, video_id INTEGER
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS videos_status ON videos (status);
CREATE INDEX IF NOT EXISTS videos_youtube_id ON videos (youtube_id);
CREATE INDEX IF NOT EXISTS videos_uploaded_at ON videos (uploaded_at);
CREATE INDEX IF NOT EXISTS ideas_status ON ideas (status, id);
"""

# Columns added after 2.0; existing databases get them on open.
VIDEO_COLUMNS = {
    "score": "INTEGER",            # Script Doctor retention score, 0-100
    "parent_id": "INTEGER",        # the original video when this one is a translated copy
    "views": "INTEGER", "likes": "INTEGER", "comments": "INTEGER", "stats_at": "TEXT",
    "resolved_place": "TEXT",      # the real place a video resolved to (for place-based de-duplication)
    "wiki_page": "TEXT",           # its Wikipedia article (canonical title): "Crystal Monastery" == "Shey Gompa"
    "lat": "REAL", "lon": "REAL",  # its coordinates: two names for one spot are the same place
    "country": "TEXT",             # its country, for the per-country pacing rule
    "topic_key": "TEXT",           # the claim (topics_used.key) this video was made from, e.g. a list place
}
TOPIC_COLUMNS = {
    "pending": "INTEGER",          # 1 = claimed by a run that has not finished with it yet (see topic_used)
}
CLAIM_STALE_HOURS = 2.0  # a pending claim this old with no video row was left by a killed run: free it again


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class History:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as c:
            # WAL lets the Studio read the library while a job is writing to it.
            try:
                c.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass  # e.g. a network drive without shared memory: the default journal still works
            c.executescript(SCHEMA)
            have = {r["name"] for r in c.execute("PRAGMA table_info(videos)")}
            for col, kind in VIDEO_COLUMNS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE videos ADD COLUMN {col} {kind}")
            have = {r["name"] for r in c.execute("PRAGMA table_info(topics_used)")}
            for col, kind in TOPIC_COLUMNS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE topics_used ADD COLUMN {col} {kind}")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, and much faster commits
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._db() as c:
            return c.execute(sql, args).lastrowid or 0

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._db() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # -- videos --
    def add_video(self, **fields) -> int:
        fields.setdefault("created_at", now_iso())
        if isinstance(fields.get("tags"), list):
            fields["tags"] = json.dumps(fields["tags"])
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        return self._exec(f"INSERT INTO videos ({cols}) VALUES ({marks})", tuple(fields.values()))

    def update_video(self, vid: int, **fields) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE videos SET {sets} WHERE id = ?", (*fields.values(), vid))

    def get(self, vid: int) -> dict | None:
        rows = self._all("SELECT * FROM videos WHERE id = ?", (vid,))
        return rows[0] if rows else None

    def recent(self, limit: int = 20) -> list[dict]:
        return self._all("SELECT * FROM videos ORDER BY id DESC LIMIT ?", (limit,))

    def delete_video(self, vid: int) -> None:
        self._exec("DELETE FROM videos WHERE id = ?", (vid,))

    def recent_titles(self, limit: int = 40) -> list[str]:
        return [r["title"] for r in self._all(
            "SELECT title FROM videos WHERE title IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))]

    def recent_places(self, days: int = 90) -> list[str]:
        """Resolved places of videos made in the last ``days`` days, so a place isn't repeated too soon."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        return [r["resolved_place"] for r in self._all(
            "SELECT resolved_place FROM videos WHERE resolved_place IS NOT NULL AND created_at >= ?", (cutoff,))]

    def country_quota_reason(self, country: str, gap_days: int = 0, week_max: int = 0) -> str:
        """Why another video about ``country`` would come too soon, or "". At most one per country within
        ``gap_days`` days, and at most ``week_max`` per country within 7 days (0 = no limit). Counts every video
        made; older rows without a country use the last part of their resolved place ("Hampi, India")."""
        want = (country or "").strip().casefold()
        if not want or not (gap_days or week_max):
            return ""
        now = datetime.now(timezone.utc)
        horizon = max(gap_days, 7 if week_max else 0)
        cutoff = (now - timedelta(days=horizon)).isoformat(timespec="seconds")
        made: list[datetime] = []
        for r in self._all("SELECT created_at, country, resolved_place FROM videos WHERE created_at >= ?", (cutoff,)):
            c = r["country"] or (r["resolved_place"] or "").rsplit(",", 1)[-1]
            if c.strip().casefold() == want:
                try:
                    made.append(datetime.fromisoformat(r["created_at"]))
                except ValueError:
                    continue
        if gap_days and any(now - t < timedelta(days=gap_days) for t in made):
            return f"a {country} video was made in the last {gap_days} days"
        if week_max and sum(now - t < timedelta(days=7) for t in made) >= week_max:
            return f"{week_max} {country} videos already this week"
        return ""

    def place_recently_made(self, *, wiki_page: str = "", lat: float | None = None, lon: float | None = None,
                            place: str = "", country: str = "", days: int = PLACE_REUSE_DAYS) -> str:
        """Why this place was already made in the last ``days`` days (see same_place), or "" if it wasn't."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        me = {"wiki_page": wiki_page, "lat": lat, "lon": lon, "place": place, "country": country}
        for r in self._all("SELECT resolved_place, wiki_page, lat, lon FROM videos WHERE created_at >= ? AND "
                           "(resolved_place IS NOT NULL OR wiki_page IS NOT NULL OR lat IS NOT NULL)", (cutoff,)):
            why = same_place(me, r)
            if why:
                return f"{r['resolved_place'] or r['wiki_page']}: {why}"
        return ""

    def videos_for_place_backfill(self) -> list[dict]:
        """Every video with the fields needed to fill in its resolved place."""
        return self._all("SELECT id, topic, subject, title, folder, resolved_place FROM videos ORDER BY id")

    def pending_uploads(self, statuses: tuple[str, ...] = ("queued",)) -> list[dict]:
        """queued = deferred by the daily quota; rendered = made with --no-upload."""
        marks = ", ".join("?" for _ in statuses)
        return self._all(f"SELECT * FROM videos WHERE status IN ({marks}) AND video_path IS NOT NULL ORDER BY id",
                         statuses)

    def uploads_since(self, since_iso: str) -> int:
        return self._all("SELECT COUNT(*) AS n FROM videos WHERE uploaded_at >= ?", (since_iso,))[0]["n"]

    def scheduled_times(self) -> list[str]:
        return [r["publish_at"] for r in self._all(
            "SELECT publish_at FROM videos WHERE publish_at IS NOT NULL AND status = 'scheduled'")]

    def stats(self) -> dict:
        rows = self._all("SELECT status, COUNT(*) AS n FROM videos GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    # -- channel performance --
    def live_video_ids(self, limit: int = 500) -> list[str]:
        return [r["youtube_id"] for r in self._all(
            "SELECT youtube_id FROM videos WHERE youtube_id IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))]

    def set_stats(self, youtube_id: str, views: int, likes: int, comments: int) -> None:
        self._exec("UPDATE videos SET views = ?, likes = ?, comments = ?, stats_at = ? WHERE youtube_id = ?",
                   (views, likes, comments, now_iso(), youtube_id))

    def performers(self, n: int = 5, *, best: bool = True, min_age_hours: int = 48) -> list[dict]:
        """Most (or least) viewed videos that have been public long enough for the numbers to mean something."""
        cutoff = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - min_age_hours * 3600,
                                        timezone.utc).isoformat(timespec="seconds")
        order = "DESC" if best else "ASC"
        return self._all(
            "SELECT id, title, style, views, likes, comments FROM videos WHERE views IS NOT NULL "
            f"AND COALESCE(publish_at, uploaded_at) <= ? ORDER BY views {order} LIMIT ?", (cutoff, n))

    def results(self, min_age_hours: int = 48) -> list[dict]:
        """Every video with view counts that has been public long enough to judge (original language only,
        so a translated copy doesn't count twice)."""
        cutoff = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - min_age_hours * 3600,
                                        timezone.utc).isoformat(timespec="seconds")
        return self._all(
            "SELECT id, title, style, score, views, likes, comments, publish_at, uploaded_at FROM videos "
            "WHERE views IS NOT NULL AND parent_id IS NULL AND COALESCE(publish_at, uploaded_at) <= ?", (cutoff,))

    # -- idea queue (planner) --
    def add_idea(self, subject: str, style: str = "", notes: str = "") -> int:
        return self._exec("INSERT INTO ideas (created_at, subject, style, notes) VALUES (?, ?, ?, ?)",
                          (now_iso(), subject.strip(), style or None, notes or None))

    def ideas(self, status: str | None = "pending", limit: int = 200) -> list[dict]:
        if status:
            return self._all("SELECT * FROM ideas WHERE status = ? ORDER BY id LIMIT ?", (status, limit))
        return self._all("SELECT * FROM ideas ORDER BY id DESC LIMIT ?", (limit,))

    def claim_idea(self) -> dict | None:
        """Take the oldest pending idea (atomically, so parallel workers never get the same one)."""
        with self._db() as c:
            row = c.execute("SELECT * FROM ideas WHERE status = 'pending' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return None
            c.execute("UPDATE ideas SET status = 'used', used_at = ? WHERE id = ?", (now_iso(), row["id"]))
            return dict(row)

    def link_idea(self, idea_id: int, video_id: int) -> None:
        self._exec("UPDATE ideas SET video_id = ? WHERE id = ?", (video_id, idea_id))

    def release_idea(self, idea_id: int) -> None:
        """Put an idea back in the queue after its video failed."""
        self._exec("UPDATE ideas SET status = 'pending', used_at = NULL WHERE id = ?", (idea_id,))

    def delete_idea(self, idea_id: int) -> None:
        self._exec("DELETE FROM ideas WHERE id = ?", (idea_id,))

    # -- de-duplication --
    def used_media(self) -> set[str]:
        return {r["key"] for r in self._all("SELECT key FROM media_used")}

    def mark_media(self, keys: list[str]) -> None:
        ts = now_iso()
        with self._db() as c:
            c.executemany("INSERT OR IGNORE INTO media_used (key, used_at) VALUES (?, ?)", [(k, ts) for k in keys])

    def topic_used(self, key: str) -> bool:
        """True when ``key`` is taken: used for good, or claimed by a run that is still going. A PENDING claim
        older than CLAIM_STALE_HOURS with no video row made from it was left by a run that never finished
        (the launcher's watchdog killed the container): it counts as free again."""
        rows = self._all("SELECT used_at, pending FROM topics_used WHERE key = ?", (key.lower(),))
        if not rows:
            return False
        if not rows[0]["pending"]:
            return True
        try:
            claimed = datetime.fromisoformat(rows[0]["used_at"])
        except (TypeError, ValueError):
            return True
        if datetime.now(timezone.utc) - claimed < timedelta(hours=CLAIM_STALE_HOURS):
            return True
        return bool(self._all("SELECT 1 FROM videos WHERE topic_key = ? LIMIT 1", (key.lower(),)))

    def mark_topic(self, key: str) -> None:
        """Use a topic for good."""
        self._exec("INSERT INTO topics_used (key, used_at, pending) VALUES (?, ?, NULL) "
                   "ON CONFLICT(key) DO UPDATE SET pending = NULL", (key.lower(), now_iso()))

    def claim_topic(self, key: str) -> None:
        """Take a topic for the run now starting: pending until the run finishes with it (finish_topic) or
        gives it back (release_topic). A stale pending claim (topic_used) is taken over with a fresh time."""
        self._exec("INSERT INTO topics_used (key, used_at, pending) VALUES (?, ?, 1) "
                   "ON CONFLICT(key) DO UPDATE SET used_at = excluded.used_at, pending = 1",
                   (key.lower(), now_iso()))

    def finish_topic(self, key: str) -> None:
        """The run is done with its claim - a video was made, or the place broke a rule: used for good."""
        self._exec("UPDATE topics_used SET pending = NULL WHERE key = ?", (key.lower(),))

    def release_topic(self, key: str) -> None:
        """Un-claim a topic (e.g. a listed place skipped only because a check could not run) so a later run
        can take it again."""
        self._exec("DELETE FROM topics_used WHERE key = ?", (key.lower(),))

    # -- small persistent state (e.g. the day YouTube last refused an upload) --
    def get_meta(self, key: str) -> str | None:
        rows = self._all("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str | None) -> None:
        if value is None:
            self._exec("DELETE FROM meta WHERE key = ?", (key,))
        else:
            self._exec("INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?) "
                       "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                       (key, value, now_iso()))
