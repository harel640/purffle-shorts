"""The ranked topic list (TOPIC_LIST_FILE): picked from first, falls back when missing/stale/used up."""

import json
from datetime import datetime, timedelta, timezone

from purffle_shorts import topics as T
from purffle_shorts.config import Settings
from purffle_shorts.history import History


def _write(tmp_path, places, days_old=0):
    gen = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat(timespec="seconds")
    p = tmp_path / "topic-list.json"
    p.write_text(json.dumps({"generated": gen, "places": places}), encoding="utf-8")
    return p


PLACES = [
    {"rank": 1, "place": "Gokarna", "region": "Karnataka", "country": "India"},
    {"rank": 2, "place": "Phong Nha", "region": "Quang Binh", "country": "Vietnam"},
    {"rank": 3, "place": "Hampi", "region": "Karnataka", "country": "India"},
]


def _setup(tmp_path, places=PLACES, days_old=0):
    s = Settings(topic_list_file=str(_write(tmp_path, places, days_old)), niches=["hidden gems in Asia"])
    return s, History(tmp_path / "history.db")


def test_list_is_picked_first_in_rank_order_and_each_entry_claimed(tmp_path):
    s, h = _setup(tmp_path)
    first = T.pick_topic(s, h)
    assert first.source == "list" and first.subject == "Gokarna, Karnataka, India"
    second = T.pick_topic(s, h)          # rank 1 was claimed when picked, even before any video is made
    assert second.subject == "Phong Nha, Quang Binh, Vietnam"


def test_list_skips_places_made_recently(tmp_path):
    s, h = _setup(tmp_path)
    h.add_video(status="uploaded", title="x", resolved_place="Mahabaleshwar Temple, Gokarna, India")
    assert T.pick_topic(s, h).subject.startswith("Phong Nha")


def test_used_up_list_falls_back_to_normal_sources(tmp_path):
    s, h = _setup(tmp_path, places=PLACES[:1])
    T.pick_topic(s, h)
    t = T.pick_topic(s, h)
    assert t.source == "niche" and t.subject == "hidden gems in Asia"


def test_stale_list_is_ignored(tmp_path):
    s, h = _setup(tmp_path, days_old=30)
    assert T.from_list(s, h) is None
    assert T.pick_topic(s, h).source == "niche"


def test_missing_empty_or_broken_list_falls_back(tmp_path):
    h = History(tmp_path / "history.db")
    assert T.from_list(Settings(topic_list_file=""), h) is None                       # not configured
    assert T.from_list(Settings(topic_list_file=str(tmp_path / "nope.json")), h) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert T.from_list(Settings(topic_list_file=str(bad)), h) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    s, h2 = _setup(empty, places=[])
    assert T.from_list(s, h2) is None
    assert T.pick_topic(s, h2).source == "niche"


def test_queued_idea_still_comes_before_the_list(tmp_path):
    s, h = _setup(tmp_path)
    h.add_idea("Bagan at dawn")
    assert T.pick_topic(s, h).source == "idea"
    assert T.pick_topic(s, h).source == "list"


def test_a_new_weeks_list_may_offer_a_place_again(tmp_path):
    # Claims are per list: a place claimed (but never made) last week can come back in this week's list.
    s, h = _setup(tmp_path, places=PLACES[:1], days_old=7)
    assert T.from_list(s, h).subject.startswith("Gokarna")
    s2 = Settings(topic_list_file=str(_write(tmp_path, PLACES[:1])), niches=["x"])
    assert T.from_list(s2, h).subject.startswith("Gokarna")


def test_env_wiring(monkeypatch):
    monkeypatch.setenv("TOPIC_LIST_FILE", "/work/data/topic-list.json")
    monkeypatch.setenv("TOPIC_LIST_MAX_AGE_DAYS", "8")
    s = Settings.from_env()
    assert s.topic_list_file == "/work/data/topic-list.json" and s.topic_list_max_age_days == 8


def test_list_subject_skips_parts_the_name_already_has():
    assert T.list_subject("Gokarna", "Karnataka", "India") == "Gokarna, Karnataka, India"
    assert T.list_subject("Singapore Botanic Gardens", "Singapore", "Singapore") == "Singapore Botanic Gardens"
    assert T.list_subject("Phong Nha", "", "Vietnam") == "Phong Nha, Vietnam"


def test_history_depth_counts_distinct_pre_1900_markers():
    text = ("Founded in 1150 by the Hoysala dynasty, rebuilt in 1565, it flourished in the 14th century. "
            "Coins from 200 BCE were found. Modern roads arrived in 1990.")
    assert T.history_depth(text) >= 5            # 1150, 1565, dynasty, 14th century, BCE (1990 doesn't count)
    assert T.history_depth("A quiet village with a waterfall, popular since 2005.") == 0
    assert T.history_depth("") == 0
