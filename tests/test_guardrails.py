"""Tests for the Maker guardrails: safety filter, channel guard and the fact-check pass (no network)."""

import pytest

from purffle_shorts import factcheck, safety, youtube
from purffle_shorts.config import Settings
from purffle_shorts.script import Scene, Script


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete_json(self, system, user, **k):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# --------------------------------------------------------------- safety filter
FEED = {"burma": 4, "india": 2, "saudi arabia": 3, "thailand": 1}


def test_safety_blocks_myanmar(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Explore Bagan, Myanmar", FakeLLM({"country": "Myanmar", "aliases": ["Burma"]}))
    assert v.unsafe and v.level == 4 and "burma" in v.country.lower()


def test_safety_blocks_mrauk_u_myanmar(monkeypatch):
    # The concrete place a video resolves to (Mrauk U is in Myanmar, listed as "Burma" at Level 4).
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Mrauk U, Myanmar", FakeLLM({"country": "Myanmar", "aliases": ["Burma"]}))
    assert v.unsafe and v.level == 4 and "burma" in v.country.lower()


def test_safety_blocks_reconsider_level3(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Riyadh desert trip", FakeLLM({"country": "Saudi Arabia", "aliases": []}))
    assert v.unsafe and v.level == 3


def test_safety_allows_india(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Hampi, India", FakeLLM({"country": "India", "aliases": [], "region": "Karnataka"}))
    assert not v.unsafe and v.level == 2


def test_safety_blocks_level4_region(monkeypatch):
    # India is overall Level 2, but Manipur is a Level 4 "Do Not Travel" area.
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("A festival in Manipur, India", FakeLLM({"country": "India", "aliases": [], "region": "Manipur"}))
    assert v.unsafe and v.level == 4 and "manipur" in v.reason.lower()


def test_safety_blocks_level3_region(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Bastar, Chhattisgarh, India",
                     FakeLLM({"country": "India", "aliases": [], "region": "Chhattisgarh"}))
    assert v.unsafe and v.level == 3


def test_safety_skips_when_region_unknown(monkeypatch):
    # Fail CLOSED: India has Level 3-4 areas, so an unresolved region is skipped.
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Somewhere in India", FakeLLM({"country": "India", "aliases": [], "region": ""}))
    assert v.unsafe and "region could not be determined" in v.reason.lower()


def test_safety_no_regional_check_for_plain_country(monkeypatch):
    # Thailand (Level 1) is not in the regional-warnings map, so a missing region is fine.
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Chiang Mai, Thailand", FakeLLM({"country": "Thailand", "aliases": [], "region": ""}))
    assert not v.unsafe and v.level == 1


# --------------------------------------------------------------- sensitive regions (separate from advisories)
def test_sensitive_region_flags_tibet_and_xinjiang():
    assert safety.sensitive_region("Lhasa, Tibet, China") == "Tibet"
    assert safety.sensitive_region("Kashgar old town") == "Xinjiang"
    assert safety.sensitive_region("A tour of the Potala Palace") == "Tibet"
    assert safety.sensitive_region("Hampi, India") is None
    assert safety.sensitive_region("") is None
    assert safety.sensitive_region("mapping the mapping") is None   # no accidental substring hits


def test_sensitive_region_is_easy_to_extend(monkeypatch):
    monkeypatch.setitem(safety.SENSITIVE_REGIONS, "Testland", ["testville"])
    assert safety.sensitive_region("A visit to Testville") == "Testland"


def test_sensitive_region_blocks_by_resolved_place_not_article_mentions():
    # The guard now runs on the RESOLVED place + country only, so a village merely NEAR a sensitive border
    # is not blocked for the border being mentioned, while a place actually IN the region is.
    assert safety.sensitive_region("Chitkul, Himachal Pradesh", "India") is None   # near the Tibet border, not in it
    assert safety.sensitive_region("Lhasa", "China") == "Tibet"
    assert safety.sensitive_region("Kashgar", "China") == "Xinjiang"


def test_sensitive_region_covers_major_places_by_name():
    # The check is now place+country only, so the region lists must name the major places themselves.
    assert safety.sensitive_region("Mount Kailash", "China") == "Tibet"
    assert safety.sensitive_region("Shigatse", "China") == "Tibet"
    assert safety.sensitive_region("Turpan", "China") == "Xinjiang"
    assert safety.sensitive_region("Urumqi", "China") == "Xinjiang"


# --------------------------------------------------------------- title length (after fact-check rewrite)
def test_final_title_caps_base_under_60_even_after_a_long_rewrite():
    from purffle_shorts.pipeline import final_title
    long = "Why does Gokarna's ancient Shiva temple genuinely look like a giant stone cow's ear?"  # 84 chars
    sc = Script(topic="Gokarna", title=long, hook_text="",
                scenes=[Scene(narration="A temple shaped like an ear.", search_query="gokarna")],
                description="d", tags=["gokarna"])
    out = final_title(sc, Settings())                      # vertical Short -> base capped, then " #shorts"
    assert out.endswith(" #shorts")
    base = out[: -len(" #shorts")]
    assert len(base) <= 60, f"{len(base)}: {base!r}"       # the fact-check rewrite no longer slips past the cap


def test_safety_ignores_non_country(monkeypatch):
    called = []
    monkeypatch.setattr(safety, "advisories", lambda: called.append(1) or FEED)
    v = safety.check("productivity tips", FakeLLM({"country": "", "aliases": []}))
    assert not v.unsafe and called == []  # no feed lookup for non-destinations


def test_safety_fails_closed_when_feed_down(monkeypatch):
    def boom():
        raise RuntimeError("feed down")
    monkeypatch.setattr(safety, "advisories", boom)
    v = safety.check("Hampi, India", FakeLLM({"country": "India", "aliases": []}))
    assert v.unsafe and "could not be verified" in v.reason  # fail CLOSED on outage


def test_safety_fails_closed_when_country_missing(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: FEED)
    v = safety.check("Visit Testlandia", FakeLLM({"country": "Testlandia", "aliases": []}))
    assert v.unsafe and "could not verify" in v.reason  # unknown country -> skip


# --------------------------------------------------------------- fact check
def _script():
    return Script(
        topic="Bagan", title="85% of temples are hidden",
        hook_text="85% of Bagan's 2,300 temples are never seen",
        scenes=[
            Scene(narration="Bagan has exactly 2,300 temples.", search_query="bagan", emphasis=["2,300"]),
            Scene(narration="Most are peaceful at dawn.", search_query="dawn", emphasis=[]),
        ],
        description="Built in the 1500s, Bagan has 2,300 temples.",
        tags=["bagan", "2300 temples"],
    )


def test_factcheck_rewrites_unsupported_stats():
    reply = {
        "title": "Most temples are hidden",
        "hook": "Most of Bagan's temples are never seen",
        "description": "Built centuries ago, Bagan has thousands of temples.",
        "tags": ["bagan", "temples"],
        "hashtags": [],
        "lines": [
            {"id": 0, "text": "Bagan has thousands of temples."},
            {"id": 1, "text": "Most are peaceful at dawn."},
        ],
        "flagged": ["removed 2,300, 85% and the 1500s date"],
    }
    out = factcheck.scrub(FakeLLM(reply), _script(), settings=None)
    assert "thousands" in out.scenes[0].narration and "2,300" not in out.scenes[0].narration
    assert out.scenes[0].emphasis == []          # "2,300" pruned — no longer in the line
    assert "85%" not in out.title and "85%" not in out.hook_text
    assert "1500s" not in out.description and "centuries" in out.description   # metadata fact-checked too
    assert out.tags == ["bagan", "temples"]


def test_ground_rejects_invented_claims_not_in_source():
    src = "Hampi was the capital of the Vijayanagara Empire. The city fell in 1565."
    sc = Script(
        topic="Hampi", title="Why was Hampi abandoned?", hook_text="",
        scenes=[
            Scene(narration="Hampi was the capital of Vijayanagara.", search_query="hampi"),
            Scene(narration="The city fell in 1565.", search_query="hampi"),
            Scene(narration="It was home to exactly 700,000 people by 1491.", search_query="hampi"),
        ],
        description="", tags=[])
    out = factcheck.ground(sc, src)
    joined = out.narration
    assert "700,000" not in joined and "1491" not in joined   # invented specifics are removed
    assert "1565" in joined                                    # a fact that IS in the source survives
    assert len(out.scenes) == 2                                # the all-invented scene is dropped


def test_ground_catches_space_separated_numbers():
    # "1 500" (space as a thousands separator) is verified as 1500, not as "1" and "500".
    src = "The site has about 1,600 surviving structures."
    sc = Script(topic="x", title="t", hook_text="",
                scenes=[Scene(narration="It spans a hill.", search_query="q"),
                        Scene(narration="Diggers found over 1 500 structures here.", search_query="q")],
                description="", tags=[])
    out = factcheck.ground(sc, src)
    assert all("1 500" not in s.narration and "1500" not in s.narration for s in out.scenes)
    assert len(out.scenes) == 1 and "hill" in out.scenes[0].narration


def test_ground_is_noop_without_a_source():
    sc = Script(topic="x", title="t", hook_text="", scenes=[Scene("In 1999 something happened.", "q")],
                description="", tags=[])
    assert factcheck.ground(sc, "").scenes[0].narration == "In 1999 something happened."


def test_factcheck_fails_open_on_error():
    sc = _script()
    before = sc.scenes[0].narration
    out = factcheck.scrub(FakeLLM(RuntimeError("boom")), sc, settings=None)
    assert out.scenes[0].narration == before     # unchanged on LLM error


# --------------------------------------------------------------- channel guard
class FakeYT:
    """Stand-in for the YouTube API client: yt.channels().list(...).execute()."""
    def __init__(self, items):
        self._items = items

    def channels(self):
        return self

    def list(self, **kwargs):
        return self

    def execute(self):
        return {"items": self._items}


def _channel(title, handle):
    return [{"id": "UC_test_123", "snippet": {"title": title, "customUrl": handle}}]


def test_channel_guard_allows_matching_handle():
    yt = FakeYT(_channel("Asia Unseen", "@asiaunseen-au"))   # Google returns the handle lower-cased
    ident = youtube.verify_channel(Settings(require_channel="@AsiaUnseen-AU"), yt)
    assert ident["title"] == "Asia Unseen"


def test_channel_guard_blocks_wrong_channel():
    yt = FakeYT(_channel("Someone Else", "@someoneelse"))
    with pytest.raises(youtube.ChannelMismatch):
        youtube.verify_channel(Settings(require_channel="@AsiaUnseen-AU"), yt)


def test_channel_guard_blocks_when_no_channel():
    with pytest.raises(youtube.ChannelMismatch):
        youtube.verify_channel(Settings(require_channel="@AsiaUnseen-AU"), FakeYT([]))


def test_channel_guard_matches_by_title_fallback():
    yt = FakeYT(_channel("Asia Unseen", ""))   # no handle set -> fall back to title match
    ident = youtube.verify_channel(Settings(require_channel="Asia Unseen"), yt)
    assert ident["id"] == "UC_test_123"
