"""Advisory feed parsing, REGIONAL_WARNINGS drift detection and the LLM-free verdict_for() (no network)."""

from purffle_shorts import safety


def _item(title, date, desc):
    return (f"<item><title>{title}</title><link>https://x</link><pubDate>{date}</pubDate>"
            f"<description><![CDATA[{desc}]]></description></item>")


AREAS = "Exercise increased caution<p>due to <b>crime</b>. Some areas have increased risk.</p>"
PLAIN = "Exercise normal precautions<p>in this country.</p>"
FEED = "<rss><channel>" + "".join([
    _item("India - Level 2: Exercise Increased Caution", "Mon, 16 Jun 2025", AREAS),
    _item("Mexico Travel Advisory - Level 2: Exercise Increased Caution", "Tue, 12 Aug 2025", AREAS),
    _item("Philippines - Level 2: Exercise Increased Caution", "Mon, 02 Nov 2026", AREAS),   # after review
    _item("Colombia - Level 3: Reconsider Travel", "Tue, 31 Mar 2026", AREAS),                # now L3 overall
    _item("Thailand - Level 1: Exercise Normal Precautions", "Tue, 07 Jul 2026", AREAS),      # unlisted areas
    _item("Georgia - Level 1: Exercise Normal Precautions", "Thu, 13 Mar 2025", AREAS),       # out of scope
    _item("Vietnam - Level 1: Exercise Normal Precautions", "Wed, 24 Jun 2026", PLAIN),
    _item("Burma - Level 4: Do Not Travel", "Fri, 08 May 2026", AREAS),                       # whole blocked
]) + "</channel></rss>"                                                                       # Egypt missing

ASIA = ["India", "Thailand", "Vietnam", "Myanmar", "Philippines"]


def test_feed_entries_parse_level_date_areas_and_normalise_titles():
    e = {a.country: a for a in safety.advisory_entries(FEED)}
    assert e["india"].level == 2 and e["india"].issued == "2025-06-16" and e["india"].areas
    assert "mexico" in e and e["mexico"].level == 2          # "Mexico Travel Advisory" -> "mexico"
    assert e["vietnam"].areas is False and e["burma"].level == 4


def test_drift_flags_stale_listed_countries_and_unlisted_risk_areas(monkeypatch):
    monkeypatch.setattr(safety, "UNCURATED_COUNTRIES", {})
    drift = dict(safety.advisory_drift(safety.advisory_entries(FEED), countries=ASIA))
    assert "philippines" in drift and "2026-11-02" in drift["philippines"]  # reissued after the review date
    assert "thailand" in drift and "not in REGIONAL_WARNINGS" in drift["thailand"]
    assert "india" not in drift                                             # unchanged since review
    assert "vietnam" not in drift and "burma" not in drift                 # no areas / blocked outright
    assert "georgia" not in drift                                           # outside the channel's countries
    assert "colombia" not in drift and "egypt" not in drift                 # listed, but outside the channel


def test_drift_without_scope_checks_every_country():
    drift = dict(safety.advisory_drift(safety.advisory_entries(FEED)))
    assert "colombia" in drift and "Level 3" in drift["colombia"]          # whole country now blocked
    assert "egypt" in drift and "missing" in drift["egypt"]
    assert "georgia" in drift and "mexico" not in drift


def test_uncurated_country_keeps_being_reported_until_listed():
    drift = dict(safety.advisory_drift(safety.advisory_entries(FEED), countries=ASIA))
    assert "thailand" in drift and "whole country blocked" in drift["thailand"]


def test_drift_is_empty_when_the_feed_matches_the_list(monkeypatch):
    monkeypatch.setattr(safety, "REGIONAL_WARNINGS", {"india": [("jammu", 4)]})
    feed = "<rss>" + _item("India - Level 2: x", "Mon, 16 Jun 2025", AREAS) + "</rss>"
    assert safety.advisory_drift(safety.advisory_entries(feed), countries=["India"]) == []


def test_country_names_adds_state_department_aliases():
    assert safety.country_names("Myanmar") == ["myanmar", "burma"]
    assert safety.country_names("Burma", ["Myanmar"]) == ["burma", "myanmar"]


def test_kyrgyzstan_matches_the_feeds_kyrgyz_republic(monkeypatch):
    # The feed files it as "The Kyrgyz Republic"; without the alias every Kyrgyz place failed closed.
    feed = "<rss>" + _item("The Kyrgyz Republic - Level 1: Exercise Normal Precautions", "Mon, 01 Jun 2026",
                           PLAIN) + "</rss>"
    assert [e.country for e in safety.advisory_entries(feed)] == ["kyrgyz republic"]
    monkeypatch.setattr(safety, "advisories", lambda: {"the kyrgyz republic": 1})   # advisories() keeps "the"
    assert not safety.verdict_for(safety.country_names("Kyrgyzstan"), "Issyk-Kul").unsafe


def test_verdict_for_matches_check_without_an_llm(monkeypatch):
    monkeypatch.setattr(safety, "advisories", lambda: {"india": 2, "burma": 4, "vietnam": 1})
    assert not safety.verdict_for(safety.country_names("India"), "Karnataka").unsafe
    assert safety.verdict_for(safety.country_names("India"), "Jammu and Kashmir").unsafe
    assert safety.verdict_for(safety.country_names("India"), "").unsafe          # region unknown: fail closed
    assert safety.verdict_for(safety.country_names("Myanmar"), "Mandalay").level == 4
    assert safety.verdict_for(["atlantis"], "").unsafe                            # not in the feed: fail closed
    assert not safety.verdict_for([], "").unsafe                                   # not a destination
