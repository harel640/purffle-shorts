"""Vigan (Nov 12): a seal from Commons was used as a scene, and an AI scene drew a parchment map."""

import pytest

from purffle_shorts import media as M
from purffle_shorts.llm import LLM


@pytest.mark.parametrize("title", [
    "Seal of Vigan, Ilocos Sur (1991-2001)", "Coat of arms of Dunhuang", "Coat-of-arms of Bhutan",
    "Flag of the Philippines", "Vigan city logo", "Emblem of Brunei", "Insignia of the province",
    "Heraldry of Gansu", "Municipal badge", "Wordmark of the museum",
])
def test_graphics_are_never_footage(title):
    assert M._SKIP_IMAGE_RE.search(title)


@pytest.mark.parametrize("title", [
    "Prayer flags over Phobjikha valley", "Crest of the ridge at Sigiriya", "Sealed cave entrance, Mogao",
    "Calle Crisologo at dusk", "Elpidio R. Quirino Monument, Vigan", "Plaza Salcedo and St. Pauls Cathedral",
])
def test_real_photos_with_similar_words_are_kept(title):
    assert not M._SKIP_IMAGE_RE.search(title)


def test_a_commons_seal_is_dropped_before_download():
    pg = {"title": "File:Seal of Vigan, Ilocos Sur (1991-2001).png", "pageid": 1,
          "imageinfo": [{"width": 2000, "height": 2000, "url": "https://upload.wikimedia.org/x.png",
                         "extmetadata": {"LicenseShortName": {"value": "Public domain"}}}]}
    assert M._wiki_item(pg, 0, 800) is None
    pg["title"] = "File:Vigan Ilocos Sur Philippines 6.jpg"
    assert M._wiki_item(pg, 0, 800) is not None


def test_the_vision_check_asks_for_a_photograph_of_the_place():
    q = LLM._vision_question(None, "Vigan, Philippines")          # needs no configured provider
    assert "real photograph of a place" in q and "seal" in q and "logo" in q and "coat of arms" in q
    assert "flag" in q and q.count("Vigan, Philippines") == 2 and q.endswith('Answer only "yes" or "no".')


@pytest.mark.parametrize("prompt", [
    "A royal decree scroll being handed to Juan de Salcedo, with Ilocos region shown on a parchment.",
    "Settlers studying a town plan",
    "A charter sealed with wax",
    "Priests reading a chart of the river",
])
def test_ai_prompts_lose_every_clause_about_scrolls_parchment_charts_and_plans(prompt):
    out = M.period_prompt(prompt)
    for word in ("scroll", "parchment", "chart", "plan", "decree", "charter"):
        assert word not in out.lower()


def test_period_scenes_without_documents_are_kept():
    p = "Augustinian friars in simple robes guiding workers building stone houses in Vigan."
    assert M.period_prompt(p) == p.rstrip(".")
