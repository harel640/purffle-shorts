"""Week 2: the recent-violence rule without years (Fuvahmulah), and the Mogao Caves review - a story, not an
encyclopedia list; enough pictures; real photos for the modern era; the article's own look; captions that
keep a phrase together."""

import pytest

from purffle_shorts import factcheck as F
from purffle_shorts import media as M
from purffle_shorts import pipeline as P
from purffle_shorts.config import Settings
from purffle_shorts.script import (Scene, Script, build_prompt, encyclopedia_phrases, lint_script,
                                   recent_violence_reason, split_scenes)
from purffle_shorts.timing import Word, caption_chunks


def _script(*lines, title="A title", hook="A hook", style="history"):
    return Script(topic="t", title=title, hook_text=hook, description="d", style=style,
                  scenes=[Scene(x, "q") for x in lines])


# ------------------------------------------------------------------ 1. violence with no year in the script
# The script that was uploaded on Nov 6 (video Yte0bDoTH_w): no year anywhere, so the old rule let it through.
FUVAHMULAH = _script(
    "Why did soldiers fire on Fuvahmulah islanders?",
    "Villagers gathered at Rasgefanno, hurling stones as soldiers stepped onto the sand.",
    "The standoff turned violent; soldiers raised rifles and opened fire, resulting in deaths and injuries.",
    "Medics from nearby Gan rushed the wounded onto a dōni and flew them to Ceylon for care.",
    "Today Rasgefanno lies quiet, its sand untouched by the past turmoil.",
    "The tragedy shows calm seas can hide fierce histories.",
    title="Why did soldiers fire on Fuvahmulah islanders?", hook="A protest turned deadly")

# A paraphrase of how the island's Wikipedia article tells it: the year is only in the paragraph's opening.
FUVAHMULAH_ARTICLE = (
    "History\n"
    "The island was part of the United Suvadive Republic, a breakaway state, from 1959 to 1963. Little "
    "happened on the island in those years apart from one incident at Rasgefanno beach. A boat carrying "
    "government soldiers tried to land there. Islanders gathered on the beach and threw stones at the soldiers "
    "in protest. The soldiers opened fire on the islanders, resulting in the death of one islander.\n\n"
    "Recent history\nA fire burned down the community centre in 2007.")


def test_the_fuvahmulah_story_is_blocked_without_its_article():
    why = recent_violence_reason(FUVAHMULAH)
    assert why.startswith("violence or deaths not shown to be old history") and "nothing dates it" in why


def test_the_fuvahmulah_story_is_dated_by_its_article():
    why = recent_violence_reason(FUVAHMULAH, FUVAHMULAH_ARTICLE)
    assert why.startswith("recent violence or deaths") and "1959" in why


@pytest.mark.parametrize("lines, article", [
    (("In 1221 the Mongols destroyed the city.", "Its people were killed or carried off."), ""),
    (("King Kashyapa killed his father and seized the throne.", "He built a palace on the rock in the 5th century."),
     ""),
    (("Kashyapa killed his father and took the throne.", "Then he moved his court to the rock."),
     "In 477 CE Kashyapa, son of King Dhatusena, killed his father and seized the throne."),
    (("Legend says an army was slaughtered at the pass.",), ""),
    (("The siege lasted eight months.", "Today pilgrims climb 600 steps."), ""),     # no violence word at all
])
def test_old_history_is_still_fine(lines, article):
    assert recent_violence_reason(_script(*lines), article) == ""


def test_undated_violence_in_a_modern_script_is_blocked():
    s = _script("In 1965 the town got its first road.", "Soon after, soldiers opened fire on a crowd.")
    assert "dated 1900 or later" in recent_violence_reason(s)


def test_a_year_in_the_sentence_still_blocks_as_before():
    assert "1989" in recent_violence_reason(_script("Protesters were shot here in 1989."))


# ------------------------------------------------------------------ 3a. the writer reads the human thread
def _article():
    lead = "The Caves are a system of 500 temples near an oasis town on an old trade route in a desert."
    filler = [f"The caves of group {i} are decorated with painted ceilings and niches of many colours, in "
              f"styles that vary from one period to the next and from one cave to another one." * 3
              for i in range(30)]
    thread = ("In 1900 the priest Wang Yuanlu discovered a sealed cave behind a wall. He reported it to the "
              "officials, who ordered it sealed again. In 1907 the explorer Stein persuaded Wang to sell "
              "thousands of scrolls, and took them to London.")
    return "\n".join([lead] + filler[:20] + [thread] + filler[20:])


def test_the_writer_reads_the_opening_and_the_human_thread():
    src = F.story_source(_article(), 3000)
    assert len(src) <= 3000
    assert src.startswith("The Caves are a system")
    assert "Wang Yuanlu discovered" in src and "persuaded Wang" in src
    assert F.story_source("short article", 3000) == "short article"


def test_an_article_without_a_thread_is_read_in_order():
    text = "\n".join(f"Paragraph {i} describes the painted ceilings and niches in some detail." * 4 for i in range(80))
    src = F.story_source(text, 2000)
    assert src.startswith("Paragraph 0") and len(src) > 1500


def test_the_history_prompt_asks_for_a_story_shape():
    _, user = build_prompt("Mogao Caves", "history", Settings(), [], context="background")
    assert "SETUP" in user and "TURNING POINT" in user and "PAYOFF" in user
    assert "encyclopedia" in user and "the site contains" in user and "At least 6 scenes" in user
    assert "exactly what the background calls them" in user


def test_encyclopedia_phrasing_is_found():
    text = ("The site contains some of the finest art spanning a thousand years. Overall, the complex comprises "
            "about 500 temples. In 1900 a priest broke open a wall.")
    assert encyclopedia_phrases(text) == ["the site contains", "spanning", "overall", "the complex comprises"]
    assert encyclopedia_phrases("In 1900 Wang broke open the sealed wall.") == []
    issues = lint_script(_script(text), Settings())
    assert any("encyclopedia" in i for i in issues)


class _HookLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete_json(self, *a, **k):
        return self.reply


def test_the_hook_rewrite_cannot_invent_a_number():
    from purffle_shorts.script import ensure_concrete_hook
    s = _script("A hidden library waited in the desert.", "In 1900 Wang Yuanlu found thousands of scrolls.",
                title="The hidden library", hook="A HIDDEN LIBRARY")
    out = ensure_concrete_hook(_HookLLM({"title": "1900 Discovery Saved 2,000 Manuscripts",
                                         "hook_text": "Thousands of scrolls found in 1900",
                                         "opening": "In 1900 one man found 2,000 scrolls."}), s)
    assert out.title == "The hidden library"                          # 2,000 is not in the narration
    assert out.hook_text == "Thousands of scrolls found in 1900"      # 1900 is
    assert out.scenes[0].narration == "A hidden library waited in the desert."


# ------------------------------------------------------------------ 3b. at least 6 pictures, none over 8 s
def test_long_scenes_are_split_at_sentence_ends():
    s = _script("One. Two two. Three three three. Four.", "Five five. Six.", "Seven seven seven. Eight.")
    s.scenes[0].historical, s.scenes[0].image_prompt = True, "a priest at a wall"
    out = split_scenes(s, [20.0, 9.0, 6.0], max_seconds=8.0, min_scenes=6)
    assert len(out.scenes) >= 6
    assert " ".join(sc.narration for sc in out.scenes) == " ".join(sc.narration for sc in s.scenes)
    assert out.scenes[0].historical and out.scenes[1].image_prompt == "a priest at a wall"


def test_a_one_sentence_scene_is_left_whole():
    s = _script("A single long sentence that runs on and on without a break")
    assert split_scenes(s, [12.0]) is s


@pytest.mark.parametrize("text, modern", [
    ("Scholars soon dispersed the collection to Beijing, London, Paris and Berlin.", True),
    ("Today the caves welcome a limited number of visitors.", True),
    ("In 1907 the explorer Stein took the scrolls away.", True),
    ("In 366 a monk carved the first cave into the cliff.", False),
    ("The cave was still sealed when the dynasty fell.", False),
])
def test_modern_era_scenes_use_real_photos(text, modern):
    assert P.modern_scene(text) is modern


# ------------------------------------------------------------------ 3c. the AI look comes from the article
def test_scholars_is_not_chola():
    assert "gopuram" not in M.look_from_text("Somewhere", "Scholars studied the scrolls.").get("architecture", "")


def test_mogao_gets_chinese_cave_temples():
    lead = ("The Mogao Caves, also known as the Thousand Buddha Grottoes, form a system of 500 temples near "
            "Dunhuang on the Silk Road. Scholars have studied them since.")
    look = M.look_from_text("Mogao Caves", lead)
    assert "cave temples" in look["architecture"] and "minaret" not in look["architecture"]


# ------------------------------------------------------------------ 3d. "366 CE after"
def _words(text):
    return [Word(t, i * 0.35, i * 0.35 + 0.3) for i, t in enumerate(text.split())]


def test_a_number_keeps_its_era_and_a_caption_never_ends_on_after():
    chunks = [c.text for c in caption_chunks(_words(
        "The first grotto was carved in 366 CE after monk Yuezun claimed a vision of a thousand Buddhas."))]
    assert "in 366 CE" in chunks and "after monk Yuezun" in chunks
    assert not any(c.split()[-1].lower() in ("after", "in", "of", "a") for c in chunks)
