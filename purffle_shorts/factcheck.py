"""Fact checks.

Three layers, strictest last:

``scrub`` - an LLM pass with NO source text: it rewrites numbers and superlatives it cannot support. Used only
when a script has no source article. Fail-open.

``ground`` - deterministic: every number, year and percentage in a story must appear in the source article;
a sentence carrying one that doesn't is dropped.

``verify_claims`` - claim-level: every sentence of the narration, the title, the on-screen hook and the
description line must be SUPPORTED by a verbatim quote from the source article. One strict-schema LLM call
proposes a verdict and a quote per sentence; the code then verifies the checker (the quote must really be in
the source, numbers and names must be there too). Unsupported sentences are removed. Fail-CLOSED: if the
checker cannot be asked, FactCheckUnavailable is raised and the caller skips the video.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from difflib import SequenceMatcher

from .utils import clean_narration, emphasis_key, redact

log = logging.getLogger("purffle")

SYSTEM = "You are a careful fact-checker for short videos. You reply only with strict JSON."

# A "hard" verifiable specific: a year, count, or percentage. These can be checked against a source text
# deterministically, so an invented one is caught even when the LLM fact-check is unavailable or lets it slip.
_FACT_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Thousands separators written as a space / thin space / non-breaking space, e.g. "1 500" -> "1500".
_DIGIT_SPACE = re.compile(r"(?<=\d)[    ](?=\d)")


def _collapse(text: str) -> str:
    return _DIGIT_SPACE.sub("", text)


def _supported(token: str, source_norm: str) -> bool:
    t = token.replace(",", "").rstrip("%").strip()
    return not t or t in source_norm


def ground(script, source_text: str, settings=None):
    """Deterministic grounding guardrail: a story's specific numbers, years and percentages must appear in the
    source article (``source_text``). Any that don't are invented — the sentence carrying one is dropped (and a
    scene left empty is removed). The prompt already tells the writer to ground every claim; this enforces it
    for the checkable facts. No-op when there is no source text. Mutates and returns the same Script."""
    if not source_text or not getattr(script, "scenes", None):
        return script
    source_norm = _collapse(source_text.replace(",", "").lower())

    def bad(text: str) -> list[str]:
        return [m.group(0) for m in _FACT_RE.finditer(_collapse(text))
                if not _supported(m.group(0).lower(), source_norm)]

    flagged: list[str] = []
    for attr in ("title", "hook_text"):                       # strip unsupported specifics from the metadata
        val = getattr(script, attr, "") or ""
        for tok in bad(val):
            val = re.sub(rf"\s*\b{re.escape(tok)}\b", "", val).strip(" ,-:–")
            flagged.append(f"{attr}: removed unsupported '{tok}'")
        setattr(script, attr, val)

    kept = []
    for sc in script.scenes:
        sents = _SENT_SPLIT.split(sc.narration.strip())
        keep = []
        for s in sents:
            b = bad(s)
            if b:
                flagged.append(f"removed unsupported claim ({', '.join(b)}): {s[:70]}")
                continue
            keep.append(s)
        new = clean_narration(" ".join(keep).strip())
        if new:
            if new != sc.narration:
                sc.narration = new
                _prune_emphasis(sc)
            kept.append(sc)
        else:
            flagged.append(f"dropped scene with only unsupported claims: {sc.narration[:70]}")
    if kept:
        script.scenes = kept
    if flagged:
        log.info("Grounding: dropped %d unsupported specific(s) not found in the source", len(flagged))
        for f in flagged[:8]:
            log.info("  %s", f[:160])
    return script


def _prune_emphasis(scene) -> None:
    """Drop highlighted words that no longer appear in the (rewritten) narration."""
    present = {emphasis_key(w) for w in scene.narration.split()}
    scene.emphasis = [w for w in scene.emphasis if emphasis_key(w) in present]


def scrub(llm, script, settings):
    """Rewrite any line with unsupported specifics. Mutates and returns the same Script."""
    lines = [{"id": i, "text": sc.narration} for i, sc in enumerate(script.scenes)]
    payload = {"title": script.title, "hook": script.hook_text, "description": script.description,
               "tags": list(script.tags), "hashtags": list(script.hashtags), "lines": lines}
    user = (
        "Check the title, hook, description, tags, hashtags and each line below. If any states a specific "
        "statistic, percentage, count, date, ranking or superlative that is NOT well-established common "
        "knowledge and that you cannot confidently support, rewrite it to remove the unsupported specifics "
        'while keeping it engaging and TRUE (e.g. "85% of X" -> "most X", "built in the 1500s" -> "built '
        'centuries ago", "the oldest" -> "one of the oldest"). NEVER invent a new number. Drop a tag or '
        "hashtag rather than keep a false one. Leave well-supported text unchanged.\n"
        'Return STRICT JSON {"title": "...", "hook": "...", "description": "...", "tags": ["..."], '
        '"hashtags": ["..."], "lines": [{"id": <int>, "text": "..."}], "flagged": ["<what changed and why>"]}.\n'
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        data = llm.complete_json(SYSTEM, user, max_tokens=5000, temperature=0.2)
    except Exception as e:
        log.warning("Fact check skipped (%s)", redact(e))
        return script
    if not isinstance(data, dict):
        log.warning("Fact check: unexpected reply; leaving script unchanged")
        return script

    changes = 0
    new_title = str(data.get("title", "")).strip()
    if new_title and new_title != script.title:
        script.title = new_title
        changes += 1
    new_hook = str(data.get("hook", "")).strip()
    if new_hook and new_hook != script.hook_text:
        script.hook_text = new_hook
        changes += 1

    by_id = {}
    for item in data.get("lines") or []:
        if isinstance(item, dict) and "id" in item:
            try:
                by_id[int(item["id"])] = str(item.get("text", "")).strip()
            except (TypeError, ValueError):
                continue
    for i, sc in enumerate(script.scenes):
        new = clean_narration(by_id.get(i, ""))
        if new and new != sc.narration:
            sc.narration = new
            _prune_emphasis(sc)
            changes += 1

    new_desc = str(data.get("description", "")).strip()
    if new_desc and new_desc != script.description:
        script.description = new_desc
        changes += 1
    for field in ("tags", "hashtags"):
        if isinstance(data.get(field), list):
            cleaned = [str(t).strip() for t in data[field] if str(t).strip()]
            if cleaned and cleaned != getattr(script, field):
                setattr(script, field, cleaned)
                changes += 1

    flagged = [str(f) for f in (data.get("flagged") or [])][:8]
    if changes:
        log.info("Fact check: rewrote %d item(s) to drop unsupported stats", changes)
        for f in flagged:
            log.info("  flagged: %s", f[:160])
    else:
        log.info("Fact check: no unsupported stats found")
    return script


# ---------------------------------------------------------------------------------- claim-level check
# The story writer once turned a traveller's remark ("these houses can be removed ... when there is a storm")
# into a storm night that swept a village downstream, and "carved" a chariot "from a single block" the article
# never mentions. Numbers were clean, so ground() passed both. verify_claims() checks every SENTENCE.
SOURCE_FETCH_CHARS = 24000        # how much of the article is fetched; quotes are verified against all of it
# How much of it the checker is shown. The free Groq tier allows 8,000 tokens a minute, and the whole request
# (article + script + instructions) has to fit inside that, so the article is cut to its most relevant passages.
# Counted in characters of plain Latin text (~4 per token); other characters cost more (see _cost).
CLAIM_SOURCE_CHARS = int(os.getenv("CLAIM_SOURCE_CHARS", "9000"))
CLAIM_REPLY_TOKENS = 3000         # max_tokens of the checker's reply (counts against the same 8,000)
LEAD_CHARS = 5000                 # the part of the article the writer saw: always shown to the checker
MIN_QUOTE_WORDS = 4               # a "quote" shorter than this proves nothing
QUOTE_RUN = 0.7                   # a quote with a slip still counts if this share of it is one verbatim run
VERDICTS = ("supported", "unsupported", "not_factual")

CLAIMS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"id": {"type": "integer"},
                       "verdict": {"type": "string", "enum": list(VERDICTS)},
                       "quote": {"type": "string"},
                       "unsupported_part": {"type": "string"}},
        "required": ["id", "verdict", "quote", "unsupported_part"]}}},
    "required": ["items"]}
HEADLINE_SCHEMA = {"type": "object", "additionalProperties": False,
                   "properties": {"title": {"type": "string"}, "hook_text": {"type": "string"}},
                   "required": ["title", "hook_text"]}

CLAIMS_SYSTEM = ("You are a strict fact-checker. You compare sentences with a SOURCE text and reply only with "
                 "strict JSON. You never use your own knowledge: only what the SOURCE states counts.")
_CLAIMS_RULES = (
    "For EACH numbered item decide ONE verdict:\n"
    "- supported: EVERY factual part of the item is stated in the SOURCE. Judge the FACTS, not the wording: "
    "a plain restatement in other words is supported (\"destroyed the city\" ~ \"wiped out the city\"; "
    "\"an enslaved Parisian goldsmith ... design the Silver Tree\" ~ \"a Parisian goldsmith made a silver "
    "tree\"), and so is a shorter summary that leaves details out. Copy into \"quote\" the passage of the "
    "SOURCE that states it, character for character, at most 30 words. If the facts sit in two different "
    "passages, copy both, separated by \" ... \".\n"
    "- unsupported: the item states an event, action, cause, date, person, quantity, material, method or "
    "superlative that the SOURCE does not state. Put the unsupported words in \"unsupported_part\". These are "
    "unsupported: turning something that CAN happen into something that DID happen; turning \"some\" or \"a "
    "few\" into \"many\" or \"thousands\"; turning a description into a dramatic event; adding weather, "
    "emotions, motives, materials or methods the SOURCE does not mention; anything true that you know from "
    "elsewhere but the SOURCE does not say.\n"
    "Words that only praise or decorate (beautiful, vibrant, breathtaking, famous, mighty) are not facts: "
    "ignore them and judge the rest of the item. A claim about people, numbers or events is never decoration "
    "(\"visitors from every continent\" is a claim).\n"
    "An item that describes the video (\"This short explores X\", \"A documentary about X\", \"Explore how X\") "
    "is judged by X: supported, with a quote, if the SOURCE states X; unsupported if it does not.\n"
    "- not_factual: the item asserts nothing checkable - a question to the viewer, direct address, or a "
    "transition (\"Here is what happened next.\").\n"
    "If only part of an item is supported, the verdict is unsupported. \"quote\" is empty unless the verdict is "
    "supported. Return every id exactly once.\n")

_STOP = frozenset("""a an and are as at be been but by can could did do does for from had has have he her his
how if in into is it its just like may more most much no not of on one only or our out over she so some such
than that the their them then there these they this those through to under up very was were what when where
which while who whom why will with would you your about after again all also any because before being between
both each ever every here once still upon today now even ever away back down around across along""".split())
# Capitalised words that are not names to be found in the source.
_NOT_NAMES = frozenset("""i earth asia asian europe european north south east west northern southern eastern
western god""".split())
_WORD_RE = re.compile(r"[^\W\d_][\w'’\-‑]*", re.UNICODE)
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\]|\[…\])\s*")


class FactCheckUnavailable(RuntimeError):
    """The claim checker could not be asked (every LLM provider failed, or its reply was unusable)."""


class NothingSupported(RuntimeError):
    """The story has no supported narration left, or its title/hook cannot be supported by the source.
    ``report`` holds the verdicts that led to it."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


# Names in their own script (Tibetan, Chinese, Devanagari, Arabic ...) and IPA sit in most articles about Asia.
# An English story never quotes them, and they cost a token or more per character: the Paro Taktsang article
# came to 9,367 tokens and Groq refused it. They are taken out of the text the checker reads and is checked
# against. Kept: Latin letters with their accents, digits, and ordinary punctuation.
_FOREIGN_RE = re.compile("[^\u0000-\u024F\u1E00-\u1EFF\u2010-\u2027\u2030-\u205E]+")
_EMPTY_BRACKETS_RE = re.compile(r"\(\s*[,;:/.\s]*\)|\[\s*[,;:/.\s]*\]")
_TOO_LARGE_RE = re.compile(r"\b413\b|too large|context length|maximum context|reduce your message", re.I)


def clean_source(text: str) -> str:
    """The article without non-Latin scripts and the empty brackets they leave behind."""
    t = _FOREIGN_RE.sub("", text or "")
    t = _EMPTY_BRACKETS_RE.sub("", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def _cost(text: str) -> int:
    """Rough size of ``text`` for the token budget, in characters of plain text: an accented or other
    non-ASCII character is counted as four."""
    plain = sum(c < "\u0080" for c in text)
    return plain + 4 * (len(text) - plain)


def _norm(text: str) -> str:
    """Lower-case, accents folded, punctuation as spaces, single-spaced: what 'the same words' means here."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[^\w\s]|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_in_source(quote: str, source_norm: str) -> bool:
    """True if ``quote`` really is a passage of the source: every piece of it (pieces are split at '...')
    appears there, and together they are at least MIN_QUOTE_WORDS long. A piece counts when it is word for
    word in the source, or - models miscopy a word now and then ("located on one of the most important
    medieval paths" for "situated on ...") - when QUOTE_RUN of its words form ONE unbroken run of the source,
    at least MIN_QUOTE_WORDS long. A made-up quote shares no such run with the article."""
    parts = [_norm(p) for p in _ELLIPSIS_RE.split(quote or "")]
    parts = [p for p in parts if p]
    if not parts or sum(len(p.split()) for p in parts) < MIN_QUOTE_WORDS:
        return False
    padded = f" {source_norm} "
    source_words: list[str] | None = None
    for p in parts:
        if f" {p} " in padded:
            continue
        words = p.split()
        if len(words) < MIN_QUOTE_WORDS:
            return False
        if source_words is None:
            source_words = source_norm.split()
        run = SequenceMatcher(None, source_words, words, autojunk=False).find_longest_match(
            0, len(source_words), 0, len(words)).size
        if run < max(MIN_QUOTE_WORDS, math.ceil(QUOTE_RUN * len(words))):
            return False
    return True


def _unsupported_numbers(text: str, source_digits: str) -> list[str]:
    return [m.group(0) for m in _FACT_RE.finditer(_collapse(text))
            if not _supported(m.group(0).lower(), source_digits)]


def _names(sentence: str, allow: frozenset[str]) -> list[str]:
    """Capitalised words that are names (people, peoples, places, dynasties): every capitalised word except the
    first of the sentence, common non-names, and the words of the place itself (``allow``, normalised)."""
    out: list[str] = []
    for k, m in enumerate(_WORD_RE.finditer(sentence)):
        w = m.group(0)
        if k == 0 or not w[0].isupper():
            continue
        key = _norm(re.sub(r"['’]s$", "", w))
        if key and key not in _NOT_NAMES and key not in allow and key not in out:
            out.append(key)
    return out


def _name_in_source(name: str, source_norm: str) -> bool:
    """A name counts as found when the source has a word starting the same way ("Mongols" ~ "Mongol",
    "Chinese" ~ "China", "Portuguese" ~ "Portugal")."""
    words = name.split()
    return all(re.search(rf"\b{re.escape(w[:max(4, len(w) - 4)])}", source_norm) for w in words)


def _missing_names(sentence: str, source_norm: str, allow: frozenset[str]) -> list[str]:
    return [n for n in _names(sentence, allow) if not _name_in_source(n, source_norm)]


def _content_words(text: str) -> list[str]:
    return [w for w in _norm(text).split() if len(w) > 3 and w not in _STOP]


def _overlap(sentence: str, quote: str) -> float | None:
    """Share of the sentence's content words that the quote also has (by 5-letter stem). Logged, not enforced:
    it shows how loosely a 'supported' sentence paraphrases its quote."""
    words = _content_words(sentence)
    if not words or not quote:
        return None
    stems = {w[:5] for w in _content_words(quote)}
    return round(sum(w[:5] in stems for w in words) / len(words), 2)


def select_source(source: str, script_text: str, budget: int = 0) -> str:
    """The part of the article shown to the checker. Short articles go whole. A long one keeps its lead (what
    the writer saw) plus the later paragraphs that share the most words with the script, in article order."""
    budget = budget or CLAIM_SOURCE_CHARS
    source = (source or "").strip()
    if _cost(source) <= budget:
        return source
    paras = [p for p in re.split(r"\n+", source) if p.strip()]
    wanted = set(_content_words(script_text))
    keep: set[int] = set()
    used = 0
    for i, p in enumerate(paras):                      # the lead, in order
        if used + _cost(p) > min(LEAD_CHARS, budget):
            break
        keep.add(i)
        used += _cost(p) + 1
    rest = [i for i in range(len(paras)) if i not in keep]
    rest.sort(key=lambda i: -(len(wanted & set(_content_words(paras[i]))) / (1 + len(paras[i]) / 600)))
    for i in rest:
        if used + _cost(paras[i]) + 1 <= budget:
            keep.add(i)
            used += _cost(paras[i]) + 1
    return "\n".join(paras[i] for i in sorted(keep)) or source[:budget // 4]


def claim_items(script, kinds: tuple[str, ...] = ("title", "hook", "description", "narration")) -> list[dict]:
    """Everything the viewer is told, as numbered items: title, on-screen hook, the description's first
    sentence, and each sentence of the narration (with its scene and position)."""
    items: list[dict] = []

    def add(kind: str, text: str, **extra) -> None:
        text = (text or "").strip()
        if text and kind in kinds:
            items.append({"id": len(items), "kind": kind, "text": text, **extra})

    add("title", script.title)
    add("hook", script.hook_text)
    add("description", _SENT_SPLIT.split((script.description or "").strip(), maxsplit=1)[0])
    for si, sc in enumerate(script.scenes):
        for k, sent in enumerate(_SENT_SPLIT.split((sc.narration or "").strip())):
            add("narration", sent, scene=si, pos=k)
    return items


_REQUOTE_NOTE = ("NOTE: you called these items supported, but the quote you gave is not word for word in the "
                 "SOURCE (do not repeat the item itself). Copy the supporting words exactly as they stand in the "
                 "SOURCE - two pieces separated by \" ... \" if needed - or answer unsupported.\n")
_BAD_QUOTE = "the checker's quote is not in the source"


def _ask(llm, items: list[dict], shown: str, note: str = "") -> dict[int, dict]:
    user = ("SOURCE (the only evidence that counts):\n<<<\n" + shown + "\n>>>\n\n" + _CLAIMS_RULES + note
            + 'Reply with JSON {"items": [{"id": <int>, "verdict": "supported|unsupported|not_factual", '
              '"quote": "...", "unsupported_part": "..."}]}.\n\nITEMS:\n'
            + json.dumps([{"id": it["id"], "text": it["text"]} for it in items], ensure_ascii=False))
    try:
        # Groq's free tier refuses a request whose prompt + max_tokens exceeds 8,000 (HTTP 413), so the reply
        # budget is kept small - and the checker is asked for short reasoning (effort="low"): with the default
        # it spent more than 3,500 hidden reasoning tokens and ran out before answering; with "low" it used
        # 634 and gave the same verdicts.
        data = llm.complete_json(CLAIMS_SYSTEM, user, CLAIMS_SCHEMA, max_tokens=CLAIM_REPLY_TOKENS, temperature=0.0,
                                 effort="low")
    except Exception as e:  # noqa: BLE001
        raise FactCheckUnavailable(f"the claim checker could not be asked ({redact(e)})") from e
    rows = data.get("items") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise FactCheckUnavailable("the claim checker's reply had no verdicts")
    out: dict[int, dict] = {}
    for r in rows:
        if isinstance(r, dict):
            try:
                out[int(r.get("id"))] = r
            except (TypeError, ValueError):
                continue
    return out


def _judge(item: dict, reply: dict | None, source_norm: str, source_digits: str, allow: frozenset[str]) -> dict:
    """The final verdict for one item: the checker's answer, then verified - the checker is not trusted."""
    text, kind = item["text"], item["kind"]
    said = str((reply or {}).get("verdict", "")).strip().lower()
    quote = str((reply or {}).get("quote", "")).strip()
    row = {**item, "llm_verdict": said or "missing", "quote": quote, "verdict": "unsupported", "reason": "",
           "overlap": None}
    sentence = kind in ("narration", "description")       # titles and hooks are title-cased: names not checked
    numbers = _unsupported_numbers(text, source_digits)
    if said not in VERDICTS:
        row["reason"] = "the checker gave no verdict for it"
    elif said == "unsupported":
        row["reason"] = str((reply or {}).get("unsupported_part", "")).strip() or "not stated in the source"
    elif said == "supported":
        missing = _missing_names(text, source_norm, allow) if sentence else []
        if not quote_in_source(quote, source_norm):
            row["reason"] = f"{_BAD_QUOTE}: \"{quote[:90]}\""
        elif numbers:
            row["reason"] = f"number not in the source: {', '.join(numbers)}"
        elif missing:
            row["reason"] = f"name not in the source: {', '.join(missing)}"
        else:
            row["verdict"], row["overlap"] = "supported", _overlap(text, quote)
    else:                                                  # not_factual: allowed only if it really asserts nothing
        names = _names(text, allow) if sentence else []
        if numbers:
            row["reason"] = f"called not factual but states a number not in the source: {', '.join(numbers)}"
        elif text.rstrip().endswith("?"):                  # a question may name what the source names
            missing = [n for n in names if not _name_in_source(n, source_norm)]
            if missing:
                row["reason"] = f"name not in the source: {', '.join(missing)}"
            else:
                row["verdict"] = "not_factual"
        elif names or re.search(r"\d", text):
            row["reason"] = "called not factual but states names or numbers without a quote"
        else:
            row["verdict"] = "not_factual"
    return row


def _rewrite_headline(llm, script, quotes: list[str], which: list[str], title_len: int) -> dict:
    """ONE rewrite of an unsupported title / on-screen hook from the supported quotes only."""
    user = (
        "The " + " and the ".join("on-screen hook_text" if w == "hook" else w for w in which) + " of this "
        "YouTube Short state something the source article does not. Rewrite so each states ONLY what the QUOTES "
        "below state - concrete stakes (who or what, and what was won or lost). Use a number, date or "
        "superlative only if a quote contains it. Never vague mystery words (mystery, secret, hidden, legend, "
        "forgotten, untold). Do NOT add any fact that is not in the quotes. "
        f"title: max {title_len} characters, no place name, no hashtags, no quotes. hook_text: max 6 words. "
        "Return both, unchanged if one of them was not named above.\n\n"
        f"Current title: {script.title}\nCurrent hook_text: {script.hook_text}\n\nQUOTES from the source:\n"
        + "\n".join(f"- {q}" for q in quotes[:14]))
    d = llm.complete_json("You are a careful YouTube Shorts editor. Reply with strict JSON.", user,
                          HEADLINE_SCHEMA, max_tokens=2000, temperature=0.2, effort="low")
    return d if isinstance(d, dict) else {}


def _tidy(text: str, limit: int) -> str:
    text = re.sub(r"[<>\"“”]", "", text or "").strip(" '-:")
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")


def accepted(reports: list[dict]) -> dict[str, dict]:
    """{sentence: its row} for everything earlier checks accepted - handed to the next check as ``known`` so a
    sentence that has not changed is not judged twice (the checker does not always repeat itself)."""
    return {r["text"]: r for rep in reports or [] for r in rep.get("items", [])
            if r.get("verdict") in ("supported", "not_factual")}


def verify_claims(llm, script, source_text: str, *, place_words: str = "",
                  kinds: tuple[str, ...] = ("title", "hook", "description", "narration"),
                  title_len: int = 60, strict: bool = True, known: dict[str, dict] | None = None):
    """Claim-level fact check against the source article. Returns ``(script, report)``; mutates the script.

    Every unsupported narration sentence is removed (a scene left empty goes with it). An unsupported
    description line is cleared (the pipeline then writes a plain one). An unsupported title or hook gets ONE
    rewrite from the supported quotes and is checked again.

    Raises FactCheckUnavailable when the checker cannot be asked, and - when ``strict`` - NothingSupported when
    no narration survives or the title/hook still cannot be supported. With ``strict=False`` (the user chose
    the topic) an unsupportable title/hook is kept with a warning."""
    source = clean_source(source_text)
    if not source:
        raise FactCheckUnavailable("no source article to check the story against")
    items = claim_items(script, kinds)
    if not items:
        raise NothingSupported("the script has nothing to check")
    source_norm = _norm(source)
    source_digits = _collapse(source.replace(",", "").lower())
    allow = frozenset(_norm(" ".join([place_words, script.topic or ""])).split())
    script_text = " ".join(it["text"] for it in items)
    shown = select_source(source, script_text)

    def judge(batch: list[dict]) -> list[dict]:
        nonlocal shown
        # Items an earlier check accepted, word for word, keep that verdict: only what is new is asked about.
        carried = {it["id"]: {**(known or {})[it["text"]], **it, "carried": True}
                   for it in batch if it["text"] in (known or {})}
        ask = [it for it in batch if it["id"] not in carried]
        if not ask:
            return [carried[it["id"]] for it in batch]
        fresh = judge_new(ask)
        by_id = {r["id"]: r for r in fresh}
        return [carried.get(it["id"]) or by_id[it["id"]] for it in batch]

    def judge_new(batch: list[dict]) -> list[dict]:
        nonlocal shown
        try:
            replies = _ask(llm, batch, shown)
        except FactCheckUnavailable as e:
            if not _TOO_LARGE_RE.search(str(e)) or _cost(shown) < 3000:
                raise
            # The provider refused the request as too big: ask once more with half as much of the article.
            shown = select_source(source, script_text, max(2500, _cost(shown) // 2))
            log.info("Claim check: the request was too large; retrying with %d chars of the article", len(shown))
            replies = _ask(llm, batch, shown)
        judged = [_judge(it, replies.get(it["id"]), source_norm, source_digits, allow) for it in batch]
        # Second chance, once: the checker said "supported" but its quote is not in the article - often it
        # echoed the sentence instead of quoting. That is the checker's slip, not the writer's, and it once
        # cost a true opening line. It is asked again for those items only; a second miss stands.
        misquoted = [r for r in judged if r["llm_verdict"] == "supported" and r["reason"].startswith(_BAD_QUOTE)]
        if misquoted:
            again = [{k: r[k] for k in ("id", "kind", "text", "scene", "pos") if k in r} for r in misquoted]
            try:
                replies = _ask(llm, again, shown, _REQUOTE_NOTE)
            except FactCheckUnavailable as e:
                log.info("Claim check: could not ask again about %d misquoted item(s) (%s)", len(again), e)
            else:
                second = {it["id"]: _judge(it, replies.get(it["id"]), source_norm, source_digits, allow)
                          for it in again}
                judged = [{**second[r["id"]], "requoted": True} if r["id"] in second else r for r in judged]
        return judged

    rows = judge(items)

    def report() -> dict:
        """The verdicts so far, logged: one summary line and one line per dropped item."""
        counts = {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS}
        overlaps = [r["overlap"] for r in rows if r["overlap"] is not None]
        log.info("Claim check: %d supported, %d dropped, %d not factual", counts["supported"],
                 counts["unsupported"], counts["not_factual"])
        for r in rows:
            if r["verdict"] == "unsupported":
                log.info("  dropped (%s): %s -- %s", r["kind"], r["text"][:90], r["reason"][:80])
        return {"source_chars": len(source), "shown_chars": len(shown), "counts": counts,
                "mean_overlap": round(sum(overlaps) / len(overlaps), 2) if overlaps else None, "items": rows}

    by_kind = {k: [r for r in rows if r["kind"] == k] for k in ("title", "hook", "description", "narration")}

    # -- narration: drop every unsupported sentence
    bad = {(r["scene"], r["pos"]) for r in by_kind["narration"] if r["verdict"] == "unsupported"}
    if bad:
        kept = []
        for si, sc in enumerate(script.scenes):
            sents = [s for s in _SENT_SPLIT.split((sc.narration or "").strip()) if s.strip()]
            keep = [s for k, s in enumerate(sents) if (si, k) not in bad]
            new = clean_narration(" ".join(keep).strip())
            if new:
                if new != sc.narration:
                    sc.narration = new
                    _prune_emphasis(sc)
                kept.append(sc)
        script.scenes = kept
    if "narration" in kinds and not script.scenes:
        raise NothingSupported("no sentence of the narration is supported by the source", report())

    # -- description line: a statement about the video, so it must be SUPPORTED (a checker that waves it
    # through as "not factual" is not believed). Otherwise it is cleared and the pipeline writes
    # "The story of <place>, <country>."
    for r in by_kind["description"]:
        if r["verdict"] != "supported":
            if r["verdict"] == "not_factual":
                r["verdict"], r["reason"] = "unsupported", "a description must be supported by a quote"
            script.description = ""

    # -- title / hook: one rewrite from the supported quotes, then checked again
    which = [k for k in ("title", "hook") if any(r["verdict"] == "unsupported" for r in by_kind[k])]
    if which:
        quotes = list(dict.fromkeys(r["quote"] for r in rows if r["verdict"] == "supported" and r["quote"]))
        fixed = False
        if not quotes:
            log.info("Claim check: nothing in the story is supported, so the %s cannot be rewritten",
                     " and ".join(which))
        if quotes:
            try:
                new = _rewrite_headline(llm, script, quotes, which, title_len)
            except Exception as e:  # noqa: BLE001
                log.warning("Claim check: title/hook rewrite failed (%s)", redact(e))
                new = {}
            cand = {"title": _tidy(str(new.get("title", "")), 60), "hook": _tidy(str(new.get("hook_text", "")), 48)}
            if all(cand[k] for k in which):
                again = [{"id": i, "kind": k, "text": cand[k]} for i, k in enumerate(which)]
                rechecked = judge(again)
                if all(r["verdict"] != "unsupported" for r in rechecked):
                    for r in rechecked:
                        setattr(script, "title" if r["kind"] == "title" else "hook_text", r["text"])
                        r["id"], r["rewritten"] = len(rows), True
                        rows.append(r)
                    fixed = True
        if not fixed:
            what = " and ".join("on-screen hook" if k == "hook" else k for k in which)
            if strict:
                raise NothingSupported(f"the {what} could not be supported by the source", report())
            log.warning("Claim check: the %s is not supported by the source; kept (the topic was chosen by hand)",
                        what)

    return script, report()
