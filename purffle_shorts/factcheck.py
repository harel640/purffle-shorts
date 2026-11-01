"""Fact-check pass.

After the script is written (and polished by the Script Doctor), a second LLM pass flags any
specific number, statistic, percentage, date, ranking or superlative it cannot support, and
rewrites those lines to keep them engaging and true WITHOUT invented stats
(e.g. "85% of temples" -> "most temples", "2,300 temples" -> "thousands of temples").
It never introduces a new number. Title, on-screen hook and every spoken line are checked.
Fail-open: on any LLM error the script is returned unchanged (with a warning).
"""

from __future__ import annotations

import json
import logging
import re

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
