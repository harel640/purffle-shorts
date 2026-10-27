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

from .utils import clean_narration, emphasis_key, redact

log = logging.getLogger("purffle")

SYSTEM = "You are a careful fact-checker for short videos. You reply only with strict JSON."


def _prune_emphasis(scene) -> None:
    """Drop highlighted words that no longer appear in the (rewritten) narration."""
    present = {emphasis_key(w) for w in scene.narration.split()}
    scene.emphasis = [w for w in scene.emphasis if emphasis_key(w) in present]


def scrub(llm, script, settings):
    """Rewrite any line with unsupported specifics. Mutates and returns the same Script."""
    lines = [{"id": i, "text": sc.narration} for i, sc in enumerate(script.scenes)]
    payload = {"title": script.title, "hook": script.hook_text, "lines": lines}
    user = (
        "Check the title, hook and each line below. If one states a specific statistic, percentage, count, "
        "date, ranking or superlative that is NOT well-established common knowledge and that you cannot "
        "confidently support, rewrite it to remove the unsupported specifics while keeping it engaging and "
        'TRUE (e.g. "85% of X" -> "most X", "2,300 temples" -> "thousands of temples", "the oldest" -> '
        '"one of the oldest"). NEVER invent a new number. Leave well-supported text unchanged.\n'
        'Return STRICT JSON {"title": "<kept or fixed>", "hook": "<kept or fixed>", '
        '"lines": [{"id": <int>, "text": "<kept or fixed>"}], "flagged": ["<what you changed and why>"]}.\n'
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

    flagged = [str(f) for f in (data.get("flagged") or [])][:8]
    if changes:
        log.info("Fact check: rewrote %d item(s) to drop unsupported stats", changes)
        for f in flagged:
            log.info("  flagged: %s", f[:160])
    else:
        log.info("Fact check: no unsupported stats found")
    return script
