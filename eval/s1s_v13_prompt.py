"""Extract v13 draft prompt from docs/S1S-PROMPT-REWRITE-DRAFT.md.

The draft markdown has a fenced code block containing the full prompt that
would be registered as `s1e` v13. This helper pulls out that text so the
A/B harness can register it into an isolated brain without copying the
prompt into Python source (single source of truth = the draft).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT_PATH = ROOT / "docs" / "S1S-PROMPT-REWRITE-DRAFT.md"


def extract_v13_prompt(draft_path: Path = DRAFT_PATH) -> str:
    """Return the v13 prompt text (fenced block after 'Prompt text')."""
    text = draft_path.read_text(encoding="utf-8")

    # Find the "# Prompt text" heading then grab the next fenced block.
    marker = re.search(r"^#\s+Prompt text.*$", text, flags=re.MULTILINE)
    if not marker:
        raise RuntimeError(
            f"no '# Prompt text' heading found in {draft_path}")
    tail = text[marker.end():]

    # The outer fence is a bare '```' on its own line. Inner fences inside
    # the prompt (remember_batch example) use '```json'. Find the first
    # bare '```\n' (opener, no language tag) and the LAST '```' line (closer)
    # after it — everything between is the prompt body, inclusive of any
    # inner ```json blocks.
    opener = re.search(r"^```\s*\n", tail, flags=re.MULTILINE)
    if not opener:
        raise RuntimeError(
            f"no opening '```' line after '# Prompt text' in {draft_path}")
    body_start = opener.end()

    # Find the last bare '```' line in the remainder of the file.
    closes = list(re.finditer(r"^```\s*$", tail[body_start:],
                              flags=re.MULTILINE))
    if not closes:
        raise RuntimeError(
            f"no closing '```' line after opener in {draft_path}")
    body_end = body_start + closes[-1].start()

    return tail[body_start:body_end].strip() + "\n"


if __name__ == "__main__":
    p = extract_v13_prompt()
    print(f"[v13] extracted {len(p)} chars, {p.count(chr(10))} lines")
    print("=" * 60)
    print(p[:600])
    print("...")
    print(p[-400:])
