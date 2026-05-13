"""Real-chat extractor — turns Claude Code JSONL session logs into a
session corpus suitable for longmem-shaped eval items.

Source: ~/.claude/projects/-Users-tpac-brain/*.jsonl (one file per session).
Output: eval/longmem/data/realchat_sessions.json — list of clean sessions:

    [
      {
        "session_id": "26a2f595-...",
        "date": "2026-05-09",
        "turn_count": 42,
        "exchanges": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
          ...
        ],
        "file_path": "/Users/tpac/.claude/projects/.../26a2f595-....jsonl"
      },
      ...
    ]

Cleaning rules (so haystacks match the shape longmem expects):
  - Drop non-message types (system, queue-operation, attachment, last-prompt).
  - Drop sidechain entries (isSidechain=True) — those are subagent threads.
  - Strip <system-reminder>...</system-reminder> blocks from user content
    (these are harness injections, not Tom's words).
  - Strip [BRAIN]...[/BRAIN] blocks from user content (these are recalled
    memories injected by hooks, not Tom's words).
  - Drop assistant `thinking` blocks (internal reasoning, not the response).
  - Drop assistant `tool_use` / `tool_result` blocks (verbose, not part of
    the conversation flow we want as haystack).
  - Keep only `text` blocks for assistant content.
  - Skip messages whose final cleaned content is empty/whitespace.

Usage:
    ./dev python3 eval/longmem/realchat_extractor.py
    ./dev python3 eval/longmem/realchat_extractor.py --min-turns 6 --max-turns 200
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_PROJECTS_DIR = os.path.expanduser("~/.claude/projects/-Users-tpac-brain")
DEFAULT_OUT = ROOT / "eval/longmem/data/realchat_sessions.json"

# Match harness-injected blocks that aren't part of the user's actual prompt.
RE_SYSTEM_REMINDER = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL)
RE_BRAIN_BLOCK = re.compile(r"\[BRAIN\].*?\[/BRAIN\]", re.DOTALL)
RE_COMMAND_TAG = re.compile(r"<command-(?:name|message|args)>.*?</command-(?:name|message|args)>",
                            re.DOTALL)


def _clean_user_text(text: str) -> str:
    """Strip injected blocks; return Tom's actual words (or empty)."""
    text = RE_SYSTEM_REMINDER.sub("", text)
    text = RE_BRAIN_BLOCK.sub("", text)
    text = RE_COMMAND_TAG.sub("", text)
    return text.strip()


def _user_content(message: Dict[str, Any]) -> str:
    """Extract Tom's actual prompt text from a user message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return _clean_user_text(content)
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                cleaned = _clean_user_text(c.get("text", "") or "")
                if cleaned:
                    parts.append(cleaned)
        return "\n\n".join(parts).strip()
    return ""


def _assistant_content(message: Dict[str, Any]) -> str:
    """Extract Anchor's response text from an assistant message.

    Only `text` blocks; skip thinking + tool_use to keep the haystack clean.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                t = (c.get("text") or "").strip()
                if t:
                    parts.append(t)
        return "\n\n".join(parts).strip()
    return ""


def _date_from_timestamp(ts: str) -> Optional[str]:
    """Return YYYY-MM-DD from an ISO-8601 timestamp, or None on parse fail."""
    if not ts or len(ts) < 10:
        return None
    return ts[:10]


def extract_session(path: str, min_turns: int = 4, max_turns: int = 200
                    ) -> Optional[Dict[str, Any]]:
    """Read one JSONL file and return a clean session dict, or None if
    the session has too few/too many turns or is unreadable.
    """
    try:
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except Exception:
        return None

    exchanges = []
    earliest_ts = None
    session_id = None
    for entry in lines:
        if entry.get("isSidechain"):
            continue
        t = entry.get("type")
        if t not in ("user", "assistant"):
            continue
        sid = entry.get("sessionId")
        if sid and not session_id:
            session_id = sid
        ts = entry.get("timestamp", "")
        if ts and (earliest_ts is None or ts < earliest_ts):
            earliest_ts = ts

        msg = entry.get("message") or {}
        if t == "user":
            text = _user_content(msg)
            if not text:
                continue  # tool-result-only or fully stripped
            # Collapse consecutive same-role messages (rare but happens)
            if exchanges and exchanges[-1]["role"] == "user":
                exchanges[-1]["content"] += "\n\n" + text
            else:
                exchanges.append({"role": "user", "content": text})
        else:  # assistant
            text = _assistant_content(msg)
            if not text:
                continue
            if exchanges and exchanges[-1]["role"] == "assistant":
                exchanges[-1]["content"] += "\n\n" + text
            else:
                exchanges.append({"role": "assistant", "content": text})

    if len(exchanges) < min_turns:
        return None
    if len(exchanges) > max_turns:
        # Truncate from the start to keep the most recent (where context is densest)
        exchanges = exchanges[-max_turns:]

    return {
        "session_id": session_id or os.path.basename(path).replace(".jsonl", ""),
        "date": _date_from_timestamp(earliest_ts or "") or "unknown",
        "turn_count": len(exchanges),
        "exchanges": exchanges,
        "file_path": path,
    }


def extract_all(projects_dir: str, min_turns: int = 4,
                max_turns: int = 200) -> List[Dict[str, Any]]:
    """Walk projects_dir, extract every session that meets length bounds."""
    out = []
    for entry in sorted(os.listdir(projects_dir)):
        if not entry.endswith(".jsonl"):
            continue
        path = os.path.join(projects_dir, entry)
        session = extract_session(path, min_turns=min_turns, max_turns=max_turns)
        if session:
            out.append(session)
    # Sort newest first — easier to find recent material
    out.sort(key=lambda s: s.get("date", ""), reverse=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-dir", default=DEFAULT_PROJECTS_DIR)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-turns", type=int, default=6,
                        help="skip sessions shorter than this (default 6)")
    parser.add_argument("--max-turns", type=int, default=200,
                        help="truncate sessions longer than this (default 200)")
    args = parser.parse_args()

    if not os.path.isdir(args.projects_dir):
        print(f"[extractor] missing projects dir: {args.projects_dir}", file=sys.stderr)
        sys.exit(1)

    sessions = extract_all(args.projects_dir, min_turns=args.min_turns,
                           max_turns=args.max_turns)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sessions, f, indent=2)

    # Summary — visual smoke test of what we got
    print(f"[extractor] {len(sessions)} sessions written to {args.out}")
    if sessions:
        from collections import Counter
        by_date = Counter(s["date"] for s in sessions)
        print(f"[extractor] date distribution (top 10):")
        for d, c in by_date.most_common(10):
            print(f"    {d}: {c} sessions")
        turn_counts = [s["turn_count"] for s in sessions]
        print(f"[extractor] turn-count: min={min(turn_counts)} "
              f"med={sorted(turn_counts)[len(turn_counts)//2]} max={max(turn_counts)}")


if __name__ == "__main__":
    main()
