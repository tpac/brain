#!/usr/bin/env python3
"""
tmemory v14 — Post-compaction session log extractor.

Called at SessionStart. Uses DENSITY-BASED gap detection: chunks time into
30-minute windows, compares user messages in the log vs nodes created in the
brain for each window. Sparse windows get extracted, well-covered windows
get skipped.

This is how a human thinks about it: "I was in a 4-hour sprint but only
remember 2 things from it... something's missing." The density check finds
exactly WHERE the brain went blind.

Design principles:
  - Density, not timestamp: don't trust "last node created" alone
  - Both voices: user messages AND Claude's reasoning matter (shared brain)
  - Code narrative: tool calls get one-line summaries, not full diffs
  - Proportional soft cap: larger gaps get more space, compressed for older
  - Surgical: only extract the sparse windows, skip well-covered ones
"""

import json
import sys
import os
import glob
from datetime import datetime, timezone, timedelta

# ── Configuration ──
WINDOW_MINUTES = 30          # Density check window size
SPARSE_THRESHOLD = 0.15      # Nodes/user_msgs ratio below this = sparse
MIN_USER_MSGS = 3            # Windows with fewer user msgs are skipped (too little to judge)
SOFT_CAP_CHARS = 15000       # Target max output size
HARD_CAP_CHARS = 25000       # Absolute max
MAX_ASSISTANT_TEXT = 500      # Max chars per assistant text block
MAX_USER_TEXT = 800           # Max chars per user message
TOOL_SUMMARY_MAX = 120        # Max chars for tool call one-liner


# ═══════════════════════════════════════════════════════════
# FINDING & READING
# ═══════════════════════════════════════════════════════════

def find_jsonl_file():
    """Find the most recent .jsonl chat log file."""
    # Cowork sessions
    for pattern in [
        "/sessions/*/mnt/.claude/projects/-sessions-*/*.jsonl",
        "/sessions/*/mnt/.claude/projects/*/*.jsonl",
    ]:
        files = glob.glob(pattern)
        if files:
            return max(files, key=os.path.getmtime)
    return None


def get_brain_nodes_by_window(brain_url, since_iso):
    """
    Get node creation counts bucketed by 30-min windows from the brain.
    Returns dict: { "2026-03-17T08:00" : 5, "2026-03-17T08:30": 0, ... }
    """
    import urllib.request

    # Get all nodes created since the given timestamp via /timeline
    nodes_by_window = {}
    try:
        # Timeline returns nodes ordered by creation time
        req = urllib.request.Request(
            f"{brain_url}/timeline",
            data=json.dumps({"limit": 500}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        events = data.get("events", [])
        for node in events:
            created = node.get("created_at", "")
            if not created or created < since_iso:
                continue
            # Bucket into 30-min window
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                window_min = (dt.minute // WINDOW_MINUTES) * WINDOW_MINUTES
                window_key = dt.strftime(f"%Y-%m-%dT%H:") + f"{window_min:02d}"
                nodes_by_window[window_key] = nodes_by_window.get(window_key, 0) + 1
            except (ValueError, AttributeError):
                pass
    except Exception:
        pass

    return nodes_by_window


def get_last_compaction_timestamp(jsonl_path):
    """
    Find the timestamp of the most recent compaction event in the log.
    Reads backwards from end for efficiency.
    """
    last_compaction_ts = None

    # Read the file and find the last "continued from a previous conversation"
    # For large files, we read backwards by checking the last ~500 lines
    try:
        with open(jsonl_path, "rb") as f:
            # Seek to near the end
            f.seek(0, 2)
            file_size = f.tell()
            # Read last 5MB (should be plenty for recent compaction)
            read_size = min(file_size, 5 * 1024 * 1024)
            f.seek(file_size - read_size)
            tail = f.read().decode("utf-8", errors="replace")

        for line in reversed(tail.strip().split("\n")):
            try:
                obj = json.loads(line)
                if obj.get("type") == "user":
                    msg = obj.get("message", {})
                    content = msg.get("content", "") if isinstance(msg, dict) else ""
                    if isinstance(content, list):
                        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
                        content = " ".join(texts)
                    if "continued from a previous conversation" in str(content):
                        ts = obj.get("timestamp", "")
                        if ts:
                            last_compaction_ts = ts
                            break
            except (json.JSONDecodeError, AttributeError):
                continue
    except Exception:
        pass

    return last_compaction_ts


# ═══════════════════════════════════════════════════════════
# EXTRACTION
# ═══════════════════════════════════════════════════════════

def summarize_tool_call(tool_name, tool_input):
    """Create a one-line summary of a tool call."""
    if tool_name in ("Read", "read"):
        path = tool_input.get("file_path", "")
        fname = os.path.basename(path) if path else "?"
        return f"Read {fname}"
    elif tool_name in ("Edit", "edit"):
        path = tool_input.get("file_path", "")
        fname = os.path.basename(path) if path else "?"
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        if old and new:
            return f"Edited {fname} ({len(old.splitlines())}→{len(new.splitlines())} lines)"
        return f"Edited {fname}"
    elif tool_name in ("Write", "write"):
        path = tool_input.get("file_path", "")
        fname = os.path.basename(path) if path else "?"
        content = tool_input.get("content", "")
        return f"Wrote {fname} ({len(content.splitlines())} lines)"
    elif tool_name in ("Bash", "bash"):
        desc = tool_input.get("description", "")
        if desc:
            return f"Bash: {desc[:TOOL_SUMMARY_MAX]}"
        cmd = tool_input.get("command", "")
        return f"Bash: {cmd.split(chr(10))[0][:TOOL_SUMMARY_MAX]}"
    elif tool_name in ("Glob", "glob"):
        return f"Searched for files: {tool_input.get('pattern', '')}"
    elif tool_name in ("Grep", "grep"):
        return f"Searched code for: {tool_input.get('pattern', '')[:80]}"
    elif tool_name == "WebSearch":
        return f"Web search: {tool_input.get('query', '')[:80]}"
    elif tool_name == "WebFetch":
        return f"Fetched: {tool_input.get('url', '')[:80]}"
    else:
        return f"Tool: {tool_name}"


def parse_timestamp(ts_str):
    """Parse an ISO timestamp string to datetime, handling various formats."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def extract_windowed_narrative(jsonl_path, since_iso, sparse_windows):
    """
    Extract narrative from JSONL, but ONLY for time windows in sparse_windows set.
    Returns list of (window_key, entries) where entries are (timestamp, speaker, content).
    """
    window_narratives = {}  # window_key -> list of (ts, speaker, content)

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = obj.get("timestamp", "")
            if not timestamp or timestamp < since_iso:
                continue

            # Determine which window this belongs to
            dt = parse_timestamp(timestamp)
            if not dt:
                continue
            window_min = (dt.minute // WINDOW_MINUTES) * WINDOW_MINUTES
            window_key = dt.strftime(f"%Y-%m-%dT%H:") + f"{window_min:02d}"

            # Only extract for sparse windows
            if window_key not in sparse_windows:
                continue

            if window_key not in window_narratives:
                window_narratives[window_key] = []

            msg_type = obj.get("type")

            if msg_type == "user":
                message = obj.get("message", {})
                content = message.get("content", "") if isinstance(message, dict) else str(message or "")
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                    content = " ".join(texts)
                if not isinstance(content, str):
                    continue
                content = content.strip()
                if "continued from a previous conversation" in content:
                    window_narratives[window_key].append((timestamp, "system", "[Compaction event]"))
                    continue
                if len(content) > 5:
                    if len(content) > MAX_USER_TEXT:
                        content = content[:MAX_USER_TEXT] + "..."
                    window_narratives[window_key].append((timestamp, "user", content))

            elif msg_type == "assistant":
                message = obj.get("message", {})
                if not isinstance(message, dict):
                    continue
                content_blocks = message.get("content", [])
                if isinstance(content_blocks, str):
                    if len(content_blocks.strip()) > 10:
                        text = content_blocks.strip()
                        if len(text) > MAX_ASSISTANT_TEXT:
                            text = text[:MAX_ASSISTANT_TEXT] + "..."
                        window_narratives[window_key].append((timestamp, "claude", text))
                    continue
                if not isinstance(content_blocks, list):
                    continue
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if len(text) > 10:
                            if len(text) > MAX_ASSISTANT_TEXT:
                                text = text[:MAX_ASSISTANT_TEXT] + "..."
                            window_narratives[window_key].append((timestamp, "claude", text))
                    elif block.get("type") == "tool_use":
                        summary = summarize_tool_call(block.get("name", ""), block.get("input", {}))
                        window_narratives[window_key].append((timestamp, "tool", summary))

    return window_narratives


# ═══════════════════════════════════════════════════════════
# DENSITY ANALYSIS
# ═══════════════════════════════════════════════════════════

def count_user_msgs_by_window(jsonl_path, since_iso):
    """
    Quick scan: count user messages per 30-min window.
    Only reads user lines (cheap).
    """
    windows = {}

    with open(jsonl_path, "r") as f:
        for line in f:
            if '"type":"user"' not in line and '"type": "user"' not in line:
                continue  # Fast skip non-user lines without full JSON parse
            line = line.strip()
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "user":
                continue

            timestamp = obj.get("timestamp", "")
            if not timestamp or timestamp < since_iso:
                continue

            # Skip compaction summaries
            msg = obj.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if isinstance(c, dict)]
                content = " ".join(texts)
            if "continued from a previous conversation" in str(content):
                continue

            dt = parse_timestamp(timestamp)
            if not dt:
                continue
            window_min = (dt.minute // WINDOW_MINUTES) * WINDOW_MINUTES
            window_key = dt.strftime(f"%Y-%m-%dT%H:") + f"{window_min:02d}"
            windows[window_key] = windows.get(window_key, 0) + 1

    return windows


def find_sparse_windows(user_msg_windows, brain_node_windows):
    """
    Compare user message density vs brain node density per window.
    Returns set of window keys that are sparse (brain missed them).
    """
    sparse = set()

    for window_key, user_count in user_msg_windows.items():
        if user_count < MIN_USER_MSGS:
            continue  # Too little activity to judge

        brain_count = brain_node_windows.get(window_key, 0)
        ratio = brain_count / user_count if user_count > 0 else 0

        if ratio < SPARSE_THRESHOLD:
            sparse.add(window_key)

    return sparse


# ═══════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════

def format_windowed_narrative(window_narratives, soft_cap=SOFT_CAP_CHARS, hard_cap=HARD_CAP_CHARS):
    """
    Format extracted narratives from sparse windows into readable output.
    Windows are sorted chronologically. More recent windows get more space.
    """
    if not window_narratives:
        return ""

    # Sort windows chronologically
    sorted_windows = sorted(window_narratives.keys())
    n_windows = len(sorted_windows)

    # Proportional budgets: recent windows get more space
    weights = []
    for i in range(n_windows):
        dist_from_end = n_windows - 1 - i
        if dist_from_end == 0:
            weights.append(1.0)
        elif dist_from_end == 1:
            weights.append(0.7)
        elif dist_from_end == 2:
            weights.append(0.5)
        else:
            weights.append(0.25)

    total_weight = sum(weights)
    char_budgets = [int(soft_cap * w / total_weight) for w in weights]

    output_parts = []

    for window_key, budget in zip(sorted_windows, char_budgets):
        entries = window_narratives[window_key]
        if not entries:
            continue

        # Window header
        section_lines = [f"── Gap: {window_key} ({len([e for e in entries if e[1] == 'user'])} user msgs, brain had few/no nodes) ──"]
        chars_used = len(section_lines[0])

        tool_buffer = []

        for timestamp, speaker, content in entries:
            if speaker == "system":
                if tool_buffer:
                    collapsed = _collapse_tools(tool_buffer)
                    section_lines.append(collapsed)
                    chars_used += len(collapsed)
                    tool_buffer = []
                section_lines.append(f"\n--- {content} ---\n")
                chars_used += len(content) + 10
                continue

            if speaker == "tool":
                tool_buffer.append(content)
                if len(tool_buffer) >= 10:
                    collapsed = _collapse_tools(tool_buffer)
                    section_lines.append(collapsed)
                    chars_used += len(collapsed)
                    tool_buffer = []
                continue

            # Flush tools before conversation
            if tool_buffer:
                collapsed = _collapse_tools(tool_buffer)
                section_lines.append(collapsed)
                chars_used += len(collapsed)
                tool_buffer = []

            if chars_used >= budget:
                remaining = len(entries) - len(section_lines)
                if remaining > 0:
                    section_lines.append(f"  [...{remaining} more entries in this window]")
                break

            prefix = "TOM:" if speaker == "user" else "CLAUDE:"
            line = f"{prefix} {content}"
            section_lines.append(line)
            chars_used += len(line)

        # Flush remaining tools
        if tool_buffer:
            section_lines.append(_collapse_tools(tool_buffer))

        output_parts.append("\n".join(section_lines))

    full_output = "\n\n".join(output_parts)

    if len(full_output) > hard_cap:
        full_output = "...[older gaps truncated]\n\n" + full_output[-hard_cap:]

    return full_output


def _collapse_tools(tool_buffer):
    """Collapse a list of tool summaries into a single line."""
    shown = "; ".join(tool_buffer[:5])
    extra = f" +{len(tool_buffer)-5} more" if len(tool_buffer) > 5 else ""
    return f"  [code: {shown}{extra}]"


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """
    Main entry point. Called from boot-brain.sh or directly for testing.

    Usage:
      python3 extract-session-log.py                     # auto: density-based from last compaction
      python3 extract-session-log.py --since TIMESTAMP   # override start point
      python3 extract-session-log.py --last-n-hours 6    # override: last N hours
      python3 extract-session-log.py --density-report     # show density analysis without extracting
      python3 extract-session-log.py --stats              # basic stats only
    """
    import argparse
    parser = argparse.ArgumentParser(description="Density-based session log extractor")
    parser.add_argument("--since", type=str, help="ISO timestamp to analyze from")
    parser.add_argument("--last-n-hours", type=float, help="Analyze last N hours")
    parser.add_argument("--density-report", action="store_true", help="Show density analysis per window")
    parser.add_argument("--stats", action="store_true", help="Basic stats only")
    parser.add_argument("--jsonl", type=str, help="Path to JSONL file (override auto-detect)")
    parser.add_argument("--threshold", type=float, default=SPARSE_THRESHOLD, help="Sparse threshold ratio")
    args = parser.parse_args()

    brain_url = os.environ.get("BRAIN_URL", "http://127.0.0.1:7437")

    # Find the chat log
    jsonl_path = args.jsonl or find_jsonl_file()
    if not jsonl_path:
        sys.exit(0)

    # Determine analysis window
    if args.since:
        since_iso = args.since
    elif args.last_n_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.last_n_hours)
        since_iso = cutoff.isoformat()
    else:
        # Default: from last compaction event
        since_iso = get_last_compaction_timestamp(jsonl_path)
        if not since_iso:
            # No compaction found — use last 2 hours as fallback
            cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
            since_iso = cutoff.isoformat()

    # Step 1: Count user messages per window (fast scan)
    user_windows = count_user_msgs_by_window(jsonl_path, since_iso)

    if not user_windows:
        sys.exit(0)

    # Step 2: Get brain node counts per window
    brain_windows = get_brain_nodes_by_window(brain_url, since_iso)

    # Step 3: Find sparse windows
    sparse = find_sparse_windows(user_windows, brain_windows)

    if args.density_report:
        print(f"Analysis since: {since_iso}")
        print(f"Sparse threshold: {args.threshold} (nodes/user_msgs ratio)")
        print(f"Window size: {WINDOW_MINUTES} minutes")
        print()
        all_windows = sorted(set(list(user_windows.keys()) + list(brain_windows.keys())))
        for w in all_windows:
            u = user_windows.get(w, 0)
            b = brain_windows.get(w, 0)
            ratio = b / u if u > 0 else float('inf')
            is_sparse = w in sparse
            marker = " ← SPARSE (will extract)" if is_sparse else ""
            skip = " (< min msgs)" if u < MIN_USER_MSGS else ""
            print(f"  {w}  user:{u:3d}  brain:{b:3d}  ratio:{ratio:.2f}{skip}{marker}")
        print()
        print(f"Sparse windows: {len(sparse)} / {len(all_windows)}")
        sys.exit(0)

    if args.stats:
        print(f"Since: {since_iso}")
        print(f"Windows analyzed: {len(user_windows)}")
        print(f"Sparse windows: {len(sparse)}")
        total_user = sum(user_windows.values())
        total_brain = sum(brain_windows.get(w, 0) for w in user_windows)
        print(f"Total user msgs: {total_user}, brain nodes in range: {total_brain}")
        if sparse:
            sparse_user = sum(user_windows.get(w, 0) for w in sparse)
            print(f"User msgs in sparse windows: {sparse_user}")
        sys.exit(0)

    if not sparse:
        # No gaps — brain coverage is healthy
        sys.exit(0)

    # Step 4: Extract narrative only for sparse windows
    window_narratives = extract_windowed_narrative(jsonl_path, since_iso, sparse)

    if not window_narratives:
        sys.exit(0)

    # Step 5: Format with proportional soft cap
    output = format_windowed_narrative(window_narratives)

    if output:
        total_entries = sum(len(v) for v in window_narratives.values())
        user_count = sum(1 for entries in window_narratives.values() for _, s, _ in entries if s == "user")
        claude_count = sum(1 for entries in window_narratives.values() for _, s, _ in entries if s == "claude")

        header = (
            f"BRAIN GAP DETECTED — {len(sparse)} sparse windows found "
            f"({user_count} user msgs, {claude_count} claude msgs in gaps)\n"
            f"The brain missed these periods. Review and encode important learnings.\n"
            f"{'=' * 60}\n"
        )

        print(header + output)


if __name__ == "__main__":
    main()
