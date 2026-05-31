"""S0 Conversation API — single source of truth for conversation context.

Any layer that needs conversation context (S1E encoder, S2 Healer, S3,
eval tools) uses this API. Abstracts the data source: S0 traces (post-April 5)
or JSONL conversation logs (pre-April 5).

This is the S0 layer's service to upper layers. Each scale exposes its
services to the scale above.

Usage:
    from servers.scales.s0.conversation import get_conversation_around

    # By node ID — finds conversation around when the node was created
    turns = get_conversation_around(brain, node_id='abc12345')

    # By timestamp — finds conversation at a specific time
    turns = get_conversation_around(brain, timestamp='2026-03-25T16:00:00')

    # By session ID — gets turns from a specific session
    turns = get_conversation_around(brain, session_id='8c4cc185-...',
                                     timestamp='2026-04-12T18:00:00')

    # Control window size
    turns = get_conversation_around(brain, node_id='abc', before=5, after=3)

Returns: [{role: 'user'|'assistant', content: str, timestamp: str}]
"""

import json
import os
from bisect import bisect_left
from typing import Dict, List, Optional

# Conversation JSONL directory — relative to brain repo root
_CONV_DIR = None


def _get_conv_dir():
    """Resolve conversation directory path. Cached after first call."""
    global _CONV_DIR
    if _CONV_DIR is not None:
        return _CONV_DIR

    # Try: brain repo / conversations
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    candidate = os.path.join(repo_root, 'conversations')
    if os.path.isdir(candidate):
        _CONV_DIR = candidate
        return _CONV_DIR

    _CONV_DIR = ''  # Not found
    return _CONV_DIR


def get_conversation(brain, session_id: str, limit: int = 20) -> List[Dict]:
    """Get recent conversation turns for a session.

    The simple path — S1E, hooks, anything that knows its session_id and
    wants the last N turns. No timestamp resolution, no JSONL fallback.

    Args:
        brain: Brain instance
        session_id: Full session UUID
        limit: Max turns to return (most recent)

    Returns: [{role, content, timestamp, trace_id, judge_output}]
        trace_id: 8-char hex id from trace_events (v29) — used by S1 encoder
                  to populate source_refs via `[trace:<hex>]` inline markers.
        judge_output: surface selection from S1R for the user turn (if any).
    """
    try:
        turns = brain._trace_dal.get_session_turns(session_id, limit=limit)
        return [{'role': t['role'],
                 'trace_id': t.get('trace_id'),
                 'content': t.get('content', ''),
                 'timestamp': t.get('timestamp', ''),
                 'judge_output': t.get('judge_output', '')}
                for t in turns]
    except Exception:
        return []


def get_conversation_around(brain, node_id: str = None,
                            session_id: str = None,
                            timestamp: str = None,
                            before: int = 10, after: int = 5) -> List[Dict]:
    """Get conversation exchanges around a point in time.

    Resolution order:
    1. If session_id + timestamp given: query that session directly
    2. If node_id given: find encoding trace → get session_id + timestamp
    3. If only timestamp given: find nearest session from traces
    4. If traces fail: fall back to JSONL conversation logs

    Args:
        brain: Brain instance
        node_id: Node ID — resolves to the conversation that created it
        session_id: Full session UUID — skip searching, query directly
        timestamp: ISO timestamp to center the window on
        before: Exchanges before the timestamp (default 10)
        after: Exchanges after the timestamp (default 5)

    Returns: [{role: 'user'|'assistant', content: str, timestamp: str}]
             Chronological order. Empty list if no conversation found.
    """
    # Resolve timestamp from node if needed
    resolved_session = session_id
    resolved_timestamp = timestamp

    if node_id and not timestamp:
        resolved_timestamp = _resolve_node_timestamp(brain, node_id)

    if not resolved_timestamp:
        return []

    # If we have node_id but no session, find the encoding session
    if node_id and not resolved_session:
        resolved_session, enc_ts = _find_encoding_session(brain, node_id, resolved_timestamp)
        if enc_ts:
            resolved_timestamp = enc_ts  # Use encoding timestamp, more precise than created_at

    # Strategy 1: S0 traces (post-April 5)
    if resolved_session:
        turns = _from_traces_by_session(brain, resolved_session, resolved_timestamp, before, after)
        if turns:
            return turns

    # Strategy 2: Find session by timestamp proximity
    turns = _from_traces_by_timestamp(brain, resolved_timestamp, before, after)
    if turns:
        return turns

    # Strategy 3: JSONL conversation logs (pre-April 5)
    return _from_jsonl(resolved_timestamp, before, after)


def _resolve_node_timestamp(brain, node_id):
    """Get a node's created_at timestamp."""
    row = brain.conn.execute(
        "SELECT created_at FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if row:
        return row[0]

    # Try short ID
    row = brain.conn.execute(
        "SELECT created_at FROM nodes WHERE id LIKE ?", (node_id[:8] + '%',)
    ).fetchone()
    return row[0] if row else None


def _find_encoding_session(brain, node_id, node_created_at):
    """Find which session encoded this node.

    Checks S1E traces for this node_id. Returns (session_id, encoding_timestamp).
    """
    short_id = node_id[:8]
    try:
        # Direct match: S1E trace metadata contains this node ID
        hit = brain._trace_dal.find_by_metadata_substring(
            's1', 'encoding_run', short_id)
        if hit and hit['session_id']:
            return hit['session_id'], hit['created_at']

        # Fallback: nearest S1E trace before node creation, SAME DAY
        # (prevents matching traces from completely different sessions)
        node_date = node_created_at[:10]
        hit = brain._trace_dal.latest_in_window(
            's1', 'encoding_run', node_created_at, node_date + 'T00:00:00')
        if hit and hit['session_id']:
            return hit['session_id'], hit['created_at']

    except Exception:
        pass

    return None, None


def _from_traces_by_session(brain, session_id, timestamp, before, after):
    """Get conversation from S0 traces for a specific session."""
    try:
        turns = brain._trace_dal.get_session_turns(
            session_id,
            around_timestamp=timestamp,
            before=before,
            after=after,
        )
        if turns:
            return [{'role': t['role'], 'content': t.get('content', ''),
                      'timestamp': t.get('timestamp', '')} for t in turns]
    except Exception:
        pass
    return []


def _from_traces_by_timestamp(brain, timestamp, before, after):
    """Find the session active at a timestamp and get its conversation."""
    try:
        # Find the S0 trace closest to this timestamp
        hit = brain._trace_dal.latest_in_window(
            's0', 'user_message', timestamp, timestamp[:10] + 'T00:00:00')
        if hit and hit['session_id']:
            return _from_traces_by_session(brain, hit['session_id'], timestamp, before, after)
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════════════
# JSONL Conversation Log Support (pre-trace history)
# ═══════════════════════════════════════════════════════════════

def _from_jsonl(timestamp, before, after):
    """Get conversation from JSONL log files."""
    conv_dir = _get_conv_dir()
    if not conv_dir:
        return []

    target_file = _find_conversation_file(conv_dir, timestamp)
    if not target_file:
        return []

    return _extract_window(target_file, timestamp, before, after)


def _find_conversation_file(conv_dir, timestamp):
    """Find which JSONL file covers a timestamp.

    Checks files by their internal message timestamps, not just filenames.
    Caches time ranges to avoid re-scanning.
    """
    target_date = timestamp[:10]
    best_file = None
    best_distance = float('inf')

    for fname in os.listdir(conv_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(conv_dir, fname)

        first_ts, last_ts = _get_file_time_range(path)
        if not first_ts or not last_ts:
            continue

        # Check if target falls within this file's range
        if first_ts[:10] <= target_date <= last_ts[:10]:
            return path  # Exact match

        # Track closest file for near-misses
        if first_ts[:10] <= target_date:
            distance = ord(target_date[9]) - ord(last_ts[9]) if last_ts else 99
            if distance < best_distance:
                best_distance = distance
                best_file = path

    return best_file


def _get_file_time_range(path):
    """Get first and last message timestamps from a JSONL file."""
    first_ts = None
    last_ts = None

    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i > 50:
                    break
                try:
                    obj = json.loads(line.strip())
                    if obj.get('type') in ('user', 'assistant', 'human'):
                        ts = obj.get('timestamp', '')
                        if ts and not first_ts:
                            first_ts = ts
                except (json.JSONDecodeError, KeyError):
                    pass

        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20480))
            tail = f.read().decode('utf-8', errors='ignore')
            for line in reversed(tail.split('\n')):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get('type') in ('user', 'assistant', 'human'):
                        ts = obj.get('timestamp', '')
                        if ts:
                            last_ts = ts
                            break
                except (json.JSONDecodeError, KeyError):
                    pass
    except (IOError, OSError):
        pass

    return first_ts, last_ts


def _extract_window(path, timestamp, before, after):
    """Extract conversation messages around a timestamp from JSONL file."""
    messages = []

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get('type') not in ('user', 'assistant', 'human'):
                        continue

                    ts = obj.get('timestamp', '')
                    if not ts:
                        continue

                    role = 'user' if obj['type'] in ('user', 'human') else 'assistant'

                    msg = obj.get('message', {})
                    if isinstance(msg, dict):
                        content = msg.get('content', '')
                    else:
                        content = obj.get('content', '')

                    if isinstance(content, list):
                        texts = [p.get('text', '') for p in content
                                 if isinstance(p, dict) and p.get('type') == 'text']
                        content = ' '.join(texts)

                    if not content or len(content.strip()) < 2:
                        continue

                    messages.append({
                        'role': role,
                        'content': content[:500],
                        'timestamp': ts,
                    })

                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except (IOError, OSError):
        return []

    if not messages:
        return []

    # Find message closest to target timestamp
    timestamps = [m['timestamp'] for m in messages]
    idx = bisect_left(timestamps, timestamp)
    idx = min(idx, len(messages) - 1)

    # Window: before × 2 and after × 2 (user + assistant = 2 per exchange)
    start = max(0, idx - before * 2)
    end = min(len(messages), idx + after * 2 + 1)

    return messages[start:end]
