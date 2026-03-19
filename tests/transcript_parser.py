#!/usr/bin/env python3
"""
brain — Transcript Parser

Reads Claude session .jsonl transcripts and extracts structured conversation data
for the Relearning Comparison simulation.

Extracts:
  - User messages (with timestamps)
  - Assistant responses (text only, stripped of tool scaffolding)
  - File edit contexts (which files were edited)
  - Session boundaries (date changes, long gaps)
  - Historical remember() calls (for ground truth comparison)
"""

import json
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple


# Gap threshold for session boundary detection (hours)
SESSION_GAP_HOURS = 2


def parse_transcript(jsonl_path: str, max_lines: int = 0) -> Dict[str, Any]:
    """
    Parse a Claude session .jsonl transcript into structured conversation data.

    Args:
        jsonl_path: Path to the .jsonl transcript file
        max_lines: Max lines to parse (0 = all)

    Returns:
        Dict with:
          - turns: List of conversation turns (user msg + assistant response)
          - sessions: List of session groups (turns grouped by date/gap)
          - remember_history: Historical remember() calls for ground truth
          - file_edits: Files that were edited during the conversation
          - stats: Parsing statistics
    """
    raw_messages = _load_raw_messages(jsonl_path, max_lines)
    turns = _extract_turns(raw_messages)
    sessions = _group_into_sessions(turns)
    remember_history = _extract_remember_calls(raw_messages)
    file_edits = _extract_file_edits(raw_messages)

    return {
        'turns': turns,
        'sessions': sessions,
        'remember_history': remember_history,
        'file_edits': file_edits,
        'stats': {
            'total_raw_messages': len(raw_messages),
            'total_turns': len(turns),
            'total_sessions': len(sessions),
            'total_remember_calls': len(remember_history),
            'total_file_edits': len(file_edits),
            'date_range': _get_date_range(turns),
        }
    }


def _load_raw_messages(jsonl_path: str, max_lines: int = 0) -> List[Dict]:
    """Load raw JSONL lines, keeping only user/assistant messages."""
    messages = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_lines > 0 and i >= max_lines:
                break
            try:
                d = json.loads(line.strip())
                msg_type = d.get('type', '')
                if msg_type in ('user', 'assistant'):
                    messages.append(d)
            except (json.JSONDecodeError, ValueError):
                continue
    return messages


def _extract_turns(raw_messages: List[Dict]) -> List[Dict[str, Any]]:
    """
    Group raw messages into conversation turns (user message + assistant response).

    A turn is: one user message followed by the assistant's complete response.
    We collapse multi-block assistant messages into a single text summary.
    """
    turns = []
    i = 0

    while i < len(raw_messages):
        msg = raw_messages[i]
        role = msg.get('message', {}).get('role', '')

        if role == 'user':
            user_text = _extract_text_from_message(msg)
            user_ts = msg.get('timestamp', '')

            # Skip system-injected messages (hook outputs, tool results without real user input)
            if _is_system_injected(user_text):
                i += 1
                continue

            # Collect assistant response(s)
            assistant_text = ''
            assistant_files = []
            j = i + 1
            while j < len(raw_messages):
                next_msg = raw_messages[j]
                next_role = next_msg.get('message', {}).get('role', '')
                if next_role == 'user':
                    break
                if next_role == 'assistant':
                    text, files = _extract_assistant_content(next_msg)
                    if text:
                        assistant_text += text + '\n'
                    assistant_files.extend(files)
                j += 1

            # Only add turn if user said something meaningful
            if user_text and len(user_text.strip()) > 5:
                turns.append({
                    'index': len(turns),
                    'timestamp': user_ts,
                    'date': user_ts[:10] if user_ts else '',
                    'user_message': user_text.strip(),
                    'assistant_response': assistant_text.strip()[:2000],  # Cap to avoid bloat
                    'files_edited': assistant_files,
                })

            i = j
        else:
            i += 1

    return turns


def _extract_text_from_message(msg_wrapper: Dict) -> str:
    """Extract plain text from a message, handling both string and block formats."""
    content = msg_wrapper.get('message', {}).get('content', '')

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    parts.append(block.get('text', ''))
                elif block.get('type') == 'tool_result':
                    # Include tool result content if it's user-provided (like uploaded files)
                    sub_content = block.get('content', '')
                    if isinstance(sub_content, str) and len(sub_content) < 500:
                        parts.append(sub_content)
                    elif isinstance(sub_content, list):
                        for sub in sub_content:
                            if isinstance(sub, dict) and sub.get('type') == 'text':
                                text = sub.get('text', '')
                                if len(text) < 500:
                                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return '\n'.join(parts)

    return ''


def _extract_assistant_content(msg_wrapper: Dict) -> Tuple[str, List[str]]:
    """
    Extract meaningful text and file edit targets from assistant message.
    Strips tool_use blocks but keeps text responses. Returns (text, files_edited).
    """
    content = msg_wrapper.get('message', {}).get('content', '')
    text_parts = []
    files = []

    if isinstance(content, str):
        return content, []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get('type') == 'text':
                text = block.get('text', '')
                # Skip very short fragments (often just whitespace between tool calls)
                if len(text.strip()) > 10:
                    text_parts.append(text.strip())

            elif block.get('type') == 'tool_use':
                name = block.get('name', '')
                inp = block.get('input', {})

                # Track file edits
                if name in ('Edit', 'Write'):
                    file_path = inp.get('file_path', '')
                    if file_path:
                        # Simplify path
                        files.append(os.path.basename(file_path))

    return '\n'.join(text_parts), files


def _is_system_injected(text: str) -> bool:
    """Check if a user message is actually system-injected (hook output, system reminder)."""
    if not text:
        return True

    # Common system-injected patterns
    system_patterns = [
        r'^<system-reminder>',
        r'^\[hook:',
        r'^Hook output:',
        r'^\{"results":',  # Raw JSON from hooks
        r'^\{"status":',
        r'^IMPORTANT:.*context may or may not be relevant',
    ]

    for pattern in system_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True

    return False


def _group_into_sessions(turns: List[Dict]) -> List[Dict[str, Any]]:
    """
    Group turns into logical sessions based on date changes and time gaps.

    A new session starts when:
      - Date changes
      - Gap > SESSION_GAP_HOURS between consecutive turns
    """
    if not turns:
        return []

    sessions = []
    current_session = {
        'session_index': 0,
        'date': turns[0].get('date', ''),
        'start_ts': turns[0].get('timestamp', ''),
        'turns': [turns[0]],
    }

    for i in range(1, len(turns)):
        turn = turns[i]
        prev_turn = turns[i - 1]

        # Check for session boundary
        new_session = False

        # Date change
        if turn.get('date') != prev_turn.get('date'):
            new_session = True

        # Time gap
        gap = _compute_gap_hours(prev_turn.get('timestamp'), turn.get('timestamp'))
        if gap is not None and gap > SESSION_GAP_HOURS:
            new_session = True

        if new_session:
            current_session['end_ts'] = prev_turn.get('timestamp', '')
            current_session['turn_count'] = len(current_session['turns'])
            sessions.append(current_session)

            current_session = {
                'session_index': len(sessions),
                'date': turn.get('date', ''),
                'start_ts': turn.get('timestamp', ''),
                'turns': [],
            }

        current_session['turns'].append(turn)

    # Close final session
    current_session['end_ts'] = turns[-1].get('timestamp', '') if turns else ''
    current_session['turn_count'] = len(current_session['turns'])
    sessions.append(current_session)

    return sessions


def _compute_gap_hours(ts1: str, ts2: str) -> Optional[float]:
    """Compute gap in hours between two ISO timestamps."""
    if not ts1 or not ts2:
        return None
    try:
        dt1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
        dt2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
        return (dt2 - dt1).total_seconds() / 3600
    except (ValueError, TypeError):
        return None


def _extract_remember_calls(raw_messages: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract historical remember() calls from the transcript for ground truth.
    Handles both curl-based (HTTP) and Skill-based remember calls.
    """
    calls = []

    for msg in raw_messages:
        role = msg.get('message', {}).get('role', '')
        content = msg.get('message', {}).get('content', '')
        ts = msg.get('timestamp', '')

        if role != 'assistant' or not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'tool_use':
                continue

            name = block.get('name', '')
            inp = block.get('input', {})

            # Curl-based remember
            if name == 'Bash':
                cmd = inp.get('command', '')
                if '/remember' in cmd and 'curl' in cmd:
                    payload = _extract_curl_json(cmd)
                    if payload:
                        calls.append({
                            'timestamp': ts,
                            'source': 'curl',
                            'type': payload.get('type', ''),
                            'title': payload.get('title', ''),
                            'content': payload.get('content', ''),
                            'keywords': payload.get('keywords', ''),
                            'locked': payload.get('locked', False),
                            'connections': payload.get('connections', []),
                            'project': payload.get('project', ''),
                        })

            # Skill-based remember
            elif name == 'Skill':
                skill = inp.get('skill', '')
                if 'remember' in skill:
                    args = inp.get('args', '')
                    calls.append({
                        'timestamp': ts,
                        'source': 'skill',
                        'type': '',  # Not structured
                        'title': args[:200] if args else '',
                        'content': args if args else '',
                        'keywords': '',
                        'locked': False,
                        'connections': [],
                        'project': '',
                    })

    return calls


def _extract_curl_json(cmd: str) -> Optional[Dict]:
    """Extract JSON payload from a curl -d command."""
    # Match both -d '...' and -d "..."
    match = re.search(r"-d\s+'({.*?})'", cmd, re.DOTALL)
    if not match:
        match = re.search(r'-d\s+"({.*?})"', cmd, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_file_edits(raw_messages: List[Dict]) -> List[Dict[str, str]]:
    """Extract all file edit operations from the transcript."""
    edits = []

    for msg in raw_messages:
        content = msg.get('message', {}).get('content', '')
        ts = msg.get('timestamp', '')

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'tool_use':
                continue

            name = block.get('name', '')
            inp = block.get('input', {})

            if name in ('Edit', 'Write'):
                file_path = inp.get('file_path', '')
                if file_path:
                    edits.append({
                        'timestamp': ts,
                        'tool': name,
                        'file': os.path.basename(file_path),
                        'full_path': file_path,
                    })

    return edits


def _get_date_range(turns: List[Dict]) -> Dict[str, str]:
    """Get first and last date from turns."""
    if not turns:
        return {'start': '', 'end': ''}

    dates = [t.get('date', '') for t in turns if t.get('date')]
    if not dates:
        return {'start': '', 'end': ''}

    return {'start': min(dates), 'end': max(dates)}


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python transcript_parser.py <transcript.jsonl>')
        sys.exit(1)

    result = parse_transcript(sys.argv[1])
    stats = result['stats']

    print(f"Parsed transcript:")
    print(f"  Raw messages: {stats['total_raw_messages']}")
    print(f"  Turns: {stats['total_turns']}")
    print(f"  Sessions: {stats['total_sessions']}")
    print(f"  Remember calls: {stats['total_remember_calls']}")
    print(f"  File edits: {stats['total_file_edits']}")
    print(f"  Date range: {stats['date_range']['start']} → {stats['date_range']['end']}")

    print(f"\nSessions:")
    for s in result['sessions']:
        print(f"  [{s['session_index']}] {s['date']} | {s['turn_count']} turns | {s['start_ts'][:19]}")
