"""Clock contract: no direct datetime.now() / date.today() in S1/S2 code.

Why this file exists
--------------------
Bug surfaced 2026-05-11: build_muster_context() silently fell back to
datetime.now() when no current_date was passed. In eval replays of
historical conversations, this resolved "today/yesterday" against the
real wall-clock instead of the conversation date — corrupting temporal
scout candidates.

Fix: servers/clock.py is the single source of truth for "now". This test
keeps it that way by failing if anyone adds a direct wall-clock call in
S1/S2/scout code.

Allowed exemptions:
  - servers/clock.py itself (the implementation)
  - Trace timestamps, perf timers, "created_at" stamps where wall-clock
    IS the right answer (these patterns are excluded by suffix matching
    on the surrounding line — e.g. lines doing `created_at = ...isoformat()`)
  - DAL code (system bookkeeping is wall-clock by design)

The intent: anything that needs to know "what date is THIS CONVERSATION
happening" must route through brain_now / conversation_now.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent

# Directories where wall-clock calls are forbidden (semantic-time code).
PROTECTED_DIRS = [
    'servers/scales/s1',
    'servers/scales/s2',
]

# Files within protected dirs where wall-clock is legitimate (logs/traces).
# Keep this list TIGHT — every entry is a foothold for the bug to come back.
EXEMPT_FILES = {
    # None at the moment — all S1/S2 semantic-time should go through clock.
}

# Patterns that flag a direct wall-clock call.
FORBIDDEN_PATTERNS = [
    re.compile(r'\bdatetime\.now\s*\('),
    re.compile(r'\b_dt\.datetime\.now\s*\('),
    re.compile(r'\bdate\.today\s*\('),
    re.compile(r'\b_dt\.date\.today\s*\('),
]

# Lines that ARE allowed even in protected dirs — bookkeeping wall-clock
# uses where conversation_now() would be semantically wrong.
LEGITIMATE_USES = [
    # Trace event created_at — these are system clocks, not conversation time
    re.compile(r'created_at\s*[:=].*now'),
    re.compile(r'set_at\s*[:=].*now'),
    re.compile(r'updated_at\s*[:=].*now'),
    re.compile(r'timestamp\s*[:=].*now'),
    # Latency / perf
    re.compile(r't0\s*=.*now'),
    re.compile(r'started_at\s*[:=].*now'),
    # Inside isoformat() chains for trace metadata are OK if marked
    re.compile(r'#\s*clock-ok'),
]


def _is_legit(line: str) -> bool:
    return any(p.search(line) for p in LEGITIMATE_USES)


def _scan_file(p: Path):
    """Yield (line_no, line) tuples that violate the contract."""
    violations = []
    try:
        text = p.read_text()
    except Exception:
        return violations
    for i, line in enumerate(text.splitlines(), start=1):
        if any(p.search(line) for p in FORBIDDEN_PATTERNS):
            if _is_legit(line):
                continue
            violations.append((i, line.strip()))
    return violations


def test_no_direct_wallclock_in_s1_s2():
    """Scan S1/S2 code for direct datetime.now() / date.today() calls.

    All semantic-time calls should go through servers/clock.py
    (brain_now or conversation_now). Bookkeeping uses are tagged as
    LEGITIMATE_USES above.
    """
    failures = []
    for prot in PROTECTED_DIRS:
        for root, dirs, files in os.walk(ROOT / prot):
            # Skip __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('__')]
            for f in files:
                if not f.endswith('.py'):
                    continue
                rel = Path(root, f).relative_to(ROOT)
                if str(rel) in EXEMPT_FILES:
                    continue
                violations = _scan_file(Path(root, f))
                for lineno, line in violations:
                    failures.append(f'{rel}:{lineno}: {line}')
    assert not failures, (
        'Clock contract violation: direct wall-clock call in S1/S2 code.\n'
        'Route through servers/clock.py (brain_now or conversation_now).\n'
        'If genuinely bookkeeping (trace timestamp etc), tag the line with '
        '"# clock-ok" or add a LEGITIMATE_USES pattern.\n\n'
        + '\n'.join(failures)
    )


def test_clock_functions_return_tz_aware():
    """brain_now and conversation_now must always return tz-aware datetimes."""
    from servers.clock import brain_now, conversation_now
    assert brain_now().tzinfo is not None
    assert conversation_now().tzinfo is not None
    assert conversation_now(messages=[]).tzinfo is not None


def test_conversation_now_parses_replay_prefix():
    """The eval replay's '[Current date: YYYY/MM/DD ...]' prefix must be
    recognized — this is the load-bearing case for the eval path."""
    from servers.clock import conversation_today
    import datetime as dt
    msgs = [
        {'role': 'user',
         'content': '[Current date: 2023/03/19 (Sun) 07:16]\n\nI did a 5K today'},
    ]
    assert conversation_today(messages=msgs) == dt.date(2023, 3, 19)


def test_conversation_now_uses_latest_user_message():
    """When multiple user messages have prefixes, the most RECENT one wins."""
    from servers.clock import conversation_today
    import datetime as dt
    msgs = [
        {'role': 'user', 'content': '[Current date: 2023/03/19]\n\nfirst'},
        {'role': 'assistant', 'content': 'ok'},
        {'role': 'user', 'content': '[Current date: 2023/03/26]\n\nsecond'},
    ]
    assert conversation_today(messages=msgs) == dt.date(2023, 3, 26)


def test_conversation_now_falls_back_to_brain_now():
    """Without a replay prefix and without session_started_at, falls back to
    operator wall-clock (matches brain_now)."""
    from servers.clock import conversation_now, brain_now
    c = conversation_now(messages=[{'role': 'user', 'content': 'hello'}])
    n = brain_now()
    # Within a few seconds of each other — exact equality is impossible
    delta_s = abs((c - n).total_seconds())
    assert delta_s < 5.0, f'fallback should match brain_now ± 5s; delta {delta_s}s'
