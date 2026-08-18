"""Temporal scout — algorithmic date extraction using dateparser.

Pure algorithmic in v1 (no LLM). Runs six extraction passes against each
turn's text, unions results, deduplicates by (phrase, ISO date):

  - dateparser.search_dates + phrase-shape filter (digits, months, etc.)
  - regex supplements for patterns dateparser mis-handles with RELATIVE_BASE:
    weekday+modifier ("last Tuesday"), modifier+unit ("next month"),
    word-number relatives ("three weeks ago"), vague quantifiers
    ("a few weeks ago"), fuzzy anchors ("recently", "a while back")

Each candidate is resolved to an ISO date and given a `precision` flag:
explicit, relative, or approximate. Approximate uses best-guess offsets
so S1S knows the anchor is fuzzy.

Event relations (`before`/`after` between events) are NOT surfaced here —
temporal scout's job stays date anchors only. S1S composes relational
edges from anchor candidates + catalog endpoints. Roadmap in
servers/scales/s1/ARCHITECTURE.md.

Haiku fallback: temporal_prompt.py holds a fallback prompt but it is NOT
wired in v1. Reserved for v2 on truly ambiguous phrases (liturgical
calendar events, seasons with fuzzy resolution).

Scout-specific candidate fields: source_phrase, resolution,
event_description, existing_anchor_id, catalog_tension (v2), precision.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
import time
from typing import Any, Dict, List, Optional

from . import contract as sc
from .base import (
    SCOUT_ERROR_KEY,
    SCOUT_LATENCY_KEY,
    SCOUT_TOKEN_USAGE_KEY,
    SCOUT_WARNING_KEY,
    _log_error,
)

_dateparser_warned = False


def _warn_dateparser_missing():
    """dateparser is a DECLARED runtime dep (requirements.txt); its absence means
    a broken venv, not a benign no-op. Log once — so temporal date extraction
    going dark is VISIBLE instead of silently returning no anchors."""
    global _dateparser_warned
    if not _dateparser_warned:
        _dateparser_warned = True
        logging.getLogger(__name__).error(
            "dateparser not installed — temporal date extraction disabled. It is "
            "a declared runtime dependency; reinstall the venv "
            "(hooks/scripts/ensure-runtime.sh / requirements.txt).")


# ─── Filter vocab ─────────────────────────────────────────────────────────

_TIME_ONLY_RE = re.compile(
    r'^\s*\d+\s*:?\s*\d*\s*(am|pm|a\.?m\.?|p\.?m\.?|hours?|hrs?|minutes?|mins?)\s*$',
    re.IGNORECASE,
)

# Leading prepositions dateparser greedily includes. Strip these BEFORE
# running the time-only check so 'at 7 PM' gets normalized to '7 PM' and
# correctly rejected.
_LEADING_NOISE_RE = re.compile(
    r'^((at|on|in|of|to|from|by|for|with|and|the|a|an)\s+)+',
    re.IGNORECASE,
)

_TEMPORAL_KEYWORDS = frozenset({
    'today', 'yesterday', 'tomorrow', 'tonight', 'now', 'recently',
})

_MONTHS = frozenset({
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
    'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
})

_WEEKDAYS = frozenset({
    'monday', 'tuesday', 'wednesday', 'thursday',
    'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
})

_DATE_MODIFIERS = frozenset({
    'last', 'next', 'this', 'past', 'coming', 'upcoming',
})

_WEEKDAY_MODIFIER_RE = re.compile(
    r'\b(last|next|this|past|coming|upcoming)\s+'
    r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    re.IGNORECASE,
)

_WEEKDAY_NUMBERS = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6,
}

# Spelled-out numbers so "three weeks ago" works. Dateparser's search_dates
# only catches digit relatives reliably — word relatives often get dropped
# or returned as a whole-sentence match with noise.
_NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12,
}

# Vague quantifiers — no precise count. Scout emits precision='approximate'
# so S1S knows the anchor is fuzzy. The midpoint convention matches how
# operators use the words in practice.
_VAGUE_QUANTIFIERS = {
    'a few': 3,
    'a couple': 2,
    'a couple of': 2,
    'several': 5,
    'many': 8,
    'numerous': 10,
    'a lot of': 10,
}

# Unresolved fuzzy anchors get a best-guess offset. 'recently' is closer
# (~1 week) than 'a while ago' (~4 weeks). These are defaults — S1S can
# override based on surrounding context if we ever add LLM fallback.
_FUZZY_ANCHORS = {
    'recently': ('week', 1),
    'lately': ('week', 1),
    'a while ago': ('week', 4),
    'a while back': ('week', 4),
    'some time ago': ('week', 4),
    'some time back': ('week', 4),
    'some time earlier': ('week', 4),
}

# Map coarse units to timedeltas. Months/years are approximate — good
# enough for fuzzy anchors that are already imprecise.
_UNIT_TO_DELTA = {
    'day': _dt.timedelta(days=1),
    'week': _dt.timedelta(weeks=1),
    'month': _dt.timedelta(days=30),
    'year': _dt.timedelta(days=365),
}

# "three weeks ago" style — word-count + unit + "ago"
_WORD_RELATIVE_RE = re.compile(
    r'\b(' + '|'.join(_NUMBER_WORDS.keys()) + r')\s+'
    r'(day|week|month|year)s?\s+ago\b',
    re.IGNORECASE,
)

# "a few weeks ago" / "several months ago" / "a couple of days ago"
_VAGUE_RELATIVE_RE = re.compile(
    r'\b(a few|a couple of|a couple|several|many|numerous|a lot of)\s+'
    r'(day|week|month|year)s?\s+ago\b',
    re.IGNORECASE,
)

# "recently", "a while ago", "lately" — no count, approximate anchor
_FUZZY_ANCHOR_RE = re.compile(
    r'\b(recently|lately|'
    r'a while (?:ago|back)|'
    r'some time (?:ago|back|earlier))\b',
    re.IGNORECASE,
)

# "last month", "next year", "this week" — modifier + unit
_MODIFIER_UNIT_RE = re.compile(
    r'\b(last|next|this|past|coming|upcoming)\s+'
    r'(day|week|month|year)\b',
    re.IGNORECASE,
)

# Absolute "Month Day[, Year]" phrases — dateparser.search_dates misses these
# in longer sentences. We catch them with a targeted regex and parse each
# match individually.
#
# (?!\d) after the day group prevents greedy matches like "May 2021" being
# read as "May 20" (day=20, year dropped). Requires day to be followed by
# a non-digit (comma, space, punctuation, or end-of-string).
_ABSOLUTE_MONTH_DAY_RE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|'
    r'oct|nov|dec)\s+'
    r'(\d{1,2})(?!\d)(?:st|nd|rd|th)?'
    r'(?:,?\s+(\d{4}))?\b',
    re.IGNORECASE,
)

# Relational markers that connect one event's date to a different event.
# When these appear in an anchor's event_description, S1S should search
# its catalog for the reference event and compose a cross-event edge
# (Allen interval algebra: meets/before/after/during/...). The scout
# does NOT resolve the reference event — that's S1S's semantic match job.
#
# Longer forms (e.g. "just before") are matched first; shorter fallbacks
# ("before", "after") apply when no qualifier is present.
_RELATIONAL_MARKER_RE = re.compile(
    r'\b(' +
    # Adjacency (maps to Allen's meets/met-by at S1S level)
    r'just before|just after|right before|right after|shortly before|'
    r'shortly after|moments before|moments after|immediately before|'
    r'immediately after|'
    # Containment / during
    r'during|while|when|at the time|in the middle of|'
    # Priority / sequence
    r'prior to|subsequent to|following|preceding|'
    # Bare sequence — last resort; may false-match
    r'before|after'
    r')\b',
    re.IGNORECASE,
)


def _resolve_modifier_weekday(
    modifier: str, weekday: str, base: _dt.datetime,
) -> Optional[_dt.datetime]:
    """Resolve 'last/next/this/... + weekday' to a concrete date.

    Dateparser.parse returns None for these phrases when a RELATIVE_BASE
    is set (observed on dateparser 1.4.0), so we handle the calendar math
    locally.

    - last/past Tuesday: most recent Tuesday strictly before base
    - next/coming/upcoming Tuesday: first Tuesday strictly after base
    - this Tuesday: Tuesday of the current week (nearest past, or today
      if today IS Tuesday)
    """
    wd = _WEEKDAY_NUMBERS.get(weekday.lower())
    if wd is None:
        return None
    base_wd = base.weekday()
    mod = modifier.lower()
    if mod in ('last', 'past'):
        delta = (base_wd - wd) % 7
        if delta == 0:
            delta = 7
        return base - _dt.timedelta(days=delta)
    if mod in ('next', 'coming', 'upcoming'):
        delta = (wd - base_wd) % 7
        if delta == 0:
            delta = 7
        return base + _dt.timedelta(days=delta)
    if mod == 'this':
        delta = (base_wd - wd) % 7
        return base - _dt.timedelta(days=delta)
    return None


def _resolve_modifier_unit(
    modifier: str, unit: str, base: _dt.datetime,
) -> Optional[_dt.datetime]:
    """Resolve 'last/next/this + day/week/month/year' to a concrete date.

    Conventions:
    - last week / last month / last year: one unit before base
    - next week / next month / next year: one unit after base
    - this week / this month: base itself (the current period)
    """
    delta = _UNIT_TO_DELTA.get(unit.lower())
    if delta is None:
        return None
    mod = modifier.lower()
    if mod in ('last', 'past'):
        return base - delta
    if mod in ('next', 'coming', 'upcoming'):
        return base + delta
    if mod == 'this':
        return base
    return None

# Trailing prepositions/articles that dateparser often grabs greedily.
_TRAILING_NOISE_RE = re.compile(
    r'[\s,]+((at|on|in|of|to|from|by|for|with|and|the|a|an)(\s+|$))+$',
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


# ─── Filtering ────────────────────────────────────────────────────────────


# A digit token only counts toward "is a date" if it carries date SHAPE: a
# 4-digit year, an ISO/slash numeric date, an ordinal day, or a digit
# relative ("5 days ago"). A bare integer (line numbers, %s, PIDs, arxiv
# ids, section numbers) is NOT a date — the old gate accepted any
# digit-bearing token, which is ~73% of the scout's SKIPPED triage tax.
_DATE_SHAPE_RE = re.compile(
    r'\b(?:19|20)\d{2}\b'                       # 4-digit year 1900-2099
    r'|\b\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?\b'  # 2026-06-28, 5/28/2026, 6/28
    r'|\b\d{1,2}(?:st|nd|rd|th)\b'              # 5th, 21st
    r'|\b\d+\s*(?:day|week|month|year)s?\b',    # "5 days", "2 weeks" relatives
    re.IGNORECASE,
)


def _looks_like_date(phrase: str) -> bool:
    """Gate a dateparser match against noise.

    Accepts phrases carrying a date-shaped token (year / numeric date /
    ordinal / digit-relative), a temporal keyword, a month name, or a
    weekday paired with a modifier. Rejects time-of-day phrases, bare
    integers, and bare function words.
    """
    p = phrase.lower().strip()
    if not p:
        return False
    if _TIME_ONLY_RE.match(p):
        return False
    words = set(re.split(r'[\s,]+', p))
    has_date_shape = bool(_DATE_SHAPE_RE.search(p))
    has_keyword = bool(words & _TEMPORAL_KEYWORDS)
    has_month = bool(words & _MONTHS)
    has_weekday = bool(words & _WEEKDAYS)
    has_modifier = bool(words & _DATE_MODIFIERS)
    return has_date_shape or has_keyword or has_month or (has_weekday and has_modifier)


def _trim_trailing_noise(phrase: str) -> str:
    """Strip trailing prepositions/articles dateparser grabbed greedily.

    "2 weeks ago at the" → "2 weeks ago"
    Idempotent — applies the regex until no more match.
    """
    s = phrase.strip()
    for _ in range(5):  # cap iterations; phrases are short
        new = _TRAILING_NOISE_RE.sub('', s).rstrip(' ,')
        if new == s:
            break
        s = new
    return s


def _trim_leading_noise(phrase: str) -> str:
    """Strip leading prepositions/articles dateparser prepended ('at 7 PM')."""
    return _LEADING_NOISE_RE.sub('', phrase.strip(), count=1).strip()


def _parse_current_date(current_date: str) -> _dt.datetime:
    """Accepts 'YYYY-MM-DD' or ISO with time. Returns datetime at 00:00."""
    d = current_date.strip()[:10]  # "2026-04-23"
    year, month, day = d.split('-')
    return _dt.datetime(int(year), int(month), int(day))


def _resolution_summary(base: _dt.datetime, resolved: _dt.datetime) -> str:
    """Short human-readable resolution like 'base=2026-04-23, offset=-14d'."""
    delta = (resolved.date() - base.date()).days
    sign = '' if delta == 0 else ('+' if delta > 0 else '')
    return f'base={base.date().isoformat()}, offset={sign}{delta}d'


def _specificity_score(phrase: str, is_explicit_date: bool,
                       precision: str = 'relative') -> int:
    """Rank candidates: absolute dates > relative-precise > relative-vague.

    Higher is better. Used to pick the top N when over cap. Uses the
    extraction pass's precision flag as the primary signal, with phrase
    pattern bumps for well-formed relatives.
    """
    if is_explicit_date:
        return 100
    p = phrase.lower().strip()
    # Approximate/fuzzy anchors rank lowest
    if precision == 'approximate':
        return 25
    # Precise relatives get a pattern-based bump
    if re.search(r'\d+\s+(day|week|month|year)s?\s+ago', p):
        return 65
    # "today", "yesterday", "tomorrow" — anchor words
    if any(kw == p or p.startswith(kw + ' ') or p.endswith(' ' + kw)
           for kw in _TEMPORAL_KEYWORDS):
        return 60
    # Weekday with modifier — moderately precise
    if _WEEKDAY_MODIFIER_RE.search(phrase):
        return 55
    # Unit with modifier ("next month")
    if _MODIFIER_UNIT_RE.search(phrase):
        return 50
    # Word-number relative ("three weeks ago")
    if _WORD_RELATIVE_RE.search(phrase):
        return 45
    return 35


# ─── Event context extraction ─────────────────────────────────────────────


def _extract_event_sentence(turn_text: str, phrase: str,
                            max_chars: int = 100) -> str:
    """Return the sentence containing the phrase, trimmed to max_chars.

    Best-effort: splits turn_text on sentence terminators, picks the first
    sentence containing a case-insensitive substring match of the phrase.
    Falls back to the turn text itself (truncated) if no sentence boundary.
    """
    if not phrase or not turn_text:
        return ''
    p_lower = phrase.lower().strip()
    for sentence in _SENTENCE_SPLIT_RE.split(turn_text):
        if p_lower in sentence.lower():
            s = sentence.strip()
            return s[:max_chars]
    # No sentence boundary — use truncated turn text
    return turn_text.strip()[:max_chars]


def _detect_relational_marker(event_description: str) -> Optional[str]:
    """Return the relational marker present in event_description, or None.

    When the sentence contains a temporal connector like "just before" or
    "during", the event's date anchors ONE event but the sentence also
    references ANOTHER event — a candidate for a cross-event edge.

    Scout surfaces the raw marker. S1S parses the context around it to
    find the reference event in its catalog and composes the appropriate
    edge (Allen's `meets`, `before`, `during`, etc.).
    """
    if not event_description:
        return None
    m = _RELATIONAL_MARKER_RE.search(event_description)
    return m.group(0).lower() if m else None


# ─── Catalog lookup ───────────────────────────────────────────────────────


def _find_existing_time_anchor(catalog_nodes: List[Dict[str, Any]],
                               iso_date: str) -> Optional[str]:
    """Return node_id of a catalog time_anchor matching the ISO date, or None."""
    for node in catalog_nodes or []:
        if node.get('type') != 'time_anchor':
            continue
        title = (node.get('title') or '').strip()
        if title[:10] == iso_date:  # accept title prefix match (forgiving)
            return node.get('id')
    return None


# ─── Phrase extraction (core algorithm) ───────────────────────────────────


def _extract_candidates_from_text(
    text: str,
    base_date: _dt.datetime,
) -> List[Dict[str, Any]]:
    """Extract date candidates from one block of text.

    Returns list of:
        {phrase_clean, phrase_raw, resolved: datetime,
         iso_date: str, is_explicit: bool}

    Deduped on (phrase_clean, iso_date). Sorted by specificity descending.
    """
    if not text or not text.strip():
        return []

    try:
        from dateparser.search import search_dates
    except ImportError:
        _warn_dateparser_missing()
        return []

    settings = {
        'RELATIVE_BASE': base_date,
        'PREFER_DATES_FROM': 'past',
        # Drop dateparser's 'timestamp' parser: it reads the first 10 digits
        # of any number as a Unix epoch, turning line numbers, PIDs, and
        # arxiv ids into "dates" — the dominant scout false-positive source.
        # Keep the date-shaped parsers (default list minus 'timestamp').
        'PARSERS': ['relative-time', 'custom-formats', 'absolute-time'],
    }

    raw = search_dates(text, settings=settings) or []

    seen = set()
    results: List[Dict[str, Any]] = []

    # Primary: dateparser's extraction, filtered to date-looking phrases.
    # Dateparser's search_dates greedily grabs surrounding prepositions
    # ("at 7 PM" instead of "7 PM", "2 weeks ago at the" with trailing
    # noise). Trim both sides before the filter so 'at 7 PM' gets
    # normalized to '7 PM' and correctly rejected as time-only.
    for phrase, resolved in raw:
        cleaned = _trim_leading_noise(_trim_trailing_noise(phrase))
        if not cleaned or not _looks_like_date(cleaned):
            continue
        iso = resolved.date().isoformat()
        key = (cleaned.lower(), iso)
        if key in seen:
            continue
        seen.add(key)
        # Heuristic: explicit if contains a month name or a 4-digit year
        p_low = cleaned.lower()
        is_explicit = any(m in p_low for m in _MONTHS) or bool(
            re.search(r'\b\d{4}\b', cleaned))
        results.append({
            'phrase_raw': phrase,
            'phrase_clean': cleaned,
            'resolved': resolved,
            'iso_date': iso,
            'is_explicit': is_explicit,
            'precision': 'explicit' if is_explicit else 'relative',
        })

    def _emit(phrase: str, resolved: Optional[_dt.datetime],
              precision: str, is_explicit: bool = False):
        """Dedup-add helper for the supplementary passes below."""
        if resolved is None:
            return
        iso = resolved.date().isoformat()
        key = (phrase.lower().strip(), iso)
        if key in seen:
            return
        seen.add(key)
        results.append({
            'phrase_raw': phrase,
            'phrase_clean': phrase,
            'resolved': resolved,
            'iso_date': iso,
            'is_explicit': is_explicit,
            'precision': precision,
        })

    # Supplements for patterns dateparser mis-handles with RELATIVE_BASE set:
    # it returns None for modifier+weekday / modifier+unit, misses
    # "Month Day" in complex sentences, and is hit-or-miss for word-number
    # relatives. Each pass resolves locally or via dateparser.parse on the
    # isolated phrase (parse works where search_dates fails).

    # Absolute "Month Day"
    try:
        import dateparser as _dp
    except ImportError:
        _warn_dateparser_missing()
        _dp = None
    if _dp is not None:
        for m in _ABSOLUTE_MONTH_DAY_RE.finditer(text):
            phrase = m.group(0)
            try:
                resolved = _dp.parse(phrase, settings={'RELATIVE_BASE': base_date})
            except Exception:
                resolved = None
            _emit(phrase, resolved, 'explicit', is_explicit=True)

    for m in _WEEKDAY_MODIFIER_RE.finditer(text):
        _emit(m.group(0),
              _resolve_modifier_weekday(m.group(1), m.group(2), base_date),
              'relative')

    for m in _MODIFIER_UNIT_RE.finditer(text):
        _emit(m.group(0),
              _resolve_modifier_unit(m.group(1), m.group(2), base_date),
              'relative')

    for m in _WORD_RELATIVE_RE.finditer(text):
        count = _NUMBER_WORDS.get(m.group(1).lower(), 1)
        delta = _UNIT_TO_DELTA[m.group(2).lower()]
        _emit(m.group(0), base_date - count * delta, 'relative')

    for m in _VAGUE_RELATIVE_RE.finditer(text):
        count = _VAGUE_QUANTIFIERS.get(m.group(1).lower(), 3)
        delta = _UNIT_TO_DELTA[m.group(2).lower()]
        _emit(m.group(0), base_date - count * delta, 'approximate')

    for m in _FUZZY_ANCHOR_RE.finditer(text):
        unit, count = _FUZZY_ANCHORS.get(
            m.group(0).lower().strip(), ('week', 4))
        _emit(m.group(0), base_date - count * _UNIT_TO_DELTA[unit],
              'approximate')

    return results


# ─── Main entry ───────────────────────────────────────────────────────────


def run_temporal_scout(
    brain,
    turns: List[Dict[str, Any]],
    catalog_nodes: Optional[List[Dict[str, Any]]] = None,
    surfaced_node_ids_by_turn: Optional[Dict[str, List[str]]] = None,
    current_date: Optional[str] = None,
    log_fn=None,
) -> Dict[str, Any]:
    """Algorithmic temporal scout.

    Args:
        brain: for interaction lookup + error logging (same as LLM scouts).
        turns: list of turn dicts. Each must carry at least:
               {'turn_id': 't<N>', 'role': 'user'|'assistant', 'text': str}
               turn_id shape is caller-defined ('t1', 'turn-0', ...).
        catalog_nodes: list of catalog node dicts for existing_anchor lookup.
                       Each must at minimum carry id, type, title.
        surfaced_node_ids_by_turn: reserved for catalog_tension in v2.
        current_date: ISO date string 'YYYY-MM-DD' used as base for relative
                      resolution. Defaults to today if None.
        log_fn: optional line-logger; receives f'[s1_scout_temporal] ...'.

    Returns:
        A scout envelope dict matching contract.validate_scout_output:
        {scout, category_statement, candidates[], scanned, _usage,
         _latency_ms, _errors, _warnings}

    Never raises. All failures log to brain_errors and return a stub
    envelope. Dateparser import failure is also graceful.
    """

    def _log(msg):
        if log_fn:
            log_fn(f'[s1_scout_temporal] {msg}')

    scout_name = 'temporal'
    stub = {
        'scout': scout_name,
        'category_statement': '',
        'candidates': [],
        'scanned': {'turns': len(turns or []), 'date_phrases_found': 0,
                    'passed_threshold': 0},
        SCOUT_TOKEN_USAGE_KEY: {},  # no LLM in v1 — empty
        SCOUT_LATENCY_KEY: 0,
        SCOUT_ERROR_KEY: [],
        SCOUT_WARNING_KEY: [],
    }

    t0 = time.time()

    # 1. Interaction config — resolved (override overlaid on code default)
    params = brain.get_interaction_config(sc.interaction_name(scout_name))
    category_statement = params['category_statement']
    max_candidates = int(params['max_candidates'])
    stub['category_statement'] = category_statement

    # 2. Base date — fall back to operator wall-clock if caller didn't pass
    # one. Production: this is correct. Eval: caller MUST pass the
    # conversation date via conversation_now(messages) to avoid resolving
    # historical "today/yesterday" against real now. See servers/clock.py.
    if current_date is None:
        from servers.clock import brain_today
        current_date = brain_today().isoformat()
    try:
        base_date = _parse_current_date(current_date)
    except Exception as e:
        msg = f'invalid current_date {current_date!r}: {e}'
        _log(msg)
        _log_error(brain, scout_name, 'bad_current_date', msg)
        stub[SCOUT_ERROR_KEY].append({'type': 'bad_current_date', 'msg': msg})
        return stub

    # 3. Extract candidates per turn
    try:
        candidates: List[Dict[str, Any]] = []
        date_phrases_found = 0
        for turn in (turns or []):
            tid = turn.get('turn_id') or turn.get('id') or ''
            text = turn.get('text') or turn.get('content') or ''
            # source_role attribution — 'user' (operator-stated dates) vs
            # 'assistant' (paraphrases / fabricated dates the assistant
            # added). Without this tag, S1E can't distinguish operator-
            # attributed temporal anchors from assistant retrospective
            # inferences. The Universal Studios regression (gpt4_85da3956,
            # 2026-05-13 comparison) was exactly this: the assistant said
            # "you went three weeks ago" paraphrasing the user's "just got
            # back", and the encoder treated that as gospel. Tagging here
            # lets the encoder choose conversation_now over assistant-only
            # dates for proximal phrases.
            role = turn.get('role') or turn.get('speaker') or ''
            if not text or not tid:
                continue
            extracted = _extract_candidates_from_text(text, base_date)
            date_phrases_found += len(extracted)
            for ext in extracted:
                existing_id = _find_existing_time_anchor(
                    catalog_nodes or [], ext['iso_date'])
                event = _extract_event_sentence(text, ext['phrase_clean'])
                marker = _detect_relational_marker(event)
                candidates.append({
                    'handle': ext['iso_date'],
                    'evidence_quote': event,  # first pass — sentence as evidence
                    'evidence_turns': [tid],
                    'why_candidate': _why_candidate(ext, existing_id, marker),
                    'source_phrase': ext['phrase_clean'][:50],
                    'source_role': role,  # 'user' | 'assistant' | ''
                    'evidence_roles': [role] if role else [],
                    'resolution': _resolution_summary(base_date, ext['resolved']),
                    'event_description': event,
                    'existing_anchor_id': existing_id,
                    'catalog_tension': [],  # v2
                    'precision': ext.get('precision', 'relative'),
                    'relational_marker': marker,  # None unless sentence has one
                    '_specificity': _specificity_score(
                        ext['phrase_clean'], ext['is_explicit'],
                        ext.get('precision', 'relative')),
                })
    except Exception as e:
        msg = f'extraction error: {type(e).__name__}: {e}'
        _log(msg)
        _log_error(brain, scout_name, 'extraction_error', msg)
        stub[SCOUT_ERROR_KEY].append({'type': 'extraction_error', 'msg': msg})
        return stub

    # 4. Dedup by ISO date — session-date tags prepended to every user turn
    # produce N identical candidates for one anchor. Keep the highest-
    # specificity version per date, merge evidence_turns + evidence_roles
    # to preserve where the date was referenced and by whom. If the same
    # date appears in BOTH a user and an assistant turn, source_role
    # promotes to 'user' (operator attribution wins) — even when the
    # highest-specificity wording came from the assistant turn — because
    # the encoder treats user attribution as authoritative.
    by_iso: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        key = c['handle']  # handle IS the ISO date
        if key in by_iso:
            existing = by_iso[key]
            # Merge turn references
            existing['evidence_turns'] = list(dict.fromkeys(
                existing['evidence_turns'] + c['evidence_turns']))
            # Merge role references
            existing['evidence_roles'] = list(dict.fromkeys(
                (existing.get('evidence_roles') or []) +
                (c.get('evidence_roles') or [])))
            if c['_specificity'] > existing['_specificity']:
                # Swap in the better candidate but preserve merged turns/roles
                turns_merged = existing['evidence_turns']
                roles_merged = existing['evidence_roles']
                by_iso[key] = dict(c)
                by_iso[key]['evidence_turns'] = turns_merged
                by_iso[key]['evidence_roles'] = roles_merged
            # User attribution wins for source_role even when the
            # higher-specificity wording came from a different role.
            if 'user' in by_iso[key].get('evidence_roles', []):
                by_iso[key]['source_role'] = 'user'
        else:
            by_iso[key] = dict(c)
    deduped = list(by_iso.values())

    # 5. Rank + cap
    deduped.sort(key=lambda c: (-c['_specificity'], c['handle']))
    capped = deduped[:max_candidates]
    for c in capped:
        c.pop('_specificity', None)  # internal field — don't emit

    stub['candidates'] = capped
    stub['scanned'] = {
        'turns': len(turns or []),
        'date_phrases_found': date_phrases_found,
        'passed_threshold': len(capped),
    }

    # 6. Validate through the same contract as LLM scouts
    envelope = {
        'scout': scout_name,
        'category_statement': category_statement,
        'candidates': capped,
        'scanned': stub['scanned'],
    }
    ok, normalized, errors, warnings = sc.validate_scout_output(envelope, scout_name)

    elapsed_ms = int((time.time() - t0) * 1000)
    normalized[SCOUT_TOKEN_USAGE_KEY] = {}
    normalized[SCOUT_LATENCY_KEY] = elapsed_ms
    normalized[SCOUT_ERROR_KEY] = []
    normalized[SCOUT_WARNING_KEY] = list(warnings)

    if not ok:
        _log(f'validation failed: {errors}')
        _log_error(brain, scout_name, 'schema_invalid', '; '.join(errors))
        normalized[SCOUT_ERROR_KEY] = [
            {'type': 'schema_invalid', 'msg': e} for e in errors
        ]
        return normalized

    _log(f'ok — {len(capped)} candidates from {date_phrases_found} '
         f'phrases in {elapsed_ms}ms (scanned {len(turns or [])} turns)')
    return normalized


def _why_candidate(ext: Dict[str, Any], existing_id: Optional[str],
                   relational_marker: Optional[str] = None) -> str:
    """One-line rationale per candidate (≤150 chars)."""
    marker_hint = ''
    if relational_marker:
        marker_hint = (
            f' Event_description has "{relational_marker}" — '
            'consider cross-event edge if ref in catalog.')
    if existing_id:
        return (
            f'Date resolves to existing time_anchor {existing_id[:8]} — '
            f'reuse, don\'t duplicate.{marker_hint}')[:150]
    if ext['is_explicit']:
        return ('Explicit date reference — create time_anchor bridge.'
                + marker_hint)[:150]
    return ('Relative date resolved to ISO — create new time_anchor bridge.'
            + marker_hint)[:150]


__all__ = [
    'run_temporal_scout',
]
