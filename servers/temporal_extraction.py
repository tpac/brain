"""Temporal interval extraction from nodes (and edges, pending design).

Called by the embed_queue worker at write time. Scans text fields for
explicit-year date references and converts each into a (start_ts, end_ts)
interval based on detected precision. Intervals are written to the
`entity_dates` table for `recall_by_time` queries.

Storage convention:
  - Inclusive Unix-second intervals.
  - "2023"          -> (year-start, year-end)
  - "Q1 2024"       -> (quarter-start, quarter-end)
  - "May 2023"      -> (month-start, month-end)
  - "May 22, 2023"  -> (day-start, day-end)
  - "2023-05"       -> (month-start, month-end)
  - "2023-05-22"    -> (day-start, day-end)

Design: regex-first, year-required. We deliberately don't use
`dateparser.search` for extraction because it defaults missing year to
the current year — bad for historical-text scanning. NL date parsing
("last month", "around March") lives on the *query* side (recall_by_time
input), not the extraction side.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple


# ── Regex patterns ──────────────────────────────────────────────────────
#
# Date references are matched directly (no `dateparser.search`) so we can
# guarantee precision detection and avoid dateparser's "fill missing year
# with current year" behavior. All patterns require explicit year context.

_MONTH_NAME = (
    r'(?:january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|'
    r'oct|nov|dec)'
)

_PATTERN_QUARTER = re.compile(
    r'\b(q[1-4])\s+(\d{4})\b', re.IGNORECASE
)
_PATTERN_ISO_DAY = re.compile(
    r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b'
)
_PATTERN_ISO_MONTH = re.compile(
    r'\b(\d{4})[-/](\d{1,2})\b(?![-/]\d)'   # avoid matching YYYY-MM-DD
)
_PATTERN_MONTH_DAY_YEAR = re.compile(
    rf'\b({_MONTH_NAME})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b',
    re.IGNORECASE
)
_PATTERN_MONTH_YEAR = re.compile(
    rf'\b({_MONTH_NAME})\s+(\d{{4}})\b',
    re.IGNORECASE
)
# Range: "between Feb and April 2023" / "from Jan to March 2023" /
# "January-March 2023" / "Jan-March 2023" / "January through March 2023".
# Same-year only; both months share the trailing year.
# Separator spacing: words (to/and/through) require whitespace on both
# sides; dashes (-, –, —) allow either spacing — matches compact forms
# like "Jan-March 2023" as well as "Jan - March 2023".
_PATTERN_MONTH_RANGE = re.compile(
    rf'\b(?:between\s+|from\s+)?({_MONTH_NAME})'
    rf'(?:\s*[-–—]\s*|\s+(?:to|and|through)\s+)'
    rf'({_MONTH_NAME})\s+(\d{{4}})\b',
    re.IGNORECASE
)

_MONTH_NAME_TO_NUM = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7,
    'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12,
    'december': 12,
}


def _interval_for_year(year: int) -> Tuple[int, int]:
    s = datetime(year, 1, 1, tzinfo=timezone.utc)
    e = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def _interval_for_quarter(year: int, q: int) -> Tuple[int, int]:
    start_month = (q - 1) * 3 + 1
    end_month = q * 3
    s = datetime(year, start_month, 1, tzinfo=timezone.utc)
    last_day = monthrange(year, end_month)[1]
    e = datetime(year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def _interval_for_month(year: int, month: int) -> Tuple[int, int]:
    last_day = monthrange(year, month)[1]
    s = datetime(year, month, 1, tzinfo=timezone.utc)
    e = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def _interval_for_day(year: int, month: int, day: int) -> Tuple[int, int]:
    s = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
    e = datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def extract_intervals_from_text(text: Optional[str]) -> List[Tuple[int, int, str]]:
    """Find date references in `text` and return interval tuples.

    Returns: [(start_ts, end_ts, raw_text), ...]. Empty list if no dates
    or if `text` is empty.

    Requires explicit year context — dateless month names ("May") and
    bare day numbers are skipped to avoid current-year fallback errors.

    Deduplicates by (start_ts, end_ts) — multiple raw spellings of the
    same interval collapse to the first match.

    Detection patterns (in priority order):
      1. Q[1-4] YYYY                  -> quarter
      2. YYYY-MM-DD or YYYY/MM/DD     -> day
      3. YYYY-MM (no day)             -> month
      3b. MonthName .. MonthName YYYY -> range (same-year, month-precision span)
      4. MonthName DD, YYYY           -> day
      5. MonthName YYYY               -> month

    Standalone-year matching (e.g., bare "2024") is intentionally NOT
    implemented in v1. Bare 4-digit numbers in technical text (`2000 chars`,
    `4096 tokens`, `2048 ms`) generate too many false positives. Every
    year-bearing interval still flows through patterns 1-5 above, all of
    which require explicit month or quarter context.
    """
    if not text:
        return []
    out: List[Tuple[int, int, str]] = []
    seen: set = set()
    consumed_spans: List[Tuple[int, int]] = []  # char ranges already matched

    def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def _is_consumed(span: Tuple[int, int]) -> bool:
        return any(_spans_overlap(span, c) for c in consumed_spans)

    def _add(span: Tuple[int, int], start_ts: int, end_ts: int, raw: str) -> None:
        key = (start_ts, end_ts)
        if key in seen:
            return
        seen.add(key)
        consumed_spans.append(span)
        out.append((start_ts, end_ts, raw.strip()))

    # 1. Quarter
    for m in _PATTERN_QUARTER.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            q = int(m.group(1)[1])
            year = int(m.group(2))
            s, e = _interval_for_quarter(year, q)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 2. ISO day (must come before ISO month — overlap guard handles it)
    for m in _PATTERN_ISO_DAY.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            s, e = _interval_for_day(year, month, day)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 3. ISO month (YYYY-MM without trailing -DD)
    for m in _PATTERN_ISO_MONTH.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            year, month = int(m.group(1)), int(m.group(2))
            if not (1 <= month <= 12):
                continue
            s, e = _interval_for_month(year, month)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 3b. MonthName .. MonthName YYYY  (range, same year)
    for m in _PATTERN_MONTH_RANGE.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            mname1 = m.group(1).lower()
            mname2 = m.group(2).lower()
            month1 = _MONTH_NAME_TO_NUM.get(mname1)
            month2 = _MONTH_NAME_TO_NUM.get(mname2)
            year = int(m.group(3))
            if not (month1 and month2):
                continue
            # Require forward ordering (skip cross-year guesses for now).
            if month1 > month2:
                continue
            s_start = datetime(year, month1, 1, tzinfo=timezone.utc)
            last_day = monthrange(year, month2)[1]
            s_end = datetime(year, month2, last_day, 23, 59, 59,
                              tzinfo=timezone.utc)
            _add(m.span(), int(s_start.timestamp()),
                  int(s_end.timestamp()), m.group(0))
        except Exception:
            continue

    # 4. MonthName DD, YYYY
    for m in _PATTERN_MONTH_DAY_YEAR.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            mname = m.group(1).lower()
            month = _MONTH_NAME_TO_NUM.get(mname)
            if not month:
                continue
            day, year = int(m.group(2)), int(m.group(3))
            s, e = _interval_for_day(year, month, day)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 5. MonthName YYYY
    for m in _PATTERN_MONTH_YEAR.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            mname = m.group(1).lower()
            month = _MONTH_NAME_TO_NUM.get(mname)
            if not month:
                continue
            year = int(m.group(2))
            s, e = _interval_for_month(year, month)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    return out


# ── Node-level extraction ────────────────────────────────────────────────

# KV keys that explicitly carry temporal anchors. Highest-confidence source.
# Values here are parsed as standalone date strings, not searched.
_EXPLICIT_TEMPORAL_KV_KEYS = ('event_time', 'event_date', 'when')

# KV keys with free-text content that may contain date references. Scanned
# the same way as title/content. Conservative list — expand if needed.
_SCANNED_TEXT_KV_KEYS = (
    'situation', 'reasoning', 'source_context',
    'user_raw_quote', 'anchor_raw_quote',
)


def extract_node_intervals(
    title: Optional[str],
    content: Optional[str],
    kv_pairs: Iterable[Tuple[str, str]],
) -> List[Tuple[int, int, str, str]]:
    """Extract all temporal intervals for one node.

    Returns: [(start_ts, end_ts, extraction_source, raw_text), ...]

    `extraction_source` is qualified — 'node.title', 'node.content',
    'node.kv:event_time', 'node.kv_scan:situation', etc.

    Deduplicates within a node by (start_ts, end_ts, extraction_source)
    — same interval found by two distinct sources keeps both rows; same
    source emitting the same interval twice keeps one.
    """
    out: List[Tuple[int, int, str, str]] = []
    seen_per_source: set = set()

    def _emit(start_ts: int, end_ts: int, source: str, raw: str) -> None:
        key = (start_ts, end_ts, source)
        if key in seen_per_source:
            return
        seen_per_source.add(key)
        out.append((start_ts, end_ts, source, raw))

    def _scan(text: Optional[str], source: str) -> None:
        for s, e, raw in extract_intervals_from_text(text):
            _emit(s, e, source, raw)

    _scan(title, 'node.title')
    _scan(content, 'node.content')

    kv_dict = dict(kv_pairs)
    # Explicit temporal KV keys — value is a canonical date string.
    for k in _EXPLICIT_TEMPORAL_KV_KEYS:
        v = kv_dict.get(k)
        if not v:
            continue
        for s, e, raw in extract_intervals_from_text(v):
            _emit(s, e, f'node.kv:{k}', raw)

    # Free-text KV keys — scan for embedded date references.
    for k in _SCANNED_TEXT_KV_KEYS:
        v = kv_dict.get(k)
        for s, e, raw in extract_intervals_from_text(v):
            _emit(s, e, f'node.kv_scan:{k}', raw)

    return out


# ── DB writer (called by embed_queue) ────────────────────────────────────

# Sentinel row marker — written when extraction ran but found no dates.
# Distinguishes "processed, no dates" from "not yet processed" so the
# left-join "find unprocessed" query doesn't re-scan empty-result
# entities forever. `recall_by_time` filters out sentinel rows via
# extraction_source.
_SENTINEL_SOURCE = '_no_dates_found'

# Defensive cap on rows per entity. A single node/edge with very long
# text containing many date references would otherwise be able to write
# hundreds of rows into entity_dates. Real-corpus distribution shows
# >95% of dated entities have ≤4 intervals — 20 is generous.
MAX_INTERVALS_PER_ENTITY = 20


def write_entity_dates(
    conn,
    entity_kind: str,
    entity_id: str,
    intervals: Iterable[Tuple[int, int, str, str]],
) -> int:
    """Replace all entity_dates rows for (entity_kind, entity_id).

    If `intervals` is empty, writes a single sentinel row marking the
    entity as processed-no-dates so the S2 indexer's "find unprocessed"
    query treats it as done.

    Returns the number of REAL interval rows written (sentinel doesn't
    count). Idempotent.
    """
    conn.execute(
        'DELETE FROM entity_dates WHERE entity_kind = ? AND entity_id = ?',
        (entity_kind, entity_id),
    )
    rows = [
        (entity_kind, entity_id, s, e, src, raw)
        for (s, e, src, raw) in intervals
    ]
    if not rows:
        conn.execute(
            '''INSERT INTO entity_dates
               (entity_kind, entity_id, start_ts, end_ts, extraction_source,
                raw_text)
               VALUES (?, ?, 0, 0, ?, NULL)''',
            (entity_kind, entity_id, _SENTINEL_SOURCE),
        )
        return 0
    # Defensive cap. Pathological inputs (e.g., a content field listing 500
    # dates) shouldn't be able to bloat the index. Keep first N — order
    # reflects emission order in extract_node_intervals/extract_edge_intervals
    # which is title → content → KV, so earlier-priority sources win.
    if len(rows) > MAX_INTERVALS_PER_ENTITY:
        rows = rows[:MAX_INTERVALS_PER_ENTITY]
    conn.executemany(
        '''INSERT OR REPLACE INTO entity_dates
           (entity_kind, entity_id, start_ts, end_ts, extraction_source,
            raw_text)
           VALUES (?, ?, ?, ?, ?, ?)''',
        rows,
    )
    return len(rows)


# ── Edge-level extraction ───────────────────────────────────────────────

def extract_edge_intervals(
    description: Optional[str],
    relation: Optional[str],
) -> List[Tuple[int, int, str, str]]:
    """Extract temporal intervals for one edge.

    Edges carry temporal claims primarily in `edge_relations.description`
    (free text). `relation` is usually a short verb ('produces',
    'validates') with low date density, but cheap to scan in case dates
    leak in.

    Returns: [(start_ts, end_ts, extraction_source, raw_text), ...]
    where extraction_source is 'edge.description' or 'edge.relation'.

    Edge dates work the same as node dates — when the tool surfaces an
    edge matching a time range, the caller is expected to also fetch the
    edge's source/target nodes for context.
    """
    out: List[Tuple[int, int, str, str]] = []
    seen: set = set()

    def _emit(start_ts: int, end_ts: int, source: str, raw: str) -> None:
        key = (start_ts, end_ts, source)
        if key in seen:
            return
        seen.add(key)
        out.append((start_ts, end_ts, source, raw))

    for s, e, raw in extract_intervals_from_text(description):
        _emit(s, e, 'edge.description', raw)
    for s, e, raw in extract_intervals_from_text(relation):
        _emit(s, e, 'edge.relation', raw)

    return out


def backfill_edge_dates(brain, edge_id: str, conn=None) -> int:
    """Extract intervals for one edge and write to entity_dates.

    Pulls all (relation, description) pairs from edge_relations for the
    given edge_id and scans each. Returns the number of intervals
    written.

    Edges with multiple relations on the same edge_id get all relations
    scanned — each row's description is its own date-bearing surface.

    Args:
        brain: Brain instance.
        edge_id: edge to scan.
        conn: optional sqlite3 connection. If None, uses brain.conn
            (legacy path, requires caller holds brain.write_lock).
            Pass brain.conn_bg_writer to route writes off the foreground
            writer slot.

    The caller is responsible for transaction lifecycle (BEGIN/COMMIT)
    on the connection passed in. This function only executes statements.
    """
    conn = conn if conn is not None else brain.conn
    rows = conn.execute(
        '''SELECT relation, description FROM edge_relations
           WHERE edge_id = ? AND (archived IS NULL OR archived = 0)''',
        (edge_id,),
    ).fetchall()
    if not rows:
        return 0
    # Aggregate intervals across all relations on this edge_id; dedupe
    # by (start_ts, end_ts, source) so multiple relations mentioning
    # the same date don't multiply rows.
    seen: set = set()
    intervals: List[Tuple[int, int, str, str]] = []
    for relation, description in rows:
        for s, e, src, raw in extract_edge_intervals(description, relation):
            key = (s, e, src)
            if key in seen:
                continue
            seen.add(key)
            intervals.append((s, e, src, raw))
    return write_entity_dates(conn, 'edge', edge_id, intervals)


def backfill_node_dates(brain, node_id: str, conn=None) -> int:
    """Extract intervals for one node and write to entity_dates.

    Pulls title + content from `nodes` and all KV pairs from
    `node_metadata_kv`. Returns the number of intervals written.

    Args:
        brain: Brain instance.
        node_id: node to scan.
        conn: optional sqlite3 connection. If None, uses brain.conn
            (legacy path, requires caller holds brain.write_lock).
            Pass brain.conn_bg_writer to route writes off the foreground
            writer slot.

    The caller is responsible for transaction lifecycle (BEGIN/COMMIT)
    on the connection passed in.
    """
    conn = conn if conn is not None else brain.conn
    row = conn.execute(
        'SELECT title, content FROM nodes WHERE id = ?', (node_id,)
    ).fetchone()
    if not row:
        return 0
    title, content = row[0], row[1]
    kv_pairs = conn.execute(
        'SELECT key, value FROM node_metadata_kv WHERE node_id = ?',
        (node_id,),
    ).fetchall()
    intervals = extract_node_intervals(title, content, kv_pairs)
    return write_entity_dates(conn, 'node', node_id, intervals)


def backfill_entity_dates(
    brain,
    node_ids: Iterable[str],
    edge_ids: Optional[Iterable[str]] = None,
    conn=None,
) -> dict:
    """Batch entry point — called by embed_queue._drain_once.

    Returns stats dict: {'nodes_processed': N, 'edges_processed': N,
    'intervals_written': N}.

    Nodes and edges are processed symmetrically — edges scan their own
    description + relation text fields (no inheritance from endpoint
    nodes). The tool layer (recall_by_time) is responsible for unwrapping
    matched edges into (edge + source_node + target_node) responses.

    Args:
        brain: Brain instance.
        node_ids: nodes to scan.
        edge_ids: edges to scan (optional).
        conn: optional sqlite3 connection to write through. None →
            brain.conn (legacy, requires brain.write_lock). For the
            embed_queue worker post-2026-05-18, pass brain.conn_bg_writer
            so writes don't race with foreground MCP writes at the WAL
            writer slot.

    The caller owns transaction lifecycle on `conn` — this function
    only executes statements and does not commit.
    """
    nodes_processed = 0
    edges_processed = 0
    intervals_written = 0
    for nid in (node_ids or []):
        if not nid:
            continue
        try:
            intervals_written += backfill_node_dates(brain, nid, conn=conn)
            nodes_processed += 1
        except Exception as e:
            brain._log_error('temporal_extract_node', e,
                              f'node_id={nid}')
    for eid in (edge_ids or []):
        if not eid:
            continue
        try:
            intervals_written += backfill_edge_dates(brain, eid, conn=conn)
            edges_processed += 1
        except Exception as e:
            brain._log_error('temporal_extract_edge', e,
                              f'edge_id={eid}')
    return {
        'nodes_processed': nodes_processed,
        'edges_processed': edges_processed,
        'intervals_written': intervals_written,
    }
