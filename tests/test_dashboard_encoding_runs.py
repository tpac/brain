"""Encoding-run card reads the authoritative delta, not the first one.

Regression guard for the 2026-07-03 "0 actions" bug: S1E (v29) writes many
delta rows per chain (edge_relation_revised per edge, node_revised per node,
journal_note per residue item, plus the encoding_run summary). The card query
must select the `encoding_run` delta — an unfiltered fetchone() grabs the first
row (an edge revision, no created/revised rollup) and the card shows "0 actions"
for runs that actually wrote. Also pins the journal-notes collection (rendered
first in the card details).

Calls the undecorated `_query_encoding_chains.__wrapped__` with an in-memory
logs_db so it exercises the SQL without a real brain.
"""
import json
import sqlite3

from dashboard.queries import encoding as enc


def _conn(rows):
    """In-memory trace_events with (chain_id, event_type, ref_type, summary,
    metadata, session_id, created_at) rows, inserted in list order (so rowid
    order == list order — the ordering the bug depended on)."""
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE trace_events (id INTEGER PRIMARY KEY, chain_id TEXT, "
              "scale TEXT, event_type TEXT, ref_type TEXT, ref_id TEXT, "
              "summary TEXT, metadata TEXT, session_id TEXT, created_at TEXT)")
    for i, r in enumerate(rows):
        c.execute("INSERT INTO trace_events (id, chain_id, scale, event_type, "
                  "ref_type, ref_id, summary, metadata, session_id, created_at) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i, r['chain_id'], r.get('scale', 's1'), r['event_type'],
                   r['ref_type'], r.get('ref_id', ''), r.get('summary', ''),
                   json.dumps(r.get('metadata', {})), r.get('session_id', ''),
                   r.get('created_at', '')))
    c.commit()
    return c


run_chains = enc._query_encoding_chains.__wrapped__   # bypass @safe_query conn-open

CHAIN = 's1e-testsess-7'
TS = '2026-07-03T16:37:00+00:00'


def _base_rows():
    # Order matters: the O/K first, then an edge_relation_revised delta BEFORE
    # the encoding_run summary delta — this is the ordering that made the old
    # unfiltered fetchone() pick the wrong row.
    return [
        {'chain_id': CHAIN, 'event_type': 'O', 'ref_type': 'encoding_prompt',
         'ref_id': '/tmp/none.json', 'summary': '19 turns', 'created_at': TS,
         'session_id': 'testsess'},
        {'chain_id': CHAIN, 'event_type': 'K', 'ref_type': 'node_catalog',
         'summary': '0 unique nodes', 'created_at': TS},
        {'chain_id': CHAIN, 'event_type': 'delta', 'ref_type': 'edge_relation_revised',
         'summary': '3 field(s): description, weight, encoding_source',
         'metadata': {'edge_id': 'e1', 'relation': 'co_anchored'}, 'created_at': TS},
        {'chain_id': CHAIN, 'event_type': 'delta', 'ref_type': 'journal_note',
         'summary': 'a doubt', 'metadata': {'note': 'ships to a corpus with zero cues',
                                            'tag': 'doubt'}, 'created_at': TS},
        {'chain_id': CHAIN, 'event_type': 'delta', 'ref_type': 'encoding_run',
         'summary': '1 actions (1 writes) in 2 rounds',
         'metadata': {'created': ['aaaa1111', 'bbbb2222'], 'revised': ['cccc3333']},
         'created_at': '2026-07-03T16:40:00+00:00'},
    ]


def test_reads_encoding_run_delta_not_first():
    runs = run_chains(_conn(_base_rows()), limit=10, session_id='', hours=999999)
    assert len(runs) == 1
    r = runs[0]
    # Summary + rollup come from the encoding_run delta, NOT the edge revision.
    assert r['summary'].startswith('1 actions')
    assert r['created_ids'] == ['aaaa1111', 'bbbb2222']
    assert r['revised_ids'] == ['cccc3333']
    # delta_ts is the run's completion, used to bound the edge-window query.
    assert r['delta_ts'] == '2026-07-03T16:40:00+00:00'


def test_collects_journal_notes():
    runs = run_chains(_conn(_base_rows()), limit=10, session_id='', hours=999999)
    notes = runs[0]['journal_notes']
    assert notes == [{'note': 'ships to a corpus with zero cues', 'tag': 'doubt'}]


def test_in_progress_run_has_no_summary_delta():
    # A chain whose encoding_run delta hasn't landed yet (still encoding) →
    # graceful fallback, empty rollup, no crash.
    rows = [r for r in _base_rows() if r['ref_type'] != 'encoding_run']
    runs = run_chains(_conn(rows), limit=10, session_id='', hours=999999)
    r = runs[0]
    assert r['summary'] == '(encoding in progress or no actions)'
    assert r['created_ids'] == [] and r['revised_ids'] == []
    # journal notes still surface even before the summary delta lands.
    assert len(r['journal_notes']) == 1
