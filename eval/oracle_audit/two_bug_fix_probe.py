#!/usr/bin/env python3
"""MINIMAL two-bug-fix probe — no new machinery, isolated copy only.

The claim: the EX.CO burial is two BUGS in channels we already run, not a missing lane.
  bug 1 (tokenizer): nodes_fts uses 'porter unicode61' → "ex.co" shatters to ex+co.
  bug 2 (scoring):   fts5 hits get a flat 0.20, discarding bm25's IDF magnitude.
Fix = an entity-preserving FTS5 table (unicode61 tokenchars '.-_/', no porter) scored by bm25().

THE DECISIVE QUESTION: for the FULL #11 query, does bm25 (with the entity tokenizer) rank the
EX.CO nodes HIGH — i.e. does bm25's per-term IDF do the "extraction" implicitly, with ZERO
extraction code? If yes, the cathedral (RLR / entity-extraction / de-pooled lanes) is unnecessary.

Never touches live (IsolatedBrain copies to temp). Usage: ./dev python3 eval/oracle_audit/two_bug_fix_probe.py
"""
import os, sys, json, re
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
KNOWN_EXCO = {'e62cc595','dabb3078','af92b2cb','30d88dd0','b3bda662','5fe121db',
              '8359cf1d','5410f4be','ef2f3276','41d31ca5','671d1f22','598d78a8'}
EXCO_RANKS = {2, 11, 12}

def sanitize(q):
    """Query -> FTS5 MATCH OR-string of quoted terms, keeping dotted/hyphenated compounds."""
    terms = re.findall(r'[a-z0-9][a-z0-9._\-/]*', q.lower())
    terms = [t for t in terms if len(t) >= 2]
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t); out.append('"%s"' % t)
    return ' OR '.join(out)

def bm25_rank(conn, table, match, limit=250):
    rows = conn.execute(
        "SELECT node_id, bm25(%s, 0, 10.0, 1.0) AS s FROM %s WHERE %s MATCH ? ORDER BY s LIMIT ?"
        % (table, table, table), (match, limit)).fetchall()
    return [r[0][:8] for r in rows]

def best_exco(ids):
    return next((i for i, x in enumerate(ids, 1) if x in KNOWN_EXCO), None)

def exco_ranks(ids):
    return {x: i for i, x in enumerate(ids, 1) if x in KNOWN_EXCO}

with IsolatedBrain() as env:
    brain = env.brain
    conn = brain.conn
    # build the entity-preserving FTS5 table on the COPY
    conn.execute("DROP TABLE IF EXISTS nodes_fts_entity")
    conn.execute("""CREATE VIRTUAL TABLE nodes_fts_entity USING fts5(
        node_id UNINDEXED, title, content,
        tokenize="unicode61 tokenchars '.-_/'"
    )""")
    conn.execute("""INSERT INTO nodes_fts_entity (node_id, title, content)
                    SELECT id, title, COALESCE(content,'') FROM nodes WHERE archived = 0""")
    n = conn.execute("SELECT COUNT(*) FROM nodes_fts_entity").fetchone()[0]
    print("\n=== two-bug-fix probe (isolated, %d nodes indexed in entity-FTS) ===" % n)

    q11 = next(it['prompt'] for it in CORPUS if it['rank'] == 11)
    m11 = sanitize(q11)
    print("\n#11: %s" % q11)
    print("  MATCH: %s" % m11[:140])

    ent = bm25_rank(conn, 'nodes_fts_entity', m11)
    por = bm25_rank(conn, 'nodes_fts', m11)
    print("\n  -- best EX.CO rank in bm25 ranking, full #11 query --")
    print("  entity-FTS (FIXED tokenizer): best=%s   EX.CO ranks=%s" % (best_exco(ent), exco_ranks(ent)))
    print("  porter-FTS (current/broken):  best=%s   EX.CO ranks=%s" % (best_exco(por), exco_ranks(por)))

    # baseline: full recall pipeline, where does EX.CO land?
    if hasattr(brain, '_recall_cache'):
        try: brain._recall_cache.clear()
        except Exception: pass
    out = brain.recall(query=q11, limit=30)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    rec_ids = [(r.get('id') or r.get('node_id') or '')[:8] for r in res]
    print("  baseline brain.recall(#11) top-30: best EX.CO rank=%s" % (best_exco(rec_ids) or '— none —'))

    # sanity: "ex.co" alone on entity-FTS
    print("\n  -- sanity: 'ex.co' alone on entity-FTS, top-6 --")
    for nid in bm25_rank(conn, 'nodes_fts_entity', '"ex.co"', 6):
        t = conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        print("    %s %s %s" % (nid, '[EXCO]' if nid in KNOWN_EXCO else '      ', (t[0] if t else '')[:50]))

    # all EX.CO queries
    print("\n  -- all EX.CO queries: best EX.CO rank in entity-FTS bm25 --")
    for it in CORPUS:
        if it['rank'] in EXCO_RANKS:
            ids = bm25_rank(conn, 'nodes_fts_entity', sanitize(it['prompt']))
            print("    #%-2d %-10s best_exco=%s" % (it['rank'], it['src'], best_exco(ids)))

    # controls: what does entity-FTS surface (junk check)
    print("\n  -- 9 controls: entity-FTS bm25 top-5 (junk/relevance check) --")
    for it in CORPUS:
        if it['rank'] in EXCO_RANKS:
            continue
        ids = bm25_rank(conn, 'nodes_fts_entity', sanitize(it['prompt']), 5)
        titles = []
        for nid in ids[:5]:
            t = conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
            titles.append((t[0] if t else '')[:30])
        print("    #%-2d %-8s: %s" % (it['rank'], it['src'], ' | '.join(titles)))
