#!/usr/bin/env python3
"""TOKENIZER TEST — does the 2nd FTS5 table (tokenchars) earn its complexity, or should arm A
just score the existing porter lane? Settles the fork with data (CLAUDE.md: test, don't deliberate).

Compares THREE tokenizers on the SAME bm25 scoring, over the corpus's entity/literal queries:
  porter       = production nodes_fts (porter unicode61)          — shatters ex.co -> [ex][co], stems
  tokenchars   = nodes_fts_exact (unicode61 tokenchars '.-_/')     — keeps ex.co whole, NO stemming
  both         = nodes_fts_both (porter unicode61 tokenchars ...)  — keep compound AND stem (best-of-both?)

Two effects, opposite directions:
  (A) compound-IDF: punctuated tokens (ex.co) — tokenchars keeps them rare/discriminating; porter
      scores on common stems [ex][co]. Expect tokenchars/both to win HERE.
  (B) stemming: pricing->price, running->run — porter/both match morphological variants tokenchars misses.
Verdict rule: if tokenchars beats porter on gold lex-RANK for punctuated queries AND 'both' keeps
stemming, keep a 2nd table (ideally 'both'). If gold-rank is ~identical, drop it — score porter.

Daemon-safe (IsolatedBrain). Usage: ./dev python3 eval/oracle_audit/lexical_tokenizer_test.py
"""
import sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers.dal import Fts5DAL                  # noqa: E402

CORPUS = {q['id']: q for q in json.load(open(f'{ROOT}/eval/oracle_audit/recall_corpus_v2.json'))['queries']}
# (id, has-punctuated-token?)
TESTS = [('E1b', True), ('E2', False), ('L1', False), ('L2', True), ('T2', False)]


def scored(conn, tbl, safe, limit=80):
    try:
        rows = conn.execute(
            "SELECT node_id, bm25(%s,0,10.0,1.0) s FROM %s WHERE %s MATCH ? ORDER BY s LIMIT ?"
            % (tbl, tbl, tbl), (safe, limit)).fetchall()
    except Exception as e:
        print("   [%s scoring err: %s]" % (tbl, e)); return {}
    hits = [(r[0], -float(r[1])) for r in rows]
    if not hits:
        return {}
    top = max(s for _, s in hits) or 1.0
    return {nid[:8]: (rk, round(s / top, 3)) for rk, (nid, s) in enumerate(hits, 1)}


def df(conn, tbl, expr):
    try:
        return conn.execute("SELECT count(*) FROM %s WHERE %s MATCH ?" % (tbl, tbl), (expr,)).fetchone()[0]
    except Exception:
        return 'ERR'


with IsolatedBrain() as env:
    b = env.brain
    c = b.conn
    # build tokenchars table inline (Fts5ExactDAL was removed after this one-shot verdict, node 3c315383)
    c.execute("DROP TABLE IF EXISTS nodes_fts_exact")
    c.execute("CREATE VIRTUAL TABLE nodes_fts_exact USING fts5("
              "node_id UNINDEXED, title, content, tokenize=\"unicode61 tokenchars '.-_/'\")")
    c.execute("INSERT INTO nodes_fts_exact (node_id,title,content) "
              "SELECT id,title,COALESCE(content,'') FROM nodes WHERE archived=0")
    # build the 'both' table (porter + tokenchars)
    c.execute("DROP TABLE IF EXISTS nodes_fts_both")
    c.execute("CREATE VIRTUAL TABLE nodes_fts_both USING fts5("
              "node_id UNINDEXED, title, content, tokenize=\"porter unicode61 tokenchars '.-_/'\")")
    c.execute("INSERT INTO nodes_fts_both (node_id,title,content) "
              "SELECT id,title,COALESCE(content,'') FROM nodes WHERE archived=0")

    print("=== (A) compound-IDF: document frequency of the rare compound vs its porter stems ===")
    print("  'ex.co'  as ONE token (tokenchars df): %s   |  porter has NO 'ex.co' token -> stems:" % df(c, 'nodes_fts_exact', '"ex.co"'))
    print("     df('ex')=%s  df('co')=%s  (porter scores ex.co on THESE common stems = low IDF = weak/blunt)"
          % (df(c, 'nodes_fts', 'ex'), df(c, 'nodes_fts', 'co')))
    print("  'create_media_buy' tokenchars df: %s  | porter stems df('media')=%s df('buy')=%s"
          % (df(c, 'nodes_fts_exact', '"create_media_buy"'), df(c, 'nodes_fts', 'media'), df(c, 'nodes_fts', 'buy')))

    print("\n=== (B) stemming reach: does porter match morphological variants tokenchars misses? ===")
    for stem_q in ('pricing', 'running', 'consolidated'):
        print("  '%-12s' porter df=%-4s tokenchars df=%-4s (porter>tokenchars => stemming adds reach)"
              % (stem_q, df(c, 'nodes_fts', stem_q), df(c, 'nodes_fts_exact', stem_q)))

    print("\n=== per-query GOLD lexical rank/norm under each tokenizer (lower rank = better lift) ===")
    for qid, punct in TESTS:
        q = CORPUS[qid]
        safe = Fts5DAL._sanitize_query(q['query_rich'])
        sp = scored(c, 'nodes_fts', safe)
        sx = scored(c, 'nodes_fts_exact', safe)
        sb = scored(c, 'nodes_fts_both', safe)
        print("\n#%-4s %s  [punctuated-token: %s]" % (qid, q['query_rich'][:52], punct))
        print("   sanitized MATCH: %s" % safe)
        for g in q['node_gold_primary']:
            def fmt(d):
                v = d.get(g)
                return "rk%-2d/%.2f" % (v[0], v[1]) if v else "  miss "
            print("   gold %s  porter=%s  tokenchars=%s  both=%s" % (g, fmt(sp), fmt(sx), fmt(sb)))
