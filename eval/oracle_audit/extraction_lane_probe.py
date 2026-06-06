#!/usr/bin/env python3
"""MINIMAL extraction lane — end-to-end test (isolated copy, no production changes).

The right-sized design (after the two-bug-fix was falsified): isolate the rarest query
term, retrieve on IT, additively merge into the candidate window. No regex-by-typography,
no RLR cathedral. This tests it whole:

  Step 1  IDF-extract: rank the query's terms by corpus doc-frequency (via the entity-FTS),
          pick the RAREST. (Does it land on 'ex.co' for #11, automatically?)
  Step 2  Retrieve on the isolated term: embed_query(term) -> cosine vs _primary (semantic),
          AND entity-FTS bm25 on the term (lexical). Union.
  Step 3  Additive merge: spine = brain.recall top-30 (UNTOUCHED); append <=5 lane hits not
          already in spine -> augmented window. Spine top-5 cannot move (control-safe).
  Step 4  Measure: ex.co surfaces for the EX.CO queries? controls unharmed (no ex.co injected,
          spine stable)?

Never touches live. Usage: ./dev python3 eval/oracle_audit/extraction_lane_probe.py
"""
import os, sys, json, re
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
KNOWN_EXCO = {'e62cc595','dabb3078','af92b2cb','30d88dd0','b3bda662','5fe121db',
              '8359cf1d','5410f4be','ef2f3276','41d31ca5','671d1f22','598d78a8'}
EXCO_RANKS = {2, 11, 12}
RESERVE = 5
LANE_TOPK = 8

def terms_of(q):
    ts = re.findall(r'[a-z0-9][a-z0-9._\-/]*', q.lower())
    seen, out = set(), []
    for t in ts:
        if len(t) >= 2 and t not in seen:
            seen.add(t); out.append(t)
    return out

with IsolatedBrain() as env:
    brain = env.brain
    conn = brain.conn
    from servers import embedder
    # entity-preserving FTS5 on the copy (rarity oracle + lexical lane)
    conn.execute("DROP TABLE IF EXISTS nodes_fts_entity")
    conn.execute("""CREATE VIRTUAL TABLE nodes_fts_entity USING fts5(
        node_id UNINDEXED, title, content, tokenize="unicode61 tokenchars '.-_/'" )""")
    conn.execute("""INSERT INTO nodes_fts_entity (node_id, title, content)
                    SELECT id, title, COALESCE(content,'') FROM nodes WHERE archived=0""")

    def df(term):
        try:
            return conn.execute("SELECT count(*) FROM nodes_fts_entity WHERE nodes_fts_entity MATCH ?",
                                ('"%s"' % term,)).fetchone()[0]
        except Exception:
            return 10**9

    def rarest(q, k=2):
        scored = sorted(((df(t), t) for t in terms_of(q)), key=lambda x: x[0])
        return scored[:k], scored

    # load all _primary doc vectors once
    model = embedder.stats.get('model_name') or ''
    vrows = brain._vec_dal.get_all_vectors(vector_types=['_primary'], model=model or None)
    vecs = [(r['node_id'], r['embedding']) for r in vrows if r['embedding']]

    def lane(term):
        qv = embedder.embed_query(term)
        sem = sorted(((embedder.cosine_similarity(qv, b), nid) for nid, b in vecs),
                     key=lambda x: -x[0])
        sem_ids = [nid[:8] for _, nid in sem[:LANE_TOPK]]
        try:
            lex = conn.execute(
                "SELECT node_id FROM nodes_fts_entity WHERE nodes_fts_entity MATCH ? "
                "ORDER BY bm25(nodes_fts_entity,0,10.0,1.0) LIMIT ?", ('"%s"' % term, LANE_TOPK)).fetchall()
            lex_ids = [r[0][:8] for r in lex]
        except Exception:
            lex_ids = []
        merged, seen = [], set()
        for nid in sem_ids + lex_ids:
            if nid not in seen:
                seen.add(nid); merged.append(nid)
        return merged

    def best_exco(ids):
        return next((i for i, x in enumerate(ids, 1) if x in KNOWN_EXCO), None)

    def title(nid):
        r = conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        return (r[0] if r else '')[:42]

    def recall_spine(q):
        if hasattr(brain, '_recall_cache'):
            try: brain._recall_cache.clear()
            except Exception: pass
        out = brain.recall(query=q, limit=30)
        res = out.get('results', []) if isinstance(out, dict) else (out or [])
        return [(r.get('id') or r.get('node_id') or '')[:8] for r in res]

    print("\n=== MINIMAL EXTRACTION LANE — end-to-end (isolated) ===")
    fp_controls = 0
    lift = 0
    print("\n--- EX.CO queries ---")
    for it in CORPUS:
        if it['rank'] not in EXCO_RANKS:
            continue
        top2, _ = rarest(it['prompt'])
        anchor = top2[0][1]
        spine = recall_spine(it['prompt'])
        ln = lane(anchor)
        reserved = [x for x in ln if x not in set(spine)][:RESERVE]
        aug = spine + reserved
        b_base, b_aug = best_exco(spine), best_exco(aug)
        if b_aug and (not b_base or b_aug <= 30 + RESERVE):
            lift += 1
        print("  #%-2d %-11s rarest=%-10s(df=%d) [next:%s df=%d]  base_exco=%s  aug_exco=%s"
              % (it['rank'], it['src'], anchor, top2[0][0],
                 top2[1][1] if len(top2) > 1 else '-', top2[1][0] if len(top2) > 1 else -1,
                 str(b_base or '—'), str(b_aug or '—')))
        rtag = [('%s%s' % (x, '*' if x in KNOWN_EXCO else '')) for x in reserved]
        print("       reserved tail: %s" % rtag)

    print("\n--- 9 controls (does extraction harm them?) ---")
    for it in CORPUS:
        if it['rank'] in EXCO_RANKS:
            continue
        top2, _ = rarest(it['prompt'])
        anchor = top2[0][1]
        spine = recall_spine(it['prompt'])
        ln = lane(anchor)
        reserved = [x for x in ln if x not in set(spine)][:RESERVE]
        injected_exco = [x for x in reserved if x in KNOWN_EXCO]
        if injected_exco:
            fp_controls += 1
        print("  #%-2d %-8s rarest=%-14s(df=%d)  exco_injected=%s"
              % (it['rank'], it['src'], anchor, top2[0][0], injected_exco or 'none'))
        print("       reserved: %s" % [title(x) for x in reserved[:3]])

    print("\n=== SUMMARY ===")
    print("  EX.CO surfaced into augmented window: %d/3" % lift)
    print("  controls with ex.co injected (false positive): %d/9" % fp_controls)
    print("  control spine top-5: UNMOVED by construction (additive tail; spine = untouched recall top-30)")
