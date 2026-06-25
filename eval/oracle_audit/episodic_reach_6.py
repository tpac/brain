#!/usr/bin/env python3
"""EPISODIC REACH for the 6 never-recalled essentials.

Content-cosine recall never pulled these 6 gold-essential nodes into the 200-candidate
pool. Tom's hypothesis: they are EPISODIC misses — reachable via the trace store, which
content-node recall doesn't touch. Test the two coverage-independent bridges (the ones
that work on OLD nodes lacking source_refs/co_anchored edges):

  cosine(query, trace_embeddings) -> top-K dialogue traces, then for each gold node:
    (1) STRUCTURAL: is the gold node in node_source_refs of a top-K trace?
    (2) TEMPORAL  : is a top-K query-matched trace within +/-W days of the gold node's
                    created_at / revised_at? (Tom: "nodes encoded/revised associated by
                    time to that text")

Report, per gold node, the best trace-rank at which it becomes reachable by EITHER bridge
(vs content-cosine, which never reached it at all). Dialogue-only (drop tool_result, per
episodic-lane hygiene). Daemon-safe (IsolatedBrain).
Usage: ./dev python3 eval/oracle_audit/episodic_reach_6.py
"""
import sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

# (qid, gold 8-char, query)
SIX = [
    ('TR6', 'f58e9b12', "let's clean up the old backups"),
    ('TO4', '4b35293c', "how does the fatigue thing work?"),
    ('TO5', '51ff0a92', "what's the real value you add over plain Claude?"),
    ('TO6', 'e49766ac', "how are the scouts built — do their prompts have examples?"),
    ('HV6', '94f6e01a', "what have we already tried and ruled out on recall burial?"),
    ('EP4', '90e27c77', "what did we conclude early on about similar_to edges and old artifacts"),
]
TOPK = 25            # trace-rank horizon
WINDOW_DAYS = 2.0    # temporal-association half-window
DIALOGUE = ('user_message', 'assistant_message')


def daynum(iso):
    # crude day index from 'YYYY-MM-DD...' for delta math
    if not iso:
        return None
    s = str(iso)[:10].replace('-', '')
    try:
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        return y * 365 + m * 30 + d
    except Exception:
        return None


with IsolatedBrain() as env:
    brain = env.brain
    from servers import embedder
    lc, bc = brain.logs_conn, brain.conn

    # dialogue-only embedded traces, with ref_type + created_at from trace_events
    rows = lc.execute(
        "SELECT te.trace_id, te.vector, te.text, ev.created_at, ev.ref_type "
        "FROM trace_embeddings te JOIN trace_events ev ON ev.id = te.trace_id"
    ).fetchall()
    traces = [(tid, vec, txt, ca, rt) for (tid, vec, txt, ca, rt) in rows if rt in DIALOGUE]
    print("dialogue trace_embeddings: %d (of %d embedded)\n" % (len(traces), len(rows)))

    def resolve(prefix):
        r = bc.execute("SELECT id, created_at, revised_at FROM nodes WHERE id LIKE ?",
                       (prefix + '%',)).fetchone()
        return r  # (full_id, created_at, revised_at) or None

    for qid, gold8, query in SIX:
        g = resolve(gold8)
        if not g:
            print("%-4s %s  [gold node not found in brain]\n" % (qid, gold8))
            continue
        gid, gca, grev = g
        gdays = [d for d in (daynum(gca), daynum(grev)) if d is not None]

        qv = embedder.embed_query(query)
        scored = sorted(
            ((embedder.cosine_similarity(qv, vec), tid, txt, ca) for tid, vec, txt, ca, rt in traces),
            key=lambda x: -x[0])[:TOPK]

        # source_refs of the gold node (which traces point to it)
        gold_trace_ids = set(r[0] for r in bc.execute(
            "SELECT trace_id FROM node_source_refs WHERE node_id = ?", (gid,)).fetchall())

        struct_rank = temporal_rank = None
        struct_hit = temporal_hit = None
        for rank, (cos, tid, txt, ca) in enumerate(scored, 1):
            if struct_rank is None and tid in gold_trace_ids:
                struct_rank, struct_hit = rank, (cos, txt, ca)
            if temporal_rank is None and gdays:
                td = daynum(ca)
                if td is not None and min(abs(td - d) for d in gdays) <= WINDOW_DAYS:
                    temporal_rank, temporal_hit = rank, (cos, txt, ca, min(abs(td - d) for d in gdays))
        best = min([r for r in (struct_rank, temporal_rank) if r is not None], default=None)

        print("%-4s gold=%s  created=%s" % (qid, gold8, str(gca)[:10]))
        print("     query: %r" % query)
        print("     top trace cos=%.3f  | source_refs on gold: %d" % (scored[0][0], len(gold_trace_ids)))
        print("     STRUCTURAL reach: %s   TEMPORAL reach: %s   -> EPISODIC-REACHABLE @ rank %s"
              % (("rank %d" % struct_rank) if struct_rank else "no",
                 ("rank %d (Δ%.0fd)" % (temporal_rank, temporal_hit[3])) if temporal_rank else "no",
                 best if best else "NO (not in top-%d traces)" % TOPK))
        show = struct_hit or temporal_hit
        if show:
            print("     matched trace: %r" % (str(show[1])[:110]))
        print()
