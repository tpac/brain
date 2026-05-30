#!/usr/bin/env python3
"""Frame replay harness — capture surface output for before/after comparison.

Validates Phase 2+ Frame work against the Phase 1 baseline. Each `capture`
runs the corpus through the production surface pipeline against an isolated
brain copy and saves a labeled snapshot. `compare` diffs two snapshots.

Usage:
    ./dev python3 eval/frame_replay.py capture phase1_baseline
    ./dev python3 eval/frame_replay.py compare phase1_baseline phase2_v1
    ./dev python3 eval/frame_replay.py list

Snapshots: eval/replay_snapshots/{label}.json

Limitation (v1): each query runs in a fresh session, so queries that depend
on multi-turn context ("Where were we?") will surface the empty-context floor.
That IS the right measurement for "what does cold recall do today?" — and
when Phase 2 lands, the same fresh-session capture will measure how much
the Frame restores presence without leaning on session continuity.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Test corpus from FRAME-DESIGN.md Appendix A ──
# Stable IDs — never renumber, so historic snapshots stay comparable.
CORPUS = [
    {
        "id": "exco_cold",
        "query": "What is EX.CO?",
        "expected": "EX.CO recognized via partnership_frame, not Anchor-meta nodes",
    },
    {
        "id": "self_intro",
        "query": "What do you know about you?",
        "expected": "operator_frame + partnership_frame answer (operator context, not pure Anchor-meta)",
    },
    {
        "id": "exco_pivot",
        "query": "Should we go back to EX.CO sales kit?",
        "expected": "Recognized as pivot-probe, current_focus tracked",
    },
    {
        "id": "where_were_we",
        "query": "Where were we?",
        "expected": "current_focus slot answers directly (Phase 2+)",
    },
    {
        "id": "open_last_week",
        "query": "What's still open from last week?",
        "expected": "find_open_loops + temporal filter (Phase 4 tool)",
    },
]

SNAPSHOT_DIR = ROOT / "eval" / "replay_snapshots"


def _capture_one(brain, ctx, spec):
    """Run a single query through recall + run_surface, return snapshot dict."""
    from servers.scales.s1.surface import run_surface
    from servers.pipeline_contract import CANDIDATES_FILE
    from servers.scales.s1.surface_contract import select_edges
    import numpy as np

    qid = spec["id"]
    query = spec["query"]

    t0 = time.time()
    result = brain.recall(query=query, limit=CANDIDATES_FILE['max_candidates'],
                          session_id=ctx.session_id, source='replay')
    results = result.get("results", [])
    recall_ms = (time.time() - t0) * 1000

    if not results:
        return {
            "id": qid, "query": query, "expected": spec["expected"],
            "skipped": True, "reason": "no recall results",
            "recall_ms": round(recall_ms),
        }

    # Mirror the candidate-enrichment logic from daemon_hooks.hook_recall —
    # same shape so run_surface receives what production gives it.
    capped = results[:CANDIDATES_FILE['max_candidates']]
    node_ids = [r.get("id", "") for r in capped if r.get("id")]
    rich_nodes = brain.get_node(node_ids)

    _query_emb = result.get("_query_embedding")
    _query_vec = None
    if _query_emb is not None:
        _query_vec = (np.frombuffer(_query_emb, dtype=np.float32)
                      if isinstance(_query_emb, bytes)
                      else np.array(_query_emb, dtype=np.float32))

    candidates_data = []
    for r in capped:
        nid = r.get("id", "")
        node_data = rich_nodes.get(nid) or {
            "id": nid, "type": r.get("type", ""),
            "title": r.get("title", ""), "content": r.get("content", ""),
            "confidence": r.get("confidence", 0), "locked": r.get("locked", False),
            "created_at": r.get("created_at"), "revised_at": r.get("revised_at"),
        }
        if _query_vec is not None and node_data.get('connections'):
            node_data['connections'] = select_edges(
                node_data['connections'], _query_vec,
                limit=10, prior_vecs=[],
                brain_conn=brain.conn, brain=brain)
        node_data["score"] = r.get("effective_activation", 0)
        node_data["discovery"] = r.get("_discovery", "embedding")
        node_data["_all_connections"] = rich_nodes.get(nid, {}).get('connections', [])
        candidates_data.append(node_data)

    recall_ref = "replay-%s" % qid

    # Frame Phase 2 (2026-05-02): pass the Frame through, mirroring production
    # daemon_hooks.hook_recall. Falls back to '' if get_frame fails — surface
    # then degrades to Phase 1 layout (session_context + encoding_journal only).
    try:
        _frame = ctx.get_frame(brain)
    except Exception:
        _frame = ''

    t1 = time.time()
    surface_err = None
    additional_context = None
    try:
        additional_context = run_surface(
            brain, ctx, candidates_data, query,
            recent_messages=[],
            result=result, enriched=query, results=results,
            recall_ref=recall_ref, session_id=ctx.session_id,
            graph_changes=None,
            query_vec=_query_vec, prior_vecs=[],
            frame=_frame)
    except Exception as e:
        surface_err = repr(e)
    surface_ms = (time.time() - t1) * 1000

    # Read back what surface saved (selected_ids file). The path encodes
    # session + stop_counter — both written by run_surface.
    selected_path = "/tmp/brain-%s-%d-surface-selected.json" % (
        ctx.session_id, ctx.stop_counter)
    selected_ids = []
    if os.path.exists(selected_path):
        try:
            selected_ids = json.load(open(selected_path)).get("selected_ids", [])
        except Exception:
            pass

    cand_summary = [{
        "rank": i + 1,
        "id": (c.get("id") or "")[:8],
        "type": c.get("type", ""),
        "title": (c.get("title") or "")[:80],
        "score": round(c.get("score", 0), 3),
        "discovery": c.get("discovery", ""),
    } for i, c in enumerate(candidates_data[:25])]

    return {
        "id": qid,
        "query": query,
        "expected": spec["expected"],
        "recall_ms": round(recall_ms),
        "surface_ms": round(surface_ms),
        "n_candidates": len(candidates_data),
        "candidates": cand_summary,
        "selected_ids": sorted(selected_ids),
        "additional_context": additional_context or "",
        "surface_error": surface_err,
    }


def capture(label, brain_dir=None, verbose=False):
    """Run corpus through production surface pipeline, save labeled snapshot."""
    from tests.isolated_brain import IsolatedBrain

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    out_path = SNAPSHOT_DIR / ("%s.json" % label)
    if out_path.exists():
        sys.stderr.write("[capture] WARN: %s exists — overwriting\n" % out_path)

    snapshot = {
        "label": label,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queries": [],
    }

    with IsolatedBrain(production_dir=brain_dir) as env:
        brain = env.brain
        for spec in CORPUS:
            # Fresh session per query (see module docstring limitation).
            session_id = "replay_%s_%s_%d" % (label, spec["id"], int(time.time()))
            ctx = brain.get_or_create_session(session_id)
            sys.stderr.write("[capture] %s: %s\n" % (
                spec["id"], spec["query"][:60]))
            try:
                row = _capture_one(brain, ctx, spec)
            except Exception as e:
                row = {
                    "id": spec["id"], "query": spec["query"],
                    "expected": spec["expected"],
                    "skipped": True, "reason": "capture exception: %r" % e,
                }
            snapshot["queries"].append(row)
            if not row.get("skipped"):
                sys.stderr.write(
                    "  → %d selected, %dc context, %d+%dms\n" % (
                        len(row.get("selected_ids") or []),
                        len(row.get("additional_context") or ""),
                        row.get("recall_ms", 0), row.get("surface_ms", 0)))

    out_path.write_text(json.dumps(snapshot, indent=2, default=str))
    sys.stderr.write("[capture] Saved %s (%d queries)\n" % (
        out_path, len(snapshot["queries"])))
    # Loud-by-default: an all-errored capture is a useless snapshot that used to
    # exit 0 and look successful — that's how a stale select_edges kwarg went
    # unnoticed. Fail loudly when nothing captured; warn on partial skips.
    skipped_n = sum(1 for q in snapshot["queries"] if q.get("skipped"))
    total_n = len(snapshot["queries"])
    if total_n and skipped_n == total_n:
        sys.exit("[capture] FAILED: all %d queries errored — snapshot unusable. "
                 "First reason: %s" % (total_n, snapshot["queries"][0].get("reason", "?")))
    if skipped_n:
        sys.stderr.write("[capture] WARN: %d/%d queries skipped\n" % (skipped_n, total_n))


def compare(label_a, label_b):
    """Print side-by-side diff of two snapshots."""
    pa = SNAPSHOT_DIR / ("%s.json" % label_a)
    pb = SNAPSHOT_DIR / ("%s.json" % label_b)
    if not pa.exists():
        sys.exit("Missing snapshot: %s" % pa)
    if not pb.exists():
        sys.exit("Missing snapshot: %s" % pb)
    a = json.loads(pa.read_text())
    b = json.loads(pb.read_text())

    a_q = {q["id"]: q for q in a["queries"]}
    b_q = {q["id"]: q for q in b["queries"]}

    print("\n" + "=" * 100)
    print("COMPARE  A: %s  (%s)" % (label_a, a["captured_at"]))
    print("         B: %s  (%s)" % (label_b, b["captured_at"]))
    print("=" * 100 + "\n")

    for qid in sorted(set(a_q) | set(b_q)):
        qa, qb = a_q.get(qid), b_q.get(qid)
        if not qa or not qb:
            print("⚠ %s: only in %s\n" % (qid, "A" if qa else "B"))
            continue

        print("━━━ %s: %s ━━━" % (qid, qa["query"]))
        print("  Expected: %s" % qa.get("expected", ""))

        if qa.get("skipped") or qb.get("skipped"):
            print("  Skipped:  A=%s  B=%s" % (
                qa.get("reason", "-") if qa.get("skipped") else "no",
                qb.get("reason", "-") if qb.get("skipped") else "no"))
            print()
            continue

        print("  Latency:  A=%d+%dms  B=%d+%dms" % (
            qa.get("recall_ms", 0), qa.get("surface_ms", 0),
            qb.get("recall_ms", 0), qb.get("surface_ms", 0)))

        sa = set(qa.get("selected_ids", []))
        sb = set(qb.get("selected_ids", []))
        if sa == sb:
            print("  Selected: same  %s" % sorted(sa))
        else:
            print("  Selected:")
            print("    A only: %s" % sorted(sa - sb))
            print("    B only: %s" % sorted(sb - sa))
            print("    both:   %s" % sorted(sa & sb))

        cb = {c["id"]: c["rank"] for c in qb.get("candidates", [])[:25]}
        rank_changes = []
        for c in qa.get("candidates", [])[:10]:
            new_rank = cb.get(c["id"])
            if new_rank is None:
                rank_changes.append(
                    "    %s '%s' A#%d → B(out)" % (
                        c["id"], c["title"][:50], c["rank"]))
            elif new_rank != c["rank"]:
                arrow = "↑" if new_rank < c["rank"] else "↓"
                rank_changes.append(
                    "    %s '%s' A#%d → B#%d %s" % (
                        c["id"], c["title"][:50], c["rank"], new_rank, arrow))
        if rank_changes:
            print("  Top-10 rank changes:")
            for line in rank_changes[:8]:
                print(line)

        ac = len(qa.get("additional_context") or "")
        bc = len(qb.get("additional_context") or "")
        print("  Context:  A=%dc  B=%dc  Δ=%+dc" % (ac, bc, bc - ac))
        print()


def list_snapshots():
    if not SNAPSHOT_DIR.exists():
        print("(no snapshots)")
        return
    for f in sorted(SNAPSHOT_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            print("  %s  (%s, %d queries)" % (
                f.stem, d.get("captured_at", "?"),
                len(d.get("queries", []))))
        except Exception:
            print("  %s  (unreadable)" % f.stem)


def main():
    p = argparse.ArgumentParser(description="Frame replay harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capture", help="Run corpus, save labeled snapshot")
    pc.add_argument("label", help="Snapshot label (e.g. phase1_baseline)")
    pc.add_argument("--brain-dir", default=None,
                    help="Production brain dir (auto-detected)")
    pc.add_argument("-v", "--verbose", action="store_true")

    pcm = sub.add_parser("compare", help="Diff two snapshots side-by-side")
    pcm.add_argument("label_a")
    pcm.add_argument("label_b")

    sub.add_parser("list", help="List captured snapshots")

    args = p.parse_args()
    if args.cmd == "capture":
        capture(args.label, brain_dir=args.brain_dir, verbose=args.verbose)
    elif args.cmd == "compare":
        compare(args.label_a, args.label_b)
    elif args.cmd == "list":
        list_snapshots()


if __name__ == "__main__":
    main()
