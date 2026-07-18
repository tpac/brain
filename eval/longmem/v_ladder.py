"""P4 validation ladder (§20.18 V0–V5) — pooled-corpus build gates.

Runs the substrate rungs against a FINISHED pooled corpus + its walker
artifacts. Each rung is GREEN/RED with exact counts; any RED exits nonzero.
Rungs V3 (=C2 empty-stack identity) and V4 (=C3 walker parity) are engine
controls — they fire with Leg B, not here.

  V0 — echoes the build's own audit from the manifest (already gated at
       build time; re-reported so the ladder is one document).
  V1 — trace-vector coverage: every conversational s0 event embedded, and
       walker op_vec_source=='store' on every labeled turn (the silent-
       component-death class: a missing substrate degrades A1 toward A0).
  V2 — moment-substrate: session-first turns have empty stacks; every
       non-first turn's stack rows (preceding completed turns) resolve to
       stored vectors; machine turns counted, not stacked (§20.17 W2).
  V5 — label sanity: soft_max non-degenerate; response-resolution ledger;
       all oracle has_answer turns present in the walker cue set.

Run (after walker extract/embed/scores/soft_usage with
WALKER_OUT_DIR=<corpus>/walker):
  ./dev python3 eval/longmem/v_ladder.py --corpus 74aea3
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import corpus_dir  # noqa: E402


def open_ro(path):
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def rung(name, green, detail):
    print("[%s] %s — %s" % (name, "GREEN" if green else "RED", detail))
    return green


def evidence_sids(oracle_path, qids):
    """(sid, content) for every has_answer USER turn — only user turns are
    op-cues. has_answer turns on the ASSISTANT side (2 in dev20: the evidence
    lives in the model's reply) are returned separately as a ledger count —
    the pre-reg's "28 evidence turns" = 26 op-cue + 2 assistant-side.
    Sids via the build's own deterministic scheme
    (build_corpus._pooled_session_plan)."""
    items = {i["question_id"]: i for i in json.loads(Path(oracle_path).read_text())}
    out, assistant_side = [], 0
    for qid in qids:
        item = items[qid]
        for sess_idx, session in enumerate(item.get("haystack_sessions", [])):
            h = hashlib.sha1(("%s|%d" % (qid, sess_idx)).encode()).hexdigest()
            sid = "i%s-%s-s%d" % (h[:7], qid, sess_idx)
            for t in session:
                if not t.get("has_answer"):
                    continue
                if t.get("role") == "user":
                    out.append((sid, t["content"]))
                else:
                    assistant_side += 1
    return out, assistant_side


_DATE_PREFIX = __import__("re").compile(r"^\[Current date:[^\]]*\]\s*")


def _strip_date_prefix(text):
    """The harness prepends '[Current date: …]' to every replayed user turn
    (conversation-time discipline) — strip it before oracle text matching."""
    return _DATE_PREFIX.sub("", text or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--walker-db", default=None,
                   help="default: <corpus_dir>/walker/walker.db")
    p.add_argument("--oracle", default=str(Path(__file__).parent /
                                           "data/longmemeval_oracle.json"))
    args = p.parse_args()

    cdir = Path(corpus_dir(args.corpus))
    manifest = json.loads((cdir / "manifest.json").read_text())
    logs = open_ro(cdir / "pooled" / "brain_logs.db")
    wpath = Path(args.walker_db) if args.walker_db else cdir / "walker" / "walker.db"
    walker = open_ro(wpath) if wpath.exists() else None
    greens = []

    # ── V0 — echo the build's audit ─────────────────────────────────────
    a = manifest["pooled_audit"]
    be = a["build_errors"]
    red = be.get("red_count", be["count"])
    greens.append(rung(
        "V0", a.get("green", False) and red == 0,
        "sessions=%d turns=%d/%d monotonic=%s errors=%d (red=%d benign=%d) "
        "nodes=%s" % (a["sessions"], a["user_turns_replayed"], a["user_turns"],
                      a["dates_monotonic"], be["count"], red,
                      be.get("benign_count", 0), a.get("node_count"))))

    # ── V1 — trace-vector coverage ──────────────────────────────────────
    cov = {}
    for rt, total, emb in logs.execute(
            "SELECT te.ref_type, COUNT(*), "
            "SUM(CASE WHEN e.trace_id IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM trace_events te LEFT JOIN trace_embeddings e "
            "ON e.trace_id = te.id "
            "WHERE te.ref_type IN ('user_message','assistant_message') "
            "GROUP BY te.ref_type"):
        cov[rt] = (emb, total)
    v1_logs = all(e == t for e, t in cov.values()) and len(cov) == 2
    detail = " ".join("%s=%d/%d" % (k, *v) for k, v in sorted(cov.items()))
    if walker is not None:
        src = dict(walker.execute(
            "SELECT COALESCE(op_vec_source,'NULL'), COUNT(*) FROM turns "
            "WHERE labeled=1 GROUP BY 1"))
        v1_walk = set(src) == {"store"}
        detail += " | walker op_vec_source: %s" % src
    else:
        v1_walk = False
        detail += " | walker.db MISSING at %s" % wpath
    greens.append(rung("V1", v1_logs and v1_walk, detail))

    # ── V2 — moment-substrate ───────────────────────────────────────────
    if walker is not None:
        rows = walker.execute(
            "SELECT session_id, epoch, seq, op_vec IS NOT NULL, "
            "anchor_vec IS NOT NULL, flags FROM turns "
            "ORDER BY session_id, epoch, seq").fetchall()
        by_se = {}
        for sid, ep, seq, has_op, has_anchor, flags in rows:
            by_se.setdefault((sid, ep), []).append(
                (seq, has_op, has_anchor, json.loads(flags or "[]")))
        firsts = len(by_se)
        machine = sum(1 for turns in by_se.values()
                      for t in turns if "machine_turn" in t[3])
        stack_sizes, unresolved = [], 0
        for turns in by_se.values():
            # a turn's stack = preceding COMPLETED turns (op+anchor vec);
            # machine turns contribute anchor-only rows (op emptied, W2)
            for i in range(len(turns)):
                stack = turns[:i]
                stack_sizes.append(len(stack))
                unresolved += sum(
                    1 for s in stack
                    if not (s[2] and (s[1] or "machine_turn" in s[3])))
        nonzero = sum(1 for s in stack_sizes if s)
        v2 = unresolved == 0
        greens.append(rung(
            "V2", v2,
            "sessions(epochs)=%d turns=%d stacks nonzero=%d zero=%d "
            "(session-firsts=%d) machine_turns=%d unresolved_stack_rows=%d"
            % (firsts, len(stack_sizes), nonzero,
               len(stack_sizes) - nonzero, firsts, machine, unresolved)))
    else:
        greens.append(rung("V2", False, "walker.db missing"))

    # ── V5 — label sanity + evidence-turn presence ──────────────────────
    if walker is not None:
        try:
            vals = [r[0] for r in walker.execute(
                "SELECT soft_max FROM soft_usage WHERE soft_max IS NOT NULL")]
        except sqlite3.OperationalError:
            vals = []
        if vals:
            import statistics
            std = statistics.pstdev(vals)
            degenerate = std < 0.01 or len(set(round(v, 4) for v in vals)) < 10
        else:
            std, degenerate = 0.0, True
        ev, assistant_side = evidence_sids(args.oracle, manifest["config"]["qids"])
        ops_by_sid = {}
        for sid, op in walker.execute(
                "SELECT session_id, op_text FROM turns WHERE labeled=1"):
            ops_by_sid.setdefault(sid, []).append(_strip_date_prefix(op))
        found = 0
        missing = []
        for sid, content in ev:
            if any(o.startswith(content[:200]) for o in ops_by_sid.get(sid, [])):
                found += 1
            else:
                missing.append((sid, content[:60]))
        v5 = (not degenerate) and found == len(ev)
        greens.append(rung(
            "V5", v5,
            "soft rows=%d std=%.4f degenerate=%s | evidence op-cues %d/%d in "
            "cue set (+%d assistant-side, not cues)%s"
            % (len(vals), std, degenerate, found, len(ev), assistant_side,
               "" if not missing else " MISSING: %s" % missing[:5])))
    else:
        greens.append(rung("V5", False, "walker.db missing"))

    print("\nladder: %s  (V3=C2, V4=C3 fire with Leg B)" %
          ("ALL GREEN" if all(greens) else "RED"))
    return 0 if all(greens) else 1


if __name__ == "__main__":
    sys.exit(main())
