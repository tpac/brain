#!/usr/bin/env python3
"""Build CLEAN recall-moment bundles for the lens-independent gold re-mint (§18.19).

A recall moment is a CONVERSATION WINDOW, not a bag of fields (Tom, 2026-06-28): the judge
needs the actual back-and-forth — ~3 turns before (operator+Anchor interleaved) → the cue turn
where recall fires → ~1-2 turns after (the outcome, incl. the operator's reaction / any
interjection). The frozen endo corpus instead reconstructed lossy disjoint fields (tail-sliced
prompt, char-capped prev_*, a single next turn), and prev_operator silently duplicated the cue
turn for 27/39 operator cues. This rebuilds the real window from the raw traces.

  conversation_before  ~6 messages before the cue, labeled OPERATOR/ANCHOR, chronological (<= cutoff)
  cue                  the turn recall fires on (operator prompt OR Anchor's stop)
  conversation_after   ~4 messages after the cue — the OUTCOME the judge reasons from (> cutoff)
  cutoff               only nodes created_at <= cutoff existed at recall time

Deterministic, READ-ONLY over brain_logs.db, no embedder, no LLM spend. Cue set + (session,cutoff)
come from the committed state_cues.json. old_gold is carried in a SEPARATE section the judge must
NOT see — kept only to measure overlap(old_essential, new_essential) after the re-mint.

Run: ./dev python3 eval/oracle_audit/gold_remint/build_moments.py
Out: eval/oracle_audit/gold_remint/moments.json
"""
import json, os, sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "endo_corpus")
OUT = os.path.join(HERE, "moments.json")
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")

BEFORE_MSGS = 6           # ~3 turns each side before the cue
AFTER_MSGS = 4            # ~1-2 turns after the cue (the outcome window)
TURN_CAP = 3000           # per-message cap inside the window
CUE_CAP = 4000            # the cue + immediate move get the full stored turn (<=4000 at source)


def content(md):
    try:
        return (json.loads(md).get("content") or "").strip()
    except Exception:
        return ""


def head_snap(text, cap):
    """Head-slice to <=cap, snap END back to a clean boundary (no mid-word cut)."""
    t = (text or "").strip()
    if len(t) <= cap:
        return t
    cut = t[:cap]
    for sep in ("\n\n", "\n", ". ", "! ", "? "):
        i = cut.rfind(sep)
        if i > cap * 0.6:
            return cut[:i + len(sep)].strip()
    i = cut.rfind(" ")
    return ((cut[:i] if i > 0 else cut).strip()) + " …"


def render(rows, cap):
    """Labeled, chronological transcript of trace rows."""
    return "\n\n".join(
        "%s: %s" % ("OPERATOR" if r["ref_type"] == "user_message" else "ANCHOR",
                    head_snap(content(r["metadata"]), cap))
        for r in rows if content(r["metadata"]))


def main():
    state = json.load(open(os.path.join(CORPUS, "state_cues.json")))
    corpus = {c["id"]: c for c in json.load(open(os.path.join(CORPUS, "endo_gold_corpus.json")))}
    con = sqlite3.connect("file:%s?mode=ro" % os.path.join(DBDIR, "brain_logs.db"), uri=True)
    con.row_factory = sqlite3.Row

    out, miss = [], []
    for st in state:
        cid, sess, cutoff, src = st["cue_id"], st["session"], st["cutoff"], st["source"]
        c = corpus.get(cid, {})
        rows = con.execute(
            "SELECT created_at, ref_type, metadata FROM trace_events "
            "WHERE session_id=? AND ref_type IN ('user_message','assistant_message') "
            "ORDER BY created_at", (sess,)).fetchall()
        if not rows:
            miss.append((cid, "no turns")); continue

        # locate the cue turn. operator_msg: the user message that fired recall = last
        # user_message <= cutoff. anchor_turn: the assistant turn AT the cutoff.
        want = "user_message" if src == "operator_msg" else "assistant_message"
        idxs = [i for i, r in enumerate(rows)
                if r["ref_type"] == want and r["created_at"] <= cutoff]
        if not idxs:
            miss.append((cid, "no cue turn")); continue
        ci = idxs[-1]

        before = rows[max(0, ci - BEFORE_MSGS):ci]
        after = rows[ci + 1: ci + 1 + AFTER_MSGS]
        cue_text = head_snap(content(rows[ci]["metadata"]), CUE_CAP)
        outcome = render(after, CUE_CAP)
        if not cue_text or not outcome:
            miss.append((cid, "empty cue or outcome")); continue

        out.append({
            "cue_id": cid,
            "source": src,
            "query_type": c.get("query_type"),
            "cutoff": cutoff,
            "conversation_before": render(before, TURN_CAP),   # <= cutoff — what recall saw
            "cue": {"speaker": "OPERATOR" if src == "operator_msg" else "ANCHOR", "text": cue_text},
            "conversation_after": outcome,                      # > cutoff — the OUTCOME / label
            # JUDGE MUST NOT SEE THIS — carried only for post-hoc overlap measurement.
            "_old_gold": {"essential": c.get("gold_essential", []),
                          "helpful": c.get("gold_helpful", []),
                          "lens": c.get("gold_lens", {})},
        })
    con.close()
    json.dump(out, open(OUT, "w"), indent=1)

    # ---- verification summary ----
    import statistics as S
    print("built %d conversation-window moments -> %s  (%d skipped: %s)"
          % (len(out), OUT, len(miss), miss[:5]))
    by = {}
    for m in out:
        by.setdefault(m["source"], []).append(m)
    for src, ms in by.items():
        bl = [len(m["conversation_before"]) for m in ms]
        al = [len(m["conversation_after"]) for m in ms]
        nb = [m["conversation_before"].count("OPERATOR:") + m["conversation_before"].count("ANCHOR:") for m in ms]
        print("  %-12s n=%d  before chars med=%d (med %d msgs)  after chars med=%d"
              % (src, len(ms), int(S.median(bl)), int(S.median(nb)), int(S.median(al))))
    print("\n=== SAMPLE: operator_msg_1373 (the 'remove connect' cue) ===")
    ex = next((m for m in out if m["cue_id"] == "operator_msg_1373"), None)
    if ex:
        print("--- conversation_before ---\n%s" % ex["conversation_before"][-900:])
        print("\n--- CUE (%s) ---\n%s" % (ex["cue"]["speaker"], ex["cue"]["text"][:400]))
        print("\n--- conversation_after ---\n%s" % ex["conversation_after"][:700])


if __name__ == "__main__":
    main()
