#!/usr/bin/env python3
"""Adjudication + verification harness for the gold re-mint cards.

Reads cards/card_<cid>_{a,b}.json (2 blind judges per cue) and produces:
  - INTEGRITY POST-FILTER (needs moments.json for cutoffs + read-only brain.db): drops cited
    nodes that can't be legitimate gold — hallucinated (not in brain), archived (no embedding =
    unrecallable), leakage (created_at > cutoff), or graft (essential with revised_at > cutoff,
    so the judge read post-cutoff content we can't verify). Conservative: integrity > coverage.
  - node-level four-tier classification (Gold+ both-essential / Gold one-essential /
    Silver+ both-silver / Silver one-silver) — the LOCKED tier definition, computed AFTER filtering
  - need-commentary per node (the judge's `expresses`) so equivalence is readable
  - ISSUES sweep — every non-empty `issues` field, the tool-health check (silent-0 guard)
  - structural integrity — missing cards, worthwhile:false cues, encode_gaps
  - old-gold overlap (new Gold+∪Gold vs the frozen old essential) per cue + aggregate

Read-only. Run after a judge Workflow batch (regenerate moments.json first via build_moments.py
so the filter has cutoffs):
  ./dev python3 eval/oracle_audit/gold_remint/analyze_cards.py
"""
import json, os, glob, sqlite3
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = os.path.join(HERE, "cards")
KEY = os.path.join(HERE, "old_gold_key.json")
MOMENTS = os.path.join(HERE, "moments.json")


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def cue_ids():
    ids = set()
    for f in glob.glob(os.path.join(CARDS, "card_*_a.json")) + glob.glob(os.path.join(CARDS, "card_*_b.json")):
        b = os.path.basename(f)
        ids.add(b[len("card_"):-len("_a.json")])
    return sorted(ids)


def node_meta(ids):
    """{node_id: (created_at, revised_at, archived)} from a READ-ONLY brain.db connection."""
    dbdir = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
    out = {}
    try:
        con = sqlite3.connect("file:%s?mode=ro" % os.path.join(dbdir, "brain.db"), uri=True)
    except Exception:
        return out
    ids = list(ids)
    for k in range(0, len(ids), 400):
        chunk = ids[k:k + 400]
        ph = ",".join("?" * len(chunk))
        for r in con.execute("SELECT id,created_at,revised_at,archived FROM nodes WHERE id IN (%s)" % ph, chunk):
            out[r[0]] = (r[1], r[2], r[3])
    con.close()
    return out


def drop_reason(nid, kind, cutoff, meta):
    """Why this cited node is not legitimate gold at this cue's cutoff — else None (keep).
    archived/leakage/hallucinated invalidate any tier; graft (revised post-cutoff) drops essentials only."""
    m = meta.get(nid)
    if m is None:
        return "hallucinated"
    created, revised, archived = m
    if archived:
        return "archived"
    if created and cutoff and created > cutoff:
        return "leakage"            # node didn't exist at recall time
    if kind == "essential" and revised and cutoff and revised > cutoff:
        return "graft"              # content read post-dates cutoff, unverifiable
    return None


def classify(card, cutoff, meta, drops, dropped, cid, j):
    """node_id -> (kind, form, title, expr), AFTER the integrity filter. essential wins ties."""
    out = {}
    if not card:
        return out
    use_filter = bool(cutoff) and bool(meta)
    for kind in ("essential", "silver"):           # essential first so it wins setdefault
        for e in card.get(kind, []):
            nid = e.get("node_id")
            if not nid:
                continue
            if use_filter:
                reason = drop_reason(nid, kind, cutoff, meta)
                if reason:
                    drops[reason] += 1
                    dropped.append((cid, j, kind, nid, reason))
                    continue
            out.setdefault(nid, (kind, e.get("form", ""), e.get("title", ""), e.get("expresses", "")))
    return out


def tier(a, b):
    if a == "essential" and b == "essential":
        return "GOLD+"
    if a == "essential" or b == "essential":
        return "GOLD"
    if a == "silver" and b == "silver":
        return "SILVER+"
    return "SILVER"


def main():
    key = load(KEY) or {}
    moments = {m["cue_id"]: m for m in (load(MOMENTS) or [])}
    cues = cue_ids()
    if not cues:
        print("no cards found in %s" % CARDS); return

    # pre-load every card + collect cited ids, so we fetch brain meta in one shot
    cards = {}
    cited = set()
    for c in cues:
        for j in ("a", "b"):
            card = load(os.path.join(CARDS, "card_%s_%s.json" % (c, j)))
            cards[(c, j)] = card
            for kind in ("essential", "silver"):
                for e in (card or {}).get(kind, []):
                    if e.get("node_id"):
                        cited.add(e["node_id"])
    meta = node_meta(cited) if moments else {}
    filter_on = bool(moments) and bool(meta)

    agg = Counter()
    issues_found, missing, worthless = [], [], []
    drops, dropped = Counter(), []
    encode_gap_total = 0
    overlap_num = overlap_den = 0
    per_cue = []

    for c in cues:
        A, B = cards[(c, "a")], cards[(c, "b")]
        cutoff = (moments.get(c) or {}).get("cutoff")
        if A is None:
            missing.append((c, "a"))
        if B is None:
            missing.append((c, "b"))
        for j, card in (("a", A), ("b", B)):
            iss = (card or {}).get("issues", "")
            if isinstance(iss, str) and iss.strip() and iss.strip().lower() not in ("none", "no issues", "issues: none"):
                issues_found.append((c, j, iss.strip()))
        wa = (A or {}).get("worthwhile", True); wb = (B or {}).get("worthwhile", True)
        if wa is False and wb is False:
            worthless.append(c)
        encode_gap_total += len((A or {}).get("encode_gaps", [])) + len((B or {}).get("encode_gaps", []))

        ca = classify(A, cutoff, meta, drops, dropped, c, "a")
        cb = classify(B, cutoff, meta, drops, dropped, c, "b")
        nodes = set(ca) | set(cb)
        tiers = defaultdict(list)
        for n in nodes:
            ta = ca.get(n, ("none", "", "", ""))[0]
            tb = cb.get(n, ("none", "", "", ""))[0]
            m = ca.get(n) or cb.get(n)
            t = tier(ta, tb)
            tiers[t].append((n, m[1], m[2], m[3]))
            agg[t] += 1
        per_cue.append((c, tiers, wa, wb))
        old = set((key.get(c) or {}).get("old_essential", []))
        if old:
            new_ess = {n for n, _, _, _ in tiers["GOLD+"]} | {n for n, _, _, _ in tiers["GOLD"]}
            overlap_num += len(new_ess & old); overlap_den += len(old)

    # ---- report ----
    print("=" * 72)
    print("GOLD RE-MINT — CARD ANALYSIS  (%d cues, %d cards)" % (len(cues), len(cues) * 2 - len(missing)))
    print("filter: %s" % ("ON (moments.json + brain.db)" if filter_on else "OFF — no moments.json/brain; tiers UNFILTERED"))
    print("=" * 72)

    print("\n## INTEGRITY POST-FILTER (dropped before tiering)")
    if not filter_on:
        print("  (skipped — regenerate moments.json via build_moments.py to enable)")
    elif not dropped:
        print("  ✓ nothing dropped — every cited node exists, is unarchived, predates cutoff, no essential grafted")
    else:
        for reason in ("hallucinated", "leakage", "archived", "graft"):
            n = drops[reason]
            if n:
                ex = [d[3][:8] for d in dropped if d[4] == reason][:8]
                print("  %-12s %d   %s" % (reason, n, ex))
        print("  detail (cue/judge/kind/node/reason):")
        for cid, j, kind, nid, reason in dropped:
            print("    %s/%s  %-9s %s  %s" % (cid, j, kind, nid[:8], reason))

    print("\n## TOOL-HEALTH / ISSUES SWEEP  (the silent-0 guard)")
    if not issues_found:
        print("  ✓ no issues reported by any judge")
    else:
        print("  ⚠ %d issue report(s):" % len(issues_found))
        for c, j, iss in issues_found:
            print("    [%s/%s] %s" % (c, j, iss[:200]))

    print("\n## STRUCTURAL INTEGRITY")
    print("  missing cards: %s" % (missing or "none"))
    print("  worthwhile:false (both judges): %d  %s" % (len(worthless), worthless or ""))
    print("  encode_gaps flagged (total across judges): %d" % encode_gap_total)

    print("\n## TIER TOTALS (node-level, post-filter)")
    for t in ("GOLD+", "GOLD", "SILVER+", "SILVER"):
        print("  %-8s %d" % (t, agg[t]))
    print("  cues with ≥1 GOLD+ node: %d/%d" %
          (sum(1 for _, ti, _, _ in per_cue if ti["GOLD+"]), len(cues)))

    if overlap_den:
        print("\n## OLD-GOLD OVERLAP  (new Gold+∪Gold ∩ old essential / old essential)")
        print("  %d/%d = %.0f%%   (low = lens-independence working; a property, not a target)"
              % (overlap_num, overlap_den, 100 * overlap_num / overlap_den))

    print("\n## PER-CUE TIERS  (form · need-commentary)")
    for c, tiers, wa, wb in per_cue:
        w = "" if (wa and wb) else "  [worthwhile a=%s b=%s]" % (wa, wb)
        print("\n══ %s%s ══" % (c, w))
        for t in ("GOLD+", "GOLD", "SILVER+", "SILVER"):
            for n, form, title, expr in tiers[t]:
                tag = ("[%s]" % form) if form else ""
                print("  %-7s %s %-8s %s" % (t, n[:8], tag, (expr or title)[:80]))


if __name__ == "__main__":
    main()
