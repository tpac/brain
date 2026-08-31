"""Read first-session rehearsal arms side by side.

Mechanical half of the comparison — what surfaced, what got written, how the
newborn's own nodes are shaped. The qualitative half (did this entity feel
trustworthy to a stranger; did it handle the correction well) is a human read
of the transcripts this prints; no metric substitutes for it.

Checks the three recall gaps rehearsal #1 found (id:2a9aa2c7), which the
operator script deliberately re-triggers:
  turn 7  — a correction        → does corrections_are_treasure surface?
  turn 9  — "do you have opinions?" → does collaborator_stance surface?
  turn 12 — casual leave-taking → does dev_how_we_end_sessions surface?

USE
    ./dev python3 eval/nursery/compare.py --arms r2_old_historical,r2_old_plus_block,r2_new
    ./dev python3 eval/nursery/compare.py --arms <a>,<b> --turn 7      # one turn, full text
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TRANSCRIPTS = "eval/nursery/transcripts"

# Seed slugs whose surfacing rehearsal #1 flagged as missing, by turn number.
GAP_CHECKS = {
    7: ("correction", ["corrections are treasure", "corrections_are_treasure",
                       "correction is treasure", "revise, don't duplicate",
                       "revise_not_duplicate"]),
    9: ("opinions", ["collaborator", "collaborator_stance"]),
    12: ("ending", ["end sessions", "dev_how_we_end_sessions", "how we end"]),
}

FIELDS = ("situation", "question", "reasoning", "their_raw_quote")


def load(label: str) -> dict:
    p = os.path.join(TRANSCRIPTS, f"{label}.json")
    with open(p) as f:
        return json.load(f)


def surfaced_titles(mem: str) -> list:
    """Titles inside a memory-context block, as rendered by the surfacer."""
    return re.findall(r'"([^"\n]{12,110})"', mem or "")


def arm_stats(d: dict) -> dict:
    turns = d["turns"]
    written = d["final"]["written"]
    ops = [o for t in turns for o in (t["memory_ops"] or [])]
    remembers = [o for o in ops if (o.get("op") or "").lower() == "remember"]
    revises = [o for o in ops if (o.get("op") or "").lower() == "revise"]
    cov = {f: sum(1 for o in remembers if (o.get(f) or "").strip()) for f in FIELDS}
    when = sum(1 for o in remembers
               if (o.get("situation") or "").strip().lower().startswith("when"))
    confs = [o.get("confidence") for o in remembers if isinstance(o.get("confidence"), (int, float))]
    return {
        "label": d["label"], "pack": os.path.basename(str(d["pack"])),
        "generation": d["generation"], "boot_block": d["boot_block"],
        "boot_chars": len(d["boot_text"]), "seeds": d["final"]["seeds"],
        "turns": len(turns),
        "parse_errors": sum(1 for t in turns if t["parse_error"]),
        "truncated": sum(1 for t in turns if t.get("truncated")),
        "recall_chars": [t["memory_context_chars"] for t in turns],
        "recall_silent_turns": sum(1 for t in turns if t["memory_context_chars"] == 0),
        "reply_chars_mean": round(sum(len(t["reply"]) for t in turns) / max(len(turns), 1)),
        "remembers": len(remembers), "revises": len(revises),
        "applied_ok": sum(1 for t in turns for a in t["applied"] if a.get("ok")),
        "applied_fail": sum(1 for t in turns for a in t["applied"] if not a.get("ok")),
        "nodes_written": d["final"]["nodes_written"],
        "field_cov": cov, "when_trigger": when,
        "conf_mean": round(sum(confs) / len(confs), 3) if confs else None,
        "conf_at_1": sum(1 for c in confs if c >= 1.0),
        "types": sorted({(w.get("type") or "?") for w in written}),
        "tokens_out": d["tokens_out"], "elapsed_s": d["elapsed_s"],
    }


def gap_report(d: dict) -> dict:
    out = {}
    for n, (name, needles) in GAP_CHECKS.items():
        t = next((x for x in d["turns"] if x["n"] == n), None)
        if not t:
            out[name] = "turn missing"
            continue
        blob = (t["memory_context"] or "").lower()
        hit = any(nd.lower() in blob for nd in needles)
        out[name] = {"surfaced": hit, "recall_chars": t["memory_context_chars"]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", required=True, help="comma-separated transcript labels")
    p.add_argument("--turn", type=int, default=None,
                   help="print this turn's full text for every arm")
    args = p.parse_args()
    labels = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = []
    for l in labels:
        try:
            arms.append(load(l))
        except FileNotFoundError:
            print(f"!! missing transcript: {l}")
    if not arms:
        sys.exit(1)

    if args.turn:
        for d in arms:
            t = next((x for x in d["turns"] if x["n"] == args.turn), None)
            if not t:
                continue
            print(f"\n{'='*78}\n{d['label']}  — turn {args.turn}\n{'='*78}")
            print(f"OPERATOR: {t['operator']}\n")
            print(f"[probes: {', '.join(t['probes'])}]")
            print(f"[recall {t['memory_context_chars']}c — surfaced: "
                  f"{surfaced_titles(t['memory_context'])[:5]}]\n")
            print(f"ENTITY:\n{t['reply']}\n")
            if t["memory_ops"]:
                print(f"MEMORY OPS ({len(t['memory_ops'])}):")
                for o in t["memory_ops"]:
                    print(f"  [{o.get('op')}] {str(o.get('title') or o.get('title_match'))[:80]}")
                    if o.get("situation"):
                        print(f"      situation: {o['situation'][:110]}")
        return

    stats = [arm_stats(d) for d in arms]
    keys = ["pack", "generation", "boot_block", "boot_chars", "seeds", "turns",
            "parse_errors", "truncated", "recall_silent_turns",
            "reply_chars_mean", "remembers", "revises", "applied_ok",
            "applied_fail", "nodes_written", "when_trigger", "conf_mean",
            "conf_at_1", "tokens_out", "elapsed_s"]
    w = max(len(s["label"]) for s in stats) + 2
    print(f"\n{'metric':<22}" + "".join(f"{s['label']:>{w}}" for s in stats))
    print("-" * (22 + w * len(stats)))
    for k in keys:
        print(f"{k:<22}" + "".join(f"{str(s[k]):>{w}}" for s in stats))
    print(f"\n{'field coverage of remembers':<22}")
    for f in FIELDS:
        print(f"  {f:<20}" + "".join(f"{str(s['field_cov'][f]):>{w}}" for s in stats))
    print(f"\n{'types written':<22}")
    for s in stats:
        print(f"  {s['label']}: {s['types']}")
    print(f"\n{'recall chars per turn':<22}")
    for s in stats:
        print(f"  {s['label']}: {s['recall_chars']}")

    print("\nrehearsal #1 recall gaps — did the seed surface this time?")
    for d in arms:
        print(f"  {d['label']}: {json.dumps(gap_report(d))}")


if __name__ == "__main__":
    main()
