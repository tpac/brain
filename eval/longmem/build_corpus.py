"""Stage 1 — build a frozen corpus (encode once, content-addressed).

This is the expensive stage: it replays each item's haystack through the full
S0/S1/S2 loop and KEEPS the resulting brain on disk. It runs once per encoding
config; a second invocation with the same inputs is a cache hit and does no
work. Every recall experiment (Stage 2, sweep.py) then reuses these frozen
brains instead of re-encoding — that's the ~100× speedup and the source of
honest A/B attribution.

Two diagnostics are computed here, where the brain is already built:
  - Answerability: `_scan_brain_for_gold` on the frozen brain. Items whose
    gold fact never got encoded are marked unanswerable and excluded from
    recall scoring downstream — an ENCODE coverage finding, not a recall miss.
  - S2 delta: what consolidation/community/healer actually did (and any
    errors they swallowed) during the build.

USE
    ./dev python3 eval/longmem/build_corpus.py --items 2 --label baseline
    ./dev python3 eval/longmem/build_corpus.py --qids temporal_a1b2 --s1e eval/prompts/s1e_v24.txt --label s1e_v24
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.longmem.harness import (
    stratified_sample, _item_axis, _apply_s1e_override, _apply_surface_override,
)
from eval.longmem.replay import replay_item
from eval.longmem.fresh_brain import create_fresh_eval_brain
from eval.longmem.classifier import _scan_brain_for_gold
from eval.longmem.corpus import (
    corpus_config_hash, corpus_dir, corpus_item_dir, manifest_path,
    load_manifest, save_manifest, source_token, summarize_s2_deltas,
    merge_s2_totals,
)


def _load_env() -> None:
    """Load .env (override empty vars — setdefault skips empty strings)."""
    envf = Path(".env")
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            key, val = k.strip(), v.strip().strip('"').strip("'")
            if not os.environ.get(key):
                os.environ[key] = val


def _gold_str(item: dict) -> str:
    a = item["answer"]
    return a if isinstance(a, str) else json.dumps(a)


def _read_build_errors(brain) -> dict:
    """Snapshot errors logged during this item's build.

    Errors land in `debug_log` (event_type='error') via `brain._log_error` —
    NOT a `brain_errors` table (the harness's `brain_errors` query is a dead
    no-op against a non-existent table). Each item builds a fresh brain, so
    every error row here is from this build. Catches the guard's
    `brain_batch_stale_txn` and any unit-level S2 exception the coordinator
    logged — the loud half of "spot S2 issues".
    """
    try:
        count = brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='error'").fetchone()[0]
        rows = brain.logs_conn.execute(
            "SELECT source, metadata FROM debug_log WHERE event_type='error' "
            "ORDER BY id DESC LIMIT 10").fetchall()
    except Exception:
        return {"count": 0, "samples": []}
    samples = []
    for source, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}
        samples.append({
            "source": source,
            "error": (meta.get("error") or "")[:160],
            "context": (meta.get("context") or "")[:120],
        })
    return {"count": count, "samples": samples}


def build_corpus(items_per_axis: int, seed: int, oracle: str,
                 s1e: str, ingest_surface: str, s2_every_n: int,
                 label: str, qids: str = None, force: bool = False) -> str:
    _load_env()

    with open(oracle) as f:
        data = json.load(f)

    if qids:
        wanted = [q.strip() for q in qids.split(",") if q.strip()]
        by_id = {it["question_id"]: it for it in data}
        picked = [by_id[q] for q in wanted if q in by_id]
        missing = [q for q in wanted if q not in by_id]
        if missing:
            print(f"[corpus] WARN qids not in oracle: {missing}", flush=True)
        if not picked:
            print("[corpus] no valid qids — exiting", file=sys.stderr)
            sys.exit(1)
    else:
        picked = stratified_sample(data, per_axis=items_per_axis, seed=seed)

    qid_list = sorted(it["question_id"] for it in picked)

    # Content address: everything that determines the encoded graph.
    config = {
        "s1e": source_token(s1e),
        "ingest_surface": source_token(ingest_surface),
        "s2_every_n": s2_every_n,
        "oracle": os.path.basename(oracle),
        "qids": qid_list,
    }
    h = corpus_config_hash(config)
    print(f"[corpus] config hash = {h}  ({config['s1e']} / surface={config['ingest_surface']} "
          f"/ s2_every_n={s2_every_n} / {len(qid_list)} items)", flush=True)

    if load_manifest(h) and not force:
        print(f"[corpus] CACHE HIT — manifest exists at {manifest_path(h)}; "
              f"0 re-encoding. Pass --force to rebuild.", flush=True)
        return h

    # Surface override needs the agentic-loop env var, same as harness.
    if ingest_surface != "active":
        os.environ["BRAIN_SURFACE_VARIANT"] = "v5_agentic"

    os.makedirs(corpus_dir(h), exist_ok=True)
    items_manifest = []
    t_run0 = time.time()

    def _save_manifest_now():
        """Rebuild + persist the manifest from items-so-far. Called after every
        item so a crash during a long unattended build preserves completed items
        (their frozen brains already exist on disk; this keeps the index in sync)."""
        ac = sum(1 for it in items_manifest if it.get("answerable"))
        m = {
            "corpus_hash": h, "label": label, "created_at_epoch": time.time(),
            "config": config, "items": items_manifest,
            "answerable_count": ac,
            "unanswerable_count": len(items_manifest) - ac,
            "s2_totals": merge_s2_totals(items_manifest),
            "build_errors_total": sum(
                it.get("build_errors", {}).get("count", 0) for it in items_manifest),
            "build_ms": int((time.time() - t_run0) * 1000),
        }
        return save_manifest(h, m), m

    for idx, item in enumerate(picked):
        qid = item["question_id"]
        axis = _item_axis(item)
        n_turns = sum(len(s) for s in item.get("haystack_sessions", []))
        print(f"\n{'='*70}")
        print(f"[corpus] item {idx+1}/{len(picked)} qid={qid} axis={axis} turns={n_turns}", flush=True)

        path = corpus_item_dir(h, qid)
        brain = create_fresh_eval_brain(path=path, wipe=True)

        if s1e != "active":
            _apply_s1e_override(brain, s1e)
        if ingest_surface != "active":
            _apply_surface_override(brain, ingest_surface)

        t0 = time.time()
        stats = replay_item(
            brain, f"ingest-{qid}", item["haystack_sessions"],
            haystack_dates=item.get("haystack_dates"),
            log_prefix=f"[item {idx+1}]", s2_every_n=s2_every_n,
        )
        encode_ms = int((time.time() - t0) * 1000)

        # Answerability gate — does the frozen graph carry the gold fact?
        scan = _scan_brain_for_gold(brain, _gold_str(item))
        answerable = bool(scan.get("found"))
        verdict = "ANSWERABLE" if answerable else "UNANSWERABLE (ENCODE_MISS)"
        print(f"[corpus]   answerability: {verdict} "
              f"(terms={scan.get('terms_used')})", flush=True)

        s2_delta = summarize_s2_deltas(stats.get("s2_deltas", []))
        s2_errs = sum(u.get("errors", 0) for u in s2_delta.values())
        if s2_errs:
            print(f"[corpus]   ⚠ S2 logged {s2_errs} error(s) during build — see manifest", flush=True)

        # Loud half: any error written to debug_log during this build (the
        # guard's brain_batch_stale_txn, coordinator-level S2 exceptions, ...).
        build_errors = _read_build_errors(brain)
        if build_errors["count"]:
            srcs = ', '.join(sorted({s["source"] for s in build_errors["samples"]}))
            print(f"[corpus]   ⚠ {build_errors['count']} error(s) in debug_log during "
                  f"build: {srcs}", flush=True)

        # Checkpoint WAL so the frozen dir is a clean, copyable snapshot.
        try:
            brain.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        try:
            brain.close()
        except Exception:
            pass

        items_manifest.append({
            "qid": qid,
            "axis": axis,
            "question": item["question"],
            "gold": _gold_str(item),
            "question_date": item.get("question_date"),
            "turns": n_turns,
            "encode_ms": encode_ms,
            "s1e_runs": stats.get("s1e_runs"),
            "s2_runs": stats.get("s2_runs"),
            "answerable": answerable,
            "gold_scan": {
                "found": scan.get("found"),
                "terms_used": scan.get("terms_used"),
                "phrase_used": scan.get("phrase_used"),
                "matches": scan.get("matches", []),
            },
            "s2_delta": s2_delta,
            "build_errors": build_errors,
            "brain_dir": path,
        })
        _save_manifest_now()  # incremental — a crash mid-build keeps prior items

    path, manifest = _save_manifest_now()

    print(f"\n[corpus] done in {manifest['build_ms']/1000:.1f}s", flush=True)
    print(f"[corpus] ANSWERABLE: {manifest['answerable_count']}/{len(items_manifest)}   "
          f"UNANSWERABLE: {manifest['unanswerable_count']}/{len(items_manifest)}", flush=True)
    print(f"[corpus] S2 totals: {json.dumps(manifest['s2_totals'])}", flush=True)
    print(f"[corpus] manifest → {path}", flush=True)
    print(f"[corpus] sweep it:  ./dev python3 eval/longmem/sweep.py --corpus {h} --label <run>", flush=True)
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=int, default=2, help="per-axis item count (total = items × 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oracle", default="eval/longmem/data/longmemeval_oracle.json")
    p.add_argument("--s1e", default="active",
                   help="'active' (seeded v1) or a path to an s1e prompt file to encode with")
    p.add_argument("--ingest-surface", dest="ingest_surface", default="active",
                   help="'active' or a path to a surface prompt file used during ingest recall")
    p.add_argument("--s2-every-n", dest="s2_every_n", type=int, default=2,
                   help="S2 fires every N encodings during ingest (default 2)")
    p.add_argument("--label", default="corpus", help="human label stored in the manifest")
    p.add_argument("--qids", default=None, help="comma-separated qids (overrides stratified sampling)")
    p.add_argument("--force", action="store_true", help="rebuild even if the corpus already exists")
    args = p.parse_args()

    build_corpus(args.items, args.seed, args.oracle, args.s1e, args.ingest_surface,
                 args.s2_every_n, args.label, qids=args.qids, force=args.force)


if __name__ == "__main__":
    main()
