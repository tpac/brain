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


def _is_benign_build_error(source: str, context: str) -> bool:
    """Benign = Haiku cited an ID outside its candidate menu but it resolved
    to a real node anyway — the selection landed, nothing degraded. Everything
    else (recall_laf fallback, S2 unit exceptions, stale txns) is RED."""
    return (source == "haiku_id_outside_candidates"
            and "resolved=" in (context or ""))


def _read_build_errors(brain) -> dict:
    """Snapshot errors logged during this item's build.

    Errors land in `debug_log` (event_type='error') via `brain._log_error` —
    NOT a `brain_errors` table (the harness's `brain_errors` query is a dead
    no-op against a non-existent table). Each item builds a fresh brain, so
    every error row here is from this build. Catches the guard's
    `brain_batch_stale_txn` and any unit-level S2 exception the coordinator
    logged — the loud half of "spot S2 issues".

    Rows are classified RED vs benign (V0 gates on red_count, §20.18).
    """
    try:
        rows = brain.logs_conn.execute(
            "SELECT source, metadata FROM debug_log WHERE event_type='error' "
            "ORDER BY id DESC").fetchall()
    except Exception:
        return {"count": 0, "red_count": 0, "benign_count": 0, "samples": []}
    samples = []
    red = 0
    for source, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}
        context = (meta.get("context") or "")[:120]
        benign = _is_benign_build_error(source, context)
        if not benign:
            red += 1
        if len(samples) < 10:
            samples.append({
                "source": source,
                "error": (meta.get("error") or "")[:160],
                "context": context,
                "benign": benign,
            })
    return {"count": len(rows), "red_count": red,
            "benign_count": len(rows) - red, "samples": samples}


def _fetch_interaction_template(name: str, version: int) -> str:
    """Fetch a registered interaction version's template from the LIVE daemon.

    Used to pull DORMANT prompt versions (e.g. s1e v24, s1_scout_facts v7) so the
    eval can A/B them against the active ones. Reads production's interactions
    table via TCP; the eval brain is untouched until _apply_interaction_override.
    """
    from servers.daemon_client import send_command
    r = send_command('get_interaction', {'name': name, 'version': int(version)})
    tmpl = (r.get('result') or {}).get('template') if isinstance(r, dict) else None
    if not tmpl:
        raise RuntimeError("could not fetch %s v%s from daemon: %s" % (name, version, r))
    return tmpl


def _apply_interaction_override(brain, name: str, template: str) -> None:
    """Register + activate `template` as a new version of `name` in THIS eval
    brain only (production daemon untouched). Generalizes harness._apply_s1e_override
    to any interaction; mirrors eval.encoder_eval.targeted_v24_eval."""
    existing = brain._interaction_dal.get_active(name)
    params = existing.get('parameters', '') if existing else ''
    result = brain._interaction_dal.register(
        name, template=template, parameters=params, created_by='eval-override-%s' % name)
    if result.get('version', 1) > 1:
        brain._interaction_dal.set_active(
            name, result['version'], set_by='eval-override-%s' % name)


def build_corpus(items_per_axis: int, seed: int, oracle: str,
                 s1e: str, ingest_surface: str, s2_every_n: int,
                 label: str, qids: str = None, force: bool = False,
                 interaction_overrides: dict = None, lived: bool = False) -> str:
    _load_env()

    # The lived arm (BRAIN_S1E_LIVED_SEQUENCE) changes the ENCODED GRAPH — the
    # XML lived-sequence input, widened catalog, 2-scout muster, inline scout
    # notes all shape what S1E writes — so it must be pinned here, not inherited
    # from the shell: set it explicitly per the arg, and clear any leaked env
    # on the control arm so a stray export can't silently flip a build's arm.
    if lived:
        os.environ["BRAIN_S1E_LIVED_SEQUENCE"] = "1"
    else:
        os.environ.pop("BRAIN_S1E_LIVED_SEQUENCE", None)

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
    # Interaction overrides (e.g. DORMANT s1e v24 + s1_scout_facts v7) change the
    # encoded graph, so they're part of the content address — a v22 corpus and a
    # v24+v7 corpus get distinct hashes.
    if interaction_overrides:
        config["interaction_overrides"] = {
            k: int(v) for k, v in sorted(interaction_overrides.items())}
    # Lived arm joins the content address ONLY when on (key absent on control →
    # every pre-existing control corpus keeps its hash; no cache invalidation).
    # Without this, a lived and a control build with the same versions would
    # collide on one hash and the cache would hand back the wrong arm's corpus.
    if lived:
        config["s1e_lived"] = True
    h = corpus_config_hash(config)
    ov_str = (" / overrides=%s" % config["interaction_overrides"]) if interaction_overrides else ""
    print(f"[corpus] config hash = {h}  ({config['s1e']} / surface={config['ingest_surface']} "
          f"/ s2_every_n={s2_every_n} / {len(qid_list)} items{ov_str})", flush=True)

    if load_manifest(h) and not force:
        print(f"[corpus] CACHE HIT — manifest exists at {manifest_path(h)}; "
              f"0 re-encoding. Pass --force to rebuild.", flush=True)
        return h

    # Surface override needs the agentic-loop env var, same as harness.
    if ingest_surface != "active":
        os.environ["BRAIN_SURFACE_VARIANT"] = "v5_agentic"

    # Fetch DORMANT override templates from the live daemon once (reused per item).
    override_templates = {}
    if interaction_overrides:
        for name, version in sorted(interaction_overrides.items()):
            override_templates[name] = _fetch_interaction_template(name, version)
            print(f"[corpus] override: {name} → v{version} ({len(override_templates[name])} chars)",
                  flush=True)

    os.makedirs(corpus_dir(h), exist_ok=True)

    # Full-prompt capture (Tom, 2026-07-02): every S1E encode in a corpus build
    # dumps its literal per-round payload — the REAL prompt fed to S1Scribe, not
    # a rebuild — under the corpus itself. Files key as
    # {arm}__ingest-{qid}__stop{n}-r{round}-{pid}-{seq}.json, so any regression
    # in the A/B is inspectable at the exact prompt that produced it.
    prompts_dir = os.path.join(corpus_dir(h), "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    os.environ["BRAIN_PROMPT_CAPTURE_DIR"] = prompts_dir
    print(f"[corpus] full-prompt capture → {prompts_dir}", flush=True)

    items_manifest = []
    t_run0 = time.time()

    def _save_manifest_now():
        """Rebuild + persist the manifest from items-so-far. Called after every
        item so a crash during a long unattended build preserves completed items
        (their frozen brains already exist on disk; this keeps the index in sync)."""
        ac = sum(1 for it in items_manifest if it.get("answerable"))
        m = {
            "corpus_hash": h, "label": label, "created_at_epoch": time.time(),
            "prompts_dir": prompts_dir,
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
        for ov_name, ov_template in override_templates.items():
            _apply_interaction_override(brain, ov_name, ov_template)

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


def _pooled_session_plan(picked):
    """Flatten items' (session, date) pairs and sort by date — the §20.18
    pooled interleave. Tie-break on (qid, sess_idx) for determinism. Dates
    are LongMemEval-shaped ('2023/05/22 (Mon) 16:46'); parse properly rather
    than trusting lexicographic luck.

    Each entry carries its brain session id (`sid`). Chain ids use
    session_id[:8] (SessionContext.session_short) — production UUIDs are
    prefix-unique by construction, so the eval scheme must be too: a readable
    'ingest-{qid}-s{n}' collides ('ingest-2…' for every 2311e44b session) and
    cross-attaches chain-keyed joins (judge_output corruption, smoke cfd549
    v2). Hence the sha1 prefix; uniqueness is ASSERTED, not assumed."""
    import hashlib
    from datetime import datetime

    def _parse(d):
        try:
            return datetime.strptime(d, "%Y/%m/%d (%a) %H:%M")
        except Exception:
            raise ValueError("unparseable haystack date %r — the pooled sort "
                             "must not fall back silently" % (d,))

    plan = []
    for item in picked:
        qid = item["question_id"]
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        if len(dates) != len(sessions):
            raise ValueError("item %s: %d sessions but %d dates — pooled "
                             "interleave needs one date per session"
                             % (qid, len(sessions), len(dates)))
        for sess_idx, (session, date) in enumerate(zip(sessions, dates)):
            h = hashlib.sha1(("%s|%d" % (qid, sess_idx)).encode()).hexdigest()
            plan.append({"qid": qid, "sess_idx": sess_idx, "date": date,
                         "sid": "i%s-%s-s%d" % (h[:7], qid, sess_idx),
                         "sort_key": (_parse(date), qid, sess_idx),
                         "session": session})
    plan.sort(key=lambda e: e["sort_key"])
    shorts = {e["sid"][:8] for e in plan}
    if len(shorts) != len(plan):
        raise ValueError("pooled session shorts collide — chain ids would "
                         "cross-attach; refusing to build")
    return plan


def build_pooled_corpus(oracle: str, qids: str, s1e: str, ingest_surface: str,
                        s2_every_n: int, label: str, force: bool = False,
                        items_per_axis: int = None, seed: int = 42) -> str:
    """§20.18 pooled-brain build: the picked items' haystack sessions
    interleaved by date into ONE fresh brain — organic cross-topic
    distractors, per-conversation session ids (the moment stack's window must
    never cross a conversation boundary), one finalize_item at the end.

    Manifest shape stays sweep.py-compatible: one entry per item (question,
    gold, answerability re-scanned on the POOLED brain), every entry's
    brain_dir pointing at the same pooled brain. The V0 audit (§20.18) is
    computed here, printed, and stored under manifest['pooled_audit'] —
    downstream legs refuse to run unless it's green.
    """
    from eval.longmem.replay import finalize_item

    _load_env()
    with open(oracle) as f:
        data = json.load(f)
    if qids:
        wanted = [q.strip() for q in qids.split(",") if q.strip()]
        by_id = {it["question_id"]: it for it in data}
        missing = [q for q in wanted if q not in by_id]
        if missing:
            print(f"[pooled] WARN qids not in oracle: {missing}", flush=True)
        picked = [by_id[q] for q in wanted if q in by_id]
    else:
        picked = stratified_sample(data, per_axis=items_per_axis or 4,
                                   seed=seed)
    if not picked:
        print("[pooled] no items — exiting", file=sys.stderr)
        sys.exit(1)

    plan = _pooled_session_plan(picked)
    n_user_turns = sum(sum(1 for t in e["session"] if t.get("role") == "user")
                       for e in plan)
    qid_list = sorted(it["question_id"] for it in picked)
    config = {
        "pooled": True,
        # Harness generation joins the content address: a graph-changing
        # harness fix (v2: per-turn embed drains — decode-alive ingest,
        # trace-embedding substrate; v3: collision-free session ids) must
        # never collide with a corpus cached under the older behavior.
        "harness": 3,
        "s1e": source_token(s1e),
        "ingest_surface": source_token(ingest_surface),
        "s2_every_n": s2_every_n,
        "oracle": os.path.basename(oracle),
        "qids": qid_list,
    }
    h = corpus_config_hash(config)
    print(f"[pooled] config hash = {h}  ({len(qid_list)} items / "
          f"{len(plan)} sessions / {n_user_turns} user turns / "
          f"span {plan[0]['date']} → {plan[-1]['date']})", flush=True)
    if load_manifest(h) and not force:
        print(f"[pooled] CACHE HIT — manifest exists at {manifest_path(h)}; "
              f"0 re-encoding. Pass --force to rebuild.", flush=True)
        return h

    if ingest_surface != "active":
        os.environ["BRAIN_SURFACE_VARIANT"] = "v5_agentic"
    os.makedirs(corpus_dir(h), exist_ok=True)
    prompts_dir = os.path.join(corpus_dir(h), "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    os.environ["BRAIN_PROMPT_CAPTURE_DIR"] = prompts_dir

    pooled_path = corpus_item_dir(h, "pooled")
    brain = create_fresh_eval_brain(path=pooled_path, wipe=True)
    if s1e != "active":
        _apply_s1e_override(brain, s1e)
    if ingest_surface != "active":
        _apply_surface_override(brain, ingest_surface)

    t_run0 = time.time()
    totals = {"turns": 0, "user_turns": 0, "s1e_runs": 0, "s2_runs": 0,
              "s1r_ms_total": 0, "s1e_ms_total": 0, "s2_ms_total": 0,
              "s2_deltas": []}
    carry = 0
    for i, e in enumerate(plan):
        sid = e["sid"]
        print(f"\n[pooled] session {i+1}/{len(plan)} {sid} "
              f"({e['date']}, {len(e['session'])} turns)", flush=True)

        stats = replay_item(
            brain, sid, [e["session"]], haystack_dates=[e["date"]],
            log_prefix=f"[pooled {i+1}/{len(plan)}]",
            s2_every_n=s2_every_n, final_flush=False, s2_carry=carry)
        carry = stats.get("encodings_since_s2", 0)
        for k in ("turns", "user_turns", "s1e_runs", "s2_runs"):
            totals[k] += stats.get(k) or 0
        totals["s2_deltas"].extend(stats.get("s2_deltas", []))

        # Per-session checkpoint: a RED-class error means the rest of the
        # spend builds a contaminated corpus — abort now, not at V0.
        errs = _read_build_errors(brain)
        if errs["red_count"]:
            print(f"\n[pooled] ✗ ABORT after session {i+1}/{len(plan)}: "
                  f"{errs['red_count']} RED-class error(s) in debug_log — "
                  f"corpus would be contaminated. Samples:", flush=True)
            for s in errs["samples"]:
                if not s["benign"]:
                    print(f"    {s['source']}: {s['error']}", flush=True)
            try:
                brain.close()
            except Exception:
                pass
            sys.exit(2)

    print(f"\n[pooled] all sessions ingested — finalize (S2 flush + backfill)",
          flush=True)
    finalize_item(brain, totals, carry, log_prefix="[pooled]")

    # ── V0 audit (§20.18): each check printed; the block is the gate ──
    node_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes").fetchone()[0]
    build_errors = _read_build_errors(brain)
    dates_sorted = all(plan[i]["sort_key"] <= plan[i + 1]["sort_key"]
                       for i in range(len(plan) - 1))
    audit = {
        "sessions": len(plan),
        "user_turns": n_user_turns,
        "user_turns_replayed": totals["user_turns"],
        "dates_monotonic": dates_sorted,
        "build_errors": build_errors,
        "node_count": node_count,
        "span": [plan[0]["date"], plan[-1]["date"]],
    }
    v0_green = (dates_sorted and build_errors["red_count"] == 0
                and totals["user_turns"] == n_user_turns)
    print(f"[pooled] V0 audit: sessions={audit['sessions']} "
          f"user_turns={audit['user_turns_replayed']}/{audit['user_turns']} "
          f"dates_monotonic={dates_sorted} "
          f"errors={build_errors['count']} "
          f"(red={build_errors['red_count']} "
          f"benign={build_errors['benign_count']}) "
          f"nodes={node_count} → {'GREEN' if v0_green else 'RED'}", flush=True)
    audit["green"] = v0_green

    # Per-item answerability, re-scanned on the POOLED brain (§20.18 V0:
    # answerability WILL differ from per-item builds — report, don't assume)
    items_manifest = []
    for item in picked:
        scan = _scan_brain_for_gold(brain, _gold_str(item))
        items_manifest.append({
            "qid": item["question_id"],
            "axis": _item_axis(item),
            "question": item["question"],
            "gold": _gold_str(item),
            "question_date": item.get("question_date"),
            "turns": sum(len(s) for s in item.get("haystack_sessions", [])),
            "answerable": bool(scan.get("found")),
            "gold_scan": {
                "found": scan.get("found"),
                "terms_used": scan.get("terms_used"),
                "phrase_used": scan.get("phrase_used"),
                "matches": scan.get("matches", []),
            },
            "brain_dir": pooled_path,
        })

    try:
        brain.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    try:
        brain.close()
    except Exception:
        pass

    ac = sum(1 for it in items_manifest if it["answerable"])
    manifest = {
        "corpus_hash": h, "label": label, "created_at_epoch": time.time(),
        "prompts_dir": prompts_dir,
        "config": config, "items": items_manifest,
        "answerable_count": ac,
        "unanswerable_count": len(items_manifest) - ac,
        "s2_totals": summarize_s2_deltas(totals["s2_deltas"]),
        "build_errors_total": build_errors["count"],
        "build_ms": int((time.time() - t_run0) * 1000),
        "pooled_audit": audit,
        "pooled_totals": {k: totals[k] for k in
                          ("turns", "user_turns", "s1e_runs", "s2_runs")},
    }
    path = save_manifest(h, manifest)
    print(f"\n[pooled] done in {manifest['build_ms']/1000:.1f}s — "
          f"ANSWERABLE {ac}/{len(items_manifest)} on the pooled brain",
          flush=True)
    print(f"[pooled] manifest → {path}", flush=True)
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
    p.add_argument("--lived", action="store_true",
                   help="build the LIVED arm (BRAIN_S1E_LIVED_SEQUENCE on: XML lived-sequence "
                        "input, widened catalog, 2 scouts, inline notes). Joins the content "
                        "address — a lived corpus never collides with a control corpus.")
    p.add_argument("--interaction-override", dest="interaction_override", default=None,
                   help="Comma-separated name=version pairs, fetched from the live daemon's "
                        "registered (incl. DORMANT) versions and activated in each eval brain. "
                        "e.g. 's1e=24,s1_scout_facts=7'. Part of the corpus hash.")
    p.add_argument("--pooled", action="store_true",
                   help="§20.18 pooled build: interleave the picked items' haystack "
                        "sessions by date into ONE brain (per-conversation session ids, "
                        "one final S2 flush, V0 audit in the manifest). Incompatible "
                        "with --lived/--interaction-override for now — the pooled arm "
                        "is the moment-stack validation substrate, not an encoder A/B.")
    args = p.parse_args()

    overrides = {}
    if args.interaction_override:
        for pair in args.interaction_override.split(","):
            if "=" in pair:
                n, v = pair.split("=", 1)
                overrides[n.strip()] = int(v.strip())

    if args.pooled:
        if args.lived or overrides:
            p.error("--pooled does not compose with --lived/--interaction-override")
        build_pooled_corpus(args.oracle, args.qids, args.s1e, args.ingest_surface,
                            args.s2_every_n, args.label, force=args.force,
                            items_per_axis=args.items, seed=args.seed)
        return

    build_corpus(args.items, args.seed, args.oracle, args.s1e, args.ingest_surface,
                 args.s2_every_n, args.label, qids=args.qids, force=args.force,
                 interaction_overrides=overrides or None, lived=args.lived)


if __name__ == "__main__":
    main()
