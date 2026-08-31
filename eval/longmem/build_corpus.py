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
from eval.longmem.fresh_brain import create_fresh_eval_brain, enable_round_capture
from eval.longmem.classifier import _scan_brain_for_gold
from eval.longmem.corpus import (
    corpus_config_hash, corpus_dir, corpus_item_dir, manifest_path,
    load_manifest, save_manifest, source_token, interaction_token,
    summarize_s2_deltas, merge_s2_totals, ingest_session_id,
    require_variant_pins, address_variants,
)
from tests.interaction_override import override_interaction


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


# Benign family = error classes that are ROUTINE in the production brain's
# own debug_log (30-day audit, 2026-07-18): designed loud-skips and
# degradations the live system experiences weekly. The eval should
# experience what production experiences (probe-input fidelity); aborting
# on them enumerates dice rolls at $3 a throw. Value = per-build cap
# (None = uncapped); beyond the cap the class counts RED — one flake is
# production-faithful, a cluster is systemic. Everything NOT listed
# (recall_laf fallback, brain_batch_*, S2 unit exceptions: zero production
# occurrences in 30d) stays RED — unknown classes fail loud, per
# Loud-by-Default.
_BENIGN_CAPS = {
    "haiku_id_outside_candidates": None,   # resolved= only, see below
    "surface_inject_overflow": None,       # deterministic byte-cap truncation
    "surface_malformed_tool_arg": None,    # designed drop-and-continue
    "revise_immutable": None,              # write-boundary rejection working
    "keepalive_tick": None,                # infra noise, not build data
    "warmup_anthropic": None,              # infra noise, not build data
    # Caps below are calibrated for up-to-600-turn pooled builds (observed
    # base rates: spiral ~1/170 turns, connect_to ~1/100 turns, aspect
    # filters front-loaded while a fresh taxonomy fills). They distinguish
    # dice from systemic — a systemic failure produces dozens, not these.
    "surface_haiku_unparseable": 4,        # max_tokens spiral (the trailing-
                                           # comma subclass is now parsed
                                           # tolerantly and shouldn't appear)
    "connect_to_*": 15,                    # the write boundary's designed
                                           # rejection FAMILY (_unresolved,
                                           # _self, _invalid, _failed — see
                                           # brain_remember._resolve_connect_to
                                           # _entry: contained, never raises,
                                           # per-entry independent); prefix-
                                           # matched so new members don't
                                           # cost another aborted build
    "aspect_integration": 12,              # designed loud-FILTERS only (see
                                           # _BENIGN_MESSAGES — the same source
                                           # also logs REAL classify/IO
                                           # failures, which stay RED); fresh
                                           # brains mint new strings, so rate
                                           # is structurally above production's
    "bg_writer_worker_stalled": 2,         # watchdog observation, self-recovers
    "s1_scout_facts_api_error": 3,         # connection blip → scout no-op
    "s1_scout_quote_api_error": 3,
    "s1_scout_temporal_api_error": 3,
}

# Sources whose benign membership is NARROWER than the source: only these
# message prefixes are designed degradations; anything else logged under the
# same source (real classify failures, IO errors) stays RED.
_BENIGN_MESSAGES = {
    "aspect_integration": ("aspect/category mismatch",
                           "noise + semantic aspect"),
}


# Constructed contamination ALARMS: logged with never-raised exceptions
# (so the frame rule below would call them benign) but they signal exactly
# what the gate exists to catch. Always RED.
_RED_OVERRIDE_PREFIXES = ("brain_batch", "daemon_crash")

# Global cap for the constructed-exception class (designed loud-logs not
# individually listed above). One-off LLM/output flakes are dice; a flood
# of them is a systemic problem even when each is individually contained.
_DESIGNED_LOUDLOG_CAP = 25


def _benign_cap_key(source: str) -> str:
    """Family classes share one cap under their prefix key; constructed
    loud-logs without an explicit entry share the global designed cap."""
    if source.startswith("connect_to_"):
        return "connect_to_*"
    if source in _BENIGN_CAPS or source in _BENIGN_MESSAGES \
            or source == "haiku_id_outside_candidates":
        return source
    return "designed_loudlog"


def _is_benign_build_error(source: str, context: str, error: str = "",
                           traceback: str = "") -> bool:
    """Membership check only — the per-build cap is applied by the caller
    (_read_build_errors), which counts occurrences across the build.

    Classification is by MECHANISM, not enumeration (the 2026-07-18 audit
    of all ~150 _log_error sources): designed loud-logs pass a CONSTRUCTED
    exception that was never raised — its traceback has no stack frames —
    while real failures pass a caught exception with 'File \"...\"' frames.
    Caught-exception classes stay RED unless explicitly listed (infra
    blips); constructed classes are benign under _DESIGNED_LOUDLOG_CAP
    unless they're contamination alarms (_RED_OVERRIDE_PREFIXES)."""
    if any(source.startswith(p) for p in _RED_OVERRIDE_PREFIXES):
        return False
    if source == "haiku_id_outside_candidates":
        return "resolved=" in (context or "")
    if source in _BENIGN_MESSAGES:
        return (error or "").startswith(_BENIGN_MESSAGES[source])
    if _benign_cap_key(source) in _BENIGN_CAPS:
        return True
    return 'File "' not in (traceback or "")


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
    benign_seen = {}
    for source, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}
        context = (meta.get("context") or "")[:120]
        benign = _is_benign_build_error(source, context, meta.get("error") or "",
                                        meta.get("traceback") or "")
        cap_key = _benign_cap_key(source)
        cap = (_DESIGNED_LOUDLOG_CAP if cap_key == "designed_loudlog"
               else _BENIGN_CAPS.get(cap_key))
        if benign and cap is not None:
            benign_seen[cap_key] = benign_seen.get(cap_key, 0) + 1
            if benign_seen[cap_key] > cap:
                benign = False    # over the cap → systemic, not dice
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
    table via TCP; the eval brain is untouched until override_interaction.
    """
    from servers.daemon_client import send_command
    r = send_command('get_interaction', {'name': name, 'version': int(version)})
    tmpl = (r.get('result') or {}).get('template') if isinstance(r, dict) else None
    if not tmpl:
        raise RuntimeError("could not fetch %s v%s from daemon: %s" % (name, version, r))
    return tmpl


def _seed_pack_token() -> dict:
    """The seed pack joins the content address: it is the encoder's only
    early catalog, so it shapes every encoded graph. Fingerprint the RESOLVED
    module state — after any --seed-pack override — so a corpus built under
    one pack never cache-hits a corpus built under another (the same stale-
    default masquerade _k_fingerprints closes for prompts; brain id:5f935ada
    for the seed flavor). One-time cache invalidation for pre-existing
    corpora: intended — they all predate the Nursery pack anyway."""
    import hashlib
    import servers.seed_pack as sp
    blob = json.dumps({"nodes": sp.SEED_NODES, "edges": sp.SEED_EDGES,
                       "generation": sp.SEED_PACK_GENERATION},
                      sort_keys=True, ensure_ascii=False)
    return {"generation": sp.SEED_PACK_GENERATION,
            "digest": hashlib.sha1(blob.encode()).hexdigest()[:12]}


def _apply_seed_pack_override(path: str) -> None:
    """Swap servers.seed_pack's pack DATA (nodes/edges/generation) for another
    pack file's, before any eval Brain is created — the loader code stays
    current, only the data changes. Pack files are data-only modules
    (SEED_NODES/SEED_EDGES at module level, no imports required).

    A pack without its own SEED_PACK_GENERATION (pre-Nursery) gets a distinct
    eval marker: the frozen brains then carry a foreign generation, so the
    sweep-time open's seeding pass reads them as not-born-from-the-current-pack
    and leaves them untouched — otherwise the current pack's gap-fill would
    inject its nodes into the frozen graph at open (id:5f935ada)."""
    import hashlib
    import servers.seed_pack as sp
    src = Path(path).read_text()
    ns: dict = {"__name__": "seed_pack_override"}
    exec(compile(src, path, "exec"), ns)
    sp.SEED_NODES = ns["SEED_NODES"]
    sp.SEED_EDGES = ns["SEED_EDGES"]
    gen = ns.get("SEED_PACK_GENERATION") or (
        "eval_ext_" + hashlib.sha1(src.encode()).hexdigest()[:8])
    sp.SEED_PACK_GENERATION = gen
    print(f"[corpus] seed-pack override: {path} → {len(sp.SEED_NODES)} nodes, "
          f"{len(sp.SEED_EDGES)} edges, generation={gen}", flush=True)


def _k_fingerprints(override_templates: dict = None) -> dict:
    """Resolved-K fingerprints for the content address, per encoding name.

    "active" alone stopped being a complete address when "no override" came to
    mean "code default": the default changes by merge, and a cached corpus
    built under an older default would cache-HIT and silently masquerade as
    what production runs now (brain id:f36def04). Fingerprint the EFFECTIVE
    template+config — override template (if any) over the code-default config,
    which is exactly what the fresh eval brain resolves. Doubles as the
    manifest's build-time record of the K each arm ran (the runbook's stamp
    check resolves at READ time and cannot see a stale build). One-time cache
    invalidation for every pre-existing corpus: intended — all predate the
    970cdfc replay fix anyway.
    """
    from servers.interaction_defaults import (INTERACTION_DEFAULTS,
                                              interaction_fingerprint)
    ovr = override_templates or {}
    fps = {}
    for name in ("s1e", "surface"):
        d_tpl, d_cfg = INTERACTION_DEFAULTS[name]
        fps[name] = interaction_fingerprint(name, ovr.get(name, d_tpl), d_cfg)
    return fps


def _resolve_build_pins(ingest_surface: str) -> dict:
    """Effective variant pair for this build, env applied BEFORE the hash.

    A surface-override build runs the agentic loop, so the env pin is applied
    here — the one place — and the returned pair is what the address and the
    run both see. Guard + addressing live in corpus.py (the addressing owner).
    """
    pins = require_variant_pins()
    if ingest_surface != "active":
        os.environ["BRAIN_SURFACE_VARIANT"] = "v5_agentic"
        pins["surface_variant"] = "v5_agentic"
    return pins


def build_corpus(items_per_axis: int, seed: int, oracle: str,
                 s1e: str, ingest_surface: str, s2_every_n: int,
                 label: str, qids: str = None, force: bool = False,
                 interaction_overrides: dict = None, lived: bool = True,
                 seed_pack: str = None) -> str:
    _load_env()

    # Seed-pack override must land before any eval Brain is created — the
    # loader reads the module globals at Brain init.
    if seed_pack:
        _apply_seed_pack_override(seed_pack)

    # The lived arm (BRAIN_S1E_LIVED_SEQUENCE) changes the ENCODED GRAPH — the
    # XML lived-sequence input, widened catalog, 2-scout muster, inline scout
    # notes all shape what S1E writes — so it must be pinned here, not inherited
    # from the shell: set it explicitly per the arg, and clear any leaked env
    # on the control arm so a stray export can't silently flip a build's arm.
    # Default is LIVED — production's arm since v29 activation (brain-env.sh
    # exports BRAIN_S1E_LIVED_SEQUENCE=1, 2026-07-03). A control-arm default
    # made every post-activation default build run the retired legacy branch:
    # no `## Arc` harvest → empty session_context_{sid} digests (the v-next.5
    # gate corpora fcc338/a3be7a shipped arc-blind that way, 2026-08-24).
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

    # Fetch DORMANT override templates from the live daemon once (reused per
    # item) — BEFORE the content address, because the address is over the
    # template CONTENT, not the version number that names it.
    override_templates = {}
    if interaction_overrides:
        for name, version in sorted(interaction_overrides.items()):
            override_templates[name] = _fetch_interaction_template(name, version)
            print(f"[corpus] override: {name} → v{version} ({len(override_templates[name])} chars)",
                  flush=True)

    # Content address: everything that determines the encoded graph.
    variant_pins = _resolve_build_pins(ingest_surface)
    config = {
        "s1e": source_token(s1e),
        "ingest_surface": source_token(ingest_surface),
        "s2_every_n": s2_every_n,
        "oracle": os.path.basename(oracle),
        "qids": qid_list,
    }
    address_variants(config, variant_pins)
    # Interaction overrides (e.g. DORMANT s1e v24 + s1_scout_facts v7) change the
    # encoded graph, so they're part of the content address — a v22 corpus and a
    # v24+v7 corpus get distinct hashes. Addressed on the template CONTENT: a
    # version int is install-local and, once "version absent" means "code
    # default", not a complete address — two corpora built against different
    # code-default generations would hash identically and load_manifest would
    # hand back the wrong arm's corpus.
    if interaction_overrides:
        config["interaction_overrides"] = {
            name: interaction_token(version, override_templates[name])
            for name, version in sorted(interaction_overrides.items())}
    # Lived arm joins the content address ONLY when on (key absent on control →
    # every pre-existing control corpus keeps its hash; no cache invalidation).
    # Without this, a lived and a control build with the same versions would
    # collide on one hash and the cache would hand back the wrong arm's corpus.
    if lived:
        config["s1e_lived"] = True
    config["k_fingerprints"] = _k_fingerprints(override_templates)
    config["seed_pack"] = _seed_pack_token()
    h = corpus_config_hash(config)
    ov_str = (" / overrides=%s" % config["interaction_overrides"]) if interaction_overrides else ""
    print(f"[corpus] config hash = {h}  ({config['s1e']} / surface={config['ingest_surface']} "
          f"/ s2_every_n={s2_every_n} / {len(qid_list)} items{ov_str})", flush=True)

    if load_manifest(h) and not force:
        print(f"[corpus] CACHE HIT — manifest exists at {manifest_path(h)}; "
              f"0 re-encoding. Pass --force to rebuild.", flush=True)
        return h

    os.makedirs(corpus_dir(h), exist_ok=True)

    # Full-prompt capture (Tom, 2026-07-02): every S1E encode in a corpus build
    # records its literal per-round payload — the REAL prompt fed to S1Scribe,
    # not a rebuild — via the payload recorder (enable_round_capture flips the
    # round_payload gate per item brain; never debug mode). Payloads ship
    # FROZEN inside each item's brain dir at {item_dir}/payloads/, so any
    # regression in an A/B is inspectable at the exact prompt that produced it.
    print("[corpus] full-prompt capture → per-item {item_dir}/payloads/",
          flush=True)

    items_manifest = []
    t_run0 = time.time()

    def _save_manifest_now():
        """Rebuild + persist the manifest from items-so-far. Called after every
        item so a crash during a long unattended build preserves completed items
        (their frozen brains already exist on disk; this keeps the index in sync)."""
        ac = sum(1 for it in items_manifest if it.get("answerable"))
        m = {
            "corpus_hash": h, "label": label, "created_at_epoch": time.time(),
            # Build-time record of the effective variants (in the address only
            # when non-baseline — this key is the always-present stamp).
            "variant_pins": variant_pins,
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
        enable_round_capture(brain)   # payloads → {path}/payloads/, frozen with the item

        if s1e != "active":
            _apply_s1e_override(brain, s1e)
        if ingest_surface != "active":
            _apply_surface_override(brain, ingest_surface)
        for ov_name, ov_template in override_templates.items():
            override_interaction(brain, ov_name, template=ov_template)

        t0 = time.time()
        stats = replay_item(
            brain, ingest_session_id(qid), item["haystack_sessions"],
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
            # Captured round payloads, frozen with the item's brain (replaces
            # the corpus-level prompts_dir of the retired capture-dir era).
            "payloads_dir": os.path.join(path, "payloads"),
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
    # Pooled is the moment-stack VALIDATION substrate — it always builds the
    # production arm. Pin lived explicitly (eval shells don't source
    # brain-env.sh, so inheriting the shell silently built control-arm pooled
    # corpora); there is no control-arm pooled build — pooled is not an
    # encoder A/B, so no `lived` arg mirrors build_corpus's.
    os.environ["BRAIN_S1E_LIVED_SEQUENCE"] = "1"
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
    variant_pins = _resolve_build_pins(ingest_surface)
    config = {
        "pooled": True,
        # Harness generation joins the content address: a graph-changing
        # harness fix (v2: per-turn embed drains — decode-alive ingest,
        # trace-embedding substrate; v3: collision-free session ids) must
        # never collide with a corpus cached under the older behavior.
        "harness": 3,
        # Pooled always builds lived (pinned above). Addressed unconditionally:
        # pre-pin pooled corpora inherited the shell (control arm in practice)
        # and keep their hashes — a fresh lived build never cache-hits one.
        "s1e_lived": True,
        "s1e": source_token(s1e),
        "ingest_surface": source_token(ingest_surface),
        "s2_every_n": s2_every_n,
        "oracle": os.path.basename(oracle),
        "qids": qid_list,
    }
    address_variants(config, variant_pins)
    config["k_fingerprints"] = _k_fingerprints()
    config["seed_pack"] = _seed_pack_token()
    h = corpus_config_hash(config)
    print(f"[pooled] config hash = {h}  ({len(qid_list)} items / "
          f"{len(plan)} sessions / {n_user_turns} user turns / "
          f"span {plan[0]['date']} → {plan[-1]['date']})", flush=True)
    if load_manifest(h) and not force:
        print(f"[pooled] CACHE HIT — manifest exists at {manifest_path(h)}; "
              f"0 re-encoding. Pass --force to rebuild.", flush=True)
        return h

    os.makedirs(corpus_dir(h), exist_ok=True)

    pooled_path = corpus_item_dir(h, "pooled")
    brain = create_fresh_eval_brain(path=pooled_path, wipe=True)
    enable_round_capture(brain)   # payloads → {pooled_path}/payloads/
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
        # Round payloads live inside the pooled brain dir (recorder layout).
        "prompts_dir": os.path.join(pooled_path, "payloads"),
        # Build-time record of the effective variants (in the address only
        # when non-baseline — this key is the always-present stamp).
        "variant_pins": variant_pins,
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
                   help="'active' (the resolved default) or a path to an s1e prompt file to encode with")
    p.add_argument("--ingest-surface", dest="ingest_surface", default="active",
                   help="'active' or a path to a surface prompt file used during ingest recall")
    p.add_argument("--s2-every-n", dest="s2_every_n", type=int, default=2,
                   help="S2 fires every N encodings during ingest (default 2)")
    p.add_argument("--label", default="corpus", help="human label stored in the manifest")
    p.add_argument("--qids", default=None, help="comma-separated qids (overrides stratified sampling)")
    p.add_argument("--force", action="store_true", help="rebuild even if the corpus already exists")
    p.add_argument("--lived", action=argparse.BooleanOptionalAction, default=None,
                   help="arm pin: --lived / --no-lived (BRAIN_S1E_LIVED_SEQUENCE: XML "
                        "lived-sequence input, widened catalog, 2 scouts, inline notes, "
                        "`## Arc` residue). Default LIVED — production's arm since v29 "
                        "activation. Joins the content address — a lived corpus never "
                        "collides with a control corpus.")
    p.add_argument("--seed-pack", dest="seed_pack", default=None,
                   help="Path to an alternate seed-pack module (SEED_NODES/SEED_EDGES "
                        "at module level) to seed each eval brain with, e.g. a "
                        "git-show of a prior generation. Part of the corpus hash; "
                        "frozen brains carry the pack's generation marker so sweeps "
                        "never gap-fill them with the current pack.")
    p.add_argument("--interaction-override", dest="interaction_override", default=None,
                   help="Comma-separated name=version pairs, fetched from the live daemon's "
                        "registered (incl. DORMANT) versions and activated in each eval brain. "
                        "e.g. 's1e=24,s1_scout_facts=7'. Part of the corpus hash.")
    p.add_argument("--pooled", action="store_true",
                   help="§20.18 pooled build: interleave the picked items' haystack "
                        "sessions by date into ONE brain (per-conversation session ids, "
                        "one final S2 flush, V0 audit in the manifest). Always builds "
                        "the production (lived) arm — pooled is the moment-stack "
                        "validation substrate, not an encoder A/B, so it takes no "
                        "--lived/--no-lived/--interaction-override.")
    args = p.parse_args()

    overrides = {}
    if args.interaction_override:
        for pair in args.interaction_override.split(","):
            if "=" in pair:
                n, v = pair.split("=", 1)
                overrides[n.strip()] = int(v.strip())

    if args.pooled:
        # args.lived is None unless the user pinned an arm explicitly — pooled
        # takes no arm pin (it always builds lived; there is no control pooled).
        if args.lived is not None or overrides or args.seed_pack:
            p.error("--pooled does not compose with --lived/--no-lived/"
                    "--interaction-override/--seed-pack")
        build_pooled_corpus(args.oracle, args.qids, args.s1e, args.ingest_surface,
                            args.s2_every_n, args.label, force=args.force,
                            items_per_axis=args.items, seed=args.seed)
        return

    build_corpus(args.items, args.seed, args.oracle, args.s1e, args.ingest_surface,
                 args.s2_every_n, args.label, qids=args.qids, force=args.force,
                 interaction_overrides=overrides or None,
                 lived=(True if args.lived is None else args.lived),
                 seed_pack=args.seed_pack)


if __name__ == "__main__":
    main()
