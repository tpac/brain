"""Coupled-arm S1E encoding A/B viewer — the "see it before we do a lot" tool.

Runs the SAME conversation through two coupled arms and dumps what each
encoded, side by side, with the ACTUAL full prompts captured (not a rebuild):

  control = BRAIN_S1E_LIVED_SEQUENCE off  + s1e <control-version>  (v25)
  new     = BRAIN_S1E_LIVED_SEQUENCE on   + s1e <new-version>      (v26)

The arms are COUPLED by design (the v-next prompt describes the lived input
structures P1–P4 produce), so we flip the flag AND the prompt together — never
one without the other. Prompt injection lands in the EVAL brain only; the live
daemon's s1e is never mutated (unlike the legacy diff_encoding.register_prompt).

Usage:
  ./dev python3 eval/longmem/ab_encode.py --qids Q1 Q2 \\
      --control-version 25 --new-version 26

  # plumbing dry-run before v26 exists (both arms same prompt; flag still flips):
  ./dev python3 eval/longmem/ab_encode.py --qids Q1 --control-version 25 --new-version 25

Encode-side only — no query/answer phase (that's the full LongMemEval gate).
This viewer answers "what does each arm encode, and does the new arm mint fewer
twins?" plus proves the pipeline works internally (embeddings, recall, capture).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fetch_prompt(version: int) -> tuple:
    """Read s1e@version from the live daemon (READ-ONLY). Returns (template, parameters)."""
    from servers.daemon_client import send_command
    r = send_command('get_interaction', {'name': 's1e', 'version': version})
    if not r.get('ok'):
        raise RuntimeError("failed to fetch s1e v%d from daemon: %s" % (version, r))
    res = r['result']
    return res['template'], res.get('parameters', '') or ''


def inject_prompt(brain, template: str, parameters: str) -> int:
    """Register `template` into the EVAL brain and make it active. Returns the
    eval-brain version. Never touches the daemon."""
    reg = brain._interaction_dal.register(
        's1e', template=template, parameters=parameters or '',
        created_by='ab_encode:injected')
    ver = reg['version'] if isinstance(reg, dict) else reg
    brain.set_interaction_active('s1e', ver, set_by='ab_encode')
    # Confirm the runtime read path returns what we injected.
    active = brain.get_interaction('s1e')
    assert active and active.get('template') == template, \
        "injection failed: active s1e template != injected"
    return ver


def dump_nodes(brain) -> list:
    """Non-seed nodes with key fields (from diff_encoding, kept local)."""
    rows = brain.conn.execute("""
        SELECT n.id, n.type, n.title, n.content, n.encoding_source
        FROM nodes n
        WHERE n.encoding_source != 'anchor:seed' OR n.encoding_source IS NULL
        ORDER BY n.created_at
    """).fetchall()
    out = []
    for nid, ntype, title, content, src in rows:
        kv = dict(brain.conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id=? "
            "AND key IN ('situation','event_time')", (nid,)).fetchall())
        out.append({"id": nid[:8], "type": ntype, "title": title or "",
                    "content": (content or "")[:220],
                    "situation": (kv.get("situation") or "")[:120],
                    "event_time": kv.get("event_time", ""), "src": src})
    return out


def dump_edges(brain) -> list:
    rows = brain.conn.execute("""
        SELECT e.source_id, e.target_id, er.relation, er.description
        FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE er.archived = 0
          AND (e.source_id IN (SELECT id FROM nodes WHERE encoding_source != 'anchor:seed')
            OR e.target_id IN (SELECT id FROM nodes WHERE encoding_source != 'anchor:seed'))
    """).fetchall()
    return [{"source": s[:8], "target": t[:8], "relation": r, "why": (d or "")[:70]}
            for s, t, r, d in rows]


def _norm_title(t: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', (t or '').lower()).strip()


def dedup_signal(nodes: list) -> dict:
    """Crude near-duplicate signal (a SIGNAL for eyeballing, not a gate): pairs of
    created nodes whose normalized titles share >=60% of their word set. The new
    arm minting fewer of these — because it revised instead — is the headline value."""
    titles = [(_norm_title(n["title"]), n["title"]) for n in nodes]
    pairs = []
    for i in range(len(titles)):
        wi = set(titles[i][0].split())
        if not wi:
            continue
        for j in range(i + 1, len(titles)):
            wj = set(titles[j][0].split())
            if not wj:
                continue
            overlap = len(wi & wj) / max(1, min(len(wi), len(wj)))
            if overlap >= 0.6:
                pairs.append((titles[i][1], titles[j][1], round(overlap, 2)))
    return {"count": len(pairs), "pairs": pairs[:8]}


def _cap_sort_key(path: str) -> tuple:
    """Order capture files by (stop, round, seq) NUMERICALLY. A lexical sort puts
    'stop5' after 'stop10'/'stop12' — so cap_files[-1] would read the FIRST run's
    thin prompt, not the last. This bug false-failed the lived-structure checks in
    the first dry-run; the numeric key is the fix."""
    m = re.search(r'stop(\d+)-r(\d+)-\d+-(\d+)', os.path.basename(path))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (-1, -1, -1)


def capture_files_for(capture_dir: str, arm: str, session: str) -> list:
    return sorted(glob.glob(os.path.join(capture_dir, "%s__%s__*.json" % (arm, session))),
                  key=_cap_sort_key)


def run_assertions(brain, arm: str, question: str, cap_files: list,
                   session: str = '') -> dict:
    """Verify the pipeline worked INTERNALLY. Hard checks gate validity; soft
    checks (lived structures) warn — they depend on haystack length + the
    deferred-embedding reality of eval ingest."""
    a = {}

    # HARD: the session arc populated (Guardrail #1 of the eval design — the
    # v26 A/B regression this tool missed: the lived arm wrote zero
    # session_context_* rows and only the corpus-stage analysis caught it).
    # Both arms must produce it: control via the SESSION_CONTEXT: line, new
    # via the `## Arc` fence → write_session_arc.
    if session:
        try:
            ctx = (brain.session_context_for(session) or '').strip()
            a["arc_produced"] = (bool(ctx), "%d chars of session arc" % len(ctx))
        except Exception as e:
            a["arc_produced"] = (False, "arc probe failed: %s" % e)

    # HARD: embeddings computed (the classic '0% baseline from NULL vectors' trap).
    try:
        # v23+ schema: vectors live in node_enrichments keyed by vector_type;
        # the primary lane is '_primary', stored in the `embedding` column.
        null_vec = brain.conn.execute(
            "SELECT COUNT(*) FROM nodes n WHERE (n.encoding_source != 'anchor:seed' "
            "OR n.encoding_source IS NULL) AND NOT EXISTS "
            "(SELECT 1 FROM node_enrichments e WHERE e.node_id = n.id "
            " AND e.vector_type = '_primary' AND e.embedding IS NOT NULL)").fetchone()[0]
        a["embeddings_complete"] = (null_vec == 0, "%d non-seed nodes missing _primary vector" % null_vec)
    except Exception as e:
        a["embeddings_complete"] = (False, "probe failed: %s" % e)

    # HARD: recall works end-to-end post-backfill (proves embeddings + pipeline).
    try:
        res = brain.recall(query=question, limit=5)
        hits = res.get("results", res) if isinstance(res, dict) else res
        n = len(hits) if hits else 0
        a["recall_live"] = (n > 0, "%d candidates for the question" % n)
    except Exception as e:
        a["recall_live"] = (False, "recall raised: %s" % e)

    # HARD: the full prompt was captured (this arm produced ≥1 dump).
    a["prompt_captured"] = (len(cap_files) > 0, "%d captured prompt files" % len(cap_files))

    # SOFT (new arm only): the lived input structures actually populated. Read the
    # LAST captured prompt (highest stop → most prior context) and look for a
    # populated <node_catalog> and a <provenance> with a trace-based encoded ref.
    if arm == "new" and cap_files:
        try:
            last = json.load(open(cap_files[-1]))
            body = "\n".join(b.get("text", "") for m in last["messages"]
                             for b in m.get("content", []) if isinstance(b, dict))
            cat_ok = "<node_catalog>" in body and "Node Catalog" in body
            prov_ok = "encoded(S1S)" in body or "surfaced:" in body
            tl_ok = "<timeline>" in body and "<turn" in body
            a["lived_catalog_populated"] = (cat_ok, "widened <node_catalog> present in last prompt")
            a["lived_timeline_xml"] = (tl_ok, "<timeline>/<turn> XML present")
            a["lived_provenance_present"] = (prov_ok, "<provenance> carried a real ref (soft — needs 2+ encodes)")
        except Exception as e:
            a["lived_catalog_populated"] = (False, "capture read failed: %s" % e)
    return a


HARD_CHECKS = {"embeddings_complete", "recall_live", "prompt_captured",
               "lived_timeline_xml", "arc_produced"}


def run_arm(arm, version, qid, item, capture_dir, control_lived=False):
    from eval.longmem.fresh_brain import create_fresh_eval_brain
    from eval.longmem.replay import replay_item

    # Arm env — set BEFORE the brain/replay so the in-process encoder reads it.
    # control_lived=True runs BOTH arms lived — for same-template A/Bs where
    # the axis is the interaction's parameters (e.g. effort high vs medium),
    # not the input structure. The coupled flag-off control only makes sense
    # when comparing prompt GENERATIONS (v25-style vs lived-style).
    if arm == "new" or control_lived:
        os.environ["BRAIN_S1E_LIVED_SEQUENCE"] = "1"
    else:
        os.environ.pop("BRAIN_S1E_LIVED_SEQUENCE", None)
    os.environ["BRAIN_PROMPT_CAPTURE_DIR"] = capture_dir
    # Explicit arm name for capture labels — with --control-lived both arms
    # run lived, so encode.py's flag-derived arm would label everything
    # 'new__' and the control arm's prompt_captured assertion goes falsely
    # INVALID (seen 2026-07-03, effort A/B run 1).
    os.environ["BRAIN_PROMPT_CAPTURE_ARM"] = arm

    tmpl, params = FETCHED[version]
    path = os.path.expanduser("~/AgentsContext/brain-ab-%s-%s" % (arm, qid))
    brain = create_fresh_eval_brain(path=path, wipe=True)   # resets BRAIN_DB_DIR/TMP_DIR
    os.environ["BRAIN_PROMPT_CAPTURE_DIR"] = capture_dir     # re-assert (create_* set TMP_DIR)
    inject_prompt(brain, tmpl, params)

    session = "ab-%s" % qid
    print("\n[ab] === arm=%s v%d qid=%s ===" % (arm, version, qid), flush=True)
    replay_item(brain, session, item["haystack_sessions"],
                haystack_dates=item.get("haystack_dates"),
                log_prefix="[%s/%s]" % (arm, qid))

    nodes, edges = dump_nodes(brain), dump_edges(brain)
    cap_files = capture_files_for(capture_dir, arm, session)
    asserts = run_assertions(brain, arm, item["question"], cap_files,
                             session=session)
    try:
        brain.close()
    except Exception:
        pass
    return {"nodes": nodes, "edges": edges, "dedup": dedup_signal(nodes),
            "asserts": asserts, "cap_files": cap_files, "path": path}


FETCHED = {}  # {version: (template, parameters)} — fetched once up front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", nargs="+", required=True)
    ap.add_argument("--control-version", type=int, default=25)
    ap.add_argument("--new-version", type=int, default=26)
    ap.add_argument("--oracle", default="eval/longmem/data/longmemeval_oracle.json")
    ap.add_argument("--capture-dir",
                    default=os.path.expanduser("~/AgentsContext/ab-prompts"))
    ap.add_argument("--control-lived", action="store_true",
                    help="run the control arm lived too (same-template A/B — "
                         "e.g. effort via interaction parameters)")
    args = ap.parse_args()

    os.makedirs(args.capture_dir, exist_ok=True)
    with open(args.oracle) as f:
        data = json.load(f)
    items = {it["question_id"]: it for it in data if it["question_id"] in args.qids}
    missing = set(args.qids) - set(items)
    assert not missing, "qids not in oracle: %s" % missing

    # Fetch both arms' prompts ONCE (read-only from the daemon).
    for v in {args.control_version, args.new_version}:
        FETCHED[v] = fetch_prompt(v)
        print("[ab] fetched s1e v%d (%d chars)" % (v, len(FETCHED[v][0])), flush=True)

    arms = [("control", args.control_version), ("new", args.new_version)]
    results = {}  # {qid: {arm: run_result}}
    for qid in args.qids:
        results[qid] = {}
        for arm, ver in arms:
            results[qid][arm] = run_arm(arm, ver, qid, items[qid],
                                        args.capture_dir,
                                        control_lived=args.control_lived)

    # ── Report ──────────────────────────────────────────────────────────
    print("\n\n%s\n# S1E COUPLED A/B — ENCODING DIFF\n%s" % ("#" * 70, "#" * 70))
    for qid in args.qids:
        it = items[qid]
        print("\n\n=== %s — '%s' ===" % (qid, it["question"][:100]))
        for arm, ver in arms:
            r = results[qid][arm]
            hard_fail = [k for k in r["asserts"]
                         if k in HARD_CHECKS and not r["asserts"][k][0]]
            valid = "VALID" if not hard_fail else "INVALID(%s)" % ",".join(hard_fail)
            print("\n--- %s (s1e v%d) — %d nodes, %d edges, dup-signal=%d — %s"
                  % (arm, ver, len(r["nodes"]), len(r["edges"]),
                     r["dedup"]["count"], valid))
            print("    internals:")
            for k, (ok, detail) in r["asserts"].items():
                print("      [%s] %s — %s" % ("✓" if ok else "✗", k, detail))
            if r["dedup"]["pairs"]:
                print("    near-dup title pairs (eyeball — new arm should have fewer):")
                for t1, t2, ov in r["dedup"]["pairs"]:
                    print("      %.2f  %r  ~  %r" % (ov, t1[:50], t2[:50]))
            for n in r["nodes"]:
                tm = " @%s" % n["event_time"] if n["event_time"] else ""
                print("      [%s] %s%s" % (n["type"], n["title"][:80], tm))
            print("    full prompts (actual, per round): %d files under %s"
                  % (len(r["cap_files"]), args.capture_dir))

    # Headline: dedup delta (the encode-side value signal a recall-QA A/B can't see)
    print("\n\n%s\n# DEDUP DELTA (headline encode-side signal)\n%s" % ("#" * 70, "#" * 70))
    for qid in args.qids:
        c = results[qid]["control"]["dedup"]["count"]
        n = results[qid]["new"]["dedup"]["count"]
        print("  %s: control=%d  new=%d  Δ=%+d" % (qid, c, n, n - c))


if __name__ == "__main__":
    main()
