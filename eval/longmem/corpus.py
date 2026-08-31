"""Frozen Corpus — the durable artifact between Stage 1 (encode) and Stage 2 (recall).

A corpus is a set of fully-encoded, per-item brains plus a manifest. It is
content-addressed by the inputs that determine the encoded graph: the s1e
(encoder) prompt, the ingest-time surface prompt, the S2 cadence, the oracle,
and the exact item set. Same inputs → same `corpus_hash` → reuse on disk
instead of paying the (expensive) encode again.

This is what makes A/B honest and fast at once:
  - RECALL experiments hold corpus_hash fixed for both arms → encoding can't
    contribute to the delta (it's the same frozen bytes).
  - ENCODE experiments build two corpora (different s1e) once each, reuse
    across every recall sweep.

Layout: ~/AgentsContext/eval-corpus/{corpus_hash}/
          manifest.json
          {qid}/                 ← a complete, closed brain (brain.db + WAL)
          {qid}/ ...

The manifest also carries the answerability verdict (the gold-scan run on the
frozen brain) and the S2 delta (what consolidation/community/healer actually
did) so a corpus build is itself a diagnostic, not just a cache.
"""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional


CORPUS_ROOT = os.path.expanduser("~/AgentsContext/eval-corpus")


# ─── Paths ────────────────────────────────────────────────────────────────

def corpus_dir(corpus_hash: str) -> str:
    return os.path.join(CORPUS_ROOT, corpus_hash)


def corpus_item_dir(corpus_hash: str, qid: str) -> str:
    """The per-item frozen brain directory (a complete brain.db)."""
    return os.path.join(corpus_dir(corpus_hash), qid)


def ingest_session_id(qid: str) -> str:
    """THE per-item ingest session id — the key session_context / journal /
    trace rows are stored under in an item brain. One definition; builders
    and readers that hand-rolled 'ingest-{qid}' drifted apart is exactly the
    bug class this module's conventions exist to prevent."""
    return "ingest-%s" % qid


def manifest_path(corpus_hash: str) -> str:
    return os.path.join(corpus_dir(corpus_hash), "manifest.json")


def load_manifest(corpus_hash: str) -> Optional[Dict[str, Any]]:
    path = manifest_path(corpus_hash)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_manifest(corpus_hash: str, manifest: Dict[str, Any]) -> str:
    os.makedirs(corpus_dir(corpus_hash), exist_ok=True)
    path = manifest_path(corpus_hash)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ─── Content addressing ────────────────────────────────────────────────────

def source_token(spec: Optional[str]) -> str:
    """Reduce a prompt source ('active' or a file path) to a stable token.

    'active' → whatever the fresh eval brain resolves for the name (the code
    default — a fresh brain carries no interaction rows). A file path → a
    content hash, so editing the override file changes the corpus hash and
    forces a rebuild.
    """
    if not spec or spec == "active":
        return "active"
    try:
        data = open(spec, "rb").read()
        return "file:" + hashlib.sha1(data).hexdigest()[:8]
    except Exception:
        return "missing:" + os.path.basename(spec)


def interaction_token(version: int, template: str) -> str:
    """Reduce an interaction override to a stable token: 'v24:ab12cd34'.

    Sibling of source_token for a prompt that comes from the interactions
    table rather than a file. The version number alone is not an address: it
    is an install-local counter, and once "no version" means "the code
    default", two corpora built against different code-default generations
    share one hash and load_manifest returns the wrong arm's corpus. Keeping
    the version in the token preserves the readable label; the content hash
    is what makes it an address.
    """
    return "v%s:%s" % (
        version, hashlib.sha1((template or "").encode()).hexdigest()[:8])


def corpus_config_hash(config: Dict[str, Any]) -> str:
    """6-hex content address over everything that determines the encoded graph."""
    blob = json.dumps(config, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:6]


# ─── Variant pins (launcher parity) ─────────────────────────────────────────

# Frozen historical baseline for the variant address keys: the implicit value
# of every valid corpus built before the variants joined the address
# (production has exported both via brain-env.sh throughout). NEVER update
# this pair to track production — it exists so pre-fix corpora keep their
# hashes, and a future production variant flip joins the address precisely
# BECAUSE it differs from this frozen pair.
_ADDRESS_BASELINE_VARIANTS = {"surface_variant": "v5_agentic",
                              "recall_variant": "laf_v1"}


def require_variant_pins() -> Dict[str, str]:
    """Refuse to run in an unpinned shell; return the effective variant pair.

    Every longmem leg encodes or recalls under BRAIN_SURFACE_VARIANT /
    BRAIN_RECALL_VARIANT. The code defaults (v4 / baseline) differ from the
    production pins, so an unpinned shell silently measures a different
    pipeline than the one it reports — same command, same corpus hash,
    different graph. Every entry point (build, sweep, ab, requery) calls this
    before touching a brain.
    """
    surface = os.environ.get("BRAIN_SURFACE_VARIANT")
    recall = os.environ.get("BRAIN_RECALL_VARIANT")
    if not surface or not recall:
        raise SystemExit(
            "[longmem] BRAIN_SURFACE_VARIANT / BRAIN_RECALL_VARIANT unset — "
            "this shell is not production-pinned (launch via ./dev, which "
            "sources hooks/scripts/brain-env.sh). Refusing: an unpinned run "
            "measures a different pipeline than the one it reports.")
    return {"surface_variant": surface, "recall_variant": recall}


def address_variants(config: Dict[str, Any], pins: Dict[str, str]) -> None:
    """Join non-baseline variants to the content address. Absent-on-baseline
    (same pattern as s1e_lived), so every valid pre-fix corpus — all built
    v5_agentic/laf_v1 via ./dev — keeps its hash."""
    for key, baseline in _ADDRESS_BASELINE_VARIANTS.items():
        if pins[key] != baseline:
            config[key] = pins[key]


def check_variant_pins(manifest: Dict[str, Any], pins: Dict[str, str],
                       leg: str) -> None:
    """Read side of the build-time variant_pins stamp: refuse to score a
    corpus under a different pipeline than it was built with. Pre-stamp
    manifests carry no variant_pins and pass (their arms were verified via
    per-item traces)."""
    stamped = manifest.get("variant_pins")
    if stamped and stamped != pins:
        raise SystemExit(
            "[%s] variant mismatch: corpus built under %s, shell is %s — "
            "refusing to score one pipeline as another." % (leg, stamped, pins))


# ─── S2 delta ────────────────────────────────────────────────────────────

def summarize_s2_deltas(deltas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate every run_s2() return from a build into a per-unit summary.

    Each delta is run_s2()'s return — `{'units': {unit_name: result_dict},
    'elapsed_ms': ...}`. We roll up, per unit: how many times it fired, how
    often it did real work, total actions, errors (with samples), and skips.
    Errors are the gold here — the coordinator swallows unit exceptions into
    `{'error': ...}`, so this is where an S2 unit that's quietly broken
    during ingest becomes visible.
    """
    summary: Dict[str, Any] = {}
    for d in deltas or []:
        if not isinstance(d, dict):
            continue
        for unit, res in (d.get("units") or {}).items():
            if not isinstance(res, dict):
                continue
            s = summary.setdefault(unit, {
                "fires": 0, "did_work": 0, "actions": 0,
                "errors": 0, "skipped": 0, "sample_errors": [],
            })
            s["fires"] += 1
            if "error" in res:
                s["errors"] += 1
                if len(s["sample_errors"]) < 3:
                    s["sample_errors"].append(str(res["error"])[:200])
            elif res.get("skipped"):
                s["skipped"] += 1
            else:
                acts = int(res.get("actions", 0) or 0)
                s["actions"] += acts
                if acts > 0:
                    s["did_work"] += 1
    return summary


def merge_s2_totals(per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum per-item S2 summaries across the whole corpus."""
    totals: Dict[str, Any] = {}
    for item in per_item:
        for unit, s in (item.get("s2_delta") or {}).items():
            t = totals.setdefault(unit, {
                "fires": 0, "did_work": 0, "actions": 0, "errors": 0, "skipped": 0,
            })
            for k in ("fires", "did_work", "actions", "errors", "skipped"):
                t[k] += int(s.get(k, 0) or 0)
    return totals
