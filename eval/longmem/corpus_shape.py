"""L1 encode-shape scorer — what did the encoder actually WRITE into a corpus?

The recall legs of an s1e A/B measure whether an answer comes back. This one
measures the graph that was written, independent of any query: field
population, voice, edge topology, edge-description craft, catalog targeting.
Run it on two corpus arms (baseline vs candidate prompt) and compare the
per-arm aggregates.

READ PATH — brain API only. Nodes come from `brain.filter_nodes` (structural
sweep) + `brain.get_node` (the canonical pull: KV, situation, corrections,
connections); source_refs through `brain.get_source_refs`, the owner's door.
No hand-rolled SQL against the node/edge tables — three such dumpers drifted
and one hard-broke on a dropped column (B2 ruling).

Every item's frozen brain is COPIED to a scratch dir before opening: Brain()
migrates schema on init, so reading a corpus in place would mutate the frozen
artifact. Same precedent as sweep.py's work-dir copy.

USE
    ./dev python3 eval/longmem/corpus_shape.py 595274
    ./dev python3 eval/longmem/corpus_shape.py <baseline_hash> <candidate_hash>
    ./dev python3 eval/longmem/corpus_shape.py <hash> --json-only > shape.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from eval.longmem.corpus import corpus_item_dir, load_manifest


# ─── What counts as what ──────────────────────────────────────────────────

# Node sources that are NOT the encoder's output. Prefix-matched. The seed
# pack loads at Brain init; s2:* / migration:* / hook:* are other writers.
NON_ENCODER_SOURCE_PREFIXES = ('anchor:seed', 's2:', 'migration:', 'hook:')

# MUST be 0 — a generic relation matches no query and pollutes activation.
GENERIC_RELATIONS = {'related', 'related_to'}

# The verbs the prompt reaches for when it lands an edge on the catalog.
RESCUE_VERBS = {'similar_to', 'corrects', 'supersedes', 'refines',
                'grounds', 'contradicts'}

# Edges NOT written by the encoder: S2's community membership and the
# noise-aspect co_anchored/co_accessed family. Their descriptions are
# machine-templated (22-66 chars, measured), so leaving them in would swamp
# every edge-craft metric an s1e A/B is trying to read. Counted separately.
NON_ENCODER_RELATIONS = {'community_member', 'co_anchored', 'co_accessed',
                         'emergent_bridge'}

# Edge `why` craft bands (chars).
WHY_BAND = (120, 180)
WHY_THIN = 80

# v29 source_ref shape: an 8-char hex trace_event id.
HEX8 = re.compile(r'^[0-9a-f]{8}$')

# emotion_label's contract default — present means the encoder set something.
NEUTRAL_LABELS = {'', 'neutral', 'none'}

KV_FIELDS = ('situation', 'reasoning', 'question', 'thought', 'event_time',
             'their_raw_quote', 'my_raw_quote')

# ── Journal guardrails ──
# MEASURE THE STORED OBJECT, NOT THE RENDER. Two iterations of this block
# were rebuilt because they parsed the captured s1e prompt: the first read
# the `## Review` heading to the next h2 and swallowed the payload's own
# `### Node Catalog` render as 20-37K of "over-production"; the second fixed
# the boundary but still measured heading FORMAT — and the two gate arms
# journal identically while rendering differently (legacy `## Encoding
# Journal` per-run headers vs the `## Arc`/`## Review` contract sections),
# so it reported the baseline as journaling nothing when it had written 45
# notes. Notes now come from the journal_note trace rows and the arc from
# the persisted journal blob. Neither can be spoofed by a render change.
#
# Below this the arc section is a stub/placeholder, not an arc.
ARC_MIN_CHARS = 20
# The arc inside the stored journal blob. Bounded by the next h2/h3 or the
# next run delimiter — the blob concatenates runs, and mixed heading levels
# (`### Node Catalog` under a `## Review`) mean h2 alone is not a boundary.
ARC_SECTION_RE = re.compile(r'^## Arc\s*$(.*?)(?=^#{2,3} |^--- Run |\Z)',
                            re.M | re.S)
# Per-run delimiter written into `encoding_journal_{sid}`.
JOURNAL_RUN_DELIM = '--- Run '
# The blob is read whole, not through the surface's 1500-char recency slice.
JOURNAL_BLOB_MAX = 1000000

# Metrics that CANNOT come from a built corpus — the brain stores state, not
# the ops that produced it. Reported in the output so nobody reads their
# absence as a zero.
NOT_COMPUTABLE = [
    "content_edits vs full-content rewrite on revise — an op shape, not "
    "state. Needs the batch op capture (round payloads under "
    "{item}/payloads/) or node_revised trace deltas.",
    "absorb / disconnect / archive-with-survivor op mix — the encoding_run "
    "trace tallies created+revised+archived (reported as ops_*), but not "
    "which write op produced them.",
    "connect_to (edge declared at creation) vs a separate connect op — both "
    "land as the same edge row.",
    "Encoder rounds, tokens, latency, scout notes — payload/trace facts, "
    "not graph facts. (The journal's arc/review IS read, from the captured "
    "prompts — see the journal block.)",
    "Nodes the encoder created and S2 later archived — filter_nodes reads "
    "archived=0, so consolidation casualties are out of the scored set.",
    "Whether a source_ref points at the turn that actually GENERATED the "
    "node (load-bearing-ness). Only shape (8-char hex) is checkable here.",
]


# ─── Item scoring ─────────────────────────────────────────────────────────

def _all_live_node_ids(brain) -> List[str]:
    """Every non-archived node id, paged past filter_nodes' 200-row cap.

    `gt` filters the queried field, so the cursor field IS created_at. The
    collected count is reconciled against the DAL's exact total_count from
    the first (unbounded) page — a silent undercount here would deflate every
    rate downstream.
    """
    collected: Dict[str, str] = {}
    cursor: Optional[str] = None
    total: Optional[int] = None
    while True:
        res = brain.filter_nodes(field='created_at', gt=cursor, rich=False,
                                 limit=200, sort_by='created_at',
                                 sort_order='asc')
        if 'error' in res:
            raise RuntimeError('filter_nodes: %s' % res['error'])
        if total is None:
            total = res.get('total_count', 0)
        rows = res.get('nodes') or []
        fresh = [r for r in rows if r['id'] not in collected]
        if not fresh:
            break
        for r in fresh:
            collected[r['id']] = r['created_at']
        cursor = rows[-1]['created_at']
    if total is not None and len(collected) != total:
        raise RuntimeError(
            'node sweep collected %d of %d live nodes — paging lost rows '
            '(duplicate created_at at a page boundary?)'
            % (len(collected), total))
    return list(collected)


def _is_encoder(source: str) -> bool:
    src = source or ''
    return not any(src.startswith(p) for p in NON_ENCODER_SOURCE_PREFIXES)


def score_journal(brain, session_ids: List[str]) -> Dict[str, Any]:
    """Journal guardrails off the STORED objects (see the block comment).

    Notes: `journal_note` trace rows, through `query_traces` — the public
    trace door, the same one `_runs` uses. NOT `brain.journal_notes()`: that
    is the encoder's continuity VIEW (resolve-filtered, last-K-runs, open
    pins) and would undercount what the run actually wrote. Full prose is
    `metadata['note']`; `summary` is capped at 80 chars and would truncate
    every note's length.

    Arc: the persisted per-session journal blob (`encoding_journal_{sid}`,
    read through `brain.get_recent_encoding_journal`), whose run bodies carry
    the `## Arc` sections. The dedicated arc digest
    (`session_context_{sid}` via `brain.session_context_for`) is preferred
    when populated — it is the arc's own storage door — but `write_session_arc`
    left it empty in both gate arms, so `arc_basis` names which one answered.
    """
    res = brain.query_traces(ref_type='journal_note', hours=None, limit=20000)
    if res.get('truncated'):
        raise RuntimeError('journal_note trace pull truncated: %s'
                           % res['truncated'])
    # S1E's guardrail counts S1E's notes: S2 units journal through the same
    # ref_type at scale 's2' (on the lived path both write — mixing them
    # inflated the count 15 vs the true 6 on the first lived smoke).
    notes = [e for e in (res.get('events') or []) if e.get('scale') == 's1']
    note_chars = 0
    for ev in notes:
        md = ev.get('metadata')
        if isinstance(md, str):
            md = json.loads(md or '{}')
        note_chars += len((md or {}).get('note') or '')

    digest = ''.join(brain.session_context_for(sid) for sid in session_ids)
    blob = ''.join(brain.get_recent_encoding_journal(sid, JOURNAL_BLOB_MAX)
                   for sid in session_ids)
    if digest.strip():
        arcs, basis = [digest.strip()], 'session_context_digest'
    else:
        arcs = [m.strip() for m in ARC_SECTION_RE.findall(blob)]
        basis = 'stored_journal_blob' if blob else 'none'

    if not notes and not blob and not digest:
        # No journal object of any kind — an older corpus, not a silent zero.
        return {'journal_stored': None, 'arc_basis': 'none',
                'arc_produced': None, 'arc_chars': None,
                'arc_sections': None, 'journal_runs': None,
                'review_chars': None, 'review_notes_count': None}

    return {
        'journal_stored': bool(notes or blob or digest),
        'arc_basis': basis,
        'arc_produced': any(len(a) > ARC_MIN_CHARS for a in arcs),
        'arc_chars': sum(len(a) for a in arcs),
        'arc_sections': len(arcs),
        'journal_runs': blob.count(JOURNAL_RUN_DELIM),
        'review_chars': note_chars,
        'review_notes_count': len(notes),
    }


def _journal_aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rates over items that stored a journal object at all."""
    scored = [it for it in items if it['journal_stored'] is not None]
    if not scored:
        return {'journal': False, 'scored_items': 0,
                'arc_basis': 'none', 'arc_produced_rate': None,
                'notes_total': None, 'notes_mean': None,
                'note_chars_total': None, 'note_chars_mean': None,
                'note_chars_max': None}
    notes = [it['review_notes_count'] for it in scored]
    chars = [it['review_chars'] for it in scored]
    return {
        'journal': True,
        'scored_items': len(scored),
        'arc_basis': ','.join(sorted({it['arc_basis'] for it in scored
                                      if it['arc_basis'] != 'none'})) or 'none',
        'arc_produced_rate': _rate(
            sum(1 for it in scored if it['arc_produced']), len(scored)),
        'arc_chars_mean': round(mean(it['arc_chars'] for it in scored), 1),
        'notes_total': sum(notes),
        'notes_mean': round(mean(notes), 1),
        'note_chars_total': sum(chars),
        'note_chars_mean': round(mean(chars), 1),
        'note_chars_max': max(chars),
    }


def _runs(brain) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """(node_id → s1e run ordinal, per-item op tallies) from `encoding_run`.

    One encoding_run event per s1e run (chain `s1e-{sid8}-{stop}`), its
    metadata carrying the run's `created` / `revised` / `archived` node ids.
    That membership list is the run boundary — NOT the node_created chain,
    which the mutation emitter writes under a single `s0-{date}-mutation`
    chain for the whole item and would collapse every run into one.

    Empty index when a corpus predates the trace (older builds) — the caller
    then falls back to created_at ordering for run comparisons.
    """
    res = brain.query_traces(ref_type='encoding_run', hours=None, limit=20000)
    if res.get('truncated'):
        raise RuntimeError('encoding_run trace pull truncated: %s'
                           % res['truncated'])
    events = sorted(res.get('events') or [],
                    key=lambda e: e.get('created_at') or '')
    index: Dict[str, int] = {}
    tally: Dict[str, Any] = {
        's1e_runs': len(events), 'ops_created': 0, 'ops_revised': 0,
        'ops_archived': 0,
        # The sessions that encoded this item — the key the journal blob and
        # the arc digest are both stored under.
        'session_ids': sorted({ev.get('session_id') or '' for ev in events
                               if ev.get('session_id')}),
    }
    for i, ev in enumerate(events):
        md = ev.get('metadata')
        if isinstance(md, str):
            md = json.loads(md or '{}')
        md = md or {}
        for nid in md.get('created') or []:
            index.setdefault(nid, i)
        for key in ('created', 'revised', 'archived'):
            tally['ops_' + key] += len(md.get(key) or [])
    return index, tally


def _rate(hits: int, n: int) -> float:
    return round(100.0 * hits / n, 1) if n else 0.0


def score_item(brain, qid: str) -> Dict[str, Any]:
    """Shape metrics for one frozen brain."""
    live_ids = _all_live_node_ids(brain)
    all_nodes = brain.get_node(live_ids) if live_ids else {}
    source_mix = Counter(nd.get('encoding_source') or ''
                         for nd in all_nodes.values())
    nodes = {nid: nd for nid, nd in all_nodes.items()
             if _is_encoder(nd.get('encoding_source'))}
    n = len(nodes)

    runs, tally = _runs(brain)
    ordering = 'encode_run_chain' if runs else 'created_at'
    created_at = {nid: (nd.get('created_at') or '') for nid, nd in nodes.items()}

    field_hits = Counter()
    refs_total = refs_conforming = 0
    content_chars: List[int] = []
    title_chars: List[int] = []
    emotion_only = label_only = 0
    neighbors: Dict[str, set] = {nid: set() for nid in nodes}
    neighbors_all: Dict[str, set] = {nid: set() for nid in nodes}

    # (source, target, relation) → the edge row, deduped across both owners.
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for nid, nd in nodes.items():
        kv = nd.get('_metadata') or {}
        for f in KV_FIELDS:
            if str(kv.get(f) or '').strip():
                field_hits[f] += 1

        emo = nd.get('emotion')
        has_emo = emo is not None and float(emo) != 0.0
        has_label = str(nd.get('emotion_label') or '').strip().lower() \
            not in NEUTRAL_LABELS
        if has_emo and has_label:
            field_hits['emotion_pair'] += 1
        elif has_emo:
            emotion_only += 1
        elif has_label:
            label_only += 1

        refs = brain.get_source_refs(nid)
        if refs:
            field_hits['source_refs'] += 1
            refs_total += len(refs)
            refs_conforming += sum(1 for r in refs if HEX8.match(str(r)))

        content_chars.append(len(nd.get('content') or ''))
        title_chars.append(len(nd.get('title') or ''))

        for c in nd.get('connections') or []:
            outgoing = c.get('direction') == 'outgoing'
            src, tgt = (nid, c.get('id')) if outgoing else (c.get('id'), nid)
            neighbors_all[nid].add(c.get('id'))
            for rel in c.get('relations') or []:
                relation = rel.get('relation') or ''
                if relation not in NON_ENCODER_RELATIONS:
                    neighbors[nid].add(c.get('id'))
                key = (src, tgt, relation)
                if key in edges:
                    continue
                # Neighbor birth time comes off the connection row; the owner's
                # off its node. Both endpoints therefore always resolvable.
                birth = {nid: created_at.get(nid, ''),
                         c.get('id'): c.get('created_at') or ''}
                edges[key] = {
                    'relation': key[2],
                    'why': rel.get('description') or '',
                    'src_born': birth.get(src, ''),
                    'tgt_born': birth.get(tgt, ''),
                    'src_run': runs.get(src),
                    'tgt_run': runs.get(tgt),
                    'self': src == tgt,
                }

    enc_edges = [e for e in edges.values()
                 if e['relation'] not in NON_ENCODER_RELATIONS]
    other_mix = Counter(e['relation'] for e in edges.values()
                        if e['relation'] in NON_ENCODER_RELATIONS)
    rel_mix = Counter(e['relation'] for e in enc_edges)
    why_lens = [len(e['why']) for e in enc_edges]
    n_edges = len(enc_edges)

    catalog_hits = 0
    catalog_scored = 0
    for e in enc_edges:
        if e['self']:
            continue
        catalog_scored += 1
        if runs:
            # A node with no run (a seed) predates every run by definition.
            src_r = e['src_run'] if e['src_run'] is not None else -1
            tgt_r = e['tgt_run'] if e['tgt_run'] is not None else -1
            if tgt_r < src_r:
                catalog_hits += 1
        elif e['tgt_born'] and e['src_born'] and e['tgt_born'] < e['src_born']:
            catalog_hits += 1

    degrees = [len(s) for s in neighbors.values()]
    deg_hist = Counter(degrees)
    deg_all = [len(s) for s in neighbors_all.values()]

    ops_write = tally['ops_created'] + tally['ops_revised']
    rec = {
        'qid': qid,
        'nodes_live_total': len(all_nodes),
        'nodes_scored': n,
        'encoding_source_mix': dict(source_mix),
        # ── s1e op mix (encoding_run traces, not graph state) ──
        's1e_runs': tally['s1e_runs'],
        'ops_created': tally['ops_created'],
        'ops_revised': tally['ops_revised'],
        'ops_archived': tally['ops_archived'],
        'revise_share_pct': _rate(tally['ops_revised'], ops_write),
        # ── field population (% of scored nodes) ──
        'situation_pct': _rate(field_hits['situation'], n),
        'reasoning_pct': _rate(field_hits['reasoning'], n),
        'question_pct': _rate(field_hits['question'], n),
        'thought_pct': _rate(field_hits['thought'], n),
        'event_time_pct': _rate(field_hits['event_time'], n),
        'emotion_pair_pct': _rate(field_hits['emotion_pair'], n),
        'emotion_singleton_emotion_only': emotion_only,
        'emotion_singleton_label_only': label_only,
        'source_refs_pct': _rate(field_hits['source_refs'], n),
        'source_refs_total': refs_total,
        'source_refs_hex8_pct': _rate(refs_conforming, refs_total),
        # ── voice ──
        'their_raw_quote_pct': _rate(field_hits['their_raw_quote'], n),
        'my_raw_quote_pct': _rate(field_hits['my_raw_quote'], n),
        # ── content ──
        'content_chars_mean': round(mean(content_chars), 1) if content_chars else 0.0,
        'title_chars_mean': round(mean(title_chars), 1) if title_chars else 0.0,
        # ── edges (encoder-written only; see NON_ENCODER_RELATIONS) ──
        'edges_total': n_edges,
        'edges_per_node': round(n_edges / n, 2) if n else 0.0,
        'degree_hist': {str(k): v for k, v in sorted(deg_hist.items())},
        'degree_mean': round(mean(degrees), 2) if degrees else 0.0,
        'degree0_pct': _rate(deg_hist[0], n),
        'degree1_pct': _rate(deg_hist[1], n),
        'degree2_pct': _rate(deg_hist[2], n),
        'degree_0_2_pct': _rate(deg_hist[0] + deg_hist[1] + deg_hist[2], n),
        'why_chars_mean': round(mean(why_lens), 1) if why_lens else 0.0,
        'why_in_band_pct': _rate(
            sum(1 for L in why_lens if WHY_BAND[0] <= L <= WHY_BAND[1]), n_edges),
        'why_thin_pct': _rate(sum(1 for L in why_lens if L < WHY_THIN), n_edges),
        'why_empty': sum(1 for L in why_lens if L == 0),
        'relation_mix': dict(rel_mix.most_common()),
        'relations_distinct': len(rel_mix),
        'generic_relation_count': sum(rel_mix[r] for r in GENERIC_RELATIONS),
        'rescue_verb_pct': _rate(
            sum(rel_mix[r] for r in RESCUE_VERBS), n_edges),
        'catalog_target_pct': _rate(catalog_hits, catalog_scored),
        'catalog_ordering': ordering,
        # ── everything else on the graph, for context ──
        'non_encoder_edges': sum(other_mix.values()),
        'non_encoder_relation_mix': dict(other_mix.most_common()),
        'degree_all_mean': round(mean(deg_all), 2) if deg_all else 0.0,
        'degree_all_0_2_pct': _rate(
            sum(1 for d in deg_all if d <= 2), n),
    }
    rec.update(score_journal(brain, tally['session_ids']))
    return rec


# ─── Corpus scoring ───────────────────────────────────────────────────────

def _item_brain_dirs(corpus_hash: str) -> List[Tuple[str, str]]:
    """[(label, brain_dir)] for a corpus. Pooled corpora point every manifest
    item at one shared brain — deduped so it's scored once."""
    manifest = load_manifest(corpus_hash)
    if not manifest:
        raise SystemExit('no manifest for corpus %s' % corpus_hash)
    out: List[Tuple[str, str]] = []
    seen = set()
    for it in manifest.get('items') or []:
        qid = it['qid']
        d = it.get('brain_dir') or corpus_item_dir(corpus_hash, qid)
        real = os.path.realpath(d)
        if real in seen:
            continue
        seen.add(real)
        out.append((qid, d))
    return out


def _open_copy(src: str, work: str):
    """Copy a frozen item brain into `work` and open it read-only-ish.

    skip_embedder=True: shape scoring needs no vectors, and it also skips the
    seed-pack top-up Brain init would otherwise write into the copy.
    """
    shutil.copytree(src, work,
                    ignore=shutil.ignore_patterns('brain-*.json', 'payloads'))
    os.environ['BRAIN_DB_DIR'] = work
    os.environ['BRAIN_TMP_DIR'] = work
    from servers.brain import Brain
    return Brain(db_path=os.path.join(work, 'brain.db'), skip_embedder=True)


def score_corpus(corpus_hash: str) -> Dict[str, Any]:
    manifest = load_manifest(corpus_hash) or {}
    items: List[Dict[str, Any]] = []
    for qid, src in _item_brain_dirs(corpus_hash):
        if not os.path.isdir(src):
            raise SystemExit('missing item brain dir: %s' % src)
        work = tempfile.mkdtemp(prefix='corpus-shape-%s-' % corpus_hash)
        item_work = os.path.join(work, 'brain')
        try:
            # Brain init is chatty on stdout; keep stdout clean for the JSON.
            with contextlib.redirect_stdout(sys.stderr):
                brain = _open_copy(src, item_work)
                try:
                    rec = score_item(brain, qid)
                finally:
                    brain.close()
            print('[shape] %s/%s  nodes=%d edges=%d journal_notes=%s'
                  % (corpus_hash, qid, rec['nodes_scored'], rec['edges_total'],
                     rec['review_notes_count']),
                  file=sys.stderr, flush=True)
            items.append(rec)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    return {
        'corpus_hash': corpus_hash,
        'label': manifest.get('label', ''),
        'config': manifest.get('config', {}),
        'items': items,
        'aggregate': _aggregate(items),
        'pooled': _pooled(items),
        'journal': _journal_aggregate(items),
        'not_computable': NOT_COMPUTABLE,
    }


# Scalar metrics aggregated as mean/sd/min/max across items.
AGG_KEYS = (
    'nodes_scored', 's1e_runs', 'ops_revised', 'revise_share_pct',
    'situation_pct', 'reasoning_pct', 'question_pct',
    'thought_pct', 'event_time_pct', 'emotion_pair_pct', 'source_refs_pct',
    'source_refs_hex8_pct', 'their_raw_quote_pct', 'my_raw_quote_pct',
    'content_chars_mean', 'title_chars_mean', 'edges_total', 'edges_per_node',
    'degree_mean', 'degree0_pct', 'degree1_pct', 'degree2_pct',
    'degree_0_2_pct', 'why_chars_mean', 'why_in_band_pct', 'why_thin_pct',
    'rescue_verb_pct', 'catalog_target_pct',
)


def _aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for k in AGG_KEYS:
        vals = [float(it[k]) for it in items]
        if not vals:
            agg[k] = {'mean': 0.0, 'sd': 0.0, 'min': 0.0, 'max': 0.0, 'n': 0}
            continue
        agg[k] = {'mean': round(mean(vals), 2),
                  'sd': round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
                  'min': round(min(vals), 2), 'max': round(max(vals), 2),
                  'n': len(vals)}
    return agg


def _pooled(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counters summed corpus-wide — per-item relation mixes are too thin to
    read individually."""
    rel: Counter = Counter()
    other: Counter = Counter()
    deg: Counter = Counter()
    src: Counter = Counter()
    for it in items:
        rel.update(it['relation_mix'])
        other.update(it['non_encoder_relation_mix'])
        deg.update({int(k): v for k, v in it['degree_hist'].items()})
        src.update(it['encoding_source_mix'])
    n_edges = sum(rel.values())
    n_nodes = sum(deg.values())
    return {
        'nodes_scored': n_nodes,
        'edges_total': n_edges,
        'encoding_source_mix': dict(src.most_common()),
        'relation_mix': dict(rel.most_common()),
        'non_encoder_relation_mix': dict(other.most_common()),
        'relations_distinct': len(rel),
        'generic_relation_count': sum(rel[r] for r in GENERIC_RELATIONS),
        'rescue_verb_pct': _rate(sum(rel[r] for r in RESCUE_VERBS), n_edges),
        'degree_hist': {str(k): deg[k] for k in sorted(deg)},
        'degree_0_2_pct': _rate(deg[0] + deg[1] + deg[2], n_nodes),
        'source_refs_total': sum(it['source_refs_total'] for it in items),
        'why_empty': sum(it['why_empty'] for it in items),
        'emotion_singletons': sum(it['emotion_singleton_emotion_only']
                                  + it['emotion_singleton_label_only']
                                  for it in items),
    }


# ─── Rendering ────────────────────────────────────────────────────────────

ITEM_COLS = (
    ('qid', 'qid', '%-16s'), ('nodes', 'nodes_scored', '%5s'),
    ('runs', 's1e_runs', '%4s'), ('rev%', 'revise_share_pct', '%5s'),
    ('edg', 'edges_total', '%4s'), ('e/n', 'edges_per_node', '%5s'),
    ('sit', 'situation_pct', '%5s'), ('rsn', 'reasoning_pct', '%5s'),
    ('qst', 'question_pct', '%5s'), ('tht', 'thought_pct', '%5s'),
    ('evt', 'event_time_pct', '%5s'), ('emo', 'emotion_pair_pct', '%5s'),
    ('refs', 'source_refs_pct', '%5s'), ('hex8', 'source_refs_hex8_pct', '%5s'),
    ('their', 'their_raw_quote_pct', '%6s'), ('mine', 'my_raw_quote_pct', '%5s'),
    ('deg02', 'degree_0_2_pct', '%6s'), ('why', 'why_chars_mean', '%6s'),
    ('band', 'why_in_band_pct', '%5s'), ('resc', 'rescue_verb_pct', '%5s'),
    ('cat', 'catalog_target_pct', '%5s'),
    # journal guardrails — null (printed '-') when no journal object stored
    ('jrn', 'journal_stored', '%4s'), ('arc', 'arc_chars', '%5s'),
    ('note', 'review_notes_count', '%5s'), ('nchr', 'review_chars', '%6s'),
)

# Columns fed by the journal block: no aggregate row (their rates are over
# journal-bearing items only, which the mean/sd row cannot express).
JOURNAL_COLS = {'journal_stored', 'arc_chars', 'review_chars',
                'review_notes_count'}


def _cell(v: Any) -> Any:
    """Table cell: None → '-', bool → Y/N (a null metric must not read 0)."""
    if v is None:
        return '-'
    if isinstance(v, bool):
        return 'Y' if v else 'N'
    return v


def render_corpus(rep: Dict[str, Any]) -> str:
    L = []
    L.append('=' * 140)
    L.append('CORPUS %s  label=%s  items=%d'
             % (rep['corpus_hash'], rep['label'] or '-', len(rep['items'])))
    L.append('=' * 140)
    L.append(' '.join(fmt % name for name, _, fmt in ITEM_COLS))
    for it in rep['items']:
        L.append(' '.join(fmt % _cell(it[key]) for _, key, fmt in ITEM_COLS))
    L.append('-' * 140)
    agg = rep['aggregate']
    L.append('mean ' + ' '.join(
        fmt % ('' if key in JOURNAL_COLS else agg[key]['mean'])
        for _, key, fmt in ITEM_COLS[1:]))
    L.append('sd   ' + ' '.join(
        fmt % ('' if key in JOURNAL_COLS else agg[key]['sd'])
        for _, key, fmt in ITEM_COLS[1:]))
    p = rep['pooled']
    L.append('')
    L.append('pooled: %d nodes, %d edges, %d source_refs, %d empty edge-why, '
             '%d emotion singletons'
             % (p['nodes_scored'], p['edges_total'], p['source_refs_total'],
                p['why_empty'], p['emotion_singletons']))
    L.append('degree histogram (encoder edges): %s   (deg 0-2 = %.1f%% of nodes)'
             % (p['degree_hist'], p['degree_0_2_pct']))
    L.append('encoding_source mix: %s' % p['encoding_source_mix'])
    L.append('relations (%d distinct): %s'
             % (p['relations_distinct'],
                ', '.join('%s=%d' % kv for kv in
                          list(p['relation_mix'].items())[:20])))
    L.append('excluded (S2 / noise-aspect, not encoder output): %s'
             % (', '.join('%s=%d' % kv for kv in
                          p['non_encoder_relation_mix'].items()) or 'none'))
    verdict = 'OK' if p['generic_relation_count'] == 0 else 'LOUD FAIL'
    L.append('generic related/related_to: %d  → %s'
             % (p['generic_relation_count'], verdict))
    # Only items that HAVE edges say anything about the ordering used.
    orderings = sorted({it['catalog_ordering'] for it in rep['items']
                        if it['edges_total']})
    L.append('rescue-verb share: %.1f%%   catalog-targeting ordering: %s'
             % (p['rescue_verb_pct'], ','.join(orderings) or '-'))
    if 'created_at' in orderings:
        L.append('  ⚠ created_at ordering OVERCOUNTS catalog targeting — with '
                 'no encoding_run traces, an edge to a sibling written '
                 'moments earlier in the SAME run scores as a hit. Only '
                 'compare arms that share an ordering.')
    L.append(_render_journal(rep['journal']))
    return '\n'.join(L)


def _render_journal(j: Dict[str, Any]) -> str:
    if not j['journal']:
        return ('journal guardrails — none: no stored journal object '
                '(no journal_note rows, no encoding_journal blob)')
    line = ('journal guardrails (%d items, stored objects): notes %d total '
            '(mean %.1f/item, %d chars total, max %d) | arc produced %.1f%% '
            '(mean %.0f chars, basis=%s)'
            % (j['scored_items'], j['notes_total'], j['notes_mean'],
               j['note_chars_total'], j['note_chars_max'],
               j['arc_produced_rate'], j['arc_chars_mean'], j['arc_basis']))
    if ',' in j['arc_basis']:
        line += ('\n  ⚠ items in this corpus answered from DIFFERENT arc '
                 'stores — the digest and the journal blob are not the same '
                 'measurement; arc_chars is not comparable across them.')
    return line


COMPARE_KEYS = AGG_KEYS


def render_compare(reps: List[Dict[str, Any]]) -> str:
    L = []
    L.append('=' * 100)
    L.append('ARM COMPARISON  (per-item mean ± sd)')
    L.append('=' * 100)
    head = '%-24s' % 'metric' + ''.join(
        '%22s' % (('%s%s' % (r['corpus_hash'],
                             ('/' + r['label']) if r['label'] else ''))[:21])
        for r in reps)
    L.append(head)
    for k in COMPARE_KEYS:
        row = '%-24s' % k
        for r in reps:
            a = r['aggregate'][k]
            row += '%22s' % ('%.2f ± %.2f' % (a['mean'], a['sd']))
        L.append(row)
    L.append('-' * 100)
    for label, key in (('generic relations', 'generic_relation_count'),
                       ('distinct relations', 'relations_distinct'),
                       ('pooled edges', 'edges_total'),
                       ('pooled nodes', 'nodes_scored')):
        L.append('%-24s' % label
                 + ''.join('%22s' % r['pooled'][key] for r in reps))
    L.append('-' * 100)
    L.append('%-24s' % 'JOURNAL (stored objects)'
             + ''.join('%22s' % ('n=%s' % r['journal']['scored_items'])
                       for r in reps))
    for label, key in (('notes_total', 'notes_total'),
                       ('notes_mean', 'notes_mean'),
                       ('note_chars_total', 'note_chars_total'),
                       ('arc_produced_rate', 'arc_produced_rate'),
                       ('arc_chars_mean', 'arc_chars_mean'),
                       ('arc_basis', 'arc_basis')):
        L.append('%-24s' % label
                 + ''.join('%22s' % str(_cell(r['journal'].get(key)))[:21]
                           for r in reps))
    if len({r['journal']['arc_basis'] for r in reps}) > 1:
        L.append('⚠ arms read the arc from DIFFERENT stores — compare '
                 'arc_produced_rate only across a shared arc_basis.')
    return '\n'.join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('corpus_hash', nargs='+')
    p.add_argument('--json-only', action='store_true',
                   help='emit only the JSON report on stdout')
    args = p.parse_args()

    reps = [score_corpus(h) for h in args.corpus_hash]

    if not args.json_only:
        for r in reps:
            print(render_corpus(r))
            print()
        if len(reps) > 1:
            print(render_compare(reps))
            print()
        print('NOT COMPUTABLE FROM CORPUS STATE:')
        for note in NOT_COMPUTABLE:
            print('  - %s' % note)
        print()
        print('=== JSON ===')
    print(json.dumps({'corpora': reps}, indent=2, default=str))


if __name__ == '__main__':
    main()
