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

# ── Journal guardrails (encode-side, read off the captured prompt) ──
# A run's journal is rendered into the NEXT run's prompt, so the LAST s1e
# prompt capture is where run N-1's arc/notes are read. An item with fewer
# than this many captures never had a journal rendered at all — it reports
# null, not zero, and drops out of the rates.
JOURNAL_MIN_RUNS = 2
# Below this the `## Arc` section is a stub/placeholder, not an arc.
ARC_MIN_CHARS = 20
# The prompt MIXES heading levels: `## Arc` / `## Review` are h2, but what
# follows them (`### Node Catalog`, `### Conversation Timeline`) is h3. A
# section therefore ends at the next h2 OR h3 — stopping only at `## ` reads
# the catalog and transcript renders as Review content and reports a 20-37K
# "over-production" that is really the payload's own render.
JOURNAL_SECTION_RE = r'^## %s\s*$(.*?)(?=^#{2,3} |\Z)'

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


def _journal_section(text: str, name: str) -> Optional[str]:
    """The `## {name}` body, bounded by the next h2/h3 heading or EOF.

    None when the heading is absent — the renderer omits an empty section, so
    absent and present-but-empty are different facts and must not collapse.
    """
    m = re.search(JOURNAL_SECTION_RE % name, text, re.M | re.S)
    return m.group(1).strip() if m else None


def score_journal(item_dir: str) -> Dict[str, Any]:
    """Journal guardrails off the item's captured s1e prompts.

    `capture_files_for` owns the recorder layout
    ({item}/payloads/{date}/s1e-{sid8}-{stop}/000-prompt.md) and orders by
    (stop, seq, attempt), so [-1] is the LAST run's prompt — the one carrying
    the previous run's journal. Items below JOURNAL_MIN_RUNS captures report
    null and are excluded from the rates; corpora with no captures at all
    (older builds) report journal_captures=0 and null metrics.
    """
    from eval.longmem.fresh_brain import capture_files_for
    files = capture_files_for(item_dir, prefix='s1e', kind='prompt')
    rec: Dict[str, Any] = {
        'journal_captures': len(files),
        'journal_rendered': None, 'arc_produced': None, 'arc_chars': None,
        'review_chars': None, 'review_notes_count': None,
    }
    if len(files) < JOURNAL_MIN_RUNS:
        return rec

    text = open(files[-1]).read()
    arc = _journal_section(text, 'Arc')
    review = _journal_section(text, 'Review')
    rec.update({
        # Neither heading present = the journal-dead case: the previous run
        # wrote nothing for the renderer to emit.
        'journal_rendered': arc is not None or review is not None,
        'arc_produced': len(arc or '') > ARC_MIN_CHARS,
        'arc_chars': len(arc or ''),
        'review_chars': len(review or ''),
        'review_notes_count': sum(1 for line in (review or '').splitlines()
                                  if line.strip()),
    })
    return rec


def _journal_aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rates over the ELIGIBLE items only (>= JOURNAL_MIN_RUNS captures)."""
    eligible = [it for it in items if it['arc_produced'] is not None]
    captures = sum(it['journal_captures'] for it in items)
    if not eligible:
        return {'payloads': captures > 0, 'eligible_items': 0,
                'items_with_captures': sum(1 for it in items
                                           if it['journal_captures']),
                'journal_rendered_rate': None, 'arc_produced_rate': None,
                'review_chars_mean': None, 'review_chars_max': None,
                'review_notes_mean': None}
    reviews = [it['review_chars'] for it in eligible]
    return {
        'payloads': True,
        'eligible_items': len(eligible),
        'items_with_captures': sum(1 for it in items if it['journal_captures']),
        'journal_rendered_rate': _rate(
            sum(1 for it in eligible if it['journal_rendered']), len(eligible)),
        'arc_produced_rate': _rate(
            sum(1 for it in eligible if it['arc_produced']), len(eligible)),
        'arc_chars_mean': round(mean(it['arc_chars'] for it in eligible), 1),
        'review_chars_mean': round(mean(reviews), 1),
        'review_chars_max': max(reviews),
        'review_notes_mean': round(
            mean(it['review_notes_count'] for it in eligible), 1),
    }


def _runs(brain) -> Tuple[Dict[str, int], Dict[str, int]]:
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
    tally = {'s1e_runs': len(events), 'ops_created': 0, 'ops_revised': 0,
             'ops_archived': 0}
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
    return {
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
            # Journal guardrails read the ORIGINAL item dir — payloads are
            # deliberately not copied into the work dir.
            rec.update(score_journal(src))
            print('[shape] %s/%s  nodes=%d edges=%d journal_prompts=%d'
                  % (corpus_hash, qid, rec['nodes_scored'], rec['edges_total'],
                     rec['journal_captures']),
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
    # journal guardrails — null (printed '-') on items with < 2 captures
    ('jrn', 'journal_captures', '%4s'), ('rndr', 'journal_rendered', '%5s'),
    ('arc', 'arc_chars', '%5s'), ('rvw', 'review_chars', '%5s'),
    ('note', 'review_notes_count', '%5s'),
)

# Columns fed by the journal block: no aggregate row (their rates are over
# eligible items only, which the mean/sd row cannot express).
JOURNAL_COLS = {'journal_captures', 'journal_rendered', 'arc_chars',
                'review_chars', 'review_notes_count'}


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
    if not j['payloads']:
        return 'journal guardrails — payloads: none (corpus has no s1e prompt captures)'
    if not j['eligible_items']:
        return ('journal guardrails — %d item(s) captured but none reached %d '
                'runs; no journal was ever rendered'
                % (j['items_with_captures'], JOURNAL_MIN_RUNS))
    return ('journal guardrails (%d eligible items): journal rendered %.1f%% '
            '| arc produced %.1f%% (mean %.0f chars) | review chars mean %.0f '
            'max %d | review notes mean %.1f'
            % (j['eligible_items'], j['journal_rendered_rate'],
               j['arc_produced_rate'], j['arc_chars_mean'],
               j['review_chars_mean'], j['review_chars_max'],
               j['review_notes_mean']))


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
    L.append('%-24s' % 'JOURNAL (eligible only)'
             + ''.join('%22s' % ('n=%s' % r['journal']['eligible_items'])
                       for r in reps))
    for label, key in (('journal_rendered_rate', 'journal_rendered_rate'),
                       ('arc_produced_rate', 'arc_produced_rate'),
                       ('review_chars_mean', 'review_chars_mean'),
                       ('review_chars_max', 'review_chars_max'),
                       ('review_notes_mean', 'review_notes_mean')):
        L.append('%-24s' % label
                 + ''.join('%22s' % _cell(r['journal'].get(key))
                           for r in reps))
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
