"""Measurement-validity layer — is a sweep result a MEASUREMENT or an artifact?

The eval's core blindness class (Tom, 2026-08-24): infrastructure that breaks
during a run degrades to legitimate-looking empties — a dead embedder returns
zero candidates, a missing digest reads as "nothing progressed", a swallowed
error scores as a brain miss — and the aggregates treat the skew as signal.
The substrate usually KNOWS (recall carries `_recall_mode`, errors land in
debug_log, coverage is countable); this module is where the eval ASKS.

Design rules, in force here more than anywhere: the watcher must be louder
than the watched (no check may fail into ok; every skipped check is recorded
and printed), and every question is asked through the OWNER's door — vector
health via `brain.vector_coverage_sweep` (the one repair+report door,
model-aware), journal/arc via `corpus_shape.score_journal` (the stored-object
scorer; `brain.journal_notes` is the encoder's resolve-filtered VIEW and
undercounts), failure modes via `servers.brain_recall.RECALL_FAILURE_MODES`.
No copied thresholds, no second extraction that can drift.

Consumed by sweep.py:

  • preflight_item(brain, ...)  — gate BEFORE measuring a frozen work copy.
    NOT read-only: it runs the vector repair door first (the copy is
    disposable; a seed-pack gap-fill on open defers vectors to a queue no
    eval worker drains). A failure means the item cannot yield valid recall
    measurements — the sweep refuses, or proceeds annotated under
    --force-preflight.

  • suspect_reasons(...)        — per-rep marking AT measurement time.
    A non-empty list stamps the row `measurement_suspect: [...]`; the row
    KEEPS its judge verdict — reports partition aggregates (VALID vs
    SUSPECT), they never drop data. Known residual: the debug_log error
    delta is blunted by Brain's error dedup (60s fingerprint window +
    per-source hourly caps), so `brain_errors_new == 0` is not proof of
    health — a persistent identical failure marks the first rep and can
    read quiet on the rest. recall_mode / harness_error are dedup-proof.

  • canaries (inside preflight) — negative controls for what invariants
    can't see (the P1 blind spot: a bug whose trigger never occurs in the
    inputs is invisible to coverage checks). Degenerate-query probe first —
    it names the actual cause (embedder down) instead of three misleading
    self-retrieval misses; then self-retrieval: a node queried by its own
    title must surface in its own brain. A canary miss means the RULER
    broke, not the brain.
"""
from typing import Any, Dict, List, Optional

from servers.brain_recall import RECALL_FAILURE_MODES  # single source

# Self-retrieval canary: query each sampled node's title; the node must
# appear in this many results. Misses mean recall's vector/rank machinery is
# broken for THIS brain — not that the brain "knows nothing".
CANARY_SAMPLE_NODES = 3
CANARY_TOP_K = 10


def preflight_item(brain, ingest_session_id: str = '') -> Dict[str, Any]:
    """Validity gate over one frozen item work copy.

    Returns {'ok': bool, 'failures': [full sentences], 'reasons': [short
    labels for suspect-marking], 'checks': {...}}. Skipped checks are
    recorded in `checks`, never silent.

    `ingest_session_id` empty (pooled corpora — per-conversation session ids
    aren't reconstructable from the qid) skips the journal check, recorded
    as `journal_checked: False`.
    """
    checks: Dict[str, Any] = {}
    failures: List[str] = []

    # 1. Nodes present — an empty graph can't answer anything. A failed scan
    # is its own failure, never a pass.
    try:
        res = brain.filter_nodes(field='created_at', rich=False, limit=1)
        if 'error' in res:
            raise RuntimeError(res['error'])
        checks['total_nodes'] = int(res.get('total_count', 0))
        if checks['total_nodes'] == 0:
            failures.append('no_live_nodes: graph is empty')
    except Exception as e:
        failures.append('node_scan_failed: %s' % e)

    # 2. Vector health — through the ONE repair+report door (model-aware;
    # a hand-rolled coverage count reads a model swap as "fully covered").
    # Repair is correct here: the work copy is disposable, and the open
    # itself can defer vectors (seed-pack gap-fill) that no worker drains.
    try:
        cov = brain.vector_coverage_sweep(batch_size=50)
        checks['vector_sweep'] = {k: cov.get(k) for k in
                                  ('repaired', 'remaining')}
        if (cov.get('by_type') or {}).get('error'):
            failures.append('embedder_unavailable: %s'
                            % cov['by_type']['error'])
        elif cov.get('stuck') or cov.get('remaining'):
            failures.append(
                'embedding_gap: unembeddable/undrained vectors remain '
                '(stuck=%s remaining=%s) — recall over this brain is '
                'partially blind' % (bool(cov.get('stuck')),
                                     bool(cov.get('remaining'))))
    except Exception as e:
        failures.append('vector_sweep_failed: %s' % e)

    # 3. Journal + arc — the stored-object scorer (three-valued:
    # journal_stored None = pre-journal corpus, don't judge; False = the
    # encoder wrote NO journal object at all — a real gap on any current
    # build; True = present, arc detail recorded for the report).
    checks['journal_checked'] = bool(ingest_session_id)
    if ingest_session_id:
        from eval.longmem.corpus_shape import score_journal
        try:
            j = score_journal(brain, [ingest_session_id])
            checks['journal'] = {k: j.get(k) for k in
                                 ('journal_stored', 'review_notes_count',
                                  'arc_chars', 'arc_basis')}
            if j.get('journal_stored') is False:
                failures.append('no_journal_object: encoder wrote neither '
                                'journal_note traces nor an encoding_journal '
                                'blob for this session')
        except Exception as e:
            failures.append('journal_check_failed: %s' % e)

    # 4. Canaries.
    canary = run_canaries(brain)
    checks['canaries'] = canary
    failures.extend(canary.get('failures', []))

    # Short labels (the part before the first colon) — the suspect
    # vocabulary sweep rows carry; full sentences stay in `failures`.
    reasons = sorted({'preflight:%s' % f.split(':')[0] for f in failures})
    return {'ok': not failures, 'failures': failures, 'reasons': reasons,
            'checks': checks}


def run_canaries(brain) -> Dict[str, Any]:
    """Negative-control probes (see module docstring). mark_accessed=False —
    a probe must not perturb access stats on the brain it measures.

    Returns {'self_retrieval': [(id8, hit)], 'degenerate_mode': str,
             'failures': [str]}.
    """
    failures: List[str] = []
    out: Dict[str, Any] = {'self_retrieval': [], 'degenerate_mode': ''}

    # Degenerate query FIRST: with a dead embedder this names the actual
    # cause; self-retrieval after it would add three misleading misses.
    try:
        res = brain.recall('zxqv gibberish nonesuch canary probe',
                           limit=3, mark_accessed=False) or {}
        out['degenerate_mode'] = res.get('_recall_mode', '') or ''
        if out['degenerate_mode'] in RECALL_FAILURE_MODES:
            failures.append('canary_recall_degraded: %s'
                            % out['degenerate_mode'])
            out['failures'] = failures
            return out
    except Exception as e:
        failures.append('canary_degenerate_query_error: %s' % e)
        out['failures'] = failures
        return out

    # Self-retrieval over encoder-authored nodes ONLY: S2-authored types
    # (community, …) are excluded from recall BY DESIGN (aspects
    # get_excluded_types) — sampling them fails healthy brains, verified on
    # a real corpus item. Both encoder stamps: production Scribe writes
    # encoder:sonnet; the eval replay dispatch stamps plain 'anchor'
    # (fidelity gap, tracked separately).
    try:
        res = brain.filter_nodes(field='encoding_source',
                                 include=['encoder:sonnet', 'anchor'],
                                 rich=False, limit=CANARY_SAMPLE_NODES,
                                 sort_by='created_at', sort_order='desc')
        sample = res.get('nodes') or []
    except Exception as e:
        failures.append('canary_sample_failed: %s' % e)
        out['failures'] = failures
        return out

    for n in sample:
        title = (n.get('title') or '').strip()
        if not title:
            continue
        try:
            got = brain.recall(title, limit=CANARY_TOP_K,
                               mark_accessed=False) or {}
            ids = {(r.get('id') or '')[:8]
                   for r in got.get('results', []) if isinstance(r, dict)}
            hit = (n['id'] or '')[:8] in ids
            out['self_retrieval'].append(((n['id'] or '')[:8], hit))
            if not hit:
                failures.append(
                    'canary_self_retrieval_miss: node %s not in top-%d for '
                    'its own title — recall machinery broken for this brain'
                    % (n['id'][:8], CANARY_TOP_K))
        except Exception as e:
            failures.append('canary_recall_error: %s' % e)

    # A canary that ran zero probes must not report healthy.
    if not out['self_retrieval']:
        failures.append('canary_no_probes_ran: sampled %d encoder node(s), '
                        'none usable for self-retrieval' % len(sample))

    out['failures'] = failures
    return out


def suspect_reasons(*, recall_mode: str = '',
                    new_errors: Optional[List[Dict[str, Any]]] = None,
                    answerer_error: str = '',
                    judge_parse_failed: bool = False,
                    harness_error: str = '') -> List[str]:
    """The per-rep suspect vocabulary — one place, so sweep rows and report
    partitioning can never drift on what 'suspect' means. Empty list = VALID.
    (Preflight labels join via preflight_item's 'reasons'.)"""
    reasons = []
    if harness_error:
        reasons.append('harness_error')
    if recall_mode in RECALL_FAILURE_MODES:
        reasons.append('recall_degraded:%s' % recall_mode)
    for e in (new_errors or []):
        reasons.append('brain_error:%s' % (e.get('type') or e.get('source')
                                           or 'unknown'))
    if answerer_error:
        reasons.append('answerer_error')
    if judge_parse_failed:
        reasons.append('judge_parse_failed')
    return reasons
