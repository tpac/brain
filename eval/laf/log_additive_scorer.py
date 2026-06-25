"""LAF parity scorer — re-express production recall scoring as log-additive layers.

Piece 1 of the LAF MVP (docs/RECALL-SR-REDESIGN.md §18.6). Goal: prove the production
STEP-6 score can be recomposed as a stack of INDEPENDENT log-additive layers without
scrambling rankings — the control baseline every later LAF layer is measured against.

Production STEP 6 (brain_recall.py ~1759-1867) assembles, per candidate:

    blended  = base                          # discovery blend (cosine / keyword / fts5)
    blended += idf2_boost                     # additive    (title rare-token boost)
    if critical:  blended *= CRITICAL_BOOST   # multiplicative
    if mismatch:  blended *= 0.7              # multiplicative
    blended += situation_boost                # additive

i.e.  production = (base + idf2)*C*M + situation .

This module is NON-INVASIVE — it never touches the recall hot path. It reconstructs the
layer features from what recall() already returns (embedding_similarity, _keyword_score,
_source, _context_mismatch, effective_activation, critical) and backs out the one hidden
additive term (idf2+situation, post-multiplier) as a RESIDUAL. A clean (>=0, bounded)
residual proves the extraction is faithful (and a missing multiplier shows up loud as a
negative/oversized residual).

Two compositions per candidate so the harness can compare rankings:
  - replay : base*C*M + additive_eff              == production           (validation)
  - logadd : base*C*M*(1 + additive_eff/base_ref)  (the reformulation under test)

The ONLY difference is whether the additive boost couples to each node's own base
(production, additive) or is applied as a base-independent factor (log-additive). The
ranking divergence between them IS the measurement.
"""
import math

from servers.brain_constants import (
    EMBEDDING_PRIMARY_WEIGHT,
    KEYWORD_FALLBACK_WEIGHT,
    FTS5_PASSTHROUGH_SCORE,
    CRITICAL_BOOST,
    TITLE_MATCH_BOOST,
    SITUATION_WEIGHT,
)

# Loose upper bound on the legitimate additive contribution for a non-critical,
# non-mismatched node: idf2 (<= TITLE_MATCH_BOOST) + situation (<= SITUATION_WEIGHT).
# A residual far outside [-EPS, ADDITIVE_MAX] for such a node means a layer we failed
# to model (the loud-by-default check).
ADDITIVE_MAX = TITLE_MATCH_BOOST + SITUATION_WEIGHT   # 0.5
RESIDUAL_EPS = 0.01   # tolerance: recall rounds embedding_similarity to 3 dp


def reconstruct_base(emb, kw, source):
    """Rebuild the STEP-6 discovery blend from the exposed raw signals."""
    emb = emb or 0.0
    kw = kw or 0.0
    if source == 'fts5_only':
        return FTS5_PASSTHROUGH_SCORE
    if emb > 0 and kw > 0:
        return EMBEDDING_PRIMARY_WEIGHT * emb + KEYWORD_FALLBACK_WEIGHT * kw
    if emb > 0:
        return emb
    return KEYWORD_FALLBACK_WEIGHT * kw   # keyword_only_fallback


def extract_features(node):
    """A recall() result node -> per-layer feature vector. None if no production score."""
    production = node.get('effective_activation')
    if production is None:
        return None
    base = reconstruct_base(node.get('embedding_similarity'),
                            node.get('_keyword_score'),
                            node.get('_source', ''))
    C = CRITICAL_BOOST if node.get('critical') else 1.0
    M = 0.7 if node.get('_context_mismatch') else 1.0
    # idf2 (inside *C*M) + situation (added outside), backed out post-multiplier:
    additive_eff = production - base * C * M
    return {
        'id': node.get('id') or node.get('node_id'),
        'base': base, 'C': C, 'M': M,
        'production': production,
        'additive_eff': additive_eff,
    }


def replay(f):
    """Exact production reconstruction (validation): should equal effective_activation."""
    return f['base'] * f['C'] * f['M'] + f['additive_eff']


def log_additive(f, base_ref):
    """The reformulation under test: every layer an independent log contribution.

        score = exp( log(base) + log(C) + log(M) + log(1 + additive_eff/base_ref) )
              = base * C * M * (1 + additive_eff/base_ref)

    Returns (score, contribs) with contribs in LOG space, so each layer's marginal
    effect on the ranking is a clean addend (per-layer attribution — what LAF needs).
    """
    base = max(f['base'], 1e-9)
    add_factor = 1.0 + max(f['additive_eff'], 0.0) / base_ref
    contribs = {
        'base': math.log(base),
        'critical': math.log(f['C']),
        'mismatch': math.log(f['M']),
        'additive': math.log(add_factor),
    }
    return math.exp(sum(contribs.values())), contribs


def residual_health(f):
    """Validation flag on the backed-out additive residual."""
    a = f['additive_eff']
    if a < -RESIDUAL_EPS:
        return 'negative'                       # over-counted a multiplier
    if f['C'] == 1.0 and a > ADDITIVE_MAX + RESIDUAL_EPS:
        return 'oversized'                      # unmodeled additive layer
    return 'ok'
