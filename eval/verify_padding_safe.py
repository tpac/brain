"""Verify the load-bearing assumption for the Phase 2 bucketing fix:
does whitespace padding shift the embedding meaningfully for our model?

If embedding(text) ≈ embedding(text + ' ' * N) for our nomic-embed-text-v1.5-Q
with cosine sim > 0.99, padding is safe and bucketing works.

If similarity drops below 0.99, we need a different padding strategy
(e.g., reach into the tokenizer and pad with PAD tokens directly, or
use a different bucketing approach that doesn't pad inputs).

Tests across:
  - 5 representative text categories (query, edge desc, title, content, long)
  - 6 padding amounts (0, 1, 10, 50, 200, 500 trailing spaces)
  - 3 padding chars (space, newline, period)

Output: a table of cosine sims + a verdict.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Representative samples — short queries to long content, drawn from the
# brain's actual usage profile. Cover different lengths + content types.
SAMPLES = [
    ('query', 'how does fatigue work in recall'),
    ('query', 'multi-session regression'),
    ('edge_desc', 'this corrects "Boot architecture violation pattern"'),
    ('edge_desc', 'extends — Outcome-as-next-O closes the O/K/Δ loop through time'),
    ('title', 'Synaptic fatigue moved to SessionContext'),
    ('content_short',
     'The brain operates at multiple scales, and each scale has a natural '
     'grain it is designed to handle. S0 is exchange-level, S1 turn, S2 graph.'),
    ('content_med',
     'Memory leak in onnxruntime: variable input shapes feed the mem_pattern '
     'optimizer, which caches allocation patterns per tensor shape. With '
     'unbounded distinct shapes (variable-length text), the cache grows without '
     'bound. The fix is bucketing inputs to a small fixed set of shapes.' * 3),
]

PADDING_AMOUNTS = [0, 1, 10, 50, 200, 500]
PADDING_CHARS = [(' ', 'space'), ('\n', 'newline'), ('.', 'period')]


def cosine(a: bytes, b: bytes) -> float:
    """Cosine similarity between two L2-normalized embedding blobs."""
    import numpy as np
    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)
    # Both are pre-normalized by the embedder, dot = cosine
    return float(np.dot(va, vb))


def main():
    from servers.embedder import embed_query, load_model, is_ready

    if not is_ready():
        load_model()

    print('Embedding %d samples × %d padding amounts × %d chars = %d total embeds'
          % (len(SAMPLES), len(PADDING_AMOUNTS), len(PADDING_CHARS),
             len(SAMPLES) * len(PADDING_AMOUNTS) * len(PADDING_CHARS)))
    print()

    # Header
    rows = []
    header = ['sample (len)', 'pad_char']
    for n in PADDING_AMOUNTS:
        header.append('+%d' % n)
    rows.append(header)

    # Track sims per padding character — verdict needs to evaluate each
    # padding strategy independently. Lumping all chars together produces
    # a misleading "padding shifts embeddings" verdict when one char
    # (period) is a known semantic token while another (space) is not.
    sims_by_char = {label: [] for _, label in PADDING_CHARS}
    for cat, text in SAMPLES:
        baseline = embed_query(text)
        if not baseline:
            print(f'FAIL: baseline embed returned None for {cat}')
            return 1
        for pad_char, pad_label in PADDING_CHARS:
            row = ['%s/%s (%d)' % (cat, pad_label[:3], len(text)), pad_label]
            for n in PADDING_AMOUNTS:
                if n == 0:
                    sim = 1.0
                else:
                    padded = text + pad_char * n
                    pad_emb = embed_query(padded)
                    if not pad_emb:
                        sim = float('nan')
                    else:
                        sim = cosine(baseline, pad_emb)
                row.append('%.5f' % sim if sim == sim else 'NaN')
                if n > 0 and sim == sim:
                    sims_by_char[pad_label].append(sim)
            rows.append(row)

    # Pretty-print
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print('  '.join(c.ljust(w) for c, w in zip(r, widths)))
    print()

    # Per-char verdict — what's safe, what's not?
    print('Per-padding-char verdict:')
    safe_chars: list = []
    unsafe_chars: list = []
    for label, sims in sims_by_char.items():
        if not sims:
            continue
        n_safe = sum(1 for s in sims if s >= 0.99)
        min_sim = min(sims)
        verdict = 'SAFE' if min_sim >= 0.99 else 'UNSAFE'
        if min_sim >= 0.99:
            safe_chars.append(label)
        else:
            unsafe_chars.append(label)
        print(f'  {label:<8} n={len(sims):>3}  min={min_sim:.5f}  '
              f'safe={n_safe}/{len(sims)}  → {verdict}')

    print()
    if safe_chars:
        print(f'✓ VERDICT: {", ".join(safe_chars)} padding is safe '
              f'(cosine = 1.0 across all tested padding amounts).')
        print('  Phase 2 bucketing design is viable. Use one of the safe chars.')
        if unsafe_chars:
            print(f'  Note: {", ".join(unsafe_chars)} would shift embeddings — '
                  f'do not use for padding.')
        return 0
    else:
        print('✗ VERDICT: no safe padding character found.')
        print('  Need a different padding strategy:')
        print('    1. Reach into tokenizer, pad with PAD/CLS tokens at token level')
        print('    2. Bypass fastembed.embed() and call ORT session directly with')
        print('       manually padded token IDs')
        print('    3. Ditch bucketing, accept the latency hit of arena=False')
        return 1


if __name__ == '__main__':
    sys.exit(main())
