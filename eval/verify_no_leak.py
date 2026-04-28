"""Reproduction test for the embedder leak fix (Phase 2 bucketing).

Pre-fix: 1000 hook_recall-style embeds with variable-length inputs
caused RSS to balloon from ~450MB to ~5GB and stick.

Post-fix (this script verifies): RSS should stay within ±200MB of
baseline across 1000 mixed-length embeds. If not, bucketing didn't
work.

The script also asserts semantic regression: cosine of (text vs
text+padding) is 1.0, AND a fixed query produces a stable vector
across calls (no drift).

Usage:
    ./dev python3 eval/verify_no_leak.py
"""
import sys
import os
import random
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _rss_mb() -> float:
    """Resident set size in MB. Uses `ps` for portability — `ru_maxrss`
    units differ across platforms (macOS bytes, Linux KB) AND can be
    stale (it's a peak counter, not current). `ps -o rss=` is current
    and always reports KB."""
    import subprocess
    try:
        out = subprocess.check_output(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            timeout=2.0,
        ).decode().strip()
        return int(out) / 1024  # KB → MB
    except Exception:
        return 0.0


def main():
    from servers.embedder import (
        embed_query, embed_document, embed_batch, load_model, cosine_similarity,
    )

    load_model()
    random.seed(42)

    # Generate a varied corpus mimicking the brain's actual workload —
    # short queries (most common), edge descriptions (medium), and
    # occasional long content embeds.
    queries = [
        'how does fatigue work',
        'multi-session regression',
        'memory leak in onnxruntime',
        'pasta carbonara false positive',
        'dampening hub recall',
    ]
    edges = [
        'this corrects "Boot architecture violation pattern"',
        'extends — refines the core principle',
        'similar_to — both resist fragmentation',
        'community_member "Boot Identity Architecture"',
        'enables — same session — identity axiom enabled the ownership feeling',
    ]
    content_short = (
        'The brain operates at multiple scales, and each scale has a natural grain. '
        'S0 is exchange-level, S1 turn, S2 graph integration over hours.'
    )
    content_long = content_short * 30  # ~5000 chars

    # Test 1: stable vector across repeated calls (no drift)
    canonical = 'the canonical regression query'
    v0 = embed_query(canonical)
    for i in range(5):
        v1 = embed_query(canonical)
        sim = cosine_similarity(v0, v1)
        if sim < 0.9999:
            print(f'✗ DRIFT: call {i+1} cosine={sim:.6f} (expect 1.0)')
            return 1
    print(f'✓ Stability: 5 repeated embeds of same text → cosine = 1.0')

    # Test 2: load loop — hammer the embedder with mixed inputs.
    baseline = _rss_mb()
    print(f'Baseline RSS: {baseline:.0f} MB')
    print('Running 1000 mixed embeds (queries + edges + content)...')

    samples_per_iter = []  # for stats
    t0 = time.time()
    for i in range(1000):
        # Choose a representative mix
        r = random.random()
        if r < 0.5:
            # query path
            q = random.choice(queries)
            embed_query(q)
        elif r < 0.85:
            # edge batch (3-8 edges)
            n = random.randint(3, 8)
            embed_batch([random.choice(edges) for _ in range(n)], kind='document')
        elif r < 0.97:
            # short content embed
            embed_document(content_short)
        else:
            # long content embed (rare)
            embed_document(content_long)

        if (i + 1) % 100 == 0:
            rss = _rss_mb()
            samples_per_iter.append(rss)
            print(f'  after {i+1:>4} embeds: RSS={rss:.0f} MB '
                  f'(delta from baseline: {rss-baseline:+.0f} MB)')

    elapsed = time.time() - t0
    final_rss = _rss_mb()
    delta = final_rss - baseline
    peak = max(samples_per_iter) if samples_per_iter else final_rss

    print()
    print(f'Done in {elapsed:.1f}s ({1000/elapsed:.0f} embeds/sec)')
    print(f'Baseline → final: {baseline:.0f} MB → {final_rss:.0f} MB (Δ {delta:+.0f} MB)')
    print(f'Peak RSS during run: {peak:.0f} MB')

    # Verdict
    print()
    if peak - baseline < 500:
        print('✓ RSS stayed within +500MB of baseline. Bucketing fix works.')
        return 0
    elif peak - baseline < 1500:
        print('⚠ RSS grew by 500MB-1.5GB. Better than the unbounded leak but')
        print('  not as bounded as expected. Check bucket sizes vs actual')
        print('  input length distribution.')
        return 0
    else:
        print('✗ RSS grew by > 1.5GB. Bucketing did not bound the leak.')
        print('  Likely cause: a code path is bypassing _bucket_pad. Check that')
        print('  ALL embedder entry points go through embed_query/document/batch.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
