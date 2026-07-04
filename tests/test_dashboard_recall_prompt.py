"""Recall surface-prompt lazy endpoint — guard + fallback.

The decoding feed stopped shipping the ~35KB surface prompt inline (75% of a
2.3MB polled payload); it's fetched per-card on expand via query_recall_prompt.
recall_ref lands in a filename, so the path-traversal guard is security-relevant
— pin it. Pure function (reads a /tmp file), no DB needed.
"""
from dashboard.queries import recalls


def test_rejects_path_traversal():
    for bad in ('../../etc/passwd', 'a/b', '..', '', 'x/../y'):
        out = recalls.query_recall_prompt(bad)
        assert out == {'error': 'bad recall_ref'}, 'unsafe recall_ref accepted: %r' % bad


def test_missing_file_returns_error_not_crash():
    # A well-formed ref with no judge-result file on disk → clean error dict.
    out = recalls.query_recall_prompt('definitely-not-a-real-ref-9999')
    assert 'error' in out and 'judge_prompt' not in out
