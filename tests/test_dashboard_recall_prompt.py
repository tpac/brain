"""Recall surface-prompt lazy endpoint — guard + fallback.

The decoding feed stopped shipping the ~35KB surface prompt inline (75% of a
2.3MB polled payload); it's fetched per-card on expand via query_recall_prompt.
Keyed by chain_id (judge payload under {BRAIN_DB_DIR}/payloads/) with the
recall_ref /tmp read as the time-bounded legacy branch. Both keys land in
filesystem paths, so the traversal guards are security-relevant — pin the
PROPERTY: hostile input never reaches a file read, and the reply is a clean
error dict either way.
"""
import json
import os

from dashboard import db as dash_db
from dashboard.queries import recalls


def test_rejects_path_traversal_both_keys():
    for bad in ('../../etc/passwd', 'a/b', '..', 'x/../y'):
        out = recalls.query_recall_prompt(recall_ref=bad)
        assert 'error' in out and 'judge_prompt' not in out, \
            'unsafe recall_ref accepted: %r' % bad
        out = recalls.query_recall_prompt(chain_id=bad)
        assert 'error' in out and 'judge_prompt' not in out, \
            'unsafe chain_id accepted: %r' % bad
        assert dash_db.chain_payload_files(bad) == [], \
            'unsafe chain_id reached the filesystem: %r' % bad


def test_missing_payload_returns_error_not_crash():
    # Well-formed keys with nothing on disk → clean error dict.
    out = recalls.query_recall_prompt(recall_ref='definitely-not-a-real-ref',
                                      chain_id='s1r-notreal-9999')
    assert 'error' in out and 'judge_prompt' not in out


def test_reads_judge_payload_by_chain(tmp_path, monkeypatch):
    # Recorder layout: payloads/{date}/{chain}/000-judge.json
    monkeypatch.setenv('BRAIN_DB_DIR', str(tmp_path))
    chain_dir = tmp_path / 'payloads' / '2026-08-03' / 's1r-abcd1234-7'
    chain_dir.mkdir(parents=True)
    (chain_dir / '000-judge.json').write_text(json.dumps({
        'recall_ref': 'abcd1234-7',
        'surface_prompt': 'THE PROMPT',
        'surface_output': 'THE CONTEXT',
    }))
    out = recalls.query_recall_prompt(chain_id='s1r-abcd1234-7')
    assert out == {'judge_prompt': 'THE PROMPT'}
    prompt, output = recalls.read_judge_payload('s1r-abcd1234-7')
    assert (prompt, output) == ('THE PROMPT', 'THE CONTEXT')


def test_legacy_tmp_fallback(tmp_path, monkeypatch):
    # Pre-migration rows have no payload — the recall_ref /tmp file answers.
    monkeypatch.setenv('BRAIN_DB_DIR', str(tmp_path / 'empty-db-dir'))
    monkeypatch.setenv('BRAIN_TMP_DIR', str(tmp_path))
    (tmp_path / 'brain-judge-result-legacy99-3.json').write_text(json.dumps({
        'surface_prompt': 'OLD PROMPT', 'surface_output': 'OLD OUT'}))
    prompt, output = recalls.read_judge_payload('s1r-legacy99-3',
                                                recall_ref='legacy99-3')
    assert (prompt, output) == ('OLD PROMPT', 'OLD OUT')
