"""make_scale_dispatch must stamp encoding_source on EVERY attributed write.

Regression for the gap where brain_batch (and revise_batch / connect /
connect_batch) escaped the encoding_source injection in make_scale_dispatch:
the wrapper only covered remember / remember_batch / revise, so S1 Scribe nodes
written via brain_batch reached the daemon with no source and defaulted to
'anchor'. They were then mislabelled brain-wide and invisible to the dashboard's
encoder view.

The contract: a scale dispatch tags its encoding_source on every command that
mints or attributes a node/edge, regardless of which write tool the agent chose.
"""
import servers.scales.dispatch as D


def _capturing_dispatch(monkeypatch, source='encoder:sonnet'):
    captured = []
    monkeypatch.setattr(
        D, 'daemon_tcp_send',
        lambda cmd, args: captured.append((cmd, args)) or {"ok": True},
    )
    return D.make_scale_dispatch(read_brain=None, encoding_source=source), captured


# One representative arg shape per attributed write command. Asserted to match
# ATTRIBUTED_WRITE_COMMANDS exactly, so adding a command to the set without a
# sample here fails loudly rather than going untested.
_SAMPLES = {
    'remember': {'title': 't', 'content': 'c', 'type': 'fact'},
    'remember_batch': {'nodes': [{'title': 't', 'content': 'c', 'type': 'fact'}]},
    'revise': {'node_id': 'x', 'reason': 'r'},
    'revise_batch': {'revisions': [{'node_id': 'x', 'reason': 'r'}]},
    'connect': {'source_id': 'a', 'target_id': 'b', 'relation': 'r'},
    'connect_batch': {'connections': [{'source_id': 'a', 'target_id': 'b', 'relation': 'r'}]},
    'brain_batch': {'operations': [{'op': 'remember', 'title': 't', 'content': 'c', 'type': 'fact'}]},
}


def test_sample_set_matches_contract():
    assert set(_SAMPLES) == D.ATTRIBUTED_WRITE_COMMANDS, (
        "sample set drifted from ATTRIBUTED_WRITE_COMMANDS — add a sample for any "
        "new attributed write so its tagging is covered"
    )


def test_attributed_is_derived_subset_of_write_commands():
    # ATTRIBUTED is derived from WRITE_COMMANDS minus the non-attributed writes,
    # so it can never drift above the write surface — a new attributed write added
    # to WRITE_COMMANDS is tagged automatically.
    assert D.ATTRIBUTED_WRITE_COMMANDS <= D.WRITE_COMMANDS
    assert D.ATTRIBUTED_WRITE_COMMANDS == D.WRITE_COMMANDS - D.NON_ATTRIBUTED_WRITES


def test_non_attributed_write_is_not_tagged(monkeypatch):
    # A write that doesn't mint attributed nodes/edges (e.g. enrich) must reach
    # the daemon WITHOUT an injected encoding_source — the allow-list is positive.
    dispatch, captured = _capturing_dispatch(monkeypatch)
    dispatch('enrich', {'node_id': 'x', 'vector_type': 'content'})
    assert captured and 'encoding_source' not in captured[0][1]


def test_every_attributed_write_is_tagged(monkeypatch):
    dispatch, captured = _capturing_dispatch(monkeypatch)
    for cmd, args in _SAMPLES.items():
        dispatch(cmd, dict(args))
    assert len(captured) == len(_SAMPLES)
    for cmd, args in captured:
        assert args.get('encoding_source') == 'encoder:sonnet', (
            "%s reached the daemon without encoding_source" % cmd
        )


def test_brain_batch_specifically_tagged(monkeypatch):
    # The proven failure case: a brain_batch encode must arrive tagged so its
    # nodes are attributed to the encoder, not silently defaulted to 'anchor'.
    dispatch, captured = _capturing_dispatch(monkeypatch)
    dispatch('brain_batch', dict(_SAMPLES['brain_batch']))
    assert captured[0][1]['encoding_source'] == 'encoder:sonnet'


def test_explicit_encoding_source_is_preserved(monkeypatch):
    # setdefault semantics: a caller that sets its own source keeps it.
    dispatch, captured = _capturing_dispatch(monkeypatch)
    dispatch('brain_batch', {'operations': [], 'encoding_source': 's2:consolidation'})
    assert captured[0][1]['encoding_source'] == 's2:consolidation'
