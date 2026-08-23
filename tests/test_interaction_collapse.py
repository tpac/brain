"""Tests for servers/interaction_collapse.py — the one-time install collapse.

The collapse's success signal is forgeable: "no override pointers, every name
on its code default" is also what a brain that NEVER held overrides reads like.
So the behavioral tests here assert the BEFORE-state explicitly before trusting
any after-state, and run against a copy of the production brain (which really
does carry the pointers) rather than a synthetic one.

Run: ./dev pytest tests/test_interaction_collapse.py -v
"""
import json

import pytest

from servers.interaction_collapse import (ADOPT, AUDIT_KEY, COLLAPSE_POLICY,
                                          COLLAPSE_VERSION_KEY, COMPARE, PIN,
                                          RETIRE, SKIP, _collapse_overrides,
                                          collapse_seeded_overrides)
from servers.interaction_defaults import (INTERACTION_DEFAULTS,
                                          interaction_fingerprint)
from tests.isolated_brain import IsolatedBrain


class TestPolicyTableShape:
    """The table must not drift from the defaults registry. RETIRE means
    'this name has no code default at all', so the two are one fact stated
    twice — and a name that gains or loses a default must move buckets."""

    def test_retire_is_exactly_the_names_without_a_code_default(self):
        retired = {n for n, v in COLLAPSE_POLICY.items() if v == RETIRE}
        with_default = {n for n in COLLAPSE_POLICY
                        if n in INTERACTION_DEFAULTS}
        assert retired == set(COLLAPSE_POLICY) - with_default, (
            "RETIRE must be exactly the policy names with no code default; "
            "retired=%s without_default=%s"
            % (sorted(retired), sorted(set(COLLAPSE_POLICY) - with_default)))

    def test_every_non_retire_verdict_has_a_code_default(self):
        for name, verdict in COLLAPSE_POLICY.items():
            if verdict != RETIRE:
                assert name in INTERACTION_DEFAULTS, (
                    "%s has verdict %s but no code default — COMPARE has "
                    "nothing to compare against and ADOPT nothing to adopt"
                    % (name, verdict))

    def test_every_registry_name_has_a_policy_entry(self):
        missing = set(INTERACTION_DEFAULTS) - set(COLLAPSE_POLICY)
        assert not missing, (
            "registry names with no collapse verdict (they would be left "
            "frozen as 'unknown' forever): %s" % sorted(missing))

    def test_verdicts_are_from_the_closed_set(self):
        allowed = {COMPARE, ADOPT, PIN, SKIP, RETIRE}
        assert set(COLLAPSE_POLICY.values()) <= allowed


class TestDaemonOnly:
    """Ruled: the collapse must never fire from Brain.__init__, or a frozen
    eval corpus collapses and then floats with future code edits."""

    def test_constructing_a_brain_does_not_collapse(self):
        with IsolatedBrain(load_env=False) as env:
            pointers = [i for i in env.brain.list_interactions()
                        if i.get('active_version') is not None]
            assert pointers, (
                "production copy has no override pointers — this fixture "
                "cannot verify anything about the collapse")

    def test_call_site_is_the_daemon(self):
        import os
        root = os.path.join(os.path.dirname(__file__), '..', 'servers')

        def reads(fn):
            with open(os.path.join(root, fn), encoding='utf-8') as f:
                return 'collapse_seeded_overrides' in f.read()

        assert reads('daemon_server.py'), "daemon must call the collapse"
        assert not reads('brain.py'), \
            "Brain.__init__ must never reach the collapse (corpus float)"
        assert not reads('schema.py'), \
            "the collapse is not a schema migration (corpus float)"


class TestCollapseBehavior:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain(load_env=False) as env:
            self.brain = env.brain
            self.before_pointers = {
                i['name']: i['active_version']
                for i in env.brain.list_interactions()
                if i.get('active_version') is not None}
            self.before_fp = {n: env.brain.get_interaction_stamp(n)['fingerprint']
                              for n in INTERACTION_DEFAULTS}
            yield

    def _pointers(self):
        return {i['name']: i['active_version']
                for i in self.brain.list_interactions()
                if i.get('active_version') is not None}

    def test_before_state_actually_holds_overrides(self):
        """The guard that makes every other assertion here mean something."""
        assert self.before_pointers, "no pointers to collapse"
        assert any(self.brain.get_interaction_stamp(n)['source'] == 'override'
                   for n in INTERACTION_DEFAULTS), \
            "no name stamps as an override — nothing to collapse"

    def test_pinned_and_skipped_pointers_survive(self):
        _collapse_overrides(self.brain)
        after = self._pointers()
        for name, verdict in COLLAPSE_POLICY.items():
            if verdict in (PIN, SKIP) and name in self.before_pointers:
                assert after.get(name) == self.before_pointers[name], \
                    "%s (%s) must never be touched" % (name, verdict)

    def test_effective_values_move_only_where_adopt_allows(self):
        _collapse_overrides(self.brain)
        for name in INTERACTION_DEFAULTS:
            now = self.brain.get_interaction_stamp(name)['fingerprint']
            if COLLAPSE_POLICY.get(name) == ADOPT:
                continue
            assert now == self.before_fp[name], (
                "%s effective value changed (%s -> %s) — the collapse must be "
                "value-preserving outside ADOPT"
                % (name, self.before_fp[name], now))

    def test_matching_compare_names_drop_their_pointer(self):
        dropped_expected = [
            n for n, v in COLLAPSE_POLICY.items()
            if v == COMPARE and n in self.before_pointers
            and self.before_fp[n] == interaction_fingerprint(
                n, *INTERACTION_DEFAULTS[n])]
        assert dropped_expected, \
            "no COMPARE name matches its default — fixture proves nothing"
        _collapse_overrides(self.brain)
        after = self._pointers()
        for name in dropped_expected:
            assert name not in after, \
                "%s matched its code default but kept its pointer" % name

    def test_differing_compare_names_keep_their_pointer(self):
        differing = [
            n for n, v in COLLAPSE_POLICY.items()
            if v == COMPARE and n in self.before_pointers
            and self.before_fp[n] != interaction_fingerprint(
                n, *INTERACTION_DEFAULTS[n])]
        _collapse_overrides(self.brain)
        after = self._pointers()
        for name in differing:
            assert after.get(name) == self.before_pointers[name], \
                "%s differs from its default and must stay an override" % name

    def test_no_interaction_rows_are_deleted(self):
        before = {i['name']: i['total_versions']
                  for i in self.brain.list_interactions()}
        _collapse_overrides(self.brain)
        after = {i['name']: i['total_versions']
                 for i in self.brain.list_interactions()}
        assert after == before, \
            "the collapse is pointer-only — version rows must survive"


class TestDriftRefusal:
    """The loud check: a verdict that would change a name's effective value
    without ADOPT's licence must restore the pointers and raise, so the
    version stays unstamped and the boot retries instead of half-collapsing."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain(load_env=False) as env:
            self.brain = env.brain
            yield

    def _misclassified_name(self):
        """A name whose row genuinely differs from its code default, so
        dropping its pointer moves the effective value."""
        for name in INTERACTION_DEFAULTS:
            stamp = self.brain.get_interaction_stamp(name)
            if stamp['source'] != 'override':
                continue
            if stamp['fingerprint'] != interaction_fingerprint(
                    name, *INTERACTION_DEFAULTS[name]):
                return name
        return None

    def test_drift_restores_pointers_and_raises(self, monkeypatch):
        name = self._misclassified_name()
        assert name, "no diverged override in the fixture — cannot force drift"
        before = {i['name']: i['active_version']
                  for i in self.brain.list_interactions()
                  if i.get('active_version') is not None}
        # RETIRE drops unconditionally and is not licensed to change a value:
        # exactly the shape of a mis-bucketed name.
        monkeypatch.setitem(COLLAPSE_POLICY, name, RETIRE)

        with pytest.raises(RuntimeError, match='effective value'):
            _collapse_overrides(self.brain)

        after = {i['name']: i['active_version']
                 for i in self.brain.list_interactions()
                 if i.get('active_version') is not None}
        assert after == before, \
            "a refused collapse must restore every pointer it dropped"

    def test_refusal_leaves_the_version_unstamped(self, monkeypatch):
        from servers.schema import read_schema_version
        name = self._misclassified_name()
        monkeypatch.setitem(COLLAPSE_POLICY, name, RETIRE)
        collapse_seeded_overrides(self.brain)  # swallows, logs, does not raise
        assert read_schema_version(self.brain.logs_conn, 'logs_meta',
                                   COLLAPSE_VERSION_KEY) == 0, \
            "a failed collapse must not stamp, or it never retries"


class TestAuditRecord:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain(load_env=False) as env:
            self.brain = env.brain
            yield

    def _audit(self):
        from servers.schema import read_meta_value
        raw = read_meta_value(self.brain.logs_conn, 'logs_meta', AUDIT_KEY)
        return json.loads(raw) if raw else None

    def test_audit_covers_every_pointer_and_replays(self):
        before = {i['name']: i['active_version']
                  for i in self.brain.list_interactions()
                  if i.get('active_version') is not None}
        _collapse_overrides(self.brain)
        audit = self._audit()
        assert audit is not None, "no audit record written"
        assert {e['name'] for e in audit} == set(before), \
            "audit must cover exactly the pointers that existed"
        for entry in audit:
            for field in ('name', 'version', 'set_by', 'set_at',
                          'row_fingerprint', 'parameters', 'verdict'):
                assert field in entry, "audit entry missing %s" % field

        # Pure replay: re-activating from the audit restores the before-state.
        for entry in audit:
            self.brain.set_interaction_active(entry['name'], entry['version'],
                                              set_by=entry['set_by'])
        restored = {i['name']: i['active_version']
                    for i in self.brain.list_interactions()
                    if i.get('active_version') is not None}
        assert restored == before, "audit record is not a faithful replay"

    def test_audit_is_never_overwritten(self):
        from servers.schema import read_meta_value, write_meta_value
        write_meta_value(self.brain.logs_conn, 'logs_meta', AUDIT_KEY,
                         '"sentinel"')
        self.brain.logs_conn.commit()
        _collapse_overrides(self.brain)
        assert read_meta_value(self.brain.logs_conn, 'logs_meta',
                               AUDIT_KEY) == '"sentinel"', \
            "a retry must not overwrite the pre-first-attempt audit record"


class TestOnceOnly:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain(load_env=False) as env:
            self.brain = env.brain
            yield

    def test_second_run_is_a_no_op(self):
        collapse_seeded_overrides(self.brain)
        first = {i['name']: i['active_version']
                 for i in self.brain.list_interactions()}
        # Re-pointing a name would be undone by a second collapse if the
        # version gate were not doing its job.
        collapse_seeded_overrides(self.brain)
        second = {i['name']: i['active_version']
                  for i in self.brain.list_interactions()}
        assert second == first

    def test_stamp_is_written_on_success(self):
        from servers.schema import read_schema_version
        collapse_seeded_overrides(self.brain)
        assert read_schema_version(self.brain.logs_conn, 'logs_meta',
                                   COLLAPSE_VERSION_KEY) >= 1
