"""Contract for daemon_config.resolve_db_dir — the Python half of D-13.

One configurable location, one resolution chain, every runtime reads it the
same way: BRAIN_DB_DIR env (trusted) → ~/.config/brain/env (the user knob;
adopted only if the dir exists) → ~/.config/brain/resolved.env (the record
the shell resolver persists; adopted only if brain.db is there — the shell's
4b guard) → legacy dir. The file both readers parse is SOURCED by shell
consumers, so the Python grammar must tolerate shell idioms (export, quotes,
$VAR, inline comments) — two readers of one file must not disagree.

The dashboard mirrors the whole chain in dashboard/db.py (its disconnection
contract forbids importing servers.*) — both implementations are covered by
the same parametrized cases so the mirror can't drift silently.

Run: ./dev python3 -m pytest tests/test_db_resolution.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.daemon_config import resolve_db_dir
from dashboard.db import _brain_dir

RESOLVERS = [resolve_db_dir, _brain_dir]
LEGACY = os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain')


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """No BRAIN_DB_DIR; XDG config at tmp; returns (configdir, make_brain)."""
    monkeypatch.delenv('BRAIN_DB_DIR', raising=False)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    confdir = tmp_path / 'xdg' / 'brain'
    confdir.mkdir(parents=True)

    def make_brain(name, with_db=True):
        d = tmp_path / name
        d.mkdir()
        if with_db:
            (d / 'brain.db').touch()
        return str(d)

    return confdir, make_brain


@pytest.mark.parametrize('resolve', RESOLVERS)
class TestResolutionChain:

    def test_env_var_wins_untested(self, resolve, cfg, monkeypatch):
        # env is trusted verbatim (hook wrappers validate it) — no dir check
        confdir, make = cfg
        (confdir / 'env').write_text(f'BRAIN_DB_DIR={make("c")}\n')
        monkeypatch.setenv('BRAIN_DB_DIR', '/env/wins')
        assert resolve() == '/env/wins'

    def test_user_config_file(self, resolve, cfg):
        confdir, make = cfg
        d = make('c', with_db=False)  # knob needs the dir, not a brain.db
        (confdir / 'env').write_text(f'ANTHROPIC_API_KEY=sk-x\nBRAIN_DB_DIR={d}\n')
        assert resolve() == d

    def test_user_config_beats_resolved_record(self, resolve, cfg):
        confdir, make = cfg
        c, r = make('c'), make('r')
        (confdir / 'env').write_text(f'BRAIN_DB_DIR={c}\n')
        (confdir / 'resolved.env').write_text(f"BRAIN_DB_DIR='{r}'\n")
        assert resolve() == c

    def test_nonexistent_config_dir_falls_through(self, resolve, cfg):
        confdir, make = cfg
        r = make('r')
        (confdir / 'env').write_text('BRAIN_DB_DIR=/does/not/exist\n')
        (confdir / 'resolved.env').write_text(f"BRAIN_DB_DIR='{r}'\n")
        assert resolve() == r

    def test_resolved_record_single_quoted(self, resolve, cfg):
        # resolve-brain-db.sh persists single-quoted values
        confdir, make = cfg
        r = make('r')
        (confdir / 'resolved.env').write_text(
            f"BRAIN_DB_DIR='{r}'\nPLUGIN_ROOT='/x'\n")
        assert resolve() == r

    def test_stale_record_without_brain_db_falls_through(self, resolve, cfg):
        # the shell's 4b guard, mirrored: a record pointing at a dir with no
        # brain.db must not be adopted (shadow-brain prevention)
        confdir, make = cfg
        (confdir / 'resolved.env').write_text(
            f"BRAIN_DB_DIR='{make('r', with_db=False)}'\n")
        assert resolve() == LEGACY

    def test_legacy_fallback(self, resolve, cfg):
        assert resolve() == LEGACY

    def test_blank_config_value_falls_through(self, resolve, cfg):
        confdir, make = cfg
        r = make('r')
        (confdir / 'env').write_text('BRAIN_DB_DIR=\n')
        (confdir / 'resolved.env').write_text(f"BRAIN_DB_DIR='{r}'\n")
        assert resolve() == r

    # -- shell grammar the same file is sourced with --

    def test_export_prefix(self, resolve, cfg):
        confdir, make = cfg
        d = make('c', with_db=False)
        (confdir / 'env').write_text(f'export BRAIN_DB_DIR={d}\n')
        assert resolve() == d

    def test_var_expansion_double_quoted(self, resolve, cfg, monkeypatch):
        confdir, make = cfg
        d = make('c', with_db=False)
        monkeypatch.setenv('MY_BASE', os.path.dirname(d))
        (confdir / 'env').write_text(
            f'BRAIN_DB_DIR="$MY_BASE/{os.path.basename(d)}"\n')
        assert resolve() == d

    def test_inline_comment_on_unquoted_value(self, resolve, cfg):
        confdir, make = cfg
        d = make('c', with_db=False)
        (confdir / 'env').write_text(f'BRAIN_DB_DIR={d} # my brain\n')
        assert resolve() == d
