"""Deploy-contract gate (DISTRIBUTION-READINESS.md 5.0b).

Executable answer to "how do I know every place a deploy touches." Hand-lists
failed twice (the 62-file build manifest; the `.claude/settings.json` permission
entry) — so every assertion here is a SHAPE-SCAN over the git-tracked tree,
never an enumeration. A new file that hardcodes a deploy-coupled name fails
this gate automatically, because the gate reads the tree, not a list.

Four assertions:
  1. version lockstep    — plugin.json.version == marketplace.json entry version
                           (drift is silent and breaks `/plugin update`)
  2. adapter-name        — for name N from plugin.json, every occurrence of
     containment           `mcp__plugin_N_`, `com.N.`, `<owner>/N` sits in a
                           small allowlist
  3. host-neutrality     — servers/ may reference the CC manifest only for the
     (D-11)                embedder block
  4. name derivation     — capital `Anchor` appears only where config owns it
     (D-12)                (xfail until 5.0c consolidates the literals)

The `com.N.` sub-shape is degenerate while N equals the service-layer name
(`brain` — D-11 keeps launchd labels `com.brain.*` forever, so today every
service file legitimately matches). It arms itself the moment 5.2 renames the
adapter: N becomes `entity` and any `com.entity.` occurrence is a leak.

Run: ./dev python3 -m pytest tests/test_deploy_contract.py -v
"""
import ast
import functools
import json
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The name the service layer keeps regardless of what the CC adapter is called
# (D-11). While the adapter name equals this, the `com.N.` shape cannot
# distinguish adapter leaks from legitimate service code.
SERVICE_NAME = 'brain'

# Never-ships content (the 5.1 denylist — tests/archive/ and tests/results/
# are denylisted there for the same reason they are excluded here: they carry
# dev-machine paths and would fail 5.1's scrub-grep). Everything else tracked
# is in scope — including tests/ (they ship, D-8) and dotfiles (the
# permission-entry miss lived in `.claude/settings.json`). This file excludes
# itself: the scanner must name the shapes it hunts, so scanning it self-trips
# (its own docstrings would arm assertion 2's post-rename check forever).
SCOPE_EXCLUDE_PREFIXES = ('docs/', 'eval/', 'tests/archive/', 'tests/results/')
SCOPE_EXCLUDE_FILES = frozenset({'CLAUDE.md', 'tests/test_deploy_contract.py'})


def _tracked_files():
    try:
        out = subprocess.run(
            ['git', 'ls-files'], cwd=REPO, capture_output=True, text=True,
            timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip('not a git checkout — deploy gate only runs in the dev repo',
                    allow_module_level=True)
    return [line for line in out.splitlines() if line]


TRACKED = _tracked_files()
if '.claude-plugin/plugin.json' not in TRACKED:
    # `git ls-files` succeeded but against some OTHER repo (e.g. an installed
    # copy nested in a user's dotfiles checkout) — scanning that tree would
    # pass vacuously, not meaningfully.
    pytest.skip('tracked tree is not the plugin repo — deploy gate only runs '
                'in the dev repo', allow_module_level=True)
SCOPE = [
    p for p in TRACKED
    if not p.startswith(SCOPE_EXCLUDE_PREFIXES) and p not in SCOPE_EXCLUDE_FILES
]


@functools.lru_cache(maxsize=None)
def _read(rel_path):
    # the tree does not change during a run; several scans read every file
    with open(os.path.join(REPO, rel_path), 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')


@functools.lru_cache(maxsize=None)
def _manifest():
    """The package manifest — `build-plugin.sh --list`, the ONE owner of
    "what ships". Both the manifest sanity test and the reachability scan
    read it from here; one subprocess per run."""
    out = subprocess.run(
        ['bash', os.path.join(REPO, 'build-plugin.sh'), '--list'],
        capture_output=True, text=True, timeout=60, cwd=REPO,
    )
    assert out.returncode == 0, out.stderr
    return [l for l in out.stdout.splitlines() if l.strip()]


def _load_json(rel_path):
    return json.loads(_read(rel_path))


PLUGIN = _load_json('.claude-plugin/plugin.json')
MARKETPLACE = _load_json('.claude-plugin/marketplace.json')
PLUGIN_NAME = PLUGIN['name']
OWNER = re.search(r'github\.com/([^/]+)', PLUGIN['repository']).group(1)


def _files_matching(pattern):
    """Scope files whose path or content matches `pattern` (compiled regex)."""
    hits = []
    for rel in SCOPE:
        if pattern.search(rel) or pattern.search(_read(rel)):
            hits.append(rel)
    return hits


class TestVersionLockstep:
    """plugin.json and marketplace.json must carry the same version, and that
    version must be the one the release is meant to ship.

    `/plugin update` compares the two; drift is silent until a user's update
    no-ops. This is a risk on every release, forever. Agreement alone is not
    enough: gate C in the export script checked only that the pair agreed, so
    the pair could agree on the private build counter (9.7.x) and ship it as
    the public launch. The value below is the ONE home of the expected
    version — a release bump edits it in the same commit as the manifests,
    which is the point: the bump becomes a reviewable line, not a drift.
    """

    # D-10: public launches at 0.9.0 — "not yet v1" is a claim about delivered
    # value, and the private plugin's 9.x counter means nothing to a stranger.
    EXPECTED_VERSION = '0.9.0'

    def test_marketplace_entry_exists_for_plugin(self):
        names = [p.get('name') for p in MARKETPLACE['plugins']]
        assert PLUGIN_NAME in names, (
            f'marketplace.json has no entry named {PLUGIN_NAME!r} (entries: {names})')

    def test_versions_match(self):
        entry = next(p for p in MARKETPLACE['plugins'] if p.get('name') == PLUGIN_NAME)
        assert entry.get('version') == PLUGIN['version'], (
            f"version drift: plugin.json={PLUGIN['version']!r} "
            f"marketplace.json={entry.get('version')!r} — breaks /plugin update")

    def test_version_is_the_expected_release(self):
        assert PLUGIN['version'] == self.EXPECTED_VERSION, (
            f"plugin.json={PLUGIN['version']!r} but the release is pinned at "
            f"{self.EXPECTED_VERSION!r} — bump EXPECTED_VERSION in the same "
            'commit as the manifests, or the export ships the wrong version')


class TestAdapterNameContainment:
    """Every occurrence of an adapter-name shape sits in a small allowlist.

    The allowlists are the ONLY enumeration here, and they list where the name
    is SUPPOSED to live — a new file hardcoding the shape fails without anyone
    updating anything.
    """

    def test_mcp_tool_prefix_contained(self):
        # The permission entry is the shape's one legitimate home in CODE —
        # this exact file was the miss that motivated the gate. The migration
        # guide names the prefix once because its verification step tells the
        # user what a correct install looks like.
        allowed = {'.claude/settings.json', 'MIGRATING.md'}
        pattern = re.compile(re.escape(f'mcp__plugin_{PLUGIN_NAME}_'))
        leaks = set(_files_matching(pattern)) - allowed
        assert not leaks, (
            f'adapter tool-prefix mcp__plugin_{PLUGIN_NAME}_ leaked into: {sorted(leaks)}')

    def test_launchd_namespace_contained(self):
        if PLUGIN_NAME == SERVICE_NAME:
            pytest.skip(
                'com.N. shape is degenerate while adapter name == service name '
                f'({SERVICE_NAME!r}); arms automatically at the 5.2 rename')
        # Post-rename: the service namespace stays com.brain.* (D-11); the
        # adapter name must never grow a launchd namespace of its own.
        pattern = re.compile(re.escape(f'com.{PLUGIN_NAME}.'))
        leaks = _files_matching(pattern)
        assert not leaks, (
            f'adapter launchd namespace com.{PLUGIN_NAME}.* appeared in: {sorted(leaks)} '
            f'— service labels stay com.{SERVICE_NAME}.* (D-11)')

    def test_repo_slug_contained(self):
        # The GitHub slug belongs in the manifest (homepage/repository), in
        # the two install commands users are handed (README, and the migration
        # guide an old-plugin user gives to Claude on its own), and in gate B's
        # allowlist entries naming those strings — nowhere else in shipped
        # code. Also catches /Users/<owner>/<name> personal paths, which double
        # as a scrub-grep (5.1) early warning.
        allowed = {'.claude-plugin/plugin.json', 'README.md', 'MIGRATING.md',
                   'scripts/export-public-tree.sh'}
        pattern = re.compile(rf'{re.escape(OWNER)}/{re.escape(PLUGIN_NAME)}\b')
        leaks = set(_files_matching(pattern)) - allowed
        assert not leaks, (
            f'repo slug {OWNER}/{PLUGIN_NAME} leaked into: {sorted(leaks)}')


class TestHostNeutrality:
    """D-11: the service layer must not know it runs under Claude Code.

    servers/ may reference the CC manifest only for the embedder block. The
    day a service name derives from plugin.json, the rename hazard returns.
    `servers/embedder.py` IS the embedder block, so it is exempt wholesale;
    any other servers/ file must keep each manifest reference within an
    embedder context, so a new non-embedder reference in brain.py fails too.
    """

    EMBEDDER_FILE = 'servers/embedder.py'
    CONTEXT_LINES = 2

    def test_servers_reference_manifest_only_for_embedder(self):
        pattern = re.compile(r'plugin\.json|\.claude-plugin')
        leaks = []
        for rel in SCOPE:
            if not rel.startswith('servers/') or rel == self.EMBEDDER_FILE:
                continue
            lines = _read(rel).splitlines()
            for i, line in enumerate(lines):
                if not pattern.search(line):
                    continue
                lo = max(0, i - self.CONTEXT_LINES)
                window = lines[lo:i + self.CONTEXT_LINES + 1]
                if not any('embed' in w.lower() for w in window):
                    leaks.append(f'{rel}:{i + 1}')
        assert not leaks, (
            f'servers/ manifest references outside an embedder context: '
            f'{leaks} — D-11 forbids the service layer deriving anything '
            f'else from plugin.json')

    def test_no_host_install_layout_in_neutral_layers(self):
        # The host-neutral layers must not know where any host installs
        # plugins. skills/ and hooks/ ARE the CC adapter — exempt by scope.
        # (Caught live: daemon_launch.py hardcoded a marketplace install path
        # as an interpreter candidate; the manifest-ref check above was blind
        # to it because install paths never mention plugin.json.)
        pattern = re.compile(r'\.claude/plugins|plugins/marketplaces')
        leaks = [
            rel for rel in SCOPE
            if rel.startswith(('servers/', 'dashboard/')) and pattern.search(_read(rel))
        ]
        assert not leaks, (
            f'host install-layout paths in host-neutral layers: {leaks}')


class TestNameDerivation:
    """D-12: instance names derive from config; no shipped `Anchor` literal.

    xfail(strict) until 5.0c consolidates the literals into BRAIN_AGENT_NAME
    (143 across 43 shipped files, measured 2026-08-31) — the day 5.0c
    completes, this XPASSes and the marker must come off, making the gate
    permanent.

    ⚠ WHAT IS NOT BEING CAUGHT WHILE THIS IS DISARMED. A strict-xfail gate is
    worse than an absent one, because a green suite reads as coverage: a
    reader concludes the literal check ran. It did not. Until the marker comes
    off, **a newly added `Anchor` literal in any shipped file produces no
    signal from this suite** — not a failure, not a warning. New code is on
    the honour system, and the cost lands as a bigger 5.0c sweep later.
    """

    # Where the name may live as CONFIG: the manifests and the userConfig
    # description text. (`displayName` is `Entity`, not the instance name —
    # D-12: the manifest names the product, config names the instance.)
    ALLOWED = {'.claude-plugin/plugin.json', '.claude-plugin/marketplace.json'}

    @pytest.mark.xfail(
        strict=True,
        reason='5.0c open — Anchor literals not yet consolidated to config (D-12)')
    def test_agent_name_only_in_config(self):
        pattern = re.compile(r'\bAnchor\b')
        leaks = set(_files_matching(pattern)) - self.ALLOWED
        assert not leaks, (
            f'`Anchor` literal outside the config allowlist in {len(leaks)} files: '
            f'{sorted(leaks)[:10]}{" …" if len(leaks) > 10 else ""}')


class TestShippedScriptsReachable:
    """Every shipped `hooks/scripts/*` file must be REACHABLE from the wiring
    that actually runs things: hooks.json, the plist templates, skills, servers,
    the root build/deploy scripts — or the named allowlist for files wired
    outside the tree entirely. Five dead scripts (a retired encoding path, two
    client libs, two utilities) shipped for months because nothing asserted
    this; this pins the class, not the instances.

    Reachability, not flat reference-counting: a pair of dead scripts that
    reference each other (encoding-hook.sh ⇄ encoding_hook.py) must not
    vouch for itself. Seeds = every in-scope file OUTSIDE hooks/scripts/;
    a script joins the live set only when its basename or stem appears in a
    seed or an already-live script. Matching includes the stem (name minus
    extension) because references are often constructed — Python imports drop
    `.py`, the installers build plist paths from a `$LABEL` variable.
    """

    # Wired via the user's own settings/keybindings, invisible to any tracked
    # file: the statusline command and the /watch live listener launcher.
    ALLOW = frozenset({'brain-statusline.sh', 'brain-watch'})

    def test_every_shipped_script_reachable(self):
        scripts = [p for p in TRACKED if p.startswith('hooks/scripts/')]
        seeds = [p for p in SCOPE if not p.startswith(('hooks/scripts/', 'tests/'))]
        seed_text = '\n'.join(_read(p) for p in seeds)

        def names(rel):
            base = os.path.basename(rel)
            stem = os.path.splitext(base)[0]
            return {base, stem}

        live = {p for p in scripts if os.path.basename(p) in self.ALLOW}
        live |= {p for p in scripts if any(n in seed_text for n in names(p))}
        # Fixpoint: scripts referenced by live scripts become live.
        while True:
            live_text = '\n'.join(_read(p) for p in live)
            grown = live | {p for p in scripts
                            if any(n in live_text for n in names(p))}
            if grown == live:
                break
            live = grown

        orphans = sorted(set(scripts) - live)
        assert not orphans, (
            'shipped hooks/scripts files with no wiring path (dead on every '
            f'install — delete them or name their external wiring in ALLOW): {orphans}')

    # ── servers/: reachability by IMPORT, not by name ──
    # hooks/scripts/* are wired by NAME (hooks.json, plists, skills), so text
    # containment is the right edge above. servers/* are wired by IMPORT, and
    # a module's stem (`dal`, `clock`, `contract`) appears in prose
    # everywhere, so name-matching would vouch for anything. The edge here is
    # the import graph: ast over every shipped .py, relative and absolute
    # imports resolved to files, `from pkg import submodule` included, lazy
    # in-function imports included (ast sees them; a runtime probe would
    # not). Seeds are the non-servers shipped code (hooks/scripts/*.py,
    # dashboard/) plus module paths named as strings in the wiring
    # (`-m servers.daemon_server` in brain-daemon, `servers/brain_mcp.py` in
    # mcp-launch.sh). `git ls-files` guarantees the manifest cannot ROT and
    # says nothing about whether a listed file is REACHED; the suite cannot
    # tell either — it runs inside the tree and exercises only what a test
    # imports. Eval-only instruments have shipped to every install this way.

    @staticmethod
    def _imports(rel):
        """Dotted names a Python file imports. Relative imports resolve against
        the file's package; `from X import y` yields X and X.y (y may be a
        submodule)."""
        try:
            tree = ast.parse(_read(rel), filename=rel)
        except SyntaxError:
            return set()
        pkg = rel.split('/')[:-1]
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg[:len(pkg) - (node.level - 1)]
                    mod = '.'.join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ''
                if mod:
                    out.add(mod)
                    out.update(f'{mod}.{a.name}' for a in node.names)
        return out

    def test_every_shipped_servers_module_reachable(self):
        manifest = _manifest()
        shipped = {p for p in manifest if p.startswith('servers/')}
        py = {p for p in shipped if p.endswith('.py')}

        def resolve(dotted):
            rel = dotted.replace('.', '/')
            return {c for c in (rel + '.py', rel + '/__init__.py') if c in py}

        def packages_of(rel):
            # importing servers.a.b runs servers/__init__ and servers/a/__init__
            parts = rel.split('/')[:-1]
            return {'/'.join(parts[:i]) + '/__init__.py'
                    for i in range(1, len(parts) + 1)} & py

        def edges(rel):
            out = set(packages_of(rel))
            for d in imports[rel]:
                out |= resolve(d)
            return out

        imports = {p: self._imports(p) for p in py}   # parse each file once
        seeds = [p for p in manifest if not p.startswith('servers/')]
        seed_text = '\n'.join(_read(p) for p in seeds)
        live = set()
        for p in seeds:
            if p.endswith('.py'):
                for d in self._imports(p):
                    live |= resolve(d)
        for d in re.findall(r'\bservers(?:\.[A-Za-z_][A-Za-z0-9_]*)+', seed_text):
            live |= resolve(d)
        live |= {f for f in re.findall(r'\bservers/[A-Za-z0-9_/]+\.py\b', seed_text) if f in py}
        # Fixpoint over the frontier: whatever a live module imports is live.
        frontier = set(live)
        while frontier:
            grown = set()
            for m in frontier:
                grown |= edges(m)
            frontier = grown - live
            live |= grown

        orphans = sorted(py - live)
        # data files under servers/ (a .json today) follow the hooks/scripts
        # rule: named by basename or stem somewhere live
        live_text = seed_text + '\n' + '\n'.join(_read(p) for p in live)
        for d in sorted(shipped - py):
            base = os.path.basename(d)
            if base not in live_text and os.path.splitext(base)[0] not in live_text:
                orphans.append(d)
        assert not orphans, (
            'shipped servers/ files nothing imports from the entrypoints '
            '(hooks, dashboard, brain_mcp, daemon_server) — dead on every '
            f'install: delete them, relocate them beside their real consumer '
            f'(eval/, scripts/), or wire them: {orphans}')


class TestMechanismContainment:
    """Duplicated mechanisms have ONE home, and this fails when a second
    appears.

    Step 6 (DISTRIBUTION-ARCH-PLAN.md) unified four mechanisms that had drifted
    across two-to-four hand-written copies. Extraction alone does not hold that
    line: `daemon-client.sh` was extracted for exactly this reason and was
    deleted in step 5 as a zero-referrer script — callers had drifted back to
    hand-rolling. What holds is a scan that fails the moment copy #2 is
    written and names the owner in the failure message (the
    `test_no_raw_popen_outside_daemon_launch` / raw-SQL-guardrail shape).

    Each row is (mechanism, regex, owners, why) — a shape-scan over the tracked
    tree, never an enumeration of callers. Adding a legitimate second owner is
    a deliberate edit to this table, which is the point.

    Two properties every row must have, because a guard that quietly stops
    guarding is the failure this class exists to prevent:
      ARMED    at least one declared owner still MATCHES the pattern. Without
               this, renaming the thing inside its owner leaves the row green
               while it guards nothing.
      SHAPED   the pattern describes the MECHANISM, not one spelling of it. A
               regex pinned to `json.dumps({"cmd"` misses single quotes, a
               `payload` variable, and byte literals — including copies this
               very repo has actually contained.
    """

    # Rows are (mechanism, regex, owners, why) or
    # (mechanism, regex, owners, why, path_regex) when the mechanism only
    # exists in one kind of file — the launchd ritual is shell, and scanning
    # Python docstrings for it only finds operator instructions, never a
    # second implementation.
    MECHANISMS = [
        (
            'API key from the plugin userConfig option',
            r'CLAUDE_PLUGIN_OPTION_(API_KEY|api_key)',
            {'hooks/scripts/api-key-env.sh'},
            'a casing fix landing in only one copy re-creates the 2026-07-15 '
            'failure: user fills the plugin key field, daemon still runs keyless',
        ),
        (
            'launchd install/reload ritual',
            # Any launchctl install/reload VERB on a line that is neither a
            # comment nor an echo. A leading-token anchor let `cp x y &&
            # launchctl bootstrap`, `sudo launchctl bootout`, and a call inside
            # a function body all walk past.
            r'(?m)^(?![ \t]*#)(?!.*\becho\b).*\blaunchctl[ \t]+(?:bootstrap|bootout|load|unload)\b',
            {'hooks/scripts/launchd-install.sh'},
            'the copy in ensure-dashboard.sh had already lost the daemon '
            "side's post-bootstrap verification; bootout-without-verify is how "
            '"file current, launchd stale" becomes permanent',
            r'^hooks/scripts/.*\.sh$',
        ),
        (
            'plist template substitution',
            r'__PLUGIN_DIR__',
            {'hooks/scripts/launchd-install.sh',
             'hooks/scripts/com.brain.daemon.plist',
             'hooks/scripts/com.brain.dashboard.plist'},
            'a second renderer is a second chance to skip identity '
            'preservation and re-point an installed service at the wrong tree '
            'or the wrong brain',
        ),
        (
            'config-knob read (~/.config/brain/env)',
            # Both shell grammars for extracting the knob: the subshell-source
            # idiom (either quoting style) and a line-parser. Pinning only the
            # first spelling let `BRAIN_DB_DIR=""` and a grep/cut parser — a
            # second GRAMMAR, which is the hazard this row names — walk past.
            r'''BRAIN_DB_DIR=(?:''|"")|\b(?:grep|sed|awk|cut)\b[^\n]*BRAIN_DB_DIR=''',
            {'hooks/scripts/resolve-brain-db.sh'},
            'this idiom had three shell copies and one had already lost the '
            "stdout discard, so a user's `echo` in the env file polluted the "
            'resolved path',
        ),
        (
            'brain path from the plugin userConfig option',
            r'CLAUDE_PLUGIN_OPTION_(BRAIN_PATH|brain_path)',
            {'hooks/scripts/resolve-brain-db.sh'},
            'same unpinned-casing trap as the API key: a copy that checks one '
            'casing is a silent no-op for half the users',
        ),
        (
            'daemon process construction',
            r'BrainDaemon\(',
            {'servers/daemon_server.py'},
            'the boot incantation was written twice (a `-c` bootstrap and a '
            'shell heredoc) and drifted on env pinning; every spawn route now '
            'execs `python -m servers.daemon_server <db>`, so constructing the '
            'daemon anywhere else is a second incantation by definition',
        ),
        (
            'raw socket construction',
            # The mechanism is "opens a socket at all", not one way of
            # spelling the payload — a payload-shaped regex let single quotes,
            # a `payload` variable and b'{"command": "ping"}' all through,
            # including a copy this very commit deleted. Every owner below is
            # a DECIDED exception with a reason; a new file here is not.
            r'socket\.(socket|create_connection)\(',
            {
                'servers/daemon_client.py',      # the client wire, the owner
                'servers/daemon_server.py',      # the server side — binds, not connects
                'servers/daemon_launch.py',      # port-occupied probe — binds, not connects
                'scripts/smoke-lib.sh',          # dev sandbox picking an ephemeral port for
                                                 # ITS daemon — never ships, never connects
                'dashboard/daemon_client.py',    # sanctioned copy: the dashboard must
                                                 # run when servers/ is absent or broken
                'hooks/scripts/post_tool_trace.py',  # fires on EVERY tool call;
                                                 # importing servers.daemon_client costs
                                                 # ~44ms (22ms of it daemon_config's
                                                 # import-time code fingerprint)
            },
            'five hand-rolled clients had three read loops and three error '
            'vocabularies; each owner here is an exception someone reasoned '
            'about, so a sixth socket in a new file has to be argued for',
        ),
    ]

    @pytest.mark.parametrize('row', MECHANISMS, ids=[m[0] for m in MECHANISMS])
    def test_mechanism_has_one_home(self, row):
        mechanism, pattern, owners, why = row[:4]
        path_rx = re.compile(row[4]) if len(row) > 4 else None
        # tests/ is exempt: a test EXERCISES a mechanism (feeding it the env
        # var, asserting the wire bytes) without implementing a second copy of
        # it. Scanning them would make covering a mechanism trip its own gate.
        rx = re.compile(pattern)
        hits = {rel for rel in SCOPE
                if not rel.startswith('tests/')
                and (path_rx is None or path_rx.search(rel))
                and rx.search(_read(rel))}
        for owner in owners:
            assert owner in TRACKED, (
                f'{mechanism}: declared owner {owner} is not tracked — the '
                'table is naming a file that no longer exists')
        # ARMED: if no owner matches any more, the row is a no-op guarding
        # nothing, and every future copy #2 passes it silently.
        assert hits & owners, (
            f'{mechanism}: no declared owner matches {pattern!r} — this row is '
            f'DISARMED. Either the mechanism moved (update owners) or it was '
            f'renamed/removed (update the pattern or delete the row); leaving '
            f'it as-is means the next copy of it ships unnoticed.')
        strays = sorted(hits - owners)
        assert not strays, (
            f'{mechanism} is owned by {sorted(owners)} — call it, do not '
            f'copy it. Second copies found in: {strays}. Why it matters: {why}')


class TestPublicTreeExport:
    """5.1: the export script's three gates.

    Two kinds of test live here and they check different things. The sandbox
    tests plant a leak in a temp dir and assert the gate fires — they pin the
    gate's MECHANICS. `test_live_tree_exports_clean` runs the real export over
    the real repo — it pins the TREE. Both are needed: a green mechanics test
    on a red tree proves only that the alarm works while the house burns."""

    SCRIPT = os.path.join(os.path.dirname(__file__), '..',
                          'scripts', 'export-public-tree.sh')

    def _run(self, *args, timeout=60):
        # A developer shell with EXPECT_VERSION exported would turn gate C's
        # release pin into a spurious drift failure in every export test.
        env = {k: v for k, v in os.environ.items() if k != 'EXPECT_VERSION'}
        return subprocess.run(['bash', self.SCRIPT, *args], env=env,
                              capture_output=True, text=True, timeout=timeout)

    def test_manifest_list_mode_is_sane(self):
        files = _manifest()
        assert len(files) > 100, 'manifest suspiciously small'
        assert '.claude-plugin/plugin.json' in files
        leaked = [f for f in files
                  if f.startswith(('docs/', 'eval/', 'scripts/'))]
        assert not leaked, f'dev-only paths in the package manifest: {leaked}'

    def test_scrub_gate_catches_planted_leak(self, tmp_path):
        (tmp_path / 'mod.py').write_text('# see /Users/tpac/brain for setup\n')
        r = self._run('--scrub-only', str(tmp_path))
        assert r.returncode != 0, 'planted personal path must fail the gate'
        assert 'mod.py' in r.stderr

    def test_scrub_gate_allows_attribution_in_named_files_only(self, tmp_path):
        (tmp_path / 'LICENSE').write_text('Copyright (c) 2026 Tom Pachys\n')
        (tmp_path / 'README.md').write_text('Built by Tom Pachys.\n')
        assert self._run('--scrub-only', str(tmp_path)).returncode == 0
        # the same string anywhere else is a leak — the allowlist is per-file
        (tmp_path / 'other.py').write_text('# author: Tom Pachys\n')
        r = self._run('--scrub-only', str(tmp_path))
        assert r.returncode != 0
        assert 'other.py' in r.stderr

    def test_denylist_gate(self, tmp_path):
        assert self._run('--denylist-only', str(tmp_path)).returncode == 0
        (tmp_path / 'eval').mkdir()
        (tmp_path / 'eval' / 'x.py').write_text('')
        r = self._run('--denylist-only', str(tmp_path))
        assert r.returncode != 0
        assert 'eval' in r.stderr
        (tmp_path / 'eval' / 'x.py').unlink(); (tmp_path / 'eval').rmdir()
        (tmp_path / 'tests' / 'conversations').mkdir(parents=True)
        assert self._run('--denylist-only', str(tmp_path)).returncode != 0, \
            'real-session fixture dir must be denylisted'

    def test_secrets_gate(self, tmp_path):
        assert self._run('--secrets-only', str(tmp_path)).returncode == 0
        (tmp_path / 'mod.py').write_text('KEY = "sk-ant-api03-' + 'A' * 40 + '"\n')
        r = self._run('--secrets-only', str(tmp_path))
        assert r.returncode != 0 and 'mod.py' in r.stderr, r.stderr
        (tmp_path / 'mod.py').unlink()
        # files that have no business in a public tree, whatever they hold
        (tmp_path / 'scratch.db').write_bytes(b'')
        r = self._run('--secrets-only', str(tmp_path))
        assert r.returncode != 0 and 'scratch.db' in r.stderr, r.stderr

    def test_secrets_gate_ignores_short_fixture_keys(self, tmp_path):
        # the suite ships fixture keys shaped like sk-ant-test-abc123 by design;
        # the gate looks for real key lengths, not the assignment
        (tmp_path / 't.py').write_text(
            "os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-test-abc123'\n")
        assert self._run('--secrets-only', str(tmp_path)).returncode == 0

    def test_scrub_allowlist_cannot_mask_a_colocated_leak(self, tmp_path):
        # review finding: line-level subtraction hid a leak sharing a line
        # with an allowed attribution — the gate must strip only the allowed
        # pattern and re-test the remainder
        (tmp_path / 'LICENSE').write_text(
            'Copyright (c) 2026 Tom Pachys, /Users/tpac/secret\n')
        r = self._run('--scrub-only', str(tmp_path))
        assert r.returncode != 0, 'co-located leak masked by attribution'

    def test_live_tree_exports_clean(self, tmp_path):
        """THE RATCHET: the LIVE repo must export clean, not just sandboxes.

        Gate B drifted 67 → 69 within hours of being cleared, from another
        stream merging two comments nobody reviewed. Nothing stops any stream
        from writing a name into a comment, so a one-time sweep starts rotting
        the moment it lands. This is what makes cleanliness hold.

        A failure names the file and line. Fix the LINE — reword the comment,
        drop the attribution, rename the fixture. Do NOT add it to the export
        script's ALLOWLIST unless the string is genuinely shipped behaviour
        (the legacy `AgentsContext` rung), deliberate attribution (LICENSE),
        or a test that asserts ON the literal and would assert nothing without
        it. Widening the allowlist to get green defeats the gate.
        """
        out = tmp_path / 'public-tree'
        r = self._run(str(out), timeout=300)
        # Don't name the cause in the headline: the export also dies when a
        # tracked file is missing from the working tree (a half-finished `git
        # mv`), and "grew a personal-information hit" would send the reader
        # hunting for a leak that isn't there. The gate's own output says which.
        assert r.returncode == 0, (
            'the live tree no longer exports clean — the failing gate names '
            'itself below (gate B = a personal-information hit; gate A = a '
            'denylisted path; gate D = a credential shape or a forbidden file; '
            'a `cp` error = a tracked file missing on disk).\n'
            f'--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}')
        # D-8's graceful skip must not rot: eval/ is absent from this tree,
        # and a test module that reaches it without require_eval() aborts
        # the WHOLE exported suite at collection. Collecting the exported
        # tests catches every coupling shape — a module import, a bare
        # import via a sys.path insert, a helper module, an in-body path —
        # by running the thing instead of pattern-matching for it. The
        # export's conftest warns that its own venv is absent and proceeds.
        c = subprocess.run(
            [sys.executable, '-m', 'pytest', '--collect-only', '-q',
             '-p', 'no:cacheprovider', 'tests'],
            cwd=str(out), capture_output=True, text=True, timeout=120)
        assert c.returncode == 0, (
            'the exported suite does not collect — a test reaches eval/ (or '
            'another excluded path) without the D-8 graceful skip; add '
            '`from tests.eval_optional import require_eval; require_eval()` '
            f'above the import.\n--- stdout ---\n{c.stdout[-3000:]}\n'
            f'--- stderr ---\n{c.stderr[-2000:]}')

    # ── the allowlist must not be able to grow quietly ──

    @staticmethod
    def _allowlist():
        """The ALLOWLIST pairs, parsed out of the export script."""
        src = open(TestPublicTreeExport.SCRIPT, encoding='utf-8').read()
        body = src.split('ALLOWLIST=(', 1)[1].split('\n)', 1)[0]
        return [tuple(m.split(':', 1))
                for m in re.findall(r'^\s*"([^"]+)"', body, re.M)]

    # Bumping this is the point: a new allowlist entry is a deliberate,
    # reviewable line in a diff, never a quiet way to turn a red gate green.
    ALLOWLIST_SIZE = 17

    def test_allowlist_cannot_grow_quietly(self):
        """The one way to make gate B green WITHOUT fixing the leak is to add
        an allowlist entry — two lines, no review, and the leak still ships.
        Pinning the count makes that an explicit diff someone has to defend."""
        entries = self._allowlist()
        assert len(entries) == self.ALLOWLIST_SIZE, (
            f'gate B allowlist is {len(entries)} entries, pinned at '
            f'{self.ALLOWLIST_SIZE}. Adding one EXEMPTS a real string from the '
            'personal-information gate — if that is genuinely what you mean '
            '(shipped behaviour, deliberate attribution, or a test asserting ON '
            'the literal), bump this number in the same commit and say why.')

    def test_no_stale_allowlist_entries(self):
        """A stale entry is an exemption nothing is using — dead permission that
        silently covers whatever lands in that file next. Same discipline as
        test_capture_grep_pin.test_allowlist_entries_still_exist."""
        repo = os.path.join(os.path.dirname(__file__), '..')
        for rel, pat in self._allowlist():
            path = os.path.join(repo, rel)
            assert os.path.exists(path), (
                f'allowlisted file is gone: {rel} — drop the entry')
            with open(path, encoding='utf-8', errors='replace') as f:
                assert pat in f.read(), (
                    f'allowlisted pattern {pat!r} no longer appears in {rel} — '
                    'drop the entry rather than leaving a dead exemption')

    def test_gate_c_rejects_unexpected_version(self, tmp_path):
        # The release command passes the version it is releasing; agreement on
        # the wrong value must fail before anything is materialized.
        r = subprocess.run(['bash', self.SCRIPT, str(tmp_path / 'out')],
                           capture_output=True, text=True, timeout=60,
                           env={**os.environ, 'EXPECT_VERSION': '0.0.0-never'})
        assert r.returncode != 0
        assert 'gate C' in r.stderr and '0.0.0-never' in r.stderr
        assert not (tmp_path / 'out').exists(), 'gate C must fail before the copy'

    def test_export_refuses_to_clobber_foreign_dir(self, tmp_path):
        (tmp_path / 'precious.txt').write_text('mine')
        r = self._run(str(tmp_path))
        assert r.returncode != 0
        assert (tmp_path / 'precious.txt').exists()


class TestReleaseCommand:
    """5.7: the release command's REFUSALS, which are its whole value.

    Only the preflight is exercised: it runs before anything expensive and
    before the tree is staged, so a refusal here proves the door stays shut
    without paying for the export, the suite or the smoke. The commit the
    release adds — author, e-mail, messages, target URL — is the one artifact
    the export gates never see; these pin that it is gated anyway."""

    SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'release.sh')

    def _run(self, tmp_path, *args, email=None, previous='none'):
        env = {k: v for k, v in os.environ.items()
               if k not in ('RELEASE_AUTHOR_EMAIL', 'RELEASE_PREVIOUS')}
        env['RELEASE_STAGE'] = str(tmp_path / 'stage')
        if email is not None:
            env['RELEASE_AUTHOR_EMAIL'] = email
        if previous is not None:
            env['RELEASE_PREVIOUS'] = previous
        return subprocess.run(['bash', self.SCRIPT, *args], capture_output=True,
                              text=True, timeout=120, env=env)

    def test_refuses_without_explicit_author_email(self, tmp_path):
        # git's configured identity must never be the fallback — on the dev
        # machine it is a work address
        r = self._run(tmp_path, '0.9.0')
        assert r.returncode != 0
        assert 'RELEASE_AUTHOR_EMAIL' in r.stderr
        assert not (tmp_path / 'stage' / 'tree').exists(), 'refused, yet staged'

    def test_refuses_without_a_predecessor_statement(self, tmp_path):
        # the upgrade test cannot be forgotten, only declined by name
        r = self._run(tmp_path, '0.9.0', email='a@example.org', previous=None)
        assert r.returncode != 0
        assert 'RELEASE_PREVIOUS' in r.stderr
        r = self._run(tmp_path, '0.9.0', email='a@example.org',
                      previous=str(tmp_path / 'not-a-tree'))
        assert r.returncode != 0
        assert 'not a plugin tree' in r.stderr

    def test_refuses_personal_author_email(self, tmp_path):
        r = self._run(tmp_path, '0.9.0', email='someone@playbuzz.com')
        assert r.returncode != 0
        assert 'gate B' in r.stderr, r.stderr
        assert not (tmp_path / 'stage' / 'tree').exists()

    def test_refuses_malformed_version(self, tmp_path):
        r = self._run(tmp_path, '9.7', email='a@example.org')
        assert r.returncode != 0
        assert 'X.Y.Z' in r.stderr

    def test_publish_refuses_a_foreign_remote(self, tmp_path):
        # a mis-aimed push is the one way history could leak
        r = self._run(tmp_path, '0.9.0', '--publish', 'git@github.com:someone/else.git',
                      email='a@example.org')
        assert r.returncode != 0
        assert 'repository' in r.stderr and 'someone/else' in r.stderr

    def test_publish_refuses_skipped_steps(self, tmp_path):
        r = self._run(tmp_path, '0.9.0', '--publish', PLUGIN['repository'],
                      '--skip', 'suite', email='a@example.org')
        assert r.returncode != 0
        assert '--skip' in r.stderr
