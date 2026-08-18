"""Seed the interactions table on a fresh brain, and keep shipped prompts
reaching installs that never diverged from them.

**DB is authoritative at runtime.** `seed_interactions` only writes an
interaction the first time — once an entry exists (any version), it is a no-op.
S3 / operator / anchor can register new versions later via register_interaction
and those versions are what the encoders will read.

**`reconcile_seeded_prompts` closes the freeze that create-only seeding caused.**
Seeding alone meant an install captured the prompts of its install date forever:
31 commits of prompt improvement in 90 days reached only brains created after
each commit. Reconcile advances a prompt ONLY while the install is still running
the shipped default; the moment a human registers or activates anything for that
name, it is hands-off permanently. Gated by SEED_PROMPTS_VERSION so it runs once
per bump, and called from the daemon only — never from `Brain()`, which would
mutate frozen eval corpora.

The actual prompt text for the seeded LLM interactions lives in sibling files:
    servers/scales/s1/encoding_prompt.py                 (s1e)
    servers/scales/s1/surface_prompt.py                  (surface)
    servers/scales/s2/community_enrichment_prompt.py     (s2_community_enrichment)
    servers/scales/s2/consolidation_enrichment_prompt.py (s2_consolidation_enrichment)
    servers/scales/s2/healer_prompt.py                   (s2_healer)
    servers/scales/s2/aspect_prompt.py                   (s2_aspects)
    servers/recall_expansion_prompt.py                   (recall_query_expansion)

Those files are mirrored FROM the DB's latest version by:
    ./dev python3 -m servers.tools.sync_prompts

Run the sync after any register_interaction call so a fresh clone of the
repo boots with the mature prompts — not a stale v1 baseline. See
tests/test_prompt_sync.py for the contract check.

Config-only interactions (trace_recording, scopes, s2_community) have no
template files — their behavior lives mostly in code, not in a prompt.
"""
import json
import os

from .dal_logs import (AUTO_V1_PROVENANCE, BACKSTOP_PROVENANCE,
                       RECONCILE_PROVENANCE)


# Surface prompt seed lives in scales/s1/surface_prompt.py (beside its
# consumer), mirrored from the DB ACTIVE version by ./dev sync-prompts and
# shipped to the fleet via shipped_prompts() — fresh brains boot with the
# mature prompt, existing pristine installs advance on a version bump.


# S2_NODE_FAMILIES_PROMPT and S2_EDGE_FAMILIES_PROMPT — REMOVED 2026-05-04
# (Step 12 of unified-aspects). Replaced by the unified aspect taxonomy in
# aspects_v1.json, which AspectRegistry reads directly (no aspect-nodes). The
# AspectIntegration maintenance unit's prompt is the s2_aspects interaction.


# ═══════════════════════════════════════════════════════════════════════
# Parameter defaults per interaction.
#
# Each config default lives in its CONSUMER's contract file (one default home
# per interaction) and is imported here by the seed/shipped roster:
#   s1e                    → scales/s1/encode_contract.S1E_INTERACTION_DEFAULT
#   surface                → scales/s1/surface_contract.SURFACE_INTERACTION_DEFAULT
#   s1_scout_*             → scales/s1/scouts/contract.SCOUT_*_INTERACTION_DEFAULT
#   s2_community_enrichment    → scales/s2/community_contract.COMMUNITY_ENRICHMENT
#   s2_consolidation_enrichment → scales/s2/consolidation_contract.CONSOLIDATION_ENRICHMENT
#   s2_healer              → scales/s2/healer_contract.HEALER_INTERACTION_DEFAULT
#   s2_aspects             → scales/s2/aspect_contract.ASPECT_INTERACTION_DEFAULT
#   recall_query_expansion → recall_expansion_prompt.RECALL_EXPANSION_INTERACTION_DEFAULT
#   trace_recording / scopes / s2_community → already contract-owned
#     (trace_contract / scopes / community_contract)
#
# boot, voice_surface, pre_edit, signal_assembler carry NO config default and
# are no longer seeded at all: every key was grepped reader-less (pre_edit's
# plausible keys are read from brain_meta via Brain.get_config, a different
# store). Their rows on existing installs are inert history.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Shipped-prompt reconciliation.
#
# Bump SEED_PROMPTS_VERSION when a prompt or config change in this repo should
# reach installs that are still running the shipped default. The bump IS the
# deployment decision, deliberately explicit: a constant in a reviewable diff,
# not an implicit consequence of editing a prompt. Same contract BRAIN_VERSION
# has — code owns the default, each install migrates itself forward at open.
#
# tests/test_seed_prompt_reconcile.py holds a fingerprint of the shipped
# templates and configs and fails when they change without a bump, because a
# forgotten bump silently rebuilds the exact freeze this mechanism removes.
# ═══════════════════════════════════════════════════════════════════════

# Starts at 2, not 1: version 1 is BURNED. A reverted first attempt at this
# mechanism (dfc74ee, 2026-08-09) booted on at least one real install and
# stamped `seed_prompts_version = 1` before being reverted — the code went away,
# the row did not. Shipping at 1 would read as "already reconciled" on exactly
# the installs the mechanism exists for, and forward-only counters cannot reuse
# a burned number. Attempt 2's generation-1 content also differs from what
# attempt 1 stamped: this version ships `parameters` alongside the template,
# where attempt 1 carried the install's old config forward.
#
# 3 carries the facts scout's `output_schema` to the fleet. Generation 2 shipped
# the config channel but not that key, so installs stamped at 2 still run the
# one mustered scout on the free-text parsing path.
#
# 4 ships Step 4's s1e `model` key (table-driven model resolution) and brings
# `surface` into shipped_prompts() — template + layout config, v15 as the
# shipped default.
SEED_PROMPTS_VERSION = 4
SEED_PROMPTS_VERSION_KEY = 'seed_prompts_version'

# Pointer provenance that proves the install is still running what WE put there.
# A strict subset of SYSTEM_PROVENANCE (asserted in the tests): every value here
# is reserved, but not every reserved value is pristine.
PRISTINE_ACTIVATIONS = (AUTO_V1_PROVENANCE, RECONCILE_PROVENANCE)


def shipped_prompts():
    """name -> (template, config dict) for every prompt the fleet should receive.

    Scope is prompts that production actually reads. Three exclusions,
    all for the same reason — advancing content for machinery that never runs
    widens the blast radius for nothing:

    • Config-only interactions (`boot`, `trace_recording`, the signal/scope
      configs). Several are dead config with no reader at all.
    • `s1_scout_quote` and `s1_scout_temporal`. Both are still registered in
      `SCOUT_RUNNERS`, but production runs the lived arm
      (`BRAIN_S1E_LIVED_SEQUENCE=1` in hooks/scripts/brain-env.sh) and
      `encode.py` musters that arm with `exclude_scouts=('quote', 'temporal')`,
      so neither ever fires. `temporal` was retired outright. `facts` is the one
      live scout and stays. `seed_interactions` still CREATES all three so the
      interactions exist if an arm re-enables them — being seeded and being
      advanced are separate questions.
    • `recall_query_expansion` — env-gated off by default
      (BRAIN_QUERY_EXPANSION); joins this roster the day the flag defaults on.

    `surface` ships template + `layout` config (both live reads on the recall
    hot path). Its MODEL is deliberately not config — see SURFACE_MODEL in
    surface_contract.py.

    The template files are mirrored from the DB's ACTIVE version by
    `./dev sync-prompts`; the config dicts are the shipped config line,
    hand-maintained here and drift-checked by `sync-prompts --check`.
    """
    from .scales.s1.encoding_prompt import SYSTEM_PROMPT as S1E
    from .scales.s2.community_enrichment_prompt import SYSTEM_PROMPT as COMM
    from .scales.s2.consolidation_enrichment_prompt import SYSTEM_PROMPT as CONS
    from .scales.s2.healer_prompt import SYSTEM_PROMPT as HEAL
    from .scales.s2.aspect_prompt import SYSTEM_PROMPT as ASP
    from .scales.s1.scouts.prompts.facts_prompt import SYSTEM_PROMPT as SF
    from .scales.s1.surface_prompt import SYSTEM_PROMPT as SURF
    from .scales.s1.encode_contract import S1E_INTERACTION_DEFAULT
    from .scales.s1.surface_contract import SURFACE_INTERACTION_DEFAULT
    from .scales.s1.scouts.contract import SCOUT_FACTS_INTERACTION_DEFAULT
    from .scales.s2.community_contract import COMMUNITY_ENRICHMENT
    from .scales.s2.consolidation_contract import CONSOLIDATION_ENRICHMENT
    from .scales.s2.healer_contract import HEALER_INTERACTION_DEFAULT
    from .scales.s2.aspect_contract import ASPECT_INTERACTION_DEFAULT
    return {
        's1e': (S1E, S1E_INTERACTION_DEFAULT),
        'surface': (SURF, SURFACE_INTERACTION_DEFAULT),
        's2_community_enrichment': (COMM, COMMUNITY_ENRICHMENT),
        's2_consolidation_enrichment': (CONS, CONSOLIDATION_ENRICHMENT),
        's2_healer': (HEAL, HEALER_INTERACTION_DEFAULT),
        's2_aspects': (ASP, ASPECT_INTERACTION_DEFAULT),
        's1_scout_facts': (SF, SCOUT_FACTS_INTERACTION_DEFAULT),
    }


def _parse_params(raw):
    """Interaction parameters as a dict. Unparseable reads as {}."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _matches_shipped(interaction, template, config):
    """True when this version already carries the shipped template AND config.

    Both halves matter. A comparison on template alone silently drops a
    config-only update — which is the motivating case for shipping config at
    all: a frozen install keeps a dated model ID that the API will retire, and
    the mechanism built to reach the fleet cannot fix it.
    """
    if (interaction.get('template') or '') != template:
        return False
    return _parse_params(interaction.get('parameters')) == config


def _pointer_is_pristine(info):
    """True when the active pointer was placed by the system, not by a human.

    `BACKSTOP_PROVENANCE` is reserved but not pristine in general: it fills a
    missing pointer with MAX(version), which on an install predating
    `interaction_active` could be a version a human registered by hand.

    The exception is decidable. With exactly one version on record, nothing but
    the seed ever registered for that name, so MAX(version) IS the shipped
    default and the pointer carries no human decision. Those are the oldest
    installs in the fleet — the ones frozen longest, and the whole reason this
    mechanism exists. Freezing them out on an ambiguity that resolves would
    exclude precisely the population it was built to reach.
    """
    set_by = info.get('active_set_by')
    if set_by in PRISTINE_ACTIVATIONS:
        return True
    return (set_by == BACKSTOP_PROVENANCE
            and info.get('total_versions') == 1)


def _pristine_advance_target(brain, name, active_version):
    """None if a human touched this name; else a reconcile-created version
    above `active_version` that can be re-adopted, or 0 for "advance freshly".

    Pristine means the install is still running what WE put there:
      • the active pointer was set by the system (SYSTEM_PROVENANCE), and
      • every version above active was created by a previous reconcile.

    That second clause is the load-bearing one. A registered-but-inactive
    version normally means a human made a deployment decision — `trace_recording`
    sits at active=1 with a dormant v2 exactly like this — and publishing over
    it is the precise registration/activation conflation `interaction_active`
    exists to prevent: a write is not a deployment decision. The one exception
    is our own crash residue: a reconcile that registered and died before
    flipping the pointer. That is recognisable by `created_by`, and re-adopting
    it avoids stacking a duplicate version on every retry.
    """
    above = [v for v in brain.list_interaction_versions(name)
             if v['version'] > active_version]
    if any(v.get('created_by') != RECONCILE_PROVENANCE for v in above):
        return None
    return max([v['version'] for v in above], default=0)


def _reconcile_pristine_prompts(brain):
    """Advance installs still running the shipped default; never touch the rest."""
    state = {i['name']: i for i in brain.list_interactions()}
    advanced, held = [], []

    for name, (template, config) in sorted(shipped_prompts().items()):
        info = state.get(name)
        if not info:
            continue  # absent — seed_interactions owns creating it

        active_version = info.get('active_version')
        if active_version is None:
            # No pointer row at all. get_active falls back to MAX(version), so
            # the runtime is reading something nobody deployed on purpose;
            # leave it to the pointer backstop in ensure_logs_schema.
            held.append('%s(no active pointer)' % name)
            continue
        if not _pointer_is_pristine(info):
            held.append('%s(activated by %s)' % (name, info.get('active_set_by')))
            continue

        residue = _pristine_advance_target(brain, name, active_version)
        if residue is None:
            held.append('%s(v%s+ registered by a human)'
                        % (name, active_version + 1))
            continue

        if _matches_shipped(brain.get_interaction(name) or {}, template, config):
            continue  # already current — no write

        if residue:
            candidate = brain.get_interaction(name, version=residue) or {}
            if _matches_shipped(candidate, template, config):
                # Crash residue that already carries this exact content: adopt
                # it instead of registering a duplicate.
                brain.set_interaction_active(name, residue,
                                             set_by=RECONCILE_PROVENANCE)
                advanced.append('%s->v%d (adopted)' % (name, residue))
                continue

        new = brain.register_interaction(name, template=template,
                                         parameters=json.dumps(config),
                                         created_by=RECONCILE_PROVENANCE)
        brain.set_interaction_active(name, new['version'],
                                     set_by=RECONCILE_PROVENANCE)
        advanced.append('%s->v%d' % (name, new['version']))

    # Loud on both outcomes: a silent reconcile is indistinguishable from a
    # broken one, and a silent skip is how the original freeze went unnoticed.
    if advanced:
        print('[seed-reconcile] advanced shipped prompts: %s'
              % ', '.join(advanced), flush=True)
    if held:
        print('[seed-reconcile] left alone (locally owned): %s'
              % ', '.join(held), flush=True)


def reconcile_seeded_prompts(brain):
    """Version-gated entry point — runs once per SEED_PROMPTS_VERSION bump.

    Daemon-only by design. `Brain()` must never call this: eval corpora,
    IsolatedBrain copies, tests, and the daemon-dead fallback in
    hooks/scripts/boot_brain.py all construct a Brain directly, and reconciling
    there would mutate frozen corpora and race two processes on
    UNIQUE(name, version). The daemon is a singleton and runs this before it
    serves, so there is exactly one writer.
    """
    from .schema import run_versioned_migrations
    try:
        run_versioned_migrations(
            brain.logs_conn, 'logs_meta', SEED_PROMPTS_VERSION_KEY,
            SEED_PROMPTS_VERSION,
            [(SEED_PROMPTS_VERSION,
              lambda _conn: _reconcile_pristine_prompts(brain))],
            label='seed prompts')
        brain.logs_conn.commit()
    except Exception as e:
        # Never block boot on prompt reconciliation: the install keeps running
        # its current prompts, and the unstamped version retries next open.
        try:
            brain.logs_conn.rollback()
        except Exception:
            pass
        print('[seed-reconcile] WARNING: skipped (%s)' % e, flush=True)
        # stdout alone is invisible to query_logs, and this can fail silently
        # for many boots in a row — the freeze it exists to remove, wearing a
        # different mask. Every comparable boot-path failure routes here.
        try:
            brain._log_error('seed_reconcile_failed', e,
                             'seed_prompts_version=%d' % SEED_PROMPTS_VERSION)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Seed entry point — called from Brain.__init__ on boot.
# ═══════════════════════════════════════════════════════════════════════

def seed_interactions(brain):
    """Register v1 templates for any interaction not already present.

    Idempotent: skips anything the DB already knows about. Never overrides.
    """
    # Imported here so seed failures surface on boot, not at import time.
    from .scales.s1.encoding_prompt import SYSTEM_PROMPT as S1E_PROMPT
    from .scales.s2.community_enrichment_prompt import SYSTEM_PROMPT as S2_COMMUNITY_PROMPT
    from .scales.s2.consolidation_enrichment_prompt import SYSTEM_PROMPT as S2_CONSOLIDATION_PROMPT
    from .scales.s2.healer_prompt import SYSTEM_PROMPT as S2_HEALER_PROMPT
    from .scales.s2.aspect_prompt import SYSTEM_PROMPT as S2_ASPECTS_PROMPT
    from .scales.s1.scouts.prompts.quote_prompt import SYSTEM_PROMPT as S1_SCOUT_QUOTE_PROMPT
    from .scales.s1.scouts.prompts.temporal_prompt import SYSTEM_PROMPT as S1_SCOUT_TEMPORAL_PROMPT
    from .scales.s1.scouts.prompts.facts_prompt import SYSTEM_PROMPT as S1_SCOUT_FACTS_PROMPT
    from .scales.s1.surface_prompt import SYSTEM_PROMPT as SURFACE_PROMPT
    from .recall_expansion_prompt import (SYSTEM_PROMPT as RECALL_EXPANSION_PROMPT,
                                          RECALL_EXPANSION_INTERACTION_DEFAULT)
    from .scales.s1.encode_contract import S1E_INTERACTION_DEFAULT
    from .scales.s1.surface_contract import SURFACE_INTERACTION_DEFAULT
    from .scales.s1.scouts.contract import (SCOUT_QUOTE_INTERACTION_DEFAULT,
                                            SCOUT_TEMPORAL_INTERACTION_DEFAULT,
                                            SCOUT_FACTS_INTERACTION_DEFAULT)
    from .scales.s2.community_contract import COMMUNITY_ENRICHMENT
    from .scales.s2.consolidation_contract import CONSOLIDATION_ENRICHMENT
    from .scales.s2.healer_contract import HEALER_INTERACTION_DEFAULT
    from .scales.s2.aspect_contract import ASPECT_INTERACTION_DEFAULT

    existing = {i['name'] for i in brain.list_interactions()}

    def _register(name, template, params_dict, created_by):
        if name in existing:
            return
        brain.register_interaction(name,
                                   template=template,
                                   parameters=json.dumps(params_dict),
                                   created_by=created_by)

    # Encoder agents — prompts seeded from sibling .py files.
    # 's1e' is the current name (only 's1e' is seeded / read at runtime).
    # 'encoding_agent' was the legacy name; its DB rows are inert history and
    # the runtime fallback to it was removed (see s1/encode.py).
    _register('s1e', S1E_PROMPT, S1E_INTERACTION_DEFAULT, 'anchor')
    _register('s2_community_enrichment', S2_COMMUNITY_PROMPT,
              COMMUNITY_ENRICHMENT, 's2:community_detection')
    _register('s2_consolidation_enrichment', S2_CONSOLIDATION_PROMPT,
              CONSOLIDATION_ENRICHMENT, 's2:consolidation')
    _register('s2_healer', S2_HEALER_PROMPT,
              HEALER_INTERACTION_DEFAULT, 's2:healer')
    _register('s2_aspects', S2_ASPECTS_PROMPT,
              ASPECT_INTERACTION_DEFAULT, 's2:aspect_integration')

    # S1 Scouts — each is its own interaction entry. The runtime reads
    # interaction.template for the per-scout task prompt (LLM scouts only;
    # temporal is algo) and interaction.parameters.category_statement for
    # the single-line teaching S1S sees in every scout report. Learnable
    # boundary — S3 will optimize each scout independently once built.
    _register('s1_scout_quote',     S1_SCOUT_QUOTE_PROMPT,
              SCOUT_QUOTE_INTERACTION_DEFAULT,     'anchor')
    _register('s1_scout_temporal',  S1_SCOUT_TEMPORAL_PROMPT,
              SCOUT_TEMPORAL_INTERACTION_DEFAULT,  'anchor')
    _register('s1_scout_facts',     S1_SCOUT_FACTS_PROMPT,
              SCOUT_FACTS_INTERACTION_DEFAULT,     'anchor')

    # Recall-lane query expansion — read by brain_recall._expand_query_via_llm
    # when BRAIN_QUERY_EXPANSION is enabled. Registered on every install (this
    # seed runs per-name-if-missing at each boot), advanced by the fleet only
    # if it ever joins shipped_prompts() (see RECALL_EXPANSION_INTERACTION_DEFAULT
    # in recall_expansion_prompt.py).
    _register('recall_query_expansion', RECALL_EXPANSION_PROMPT,
              RECALL_EXPANSION_INTERACTION_DEFAULT, 'anchor')

    # Short-template / config-only interactions (prompts inline).
    # 'judge' was renamed to 'surface' in commit 620fb4f (2026-05-03);
    # this seed only knows about 'surface'. Old 'judge' rows in older
    # brains are orphans — clean them out manually if they exist.
    _register('surface', SURFACE_PROMPT, SURFACE_INTERACTION_DEFAULT, 'anchor')
    # Payload-recorder gates (docs/TRACE-MODES-DESIGN.md): modes as named
    # config versions — v1 normal (auto-activates), v2 debug (dormant).
    # "Entering debug" = set_interaction_active('trace_recording', 2).
    # Each registration guards on its own absence (version count, not just
    # the name) so a boot that crashed between the two self-heals on the
    # next seed instead of losing the debug version forever; >= 2 versions
    # (including externally-registered ones) → never add more. Deliberate
    # edge: a single OPERATOR-registered v1 gets the contract debug appended
    # as v2 — an ADD, never an override (active pointer untouched), and it
    # keeps the documented recipe (activate v2 = enter debug) coherent.
    from .trace_contract import (TRACE_RECORDING_DEBUG,
                                 TRACE_RECORDING_NORMAL)
    _tr_versions = next((i['total_versions'] for i in brain.list_interactions()
                         if i['name'] == 'trace_recording'), 0)
    if _tr_versions == 0:
        _register('trace_recording', '', TRACE_RECORDING_NORMAL, 'anchor')
    if _tr_versions < 2:
        brain.register_interaction('trace_recording', template='',
                                   parameters=json.dumps(TRACE_RECORDING_DEBUG),
                                   created_by='anchor')
    # boot / voice_surface / pre_edit / signal_assembler — NOT seeded: every
    # config key was grepped reader-less (see the defaults note at the top of
    # this file), and a row with no template and no config serves nobody.
    # Rows on existing installs are inert history (collapse policy: ADOPT).
    # Scope policy (servers/scopes.py) — config-only. Per-dimension modes
    # (open/scoped/isolated) + per-value overrides; 'scoped' everywhere is
    # the behavior-neutral default (the LAF lane is unfitted, isolation is
    # opt-in). Edit via register_interaction + set_interaction_active.
    from .scopes import SCOPES_CONFIG_V1
    _register('scopes', '', SCOPES_CONFIG_V1, 'anchor')
    # s2_community config knob (distinct from enrichment prompt — this is
    # decoder parameters, not an LLM template).
    from .scales.s2.community_contract import COMMUNITY_DETECTION
    _register('s2_community', '', COMMUNITY_DETECTION, 's2:community_detection')

    # s2_edge_families and s2_node_families seeds — REMOVED 2026-05-04
    # (Step 12 of unified-aspects). Replaced by the unified aspect taxonomy in
    # servers/scales/s2/aspects_v1.json, which AspectRegistry reads directly at
    # Brain.__init__ and AspectIntegration maintains. (The one-shot
    # scripts/migrate_to_aspects.py bridge and servers/aspect_migration.py were
    # retired 2026-05-29 — the live registry reads JSON, never aspect-nodes, so
    # the migration's node output was inert.)
