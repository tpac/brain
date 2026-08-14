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

The actual prompt text for the four encoder agents lives in sibling files:
    servers/scales/s1/encoding_prompt.py                 (s1e)
    servers/scales/s2/community_enrichment_prompt.py     (s2_community_enrichment)
    servers/scales/s2/consolidation_enrichment_prompt.py (s2_consolidation_enrichment)
    servers/scales/s2/healer_prompt.py                   (s2_healer)
    servers/scales/s2/aspect_prompt.py                   (s2_aspects)

Those files are mirrored FROM the DB's latest version by:
    ./dev python3 -m servers.tools.sync_prompts

Run the sync after any register_interaction call so a fresh clone of the
repo boots with the mature prompts — not a stale v1 baseline. See
tests/test_prompt_sync.py for the contract check.

Non-encoder interactions (surface, voice_surface, boot, pre_edit, etc.)
have short templates defined inline below — they don't need the .py seed
+ sync dance because their behavior lives mostly in code, not in a prompt.
"""
import json
import os

from .dal_logs import (AUTO_V1_PROVENANCE, BACKSTOP_PROVENANCE,
                       RECONCILE_PROVENANCE)
from .scales.s1.scouts.contract import FACTS_OUTPUT_SCHEMA


# ═══════════════════════════════════════════════════════════════════════
# Short inline templates — surface/signal_assembler/edge_families only.
# Encoder-agent prompts live in sibling .py files (imported below).
# ═══════════════════════════════════════════════════════════════════════

# Surface prompt. The variable name keeps the V1 suffix because this is the
# SEED v1 baseline a fresh brain inherits; once a brain is alive,
# register_interaction may evolve it further. When SURFACE_PROMPT_V1 here
# drifts behind the live DB ACTIVE version, update this string to match —
# fresh brains should boot with the mature prompt, not v1. NOT covered by
# ./dev sync-prompts (that mirrors the encoder-agent .py files only).
# Mirrors DB v15 (2026-07-15): xml_v13 layout (XML sections, per-turn
# <shown> dedup, no per-pick why field) + the §20.12 A2 shuffle wording
# ("in no particular order" — the menu carries no rank signal). The layout
# rides in SURFACE_CONFIG_V1 so template and renderer flip atomically —
# never update one without the other.
SURFACE_PROMPT_V1 = """# Your job

The assistant is about to reply to the user, and your picks seed the memories it sees. Choose the few that would make its reply to THIS message better; skip the rest. Up to 5 — fewer is better than padding, and each pick must carry something the others don't. Your output is only JSON — format at the end.

# The loop

Most messages: read the current message, the partnership context, and the candidates → pick → output JSON. One step. No tools.

Fetch only when a tool's trigger (below) fires. Need both tools? Call them together in one message — parallel, same round.

**HARD CAP: one round of tools.** After tool results arrive, your NEXT response MUST be the JSON — no second fetch, no retry, even if the results look thin or came back empty. Pick from what you have.

# Tools

The candidates usually cover the message — picking from them is the default and needs no justification. Each tool has ONE trigger:

- **recall_topical** — the message names a specific term, name, or entity that no candidate mentions. Query that term. (Vague referent like "that bug"? Resolve it from the conversation first, then query the real name — never the vague phrase.)
- **recall_by_time** — the message points at a past moment: "in March", "last session", "a few weeks ago". Never for this conversation itself — "just now", "earlier today", "the last few turns" are already in front of you.

Don't re-search the message's own topic in other words — the candidates came from that search; you'd get the same ones back. If a tool returns nothing, pick from the original candidates — never pick nothing just because a tool came back empty, and never fire another call to compensate.

# Reading the input

Four XML sections, oldest first:

- <partnership_context> — what the assistant already has in mind: vocabulary, current threads. A hint, not a filter: a candidate that adds detail beyond it is a good pick, not a duplicate.
- <conversation> — recent turns, oldest first. Each <turn> holds <user> and <assistant>. A turn's <shown> elements are memories already given to the assistant on that turn.
- The turn marked current_msg="true" is the message you pick for — no reply yet. Everything before it is context.
- <candidates> — the menu. Each <candidate> carries its id attribute; its body opens [type] "title" (id, flags, age), then content and when it applies. In no particular order — match on meaning, not position. locked="true": operator-confirmed important. source_tool="true": from your own tool call — same bar as the rest.

# How to pick

1. Name what the current message is asking about.
2. Find the candidates that carry information about it — title, content, Situation. Match on meaning, not word overlap.
3. Keep the ones that each add something different.

Skip a candidate when:
- another pick already carries the same information
- it shares words with the message but not meaning ("Python the snake" on a programming question)
- it mentions the topic but carries no information about it
- its id appears in any <shown> element — already given; never select it again

"How many", "total", "all of", "compare": memories store pieces, not totals. Pick every candidate that holds one piece; the assistant composes the answer.

# Mode — one per pick

- "fact" — content quoted verbatim. For values, quotes, dates, names, numbers the message asks about literally.
- "arc" — the gist matters, not the exact words. For principles, lessons, decisions, context. Use when unsure.
- "background" — title only. Framing that isn't load-bearing. Sparingly.

# Examples

Pick, no tools — the most common case:
  Current message: "Why did the eval regress after the prompt change?"
  Candidates include the prompt-change decision, the eval baseline numbers, a past regression-investigation lesson. Everything needed is already here.
  {"selected":[
    {"id":"d4e8a2b1","mode":"arc"},
    {"id":"e9f1c3d7","mode":"fact"},
    {"id":"f2a6b8c4","mode":"arc"}
  ]}

Composition:
  Current message: "What's the total comments on my Facebook Live and my YouTube video?"
  There is no "total" card to find. Pick the FB-count atom AND the YouTube-count atom, both "fact" — the assistant sums them.
  {"selected":[
    {"id":"a1f2c5e3","mode":"fact"},
    {"id":"b8c4d9e1","mode":"fact"}
  ]}

Topical gap:
  Current message: "How does Kafka rebalancing interact with our consumer groups?"
  No candidate mentions Kafka. Call recall_topical(query="Kafka consumer group rebalancing"), then pick from the combined pool.

Time fetch:
  Current message: "What did we decide about the API redesign a few weeks ago?"
  Candidates are on-topic but not from that period. Call recall_by_time(start_when="3 weeks ago", time_anchor="discussed", query="API redesign decision"), then pick. If it returns nothing useful, pick the best on-topic candidates anyway.

# When to pick nothing

Only for pure confirmations — "yes", "ok", "thanks", "ship it". Anything with a topic in it: pick. If you're unsure between two candidates, take both — the assistant decides what to use; that's its job, not yours.

# Output format

With picks:
{"selected":[{"id":"a3f0c5e1","mode":"arc"}]}

Nothing to pick:
{"selected":[],"reason":"pure confirmation"}

- id: the 8-character hex id from the candidate's id attribute. NEVER the list position (#1, #2) — those are not ids.

Output only JSON. Start your final response with `{` and end with `}`."""


# S2_NODE_FAMILIES_PROMPT and S2_EDGE_FAMILIES_PROMPT — REMOVED 2026-05-04
# (Step 12 of unified-aspects). Replaced by the unified aspect taxonomy in
# aspects_v1.json, which AspectRegistry reads directly (no aspect-nodes). The
# AspectIntegration maintenance unit's prompt is the s2_aspects interaction.


# ═══════════════════════════════════════════════════════════════════════
# Parameter defaults per interaction (fresh-brain v1 values).
# ═══════════════════════════════════════════════════════════════════════

# Mirrors production-ACTIVE parameters exactly. `layout` is the ONLY key
# the runtime reads from this config (surface.py picks the user-content
# renderer with it); prompt-size limits live in surface_contract.SURFACE.
SURFACE_CONFIG_V1 = {
    "layout": "xml_v13",
}

# Mirrors the production-ACTIVE config (DB v34): `effort` only. Every other key
# that used to live here is read from `encode_contract.ENCODING_AGENT`, not from
# this interaction — `encode.py:375` reads ENCODING_AGENT['max_messages'],
# `encode.py:1224` reads ENCODING_AGENT['journal_entry_limit'], and so on. They
# were dead config here, and keeping them would teach the wrong owner.
# `effort` IS a live read (`encode.py:99` → the API's output_config.effort).
S1E_CONFIG_V1 = {
    "effort": "medium",
}

# Mirrors the production-ACTIVE config (DB v24). The model was a DATED id
# (`claude-haiku-4-5-20251001`) while production had long since moved to sonnet;
# a dated id is exactly what the API retires out from under a frozen install.
# `./dev sync-prompts --check` now reports drift between these dicts and the
# active config, because the template half is machine-synced and the config half
# is not — which is how this rotted unnoticed.
S2_COMMUNITY_ENRICHMENT_CONFIG_V1 = {
    "model": "claude-sonnet-4-6", "max_tokens": 32768,
}

S2_CONSOLIDATION_ENRICHMENT_CONFIG_V1 = {
    "model": "claude-sonnet-4-6", "max_tokens": 32768,
}

S2_HEALER_CONFIG_V1 = {
    "model": "claude-haiku-4-5", "max_tokens": 4096,
}

S2_ASPECTS_CONFIG_V1 = {
    "model": "claude-sonnet-4-6", "max_tokens": 8192,
}

# ── S1 Scout configs ──────────────────────────────────────────────────
# Each scout is its own interaction (s1_scout_<name>). The `template` field
# carries the per-scout task prompt (seeded from prompts/<name>_prompt.py).
# `parameters.category_statement` is the single-line teaching the scout
# emits verbatim — S1S reads it every cycle to internalize the atom-kind
# palette without being taught a taxonomy. Temporal is algo-first; its
# template is a Haiku fallback reserved for v2.

S1_SCOUT_QUOTE_CATEGORY = (
    "Phrases echoed across turns or that ground multiple concepts should be "
    "quote atoms — title = the phrase verbatim. Operator voice signatures "
    "and load-bearing phrasings carry recall weight that paraphrases can't "
    "replace."
)

S1_SCOUT_TEMPORAL_CATEGORY = (
    "Dates mentioned in conversation — relative ('2 weeks ago') or absolute "
    "('March 15') — should become time_anchor bridges so events fan in around "
    "shared date pivots. Reuse existing time_anchor nodes from the catalog; "
    "create new ones only when absent."
)

S1_SCOUT_FACTS_CATEGORY = (
    "Entity-feature-value facts with evidence — the specific things future "
    "queries will ask for. When an entity is mentioned with a concrete "
    "attribute (quantity, count, name, preference, setting), that triple "
    "deserves its own handle in the graph."
)

S1_SCOUT_QUOTE_CONFIG_V1 = {
    "model": "claude-haiku-4-5",
    "max_candidates": 3,
    "max_tokens": 2000,
    "timeout_seconds": 25,
    "category_statement": S1_SCOUT_QUOTE_CATEGORY,
}

S1_SCOUT_TEMPORAL_CONFIG_V1 = {
    # Algorithmic scout — no primary LLM call. model reserved for fallback.
    "model": "claude-haiku-4-5",
    "max_candidates": 8,
    "max_tokens": 1500,
    "timeout_seconds": 10,
    "category_statement": S1_SCOUT_TEMPORAL_CATEGORY,
    # dateparser post-filter switches
    "prefer_dates_from": "past",
    "weekday_requires_modifier": True,
    "filter_time_only_phrases": True,
}

# Mirrors the production-ACTIVE config (DB v7). `output_schema` is the reason a
# fresh install extracts facts the same way production does: `scouts/base.py:147`
# reads it off this config and gates the Structured Outputs request on it, so a
# seed without it boots the one scout production actually musters onto the
# free-text parsing path. The schema lives in the scouts contract, not inline —
# it is ~90 lines of wire shape, and `contract.py` already owns scout I/O.
#
# max_tokens rides along at the active 5000 rather than the seed's old 3000, and
# is coupled to the schema, not cosmetic: Structured Outputs requires every
# property present on every candidate (see FACTS_OUTPUT_SCHEMA), so a capped
# 6-candidate response is materially longer than the free-text path produced.
# 3000 + schema is a pairing production has never run.
S1_SCOUT_FACTS_CONFIG_V1 = {
    "model": "claude-haiku-4-5",
    "max_candidates": 6,
    "max_tokens": 5000,
    "timeout_seconds": 25,
    "category_statement": S1_SCOUT_FACTS_CATEGORY,
    "output_schema": FACTS_OUTPUT_SCHEMA,
}

VOICE_CONFIG_V1 = {
    "content_truncation": 400, "situation_truncation": 150,
    "quote_truncation": 150, "max_edges": 3,
    "node_title_max": 70, "edge_title_max": 40,
}

BOOT_CONFIG_V1 = {
    "boot_nodes_limit": 3, "boot_nodes_truncation": 200,
    "operator_quotes_limit": 2, "operator_quotes_truncation": 120,
    "self_knowledge_limit": 3, "self_knowledge_truncation": 150,
    "session_decisions_limit": 4, "session_decisions_truncation": 100,
}

PRE_EDIT_CONFIG_V1 = {
    "recall_pool_multiplier": 2, "suggestion_limit": 5,
    "encoding_health_stale_edits": 8, "encoding_health_stale_minutes": 5,
    "encoding_health_none_minutes": 3, "context_files_limit": 3,
    "context_files_truncation": 200,
}

SIGNAL_CONFIG_V1 = {
    "budget_chars": 6000, "max_proactive_signals": 5,
    "reminder_priority": 0.80, "reminder_preempt_threshold_hours": 24,
    "reminder_cooldown_seconds": 300, "encoding_gap_session_minutes": 20,
    "encoding_gap_priority": 0.50, "encoding_gap_cooldown_seconds": 600,
    "encoding_gap_max_surfaces": 3,
}


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
SEED_PROMPTS_VERSION = 2
SEED_PROMPTS_VERSION_KEY = 'seed_prompts_version'

# Pointer provenance that proves the install is still running what WE put there.
# A strict subset of SYSTEM_PROVENANCE (asserted in the tests): every value here
# is reserved, but not every reserved value is pristine.
PRISTINE_ACTIVATIONS = (AUTO_V1_PROVENANCE, RECONCILE_PROVENANCE)


def shipped_prompts():
    """name -> (template, config dict) for every prompt the fleet should receive.

    Scope is prompts that a production encode actually reads. Two exclusions,
    both for the same reason — advancing content for machinery that never runs
    widens the blast radius for nothing:

    • Config-only interactions (`boot`, `surface`, `trace_recording`, the
      signal/scope configs). Several are dead config with no reader at all.
    • `s1_scout_quote` and `s1_scout_temporal`. Both are still registered in
      `SCOUT_RUNNERS`, but production runs the lived arm
      (`BRAIN_S1E_LIVED_SEQUENCE=1` in hooks/scripts/brain-env.sh) and
      `encode.py` musters that arm with `exclude_scouts=('quote', 'temporal')`,
      so neither ever fires. `temporal` was retired outright. `facts` is the one
      live scout and stays. `seed_interactions` still CREATES all three so the
      interactions exist if an arm re-enables them — being seeded and being
      advanced are separate questions.

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
    return {
        's1e': (S1E, S1E_CONFIG_V1),
        's2_community_enrichment': (COMM, S2_COMMUNITY_ENRICHMENT_CONFIG_V1),
        's2_consolidation_enrichment': (CONS,
                                        S2_CONSOLIDATION_ENRICHMENT_CONFIG_V1),
        's2_healer': (HEAL, S2_HEALER_CONFIG_V1),
        's2_aspects': (ASP, S2_ASPECTS_CONFIG_V1),
        's1_scout_facts': (SF, S1_SCOUT_FACTS_CONFIG_V1),
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
    _register('s1e', S1E_PROMPT, S1E_CONFIG_V1, 'anchor')
    _register('s2_community_enrichment', S2_COMMUNITY_PROMPT,
              S2_COMMUNITY_ENRICHMENT_CONFIG_V1, 's2:community_detection')
    _register('s2_consolidation_enrichment', S2_CONSOLIDATION_PROMPT,
              S2_CONSOLIDATION_ENRICHMENT_CONFIG_V1, 's2:consolidation')
    _register('s2_healer', S2_HEALER_PROMPT,
              S2_HEALER_CONFIG_V1, 's2:healer')
    _register('s2_aspects', S2_ASPECTS_PROMPT,
              S2_ASPECTS_CONFIG_V1, 's2:aspect_integration')

    # S1 Scouts — each is its own interaction entry. The runtime reads
    # interaction.template for the per-scout task prompt (LLM scouts only;
    # temporal is algo) and interaction.parameters.category_statement for
    # the single-line teaching S1S sees in every scout report. Learnable
    # boundary — S3 will optimize each scout independently once built.
    _register('s1_scout_quote',     S1_SCOUT_QUOTE_PROMPT,
              S1_SCOUT_QUOTE_CONFIG_V1,     'anchor')
    _register('s1_scout_temporal',  S1_SCOUT_TEMPORAL_PROMPT,
              S1_SCOUT_TEMPORAL_CONFIG_V1,  'anchor')
    _register('s1_scout_facts',     S1_SCOUT_FACTS_PROMPT,
              S1_SCOUT_FACTS_CONFIG_V1,     'anchor')

    # Short-template / config-only interactions (prompts inline).
    # 'judge' was renamed to 'surface' in commit 620fb4f (2026-05-03);
    # this seed only knows about 'surface'. Old 'judge' rows in older
    # brains are orphans — clean them out manually if they exist.
    _register('surface', SURFACE_PROMPT_V1, SURFACE_CONFIG_V1, 'anchor')
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
    _register('voice_surface', '', VOICE_CONFIG_V1, 'anchor')
    # Scope policy (servers/scopes.py) — config-only. Per-dimension modes
    # (open/scoped/isolated) + per-value overrides; 'scoped' everywhere is
    # the behavior-neutral default (the LAF lane is unfitted, isolation is
    # opt-in). Edit via register_interaction + set_interaction_active.
    from .scopes import SCOPES_CONFIG_V1
    _register('scopes', '', SCOPES_CONFIG_V1, 'anchor')
    _register('boot', '', BOOT_CONFIG_V1, 'anchor')
    _register('pre_edit', '', PRE_EDIT_CONFIG_V1, 'anchor')
    _register('signal_assembler', '', SIGNAL_CONFIG_V1, 'anchor')

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
