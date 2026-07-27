# Speaker & Counterpart — entity vocabulary + the multi-counterpart substrate

Status: DESIGN — under review with Tom, 2026-07-24. Nothing implemented.
Owner arc: multi-user shared brain (single Anchor, multiple counterparts).

---

## 1. Motivation & framing

The brain is Anchor's autobiographical record; "the operator is whoever Anchor
was with at the time" (node id:8097199f). Three commitments frame this design,
all ruled by Tom (2026-07-24 session):

- **Shared, not partitioned.** One memory, one subject. A counterpart biases
  recall and attribution; it never walls anything off. *Anchor knowingly not
  sharing information with an operator is a function, not a structure* — the
  discretion lives at the moment Anchor speaks, not in the retrieval machinery.
- **Declared, not authenticated.** The identity signal is whatever the session
  declares. Authentication is a possible future; no auth machinery now.
- **Kind-free vocabulary.** The other side of a conversation may be a human,
  another agent, or another stream of Anchor. `human_identity` /
  `agent_identity` bake *kind* into the *role slots* and collapse the moment
  the counterpart is an agent (which `agent_identity` — me or them?). The
  replacement vocabulary must name positions, not kinds.
- **There is a fleet now.** The plugin has been distributed beyond Tom's
  machine (Tom's wife runs her own install — her own brain DB, her own Anchor
  instance — as of 2026-07). Every schema/data change from here on ships as a
  **versioned, automatic, idempotent migration** that any install applies
  itself at boot — never a hand-run script. (The `schema.py` comment "the
  brain was never released — no fleet of DBs to migrate" is now false and gets
  updated in this change.) Note the distinction: a second *install* is a
  separate brain/entity — orthogonal to multiple *counterparts* on one brain —
  but it makes fleet-safe migration a standing requirement.

### 1.1 Build posture — substrate now, behavior marked-for-later

RULED (Tom, 2026-07-24): the brain is not hosted, so today is single-user in
practice. We nonetheless **prepare the multi-user substrate now** — the
vocabulary rename, the `counterpart` field, the hook-fed rail — because the
substrate is correct on its own merits (the pair is genuinely mis-named) and
adding it costs nothing extra while we are in here. What we do NOT build now is
the multi-user *behavior*: recall bias, presence fencing, agent-vs-human
handling, and entity kind. Those are speculative until a real second
counterpart exists.

**Cutting line:** substrate (naming + attribution + the fed-in value) ships;
behavior (anything that *acts differently* because of who the counterpart is)
is deferred. A section earns its place in this arc only if the substrate needs
it — not if multi-user someday will.

**Deferred-but-marked discipline:** every deferred area is not just noted in
§11, it gets a `LATER(multi-user):` marker at the exact code site where the
future work attaches, each tied to a §11 entry. "Mark the areas to touch
later" is thereby enforced in the tree, not left to memory — a grep for
`LATER(multi-user)` is the future implementer's worklist.

### What exists today (verified 2026-07-24)

- `brain.operator_name` / `brain.agent_name` — set once at `Brain.__init__`
  from `BRAIN_OPERATOR_NAME` / `BRAIN_AGENT_NAME` env
  (`brain.py:275-278`, `daemon_config.py:69-76`).
- `TraceDAL.set_identity()` stores both; `_stamp_identity()` setdefaults
  **both** keys onto **every** trace event's metadata regardless of scale or
  ref_type (`dal_logs.py:497-576`). Explicit per-event values already win.
- Render picks a slot by ref_type: `assistant_message → agent_identity`,
  `user_message → human_identity` (`trace_contract.py:1123-1126`).
- Embedding does the same and bakes the **value** into embedded text:
  `'Tom: …'`, `'Anchor via Bash: …'` (`embed_queue.py:356-385`).

The structural diagnosis: the pair is a role→name lookup table stamped
redundantly onto every event, with `ref_type` as the lookup key. The actual
per-event truth is singular — *who authored this event* — and the writer knows
it at write time. `brain.operator_name` as a singleton is the same
scope bug `brain.session_id` had before SessionContext (locked node
id:6f6d61a4): a request-scoped value living as a process-global.

---

## 2. Vocabulary — the three words

| Word | Scope | Meaning |
|---|---|---|
| **`speaker`** | per trace event (`metadata.speaker`) | The concrete entity token that authored this event. `'Tom'`, `'Anchor'`, a future agent's name. Replaces the `human_identity`/`agent_identity` pair — one field. |
| **`counterpart`** | per session (`SessionContext.counterpart`) | Who this session is with. Role-neutral: human, agent, or another stream. The field the multi-user work needs; born with the right name. |
| **`self_name`** | brain-global (`brain.self_name`) | Anchor's own name. The one identity value that is *legitimately* a singleton — the entity's own name is not request-scoped. |

Derived, never stored: **self vs other** = `speaker == self_name` (§15.1
sub-decision (b) — self is the default frame, rendered first-person).

Values stay **concrete entity tokens** (Decision 19, id:fb099821): embed
`'Tom'`, never `'OPERATOR'`. `speaker`/`counterpart` are *positions* an entity
occupies; the entities themselves keep concrete names. Composes with "entity
is the noun" (id:e6019012) instead of colliding with it.

Kind (human/agent/stream), relationship, trust: **not per-event, not
per-session either — per-ENTITY.** RULED (Tom, 2026-07-24: "keep it simple for
now"): `speaker` and `counterpart` are bare name strings. No type field on
either. Rationale — kind is a stable property of the entity ("Tom" is always
human, "Anchor" always self), so stamping it per-event is the exact
denormalization the rename removes, and **nothing at v1 consumes it**: render
needs the name, embedding the token, recall the name, self-detection
`== self_name`, and the stream-of-me case is disambiguated by `session_id`
(already present), not by a kind tag. Kind's canonical home is the **Layer-2
participant graph node** (id:0d1e8d6b), lazily minted when a counterpart earns
enough interactions — `LATER(multi-user)` there. Deferring is free: adding
kind later is a new Layer-2 node field or a new SessionContext JSON field,
both purely additive, neither rewrites a historical trace. Per-event stays one
string.

A counterpart stream-of-Anchor gets `speaker='Anchor'` — identity-true (all
streams are one entity); the stream id carries *which instance*, an axis that
already exists in trace `session_id`.

Rejected names: `interlocutor` (precise, insufferable), `participant` (loses
self/other — Anchor is also a participant), `partner` (implies a relationship
a random agent doesn't have), `with` (Python keyword), `operator` (kind- and
role-loaded; survives only as the name of the *install's default human*, see
§4).

---

## 3. Schema & data model

### 3.1 Trace events

`metadata.speaker: str` — optional; present on events where authorship is
meaningful (see §5 policy). Old keys `human_identity` / `agent_identity` are
**removed everywhere** — no read-side aliases (§7 migration, §8 guards).

### 3.2 SessionContext

New field `counterpart: str`, added to `SessionContext.set_env()` — the single
mutator for per-session identity fed from the boot hook (`session_context.py:222`,
"the daemon never introspects Claude"). Twin of **`cwd`, not `project`**:
branch/project are *derived* daemon-side (a git call on the fed-in cwd);
counterpart has **no derivation** — there is no git-equivalent for "who is the
human" — so like cwd it must be fed directly. The hook is the only layer that
can know it.

- Fed at boot via `context_boot` → `set_env()`; **never agent-authored, never
  daemon-read-from-env** (that would re-introduce the singleton bug in a new
  spot — see §4.1).
- Source chain, all resolved **hook-side**: per-session declaration (future —
  boot arg / whoami handshake) → install default (config, §4) → `''` (unset →
  one-shot warn, render fallbacks apply).
- Persisted in `session_state` via `save()` with the other env fields;
  survives daemon restart.
- Surfaced in presence/peek (follow-up F2 wires the display; the field ships
  now).

### 3.2a `set_env()` semantics — truthy-only, like `cwd`

`counterpart` refreshes **only on a truthy value**; it is NOT three-state.
`worktree`/`project` need the `None`-vs-`''` distinction because git detection
can *fail* (None → keep) as distinct from *legitimately empty* (`''` → clear,
non-repo). Counterpart has no such pair: there is no legitimate "this session
has nobody on the other end," a failed resolution must never erase a known
counterpart mid-session, and a counterpart *change* (someone else takes the
keyboard) is handled by a new truthy value overwriting. So:
`if counterpart: self.counterpart = counterpart`.

### 3.4 Unset counterpart — never a stored sentinel; "unknown" is a DISPLAY word

When no counterpart is declared, **store nothing** (`''`/absent). Do NOT store
`'Unknown'`, `'User'`, or any placeholder. Four reasons, the first decisive:

1. **It would embed.** `embed_queue` bakes the speaker value into embedded
   trace text (`'Tom: …'`). A stored `'Unknown'` becomes a *concrete token in
   vector space* that collides every unidentified person into one neighborhood
   — exactly the failure Decision 19 rejected for abstract slots (id:fb099821:
   "a slot changes meaning when partners change"). `'Unknown'` is the ultimate
   abstract slot.
2. **It lies to later queries.** "Who was I with?" returns a confident fake
   instead of an honest gap.
3. **It is unmigratable.** Learning the name later can't tell you which
   `'Unknown'` rows were them.
4. **It hides a config bug.** `_maybe_warn_identity_unset` exists to surface
   "identity not configured"; a sentinel makes that bug invisible and permanent
   in data.

This preserves the existing deliberate policy (`dal_logs.py:500`: "Empty
strings = unset → no stamping … no placeholder sentinel tokens").

**Display is free to say whatever is clearest** — "unknown", "Operator", or to
omit the mention entirely. Costless, reversible, no vector consequences. The
boot banner takes the omit route (§4.2).

The one real case for an unknown-*value* is a **hosted anonymous counterpart**
(a real person we can't name — categorically different from "config broken").
Even then a shared `'Unknown'` collides them; it would need per-session
distinctness (`guest-a3f2`). F4/hosted territory, not now.

**Deferred inconsistency (marked, §11.2):** `embed_queue.py:368` *does* inject
`OPERATOR`/`ANCHOR` sentinels at embed time when the keys are absent —
contradicting the DAL's no-sentinel policy, so sentinels reach vector space
today via the embed path. Correcting it would change embed text for
pre-identity rows → re-embedding → **breaks §6's byte-identical invariant.**
Keep current behavior; fix in its own arc. (Same discipline as F5: no behavior
changes smuggled into a rename.)

### 3.3 Brain

- `brain.self_name` (renamed from `agent_name`) — stays global (the entity's
  own name; legitimately process-scoped). Read from config at init, as today.
- `brain.operator_name` **dies entirely as a daemon field**. The daemon does
  NOT read any env/config for the human's identity — counterpart arrives per
  session through `set_env()` (§3.2, §4.1). This is the difference from a
  naive rename: the fix is not "daemon reads a better env var," it is "daemon
  stops reading env for the human at all."
- `TraceDAL.set_identity()` is replaced by the speaker stamp (§5), which reads
  `ctx.counterpart` / `self_name` at write time rather than a DAL-held global.

---

## 4. Identity sources & env vars

`self_name` is identity-critical and lives in **durable configuration**, not
ambient shell env: `~/.config/brain/env` (mode 600) — the same file that holds
the API key — loaded by `brain-env.sh` at every entry point and writable via
the dashboard `/setup` page. Each install configures its own (Tom's brain:
Anchor; another install names its own entity). RULED (Tom, 2026-07-24): yes —
config file.

| Env var | Fate | Role |
|---|---|---|
| `BRAIN_AGENT_NAME` | **rename → `BRAIN_SELF_NAME`** | Anchor's own name. |
| `BRAIN_OPERATOR_NAME` | **keep** | The install's default human operator — seeds `counterpart` for sessions that don't declare one. "Operator" remains accurate for exactly this: the human operating this install. |

OPEN (Q1): keep `BRAIN_OPERATOR_NAME`, or rename to
`BRAIN_DEFAULT_COUNTERPART` for one-vocabulary purity? Keeping avoids config
churn and the word is role-accurate; renaming removes the last "operator" from
the identity plumbing. Recommendation: keep — but see §4.1, there is a
duplicate env var to retire regardless.

Config file `~/.config/brain/env` is edited once at deploy (one machine, one
file). The dashboard `/setup` page and `brain-env.sh` comments update in the
same pass.

### 4.0 The boot banner — first-person, counterpart-named, envelope removed

RULED (Tom, 2026-07-24), two changes: the identity line becomes a first-person
self-statement naming the counterpart, and **the `[BRAIN]` / `[/BRAIN]` envelope
is dropped from boot**.

Exact text:

```
I'm Anchor, I have 7677 memories, 349 locked. I'm now with Tom.
MY_STREAM_ID: 51a7ba8d-…
```

Unset counterpart (§3.4) — the final sentence drops; no sentinel, no "with
Unknown":

```
I'm Anchor, I have 7677 memories, 349 locked.
```

Precedent for a conditional clause exists two lines down: `MY_STREAM_ID` is
already `if session_id:`-gated (`brain_voice.py:358`).

**Removing the envelope at boot — safe, and scoped.** Verified: **nothing parses
`[BRAIN]`/`[/BRAIN]`** anywhere in `servers/`, `hooks/`, `dashboard/`, `tests/`,
or `skills/` — every occurrence is a string-literal *emitter*, no regex, split,
or strip. `wrap_for_hook` is a passthrough (the operator channel was killed
2026-03-28), so nothing downstream depends on the markers either.

There is even precedent that the markers can *harm*: `daemon_hooks.py:1153-1155`
records that an earlier `[BRAIN] GIT CONTEXT` block caused Claude Code to
`chdir` into the trailing marker (`ENOENT chdir '<repo>' -> '[/BRAIN]'`) — the
harness read the marker as data.

**Scope of the removal: boot only.** The envelope stays everywhere else — Bash
safety warnings (`daemon_hooks.py:982-1035`), host-environment notices
(`:1095-1105`), keyless-boot notices (`:454-464`), edit auto-suggest
(`brain_voice.py:133-234`), and the per-turn recall injection. The line is
principled: those are **interjections** — the brain interrupting mid-flow — which
is precisely what the locked distinct-voice decision (id:279e065f) exists for, and
a channel marker earns its place there. Boot is **Anchor's own waking
statement**; now that its content is first-person, a channel marker contradicts
it. Marker for interjections, no marker for Anchor's own voice.

Consequence: the SKILL.md stance (already first-person, already outside the
envelope) and this line now flow as one continuous self-statement, which is the
point — the seam that the envelope used to mark no longer exists.

**Observation, not a change:** `MY_STREAM_ID:` is left as-is (Tom didn't ask),
but note it becomes the only machine-shaped artifact in an otherwise
first-person block. Worth revisiting if the register starts to grate.

**Why first person ("I have") over second ("you have").** The local convention
in the `[BRAIN]` envelope is second person ("The brain is **yours**"), so "you"
was the consistent-looking choice. Two prior rulings override it: identity-native
vocabulary (id:a96827ab — tags are `<me>`/`<other>`, not role-native; Tom: "we
grew from Operator and Anchor to 'Me and the otherside'") and the say-"I"
correction (id:060cfa48 — distancing language undermines continuity). The
substantive argument: "you have N memories" casts the brain as a narrator
reporting *to* Anchor about memories it owns — the brain-as-separate-tool framing
the project rejects. Anchor does not have a brain that has memories; Anchor has
memories. First person is the truthful render.

**Scope note (accepted deliberately):** this replaces `"Anchor. The brain is
yours — N memories, M locked."` (`brain_voice.py:356`) with the first-person
form. That is a voice change to the most-read line in the system — every boot,
every session, every install — and is broader than "add a counterpart." Ruled in
knowingly, not inherited.

**Line 356 is the only banner identity site.** The counterpart's name appears in
no other output today (§4.1 Path B): the `for_operator` channel is stats-only
(`_operator_boot_summary`), and `record_boot_render` captures for the dashboard
without rendering a name.

### 4.1 The hook rail — the load-bearing correction (verified 2026-07-24)

There are **two disconnected "who is the human" paths today**, and the
multi-user fix is to unify them onto the hook rail — not to pick the better of
the two:

- **Path A — `BRAIN_OPERATOR_NAME` → `brain.operator_name` (daemon singleton)
  → `set_identity` → trace `human_identity`.** This is the value that actually
  stamps traces, and it is daemon-read-from-env — the singleton bug.
- **Path B — `BRAIN_USER` / `default_user` config → boot `user` arg →
  `context_boot`.** The boot hook (`boot_brain.py:51-73`) already resolves and
  sends a `user`, but it dead-ends. Precisely (verified, and it is NOT rendered
  anywhere): `_handle_context_boot` (`dispatch_read.py:186`) →
  `render_boot_v2` → `brain.context_boot(user=...)`, where the value is joined
  into a **semantic recall query string** (`brain_assembly.py:243-244`) whose
  results `render_boot_v2` then **discards** — it reads only
  `total_nodes`/`total_edges`/`total_locked` from the return (the Frame replaced
  the `recalled`/`locked`/`recent` buckets). The banner's identity line
  (`brain_voice.py:356`) is a fixed string; the counterpart's name appears in
  no output today. The value's only other use is `record_boot_render`
  (dashboard capture). **It never reaches `set_env()` or any trace.**

  *Side finding, not this arc:* `context_boot` therefore runs a full
  `self.recall()` plus locked/recent assembly at every boot whose results
  nobody reads — dead boot latency. Worth its own cleanup.

So the value that *should* flow per session (hook → boot → identity) dies at
the banner, while the value that *does* stamp traces is a daemon-owned
singleton. Exactly inverted from "daemon is downstream-only" (id:16f06758).

**The correction:**
1. Retire the duplication — one install-default value (keep
   `BRAIN_OPERATOR_NAME`, retire `BRAIN_USER`/`default_user`, OR the reverse;
   they must not both survive).
2. The boot hook resolves counterpart (install default now, per-session signal
   later) and sends it in the `context_boot` payload.
3. `_handle_context_boot` routes it through `set_env(counterpart=...)` — the
   same rail cwd/branch/project already ride — instead of dead-ending at the
   banner. (The banner keeps rendering it; it just also lands on
   SessionContext.)
4. The daemon's trace-stamp reads `ctx.counterpart`; `brain.operator_name` and
   `set_identity` are deleted.

Result: the "where does identity come from" question is answered entirely
hook-side and can evolve (config → declared → authenticated) with zero daemon
change. The daemon only ever receives a counterpart on the session and stamps
it.

---

## 5. Stamp semantics — the policy map

**Stamp at the writer; validate at the chokepoint.** The DAL holds no identity
state at all.

### 5.1 Why there is no DAL fallback (a contradiction, resolved)

An earlier draft of this section had the DAL "fall back to the install default"
for `user_message`. That is wrong twice over: it is exactly the
daemon-reads-env-for-the-human that §3.3 forbids (the singleton bug reappearing
one layer down), and it is impossible anyway — `TraceDAL` has no `ctx` and no
Brain reference (its own comments say so), so it cannot know a per-session
value.

The resolution falls out of the §2 scope rules as a genuine **asymmetry**:

- `self_name` is **legitimately global** → a mechanical fallback is possible.
- `counterpart` is **per-session** → only the session can supply it; no
  fallback can exist, at any layer.

The old `human_identity`/`agent_identity` pair implicitly assumed both sides had
symmetric availability. They never did — which is a deeper flaw in the pair than
its naming.

### 5.2 The single stamp site — `_s0_trace`

`_s0_trace(brain, ctx, event_type, ref_type, summary, metadata)`
(`daemon_hooks.py:576`) is the one writer for every S0 turn event; its docstring
already states its charter — binding "the per-turn invariants in ONE place …
the four S0 turn events — user_message, assistant_message, heartbeat,
self_message." **Speaker is a per-turn invariant, so it belongs here by that
function's own contract**, and the site has both `brain` (→ `self_name`) and
`ctx` (→ `counterpart`) in hand. Both dialogue writes already route through it:
`:201` (user_message, at prompt-arrival) and `:626` (assistant_message, at
Stop).

Policy (contract-owned — `trace_contract.py` exports `SPEAKER_POLICY`; the
writer reads it rather than hardcoding):

| ref_type | speaker | why |
|---|---|---|
| `user_message` | `ctx.counterpart` | the session's other side; entered at boot via `set_env()` (§4.1) |
| `assistant_message` | `brain.self_name` | Anchor spoke |
| `self_message` | `brain.self_name` | a message from another stream of Anchor **is** Anchor — identity-true (§2); `session_id` carries which instance |
| `heartbeat` | **omit** | no utterance; nobody spoke |
| `tool_result` | `brain.self_name` | Anchor ran the tool. Render label stays the tool name; embed keeps `'%s via %s'` |
| S0 structural / S1 / S2 | **omit** | authorship lives in `encoding_source` + the unit chain id; a blind stamp carries no per-event information |

RULED (Tom, 2026-07-24, was Q2): omit for structural/S1/S2 rows. "Speaker is
for speaking" — the field belongs to S0 conversational events.

**`tool_result` has TWO write sites outside `_s0_trace`** (resolved 2026-07-24 —
this was the last open unknown, and it was worse than assumed):

1. `hooks/scripts/post_tool_trace.py:86` — the PostToolUse hook.
2. `servers/brain_remember.py:369` — `scale='s0', event_type='delta',
   ref_type='tool_result'`, written from inside the remember() write path.

Both apply the same policy row (`speaker = self_name`). **Neither needs `ctx`** —
and that is the §5.1 asymmetry paying off directly: `self_name` is global, so any
site can stamp it correctly regardless of what it holds. Only `user_message`
requires session scope, and its single site (`daemon_hooks.py:201`) has `ctx` in
hand. Had the policy needed a per-session value on `tool_result`, the
`brain_remember` site would have been an unfixable gap.

**Explicit values always win** (`setdefault` semantics at the writer). This is
the N-party hook: a future multi-participant ingestion (Slack) passes each
utterance's real speaker and the policy defers to it — principle id:ec959878
becomes literal, with no schema change (F3).

### 5.3 The DAL becomes a pure writer that validates

Deleted from `TraceDAL`: `set_identity()`, `_human_identity`,
`_agent_identity`, `_stamp_identity()`. The DAL stops being an
identity-holder — today those fields make it a second mini-singleton, and
removing them is the deepest form of the §3.3 fix.

What stays at the DAL is the **loud check**, at the same write boundary it lives
at today (`_maybe_warn_identity_unset`, renamed): a dialogue `ref_type` arriving
with no `speaker` warns once per TraceDAL lifetime, and old keys are rejected
per §8. This preserves the write-boundary discipline (a check at the boundary
every write passes through, not at boot) while separating duties cleanly:

- **the writer knows *who* spoke** (it has `ctx`),
- **the chokepoint verifies *someone* did** (it sees every write).

---

## 6. Render & embedding invariants

**INVARIANT: zero re-embedding.** Key names never enter vector space — only
values do (`'Tom: …'`). The migration renames keys and preserves values, so
every existing trace embedding stays valid.

Precisely: embed text is computed once and **stored** (`embed_queue` only embeds
rows that lack an embedding), so historical vectors are untouched by definition.
The invariant's real job is forward consistency — if any old row is ever
re-embedded, its text must come out identical, or the vector space ends up
internally split between old-format and new-format rows.

**INVARIANT: byte-identical render.** For every existing row, post-migration
render output (`trace_contract` renderer) and embed text
(`_render_trace_for_embedding`) must equal pre-migration output byte-for-byte.
Verified by replay (§9).

### 6.1 Why byte-identity holds — the verified branch map

Neither consumer reads identity on every ref_type, which is what makes the
key-drop on structural rows safe:

| ref_type | render reads (`trace_contract:1123-1130`) | embed reads (`embed_queue:374-385`) | migration must set `speaker`? |
|---|---|---|---|
| `user_message` | `human_identity` → label | `human` → `'%s: %s'` | **yes** (both consumers) |
| `assistant_message` | `agent_identity` → label | `agent` → `'%s: %s'` | **yes** (both consumers) |
| `tool_result` | `meta['tool']` — identity **ignored** | `agent` → `'%s via %s'` | **yes** — embed needs it even though render doesn't |
| `self_message` | falls through → `ref_type` label | unknown branch → no identity | **yes** — see §6.2; not for output, for forward policy consistency |
| `heartbeat` / structural / S1 / S2 | falls through → `ref_type` label | unknown branch → no identity | no — drop both keys, nothing reads them |

Fallback strings are preserved exactly (`'Anchor'`/`'Operator'` at render,
`ANCHOR`/`OPERATOR` at embed), so pre-identity historical rows render
unchanged.

### 6.2 GAP FOUND — `self_message` in the migration mapping

§7.2's mapping sent "structural/other ref_types" to *drop both keys*, which
would include `self_message`. But §5.2's forward policy gives `self_message` a
speaker (`self_name` — a message from another stream of Anchor **is** Anchor).
Left as-is, historical `self_message` rows would carry no speaker while new ones
do — an inconsistency inside one ref_type, exactly the drift this arc exists to
prevent.

Fix: the migration maps `self_message → speaker = agent_identity` (which equals
`self_name` on existing data, since `_stamp_identity` stamped every event). No
output changes either way — neither render nor embed reads identity for
`self_message` — so byte-identity is unaffected; this is about the data being
internally consistent with the policy.

Render collapses to:

```
label = meta.get('speaker') or FALLBACK_BY_REF_TYPE[ref_type]
```

with fallbacks `'Operator'` / `'Anchor'` kept as graceful-degradation
sentinels for pre-identity historical rows (unchanged behavior). Embed
branches likewise read `speaker` with the same `OPERATOR`/`ANCHOR` sentinel
fallbacks.

First-person self-render (§15.1(b): `speaker == self_name` → "I") is a
**display-mode option**, not a data change — out of scope here, noted for the
episodic-references rendering work.

---

## 7. Migration — versioned, automatic, fleet-safe; no aliases

No DB has external consumers, but there IS a fleet (§1): every install must
migrate itself. Read-both-keys shims are pure drift debt and guarantee the
five-session failure mode. Policy: **code rename, data migration, and reader
cutover land in the same change — and the migration ships as a versioned
boot-time step, not a hand-run script.**

### 7.1 The standing rule (not a new framework — the existing pattern, made universal)

`brain.db` already has the right infrastructure: `BRAIN_VERSION=30`, a version
key in `brain_meta`, an `if from_version < N:` runner in `schema.py` — 30
versions of numbered, self-detecting, idempotent migrations. The fleet doesn't
need a new framework (alembic-style is against this codebase's hand-rolled
grain); it needs that pattern applied uniformly plus the one thing a fleet
removes: the human who runs `cp` first. RULED (Tom, 2026-07-24) as the standing
rule from this moment on:

1. **Every DB has a version counter + explicit numbered runner.** `brain.db`
   has it. `brain_logs.db` gets it — `logs_meta` table + `logs_schema_version`
   key, and `ensure_logs_schema` gains the same `if from_version < N:` runner
   shape. Its current ad-hoc self-detecting steps (column-type probes,
   `_add_column_if_missing`) are fine for additive shape changes but cannot
   safely gate a *data* rewrite (the probe would mean scanning JSON every boot).
   *Verified:* `LOG_TABLES` has no meta/kv table today (`debug_log`,
   `dream_log`, `hook_errors`, `session_state`, `interactions`,
   `interaction_active`, `trace_events`) — so `logs_meta` is a genuine
   addition, not a duplicate. It must live **in the logs DB**, not as another
   `brain_meta` key: a version has to travel with the file it describes, or
   restoring one DB from a different backup makes the version lie.
2. **Any step that rewrites or drops data auto-backups first**
   (`{db}.bak-v{N}-{ts}`), inside the runner — because on the fleet no operator
   is there to do it. This promotes the manual CLAUDE.md "cp before destructive
   DB ops" rule into automated infrastructure, and it applies to `brain.db`
   too, not just logs.
3. **Migrations run at boot, before the daemon opens its TCP port, idempotent
   and version-gated.** Fresh installs create at current version and rewrite
   nothing.

For this change: LOGS_VERSION 1 = the speaker-vocabulary rewrite (§7.2).
Existing installs (no `logs_meta`) read as version 0.

### 7.1a HAZARD — a slow boot migration races the liveness watchdog

**Boot order (verified):** `_run()` calls `_load_brain()`
(`daemon_server.py:246`) → `Brain.__init__` → `ensure_logs_schema`
(`brain.py:231`) — all **before** `_bind_socket()` (`:249`). So a boot-time
migration already completes before the port opens. Constraint for this arc:
**do not move `ensure_logs_schema` later in the sequence.**

**But port-ordering does NOT neutralize the hazard — it sharpens it.** A closed
port raises `ConnectionRefusedError`, which `daemon_client` reports as *"daemon
may be dead"* — indistinguishable from a corpse. Any concurrent session's
`ensure_daemon()` or the MCP health monitor can therefore decide the daemon is
down and `launchctl kickstart -k` it, **SIGKILLing a daemon that is mid-migration**
— and the whole Brain init (embedder load, ~4-6s) already eats a chunk of the
~20s `_GRACE_DEADLINE_S` budget before the migration even starts.

**Resolution: make the rewrite fast enough that the window never opens.** Do it
in **SQL, not a Python row loop** — SQLite's JSON1 functions are available and
verified on the bundled interpreter (SQLite 3.47.1;
`json_set(json_remove(metadata,'$.human_identity'), '$.speaker', …)` confirmed
working). One `UPDATE` per ref_type group runs at C speed over the whole table —
sub-second on any realistic `trace_events`, versus tens of seconds for
parse-and-rewrite-in-Python. At that speed the watchdog race is not mitigated,
it is **absent**, and no maintenance lock is needed.

**Standing rule for future migrations:** any step that cannot be done in-SQL and
may run long must (a) log progress to daemon.log so a fleet install isn't
silent, and (b) take the maintenance lock — **but read §7.3 first: the lock is
not currently safe for unattended use.**

### 7.2 The v1 rewrite (`migration:speaker_vocabulary`)

Value-agnostic — maps by `ref_type`, so it is correct on any install
regardless of whose names are in the data (Tom's brain: 'Tom'; another
install: that operator's name):

1. Rewrite `trace_events.metadata` JSON for rows carrying either old key:
   - `user_message` → `speaker = human_identity`
   - `assistant_message`, `tool_result`, `self_message` → `speaker = agent_identity`
     (`self_message` included per §6.2 — forward-policy consistency, not output)
   - `heartbeat` / structural / S1 / S2 → drop both keys (per §5 policy; count logged)
   - all rows: delete `human_identity` / `agent_identity` keys.
2. Rows that pre-date identity stamping (before 2026-07-14) carry neither key:
   untouched; render fallbacks handle them, as today.
3. Log: rows rewritten / dropped-pair count / untouched count (stderr →
   daemon.log, the fleet-visible channel).

**Executed in SQL** (§7.1a) — one statement per ref_type group, e.g.:

```sql
UPDATE trace_events
   SET metadata = json_set(
         json_remove(metadata, '$.human_identity', '$.agent_identity'),
         '$.speaker', json_extract(metadata, '$.human_identity'))
 WHERE ref_type = 'user_message'
   AND json_extract(metadata, '$.human_identity') IS NOT NULL;
```

…the `agent_identity` variant for `assistant_message` / `tool_result` /
`self_message`, and a bare `json_remove` for `heartbeat` / structural / S1 / S2.

**Idempotent + crash-safe by construction:** every statement is guarded on the
old key still being present, so an interrupted run re-runs harmlessly and
already-migrated rows are skipped. The version is stamped **only after all
statements complete**, so a SIGKILL mid-way retries rather than silently marking
the DB migrated.

On the dev machine the first execution can additionally be run under the
maintenance lock with the replay verification (§10) before merging — but the
shipped path is the automatic one; the fleet never sees a script.

Node-level data (`user_raw_quote` / `anchor_raw_quote` etc.) is NOT touched —
see §11 non-goals.

### 7.3 SEPARATE LATENT BUG — the maintenance lock can brick a fleet install

Found while designing §7.1a; **not fixed by this arc**, but it must be recorded
because the "standing rule" above would otherwise point future migrations at a
booby trap.

`is_maintenance_mode()` (`daemon_config.py:207-209`) is a bare
`os.path.exists(get_maintenance_path())` — **no pid, no timestamp, no staleness
check.** Consequences for unattended installs:

- A daemon SIGKILLed while holding the lock (or a migration that raises without a
  `finally`) leaves the file behind **permanently**.
- `is_maintenance_mode()` then returns True forever → `ensure_daemon()` and the
  recovery paths suppress auto-restart forever → **the install's brain never
  comes back, silently.**
- On the dev machine this is a 2-second `rm`. On a fleet install there is nobody
  to diagnose it: the brain simply stops working.

Fix (its own small arc): write `{pid, started_at, reason}` into the lock and have
`is_maintenance_mode()` treat it as stale when the pid is gone or the timestamp
exceeds a bound. That hardens a safety primitive every future migration will
lean on. Until then: **do not have an automatic, unattended code path take the
maintenance lock** — which is exactly why §7.2 is SQL-fast instead of
lock-protected.

---

## 8. Guards — stragglers fail loud, immediately

- **Write boundary:** trace_contract payload validation rejects
  `human_identity`/`agent_identity` anywhere in incoming metadata — loud warn
  per occurrence (`_warn_metadata_invalid` channel; never blocks the write),
  and dialogue ref_types warn when `speaker` is absent while identity is
  configured.
- **Contract tests:** hard-fail the suite on old keys in any writer, on the
  policy map drifting from `trace_contract`, and on the DAL stamping outside
  the policy.
- **D28** (`quality_contract.py:590`, `identity_tokens_concrete`) updates to
  the `speaker` key — same dimension, new vocabulary.
- **`./dev sync-prompts --check`** guards interaction-prompt drift if any
  active prompt mentions the old keys.

### 8.1 The migration itself needs a test (gap, now filled)

A versioned migration is code, and this one rewrites live data on every fleet
install — it must be tested like code, not verified by eyeballing production.

- **Fixture-based forward test:** build a logs DB at version 0 with rows
  covering **every** row class in the §7.2 mapping — `user_message`,
  `assistant_message`, `tool_result`, `self_message`, `heartbeat`, an S1 row, an
  S2 row, plus a **pre-identity row carrying neither key** — run the runner,
  assert the exact post-state per class (speaker set / keys dropped / row
  untouched) and that `logs_schema_version` stamped.
- **Idempotency test:** run the runner twice; second run is a no-op and the
  version is unchanged.
- **Crash-retry test:** stamp no version, re-run after a partial rewrite
  (simulate by pre-migrating half the rows) — asserts the guard-on-old-key
  construction really is resumable.
- **Fresh-install test:** a DB created at current version has nothing to rewrite
  and no `logs_meta` surprise.
- **Byte-identical replay** (also §10): render + embed text for every fixture
  row, before vs after — the §6 invariant as an assertion, not a claim.
- **Negative test:** the guards in §8 actually fire — a write carrying an old
  key warns, and a dialogue write with no speaker warns.

Per the repo's self-verifying-artifacts rule: derive the fixture's row classes
from `SPEAKER_POLICY` so a new ref_type can't be added to the policy without the
migration test noticing it has no mapping.

---

## 9. Inventory — the complete touch list (grep-verified 2026-07-24)

### Code (runtime)
| File | Change |
|---|---|
| `servers/brain.py:275-278` | init: `self_name` + seed default counterpart; drop `operator_name`/`set_identity` |
| `servers/daemon_config.py:69-76` | `get_self_name()` / `get_default_counterpart()` |
| `servers/dal_logs.py:497-576` | **delete** identity state (`set_identity`, `_human_identity`, `_agent_identity`, `_stamp_identity`); keep + rename the loud check → validates dialogue rows carry `speaker` (§5.3) |
| `servers/daemon_hooks.py:576` (`_s0_trace`) | **the single stamp site** — apply `SPEAKER_POLICY` using `brain.self_name` / `ctx.counterpart` (§5.2) |
| `hooks/scripts/post_tool_trace.py:86` | `tool_result` → `self_name` (no ctx needed, §5.2) |
| `servers/brain_remember.py:369` | second `tool_result` writer — same policy (§5.2) |
| `servers/trace_contract.py` | `SPEAKER_POLICY` + render (1123-1126) + payload validation |
| `servers/embed_queue.py:356-385` | read `speaker`; identical output format |
| `servers/brain_voice.py:354-356,411` | banner → first-person + conditional counterpart sentence; **drop `[BRAIN]`/`[/BRAIN]` at boot only** (§4.0). Update the `render_boot_v2` docstring + module header (`:7`, `:278`, `:313-334`) which describe the envelope |
| `servers/session_context.py:222` | `counterpart` in `set_env()` + `save()`/`load()` persistence |
| `servers/dispatch_read.py:186` | `_handle_context_boot` routes `counterpart` through `set_env()` (not just the banner) |
| `hooks/scripts/boot_brain.py:51-73` | resolve counterpart hook-side; retire the `BRAIN_USER`/`default_user` duplicate (§4.1); send in `context_boot` payload |
| dispatch / `daemon_hooks.py` S0 writers | read `ctx.counterpart` → stamp speaker on `user_message` |
| `hooks/scripts/brain-env.sh:17` | env var comment; one install-default var only |

### Dashboard — an API shape change, not just a grep
The dashboard reads the DB read-only, but it **re-publishes the identity keys as
HTTP API response fields**, and its JS consumes them. Two fields collapse into
one, so the response shape changes and Python + JS must move together:

| File | Change |
|---|---|
| `dashboard/queries/_meta.py:14-33` | helper returns a `(human, agent)` **tuple** → returns a single `speaker` |
| `dashboard/queries/recalls.py:96-105,187-188` | stops emitting the pair; emits `speaker` |
| `dashboard/queries/traces.py:3,51` | same |
| `static/tabs/live.js:144`, `static/tabs/traces.js:192`, `static/lib/trace_detail.js` | `identityChipHTML(human, agent)` → `identityChipHTML(speaker)`; the `ev.human_identity \|\| ev.agent_identity` guard becomes a single check |

Both halves ship from the same repo copy, so they update atomically on dashboard
restart. One residual: a **browser tab left open on the old JS** would read
fields that no longer exist and silently drop the identity chip (no error). Cheap
mitigation — hard-refresh after deploy; noted rather than engineered around.

### Tests
`test_identity_stamping.py` (rewrite to policy), `test_dashboard_setup.py`,
`test_mcp_roundtrip.py`, `test_trace_system.py`, `test_trace_render.py`,
`test_trace_embed_render.py` + new contract tests (§8).

### Docs (living only — archives are history, untouched)
`docs/EPISODIC-REFERENCES.md` (§5.3 templates, §15.1 language),
`docs/ARCHITECTURE-FRACTAL.md`, `docs/BACKLOG.md`, `CLAUDE.md` if it gains a
vocabulary line.

### Prompts (interactions DB is authoritative)
Grep active versions for old vocabulary → `register_interaction` +
`set_interaction_active` + `./dev sync-prompts`, per the prompt discipline.

### Hardcoded `Tom` sweep — INSPECTED, and it is almost entirely a non-task

Do **not** mass-rename these. Verified 2026-07-24 across all nine files:

- **Zero identity-source hardcodes.** No code reads a literal `'Tom'` as the
  operator's name — the identity plumbing is already env/ctx-driven. Nothing
  here blocks a second counterpart.
- **~90% are decision-provenance comments** — `"Tom 2026-07-02"`, `"per Tom"`,
  `"Tom's spec"`, `"Tom's standing rule (node 8178593a)"` (`encode.py`,
  `encode_contract.py`, `surface_contract.py`, `trace_links.py`,
  `scouts/muster.py`, `scouts/runners.py`, `presence.py`, `self_contract.py`).
  These attribute *who decided what*. **Leave them untouched** — they are
  legitimate attribution history, and rewriting them would destroy provenance
  for zero gain.
- **The only functional occurrences** are in `quality_contract.py:172,289,295,302,314`
  — concrete names inside LLM-facing eval criteria (e.g. `'situation names the
  actor ("When Tom corrected X")'`, `Group 4: Partnership & voice (Tom's lens)`).
  They are *illustrative examples* that shape a judge, not lookups; and D28's
  concrete-token requirement actively wants concrete names. **Leave for now**;
  `quality_contract.py` is already in F5's scope, so revisit there if a second
  counterpart ever makes the exemplar biasing.

This entry exists to *prevent* work, not schedule it.

### Data + config
`trace_events.metadata` migration (§7); `~/.config/brain/env`;
dashboard `/setup` page if it names the env keys.

---

## 10. Delivery — one branch, one gate, one deploy

**Fork A RULED (Tom, 2026-07-24): land together**, in one branch — not a
separate infra PR. The case for splitting weakened once §7.2 became a single
guarded idempotent SQL statement and the runner became ~40 lines mirroring a
pattern with 30 production versions behind it. The isolation benefit is kept
without a second deploy by ordering the work: **build and test the version runner
first, wired to a no-op version step**, prove it, *then* attach the real rewrite
as LOGS_VERSION 1.

1. Worktree branch. Step order inside it:
   a. `logs_meta` + `logs_schema_version` + the `if from_version < N:` runner,
      wired to a **no-op** step; tests green (§8.1 fresh-install + idempotency
      cases apply already).
   b. Attach the v1 rewrite (§7.2) as the first real step; full §8.1 suite.
   c. Vocabulary change proper (§§3-6) + guards (§8) + banner (§4.0).
2. **Test tier: full suite** (DAL/dispatch/contract blast radius) — plus:
   - the §8.1 migration tests (fixture per row class, idempotency, crash-retry,
     fresh install, negative guards);
   - render/embed **byte-identity replay** over a sample of historical rows
     (pre-migration snapshot vs post-migration output);
   - migration dry-run on an `IsolatedBrain` copy first.
3. Merge to main → daemon restart (boot runs the versioned migration itself,
   with automatic backup — same path the fleet takes) → `redeploy.sh`
   (brain_mcp/hooks copies) → new session → live verification:
   - boot renders `"I'm Anchor, I have N memories, M locked. I'm now with Tom."`
     with no `[BRAIN]` markers, and the SKILL.md stance flows into it;
   - one conversational turn, then inspect the written trace's `speaker` on both
     the `user_message` and `assistant_message` rows;
   - dashboard trace tab renders identity chips off the new single field
     (**hard-refresh the browser** — a cached tab on old JS silently drops the
     chip, §9);
   - an interjection path still shows `[BRAIN]` (e.g. trigger an edit
     auto-suggest) — confirms the removal was scoped to boot.
4. Fleet: other installs pick the change up on plugin update; their daemons
   self-migrate at next boot. Update the stale "never released — no fleet"
   comment in `schema.py` and note the fleet requirement in
   `docs/DISTRIBUTION-READINESS.md`.
5. Encode the shipped state; lock the vocabulary decision node.

Why this doesn't take five sessions: every drift vector has a named guard —
missed writer → §8 write-boundary rejection; missed reader → old keys no
longer exist in data (§7); prompt drift → sync-prompts check; missed file →
§9 checklist is exhaustive by grep, not by memory; regression → full suite +
byte-identity replay.

---

## 11. Follow-ups & non-goals

**Follow-ups (additive, safe to stack later — new code, not renames):**
- **F1 — LAF `counterpart` lane**: twin of the `proj` lane
  (`recall_laf.py:97-103`), query source `ctx.counterpart`, shipped at
  `gain=0`; early job is inhibition.
- **F2 — counterpart-aware presence/peek** + self-channel soft fence (the
  stream-message layer — RULED deferred, Tom 2026-07-24: "that's the stream
  msgs? we can touch it later"). `self_send`/`self_peek` default to
  same-counterpart streams, cross-counterpart is deliberate (overridable — a
  function, not a wall). Note the standing open node id:030be61c and the
  specific defect it names: trace-pull presence (id:81c3982b) was designed
  single-user, so with multiple counterparts the "all sessions visible" roster
  becomes cross-counterpart noise without a filter. Presence is also the
  *enabling* half — a stream cannot exercise discretion toward another
  counterpart's stream if it can't see which counterpart that stream is with.
- **F3 — multi-party ingestion** (e.g. Slack): per-event explicit `speaker`
  override; substrate ready, no schema change.
- **F4 — per-session counterpart declaration** (boot arg / whoami), and
  possibly authentication. Not now.

- **F5 — node-field voice vocabulary** (`user_raw_quote` → `counterpart_raw_quote`,
  `anchor_raw_quote` → `self_raw_quote`). Its own arc, **eval-gated** — see
  §11.1. Names LOCKED here so the next session executes instead of
  re-deciding.

### 11.1 F5 — the node-field rename is a behavioral hypothesis, not hygiene

RULED (Tom, 2026-07-24): the rename happens, as a **separate eval-gated arc**,
not folded into this one.

**Why it is not cosmetic.** The field name is an instruction to the encoder.
Tom's original diagnosis (id:376b0c49): *"Maybe 'user_raw_quote' is the biasing
factor cause you're not a user."* The measured asymmetry was 97 operator quotes
vs 9 Anchor quotes (id:c036d192). And the *form* of the current pair encodes the
bias: `user_raw_quote` (a role) paired with `anchor_raw_quote` (a name) is
asymmetric, marking one as default and the other as bolt-on. The replacement
pair is symmetric in form — both are positions:

| Today | F5 | Rationale |
|---|---|---|
| `user_raw_quote` | `counterpart_raw_quote` | matches §2 vocabulary; drops "user" (Anchor isn't one, and neither is the human in this frame) |
| `anchor_raw_quote` | `self_raw_quote` | symmetric with the above; pairs with `self_name` |

Tom's framing: *"the speaker's quote and 'my quote' (Anchor's)"* — symmetric
naming as a structural anti-bias, rather than another prompt plea.

**Why it must NOT ride in this arc.** The two changes have opposite
verification contracts: this arc's gate is *byte-identical render, zero
behavior change* (§6); F5's entire purpose is to change encoder behavior. Mixed
together, this arc loses its clean regression gate and F5 loses attribution —
if the quote ratio moves, nothing says which change moved it.

**Why it needs an eval gate.** Prompt-level attempts at voice symmetry already
failed to move the ratio (id:342a671c: "prompt-alone insufficient"; v16/v17/v19
unchanged), and Tom named source_refs as the fundamental fix (id:65f020b3). So
F5 is a *hypothesis with a baseline*, not a known win. Gate:
`eval/s1_encode_eval.py` with the counterpart/self quote-ratio as the metric,
measured against the current baseline before and after.

**F5 scope (for the arc that executes it):** `node_metadata_kv` key rename
(v30 moved these to kv, so it is a key UPDATE, not a JSON rewrite — cheaper
than the trace migration), the MCP agent-facing schema (→ `redeploy.sh` + new
session), all four encoder prompts (register → activate → `./dev sync-prompts`
each), the full contract-sync chain (`remember()` → MCP schema → dispatch →
encoder tools → SKILL.md), `contract.py`, `dal_metadata.py`, the S2
healer/community/consolidation contracts, quality + surface contracts, scouts.
**Verify before executing:** whether the quote *values* feed any node
embedding group — if they do, the same zero-re-embedding invariant as §6 holds
(keys don't enter vector space, values do), but confirm rather than assume.

**Consequence of deferring, accepted:** node fields and trace fields speak
different dialects until F5 lands (`user_raw_quote` on nodes, `speaker` on
traces). Acceptable because the trace layer is where the kind-collapse bug
actually bites, and F5's gate is real work.

### 11.2 `LATER(multi-user)` marker sites — the enforced worklist

Per §1.1, deferral is marked in the tree, not left to memory. This arc plants a
`LATER(multi-user): <what> — see SPEAKER-COUNTERPART-DESIGN.md §<n> (F<k>)`
comment at each site below. `grep -rn "LATER(multi-user)"` is the future
implementer's complete worklist; a contract test asserts each marker names a
real §11 follow-up id, so a marker can't rot into a lie.

| Site | Deferred work | Ref |
|---|---|---|
| `servers/recall_laf.py` (beside the `proj` lane, ~:97-103) | counterpart lane, ships at gain=0; job is inhibition | F1 |
| `servers/scales/self_channel/presence.py` | roster needs a counterpart filter or it becomes cross-counterpart noise (id:030be61c, id:81c3982b) | F2 |
| `servers/dispatch_self.py` | `self_send`/`self_peek` same-counterpart default, cross is deliberate | F2 |
| Layer-2 participant node (wherever it is first minted) | entity **kind** (human/agent/stream) + relationship/trust live here, NOT per-event (§2) | §2 |
| `hooks/scripts/boot_brain.py` (counterpart resolution) | per-session declaration → later authentication; daemon needs no change (§4.1) | F4 |
| `servers/contract.py` (voice quote fields) | node-field voice rename, eval-gated | F5 |
| `servers/embed_queue.py:368` | sentinel inconsistency: DAL refuses to stamp placeholder tokens (`dal_logs.py:500`) but embed injects `OPERATOR`/`ANCHOR` when absent — so sentinels DO reach vector space. Fixing changes embed text for pre-identity rows → re-embedding → breaks §6. Deferred deliberately | §3.4 |

**Non-goals (explicitly out of scope):**
- Authentication of any kind (declared-only, per Tom).
- ACL / partition / per-node visibility — contradicts the shared-autobiography
  commitment; discretion is a runtime function at the speech boundary.

---

## 12. Decision log

All review questions are now resolved. No open questions remain for §§1-11;
what's left is the section-by-section walk of §§3, 5, 6, 8, 9, 10 mechanics.

Resolved (all Tom, 2026-07-24):
- ~~Q1~~ → **keep `BRAIN_OPERATOR_NAME`** as the install-default env (§4); the
  duplicate `BRAIN_USER`/`default_user` path is retired regardless (§4.1).
- ~~Q3~~ → node-field voice rename **happens, as a separate eval-gated arc**;
  names locked now (`counterpart_raw_quote` / `self_raw_quote`) — §11.1 (F5).
- ~~Q4~~ → self-channel / stream-message fencing **deferred**, marked at the
  code site — §11.2 (F2).
- **Boot banner** → `"I'm Anchor, I have N memories, M locked. I'm now with
  Tom."`; final sentence drops when counterpart is unset; **`[BRAIN]` envelope
  removed at boot only** (kept for interjections) — §4.0. Includes the
  deliberate voice change away from "The brain is yours".
- **Fork A** → migration infra and the vocabulary change **land together**, but
  the version runner is built and tested first behind a no-op version step
  (§10).
- **Fork B** → old keys at the write boundary **warn, never block** (option 1) —
  consistent with "a malformed trace beats a lost one"; CI contract tests are
  what actually catch a straggler writer (§8).
- **Unset counterpart** → store nothing; "unknown" is a display word only —
  §3.4. `set_env` is truthy-only — §3.2a.
- ~~Q2~~ → omit speaker on structural/S1/S2 rows ("speaker is for speaking").
- **Type vs name** → name only; kind is per-entity, home is Layer-2, deferred
  free (§2).
- **Build posture** → prepare multi-user substrate now, mark behavior for
  later via `LATER(multi-user)` code markers (§1.1).
- **Migration infra** → existing `brain.db` version-runner pattern made
  universal (both DBs) + auto-backup for data-rewrites + port-before-migration
  ordering (§7.1, §7.1a).
- Decisions #1 (single `speaker` field) and #2 (`self_name` global, in config
  file) confirmed.
