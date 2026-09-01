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

### 3.2 Counterpart lives behind ONE accessor — the SessionContext field is DEFERRED

REVISED 2026-07-26 (Tom ruled). An earlier draft added `counterpart: str` to
`SessionContext.set_env()` and threaded it through `save()`/`load()`,
`context_boot`, and the boot hook payload. **That is deferred.** What ships is a
single accessor:

```python
# Brain — the ONE place that answers "who spoke this event?"
def speaker_for(self, ctx, ref_type) -> str:
    """Resolve the per-event speaker. TODAY: self_name for Anchor's own events,
    the install-default counterpart for the other side. `ctx` is accepted and
    deliberately unused for the counterpart branch — that is the seam.
    LATER(multi-user): return ctx.counterpart instead of the install default."""
```

**Why the field is deferred.** `counterpart` today carries a *constant*: one
counterpart, one config value, identical in every session. Threading a constant
through six files buys nothing observable and costs exactly what Tom flagged —
a fifth SessionContext env field, each with its own truthy/three-state
semantics, hand-listed in `save()` and `load()`, plus a new boot-payload field
and dispatch routing. That is "patch a new parameter everywhere blindly."

**Why the accessor is not just deferral.** The §5.1 asymmetry says a
per-session value can have no fallback at any layer — so shipping a global and
going per-session later would normally mean revisiting every stamp site
(patching it twice, worse than doing it once). The accessor collapses that to
**one line in one place**: today it reads the install default, later it reads
`ctx.counterpart`. `ctx` is already threaded to the call site, so no plumbing is
needed when the switch happens.

What this preserves from the original design, unchanged:
- The daemon still never holds a *mutable identity singleton*:
  `brain.operator_name`, `TraceDAL.set_identity`, `_human_identity`,
  `_agent_identity`, `_stamp_identity` are all still deleted (§3.3, §5.3).
  Reading an install-default config value inside one accessor is not the same
  bug — the bug was identity state scattered across Brain and the DAL with
  every writer trusting it.
- `self_name` stays global and legitimately so (§2).
- Unset semantics are unchanged (§3.4): the accessor returns `''`, never a
  sentinel.

**When the field lands** (F4, or the first real second counterpart): add
`counterpart` to `set_env()` as a twin of **`cwd`, not `project`** —
branch/project are *derived* daemon-side from the fed-in cwd, while counterpart
has no derivation (there is no git-equivalent for "who is the human"), so like
cwd it must be fed from the hook, resolved hook-side (declaration → install
default → `''`), persisted in `session_state`, and surfaced in presence (F2).
That paragraph is the spec for later; it is not this arc.

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

**Removing the envelope at boot — safe in production, and scoped.** Verified
(re-verified 2026-07-27): **no production code parses `[BRAIN]`/`[/BRAIN]`** —
every occurrence in `servers/`, `hooks/`, `dashboard/`, `skills/` is a
string-literal *emitter*, no regex, split, or strip. `wrap_for_hook` is a
passthrough (the operator channel was killed 2026-03-28), so nothing downstream
depends on the markers either. **Tests DO parse them** (earlier "nothing
anywhere" claim was wrong for `tests/`): `tests/test_brain_voice.py`
`.index()`/`.find()`s the markers in boot output — five envelope tests break on
removal, and two more assert the exact `"Anchor. The brain is yours"` text
broken by the reword. Those seven tests update in this arc (§9).

There is even precedent that the markers can *harm*: `daemon_hooks.py:1153-1155`
records that an earlier `[BRAIN] GIT CONTEXT` block caused Claude Code to
`chdir` into the trailing marker (`ENOENT chdir '<repo>' -> '[/BRAIN]'`) — the
harness read the marker as data.

**Scope of the removal: boot only.** The envelope stays at the live
interjection sites — Bash safety warnings (`daemon_hooks.py:982-1035`),
keyless-boot notices (`:454-464`), edit auto-suggest (`brain_voice.py:133-234`),
and the recall-failure notice (`pre_response_recall.py:98`). Two census
corrections (2026-07-27): the host-environment notice (`:1095-1105`) builds the
markers but is **log-only** (`return {"output": ""}` — never reaches context),
and the per-turn recall injection is **already unmarked** ("Brain activated N
memories:", `surface_contract.py`) — so "marker for interjections" is the
prescriptive rule, not a description of the highest-volume path today. The line
is principled: those are **interjections** — the brain interrupting mid-flow — which
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
  nobody reads — dead boot latency. Correction (2026-07-27): the recall is
  driven by the hardcoded `task="session start"` (`brain_voice.py:341`), NOT by
  `user` — removing the `user` arg does not stop it. Tracked as node
  id:ba210861; its own cleanup.

So the value that *should* flow per session (hook → boot → identity) dies at
the banner, while the value that *does* stamp traces is a daemon-owned
singleton. Exactly inverted from "daemon is downstream-only" (id:16f06758).

**The correction (revised 2026-07-26 for the §3.2 accessor scope):**
1. Retire the duplication — one install-default value (keep
   `BRAIN_OPERATOR_NAME`, retire `BRAIN_USER`/`default_user`, OR the reverse;
   they must not both survive). This happens **now** — two live values for one
   concept is the actual mess, independent of multi-user.
2. `brain.speaker_for(ctx, ref_type)` (§3.2) becomes the single reader of that
   install default. `brain.operator_name`, `TraceDAL.set_identity` and the DAL's
   identity fields are deleted.
3. **Deferred** (F4): the boot hook resolving a per-session counterpart and
   sending it in the `context_boot` payload; `_handle_context_boot` routing it
   through `set_env(counterpart=…)`. Only the accessor's counterpart branch
   changes when this lands.
4. Path B's dead end is *not* wired up in this arc — it is **removed**. The boot
   `user` arg stops being resolved and sent, since the banner (§4.0) reads the
   accessor and nothing else consumed it. That deletes code rather than adding
   it. **The path has TWO heads** (2026-07-27): `boot_brain.py:51-73` (daemon
   path) AND `boot_brain.py:117-119` (`_boot_via_direct`, the no-daemon
   fallback — resolves `BRAIN_USER` again and calls
   `format_boot_context(user=...)` in-process). Remove both, plus the `user=`
   params in the `context_boot` / `format_boot_context` / `record_boot_render`
   signatures and their test callers (`tests/test_core.py:541`,
   `tests/relearning.py:912`, `tests/test_brain_voice.py:162`).

Result: the "where does identity come from" question has exactly **one** answer
site. It can evolve (install config → hook-declared → authenticated) by changing
that site, and the eventual hook-side resolution stays the right long-term shape
— the daemon receives the value rather than introspecting Claude.

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
function's own contract**, and the site has both `brain` and `ctx` in hand. Both
dialogue writes already route through it: `:201` (user_message, at
prompt-arrival) and `:626` (assistant_message, at Stop).

**Two census corrections (2026-07-27) to the "one writer" claim:**
1. `_s0_trace` also emits a FIFTH ref_type the policy must own:
   `anchor_touched` (`daemon_hooks.py:674`) — takes the structural row (omit),
   and the §8.1 policy-derived fixture must include it.
2. One S0 turn event bypasses `_s0_trace` entirely: `Brain.stamp_boot_liveness`
   (`brain.py:1002-1006`, called from `brain_voice.py:349`) writes the boot
   heartbeat via a direct `_trace_dal.append` — already on the writer list in
   `test_trace_contract_sync.py:21`. Its policy row is `heartbeat → omit`, so
   no stamp is needed there; it just bounds the "single writer" claim and must
   stay on the §8 contract-test writer list.

It stamps through the one accessor — `speaker = brain.speaker_for(ctx,
ref_type)` (§3.2). Passing `ctx` from day one is precisely what makes the later
per-session switch a one-line change *inside the accessor* instead of a sweep
across every writer.

Policy (contract-owned — `trace_contract.py` exports `SPEAKER_POLICY`; the
writer reads it rather than hardcoding):

| ref_type | speaker | why |
|---|---|---|
| `user_message` | `brain.speaker_for(ctx, …)` → install default today, `ctx.counterpart` later (§3.2) | the session's other side |
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

Mechanics note (2026-07-27): `post_tool_trace.py` never touches `TraceDAL` — it
sends `{"cmd": "trace_append"}` over TCP into the generic, ref_type-agnostic
`_handle_trace_append` (`dispatch_observability.py:11-45`). "Stamping at :86"
means writing `speaker` into the wire payload; the hook process reads
`BRAIN_SELF_NAME` from env (global — exactly the §5.1 asymmetry). The §5.3
chokepoint sits at the DAL, which that wire path crosses, so stragglers through
this open door still hit the guard. Also: the `brain_remember` row is degenerate
by design — session-less (`chain_id='archive-…'`), no `tool` key — it renders
under the `tool_result` label and embeds as `'Anchor via tool: …'`; leave it,
don't "fix" it mid-arc.

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
   *Verified (census corrected 2026-07-27):* `LOG_TABLES` has ELEVEN tables
   (`debug_log`, `dream_log`, `hook_errors`, `session_state`, `interactions`,
   `interaction_active`, `trace_events`, `trace_embeddings`, `self_inflight`,
   `self_delivered`, `boot_renders`) — and no meta/kv table among them, so
   `logs_meta` is a genuine addition. Caveat from the LIVE db: the actual
   `brain_logs.db` carries an **orphan `schema_migrations` table** (3 rows from
   2026-03-31, a pre-Python migration system; zero code references).
   Recommendation: v1 drops it (the runner's auto-backup covers recovery), so
   `logs_meta` isn't the second half-dead version store in the same file.
   Two mechanics facts: `ensure_logs_schema` lives in `schema.py:1527` (not
   `dal_logs.py`), and its signature is `ensure_logs_schema(conn)` — **no
   `db_path`** — so the auto-backup rule requires a signature change at the
   `brain.py:231` call site. `logs_meta` must live **in the logs DB**, not as
   another `brain_meta` key: a version has to travel with the file it
   describes, or restoring one DB from a different backup makes the version
   lie.
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
may be dead"* — indistinguishable from a corpse. **Corrected mechanism
(2026-07-27):** `ensure_daemon()` is NOT the assassin — the booting daemon holds
the fcntl singleton flock from before `_load_brain()`
(`daemon_server.py:149-151`), and `ensure_daemon` blocks on that same flock
before any kickstart (`daemon_client.py:220-225`), then re-checks under it. The
real kill path is **`recover_daemon()`** (`daemon_client.py:404-453`), which
never touches the flock: the MCP health monitor (`brain_mcp.py:1249` — its own
10×2s ≈ 20s budget, a distinct constant from `ensure_daemon`'s
`_GRACE_DEADLINE_S`) and every hook via `hook_common.py:295` call it, and after
~20s of refused pings it `launchctl kickstart -k`s — **SIGKILLing a daemon that
is mid-migration**. Brain init (embedder load, ~4-6s) already eats a chunk of
that budget before the migration even starts. The recovery cooldown/breaker
(`_RECOVERY_COOLDOWN_S=30`, max 5 attempts per 10 min) bounds the storm — with
an idempotent migration the boot eventually completes after repeated kills
rather than never — but "eventually, after up to five SIGKILLs mid-rewrite" is
not a plan.

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

**Half-keyed rows (gap found 2026-07-27):** `_stamp_identity` stamps each key
*independently* (`if self._human_identity:` / `if self._agent_identity:`), so a
fleet install with only one env var configured produces rows carrying ONE old
key — e.g. a `user_message` with `agent_identity` but no `human_identity`. The
guarded statements above skip such rows forever (guard key absent, ref_type
mismatch on the others), leaving old-key residue in migrated data. Fix: a final
sweep statement — bare `json_remove` of both keys wherever either survives
after the speaker-setting statements, count logged. Nothing is lost: the
skipped key is the *wrong-side* name for that ref_type (useless under §5
policy). §8.1's fixture gains a half-keyed row class.

**Idempotent + crash-safe by construction:** every statement is guarded on the
old key still being present, so an interrupted run re-runs harmlessly and
already-migrated rows are skipped. The version is stamped **only after all
statements complete**, so a SIGKILL mid-way retries rather than silently marking
the DB migrated. **Do not copy the ordering from `schema.py`** (2026-07-27):
the brain.db runner this mirrors stamps its version (`:1267-1271`) *before*
`_backfill_data` runs (`:1286`) — the exact inversion of this rule. Mirror the
pattern, not that ordering.

On the dev machine the first execution can additionally be run under the
maintenance lock with the replay verification (§10) before merging — but the
shipped path is the automatic one; the fleet never sees a script.

Node-level data (`their_raw_quote` / `my_raw_quote` etc.) is NOT touched —
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
- **D28** (`servers/scales/s1/quality_contract.py:590`,
  `D28_identity_tokens_concrete`) updates to the `speaker` key — same
  dimension, new vocabulary.
- **Prompt sweep** — no automated drift gate exists: grep the code-default
  prompts (indexed by `servers/interaction_defaults.py`) for the old keys, and
  check `list_interactions` for any deployed override that still carries them.

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
| `servers/brain.py` (new) | `speaker_for(ctx, ref_type)` — the ONE identity answer site (§3.2), carrying the `LATER(multi-user)` seam |
| `hooks/scripts/boot_brain.py:51-73` | **remove** the now-dead `user` resolution + payload arg; retire the `BRAIN_USER`/`default_user` duplicate (§4.1). Deletes code |
| `servers/dispatch_read.py:186` | drop the `user` arg plumbing from `_handle_context_boot` / `record_boot_render` if nothing else consumes it |
| ~~`servers/session_context.py`~~ | **DEFERRED (F4)** — no `counterpart` field, no `save()`/`load()` change, no 5th env field in this arc (§3.2) |
| `hooks/scripts/brain-env.sh:17` | env var comment; one install-default var only |
| `hooks/scripts/boot_brain.py:117-119` | second `BRAIN_USER` head (`_boot_via_direct`, no-daemon fallback) — removed together with `:51-73` (§4.1) |
| `servers/brain.py:1002-1006` (`stamp_boot_liveness`) | second heartbeat writer, bypasses `_s0_trace` — policy row omit, no stamp; stays on the §8 writer list (§5.2) |
| `servers/brain_assembly.py:745` | stale envelope-describing docstring (`[BRAIN-To-*]` merge that no longer happens) — update with §4.0 |
| `eval/laf/walker/embed.py:29` | `HUMAN_IDENTITY = 'Tom'` keyed to the OLD metadata key — update to `speaker` or the walker eval silently breaks post-migration |

### Dashboard — an API shape change, not just a grep
The dashboard reads the DB read-only, but it **re-publishes the identity keys as
HTTP API response fields**, and its JS consumes them. Two fields collapse into
one, so the response shape changes and Python + JS must move together:

| File | Change |
|---|---|
| `dashboard/queries/_meta.py:14-33` | helper returns a `(human, agent)` **tuple** → returns a single `speaker` |
| `dashboard/queries/recalls.py:96-105,187-188` | stops emitting the pair; emits `speaker` |
| `dashboard/queries/traces.py:3,51` | same |
| `static/lib/dom.js:128-143` | the chip helper pair lives HERE (not trace_detail.js): `identityChip()` / `identityChipHTML(human, agent)` → single `speaker` arg |
| `static/tabs/live.js:144`, `static/tabs/traces.js:192` | call sites; the `ev.human_identity \|\| ev.agent_identity` guard becomes a single check |
| `static/lib/trace_detail.js:148,320` | the keys appear as SUPPRESSORS (`_DELTA_KNOWN`; the `_renderGeneric` filter) — swap to `speaker` or the new key leaks into the generic metadata dump |

Both halves ship from the same repo copy, so they update atomically on dashboard
restart. One residual: a **browser tab left open on the old JS** would read
fields that no longer exist and silently drop the identity chip (no error). Cheap
mitigation — hard-refresh after deploy; noted rather than engineered around.

### Tests
`test_identity_stamping.py` (rewrite to policy), `test_dashboard_setup.py`,
`test_mcp_roundtrip.py` (`:500-518` pins `agent_identity` rendering verbatim —
the strongest rename guard in the suite), `test_trace_system.py`,
`test_trace_render.py`, `test_trace_embed_render.py`,
**`test_brain_voice.py`** (five envelope assertions + two exact-banner-text
assertions — §4.0), `test_core.py:541` / `relearning.py:912`
(`context_boot(user=…)` callers — §4.1) + new contract tests (§8).

### Docs (living only — archives are history, untouched)
`docs/EPISODIC-REFERENCES.md` (§5.3 templates, §15.1 language),
`docs/ARCHITECTURE-FRACTAL.md`, `docs/BACKLOG.md`, `CLAUDE.md` if it gains a
vocabulary line.

### Prompts (code owns the default)
Grep the prompt `.py` defaults for old vocabulary and edit `SYSTEM_PROMPT` in
place — the edit *is* the deployment. Also check `list_interactions` for any
deployed override carrying the old keys.

### Hardcoded `Tom` sweep — INSPECTED, and it is almost entirely a non-task

Do **not** mass-rename these. Verified 2026-07-24 across all nine files:

- **Daemon identity plumbing is clean** — no `servers/` or `hooks/` code reads
  a literal `'Tom'` as the operator's name; identity flows env/ctx only.
  One real behavioral hardcode exists OUTSIDE it (found 2026-07-27):
  `dashboard/queries/stats.py:136` keys a quote-capture insight on SQL
  `LIKE '%Tom said%'` — already silently wrong on the second install.
  Independent of this arc (node content, not trace keys); being fixed as its
  own change.
- **~90% are decision-provenance comments** — `"Tom 2026-07-02"`, `"per Tom"`,
  `"Tom's spec"`, `"Tom's standing rule (node 8178593a)"` (`encode.py`,
  `encode_contract.py`, `surface_contract.py`, `trace_links.py`,
  `scouts/muster.py`, `scouts/runners.py`, `presence.py`, `self_contract.py`).
  These attribute *who decided what*. **Leave them untouched** — they are
  legitimate attribution history, and rewriting them would destroy provenance
  for zero gain. (Census note 2026-07-27, **revised 2026-08-09**: the runtime
  footprint was ~29 files — the nine above, 4 dashboard files, ~8 more
  `servers/` comment sites, and 8 `servers/scales/s1/examples/` few-shot files.
  **Those 8 no longer exist**: the directory was deleted (654b2e2) after it was
  found to ship reconstructed real operator conversations to every install with
  zero runtime consumers, and to have diverged from the live §7.6 block it once
  generated. Current shipped-tree count is **~185 `Anchor` lines across ~43
  files** (exact figure moves with the manifest — recount before acting); D-12 now requires all of them to derive from `BRAIN_AGENT_NAME`.)
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
longer exist in data (§7); prompt drift → the §9 prompt-default grep; missed file →
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

- **F5 — node-field voice vocabulary** (`their_raw_quote` → `counterpart_raw_quote`,
  `my_raw_quote` → `self_raw_quote`). Its own arc, **eval-gated** — see
  §11.1. Names LOCKED here so the next session executes instead of
  re-deciding.

### 11.1 F5 — the node-field rename is a behavioral hypothesis, not hygiene

RULED (Tom, 2026-07-24): the rename happens, as a **separate eval-gated arc**,
not folded into this one.

**Why it is not cosmetic.** The field name is an instruction to the encoder.
Tom's original diagnosis (id:376b0c49): *"Maybe 'their_raw_quote' is the biasing
factor cause you're not a user."* The measured asymmetry was 97 operator quotes
vs 9 Anchor quotes (id:c036d192). And the *form* of the current pair encodes the
bias: `their_raw_quote` (a role) paired with `my_raw_quote` (a name) is
asymmetric, marking one as default and the other as bolt-on. The replacement
pair is symmetric in form — both are positions:

| Today | F5 | Rationale |
|---|---|---|
| `their_raw_quote` | `counterpart_raw_quote` | matches §2 vocabulary; drops "user" (Anchor isn't one, and neither is the human in this frame) |
| `my_raw_quote` | `self_raw_quote` | symmetric with the above; pairs with `self_name` |

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
session), all four encoder prompt defaults (edit `SYSTEM_PROMPT` in each
prompt `.py`), the full contract-sync chain (`remember()` → MCP schema → dispatch →
encoder tools → SKILL.md), `contract.py`, `dal_metadata.py`, the S2
healer/community/consolidation contracts, quality + surface contracts, scouts.
**Verify before executing:** whether the quote *values* feed any node
embedding group — if they do, the same zero-re-embedding invariant as §6 holds
(keys don't enter vector space, values do), but confirm rather than assume.

**Consequence of deferring, accepted:** node fields and trace fields speak
different dialects until F5 lands (`their_raw_quote` on nodes, `speaker` on
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
| `servers/channels/self_channel/presence.py` | roster needs a counterpart filter or it becomes cross-counterpart noise (id:030be61c, id:81c3982b) | F2 |
| `servers/dispatch_self.py` | `self_send`/`self_peek` same-counterpart default, cross is deliberate | F2 |
| Layer-2 participant node (wherever it is first minted) | entity **kind** (human/agent/stream) + relationship/trust live here, NOT per-event (§2) | §2 |
| `servers/brain.py` `speaker_for()` counterpart branch | **the primary seam** — return `ctx.counterpart` instead of the install default; this is the one-line switch the whole deferral rests on (§3.2) | F4 |
| `servers/session_context.py:222` (`set_env`) | add `counterpart` as a `cwd`-twin (fed, not derived) + `save()`/`load()`; spec written in §3.2 | F4 |
| `hooks/scripts/boot_brain.py` | resolve a per-session counterpart hook-side and send it in the boot payload → later authentication | F4 |
| `servers/contract.py` (voice quote fields) | node-field voice rename, eval-gated | F5 |
| `servers/embed_queue.py:368` | sentinel inconsistency: DAL refuses to stamp placeholder tokens (`dal_logs.py:500`) but embed injects `OPERATOR`/`ANCHOR` when absent — so sentinels DO reach vector space. Fixing changes embed text for pre-identity rows → re-embedding → breaks §6. Deferred deliberately | §3.4 |

**Non-goals (explicitly out of scope):**
- Authentication of any kind (declared-only, per Tom).
- ACL / partition / per-node visibility — contradicts the shared-autobiography
  commitment; discretion is a runtime function at the speech boundary.

---

## 12. Decision log

All 2026-07-24/26 review questions are resolved. Two amendments from the
2026-07-27 fresh-eyes pass are AWAITING RULING (below); what's left besides
them is the section-by-section walk of §§3, 5, 6, 8, 9, 10 mechanics.

**REVIEW, 2026-07-27 (fresh-eyes verification pass, separate stream).** Every
code claim re-verified against HEAD by four independent traces. Factual
corrections folded in place (§§4.0, 4.1, 5.2, 7.1, 7.1a, 7.2, 8, 9): the
bypass heartbeat writer + `anchor_touched`, the second `BRAIN_USER` head,
`test_brain_voice.py` breakage, LOG_TABLES census, the watchdog-path
correction (`recover_daemon`, not `ensure_daemon`), dashboard chip/suppressor
attribution, the walker-eval hardcode, the orphan `schema_migrations` table,
and the `schema.py` stamp-ordering warning. **Architecture endorsed.** Two
amendments NOT folded, awaiting Tom's ruling:

1. **Accessor shape:** rename `speaker_for(ctx, ref_type)` →
   **`counterpart_for(ctx)`**. No real call site uses the `ref_type` arg — the
   stamp site holds `SPEAKER_POLICY` itself; the banner wants "who am I with,"
   not "who spoke a user_message"; the no-ctx `tool_result` writers never call
   the accessor at all. Same ruling preserved (one accessor, ctx-threaded
   seam, one-line F4 switch); the signature stops promising a resolution it
   doesn't perform. Speaker resolution = `SPEAKER_POLICY[ref_type]` →
   `self_name` / `counterpart_for(ctx)` / omit, applied at the writer.
2. **Runner shape:** the logs runner shares a ~30-line versioned-migration
   envelope with `ensure_schema` (read version → backup-if-rewriting → run due
   steps → stamp LAST), extracted in `schema.py`; logs-side consumer now,
   brain.db moves onto it as its own later step. Rationale: the envelope is
   where the crash-safety ordering lives, and the existing brain.db runner
   gets that ordering wrong (§7.2) — a parallel hand-copy would inherit or
   re-derive it.

**REVISION, 2026-07-26 (Tom ruled) — scope narrowed to one accessor.** Three
review spots resolved together:

- **`counterpart` on SessionContext is DEFERRED to F4** (§3.2). It carries a
  constant today (one counterpart, one config value, same in every session), so
  threading it through `set_env`/`save`/`load`/`context_boot`/boot-payload buys
  nothing observable and *is* the "patch a parameter everywhere" failure. What
  ships is `brain.speaker_for(ctx, ref_type)` — one accessor, `ctx` passed from
  day one, so the later per-session switch is one line in one place. This also
  resolves the fifth-env-field concern and makes the arc ~4 files smaller (and
  net-deletes the dead boot `user` path).
- **`speaker` IS stored on Anchor's own events** — 93% of stored values are the
  same constant (`tool_result` alone is 36,588 of 44,300 s0 rows, and render
  never reads its speaker), *but* `tool_result` is eagerly embedded
  (`EAGER_TRACE_REF_TYPES = SAID_AND_DID_REF_TYPES`), so the name is **already
  frozen into those vectors** as `'Anchor via Bash: …'`. Not storing it in
  metadata would let render and vector disagree on a rename. And since the
  backfill (`5cff407`) already stamped all 57,672 traces with *two* identity
  fields, one field is a **reduction** of what is stored today, not an addition.
  (Correction: an earlier note claimed this "shrinks the migration" — it does
  not, materially; both variants touch every row.)
- **The arc's two halves are separable, and both still ship**: the rename earns
  its place unconditionally (the other side can already be an agent or another
  stream — 134 `self_message` rows exist now); the per-session *plumbing* does
  not, and is deferred.

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
