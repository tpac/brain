# Thalamus — design v2

**Status: settled with Tom 2026-08-28, ready to build.** Supersedes v1 (rejected
2026-08-10 on corpus evidence); v1's fatal findings survive at the bottom as the
constraints they've become.

## The problem

Encoders raise things only a decision-maker can settle, into a store whose only
reader is the next run of the same encoder. Canonical case: consolidation asked,
three runs running, whether the decoder's suppression-verb set is meant to be
operator-configurable — a real inconsistency, correctly diagnosed, unread for
weeks. The old exit — `open ×5` → `journals-escalation` node → boot injection —
fires on **persistence, not need**, and lands in a passive channel: it saturated
all 10 boot slots with stale escalations while the real asks never arrived.

## What the Thalamus is

**The brain speaking to its streams.** A durable store of standing intents with
delivery policy — beside, not inside, the msgs layer (which is streams speaking
to each other, ephemeral, consume-once).

| | Msgs layer (exists, untouched) | Thalamus |
|---|---|---|
| What it is | a stream speaking to a stream | the brain speaking to its streams |
| Lifetime | ephemeral — TTL (1h/24h), reaped | durable — until answered / withdrawn / expired |
| Cardinality | one send, one pickup | **one item → N deliveries** |
| Memory | receipt dies at TTL | ledger `(item, session, when, via)`, forever |

One item ≠ one message: "every session should know X on boot for the next
month" is one row, delivered N times — once per session that boots inside the
window. The msgs layer cannot model that; the Thalamus exists because of it.

## The shape

### One door, three entrances

`file()` (daemon-side) is the single producer door.

- **Anchor** — the MCP tools (below), from any session.
- **LLM agents** (S2 units, S1 Scribe) — the same `remind` tool added to their
  agent-loop toolsets, wired to the same daemon function.
- **Code** (hooks, system producers, later) — direct call.

Producers state intent; the door routes:

| Producer intent | Route |
|---|---|
| everyone alive **now** | delegate to `signal.send(broadcast)`; row goes terminal `sent` — fire-and-forget, the courier owns its death |
| everyone in a window | queue: `expires_at`, audience `window` |
| next session / a specific stream | queue: audience `next-boot` / `directed` |
| at a time ("Tuesday", "in 3 days") | queue: `deliver_at` |
| needs a decision | queue: `needs_answer=true`, loud expiry |

v-first: live-now delegation is Anchor-filed only — `from_session` is real, so
the locked stream-speech render stays honest. Machine live-now needs the
`brain`-origin render (Phase 3).

### Object

```
thalamus_items:      id (th_xxxx) · source · body · refs · audience ·
                     deliver_at · expires_at · needs_answer · dedup_key? ·
                     state · answer · created_at
thalamus_deliveries: item_id · session_id · delivered_at · via (boot|stop)
```

Item states: `open` → `answered` | `dismissed` (by Anchor) | `withdrawn` (by its
own producer) | `expired` (window ended — **loud** terminal for an unanswered
ask, the dead-letter fix; a natural death for a notice). `sent` is the immediate
terminal for delegated broadcasts.

Delivery state and producer-condition state are **two orthogonal machines**
(id:e63c41dd): the ledger says who saw it; whether the underlying condition is
still true stays the producer's business. Retry (Phase 3) gates on *unacked*,
never on *still-open*.

### Identity & volume — the two v1 killers, and where they live now

- **Identity is never system-invented.** `dedup_key` is optional and
  producer-owned; a repeat `(source, dedup_key)` updates the item instead of
  inserting. No key is ever derived from note text — the corpus proved that
  impossible (91 subjects for one concern; 850/2,381 run-local — id:6789e133).
- **Volume is owned at the write boundary.** A per-source open-item cap
  (contract constant). At the cap, `file()` **rejects synchronously with
  guidance** — "you have N open asks: resolve, update, or withdraw." This is
  the argument for a tool over a journal protocol: a protocol's violation is
  found by a sweeper hours later with nobody to tell; a tool rejection lands in
  the agent's loop mid-run, where it can adapt. Per-render caps bound the read
  side.

### Delivery — pull at the two proven moments

Sessions pull; the Thalamus never enumerates sessions, never pushes, holds no
roster.

- **Stop drain** — beside stream mail, in its own render section. The locked
  `render_signal` containment contract ("other stream says:") is untouched.
- **Boot render** — replaces the `journals-escalation` standing-items block.

Pull predicate: `state=open ∧ deliver_at ≤ now < expires_at ∧ audience matches
∧ no ledger row (item, this session)`. The ledger is written at render —
annotate-at-render is the only visibility mechanism that survives receipt
expiry (id:8a170558). Each delivery also writes a trace event (new ref_type;
sync `trace_contract` + its test) — untraced delivery *is* the visibility
problem this system exists to fix.

Asks default to `next-boot`: an architecture question arriving mid-thread
trains reflex-deferral; at boot there is no thread to protect.

### Return surfaces — render-joins, never write-backs

- **To Anchor**: the two renders plus `thalamus_list()` (pullable any time).
  The render is answerable without a fetch: question + resolved refs
  (`id · title · gist`) + the pre-filled `thalamus_resolve` call.
- **To producers**: same-run, the tool result ("filed th_4a2f" / the
  rejection). Cross-run, the unit's journal-view assembler joins items on
  `source` and renders a live block — open items with current state, answers
  inline. **No journal notes are written back**: they would evict real residue
  (`JOURNAL_CONTINUITY_RUNS = 3`), go stale as snapshots (id:defbdf8b), and put
  machine-authored lines in the encoder's own voice.

### Anchor's surface — three MCP tools; the budget is three

- `thalamus_list()`
- `thalamus_resolve(id, answer=… | defer_until=… | dismiss=true)`
- `remind(what, when=…, for_whom=…, needs_answer=…, refs=…)` — **the** producer
  verb, for Anchor and agents alike: a notice, a reminder, and an ask are the
  same call with different params.

Producers may update or **withdraw their own items** (source-scoped) through
the same door. Without retraction, a condition that resolved itself waits as a
stale ask — the wallpaper defect, one level up.

## Journals under this design

There is **no journal consumer and no triager**. Journals stay purely the
encoder's self-record. An encoder that wants a decision files deliberately, by
tool, mid-run — the judgment lives in the producer that has the context,
informed by its own open items rendered into its view. The triager's former
jobs: identity → producer `dedup_key` + open-items visibility; volume → the
door cap; plain language → the tool description ("written for a reader with
none of your context — no internal vocabulary").

## Phases

**Phase 1 — core, zero LLM.** Schema (`servers/schema.py` discipline) +
contract constants; `file()` with routing + budget guard; both pulls; the three
MCP tools. Prove `remind()` end to end: file → due → delivered at Stop and at
boot → resolve → ledger + trace verified. Deploy is two-step — daemon restart
(`servers/*`) AND `./redeploy.sh` + new session (`brain_mcp.py`) — before any
behavioral test can run.

**Phase 2 — producers.** `remind` into consolidation's and the Scribe's
toolsets; the journal-view join; retire `journals-escalation`, the `open ×5`
promotion, and the standing-items renderer **in the same commit as their
replacement**. Then measure the two behavioral unknowns: do encoders file
sanely (spam / under-use), and does Anchor drain and answer.
`bridge_proposals` died built-but-unused (id:bfc6d106) — delivery alone is not
success.

**Phase 3 — policy.** Retry-on-unacked; machine live-now (`brain`-origin
render); the `on_topic` moment — deliver when a session touches related ground,
the salience gate that fully earns the name.

## Open / carried constraints

- **Scope veil — deliberately deferred.** Items aren't nodes and carry no
  provenance stamp. v-first producers are global (Anchor, S2 units, the
  Scribe), so nothing walled can leak; the moment a session-scoped producer
  files, items need a scope story.
- Stale comments still describing the removed PreToolUse leg —
  `daemon_hooks.py:530`, `signal.py:266` — clean up in passing during Phase 1.
- Prior ruling to discharge in Phase 2 verification: LATERAL-SCALES' *"the
  burden is on it to prove it's not just async S0."* The proof is behavioral,
  not architectural.

## v1 evidence, kept as constraints

Measured against 2,381 journal notes (2026-06-24 → 2026-08-10):

| Finding | Number | Constraint it produced |
|---|---|---|
| subjects are not identities (id:6789e133) | 91 subjects, one concern; 850/2,381 run-local | `dedup_key` producer-owned or absent — never derived from text |
| no viable volume setting | floor ~0.1/day, ceiling ~10.6/day on a 2.6/day channel | volume owned at the door, not by encoder discretion |
| delivered ≠ acted on (id:ab520c77) | a consumed self-message, still missed | Stop buys one turn, not a decision — asks go to boot; Phase 2 measures draining, not delivery |
| receipts expire (id:8a170558) | 1h / 24h TTL | Thalamus owns its ledger; annotate-at-render |
| passive injection fails (removed PreToolUse leg, `364269f`) | — | pull renders at the two moments that provably land |
