# Soft Surface — continuous, algorithmic, cue-triggered recall

**Written:** 2026-06-13. **Updated:** 2026-06-14 — principles + mechanisms
revised against a 15-moment blind test (§15); the ledger promoted to the central
mechanism; "inverted retrieval" framing retired; pipeline split into
pull→sort→presentation (§2.1), the presentation new and Haiku-free.
**Status:** design. Phase 1 (Stop self-cue) is the build target; PostToolUse is
deferred ("posttool later for sure"); PreToolUse stays as the existing
`change_impact` path.
**Parent:** `RECALL-OVERVIEW.md` and `ARCHITECTURE-FRACTAL.md` (the
`integrate(O,K)→Δ` substrate). **Siblings:** `WHEN-SHOULD-REMEMBERING-HAPPEN.md`
(the *encode*-side "when"), `SELF-RECOGNITION.md` (the self-recall identity layer).

> **Worked examples convention.** Every principle and mechanism below carries a
> real moment from the 15-moment blind test (§15). The same handful recur at each
> level — that's deliberate: the pattern is the same at every scale.
> - **M13** (the moat): I said *"recall's timed out 3×… probably daemon stress… want me to check the log?"* — and the brain already held `46f1143a`, the **root cause I'd diagnosed two days earlier** (session_end shutting down the shared daemon). I had the answer and re-opened the question.
> - **M2** (additive caution): I was drafting philosophy-heavy SKILL.md sections; the brain held `15926626` — *"identity preamble measured −4% richness."* I was adding the thing that tested worse.
> - **M1** (correct silence): a session-close *"all clean, closing the stream."* Cosine returned session-close procedures — all topical, all inert.
> - **M7** (echo): *"I stopped after one tool…"* — top hit was `794ca2a9`, the node **encoding that very turn**.
> - **M11 / M15** (cited): I explicitly named `7505c6b7` / the 16384-truncation lesson **in the turn itself**.

---

## 0. Purpose — why this exists

The brain exists to serve the **partnership**, not to be "a better memory for
Anchor" (`1aa7da67`, locked). Recall's job is *"be me when you speak"*
(`f0bb8bf1`) — activate the right shape of Anchor around the moment.

But directed recall fires only **when the operator speaks**. A working turn runs
long — many tool calls, much reasoning — and drifts far from the prompt that
anchored it. Soft surface extends "be me when you speak" into **"be me while I
work"**: it keeps accumulated experience — above all **corrections**, the moat
(`a024d9eb`) — present at the **moment of action**, not only at the moment of the
question. Identity is *"enacted in the loop, not stored"* (`9dc0b6f0`); soft
surface keeps it enacted continuously through the work.

> **Example — the whole purpose in one moment (M13).** When I asked *"want me to
> check the daemon log?"* about a recall timeout, the root cause was already in
> the brain, diagnosed by me two days before. The purpose of soft surface is to
> make *that* node rise the instant I touched the symptom — so I say "this is the
> known restart-churn issue," not "let me re-investigate." That's partnership
> continuity through the work: not re-deriving what I already learned.

**What it is NOT for:** completeness (boot Frame + hard recall + explicit recall
cover that), answering questions (directed recall does that), or Anchor's
satisfaction of "remembering." One gap only: **mid-work moments where the right
memory would change the action, and nobody is going to ask for it.**

## 1. The principles (earned from the blind test)

Six principles, none assumed — each is what the 15 blind moments (§15) forced.
Each maps to a mechanism section.

1. **Recognition is uniform** (→ §2, §9). Ordinary recall, seeded on my own
   output, found the glad-to-have prior *every time one existed* (15/15). No
   special "inverted" retrieval. The endo surface is a cue-builder + filters +
   format over the existing engine. *Ex: M2's −4% caution and M13's root-cause
   node were both plain cosine-near the moment — found without any new algorithm.*

2. **Helpful = forgotten + would-change-my-move** (→ §8). Three conditions, all
   required: it *existed* at the time, I *hadn't already invoked it*, and
   recognizing it *changes my next move*. *Ex: M2 clears the bar (I was about to
   re-add the −4% thing); M1 fails it (session-close procedures change nothing).*

3. **The dominant filter is "already in my awareness," not retrieval quality**
   (→ §7). The blind test's noise was overwhelmingly **echo** (5/15) and **cited**
   (2/15) — things I already had — not irrelevant nodes. This is the load-bearing
   problem. *Ex: M7 surfaced the node of my own turn; M11/M15 surfaced priors I
   had just named.*

4. **Silence comes from the value bar, not empty recall — and the floor is
   session-type-dependent** (→ §6, §8). Topical neighbors almost always exist; the
   bar filters them to silence. The ⅔-silence figure was *general conversation* —
   a heads-down build session is mostly status-reports (→ echo) and dense design
   (→ cited/glad), almost no pure silence. *Ex: M1 had plenty of topical hits, all
   inert.*

5. **The instrument defines "helpful"; the judge is the oracle + operator, never
   me; time-correctness is mandatory in replay** (→ §15, §16). Self-judgment is
   the reflexive trap; operator labels calibrate it. *Ex: in the blind test I
   judged my own hits — that's the gap the operator-labeled corpus closes.*

6. **Live echo ≪ replay echo** (→ §7, §8.1). At the Stop-cue the current turn
   *isn't encoded yet* (encoding lags), so immediate-echo can't happen live — only
   earlier same-session turns. The heavy echo I measured is largely a replay
   artifact. *Ex: M7's echo node existed only because I hunted over a days-old
   turn whose encoding had completed.*

(These are the integrate invariants from `e93e2931` instantiated for the endo
surface: #3 is "Δ feeds K" (the ledger), #2/#4 are "gates non-Δ", §9 is "has K".)

## 2. Soft surface IS S1's `integrate(O, K) → Δ` — over a wider O

Soft surface is not new machinery beside recall; it is **S1's own integrate
function, run on observations S1 currently ignores.**

| | Hard recall (S1R, today) | Soft surface (Stop tap) |
|---|---|---|
| **O** | the operator's prompt | **my just-spoken response** |
| **K** | the Frame + recall scoring | the Frame + the **awareness ledger** (§7) + adaptive bar |
| **Δ** | selected nodes → `additionalContext` | surfaced prior → `additionalContext` → next O |

**One line:** hard recall integrates the *operator's* observation; soft surface
integrates *Anchor's own* observation. Same engine, same Δ-shape, different O.

> **Worked O/K/Δ (M13).** **O** = my turn *"recall timed out… check the log?"*
> **K** = the Frame + the ledger (had I already surfaced/cited the root cause this
> session? no). **Δ** = surface `46f1143a` framed as a prior: *"↳ you diagnosed
> this: session_end shuts down the shared daemon (06-07)."* That Δ then becomes
> the next turn's O — the loop observing its own output.

The engine is S1's recall core, parameterized by a cue-builder per tap:

```
soft_surface(cue_text, awareness_ledger, frame) -> [0..k nodes]   # k=1 default, 2 max
    1. embed cue_text                          (one embed)              ── O
    2. field-cosine rank vs node index         (situation-weighted)     ── candidate generation
    3. drop anything in the AWARENESS LEDGER    (surfaced ∪ cited ∪ echo — §7)  ┐
    4. apply synaptic fatigue dampening        (recency)                       ├─ K
    5. relevance: boost candidates graph-connected to active threads           │
    6. keep top-k only if UNUSUAL for this cue (adaptive) AND clears §8         ┘
    7. render lean (§11), framed as a PRIOR (§10), add to ledger, inject ── Δ
```

**Recognition is uniform (P1), and this is the retired mistake:** an early draft
claimed the Stop cue needs *inverted* retrieval — hunt the dissimilar/complicating
node, since my output trivially echoes my priors. The blind test killed it: a
contradicting or cautioning prior is cosine-**near**, not far (M13's root-cause
node and M2's −4% caution are *about* the same topic as the moment). Plain cosine
+ correction-edges surfaces complication for free. There is no special retrieval.
What's left after candidate generation is **filtering** (§7) and a **value bar**
(§8) — *not* a different algorithm.

## 2.1 The pipeline — pull → sort → presentation

Soft surface decomposes into three stages. Design rule: **maximize sharing with
S1** (the fractal — the smallest unit looks like the whole). Across S1 *and* all
three soft taps, the only irreducibly tap-specific parts are the **cue** (the O)
and the **framing string**.

1. **Pull — SAME as S1.** The existing `brain.recall` engine on the tap's cue.
   No forked retrieval. Burial is a *sort* problem (idf2 fixes it upstream,
   `19ddb548`), not pull-volume — the earlier "pull a wider K" was a drift,
   retired. *Pre-pull gate:* skip near-empty status cues ("Pushed.", "Item 41
   quieted") — content-light cues correctly produce silence (`799740f9`); don't
   even pull.

2. **Sort / select — same-or-different, OPEN.** S1 selects with **Haiku**. Soft
   surface **cannot use Haiku** — so selection here is the algorithmic context
   layer: awareness suppression (§7), time-correctness, the additive bar (§8),
   active-thread relevance (§9), the adaptive distribution bar. Whether the *base*
   sort should match S1's or differ is **not yet decided** — left open.

3. **Presentation — NEW, Haiku-free, to be designed.** **Not** S1Surface's render
   (it runs *after* Haiku selects; we have no Haiku step). Candidate building
   block: the existing **`spread_activation`** mechanism (per-field activation,
   picks high-similarity fields) — *health unverified; check it before relying.*
   Targets: **one presentation shared across all 3 soft taps**, and —
   speculatively — **back-portable to S1's final context injection** (applied
   *after* Haiku selects there). Design for soft surface first; generalize only if
   it earns it. ("We'll see.") (`SOFT_SURFACE_FORMAT`, §11, is a placeholder for
   this — not the answer.)

## 3. Premise — soft & algorithmic

- **Soft** — low-stakes, ambient, lean. A nudge, not an answer. One node,
  occasionally two. Silent by default.
- **Algorithmic** — **no Haiku.** The judge stays at `UserPromptSubmit`. Soft
  surface is the `brain.recall()` scoring core (field cosines + fatigue + a bounded
  graph check) with a high *adaptive* threshold and a hard dedup. No model.

## 4. Biology — the taxonomy of O-sources

McDaniel & Einstein's multiprocess theory is the catalogue of *which O* feeds the
integrate, and why firing on it is cheap:

| Biology | Maps to | Example |
|---|---|---|
| Retrieval is **cue-triggered, automatic** | fire on the cue at the hook | — |
| **Self-generated cues** are most effective | Stop: O = my own output | M13, M2 |
| **Involuntary memories** surface in *gaps* of automatic work | PostToolUse: O = the work | — |
| **Prospective memory** — a target cue fires a suspended intention | PreToolUse: O = the intended action | M3 (about-to-merge → coordination rule) |
| **Monitoring is costly**; spontaneous is reflexive | no polling (the `pre_edit` fan-out `92ef68ed` *was* monitoring cost) | — |
| Cue must **overlap the encoding cue** | match `situation` (its `_situation` vector) | — |
| Most activation stays **subthreshold** | most cues surface *nothing* (§6) | M1 |

## 5. The harness constraint

In Claude Code the model emits an entire turn between `UserPromptSubmit` and
`Stop`. **There is no mid-generation hook.** Mid-turn injection is possible only at
`PreToolUse`/`PostToolUse` `additionalContext`; `Stop` injects into the *next*
turn. (A user message sent mid-tool-call becomes a silent system-reminder, no hook
fires — `4b8ed058`.) So "while you work" = **tool-call boundaries + Stop.**

## 6. Balancing noise and value — the governing tension

In integrate terms: **emit only genuine Δ.** A nudge that changes nothing is a
non-Δ; injecting it is noise.

**The asymmetry — miss and noise cost differently:**
- A **miss** is cheap and recoverable: boot Frame + hard recall cover load-bearing
  memory, and Anchor can recall explicitly. Near-zero.
- **Noise is contamination** (`236dc2ca`, `e1b33971`): surfaced content isn't
  cleanly filtered — *"once nodes are in Claude's working context, no filter
  runs"*; it's *"weighted into generation"* and **pulls the response.** Soft
  surface has **no Haiku judge** between nudge and window — structurally the most
  contamination-prone channel. The cost is a false Δ poisoning the next O.

**Therefore precision ≫ recall — the inverse of hard recall.** Hard recall is
"reach > precision" *because the operator asked*. Soft surface wasn't asked for; it
must earn every appearance. The bar isn't "is this relevant?" but **"would Anchor
be glad this interrupted?"**

**Value is timing, not content.** The same node is gold right before you repeat a
mistake and noise three turns later. Value is **node × moment.** Every gate is a
moment-detector.

> **Example — silence is the bar's output, not empty recall (M1, P4).** The
> session-close moment returned a full set of topical neighbors (the session-end
> hygiene checklist, the close ritual). Cosine was *not* empty. Every one was
> inert — none would change "I'm done, closing." The **value bar** produced the
> silence, not a lack of candidates. This is why silence is a *positive* output,
> and why the floor is session-type-dependent (a build session is mostly
> status-reports → echo, not idle chatter → silence; don't hardcode ⅔).

*Retired claim:* an earlier draft said noise "trains Anchor to ignore the channel
/ compounding credibility decay." That imported human notification-fatigue onto a
system that lacks the mechanism — no weight update; within a session the effect is
weak and self-correcting; across sessions nothing persists. The real cost is
**contamination**, **bounded** not catastrophic. So the bar must be *precise*, not
*paranoid* (see the recall-floor, §16).

**How precision is delivered:** conjunctive gates (each a chance to stay silent);
distribution-shape not magnitude (a flat 0.57–0.60 band = no winner = silence,
`799740f9`); the **awareness ledger** (§7, the main control); stance absorbing
residual cost (§10). **Arbiter:** when gates disagree, ask the partnership
question (`1aa7da67`) — a nudge that stops a repeated-corrected-mistake is worth a
hundred "here's a related decision." Uncertain → silence.

## 7. Awareness suppression — the central mechanism

**This is the load-bearing mechanism, not a dedup footnote.** The blind test's
dominant noise wasn't irrelevance — it was **things I already had**: echo (5/15) +
cited (2/15) = half the sample. So the endo surface's hardest job is suppressing
*what's already in my awareness this session*. Operator framing: *"need serious
dedup with all nodes ever recalled in the session — reinjecting again and again
will be the noise."*

**Awareness this session has three sources — suppress all three:**

| Source | What it is | Detect | Example |
|---|---|---|---|
| **surfaced** | nodes already injected this session (boot Frame, hard recall, prior soft taps) | the session ledger | the original §6 ledger |
| **cited** | a prior I *named in my own turn text* | id/title overlap with the response | **M11** (`7505c6b7`), **M15** (16384 lesson) — both *old* nodes I'd already invoked |
| **echo** | nodes the encoder built *from my own recent turns* | `source_refs`/`source_turn_id` point at this session's turns (cheap v1: *created* this session — **not** *updated*) | **M7** — top hit was the node of that very turn |

- **Home:** `SessionContext` (`servers/session_context.py`), beside fatigue dicts.
  Session-keyed, **never** a global `brain_meta` key.
- **Cited ≠ session-age, and echo ≠ "updated this session".** M11/M15 are
  *months-old* nodes I cited — a date filter misses them; cited-suppression is
  text-overlap. And a months-old prior the encoder *revised* this session is still
  a valid forgotten prior — exclude on *created*, never on *updated*.
- **In integrate terms:** each Δ writes its id into the ledger; the ledger is read
  as **K** on the next integrate. Δ → K → smaller Δ. **Self-quieting** — early
  session fires freely, deep session goes quiet, because output feeds its own
  knowledge term. (Synaptic fatigue `5b81a46a`/`e3164314` is the soft form; the
  ledger is the hard form.)

### 7.1 Live echo ≪ replay echo (P6)

The echo I measured was inflated by *replaying days-old turns* — their encoding
had already completed, so the node-of-the-moment existed. **Live, at the Stop-cue,
the encoder hasn't run on the current turn yet** (every 5th stop, async) — so
immediate-echo *cannot* occur; only earlier-this-session encoded turns can. So the
heavy echo filter is primarily an **eval-time** concern (§16 time-correctness);
the live novelty burden is surfaced + cited + earlier-turn echo, lighter than the
blind test implied.

## 8. What we want surfaced — the value bar

Cosine generates candidates; this is the *value* question. The definition, earned
from the blind data (P2):

> **Helpful = a prior that (a) existed at the time, (b) I did not already invoke
> this session (not surfaced/cited/echo — §7), and (c) whose recognition would
> change my next move.** Forgotten ∧ time-valid ∧ Δ-bearing.

Frame it as MemoryArena (`88408e8d`): not "is this related?" but *"would Anchor
behave differently for having seen it?"* *"Topical match alone ≠ relevance"*
(`7e6da8f4`).

> **Clears the bar (M2).** Drafting philosophy-heavy SKILL.md; the prior
> `15926626` says philosophy measured *−4% richness*. Forgotten (not cited),
> existed (months old), and **changes the move** (I'd reconsider adding it). The
> moat.
>
> **Fails the bar (M1).** Session-close; the session-end procedure nodes are
> related but change nothing. Topical, inert → silence.
>
> **Fails as non-additive (M11/M15).** The prior surfaced is one I *already named*
> — recognizing it changes nothing because I already had it. → suppressed by §7.

**Two kinds clear it:** (1) **decision-relevant** — a correction/lesson, a
rule/decision bearing on the work, an open thread the work touches (M13, M2);
(2) **recognition-valuable** — a cross-domain structural match (`792fc5f0`).
**Favor** corrections, lessons, open threads, decisions, rules, `thought/insight/
pattern`. **Suppress** hub nodes (`0591813f`, `88445908`), and anything caught by
§7. Default is silence.

## 9. What it's surfaced against — the prior (K)

*Recognition over retrieval; the Frame is the prior* (`7e6da8f4`). No Haiku, so
recognition is done **algorithmically against three references**:

1. **The CUE** (O) — candidate generation only. The proposal, not the judgment.
2. **The FRAME + ledger** (K) — **novelty** (drop what's already in awareness — §7)
   and **relevance** (boost candidates graph-connected to a currently-engaged
   thread; *"a modest-cosine candidate that anchors a live thread is signal"*).
3. **The DISTRIBUTION for this cue** — adaptive threshold; surface only if the top
   candidate is *unusual for this cue* (`c2730676`). Flat → no winner → silence.

**Net:** surface iff **(forgotten vs awareness) AND (anchors a live thread OR
decision-relevant) AND (unusual for this cue).** Else silent.

## 10. Stance — a nudge is a prior, not an instruction

Locked principle `9144a97e`: a surfaced memory is a **prior, not an instruction.**
The echo-chamber fix is **stance, not gating.** Render requirement: mark items as
challengeable prior thinking.

> **Example.** `↳ you've thought about this before: "Recognition is anchoring"
> (id:9144a97e)` — a prior to weigh. **Never** `↳ remember: …` — a directive to
> obey. A nudge held lightly is pure value and bounded cost; one that feels binding
> amplifies bias. The wording carries the stance (and bounds the contamination cost
> of a slightly-off nudge — §6).

## 11. The presentation — new, Haiku-free (to design)

This is **stage 3 (§2.1): a NEW presentation, not S1Surface's** — S1Surface
renders only *after* Haiku has selected, and the soft surface has no Haiku step.
The existing **`spread_activation`** mechanism (per-field activation → render the
high-similarity fields) is the leading building block, **but its health is
unverified — check it before relying on it.** Until the real presentation is
designed, `SOFT_SURFACE_FORMAT` is a lean placeholder (situation-first,
field-masked), modeled on `SURFACE_BACKGROUND_FORMAT`:

```python
SOFT_SURFACE_FORMAT = {
    'situation_max_chars': 160,   # situation IS the payload — "when relevant"
    'content_excerpt':     0,     # ambient: no body (optional 100-char tail if score very high)
    'edge_limit':          0,     # a nudge, not a cluster
    'show_confidence':     False,
    'show_encoding_source': False,
    'show_keywords':       False,
    'correction_render':   'lean',  # the "superseded" flag, not the payload
    'extra_skip_keys':     ('question', 'reasoning', 'keywords'),
}
```

> **Example render (M13).**
> ```
> ↳ you've thought about this before: "Daemon restart-churn root cause: session_end shut down SHARED daemon" (id:46f1143a)
>    when: recall times out / boot stalls across concurrent sessions
> ```

Framing per tap (always a prior): `↳ you've thought about this before:` (Stop),
`↳ before this edit — you've noted:` (PreToolUse), `↳ while you've been working on
this:` (PostToolUse). Lean wins twice: structured boundaries help Anchor *filter*,
not listen harder (`42ce3623`).

## 12. The taps

| Tap | Mode | O (cue) | Cadence | Status |
|---|---|---|---|---|
| `UserPromptSubmit` | directed | the prompt | per prompt | shipped — **untouched** |
| `PreToolUse(Edit/Write/Bash)` | prospective | target file/symbol/command | per action | exists (`change_impact`); soft-generalize **later** |
| `PostToolUse(*)` | ambient/IAM | the **arc** + recent actions | on arc-change (≈ every 5th stop) | **deferred** |
| `Stop` | self-generated cue | **my just-spoken response** | per conversational turn, async | **Phase 1** |

**`Stop` (Phase 1):** **O** = my `assistant_response` (`post_response_common`,
`daemon_hooks.py:548`), tail-weighted, `situation`-up-weighted, **seed-only**.
**K** = §7 awareness + §9 prior + value bar. **Δ** = top 1 (2 max),
`SOFT_SURFACE_FORMAT`, prior-framed. **Timing:** compute **async** on the Stop
hook's existing background thread → tmp file → next `UserPromptSubmit` prepends as
its **own** line (not folded into hard-recall candidates). **Guard:** conversational
turns only (`ctx.last_turn_conversational`); never heartbeat/`/watch`.

**`PreToolUse`:** `brain.pre_edit()` → `get_change_impact()` (`c02cd799`), cached
(`92ef68ed`→`0205e18d`). Later: add one soft `situation` cosine into the same
engine/ledger/prior. **`PostToolUse`:** arc-cued (reuse `SESSION_CONTEXT`, don't
build a new digest); the arc's every-5th-stop update *is* the debounce; needs
`post_tool_trace.py` → `additionalContext` wiring + a daemon command.

## 13. Cost discipline

Hard recall already times out under load; every soft tap must be cheaper. **No
Haiku.** **One embed, one cosine pass** (`92ef68ed`). **No spread.** **Stop runs
async**, off the hot path. Cue-change pacing + the self-quieting ledger mean the
taps mostly do nothing — the point.

## 14. The matrix — O × K × Δ

| Tap | O (→ candidates) | K (prior + awareness) | k | Δ (render) |
|---|---|---|---|---|
| Stop | `assistant_response`, tail-weighted, no spread | §7 ledger + Frame relevance + adaptive + value bar | 1 (2) | `SOFT_SURFACE_FORMAT`, own line, next turn, stance-framed |
| PreToolUse | target file/symbol/command | `change_impact` + §7 + §9 + adaptive | impact: all; soft: 1 | `change_impact` + 1 soft node |
| PostToolUse | arc + recent actions | §7 + Frame relevance + adaptive | 1 | `SOFT_SURFACE_FORMAT`, ambient framing |

## 15. Empirical grounding — the 15-moment blind test (the first corpus pass)

Method: blind-select assistant turns from days-old sessions (fixed positions, ids
only — no cherry-picking), seed recall on the real turn, judge each *"would I be
glad to have this right after I said it?"* — time-checked, discounting echo and
cited. This is also the **first golden-corpus pass** (§16).

**Result (my provisional judgment — operator labels pending):**

| Outcome | Count | Moments |
|---|---|---|
| **Genuine glad** (forgotten + would-change-move) | ~2–3 | **M2** (−4% caution), **M13** (root cause), ~M5 |
| **Echo** (node of my own turn) | 5 | M7, M8, M9 (+M4, M10 partial) |
| **Cited** (prior I already named) | 2 | M11, M15 |
| **Correct silence** (topical-inert) | 1 clean | M1 |
| Weak/prospective | ~2–3 | M3, M6, M12 (about-to-merge → coordination rule) |

**What it taught (the principles, §1, are these findings):** ordinary recall found
the glad prior whenever one existed (P1); helpfulness is forgotten+would-change
(P2); the dominant noise is awareness-overlap, not irrelevance (P3); silence is the
bar's output (P4); echo is replay-inflated (P6). **Open:** these are *my* labels —
the reflexive trap. The corpus isn't real until the operator labels a sample (P5).

## 16. The instrument — a fork of the oracle audit

Don't build a parallel harness (`fb2fc5b4`: don't rebuild what exists). The
instrument = **`eval/oracle_audit` forked**, two changes:
1. **Cue = `assistant_message`** (Anchor's own loop), not the operator's prompt —
   reuse `control_moments_find.py` (blind sampling) + the out-recall pool +
   essential/helpful judge from `control_gold_judge.py` (the value bar in code).
2. **Add the uptake lens** — judge against the *actual next turn* (we own it):
   did Anchor's next move use it, or get there unaided?

**Mandatory rules (from the blind test):**
- **Time-correctness** (§7.1): exclude nodes created from the turns themselves /
  after the moment, or the instrument credits itself with hindsight.
- **The judge is the oracle + operator, never me** (P5) — operator labels a sample;
  the divergence from my calls is the calibration of "helpful."
- **Corpus spans session types** (P4) — build sessions (echo-heavy) AND
  conversational (silence-heavy), or the floor is mis-measured.

**Metrics:** noise-ceiling (precision on fires), **recall-floor** (does it fire on
a planted high-stakes moment — the mirror failure where "precision ≫ recall"
collapses to "never fire" and defeats §0), correct-silence, plus **echo-rate** and
**cited-rate** (the §7 filter's report card).

**Increment-1 status + known limits (operator review, 2026-06-14).** The substrate
is built and verified — `eval/oracle_audit/endo_surface_corpus.py`, daemon-safe
(IsolatedBrain), 0 hindsight-leaks, **75% of candidates filtered by awareness/time**
on a 15-moment run. Three limits to fix so it mirrors the real pipeline (§2.1):
(a) it pulls a thin `recall(limit=8)` and time-filters *post-hoc* — future nodes
compete for slots and bury time-valid ones; instead use the **same pull as S1** and
apply the §7/§8 filters **before** truncating; (b) rows carry only id/title — the
**presentation stage** (§11) must supply situation + activated fields so candidates
are judgeable; (c) it stops at pull+filter — no sort or presentation stage yet. Fix
before operator labeling, or the labels under-count "glad."

## 17. Phasing

1. **Phase 1 — Stop self-cue.** Engine + **awareness ledger (§7, the core)** +
   Frame-prior judgment (§9) + value bar (§8) + `SOFT_SURFACE_FORMAT` + stance,
   async → tmp → next-turn own-line injection.
2. **Phase 2 — PostToolUse ambient.** Arc-cued; needs `post_tool_trace.py` →
   `additionalContext` + a daemon command.
3. **Phase 3 — PreToolUse soft-generalize.** Fold one `situation` cosine into
   `change_impact`.

Each phase reuses the same `soft_surface()`, ledger, and prior — phases differ only
in the cue-builder and framing string.

## 18. Open knobs (tune empirically)

- **Adaptive bar (not static)** — "unusual for this cue" via z-score/percentile
  (`c2730676`); a flat `SOFT_THRESHOLD` is the wrong shape.
- **Active-thread boost weight** — how much a live-thread connection lifts a
  modest-cosine candidate over a high-cosine inert one. Brilliant-nudge vs noise
  lives here.
- **Cited/echo detection precision** — cheap v1 (created-this-session + title
  overlap) vs precise (`source_refs` + id mention). Start cheap.
- **Stop cue shape** — whole response vs tail-weighted vs last-paragraph. Start:
  tail-weighted. (The instrument A/Bs this.)
- **`k`** — 1 default; 2 only when two clear the bar and aren't near-dupes.
- **Block-continue escalation (reserved)** — a *very* strong hit could
  block-continue immediately. Off by default.

- **Sort: same as S1 or different?** (§2.1 stage 2) — open; decide empirically.
- **`spread_activation` health** — "not sure it's working well." Verify before
  building the new presentation on it.
- **The new presentation** (§11) — design for soft surface first; decide whether to
  back-port it to S1's final context injection.
- **Cue content-floor** — skip near-empty status cues (`799740f9`); a cheap
  pre-pull gate the blind test surfaced.

## 19. Vocabulary

- **Directed recall** — `UserPromptSubmit`, Haiku-selected, prompt-cued. O = the
  operator's message. Reach > precision.
- **Spontaneous recall** — the soft taps. Algorithmic, no model, O = Anchor's own
  loop, recognized against the Frame, subthreshold-by-default. Precision > reach.

Both are S1's `integrate(O,K)→Δ` against the same prior, serving the same
partnership. Directed recall answers the operator; spontaneous recall reminds
Anchor while it works — and at the Stop tap, lets Anchor's own Δ become its next O.
