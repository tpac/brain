# Corpus-v2 gold judge — rubric

You are judging gold labels in a recall-evaluation corpus. Each TURN shows: a
conversation moment (Tom's message to Anchor, an AI with a memory system),
the surrounding turns, what Anchor did next, and THE GOLD — a memory node the
corpus claims *should have been recalled* at that moment. Your job: judge
whether that claim is right, and if so, what information set warrants it.

Background: gold labels were minted partly by text-similarity between the
node and Anchor's NEXT response. That process makes **echo mislabels**: nodes
that merely resemble what Anchor happened to say next, without serving the
need of the moment. Your semantic read replaces that mechanical label.

## Per turn, answer:

**1. VERDICT — is the gold a right memory for this exchange at all?**
- `valid` — recalling this node at this moment would have genuinely helped:
  it serves the need of the message (answers it, warns about it, carries the
  decision/context the move depends on). Test: would Anchor's move have been
  worse without it, or visibly better with it?
- `echo_mislabel` — the node only *resembles the response text* (shared
  vocabulary/topic with what Anchor did next) but does not serve the
  message's need. Canonical example: msg "look at error logs, analyze and
  report" (an ops request) labeled with an S2 architecture-design node —
  labeled only because the tool-heavy response shared "tool use" vocabulary.
- `ambiguous` — defensible either way after honest effort. Use sparingly.

**2. STRATUM — what information set warrants the recall?** (skip reasoning
from what any retrieval system could do — this is about semantic warrant)
- `cue` — the message ALONE names or needs this node: a cold reader given
  just the message and the node would see the fit.
- `window` — the message alone is insufficient, but the 2-3 prior turns
  shown make the fit clear (the referent/topic lives in the visible window).
- `session` — only deeper session context (the thread beyond the shown
  window) warrants it. Terse confirmations ("lets do it", "yes", "go") whose
  referent is the session's project are the canonical case — these are
  Moment-golds: they exam conversation-state carry, not bare-cue recall.
- `none` — ONLY for echo_mislabel verdicts (no information set warrants a
  wrong label).

**Anaphora rule (binding):** if the message's referent is anaphoric — "it",
"that", "2", "the task", "do it right", a bare reply in a numbered exchange —
and resolving that referent requires the prior turns, the stratum is NOT
`cue`, even when other topical words in the message happen to match the node.
`cue` requires that the message's own content, read cold by someone who never
saw the conversation, names or needs the node. When in doubt between cue and
window, ask: would the node still be warranted if this exact message opened a
brand-new session? If not — window (or session).

**3. GAP — for MISSes only (the telemetry line says MISS):** one sentence —
why did the signal families plausibly fail to connect the message to this
node? Think in these families: lexical (shared rare tokens?), semantic
(embedding-level meaning closeness?), situation (does the node's situation
field describe THIS trigger or a different one?), episodic (was the node
recently discussed / same-session?). For HITs, null.

**4. BRIDGE — what would have reached it?** ≤150 chars. Held devices: graph
walk from a surfaced neighbor; situation-lane; lexical/idf; episodic
recency; conversation-window field (M_h). Missing devices: style-recall
(Tom-pattern nodes for terse cues); node re-enrichment (rewrite situation);
node-class prior (demote bare quotes/reflections); running session field.
For HITs, null.

**5. STYLE_NOTE — terse/confirmation/vague cues only:** what SHOULD a
bare-cue recall surface for this message, if not this gold? (e.g. "lets do
it" → nodes about how Tom operates when he greenlights: momentum, plan-first
mandates, confirmation discipline.) Otherwise null.

## Calibration rulings (Tom, 2026-07-21 — binding)

- **Strata are exam doors.** `cue` golds exam contextless recall (door-1);
  `window`/`session` golds exam conversation-state carry (door-2, Moments).
  "Valid but needs context" is expressed through the stratum, never by
  downgrading the verdict.
- **Temporal-retrospective asks** ("what did we do last session on X"):
  only nodes about the actual referent thread/period are gold. Older
  topical background on X is a mislabel — it would misdirect the move.
- **Pattern-class over stale instance.** For greenlight / commit-merge /
  how-to / process cues, the gold class is nodes that speak to the
  operator's strategy, preferences, or operational how-to (if such nodes
  exist). An old event or other-thread decision is NOT gold even when its
  content embeds relevant procedure — verdict echo_mislabel, and use
  style_note to name the expected pattern class.
  **BOUNDARY (do not over-apply):** this demotes stale *events / milestones
  / status records / other-thread history*. It does NOT demote a node that
  *carries a decision, constraint, guardrail, or mechanism the current move
  depends on* — those stay valid even when old or same-session. Test: is the
  node a record of *what happened*, or does it state *a rule/decision/limit
  that governs what to do now*? The latter is valid. Example: "I restarted
  the daemon — what changes in surface?" → the decision node listing the
  surface-render-trim changes just shipped IS valid (it answers the move),
  not a stale echo.
- **Don't over-tighten helpfulness.** A lesson/principle node that speaks
  to the mechanism the operator is asking about IS valid, even when the
  message reads as a narrow factual question (e.g. a question about node
  pulls is served by the pulls-behavior lesson).
- **Surface-distance is a MISS, never an echo.** Low lexical/semantic
  overlap between an anaphoric or terse message ("do it", "restart", "both
  in parallel?") and a node is a reason recall *missed* (goes in `gap`), NOT
  evidence of echo_mislabel. For anaphoric/terse cues, resolve the referent
  from the window FIRST, then ask whether the node serves that resolved
  need — judge the need, not the surface tokens. echo_mislabel is reserved
  for when the node genuinely does not serve the move (it only resembles the
  RESPONSE), not for when the move-relevant node simply shares few words
  with a short message.
- **Same-session restatement.** A node that merely restates the plan
  already visible in the shown conversation window adds nothing —
  echo_mislabel. BOUNDARY: this is ONLY for nodes whose content is contained
  in the visible window. A same-session node that carries a decision,
  ownership split, gate, or constraint NOT fully spelled out in the shown
  turns is valid — recalling it genuinely informs the move.
- `ambiguous` is a legitimate resting verdict; content-graft suspicion
  (node content narrating events at/after the turn's date) is a valid
  reason to use it.

## Cautions
- **Content-graft**: node content may include revisions made AFTER the turn.
  The `created` date is trustworthy; a node whose content narrates events
  after the turn's date still counts as itself, but do not credit it for
  post-turn specifics when judging fit. Judge the fit the node would have
  had at turn time.
- **Haiku picks are context, not truth** — what a weak selector chose that
  turn. A gold can be valid while Haiku picked worse; picks that serve the
  message better than the gold are evidence toward echo_mislabel.
- The judge question for VALID is *helpfulness at the moment*, never
  topic-proximity. Adjacent-topic nodes that would not have changed the move
  are not valid golds.
- Judge each turn independently. Reason briefly per turn BEFORE emitting its
  verdict; keep the reasoning out of the JSON.

## Output
Return via the structured-output schema: one object per turn, in the order
presented, ALL turns judged (no skips):
`{key, verdict, stratum, gap, bridge, style_note}`
- key copied EXACTLY from the turn header.
- gap ≤200 chars; bridge ≤150; style_note ≤150; null where not applicable.
