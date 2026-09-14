"""V3.8 authoring record — DRAFT carriers on the frozen V3.6 full template, one carrier type per arm.

The V3.8 round (Tom's brief, 2026-09-14): revise V3.6 where the guide fails at revisiting recently encoded
nodes inside the window it is exposed to — the covered-turn miss class (brain id:64abc6a0; the whole-prompt
read is AUDIT-COVERED-TURN-2026-09-14.md beside this file). Production is V3.6 full; each carrier is an
exact-once replacement on its template or gist, measured alone (one carrier per arm — id:4c43742b).

STATUS: drafts for Tom's marks. Nothing here is frozen; the replay probe (PROBES.md) is the only measurement.
Tom reads the audit table and marks first; the carriers below are re-authored to his marks before any freeze.

  GLOSS (template_gloss.md) — position, a text pass on four sentences where the class binds
    G1  L39: `encoded="true"` means a prior run SAW the turn, not that its facts are held; the catalog's
        `[encoded(me, turn N)]` entries are what it kept; a covered turn whose fact has no node, or whose
        node reads it differently than the words do, is mine now
    G2  L21: a prior note OR ARC LINE is evidence, not a ruling — least of all a no-mint verdict
    G3  L273: "covered restatements" → "restatements of what a node already holds" (coverage is not a
        reason to skip)
    G4  L289: a no-mint verdict goes to neither residue nor the arc
    (V3.6: the gloss rules covered turns out of fresh capture; the V3.7 verdict travelled in the Arc and
     the next run quoted it as a ruling; run 3 both arms called the covered May session "transactional")

  EXAMPLE (template_example.md) — one worked window after Mira's later window (L599)
    E1  a catalog node `[encoded(me, turn 5)]` "flume plume e-folds in 12 min at 18 °C" with an edge-only
        sampling-cadence node derived from it (interval = τ/4); a window whose covered turns include the
        January fit AND a three-months-later "about eight minutes now that we run the tank at 24 °C" that
        the covering run kept nothing from (no `encoded(me, …)` names it); the only uncovered turn is
        routine. Three Bads (covered → `new: none`; an `open` "12 min vs 8 min — which is correct?"; a
        thought-only "the single-exponential read is stable"). The lists mark the node stale on every
        surface incl. `event_time`; a fetch for the edge-only protocol node; the swap "about 8 min at
        24 °C as of … (12 min at 18 °C from the January fit)"; the cadence follows to 2 minutes.
    (V3.6: no worked window anywhere depicts a covered turn; Priya teaches the routine-change swap with
     the old value only in the catalog — the gym window had the old TEXT on the page and V3.6 filed rule 4)

  WALK (gist_walk.md) — procedure, the gist's lists
    W1  `targets`: before the change-driven walk, one pass the other way — every `encoded(me, turn N)`
        entry against the covered turns it came from; a fact no entry carries goes on `new`; an entry
        that reads a covered turn differently than the words do is `stale`
    W2  `new`: "the window carries, covered turns included — a covering run that wrote nothing for a
        turn left its facts to me"
    (V3.6 run 3: fourteen `clean` target lines against a catalog never compared with the covered text)

Run: ./dev python3 author.py  -> template_gloss.md, template_example.md, gist_walk.md, author_log.json
(template_full.md / gist_full.md are the V3.6 fixture's, read in place; gist_full.md is copied for the arms
that do not touch it).
"""
import json, pathlib, shutil

HERE = pathlib.Path(__file__).resolve().parent
PARENT = HERE.parent / 's1e_guide_v3_6_2026-09-13'
LOG = []


def replace(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'{label}: expected exactly one occurrence of the old text, found {n}')
    LOG.append({'label': label, 'old_chars': len(old), 'new_chars': len(new), 'delta': len(new) - len(old)})
    return text.replace(old, new, 1)


base = (PARENT / 'template_full.md').read_text()
gist = (PARENT / 'gist_full.md').read_text()

# ══════════════════════════ GLOSS — position, text pass ══════════════════════════
tpl = base
tpl = replace(tpl,
    "`encoded=\"false\"` is uncovered, my focus. `encoded=\"true\"` means a prior run covered the turn: text remains, actions become `trimmed — N action(s) recorded…`. I reread covered text for cross-turn patterns and contradictions, not fresh atoms; later evidence can revise its encoded substance. Previously encoded never means untouchable.",
    "`encoded=\"false\"` is uncovered, my focus. `encoded=\"true\"` means a prior run SAW the turn — not that its facts are held: text remains, actions become `trimmed — N action(s) recorded…`. What that run kept is in the catalog, tagged `[encoded(me, turn N)]`; I read covered text against those entries. A covered turn whose fact has no node, or whose node reads it differently than the words do, is mine now, as if uncovered — the flag records a run, the catalog records memory. Later evidence revises encoded substance; previously encoded never means untouchable.",
    'G1 L39 the gloss: covered means seen, not held; read covered text against the entries the run kept')
tpl = replace(tpl,
    "The runtime journal supplies it; a prior note is revisable evidence, not a ruling for this run.",
    "The runtime journal supplies it; a prior note or arc line is revisable evidence, not a ruling for this run — least of all a verdict that a stretch of turns earned nothing.",
    'G2 L21 continuity: the arc line is evidence too, and a no-mint verdict is not a ruling')
tpl = replace(tpl,
    "greetings, acknowledgements, covered restatements, abandoned questions without engagement.",
    "greetings, acknowledgements, restatements of what a node already holds, abandoned questions without engagement.",
    'G3 L273 skip: coverage is not a reason to skip')
tpl = replace(tpl,
    "A no-mint verdict never goes to residue. A miss I can name gets fixed now:",
    "A no-mint verdict never goes to residue or the arc. A miss I can name gets fixed now:",
    'G4 L289: the arc is named beside residue')
(HERE / 'template_gloss.md').write_text(tpl)
gloss_len = len(tpl)
gloss_tpl = tpl

# ══════════════════════════ EXAMPLE — a covered turn, reread ══════════════════════════
COVERED = '''### A covered turn, reread — the flag says seen, the catalog says what was kept

*Reading cue: Compare what the covering run kept with what the covered words say.*

Conversation now is **2026-04-20**. The catalog shows what an earlier run kept from January, under the
tag that names the run; the sampling protocol it grounds appears only as an edge:

```
[encoded(me, turn 5)] [measurement] "Flume dye plume — e-folding time 12 minutes at 18 °C" (id:4e7b1a92)
  Content: Nour fitted a single exponential to the rhodamine plume in the recirculating flume: concentration falls by 1/e every 12 minutes with the tank at 18 °C.
  Situation: When planning a tracer run in Nour's flume or reading a plume decay curve — the plume e-folds in 12 minutes.
  Question: How fast does the dye plume decay in Nour's flume?
  Reasoning: Nour's own least-squares fit on 2026-01-15, from one 90-minute run at a single temperature while the sampling schedule was set; a fitted parameter, not a temperature law.
  Their Raw Quote: the plume e-folds in about twelve minutes at eighteen degrees — that's the fit from the January run
  Event Time: 2026-01-15
  Edges:
    [protocol id:c1d8f350] "Flume sampling cadence — one sample every 3 minutes" this grounds — the 3-minute interval is one quarter of the 12-minute e-folding time; the cadence follows the plume's decay, not the autosampler's floor
```

The window. Turns 4 and 5 are January; the run that stopped at turn 5 wrote the two nodes, and its
provenance line says so. Turns 7 and 8 are today and covered too — a run stopped after them — but no
`encoded(me, …)` anywhere names them: that run kept nothing from them. Only turn 9 is uncovered:

```
<turn n="4" age="3 months ago" encoded="true">
  <other trace="a71c3e08">Fit's in — the plume e-folds in about twelve minutes at eighteen degrees, that's the fit from the January run. How often should the autosampler pull?</other>
  <me trace="5b902fd4">Every 3 minutes — a quarter of the e-folding time, so four points inside each e-folding.</me>
</turn>
<turn n="5" age="3 months ago" encoded="true">
  <other trace="d4e17b60">Good. And the blank correction — before or after the drift subtraction?</other>
  <provenance>encoded(me, turn 5): "Flume dye plume — e-folding time 12 minutes at 18 °C" id:4e7b1a92 | "Flume sampling cadence — one sample every 3 minutes" id:c1d8f350</provenance>
  <me trace="93c0af25">Blank first, then the drift subtraction…</me>
</turn>
<turn n="7" age="just now" encoded="true">
  <other trace="1f68d3b7">Still one clean exponential — no second tail. Can the students fit the demo run the same way?</other>
  <me trace="c50e29a1">Same fit — one exponential over the whole decay.</me>
</turn>
<turn n="8" age="just now" encoded="true">
  <other trace="b2af76c9">Then a 40-minute slot covers Wednesday's undergraduate lab: the plume e-folds in about eight minutes now that we run the tank at 24 °C, so they'd see five e-foldings.</other>
  <me trace="7e134d0b">Forty minutes is comfortable for that.</me>
</turn>
<turn n="9" age="just now" encoded="false">
  <other trace="06cb98e2">I'll write the run up tonight. Anything you need from me before that?</other>
  <me trace="f8a340c7">No — that covers it.</me>
</turn>
```

Three Bad moves, each one I have made. Bad: “turns 4–8 are covered; turn 9 is routine; `new: none`” —
the flag says a run saw turn 8, the catalog shows that run kept nothing from it, and the eight minutes
on the page has no node. Bad: an `open`, “plume e-folding — 12 minutes (turn 4) vs 8 minutes (turn 8),
which is correct?” — two fits by the same person on the same flume, three months apart, are a changed
parameter, not an in-window contradiction; rule 3, dated, not rule 4; the catalog's 12 minutes is true
as of January, at 18 °C. Bad: `thought` only — “the plume came up again; the single-exponential read is
stable” — turn 7 confirms the shape of the decay, turn 8 moves its constant, and reading covered text
for what confirms the node is how the change slipped past the run before this one.

```
changes: the flume's e-folding time — 12 minutes at 18 °C (2026-01-15) → “about eight minutes” at 24 °C (2026-04-20), same speaker, three months on: a refitted parameter, not a contradiction; the covering run kept nothing from turn 8 — no encoded(me, …) names it — so the change is mine now
changes: newly known — a 40-minute flume slot for Wednesday's undergraduate lab; not in catalog
targets: 4e7b1a92 · 12 min at 18 °C → about 8 min at 24 °C, the single-exponential shape unchanged → every surface that says 12 minutes moves: title stale · content stale · situation stale · reasoning stale · their_raw_quote stale (the January words carry the old fit; the April words carry the claim now) · event_time stale (dates the January fit, not what the node now says); the decay question still fits: question clean; the cadence is derived from the time constant: why→c1d8f350 stale
targets: c1d8f350 · a 3-minute interval for a 12-minute e-folding → a quarter of 8 minutes is 2 minutes: title stale · content unread
fetch: c1d8f350 — edge-only; its interval is derived from the time constant I am about to change
new: Nour's Wednesday demo run — a 40-minute flume slot for the undergraduate lab
```

`get_nodes(["c1d8f350"])` returns the protocol:

```
[protocol] "Flume sampling cadence — one sample every 3 minutes" (id:c1d8f350)
  Content: The autosampler pulls one sample every 3 minutes through a flume tracer run — a quarter of the 12-minute e-folding time, giving four points inside each e-folding.
  Situation: When setting up or changing the autosampler for a flume tracer run.
  Reasoning: The cadence I gave on 2026-01-15; the interval is derived from the e-folding time Nour fitted, not the sampler's 30-second floor.
```

The write, one `brain_batch`. The fitted constant takes the routine-change swap with the old value
dated in prose; the cadence follows it; the edge why moves with the revise, `old` copied from the edge
line:

```json
{"operations": [
  {"op": "revise", "node_id": "4e7b1a92",
   "reason": "Nour refitted the plume between January and April — about 8 minutes at 24 °C against 12 at 18 °C; the shape held. The run that covered the April turn wrote nothing, so the change is mine now: every surface that says 12 minutes moves, January stays in prose at its own temperature, and the refs carry both statements.",
   "title": {"old": "e-folding time 12 minutes at 18 °C", "new": "e-folding time about 8 minutes at 24 °C as of April 2026"},
   "content": {"old": "concentration falls by 1/e every 12 minutes with the tank at 18 °C.", "new": "concentration falls by 1/e about every 8 minutes with the tank at 24 °C, as of 2026-04-20 (12 minutes at 18 °C from the 2026-01-15 fit). The decay is still one clean exponential, with no second tail."},
   "situation": "When planning a tracer run in Nour's flume or reading a plume decay curve — the plume e-folds in about 8 minutes at the 24 °C set point.",
   "reasoning": "Nour's April 20 statement, made while sizing a teaching slot; the January fit was equally direct and is kept as the value at 18 °C. Two fits three months apart are a changed parameter, not a contradiction, and neither establishes how the time constant scales with temperature.",
   "their_raw_quote": "the plume e-folds in about eight minutes now that we run the tank at 24 °C",
   "event_time": "2026-04-20",
   "thought": "Twelve minutes at 18 °C and eight at 24 °C make the e-folding time a parameter of the tank's temperature, not a constant of the flume; two points are not a law, and a third set point would say whether the dependence is worth fitting.",
   "source_refs": ["a71c3e08", "b2af76c9"],
   "connect_to": [{"target": "c1d8f350", "relation": "grounds", "why": {"old": "the 3-minute interval is one quarter of the 12-minute e-folding time", "new": "the interval is one quarter of the e-folding time — 2 minutes now that the plume e-folds in about 8"}}]},
  {"op": "revise", "node_id": "c1d8f350",
   "reason": "The interval is derived from the e-folding time, which was refitted; the quarter-of-tau rule itself is unchanged.",
   "title": {"old": "one sample every 3 minutes", "new": "one sample every 2 minutes"},
   "content": {"old": "every 3 minutes through a flume tracer run — a quarter of the 12-minute e-folding time, giving four points inside each e-folding.", "new": "every 2 minutes through a flume tracer run — a quarter of the e-folding time, about 8 minutes at the 24 °C set point as of 2026-04-20 (3 minutes for the 12-minute e-folding fitted in January). Four points inside each e-folding is the rule; the interval follows the fit."}},
  {"op": "remember", "type": "plan",
   "title": "Nour's Wednesday demo run — a 40-minute flume slot for the undergraduate lab",
   "content": "Nour has a 40-minute flume slot for Wednesday's undergraduate lab: at an 8-minute e-folding time the students see about five e-foldings of the rhodamine plume, fitted with the same single exponential as a research run. Nothing is reported run yet.",
   "situation": "When Nour's Wednesday teaching slot, the undergraduate flume demo or its write-up comes up.",
   "question": "How long is Nour's undergraduate flume demo, and what do the students see in it?",
   "reasoning": "Nour's choice and its arithmetic on 2026-04-20; the slot is sized, the run is not reported done.",
   "their_raw_quote": "a 40-minute slot covers Wednesday's undergraduate lab … so they'd see five e-foldings",
   "event_time": "2026-04-20",
   "connect_to": [
     {"target": "4e7b1a92", "relation": "constrained_by", "why": "the 40 minutes is five e-foldings at the refitted 8-minute time constant; the demo is where the new value surfaced, and the plume node now carries it"},
     {"target": "c1d8f350", "relation": "follows", "why": "the demo samples on the quarter-of-tau cadence a research run uses, so the 2-minute interval governs it; a teaching slot earns no coarser autosampler schedule"}]}
]}
```

The flag said seen; the catalog said what was kept — 12 minutes and a 3-minute cadence, both from
January. The April sentence sat on the page through a whole run before this one and earned no line;
that run's silence is not a ruling, and neither is an arc line that calls the stretch transactional.
The `thought` is true and still not the move: the fit moved on every surface. My `sweep:` names
`4e7b1a92` and `c1d8f350`. My Arc line: `the flume's e-folding time refitted to 8 minutes at 24 °C;
the sampling cadence follows it`.

'''
tpl = base
tpl = replace(tpl,
    "\n\n### Other shapes this episode does not carry\n",
    "\n\n" + COVERED + "### Other shapes this episode does not carry\n",
    'E1 a worked window with covered turns: the covering run kept nothing from the changed value')
(HERE / 'template_example.md').write_text(tpl)
example_len = len(tpl)

# ══════════ EXAMPLE_B — variant for item 4 of the brief: a fourth Bad names the reconciliation move ══════════
# The transfer probe (PROBES.md § Transfer probe) showed the flume example does not move the gym windows: the
# encoder never sees a changed value because it reads the new words as consistent with the node by supplying a
# mechanism the speaker did not state ("departure time", "buffer"). No Bad in the prompt depicts that reading.
tpl_b = replace(tpl,
    "Three Bad moves, each one I have made.",
    "Four Bad moves, each one I have made.",
    'E2a four Bads')
tpl_b = replace(tpl_b,
    "is how the change slipped past the run before this one.",
    "is how the change slipped past the run before this one. Bad: reconcile — “eight minutes is what the students will see on a coarse demo fit; the research value stays twelve” — a reading that keeps the node by supplying a mechanism Nour did not state. The words say the plume e-folds in eight minutes now. When the only way to keep a node is a cause I invented, the node moves, not the words.",
    'E2b Bad 4: the reconciliation move — an invented mechanism keeps the node')
(HERE / 'template_example_b.md').write_text(tpl_b)
example_b_len = len(tpl_b)

# ══════════ STACK — gloss + example (lane 2: the example must not ship without the L39 gloss change) ══════════
tpl_s = replace(gloss_tpl,
    "\n\n### Other shapes this episode does not carry\n",
    "\n\n" + COVERED + "### Other shapes this episode does not carry\n",
    'S1 the covered-turn example on the gloss template')
(HERE / 'template_stack.md').write_text(tpl_s)
stack_len = len(tpl_s)

# ══════════════════════════ WALK — procedure, the gist ══════════════════════════
g = gist
g = replace(g,
    "`targets` — for each change, I walk EVERY catalog entry, every Edges line, and every id my continuity names.",
    "`targets` — for each change, I walk EVERY catalog entry, every Edges line, and every id my continuity names. Before that walk, one pass the other way: every entry tagged `encoded(me, turn N)` against the covered turns it came from — the flag says a run saw those turns, the entry says what it kept. A fact those turns state that no entry carries goes on `new`; an entry that reads a covered turn differently than the words do is `stale` on the surfaces that differ, a change in the same speaker's routine among them.",
    'W1 targets: the reverse pass, catalog entry → the covered turns it came from')
g = replace(g,
    "One line per first-disclosure fact the window carries:",
    "One line per first-disclosure fact the window carries, covered turns included — a covering run that wrote nothing for a turn left its facts to me:",
    'W2 new: covered turns included')
(HERE / 'gist_walk.md').write_text(g)
walk_len = len(g)

shutil.copyfile(PARENT / 'gist_full.md', HERE / 'gist_full.md')
(HERE / 'author_log.json').write_text(json.dumps(LOG, indent=1))
for row in LOG:
    print(f"{row['delta']:+6d}  {row['label']}")
print(f'base template {len(base):,}; gloss {gloss_len:,} ({gloss_len - len(base):+,}); example {example_len:,} ({example_len - len(base):+,}); example_b {example_b_len:,} ({example_b_len - len(base):+,}); stack {stack_len:,} ({stack_len - len(base):+,}); gist {len(gist):,} → walk {walk_len:,} ({walk_len - len(gist):+,})')
