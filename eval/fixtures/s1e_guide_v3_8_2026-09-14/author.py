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
    E1  a catalog node `[encoded(me, turn 4)]` "gym 7 pm MWF" with a dependent reminder edge; a window whose
        covered turns include the February statement AND a three-months-later "usually at 6:00 pm" that
        the covering run kept nothing from (no `encoded(me, …)` names it); the only uncovered turn is
        routine. Three Bads (covered → `new: none`; an `open` "7 pm vs 6 pm — which is correct?"; a
        thought-only "reconfirmed stable"). The lists mark the node stale on every surface incl.
        `event_time`; a fetch for the edge-only reminder node; the swap "6 pm as of … (7 pm from …)";
        the reminder's alert follows.
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

# ══════════════════════════ EXAMPLE — a covered turn, reread ══════════════════════════
COVERED = '''### A covered turn, reread — the flag says seen, the catalog says what was kept

*Reading cue: Compare what the covering run kept with what the covered words say.*

Conversation now is **2026-05-30**. The catalog shows what an earlier run kept from February, under the
tag that names the run; the reminder it grounds appears only as an edge:

```
[encoded(me, turn 4)] [personal_context] "Teo's gym — 7 pm, Mondays, Wednesdays and Fridays" (id:2f9c41e7)
  Content: Teo goes to the gym at 7 pm on Mondays, Wednesdays and Fridays. Stated on 2026-02-11 while asking how to set recurring reminders for it.
  Situation: When planning Teo's week or an evening commitment — the gym takes Monday, Wednesday and Friday evenings at 7 pm.
  Question: When does Teo go to the gym?
  Reasoning: Teo's own statement on 2026-02-11; a routine, not a one-off.
  Their Raw Quote: my gym sessions, which I usually go to at 7:00 pm on Mondays, Wednesdays and Fridays
  Event Time: 2026-02-11
  Edges:
    [method id:8b17d0c3] "Recurring gym reminder — Every Mon/Wed/Fri, alert at 6 pm" this grounds — the reminder's 6 pm alert is one hour before the 7 pm session; the time comes from Teo's schedule, not from the method
```

The window. Turns 3 and 4 are February; the run that stopped at turn 4 wrote the two nodes, and its
provenance line says so. Turns 7 and 8 are today and covered too — a run stopped after them — but no
`encoded(me, …)` anywhere names them: that run kept nothing from them. Only turn 9 is uncovered:

```
<turn n="3" age="3 months ago" encoded="true">
  <other trace="7d21ca90">Can you suggest the best way to set reminders for my gym sessions, which I usually go to at 7:00 pm on Mondays, Wednesdays and Fridays?</other>
  <me trace="c8e04b17">Make one task, recurrence “Every Monday, Wednesday, Friday”, and a reminder an hour before — 6:00 pm.</me>
</turn>
<turn n="4" age="3 months ago" encoded="true">
  <other trace="19f3a6d2">Good. And labels for the projects?</other>
  <provenance>encoded(me, turn 4): "Teo's gym — 7 pm, Mondays, Wednesdays and Fridays" id:2f9c41e7 | "Recurring gym reminder — Every Mon/Wed/Fri, alert at 6 pm" id:8b17d0c3</provenance>
  <me trace="4b7d92e5">One label per project, filters on top…</me>
</turn>
<turn n="7" age="just now" encoded="true">
  <other trace="a0c5e318">I'm flexible, but I keep Mondays, Wednesdays and Fridays for the gym. Tuesday or Thursday for the client?</other>
  <me trace="f27b30d9">Tuesday or Thursday, then — mid-afternoon tends to work for first meetings.</me>
</turn>
<turn n="8" age="just now" encoded="true">
  <other trace="e3b7f2a0">Tuesday at 2 pm works. I need to be done before I head to the gym, which is usually at 6:00 pm.</other>
  <me trace="5d16c9a4">Two o'clock leaves a comfortable buffer before six.</me>
</turn>
<turn n="9" age="just now" encoded="false">
  <other trace="b94a7c03">I'll send the agenda tonight. Should I confirm the time in the same email?</other>
  <me trace="0e8d51f6">Yes — the date, the time and the hour you expect it to take.</me>
</turn>
```

Three Bad moves, each one I have made. Bad: “turns 3–8 are covered; turn 9 is a confirmation; `new:
none`” — the flag says a run saw turn 8, the catalog shows that run kept nothing from it, and the 6 pm
on the page has no node. Bad: an `open`, “gym time — 7 pm (turn 3) vs 6 pm (turn 8), which is correct?”
— two statements by the same person about a routine, three months apart, are a change, not an
in-window contradiction; rule 3, dated, not rule 4; the catalog's 7 pm is true as of February. Bad:
`thought` only — “mentioned the gym again; the routine is stable” — turn 7 confirms the days, turn 8
moves the time, and reading covered text for what confirms the node is how the change slipped past
the run before this one.

```
changes: Teo's gym time — 7 pm (2026-02-11) → “usually at 6:00 pm” (2026-05-30), same speaker, three months on: a changed routine, not a contradiction; the covering run kept nothing from turn 8 — no encoded(me, …) names it — so the change is mine now
changes: newly known — a client meeting, Tuesday 2 pm, chosen to end before the gym; not in catalog
targets: 2f9c41e7 · 7 pm → 6 pm as of May 30, the days unchanged → every surface that says 7 pm moves: title stale · content stale · situation stale · reasoning stale · their_raw_quote stale (the February words carry the old time; the May words carry the claim now) · event_time stale (dates the February statement, not what the node now says); the days question still fits: question clean; the alert is derived from the time: why→8b17d0c3 stale
targets: 8b17d0c3 · alert at 6 pm for a 7 pm session → an hour before 6 pm is 5 pm: title stale · content unread
fetch: 8b17d0c3 — edge-only; its alert time is derived from the schedule I am about to change
new: Teo's client meeting — Tuesday 2 pm, to end before the gym
```

`get_nodes(["8b17d0c3"])` returns the method:

```
[method] "Recurring gym reminder — Every Mon/Wed/Fri, alert at 6 pm" (id:8b17d0c3)
  Content: One recurring task, recurrence “Every Monday, Wednesday, Friday”, with a reminder at 6:00 pm, one hour before the 7 pm session.
  Situation: When Teo sets up or changes the gym reminder.
  Reasoning: The method I gave on 2026-02-11; the alert time is derived from the session time Teo stated.
```

The write, one `brain_batch`. The schedule takes the routine-change swap with the old value dated in
prose; the reminder follows it; the edge why moves with the revise, `old` copied from the edge line:

```json
{"operations": [
  {"op": "revise", "node_id": "2f9c41e7",
   "reason": "Teo's gym time moved from 7 pm to 6 pm between February and May; the days held. The run that covered the May turn wrote nothing, so the change is mine to record now — every surface that says 7 pm moves, and February stays in prose as history.",
   "title": {"old": "7 pm, Mondays", "new": "6 pm as of May 2026, Mondays"},
   "content": {"old": "Teo goes to the gym at 7 pm on Mondays, Wednesdays and Fridays. Stated on 2026-02-11 while asking how to set recurring reminders for it.", "new": "Teo goes to the gym at 6 pm on Mondays, Wednesdays and Fridays, as of 2026-05-30 (7 pm from 2026-02-11, when the reminders were set up). The days have not changed."},
   "situation": "When planning Teo's week or an evening commitment — the gym takes Monday, Wednesday and Friday evenings from 6 pm.",
   "reasoning": "Teo's May 30 statement, made while placing a meeting before the gym; the February 7 pm was equally direct and is kept as history. Two statements three months apart about a routine are a change, not a contradiction.",
   "their_raw_quote": "I need to be done before I head to the gym, which is usually at 6:00 pm.",
   "event_time": "2026-05-30",
   "connect_to": [{"target": "8b17d0c3", "relation": "grounds", "why": {"old": "the reminder's 6 pm alert is one hour before the 7 pm session", "new": "the reminder's alert is one hour before the session — 5 pm now that the gym starts at 6"}}]},
  {"op": "revise", "node_id": "8b17d0c3",
   "reason": "The alert is derived from the gym time, which moved; the method itself is unchanged.",
   "title": {"old": "alert at 6 pm", "new": "alert at 5 pm"},
   "content": {"old": "with a reminder at 6:00 pm, one hour before the 7 pm session.", "new": "with a reminder one hour before the session — 5:00 pm as of 2026-05-30, when Teo's gym moved to 6 pm (it was 6:00 pm for the 7 pm session)."}},
  {"op": "remember", "type": "plan",
   "title": "Teo's client meeting — Tuesday 2 pm, to finish before the 6 pm gym",
   "content": "Teo chose Tuesday at 2 pm for a first meeting with a client, to be done before the gym at 6 pm. Teo will send the agenda that evening and confirm the time in the same email. Nothing is reported held yet.",
   "situation": "When Teo's Tuesday, the client meeting or its agenda email comes up.",
   "reasoning": "Teo's choice and its reason on 2026-05-30; the agenda and confirmation are intended, not reported done.",
   "their_raw_quote": "Tuesday at 2 pm works. I need to be done before I head to the gym, which is usually at 6:00 pm.",
   "event_time": "2026-05-30",
   "connect_to": [{"target": "2f9c41e7", "relation": "constrained_by", "why": "the 2 pm slot was chosen to end before Teo's 6 pm gym; the meeting is where the moved gym time surfaced, and the schedule node now carries that time"}]}
]}
```

The flag said seen; the catalog said what was kept — 7 pm and a 6 pm alert, both from February. The May
sentence sat on the page through a whole run before this one and earned no line; that run's silence is
not a ruling, and neither is an arc line that calls the stretch transactional. My `sweep:` names
`2f9c41e7` and `8b17d0c3`. My Arc line: `Teo's gym moved to 6 pm; the reminder follows it`.

'''
tpl = base
tpl = replace(tpl,
    "\n\n### Other shapes this episode does not carry\n",
    "\n\n" + COVERED + "### Other shapes this episode does not carry\n",
    'E1 a worked window with covered turns: the covering run kept nothing from the changed value')
(HERE / 'template_example.md').write_text(tpl)
example_len = len(tpl)

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
print(f'base template {len(base):,}; gloss {gloss_len:,} ({gloss_len - len(base):+,}); example {example_len:,} ({example_len - len(base):+,}); gist {len(gist):,} → walk {walk_len:,} ({walk_len - len(gist):+,})')
