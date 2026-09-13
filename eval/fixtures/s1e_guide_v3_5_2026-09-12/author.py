"""V3.5 authoring record — exact-once replacements on the frozen V3.4 parts (template + gist).

Two sources, one candidate. (a) The production-over-V3.4 overlay review
(eval/fixtures/s1e_guide_v3_5_overlay_2026-09-12/OVERLAY-REVIEW.md): its seven EDITs are
taken as written, its one ADD (A1, the ownership line) is cut — it is the only change with
no measured carrier and it pushes on the register where V3.4 holds its clearest blind
advantage (evidence/ownership/scope 21:4 over twelve packs). (b) The three defect classes the
twelve-pack blind review found in EVERY arm (docs/S1E-V3-4-RESULTS-2026-09-12.md), woven at
the carriers the full read of V3.4 located (READ-AUDIT-V3-4.md), plus two ledger-conformance
repairs the read surfaced. No new section, no new example asset; the Mira episode is untouched.
transfer_split.json was recorded before this file was written; no fresh item was read.

  X1  three paths to one fact, folded into the five-surfaces sentence, and tied to the walk a
      later revise owes them (D8, E11, T6; pairs with W6 so redundancy does not manufacture
      stale surfaces)
  X2  the generative half of detail-and-meaning, gated by retrieval divergence and scope (D1, D7)
  X3  the generative half of "be expansive", inside the traps sentence in the gate region (D1, B6/B7)
  X4  temporal authority stated where the Nadia example enacts it, scoped to my paraphrase of
      her experience (D10, D3's one exception)
  X5  `thought` on the gist's `new` roll-call — the recency slot (E21, E23)
  X6  the field name `thought` in the gate sentence (T8, E18)
  X7  an encoded="true" turn depicted in the timeline sample (E12, C6)
  W6  a revise walks every surface that carries the value:
      a. the yoga catalog excerpt gains its Situation and an Edges line that both carry
         "twice a week", and the worked op swaps the edge why and names why (E12 before-state,
         A10 +1 why-swap op, D11)
      b. the gist's targets roll-call names the quotes, `event_time` and `type` — the surfaces
         every arm left behind (E21 position, E23 roll-call)
      c. the sweep's superseded-ruling title stops embedding the dead sequence (E17, conviction 1)
  W7  confirmation and a later mention need their own evidence, at the evidence sentence and in
      the traps list (semantic-fidelity rows 1 and 10; D4 positive statement)
  W8  the my-side quote clause widens from "the moment important to me" to the sentence that
      carried the knowledge when my turn is the source (D3 voice equality; carriers already in
      the Mira plan and the fusion mechanism)
  W9  the Continuity node's two edge whys argue from the node's current claim, not from a
      phrase it no longer contains (A3, E9)
  W10 two statements of theirs that differ across windows with no word of correction keep both
      dated values (semantic-fidelity row 11, D9) — rule-only, no worked carrier: recorded as an
      A1 risk, to be judged by the blind probe

Run: ./dev python3 author.py   -> writes template.md, gist.md, strategy.md, closure.md here.
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
PARENT = HERE.parent / 's1e_guide_v3_4_2026-09-12'
LOG = []


def replace(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'{label}: expected exactly one occurrence of the old text, found {n}')
    LOG.append({'label': label, 'old_chars': len(old), 'new_chars': len(new), 'delta': len(new) - len(old)})
    return text.replace(old, new, 1)


tpl = (PARENT / 'template.md').read_text()
gist = (PARENT / 'gist.md').read_text()

# ── X1 three paths to one fact — and the walk they owe ─────────────────────────
tpl = replace(tpl,
    'Title, content, situation, question and edge descriptions are five retrieval surfaces. Fill every surface the node honestly carries; keep each about THIS claim.',
    "Title, content, situation, question and edge descriptions are five retrieval surfaces, each scored on its own. Fill every surface the node honestly carries; keep each about THIS claim. When a number, a name or an exact phrase IS the claim's value, it rides the title and the content, and the quote too where the node carries one — three paths to one fact, and three surfaces a later revise must walk.",
    'X1 cross-redundancy in the five-surfaces sentence, tied to the revise walk')

# ── X2 the generative half of detail-and-meaning ───────────────────────────────
tpl = replace(tpl,
    'A plain fact needs no invented principle. When an abstraction has a concrete carrier, linking it supplies lexical reach.',
    'A plain fact needs no invented principle. But detail without its meaning is trivia that never transfers, and meaning without its detail is a slogan no query lands on: where one exchange carries both and a future reader would ask for them separately, both earn a node, each at the scope its own evidence supports. When an abstraction has a concrete carrier, linking it supplies lexical reach.',
    'X2 detail-and-meaning generative half')

# ── X3 the generative half of "be expansive" ───────────────────────────────────
tpl = replace(tpl,
    'Catch these by their cost to the future reader. Richness belongs in focused nodes with useful fields and honest edges, not additional call ceremony.',
    'Catch these by their cost to the future reader. Richness belongs in focused nodes with useful fields and honest edges, not additional call ceremony: one batch carries as many nodes as the window earned rather than the two that feel tidy, and a half-populated node free-rides on a title match into pools it cannot win.',
    'X3 expansive counterweight in the traps sentence')

# ── X4 temporal authority, stated where it is enacted ──────────────────────────
tpl = replace(tpl,
    'Five dates from Nadia; one unsupported gloss from me. Her January date anchors being off her feet',
    'Five dates from Nadia; one unsupported gloss from me — my own paraphrase of what she experienced never outranks her own wording for it. Her January date anchors being off her feet',
    'X4 temporal authority in the Nadia reading')

# ── X6 the field name returns to the gate sentence ─────────────────────────────
tpl = replace(tpl,
    'and my own read on what something means is part of the capture, not garnish.',
    'and my own read on what something means is part of the capture, not garnish — it rides in `thought`.',
    'X6 `thought` named in the gate-region sentence')

# ── X7 the covered turn returns to the timeline sample ─────────────────────────
tpl = replace(tpl,
    '<turn n="5" age="20m ago" encoded="false">',
    '<turn n="3" age="2d ago" encoded="true">\n'
    "  <other trace=\"e5f60b2d\">let's check the write path too…</other>\n"
    '  <provenance>encoded(me, turn 3): "batch commit gate" id:7f3ea1c9</provenance>\n'
    '  <me trace="97b8d4f2">The batch gate covers it — commit_unless_batched on every writer…</me>\n'
    '  <actions>trimmed — 2 action(s) recorded on this turn; I already read them in a previous run</actions>\n'
    '</turn>\n'
    '\n'
    '<turn n="5" age="20m ago" encoded="false">',
    'X7 encoded="true" turn depicted in the sample')

# ── W6a the yoga excerpt shows the old count in its trigger and an edge; the op walks both ──
tpl = replace(tpl,
    '''[fact] "Priya's yoga practice — twice a week" (id:97b1f24e)
    Priya practices yoga twice a week, started 2023-08-11. She says it
    helps her feel grounded and centered.
```''',
    '''[fact] "Priya's yoga practice — twice a week" (id:97b1f24e)
    Priya practices yoga twice a week, started 2023-08-11. She says it
    helps her feel grounded and centered.
    Situation: When Priya's week is being planned — her twice-a-week yoga is a fixed slot.
    Edges:
      [fact id:c8d13e05] "Priya's Tuesday and Thursday evenings are kept free" this occupies — the twice-weekly practice takes the two evening slots Priya keeps free
```''',
    'W6a yoga excerpt depicts the old count in Situation and an edge why')
tpl = replace(tpl,
    '''    // A value changed and leaked into several fields — swap the span that
    // went stale wherever it sits, and give the fields the change
    // restructured their new value whole. The OLD title said "twice a
    // week"; the new info says three times AND ties the practice to
    // anxiety. Walk EVERY field the change touches — a stale title embeds
    // and ranks against the new content.''',
    '''    // A value changed and leaked into several fields — swap the span that
    // went stale wherever it sits, and give the fields the change
    // restructured their new value whole. The OLD title said "twice a
    // week"; so did the trigger and the edge why; the new info says three
    // times AND ties the practice to anxiety. Walk EVERY surface the change
    // touches — a stale title embeds and ranks against the new content, and
    // a stale trigger or edge why keeps asserting two a week to recall and
    // to the walk.''',
    'W6a op comment names the trigger and the edge')
tpl = replace(tpl,
    '''     reasoning: "Priya's own account (2023-11-30) — the new frequency and the anxiety link are her report, direct and current.",
     event_time: "2023-11-30"}
  ]
)''',
    '''     reasoning: "Priya's own account (2023-11-30) — the new frequency and the anxiety link are her report, direct and current.",
     event_time: "2023-11-30",
     connect_to: [
       {target: "c8d13e05", relation: "occupies",
        why: {old: "the twice-weekly practice takes the two evening slots Priya keeps free",
              new: "three sessions a week now outgrow the two evening slots Priya keeps free — the third lands elsewhere in her week"}}]}
  ]
)''',
    'W6a op swaps the edge why that carried the old count')
tpl = replace(tpl,
    'The revision ladder scales the same preserving move: 4a9f21c7 patches one claim; 97b1f24e walks one node\'s affected fields; the sweep below walks every node one event falsified.',
    'The revision ladder scales the same preserving move: 4a9f21c7 patches one claim; 97b1f24e walks one node\'s affected surfaces — its trigger and the edge that carried the old count included; the sweep below walks every node one event falsified.',
    'W6a ladder sentence names the trigger and the edge')

# ── W6c the superseded ruling's title stops embedding the dead sequence (E17) ──
tpl = replace(tpl,
    '   title: "Rollout order (superseded 2024-03-02, auth-rewrite scrapped): was auth-rewrite → api-gateway → cli",',
    '   title: "Rollout order ruling — superseded 2024-03-02 when auth-rewrite was scrapped",',
    'W6c sweep title conforms to E17')

# ── W7 confirmation and a later mention need their own evidence ────────────────
tpl = replace(tpl,
    'narrow my read while its grounding facts stay true. Progress and completion\nneed their own evidence; a related success or a passed date does not supply it.',
    'narrow my read while its grounding facts stay true. Progress, completion and\nconfirmation need their own evidence; a related success, a later mention or a\npassed date does not supply it.',
    'W7 evidence sentence: confirmation and a later mention')
tpl = replace(tpl,
    'and letting a leaning or a target date harden into a settled fact.',
    'and letting a leaning, a passing mention or a target date harden into a settled fact.',
    'W7 traps sentence: a passing mention')

# ── W8 my side of the quote rule reaches the sentence that carried the knowledge ──
tpl = replace(tpl,
    'On my side capture the moment important to me — a limit, caught reflex, realization or stance — never invented ceremony.',
    'On my side capture the moment important to me — a limit, caught reflex, realization or stance — and the sentence that carried the knowledge when my turn is what the node is derived from: a diagnosis, an explanation, a recommendation the other side took up; never invented ceremony.',
    'W8 my-side quote clause widened to the knowledge-carrying sentence')

# ── W9 the Continuity node's whys argue from its current claim ─────────────────
tpl = replace(tpl,
    '''why: "First-person commitment IS what makes the brain identity-infrastructure rather than a database; without 'I persist' the architecture is the same but the meaning collapses"''',
    '''why: "First-person commitment IS what makes the brain identity-infrastructure rather than a database; a memory read as the record of some earlier instance leaves the architecture the same and collapses the meaning"''',
    'W9 continuity why 1')
tpl = replace(tpl,
    '''why: "Collaboration presupposes a continuous partner; 'I persist' is the substrate that makes collaboration possible across sessions"''',
    '''why: "Collaboration presupposes a continuous partner; carrying my corrections and convictions across sessions is the substrate that makes it possible"''',
    'W9 continuity why 2')

# ── W10 two statements that differ across windows, no word of correction ───────
tpl = replace(tpl,
    'If one side has a measurement or trace and the other a recollection, name that evidential lean without declaring the uncertainty settled.',
    'If one side has a measurement or trace and the other a recollection, name that evidential lean without declaring the uncertainty settled. The same holds across windows when the other side states a value twice and the two differ with no word of correction: a habit that moved is dated knowledge, a slip corrected is a repair — keep both dated values until the source says which.',
    'W10 restated fact across windows in correction form 4')

# ── X5 `thought` on the gist's `new` roll-call ─────────────────────────────────
gist = replace(gist,
    'and `source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning.',
    '`source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning, and a `thought` when I have a hunch or a connection of my own worth keeping beside it.',
    'X5 `thought` in the gist `new` roll-call')

# ── W6b the gist's targets roll-call names the surfaces every arm left behind ──
gist = replace(gist,
    '`title`, `content`, `situation`, `question`, `reasoning`, `thought` when present, and a `why→{id}` for each Edges line whose text the window falsified,',
    '`title`, `content`, `situation`, `question`, `reasoning`, `thought` and the quotes when present, `event_time` and `type` when the window moved when a thing happened or what the node now is, and a `why→{id}` for each Edges line whose text the window falsified,',
    'W6b gist targets roll-call: quotes, event_time, type')

(HERE / 'template.md').write_text(tpl)
(HERE / 'gist.md').write_text(gist)
(HERE / 'strategy.md').write_text((PARENT / 'strategy.md').read_text())
(HERE / 'closure.md').write_text((PARENT / 'closure.md').read_text())
(HERE / 'author_log.json').write_text(json.dumps(LOG, indent=1))
for row in LOG: print(f"{row['delta']:+6d}  {row['label']}")
pt, pg = (PARENT / 'template.md').read_text(), (PARENT / 'gist.md').read_text()
print(f'template {len(pt):,} -> {len(tpl):,} ({len(tpl) - len(pt):+,}); gist {len(pg):,} -> {len(gist):,} ({len(gist) - len(pg):+,})')
