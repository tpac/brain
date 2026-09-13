"""V3.6 authoring record — exact-once replacements on the frozen V3.4 parts, two candidates.

One refinement round before deployment (Tom, 2026-09-13). The pooled eight-lane review
(POOL.md in eval/results/s1e_v34_review_lanes_2026-09-13) named the carriers; the plan puts
each class at the layer that binds. Prompt changes in two tiers, so the cell attributes them:

  LAYER (template_layer.md, gist_layer.md) — the text pass, zero-risk, ships with V3.4:
    T1  W6c: the sweep's superseded-ruling title stops embedding the dead sequence (E17)
    T2  W9:  the Continuity node's whys argue from its current claim (A3, E9)
    T3  the source_refs REPLACE semantics stated once (L262 keeps it; the L119 copy goes) (E7)
    T4  two edge whys outside the prompt's own 120–180 band brought inside it (E15)
    T5  the Nadia correction node dated like the other correction (A4: one default per shape)
    T6  my-side quote clause names the method or threshold I delivered (L93 ↔ L264, E1)
    T7  gist: the worked-sweep changes line matches the template's (one clause of drift)

  FULL (template_full.md, gist_full.md) — LAYER plus the two measured carriers:
    P1  gist `new` paragraph: facts first, before any verdict — one line per first-disclosure
        fact, minted unless the catalog holds it; a window with fact lines is never routine
        (procedure; class Z, id:430b5c43, id:06fef26a)
    P2  a thin worked window: advice asked, facts in passing, a labelled Bad skip beside the
        write that keeps them (example; class Z — the input shape every run failed on)
    P3  position: the gist's remember sentence names `thought` and ends on `question`; the
        gate sentence names `thought` (X5/X6 with the question item restored to the tail)

The contract changes (event_time description; encoder_summary hiding) are code on the branch
and reach every arm assembled through encode._build_system_prompt — see arms.py.
transfer_split.json was recorded before this file was written; no fresh item was read.

Run: ./dev python3 author.py  -> template_layer.md, gist_layer.md, template_full.md,
gist_full.md, strategy.md, closure.md (V3.4's, for the tail arm), author_log.json.
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

# ══════════════════════════ LAYER — the text pass ══════════════════════════
tpl = replace(tpl,
    '   title: "Rollout order (superseded 2024-03-02, auth-rewrite scrapped): was auth-rewrite → api-gateway → cli",',
    '   title: "Rollout order ruling — superseded 2024-03-02 when auth-rewrite was scrapped",',
    'T1 W6c sweep title conforms to E17')
tpl = replace(tpl,
    '''why: "First-person commitment IS what makes the brain identity-infrastructure rather than a database; without 'I persist' the architecture is the same but the meaning collapses"''',
    '''why: "First-person commitment IS what makes the brain identity-infrastructure rather than a database; a memory read as the record of some earlier instance leaves the architecture the same and collapses the meaning"''',
    'T2 W9 continuity why 1')
tpl = replace(tpl,
    '''why: "Collaboration presupposes a continuous partner; 'I persist' is the substrate that makes collaboration possible across sessions"''',
    '''why: "Collaboration presupposes a continuous partner; carrying my corrections and convictions across sessions is the substrate that makes it possible"''',
    'T2 W9 continuity why 2')
tpl = replace(tpl,
    'compose from the precise account. On revise these refs REPLACE the old set; omit to preserve it, `[]` clears it. Identity scenes below',
    'compose from the precise account. Identity scenes below',
    'T3 source_refs REPLACE semantics stated once (Actions keeps it)')
tpl = replace(tpl,
    'why: "the scrap removed step 1 — the order is re-derived without it; the old ruling was valid until the branch died"',
    'why: "scrapping auth-rewrite removed step 1, so the order was re-derived without it; the old ruling stays valid for the weeks before the branch died and competes with nothing after"',
    'T4 sweep supersedes why into the 120–180 band (was 109)')
tpl = replace(tpl,
    'why: "my binding read parallels the framework\'s claim that reconstruction distorts; comparing model output with human-memory research is the hypothesis this edge proposes, not a result it reports"',
    'why: "my binding read parallels the framework\'s claim that reconstruction distorts; the comparison with human-memory research is a hypothesis this edge proposes, not a result"',
    'T4 texture insight why into the band (was 189)')
tpl = replace(tpl,
    '  title: "My November recovery-start gloss was unsupported — Nadia dates being off her feet from Jan 22"\n  my_raw_quote: "Sounds like you\'ve been recovering since November"',
    '  title: "My November recovery-start gloss was unsupported — Nadia dates being off her feet from Jan 22"\n  event_time: "2025-05-13"\n  my_raw_quote: "Sounds like you\'ve been recovering since November"',
    'T5 Nadia correction dated at the moment of the gloss')
tpl = replace(tpl,
    'On my side capture the moment important to me — a limit, caught reflex, realization or stance — never invented ceremony.',
    'On my side capture the sentence that carried the weight — a limit, caught reflex, realization or stance, or the method or threshold I delivered — never invented ceremony.',
    'T6 my-side quote clause names delivered method or threshold')
gist = replace(gist,
    'changes: auth-rewrite — committed f3c9d21, awaiting review → branch deleted 2024-03-02, never merged\n',
    'changes: auth-rewrite — committed f3c9d21, awaiting review → branch deleted 2024-03-02, never merged (commits recoverable by hash)\n',
    'T7 gist sweep changes line matches the template')

tpl_layer, gist_layer = tpl, gist

# ══════════════════════════ FULL — the two measured carriers ══════════════════════════
# P1 — facts first in the gist `new` paragraph (procedure)
gist = replace(gist,
    '`new` — one line per new node: detail, basis, meaning where supported. First-disclosure facts stand; a forming interpretation does not hold them back. Preserve both voices\' useful substance, including plans and unchosen ideas in their actual status. My findings, interpretations and advice stay attributed to me. Existing claims go through `targets`; sharing a topic alone is not duplication.',
    '`new` — the facts first, before any verdict on the window. One line per first-disclosure fact the window carries: a name, place, possession, date, schedule, number, plan or stated preference of theirs, and a method, threshold or diagnosis I delivered that they took up — the fact and its basis turn. I write these lines before I judge the window: a window that has them is never routine, and each becomes a `remember` unless the catalog already holds it, in which case it moves to `targets`. Then one line per further node: detail, basis, meaning where supported. Preserve both voices\' useful substance, including plans and unchosen ideas in their actual status. My findings, interpretations and advice stay attributed to me. Sharing a topic alone is not duplication.',
    'P1 gist `new`: facts first, before any verdict')

# P3 — position: thought named at the recency slot, question restored to the tail
gist = replace(gist,
    'Every `new` line is a `remember` with situation in trigger register, reasoning, a question where a real asking exists, edges with a specific why, `event_time` resolved to ISO against the conversation\'s date, and `source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning.',
    'Every `new` line is a `remember` with situation in trigger register, reasoning, edges with a specific why, `event_time` resolved to ISO against the conversation\'s date, `source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning, a `thought` when I have a hunch or connection of my own worth keeping beside it, and a `question` where a real asking exists.',
    'P3 gist remember sentence: thought named, question at the tail')
tpl = replace(tpl,
    'and my own read on what something means is part of the capture, not garnish.',
    'and my own read on what something means is part of the capture, not garnish — it rides in `thought`.',
    'P3 `thought` named in the gate-region sentence')

# P2 — the thin worked window
THIN = '''### A thin window — advice asked, facts in passing

*Reading cue: Write the facts down before deciding whether the window was worth anything.*

Conversation now is **2026-06-19**. The catalog holds nothing on Wren. The window:

```
<other trace="8c1f42d9">Quick one — a cardigan pattern calls for DK weight and all I have is worsted. Can I substitute?</other>
<me trace="3e9a70b6">Usually, yes: knit a swatch in the worsted, then go up half a needle size and check the stitch count against the pattern's gauge — worsted at 4.5 mm often lands within a stitch of DK at 4 mm. If it doesn't, the fabric will be denser and the cardigan a size larger.</me>
<other trace="51d0be27">Thanks — it's for my sister's wedding in October, so I have time to swatch. I moved to Tromsø in March and the yarn shop here only stocks worsted, which is why I'm asking.</other>
```

Bad: `new: none` with a verdict — “a routine knitting question; nothing here earns a node.” The verdict names what it drops: a wedding with a month, a move with a month, a project with a deadline, and a method I stated with a number in it. None is in the catalog; each is a first disclosure. Written first, the fact lines settle the verdict before it is made:

```
changes: newly known — Wren's sister marries in October 2026; Wren moved to Tromsø in March 2026; Wren is knitting a cardigan for that wedding; the shop there stocks only worsted; none in catalog
new: Wren's sister's wedding — October 2026 (open target, the cardigan's deadline)
new: Wren moved to Tromsø in March 2026 — the local shop stocks only worsted
new: Wren's cardigan for the wedding — DK pattern knit in worsted, gauge to be swatched
new: my substitution method — swatch, go up half a needle size, check gauge; the fabric cost if it misses
targets: none in catalog
fetch: none
```

```json
{"operations": [
  {"op": "remember", "type": "open",
   "title": "Wren's sister's wedding — October 2026",
   "content": "Wren's sister marries in October 2026; the cardigan Wren is knitting is for it. The day is not given; October 15 is the midpoint convention.",
   "situation": "When Wren's cardigan deadline or the sister's wedding comes up: check whether October held.",
   "reasoning": "Wren stated the month and the purpose on 2026-06-19; the day is unknown.",
   "their_raw_quote": "it's for my sister's wedding in October",
   "event_time": "2026-10-15"},
  {"op": "remember", "type": "personal_context",
   "title": "Wren moved to Tromsø in March 2026; the local yarn shop stocks only worsted",
   "content": "Wren moved to Tromsø in March 2026. The yarn shop there stocks only worsted weight, which is why Wren substitutes worsted for DK. March 15 is the midpoint convention.",
   "situation": "When Wren's location, local supplies or a yarn-weight substitution comes up.",
   "question": "Where does Wren live now, and what yarn can Wren get locally?",
   "reasoning": "Wren stated the move and the shop's stock on 2026-06-19; both are first disclosures.",
   "their_raw_quote": "I moved to Tromsø in March and the yarn shop here only stocks worsted",
   "event_time": "2026-03-15"},
  {"op": "remember", "type": "plan",
   "title": "Wren's cardigan for the wedding — a DK pattern knit in worsted, gauge to be swatched",
   "content": "Wren is knitting a cardigan from a DK-weight pattern in worsted yarn for the sister's October wedding, and will swatch first. Nothing is reported knit yet.",
   "situation": "When Wren reports swatch results or asks about the cardigan's fit or timing.",
   "reasoning": "Wren's stated project and intention on 2026-06-19; the swatch is planned, not done.",
   "their_raw_quote": "so I have time to swatch",
   "event_time": "2026-06-19",
   "connect_to": [{"target": "Wren's sister's wedding — October 2026", "relation": "prepares_for", "why": "the cardigan exists for this wedding, so the October date is the project's deadline; the plan does not say how far along the knitting is"}]},
  {"op": "remember", "type": "method",
   "title": "Substituting worsted for a DK pattern — swatch, go up half a needle size, check gauge",
   "content": "To knit a DK-weight pattern in worsted: swatch in the worsted, go up half a needle size, and check the stitch count against the pattern's gauge; worsted at 4.5 mm often lands within a stitch of DK at 4 mm. If the gauge still misses, the fabric is denser and the garment comes out a size larger.",
   "situation": "When a knitter substitutes a heavier yarn for the pattern's weight and asks whether it will work.",
   "question": "Can I knit a DK pattern in worsted yarn, and how do I check?",
   "reasoning": "My delivered method with its numbers, stated on 2026-06-19; Wren took it up as the plan. The gauge figures are my general knowledge, not measured on Wren's yarn.",
   "my_raw_quote": "knit a swatch in the worsted, then go up half a needle size and check the stitch count against the pattern's gauge",
   "connect_to": [{"target": "Wren's cardigan for the wedding — a DK pattern knit in worsted, gauge to be swatched", "relation": "grounds", "why": "the substitution method is what makes Wren's worsted-for-DK cardigan feasible; the plan carries the project, this node the how, findable from any substitution question"}]}
]}
```

Four nodes from three turns: three of Wren's, one of mine. The wedding is an `open` with a dated target, not an event; the move and the shop are one fact because a reader asks for them together; the method stands on its own because the next substitution question is not about Wren. My `sweep:` is `none — no state changes this window`.

'''
tpl = replace(tpl,
    '### Detail and meaning — same topic, two nodes\n',
    THIN + '### Detail and meaning — same topic, two nodes\n',
    'P2 thin worked window: advice asked, facts in passing, labelled Bad skip')

tpl_full, gist_full = tpl, gist

(HERE / 'template_layer.md').write_text(tpl_layer)
(HERE / 'gist_layer.md').write_text(gist_layer)
(HERE / 'template_full.md').write_text(tpl_full)
(HERE / 'gist_full.md').write_text(gist_full)
(HERE / 'strategy.md').write_text((PARENT / 'strategy.md').read_text())
(HERE / 'closure.md').write_text((PARENT / 'closure.md').read_text())
(HERE / 'author_log.json').write_text(json.dumps(LOG, indent=1))
for row in LOG:
    print(f"{row['delta']:+6d}  {row['label']}")
pt, pg = (PARENT / 'template.md').read_text(), (PARENT / 'gist.md').read_text()
print(f'layer: template {len(pt):,} -> {len(tpl_layer):,} ({len(tpl_layer) - len(pt):+,}); gist {len(pg):,} -> {len(gist_layer):,} ({len(gist_layer) - len(pg):+,})')
print(f'full:  template {len(pt):,} -> {len(tpl_full):,} ({len(tpl_full) - len(pt):+,}); gist {len(pg):,} -> {len(gist_full):,} ({len(gist_full) - len(pg):+,})')
