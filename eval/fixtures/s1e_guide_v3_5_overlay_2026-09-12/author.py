"""V3.5 PRODUCTION-OVERLAY candidate — exact-once replacements on the frozen V3.4 parts.

A reviewable candidate, not a release. It answers one question: laying the deployed
production S1E prompt on top of frozen V3.4, which production passages that V3.4
dropped or reduced should come back, and in what form? Every decision — ADD, EDIT
and REJECT — is recorded in OVERLAY-REVIEW.md beside its evidence.

Eight changes: one ADD and seven EDITs. Every one either folds production substance
into a sentence V3.4 already carries (E16: new teaching rides an existing
explanation) or restores a depiction an existing V3.4 rule needs (E12). No new
bullet, no new section, no new example asset. The Mira episode is untouched; every
V3.3/V3.4 weave survives (quote moves with the fact; question on the remembers;
Skip sentence; refused-op sentence; hunch thought on the plan revise).

  A1  Ownership at the head of Actions — production's "I am the source" opener,
      the one true addition. (E19 ownership audit; B1 named-position attention.)
  X1  Cross-redundancy for facts, folded into the five-surfaces sentence, with the
      consequence E11 asks a menu to attach. Production taught it twice; V3.4 zero.
  X2  The generative half of detail-and-meaning, restored beside V3.4's restraint
      half and gated by scope. (D1; the measured folding stance.)
  X3  The generative half of "be expansive", folded into the traps sentence in the
      gate region. (D1; B6/B7.)
  X4  The temporal-authority rule stated where the Nadia example already enacts it,
      scoped to paraphrase-of-their-experience. (D10; D3's one exception.)
  X5  `thought` enters the gist's `new`-line field roll-call — the recency slot,
      which is the measured lever for this field, not the template. (E21; 7bdb1c25.)
  X6  The field name `thought` returns to the gate-region sentence that names the
      act without naming the surface. (T8; A1/E23.)
  X7  An `encoded="true"` turn returns to the timeline sample, so the rule below it
      has its trigger visible in the depicted input. (E12; C6.)

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
    LOG.append({'label': label, 'old_chars': len(old), 'new_chars': len(new),
                'delta': len(new) - len(old)})
    return text.replace(old, new, 1)


tpl = (PARENT / 'template.md').read_text()
gist = (PARENT / 'gist.md').read_text()

# -- A1 ownership at the head of Actions -------------------------------------
# Production opens `## Actions` with the ownership line and only then reaches
# mechanics; V3.4 opens with mechanics. E19: a surface the agent does not
# experience as its own is one it will not repair. Marked deliberate
# reinforcement of the opener (E7), because it lands at the write moment.
tpl = replace(tpl,
    'The catalog is a view, not the whole brain. Read what the decision lacks:',
    "I am the source — the graph's shape this turn is my call. The catalog is a view, not the whole brain. Read what the decision lacks:",
    'A1 ownership line opens Actions')

# -- X1 cross-redundancy for facts -------------------------------------------
# Production carries this twice (the canonical field roll-call and the
# "Numbers cross-redundant" demonstration bullet); V3.4 carries it zero times.
# D8 names it a measured lever. Folded into the five-surfaces sentence, with
# the per-surface consequence E11 asks a menu to attach.
tpl = replace(tpl,
    'Title, content, situation, question and edge descriptions are five retrieval surfaces. Fill every surface the node honestly carries; keep each about THIS claim.',
    "Title, content, situation, question and edge descriptions are five retrieval surfaces, each scored on its own. Fill every surface the node honestly carries; keep each about THIS claim. When a number, a name or an exact phrase IS the claim's value, it rides the title and the content, and the quote too where the node carries one — three paths to one fact.",
    'X1 cross-redundancy in the five-surfaces sentence')

# -- X2 the generative half of detail-and-meaning ----------------------------
# V3.4 kept the restraint half ("a plain fact needs no invented principle") and
# dropped the generative half ("the pair is the unit"). D1: a restraint rule
# without an equally prominent generative counterpart over-generalizes. The
# restored half is gated by the retrieval-divergence test and by scope, so it
# cannot become production's standing "and the principle or concept each one
# points to" — the clause behind production's lesson/principle/framework
# surplus on assistant-heavy material. Production's "roughly half the rate"
# figure is deliberately NOT imported: no eval behind it (A9).
tpl = replace(tpl,
    'A plain fact needs no invented principle. When an abstraction has a concrete carrier, linking it supplies lexical reach.',
    'A plain fact needs no invented principle. But detail without its meaning is trivia that never transfers, and meaning without its detail is a slogan no query lands on: where one exchange carries both and a future reader would ask for them separately, both earn a node, each at the scope its own evidence supports. When an abstraction has a concrete carrier, linking it supplies lexical reach.',
    'X2 detail-and-meaning generative half')

# -- X3 the generative half of "be expansive" --------------------------------
# Production's bolded "Be expansive here" block is gone from V3.4, which keeps
# only its restraint tail ("not additional call ceremony"). The two clauses
# that carry behaviour — one batch holds what the window earned, and the cost
# of a half-populated node — fold into that same sentence, in the gate region
# B6 measured as the prompt's centre of gravity (B7: ride the carrier, do not
# bolt on). Production's "I don't ration" is deliberately NOT imported.
tpl = replace(tpl,
    'Catch these by their cost to the future reader. Richness belongs in focused nodes with useful fields and honest edges, not additional call ceremony.',
    'Catch these by their cost to the future reader. Richness belongs in focused nodes with useful fields and honest edges, not additional call ceremony: one batch carries as many nodes as the window earned rather than the two that feel tidy, and a half-populated node free-rides on a title match into pools it cannot win.',
    'X3 expansive counterweight in the traps sentence')

# -- X4 temporal authority, stated where it is enacted -----------------------
# The Nadia example enacts the rule and never states it; D10 carries it as a
# standing law and D3 records it as voice equality's one scoped exception.
# Scoped deliberately to MY PARAPHRASE of THEIR experience, so it cannot be
# read as "a later statement of theirs corrects an earlier one" — the reading
# every arm of the V3.4 cell made on the tennis item.
tpl = replace(tpl,
    'Five dates from Nadia; one unsupported gloss from me. Her January date anchors being off her feet',
    'Five dates from Nadia; one unsupported gloss from me — my own paraphrase of what she experienced never outranks her own wording for it. Her January date anchors being off her feet',
    'X4 temporal authority in the Nadia reading')

# -- X6 the field name returns to the gate-region sentence -------------------
# V3.4 names the act ("my own read ... is part of the capture") and hides the
# surface. T8: an English synonym for an act whose field exists breaks the
# intent-to-field lookup exactly there.
tpl = replace(tpl,
    'and my own read on what something means is part of the capture, not garnish.',
    'and my own read on what something means is part of the capture, not garnish — it rides in `thought`.',
    'X6 `thought` named in the gate-region sentence')

# -- X7 the covered turn returns to the timeline sample ----------------------
# V3.4 states the encoded="true" rule in the paragraph below the sample and
# depicts only an encoded="false" turn. E12: an example teaches recognition
# only if its input shows the thing to be recognised — the trimmed-actions
# stub and the encoded(me, turn N) provenance form are both invented notations
# (C6) that the prose names and the depiction does not show.
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

# -- X5 `thought` enters the gist's `new`-line field roll-call ---------------
# The position lever. The template paragraph moved the count by nothing across
# V3.2 -> V3.3; the example shape (V3.4 W5) moved the SHAPE but not the count;
# fill rate is a roll-call question, and the gist is the last thing read before
# the conversation. Gated so it cannot read as a quota.
gist = replace(gist,
    'and `source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning.',
    '`source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning, and a `thought` when I have a hunch or a connection of my own worth keeping beside it.',
    'X5 `thought` in the gist `new` roll-call')

(HERE / 'template.md').write_text(tpl)
(HERE / 'gist.md').write_text(gist)
(HERE / 'strategy.md').write_text((PARENT / 'strategy.md').read_text())
(HERE / 'closure.md').write_text((PARENT / 'closure.md').read_text())
(HERE / 'author_log.json').write_text(json.dumps(LOG, indent=1))
print(json.dumps(LOG, indent=1))
parent_tpl = (PARENT / 'template.md').read_text()
parent_gist = (PARENT / 'gist.md').read_text()
print(f'template {len(parent_tpl)} -> {len(tpl)} ({len(tpl) - len(parent_tpl):+d})')
print(f'gist     {len(parent_gist)} -> {len(gist)} ({len(gist) - len(parent_gist):+d})')
print('template words:', len(tpl.split()), '(V3.4:', len(parent_tpl.split()), ')')
