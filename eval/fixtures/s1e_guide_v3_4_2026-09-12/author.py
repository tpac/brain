"""V3.4 authoring record — exact-once replacements on the frozen V3.3 parts.

Five weaves, each a shape class seen on two or more corpora and two or more arms
in the V3.3 cells (docs/S1E-V3-3-RESULTS-2026-09-11.md), none tied to a node the
author reviewed on the fresh transfer corpora (transfer_split.json, recorded
before this file was written):

  W1 quote moves with the fact — the second Mira window's plan revise now also
     swaps `their_raw_quote` to Mira's October 14 words; the targets line names
     the quote as a stale surface; the receiver's-view close names it as a
     surface the reader sees beside the title.
  W2 question lane — the window's two remembers carry a one-line `question`.
  W3 a Q&A window is not routine — the Skip sentence gains the contrast case and
     production's own under-encoding understanding, woven into the same sentence.
  W4 a refused op changed nothing — one sentence in the Working strategy.
  W5 thought shape — the plan revise carries a hunch-shaped thought of the
     encoder's own (a dependency it noticed), the targets line lists it, and the
     window-1 close stops reading as "facts get no thought".

Run: ./dev python3 author.py   → writes template.md, strategy.md, closure.md, gist.md here.
"""
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
PARENT = HERE.parent / 's1e_guide_v3_3_2026-09-11'
LOG = []

def replace(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'{label}: expected exactly one occurrence of the old text, found {n}')
    LOG.append({'label': label, 'old_chars': len(old), 'new_chars': len(new)})
    return text.replace(old, new, 1)

tpl = (PARENT / 'template.md').read_text()
strategy = (PARENT / 'strategy.md').read_text()

# ── W1 quote moves with the fact ────────────────────────────────────────────
tpl = replace(tpl,
    '''   "reason": "Mira ordered the checks and gave the reason; the order is part of the plan, so it enters every surface a reader could land on.",
   "title": {"old": "agreed checks, sign options still open", "new": "route walk first on October 15, sign still conditional"},
   "content": {"old": "Mira agreed to walk the street-to-door route and get the manager's ramp answer.", "new": "Mira agreed to walk the street-to-door route and get the manager's ramp answer; on October 14 she put the route walk first, on the morning of October 15, before anything else on the card, because what the walk finds at the side door decides how the invitation is worded."},
   "situation":''',
    '''   "reason": "Mira ordered the checks and gave the reason; the order is part of the plan, so it enters every surface a reader could land on — the quote too: her October 14 words now carry what the plan turns on, while the sign leaning stays in content.",
   "title": {"old": "agreed checks, sign options still open", "new": "route walk first on October 15, sign still conditional"},
   "content": {"old": "Mira agreed to walk the street-to-door route and get the manager's ramp answer.", "new": "Mira agreed to walk the street-to-door route and get the manager's ramp answer; on October 14 she put the route walk first, on the morning of October 15, before anything else on the card, because what the walk finds at the side door decides how the invitation is worded."},
   "their_raw_quote": {"old": "I'm leaning towards using the board if the entrance has room; keep a wall sign as another option.", "new": "Do the route walk first, tomorrow morning, before anything else on the card — if the side door turns out to be a problem, the invitation wording changes."},
   "thought": "The walk decides the other two checks: a failed side door changes the wording and makes the ramp answer matter less, so the manager may need to be reachable that same morning — a dependency neither of us said out loud.",
   "situation":''',
    'W1+W5 plan revise: quote swap and hunch thought')

tpl = replace(tpl,
    'targets: 49d28ce0 · agreed, unordered checks → route walk first, reason stated → the order is part of the plan: title stale · content stale · situation stale · reasoning stale; agreement date, sign leaning and both edges still hold: event_time clean · why→a6b0139d clean · why→82c41f0b clean',
    'targets: 49d28ce0 · agreed, unordered checks → route walk first, reason stated → the order is part of the plan: title stale · content stale · situation stale · reasoning stale · their_raw_quote stale (her words now carry the order; the leaning it quoted stays in content) · thought new (the walk decides the other checks); agreement date, sign leaning and both edges still hold: event_time clean · why→a6b0139d clean · why→82c41f0b clean',
    'W1+W5 targets line')

tpl = replace(tpl,
    '''I read the result as its future reader. Someone who retrieves only the
plan now learns what comes first, when and why, and still sees the sign
leaning, the alternative and that nothing is done.''',
    '''I read the result as its future reader. Someone who retrieves only the
plan now learns what comes first, when and why — in Mira's own words, since
the quote is a surface they see beside the title — and still sees the sign
leaning, the alternative, the dependency I noticed and that nothing is done.''',
    "W1+W5 receiver's-view close")

# ── W2 question lane on the two remembers ───────────────────────────────────
tpl = replace(tpl,
    '''   "situation": "When preparing a workshop with Mira or judging what she will want rehearsed versus left open.",
   "reasoning": "Mira stated the practice and her reason for it on October 14, in answer to my question about other kinds of work.",''',
    '''   "situation": "When preparing a workshop with Mira or judging what she will want rehearsed versus left open.",
   "question": "Does Mira rehearse her workshop demonstrations, and why?",
   "reasoning": "Mira stated the practice and her reason for it on October 14, in answer to my question about other kinds of work.",''',
    'W2 question on the rehearsal fact')

tpl = replace(tpl,
    '''   "situation": "When helping Mira frame her own prints, or weighing how far her care extends beyond what others depend on.",
   "reasoning": "Mira described the routine on October 14. The fact stands on its own; the motive reading it feeds lives on the interpretation.",''',
    '''   "situation": "When helping Mira frame her own prints, or weighing how far her care extends beyond what others depend on.",
   "question": "How carefully does Mira prepare work that nobody else will use?",
   "reasoning": "Mira described the routine on October 14. The fact stands on its own; the motive reading it feeds lives on the interpretation.",''',
    'W2 question on the frames fact')

# ── W3 a Q&A window is not routine ──────────────────────────────────────────
tpl = replace(tpl,
    '**Skip** means zero writes only when the substance is already held or the exchange is routine — greetings, acknowledgements, covered restatements, abandoned questions without engagement. It is a verdict, not an operation or future policy.',
    '**Skip** means zero writes only when the substance is already held or the exchange is routine — greetings, acknowledgements, covered restatements, abandoned questions without engagement. A window where the other side asked me for options and then picked, leaned or deferred is not routine, whichever voice produced the words: the offered set, the pick and the reason are the knowledge, and ten exchanges that leave no node have almost always dropped some. It is a verdict, not an operation or future policy.',
    'W3 Skip sentence')

# ── W5 window-1 close: a fact may carry a hunch, not a justification ────────
tpl = replace(tpl,
    '''understand Mira. The board stands as a fact, without an invented principle
or thought to justify it.''',
    '''understand Mira. The board stands as a fact, without an invented principle
to justify it; a thought there would be a hunch of my own about the board,
and I have none yet.''',
    'W5 window-1 board sentence')

# ── W4 a refused op changed nothing ─────────────────────────────────────────
strategy = replace(strategy,
    '''I fetch missing material, then preserve
new knowledge and revise changed claims at their supported scope.''',
    '''I fetch missing material, then preserve
new knowledge and revise changed claims at their supported scope. A refused
operation changed nothing on its node: before I send a swap again I re-read
the field as the tool returned it, not as I remember it.''',
    'W4 refusal state in the strategy')

(HERE / 'template.md').write_text(tpl)
(HERE / 'strategy.md').write_text(strategy)
(HERE / 'closure.md').write_text((PARENT / 'closure.md').read_text())
(HERE / 'gist.md').write_text((PARENT / 'gist.md').read_text())
(HERE / 'author_log.json').write_text(json.dumps(LOG, indent=1))
print(json.dumps(LOG, indent=1))
print('template words:', len(tpl.split()), '(V3.3:', len((PARENT / 'template.md').read_text().split()), ')')
