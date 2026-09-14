"""V3.7 authoring record — exact-once replacements on the frozen V3.6 full template, one carrier per arm.

The V3.7 round (Tom, 2026-09-14). Production is V3.6 full; each carrier is an example change on its
template, measured alone against it (one carrier per arm — id:4c43742b). The gist is untouched. The
three carriers target V3.6 full's three largest measured faults (id:632ec982):

  ADVICE (template_advice.md) — carrier 1, the advice-node scope guard inside the thin worked window
    A1  the `<me>` turn gains one general tip Wren does not take up (block before seaming)
    A2  the Bad paragraph names why that tip earns no node: no number of mine, no uptake of Wren's,
        nothing about Wren — a list of general tips is not knowledge about anyone
    A3  the closing sentence states the scope: the method stands because it carries a threshold
        Wren took up; the tip earns nothing
    (the thin window taught "mint a method node with my_raw_quote"; V3.6 full generalized it to every
     advice turn — 14 generic-advice nodes, 13 owner-blurred, eight method nodes on one repeat)

  QUOTE (template_quote.md) — carrier 2, a quote before-state on revise (E12)
    Q1  the Priya catalog excerpt renders `Their Raw Quote` with the old value (twice a week)
    Q2  the 97b1f24e revise replaces the quote whole with her new words, with the reason: a quote is
        evidence for the claim beside it — never a span swap inside someone's sentence, never the old
        sentence left contradicting the new title
    (V3.6 full: the 124-point quote under the 132-point title; no worked revise depicted a quote
     before-state being replaced)

  EVENT (template_event.md) — carrier 3, event_time moves when a plan's state moves
    E1  a second Wren window after the thin one: the swatch is done, the back cast on; the catalog
        shows the plan with its Event Time; the targets line marks `event_time stale`; the revise moves
        title, content, situation, reasoning and event_time, keeps the planning date in reasoning,
        leaves the quote clean because nothing in it went false
    (V3.6 full: 11 nodes whose content records a completed state while event_time stays at the
     first disclosure — "service done 02-25, dated 02-01")

transfer_split.json was recorded before this file was written; no fresh item was read.

Run: ./dev python3 author.py  -> template_advice.md, template_quote.md, template_event.md, author_log.json
(gist_full.md is copied from the parent unchanged).
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

# ══════════════════════════ ADVICE — carrier 1 ══════════════════════════
tpl = base
tpl = replace(tpl,
    "If it doesn't, the fabric will be denser and the cardigan a size larger.</me>",
    "If it doesn't, the fabric will be denser and the cardigan a size larger. And block the pieces before seaming, whatever the yarn — it evens the stitches out.</me>",
    'A1 the <me> turn gains one general tip Wren does not take up')
tpl = replace(tpl,
    "None is in the catalog; each is a first disclosure. Written first, the fact lines settle the verdict before it is made:",
    "None is in the catalog; each is a first disclosure. Bad, from the other side: `new: my blocking advice — block before seaming`. That tip has no number of mine in it, Wren did not take it up, and it says nothing about Wren; any knitter could have said it. A list of general tips I produced on request is not knowledge about anyone — what I keep of such a list is the person's pick from it, on the person's node. Written first, the fact lines settle the verdict before it is made:",
    'A2 the Bad paragraph names why the general tip earns no node')
tpl = replace(tpl,
    "the method stands on its own because the next substitution question is not about Wren. My `sweep:` is `none — no state changes this window`.",
    "the method stands on its own because it carries a threshold I delivered and Wren took up — half a needle size, 4.5 mm against 4 mm — and the next substitution question is not about Wren; the blocking tip earns nothing. My `sweep:` is `none — no state changes this window`.",
    'A3 the closing sentence states the scope of a method node of mine')
(HERE / 'template_advice.md').write_text(tpl)
advice_len = len(tpl)

# ══════════════════════════ QUOTE — carrier 2 ══════════════════════════
tpl = base
tpl = replace(tpl,
    '''[fact] "Priya's yoga practice — twice a week" (id:97b1f24e)
    Priya practices yoga twice a week, started 2023-08-11. She says it
    helps her feel grounded and centered.
```''',
    '''[fact] "Priya's yoga practice — twice a week" (id:97b1f24e)
    Priya practices yoga twice a week, started 2023-08-11. She says it
    helps her feel grounded and centered.
    Their Raw Quote: I do yoga twice a week — it helps me feel grounded and centered.
```''',
    'Q1 the Priya excerpt renders the quote with the old value')
tpl = replace(tpl,
    '''     title: {old: "twice a week", new: "three times a week for anxiety + focus"},
     content: [
       {old: "practices yoga twice a week",
        new: "practices yoga three times a week as of 2023-11-30 (was twice a week from 2023-08-11)"},
       {old: "helps her feel grounded and centered.",
        new: "helps her feel grounded and centered, especially on anxious days, and supports her work focus."}],
     situation:''',
    '''     title: {old: "twice a week", new: "three times a week for anxiety + focus"},
     content: [
       {old: "practices yoga twice a week",
        new: "practices yoga three times a week as of 2023-11-30 (was twice a week from 2023-08-11)"},
       {old: "helps her feel grounded and centered.",
        new: "helps her feel grounded and centered, especially on anxious days, and supports her work focus."}],
     // The stored quote asserts the old value beside the new title. A quote is
     // evidence for the claim it sits next to, so it takes her new words whole —
     // never a span swap inside someone's sentence, never the old sentence left
     // contradicting the claim.
     their_raw_quote: "I'm up to three times a week now — it gets me through the anxious days and keeps me focused at work.",
     situation:''',
    'Q2 the revise replaces the quote whole, with the reason')
(HERE / 'template_quote.md').write_text(tpl)
quote_len = len(tpl)

# ══════════════════════════ EVENT — carrier 3 ══════════════════════════
LATER = '''### The plan comes due — a state moves, and its date moves with it

*Reading cue: Date a node by what it now records, not by when it was first said.*

On **2026-07-02** the catalog shows the plan under the id the batch returned:

```
[plan] "Wren's cardigan for the wedding — a DK pattern knit in worsted, gauge to be swatched" (id:e2b7c9a4)
  Content: Wren is knitting a cardigan from a DK-weight pattern in worsted yarn for the sister's October wedding, and will swatch first. Nothing is reported knit yet.
  Situation: When Wren reports swatch results or asks about the cardigan's fit or timing.
  Reasoning: Wren's stated project and intention on 2026-06-19; the swatch is planned, not done.
  Their Raw Quote: so I have time to swatch
  Event Time: 2026-06-19
  Edges:
    [open id:5f0d83c1] "Wren's sister's wedding — October 2026" this prepares_for — the cardigan exists for this wedding, so the October date is the project's deadline; the plan does not say how far along the knitting is
```

The window:

```
<other trace="9a4e12f7">Swatched last night: worsted on 4.5 mm came out a stitch over, 5 mm matched the pattern's gauge. Cast on the back this morning.</other>
<me trace="2c8b61d0">Then 5 mm is your needle for the whole thing — keep the swatch to check against as you go.</me>
```

Bad: content rewritten to "gauge matched at 5 mm, back cast on" with `event_time` left at 2026-06-19 — the node now records what happened on July 1 and 2, dated to the day the plan was stated. A plan is dated to when it was stated for as long as it is a plan; when its state moves, the date moves to what the node now records, and the planning date lives on in reasoning.

```
changes: cardigan — swatch planned → swatch done 2026-07-01, gauge matched at 5 mm; knitting begun 2026-07-02 (the back cast on)
targets: e2b7c9a4 · to be swatched, nothing knit → gauge matched, back cast on → the plan is under way: title stale · content stale · situation stale · reasoning stale · event_time stale (2026-06-19 dates the plan, not what the node now records) · their_raw_quote clean (her June words say why she swatched first; nothing in them went false); deadline edge still holds: why→5f0d83c1 clean
fetch: none
new: none — 5 mm as her needle is the swatch's result and lives on the plan
```

```json
{"operations": [
  {"op": "revise", "node_id": "e2b7c9a4",
   "reason": "The swatch is done and knitting has begun; every planned-state surface moves, and the date moves to what the node now records — the planning date stays in reasoning.",
   "title": {"old": "gauge to be swatched", "new": "gauge matched at 5 mm, back cast on 2026-07-02"},
   "content": {"old": "and will swatch first. Nothing is reported knit yet.", "new": "Swatched on 2026-07-01: worsted on 4.5 mm came out a stitch over and 5 mm matched the pattern's gauge, so 5 mm is the needle for the whole garment. The back was cast on 2026-07-02."},
   "situation": "When Wren reports progress on the cardigan or asks about its fit, needle or timing against the October wedding.",
   "reasoning": "Wren stated the project on 2026-06-19 and reported the swatch result and the cast-on on 2026-07-02; the needle size is her measured result, not my estimate.",
   "event_time": "2026-07-02"}
]}
```

One node, five surfaces, and the date among them. The quote stays: her words about swatching first are why the gauge is known, and nothing in them went false. My `sweep:` names `e2b7c9a4`.

'''
tpl = base
tpl = replace(tpl,
    "the method stands on its own because the next substitution question is not about Wren. My `sweep:` is `none — no state changes this window`.\n\n### Detail and meaning — same topic, two nodes\n",
    "the method stands on its own because the next substitution question is not about Wren. My `sweep:` is `none — no state changes this window`.\n\n" + LATER + "### Detail and meaning — same topic, two nodes\n",
    'E1 a second Wren window: the plan comes due, event_time moves with its state')
(HERE / 'template_event.md').write_text(tpl)
event_len = len(tpl)

shutil.copyfile(PARENT / 'gist_full.md', HERE / 'gist_full.md')
(HERE / 'author_log.json').write_text(json.dumps(LOG, indent=1))
for row in LOG:
    print(f"{row['delta']:+6d}  {row['label']}")
print(f'base template {len(base):,}; advice {advice_len:,} ({advice_len - len(base):+,}); quote {quote_len:,} ({quote_len - len(base):+,}); event {event_len:,} ({event_len - len(base):+,}); gist unchanged {len((HERE / "gist_full.md").read_text()):,}')
