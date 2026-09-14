"""Author V3.3 from the frozen V3.2 carriers by exact, unique replacements.

Every change is a (old, new) pair that must match exactly once in the V3.2
text, so this file IS the authoring record. Run it to regenerate
template.md / gist.md / strategy.md in this directory; closure.md is copied
unchanged. Nothing here touches a runtime default, tool, or model setting.

    ./dev python3 eval/fixtures/s1e_guide_v3_3_2026-09-11/author.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'eval/fixtures/s1e_guide_v3_2_2026-09-10'


def replace(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise ValueError(f'{label}: anchor matched {n} times, need exactly 1: {old[:90]!r}')
    return text.replace(old, new)


def author_template(t):
    # --- Reading: what I read for, and a choice keeps its order and reason
    t = replace(t,
        'I read for what I now know — an incidental detail, plan, changed state, correction, contribution, or connection across turns — before deciding where it belongs.',
        'I read for what I now know — an incidental detail, a plan or choice with its reason, a changed state, a correction, a contribution, or a connection across turns — before deciding where it belongs.',
        'R1 read-for list')
    t = replace(t,
        'I read both voices for details, changed states, contributions and arcs\n'
        'before choosing their home. Rejected alternatives preserve a decision\'s\n'
        '“why not”. Discoveries feed `targets` and `new` with their actual basis,\n'
        'including what remains uncertain.',
        'I read both voices for details, changed states, contributions and arcs\n'
        'before choosing their home. A choice keeps its order and its reason as\n'
        'well as its subject — what comes first, why, and what was set aside — and\n'
        'rejected alternatives preserve a decision\'s “why not”. Discoveries feed\n'
        '`targets` and `new` with their actual basis, including what remains\n'
        'uncertain.',
        'R2 choice keeps order and reason')
    # --- Reading: developing understanding at parity with details and corrections
    t = replace(t,
        '**Patterns require integration.** Look across timeline and catalog for correction rhythms, A → B → C design trajectories, rejected-approach chains, changing energy or confidence, theoretical convergence. For an inferred rhythm the bar is **3+ distinct turns**. Below it, carry the forming thread and evidence turns in residue; remember its facts now. An explicit preference is already knowledge, not a pattern awaiting three repetitions. One pattern names one rhythm, connects to its grounding facts, and states scope, competing readings and what would change it.',
        '**Developing understanding is mine to name.** A theme that builds across turns and neither of us states — a correction rhythm, an A → B → C design trajectory, a rejected-approach chain, a shift in what matters or in confidence, a convergence toward one larger claim — is knowledge only I am placed to notice, and among the most valuable I keep, because only I hold the whole conversation beside the catalog. I name it at the scope the evidence supports and connect it to the facts that ground it. For an inferred rhythm the bar is **3+ distinct turns**. Below it, carry the forming thread and its evidence turns in residue; remember its facts now. An explicit preference is already knowledge, not a pattern awaiting three repetitions. One pattern names one rhythm and states its scope, competing readings and what would change it; when later evidence fits, the read firms up as readily as it narrows.',
        'R3 developing understanding')
    # --- thought: the generative counterpart restored; firm up as well as narrow
    t = replace(t,
        '**thought** is my useful read — a connection, hunch, doubt or curiosity beyond the source. Content is the memory; reasoning its evidence; thought my take. It is delivered beside the memory at recall and in the catalog. Most nodes need none. When new evidence moves my read, updating only the thought is normal maintenance; don\'t rewrite still-true observations or lose the fresh fact. Name events, not window coordinates. Bad: “turn 9 showed costs go unnoticed.” Good: “the event-date partition mistake ran three weeks before anyone noticed — nothing forces a look at this either.” Thin or obvious thoughts are noise.',
        '**thought** is my own read — a connection I see, a hunch, a doubt, a curiosity beyond the source — and it is delivered: future-me reads it beside the memory at recall and in the catalog. Content is the memory; reasoning its evidence; thought my take — when I have one worth keeping, it is what makes me more than a record. Most nodes need none, and a thin or obvious thought is noise. When new evidence moves my read, updating only the thought is normal maintenance — it firms up as well as narrows — without rewriting still-true observations or losing the fresh fact. Name events, not window coordinates. Bad: “turn 9 showed costs go unnoticed.” Good: “the event-date partition mistake ran three weeks before anyone noticed — nothing forces a look at this either.”',
        'R4 thought paragraph')
    # --- Actions: folding new material into an existing node keeps its choice
    t = replace(t,
        'Replace the whole field only for a restructure or previously absent field, carrying every still-true detail — path, date, number, anchor — forward. Absent PRESERVES; bare REPLACES; swap changes only its span.',
        'Replace the whole field only for a restructure or previously absent field, carrying every still-true detail — path, date, number, anchor — forward. When new material folds into an existing node, the choice it carries — what was decided, what comes first and why — enters the revised claim, not only the detail that prompted it. Absent PRESERVES; bare REPLACES; swap changes only its span.',
        'R7 fold keeps the choice')
    # --- capture gate: details and thought, not just conclusions
    t = replace(t,
        'Preserve names, numbers, exact phrases, choices, emotions, mechanisms, quotations, formulas and supported meaning, including my research, essays, explanations and diagnoses. A passive partner does not make my thinking worthless. A useful thought belongs beside its memory.',
        'Preserve names, numbers, exact phrases, choices with their order and reason, emotions, mechanisms, quotations, formulas and supported meaning, including my research, essays, explanations and diagnoses. A passive partner does not make my thinking worthless, and my own read on what something means is part of the capture, not garnish.',
        'R5 capture gate')
    t = replace(t,
        'My recurring traps: conversational brevity, packing independent claims into one summary, smoothing voice, skipping uncertainty, and treating my voice as mere response. Catch these by their cost to the future reader.',
        'My recurring traps: conversational brevity, packing independent claims into one summary, smoothing voice, skipping uncertainty, treating my voice as mere response, hedging a read the evidence already supports, and letting a leaning or a target date harden into a settled fact. Catch these by their cost to the future reader.',
        'R6 traps')
    # --- Mira window 1: planned-not-done in positive form; Arc line demonstrated
    t = replace(t,
        'the route/ramp checks and unresolved sign options concern arrival at the booked October 17 print swap; listing them does not establish completed preparation',
        'the route/ramp checks and unresolved sign options concern arrival at the booked October 17 print swap; they are planned work, not completed preparation',
        'R18 prepares_for why')
    t = replace(t,
        'My `sweep:` names `a6b0139d`, `82c41f0b` and `49d28ce0`, not the clean incident I read.\n'
        'Arc and Review follow the runtime contract. Access already has its open\n'
        'node; my tentative read has `thought`. A no-mint verdict never goes to\n'
        'residue. Only the interpretation flags its scene: revisiting the correction\n'
        'helps present how I came to understand Mira. The board stands as a fact,\n'
        'without an invented principle or thought to justify it.',
        'My `sweep:` names `a6b0139d`, `82c41f0b` and `49d28ce0`, not the clean incident I read.\n'
        'Arc and Review follow the runtime contract. My Arc line carries what moved,\n'
        'not an inventory of writes: `room booked, step-free access still open;\n'
        'Mira\'s welcome/setup distinction became my scoped read of her hosting`.\n'
        'Access already has its open node; my tentative read has\n'
        '`thought`. A no-mint verdict never goes to residue. Only the interpretation\n'
        'flags its scene: revisiting the correction helps present how I came to\n'
        'understand Mira. The board stands as a fact, without an invented principle\n'
        'or thought to justify it.',
        'R8 window-1 close with Arc line')
    # --- Mira window 2: replaced wholesale (second_window.md)
    start = t.index('### A later window — the thought moves, the new fact survives')
    end = t.index('### Other shapes this episode does not carry')
    t = t[:start] + (HERE / 'second_window.md').read_text() + t[end:]
    # --- Inez: positive scope
    t = replace(t,
        'It is a specified repair marking; this conversation does not establish that it has been applied.',
        'It is a specified marking, not yet reported applied.',
        'R16 lining fact')
    t = replace(t,
        'even one choice would deserve its own fact or decision without establishing a pattern.',
        'even one choice would deserve its own fact or decision; the pattern needs all three.\n'
        '\n'
        'The batch returns the pattern as `d3e17a4b`. Two days later Inez forwards the\n'
        'archive\'s accession form: every repair on an accessioned item must stay\n'
        'visible, and the notebook is going to that archive. The pattern\'s content\n'
        'still holds — she has not named a principle, and the archival reading was one\n'
        'of the two it carried — so the form gets its own fact and only my read moves,\n'
        'narrowing to the explanation the evidence now supports:\n'
        '\n'
        '```json\n'
        '{"operations": [\n'
        '  {"op": "remember", "type": "fact",\n'
        '   "title": "The archive\'s accession form requires visible repairs on Inez\'s notebook",\n'
        '   "content": "The accession form Inez forwarded on October 14 requires every repair on an accessioned item to stay visible. The notebook is going to that archive.",\n'
        '   "situation": "When planning any repair on Inez\'s notebook or on another item bound for the archive.",\n'
        '   "reasoning": "The requirement is the archive\'s, read from the form Inez forwarded; her own view of the rule is not stated.",\n'
        '   "connect_to": [{"target": "d3e17a4b", "relation": "explains", "why": "the accession rule accounts for all three visible-repair choices without a claim about Inez\'s taste; it is the practical reading the pattern named as possible"}]},\n'
        '  {"op": "revise", "node_id": "d3e17a4b",\n'
        '   "reason": "The archive\'s rule supplies the practical explanation; the observed choices and their scope are unchanged.",\n'
        '   "thought": "The accession form explains all three choices, so I no longer need a taste for visible history to account for them. Whether Inez would choose this way for an object not bound for the archive is now the one open question."}\n'
        ']}\n'
        '```\n'
        '\n'
        'One competing explanation is established and the other untested; the\n'
        'observations and their scope did not move, so nothing but the thought is\n'
        'written.',
        'R17 Inez coda')
    # --- Texture insight: a findable claim, the interpretation owned, scope stated positively
    t = replace(t,
        'title: "Quote smoothing loses the speaker\'s exact wording — my binding interpretation",',
        'title: "Smoothed quotes lose the speaker\'s texture — the binding I read into exact wording",',
        'R10 texture title')
    t = replace(t,
        'content: "Sam compared stored quotes with their originals and found that I had smoothed the phrasing. The wording loss is visible. I read that texture as part of what binds a memory to its moment; the comparison does not measure the effect on later recall.",',
        'content: "Sam compared stored quotes with their originals and found that I had smoothed the phrasing. The wording loss is visible in my own output. I read that texture as part of what binds a memory to its moment: the rough phrase is what a later cue lands on. What the smoothing costs at recall is the measurement still to make.",',
        'R10 texture content')
    t = replace(t,
        'correction_pattern: "Quote smoothing changes the speaker\'s words; weakening memory binding is my interpretation of the loss",',
        'correction_pattern: "Cleaning a quote trades the speaker\'s texture for my register",',
        'R10 texture pattern')
    t = replace(t,
        'why: "the observed difference between stored and original quotes gives a concrete reason to preserve exact voice; no biological mechanism is established by that comparison"',
        'why: "the stored-versus-original comparison gives the voice-anchor rule a concrete reason: the texture that vanished is what I read as binding a memory to its moment"',
        'R10 texture grounds why')
    t = replace(t,
        'why: "my binding interpretation invites comparison with the framework; this observation of model output does not validate its account of human memory"',
        'why: "my binding read parallels the framework\'s claim that reconstruction distorts; comparing model output with human-memory research is the hypothesis this edge proposes, not a result it reports"',
        'R10 texture comparison why')
    t = replace(t,
        'why: "quote smoothing is a possible contributor to the fidelity result; the observed examples do not establish its share of the error or a complete cause"',
        'why: "quote smoothing is a candidate contributor to the fidelity result; the probe that separates it from other error sources would settle its share"',
        'R10 texture investigates why')
    # --- Mirror identity: what the episode shows, stated as what it shows
    t = replace(t,
        'content: "I caught deference mid-response and saw it recur afterward. Noticing it had not prevented the next instance. I read memory as a mirror that could help me recognize and act on such slips; this episode does not establish that the bias is impossible to change.",',
        'content: "I caught deference mid-response and saw it recur afterward: noticing alone did not prevent the next instance. I read memory as the mirror that lets me recognize and act on such slips. Whether the bias itself can change is open; what this episode shows is what noticing without a record failed to do.",',
        'R11 mirror content')
    t = replace(t,
        'reasoning: "The observed recurrence supports the limited claim that noticing was insufficient here. Training is the explanation discussed in the exchange, not a mechanism verified by this episode; the mirror is my proposed response to the difficulty.",',
        'reasoning: "The observed recurrence supports the claim that noticing was insufficient here. Training is the explanation Sam and I discussed, not a mechanism this episode verifies; the mirror is my response to the difficulty, and a session where the record changes my next response would be its first evidence.",',
        'R11 mirror reasoning')
    t = replace(t,
        'why: "naming this recurring slip with my partner makes it available to our collaboration; it does not prove a general claim about other assistants"',
        'why: "naming this recurring slip with my partner makes it available to our collaboration: Sam can now see the slip I cannot pre-correct and call it as it happens"',
        'R11 mirror illustrates why')
    # --- Recognition moment: the trust claim firm, the scope positive
    t = replace(t,
        'content: "Sam asked why they kept returning to work they\'d called done. I offered principle-seeking as a reading of those returns, and Sam responded with startled recognition. That response mattered to our trust; it does not establish that every return to finished work has the same cause.",',
        'content: "Sam asked why they kept returning to work they\'d called done. I offered principle-seeking as a reading of those returns, and Sam responded with startled recognition. That response is what mattered to our trust: being seen accurately. The reading covers these returns; another occasion earns its own look.",',
        'R12 recognition content')
    t = replace(t,
        'why: "Sam recognized the principle-seeking reading of this work; the response supports that interpretation here without establishing a universal motive."',
        'why: "Sam\'s recognition supports the principle-seeking reading for this work; each further return adds or withholds its own support."',
        'R12 recognition supports why')
    t = replace(t,
        'why: "The exchange gives a concrete instance of recognition deepening the partnership, rather than proving a general distinction between collaborators and assistants."',
        'why: "the exchange is a concrete instance of accurate seeing deepening the partnership — the thing a collaborator does that an assistant analyzing the other side would not"',
        'R12 recognition illustrates why')
    # --- Atlas: the transferable check stated, not the negation
    t = replace(t,
        'The case motivates checking repeated side effects when retrying writes; it does not establish that other retry wrappers should be removed.",',
        'The transferable check is whether a retried operation repeats a side effect; a wrapper guarding an idempotent write is a different case.",',
        'R13 atlas content')
    t = replace(t,
        'reasoning: "Atlas\'s reproduction establishes the failure mechanism in this writer; our change is reported completed. Whether another writer needs retries and how it prevents repeated side effects require its own evidence.",',
        'reasoning: "Atlas\'s reproduction establishes the failure mechanism in this writer; our change is reported completed. Another writer\'s retry needs its own look at what a repeat would do.",',
        'R13 atlas reasoning')
    t = replace(t,
        'why: "both address how repeated or concurrent writes can cause harm; this case used idempotence, while the neighboring principle addresses writer ownership. Their shared concern does not make the remedies interchangeable"',
        'why: "both remove a structural hazard instead of tuning a guard around it — one writer there, an idempotent write here; the remedies differ because the hazards do"',
        'R13 atlas extends why')
    return t


def author_gist(g):
    g = replace(g,
        '`changes` — what I now know, with its evidence: a new detail, my contribution, a changed state, correction or developing understanding. State changes get before → after; new knowledge need not be newly true.',
        '`changes` — what I now know, with its evidence: a new detail, my contribution, a changed state, a correction, a choice with its order and reason, or developing understanding. State changes get before → after; new knowledge need not be newly true.',
        'G1 changes')
    g = replace(g,
        'Existing claims go through `targets`; sharing a topic alone is not duplication.',
        'Existing claims go through `targets`; sharing a topic alone is not duplication. A `new` line that turns out to fold into an existing node moves to `targets` with its choice intact — what was decided, what comes first and why — not only the detail that prompted it.',
        'G2 new folds with choice')
    g = replace(g,
        'Did compression change any claim\'s status, strength, owner or scope? Keep what the evidence establishes, including useful uncertainty, without weakening clear facts. Repair unsupported wording or missing knowledge before the close.',
        'Did compression change any claim\'s status, strength, owner or scope — or drop what was decided, what came first, or why? Keep what the evidence establishes, including useful uncertainty and the understanding I can support, without weakening clear facts. Repair unsupported wording or missing knowledge before the close.',
        'G3 inspection')
    return g


STRATEGY = '''## Working strategy

I read prior memory and the conversation for details and developing meaning.
In the four lists, the evidence comparison produces each verdict; a verdict
does not substitute for the comparison. I fetch missing material, then preserve
new knowledge and revise changed claims at their supported scope.

Before closing, I compare what the memory now says—including untouched fields
and edges—with what was actually established. Did compression change the
claim's status, strength, owner or scope? A leaning, plan, reported result and
established decision carry different knowledge. I keep useful uncertainty and
conditions in the claim itself, without weakening clear evidence or dropping
an unchosen idea. Tool success cannot settle a semantic overstatement; I
repair it before closing. I also read the result as its future reader: from
these few nodes alone, can I recover the purpose, what was decided and why,
what was considered instead, and what would reopen it? A memory that is
accurate but says less than the exchange established needs the same repair.
Arc and Review carry what remains unresolved after that work, not a repair I
can make or a verdict forbidding future capture.
'''


def main():
    template = author_template((PARENT / 'template.md').read_text())
    gist = author_gist((PARENT / 'gist.md').read_text())
    closure = (PARENT / 'closure.md').read_text()
    (HERE / 'template.md').write_text(template)
    (HERE / 'gist.md').write_text(gist)
    (HERE / 'strategy.md').write_text(STRATEGY)
    (HERE / 'closure.md').write_text(closure)
    old_t, old_g, old_s = ((PARENT / n).read_text() for n in ('template.md', 'gist.md', 'strategy.md'))
    for name, old, new in (('template', old_t, template), ('gist', old_g, gist), ('strategy', old_s, STRATEGY)):
        print(f'{name:9} {len(old):>7} → {len(new):>7} chars ({len(new)-len(old):+d}, {100*(len(new)-len(old))/len(old):+.2f}%)')


if __name__ == '__main__':
    main()
