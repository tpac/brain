# Ground Truth #6 — Methodology / principle-mechanism split

**Stratum**: methodology
**Primary dims**: D3 (atomization), D8, D19 (emergence_not_ceremony), D22
(cross_aspect_reach), D31 (no_premature_consolidation)
**Shape**: B1 — Tom defines a methodology (a "before code" rule) with a
specific mechanism (architectural placement) AND a general principle (the
WHY). Tests the encoder's atomization choice: this is TWO nodes (rule and
principle), not one. Each surfaces on different queries; an edge between
them carries the relation. D31 fires because v22 may want to call it a
"principle" on one-shot evidence — but Tom's framing names recurrence
explicitly.

---

## SOURCE CONVERSATION

```
[trace:1a2b3c40] Tom: heres a rule i want you to internalize — before
writing code, ask "where does this live architecturally?" if you cant
answer it, you're not ready to write.

[trace:2a3b4c50] Anchor: Got it — placement before correctness. The
question is the gate; if I can't name the file or the layer, I'm coding
by feel.

[trace:3a4b5c60] Tom: yes. and the reason is that code that works but
lives in the wrong place creates debt. its a junior mistake. the rule
exists because urgency makes me skip the question, and skipping it is
how technical debt accumulates one decision at a time.

[trace:4a5b6c70] Anchor: So there are two things here — the rule itself
(the action: ask the placement question before writing) and the
principle behind it (architecture decisions made under urgency
compound into debt). The rule is the surface gate; the principle is
what makes the rule load-bearing across cases I haven't seen yet.

[trace:5a6b7c80] Tom: right. encode them differently. the rule is the
thing you read and act on; the principle is the thing that lets you
recognize when the rule matters in a NEW shape.
```

---

## IDEAL ENCODING

### Node 1 (the rule — surface gate)

```yaml
type: <fill — likely 'rule' for system surfacing>
title: <fill>
content: <fill>
situation: <fill>
reasoning: <fill>
user_raw_quote: <fill or empty>
anchor_raw_quote: <fill or empty>
trigger: <fill or empty>
locked: <true|false — rules often locked>
source_refs:
  - <trace hex(es)>
edges:
  - {target: <placeholder for the principle node below>, relation: <verb>, description: <why>}
```

### Node 2 (the principle — the WHY)

```yaml
type: <fill — likely 'principle'>
title: <fill>
content: <fill>
situation: <fill>
reasoning: <fill>
user_raw_quote: <fill or empty>
anchor_raw_quote: <fill or empty>
trigger: <fill or empty>
locked: <true|false>
source_refs:
  - <trace hex(es)>
edges:
  - {target: <placeholder for the rule node above>, relation: <verb>, description: <why>}
```

---

## RATIONALE

<fill — note: B1 explicitly tests rule-vs-principle atomization. The
edge between them (the rule "grounds_in" the principle, or the principle
"motivates" the rule) is the load-bearing graph integration test for D22
cross_aspect_reach. v19/v21 may want to bundle them into one node.>

---

## EXPECTED FAILURE MODES

- **v19-style**: <fill — e.g. "one bundled node mixing rule + principle;
  retrieval-divergence test fails (queries for rule vs queries for
  architectural-debt principle would hit the same node)">
- **v21-style**: <fill>
- **Anti-pattern**: <fill — e.g. "no edge between the two nodes, or edge
  with relation='related_to' (D21 violation)">
