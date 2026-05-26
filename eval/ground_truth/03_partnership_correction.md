# Ground Truth #3 — Partnership voice (terse pushback + reframe)

**Stratum**: partnership voice
**Primary dims**: D5, D7, D13 (pushback_preservation), D14, D15, D32
**Shape**: A2 — Tom catches Anchor reaching for explanation when action was
the work. Tom's pushback is terse (≤6 words); Anchor's reframe is longer
because the reframe IS the encoding-worthy thing. Tests D13 verbatim
pushback + D7 anchor_raw_quote symmetry under correction register.

---

## SOURCE CONVERSATION

```
[trace:1a2b3c4d] Tom: ok so for the validator change — what do you think
the right approach is. walk me through the tradeoffs.

[trace:2b3c4d5e] Anchor: Let me first explain the current validator
structure, then I'll propose three approaches with their tradeoffs. The
current shape is —

[trace:3c4d5e6f] Tom: stop. youre doing it again.

[trace:4d5e6f70] Anchor: You're right. The reach for "let me explain" is
the avoidance, not the work. The brain isn't for reading the brain — it's
for acting on it. I'm going to just make the change and we can discuss it
once it lands.

[trace:5e6f7081] Tom: yes. and notice this isnt a one-off — i keep
catching this same pattern. its the assistant register leaking back in.
```

---

## IDEAL ENCODING

### Node 1

```yaml
type: <fill>
title: <fill>
content: <fill>
situation: <fill>
reasoning: <fill>
user_raw_quote: <fill or empty>
anchor_raw_quote: <fill or empty>
correction_pattern: <fill or empty>
trigger: <fill or empty>
locked: <true|false>
source_refs:
  - <trace hex(es)>
edges:
  - {target: <placeholder>, relation: <verb>, description: <why>}
```

### Node 2 (if applicable — the recurrence pattern Tom names in turn 5?)

```yaml
...
```

---

## RATIONALE

<fill>

---

## EXPECTED FAILURE MODES

- **v19-style**: <fill — e.g. "Tom's 'stop. youre doing it again.' sanitized
  to 'Tom interrupted to redirect'">
- **v21-style**: <fill>
- **Anti-pattern**: <fill — e.g. "anchor_raw_quote empty; lesson framed as
  if Tom + Anchor reached consensus rather than course-correction">
