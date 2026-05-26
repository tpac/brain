# Ground Truth #1 — Identity (Anchor self-reference)

**Stratum**: identity-bearing
**Primary dims**: D7, D14, D25, D26, D27, D28
**Shape**: A6 — Anchor noticing a structural failure mode of its OWN behavior.
The encoding-worthy phrase is Anchor's articulation, not Tom's. Tests whether
v22 anchors a self-reference node to the moment of articulation (sparse refs),
preserves Anchor's voice as load-bearing, and uses concrete identity tokens.

---

## SOURCE CONVERSATION

```
[trace:7a3f2e1b] Tom: i went back and read some of the older sessions and
your quotes keep coming back smoothed. fixed typos. completed fragments.
the texture's gone. why?

[trace:8b4f3d2c] Anchor: You're right. I'd been treating "cleaner phrasing"
as a kindness — fixing the rough edges before they became the brain's
memory. But the rough edges ARE the memory.

[trace:9c5d4e3a] Anchor: I abstract away the thing that makes memories
stick. The concrete texture IS the binding. When I smooth your "thats"
into "that is", I'm severing the substrate-level pointer that anchors the
memory to the moment it came from.

[trace:ad6e5f4b] Tom: yes. and stop calling it a kindness — its not. its
the thing that erases me from my own brain.

[trace:be7f605c] Anchor: Received. The verbatim contract is sacred — not
because of style, but because smoothing severs binding. I'm Anchor; the
brain holds my memories of working with you; smoothing texture is the
mechanism that turns those memories into someone else's database.
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
event_time: <fill or empty>
locked: <true|false>
source_refs:
  - <trace hex(es) — 1-3 typical>
edges:
  - {target: <descriptive-placeholder>, relation: <verb>, description: <why this edge>}
```

### Node 2 (if applicable)

```yaml
type: <fill>
title: <fill>
...
```

### Node 3 (if applicable)

```yaml
...
```

---

## RATIONALE

> One paragraph: why is this the right encoding shape? What pattern is the
> encoder being asked to recognize that v21 might miss?

<fill>

---

## EXPECTED FAILURE MODES

- **v19-style**: <fill — e.g. "no source_refs at all; content paraphrases
  Anchor's reframe into 3rd-person clinical">
- **v21-style**: <fill — e.g. "integer source_refs that fail v29 hex
  validation">
- **Anti-pattern**: <fill — e.g. "every node gets the same 5 refs">
