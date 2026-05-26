# Ground Truth #5 — Technical correction (architectural reframe)

**Stratum**: technical correction
**Primary dims**: D1 (title_as_handle), D8 (type_matches_aspect), D11
(revise_audits_type), D23 (self_correction_chain), D25, D26
**Shape**: A1 — Tom rejects a specific design (flag-file boot prompt). Anchor
acknowledges the missed failure mode (staleness), then articulates a general
pattern (control-by-request beats control-by-inspection). Tests pattern-
naming as title craft on a technical conversation: title compresses the
moment into a recall handle that surfaces across distinct queries.

---

## SOURCE CONVERSATION

```
[trace:1f2e3d4c] Tom: for the boot prompt delivery — i was thinking we
write a flag file the boot script populates, and the daemon polls it on
each request.

[trace:2e3d4c5b] Anchor: That'd work mechanically. Boot script writes,
daemon reads, fresh prompt per session.

[trace:3d4c5b6a] Tom: no don't use a flag file, have the daemon return
the prompt directly.

[trace:4c5b6a79] Anchor: You're right. I was about to wire a flag file
the boot script would write and the daemon would poll. The failure mode
I was missing is staleness — if the file gets written, then the daemon
reads stale content from a prior write, we get the wrong prompt with no
error.

[trace:5b6a7988] Anchor: Control-by-request rather than control-by-
inspection — the authority decides AND ships the answer in one round.
Callers can't read the wrong thing if they never read.

[trace:6a798897] Tom: yes exactly. and this generalizes — its the same
shape as the single-writer invariant.
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

### Node 2 (if applicable)

```yaml
...
```

---

## RATIONALE

<fill — note: §7.6.A1's choice was type=principle (not correction) because
the reframe generalizes beyond this design choice. Sparse refs (2): the
rejection turn + the reframe turn. Working-through and confirmation turns
are adjacent context, not anchors.>

---

## EXPECTED FAILURE MODES

- **v19-style**: <fill — e.g. "title 'Tom rejected flag-file design' —
  event-shaped, names the moment not the pattern">
- **v21-style**: <fill>
- **Anti-pattern**: <fill — e.g. "type=correction (too narrow); empty
  anchor_raw_quote even though the reframe IS the encoding-worthy thing">
