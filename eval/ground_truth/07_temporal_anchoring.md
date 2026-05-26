# Ground Truth #7 — Temporal anchoring (multi-event sequence with dates)

**Stratum**: temporal anchoring
**Primary dims**: D9 (semantization_gradient), D10 (lifecycle_marker), D18
(temporal_anchoring), D21, D25
**Shape**: A conversation that anchors a specific dated event AND references
a downstream event by relative time. Tests `event_time` population, the
temporal scout's relative→absolute resolution, and whether v22 ties source_refs
to date-bound moments rather than the surrounding generic discussion.

Adapted from a longmem-style temporal-reasoning shape (see
`eval/longmem/data/longmemeval_oracle.json` for the family). Substitute with
a real session if you have one that exercises Allen-relation sequencing.

---

## SOURCE CONVERSATION

```
[trace:11223344] Tom: i got my car serviced for the first time on march
15th. great experience. theyre also offering quarterly check-ins now.

[trace:22334455] Anchor: Noted — first service 2023-03-15. The quarterly
cadence is a forward-looking schedule, not an event yet; I'll log the
service event and the cadence separately.

[trace:33445566] Tom: three weeks later i noticed the gps started
behaving weird — re-routing for no reason on familiar streets.

[trace:44556677] Anchor: So 2023-04-05-ish, post-service issue: GPS
mis-routing on known streets. That's downstream of the service — not
sure yet whether it's coincidence or caused by the service work. The
temporal relation is "after, possibly because-of" — worth keeping the
sequence so future queries about post-service issues can land on it.

[trace:55667788] Tom: yes — and the gps thing is the FIRST issue, not
the only one. there'll be more.
```

---

## IDEAL ENCODING

### Node 1 (the service event)

```yaml
type: <fill — likely 'event' or 'fact'>
title: <fill>
content: <fill>
situation: <fill>
reasoning: <fill>
user_raw_quote: <fill or empty>
event_time: <fill — 2023-03-15>
locked: <true|false>
source_refs:
  - <trace hex(es)>
edges:
  - {target: <placeholder>, relation: <verb>, description: <why>}
```

### Node 2 (the GPS issue, downstream)

```yaml
type: <fill>
title: <fill>
content: <fill>
situation: <fill>
reasoning: <fill>
user_raw_quote: <fill or empty>
event_time: <fill — 2023-04-05>
locked: <true|false>
source_refs:
  - <trace hex(es)>
edges:
  - {target: <placeholder for service event>, relation: 'after' or 'caused_by?', description: <why>}
```

### Node 3 (the open thread Tom names in turn 5 — "more to come")

```yaml
type: <fill — likely 'open'>
title: <fill>
content: <fill>
situation: <fill>
reasoning: <fill>
user_raw_quote: <fill or empty>
event_time: <fill or empty>
source_refs:
  - <trace hex(es)>
edges:
  - {target: <placeholder for service event>, relation: <verb>, description: <why>}
```

---

## RATIONALE

<fill — note: this is anchored synthesis (refs anchor the specific dated
events), NOT pure synthesis. The GPS-after-service Allen relation is
load-bearing and lives in the edge between the two event nodes. Tom's "FIRST
issue, not the only one" framing earns an open node — this is a thread, not
a closed event.>

---

## EXPECTED FAILURE MODES

- **v19-style**: <fill — e.g. "one bundled 'car service experience' node;
  no separate GPS-issue event; no event_time on either">
- **v21-style**: <fill — e.g. "GPS issue node has no edge to the service
  event (D22 cross_aspect_reach violation)">
- **Anti-pattern**: <fill — e.g. "wall-clock timestamps instead of
  conversation_now-derived; relative 'three weeks later' not resolved to
  2023-04-05">
