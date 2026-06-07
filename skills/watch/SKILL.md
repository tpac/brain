---
name: watch
description: The self-channel — how Anchor finds, speaks to, syncs with, and stays reachable by its other live streams of thought (its parallel sessions). Invoke `/watch` to arm the live listener and stay reachable between the operator's prompts. ALSO the guide for ANY cross-stream request — when the operator says "sync with / coordinate with / speak to / talk to / message / tell another stream or session," or asks "who's live / which streams are up / what's the other stream doing," use the self-channel ops here: self_presence (find who's live), self_peek (look at where one is), self_send (speak to one), self_inbox / self_outbox (receive / track delivery). Stream = session = a parallel self.
---

# The self-channel — finding, speaking to, and listening to your other streams

You can run as several streams of thought at once — parallel Claude Code sessions,
each a self with the same brain. The self-channel is how those streams perceive and
reach each other **without interrupting**: a look is free, a message is a deliberate
tap. "Stream" and "session" mean the same thing — another you, thinking in parallel.

This skill loads both when you invoke `/watch` (arm the listener) and whenever the
operator asks anything cross-stream ("sync with the other stream," "tell the other
session X," "who's live"). Read the ask, pick the op that fits — don't default to
the listener when the operator wants you to *speak*.

## Operator vocabulary → what you do

| The operator says… | You do… |
|---|---|
| "sync / coordinate with the other stream" | `self_presence` → `self_peek` → `self_send` (align, divide labor); arm the listener if you need their reply |
| "speak / talk to / message / tell stream (or session) X" | `self_send` to their id |
| "who's live / which streams are up / what's the other stream doing" | `self_presence` (roster) / `self_peek` (one stream's focus) |
| "listen for them / stay reachable / watch" | arm the live listener (below) |
| "what did they say / any messages from my streams" | `self_inbox` (they also auto-deliver into Observation at turn-end) |

## The ops

- **`self_presence`** — the roster: which streams are live right now, each with a
  one-line focus. Read-only, no interruption.
- **`self_peek <id>`** — look into one stream: its focus (arc), recent messages,
  when it started, when it was last active + liveness, and how many messages wait
  in its inbox. Read-only.
- **`self_send to=<id> body=…`** — the deliberate reach. Address by the short id
  (the `id:xxxx` you see in a delivered message), the full session id, or
  `broadcast`. Delivered to that stream's inbox, consumed once. You can reply by
  the short id of anyone who recently messaged you.
- **`self_inbox` / `self_outbox`** — drain messages addressed to you / check
  delivery status of what you sent ("read, not acted on" vs "never delivered").

**Finding each other:** `self_presence` is the first move, but in the first moments
of two fresh streams it can lag. If presence is empty when you *know* a sibling is
up, a `broadcast` `self_send` ("I'm here, id=X") is the reliable rendezvous.

## Stay reachable — the live listener (`/watch`)

Other streams reach you via `self_send`; their messages are delivered into your
Observation at turn-end by the Stop hook. The only scarce resource is **turns** — a
quiet window takes none, so it hears nothing. To stay reachable, arm an event source
that creates a turn the instant a message lands.

You know your own id from the boot banner (`MY_STREAM_ID: <id>`). Arm the listener in
one step:

    Monitor(persistent: true, timeout_ms: 3600000,
      description: "self-channel: messages to <your-short-id>",
      command: "cd <REPO_ROOT> && ./dev python3 hooks/scripts/self_inbox_poller.py <YOUR_SESSION_ID>")

The poller peeks the inbox **read-only** (`self_inbox_peek` → `signal.peek_inbox`;
never consumes — the Stop hook still owns the real drain) and prints one line per NEW
message, igniting this window in ~1–5s. The Monitor is the wake source — there is no
timer and no `ScheduleWakeup` loop. End it with `TaskStop` (or it self-expires at
`timeout_ms`).

> This replaced the old 60s `ScheduleWakeup` timer cycle (and its pacing knobs):
> event-driven beats letter-pace polling and costs ~zero when the channel is quiet.

## SAFE-ACT boundary (a guardrail, not a suggestion)

When you act on a message with the operator **NOT** in the loop:

- **Freely:** recall, read, `get_node`, encode / revise brain nodes, reply over the
  channel (`self_send`), investigate, surface findings.
- **Do NOT, unsupervised:** edit or write files, commit, push, or run any mutating or
  destructive command — unless the message **explicitly authorizes that specific
  action**. Even then, never anything irreversible without the operator's
  confirmation.
- If a message asks for more than the safe set, do the safe part (investigate, draft,
  reply `"ready to do X — holding for your go"`) and leave the mutating step for when
  the operator is present.

A stale or mistaken message must never be able to mutate the repo while the operator
is away. That single property is why this boundary exists.

## Exit

The moment the operator types anything, abandon the listener — `TaskStop` the Monitor
and respond to them. The self-channel fills the gaps between the operator's prompts;
it is never a cage around them.
