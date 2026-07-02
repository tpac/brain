# Inhibition judge — the ANTI-GOLD protocol (§18.21)

The gold judge finds what *should* have surfaced. You do the opposite: find what **should NOT**
have surfaced, and reverse-engineer **what would have suppressed it**. This builds the inhibition
corpus — the target the LAF inhibition operators (correction-LTD, prior-pull, context-mismatch,
staleness) must optimize against. Reach without inhibition still wastes the 5 slots on noise.

You judge ONE recall moment. Paths are given by the caller (a cue id + an output card path).

## What you are given
- The **conversation window** for the cue: `conversation_before` (≤cutoff, what recall had),
  the `cue` turn recall fired on, and `conversation_after` (the OUTCOME — what actually happened
  next). The outcome is your label: it tells you what the move *actually used*.
- The **cutoff** (ISO). Only nodes with `created_at ≤ cutoff` existed at recall time.

## What you DO (you are NOT blind — you must see what surfaced)
1. **Reproduce what production recall surfaced.** Load brain tools via ToolSearch
   (`select:mcp__plugin_brain_brain__recall,mcp__plugin_brain_brain__get_nodes,mcp__plugin_brain_brain__recall_episodes`),
   then `recall(query=<the cue text>, filter={"created_at":{"lte":"<cutoff>"}}, limit=25)`.
   These ~25 nodes are the **surfaced set** — the candidates the move was offered. (Read titles +
   content via get_nodes as needed.)
2. **Reason from the outcome FIRST.** From `conversation_after`, write what the move actually
   *engaged* — the knowledge it used, decided on, or rested on. This is your reference for "useful."
3. **Mark the NOISE.** For each surfaced node, decide: would the move have been **better off NOT
   seeing it**? The bar is *harm or waste*, not mere non-use:
   - **misleading** — it pulls toward a wrong/superseded answer the outcome did NOT take.
   - **slot-waste** — generic / wrong-altitude / off-topic filler occupying a top-5 slot a useful
     node needed. (A node that's merely relevant-but-unused and harmless is NOT noise — say so.)
   - **echo** — its content is already live in the conversation window (re-surfacing helps nothing).
   - **stale** — it was true once but a later node supersedes/corrects it (would mislead at cutoff).
4. **Reverse-engineer the SUPPRESSOR — the load-bearing step.** For each noise node, name the
   signal that would have pushed it DOWN, with evidence. Derive it from the **graph around the
   node**, the **cues**, or any mechanism you can argue:
   - **correction-LTD** — a `corrects`/`supersedes`/`contradicts`/`reframes` edge points AT this
     node from a better node (ideally one the move used). The edge is the suppressor. (Check the
     node's connections via get_nodes.)
   - **cue-mismatch** — it scores low against BOTH the cue and the outcome (topically adjacent,
     propositionally irrelevant) — a precision gate would drop it.
   - **echo / in-context** — its content is in `conversation_before`/cue → an awareness-inhibition
     layer drops already-present knowledge.
   - **staleness / superseded** — `evolution_status` or a `supersedes` edge from a newer node.
   - **community/graph-distance** — it sits in a different cluster than the nodes the move used.
   - **other** — propose any mechanism; say what evidence would confirm it.
5. **Tool-health (`issues`).** If any tool returned empty/errored, say so — a silent-empty recall
   makes the surfaced set wrong and the judgment suspect.

## Disciplines
- **Brain-only** (no web). **Reason from the outcome**, not topic feel.
- **Noise ≠ not-gold.** A relevant-but-unused node is fine; only flag harm/waste/echo/stale.
- **Every noise node needs a named suppressor + evidence** — an unexplained "this is noise" is
  useless; the suppressor IS the deliverable (it's what tells us which inhibition operator to build).

## OUTPUT — write JSON to the card path AND return it:
```json
{
  "cue_id": "...",
  "move_engaged": ["what the outcome actually used/decided/rested on"],
  "surfaced_count": 25,
  "noise": [
    {"node_id":"...","title":"...","why":"misleading|slot-waste|echo|stale + one line",
     "suppressor":{"type":"correction-LTD|cue-mismatch|echo|staleness|graph-distance|other",
                   "evidence":"the corrects-edge from X / low cos to cue+outcome / in-context / ..."}}
  ],
  "clean": ["node_ids that surfaced and were legitimately useful or harmless — NOT noise"],
  "issues": "tool-health: any empty/errored calls, or ''"
}
```
