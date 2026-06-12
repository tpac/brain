# brain_batch generation-shape probe — v0_live

model=claude-sonnet-4-6 repeats=5  (tool_choice forced; production is auto — see header note in script)

| scenario | dimension | pass | failures (first 2 distinct) |
|---|---|---|---|
| reason_swap | reason vs reasoning (the 2026-06-12 incident) | 5/5 | — |
| both_fields | reasoning as field update + reason as audit, together | 5/5 | — |
| invented_op | merge intent → absorb, not an invented op name | 5/5 | — |
| connect_to_vs_connect | new-node edges go through connect_to | 5/5 | — |
| double_emit | edge expressed exactly once | 5/5 | — |
| absorb_content | absorb is content-destructive without an override | 5/5 | — |
| relation_as_op | relations are edge fields, not op names | 5/5 | — |
| archive_required | archive carries its required id | 5/5 | — |
