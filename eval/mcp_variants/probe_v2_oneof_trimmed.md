# brain_batch generation-shape probe — v2_oneof_trimmed

model=claude-sonnet-4-6 repeats=10  (tool_choice forced; production is auto — see header note in script)

| scenario | dimension | pass | failures (first 2 distinct) |
|---|---|---|---|
| reason_swap | reason vs reasoning (the 2026-06-12 incident) | 10/10 | — |
| both_fields | reasoning as field update + reason as audit, together | 10/10 | — |
| invented_op | merge intent → absorb, not an invented op name | 10/10 | — |
| connect_to_vs_connect | new-node edges go through connect_to | 10/10 | — |
| double_emit | edge expressed exactly once | 10/10 | — |
| absorb_content | absorb is content-destructive without an override | 10/10 | — |
| relation_as_op | relations are edge fields, not op names | 10/10 | — |
| archive_required | archive carries its required id | 10/10 | — |
