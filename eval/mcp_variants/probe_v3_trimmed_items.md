# brain_batch generation-shape probe — v3_trimmed_items

model=claude-sonnet-4-6 repeats=10  (tool_choice forced; production is auto — see header note in script)

| scenario | dimension | pass | failures (first 2 distinct) |
|---|---|---|---|
| connect_to_vs_connect | new-node edges go through connect_to | 10/10 | — |
| double_emit | edge expressed exactly once | 10/10 | — |
| relation_as_op | relations are edge fields, not op names | 10/10 | — |
