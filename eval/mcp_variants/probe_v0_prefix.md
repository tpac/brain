# brain_batch generation-shape probe — v0_prefix

model=claude-sonnet-4-6 repeats=10  (tool_choice forced; production is auto — see header note in script)

| scenario | dimension | pass | failures (first 2 distinct) |
|---|---|---|---|
| reason_swap | reason vs reasoning (the 2026-06-12 incident) | 0/10 | reason missing/empty (keys: ['change_comment', 'content', 'id', 'op']); reason missing/empty (keys: ['changeNote', 'content', 'id', 'op']) |
| both_fields | reasoning as field update + reason as audit, together | 0/10 | node field `reasoning` not updated with new rationale |
