# Generic brain tool descriptions — candidate v1

**Subsequent check:** the old/new short mechanical probe completed at 24/24
per arm, with all 48 generated calls schema-valid. See the
[pre-eval review](S1E-PRE-EVAL-REVIEW-2026-09-08.md) for captures, usage,
remaining instruction conflicts and corrected context. The sections below
record the original authoring checkpoint; full-context model evaluation and
the production-shaped schema gate remain pending.

Authored at Tom's request: revise the descriptions used in the encoder eval,
keeping them generic enough for later use in the shared MCP surface. The
candidate changes tool and parameter descriptions for the same six tools.
It does not alter V3, journals, tool availability, argument shapes, dispatch
or live MCP definitions. No behavioral evaluation or deployment occurred.

Start with the [complete readable descriptions](../eval/fixtures/s1e_tool_descriptions_2026-09-08/descriptions.md).
The [exact old/new text](../eval/fixtures/s1e_tool_descriptions_2026-09-08/description_edits.json)
is saved by JSON pointer for every changed description.

## The revision

- Remove LLM-round pressure and host-specific instructions. Batching describes
  how operations execute and return results, without deciding when a caller
  should stop, defer, write a journal or begin another round.
- Make partial outcomes explicit: a successful batch response can contain a
  rejected operation or a failed edge. This is generic tool behavior, not a
  special encoder inspection phase.
- Use consistent target wording: exact IDs for existing nodes, exact titles
  for siblings created in the same remember batch. Creating the same title
  does not revise the existing node; siblings still win title collisions.
- Preserve the mechanics that prevent damage: omitted fields stay unchanged,
  whole replacements discard omitted text, swaps must match uniquely, the audit
  `reason` differs from stored `reasoning`, and reference replacement differs
  from omission. Preserve edge upsert, duplicate-edge avoidance, merge direction,
  absorbed-content preservation and locked-node restrictions.
- Describe fields by what they represent. Remove storage-table exposition and
  first-person caller assumptions. `thought` remains an optional interpretation
  that can change independently; no frequency rule or quota. Source references
  remain selective and can come from returned source records or supplied markers.
- Describe `get_nodes` by the data it returns and how `rich` changes the view.
  Remove the warning to use it sparingly. An ID can be fetched regardless of
  whether it appeared in the current result set. The size-dependent view limits
  and missing-ID reporting remain explicit.

The descriptions contain no encoder name, journal policy, stop policy, next-run
instruction or scenario-specific benchmark content. Short examples only explain
generic argument meaning, such as an edge connecting an estimate and schedule.

## Size and compatibility

| Measure | Frozen tools | Candidate |
|---|---:|---:|
| Tools | 6 | 6 |
| Description entries | 117 | 117 |
| Description characters, including repeated entries | 20,781 | 15,953 |
| Serialized tool JSON characters | 31,683 | 26,831 |

73 descriptions changed. Descriptive prose is **23.2% shorter**; the whole
serialized toolset is **15.3% shorter**. Serialization uses Python
`json.dumps(..., ensure_ascii=False)` on the parsed tool list. These are
character counts, not measured token, cost or quality improvements.

Offline validation passed:

- Removing only string-valued description annotations leaves exactly identical
  tool names, order, schemas, fields, required lists, defaults, alternatives and
  references. Fields literally named `description` are preserved.
- All six schemas validate against JSON Schema Draft 2020-12.
- All 19 actual saved calls from the 18-encode sanity cell retain the same schema
  verdict under both toolsets. The historical missing-`reason` operation remains
  invalid; validation did not turn that failure into a pass.
- The unchanged five-arm loader validates its original hashes. The candidate
  loader preserves the selected guide, gist and model settings and assigns a
  new identity for the changed tools.
- The existing batch probe can load the exported MCP definition. This checks
  format compatibility; the model-generation probe itself has not been run.

[Machine review](../eval/fixtures/s1e_tool_descriptions_2026-09-08/offline_review.json).
The first validation attempt stopped at a missing `jsonschema` dependency,
before running checks. Dependencies were installed only in
`/private/tmp/s1e-tool-schema-deps`; the project's environment was not modified.
Validation used jsonschema 4.26.0, attrs 26.1.0, referencing 0.37.0,
jsonschema-specifications 2025.9.1, rpds-py 2026.6.3 and typing-extensions 4.16.0.

## Artifacts and use in a later eval

- [API toolset](../eval/fixtures/s1e_tool_descriptions_2026-09-08/tools.api.json):
  the six `name` / `description` / `input_schema` objects.
- [MCP toolset](../eval/fixtures/s1e_tool_descriptions_2026-09-08/tools.mcp.json):
  the same six definitions with `inputSchema` spelling.
- [brain_batch probe input](../eval/fixtures/s1e_tool_descriptions_2026-09-08/brain_batch.mcp.json):
  a single MCP definition consumable by `eval/mcp_batch_probe.py --variant`.
- [Author/review/loader](../eval/fixtures/s1e_tool_descriptions_2026-09-08/candidate.py):
  deterministic description edits, offline checks and `load_candidate()`.
- [Manifest](../eval/fixtures/s1e_tool_descriptions_2026-09-08/manifest.json):
  original tool hash and authored artifact hashes. Existing files cannot be
  overwritten by the authoring command; a later text change needs a new identity.

`load_candidate('v3_titles')` returns the frozen V3 + cues arm with only its tools
replaced. Its identity is
`22109e3dbbbe780e6f45989102a703a7c0404996ed313e3a869e5d8056e67ac7`.
Use that identity in any future corpus cache key; do not label the outputs as
the original V3 + cues arm. No existing pinned runner or capture was modified.

Offline recheck, without model calls:

```sh
cd /Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242
./dev env PYTHONPATH=/private/tmp/s1e-tool-schema-deps python3 eval/fixtures/s1e_tool_descriptions_2026-09-08/candidate.py
```

## Later application to shared MCP

The reusable language belongs in the existing owners, not in an encoder-only
copy: tool descriptions in `servers/brain_mcp.py`; shared field, swap and
connect-to descriptions in `servers/contract.py`; source-reference descriptions
in their existing MCP schema generators. Apply by tool name and field meaning;
the saved JSON-pointer indices describe this frozen tool order only.

Shared field descriptions also generate text inside system prompts. That
generated system text is deliberately held fixed in this tool-only candidate.
A later shared-source application will need to inspect and evaluate that wider
change, including singular tools and other callers. This candidate does not
claim the entire MCP surface has been reviewed or rewritten.

Before promotion, run the repository's batch-generation and production-shaped
schema gates against the candidate. `mcp_batch_probe.py` already accepts its
single-tool export. `mcp_schema_gate.py` currently captures the active S2 toolset
and has no candidate argument, so its candidate integration must be made
explicit before claiming it tested this version. Behavioral comparisons should
keep V3 and factual context fixed and retain Tom's repeated sequential design.
The authoring request is complete; these model evals remain the next stage.

Tracked HEAD remains `9a1727f`; no tracked runtime change, registration, merge,
daemon restart or deployment was performed.
