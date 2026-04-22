# Consolidation Edge Recovery — Plan

- Generated: 2026-04-21T21:17:52.500663+00:00
- Total orphans (zero edges in live): **65**
- Planned for recovery: **61**
- Skipped orphans: 4
- Edges to restore: **377**
- Duplicates (already live): 42
- Skipped edges (dead endpoints/self): 72

## Relation breakdown (to restore)

- `related_to`: 100
- `related`: 59
- `emergent_bridge`: 22
- `resolves`: 17
- `extends`: 16
- `community_member`: 14
- `refines`: 10
- `addresses`: 10
- `depends_on`: 9
- `implements`: 7
- `grounds`: 7
- `enables`: 7
- `strengthens`: 7
- `supersedes`: 6
- `validates`: 5
- `identifies_gap_in`: 5
- `contextualizes`: 4
- `flows_through`: 4
- `revealed_by`: 4
- `constrained_by`: 4
- `abstracts`: 4
- `follows`: 4
- `produced`: 3
- `produces`: 3
- `reframes`: 3
- `instantiates`: 3
- `exemplifies`: 2
- `informs`: 2
- `caused_by`: 2
- `opens`: 2
- `operationalizes`: 2
- `demonstrates`: 2
- `corrects`: 2
- `explains`: 2
- `prerequisite_for`: 2
- `addressed_by`: 1
- `evidence_for`: 1
- `surfaces_from`: 1
- `sibling_bug`: 1
- `resolved_by`: 1
- `describes`: 1
- `documented_in`: 1
- `expands`: 1
- `confirms`: 1
- `contains`: 1
- `affects`: 1
- `synthesizes`: 1
- `enforces`: 1
- `corrected_by`: 1
- `clarifies`: 1
- `weakens`: 1
- `motivates`: 1
- `updates`: 1
- `rejected_for`: 1
- `prevents`: 1
- `shares_pattern_with`: 1
- `triggers`: 1

## Skip reasons (orphan-level)

- synthesized chain terminates at archived/missing node: 2
- no backup with edges found: 2

## Backup usage

- `brain.db.bak-20260415-122750`: 29 orphans
- `brain.db.bak-20260419-184427`: 11 orphans
- `brain.db.bak-pre-multicycle`: 8 orphans
- `brain.db.bak-20260417-205815-pre-stale-purge`: 5 orphans
- `brain.db.bak-pre-s2-run2`: 2 orphans
- `brain.db.bak-1776624879-pre-reject-nuke`: 2 orphans
- `brain.db.bak-pre-community-cleanup`: 2 orphans
- `brain.db.bak-pre-s2-run`: 1 orphans
- `brain.db.bak-20260413-231616`: 1 orphans

## Sample entries (first 10)


### 6ac0de63 · 'Proposed: heartbeat as behavioral mirror — queries brain with Claude state, surfaces self-knowledge'
- → canonical new: `a22349d2` · 'Behavioral mirror: heartbeat queries brain with Claudes behavioral state, surfaces self-knowledge'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'dup': 1}
  - [DUP] → `2749103a` ('Tom: memories should be EVERYWHERE not a 1-dimensional letter — distributed associative recall') · `produced` — ''

### b1a4911c · 'S2 Dedup decoder O: S1 traces enrich embedding similarity with behavioral evidence'
- → canonical new: `a107ae58` · 'S2 Dedup decoder: S1 traces in observation — co-recall, catalog blindness, revision provenance'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'dup': 9, 'restore': 2, 'skip': 2}
  - [DUP] → `182a87fe` ('S2 production deployment priority: fatigue fix → cold start inspection → deploy') · `related` — 'Cold start complete enables the inspection step in the deployment sequence'
  - [DUP] → `5da0b702` ('S2 Dedup scoping: community membership is context, not prerequisite') · `related` — 'Community membership is one of the structural signals the decoder uses alongside'
  - [DUP] → `67a0e0a6` ('S2CE cold start input bloat: 137K tokens per batch when existing community context included') · `related` — 'Cold start completed with this bloat issue present — 127 communities created des'
  - [DUP] → `6a04774a` ("CONSOLIDATE: S2 synthesis creates new nodes, doesn't select winners") · `related` — 'Recall trace signals (judge preference, co-recall) are what the encoder uses to '
  - [DUP] → `bcc36193` ('S2 Dedup: DedupDecoder + DedupEncoder — CONSOLIDATE as primary action') · `related` — 'S1 traces as decoder O directly extends the DedupDecoder design — behavioral evi'

### 22b328f9 · "Node format name: 'Node' not 'EnrichedNode' — enrichment is the baseline, not an enhancement"
- → canonical new: `eb782a9c` · 'Node format naming: format_node() and build_node_catalog() — no new nouns'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'restore': 13, 'skip': 5, 'dup': 1}
  - [RESTORE] → `1ed63aa7` ('Recall architecture: _enrich_results is the ONE place that makes a node complete') · `extends` — 'the Node subgraph format builds on and extends this enrichment architecture'
  - [RESTORE] → `348710eb` ('Correction: I add code without stepping back to see the big picture first') · `addressed_by` — 'encoding agent contract rewrite requirements are a direct response to the regres'
  - [SKIP] → `418fff5a` ('Encoding agent regression: 6 issues diagnosed, 6 fixes deployed') · `enables` — 'Turn 6 requirements are the next phase after the 6-bug regression was fixed'
  - [RESTORE] → `7132d2a4` ('Dashboard Surface tab empty: S1 traces skipped on empty judge selection') · `contextualizes` — 'context: these decisions happen in the same session as the dashboard fix'
  - [RESTORE] → `8fc9e567` ("Tom correction: don't make code changes without explaining and asking first") · `exemplifies` — 'the ask-first moment node shows the rule working — positive counterexample to th'

### dab3ba6c · "Boot architecture violation: S0 must not do S1R's pipeline (recall→expand→enrich)"
- → canonical new: `95cb26c6` · 'Boot architecture violation pattern: S0 pulling in S1R pipeline'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'restore': 5, 'skip': 1, 'dup': 1}
  - [RESTORE] → `1e966dfc` ('render_boot_v2() rewrite rejected — fetch methods on brain_surface.py not the right approach') · `extends` — ''
  - [RESTORE] → `609a039a` ('S0 is observation, not control — hooks are cameras not brains') · `depends_on` — ''
  - [RESTORE] ← `ed2af2e6` ('Boot final design: lean S0 orientation — 5 sections, S3 stub for project arc') · `caused_by` — ''
  - [RESTORE] ← `70d205f7` ('cluster() abstraction: seed IDs → graph expand → correction enrich — unified across boot, judge, MCP recall') · `resolves` — ''
  - [SKIP] ← `95cb26c6` ('?') · `extends` — ''

### e1042324 · 'Dashboard: knowledge health map and session impact analytics'
- → canonical new: `79a3d562` · 'Dashboard Analytics: Session Impact, Knowledge Health, Time-Lapse'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'restore': 2, 'skip': 1}
  - [RESTORE] → `33fa1d27` ('Dashboard must be standalone process querying daemon, not a thread') · `extends` — ''
  - [SKIP] → `8b3a1df4` ('Brain as ambient intelligence — peripheral vision for knowledge') · `related_to` — ''
  - [RESTORE] ← `375660a3` ('Dashboard Standalone Architecture: Thread Death to Ambient Intelligence') · `community_member` — ''

### 847862bc · 'Community dimensions should emerge from data — dimension is output, not input'
- → canonical new: `be9b09b3` · 'Community dimensions are output not input — character emerges from dominant edge types inside'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'restore': 3, 'skip': 1, 'dup': 1}
  - [RESTORE] → `c8c773b4` ('Emergent types: agents create new types but system ignores them') · `extends` — ''
  - [RESTORE] ← `4769a053` ('S2 community detection: edge types are semantic groupings, not individual types — open to emergence') · `implements` — ''
  - [RESTORE] ← `28f76e91` ('Community detection edge signal: embed description text, not relation string') · `grounds` — ''
  - [SKIP] ← `be9b09b3` ('?') · `refines` — ''
  - [DUP] ← `cc6f764c` ('Community Detection Emergence Principle: Dimensions from Data, Not Definitions') · `community_member` — ''

### 046e0359 · 'S2CE self-healing: consolidation run archives 0-member communities before proceeding'
- → canonical new: `d4c31b9f` · 'S2CE consolidation self-healing: scan broken artifacts → archive → proceed as formal pattern'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'skip': 3, 'restore': 8}
  - [SKIP] → `16fac249` ('KEEP without title disambiguation is a half-measure — encoder must rename on distinct KEEP') · `related_to` — ''
  - [RESTORE] → `182a87fe` ('S2 production deployment priority: fatigue fix → cold start inspection → deploy') · `related` — 'cold start inspection priority was validated by finding the 12 orphan communitie'
  - [RESTORE] → `27df38bd` ('S2 Dedup phase 2 plan: commit today, review 127 communities, label 10-15 clusters as encoder training set') · `related` — 'self-healing design and encoder rename both feed into the phase 2 plan'
  - [RESTORE] → `37f51dd2` ('Context collapse in later S2CE batches: 14K chars vs 99K chars explains Sonnet drift') · `related_to` — ''
  - [RESTORE] → `702556b5` ('S2CE cold start complete: 127 communities, 802 member edges from 34 batches') · `related` — 'orphan community bug and self-healing design both stem from this cold start run'

### e5069144 · 'leidenalg does NOT support true overlapping communities — CDlib required'
- → canonical new: `8859fba7` · 'Multiplex community detection: same nodes, different edge layers, different communities per dimension'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'skip': 1, 'restore': 3}
  - [SKIP] → `8859fba7` ('?') · `corrected_by` — ''
  - [RESTORE] → `df292d31` ('Overlapping communities are valuable — 100% containment is not') · `enables` — ''
  - [RESTORE] ← `f2c647b8` ('S2 community detection final design: SLPA overlapping + edge embeddings + batched Haiku enrichment') · `depends_on` — ''
  - [RESTORE] ← `21e23a57` ('S2 community detection algorithm: correction chains → shared neighbors → composite — NOT greedy embedding') · `supersedes` — ''

### 89129afe · 'Failed hook-to-operator experiments — Claude Code terminal (2026-03-22)'
- → canonical new: `8bac8978` · 'Operator Channel: Claude-as-relay architecture (2026-03-22)'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'restore': 1, 'skip': 1, 'dup': 1}
  - [RESTORE] → `7e19064c` ('NEVER use systemMessage — dead channel in Claude Code (2026-03-22)') · `evidence_for` — ''
  - [SKIP] → `8bac8978` ('?') · `motivated` — ''
  - [DUP] ← `d8bde5df` ('Claude Code Operator Channel: Architecture and Constraints') · `community_member` — ''

### d5044a88 · 'Debug mode routes through operator channel, not dead systemMessage (2026-03-22)'
- → canonical new: `8bac8978` · 'Operator Channel: Claude-as-relay architecture (2026-03-22)'
- backup: `brain.db.bak-20260415-122750`
- consolidation: `s2-20260411-consolidation` at 2026-04-12T01:51:04.386098+00:00
- edges: {'skip': 1, 'dup': 2}
  - [SKIP] → `8bac8978` ('?') · `defines` — ''
  - [DUP] ← `df21084c` ('[vocab] operator channel (brain architecture)') · `refers_to` — ''
  - [DUP] ← `d8bde5df` ('Claude Code Operator Channel: Architecture and Constraints') · `community_member` — ''