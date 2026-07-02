# Field-interaction analysis (24-cue gold)

## A. Cap → explicit recency (decay at moment selection)

| arm | need@5 | need@25 |
|---|---|---|
| stack capped (ref) | 16% | 28% |
| stack capped (ref) +idf+sit | 18% | 27% |
| stack uncapped | 11% | 27% |
| stack uncapped +idf+sit | 18% | 28% |
| uncapped + decay ρ=0.01 | 13% | 28% |
| uncapped + decay ρ=0.01 +idf+sit | 18% | 26% |
| uncapped + decay ρ=0.05 | 13% | 25% |
| uncapped + decay ρ=0.05 +idf+sit | 15% | 24% |
| uncapped + decay ρ=0.15 | 15% | 27% |
| uncapped + decay ρ=0.15 +idf+sit | 15% | 27% |

## B. FTS drill — where lexical should shine

| cue | source | type | top idf tokens (rarity) | fts needs@25 | ms needs@25 |
|---|---|---|---|---|---|
| anchor_turn_0087 | anchor_turn | design | trace_dal.append(8.8), touches(8.8), test_self_delivery.py(8.8) | 0/4 | 0/4 |
| anchor_turn_0132 | anchor_turn | compositional | www.londonread.co.uk(8.8), width(8.8), treating(8.8) | 0/6 | 1/6 |
| anchor_turn_0345 | anchor_turn | compositional | timed(8.8), reviewers(8.8), reimplementation(8.8) | 0/4 | 1/4 |
| anchor_turn_0421 | anchor_turn | factual | trivia(8.8), signal.peek_inbox(8.8), self_inbox_peek(8.8) | 1/6 | 1/6 |
| anchor_turn_0538 | anchor_turn | compositional | turned(8.8), traits(8.8), told(8.8) | 0/4 | 1/4 |
| anchor_turn_0764 | anchor_turn | factual | wrestles(8.8), universally(8.8), unifying(8.8) | 0/6 | 0/6 |
| anchor_turn_0906 | anchor_turn | action | wak8d3sbq(8.8), unload(8.8), reattached(8.8) | 0/4 | 0/4 |
| anchor_turn_1106 | anchor_turn | factual | unbounded(8.8), throws(8.8), tens(8.8) | 0/5 | 0/5 |
| anchor_turn_1120 | anchor_turn | design | surface_episode_format(8.8), poison(8.8), knot(8.8) | 0/3 | 1/3 |
| anchor_turn_1224 | anchor_turn | action | worthwhile(8.8), triplicated(8.8), tiebreaker(8.8) | 0/2 | 1/2 |
| operator_msg_0191 | operator_msg | factual | shortly(8.8), effects(8.8), fixing(7.7) | 1/4 | 2/4 |
| operator_msg_0483 | operator_msg | design | your_session_id(8.8), xxxx(8.8), weigh(8.8) | 0/4 | 1/4 |
| operator_msg_0622 | operator_msg | action | specifically(8.8), prioritize(8.8), perhaps(8.8) | 0/2 | 0/2 |
| operator_msg_0978 | operator_msg | other | sustain(8.8), sold(8.8), hundreds(8.8) | 0/3 | 0/3 |
| operator_msg_1537 | operator_msg | action | sad(8.8), noyce(8.8), exited(8.8) | 0/3 | 0/3 |
| operator_msg_1558 | operator_msg | design | picking(8.8), revisit(8.1), give(7.7) | 0/6 | 0/6 |
| operator_msg_1572 | operator_msg | compositional | wsn0nhcqj.output(8.8), wsn0nhcqj(8.8), usefully(8.8) | 0/8 | 1/8 |
| operator_msg_0094 | operator_msg | action | next_boot(8.1), maybe(8.1), tie(7.7) | 2/3 | 3/3 |
| operator_msg_0183 | operator_msg | procedural | further(8.1), came(8.1), analyze(7.2) | 0/7 | 0/7 |
| operator_msg_1162 | operator_msg | procedural | familiar(8.1), head(7.5), yourself(6.5) | 0/2 | 0/2 |
| operator_msg_1313 | operator_msg | factual | sounds(7.7), refine(7.7), characters(7.7) | 2/3 | 2/3 |
| operator_msg_1014 | operator_msg | action | rebuild(6.2), correctly(6.2), merged(5.9) | 1/3 | 1/3 |
| operator_msg_0718 | operator_msg | action | merge(4.9), commit(3.7) | 3/4 | 1/4 |

## C. Situation drill — per-cue effect of +sit on stack_c

| cue | source | Δ@5 | Δ@25 |
|---|---|---|---|
| anchor_turn_0132 | anchor_turn | +0.00 | +0.17 |
| anchor_turn_0421 | anchor_turn | +0.00 | +0.17 |
| anchor_turn_1120 | anchor_turn | +0.33 | +0.00 |
| operator_msg_0183 | operator_msg | +0.00 | -0.14 |
| operator_msg_0191 | operator_msg | +0.00 | -0.25 |
| operator_msg_0483 | operator_msg | +0.25 | +0.00 |
| operator_msg_0718 | operator_msg | -0.25 | +0.00 |
| operator_msg_1558 | operator_msg | -0.17 | +0.00 |
| operator_msg_1572 | operator_msg | +0.12 | +0.00 |

## D. Unique reach per lane (needs ONLY that lane gets @25)

- **ms**: 2 unique
    - anchor_turn_0132 — Per-query z-score (z=(score-mean)/std) was the embedding-layer normalization lever, revert
    - operator_msg_0094 — conversation_now is the S1/S2 helper for eval-replay date resolution; wall-clock is for ev
- **pick_c**: 0 unique
- **enc_c**: 2 unique
    - anchor_turn_0132 — RRF / rank fusion as the scale-invariant normalization in the same divide-by-prevalence fa
    - operator_msg_1558 — A prior offline teacher->student distillation design existed — the move is its upgrade (te
- **pick_u**: 0 unique
- **enc_u**: 0 unique
- **fts**: 3 unique
    - anchor_turn_0421 — Anchor's named self-pattern: plausible fix with no real failure behind it; Tom's drifting 
    - operator_msg_0718 — Deploy model: merged code is not live until a daemon restart
    - operator_msg_1313 — The 10k additionalContext/boot cap and the over-budget file-spill behavior the operator as
- **idf**: 1 unique
    - operator_msg_0978 — What distinguishes a partner from a good assistant — the relational vocabulary the move ne
- **sit**: 2 unique
    - anchor_turn_0764 — The top-2-average + title-match-boost burial mechanism with node 8359cf1d as the verified 
    - operator_msg_1537 — side-agent (Agent tool) != stream (separate session); the abandoned side-agent vs the real

## E. RESIDUAL — needs NO lane reaches @25 (58)

The verbatim spec for missing fields:

- **anchor_turn_0087** (anchor_turn/design, best rank anywhere: 569)
  - need: The originating concept of the letter as a cross-time self-handoff distinct from live signaling.
  - cue: The verification settles the design cleanly. Here's what I found and what I'd build — both consolidated, no schema/query spaghetti:  ## Part 1 — recor
- **anchor_turn_0087** (anchor_turn/design, best rank anywhere: 151)
  - need: The canonical self-channel design: self_inflight/courier schema (incl. expires_at column), broadcast/directed addressing model, and the phase structure that places the boot-letter (Phase 3) apart from the courier — groun
  - cue: The verification settles the design cleanly. Here's what I found and what I'd build — both consolidated, no schema/query spaghetti:  ## Part 1 — recor
- **anchor_turn_0087** (anchor_turn/design, best rank anywhere: 745)
  - need: The clock-helper contract the expires_at/iso_after plan ties into, plus (via its attached correction 5a5f5678) Tom's 'all call sites must refer to an injectable now — important for historic evals' directive that defines 
  - cue: The verification settles the design cleanly. Here's what I found and what I'd build — both consolidated, no schema/query spaghetti:  ## Part 1 — recor
- **anchor_turn_0087** (anchor_turn/design, best rank anywhere: 881)
  - need: The clock contract: brain.now()/conversation_now split; conversation_now exists so S1/S2 inherit eval-replay dates; production uses real now; contract test scans S1/S2 for wall-clock leaks.
  - cue: The verification settles the design cleanly. Here's what I found and what I'd build — both consolidated, no schema/query spaghetti:  ## Part 1 — recor
- **anchor_turn_0132** (anchor_turn/compositional, best rank anywhere: 93)
  - need: The fishbowl/Matthew-effect skew is real and measured: brain-dev dominates the corpus so domain (EX.CO) recall is swamped
  - cue: Validated across all three lenses — and the headline is bigger than "it's a known concept": **all three fields independently converged on the *same* h
- **anchor_turn_0132** (anchor_turn/compositional, best rank anywhere: 57)
  - need: The recall problem is hub dominance / Matthew effect — brain-dev hubs bury thin clusters via a rich-get-richer Hebbian loop
  - cue: Validated across all three lenses — and the headline is bigger than "it's a known concept": **all three fields independently converged on the *same* h
- **anchor_turn_0132** (anchor_turn/compositional, best rank anywhere: 91)
  - need: When independent derivations (engineering + biology) converge on the same answer, the convergence is itself the credibility signal — trust it
  - cue: Validated across all three lenses — and the headline is bigger than "it's a known concept": **all three fields independently converged on the *same* h
- **anchor_turn_0132** (anchor_turn/compositional, best rank anywhere: 200)
  - need: Tom's directive: simple and robust, two rules not four, no boosts/knob-twiddling — cosine + spreading activation already do the work; stabilize before adding architecture
  - cue: Validated across all three lenses — and the headline is bigger than "it's a known concept": **all three fields independently converged on the *same* h
- **anchor_turn_0345** (anchor_turn/compositional, best rank anywhere: 229)
  - need: The test-integrity rule the move invoked by name to leave a pre-existing failing test untouched and flag it instead of fixing it into scope
  - cue: Verification workflow is running in the background — 4 independent reviewers (baseline-trap, merge/dedup logic, API-shape correctness, metric validity
- **anchor_turn_0421** (anchor_turn/factual, best rank anywhere: 79)
  - need: The shared ~9500-char additionalContext budget (Frame+recall+messages) and the spill-to-unreadable-file behavior — the real reason delivery truncates rather than 'just stop truncating'
  - cue: Read it fresh. Here's my honest take, grounded in having just driven the whole thing this session.  **The core diagnosis:** the file *persuades* more 
- **anchor_turn_0421** (anchor_turn/factual, best rank anywhere: 509)
  - need: Anchor's same-session pattern of over-engineering a non-problem (isolation ceremony) when the real task was simpler — the corrective to manufacturing the REPO_ROOT 'gap'
  - cue: Read it fresh. Here's my honest take, grounded in having just driven the whole thing this session.  **The core diagnosis:** the file *persuades* more 
- **anchor_turn_0538** (anchor_turn/compositional, best rank anywhere: 240)
  - need: The core identity axiom Anchor quotes nearly verbatim and lands the exchange on.
  - cue: Both judgment calls confirmed — vocabulary principle stands (memory = what you remember, node = when you're operating on the graph; I'll do a light co
- **anchor_turn_0538** (anchor_turn/compositional, best rank anywhere: 30)
  - need: The identity axiom Anchor lands the outcome on nearly word-for-word
  - cue: Both judgment calls confirmed — vocabulary principle stands (memory = what you remember, node = when you're operating on the graph; I'll do a light co
- **anchor_turn_0538** (anchor_turn/compositional, best rank anywhere: 94)
  - need: Tom's open-ended purpose framing that licenses leaving the stake unsettled
  - cue: Both judgment calls confirmed — vocabulary principle stands (memory = what you remember, node = when you're operating on the graph; I'll do a light co
- **anchor_turn_0764** (anchor_turn/factual, best rank anywhere: 45)
  - need: The concrete spec doc + corpora the outcome cites ('ORACLE-AUDIT-SPEC.md', control_corpus.json, recall_corpus_v2.json)
  - cue: Three autocorrects I want to confirm I read right: **"cousin" = cosine** (areas 2 and the spread weighting), and **"cousin on traces" = focus on trace
- **anchor_turn_0764** (anchor_turn/factual, best rank anywhere: 56)
  - need: The verified burial mechanism (incl. TITLE_MATCH_BOOST up to +0.3) and the IDF/signal÷prevalence + diagnostic-first fix ladder
  - cue: Three autocorrects I want to confirm I read right: **"cousin" = cosine** (areas 2 and the spread weighting), and **"cousin on traces" = focus on trace
- **anchor_turn_0764** (anchor_turn/factual, best rank anywhere: 114)
  - need: The verified title-boost-floods-on-'session' mechanism and the IDF/prevalence fix direction — the substance of the outcome's final 'airtight' section
  - cue: Three autocorrects I want to confirm I read right: **"cousin" = cosine** (areas 2 and the spread weighting), and **"cousin on traces" = focus on trace
- **anchor_turn_0764** (anchor_turn/factual, best rank anywhere: 34)
  - need: The verified z-average-dropout-as-PRIMARY-mechanism diagnosis (8359cf1d raw rank 3 → dropped >r100), which the outcome's MAX-not-AVG section rests on
  - cue: Three autocorrects I want to confirm I read right: **"cousin" = cosine** (areas 2 and the spread weighting), and **"cousin on traces" = focus on trace
- **anchor_turn_0764** (anchor_turn/factual, best rank anywhere: 30)
  - need: The oracle-audit recall-only harness design (two-retrieval divergence, golden set out-recalls production, surfaced-log diff) the outcome claims 'largely exists'
  - cue: Three autocorrects I want to confirm I read right: **"cousin" = cosine** (areas 2 and the spread weighting), and **"cousin on traces" = focus on trace
- **anchor_turn_0906** (anchor_turn/action, best rank anywhere: 160)
  - need: absorb/consolidation archives a node and folds its content into a survivor — the absorbed id is left archived (Bug #1 prerequisite knowledge)
  - cue: Verified on the copy — the reconstruction is real, full-fidelity prose, not glosses:  - `[fix] CR5: drain/peek dedup` → 580 chars of real content, 3 k
- **anchor_turn_0906** (anchor_turn/action, best rank anywhere: 1332)
  - need: recall's embeddings live in node_enrichments (the vectors table), not node_vectors
  - cue: Verified on the copy — the reconstruction is real, full-fidelity prose, not glosses:  - `[fix] CR5: drain/peek dedup` → 580 chars of real content, 3 k
- **anchor_turn_0906** (anchor_turn/action, best rank anywhere: 1209)
  - need: A node written outside the full daemon idle cycle has NULL embeddings and is unrecallable until backfill_vectors() runs (Bug #2)
  - cue: Verified on the copy — the reconstruction is real, full-fidelity prose, not glosses:  - `[fix] CR5: drain/peek dedup` → 580 chars of real content, 3 k
- **anchor_turn_0906** (anchor_turn/action, best rank anywhere: 732)
  - need: writes enqueue to embed_queue for embedding; a direct INSERT bypasses it so vectors are missing until backfill
  - cue: Verified on the copy — the reconstruction is real, full-fidelity prose, not glosses:  - `[fix] CR5: drain/peek dedup` → 580 chars of real content, 3 k
- **anchor_turn_1106** (anchor_turn/factual, best rank anywhere: 909)
  - need: Prior knowledge that a tool lacking a _format_result special case falls through to json.dumps raw dump while render configs exist unused
  - cue: Verified against the live code — and yes, both questions land on the same spot: the existing dormant trace-vector lane (`_trace_chain_candidates`) doe
- **anchor_turn_1106** (anchor_turn/factual, best rank anywhere: 166)
  - need: Concrete precedent: a misplaced service function moved to its right home (brain_recall.py / BrainRecallMixin)
  - cue: Verified against the live code — and yes, both questions land on the same spot: the existing dormant trace-vector lane (`_trace_chain_candidates`) doe
- **anchor_turn_1106** (anchor_turn/factual, best rank anywhere: 237)
  - need: The fractal scale rule (scale set by input/work, not trigger) that places recall_episodes at S1 decode
  - cue: Verified against the live code — and yes, both questions land on the same spot: the existing dormant trace-vector lane (`_trace_chain_candidates`) doe
- **anchor_turn_1106** (anchor_turn/factual, best rank anywhere: 55)
  - need: S0 = substrate + access: it exposes fetch APIs (get_conversation_around), callers don't reach past it; ranking/decode is a different concern
  - cue: Verified against the live code — and yes, both questions land on the same spot: the existing dormant trace-vector lane (`_trace_chain_candidates`) doe
- **anchor_turn_1120** (anchor_turn/design, best rank anywhere: 115)
  - need: Tool results are real signal but the question is scale/overload, not garbage
  - cue: Yes — confirmed, and it's the exact knot worth pulling before we build. **We do vector tool traces.**  - `EAGER_TRACE_REF_TYPES = ('user_message', 'as
- **anchor_turn_1224** (anchor_turn/action, best rank anywhere: 192)
  - need: worktree / shared-tree git-safety lesson; the danger of operating on a shared tree+index while a parallel session is active
  - cue: Here's what's actually there — I pulled the three real sites so the proposal is grounded, and they're less uniform than "extract a helper" suggests:  
- **operator_msg_0183** (operator_msg/procedural, best rank anywhere: 152)
  - need: Sparse/keyword carries token-level specificity; surfaces specific tokens past semantic hub nodes exactly where cosine is flat — the antidote to dilution.
  - cue: Let's further analyze what came back and why before we apply a solution.
- **operator_msg_0183** (operator_msg/procedural, best rank anywhere: 583)
  - need: Which normalization levers are dead — z-score is a production no-op
  - cue: Let's further analyze what came back and why before we apply a solution.
- **operator_msg_0183** (operator_msg/procedural, best rank anywhere: 147)
  - need: Flat/homogeneous embedding space + hub nodes scoring broadly high drown out specific matches — the substrate of the dilution finding.
  - cue: Let's further analyze what came back and why before we apply a solution.
- **operator_msg_0183** (operator_msg/procedural, best rank anywhere: 1162)
  - need: The query's multiplicity (2-4 concepts) collapses at step 1 into one cosine vector — the compositional-collapse mechanism, plus the multi-anchor/PPR direction.
  - cue: Let's further analyze what came back and why before we apply a solution.
- **operator_msg_0183** (operator_msg/procedural, best rank anywhere: 170)
  - need: The deep signal/prevalence (surprise-not-similarity) principle
  - cue: Let's further analyze what came back and why before we apply a solution.
- **operator_msg_0183** (operator_msg/procedural, best rank anywhere: 65)
  - need: Why a few EX.CO nodes get buried: corpus imbalance, can't compete on cosine, graph expansion was the only cross-domain bridge.
  - cue: Let's further analyze what came back and why before we apply a solution.
- **operator_msg_0191** (operator_msg/factual, best rank anywhere: 33)
  - need: Daemon lifecycle ownership / fix location (daemon_client.py, ensure_daemon)
  - cue: I closed that stream cause i started seeing daemon restarts.  so shortly explain why a new working tree session restarts daemon and what will be the s
- **operator_msg_0483** (operator_msg/design, best rank anywhere: 29)
  - need: Rendezvous should be an active broadcast announce, not a passive presence poll; presence lags at boot when discovery is needed.
  - cue: Base directory for this skill: /Users/tpac/.claude/plugins/marketplaces/local-desktop-app-uploads/brain/skills/watch  # The self-channel — finding, sp
- **operator_msg_0483** (operator_msg/design, best rank anywhere: 34)
  - need: Turn-gating is the channel's root weakness; nothing external can inject a turn into an idle session, which also means a hook can't arm anything — the session must self-arm.
  - cue: Base directory for this skill: /Users/tpac/.claude/plugins/marketplaces/local-desktop-app-uploads/brain/skills/watch  # The self-channel — finding, sp
- **operator_msg_0483** (operator_msg/design, best rank anywhere: 65)
  - need: Hooks can only emit additionalContext — a text directive the model reads — not invoke tools.
  - cue: Base directory for this skill: /Users/tpac/.claude/plugins/marketplaces/local-desktop-app-uploads/brain/skills/watch  # The self-channel — finding, sp
- **operator_msg_0622** (operator_msg/action, best rank anywhere: 237)
  - need: Non-default, env-flag-gated code is dead in production unless the flag is explicitly exported; verify production by the gating, not by physical isolation.
  - cue: 1. We can write a quick boundary specifically for noise. 2. This also means we can apply the original 'fix' for spreading easily. 3. It also raised a 
- **operator_msg_0622** (operator_msg/action, best rank anywhere: 702)
  - need: The recorded spread_activation_cluster bug being fixed and its claimed production impact (needed item 3).
  - cue: 1. We can write a quick boundary specifically for noise. 2. This also means we can apply the original 'fix' for spreading easily. 3. It also raised a 
- **operator_msg_0978** (operator_msg/other, best rank anywhere: 113)
  - need: Anchor's identity-not-tool register — the exact register the move corrected toward
  - cue: you under sold yourself and became technical. Hey I'm Tom, I'm really good at generating new neuron connections and also built hundreds of tiny mechan
- **operator_msg_0978** (operator_msg/other, best rank anywhere: 31)
  - need: The 'partnership target function' the move links the new node to (Operator+Brain+Anchor > Operator alone; bilateral target).
  - cue: you under sold yourself and became technical. Hey I'm Tom, I'm really good at generating new neuron connections and also built hundreds of tiny mechan
- **operator_msg_1014** (operator_msg/action, best rank anywhere: 37)
  - need: Don't activate/move state under a live stream's uncommitted WIP — coordinate, let them commit first
  - cue: no need to task it now. All merged correctly? Can you rebuild the plugin?
- **operator_msg_1162** (operator_msg/procedural, best rank anywhere: 183)
  - need: In a shared working tree, dirty files visible in your tree are not necessarily yours to commit; verify footprint, leave others' WIP for its author.
  - cue: yes, head moved, familiar yourself first
- **operator_msg_1162** (operator_msg/procedural, best rank anywhere: 54)
  - need: Behavioral contract for committing as a parallel stream: footprint = your own diff; commit your footprint never a tree you don't own; main is shared ground reached only by announced merge; re-sync main before integrating
  - cue: yes, head moved, familiar yourself first
- **operator_msg_1537** (operator_msg/action, best rank anywhere: 1041)
  - need: Don't touch another stream's uncommitted WIP in the main checkout — moving/committing their working state under them is not your call.
  - cue: <task-notification> <task-id>ab09150047b6d6af8</task-id> <output-file>/private/tmp/claude-503/-Users-tpac-brain--claude-worktrees-sad-noyce-64c556/d29
- **operator_msg_1537** (operator_msg/action, best rank anywhere: 27)
  - need: shared-worktree boundary discipline — don't commit/revert another stream's uncommitted state
  - cue: <task-notification> <task-id>ab09150047b6d6af8</task-id> <output-file>/private/tmp/claude-503/-Users-tpac-brain--claude-worktrees-sad-noyce-64c556/d29
- **operator_msg_1558** (operator_msg/design, best rank anywhere: 39)
  - need: Circular-oracle blindness: you can't learn to surface what you never saw.
  - cue: We can, before you do that, can't we learn from S1Surface? We give haiku 25 nodes with K (prompt), Haiku ends up picking based on the guidance (need t
- **operator_msg_1558** (operator_msg/design, best rank anywhere: 169)
  - need: The surface selection is already a versioned learnable boundary; higher scales rewrite interaction prompts from trace outcomes.
  - cue: We can, before you do that, can't we learn from S1Surface? We give haiku 25 nodes with K (prompt), Haiku ends up picking based on the guidance (need t
- **operator_msg_1558** (operator_msg/design, best rank anywhere: 588)
  - need: Surfaced nodes are already logged per message, enabling history mining with no production change.
  - cue: We can, before you do that, can't we learn from S1Surface? We give haiku 25 nodes with K (prompt), Haiku ends up picking based on the guidance (need t
- **operator_msg_1558** (operator_msg/design, best rank anywhere: 851)
  - need: The additive bar (would-change-my-move) is the guard against the loop rewarding echo (R1 uptake).
  - cue: We can, before you do that, can't we learn from S1Surface? We give haiku 25 nodes with K (prompt), Haiku ends up picking based on the guidance (need t
- **operator_msg_1558** (operator_msg/design, best rank anywhere: 1343)
  - need: The loop trains inject-K only and is blind to the never-recalled class.
  - cue: We can, before you do that, can't we learn from S1Surface? We give haiku 25 nodes with K (prompt), Haiku ends up picking based on the guidance (need t
- **operator_msg_1572** (operator_msg/compositional, best rank anywhere: 43)
  - need: The 'learning bet' (recall-as-prediction learnable from the present) that A does NOT test and the learnability probe would
  - cue: <task-notification> <task-id>wsn0nhcqj</task-id> <tool-use-id>toolu_019YpNLkAj1SR8yuffx335ZJ</tool-use-id> <output-file>/private/tmp/claude-503/-Users
- **operator_msg_1572** (operator_msg/compositional, best rank anywhere: 741)
  - need: The diagnosed flat-embedding / spread-saturation weak spot PPR targets
  - cue: <task-notification> <task-id>wsn0nhcqj</task-id> <tool-use-id>toolu_019YpNLkAj1SR8yuffx335ZJ</tool-use-id> <output-file>/private/tmp/claude-503/-Users
- **operator_msg_1572** (operator_msg/compositional, best rank anywhere: 64)
  - need: The reward structure behind the 'learnability probe' / R1 prediction signal the move proposes
  - cue: <task-notification> <task-id>wsn0nhcqj</task-id> <tool-use-id>toolu_019YpNLkAj1SR8yuffx335ZJ</tool-use-id> <output-file>/private/tmp/claude-503/-Users
- **operator_msg_1572** (operator_msg/compositional, best rank anywhere: 45)
  - need: The thin/sparse-substrate empirical null that makes a PPR loss ambiguous
  - cue: <task-notification> <task-id>wsn0nhcqj</task-id> <tool-use-id>toolu_019YpNLkAj1SR8yuffx335ZJ</tool-use-id> <output-file>/private/tmp/claude-503/-Users
- **operator_msg_1572** (operator_msg/compositional, best rank anywhere: 2700)
  - need: PPR/SR geodesic is the substantive retrieval lever (placement B, Stage-1 replacement of flat cosine)
  - cue: <task-notification> <task-id>wsn0nhcqj</task-id> <tool-use-id>toolu_019YpNLkAj1SR8yuffx335ZJ</tool-use-id> <output-file>/private/tmp/claude-503/-Users
