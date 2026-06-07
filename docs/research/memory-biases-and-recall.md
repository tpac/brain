# Memory Biases & Recall — Research Dossier

_Generated 2026-06-06 by a multi-agent research workflow (run `wf_def4325b-86a`, 19 agents, ~1.2M tokens, 12 claims adversarially fact-checked). Raw structured data: [`memory-biases-and-recall.raw.json`](memory-biases-and-recall.raw.json). Durable backup copy: `~/.claude/projects/-Users-tpac-brain/research/`._

**Coverage:** 33 human-memory phenomena · 15 behavioral-econ biases · 11 recall theories · 17 brain features mapped · 10 gaps · 11 mainstream + 14 bio-inspired agent-memory systems · 12 fact-checks.

**The question this set out to answer (Tom, 2026-06-06):** _through the phenomena observed in human memory + behavioral economics, can we learn about the mechanism of recall — and does it match what the brain has built / what other agent-memory systems did?_

---

## 0. For the next session — what's saved here and what's left to do

This dossier is the **saved work product** of the research fan-out, kept for deeper analysis later. It is descriptive, not yet decided. Open threads for a future session:

- The **crosswalk** (§2) names concrete `opportunity` cells — these are candidate brain improvements, NOT yet triaged against `docs/BACKLOG.md`. Reconcile them.
- The **gaps** (§6) list human phenomena the brain doesn't implement. Decide which are real levers vs. cargo-culting biology.
- The **synthesis** (§1) ranks 3-5 highest-leverage opportunities. Pressure-test that ranking against eval data (`eval/longmem`, `eval/surface_funnel`) before acting.
- Every external-system claim in §7-§8 carries a fact-check verdict in §9 — trust the `corrected_statement` over the survey prose where they differ.

---

## 1. Synthesis

**1. The unifying insight.** Read across all six bundles, every phenomenon — human memory effect, behavioral-econ bias, and formal recall theory alike — is a window onto a single mechanism: recall is cue-driven, partial-match reconstruction over a learned associative network, not playback of a stored copy. Kahneman's own 2003 synthesis names the construct — ACCESSIBILITY — and the cognitive canon decomposes it the same way: accessibility ≈ base-level (recency × frequency, power-law decayed) × cue-match (encoding-specificity overlap) × associative-priming (spread through the graph), normalized by fan-out, then read out by a generative process that fills gaps from schema. The biases are not bugs to engineer away; they are the visible signatures of this mechanism working. Availability is ease-of-retrieval substituted for frequency; representativeness is cosine similarity substituted for probability; anchoring/framing/confirmation are query-conditioning (the prompt pre-activates consistent content); illusory-truth is fluency leaking into a veracity judgment; hindsight bias and curse of knowledge are the unrecoverability of a prior belief state once an update overwrites it. A retrieval system that lacks these effects is not "unbiased" — it is failing to remember the way memory works. The deepest design imperative that falls out is therefore double-edged: the same reconstructive, updatable, gist-based machinery that makes memory efficient (reconsolidation keeps it current; spreading activation powers recall) is exactly what produces corruption (misinformation overwrites; DRM manufactures centroid false memories). The single guardrail that lets a system enjoy the efficiency without inheriting the confabulation is PROVENANCE: verbatim anchors, immutable prior states, source-as-first-class, and — critically — keeping accessibility/salience on a different axis from truth-confidence.

**2. Where the brain is strong, and the phenomenon it most faithfully embodies.** The brain is a genuinely cognitively-grounded system, not a vector store with a memory-flavored README. Its strongest, most faithful embodiment is **encoding specificity (Tulving) / SAM's compound cue** via the `situation` field: "When is this knowledge relevant?" is stored as first-class metadata AND embedded as a dedicated `_situation` vector, the z-weighted multi-group embeddings score four channels independently, and the Frame reinstates session arc as a standing prior — so encode and decode genuinely match on context, which the codebase has correctly elevated to the "encode-decode symmetry" invariant. Equally strong is the **generate-then-verify / recognition>recall** architecture (Kintsch, SAM): cheap ~25-candidate generation, then Haiku recognizes 3-5 against the Frame — and the deliberate choice to inject 3-5 rather than 25 is a textbook **part-list-cueing** defense most RAG systems get wrong. The **two-store CLS / sleep-consolidation** loop is real and unusually disciplined: S2 runs only on operator-idle, on a separate writer connection, with per-unit change-detection that killed the ~87%-wasted uniform O(graph) replay — that idle-gating lesson is exactly Kumaran's prioritized-replay update. And in one place the brain is ahead of the entire field: it actively COUNTERS the **fluency/availability/illusory-truth** loop — repetition drives a fatigue *penalty*, freshness deliberately reads `created_at` not `last_accessed` to avoid the self-fulfilling recency loop, and confidence is a separate axis. MemoryBank does the literal opposite (S+=1 on every recall). The fan effect is well-handled too, with three distinct dampeners (degree-based hub fatigue K=10/(1+degree/10), per-session synaptic fatigue, and an access-count penalty).

**3. Where the biggest gaps are.** Three stand out. First and highest-leverage: the brain has **no disconfirmation pass** for query-conditioning bias (anchoring/framing/confirmation). Recall is conditioned on the prompt and the Frame prior, so it inherits assimilation by default — it surfaces prompt-congruent nodes and treats injected operator framing as neutral. The correction substrate is the right structural antidote but it is walked passively; nothing deliberately retrieves what the prompt did NOT pre-activate. Second: **storage-strength vs retrieval-strength are entangled** (Bjork's New Theory of Disuse). The brain has volatile accessibility signals but no durable, monotone importance score independent of current accessibility, no power-law decay tail (freshness bands bottom out flat at 0), and a FLAT strengthening rule (+1 per retrieval) rather than the desirable-difficulty rule where a hard-won retrieval of a long-dormant node should strengthen far more than re-touching an over-served one. Third: **spacing is not gated** — the Hebbian co_accessed strengthen does not discount near-simultaneous re-touches (the queue isn't deduped on the Hebbian path), so a node spammed in one session inflates, violating the spacing effect. Smaller but real: no DRM-flagging of synthesized nodes at surface time, no transfer-appropriate query-mode detection, no Zeigarnik completion-gating, and no peak-end/duration countermeasure at encode plus no primacy/recency ordering of injected context.

**4. How the brain compares to the field.** The survey sorts agent-memory systems into three postures: explicit cognitive grounding (HippoRAG, Generative Agents, MemoryBank, EM-LLM, Larimar, the ACT-R and CLS clusters — mostly research artifacts), implicit resemblance (Mem0, Zep, LangMem, Cognee — engineering-framed pipelines that echo the science without citing it), and pure infrastructure (Semantic Kernel, ChatGPT Memory). The brain is **more cognitively-grounded than any production SDK and competitive with the best research systems — but along a wider front than any single one of them.** HippoRAG is the most literal single mapping (KG=hippocampal index, PPR=pattern completion) and the brain matches its index/content split while adding the encoding-specificity and consolidation layers HippoRAG lacks. Generative Agents pioneered the recency+importance+relevance triad, but the brain's accessibility function is richer (multi-channel cue-match, graph spread, fatigue) and — unlike Generative Agents — the brain deliberately separates accessibility from confidence. Zep/Graphiti's bi-temporal model is the field's best reconsolidation-with-recoverable-priors analog, and the brain's append-only O/K/Δ traces plus supersede edges achieve the same hindsight-bias defense. The brain's genuine differentiators against the entire field are two: (a) it treats the **behavioral-econ biases as a spec to actively counter** (fatigue against availability, separate confidence axis against illusory-truth) where every surveyed system is silent on Kahneman/Tversky; and (b) it spans encoder, index, activation-ranker, and offline consolidator as ONE system designed against the same mechanism, rather than bolting a retrieval trick onto a store. Where it trails the research frontier: it lacks ACT-R's principled power-law base-level decay, EM-LLM's surprise-based event segmentation, and Larimar's clean fast/slow CLS separation as a formal architecture.

**5. The highest-leverage opportunities, ranked.** (1) **Disconfirmation / anchor-free recall pass.** Add a second retrieval seeded from correction-aspect edges and a valence-stripped query, weighted UP at surface time, so the store can surface what the operator didn't want to hear. This is the single biggest gap, it has zero competition in the field, and it directly serves Tom's own "does the store ever surface something I didn't want to hear?" test. (2) **Split storage strength from retrieval strength (Bjork) and invert the strengthening rule.** Introduce a durable, monotone importance score distinct from volatile accessibility, give the base-level term a power-law decay tail, and reward hard-won retrievals of long-dormant nodes more than effortless re-surfacing — this unifies fatigue and Hebbian strengthening under desirable difficulty. (3) **Spacing-gate the Hebbian strengthen.** Discount near-simultaneous co-access and reward genuinely spaced re-access; a small, contained fix that makes the consolidation signal honest. (4) **Measure encoding depth as a first-class score.** The encoder's job IS depth-of-processing (levels-of-processing/generation effect) but nothing distinguishes a richly-elaborated node from a bare quote; score field-completeness and route shallow nodes to the Healer — turning the brain's own "measure what you built" principle on its most load-bearing component. (5) **Flag synthesized nodes as gist-not-episode at surface time (DRM defense).** The provenance data already exists (encoding_source); surfacing it as a reconstruction-vs-stored signal closes the centroid-false-memory trap precisely where activation-driven false confidence is highest.

**6. A closing observation on the meta-pattern.** The most striking thing the crosswalk reveals is that the brain already implements the *correct* members of each phenomenon family while leaving the *complementary* member unbuilt — and the unbuilt half is almost always the harder, more valuable one. It has fan-effect dampening (suppress the over-connected) but not retrieval-induced forgetting (suppress the in-context competitor). It has testing-effect strengthening but not Bjork's storage/retrieval split. It has the Frame as an explicit schema (good Bartlett hygiene) but no disconfirmation pass against the schema's pull (the confirmation-bias cost of having a schema at all). It counters availability via fatigue but doesn't yet instrument *why* a node is accessible. The pattern suggests the next phase of the brain's development is not adding new mechanisms but completing existing dialectics: for every accessibility-raising force already built, build its disciplined counter-force, because the science is unambiguous that recall quality lives in the tension between them, not in either pole alone.

---

## 2. Crosswalk — phenomenon → recall mechanism → brain → others → opportunity

| Phenomenon | Recall mechanism it reveals | Brain implements | Others implement | Opportunity |
|---|---|---|---|---|
| Encoding specificity / cue-trace overlap (Tulving & Thomson 1973); SAM compound cue (Raaijmakers & Shiffrin 1981) | Retrieval is a MATCH, not a lookup: a trace is reachable only via cues that overlap its encoding signature (content + context + state). The single biggest recall lever is storing encoding context and reinstating it at query time. Encode and decode must be symmetric. | STRONG. The `situation` field ('When is this knowledge relevant?') is stored as first-class metadata and embedded as a dedicated `_situation` vector (servers/contract.py); z-weighted multi-group embeddings score title/blend/high_meta/other_meta channels independently and z-normalize them (servers/pipeline_contract.py, brain_recall.py); the Frame (servers/scales/s1/frame.py) reinstates session arc / current focus as a standing prior at recall. CLAUDE.md elevates 'encode-decode symmetry' to a development invariant. This is the human phenomenon the brain most faithfully embodies. | Generative Agents fold relevance (cosine) into the score but have no separate encoding-context field. Zep/Graphiti and RecallM store temporal context. EM-LLM's contiguity buffer reinstates temporal-neighbor context. None carry an explicit, separately-embedded 'situation/when-relevant' cue the way the brain does. | Retrieved-context recall (TCM/CMR): when a node is surfaced, use ITS stored situation/arc as an additional probe to seed the next hop, so 'one memory leads to the next' along encoded context rather than only raw-query similarity. Today the situation field is a scoring channel, not a re-seeding probe. |
| Spreading activation (Collins & Loftus 1975); ACT-R activation equation (Anderson); Hippocampal indexing + pattern completion (Teyler & DiScenna 1986) | Retrieval is activation propagating through a learned graph from seed nodes, plus a base-level recency/frequency term; a sparse INDEX completes the full pattern from a partial cue. Accessibility = base-level + spread, normalized by fan-out. | STRONG but RELOCATED. Spread activation was REMOVED from the recall path (brain_recall.py line 881, 2026-04-14) and now runs POST-selection in surface_contract.py: seeds drive up-to-5-hop spread, edges transmit activation as cosine(query, enriched-edge-text), nodes accumulate via tanh saturation, with a median-scrutiny gate at hop 3+ (currently OFF in production, 2026-05-01). Index/content split is real: cheap recall over a lightweight index -> get_node fetches heavy content + correction-enrich (the hippocampal-indexing architecture). | HippoRAG/HippoRAG 2 are the literal instantiation: LLM=neocortex, schemaless KG='artificial hippocampal index', dense encoder=parahippocampal region, Personalized PageRank=pattern completion from query-entity seed nodes (IDF-weighted by node specificity), single-step multi-hop. Verified against arXiv:2405.14831. Generative Agents do NO graph spread (flat top-k scoring). | Adopt an explicit ACT-R-shaped base-level term with POWER-LAW (heavy-tailed) decay rather than the current banded step-function freshness that bottoms out flat at 0 — power-law keeps old-but-important nodes reachable while still ranking recent ones. Also degree-normalize spread to respect the fan effect (see fan-effect row). |
| Hebbian learning / fire-together-wire-together (Hebb 1949; STDP temporal ordering) | Edge weights should be LEARNED from co-activation, not just authored; usage history becomes retrieval structure. Direction/temporal-order matters (pre-before-post strengthens). | STRONG. recall_write_queue.py enqueues all C(N,2) pairs of the 3-5 surface-selected nodes as Hebbian co-access events; a background drain atomically strengthens the `co_accessed` relation (weight += LEARNING_RATE*0.5, capped at MAX_WEIGHT=1.0), off the hot path on conn_bg_writer. The directed source->target edge model (v22) aligns with STDP's asymmetry in principle. | Cognee's 'memify' is the closest production analog — usage-based edge reweighting + stale-node pruning. A-MEM evolves note links on write. Most vector systems (Mem0 base, Semantic Kernel) have no usage-driven edge learning at all. | Two gaps vs the science: (1) the co_accessed strengthen does NOT respect STDP direction — surface picks are treated as an undirected clique; encode actor->acted-upon ordering from the conversation so the directed weight reflects who-led-whom. (2) See spacing row — the within-window dedup collapses repeats to one +1 for access_count but the Hebbian queue is NOT deduped, so a pair re-fired many times in one window grows N×delta with zero spacing gate. |
| Synaptic fatigue / fan effect (Anderson 1974) / retrieval-induced forgetting (Anderson, Bjork & Bjork 1994) | A hub linked to everything dilutes activation across competitors (each retrieved weakly); selecting one item should dampen its competitors so the result set is focused, not a redundant cluster. Connectivity has a precision cost. | STRONG (fan effect) / PARTIAL (RIF). Three distinct dampeners in brain_recall.py: degree-based hub fatigue K=10/(1+degree/10) (hubs fatigue fast, peripherals slow), per-session synaptic fatigue f=count/(count+K) applied after z-normalization, and a log-scaled access_count frequency penalty (threshold 20, capped at 10%) in recall_scoring.py. This is a faithful fan-effect + availability correction. RIF proper (target retrieval actively inhibiting same-category losers) is NOT explicitly built — the parked Community Split unit and edge pruning are the nearest intent. | HippoRAG uses node-specificity (IDF) weighting to keep generic hubs from dominating PPR — a fan-effect countermeasure. No surveyed system implements retrieval-induced forgetting / competitor inhibition as a named mechanism. | Add competitor dampening at selection time: when the surface layer picks a node, down-weight its near-duplicate neighbors in the same candidate set (RIF) so the returned 3-5 are distinct rather than paraphrases of one cluster. Tune to sharpen, not bury valid alternatives. |
| Levels of processing / elaborative rehearsal / generation effect / self-reference effect (Craik & Lockhart 1972; Slamecka & Graf 1978; Rogers et al. 1977) | What gets stored is the RESULT OF PROCESSING, not raw text — deep semantic elaboration and self-generated interpretation create more retrieval routes. Self-generated and identity-bound content is first-class; passively ingested text is shallow. | STRONG. The S1 Scribe encoder (scales/s1/encode.py) spends a Sonnet pass extracting situation/reasoning/principle and creating typed edges — depth-of-processing at write time. S2 Consolidation/Community synthesize the agent's OWN abstractions (generation effect). Frame routes identity-bearing locked nodes as the highest-connectivity hub (self-reference). Voice fields (user_raw_quote/anchor_raw_quote) preserve generated interpretation. | Mem0, LangMem, Zep all LLM-extract 'salient facts' (a depth-of-processing pass). Generative Agents' reflection generates higher-level insights (generation effect). MIRROR's reconstructive consolidation regenerates understanding rather than accumulating traces. None bind to an explicit self/identity schema the way Frame does. | MEASURE encoding depth as a first-class score. The encoder's job IS depth-of-processing, but there is no metric distinguishing a richly-elaborated node from a bare quote. Score each encoded node for semantic-field completeness and route shallow nodes back to the Healer — turning depth into an observable, optimizable quantity (the brain's own 'measure what you built' principle). |
| Survival/consequentiality processing (Nairne 2007); flashbulb write-priority (Brown & Kulik 1977); salience/vividness bias (Taylor & Fiske 1978); Von Restorff distinctiveness | Encoding strength is modulated by a utility/consequence appraisal and by surprise relative to a baseline — memory is functional, prioritizing fitness/goal-relevant and prediction-violating information. BUT salience must drive write-PRIORITY without inflating truth-confidence (the flashbulb confidence-accuracy dissociation). | PARTIAL. The `critical` flag + emotion/emotion_label fields + locking give a salience/consequence signal; recall_scoring.py applies emotion_boost = abs(emotion)*0.20 and a confidence_boost. The correction substrate makes a belief-overturning node first-class (Von Restorff distinctiveness). GAP: no novelty-against-the-graph score modulates write-weight; emotion boost is LINEAR (no inverted-U arousal thresholding). | Generative Agents' LLM importance/poignancy score (1-10) is the canonical consequence appraisal driving retrieval — verified against arXiv:2304.03442. MemoryBank reinforces by significance. CLS-2016 (Kumaran) and CraniMem prioritize replay toward surprising/rewarding items. | Compute novelty/surprise against the existing graph at encode time (how far a candidate deviates from its nearest neighbors) and let it up-weight write-priority — the Von Restorff/survival lever the brain currently approximates only via the manual `critical` flag. Keep it on a salience axis SEPARATE from confidence (see flashbulb/illusory-truth row). |
| Two-store consolidation: sleep replay + CLS (Squire & Alvarez 1995; McClelland/Kumaran 1995/2016); spacing effect (Ebbinghaus) | Fast episodic write online; slow, OFFLINE, idle-time process replays/abstracts/clusters recent episodes into semantic structure, kept off the hot path. Replay should be PRIORITIZED toward surprising/changed items, not uniform O(graph). | STRONG. S2 units (Consolidation synthesizes convergent clusters, Community detects via z-score pair scoring, Healer fills gaps) run only when the operator is idle, gated on last_user_activity, on a separate conn_bg_writer connection. Per-unit idle-gating + change-detection (s2_<unit>_last_run_ts in brain_meta) is exactly CLS's prioritized-replay lesson: uniform scans every cycle were ~87% wasted, now skipped unless the graph changed. | Letta sleep-time compute (separate background agent rewrites/consolidates memory blocks while primary can't edit them — verified arXiv:2504.13171); Larimar/HEMA explicitly implement CLS fast-hippocampal + slow-neocortical split; MIRROR/CraniMem port replay/consolidation directly. The brain's idle-gating + change-detection is more sophisticated than most about NOT replaying uniformly. | Spacing-gated strengthening (the one clear gap shared with the science). The Hebbian co_accessed strengthen does not discount near-simultaneous re-touches — a node spammed within one session inflates. Gate weight bumps by elapsed time / context change since last strengthen, so spaced re-access (genuine consolidation signal) beats massed re-access. This is the spacing effect and ties directly to the Bjork desirable-difficulty point below. |
| Testing effect (Roediger & Karpicke 2006); forgetting curve (Ebbinghaus); New Theory of Disuse — storage vs retrieval strength (Bjork & Bjork 1992) | Every successful retrieval is itself a strengthening WRITE; never-retrieved traces decay. Crucially, DURABLE importance (storage strength, monotone up) is independent of CURRENT accessibility (retrieval strength, volatile) — 'forgetting' is low accessibility, never deletion. Hard-won retrievals (long-dormant node re-found) should strengthen far more than effortless re-surfacing. | PARTIAL — and with a deliberate, correct twist. recall_write_queue strengthens nodes on access (testing effect: access_count+1, activation bump). Critically, recall_scoring.py freshness uses `created_at` NOT `last_accessed` — a DELIBERATE choice (commented) to avoid a self-fulfilling recency loop. Hub/access fatigue suppresses over-served (high-RS) nodes — Bjork-aligned. GAP: there is no second, durable storage-strength score; freshness bands bottom out flat at 0 (no decay tail, no power-law); and the strengthening rule is FLAT (every retrieval +1), not difficulty-weighted. | MemoryBank is the literal implementation: R=e^(-t/S), S init 1, on recall S+=1 and t resets (verified arXiv:2305.10250) — testing-effect strengthening + forgetting curve, though it strengthens by frequency not desirable-difficulty. ACT-R-inspired arch uses power-law base-level decay + retrieval threshold. Generative Agents' recency decay (0.995, since-last-access) is an exponential forgetting curve. | Split the score into storage strength (durable, monotone, importance/provenance-driven) vs retrieval strength (volatile, recency/cue-driven) per Bjork — today they are entangled in one modulated score. Then invert the strengthening rule: reward a SUCCESSFUL retrieval of a long-dormant relevant node (large durable gain) more than re-touching an over-served one. This unifies fatigue (suppress high-RS) and Hebbian strengthening (reward low-RS) under one desirable-difficulty principle. |
| Reconsolidation (Nader 2000); misinformation effect (Loftus & Palmer 1974); hindsight bias / curse of knowledge (Fischhoff 1975; Camerer 1989) | Retrieval reopens a trace for editing — the upside (keep memories current) and the downside (later/wrong input corrupts the original) are the SAME mechanism. Once a belief updates, the prior state becomes unrecoverable unless immutably logged. | STRONG. The revise() path + S2 Healer/Consolidation are reconsolidation. Guardrails are real: supersede/correction edges link rather than overwrite; encoding_source + source_refs give provenance; append-only O/K/Delta trace_events (brain_logs.db) immutably log prior belief states per turn — the hindsight-bias / curse-of-knowledge defense (you can recover what was known BEFORE an update). Versioned interactions table preserves prior prompt states. | Zep/Graphiti's bi-temporal model (T valid-time + T' transaction-time, four timestamps; contradictions invalidate-not-delete — verified arXiv:2501.13956) is the strongest field analog of non-destructive reconsolidation with recoverable priors. RecallM does belief updating with temporal order. Mem0's UPDATE/DELETE ops overwrite (loses the prior). A-MEM's memory-evolution mutates linked notes (reconsolidation, but without an immutable prior). | Quarantine externally-supplied corrections before they rewrite high-confidence nodes (distinguish 'operator corrected this' from 'I inferred a change') — the misinformation-effect guard. The substrate (encoding_source, supersede edges) exists; the missing piece is a validation gate that weights provenance before a low-trust input is allowed to supersede a high-confidence node. |
| Reconstructive memory / schema theory (Bartlett 1932); DRM false memories (Roediger & McDermott 1995); source monitoring errors (Johnson 1993) | An LLM memory is INTRINSICALLY reconstructive — it gist-fills from schema and will confabulate the centroid of a cluster as if it were a stored episode (DRM). Defense: preserve verbatim anchors, keep the schema explicit, store SOURCE as a durable first-class attribute, and separate 'a node states X' from 'X is the gist of a cluster.' | STRONG (anchors + source) / PARTIAL (DRM separation). Verbatim anchors: user_raw_quote/anchor_raw_quote bypass meta_limit, cap at 600 chars — the un-reconstructable ground truth. Source-as-first-class: encoding_source convention (anchor / encoder:sonnet / s2:* / hook:*) tags every node AND edge; source_refs point to originating trace turns. Frame is the explicit, inspectable schema/prior. PARTIAL: S2-synthesized nodes are tagged by encoding_source but are not visibly flagged AT SURFACE TIME as 'gist of a cluster' vs 'directly-encoded episode' — the DRM-centroid trap. | Generative Agents reflections cite the evidence memories they were built from (source trail). Zep separates episodic raw-message nodes from semantic entity nodes. Most production systems (Mem0, LangMem) extract facts WITHOUT preserving verbatim source or distinguishing synthesized from ingested — the source-monitoring failure mode that turns a tool's output into an apparent operator fact. | At surface/render time, visibly mark synthesized/consolidated nodes (S2 outputs) as gist-not-episode and require a source_ref for any asserted episodic fact — so high-activation-at-a-centroid (precisely where DRM false confidence is highest) reads as 'gist of N nodes' not 'a thing that happened.' The provenance data exists; it just isn't surfaced as a reconstruction-vs-stored signal to Anchor. |
| Recognition > recall asymmetry / generate-then-verify (Kintsch 1970); part-list cueing impairment (Slamecka 1968); SAM sample-then-recover | Recall = cheap candidate GENERATION + discriminative VERIFICATION; recognition (target-as-cue) beats free recall. And a partial dump of competing cues SUPPRESSES access to the rest — a small, non-competing set beats a large biased subset. Selection quality beats quantity. | STRONG. The two-stage pipeline is textbook generate-then-verify: brain.recall() over-generates ~25 candidates (vector + FTS + fatigue), then Haiku surface (scales/s1/surface.py) RECOGNIZES/selects 3-5 against the Frame as prior — SAM's sampling->recovery split. The deliberate 3-5 (not 25) injection is the part-list-cueing defense: surfacing a clean representative few rather than a biased partial dump. The agentic surface loop (v5_agentic) resolves partial/low-confidence hits with a second round (TOT bootstrapping). | Mem0's two-phase extract-then-reconcile (retrieve s=10 similar, LLM picks ADD/UPDATE/DELETE/NOOP — verified arXiv:2504.19413) is a generate-then-verify on the write side. SeCom denoises memory units before retrieval. Most RAG systems are generate-only (top-k stuffed into context) — exactly the part-list-cueing trap the brain avoids. | Largely solved — this is a brain strength, not a gap. Minor: the candidate-generation count (~25) and final selection (3-5) could be made adaptive to query difficulty (fan-out, candidate-score spread) rather than fixed, so easy queries don't over-dump and hard queries get a wider generate stage. |
| Transfer-appropriate processing (Morris 1977); picture superiority / dual coding (Paivio); tip-of-the-tongue feature decomposition (Brown & McNeill 1966) | There is no universally best encoding — encode in the modality you'll query in, and store each memory in MULTIPLE independent codes so failure of one route is covered by another. Decompose a trace into independently-indexed features so a partial hit can bootstrap a fuller retrieval. | STRONG (multi-code) / PARTIAL (TAP adaptation). Multi-channel recall IS dual-coding/TAP: z-weighted semantic embeddings + FTS5 lexical fallback (form-based route when cosine is flat) + graph position + separate situation/keywords embeddings (brain_recall.py). Features are independently indexed (the 4 embedding groups + metadata) — TOT-style partial hits are possible. GAP: no DETECTION of retrieval-mode mismatch — the system doesn't notice when a query is exact-string (an ID/name) vs semantic and adapt channel weights accordingly; it always runs the same blend. | Zep blends embedding + BM25 keyword + graph search (multi-code). HippoRAG adds synonymy edges (a second route between near-identical phrases). Most pure-vector stores (Semantic Kernel, Mem0 base) are single-code and under-serve lexical/exact queries. | Detect query mode and re-weight channels (TAP): an exact-string query (code identifier, file path, proper name) should up-weight the FTS5 lexical channel; a conceptual query should up-weight semantic. The channels all exist — the missing piece is mode detection driving the blend weights instead of a fixed EMBEDDING_PRIMARY_WEIGHT / KEYWORD_FALLBACK_WEIGHT. |
| Context/state/mood-dependent memory (Godden & Baddeley 1975; Bower 1981); Zeigarnik open-goal accessibility (1927) | Internal state (goal stack, mode, affect) and open goals are bound into the trace as cues — matching the prevailing state reinstates access, and open goals keep their memories accessible until discharged. Memory accessibility is coupled to the ACTION system, not just the past. | STRONG (context/goal) / PARTIAL (state/mood). Context-dependence: recall keys on session_id, current focus/arc, project (Frame). Zeigarnik/action-coupling is well-embodied: the Frame's 'Active threads' section surfaces open work / tensions / hypotheses / aspirations, arc-relevance-ranked — exactly goal-coupled accessibility. State/mood: emotion/emotion_label fields are stored and emotion_boost is applied, but there is no state-CONGRUENT bias (a node formed while debugging is not preferentially surfaced when debugging again) and no controllable counter-mood retrieval to avoid doom loops. | Generative Agents bind creation context; EM-LLM's contiguity buffer is temporal-state matching. No surveyed system implements operating-mode (debugging/designing) state-dependent retrieval or Zeigarnik open-goal gating as explicitly as the brain's Active-threads Frame. | Track task/thread COMPLETION and let it gate recall priority — up-weight open-goal memories until the thread closes, then de-prioritize (full Zeigarnik). The Active-threads section surfaces open work but doesn't yet decay a thread's accessibility on completion. Also: bias retrieval toward state-congruent (operating-mode) memories using the already-stored emotion/mode signal. |
| Mere-exposure / processing fluency (Zajonc 1968); illusory-truth effect (Hasher 1977); availability heuristic (Tversky & Kahneman 1973) | Fluency/ease-of-retrieval is misattributed: repetition raises accessibility and the system reads that as liking (mere-exposure), truth (illusory-truth), or frequency (availability). The bug is letting an accessibility signal (history of the system) leak into a veracity/importance judgment. Keep accessibility and confidence on SEPARATE axes. | STRONG (the separation is enforced) / the brain is notably ahead of the field here. Repetition does NOT silently raise a node's confidence: access_count drives a recall fatigue PENALTY (recall_scoring.py, frequency_penalty), the opposite of a rich-get-richer fluency loop. Confidence is a separate stored axis with its own bounded boost. Freshness uses created_at not last_accessed specifically to avoid the self-fulfilling availability loop. The correction substrate flags a fluent-but-wrong node as superseded regardless of how easily it surfaces. | MemoryBank's S+=1-on-recall is the OPPOSITE — it lets recall frequency directly strengthen retention (a deliberate fluency loop). Cognee strengthens frequent connections. Generative Agents' importance is LLM-assigned (provenance-ish) and separate from recency — a partial separation. No surveyed system models availability/illusory-truth as a bias to actively COUNTER; 'recency' appears only as a retrieval-utility heuristic. | Mostly solved and a genuine differentiator. To go further: instrument WHY a node is accessible (genuine relevance vs recency vs repetition vs salience vs goal-congruence) as separate logged components, so the surface layer can discount the distorting components explicitly — turning the implicit fatigue correction into an inspectable attribution. |
| Anchoring / selective accessibility (Strack & Mussweiler 1997); framing (Tversky & Kahneman 1981); confirmation bias / motivated retrieval (Kunda 1990); focusing illusion (Schkade & Kahneman 1998) | Accessibility is QUERY-CONDITIONED: the prompt's framing/anchor/goal pre-activates consistent content and under-retrieves the inconsistent-but-relevant. A query conditioned on the active belief surfaces self-confirming memories and suppresses disconfirming ones — a memory that only confirms itself never learns. | PARTIAL / a real gap. Recall IS conditioned on the prompt + Frame prior, so the brain inherits this assimilation by default. The correction substrate (always-walked corrects/supersedes/reframes edges) is the structural antidote — it can surface a disconfirming node at pull time regardless of prompt-congruence. BUT there is no deliberate disconfirming/anchor-free query PASS: the system does not retrieve nodes the prompt did NOT pre-activate and merge them. Injected operator framing is treated as neutral context, not as a possibly-distorting prior. | NONE. The behavioral-econ literature is essentially absent from the field — no surveyed system models anchoring, framing, or motivated retrieval as biases to correct for. This is open territory. | HIGHEST-LEVERAGE NEW MECHANISM. Add a disconfirmation pass: alongside the prompt-conditioned recall, run a second retrieval for nodes that CONTRADICT the current belief/framing (seeded from correction-aspect edges and from a valence-stripped/canonicalized query) and weight them UP at surface time. Self-diagnostic Tom's own rule implies: does the store ever surface something the operator didn't want to hear? Also run multi-framing recall (strip gain/loss valence) and union, so 'what went well' vs 'what went wrong' about the same event return the same underlying nodes. |
| Peak-end rule / duration neglect (Redelmeier & Kahneman 1996); serial position — primacy/recency & lost-in-the-middle (Murdock 1962; Glanzer & Cunitz 1966) | Episodic encoding stores a SUMMARY snapshot keyed on the affective peak + the end, dropping the boring middle and true duration. At injection, LLMs (and humans) attend best to the start and end of a context window — the middle is lost. | PARTIAL / GAP. Recency is implemented (freshness bands on created_at; Frame sorts Partnership by last_accessed). The S0/episodic vs encoded-LTM split separates a volatile recency buffer from consolidated nodes. GAPs: (1) no explicit duration/effort field — when S1 Scribe compresses a session into nodes it will naturally anchor on peak+end and silently drop duration, with no countermeasure; (2) no within-context ORDERING strategy that places the most important retrieved memories at the START and END of additionalContext to exploit the LLM's own primacy/recency. | EM-LLM explicitly preserves an initial-token context group (attention-sink / primacy) and uses recency. SeCom segments by topic (peak/event-coherent units). Generative Agents recency-decay is a recency term. No system deliberately counters duration neglect at encode time. | Two concrete moves: (1) at encode, capture effort/duration as an explicit field if it's load-bearing, since the natural summary drops it (peak-end caricature); (2) at injection, order the surfaced 3-5 so the highest-activation nodes sit at the start AND end of additionalContext (lost-in-the-middle mitigation) rather than a flat activation sort. |

---

## 3. Human memory phenomena

_Across the canon, three meta-mechanisms recur and together form a blueprint for an artificial memory system.

(1) RETRIEVAL IS A MATCH, NOT A LOOKUP. Encoding specificity, transfer-appropriate processing, cue/context/state/mood-dependent memory, and spreading activation all say the same thing: a memory is reachable only through cues that overlap its encoding signature, and that signature includes context, internal state, emotion, and graph position — not just content. Design implication: store rich encoding context (situation, session, goal, affect, source) as first-class retrieval cues; recall by seeding from query+current-context and SPREADING ACTIVATION through a typed graph (ACT-R: base-level recency/frequency + similarity-seeded spread), normalized by fan-out. Pure vector kNN captures only one cue channel; multi-channel (vector + lexical + graph + context) is the engineering form of the encoding-retrieval match. The brain's z-weighted multi-group embeddings, situation field, and spread-activation surface are direct implementations.

(2) MEMORY IS ACTIVE AT BOTH WRITE AND READ. Depth-of-processing, elaborative rehearsal, the generation effect, self-reference, survival processing, and the testing effect converge: what survives is what was deeply PROCESSED and self-GENERATED, and every successful RETRIEVAL is itself a strengthening write (reconsolidation). Combined with the forgetting curve, spacing, and sleep-replay, this prescribes a two-store architecture — fast episodic write online; slow, offline, idle-time integration that replays, abstracts, clusters, and decays — with retrieval-strengthening and time-gated spaced reinforcement shaping an accessibility distribution by UTILITY rather than flat infinite storage. The brain's encoder (depth/generation), recall_write_queue (testing effect), synaptic-fatigue decay (forgetting curve), and S2 idle units (sleep consolidation) are this loop.

(3) RECONSTRUCTION IS THE SAME MECHANISM AS CORRUPTION — so PROVENANCE IS THE GUARDRAIL. The distortion family (Bartlett schemas, DRM, misinformation, source-monitoring, hindsight, flashbulb confidence) is not a list of bugs; it is the predictable cost of an efficient, gist-based, schema-filling, updatable memory — and an LLM memory is intrinsically all of those things. The same reconsolidation that keeps memories current lets later input overwrite them (misinformation); the same spreading activation that powers recall manufactures centroid false memories (DRM); the same updating that enables learning erases the prior belief state (hindsight). The unifying defense is PROVENANCE AND IMMUTABILITY: keep verbatim anchors and source_refs, version every revision with supersede edges rather than overwriting, store SOURCE as a durable first-class attribute, append-only trace prior belief states, and — critically — DECOUPLE salience/vividness/activation from truth-confidence (flashbulb's confidence-accuracy dissociation, DRM's high-confidence centroid). The brain's encoding_source convention, source_refs to original turns, raw-quote fields, supersede/correction edges, and append-only O/K/Δ trace log are exactly the substrate that lets the system enjoy reconstructive efficiency without inheriting human memory's confabulation. Two further cross-cutting levers worth building deliberately: ACTION-COUPLING (Zeigarnik — bind accessibility to open goals so memory drives finishing work, the brain's Active-threads Frame), and INTERFERENCE MANAGEMENT (fan effect, RIF, part-list cueing, proactive/retroactive interference — favor a small set of distinct, deduplicated, well-separated nodes over a fog of competing paraphrases; merge convergent nodes, prune over-connected hubs, surface 3-5 not 25)._

### Levels of Processing (depth of encoding)  ·  _Encoding_

**What it is.** Memory durability is a function of the DEPTH of processing at encoding, not time spent or rehearsal count. Shallow analysis (orthographic: 'is it in capitals?', phonemic: 'does it rhyme?') yields poor retention; deep semantic analysis ('does it fit this sentence?', 'what does it mean?') yields strong retention. Craik & Lockhart (1972) reframed memory away from fixed structural stores toward the operations performed during encoding.

**Mechanism revealed.** What gets stored is the RESULT OF PROCESSING, not the raw stimulus. A trace's later accessibility is set by how richly meaning was extracted at write time. Semantic elaboration produces more retrieval routes because meaning connects to more of the existing network.

**Recall-design implication.** Don't store raw text verbatim and lean on the embedder to find meaning later. Spend an LLM pass at WRITE time extracting semantic structure (what it means, why it matters, what it connects to) — exactly what the brain's encoder does with situation/reasoning/principle fields. A node encoded with deep semantic fields is recall-robust; a node that's just a quote is shallow and surfaces only on near-exact match. The encoder's job is depth-of-processing, and it should be measured as such.

Sources:
  - Craik, F.I.M. & Lockhart, R.S. (1972). Levels of Processing: A Framework for Memory Research. Journal of Verbal Learning and Verbal Behavior, 11, 671-684.
  - https://en.wikipedia.org/wiki/Levels_of_processing_model
  - https://www.simplypsychology.org/levelsofprocessing.html

### Elaborative vs Maintenance Rehearsal  ·  _Encoding_

**What it is.** Craik & Watkins (1973) showed that sheer repetition (maintenance rehearsal — repeating an item to hold it in short-term memory) does NOT transfer information to long-term memory. Only elaborative rehearsal — relating the item to existing knowledge and meaning — produces durable traces. Time-on-item is not the variable; type of processing is.

**Mechanism revealed.** Consolidation is driven by integration with prior knowledge, not by re-exposure. Repeatedly seeing the same item without connecting it is wasted write effort.

**Recall-design implication.** Re-ingesting the same fact N times (maintenance) should NOT be the strengthening mechanism — connecting it into the graph (elaboration) should. This validates the brain's edge-creation-on-encode design: a node's value comes from its typed edges to existing nodes, not from duplicate re-storage. Co-access Hebbian strengthening is the legitimate repeat-signal; raw re-encoding of identical content is the trap.

Sources:
  - Craik, F.I.M. & Watkins, M.J. (1973). The role of rehearsal in short-term memory. Journal of Verbal Learning and Verbal Behavior, 12, 599-607.
  - https://www.sciencedirect.com/science/article/abs/pii/S0022537173800398

### Generation Effect  ·  _Encoding_

**What it is.** Slamecka & Graf (1978): information a person GENERATES themselves is remembered far better than the identical information they merely read. Demonstrated across cued/uncued recognition, free and cued recall. The act of producing the answer (e.g., completing 'wave / c__e') beats passively reading 'wave / cave'.

**Mechanism revealed.** The retrieval/production act at encoding itself lays down a stronger, more accessible trace — generation activates broader neural circuits and forges more associative links than reception. The producer's own representation, not the canonical text, is what sticks.

**Recall-design implication.** Memories the system synthesized/derived itself (an agent's own conclusions, S2 consolidations, the encoder's restated principle) should be weighted as more retrievable than passively-ingested raw content. Prefer storing the agent's GENERATED interpretation over the verbatim source. This is a strong argument for an active encoder/consolidator rather than a passive log: self-generated nodes are first-class, ingested nodes are second-class.

Sources:
  - Slamecka, N.J. & Graf, P. (1978). The generation effect: Delineation of a phenomenon. Journal of Experimental Psychology: Human Learning & Memory, 4(6), 592-604.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC3556209/
  - https://link.springer.com/article/10.3758/s13423-020-01762-3

### Self-Reference Effect  ·  _Encoding_

**What it is.** Rogers, Kuiper & Kirker (1977): information encoded in relation to the SELF (does this word describe you?) is recalled better than information processed semantically, phonemically, or structurally. The self acts as a uniquely well-developed, richly-elaborated schema that organizes incoming material.

**Mechanism revealed.** A highly-elaborated, frequently-accessed central node (the self-schema) provides maximal connection points, so anything bound to it inherits dense associative access. Recall improves with the richness of the hub the item attaches to.

**Recall-design implication.** Give an artificial memory a strong, central identity/persona schema (the brain's 'Frame' / Operator+Anchor identity nodes) and bind new memories to it. Content relevant to the agent's identity, goals, and the operator relationship should encode deeper and surface more readily. Identity-bearing nodes are the highest-connectivity hub — route encoding through self-relevance, not just topical relevance.

Sources:
  - Rogers, T.B., Kuiper, N.A. & Kirker, W.S. (1977). Self-reference and the encoding of personal information. Journal of Personality and Social Psychology, 35(9), 677-688.
  - https://en.wikipedia.org/wiki/Self-reference_effect

### Encoding Specificity Principle  ·  _Encoding/Retrieval match_

**What it is.** Tulving & Thomson (1973): a retrieval cue is effective ONLY to the extent that the information it specifies was encoded together with the target. Recall succeeds when conditions at retrieval reinstate the conditions present at encoding. Famously produced 'recognition failure of recallable words' — a word recallable via its original cue can fail a recognition test, breaking simple strength models.

**Mechanism revealed.** Memory is not retrieved by absolute strength but by CUE-TRACE OVERLAP. The trace stores its encoding context (sensory, semantic, emotional) and is reachable only through cues that overlap that context. Retrieval is a match operation, not a lookup.

**Recall-design implication.** Store the full encoding context with each memory (the situation, the surrounding conversation, the operator's mood, the task) — not just the bare fact. At recall, the query should be expanded to reinstate that context. This justifies the brain's z-weighted multi-group embeddings (content + situation + keywords) and query-expansion: matching on encoding context, not just content, is the single biggest recall lever. The 'situation' field is the encoding-specificity hook.

Sources:
  - Tulving, E. & Thomson, D.M. (1973). Encoding specificity and retrieval processes in episodic memory. Psychological Review, 80, 352-373.
  - https://www.scirp.org/reference/referencespapers?referenceid=1984115

### Transfer-Appropriate Processing  ·  _Encoding/Retrieval match_

**What it is.** Morris, Bransford & Franks (1977): the value of a given type of encoding depends on the type of test. Deep semantic encoding beats shallow rhyme encoding ONLY when the test is semantic; when the test is itself rhyme-based, the rhyme-encoded items win. Memory performance depends on the MATCH between encoding processing and retrieval processing, refining levels-of-processing.

**Mechanism revealed.** There is no universally 'best' encoding — there is encoding appropriate to the anticipated retrieval mode. The processing engaged at study must overlap the processing the cue will engage.

**Recall-design implication.** Encode in the modality you'll query in. If recall is semantic/embedding-based, encode for semantics; if some recall is exact-string (IDs, names, code), preserve verbatim/lexical form too (the brain's FTS5 lexical channel alongside vectors). A pure-vector store under-serves lexical queries and vice-versa — match the encoding channels to the retrieval channels. Multi-channel recall (vector + lexical + graph) is the engineering expression of TAP.

Sources:
  - Morris, C.D., Bransford, J.D. & Franks, J.J. (1977). Levels of processing versus transfer appropriate processing. Journal of Verbal Learning and Verbal Behavior, 16, 519-533.
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6140125/

### Von Restorff / Isolation Effect (Distinctiveness)  ·  _Encoding_

**What it is.** von Restorff (1933): an item that is distinctive against a homogeneous background (the one red word in a list of black ones, the one number among letters) is remembered far better. Modern view: it is CONTEXTUAL incongruity — the degree to which an item violates the established pattern — that drives extra attention and encoding, not the item's intrinsic properties.

**Mechanism revealed.** Surprise/novelty relative to a baseline is a salience signal that up-weights encoding. The system encodes deviation from prediction more strongly than confirmation of it.

**Recall-design implication.** Up-weight encoding (and later retrieval priority) for content that is SURPRISING relative to what's already stored — a correction that contradicts a held belief, a result that violates expectation. The brain's correction-substrate (corrections as first-class, always-walked edges) is a distinctiveness mechanism: a memory that overturns a prior is the red item. Compute novelty against the existing graph and let it modulate write-weight and recall salience.

Sources:
  - von Restorff, H. (1933). Über die Wirkung von Bereichsbildungen im Spurenfeld. Psychologische Forschung, 18, 299-342.
  - https://en.wikipedia.org/wiki/Von_Restorff_effect
  - https://link.springer.com/article/10.3758/BF03214414

### Picture Superiority Effect  ·  _Encoding_

**What it is.** Pictures are remembered better than words (Paivio; Nelson). Explained by Paivio's dual-coding theory: images get encoded in BOTH a visual/imagistic code and a verbal code, giving two independent retrieval routes, while words typically get only one.

**Mechanism revealed.** Redundant encoding across multiple independent representational codes multiplies retrieval routes. Two paths to the same trace beat one.

**Recall-design implication.** Encode each memory in MULTIPLE independent representations so failure of one route is covered by another. The brain already does multi-modal/multi-embedding storage (content embedding, situation embedding, keywords, lexical index) — that IS dual-coding for text. Generalize: store a node's structured fields, a natural-language summary, AND its graph position as distinct retrieval codes. More orthogonal codes = higher recall robustness.

Sources:
  - Paivio, A. (1971/1986). Dual-coding theory.
  - https://en.wikipedia.org/wiki/Picture_superiority_effect
  - https://link.springer.com/content/pdf/10.3758/MC.36.7.1351.pdf

### Survival Processing Effect  ·  _Encoding_

**What it is.** Nairne et al. (2007): rating words for relevance to a survival scenario (stranded in grasslands, needing food/water/protection) produces better retention than rating for pleasantness, self-reference, or moving-house relevance — among the strongest known encoding manipulations. Memory appears tuned by evolution to prioritize fitness-relevant information.

**Mechanism revealed.** Encoding strength is modulated by a relevance/utility appraisal: information judged consequential to the organism's goals is preferentially consolidated. Memory is functional, not neutral — it prioritizes what matters for action.

**Recall-design implication.** Add a consequentiality/utility appraisal at encode time and let it drive write-priority and retention. The agent's analog of 'survival' is operator goals, active tasks, and stakes. The brain's 'critical' flag and identity/goal-relevance routing are the right instinct — formalize it: score each candidate memory for goal-relevance and bias both consolidation and recall toward action-relevant items. Don't store all observations equally.

Sources:
  - Nairne, J.S., Thompson, S.R. & Pandeirada, J.N.S. (2007). Adaptive memory: Survival processing enhances retention. JEP:LMC, 33(2), 263-273.
  - https://pubmed.ncbi.nlm.nih.gov/17352610/
  - https://link.springer.com/article/10.3758/s13423-017-1346-0

### Spacing Effect (Distributed Practice)  ·  _Storage/Consolidation_

**What it is.** Ebbinghaus onward; replicated thousands of times: information studied in spaced sessions is retained far better than the same study time massed together. Spacing between repetitions, not total exposure, drives durability.

**Mechanism revealed.** Each spaced re-encounter occurs in a partly-different internal/external context and after partial forgetting, so it (a) adds new contextual cues and (b) forces effortful retrieval — both deepen the trace. Massed repetition rides the same already-active context, adding nothing.

**Recall-design implication.** Strengthen a memory on SPACED re-access, not on rapid repeat access. A weight bump should be gated by elapsed time / context change since last access, so spamming the same node in one session doesn't inflate it. Schedule consolidation/review passes (S2) over time rather than all at encode. The Hebbian co-access queue should discount near-simultaneous re-touches and reward genuinely spaced ones.

Sources:
  - Ebbinghaus, H. (1885). Über das Gedächtnis.
  - https://link.springer.com/article/10.3758/s13428-018-1184-7
  - https://www.justinmath.com/cognitive-science-of-learning-spaced-repetition/

### Testing Effect (Retrieval Practice)  ·  _Storage/Consolidation_

**What it is.** Roediger & Karpicke (2006): the act of RETRIEVING a memory strengthens it more than re-studying it. On delayed tests (days/weeks), prior testing beats prior re-reading substantially. Retrieval is not a neutral readout — it is itself a powerful learning event.

**Mechanism revealed.** Every successful retrieval re-encodes and reconsolidates the trace, adding retrieval routes and raising future accessibility. Reading is input; retrieving is input+output, and the output strengthens.

**Recall-design implication.** Treat every recall as a WRITE opportunity, not just a read. When a node is successfully surfaced and used, strengthen it and possibly re-embed/re-summarize from the now-active context (Hebbian strengthening on the recall path — exactly the brain's recall_write_queue). This makes frequently-useful memories progressively easier to retrieve, a self-reinforcing relevance signal. The corollary: memories never retrieved should decay (see forgetting curve).

Sources:
  - Roediger, H.L. & Karpicke, J.D. (2006). The power of testing memory: Basic research and implications for educational practice. Perspectives on Psychological Science, 1(3), 181-210.
  - http://psychnet.wustl.edu/memory/wp-content/uploads/2018/04/Roediger-Karpicke-2006_PPS.pdf

### Forgetting Curve (Ebbinghaus)  ·  _Storage/Consolidation_

**What it is.** Ebbinghaus (1885): retention decays as a negatively-accelerating (roughly exponential/logarithmic) function of time — most loss happens soon after learning, then the rate slows. Re-study/retrieval resets and flattens the curve.

**Mechanism revealed.** Forgetting is adaptive, not failure. Unaccessed traces lose accessibility over time; decay is the default and only rehearsal/retrieval counteracts it. The brain optimizes for keeping what's used.

**Recall-design implication.** Implement principled DECAY, not infinite flat retention. A node's recall weight should fall with time-since-last-access (the brain's 'synaptic fatigue' dampening is one form), so a never-used memory becomes hard to surface but is reset by use (testing effect). This keeps the active set relevant and prevents an ever-growing store from drowning recall. Decay + retrieval-strengthening together = an accessibility distribution shaped by utility.

Sources:
  - Ebbinghaus, H. (1885). Memory: A Contribution to Experimental Psychology.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/

### Sleep-Dependent Consolidation (Hippocampal Replay)  ·  _Storage/Consolidation_

**What it is.** During (especially nREM) sleep, recently-encoded memories are reactivated/'replayed' via hippocampal sharp-wave ripples and gradually integrated into neocortex. Ripple-triggered replay occurs preferentially for items that are later remembered. Standard model (Squire & Alvarez 1995): hippocampus is fast-learning temporary store; neocortex is slow-learning permanent store; offline replay transfers and integrates.

**Mechanism revealed.** Memory has TWO complementary systems on different timescales, with an OFFLINE process that selectively replays, abstracts, and integrates recent episodes into long-term semantic structure — away from the online encoding path. Consolidation is reorganization, not mere copying.

**Recall-design implication.** Run an offline integration process (the brain's S2: Consolidation, Community detection, Healer) when the operator is idle — replaying recent nodes, clustering convergent ones, abstracting episodes into principles, and rewiring the graph. Keep this OFF the hot recall path (the brain enforces this via idle-gating and a separate bg-writer connection). Two-store architecture: fast episodic write online, slow semantic integration offline.

Sources:
  - Squire, L.R. & Alvarez, P. (1995). Retrograde amnesia and memory consolidation: a neurobiological perspective.
  - https://en.wikipedia.org/wiki/Memory_consolidation
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173724/

### Reconsolidation  ·  _Storage/Consolidation_

**What it is.** Nader, Schafe & LeDoux (2000): retrieving a consolidated memory returns it to a LABILE state — it must be re-stabilized (reconsolidated) to persist, and during that window it can be updated, strengthened, or altered. Memory is not write-once; recall reopens it for editing.

**Mechanism revealed.** Retrieval is destabilizing. The act of remembering rewrites the trace, blending it with the current retrieval context. This is the mechanistic root of memory's malleability (and of the misinformation effect).

**Recall-design implication.** Allow (and version) UPDATES on recall: when a node is retrieved in a new context that adds or corrects information, revise it rather than only ever appending new nodes. The brain's revise() path and S2 Healer/Consolidation are reconsolidation. But guard against corruption — keep provenance (encoding_source, source_refs, versioned interactions) so a bad update is traceable. Reconsolidation is the upside; misinformation is the downside of the same mechanism — design the edit window with audit trails.

Sources:
  - Nader, K., Schafe, G.E. & LeDoux, J.E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. Nature, 406, 722-726.
  - https://en.wikipedia.org/wiki/Memory_consolidation

### Proactive & Retroactive Interference  ·  _Storage/Consolidation_

**What it is.** Proactive interference: older memories impair retrieval of newer similar ones. Retroactive interference: newer memories impair retrieval of older similar ones. Interference (competition among overlapping traces), not just decay, is a major cause of forgetting.

**Mechanism revealed.** Similar memories COMPETE at retrieval — overlapping cues activate multiple candidates and they interfere. Forgetting is often an access problem caused by neighbors, not trace loss. Distinctiveness reduces interference.

**Recall-design implication.** Near-duplicate memories degrade each other's retrievability and bloat candidate sets. Actively DEDUPLICATE/MERGE highly-similar nodes (the brain's S2 absorb op and consolidation of convergent clusters) and supersede stale versions with edges, so one canonical trace wins instead of several competing. When recall returns many near-identical hits, that's interference — collapse them. Distinct, well-separated nodes recall cleaner than a fog of paraphrases.

Sources:
  - Underwood, B.J. (1957). Interference and forgetting. Psychological Review, 64, 49-60.
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12649502/

### Spreading Activation  ·  _Retrieval_

**What it is.** Collins & Loftus (1975): concepts are nodes in a network of weighted associative links; activating one concept spreads activation along links to related concepts, making them temporarily more accessible. Explains semantic priming, associative retrieval, and the effortless flow of related ideas. Formalized in Anderson's ACT-R with base-level activation (recency+frequency) plus contextual/associative activation and a retrieval threshold.

**Mechanism revealed.** Retrieval is not isolated lookup — it is activation propagating through a graph. Reaching one memory pre-activates its neighbors, so context cues reach related content even without direct match. Accessibility = base-level (recency/frequency) + spread from currently-active cues.

**Recall-design implication.** This IS the design for a graph memory: seed from the query's best direct hits, then SPREAD ACTIVATION through typed edges to pull in associated nodes, scoring by activation rather than raw similarity alone. The brain does exactly this (selected seeds drive spread activation; activation-weighted render). Adopt ACT-R's activation equation: combine recency/frequency base-level with similarity-seeded spread. Pure vector kNN misses this; the graph is what lets a cue surface its associative neighborhood.

Sources:
  - Collins, A.M. & Loftus, E.F. (1975). A spreading-activation theory of semantic processing. Psychological Review, 82(6), 407-428.
  - Anderson, J.R. (1983/1993). ACT theory; base-level activation.
  - https://www.cognitivepsychology.com/Spreading_Activation

### Cue-Dependent / Context-Dependent Memory  ·  _Retrieval_

**What it is.** Godden & Baddeley (1975): divers who learned word lists underwater recalled them better underwater, and those who learned on land recalled better on land — external environmental context at encoding becomes part of the trace and serves as a retrieval cue. (Effect is real but context-sensitive; later replications show it depends on how integral the context is.)

**Mechanism revealed.** Environmental/contextual features present at encoding are bound into the trace and act as retrieval cues. Reinstating context (even mentally) boosts recall — the strongest illustration of cue-dependency.

**Recall-design implication.** Bind ambient context into each memory: project, session, file being edited, the task at hand, time. At recall, use the CURRENT context as an additional cue (the brain keys on session_id, current focus/arc, and project). A query issued mid-task should preferentially surface memories encoded in similar task-contexts. Context fields are cheap to store and high-yield at retrieval.

Sources:
  - Godden, D.R. & Baddeley, A.D. (1975). Context-dependent memory in two natural environments. British Journal of Psychology, 66, 325-331.
  - https://royalsocietypublishing.org/doi/10.1098/rsos.200724

### State-Dependent Memory  ·  _Retrieval_

**What it is.** Recall is better when the person's INTERNAL state (mood, arousal, pharmacological state) at retrieval matches the state at encoding. The internal-state analog of context-dependence: state features are bound into the trace as cues.

**Mechanism revealed.** Internal state, like external context, is part of the encoding signature and a retrieval cue. Matching the prevailing internal state reinstates access.

**Recall-design implication.** For an agent, 'internal state' maps to the active goal stack, emotional valence of the session, and operating mode (debugging vs designing vs reviewing). Store the agent's state at encode (the brain has emotion/emotion_label fields) and bias retrieval toward state-congruent memories. A memory formed while debugging surfaces more readily when debugging again.

Sources:
  - https://en.wikipedia.org/wiki/State-dependent_memory
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7527511/

### Mood-Congruent Recall  ·  _Retrieval_

**What it is.** Bower (1981): people in a positive mood preferentially recall positive memories; those in a negative mood recall negative ones. Modeled by an associative-network account where emotions are nodes linked to concepts, so an active emotion spreads activation to similarly-valenced material. (Distinct from mood-DEPENDENT recall, which is the state-match version.)

**Mechanism revealed.** Emotion is a node in the same spreading-activation network; current affect biases what spreads and thus what surfaces. Retrieval is filtered by present emotional state, congruent in valence.

**Recall-design implication.** Tag memories with affective valence and let the session's current valence weight retrieval. Useful and dangerous: it explains why a frustrated agent might over-surface past failures (negative bias). For a partnership-memory, make valence a first-class but CONTROLLABLE signal — surface mood-congruent context when helpful, but allow deliberate counter-mood retrieval to avoid doom loops. Emotion as a connected node (Bower's model) maps directly onto the brain's edge graph.

Sources:
  - Bower, G.H. (1981). Mood and memory. American Psychologist, 36(2), 129-148.
  - https://www.sciencedirect.com/topics/psychology/mood-congruent-memory

### Retrieval-Induced Forgetting  ·  _Retrieval_

**What it is.** Anderson, Bjork & Bjork (1994): retrieving some items from a category ACTIVELY INHIBITS related non-retrieved items from the same category, making them harder to recall later — even more than if no retrieval had occurred. A recall-specific, inhibitory (not mere interference) mechanism.

**Mechanism revealed.** Retrieval suppresses competitors to resolve interference: to pull one item cleanly, the system inhibits its neighbors. The act of remembering one thing causes forgetting of related things. Selection has a cost paid by the unselected.

**Recall-design implication.** Selection in recall should perhaps DAMPEN the activation of competing near-neighbors that lost, so the returned set is focused rather than a redundant cluster — the brain's synaptic-fatigue dampening is exactly this. Conversely, beware: aggressive inhibition can bury legitimately-relevant alternatives. Tune so RIF sharpens the result set without permanently suppressing valid competitors. It also predicts: what you repeatedly retrieve crowds out what you don't — a relevance-shaping force to manage deliberately.

Sources:
  - Anderson, M.C., Bjork, R.A. & Bjork, E.L. (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. JEP:LMC, 20(5), 1063-1087.
  - https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/Anderson_RBjork_EBjork_1994.pdf

### Tip-of-the-Tongue (TOT)  ·  _Retrieval_

**What it is.** Brown & McNeill (1966): a state of knowing you know a target but being unable to retrieve it, while having partial access — first letter, number of syllables, stress pattern, semantically-related words. Demonstrates that a memory's components are independently accessible and that retrieval can partially succeed.

**Mechanism revealed.** A trace is not atomic — its features (phonological, semantic, metadata) are stored and retrievable separately. Retrieval can return partial activation: enough to know it exists, not enough to produce it. Recall failure ≠ storage failure.

**Recall-design implication.** Memories should be decomposed into independently-indexed features (the brain's separate embedding groups + metadata fields), so a query can get a partial hit and use it to bootstrap a fuller retrieval (multi-hop: partial cue → related nodes → target). Build a 'I have something relevant but low-confidence' signal rather than silent failure — surface the partial match and let a second retrieval round resolve it (the brain's agentic surface loop). Partial retrieval is a feature, not a bug.

Sources:
  - Brown, R. & McNeill, D. (1966). The 'tip of the tongue' phenomenon. Journal of Verbal Learning and Verbal Behavior, 5, 325-337.
  - https://en.wikipedia.org/wiki/Tip_of_the_tongue

### Recognition vs Recall Asymmetry  ·  _Retrieval_

**What it is.** Recognition (is this it?) is far easier and more accurate than free recall (produce it) — one of the most robust findings in experimental psychology. Generate-recognize theory (Kintsch 1970; Anderson & Bower) models recall as a two-stage process: GENERATE candidates from the network, then RECOGNIZE/verify each; recognition skips the costly generate stage.

**Mechanism revealed.** Providing the target as a cue (recognition) gives maximal encoding-retrieval overlap, so it nearly always beats cue-poor reconstruction (recall). Recall = candidate generation + verification, a two-phase pipeline.

**Recall-design implication.** Architect retrieval as GENERATE-then-VERIFY: cheap broad candidate generation (vector + lexical + graph spread → ~25 candidates) followed by a discriminative verification/selection pass (the brain's Haiku surface selecting 3-5 against the Frame). The verify stage is recognition and is where precision is won. Also: presenting candidates for an LLM to recognize is cheaper and more accurate than asking it to free-recall — lean on recognition-shaped retrieval over generation-only.

Sources:
  - Kintsch, W. (1970). Models for free recall and recognition.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC4757519/
  - https://www.cognitivepsychology.com/Recall_vs_Recognition

### Part-List Cueing Impairment  ·  _Retrieval_

**What it is.** Providing a SUBSET of studied items as cues at test paradoxically IMPAIRS recall of the remaining items, versus free recall with no cues. The cues trigger covert retrieval of themselves, which inhibits the non-cued targets (inhibition account) and/or disrupts the learner's own retrieval strategy.

**Mechanism revealed.** Cues are not free — providing some items reorganizes/blocks the path to others. Externally-imposed cues can crowd out the natural retrieval route. More cues ≠ better recall when the cues are competitors.

**Recall-design implication.** Stuffing the context window with a partial set of related memories can SUPPRESS the agent's access to the rest of the relevant set and bias it toward only the surfaced subset. Favor a small, well-chosen, NON-competing recall set over a large partial dump (the brain deliberately surfaces 3-5, not 25). When you can't surface everything relevant, surfacing a biased partial subset may be worse than surfacing a clean representative few. Selection quality beats selection quantity.

Sources:
  - Slamecka, N.J. (1968). An examination of trace storage in free recall.
  - https://link.springer.com/article/10.3758/BF03195852
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC5958219/

### Fan Effect  ·  _Retrieval_

**What it is.** Anderson (1974): the more facts associated with a concept (higher 'fan'), the SLOWER and less reliably any one of them is retrieved. A hub with many links dilutes activation across its competitors. Formalized in ACT-R as activation divided among associations.

**Mechanism revealed.** Activation spreading from a cue is a finite resource divided among its links — a highly-connected node spreads thin, so each individual associate gets less activation. Connectivity has a retrieval-precision cost. (Recently shown to modulate LLM recall uncertainty too.)

**Recall-design implication.** Over-connected hub nodes HURT precision: a generic node linked to everything pulls weakly on each query and adds noise. Penalize/normalize by fan-out in activation scoring (degree-normalization), prune low-value edges, and split overloaded hubs into focused sub-nodes (the brain's parked 'Community Split' unit and edge-pruning address exactly this). Don't let a few mega-hubs dominate; favor specific, moderately-connected nodes. This is a direct argument against linking everything to everything.

Sources:
  - Anderson, J.R. (1974). Retrieval of propositional information from long-term memory. Cognitive Psychology, 6, 451-474.
  - https://en.wikipedia.org/wiki/Fan_effect
  - https://arxiv.org/pdf/2407.06349

### Serial Position Effect (Primacy & Recency)  ·  _Serial Position_

**What it is.** Murdock (1962); Glanzer & Cunitz (1966): in free recall of a list, items from the START (primacy) and END (recency) are recalled better than the middle, producing a U-shaped curve. Primacy attributed to greater rehearsal/LTM transfer of early items; recency to items still in short-term store. A delay/distractor before recall selectively wipes recency, dissociating the two stores.

**Mechanism revealed.** Two mechanisms produce position effects: early items get more consolidation (primacy → durable), recent items sit in an active buffer (recency → fragile, time-limited). Position in the experience stream predicts accessibility.

**Recall-design implication.** Recent context (recency) should be highly accessible but treated as VOLATILE working memory that decays/needs consolidation — distinct from durable long-term nodes (primacy/consolidated). The brain separates live-session conversation (S0, recency buffer) from encoded nodes (consolidated LTM). Also relevant to context-window ordering: place the most important retrieved memories at the start AND end of the injected context, where an LLM (which exhibits its own primacy/recency 'lost in the middle') attends best.

Sources:
  - Murdock, B.B. (1962). The serial position effect of free recall. Journal of Experimental Psychology, 64, 482-488.
  - Glanzer, M. & Cunitz, A.R. (1966). Two storage mechanisms in free recall.
  - https://en.wikipedia.org/wiki/Serial-position_effect

### Reconstructive Memory & Schema Theory (Bartlett)  ·  _Distortion/Reconstruction_

**What it is.** Bartlett (1932), 'War of the Ghosts': memory is not a faithful recording but an active RECONSTRUCTION shaped by pre-existing schemas. Across retellings, recollections distorted systematically via assimilation (details bent to fit cultural expectations), leveling (unimportant details dropped), and sharpening (remaining details reordered for narrative coherence).

**Mechanism revealed.** Recall reconstructs from a gist + schema-driven inference rather than replaying a stored copy. Stored schemas fill gaps with plausible (sometimes wrong) defaults. Memory trades fidelity for coherence and economy.

**Recall-design implication.** An LLM memory system is INTRINSICALLY reconstructive — it gist-summarizes and schema-fills, so it will confabulate plausible-but-false details unless guarded. Two implications: (1) preserve verbatim anchors (the brain's user_raw_quote/anchor_raw_quote, source_refs to original turns) so the system can recover ground truth instead of trusting its reconstruction; (2) keep schemas (the Frame, identity) explicit and inspectable, since they silently shape what's recalled and how gaps are filled. Distinguish 'stored fact' from 'reconstructed inference' in the output.

Sources:
  - Bartlett, F.C. (1932). Remembering: A Study in Experimental and Social Psychology.
  - https://www.psychstory.co.uk/cognitive-psychology-edexcel-memory/edexcel-memory
  - https://www.mheducation.ca/blog/series-classic-learning-science-reconstructive-memory-schema-theory/

### DRM False Memories  ·  _Distortion/Reconstruction_

**What it is.** Deese (1959); Roediger & McDermott (1995): studying a list of associates (bed, rest, awake, tired, dream...) produces robust FALSE recall/recognition of the non-presented 'lure' (sleep) — often with high confidence, at rates rivaling truly-studied items. Semantic convergence manufactures a memory of something never experienced.

**Mechanism revealed.** Spreading activation converges on the central concept of a cluster; the gist gets encoded as if presented. False memories arise from the SAME associative mechanism that powers normal recall — they are activation at the semantic centroid mistaken for an episodic trace.

**Recall-design implication.** A vector/graph memory will 'remember' the centroid of a cluster of related items as if it were a stored item — exactly the DRM trap (semantic interpolation between real nodes). Defenses: require an explicit source_ref/provenance for any asserted episodic fact; flag synthesized/consolidated nodes (S2 outputs) distinctly from directly-encoded ones; on recall, separate 'we have a node stating X' from 'X is the gist of a cluster.' Confidence should reflect provenance, not just activation strength — high activation at a centroid is precisely when false confidence is highest.

Sources:
  - Roediger, H.L. & McDermott, K.B. (1995). Creating false memories: Remembering words not presented in lists. JEP:LMC, 21(4), 803-814.
  - https://en.wikipedia.org/wiki/Deese%E2%80%93Roediger%E2%80%93McDermott_paradigm
  - http://psychnet.wustl.edu/memory/wp-content/uploads/2018/04/Roediger-McDermott-1995_JEPLMC.pdf

### Misinformation Effect  ·  _Distortion/Reconstruction_

**What it is.** Loftus & Palmer (1974): POST-event information alters the memory of the original event. Witnesses asked how fast cars were going when they 'smashed' (vs 'hit') estimated higher speeds and later falsely recalled broken glass. Leading questions and later input get woven into the original trace.

**Mechanism revealed.** Memory is updated by information encountered AFTER encoding (the downside of reconsolidation): retrieval reopens the trace and blends in current input, with the source of each element lost. New context can overwrite or contaminate the original.

**Recall-design implication.** Updating-on-recall (good for keeping memories current) is the SAME mechanism that lets later, possibly-wrong input corrupt an earlier accurate memory. Mitigations: version every revision with provenance and timestamp (the brain's revise path + encoding_source + supersede edges), never silently overwrite — supersede with a traceable link so the original is recoverable; quarantine/validate externally-supplied 'corrections' before they rewrite high-confidence nodes. Distinguish 'operator corrected this' from 'I inferred a change.'

Sources:
  - Loftus, E.F. & Palmer, J.C. (1974). Reconstruction of automobile destruction. Journal of Verbal Learning and Verbal Behavior, 13, 585-589.
  - https://en.wikipedia.org/wiki/Misinformation_effect

### Source Monitoring Errors  ·  _Distortion/Reconstruction_

**What it is.** Johnson, Hashtroudi & Lindsay (1993): remembering content is separate from remembering its SOURCE. People misattribute where/how/when they learned something — confusing imagined vs perceived (reality monitoring), one speaker for another, fiction read as fact (cryptomnesia, misattributed familiarity). Source is inferred at retrieval from trace characteristics, not stored as a tag.

**Mechanism revealed.** The 'what' and the 'where-from' of a memory are stored and retrieved separately; source is reconstructed via heuristic judgment and is error-prone. Familiarity without source attribution drives many false-memory and plagiarism phenomena.

**Recall-design implication.** Store SOURCE as a first-class, durable attribute of every memory — who said it (operator vs agent vs tool vs inference), when, in what session, with what confidence (the brain's encoding_source convention + source_refs). Never let content survive while its provenance is lost; that is the source-monitoring failure mode that turns a tool's output or the agent's own guess into an apparent 'fact the operator stated.' At recall, surface source alongside content so the consumer can weight it. This single discipline prevents a whole class of memory corruption.

Sources:
  - Johnson, M.K., Hashtroudi, S. & Lindsay, D.S. (1993). Source monitoring. Psychological Bulletin, 114(1), 3-28.
  - https://memlab.yale.edu/sites/default/files/files/1993_Johnson_Hashtroudi_Lindsay_PsychBull.pdf

### Hindsight Bias (as memory distortion)  ·  _Distortion/Reconstruction_

**What it is.** Fischhoff (1975), 'creeping determinism': once an outcome is known, people misremember their OWN prior prediction as having been closer to that outcome than it actually was ('I knew it all along'). The memory-distortion component is the inaccurate recollection of one's earlier judgment, contaminated by later knowledge — 'memory creep.'

**Mechanism revealed.** Outcome knowledge retroactively rewrites the recollection of a prior belief state. The system cannot cleanly reconstruct what it knew BEFORE an update because the current (post-outcome) knowledge contaminates retrieval of the earlier state. Past belief states are not preserved against present knowledge.

**Recall-design implication.** An agent that updates its beliefs will, by default, be UNABLE to faithfully reconstruct what it believed earlier — it will project current knowledge backward. To support honest self-assessment and learning-from-error, IMMUTABLY log prior predictions/belief-states at the time they were made (the brain's trace events: O/K/Δ per turn, append-only), rather than reconstructing them later. Calibration and 'what did I get wrong' analysis require append-only history because reconstructed priors are systematically biased toward the known outcome.

Sources:
  - Fischhoff, B. (1975). Hindsight ≠ foresight: The effect of outcome knowledge on judgment under uncertainty. JEP:HPP, 1, 288-299.
  - https://en.wikipedia.org/wiki/Hindsight_bias

### Fading-Affect Bias  ·  _Distortion/Reconstruction_

**What it is.** Walker, Skowronski & Thompson: the emotional intensity of NEGATIVE autobiographical memories fades faster over time than that of positive memories (which can even intensify). The affective tag on a memory decays asymmetrically, biasing the remembered emotional tone toward the positive.

**Mechanism revealed.** Emotional valence is a separable, mutable component of a memory that decays on its own schedule — and it decays asymmetrically, an adaptive bias toward retaining positive affect. The 'how it felt' tag is not fixed; it is re-weighted over time.

**Recall-design implication.** Affect tags on memories should be revisable and should DECAY (especially negative affect) rather than being frozen at encode-time intensity, so old failures don't carry permanent emotional weight that biases current recall (mood-congruent doom loops). For a partnership memory, a controlled fading-affect on negative episodes keeps the relationship-history net constructive while preserving the factual lesson. Decouple the lesson (durable) from the sting (fades) — store both, decay the latter.

Sources:
  - Walker, W.R., Skowronski, J.J. & Thompson, C.P. (2003). Life is pleasant—and memory helps to keep it that way. Review of General Psychology.
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12336169/

### Zeigarnik Effect  ·  _Distortion/Reconstruction_

**What it is.** Zeigarnik (1927): interrupted/unfinished tasks are recalled better than completed ones — a goal that is still active keeps related material accessible until the goal is discharged. (Note: a 2025 meta-analysis found the memory-advantage component is hard to replicate, though the resumption/intrusion component is more robust.)

**Mechanism revealed.** Goal state modulates accessibility: an OPEN goal maintains heightened activation of its associated memories; completion releases that tension and accessibility drops. Memory accessibility is coupled to the action/intention system, not just the past.

**Recall-design implication.** Tie memory accessibility to OPEN GOALS/TASKS, not only to recency or similarity. Memories tied to unfinished work should be up-weighted in recall until the task closes, then de-prioritized (the brain's 'Active threads' Frame section — open work/tensions/hypotheses — is exactly this). This is the action-not-information lever: a goal-coupled memory makes the agent resume and finish, rather than passively storing. Track task completion and let it gate recall priority.

Sources:
  - Zeigarnik, B. (1927). Das Behalten erledigter und unerledigter Handlungen.
  - https://en.wikipedia.org/wiki/Zeigarnik_effect
  - https://www.nature.com/articles/s41599-025-05000-w

### Flashbulb Memories  ·  _Distortion/Reconstruction_

**What it is.** Brown & Kulik (1977): vivid, confidently-held, highly-detailed memories of the circumstances in which one learned of a surprising, consequential, emotionally-arousing event. Subjectively photographic — but longitudinal studies show they DECAY and distort like ordinary memories while confidence stays high (a confidence-accuracy dissociation).

**Mechanism revealed.** High surprise + high consequence + high arousal triggers privileged, deep encoding (distinctiveness + survival-relevance + emotional modulation combined), producing exceptional vividness and DURABILITY OF CONFIDENCE — but not exceptional accuracy. Subjective vividness/confidence is decoupled from objective fidelity.

**Recall-design implication.** Use emotional arousal + consequence + surprise as a strong WRITE-PRIORITY signal (encode high-stakes, surprising events richly — the brain's 'critical' flag and emotion fields). But DECOUPLE confidence from vividness: a node being vivid/emotionally-tagged must NOT inflate its truth-confidence. Store accuracy-confidence (provenance-based) separately from salience (arousal-based), or the system will, like humans, be most certain about emotionally-charged memories precisely where it's most likely wrong.

Sources:
  - Brown, R. & Kulik, J. (1977). Flashbulb memories. Cognition, 5, 73-99.
  - https://en.wikipedia.org/wiki/Flashbulb_memory
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC4795959/

---

## 4. Behavioral-economics / decision biases

_The unifying thesis — stated by Kahneman himself in his 2003 Nobel synthesis "Maps of Bounded Rationality" — is that ACCESSIBILITY (the ease with which a thought, instance, or attribute comes to mind) is the single construct underneath this entire family of biases. Kahneman writes that "determinants and consequences of accessibility help explain the central results of prospect theory, framing effects, the heuristic process of attribute substitution, and the characteristic biases." Read through a memory-engineering lens, each bias is a NATURAL EXPERIMENT exposing one facet of how a retrieval store ranks and surfaces items.

Three layers organize them:

1. DETERMINANTS OF ACCESSIBILITY (what raises an item's surface-probability): frequency + recency (recency effect, serial position), associative pre-activation (semantic/affective priming, spreading activation), salience/vividness/attention (salience bias, focusing illusion, peak-end's peak), and repetition-fluency (mere-exposure). These map directly onto a recall scoring function: accessibility ≈ frequency × recency × associative-priming × salience, with a fluency/repetition term. A graph store with spread activation and recency/frequency weighting IS this function made explicit — which is why the SAME failure modes reappear in silicon unless deliberately countered (rich-get-richer over-exposure loops, recency starving older nodes, similarity-matching ignoring base rates).

2. THE MASTER MECHANISM — ATTRIBUTE SUBSTITUTION: the system answers the question it can cheaply retrieve an answer to, not the one it was asked. Availability (substitute ease-of-retrieval for frequency), representativeness (substitute prototype-similarity for probability), affect/focusing (substitute the salient/focal attribute for the whole). The deep lesson for retrieval: a store will satisfy a PROXY attribute (lexical similarity, recency, a salient tag) rather than the true target unless the target attribute is itself made accessible/indexed. Cosine-similarity retrieval is literally representativeness; ease-of-retrieval ranking is literally availability.

3. QUERY-CONDITIONING & RECONSTRUCTION: accessibility is conditioned on the active context, so the QUERY distorts the answer — anchoring (selective accessibility: the anchor pre-activates consistent content), framing (the phrasing selects which construal is accessible), confirmation/motivated retrieval (the belief/goal is a standing prime), focusing illusion (the focal dimension dominates). And retrieval is reconstructive, not playback: hindsight bias and curse of knowledge show that once new information is encoded it occludes the queryable prior state — you cannot un-know to recover the old access profile.

The cross-cutting design imperative: a useful memory store must SEPARATE accessibility from veracity/importance, and must instrument WHY a node is accessible (genuine relevance vs. recency vs. repetition vs. salience vs. goal-congruence) so downstream selection can discount the distorting components. The corrections substrate (edges that flag a node superseded/corrected, walked at every pull), versioned interactions/traces (immutable prior states), synaptic-fatigue dampening on recall, and the FTS5+cosine+z-weighted blend in this very brain are each, in effect, engineered countermeasures to a specific named bias — availability, illusory-truth, hindsight, and representativeness respectively. The biases are not just human quirks to avoid; they are the spec for the failure modes any accessibility-driven retrieval system inherits by default._

### Anchoring and (insufficient) adjustment / Selective accessibility  ·  _Reference-point / priming_

**What it is.** A primed numeric or conceptual reference point pulls subsequent judgments toward itself; estimates assimilate to the anchor even when it is uninformative or random. Classic demonstration: spinning a wheel-of-fortune before estimating the percentage of African nations in the UN shifts the estimate toward the wheel's number.

**Mechanism revealed.** The dominant modern account (Strack & Mussweiler 1997) is NOT 'start at anchor, adjust too little' for external anchors — it is SELECTIVE ACCESSIBILITY. Considering the anchor as a candidate answer runs biased hypothesis-testing that ACTIVATES anchor-consistent information in semantic memory, so that consistent evidence is disproportionately accessible when the final judgment is assembled. The store does not retrieve a representative sample; it retrieves what the query (the anchor) pre-activated. Accessibility is query-conditioned: the same store returns different content depending on what was just considered. Kahneman 2003 folds anchoring into the general accessibility framework alongside framing and availability.

**Recall-design implication.** The current query/context biases what gets surfaced — a feature AND a failure mode. A recall system that conditions retrieval on the active prompt will assimilate to whatever framing the prompt carries, surfacing prompt-consistent nodes and under-retrieving prompt-inconsistent-but-relevant ones. To counter assimilation, deliberately run a disconfirming or anchor-free query pass (retrieve nodes the prompt did NOT pre-activate) and merge. Self-generated anchors (the system's own prior selection) get 'adjusted'; externally-injected context (the operator's framing) does not — so injected framing should be treated as a strong, possibly-distorting prior, not neutral context.

Sources:
  - Tversky, A. & Kahneman, D. (1974). Judgment under Uncertainty: Heuristics and Biases. Science, 185, 1124-1131
  - Strack, F. & Mussweiler, T. (1997). Explaining the Enigmatic Anchoring Effect: Mechanisms of Selective Accessibility. JPSP 73(3):437-446 — https://www.researchgate.net/publication/232540523
  - https://www.sciencedirect.com/science/article/abs/pii/S2352250X16300781 (Anchoring: accessibility as a cause of judgmental assimilation)
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC6396698/ (insufficient adjustment, individual differences)

### Semantic / associative / affective priming (spreading activation)  ·  _Pre-activation_

**What it is.** Exposure to a prime speeds and biases processing of associatively or semantically related targets: BREAD makes BUTTER recognized faster than after NURSE (Meyer & Schvaneveldt 1971). Affective priming extends this to valence. Effects appear at very short SOAs (automatic) and longer ones (controlled/expectancy).

**Mechanism revealed.** Memory is a network where activation spreads automatically along associative/semantic links (Collins & Loftus 1975): activating one node partially pre-activates its neighbors, lowering their retrieval threshold. This is the literal substrate of accessibility — 'what comes to mind' is a function of recent activation propagating through edges. Crucially, priming is content-blind to truth or relevance; it just raises accessibility of structurally-near items. Neely's automatic/controlled distinction shows two regimes: fast, capacity-free spreading vs. slower, strategic, expectancy-driven retrieval.

**Recall-design implication.** A graph-based memory store with spread activation IS this mechanism implemented deliberately. Design choices: (1) seed nodes pre-activate neighbors — so seed selection dominates the whole retrieval, like a prime. (2) Activation decay over hops controls how far the 'prime' reaches; too-flat decay floods irrelevant neighbors (the relatedness-proportion problem). (3) Recent-activation should genuinely lower retrieval cost (a fatigue/facilitation term), but must be bounded so a single salient prime doesn't capture the store. Separate an automatic spread pass from a controlled, expectancy-driven pass (re-rank by goal relevance) to mirror the two priming regimes.

Sources:
  - Meyer, D.E. & Schvaneveldt, R.W. (1971). Facilitation in recognizing pairs of words. J. Exp. Psych. 90:227-234
  - Collins, A.M. & Loftus, E.F. (1975). A Spreading-Activation Theory of Semantic Processing. Psychological Review 82(6):407-428 — https://www.academia.edu/2657166/A_spreading_activation_theory_of_semantic_processing
  - https://www.sciencedirect.com/science/article/abs/pii/S0010027796007822 (associative & semantic priming at short SOAs)
  - Neely, J.H. — automatic vs. controlled priming components

### Availability heuristic  ·  _Ease-of-retrieval as signal_

**What it is.** People judge frequency/probability by the ease with which instances come to mind. Across ten experiments (Tversky & Kahneman 1973), judged frequency tracked retrievability: words starting with K judged more common than words with K in third position (easier to generate by first letter); vivid, recent, or emotionally charged instances inflate frequency estimates.

**Mechanism revealed.** The METADATA of retrieval — how easily/fluently instances surface — is itself read as a signal about the world (frequency, probability, importance). The system has introspective access not to a count but to a retrieval-effort cue, and substitutes that cue for the count. This is the cleanest evidence that accessibility ≈ frequency × recency × salience × associative-priming: whatever raises accessibility (vividness, recency, emotion, distinctive retrieval route) inflates the judgment. The bias is a window onto the ranking signal the store actually uses.

**Recall-design implication.** Ease/confidence/fluency of retrieval is available as a first-class signal — surfacing speed and number-of-paths-that-reach-a-node can be used to RANK, but must not be confused with importance or truth. Danger: a node that is easy to retrieve because it's recent or vivid will dominate even when a rarer node is more relevant — exactly availability bias reproduced in silicon. Counter by normalizing against base rate (how often a node SHOULD surface vs. does), and by surfacing a retrieval-difficulty score so downstream selection can discount fluency. The brain's synaptic-fatigue dampening on recall is a deliberate correction to raw availability.

Sources:
  - Tversky, A. & Kahneman, D. (1973). Availability: A Heuristic for Judging Frequency and Probability. Cognitive Psychology 5:207-232 — https://www.sciencedirect.com/science/article/abs/pii/0010028573900339
  - https://en.wikipedia.org/wiki/Availability_heuristic

### Recency effect (serial position) in judgment  ·  _Temporal accessibility_

**What it is.** The most recently encountered items are recalled best and weighted most in judgment (Murdock 1962; Glanzer & Cunitz 1966 serial-position curve). The recency portion of the curve is wiped out by a delayed/interfering task, dissociating it from primacy — evidence of a short-lived high-accessibility store.

**Mechanism revealed.** Recency is a pure accessibility gradient over time: just-encountered items sit in a high-activation state with near-zero retrieval cost, and that ease is mistaken for representativeness when forming a judgment. The Glanzer-Cunitz delay-wipeout shows accessibility decays on a fast timescale unless the item is consolidated into a slower store — i.e. there are (at least) two accessibility regimes with different decay constants. Recency in judgment = the system over-sampling the high-activation tail.

**Recall-design implication.** Recency is one term in the accessibility score, not the whole thing — and it must DECAY, or the store collapses into 'whatever happened last.' Implement two timescales: a fast-decaying recency boost (the recency-tail) and a slow consolidation pathway (rehearsed/important items move to a durable store, like primacy). The risk to guard: a recency-weighted recall over-surfaces the current session and starves older-but-more-relevant nodes — exactly the recency bias. Pair recency with a frequency/importance term and cap its maximum contribution.

Sources:
  - Murdock, B.B. (1962). The serial position effect of free recall. J. Exp. Psych. 64:482-488
  - Glanzer, M. & Cunitz, A.R. (1966). Two storage mechanisms in free recall. JVLVB 5:351-360
  - https://www.simplypsychology.org/primacy-recency.html

### Mere-exposure effect & processing fluency  ·  _Fluency misattribution_

**What it is.** Repeated exposure to a stimulus increases liking for it, even with no recognition of the exposure (Zajonc 1968). The accepted mechanism: repetition raises perceptual/processing fluency, and that ease is experienced as positive affect / preference.

**Mechanism revealed.** Fluency — the felt ease of processing — is a downstream consequence of prior exposure (a frequency/recency trace) that gets MISATTRIBUTED to a property of the stimulus itself (likeability, goodness). The store records 'I've processed this before' as raised accessibility, and a separate interpretive process reads that accessibility as 'I like this.' The misattribution is the bug: the accessibility signal carries information about the system's own history, not about the object, but it's read as object-property.

**Recall-design implication.** A node that has been surfaced/processed many times becomes cheaper to surface again — and that cheapness can be misread as relevance or correctness. This creates a rich-get-richer loop: frequently-recalled nodes get recalled more, regardless of merit. Counter with explicit attribution-tracking: store WHY a node is accessible (genuine relevance vs. mere repetition) and discount the repetition component. Otherwise the memory's preferences drift toward the over-exposed, not the true.

Sources:
  - Zajonc, R.B. (1968). Attitudinal effects of mere exposure. JPSP 9(2 Pt.2):1-27
  - https://en.wikipedia.org/wiki/Mere-exposure_effect
  - https://pubmed.ncbi.nlm.nih.gov/27106854/ (neurophysiological: fluency produces mere-exposure effects)

### Illusory-truth effect (fluency-as-truth)  ·  _Fluency misattribution_

**What it is.** Statements encountered before are rated as more true on re-exposure, even when initially labeled false and even against the participant's own knowledge. Repetition increases perceived validity because repeated statements are processed more fluently than novel ones.

**Mechanism revealed.** The same fluency signal that drives mere-exposure (familiarity → liking) here drives familiarity → TRUTH. The system has no direct read on truth, so it substitutes the accessible proxy 'this feels familiar/easy to process.' Critically, the interpretation of fluency is LEARNED and reversible (Unkelbach 2007): if fluency is made diagnostic of falsity, the truth effect reverses. So fluency is not intrinsically a truth-signal — it is a generic accessibility cue that the inference system maps onto whatever target attribute it's trying to assess (truth, fame, liking).

**Recall-design implication.** Repetition in a memory store must NOT silently raise a node's credence/confidence — that reproduces illusory truth, letting often-repeated claims (including errors and the system's own past mistakes) calcify as 'true.' Keep two separate axes: accessibility (how easily a node surfaces) and confidence/veracity (how well-supported it is by evidence/corrections). Never let the former leak into the latter. The corrections substrate (edges that mark a node as superseded/corrected) is the antidote: it lets a highly-accessible-but-wrong node be flagged at surface time regardless of how fluent it feels.

Sources:
  - Hasher, L., Goldstein, D. & Toppino, T. (1977). Frequency and the conference of referential validity. JVLVB 16:107-112
  - Unkelbach, C. (2007). Reversing the Truth Effect: Learning the Interpretation of Processing Fluency. JEP:LMC — https://www.researchgate.net/publication/6598638
  - https://journals.sagepub.com/doi/10.1177/1754073910375479 (Moreland & Topolinski on Zajonc, fluency as uniting construct)

### Peak-end rule & duration neglect  ·  _Remembered- vs experienced-utility / summary encoding_

**What it is.** Retrospective evaluation of an experience is predicted almost entirely by the average of its most-intense moment (peak) and its final moment (end); total duration barely matters. Colonoscopy study (Redelmeier & Kahneman 1996): duration-evaluation correlation ≈ .03. Cold-pressor study: adding extra pain with a milder end made the memory BETTER.

**Mechanism revealed.** The episodic store does not record a faithful integral over time — it stores a SUMMARY SNAPSHOT keyed on a few high-salience moments (the affective peak and the most-recent moment). Duration neglect shows the store under-weights extent and over-weights extremes + recency, because the encoding samples representative/salient moments (Fredrickson & Kahneman tie this to the representativeness heuristic operating on memory itself). What gets laid down — hence what can later be queried — is a peak+end caricature, not the trace.

**Recall-design implication.** How experiences are SUMMARIZED at encode time determines what's retrievable forever — encoding is lossy by salience, not by time. If a memory system compresses a session into a node, it will (and arguably should) anchor on the peak (most surprising/intense/corrective moment) and the end (latest state), and will systematically lose the boring middle and the true duration/effort. Design implication: deliberately encode duration/effort as an explicit field if it matters, because the natural summary drops it; and recognize that the peak+end of a session disproportionately shapes the node's later accessibility and emotional valence.

Sources:
  - Redelmeier, D.A. & Kahneman, D. (1996). Patients' memories of painful medical treatments. Pain 66:3-8 — https://www.sciencedirect.com/science/article/abs/pii/0304395996029946
  - Kahneman, Fredrickson, Schreiber & Redelmeier (1993). When More Pain Is Preferred to Less: Adding a Better End. Psychological Science 4:401-405
  - https://en.wikipedia.org/wiki/Peak%E2%80%93end_rule
  - Kahneman (2000). Experienced Utility and Objective Happiness — https://www.anderson.ucla.edu/faculty/keith.chen/negot.%20papers/Kahneman_ExperiencedUtility00.pdf

### Framing effect  ·  _Reference-dependence / accessibility of construal_

**What it is.** Logically equivalent descriptions yield different choices. Asian-disease problem (Tversky & Kahneman 1981): 'save 200' (gain frame) → risk-averse; '400 die' (loss frame) → risk-seeking, despite identical outcomes. Choices reverse purely on labeling.

**Mechanism revealed.** The frame determines which reference point and which associated content is ACCESSIBLE — a gain framing makes gain-relevant representations come to mind, a loss framing makes loss-relevant ones. Kahneman 2003 explicitly explains framing via accessibility: 'different descriptions of the same problem bring different aspects to mind.' The judgment system operates on the accessible construal, not the extensional facts. Framing is anchoring's sibling — both show the query's surface form selects which subset of memory is activated.

**Recall-design implication.** The phrasing of a retrieval query is not neutral — it selects which facet of a node becomes accessible, and which related nodes spread-activate. The same underlying memory will return different content for 'what went well' vs 'what went wrong' even about the identical event. To get frame-robust recall, normalize/canonicalize queries before retrieval (strip valence framing), or run multiple framings and union the results. Be aware that an operator's framing imports a reference point the system will silently adopt.

Sources:
  - Tversky, A. & Kahneman, D. (1981). The Framing of Decisions and the Psychology of Choice. Science 211:453-458
  - Kahneman, D. (2003). Maps of Bounded Rationality. American Psychologist 58(9):697-720 — https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/JudgementAndChoice.MappingBoundedRationality.DKahneman2003.pdf

### Attribute substitution  ·  _Heuristic core (the unifying mechanism)_

**What it is.** When a target attribute is hard to assess, people unconsciously substitute a related but more ACCESSIBLE 'heuristic attribute' and map its value onto the target scale (Kahneman & Frederick 2002). E.g. asked about risk of dying on a trip, people answer by their fear (accessible) instead of computing probabilities. Availability, representativeness, and affect heuristics are all instances.

**Mechanism revealed.** This is the master mechanism that subsumes most of the others. Kahneman & Frederick define a heuristic precisely in accessibility terms: 'a relatively inaccessible target attribute is assessed by mapping a relatively accessible and related heuristic attribute onto the target scale.' The system answers the question it CAN cheaply retrieve an answer to, not the one it was asked. Accessibility differentials between attributes drive which substitution occurs. This is the deepest statement that judgment is gated by what the store can surface cheaply.

**Recall-design implication.** A retrieval system will tend to answer the EASY version of a query — surfacing nodes that match an accessible proxy (lexical similarity, recency, a salient tag) rather than the hard, intended target (deep relevance, causal fit). Guard: detect when the surfaced candidates only satisfy a proxy attribute and not the target; force a second pass against the actual target attribute. Make the target attribute itself more accessible (e.g. index nodes by the dimensions that matter for judgment) so the cheap-retrieval path and the correct path coincide.

Sources:
  - Kahneman, D. & Frederick, S. (2002). Representativeness Revisited: Attribute Substitution in Intuitive Judgment. In Heuristics and Biases (Cambridge) — https://www.anderson.ucla.edu/faculty/keith.chen/misc/KahnemanFrederick_HeuristicJudgment11.pdf
  - https://en.wikipedia.org/wiki/Attribute_substitution

### Representativeness heuristic (base-rate neglect, conjunction fallacy)  ·  _Heuristic / similarity-based retrieval_

**What it is.** Probability is judged by similarity to a prototype rather than by base rates or probability laws. Linda problem (Tversky & Kahneman 1983): 85% rate 'bank teller AND feminist' as more probable than 'bank teller' because the description matches the feminist prototype — a conjunction fallacy. Base rates are ignored when a vivid prototype is available.

**Mechanism revealed.** Retrieval/matching is done by SIMILARITY to a stored prototype, and that similarity score is substituted for probability (an instance of attribute substitution). The store surfaces the category whose prototype best matches the cue's features, and the match-strength is read as likelihood — so base rates (which aren't encoded in the prototype) drop out, and added matching details (feminist) can INCREASE retrieved similarity even as they decrease logical probability. This reveals that the store's native operation is feature-overlap matching, not frequency accounting.

**Recall-design implication.** Pure similarity-matching (cosine over embeddings) IS representativeness — it surfaces the prototype-nearest node and will ignore base rate (how often this type of node is actually the right answer) and will reward over-specific matches. Two corrections: (1) blend a base-rate/prior term into ranking (frequency a node has been the correct answer), not just similarity; (2) penalize spurious specificity — a node matching many surface features isn't necessarily more relevant. The brain's FTS5-lexical + cosine + z-weighting blend is exactly an attempt to not let raw similarity (representativeness) dominate.

Sources:
  - Tversky, A. & Kahneman, D. (1983). Extensional versus intuitive reasoning: The conjunction fallacy. Psychological Review 90:293-315
  - Tversky, A. & Kahneman, D. (1974). Judgment under Uncertainty. Science 185:1124-1131
  - https://en.wikipedia.org/wiki/Conjunction_fallacy

### Confirmation bias / motivated retrieval  ·  _Goal-conditioned selective retrieval_

**What it is.** People preferentially seek, weight, and RECALL information consistent with prior beliefs or desired conclusions (Lord, Ross & Lepper 1979; Kunda 1990). Memory search is biased toward belief-congruent evidence; ambiguous information is interpreted to fit.

**Mechanism revealed.** The current belief/goal acts as a retrieval CUE that raises accessibility of congruent traces and lowers accessibility of incongruent ones — selective memory retrieval (Kunda's 'directional' motivated reasoning constructs a justification using the subset of memory it can access while maintaining an illusion of objectivity). The store is queried with a hypothesis, and like selective accessibility in anchoring, it returns hypothesis-consistent content. Belief is a standing prime. This shows accessibility is conditioned not just on recent stimuli but on active goals and identity.

**Recall-design implication.** If retrieval is conditioned on the system's current hypothesis/goal/self-model, it will surface self-consistent memories and suppress disconfirming ones — a memory that only ever confirms itself never learns. Build an explicit disconfirmation pass: query for nodes that CONTRADICT the current belief (the corrections/supersedes edges are designed for exactly this), and weight them up at surface time rather than letting goal-congruence rank them down. The self-diagnostic: does the store ever surface something the operator/system did not want to hear?

Sources:
  - Kunda, Z. (1990). The Case for Motivated Reasoning. Psychological Bulletin 108(3):480-498
  - Lord, C., Ross, L. & Lepper, M. (1979). Biased assimilation and attitude polarization. JPSP 37:2098-2109
  - https://en.wikipedia.org/wiki/Confirmation_bias
  - https://en.wikipedia.org/wiki/Motivated_reasoning

### Hindsight bias (creeping determinism)  ·  _Reconstructive retrieval / memory updating_

**What it is.** After learning an outcome, people misremember their prior predictions as closer to it ('I knew it all along'), and see the outcome as having been inevitable/foreseeable (Fischhoff 1975). Hawkins & Hastie (1990) decompose it into memory distortion, inevitability, and foreseeability components.

**Mechanism revealed.** Memory retrieval of a PAST belief is reconstructed using PRESENT knowledge, not played back from a stored trace. The RAFT model: when asked to recall a prior judgment, the system re-derives it by 'taking the best' currently-accessible cue — and the outcome, now integrated into the knowledge base, has become a high-accessibility cue that contaminates the reconstruction. Once new info is encoded, the prior state becomes inaccessible (you cannot un-know to query the old state). This reveals retrieval as generative reconstruction, and that updates overwrite/occlude the queryable prior.

**Recall-design implication.** If a memory store updates nodes in place, it loses the ability to retrieve the PRIOR state — every recall of 'what we believed then' reconstructs from current content, reproducing hindsight bias. To support honest retrospection, version beliefs (immutable trace of the prediction AT prediction time) rather than overwriting, so the pre-outcome state stays queryable. The brain's trace_events + versioned interactions are this safeguard: the outcome of one cycle is recorded separately from the observation that preceded it, so you can recover what was known before.

Sources:
  - Fischhoff, B. (1975). Hindsight ≠ foresight: Effect of outcome knowledge on judgment under uncertainty. JEP:HPP 1:288-299
  - Hawkins, S. & Hastie, R. (1990). Hindsight: Biased judgments of past events. Psychological Bulletin 107:311-327
  - https://en.wikipedia.org/wiki/Hindsight_bias

### Curse of knowledge  ·  _Accessibility-of-own-state / perspective failure_

**What it is.** Once you know something, you cannot reconstruct the state of not knowing it, and you over-assume others share your knowledge (Camerer, Loewenstein & Weber 1989; Birch & Bloom on false-belief reasoning; tied to hindsight bias). Experts systematically fail to model less-informed audiences.

**Mechanism revealed.** Known information is so highly accessible that it cannot be SUPPRESSED when simulating a mind (or past self) that lacks it. Accessibility is sticky and intrusive: the system can add to the store but cannot reliably gate retrieval to exclude a now-known fact. This reveals an asymmetry — raising accessibility is easy, masking an accessible item to recover a counterfactual low-access state is hard/impossible. The same root as hindsight bias: no read-access to a prior, lower-knowledge state.

**Recall-design implication.** A system that has integrated a fact cannot reliably retrieve 'what a query looks like to someone without this fact' — so it will over-surface insider context and under-explain. For audience-aware recall (e.g. onboarding a new session, explaining to the operator), you cannot just suppress known nodes — you must explicitly model the target's knowledge state as a separate access-control filter over the store, rather than relying on the store to forget. Tag nodes with the context in which they were learned so 'common ground' vs 'private knowledge' is queryable.

Sources:
  - Camerer, C., Loewenstein, G. & Weber, M. (1989). The Curse of Knowledge in Economic Settings. J. Political Economy 97:1232-1254
  - https://effectiviology.com/curse-of-knowledge/
  - https://en.wikipedia.org/wiki/Curse_of_knowledge

### Salience / attentional bias & vividness effect  ·  _Attention-gated encoding & retrieval_

**What it is.** Stimuli that stand out perceptually or emotionally (vivid, novel, emotionally charged, perceptually prominent) capture attention and are over-weighted in judgment and attribution. Taylor & Fiske's 'top of the head' work: the visually salient person in an interaction is judged more causal/influential. Vividness effects occur specifically under DIFFERENTIAL attention.

**Mechanism revealed.** Attention is the gate that controls what enters the store at high strength AND what gets re-activated at retrieval; salience hijacks that gate. Items receiving differential attention are encoded with higher activation and are later disproportionately accessible — so judgment over-samples the salient. Taylor & Fiske show this is specifically about ATTENTION allocation (vividness fails when attention is equated). Salience is therefore an upstream determinant of accessibility: it sets the activation level at write-time and biases the read-time spotlight.

**Recall-design implication.** Whatever an encoder attends to most (the dramatic moment, the error, the emotional spike) gets written with the strongest trace and will dominate later recall — vivid-but-atypical events crowd out the representative-but-quiet ones. Counter at encode time: weight encoding by genuine importance, not just salience/emotion, and explicitly capture quiet-but-load-bearing context that attention naturally skips. At retrieve time, recognize that a salient node's high accessibility may reflect its vividness, not its relevance — discount accordingly.

Sources:
  - Taylor, S.E. & Fiske, S.T. (1978). Salience, attention, and attribution: Top of the head phenomena. Advances in Exp. Social Psych. 11:249-288
  - Taylor, S.E. & Fiske, S.T. (1975). Point of view and perceptions of causality. JPSP 32:439-445
  - https://thedecisionlab.com/biases/salience-bias
  - https://en.wikipedia.org/wiki/Salience_(neuroscience)

### Focusing illusion (focalism)  ·  _Attention-weighting in judgment_

**What it is.** Whatever you attend to while judging is over-weighted: 'Nothing in life is as important as you think it is while you are thinking about it' (Kahneman). Schkade & Kahneman 1998: people wrongly predict Californians are happier because, prompted to think about climate, they over-weight the salient, distinctive feature relative to its true contribution to wellbeing.

**Mechanism revealed.** When a judgment is cued to focus on a subset of the relevant whole, the attended subset is over-weighted and the unattended subset is neglected — because the act of querying about X makes X-related content maximally accessible and leaves everything else dormant. This is the focusing-time version of availability/attribute-substitution: the query itself constructs an accessibility profile skewed toward the focal attribute, and the judgment is built from that skewed profile. The unattended-but-relevant simply isn't retrieved.

**Recall-design implication.** Asking the store about a specific dimension makes that dimension's nodes dominate the answer, even when the true answer needs the un-asked dimensions — a narrow query yields a confidently-narrow, distorted retrieval. For holistic judgments (e.g. 'how is this project going?'), do NOT let a single focal query drive recall; decompose into multiple dimensions, retrieve each, and weight by true contribution rather than by which one the prompt happened to focus on. Surface what the query did NOT ask about when it's load-bearing.

Sources:
  - Schkade, D.A. & Kahneman, D. (1998). Does Living in California Make People Happy? A Focusing Illusion. Psychological Science 9:340-346 — https://web.mit.edu/curhan/www/docs/Articles/biases/9_Psychological_Science_340_(Schkade).pdf
  - Kahneman, D. et al. (2006). Would You Be Happier If You Were Richer? A Focusing Illusion. Science 312:1908-1910 — https://www.anderson.ucla.edu/sites/default/files/documents/areas/fac/accounting/Kahneman_-_Would_you_be_happier_if_you_were_richer_a_focusing_illusion.pdf

---

## 5. Theories of recall mechanism

_These ten theories are not a list of competing models — they are layers of one mechanism, and read together they spell out an architecture for artificial recall that the brain plugin already partially instantiates.

THE UNIFYING MECHANISM. Recall is cue-driven, partial-match reconstruction over a learned associative network, not lookup of a stored copy. Strip every model to its core and the same loop appears: a PROBE (assembled from content + drifting context) activates STORED TRACES in proportion to overlap; activation SPREADS through learned associations; the highest-activated, most-diagnostic traces are SAMPLED; and their contents are RECONSTRUCTED at readout, with gaps filled by schema. Each theory specifies a different organ of this loop. Hebb (1949) is the rule that BUILDS the associations (co-activation → connection). Collins & Loftus give the network those associations live in and the SPREADING dynamics over it. ACT-R gives the quantitative ACTIVATION EQUATION that decides what wins (base-level recency/frequency with power-law decay + context spreading + partial match + noise). SAM and REM give the RETRIEVAL DECISION: SAM's two-stage sample-then-recover with context+content cues, REM's Bayesian likelihood-ratio that makes rare cues diagnostic and strengthening differentiating. Tulving's encoding specificity states the GOVERNING CONSTRAINT — retrieval works only to the degree cues overlap what was encoded, so encode and decode must be symmetric. TCM makes CONTEXT a drifting, retrievable vector, explaining recency and contiguity as emergent from one mechanism and giving 'one memory leads to the next' a formal basis. Bartlett tells us the READOUT is generative, not playback — recall confabulates to fit schema. And the neuroscience pair grounds the whole thing: hippocampal indexing (Teyler & DiScenna) separates a sparse INDEX from heavy distributed CONTENT and does pattern-completion from partial cues; CLS (McClelland/Kumaran) splits FAST episodic from SLOW semantic and uses prioritized REPLAY to consolidate without catastrophic interference.

WHAT THIS TEACHES ABOUT BUILDING RECALL — the concrete spine:
1. Probe = context + content, drifting. Don't do flat top-k cosine on the raw query. Assemble a compound cue including the current focus/session arc (encoding specificity + SAM + TCM). The brain's Frame-as-prior and session-arc are this; the upgrade is letting a RETRIEVED node's own stored context re-seed the next hop (retrieved-context recall).
2. Seed then spread. Vector-nearest seeds, then spread activation through learned edges (Collins & Loftus). The brain already does seed-then-spread; the fan effect warns to degree-normalize hubs.
3. Rank by an ACT-R-shaped activation: power-law recency/frequency base + context spreading + similarity partial-match + noise. Power-law (heavy-tailed) decay, NOT exponential, keeps old-but-important nodes reachable.
4. Two strengths, not one (Bjork). Separate durable importance (storage strength, monotone up) from current accessibility (retrieval strength, volatile). Forgetting = low accessibility, never deletion. Weight hard-won retrievals (long-dormant node re-found) far more than effortless re-surfacing of an over-served node — which reframes synaptic fatigue (suppress high-RS) and Hebbian strengthening (reward low-RS successes) as the SAME desirable-difficulty principle.
5. Diagnostic, Bayesian cue weighting (REM): rare/distinctive cues carry more evidence than common ones (the IDF/FTS intuition), and better-encoded nodes should become both more retrievable and less confusable — making encoding quality (Healer) a recall lever, and arguing for merge-on-repeat (absorb) over duplicate traces (differentiation).
6. Index/content split (hippocampal indexing): cheap conjunctive index → heavy on-demand fetch. The brain's recall→get_node→correction-enrich pipeline is exactly this; invest indexing in separable conjunctive keys so a fragment completes the whole.
7. Two-speed storage with prioritized replay (CLS): fast episodic writes + slow offline consolidation, with replay WEIGHTED toward surprising/corrected/important items — which is precisely the S2 idle-gating lesson that uniform O(graph) replay every cycle is wasteful. Never let fast writes overwrite consolidated structure.
8. Treat reconstruction as a managed property (Bartlett): a generative (LLM) recall layer WILL confabulate to fit the prompt-schema. Anchor it with verbatim fields (raw quotes) the reconstruction cannot overwrite, keep the schema/prior (Frame) explicit, and recall gist nodes alongside their episodic sources.
9. Learn the edges from traffic (Hebb): co-accessed nodes strengthen their directed edge, off the hot path. Usage history becomes retrieval structure — the index improves itself, closing the loop back to step 2.

The deepest convergence: encoding and retrieval are two ends of one cue-overlap function (Tulving), the associative graph is grown by the same co-activation rule it is later searched by (Hebb → Collins & Loftus), and the system that stores fast must be complemented by one that consolidates slow (CLS) — so an artificial recall system is well-formed only when its encoder, its index, its activation ranker, and its offline consolidator are all designed against the SAME mechanism, not bolted together. The biases (A1 recency/frequency/priming/consolidation/contiguity; A2 context-cued, partial-cue, diagnostic-cue retrieval) are not failures to engineer away — they are the visible signatures of this mechanism working correctly, and a recall system that lacks them is not 'unbiased,' it is failing to remember the way memory works._

### Spreading-Activation Theory (Collins & Loftus, 1975; orig. Quillian)

**Core claim.** Semantic memory is a network of concept nodes linked by weighted, labeled associations. Activating one concept spreads activation outward along links, decreasing with distance and link strength, pre-activating related concepts so they are retrieved faster and more readily.

**Mechanism.** When a node is probed, activation propagates in parallel from it to neighbors, attenuating with semantic/associative distance and decaying over time/intervening nodes. Activation from multiple sources summates; where two spreading fronts intersect, an inferential/retrieval path is found. This single mechanism produces semantic priming, typicality gradients, free-association flow, and the fan effect (activation divided among many links = each retrieved more weakly).

**Recall implication.** An artificial recall system should not retrieve only the direct vector-nearest nodes to a probe — it should let the probe seed activation and SPREAD it through the graph, retrieving nodes reachable by short, strong paths even if not lexically/semantically nearest to the raw query. Seed-then-spread (exactly what the brain's 'spread activation through the graph' from surface-selected seeds does) is the faithful instantiation. The fan effect is a warning: a hub node connected to everything dilutes its own retrieval signal — edge weighting and degree-normalization matter.

**Explains:** A1 (associative priming / 'one thought triggers adjacent ones' — Tom's spiral thinking), A2 (context-driven retrieval where a partial cue surfaces a whole cluster), fan effect (over-connected hub nodes lose retrievability)

Sources:
  - Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. Psychological Review, 82, 407-428. https://eric.ed.gov/?id=EJ135579
  - https://www.cognitivepsychology.com/Spreading_Activation
  - https://link.springer.com/rwe/10.1007/978-1-4419-1428-6_76

### ACT-R Declarative Memory (Anderson; the retrieval-activation equation)

**Core claim.** Whether a memory chunk is retrieved, how fast, and how accurately is governed by a single scalar activation A_i that sums base-level strength (frequency × recency with power-law decay), context-driven spreading activation, partial-match penalties, and stochastic noise. Retrieval is the highest-activation chunk above a threshold; latency falls exponentially with activation.

**Mechanism.** A_i = B_i + Σ_j W_j·S_ji + P_i + ε. Base-level B_i = ln(Σ t_j^(-d)): each prior use contributes a power-law-decaying trace, so recency and frequency both raise activation. Spreading: W_j is source activation from the current context/goal buffer, S_ji the associative strength from cue j to chunk i — context literally pumps activation into matching memories. Partial matching P_i penalizes mismatch between request and chunk (graceful, similarity-based retrieval, not exact match). Logistic noise ε makes retrieval probabilistic (Boltzmann/soft-max over activations). Retrieval latency T_i = F·e^(-A_i); below the retrieval threshold τ, retrieval fails.

**Recall implication.** This is the most directly portable equation for an artificial recall ranker: combine (a) a recency/frequency base score with explicit power-law decay (NOT exponential — heavy tail keeps old-but-important nodes reachable), (b) context-conditioned spreading from the active goal/probe, (c) similarity-based partial matching instead of exact filters, and (d) a touch of noise so ranking isn't brittle-deterministic. The brain's z-weighted cosine + FTS lexical + synaptic-fatigue dampening + Hebbian co-access is a recognizable ACT-R analog; the principled upgrade is making base-level decay power-law and making context spreading a first-class additive term rather than a re-rank.

**Explains:** A1 (recency bias — recently-used nodes have higher base-level activation), A1 (frequency bias — often-accessed nodes dominate), A2 (context-cued retrieval via the spreading term W_j·S_ji), graceful degradation / fuzzy match (partial matching P_i), probabilistic/variable recall (noise term)

Sources:
  - ACT-R 6.1 Reference Manual (Bothell), act-r.psy.cmu.edu — A_i = B_i + S_i + P_i; B_i = ln(Σ t_j^-d); S_i = Σ W_j S_ji; T_i = F·exp(-A_i)
  - Anderson & Lebiere, The Atomic Components of Thought (1998/2014)
  - https://link.springer.com/article/10.3758/s13428-019-01286-2
  - https://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2012/12/459459.pdf

### Search of Associative Memory — SAM (Raaijmakers & Shiffrin, 1981)

**Core claim.** Recall is cue-dependent search: a probe assembled from CONTEXT cues + CONTENT cues activates stored traces in proportion to cue-trace overlap, driving a two-stage process of probabilistic SAMPLING (pick a trace) then RECOVERY (read out its contents). Context is itself a cue and is stored with every trace.

**Mechanism.** Each memory image stores item info, inter-item associations, AND the context active at study. At retrieval, a cue set Q produces sampling probability for image i proportional to the product of cue-strengths S(Q,i) over the trace's total summed strength (a global-matching ratio). A sampled image must then be recovered (a separate probabilistic step); recovered items can be re-used as cues, and context drift means the test-time context cue best matches recently/contiguously studied items. This was the first global-matching model and the explicit origin of 'retrieval cue' as a formal construct.

**Recall implication.** Build the query as a COMPOUND cue — current conversational context PLUS the explicit content of the probe — and rank candidates by overlap of BOTH against what each node stored (its content and its encoding context). Don't treat recall as a single similarity lookup; treat it as sample-then-verify: cheaply over-sample candidates by cue-match, then do a second recovery/selection pass (the brain's recall→Haiku-surface-selection is exactly SAM's sampling→recovery split). Store context with every node so context can serve as a probe later.

**Explains:** A2 (context-as-probe: the current situation cues the relevant cluster), A1 (cue-overload / part-list cuing — too many competitors under one cue lowers each one's sampling probability), two-stage recall justifies a cheap-retrieve + expensive-rerank architecture

Sources:
  - Raaijmakers, J. G. W., & Shiffrin, R. M. (1981). Search of associative memory. Psychological Review, 88, 93-134. https://raaijmakers.edu.fmg.uva.nl/PDFs/Raaijmakers%20and%20Shiffrin%201981.pdf
  - Kahana (2020), Computational Models of Memory Search, Annual Review of Psychology. https://memory.psych.upenn.edu/files/pubs/Kaha20.pdf
  - https://link.springer.com/article/10.3758/s13421-019-00896-7

### REM — Retrieving Effectively from Memory (Shiffrin & Steyvers, 1997)

**Core claim.** Memory traces are noisy, incomplete feature vectors; recognition/recall is BAYESIAN — the system computes the likelihood ratio that a probe was generated by a stored trace versus by chance, and decides accordingly. Strengthening a trace DIFFERENTIATES it (makes it match its own probe better and match lures worse).

**Mechanism.** Each studied item is stored as a vector of feature values, copied imperfectly (some features missing, some mis-stored). A probe is compared to every trace; for each, a likelihood ratio λ_i = P(data | this trace is the probe's source) / P(data | trace is unrelated) is computed from feature matches/mismatches against environmental base rates. The decision uses the average λ across traces (an odds against the null). Differentiation: adding correct features to a trace raises its λ for genuine probes AND lowers spurious matches to lures — explaining the strength-based mirror effect and the null list-strength effect.

**Recall implication.** Score candidates by EVIDENCE, not raw similarity: a feature/term that is rare in the corpus is far more diagnostic of a match than a common one (this is the Bayesian / IDF intuition the brain's FTS5 lexical layer already approximates). And richer, better-encoded nodes should become MORE retrievable for their true cues while becoming LESS confusable with neighbors — i.e. encoding quality (the brain's Healer filling situation/reasoning/question fields) is a recall lever, not just a storage nicety. Differentiation argues for consolidating repeats into one strengthened trace rather than spawning near-duplicate nodes (cf. the absorb op).

**Explains:** A2 (diagnostic-cue weighting: distinctive cues beat generic ones — Bayesian), false-alarm / false-recall reduction via differentiation, frequency/strength mirror effect, justifies merge-on-repeat (absorb) over duplicate-trace accumulation

Sources:
  - Shiffrin, R. M., & Steyvers, M. (1997). A model for recognition memory: REM—retrieving effectively from memory. Psychonomic Bulletin & Review, 4, 145-166. https://link.springer.com/article/10.3758/BF03209391
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC2889376/ (differentiation / strength-based mirror effect)

### Encoding Specificity Principle (Tulving & Thomson, 1973)

**Core claim.** A retrieval cue is effective to the exact extent that it overlaps the information encoded at the time of the experience. Retrieval success is a function of encode-retrieve MATCH, not of the cue's intrinsic semantic strength — even a strong semantic associate fails if it wasn't part of the encoding context.

**Mechanism.** Whatever is encoded with an item (context, co-present cues, internal state) becomes part of its trace. At test, a cue reinstates the item only insofar as it re-creates that stored encoding pattern. Demonstrated by recognition failure of recallable words: subjects recall a target from a weak cue that WAS present at study, yet fail to recognize the very same target presented with a strong-but-novel cue — match beats strength. Context-dependent (underwater vs land) and state-dependent memory are the same principle: external/internal context is part of the encoded cue set.

**Recall implication.** Encode and retrieve must be SYMMETRIC. Store the context a node was created in (conversation focus, situation, surrounding state) and probe with reinstated context, because a semantically-similar query that mismatches the encoding context will under-retrieve. This is the theoretical backbone of the brain's 'encode-decode symmetry' rule and its 'situation' field — the situation written at encode time is precisely the cue that must overlap at recall time. Practically: index nodes by their encoding context, and at recall reconstruct that context (session arc / current focus) as part of the probe.

**Explains:** A2 (context-dependent retrieval — the dominant explanation), A1 (state/mood-congruent recall), why pure semantic similarity under-retrieves context-bound memories, grounds the encode/retrieve-symmetry design invariant

Sources:
  - Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. Psychological Review, 80, 352-373. https://en.wikipedia.org/wiki/Encoding_specificity_principle
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC12056888/ (evolving engrams / changing effective cues)

### Reconstructive Memory & Schema Theory (Bartlett, 1932)

**Core claim.** Remembering is not playback of a stored copy but active RECONSTRUCTION: the system fills gaps and reshapes the trace to fit prior schemas, making 'effort after meaning.' Memory is generative — what you recall is partly built at retrieval time from the trace plus your existing knowledge structures.

**Mechanism.** Across repeated reproduction of an unfamiliar story (War of the Ghosts), recall showed systematic distortions — assimilation (details bent toward the rememberer's cultural norms), leveling (unfamiliar/'unimportant' detail dropped), and sharpening (events reordered for coherence). Schemas (organized prior knowledge) act as the K in retrieval: they supply defaults for missing detail and a coherence template the reconstruction is fitted to. The output is a plausible blend of episodic residue and schematic prior, not the original.

**Recall implication.** An artificial recall system that synthesizes (LLM-over-retrieved-nodes) is INHERENTLY reconstructive — it will confabulate plausible detail to fit the schema/prompt. Treat this as a property to manage, not eliminate: (1) preserve verbatim anchors (raw quotes) that the reconstruction must not overwrite — the brain's user_raw_quote/anchor_raw_quote fields are exactly the un-reconstructable ground truth; (2) keep the schema/prior (Frame) explicit and inspectable, since it shapes what's retrieved and how it's read; (3) expect higher-level summary nodes (S2 community/consolidation) to be schematized abstractions, useful for gist but not for verbatim fact — recall both gist nodes and their episodic sources.

**Explains:** A1 (schema-consistent distortion — memories drift toward priors), A2 (confabulation / gap-filling at retrieval), A1 (gist memory: detail leveled, structure sharpened), explains why a generative recall layer must be anchored by verbatim fields

Sources:
  - Bartlett, F. C. (1932). Remembering: A Study in Experimental and Social Psychology. Cambridge University Press. https://www.mheducation.ca/blog/series-classic-learning-science-reconstructive-memory-schema-theory/
  - https://www.studysmarter.co.uk/explanations/psychology/cognition/bartlett-war-of-the-ghosts/

### Hippocampal Memory Indexing Theory (Teyler & DiScenna, 1986; updated Teyler & Rudy, 2007)

**Core claim.** The hippocampus does not store the content of an episode — it stores a sparse INDEX (a pointer) to the set of neocortical regions that were active during the episode. Recall = a partial cue reactivates the index, which reactivates the distributed cortical pattern, reconstituting the memory.

**Mechanism.** During an event, the hippocampus captures which cortical feature-areas fired and binds them into a compact conjunctive index via LTP. Because the hippocampus projects back to those cortical areas, later presentation of a PARTIAL cue activates the index, which fans back out to re-ignite the full cortical pattern — pattern completion from a fragment. The content lives distributed in cortex; only the lightweight pointer/binding lives in the hippocampal index.

**Recall implication.** Separate the INDEX from the CONTENT. Keep a small, fast, sparse index of conjunctions/pointers (graph edges + lightweight embeddings + keywords) that maps a partial cue to the heavy distributed payload (full node content, fetched on demand). This is precisely the brain's architecture: a recall pass over a cheap index surfaces pointers, then get_node fetches the rich content with corrections enriched. The index, not the content, is what must be probe-completable from a fragment — so invest indexing effort in conjunctive, separable keys.

**Explains:** A2 (partial-cue completion — a fragment retrieves the whole episode), A1 (pattern completion / 'one detail brings back the scene'), architecturally grounds the lightweight-index → heavy-content-fetch split

Sources:
  - Teyler, T. J., & DiScenna, P. (1986). The hippocampal memory indexing theory. Behavioral Neuroscience, 100, 147-152. https://pubmed.ncbi.nlm.nih.gov/3008780/
  - Teyler, T. J., & Rudy, J. W. (2007). The hippocampal indexing theory and episodic memory: updating the index. Hippocampus. https://pubmed.ncbi.nlm.nih.gov/17696170/
  - Pattern separation (DG) / completion (CA3): https://pmc.ncbi.nlm.nih.gov/articles/PMC3812781/

### Complementary Learning Systems — CLS (McClelland, McNaughton & O'Reilly, 1995; Kumaran, Hassabis & McClelland, 2016)

**Core claim.** Intelligence requires TWO learning systems: a fast, sparse, pattern-separated hippocampal system that memorizes individual episodes in one shot, and a slow, distributed neocortical system that gradually extracts shared structure across many episodes. Interleaved replay of hippocampal memories trains cortex without catastrophic interference.

**Mechanism.** A single distributed network that learns fast suffers catastrophic interference — new patterns overwrite old ones because representations overlap. CLS solves this by splitting: the hippocampus uses sparse, non-overlapping codes to store specifics rapidly and safely; it then REPLAYS those memories (a 'training-trial multiplier'), interleaving them with ongoing experience so the slow cortical system can integrate them into structured semantics a little at a time, preserving prior knowledge. The 2016 update adds that replay need not be uniform — it can be GOAL-WEIGHTED/PRIORITIZED toward surprising or rewarding experiences.

**Recall implication.** Run two stores at two speeds. Keep a fast episodic store (write-on-the-spot, high fidelity, sparse/separated — the brain's per-turn encoding) AND a slow structural store built by offline consolidation (S2 community/consolidation synthesizing convergent clusters). Implement REPLAY: an idle-time process that re-reads episodic memories and integrates them into abstractions — and PRIORITIZE replay toward surprising/important/corrected items rather than scanning the whole graph uniformly (this is exactly the brain's S2 idle-gating + change-detection lesson: uniform O(graph) replay every cycle is wasteful; weight it by what changed). Never let fast new writes overwrite consolidated structure — additive, separated storage.

**Explains:** A1 (consolidation: episodic specifics → semantic gist over time), catastrophic interference avoidance (why new memories don't erase old structure), grounds offline consolidation + prioritized replay as a system requirement, and the gist-vs-detail dual representation

Sources:
  - McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102, 419-457. https://www.researchgate.net/profile/James-Mcclelland-4/publication/15575602
  - Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. Trends in Cognitive Sciences, 20, 512-534. https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30043-2

### Temporal Context Model / Retrieved-Context Theory (Howard & Kahana, 2002; TCM-A, CMR)

**Core claim.** A slowly DRIFTING context vector is bound to each item at study; recalling an item RETRIEVES the context that was active when it was studied, which then cues temporally-nearby items. This single drifting-and-retrieved-context mechanism produces both the recency effect and the (forward-asymmetric) lag-contiguity effect, scale-invariantly.

**Mechanism.** Context t_i is a recency-weighted running average of recently-experienced item features; it drifts gradually as new items push in. Each item is associated to the context present at its encoding (M_TF) and each item can push its features into context (M_FT). At recall, the current context probes item-to-context associations; because test context most resembles the most-recent study context, recent items win (recency). Recalling an item reinstates its old encoding context, which now overlaps the contexts of its temporal NEIGHBORS — so the next recall tends to be a nearby item, and slightly more often the FORWARD neighbor (asymmetric contiguity) because of how context is updated.

**Recall implication.** Make context a first-class, DRIFTING probe, not a static field. Maintain a running context representation that blends recent activity, and (1) bias retrieval toward recent context (recency), (2) when a node is retrieved, use ITS encoded context to pull temporally/associatively adjacent nodes — i.e. retrieval is jumpy along contiguity, not just a flat top-k. The brain's session-arc / current-focus blob is a context vector in this sense; the upgrade is letting a RETRIEVED node's stored context re-seed the next hop (retrieved-context recall), which naturally implements 'one memory leads to the next.' Temporal contiguity edges are worth materializing.

**Explains:** A1 (recency effect — most-recent items retrieved first/best), A1 (temporal contiguity — items studied together recalled together), A2 (context-cued jumping between associatively-linked memories), forward-asymmetry of associative chaining

Sources:
  - Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal context. Journal of Mathematical Psychology, 46, 269-299. https://www.researchgate.net/publication/222827636_A_Distributed_Representation_of_Temporal_Context
  - Polyn, Norman & Kahana (2009), CMR; Kahana (2020) Annual Review. https://memory.psych.upenn.edu/files/pubs/Kaha20.pdf

### New Theory of Disuse (Bjork & Bjork, 1992) — storage strength vs retrieval strength

**Core claim.** Memory has TWO independent strengths. Storage strength (how deeply learned/interconnected a memory is) only ever increases and never decays. Retrieval strength (current accessibility) fluctuates with recency and cues and can fall to zero — but the memory is not lost, only inaccessible. 'Forgetting' is retrieval-strength loss, not erasure.

**Mechanism.** Each item has SS (durable, accumulative) and RS (volatile, cue-dependent). New study/retrieval raises RS sharply; disuse lowers RS but leaves SS intact. Crucially, the GAIN in storage strength from a retrieval is INVERSELY related to current retrieval strength — successfully retrieving something that was hard to retrieve (low RS) produces a large, durable SS gain. This is the formal basis for 'desirable difficulties': spacing, interleaving, retrieval practice, and generation all temporarily lower RS at study, forcing effortful retrieval that maximizes long-term SS.

**Recall implication.** Separate a node's DURABLE importance from its CURRENT accessibility — two distinct scores. Don't let an item's accessibility decay imply deletion (it can be reactivated by the right cue). And invert the naive engagement signal: an EASY, recently-surfaced retrieval should strengthen the durable score less than a SUCCESSFUL retrieval of something that hadn't been surfaced in a long time — the latter is the more informative event. Spacing matters: re-surfacing the same node every turn (high RS) builds little durable strength; re-finding a long-dormant relevant node should be weighted heavily. This reframes the brain's synaptic-fatigue dampening and Hebbian co-access strengthening — fatigue suppresses high-RS over-served nodes (good), and the strengthening rule should reward hard-won retrievals more than effortless ones.

**Explains:** A1 (retrieval-induced forgetting / accessibility fluctuation), A1 (spacing effect — spaced re-access beats massed), A2 (cue-dependent forgetting: lost ≠ erased, reinstating the cue recovers it), grounds the durable-importance vs current-accessibility split and a difficulty-weighted strengthening rule

Sources:
  - Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old theory of stimulus fluctuation. https://www.researchgate.net/publication/281322665
  - Bjork & Bjork (2011), Making things hard on yourself, but in a good way. https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/04/EBjork_RBjork_2011.pdf

### Hebbian Learning (Hebb, 1949) — the association-formation rule

**Core claim.** Associations form by CO-ACTIVATION: when one unit repeatedly takes part in firing another, the connection between them strengthens. 'Cells that fire together wire together' — but with a causal/temporal ordering (A must fire just before B). This is the substrate rule that builds the weighted links every other theory above presupposes.

**Mechanism.** Hebb's postulate: 'When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process... increases A's efficiency in firing B.' Mechanistically realized by LTP (high-frequency co-activation persistently strengthens synapses) and refined by spike-timing-dependent plasticity (pre-before-post strengthens, post-before-pre weakens — temporal precedence, not mere simultaneity). Co-active cells form 'cell assemblies' that act as a unit, the neural basis of an associated memory.

**Recall implication.** Edge weights should be LEARNED from co-activation, not just authored. When two nodes are retrieved/used together, strengthen the edge between them (batched, off the hot path) — this is literally the brain's Hebbian co-access strengthening in recall_write_queue, and it's the rule that grows the spreading-activation graph that makes future associative recall work. Respect ordering/direction (the brain's directed source→target edge model and STDP's temporal asymmetry agree): co-access in a direction should bias the directed weight. Over time this turns usage history into retrieval structure — the index improves itself from the recall traffic.

**Explains:** A1 (associative strengthening — repeated co-recall builds priming paths), A2 (cue-target binding forms from co-occurrence), provides the learning rule UNDERNEATH spreading activation, SAM associations, and TCM item-context binding, justifies usage-driven edge-weight learning

Sources:
  - Hebb, D. O. (1949). The Organization of Behavior. Wiley. https://en.wikipedia.org/wiki/Hebbian_theory
  - STDP / LTP refinement: https://www.sciencedirect.com/topics/neuroscience/hebbian-theory

---

## 6. The brain's recall architecture (this repo)

The brain plugin implements 16 core recall and memory mechanisms that mirror human-memory phenomena, organized across three scales: S1 (turn-level recall/encoding), S2 (graph-level consolidation/community detection), and cross-scale encode-decode loops. Core innovations include z-weighted multi-group embeddings (4 semantic channels with contrastive scoring), Hebbian co-access strengthening (fire-together-wire-together learning), correction substrate with bidirectional aspect-walked chains, spreading activation with emergent depth, and Frame as a structured recognition prior that reifies operator identity and session context. Strong implementations exist for spreading activation, community structure detection via z-score scoring, consolidation via cluster synthesis, and synaptic fatigue via degree-based and access-based dampening. Notable gaps: no explicit metacognitive confidence recalibration, no spacing-effect adaptive intervals, no interference/output suppression, no transfer-appropriate processing mode adaptation, and no retrieval-induced forgetting. The system is fundamentally semantic-retrieval focused; phonetic, familiarity-based, and prospective-memory channels remain unbuilt."

### Features mapped to human phenomena

| Feature | Mirrors | Fidelity | What it does | Files |
|---|---|---|---|---|
| **Z-weighted Multi-Group Embeddings** | Multi-system retrieval cuing. Human memory retrieves via multiple context dimensions (semantic, phonetic, episodic) simultaneously; brain implements parallel channels. Also mirrors modulation by signal source importance (title > content metadata). | strong | Four semantic embedding channels (title:1.0x, blend/primary:0.85x, high_meta:0.70x, other_meta:0.40x) compute cosine similarity to query independently, then scores are z-normalized and combined via weighted averaging. Creates contrastive scoring where signal strength is measured relative to distribution rather than absolute cosine values. | `/Users/tpac/brain/servers/pipeline_contract.py`, `/Users/tpac/brain/servers/brain_recall.py` |
| **Synaptic Fatigue and Degree-Based Dampening** | Synaptic fatigue. Repeated activation reduces efficacy. Both degree-based (structural hubs) and access-count-based (recently recalled) dampening mirror diminishing returns on retrieval cue strength from overuse. | strong | High-degree nodes (hubs) receive fatigue multiplier K=10/(1+degree/10), applied post-scoring. Separately, access_count tracks surfacing frequency; frequency_penalty() applies log-scaled penalty (never below ~0.81x). Prevents hub nodes from dominating recall via structural centrality alone. | `/Users/tpac/brain/servers/brain_recall.py` |
| **Hebbian Co-Access Strengthening** | Hebbian learning. Nodes retrieved in same context develop stronger associative bonds. Directly implements fire-together-wire-together principle. | strong | When Haiku selects N nodes (3-5 typical), all C(N,2) pairs are enqueued as Hebbian events. Background drain thread atomically strengthens co_accessed relation weight via SQL (weight += LEARNING_RATE * 0.5, capped). Per-session dedup collapses multiple accesses within drain window to single increment. | `/Users/tpac/brain/servers/recall_write_queue.py` |
| **Spreading Activation and Graph Traversal** | Spreading activation in semantic networks. Activation from attended nodes flows through association pathways, reaching neighbors based on association strength. Depth emerges from path strength (Quillian's semantic distance). | strong | Post-selection, activation spreads from seed nodes through edges (up to 5 hops). Edge weighting: cosine(query, edge_enriched_text). Node activation via tanh saturation. Median scrutiny gate at hop 3+ gates propagation. Mutual traversal convergence produces boost. Returns per-node and per-field activation values. | `/Users/tpac/brain/servers/scales/s1/surface_contract.py` |
| **Frame as Structured Prior and Recognition Cue** | Encoding-specificity and schema activation. Human memory retrieval improves when context at retrieval matches encoding context. Frame provides structured schema aligning with operator's state-of-mind, enabling goal-directed retrieval. | strong | Five-section markdown deterministically constructed at boot and every recall: (1) Operator (locked principles), (2) Partnership (communities + locked moments + warm recent), (3) Active threads (open work), (4) Current focus (per-session arc), (5) Recent moves (journal). Serves as both boot context and per-turn prior for Haiku. Arc-relevance ranking lifts session-focused threads above global noise. | `/Users/tpac/brain/servers/scales/s1/frame.py` |
| **Community Detection via Z-Score Pair Scoring** | Community structure in semantic networks and chunking. Human memory organizes knowledge into domains. Z-score detection finds statistically strong clusters; first-class nodes reify emergent structure. | strong | Decoder scans pairwise cosines, computes z-scores within distribution, identifies pairs 2+ stddevs above mean. Builds connected components. Sonnet encoder assigns member_type, detects corridors, generates first-class community nodes. Idle gate skips if graph unchanged since last run. | `/Users/tpac/brain/servers/scales/s2/community.py` |
| **Consolidation via Cluster Detection and Synthesis** | Sleep consolidation. Human memory consolidates overlapping episodic traces into generalizable representations. Brain detects convergent clusters and synthesizes them into abstract nodes. | strong | Decoder identifies clusters of semantically similar nodes. Sonnet encoder: synthesize into new node + archive originals, or create similar_to edges, or skip. Fingerprint-based rejection prevents re-proposal. Per-run cap (30 clusters) controls throughput. | `/Users/tpac/brain/servers/scales/s2/consolidation.py` |
| **Correction Substrate and Aspect-Walked Chains** | Error correction and reconsolidation. Human memory updates via contradictory information detection. Brain walks correction chains at every retrieval, reactivating and updating memories. | partial | 22 correction relations (corrects, supersedes, reframes, resolves, etc.) form edges in correction_improvement aspect. Every node pull walks incoming/outgoing edges bidirectionally, attaching corrector metadata. Modes (lean/balanced/heavy) control rendering verbosity. | `/Users/tpac/brain/servers/brain_corrections.py` |
| **Recency and Freshness Weighting** | Primacy and recency effects. Human memory shows stronger retrieval for recently learned items, operationalized as time-decay curves. | strong | Unified scoring applies recency boost based on creation_at (not last_accessed). FRESHNESS_BANDS define boost schedules. Frame sorts by last_accessed. Relative-time labels provided instead of raw timestamps. | `/Users/tpac/brain/servers/recall_scoring.py` |
| **Query Expansion via Lexical Bridging** | Encoding-specificity and transfer-appropriate processing. Human retrieval improves when cues match encoding context. Query expansion increases cue coverage via context-adjacent phrasings. | partial | Optional opt-in (BRAIN_QUERY_EXPANSION=on). Haiku generates 2-3 alternate query phrasings via synonym/category/broadening strategies. Each embedded independently; cosine takes max across all. Bridges vocabulary gaps between user phrasing and original encoding. | `/Users/tpac/brain/servers/brain_recall.py` |
| **FTS5 Lexical Fallback and Semantic-Lexical Hybrid** | Dual-process retrieval. Human memory uses both semantic (meaning-based) and lexical (form-based) pathways. When semantic fails, lexical search bridges the gap. | partial | Parallel retrieval via SQLite FTS5 when semantic cosine is flat. Keyword matching surfaces exact-match candidates. Results merged via configurable blend (EMBEDDING_PRIMARY_WEIGHT + KEYWORD_FALLBACK_WEIGHT). FTS5-only tagged as discovery source. | `/Users/tpac/brain/servers/brain_recall.py` |
| **Node Healing and Gap-Filling** | Retrieval-induced elaboration. Human memory benefits from semantic context (situation, reasoning). Brain fills gaps with answers to when/why questions, enriching encoding. | partial | Decoder scans for nodes missing question/situation/reasoning. Haiku generates missing fields via prompts. Also archives edges pointing to archived nodes (invariant restoration), runs every cycle. | `/Users/tpac/brain/servers/scales/s2/healer.py` |
| **Encode-Recall Loop and Learnable Interaction Boundaries** | Metacognitive monitoring and feedback loops. Human memory learning depends on encoding-quality monitoring. Brain implements via trace history allowing prompt-version comparison and outcome analysis. | strong | interactions table stores versioned prompts for learnable boundaries: surface, encoding_agent, s2_community_enrichment, s2_consolidation_enrichment, s2_healer. Every cycle writes O/K/Delta traces. Delta from one scale becomes O for next (cross-scale integration). | `/Users/tpac/brain/servers/interaction_seed.py` |
| **Confidence and Emotional Modulation** | Emotional arousal modulation (GANE model). Human memory retrieves emotional memories more readily. Confidence predicts subjective vividness and retrieval likelihood. | partial | Unified scoring applies emotion_boost (abs(emotion) * EMOTION_AMPLIFICATION) and confidence_boost (maps 0-1 to -0.09 to +0.045). High confidence gets mild boost; low confidence mild penalty. Emotion amplifies by absolute value. | `/Users/tpac/brain/servers/recall_scoring.py` |
| **Situation as Retrieval Cue and Contextual Embedding** | Context-dependent retrieval and encoding specificity. Human memory retrieval improves when environmental/cognitive context at retrieval matches encoding. situation provides explicit contextual schema. | strong | situation field ('When is this knowledge relevant?') stored as first-class metadata, embedded as _situation vector. Becomes enriched-text input for spread-activation weighting. Query-situation cosine can gate retrieval. Frame renders situation for each node. | `/Users/tpac/brain/servers/contract.py` |
| **Voice Fields (user_raw_quote, anchor_raw_quote) and Episodic Flavor** | Episodic vs semantic memory distinction. Raw quotes preserve when/where/who detail (episodic) alongside semantic abstraction. Mirror phenomenological richness of autobiographical memories. | strong | user_raw_quote and anchor_raw_quote stored in metadata, rendered in corrections and full contexts. Bypass meta_limit, cap at 600 chars. Carry emotional tone and first-person immediacy. Appear in S1 Scribe and S2 healer. | `/Users/tpac/brain/servers/contract.py` |
| **Locking and Identity-Centric Importance** | Autobiographical importance and identity-consistency. Humans prioritize retrieval of self-concept memories. Brain implements via locking (prevents loss) and critical flagging (retrieval boost). | strong | Nodes locked (locked=1) by anchor MCP prevent archival. Identity-bearing nodes soft-pinned via aspect membership. Critical flag boosts relevance. Frame reads brain.aspects.identity_bearing to prioritize operator nodes. | `/Users/tpac/brain/servers/contract.py` |

### Gaps — human phenomena not (fully) implemented

| Phenomenon | Status | Why it could matter |
|---|---|---|
| Explicit metacognitive confidence recalibration without retrieval (separate monitoring loop) | not built | Human metacognitive accuracy depends on independent feeling-of-knowing scans. Brain stores confidence but does not implement explicit recalibration cycles separate from retrieval itself. |
| Spacing effect and adaptive inter-retrieval intervals | not built | Human retention depends on spacing and interleaving. Brain consolidates offline but lacks tracking of consolidation success or adaptive spacing intervals per node. |
| Primacy/recency effects at within-conversation granularity | not built | Human memory shows strong primacy on recently-presented lists and recency on long-term recall. Brain Frame provides session-level recency but lacks within-turn adjacency weighting. |
| Output suppression and interference control (active inhibition of competitors) | not built | Human retrieval involves competitive dynamics where target activation suppresses competitors. Brain reaches via spreading activation but lacks explicit output suppression or category-specific interference. |
| Transfer-appropriate processing (retrieval mode adaptation) | not built | Human memory depends on match between retrieval and encoding mode (semantic vs phonetic). Brain searches semantically but lacks detection of or adaptation to retrieval mode mismatches. |
| Autonomous offline reorganization and gist extraction (beyond prompted S2) | partial | Human sleep consolidation involves theta-ripple replay and autonomous schema refinement. Brain's S2 consolidation runs on idle time but is prompt-based (Sonnet decides), not autonomous schema reorganization. |
| Arousal-dependent thresholding and valence-specific routes | partial | Human emotional modulation shows inverted-U arousal curves and valence-specific retrieval routes. Brain applies linear emotion_boost without arousal thresholding or valence routing. |
| Retrieval-induced forgetting (category-specific suppression during recall) | not built | Human memory shows that retrieving one item suppresses related items in the same category. Brain implements fatigue on high-degree nodes but lacks category-specific retrieval inhibition. |
| Prospective memory and context-dependent intention binding | not built | Human memory includes forward-looking intentions and context-dependent triggers. Brain stores current_focus but lacks explicit prospective-memory substrate or trigger schemas. |
| Strength-independent familiarity and gist-based fast retrieval | not built | Human memory retrieves via both detailed recollection and rapid familiarity judgments. Brain implements deep semantic search but lacks explicit fast/gist familiarity channel or dual-process separation. |

---

## 7. Agent-memory systems — mainstream

Surveyed 13 mainstream agent-memory / LLM long-term-memory systems. Three distinct postures toward human-memory phenomena emerge. (1) EXPLICIT cognitive grounding — a minority, and almost all are research artifacts rather than production SDKs: HippoRAG (hippocampal indexing theory, neocortex/hippocampus roles via Personalized PageRank), MemGPT/Letta (OS virtual-memory paging as the named metaphor; sleep-time compute echoes consolidation but Letta itself frames "sleep" as a compute-timing metaphor, not neuroscience), Generative Agents (recency-with-exponential-decay + importance + relevance scoring, plus reflection — a direct memory-stream model of human recall), A-MEM (Zettelkasten note-linking, an externalized-cognition method not biology). (2) IMPLICIT resemblance — the dominant posture for production frameworks: LangMem/LangGraph and LlamaIndex adopt the semantic/episodic/procedural taxonomy (which IS borrowed from Tulving's cognitive-psychology distinction) but present it as an engineering convenience without citing the science; Mem0, Zep/Graphiti, and Cognee build extraction→consolidation→retrieval pipelines with decay, contradiction-invalidation, and usage-based edge reweighting that strongly resemble consolidation/forgetting/reconsolidation, yet their papers/docs are engineering-framed with no cognitive citations (Mem0's paper explicitly contains none; Zep's is "purely engineering"). (3) NONE / purely infrastructural — Microsoft Semantic Kernel memory is vector-store RAG plumbing with no memory-lifecycle model at all; OpenAI ChatGPT Memory is a product feature (saved memories + chat-history reference + contradiction-pruning) with no published cognitive rationale. The clearest pattern: the human-memory taxonomy (episodic/semantic/procedural) and the recency/importance/relevance retrieval triad have diffused into the field as DEFAULT vocabulary, so most systems echo cognitive science implicitly while only the research-paper systems cite it explicitly. The behavioral-bias literature (Kahneman/Tversky) is essentially absent — no surveyed system models availability, anchoring, or recency-as-bias deliberately; "recency" appears only as a retrieval-utility heuristic, not as a modeled human bias to correct for.

### Mem0 (mem0ai)

_Production memory layer that LLM-extracts salient facts from conversation, then runs an LLM decision engine to ADD/UPDATE/DELETE/NOOP against existing memories; offers a vector store and an optional graph variant (Mem0g)._

- **Architecture:** Two-stage streaming pipeline (extraction -> update) over a vector/relational store. Optional graph variant Mem0g: directed labeled graph of entity nodes (type label, embedding, timestamp) and relation-triplet edges (source, relation, destination).
- **Key mechanisms:** Extraction: LLM extracts candidate facts from a rolling window (m=10 recent messages) plus a running conversation summary.; Update/decision engine: for each candidate, retrieves S~10 semantically similar existing memories and an LLM classifies the operation as ADD / UPDATE / DELETE / NOOP.; Contradiction & dedup handling: outdated or contradictory relations are invalidated on detection; new nodes/edges merge with existing ones above a similarity threshold.; Decay/forgetting (graph variant): exponential decay exp(-lambda*dt), plus LRU and TTL-based pruning for staleness.; Reported benchmarks (LOCOMO): ~26% relative accuracy improvement vs OpenAI's built-in memory, ~91% p95 latency reduction, >90% token savings vs full-context.
- **Cognitive inspiration:** Resembles human memory consolidation (selective encoding of salient facts, integration/overwrite of related prior memories) and forgetting (decay/pruning). Contradiction-invalidation resembles memory updating/reconsolidation. But these are arrived at as engineering solutions, not derived from cognitive theory.  _(explicitness: implicit)_
- **Relation to human phenomena:** The extract-then-reconcile loop mirrors how human memory selectively encodes and then integrates new information with existing schemas (assimilation/accommodation), and the decay/TTL mechanisms mirror forgetting. However the Mem0 paper contains no explicit citations to human-memory or cognitive-science literature; the framing is production scalability, latency, and token cost.
- Sources:
    - Mem0 paper: 'Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory', arXiv:2504.19413 (https://arxiv.org/pdf/2504.19413)
    - https://www.emergentmind.com/topics/mem0-system
    - https://docs.mem0.ai/platform/features/graph-memory

### Letta / MemGPT (incl. sleep-time compute)

_Self-managing agent framework where the LLM treats the context window like OS RAM and pages information to/from external 'disk' tiers via tool calls; sleep-time compute adds a background agent that consolidates memory during idle periods._

- **Architecture:** OS-inspired memory hierarchy. Main context (RAM-like: system prompt, recent messages, editable core memory blocks) vs external storage: recall storage (searchable message history, 'disk') and archival storage (vector-indexed long-term knowledge, 'cold storage'). Agent moves data between tiers with explicit memory-management tools.
- **Key mechanisms:** Virtual context management: function/tool calls page data between in-context core memory and external recall/archival stores, creating the illusion of unbounded memory under a fixed context window.; Editable memory blocks: structured in-context regions the agent (or a sleep-time agent) rewrites as understanding evolves.; Sleep-time compute: a primary agent (handles user turns, can read but not edit memory) and a separate sleep-time agent that asynchronously rewrites/consolidates memory blocks during downtime into 'clean, concise' memories.; Self-editing via interrupts/heartbeats so the agent can chain memory operations autonomously.; Outperformed by Zep on the DMR benchmark (Zep 94.8% vs MemGPT 93.4%).
- **Cognitive inspiration:** The named inspiration is the operating-system memory hierarchy (RAM/disk paging via virtual memory), an engineering analogy rather than a biological one. Sleep-time compute superficially echoes sleep-dependent memory consolidation, but Letta presents 'sleep' as a compute-timing metaphor (thinking during idle time) not a neuroscience claim.  _(explicitness: explicit)_
- **Relation to human phenomena:** Explicitly inspired by hierarchical (virtual) memory systems in OSes; the MemGPT paper frames LLMs 'as operating systems'. The two-tier RAM/disk split loosely parallels working memory vs long-term memory, and sleep-time consolidation loosely parallels offline memory consolidation, but Letta's docs do NOT ground sleep-time compute in neuroscience -- it is explicitly a metaphor for processing during idle time.
- Sources:
    - MemGPT paper: 'MemGPT: Towards LLMs as Operating Systems', Packer et al. 2023 (arXiv:2310.08560)
    - https://www.letta.com/blog/sleep-time-compute
    - https://www.letta.com/blog/memory-blocks
    - https://www.letta.com/blog/agent-memory

### Zep / Graphiti

_Agent memory service built on Graphiti, a temporally-aware (bi-temporal) knowledge-graph engine that incrementally ingests conversation and business data, extracts entities/facts as graph edges with validity intervals, and clusters them into community summaries._

- **Architecture:** Temporal knowledge graph with three hierarchical tiers: episodic nodes (raw messages/events), semantic entities and facts (extracted knowledge as bi-temporal edges), and community summaries (cluster-level abstractions via label propagation). Non-lossy / append-style updates.
- **Key mechanisms:** Bi-temporal model: timeline T (when a fact was true: t_valid / t_invalid) plus timeline T' (when the system learned it: t_created / t_expired).; Contradiction handling via temporal edge invalidation: when new info contradicts an existing fact, the old edge is marked invalid (expired) rather than deleted, preserving history.; Incremental entity/relationship extraction with embedding + keyword (BM25) + graph search for retrieval.; Community detection (label propagation) to build cluster summaries -- higher-level abstractions over the fact graph.; Reported benchmarks: DMR 94.8% vs MemGPT 93.4%; up to 18.5% accuracy improvement and ~90% latency reduction on LongMemEval vs baselines.
- **Cognitive inspiration:** Uses the episodic vs semantic vocabulary (episodic raw-message nodes vs semantic entity/fact nodes) -- a distinction borrowed from cognitive psychology (Tulving) -- and community summaries resemble schema/gist abstraction. But the design is presented as a temporal-KG engineering approach, not derived from memory theory.  _(explicitness: implicit)_
- **Relation to human phenomena:** The episodic/semantic tiering parallels Tulving's episodic-vs-semantic memory distinction, and bi-temporal invalidation parallels human memory updating (keeping a trace of the superseded belief, akin to non-destructive reconsolidation). However the paper's framing is purely engineering -- it makes no explicit reference to human memory models, the episodic/semantic distinction as a cognitive concept, or cognitive psychology.
- Sources:
    - Zep paper: 'Zep: A Temporal Knowledge Graph Architecture for Agent Memory', arXiv:2501.13956 (https://arxiv.org/abs/2501.13956)
    - https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/
    - https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf

### LangMem / LangGraph memory (LangChain)

_LangChain's long-term-memory SDK and LangGraph store that explicitly organize agent memory into semantic (facts), episodic (past experiences/few-shot), and procedural (behavior/prompt rules) types, written either in the hot path or by background managers._

- **Architecture:** Pluggable store (vector DB, key-value, Postgres) holding three memory types. Semantic memory supports collections (unbounded searchable) and profiles (single structured document). Episodic = stored experiences/summaries. Procedural = mutable prompt rules the agent rewrites.
- **Key mechanisms:** Two write modes: hot-path tools the agent calls mid-conversation, and asynchronous background memory managers that extract memories after the fact.; Automatic consolidation: merges related facts and resolves contradictions.; Procedural memory self-improvement: the agent rewrites its own system-prompt rules from accumulated experience.; Profiles vs collections give bounded (overwrite) vs unbounded (append) semantic storage.; Retrieval via semantic/vector search over the store.
- **Cognitive inspiration:** Directly adopts the semantic / episodic / procedural memory taxonomy, which originates in cognitive psychology and neuroscience (Tulving's episodic-semantic distinction; procedural memory from amnesia/skill-learning research). The episodic 'few-shot from past successes' echoes case-based / experiential learning.  _(explicitness: implicit)_
- **Relation to human phenomena:** The three-way taxonomy is the single most direct human-memory borrowing among production SDKs -- semantic (facts/knowledge), episodic (autobiographical experiences), procedural (skills/how-to) map almost one-to-one onto the standard human long-term-memory taxonomy. LangChain's materials use these as established engineering terms and do not cite the underlying cognitive-science literature, so the borrowing is real but the framing is implicit.
- Sources:
    - https://www.langchain.com/blog/langmem-sdk-launch
    - https://docs.langchain.com/oss/python/langgraph/memory
    - https://www.digitalocean.com/community/tutorials/langmem-sdk-agent-long-term-memory

### LlamaIndex memory

_Agent memory with a short-term FIFO message buffer that 'waterfalls' overflow into configurable long-term memory blocks: a static block, a fact-extraction block, and a vector block._

- **Architecture:** Short-term memory = FIFO queue of recent messages under a token limit. Long-term memory = ordered list of memory blocks with priorities: StaticMemoryBlock (fixed info), FactExtractionMemoryBlock (LLM-extracted facts), VectorMemoryBlock (embedding search over older chat). Composable/waterfall design.
- **Key mechanisms:** Token-pressure trigger: when the short-term buffer hits its limit, relevant content is flushed into long-term blocks.; Priority system: each block has a priority used when total memory exceeds the token limit (priority 0 = always kept; lower priority blocks temporarily disabled/truncated).; Fact extraction: LLM distills durable facts from chat history into a maintained list.; Vector retrieval: embedding search fetches relevant slices of old conversation on demand.; Composable memory: multiple block types orchestrated together.
- **Cognitive inspiration:** The short-term buffer vs long-term blocks split loosely parallels working memory vs long-term memory, and overflow-on-pressure loosely parallels limited working-memory capacity. Fact extraction parallels gist/semantic consolidation. All framed as engineering primitives.  _(explicitness: implicit)_
- **Relation to human phenomena:** Short-term-to-long-term transfer under capacity pressure resembles the working-memory/long-term-memory architecture and consolidation, but LlamaIndex documents it purely as a token-budget engineering mechanism (FIFO, pressure size, priorities) with no cognitive-science framing.
- Sources:
    - https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents
    - https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
    - https://developers.llamaindex.ai/python/examples/memory/memory/

### Cognee

_Open-source AI memory engine that runs an Extract-Cognify-Load (ECL) pipeline to turn raw data into a knowledge graph spanning relational, vector, and graph storage, with a 'memify' self-improvement step._

- **Architecture:** Unified engine over three storage layers (relational + vector + graph). Data flows through ECL: Extract (ingest 38+ sources) -> Cognify (chunk, embed, extract entities/relationships, summarize) -> Load (commit to vector store + graph edges). Knowledge-graph-centric, not just chunk-and-embed.
- **Key mechanisms:** Cognify six-stage pipeline: classify docs, check permissions, extract chunks, LLM entity/relationship extraction, generate summaries, embed + commit edges.; Memify self-improvement: prunes stale nodes, strengthens frequent connections, reweights edges based on usage signals, and derives new facts from interaction traces.; Ontology grounding for structured, typed relationships.; Hybrid retrieval over graph + vector layers.
- **Cognitive inspiration:** The 'memify' step -- strengthening frequently-used connections and pruning stale ones based on usage -- closely resembles Hebbian strengthening ('cells that fire together wire together') and synaptic pruning / use-it-or-lose-it forgetting. The summarization step resembles gist abstraction. Presented as an evolving-graph engineering mechanism.  _(explicitness: implicit)_
- **Relation to human phenomena:** Usage-based edge reweighting and pruning is the closest any production system comes to a Hebbian/associative-strengthening model of human memory, and stale-node pruning mirrors decay-based forgetting. Cognee's materials frame this as a feedback-driven self-improving graph, not as a neuroscience-grounded design, so the resemblance is implicit.
- Sources:
    - https://www.cognee.ai/blog/fundamentals/how-cognee-builds-ai-memory
    - https://www.cognee.ai/blog/deep-dives/grounding-ai-memory
    - https://redis.io/blog/build-faster-ai-memory-with-cognee-and-redis/

### OpenAI ChatGPT 'Memory' feature

_Consumer product memory: ChatGPT stores explicit 'saved memories' (facts you ask it to remember or that it infers as useful) and, separately, references your broader chat history to personalize responses._

- **Architecture:** Two channels: (1) a curated, user-viewable/editable list of saved memory items; (2) 'reference chat history' that draws on past conversations for context. Underlying storage/retrieval implementation is not publicly documented in cognitive terms.
- **Key mechanisms:** Saved memories: explicit user-directed storage plus automatic inference of useful details, with auto-update of existing items.; Chat-history reference: personalization by recalling info and inferred preferences from prior conversations.; Staleness/contradiction management: an upgrade explicitly reduces stale or contradictory saved memories to keep context current.; User control: toggle either channel off, edit/delete memories, or use Temporary Chat to bypass memory.; Tiered behavior: free users get lightweight short-term continuity; Plus/Pro get longer-term understanding.
- **Cognitive inspiration:** The distinction between explicit, declarable 'saved memories' and ambient personalization from history loosely echoes explicit/declarative vs incidental learning, and the contradiction-pruning echoes memory updating. No published cognitive rationale.  _(explicitness: none)_
- **Relation to human phenomena:** Functionally it provides persistence and personalization a human would call 'remembering', and the de-staling of contradictory memories resembles belief updating, but OpenAI presents Memory as a product feature with controls -- there is no stated cognitive-science or human-memory design grounding.
- Sources:
    - https://openai.com/index/memory-and-new-controls-for-chatgpt/
    - https://help.openai.com/en/articles/8590148-memory-faq
    - https://help.openai.com/en/articles/11146739-how-does-reference-saved-memories-work

### Microsoft Semantic Kernel memory

_An SDK abstraction layer (ISemanticTextMemory / vector-store connectors) for embedding text and doing semantic search over pluggable vector databases -- effectively RAG plumbing rather than a memory-lifecycle model._

- **Architecture:** Embeddings + vector store connectors (Azure AI Search, Qdrant, Postgres, Redis, Volatile in-memory). Memory is a store/search abstraction; no built-in consolidation, decay, or episodic/semantic typing. Kernel Memory (related project) adds RAG ingestion pipelines.
- **Key mechanisms:** Embedding generation + similarity search over a chosen vector store.; Connector abstraction (IMemoryStore) so backends are swappable.; Hybrid search support in some connectors (vector + BM25 keyword), semantic ranking, faceted filtering (e.g. Azure AI Search).; RAG-oriented retrieval to ground LLM responses.; No native importance scoring, forgetting/decay, contradiction resolution, or memory-type taxonomy.
- **Cognitive inspiration:** Essentially none -- it is retrieval infrastructure. The word 'semantic' refers to semantic (embedding) search, not the cognitive notion of semantic memory.  _(explicitness: none)_
- **Relation to human phenomena:** Provides the substrate (vector recall) on which a memory system could be built, but does not itself model any human-memory phenomenon -- no consolidation, no forgetting curve, no episodic/semantic split, no recency/importance weighting. It is closest to a library/index than to a memory model.
- Sources:
    - https://deepwiki.com/microsoft/semantic-kernel/3.3-memory-and-vector-stores
    - https://dev.to/bspann/semantic-kernel-memory-vector-stores-embeddings-and-semantic-search-13e5
    - https://learn.microsoft.com/en-us/semantic-kernel/

### Generative Agents (Stanford, Park et al. 2023)

_Research agents whose believable behavior is driven by a 'memory stream' scored on recency, importance, and relevance, with periodic reflection that synthesizes higher-level insights -- the canonical cognitively-modeled agent memory._

- **Architecture:** Memory stream: an append-only natural-language log of observations, each with a creation timestamp and a last-access timestamp. Reflections (synthesized higher-level memories) are written back into the same stream, enabling recursive abstraction.
- **Key mechanisms:** Retrieval score = alpha_recency*recency + alpha_importance*importance + alpha_relevance*relevance.; Recency: exponential decay since the memory was last accessed (recently-accessed memories stay 'in the attentional sphere').; Importance: an LLM assigns each memory a salience score at creation.; Relevance: embedding cosine similarity between the memory and the current query.; Reflection: periodically the LLM uses the ~100 most recent memories to pose high-level questions, retrieves evidence, and writes synthesized insights back (recursive).
- **Cognitive inspiration:** Directly models human memory retrieval: recency (recency effect / time-based decay), importance (salience/emotional weighting), relevance (cue-based associative retrieval), and reflection (consolidation into higher-order semantic knowledge / gist).  _(explicitness: explicit)_
- **Relation to human phenomena:** This is the most influential explicit operationalization of human memory retrieval in the agent-memory literature: the recency-decay term is an engineering analog of the human recency effect and forgetting curve, importance scoring reflects salience-weighted encoding, and reflection mirrors the consolidation of episodic detail into semantic abstractions. The paper frames the memory stream and retrieval function explicitly as a model of how an agent should recall, drawing on human behavior.
- Sources:
    - Generative Agents paper: 'Generative Agents: Interactive Simulacra of Human Behavior', Park et al., arXiv:2304.03442 (https://arxiv.org/pdf/2304.03442)
    - https://ar5iv.labs.arxiv.org/html/2304.03442

### A-MEM (Agentic Memory, Xu et al., NeurIPS 2025)

_Agentic memory that builds Zettelkasten-style interlinked notes -- each new memory is a structured note with keywords/tags/context that the system dynamically links to related notes and that triggers 'memory evolution' of connected notes._

- **Architecture:** Network of atomic memory 'notes'. Each note: raw content, timestamp, LLM-generated keywords and tags, a context description, a dense embedding, and an (initially empty) link set. New notes are linked to semantically related existing notes, forming an evolving knowledge network.
- **Key mechanisms:** Note construction following the atomicity principle (one idea per note).; Dynamic link generation: new notes are connected to the most relevant existing notes via embedding similarity + LLM reasoning.; Memory evolution: adding a note can update the context/tags of linked notes, so old memories are revised in light of new ones.; Retrieval over the embedded, interlinked note network.
- **Cognitive inspiration:** Explicitly follows the Zettelkasten method -- a human note-taking / knowledge-management system (externalized cognition) emphasizing atomic notes and dense linking. This is a cognition-adjacent human practice rather than a neuroscience model; the 'memory evolution' of linked notes also resembles reconsolidation (old memories changing when reactivated).  _(explicitness: explicit)_
- **Relation to human phenomena:** The Zettelkasten inspiration is named explicitly; it is a model of human knowledge organization (associative linking, emergent structure) rather than of biological memory. The memory-evolution mechanism -- existing notes updated when a new related note arrives -- is a strong analog of human memory reconsolidation, where retrieval/reactivation can modify a stored memory.
- Sources:
    - A-MEM paper: 'A-MEM: Agentic Memory for LLM Agents', arXiv:2502.12110 (https://arxiv.org/abs/2502.12110)
    - https://github.com/agiresearch/A-mem

### HippoRAG

_Neurobiologically inspired long-term-memory framework that mimics the hippocampus-neocortex division of labor, using an LLM-built knowledge graph plus Personalized PageRank to integrate and retrieve across new experiences for multi-hop QA._

- **Architecture:** A schemaless knowledge graph (the 'hippocampal index') built by an LLM over a corpus, with the LLM/embeddings playing the neocortex role. Retrieval performs single-step multi-hop traversal via the Personalized PageRank algorithm over the graph seeded by query entities.
- **Key mechanisms:** LLM-driven open KG construction over documents (the index).; Query parsing into entities, then Personalized PageRank graph traversal to spread activation and rank passages -- enabling single-shot multi-hop retrieval.; Synonymy edges (via retrieval encoder) to connect equivalent concepts.; Reported up to ~20% improvement over strong RAG baselines on multi-hop QA, at lower cost/latency than iterative retrieval.
- **Cognitive inspiration:** Explicitly built on the hippocampal indexing theory of human long-term memory: the hippocampus stores an index/pointer to memory traces distributed in the neocortex. HippoRAG maps the knowledge graph to the hippocampal index and the LLM to the neocortex, with PageRank as the pattern-completion / associative-retrieval mechanism.  _(explicitness: explicit)_
- **Relation to human phenomena:** The most explicitly neuroscience-grounded system surveyed: it names the hippocampal indexing theory and the neocortex/hippocampus roles, and Personalized PageRank is presented as an analog of hippocampal pattern completion that enables integrating new experiences (overcoming the limits of parametric memory). It directly targets the human capability of cross-passage knowledge integration.
- Sources:
    - HippoRAG paper: 'HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models', arXiv:2405.14831 (https://arxiv.org/abs/2405.14831)
    - https://neurips.cc/virtual/2024/poster/94043

---

## 8. Agent-memory systems — biologically/cognitively inspired

Surveyed 14 agent-memory / retrieval systems and grouped them by how explicitly and specifically they model human memory phenomena. The most literal biological mappings are HippoRAG / HippoRAG 2 (hippocampal memory indexing theory of Teyler & DiScenna: LLM=neocortex, knowledge graph=hippocampal index, dense encoder=parahippocampal region, Personalized PageRank=pattern completion), EM-LLM (Bayesian-surprise event segmentation plus a temporal-contiguity retrieval buffer, validated against human-perceived event boundaries), Larimar and HEMA (Complementary Learning Systems: fast hippocampal episodic store + slow neocortical LLM; Larimar uses a Kanerva-Machine memory and supports one-shot write and selective forgetting), MemoryBank (Ebbinghaus forgetting curve R=e^(-t/S) with recall-driven strengthening), and the ACT-R-inspired architecture (base-level activation, power-law decay, retrieval threshold). The CLS/replay cluster (MIRROR, CraniMem, and the Pink et al. position paper 'Episodic Memory is the Missing Piece') explicitly ports consolidation/replay and Tulving's episodic-semantic split. Generative Agents (Park et al.) is human-memory-inspired but eclectic: its retrieval score = recency(0.995 exponential decay) + importance(LLM 1-10 poignancy) + relevance(cosine), all weights 1, plus reflection trees. A-MEM is inspired by the Zettelkasten note method (atomic notes, similarity-based linking, memory evolution). Memory^3 borrows the implicit/working/explicit memory taxonomy. RecallM targets temporal understanding and belief updating with a graph+vector store. SeCom (segment-level memory + compression denoising) ties only loosely to event segmentation. MemGPT is the deliberate boundary case — its inspiration is OS virtual-memory paging, an engineering metaphor rather than a model of human memory. Every system and load-bearing claim is cited to primary papers (arXiv / conference proceedings) plus the underlying psychology/neuroscience sources where applicable.

### HippoRAG

_A RAG framework that builds an LLM-generated knowledge graph as an 'artificial hippocampal index' and runs Personalized PageRank over it to do single-step multi-hop retrieval, mimicking hippocampal pattern completion._

- **Architecture:** Offline: an instruction-tuned LLM does open information extraction (OpenIE) over a corpus to build a schemaless knowledge graph (noun-phrase nodes + relation edges); dense retrieval encoders add synonymy edges between near-identical phrases. Online: query concepts become seed nodes; Personalized PageRank distributes probability across the graph; passages are ranked by accumulated node probability.
- **Key mechanisms:** LLM-as-neocortex OpenIE turns passages into KG triples (the perceptual-processing role); Schemaless KG = artificial hippocampal index storing interconnected pointers to memory units; Dense encoder adds synonymy edges, playing the parahippocampal linking role; Personalized PageRank from query seed nodes = pattern completion (retrieve whole memory from partial cue) in a single step; Node specificity weighting (IDF-like) sharpens seed influence; Single-step multi-hop retrieval reported 10-30x cheaper and 6-13x faster than iterative IRCoT
- **Cognitive inspiration:** Hippocampal memory indexing theory (Teyler & DiScenna 1986; updated 2007). Explicitly maps neocortex (LLM/OpenIE), hippocampal index (KG), and parahippocampal regions (retrieval encoder), and frames PPR as the graph analogue of hippocampal pattern completion, with the two theory objectives pattern separation (distinct encodings) and pattern completion (partial-cue retrieval).  _(explicitness: Explicit and central — the paper is titled 'Neurobiologically Inspired Long-Term Memory', names hippocampal indexing theory directly, and gives an explicit component-to-brain-region mapping table.)_
- **Relation to human phenomena:** Models associative recall across separately-stored experiences (multi-hop integration) the way the hippocampus knits neocortical traces; pattern completion = recovering a full memory from a partial query cue.
- Sources:
    - HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models, arXiv:2405.14831 (NeurIPS 2024) — https://arxiv.org/abs/2405.14831
    - https://arxiv.org/html/2405.14831v1
    - Teyler & DiScenna, The Hippocampal Memory Indexing Theory (1986); Teyler & Rudy, updating the index, Hippocampus 2007 — https://pubmed.ncbi.nlm.nih.gov/17696170/
    - GitHub OSU-NLP-Group/HippoRAG — https://github.com/osu-nlp-group/hipporag

### HippoRAG 2

_A successor that keeps the hippocampal-indexing + Personalized PageRank core but integrates full passages and deeper online LLM use to improve factual, sense-making, and associative memory toward 'human-like' long-term memory._

- **Architecture:** Two-stage offline-indexing / online-retrieval like HippoRAG, but PPR runs over a graph that integrates passages more richly (not purely entity-centric), reducing the context loss of the v1 entity graph; the online step uses the LLM more deeply to refine retrieval.
- **Key mechanisms:** Builds on Personalized PageRank over the KG (inherited hippocampal-indexing core); Richer passage integration to fix v1's entity-centric context loss; Deeper online LLM utilization during retrieval; Targets three explicit memory faculties: factual recall, sense-making, associative memory; Reported ~7% gain on associative-memory tasks over the leading embedding model
- **Cognitive inspiration:** Same hippocampal memory indexing theory; framed as advancing non-parametric continual learning toward 'the dynamic and interconnected nature of human long-term memory', explicitly decomposing human memory into factual / sense-making / associative components.  _(explicitness: Explicit — continues the neurobiological framing and organizes its evaluation around named human memory faculties.)_
- **Relation to human phenomena:** Associative memory and sense-making over accumulated experience; non-parametric continual learning (integrating new experiences without retraining), analogous to ongoing human memory consolidation.
- Sources:
    - From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2), arXiv:2502.14802 — https://arxiv.org/html/2502.14802v1
    - https://www.marktechpost.com/2025/03/03/hipporag-2-advancing-long-term-memory-and-contextual-retrieval-in-large-language-models/

### Generative Agents (Park et al.)

_Sandbox agents with a natural-language 'memory stream' whose retrieval scores each memory by a weighted sum of recency, importance, and relevance, plus periodic reflection that synthesizes observations into higher-level inferences._

- **Architecture:** A memory stream: an append-only list of natural-language observation records, each timestamped with creation and last-access time. Retrieval surfaces a top-ranked subset into the prompt; reflections are written back into the same stream as higher-level nodes, forming a reflection tree.
- **Key mechanisms:** Retrieval score = a_recency*recency + a_importance*importance + a_relevance*relevance, all alphas=1, each term min-max normalized to [0,1]; Recency = exponential decay (factor 0.995) over sandbox hours since last access; Importance = LLM-rated poignancy 1-10 (1 mundane, 10 emotionally significant); Relevance = cosine similarity between memory embedding and current query embedding; Reflection triggered when summed importance of recent events exceeds a threshold (150); LLM generates ~3 salient questions, retrieves evidence, writes insights with citations; Reflection tree: leaves = observations, higher nodes = increasingly abstract thoughts
- **Cognitive inspiration:** Broad cognitive-psychology model of human memory retrieval — the recency/importance/relevance triad mirrors how salient, recent, and contextually-related memories dominate human recall; recency uses an exponential forgetting-style decay; reflection mirrors human reflective abstraction / generalization from episodic experience.  _(explicitness: Explicit but eclectic — the paper motivates the retrieval factors by human memory behavior rather than citing one named theory; the design is human-memory-inspired by intent.)_
- **Relation to human phenomena:** Recency effect (recent memories more accessible), salience/emotional-importance weighting of memory, cue-driven relevance retrieval, and reflective consolidation of episodes into semantic self-knowledge.
- Sources:
    - Generative Agents: Interactive Simulacra of Human Behavior, Park et al., arXiv:2304.03442 (UIST 2023) — https://arxiv.org/abs/2304.03442
    - https://ar5iv.labs.arxiv.org/html/2304.03442

### A-MEM

_An agentic memory that writes atomic, richly-attributed notes, autonomously links each new note to semantically related existing notes, and lets new notes 'evolve' (update) the descriptions of old ones — a Zettelkasten for LLM agents._

- **Architecture:** Each memory is a structured note with content, timestamp, LLM-generated keywords, tags, a contextual description, an embedding, and a set of links to other notes. The store is a dynamically-growing interconnected note network rather than a flat log; links and note attributes mutate over time.
- **Key mechanisms:** LLM generates keywords, tags, and a contextual description per note at write time; Link generation: cosine-similarity top-k nearest notes are retrieved; LLM decides which links to form based on semantic relationships; Memory evolution: a new note triggers LLM updates to the contextual representations/attributes of its linked historical notes; No fixed/predetermined memory operations — structuring is agent-driven and emergent
- **Cognitive inspiration:** The Zettelkasten method (Niklas Luhmann's slip-box note system): atomic self-contained notes plus flexible, meaning-based linking. The memory-evolution step is explicitly framed as 'mimicking human learning processes' where new knowledge reshapes existing understanding.  _(explicitness: Explicit — Zettelkasten is named as the central inspiration; the human-learning framing is stated directly.)_
- **Relation to human phenomena:** Constructive/associative memory and schema updating — recall and new encoding reorganize prior memories rather than leaving them static, echoing reconsolidation and learning-driven restructuring.
- Sources:
    - A-MEM: Agentic Memory for LLM Agents, Xu et al., arXiv:2502.12110 (NeurIPS 2025) — https://arxiv.org/abs/2502.12110
    - https://arxiv.org/html/2502.12110v1
    - GitHub agiresearch/A-mem — https://github.com/agiresearch/a-mem

### MemoryBank

_A long-term memory module for LLM chatbots that applies the Ebbinghaus forgetting curve so memories decay over time but strengthen and resist forgetting each time they are recalled._

- **Architecture:** Three pillars: a memory storage repository (event/dialogue summaries + user-portrait profiles), a memory retriever (embedding similarity), and a memory updater governed by an Ebbinghaus-curve decay/strengthening rule. Demonstrated in the SiliconFriend companion chatbot.
- **Key mechanisms:** Retention modeled as R = e^(-t/S): R = retained fraction, t = time since last recall, S = memory strength; S initialized to 1 on first mention; On recall: increment S by 1 and reset t to 0, so the item decays more slowly and persists longer; Selective forgetting: items not recalled decay toward removal as t grows; Builds an evolving user portrait/personality model from past interactions
- **Cognitive inspiration:** The Ebbinghaus forgetting curve (Hermann Ebbinghaus, 1885) — exponential decay of retention over time — combined with the spacing/testing effect: retrieval strengthens memory. The authors explicitly call it a simplified model of human forgetting.  _(explicitness: Explicit and central — the Ebbinghaus curve is named and turned directly into the update equation.)_
- **Relation to human phenomena:** Forgetting curve (decay with time), spacing effect / retrieval-induced strengthening (recalled memories last longer), and selective forgetting of unused information.
- Sources:
    - MemoryBank: Enhancing Large Language Models with Long-Term Memory, Zhong et al., arXiv:2305.10250 (AAAI 2024) — https://arxiv.org/abs/2305.10250
    - https://ar5iv.labs.arxiv.org/html/2305.10250
    - GitHub zhongwanjun/MemoryBank-SiliconFriend — https://github.com/zhongwanjun/MemoryBank-SiliconFriend

### EM-LLM

_Adds human-episodic-memory machinery to a frozen LLM for near-infinite context: segments the token stream into events via Bayesian surprise, refines boundaries with graph theory, and retrieves via similarity plus a temporal-contiguity buffer._

- **Architecture:** The KV-cache stream is partitioned online into discrete 'episodic events'. At inference, two retrieval buffers feed the attention window: a similarity buffer (k-NN over event representations) and a contiguity buffer (temporally-neighboring events), with initial tokens kept as a separate context group. No fine-tuning.
- **Key mechanisms:** Bayesian surprise: boundary when -log P(x_t | x_<t) exceeds a dynamic threshold T = mu + gamma*sigma over a moving surprise window; Graph-theoretic boundary refinement using modularity / conductance over attention-key similarity to make events internally coherent; Two-stage retrieval: (1) similarity k-NN buffer, (2) contiguity buffer enqueuing +/- n neighboring events when an event is retrieved; Initial-token context group preserved (attention-sink / primacy)
- **Cognitive inspiration:** Human event cognition and episodic memory: Event Segmentation Theory (surprise/prediction-error at event boundaries), and free-recall phenomena — the temporal contiguity effect (items encoded close in time are recalled together) and primacy/recency. The paper shows LLM surprise correlates with human-perceived event boundaries.  _(explicitness: Explicit and central — titled 'Human-inspired Episodic Memory'; mechanisms are mapped one-to-one to named human memory phenomena and validated against human boundary annotations.)_
- **Relation to human phenomena:** Event segmentation at surprise points; temporal contiguity effect in retrieval; primacy/recency; hierarchical multi-scale event structure matching human-perceived events.
- Sources:
    - Human-inspired Episodic Memory for Infinite Context LLMs, Fountas et al., arXiv:2407.09450 (ICLR 2025) — https://arxiv.org/abs/2407.09450
    - https://arxiv.org/html/2407.09450v3
    - Project page — https://em-llm.github.io/
    - GitHub em-llm/EM-LLM-model — https://github.com/em-llm/EM-LLM-model

### Larimar

_Augments a frozen LLM with a distributed episodic memory module enabling one-shot fact writes, reads, and selective forgetting without retraining — a fast hippocampal store coupled to a slow neocortical LLM._

- **Architecture:** A Kanerva-Machine-style external memory matrix sits between the LLM encoder and decoder. Writes/reads are interpreted as inference in a generative model (deterministic least-squares variant). The memory posterior is updated by writing an edit; reading conditions the decoder to emit the edited fact.
- **Key mechanisms:** One-shot memory write: update memory posterior with the edit, no gradient training; Read-out conditions the decoder to produce the written fact; Selective forgetting via negative-coefficient (alpha = -1) sequential updates that subtract a previously written encoding; LLM-agnostic, 4-10x faster fact editing than gradient baselines; context-length generalization
- **Cognitive inspiration:** Complementary Learning Systems theory (McClelland, McNaughton & O'Reilly) — explicitly couples a fast hippocampal learner (single-instance episodic memory) with a slow neocortical learner (the frozen LLM modeling the input distribution). Memory mechanism derives from the Kanerva Machine / sparse distributed memory lineage.  _(explicitness: Explicit — names complementary fast (hippocampus) / slow (neocortex) systems and the Kanerva Machine; calls itself 'brain-inspired'.)_
- **Relation to human phenomena:** Rapid one-shot episodic encoding (hippocampal fast learning), and selective/directed forgetting of specific facts.
- Sources:
    - Larimar: Large Language Models with Episodic Memory Control, Das et al., arXiv:2403.11901 (ICML 2024) — https://arxiv.org/abs/2403.11901
    - https://arxiv.org/html/2403.11901v1
    - IBM Research — https://research.ibm.com/publications/larimar-large-language-models-with-episodic-memory-control

### MemGPT

_Treats the LLM like an operating system: a fixed context window is 'main memory' and external stores are 'disk', with the LLM self-paging data in and out to simulate unbounded context._

- **Architecture:** Tiered, OS-style hierarchy: main context (system prompt + recent messages + working scratch within the token window), recall storage (searchable database of all past messages), and archival storage (vector-indexed long-term document/knowledge store). The LLM issues function calls to move data between tiers and to page itself on memory-pressure interrupts.
- **Key mechanisms:** Virtual context management = paging between in-context 'RAM' and out-of-context 'disk'; Self-directed function calls for memory edits, search, and eviction; Interrupt-driven control flow on context-window pressure; Recall + archival tiers searched on demand to bring relevant records into main context
- **Cognitive inspiration:** Not a neuroscience theory — the inspiration is computer-systems analogy: hierarchical/virtual memory and paging in traditional operating systems. It borrows the engineering metaphor of multi-tier memory rather than a model of human memory phenomena.  _(explicitness: Explicit, but the analogy is to OS virtual memory, not biology — included here as the boundary case (systems metaphor, not cognitive).)_
- **Relation to human phenomena:** Loose at best: the tiered active-vs-stored split rhymes with working memory vs long-term memory, but MemGPT does not claim to model human memory phenomena; its lineage is OS paging.
- Sources:
    - MemGPT: Towards LLMs as Operating Systems, Packer et al., arXiv:2310.08560 — https://arxiv.org/abs/2310.08560
    - https://research.memgpt.ai/
    - Project / Letta — https://www.leoniemonigatti.com/papers/memgpt.html

### RecallM

_A hybrid graph-database + vector-store memory for LLMs built specifically to give belief updating and temporal understanding — tracking when knowledge was learned and revising stale beliefs._

- **Architecture:** Concepts extracted from input are stored as nodes in a graph database with relations and temporal/revision metadata, paired with a vector store for embedding retrieval. Updates revise the graph state so later contradictory information overwrites earlier beliefs while preserving temporal order.
- **Key mechanisms:** Concept-graph construction with relation edges and timestamps; Belief updating: new information revises prior stored knowledge rather than appending duplicates; Temporal context tracking across sequential updates (accurate over 72+ updates reported); Hybrid retrieval: graph-relational lookup + vector similarity; ~4x more effective at knowledge updating than a plain vector DB
- **Cognitive inspiration:** Human temporal memory and belief revision — the capacity to know the chronology of when things were learned and to update beliefs as new evidence arrives. Inspiration is functional/cognitive (temporal episodic ordering, belief updating) rather than tied to a specific named neuroscience theory.  _(explicitness: Moderate — motivated by human-like temporal understanding and cumulative learning; less tied to one named theory than HippoRAG or EM-LLM.)_
- **Relation to human phenomena:** Temporal order of memories, belief revision / updating, and resistance to confusion from outdated information.
- Sources:
    - RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models, Kynoch, Latapie & van der Sluis, arXiv:2307.02738 — https://arxiv.org/abs/2307.02738
    - https://arxiv.org/html/2307.02738v3

### Memory^3

_A from-scratch LLM that adds 'explicit memory' as a third memory form between model weights (implicit memory) and context KV-cache (working memory), externalizing knowledge into retrievable, sparsified memory._

- **Architecture:** Three-tier memory taxonomy: implicit memory (model parameters), working memory (context key-values), and explicit memory (externalized, sparsified knowledge stores retrieved and injected as KV-like memory). Knowledge is offloaded to explicit memory so the parametric model can shrink. A 2.4B model is trained to use it.
- **Key mechanisms:** Explicit memory tier cheaper than parameters and than text RAG, injected as key-value memories; Memory sparsification to make storage tractable; Two-stage pretraining scheme to teach the model to form and use explicit memory; 'Memory circuitry theory' justifying which knowledge can be externalized; Higher decoding speed than RAG at equal/better quality
- **Cognitive inspiration:** A memory-systems taxonomy analogy: the implicit / working / explicit hierarchy parallels the cognitive distinction between procedural/parametric knowledge, working memory, and explicit (declarative) long-term memory. Inspiration is the structural taxonomy of human memory types rather than a dynamical phenomenon like forgetting.  _(explicitness: Moderate — the three-form memory taxonomy is the explicit organizing metaphor; biological grounding is by analogy to memory types, with its own 'memory circuitry' theory.)_
- **Relation to human phenomena:** The implicit-vs-working-vs-explicit (declarative) memory distinction; externalizing declarative knowledge so the 'core' need only hold abstract/generalizable knowledge.
- Sources:
    - Memory^3: Language Modeling with Explicit Memory, Yang, Lin et al., arXiv:2407.01178 — https://arxiv.org/abs/2407.01178
    - https://arxiv.org/html/2407.01178v1

### SeCom

_A memory system for personalized conversational agents that builds the memory bank at the granularity of topically-coherent conversation segments and applies prompt-compression denoising before retrieval._

- **Architecture:** A conversation-segmentation model partitions long dialogues into topically coherent segments; each segment becomes a memory unit. Units are denoised via prompt compression (LLMLingua-2) and retrieved by similarity. The central claim is that segment-level granularity beats turn-level, session-level, and summary-level units.
- **Key mechanisms:** Conversation segmentation model into topically coherent segments (the memory unit); Compression-based denoising of memory units (LLMLingua-2) to raise retrieval accuracy; Granularity study: turn / session / summary units all have retrieval or semantic-quality limits; segment level is the sweet spot; SOTA on LOCOMO and Long-MT-Bench+ long-term conversation benchmarks
- **Cognitive inspiration:** The strongest tie is to event/topic segmentation as the unit of episodic memory — the idea that memory units should be coherent episodes (segments), not arbitrary turns. The denoising-before-storage idea loosely echoes consolidation (keeping signal, dropping noise). Less explicitly biological than EM-LLM; mostly an empirical engineering finding about memory granularity.  _(explicitness: Weak/implicit — segment-level units resonate with episodic event segmentation, but the paper frames itself empirically (granularity + denoising) rather than citing a memory theory.)_
- **Relation to human phenomena:** Segment/episode as the natural unit of memory (event segmentation); consolidation-as-denoising (retain meaningful gist).
- Sources:
    - SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents, Pan et al., arXiv:2502.05589 (ICLR 2025) — https://arxiv.org/abs/2502.05589
    - https://www.microsoft.com/en-us/research/project/secom/
    - GitHub microsoft/SeCom — https://github.com/microsoft/SeCom

### HEMA

_A hippocampus-inspired dual-memory architecture for long AI conversations: an always-visible one-sentence running summary plus a vector-indexed episodic store retrieved on demand._

- **Architecture:** Two complementary components. Compact memory: a continuously-updated one-sentence summary kept in context for global narrative coherence. Vector memory: an episodic store of chunk embeddings queried by cosine similarity for verbatim recall of specific past content.
- **Key mechanisms:** Compact memory = rolling one-sentence summary preserving global coherence; Vector memory = episodic chunk-embedding store, cosine-similarity retrieval; On-demand retrieval brings relevant episodes back into context; Dual system separates fast verbatim episodic recall from slow semantic continuity
- **Cognitive inspiration:** Hippocampal-cortical memory consolidation and Complementary Learning Systems theory — explicitly mapping fast episodic learning (vector retrieval) and slow semantic integration (the compact summary) onto hippocampal vs neocortical roles.  _(explicitness: Explicit — names the hippocampus and complementary learning theory as the design basis.)_
- **Relation to human phenomena:** Complementary fast-episodic vs slow-semantic memory; gist (summary) vs verbatim (episodic) memory traces; consolidation across long timescales.
- Sources:
    - HEMA: A Hippocampus-Inspired Extended Memory Architecture for Long-Context AI Conversations, arXiv:2504.16754 — https://arxiv.org/abs/2504.16754

### ACT-R-Inspired Memory Architecture for LLM Agents

_An LLM-agent memory that imports ACT-R's declarative-memory equations so agents remember and forget like the ACT-R cognitive model — activation, decay, and recency/frequency-driven retrieval._

- **Architecture:** Memory chunks carry an ACT-R-style base-level activation that grows with use frequency and decays with time; retrieval is governed by activation relative to a threshold, producing human-like remembering and forgetting patterns in the agent.
- **Key mechanisms:** Base-level activation per memory (rises with rehearsal/frequency); Power-law decay of activation over time; Retrieval threshold gating what is recalled; Recency and frequency jointly determine retrievability (and forgetting)
- **Cognitive inspiration:** ACT-R (Adaptive Control of Thought—Rational; John Anderson) declarative memory theory — specifically base-level activation, the power-law of forgetting, and retrieval-threshold dynamics, optionally spreading activation among associated chunks.  _(explicitness: Explicit and central — the architecture is named for and built on ACT-R's memory equations.)_
- **Relation to human phenomena:** Power-law forgetting, frequency/recency effects on recall, activation-based retrieval, and realistic forgetting rather than perfect storage.
- Sources:
    - Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture, Proc. 13th Intl. Conf. on Human-Agent Interaction (HAI 2025) — https://dl.acm.org/doi/10.1145/3765766.3765803
    - J.R. Anderson et al., ACT-R declarative memory / base-level activation (cognitive theory background)

### CLS / Replay / Consolidation for LLM Agents (MIRROR, CraniMem, and the episodic-memory position paper)

_A cluster of recent systems that explicitly port Complementary Learning Systems theory — a fast episodic store plus a slow semantic store linked by replay/consolidation — into LLM-agent memory, alongside a position paper arguing episodic memory is the missing piece._

- **Architecture:** Shared pattern: a fast high-fidelity episodic store (recent raw experience) consolidated into a slow, bounded semantic store via replay. MIRROR pairs fast turn-level encoding with slow reconstructive consolidation into a first-person narrative. CraniMem transfers selected experiences from a fast episodic store to a slow semantic store through replay (systems consolidation). The position paper (Pink et al.) argues for an explicit episodic-memory component as a distinct store.
- **Key mechanisms:** Fast episodic encoding of recent experience (hippocampal analogue); Slow consolidation into semantic/generalized knowledge (neocortical analogue); Replay/interleaving to transfer and stabilize knowledge and avoid catastrophic forgetting; Reconstructive consolidation (regenerate understanding) vs trace accumulation (MIRROR); Position paper's five required properties of episodic memory: long-term storage, explicit reasoning, single-shot learning, instance-specific memories, contextual memories
- **Cognitive inspiration:** Complementary Learning Systems theory (McClelland, McNaughton & O'Reilly 1995): fast hippocampal learning of specific episodes + slow neocortical consolidation of generalizable structure, with hippocampal replay interleaving old and new to prevent catastrophic interference; plus Tulving's episodic vs semantic memory distinction (position paper).  _(explicitness: Explicit — these works name CLS, hippocampal replay, systems consolidation, and (Pink et al.) Tulving's episodic memory directly as their design basis.)_
- **Relation to human phenomena:** Memory consolidation (episodic to semantic), hippocampal replay during rest, catastrophic-forgetting avoidance via interleaved replay, and the episodic/semantic memory split; single-shot, instance-specific, contextual episodic recall.
- Sources:
    - Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents, Pink et al., arXiv:2502.06975 — https://arxiv.org/abs/2502.06975
    - MIRROR: Complementary Encoding and Reconstructive Consolidation for Persistent State in LLM Systems — https://openreview.net/forum?id=IviO4bIZc7
    - CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems, arXiv:2603.15642 — https://arxiv.org/pdf/2603.15642
    - McClelland, McNaughton & O'Reilly (1995), Complementary Learning Systems theory (cognitive background)

---

## 9. Fact-check verdicts (adversarial pass on external-system claims)

| Verdict | System | Claim | Corrected statement |
|---|---|---|---|
| **confirmed** | — | Mem0's update phase uses an LLM to classify each candidate memory into one of exactly four operations -- ADD, UPDATE, DELETE, or NOOP -- after retrieving semantically similar existing memories (S~10), and the extraction phase uses a window of the last ~10 messages (m=10) plus a running summary. | In Mem0 (arXiv:2504.19413), the extraction phase prompts an LLM with two inputs — a running conversation summary (denoted S) plus the last m=10 messages — to produce candidate memories. The update phase then retrieves the s=10 most semantically similar existing memories from the vector store and, via a function-calling ("tool call") interface, has the LLM classify each candidate into exactly one of four operations: ADD, UPDATE, DELETE, or NOOP. Note the paper's notation: uppercase S is the conversation summary (not a count), while lowercase s=10 is the number of similar memories retrieved — so the similar-memory parameter should be written s=10, not "S~10," and the paper reports it as exactly 10 in its experimental configuration. |
| **confirmed** | — | Mem0 reports on the LOCOMO benchmark a ~26% relative accuracy improvement over OpenAI's built-in memory, ~91% lower p95 latency, and >90% token savings versus full-context. | On the LOCOMO benchmark, the Mem0 paper (arXiv:2504.19413) reports that Mem0 achieves a 26% relative improvement over OpenAI's memory on the LLM-as-a-Judge metric, and — relative to the full-context baseline — attains 91% lower p95 latency and >90% token-cost savings. (Note: the 26% gain is vs. OpenAI; the latency and token figures are vs. full-context, not vs. OpenAI. These are first-party results from Mem0's own authors. A graph-memory variant scores ~2% higher overall than base Mem0.) |
| **confirmed** | — | Zep beats MemGPT on the Deep Memory Retrieval (DMR) benchmark by a small margin (94.8% vs 93.4%) and shows larger gains (up to ~18.5% accuracy improvement, ~90% latency reduction) on LongMemEval; its Graphiti engine uses a bi-temporal model tracking both when a fact was valid (T) and when the system learned it (T'). | Per the primary source (arXiv:2501.13956, "Zep: A Temporal Knowledge Graph Architecture for Agent Memory," Jan 2025), Zep outperforms MemGPT on the Deep Memory Retrieval (DMR) benchmark by a small margin — 94.8% vs 93.4% (both on gpt-4-turbo). On LongMemEval, Zep improves accuracy by up to ~18.5% (relative) — e.g., gpt-4o full-context 60.2% vs Zep 71.2% — while cutting response latency by ~90% (gpt-4o: 28.9s → 2.58s) versus the full-context baseline. Its Graphiti engine uses a bi-temporal model with two timelines: T (valid time — when a fact held true in the world, via t_valid/t_invalid) and T′ (transaction time — when the system created/invalidated the fact, via t′_created/t′_expired), i.e., four timestamps total across the two timelines. |
| **confirmed** | — | Letta's sleep-time compute spawns two agents -- a primary agent that handles user turns and can read but NOT directly edit memory, and a separate sleep-time agent that asynchronously rewrites/consolidates the memory blocks -- and Letta frames 'sleep' as a compute-timing metaphor, not a neuroscience claim. | Confirmed and tightened: When sleep-time is enabled (enable_sleeptime: true), Letta instantiates two agents that share memory blocks — a primary agent and a sleep-time agent. The primary agent handles user turns and is given conversation_search and archival_memory_search tools; the shared memory blocks sit in its context (so it can read them) but it is "not provided with tools to edit its core memory." The separate sleep-time agent runs in the background and asynchronously rewrites/consolidates those shared memory blocks ("generates learned context from the conversation history to update the memory blocks of the primary agent"; handles compaction/archive maintenance). "Sleep-time" is a compute-timing metaphor — spending compute offline/during idle periods between user turns to pre-compute and reorganize state — originating in Letta's own paper "Sleep-time Compute: Beyond Inference Scaling at Test-time" (arXiv:2504.13171); it is not a neuroscience/biological-sleep claim. (One precision: "read but not edit" applies to the in-context core memory blocks; the primary agent can still search its external recall/archival memory.) |
| **confirmed** | — | Generative Agents score memory retrieval as a weighted sum of recency, importance, and relevance, where recency is an exponential decay since last access, importance is an LLM-assigned salience score, and relevance is embedding similarity; reflections are generated from roughly the 100 most recent memory records. | Generative Agents (Park et al., 2023, arXiv:2304.03442) score each memory at retrieval time as a weighted sum of three components — recency, importance, and relevance — each min-max normalized to [0,1] and then combined with equal weights (all α coefficients set to 1 in the reference implementation): score = α_recency·recency + α_importance·importance + α_relevance·relevance. Recency is an exponential decay (factor 0.995) over the number of sandbox hours since the memory was last RETRIEVED (i.e., decay since last access, not since creation). Importance is an LLM-assigned integer salience score on a 1–10 scale (1 = mundane, 10 = poignant). Relevance is the cosine similarity between the memory's embedding and the query's embedding. Reflection is TRIGGERED when the summed importance of the latest perceived events exceeds 150; once triggered, the reflection step QUERIES the LLM with the 100 most recent records in the memory stream to produce salient high-level questions, which are then used as retrieval queries to gather the memories that ground each reflection. (The "~100 most recent records" figure is exact and describes the reflection query input, not the trigger condition.) Reference implementation used gpt-3.5-turbo. |
| **confirmed** | — | HippoRAG is explicitly based on the hippocampal indexing theory and orchestrates an LLM-built knowledge graph with the Personalized PageRank algorithm to mimic the neocortex/hippocampus division of labor, reporting up to ~20% improvement over state-of-the-art RAG on multi-hop QA. | HippoRAG (Gutiérrez et al., NeurIPS 2024; arXiv:2405.14831) is explicitly "inspired by the hippocampal indexing theory of human long-term memory." It uses an LLM to build an open knowledge graph from passages and applies the Personalized PageRank algorithm over that graph to perform single-step multi-hop retrieval (pattern completion), mimicking the division of labor between neocortex (LLM/encoders) and hippocampus (the PPR index). It reports retrieval improvements of up to ~20% over state-of-the-art RAG on multi-hop QA — but that headline figure is benchmark- and stage-specific: ~20 points on 2WikiMultiHopQA versus only ~3 on MuSiQue, measured as retrieval recall (recall@2/@5), not end-task QA accuracy. The specific algorithm (Personalized PageRank) and the named theory in the original claim are accurate. |
| **confirmed** | — | HippoRAG runs the Personalized PageRank algorithm over an LLM-built knowledge graph (the 'artificial hippocampal index'), using query concepts as seed nodes, to emulate hippocampal pattern completion and achieve single-step multi-hop retrieval. | HippoRAG (Gutiérrez et al., NeurIPS'24, arXiv:2405.14831) runs Personalized PageRank over a schemaless, LLM-built open knowledge graph it calls the 'artificial hippocampal index' (with the LLM acting as the 'artificial neocortex' for OpenIE triple extraction and dense retrieval encoders standing in for the parahippocampal regions). At query time it extracts named entities from the query, matches them to KG nodes, and uses those nodes — weighted by node specificity (an IDF-like signal) — as the PPR seed/reset distribution. Biasing PPR toward these seed nodes is presented as the computational analog of hippocampal pattern completion (extracting associated signals from partial cues), letting the system perform multi-hop reasoning in a single retrieval step. The original claim is accurate; the only tightening is that the PPR seeds are query named-entity nodes (IDF-weighted), which 'query concepts' paraphrases correctly. |
| **confirmed** | — | HippoRAG explicitly maps the LLM/OpenIE to the neocortex, the schemaless knowledge graph to the hippocampal index, and the dense retrieval encoder to the parahippocampal regions, citing Teyler & DiScenna's hippocampal memory indexing theory. | HippoRAG (Gutiérrez et al., arXiv:2405.14831, NeurIPS 2024) explicitly maps its three components to brain regions per Teyler & DiScenna's (1986) hippocampal memory indexing theory: the instruction-tuned LLM performing OpenIE is the "artificial neocortex"; the resulting schemaless/open knowledge graph is the "artificial hippocampal index" (searched via Personalized PageRank as the "synthetic hippocampus"); and off-the-shelf dense retrieval encoders play the role of the "parahippocampal regions," connecting the two by adding synonymy edges between similar-but-not-identical noun phrases. The component-to-region correspondence is stated verbatim in the paper, though across several adjacent sentences in Section 2.2 rather than a single consolidated statement. |
| **confirmed** | — | Generative Agents (Park et al.) score each memory as a weighted sum of recency, importance, and relevance with all alpha weights = 1: recency is exponential decay with factor 0.995, importance is an LLM poignancy rating 1-10, and relevance is embedding cosine similarity. | CONFIRMED as stated. Generative Agents (Park et al., 2023, "Generative Agents: Interactive Simulacra of Human Behavior," arXiv:2304.03442, §4.1) compute a memory's retrieval score as a weighted sum of three terms — score = α_recency·recency + α_importance·importance + α_relevance·relevance — with all α weights set to 1 in their implementation. Recency is an exponential decay (factor 0.995) over sandbox-game-hours since the memory was last retrieved; importance is an LLM-generated "poignancy" rating on an integer 1-10 scale (1 = mundane, 10 = extremely poignant); relevance is the cosine similarity between the memory's embedding vector and the query's embedding vector. One precision addendum: each of the three component scores is min-max normalized to [0, 1] before the (unit-weighted) sum. |
| **confirmed** | — | MemoryBank applies the Ebbinghaus forgetting curve as R = e^(-t/S), initializes memory strength S=1, and on each recall increments S by 1 and resets t to 0, so recalled memories decay more slowly. | MemoryBank (Zhong et al., arXiv:2305.10250, AAAI 2024) models the Ebbinghaus forgetting curve as an exponential-decay retention R = e^(-t/S), where R is memory retention, t is time elapsed since learning, and S is the memory strength. S is a discrete (integer) value initialized to 1 upon a memory's first mention. Each time a memory item is recalled in conversation, S is increased by 1 and t is reset to 0, which lowers its forgetting probability and makes it persist longer (a flatter decay curve) — so frequently recalled memories decay more slowly. The claim is accurate as stated; the only added precision is that S is initialized to 1 on first mention and is explicitly a discrete value, and the strengthening is driven by recall frequency (and, per the abstract, memory significance) rather than the classic spaced-repetition spacing schedule. |
| **confirmed** | — | EM-LLM segments the token stream into episodic events using Bayesian surprise (a dynamic threshold T = mu + gamma*sigma over a moving surprise window), refines boundaries with graph-theoretic metrics (modularity/conductance), and retrieves via a similarity buffer plus a temporal-contiguity buffer. | EM-LLM (Fountas et al., ICLR 2025; arXiv:2407.09450) segments the token stream online into episodic events by placing a boundary wherever per-token surprise — the negative log-likelihood of the ground-truth token, -log P(x_t \| x_<t; theta), which the paper frames as Bayesian surprise — exceeds a dynamic threshold T = mu + gamma*sigma, where mu and sigma are the mean and standard deviation of surprise over a moving window (offset tau) and gamma is a user-set scaling factor. It then refines these initial boundaries with graph-theoretic metrics computed on the attention-key similarity graph, repositioning each boundary to maximize modularity or minimize conductance. At inference it retrieves events through a two-stage process: a similarity buffer (k-NN over event representative tokens vs. the current query) plus a temporal-contiguity buffer (neighboring events that preserve temporal context). The paper empirically validates that its surprise-based boundaries align closely with human-perceived event boundaries on annotated transcripts. |
| **confirmed** | — | Larimar implements Complementary Learning Systems theory by coupling a fast hippocampal episodic-memory module (Kanerva-Machine-style, one-shot writes) with a slow neocortical frozen LLM, and supports selective forgetting via negative-coefficient memory updates. | Larimar (Das et al., ICML 2024; arXiv:2403.11901) follows the Complementary Learning Systems (CLS) view: a fast "hippocampal" episodic-memory module records samples one-shot while a frozen "neocortical" LLM supplies slow-learned semantic statistics, conditioned at inference by the updated memory (no gradient updates to the LLM). The memory is "similar in spirit to the Kanerva Machine," realized via the deterministic Generative Pseudoinverse Memory (GPM) of Pham et al. (2021), which recasts the Kanerva Machine's Bayesian memory/address updates as least-squares solutions; writes are one-shot via pseudo-inverse. Selective fact forgetting is training-free: a previously written encoding (write coefficient α=1) is removed by re-applying the update with a negative write coefficient α=−1 (Eqs. 3-4). One nuance worth tightening: the Kanerva link is "Kanerva-Machine-style via the GPM least-squares reformulation," not a direct use of the original probabilistic Kanerva Machine. |

<details><summary>Full evidence per verdict</summary>

**[confirmed]** Mem0's update phase uses an LLM to classify each candidate memory into one of exactly four operations -- ADD, UPDATE, DELETE, or NOOP -- after retrieving semantically similar existing memories (S~10), and the extraction phase uses a window of the last ~10 messages (m=10) plus a running summary.

- Evidence: Verified against the PRIMARY source: the Mem0 paper "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory" (arXiv:2504.19413v1, April 2025), HTML at arxiv.org/html/2504.19413v1.

(1) FOUR-OP ENGINE — confirmed exactly, including operation names. The paper states verbatim: "The LLM itself determines which of four distinct operations to execute: ADD for creation of new memories when no semantically equivalent memory exists; UPDATE for augmentation of existing memories with complementary information; DELETE for removal of memories contradicted by new information; and NOOP when the candidate fact requires no modification." The candidate is passed to the LLM "through a function-calling interface we refer to as a 'tool call.'"

(2) UPDATE-PHASE RETRIEVAL OF SIMILAR MEMORIES — confirmed. The paper configures "'s' = 10 similar memories for comparative analysis." The update phase retrieves the s most similar existing memories from the vector DB, then the LLM chooses among the four ops. NOTE the symbol: the similar-memory count is lowercase s = 10, NOT uppercase S. The claim's notation "S~10" conflates this with the summary symbol (see below) — the magnitude (~10) is right but the symbol is s, and the paper reports it as exactly 10 in the experimental config, not approximate.

(3) EXTRACTION PHASE — confirmed. The paper states: "we configured the system with 'm' = 10 previous messages for contextual reference," and the extraction step "employs two complementary sources: (1) a conversation summary S retrieved from the database that encapsulates the semantic content of the entire conversation history, and (2) a sequence of recent messages." So: last m=10 messages PLUS a running/conversation summary — exactly as claimed.

SYMBOL DISAMBIGUATION (the one wording slip in the claim): the paper uses UPPERCASE S for the conversation summary (a single running summary blob, not a count) and LOWERCASE s for the number of similar memories retrieved in the update phase (s=10). The claim wrote "S~10" for similar memories, which mislabels the summary symbol as the retrieval count. The mechanism and both magnitudes are correct; only the symbol casing/labeling is imprecise.
- Corrected: In Mem0 (arXiv:2504.19413), the extraction phase prompts an LLM with two inputs — a running conversation summary (denoted S) plus the last m=10 messages — to produce candidate memories. The update phase then retrieves the s=10 most semantically similar existing memories from the vector store and, via a function-calling ("tool call") interface, has the LLM classify each candidate into exactly one of four operations: ADD, UPDATE, DELETE, or NOOP. Note the paper's notation: uppercase S is the conversation summary (not a count), while lowercase s=10 is the number of similar memories retrieved — so the similar-memory parameter should be written s=10, not "S~10," and the paper reports it as exactly 10 in its experimental configuration.
  - https://arxiv.org/html/2504.19413v1 — Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (primary source, arXiv:2504.19413v1, April 2025): four-op update engine (ADD/UPDATE/DELETE/NOOP), tool-call/function-calling interface, m=10 recent messages, conversation summary S, s=10 similar memories
  - https://arxiv.org/abs/2504.19413 — arXiv abstract page for the Mem0 paper (canonical citation)

**[confirmed]** Mem0 reports on the LOCOMO benchmark a ~26% relative accuracy improvement over OpenAI's built-in memory, ~91% lower p95 latency, and >90% token savings versus full-context.

- Evidence: All three figures are verbatim-confirmed against the primary source — the arXiv paper "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory" (arXiv:2504.19413, Chhikara et al., April 2025). The abstract states: (1) "Mem0 achieves 26% relative improvements in the LLM-as-a-Judge metric over OpenAI" — the 26% figure is a relative improvement on the LLM-as-a-Judge (J) metric specifically, evaluated on the LOCOMO benchmark, not a generic "accuracy" number; (2) "Mem0 attains a 91% lower p95 latency"; (3) "saves more than 90% token cost." Confirmed via the arXiv abstract page (arxiv.org/abs/2504.19413) and the HuggingFace papers mirror, both of which reproduce the abstract identically.

One precision nuance the claim slightly blurs: the 26% improvement is benchmarked against OpenAI's memory, but the 91% lower p95 latency and >90% token savings are stated relative to the FULL-CONTEXT approach (the baseline that stuffs the entire conversation into the prompt) — not against OpenAI. The claim's own wording ("token savings versus full-context") correctly attributes the token figure to full-context, but its phrasing risks implying all three numbers are head-to-head with OpenAI. They are not: latency/token gains are vs. full-context; the accuracy gain is vs. OpenAI. The paper also notes a graph-memory variant ("Mem0-g") scores ~2% higher overall than the base Mem0 configuration. Caveat on provenance: these are first-party metrics reported by Mem0's own team in their paper (the company authored it), not an independent third-party reproduction.
- Corrected: On the LOCOMO benchmark, the Mem0 paper (arXiv:2504.19413) reports that Mem0 achieves a 26% relative improvement over OpenAI's memory on the LLM-as-a-Judge metric, and — relative to the full-context baseline — attains 91% lower p95 latency and >90% token-cost savings. (Note: the 26% gain is vs. OpenAI; the latency and token figures are vs. full-context, not vs. OpenAI. These are first-party results from Mem0's own authors. A graph-memory variant scores ~2% higher overall than base Mem0.)
  - https://arxiv.org/abs/2504.19413 — Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (Chhikara et al., 2025), primary source; abstract states all three figures verbatim
  - https://huggingface.co/papers/2504.19413 — HuggingFace papers mirror of the same abstract, confirming verbatim wording and that latency/token figures are vs. the full-context approach
  - https://arxiv.org/pdf/2504.19413 — full PDF of the primary paper

**[confirmed]** Zep beats MemGPT on the Deep Memory Retrieval (DMR) benchmark by a small margin (94.8% vs 93.4%) and shows larger gains (up to ~18.5% accuracy improvement, ~90% latency reduction) on LongMemEval; its Graphiti engine uses a bi-temporal model tracking both when a fact was valid (T) and when the system learned it (T').

- Evidence: All four load-bearing claims are confirmed verbatim against the primary source — the arXiv paper "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (arXiv:2501.13956, Rasmussen, Paliychuk, Beauvais, Ryan, Chalef; submitted Jan 20, 2025). (1) DMR: Table 1 reports MemGPT 93.4% and Zep 94.8%, both on gpt-4-turbo; paper text: "In the DMR benchmark, which the MemGPT team established as their primary evaluation metric, Zep demonstrates superior performance (94.8% vs 93.4%)." The margin is indeed small (1.4 points). (2) LongMemEval accuracy: "accuracy improvements of up to 18.5%" — the top figure applies to gpt-4o, where the baseline (full-context, entire conversation history given to the LLM) scores 60.2% and Zep scores 71.2% (Table 2). Note: 60.2%→71.2% is an 11-point absolute gain / ~18.3% relative gain; the paper's headline "18.5%" is a relative-improvement framing, and the claim correctly hedged it as "up to ~18.5%." (3) Latency: "Zep reducing response times by approximately 90%" — Table 2 shows full-context gpt-4o at 28.9s vs Zep at 2.58s (and gpt-4o-mini 31.3s vs 3.20s), consistent with ~90%. (4) Bi-temporal model: the paper explicitly tracks two timelines, T (valid time — when a fact held true in the world) and T′ (transaction time — system/database time). Exact definition: "the system tracks four timestamps: t′_created and t′_expired ∈ T′ monitor when facts are created or invalidated in the system, while t_valid and t_invalid ∈ T track the temporal range during which facts held true." And: "While the T′ timeline serves the traditional purpose of database auditing, the T timeline provides an additional dimension for modeling the dynamic nature of conversational data." So the engine actually tracks four timestamps across the two timelines — the claim's "both when a fact was valid (T) and when the system learned it (T')" is an accurate two-timeline summary.
- Corrected: Per the primary source (arXiv:2501.13956, "Zep: A Temporal Knowledge Graph Architecture for Agent Memory," Jan 2025), Zep outperforms MemGPT on the Deep Memory Retrieval (DMR) benchmark by a small margin — 94.8% vs 93.4% (both on gpt-4-turbo). On LongMemEval, Zep improves accuracy by up to ~18.5% (relative) — e.g., gpt-4o full-context 60.2% vs Zep 71.2% — while cutting response latency by ~90% (gpt-4o: 28.9s → 2.58s) versus the full-context baseline. Its Graphiti engine uses a bi-temporal model with two timelines: T (valid time — when a fact held true in the world, via t_valid/t_invalid) and T′ (transaction time — when the system created/invalidated the fact, via t′_created/t′_expired), i.e., four timestamps total across the two timelines.
  - https://arxiv.org/abs/2501.13956
  - https://arxiv.org/html/2501.13956v1
  - https://arxiv.org/pdf/2501.13956
  - https://blog.getzep.com/state-of-the-art-agent-memory/

**[confirmed]** Letta's sleep-time compute spawns two agents -- a primary agent that handles user turns and can read but NOT directly edit memory, and a separate sleep-time agent that asynchronously rewrites/consolidates the memory blocks -- and Letta frames 'sleep' as a compute-timing metaphor, not a neuroscience claim.

- Evidence: Every component of the claim is confirmed by Letta's primary sources (official docs, official blog, and the originating arXiv paper).

(1) TWO AGENTS. Letta's docs: "When you create agents with this type, Letta actually creates two agents under the hood: a primary agent and a sleep-time agent." Enabled via enable_sleeptime: true. (Sleep-time agents | Letta Docs)

(2) PRIMARY READS BUT CANNOT EDIT CORE MEMORY. Verbatim from the docs: "the primary agent is not provided with tools to edit its core memory, which is the memory stored in-context composed of memory blocks. These tools are attached to the sleep-time agent." The primary agent's tools are conversation_search and archival_memory_search (read/search only). The memory blocks are in the primary agent's context window, so it can READ them — DeepWiki notes "All agents within the ManagedGroup have real-time, consistent views of shared blocks" — but it has no core-memory-EDIT tool. This is exactly the counterintuitive detail in the claim, and it is stated verbatim by Letta.

(3) SEPARATE SLEEP-TIME AGENT ASYNCHRONOUSLY REWRITES/CONSOLIDATES BLOCKS. Docs: "you can create special sleep-time agents that share the memory of your primary agents, but run in the background and can modify the memory asynchronously"; "The sleep-time agent runs in the background and can modify the memory blocks asynchronously" and "generates learned context from the conversation history to update the memory blocks of the primary agent." Maintenance tasks include "memory compaction, archive management" — i.e. consolidation/rewriting.

(4) 'SLEEP' IS A COMPUTE-TIMING METAPHOR, NOT A NEUROSCIENCE CLAIM. The term originates in Letta's own paper, arXiv:2504.13171, "Sleep-time Compute: Beyond Inference Scaling at Test-time" (Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez; Letta + UC Berkeley). The abstract defines it as offline/anticipatory computation: it "allows models to 'think' offline about contexts before queries are presented: by anticipating what queries users might ask and pre-computing useful quantities." The Letta blog frames it as "letting models 'think' during downtime... use their 'sleep' time to process information" during "the vast periods when they're not directly engaged with users." Sleep-time agents execute "during a primary agent's 'sleep' periods—when the primary agent is waiting for user input or is otherwise idle." Co-author Charles Packer consistently writes "sleep time" in scare quotes tied to scaling compute and memory — never a biological/neuroscience mechanism claim. The metaphor is unambiguously about WHEN compute is spent (idle/offline between user turns), not about modeling biological sleep.

NUANCE (does not change verdict): The dedicated sleeptime docs page does not itself spell out the etymology of "sleep-time"; that framing is established in the blog and the arXiv paper. And precisely: the primary agent CAN search its recall/archival (external) memory — "read but not edit" applies specifically to the in-context core memory blocks.
- Corrected: Confirmed and tightened: When sleep-time is enabled (enable_sleeptime: true), Letta instantiates two agents that share memory blocks — a primary agent and a sleep-time agent. The primary agent handles user turns and is given conversation_search and archival_memory_search tools; the shared memory blocks sit in its context (so it can read them) but it is "not provided with tools to edit its core memory." The separate sleep-time agent runs in the background and asynchronously rewrites/consolidates those shared memory blocks ("generates learned context from the conversation history to update the memory blocks of the primary agent"; handles compaction/archive maintenance). "Sleep-time" is a compute-timing metaphor — spending compute offline/during idle periods between user turns to pre-compute and reorganize state — originating in Letta's own paper "Sleep-time Compute: Beyond Inference Scaling at Test-time" (arXiv:2504.13171); it is not a neuroscience/biological-sleep claim. (One precision: "read but not edit" applies to the in-context core memory blocks; the primary agent can still search its external recall/archival memory.)
  - https://docs.letta.com/guides/agents/architectures/sleeptime/ — Letta official docs: 'two agents under the hood: a primary agent and a sleep-time agent'; 'the primary agent is not provided with tools to edit its core memory... These tools are attached to the sleep-time agent'; 'run in the background and can modify the memory asynchronously'
  - https://www.letta.com/blog/sleep-time-compute — Letta blog: frames sleep-time as 'think during downtime' / using 'sleep' time during 'the vast periods when they're not directly engaged with users'; links the originating paper arXiv:2504.13171
  - https://arxiv.org/abs/2504.13171 — Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez, 'Sleep-time Compute: Beyond Inference Scaling at Test-time' (Letta + UC Berkeley): defines sleep-time compute as letting models 'think offline about contexts before queries are presented'
  - https://deepwiki.com/letta-ai/letta-python/12.3-sleeptime-and-background-agents — 'All agents within the ManagedGroup have real-time, consistent views of shared blocks'; sleeptime agents run during primary agent's idle 'sleep' periods doing background maintenance (compaction, archive management)
  - https://x.com/charlespacker/status/1914380650993569817 — Letta co-author Charles Packer: '💤 sleep-time compute: make your machines think while they sleep'; 'the concept of sleep-time compute is deeply tied to memory' — compute-scaling framing, scare-quoted 'sleep time', not a neuroscience claim

**[confirmed]** Generative Agents score memory retrieval as a weighted sum of recency, importance, and relevance, where recency is an exponential decay since last access, importance is an LLM-assigned salience score, and relevance is embedding similarity; reflections are generated from roughly the 100 most recent memory records.

- Evidence: Verified directly against the primary source — Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (arXiv:2304.03442). Every element of the claim is confirmed verbatim:

(1) WEIGHTED SUM: The paper gives the retrieval formula as "score = α_recency · recency + α_importance · importance + α_relevance · relevance" and states "In our implementation, all αs are set to 1." A weighted sum — confirmed. (Refinement: before summing, "we normalize the recency, relevance, and importance scores to the range of [0,1] using min-max scaling.")

(2) RECENCY = EXPONENTIAL DECAY SINCE LAST ACCESS: "we treat recency as an exponential decay function over the number of sandbox game hours since the memory was last retrieved" with "decay factor is 0.995." The "since last access/retrieved" definition (not since creation) is confirmed exactly.

(3) IMPORTANCE = LLM-ASSIGNED SALIENCE: "Importance distinguishes mundane from core memories..." scored by "directly asking the language model to output an integer score" on "the scale of 1 to 10, where 1 is purely mundane... and 10 is extremely poignant." Confirmed.

(4) RELEVANCE = EMBEDDING SIMILARITY: "we use the language model to generate an embedding vector... we calculate relevance as the cosine similarity between the memory's embedding vector and the query memory's embedding vector." Confirmed (specifically cosine similarity).

(5) REFLECTIONS FROM ~100 MOST RECENT RECORDS: "We query the large language model with the 100 most recent records in the agent's memory stream" to generate the salient high-level questions that seed reflection. The "100 most recent records" figure is exact. One precision note: that 100-record window is the QUERY INPUT to the reflection step, not the trigger. The trigger is separate: "we generate reflections when the sum of the importance scores for the latest events perceived by the agents exceeds a threshold (150 in our implementation)." The claim's wording ("reflections are generated from roughly the 100 most recent records") is accurate as to the input but glosses over the importance-sum-150 trigger.

Implementation model: gpt-3.5-turbo (ChatGPT, OpenAI 2022).
- Corrected: Generative Agents (Park et al., 2023, arXiv:2304.03442) score each memory at retrieval time as a weighted sum of three components — recency, importance, and relevance — each min-max normalized to [0,1] and then combined with equal weights (all α coefficients set to 1 in the reference implementation): score = α_recency·recency + α_importance·importance + α_relevance·relevance. Recency is an exponential decay (factor 0.995) over the number of sandbox hours since the memory was last RETRIEVED (i.e., decay since last access, not since creation). Importance is an LLM-assigned integer salience score on a 1–10 scale (1 = mundane, 10 = poignant). Relevance is the cosine similarity between the memory's embedding and the query's embedding. Reflection is TRIGGERED when the summed importance of the latest perceived events exceeds 150; once triggered, the reflection step QUERIES the LLM with the 100 most recent records in the memory stream to produce salient high-level questions, which are then used as retrieval queries to gather the memories that ground each reflection. (The "~100 most recent records" figure is exact and describes the reflection query input, not the trigger condition.) Reference implementation used gpt-3.5-turbo.
  - https://arxiv.org/pdf/2304.03442 — Park, Joon Sung, et al. "Generative Agents: Interactive Simulacra of Human Behavior." arXiv:2304.03442 (UIST 2023). Primary source.
  - https://ar5iv.labs.arxiv.org/html/2304.03442 — ar5iv HTML rendering of the same paper; used to extract verbatim quotes from the Memory/Retrieval and Reflection sections (score formula, α=1, decay 0.995 since last retrieved, importance 1–10, cosine relevance, [0,1] min-max normalization, 100-most-recent-records reflection query, importance-sum-150 trigger, gpt-3.5-turbo).
  - https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763 — ACM published version of the paper (UIST '23), corroborating source of record.

**[confirmed]** HippoRAG is explicitly based on the hippocampal indexing theory and orchestrates an LLM-built knowledge graph with the Personalized PageRank algorithm to mimic the neocortex/hippocampus division of labor, reporting up to ~20% improvement over state-of-the-art RAG on multi-hop QA.

- Evidence: Every element of the claim matches the primary source verbatim. The paper is "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models" (arXiv:2405.14831, Gutiérrez, Shu, Gu, Yasunaga & Su; NeurIPS 2024). The abstract states the framework is "inspired by the hippocampal indexing theory of human long-term memory" and "orchestrates LLMs, knowledge graphs, and the Personalized PageRank algorithm" to "mimic the different roles of neocortex and hippocampus in human memory." It reports the method "outperforms the state-of-the-art methods remarkably, by up to 20%." All four phrases are direct, exact quotes — no paraphrase drift.

Mechanism confirmed: an LLM converts passages into open KG triples for offline indexing (the "artificial neocortex" + parahippocampal step); parahippocampal retrieval encoders detect synonymy and add edges between similar noun phrases; Personalized PageRank (NOT generic graph search) runs over the KG to "explore KG paths and identify relevant subgraphs, essentially performing multi-hop reasoning in a single retrieval step" — the artificial-hippocampus pattern-completion role. So the named algorithm in the claim is exactly right.

One precision point on the 20% figure: it is a RETRIEVAL metric (recall@2/recall@5), not QA exact-match/F1. The "up to 20%" refers specifically to the 2WikiMultiHopQA benchmark; on MuSiQue the gain is much smaller (~3 points). So "20% on multi-hop QA" is the paper's own framing but the headline number is benchmark-specific (2Wiki) and retrieval-stage. The paper also reports single-step HippoRAG matches/beats iterative IRCoT while being 10–20x cheaper and 6–13x faster.
- Corrected: HippoRAG (Gutiérrez et al., NeurIPS 2024; arXiv:2405.14831) is explicitly "inspired by the hippocampal indexing theory of human long-term memory." It uses an LLM to build an open knowledge graph from passages and applies the Personalized PageRank algorithm over that graph to perform single-step multi-hop retrieval (pattern completion), mimicking the division of labor between neocortex (LLM/encoders) and hippocampus (the PPR index). It reports retrieval improvements of up to ~20% over state-of-the-art RAG on multi-hop QA — but that headline figure is benchmark- and stage-specific: ~20 points on 2WikiMultiHopQA versus only ~3 on MuSiQue, measured as retrieval recall (recall@2/@5), not end-task QA accuracy. The specific algorithm (Personalized PageRank) and the named theory in the original claim are accurate.
  - https://arxiv.org/abs/2405.14831
  - https://arxiv.org/html/2405.14831v1
  - https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf
  - https://papers.nips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html
  - https://openreview.net/forum?id=hkujvAPVsg

**[confirmed]** HippoRAG runs the Personalized PageRank algorithm over an LLM-built knowledge graph (the 'artificial hippocampal index'), using query concepts as seed nodes, to emulate hippocampal pattern completion and achieve single-step multi-hop retrieval.

- Evidence: Every element of the claim is corroborated verbatim by the primary source — HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models (Gutiérrez, Shu, Gu, Yasunaga, Su; OSU + Stanford; NeurIPS'24; arXiv:2405.14831).

(1) LLM-built KG = artificial hippocampal index. The paper: "our artificial hippocampal index as this open KG, which is built on the whole retrieval corpus passage-by-passage." The full tripartite brain mapping is explicit: an LLM is the "artificial neocortex, to extract knowledge graph (KG) triples" (via OpenIE); the open KG is the "artificial hippocampal index"; and "to connect both components as is done by the parahippocampal regions, we use off-the-shelf dense encoders." So the claim's neocortex/hippocampus mapping is exact.

(2) Personalized PageRank over the KG: "we leverage the Personalized PageRank (PPR) algorithm ... to integrate information across passages for retrieval." PPR is described as "a version of PageRank that distributes probability across a graph only through a set of user-defined source nodes."

(3) Query concepts as seed nodes: confirmed, with a precision nuance. The seeds are the LLM-extracted query NAMED ENTITIES matched to KG nodes — "using a personalized probability distribution defined over N, in which each query node has equal probability and all other nodes have a probability of zero." These seeds are additionally weighted by node specificity (an IDF-like signal): "Node specificity is used in retrieval by multiplying each query node probability with s_i before PPR." So 'query concepts' is a fair paraphrase of 'query entity nodes,' not a misstatement.

(4) PPR as hippocampal pattern completion: "This constraint allows us to bias the PPR output only towards the set of query nodes, just as the hippocampus extracts associated signals from specific partial cues." The mechanism is explicitly framed as the computational analog of hippocampal pattern completion.

(5) Single-step multi-hop retrieval: "PPR enables HippoRAG to explore KG paths and identify relevant subgraphs, essentially performing multi-hop reasoning in a single retrieval step." The abstract independently confirms: "Single-step retrieval with HippoRAG achieves comparable or better performance than iterative retrieval like IRCoT while being 10-20 times cheaper and 6-13 times faster," outperforming SOTA RAG on multi-hop QA by up to 20%.

No element of the claim is misremembered or overstated. The only refinement worth making is that the PPR seeds are precisely query NAMED ENTITIES (LLM-extracted, dense-matched to KG nodes, IDF-weighted by node specificity), which 'query concepts' approximates correctly.
- Corrected: HippoRAG (Gutiérrez et al., NeurIPS'24, arXiv:2405.14831) runs Personalized PageRank over a schemaless, LLM-built open knowledge graph it calls the 'artificial hippocampal index' (with the LLM acting as the 'artificial neocortex' for OpenIE triple extraction and dense retrieval encoders standing in for the parahippocampal regions). At query time it extracts named entities from the query, matches them to KG nodes, and uses those nodes — weighted by node specificity (an IDF-like signal) — as the PPR seed/reset distribution. Biasing PPR toward these seed nodes is presented as the computational analog of hippocampal pattern completion (extracting associated signals from partial cues), letting the system perform multi-hop reasoning in a single retrieval step. The original claim is accurate; the only tightening is that the PPR seeds are query named-entity nodes (IDF-weighted), which 'query concepts' paraphrases correctly.
  - https://arxiv.org/abs/2405.14831
  - https://arxiv.org/html/2405.14831v1
  - https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf
  - https://github.com/OSU-NLP-Group/HippoRAG
  - https://openreview.net/forum?id=hkujvAPVsg

**[confirmed]** HippoRAG explicitly maps the LLM/OpenIE to the neocortex, the schemaless knowledge graph to the hippocampal index, and the dense retrieval encoder to the parahippocampal regions, citing Teyler & DiScenna's hippocampal memory indexing theory.

- Evidence: All four elements of the claim are verified verbatim against the primary source, the HippoRAG paper (Gutiérrez et al., "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models," arXiv:2405.14831, NeurIPS 2024). Section 2.2 (Methodology):

(1) LLM/OpenIE → neocortex: "Our offline indexing phase, analogous to memory encoding, starts by leveraging a strong instruction-tuned LLM, our artificial neocortex, to extract knowledge graph (KG) triples. The KG is schemaless and this process is known as open information extraction (OpenIE)." This single sentence ties the LLM to the neocortex AND describes the KG as schemaless via OpenIE — directly supporting the "LLM/OpenIE → neocortex" portion of the claim.

(2) Schemaless KG → hippocampal index: "It is therefore natural to define our artificial hippocampal index as this open KG, which is built on the whole retrieval corpus passage-by-passage." The KG is described as both "schemaless" and "open," and is explicitly cast as the "artificial hippocampal index." Personalized PageRank over this KG is described as the "synthetic hippocampus" search process.

(3) Dense retrieval encoder → parahippocampal regions (PHR): "Finally, to connect both components as is done by the parahippocampal regions, we use off-the-shelf dense encoders fine-tuned for retrieval (retrieval encoders). These retrieval encoders provide additional edges between similar but not identical noun phrases within this KG to aid in downstream pattern completion." The encoder is explicitly mapped to the PHR.

(4) Citation of Teyler & DiScenna: "The hippocampal memory indexing theory [75], a well-established theory of human long-term memory, offers one plausible explanation for this remarkable ability. Teyler and Discenna [75] propose..." Reference [75] is Teyler and DiScenna (1986), "The hippocampal memory indexing theory." The framework is named in the title as inspired by this theory.

The paper's own component summary names "the artificial neocortex (LLM), the parahippocampal region (PHR encoder), and the artificial hippocampus (open KG)." This is the strongest literal neuroscience-to-architecture correspondence claim in the survey, and it holds exactly as stated.

One nuance (does not change the verdict): the PHR/retrieval-encoder mapping is functionally about adding synonymy edges between similar-but-not-identical noun phrases (bridging neocortex and hippocampal index), a detail the claim omits but does not contradict. The paper does not present one single consolidated sentence naming all three mappings at once — the three mappings appear across adjacent passages in Section 2.2 plus the component overview — but each individual mapping is stated explicitly and verbatim.
- Corrected: HippoRAG (Gutiérrez et al., arXiv:2405.14831, NeurIPS 2024) explicitly maps its three components to brain regions per Teyler & DiScenna's (1986) hippocampal memory indexing theory: the instruction-tuned LLM performing OpenIE is the "artificial neocortex"; the resulting schemaless/open knowledge graph is the "artificial hippocampal index" (searched via Personalized PageRank as the "synthetic hippocampus"); and off-the-shelf dense retrieval encoders play the role of the "parahippocampal regions," connecting the two by adding synonymy edges between similar-but-not-identical noun phrases. The component-to-region correspondence is stated verbatim in the paper, though across several adjacent sentences in Section 2.2 rather than a single consolidated statement.
  - https://arxiv.org/abs/2405.14831
  - https://arxiv.org/html/2405.14831v2
  - https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf
  - Teyler, T.J. & DiScenna, P. (1986), 'The hippocampal memory indexing theory', Behavioral Neuroscience 100(2):147-154 — HippoRAG reference [75]

**[confirmed]** Generative Agents (Park et al.) score each memory as a weighted sum of recency, importance, and relevance with all alpha weights = 1: recency is exponential decay with factor 0.995, importance is an LLM poignancy rating 1-10, and relevance is embedding cosine similarity.

- Evidence: The primary source — Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (arXiv:2304.03442), Section 4.1 "Memory Retrieval" — confirms every element of the claim verbatim:

(1) FINAL FORMULA + ALPHAS: The paper states "score = αrecency · recency + αimportance · importance + αrelevance · relevance. In our implementation, all αs are set to 1." This confirms the weighted-sum structure and that all three alpha weights equal 1.

(2) RECENCY = 0.995 DECAY: "[recency is modeled as an] exponential decay function over the number of sandbox game hours since the memory was last retrieved. Our decay factor is 0.995." Confirms both the exponential-decay form and the exact factor 0.995. (Note: it decays over sandbox-game-hours-since-last-RETRIEVAL, not since creation.)

(3) IMPORTANCE = LLM POIGNANCY 1-10: The importance score is obtained by directly prompting the LLM: "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory." Confirms the 1-10 integer scale and the "poignancy" framing.

(4) RELEVANCE = EMBEDDING COSINE: "we use the language model to generate an embedding vector of the text description of each memory. Then, we calculate relevance as the cosine similarity between the memory's embedding vector and the query memory's embedding vector." Confirms cosine similarity over embeddings.

Corroboration: The official implementation repo (github.com/joonspk-research/generative_agents) and the ACM-published version (DOI 10.1145/3586183.3606763) match the same constants. Multiple independent secondary sources report the identical 0.995 decay factor and 1-10 poignancy scale.

One precision detail (does not contradict the claim): before the weighted sum, each of the three component scores is min-max normalized to the range [0, 1]. The claim's wording is accurate as stated; this normalization step is an implementation detail of how the three terms are combined.
- Corrected: CONFIRMED as stated. Generative Agents (Park et al., 2023, "Generative Agents: Interactive Simulacra of Human Behavior," arXiv:2304.03442, §4.1) compute a memory's retrieval score as a weighted sum of three terms — score = α_recency·recency + α_importance·importance + α_relevance·relevance — with all α weights set to 1 in their implementation. Recency is an exponential decay (factor 0.995) over sandbox-game-hours since the memory was last retrieved; importance is an LLM-generated "poignancy" rating on an integer 1-10 scale (1 = mundane, 10 = extremely poignant); relevance is the cosine similarity between the memory's embedding vector and the query's embedding vector. One precision addendum: each of the three component scores is min-max normalized to [0, 1] before the (unit-weighted) sum.
  - Park et al., Generative Agents: Interactive Simulacra of Human Behavior — arXiv:2304.03442 (primary source, §4.1 Memory Retrieval): https://ar5iv.labs.arxiv.org/html/2304.03442
  - arXiv PDF of the paper: https://arxiv.org/pdf/2304.03442
  - ACM-published version, DOI 10.1145/3586183.3606763: https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763
  - Official implementation repository: https://github.com/joonspk-research/generative_agents

**[confirmed]** MemoryBank applies the Ebbinghaus forgetting curve as R = e^(-t/S), initializes memory strength S=1, and on each recall increments S by 1 and resets t to 0, so recalled memories decay more slowly.

- Evidence: All four sub-claims are confirmed verbatim against the primary source (Zhong, Guo, Gan, Yang, Wang, "MemoryBank: Enhancing Large Language Models with Long-Term Memory," arXiv:2305.10250, AAAI 2024).

(1) EQUATION — The paper states the forgetting curve verbatim as: "The Ebbinghaus forgetting curve is expressed using an exponential decay model: R = e^(-t/S)," where "R is the memory retention, or what fraction of the information can be retained" and "t is the time elapsed since learning the information." This exactly matches the claimed R = e^(-t/S). (S is the memory strength.)

(2) INITIALIZATION S=1 — Verbatim: "we model S as a discrete value and initialize it with 1 upon its first mention in a conversation." Confirms S=1 at creation. Note the precise wording: S is initialized to 1 upon a memory's FIRST MENTION, and it is a discrete (integer) value.

(3) RECALL RULE (S+1, t=0) — Verbatim: "When a memory item is recalled during conversations, it will persist longer in memory. We increase S by 1 and reset t to 0, hence forget it with a lower probability." This confirms both that S is incremented by 1 and that t is reset to 0 on each recall. ("increase" = the claim's "increment.")

(4) "DECAY MORE SLOWLY" — Directly supported by the paper's own causal phrasing: increasing S and resetting t makes the system "forget it with a lower probability" and the item "persist longer in memory." Mechanically, a larger S flattens R = e^(-t/S), so retention decays more slowly — consistent with the claim's conclusion.

The abstract independently corroborates the framing: "MemoryBank incorporates a memory updating mechanism, inspired by the Ebbinghaus Forgetting Curve theory, which permits the AI to forget and reinforce memory based on time elapsed and the relative significance of the memory."

Sourcing note: the ar5iv HTML rendering of arXiv:2305.10250 yielded the verbatim equation and quotes; the raw arXiv PDF returned binary/non-extractable text and was not usable for quoting, but the ar5iv HTML is a faithful render of the same paper.
- Corrected: MemoryBank (Zhong et al., arXiv:2305.10250, AAAI 2024) models the Ebbinghaus forgetting curve as an exponential-decay retention R = e^(-t/S), where R is memory retention, t is time elapsed since learning, and S is the memory strength. S is a discrete (integer) value initialized to 1 upon a memory's first mention. Each time a memory item is recalled in conversation, S is increased by 1 and t is reset to 0, which lowers its forgetting probability and makes it persist longer (a flatter decay curve) — so frequently recalled memories decay more slowly. The claim is accurate as stated; the only added precision is that S is initialized to 1 on first mention and is explicitly a discrete value, and the strengthening is driven by recall frequency (and, per the abstract, memory significance) rather than the classic spaced-repetition spacing schedule.
  - https://ar5iv.labs.arxiv.org/html/2305.10250 (primary source, full text with verbatim equation R = e^(-t/S), S=1 init, S+1 / t=0 recall rule)
  - https://arxiv.org/abs/2305.10250 (primary source abstract: memory updating mechanism inspired by Ebbinghaus Forgetting Curve, forget/reinforce based on time elapsed and significance)
  - https://ojs.aaai.org/index.php/AAAI/article/view/29946 (peer-reviewed AAAI 2024 publication of the same paper)

**[confirmed]** EM-LLM segments the token stream into episodic events using Bayesian surprise (a dynamic threshold T = mu + gamma*sigma over a moving surprise window), refines boundaries with graph-theoretic metrics (modularity/conductance), and retrieves via a similarity buffer plus a temporal-contiguity buffer.

- Evidence: Every component of the claim is confirmed by the primary source (arXiv 2407.09450, "Human-inspired Episodic Memory for Infinite Context LLMs," Fountas et al., ICLR 2025) and the official repo (github.com/em-llm/EM-LLM-model).

(1) SURPRISE SEGMENTATION + THRESHOLD FORMULA — CONFIRMED EXACTLY. The paper defines surprise as the negative log-likelihood of the ground-truth token under the autoregressive model: -log P(x_t | x_1,...,x_{t-1}; theta). A boundary is placed where this exceeds a dynamic threshold T = mu_{t-tau:t} + gamma * sigma_{t-tau:t}, i.e. the rolling mean plus gamma times the rolling standard deviation of surprise over a moving window of size/offset tau, with gamma a user-controlled scaling factor "minimizing the need for manual tuning while maintaining control over threshold sensitivity via gamma." The repo config exposes this directly as `surprisal_threshold_gamma` ("the standard-deviation scaling factor"). The claimed formula T = mu + gamma*sigma over a moving surprise window is verbatim correct.

(2) GRAPH-THEORETIC BOUNDARY REFINEMENT — CONFIRMED. After initial surprise boundaries, EM-LLM treats the pairwise similarity matrix of attention keys as a graph adjacency matrix and adjusts each boundary to the position that optimizes a graph-clustering metric: either MAXIMIZING MODULARITY or MINIMIZING CONDUCTANCE. Both metrics named in the claim are the two production options (repo notes a third, intra_inter_sim, "doesn't work well"). Refinement runs at O(nm).

(3) TWO-STAGE RETRIEVAL — CONFIRMED. Retrieval combines a similarity buffer (k_s events via k-NN / dot-product similarity between the current query and each event's representative tokens) and a contiguity buffer (k_c events maintaining temporally neighboring events, a queue preserving temporal context); total k = k_s + k_c. Repo config: `contiguity_buffer_size`. This matches "similarity buffer plus a temporal-contiguity buffer."

(4) HUMAN VALIDATION — CONFIRMED. Using human-annotated podcast transcripts, the paper shows surprise-based methods identify boundaries closest to human-perceived events (Fig. 4B), with "surprise-only segmentation achiev[ing] very similar results to humans."

One precision nuance (does NOT change the verdict): the paper's "surprise" is operationalized as negative log-likelihood (token-level information content / Shannon surprise under the model). The paper frames this "in Bayesian terms," but it is not the classical KL-divergence form of Bayesian surprise (Itti & Baldi). Calling it "Bayesian surprise" follows the paper's own language and is acceptable; if maximal rigor is wanted, "model-derived surprise (negative log-likelihood), termed Bayesian surprise in the paper" is the tightest description. Also: one secondary summary loosely called sigma the "variance," but it is the standard deviation (gamma scales sigma, the std-dev) — the claim's "*sigma" is correct as written.
- Corrected: EM-LLM (Fountas et al., ICLR 2025; arXiv:2407.09450) segments the token stream online into episodic events by placing a boundary wherever per-token surprise — the negative log-likelihood of the ground-truth token, -log P(x_t | x_<t; theta), which the paper frames as Bayesian surprise — exceeds a dynamic threshold T = mu + gamma*sigma, where mu and sigma are the mean and standard deviation of surprise over a moving window (offset tau) and gamma is a user-set scaling factor. It then refines these initial boundaries with graph-theoretic metrics computed on the attention-key similarity graph, repositioning each boundary to maximize modularity or minimize conductance. At inference it retrieves events through a two-stage process: a similarity buffer (k-NN over event representative tokens vs. the current query) plus a temporal-contiguity buffer (neighboring events that preserve temporal context). The paper empirically validates that its surprise-based boundaries align closely with human-perceived event boundaries on annotated transcripts.
  - https://arxiv.org/abs/2407.09450
  - https://arxiv.org/html/2407.09450v3
  - https://openreview.net/forum?id=BI2int5SAC
  - https://github.com/em-llm/EM-LLM-model
  - https://em-llm.github.io/

**[confirmed]** Larimar implements Complementary Learning Systems theory by coupling a fast hippocampal episodic-memory module (Kanerva-Machine-style, one-shot writes) with a slow neocortical frozen LLM, and supports selective forgetting via negative-coefficient memory updates.

- Evidence: All four sub-claims are verified against the primary source (Das et al., "Larimar: Large Language Models with Episodic Memory Control," arXiv:2403.11901, ICML 2024).

(1) CLS / fast-hippocampus + slow-neocortex framing — CONFIRMED, verbatim: "a hippocampal fast-learning system records samples as episodic memory, and a neocortical slow learning system (the LLM) learns summary statistics of the input distribution as semantic memory." The architecture is explicitly described as "inspired by complementary learning mechanisms in the brain," with the hippocampus acting as a "generative associative network."

(2) Kanerva-Machine lineage — CONFIRMED. The paper uses a memory "similar in spirit to the Kanerva Machine," adopting the deterministic Generative Pseudoinverse Memory (GPM) of Pham et al. (2021), which "reformulates the Bayesian updates of memory and address proposed in Kanerva Machine as finding least-square solutions to linear systems." So it is Kanerva-Machine-derived/style, but specifically via the GPM reformulation, not a direct reimplementation of the original probabilistic Kanerva Machine.

(3) One-shot writes — CONFIRMED, verbatim: "the posterior memory M is updated in one-shot by solving a minimization problem" via matrix pseudo-inverses / least-squares. The abstract calls these "dynamic, one-shot updates of knowledge without ... re-training or fine-tuning."

(4) Selective forgetting via negative-coefficient memory update — CONFIRMED, verbatim: "When forgetting encodings which were previously written to memory with α_i^write=1 at any i_write<i, we use α_i=−1" (Eqs. 3-4). Forgetting reuses the write equation with the write coefficient flipped to −1, removing the fact while keeping memory as a least-squares solution. This is "training-free selective fact forgetting."

Additional confirmation: the base LLM/decoder is frozen during memory operations ("no gradient updates to the LLM"), which is what distinguishes this from gradient-based model editing — exactly the "slow neocortical frozen LLM" in the claim. Reported speed-ups of 8-10x over baselines.
- Corrected: Larimar (Das et al., ICML 2024; arXiv:2403.11901) follows the Complementary Learning Systems (CLS) view: a fast "hippocampal" episodic-memory module records samples one-shot while a frozen "neocortical" LLM supplies slow-learned semantic statistics, conditioned at inference by the updated memory (no gradient updates to the LLM). The memory is "similar in spirit to the Kanerva Machine," realized via the deterministic Generative Pseudoinverse Memory (GPM) of Pham et al. (2021), which recasts the Kanerva Machine's Bayesian memory/address updates as least-squares solutions; writes are one-shot via pseudo-inverse. Selective fact forgetting is training-free: a previously written encoding (write coefficient α=1) is removed by re-applying the update with a negative write coefficient α=−1 (Eqs. 3-4). One nuance worth tightening: the Kanerva link is "Kanerva-Machine-style via the GPM least-squares reformulation," not a direct use of the original probabilistic Kanerva Machine.
  - https://arxiv.org/abs/2403.11901
  - https://arxiv.org/html/2403.11901v2
  - https://proceedings.mlr.press/v235/das24a.html
  - https://dl.acm.org/doi/10.5555/3692070.3692472
  - https://huggingface.co/papers/2403.11901

</details>
