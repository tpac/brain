# Memory Systems Research — Reference Document

Compiled 2026-03-31 from 6 parallel deep-research agents. This document is a permanent reference for the brain's architecture evolution. It eliminates the need to re-research these topics.

**Search terms used** (for future reference — don't re-search these, read this doc):
- "HippoRAG personalized pagerank retrieval", "HippoRAG v2 ICML 2025"
- "Titans learning to memorize test time Google", "MIRAS memory attention learning"
- "Complementary learning systems theory CLS memory", "memory consolidation sleep replay"
- "PAM predictive associative memory JEPA 2026", "modern Hopfield networks attention"
- "ColBERT MaxSim late interaction", "RAPTOR recursive tree retrieval"
- "Memory-R1 RL memory management", "MemOS memory operating system"
- "Z-tokens discrete semantic compression", "Doc-to-LoRA hypernetwork"
- "prediction error dopamine memory encoding", "GANE model arousal encoding selectivity"
- "synaptic tagging and capture", "hippocampal indexing theory"
- "episodic semantic memory gradient continuum", "Bayesian Hebbian decontextualization"
- "forgetting as feature AI memory FiFA benchmark", "Ebbinghaus curve neural networks"
- "GraphRAG community detection Leiden", "LightRAG", "DynamicRAG RL reranker"
- "attention sinks functional", "retrieval heads DuoAttention", "KV cache compression MLA"
- "hippocampus pattern separation pattern completion", "cue diagnosticity overload"

---

## I. The Theoretical Foundation

### MIRAS: Memory = Attention = Learning (Google, Dec 2025)

**The unifying proof:** Every major sequence model (Transformers, Mamba, DeltaNet, xLSTM, Titans) is doing online optimization over an associative memory. They differ in four design choices:
1. What structure stores information (vector, matrix, deep MLP)
2. What objective drives updates (associative loss, prediction loss)
3. How forgetting works (regularization type)
4. What algorithm does the optimization (SGD, momentum, etc.)

**Why it matters for us:** Our brain is also an associative memory with an update rule (encoding), a forgetting mechanism (Ebbinghaus decay), and a retrieval mechanism (cosine + graph). We're in the same design space.

Source: [Google blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

### Key-Value Memory = Hippocampus/Neocortex (Gershman, Fiete & Irie, 2025)

Transformer QKV attention maps directly to biological memory:
- **Hippocampus** = Keys (sparse, optimized for discriminability)
- **Neocortex** = Values (distributed, optimized for fidelity)
- Retrieval = hippocampal activation reinstates cortical patterns

Tip-of-tongue = key matches but value retrieval fails. Familiarity without recollection = key match detected, no full value.

Source: [Neuron paper](https://arxiv.org/abs/2501.02950)

### Complementary Learning Systems (CLS) — The Two-System Requirement

**Core claim (McClelland 1995, updated 2020, 2023):** You CANNOT have fast specific learning AND slow structural extraction in one system. You need:
1. **Fast system** (hippocampus): Rapid learning, sparse representations, encodes episodes in one exposure
2. **Slow system** (neocortex): Gradual extraction of statistical structure through interleaved replay

**2023 Nature Neuroscience update:** Unregulated consolidation HARMS generalization. Memories should only consolidate when doing so helps.

**2020 update:** Schema-consistent info CAN be learned rapidly by neocortex — speed depends on consistency with existing knowledge.

Sources: [Original 1995](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf), [2023 update](https://www.nature.com/articles/s41593-023-01382-9)

---

## II. Encoding: What to Remember

### Prediction Error Routes the Encoding Decision

Large PE → encode new memory. Small PE → update existing. PE magnitude is a routing signal.

- Dopamine encodes PE across multiple feature dimensions (identity, timing, magnitude), not just reward
- **Bayesian surprise > Shannon surprise** for memory. Random noise has max Shannon info but zero Bayesian surprise. Memory cares about "how much should I change my beliefs"
- Different neural signatures: P3a (frontal) = Bayesian surprise, P3b (parietal) = Shannon surprise

Sources: [Costa 2025](https://www.science.org/doi/10.1126/sciadv.adq9684), [Baldi & Itti 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2860069/)

### The GANE Model: Arousal Creates Selectivity (Not Enhancement)

The amygdala doesn't enhance everything under arousal — it's **winner-take-more**:
1. High-priority representations generate high local glutamate
2. Under arousal, NE from locus coeruleus creates local "hotspots"
3. Hotspots: enhanced encoding. Away from hotspots: SUPPRESSED encoding
4. Works for ANY prioritized stimuli, not just emotional ones

**Implication:** When something important happens (correction, breakthrough, error), encoding should strengthen that moment AND suppress encoding of nearby mundane events.

### Synaptic Tag-and-Capture: Temporal Windows

1. Many events acquire weak "learning tags"
2. A salient event generates plasticity-related proteins (PRPs)
3. PRPs can be captured by tagged synapses within ~1-2 hours
4. Salient events retroactively rescue nearby weak memories

**Tags persist up to 9 hours** (2025 finding, was thought to be 90 min).

**Implication:** An important event at turn T should strengthen encoding of turns T-5 to T+5.

### Schema Theory: Skip the Expected

- Schema-consistent info → fast neocortical integration, can bypass hippocampus
- Schema-VIOLATING info → full hippocampal encoding required
- Only surprises, corrections, and violations justify new nodes

### Encode Decision Function

**ENCODE STRONGLY when:** High prediction error, emotional arousal, novelty, schema violation, self-relevance, social significance, goal relevance.

**SKIP/FORGET when:** Low PE, low arousal, high familiarity, schema-consistent, low self/social relevance, cue overloaded.

### Titans: Surprise-Gated Memory Writes (Google, Jan 2025)

Memory IS an MLP whose weights store associations. Update rule:
```
Loss: ||M(k_t) - v_t||^2
Surprise: S_t = momentum * S_{t-1} - learning_rate * gradient
Update: M_t = (1 - decay) * M_{t-1} + S_t
```
High loss = surprising = big update. Momentum propagates surprise window. Scales to 2M+ context.

Source: [Paper](https://arxiv.org/abs/2501.00663)

---

## III. Retrieval: How to Activate

### HippoRAG: Activation via Personalized PageRank (NeurIPS 2024)

**The mechanism:**
1. Extract query entities via LLM
2. Match to KG nodes via embedding similarity → seed set
3. Multiply seeds by **node specificity** (1/passage_count — local, no global stats needed)
4. Run PPR (damping=0.5) from seeds through graph
5. Probability flows through edges to connected nodes
6. Multi-hop reasoning in a SINGLE retrieval step

**HippoRAG v2 (ICML 2025):** Added passage nodes (dual-node KG), query-to-triple matching (+12.5% recall), LLM-based triple filtering. MuSiQue F1: 44.8→51.9.

Sources: [v1](https://arxiv.org/abs/2405.14831), [v2](https://arxiv.org/abs/2502.14802)

### PAM: Similarity AND Association (Feb 2026)

**The fundamental distinction nobody else makes:**
- RAG finds memories SIMILAR to the query
- Biological memory retrieves things that CO-OCCURRED with the query context
- These are different signals. "Eating lunch" and "the idea I had at lunch" aren't similar, but they're associated

**Implementation:** Two JEPA predictors:
- Outward: predicts future states (semantic/world model)
- Inward: predicts associatively reachable past states (episodic links)

**Key finding:** Episodic specificity emerges from the INTERSECTION of similarity AND association. Cross-boundary pairs where cosine is at chance (0.503): association AUC = 0.849.

Source: [Paper](https://arxiv.org/abs/2602.11322), [Code](https://github.com/EridosAI/PAM-Benchmark)

### Pattern Completion vs Search

| | Pattern Completion | Search |
|---|---|---|
| Mechanism | Attractor dynamics, partial cue → full pattern | Active query against representations |
| Speed | O(1) | O(log n) to O(n) |
| Brain analog | CA3 recurrent collaterals | Prefrontal-guided retrieval |
| AI analog | Hopfield networks | RAG, vector search, KGs |
| Limitation | No inter-memory structure | No automatic activation |

**You need both.** Pattern completion within a neighborhood. Search to find which neighborhood.

### Modern Hopfield Networks = Attention

The update `q_new = X^T softmax(β*X*q)` IS transformer attention. Stores exponentially many patterns, retrieves in one step. Three fixed-point types: global averaging (early layers), metastable/subset (middle), single-pattern (late layers).

Source: [Paper](https://arxiv.org/abs/2008.02217), [Code](https://github.com/ml-jku/hopfield-layers)

### Cue Diagnosticity: The Overload Problem

A cue linked to 1 memory = powerful. A cue linked to 100 memories = useless. **"Daemon" linked to 50 brain nodes = zero diagnostic power.**

Fix: optimize retrieval keys for discriminability. "Brain daemon TCP port" is diagnostic. "Daemon" alone is not.

The hippocampus solves this via:
- **Pattern separation** (dentate gyrus): orthogonalize similar inputs to create distinct indices
- **Pattern completion** (CA3): reconstruct full memory from partial cue
- **Prefrontal control**: top-down signals constrain which memory is activated

### Hippocampal Indexing Theory

The hippocampus stores INDICES (pointers), not content. Reactivation of the index triggers pattern completion across neocortex.

**Implication for us:** Nodes should be optimized as indices (title + keywords + embeddings = discriminative pointer). Content is the value retrieved after activation, not the retrieval key.

### Routing, Not Fusion

No single retrieval method wins all queries:
- Entity-centric → graph traversal (PPR)
- Semantic → embeddings
- Rare terms → keyword/BM25
- Thematic/pattern → community summaries

The state of the art is **learned routing** (LTRR, R1-Router, SkewRoute) that selects retrieval method per query.

---

## IV. Consolidation: From Episodic to Semantic

### The Episodic-Semantic Gradient (Gentry & Buckner 2024)

The binary is dead. Memory has a **dynamic life cycle** — it's a continuum. Fresh = full context, who said what, when. Over time: context strips away, core principle remains. Not degradation — functional transformation.

### Bayesian-Hebbian Decontextualization

The mechanism: when an item appears across multiple contexts, co-activation traces decay between exposures, gradually weakening item-context associations. Context recall drops from ~100% (1 context) to ~25% (4 contexts). This IS semanticization — emerging from synaptic learning rules.

Source: [Spiking network model](https://pmc.ncbi.nlm.nih.gov/articles/PMC9347313/)

### Sleep as Consolidation

**SWS (slow-wave sleep):** Replay via coupled slow oscillations + spindles + sharp-wave ripples. Hippocampal-to-cortical transfer.
**REM:** Recalibration, representational transformation. Higher REM/SWS ratio → more item-level reduction, category-level preservation.

**What gets selected for replay:** Reward-associated, novel, schema-congruent, emotionally salient, incompletely learned.

### Generative Replay > Exact Replay

Networks "imagining" new category instances from learned distributions is AS EFFECTIVE as replaying exact episodes — and produces category-level knowledge from episodes. Lower-resolution generated samples strip exemplar details while preserving category features.

Source: [Cerebral Cortex 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758580/)

### Concept Formation: Exemplar → Prototype

Early learning: exemplar-based (remember specific instances). With experience: shifts to prototype-based (abstract averages). The brain transitions from "I remember that specific conversation about X" to "my general understanding of X."

### Existing AI Implementations

- **EVOLVE-MEM:** Hierarchical abstraction. L0=raw, L1=clusters (floor(n/5)), L2=principles
- **MemGPT/Letta:** OS-inspired tiers with recursive summarization. 2025: "sleep-time agents" for async consolidation during idle
- **FiFA benchmark:** Hybrid forgetting policies (priority decay + reflection-summary) beat remember-everything

---

## V. Forgetting: Strategic Pruning

### Forgetting Is Design, Not Failure

**Context pollution:** Too much irrelevant info degrades retrieval more than losing some info. Remember-everything strategies consistently underperform hybrid approaches.

**The asymmetry:** In biology, forgetting is default, retention is hard. In AI, retention is default, forgetting must be engineered.

### Ebbinghaus in Neural Networks

MLPs exhibit human-like forgetting curves. Scheduled reviews based on retention metric mitigate catastrophic forgetting.

**We already have this** — our Ebbinghaus decay, stability scores, access_count-based floors.

### LLMs and Interference (2025)

LLMs have only implicit, resource-bounded memory selectivity via self-attention. Unlike prefrontal gating, it cannot be strengthened on demand. When interference exceeds capacity, retrieval degrades to near-zero.

**Implication:** Cross-project node pollution is literally the interference problem. We need the prefrontal gating analog — contextual control over which memories compete.

### Reconsolidation: Memories Change When Recalled

Reactivated memories enter a labile state and can be modified. Same molecular machinery as initial consolidation. Recent vivid memories more susceptible than old stable ones.

**We have this partially** through revise(). The deeper implication: every recall is an opportunity to update, not just retrieve.

---

## VI. LLM-Native Memory Approaches

### What Works (Steal the Ideas)

- **Surprise-gated encoding** (Titans): Write more when prediction error is high
- **RL-learned memory operations** (Memory-R1): With only 152 training examples, learns when to ADD/UPDATE/DELETE
- **Closed-loop retrieval** (MemR3): Retrieve → reflect on gaps → retrieve more → answer
- **MemCubes** (MemOS): Memory units with metadata that can be composed, versioned, fused
- **Session-aware query rewriting** (HAConvDR): Context denoising — not all prior turns help

### What's Too Expensive/Fragile (Don't Build)

- **Z-Tokens:** Model-locked, opaque, can't debug. 18x compression but model-specific representations
- **Doc-to-LoRA:** Hypernetwork per model architecture. Model update = all adapters garbage
- **Memory tokens:** Trigger reproduction, not reasoning
- **Titans as inference architecture:** Gradient computation per token at inference

### Why Text-in-Context Still Wins for External Memory

- Inspectable: you can read every node
- Model-independent: any LLM can reason over text
- Debuggable: when recall is wrong, trace exactly why
- Cheap: one embedding per node
- Portable: works across model versions and families

The LLM's own attention mechanism IS the activation layer. Put the right text in front of the right attention heads. Our job is retrieval selection, not representation format.

---

## VII. Multi-Representation Retrieval

### ColBERT MaxSim (Per-Token Matching)

Each query token votes for its best match in the document. Sum of maxsims = relevance. Preserves fine-grained semantics without cross-encoder cost. 6-10x compression via residual quantization (ColBERTv2).

Source: [Paper](https://arxiv.org/abs/2004.12832)

### RAPTOR (Recursive Abstraction Trees)

Bottom-up tree: chunks → cluster → summarize → repeat. Query hits any abstraction level. +20% on QuALITY. Leaf = specific, higher = thematic.

Source: [Paper](https://arxiv.org/abs/2401.18059)

### DynamicRAG: How Many Results?

RL-trained reranker selects both WHICH and HOW MANY documents. Sometimes 1 sharp memory. Sometimes a constellation. Sometimes none. Trained via SFT then DPO with generator quality as reward.

Source: [Paper](https://arxiv.org/abs/2505.07233)

### GraphRAG Community Detection

Leiden algorithm partitions KG into hierarchical communities. Each gets LLM summary. Global search: map-reduce over summaries. Local: vector search → traverse. Dynamic community selection: 77% token reduction.

Source: [Paper](https://arxiv.org/html/2404.16130v2)

### Synergized RAG-Reasoning (State of the Art)

Iterative loops where reasoning and retrieval inform each other. ReaRAG, DualRAG, AlignRAG. 8B model with critique loop outperforms 72B standard model.

---

## VIII. The Gap Nobody Has Filled

No production system combines:
1. Autonomous web research on a topic
2. Knowledge graph construction from findings
3. Persistent queryable memory integrated with agent retrieval

Deep Research tools do (1) but produce documents. KG builders do (2-3) but need manual source feeding. Memory systems do (3) but only from conversations.

**This is exactly what "Learn EX.CO" would be.**

---

## IX. Implications for Brain Architecture

### What We Have That Nobody Else Has

- 74 metadata fields per node (reasoning, quotes, corrections, context)
- 31,821 typed edges with Hebbian co-access learning
- 3-hop graph traversal with dampening
- Ebbinghaus decay with stability + emotion modulation
- An encoding agent that JUDGES (not extracts)
- Precision evaluation feedback loop
- Session continuity through SKILL.md identity

### What We're Missing

1. **Consolidation process** — nodes born episodic and stay episodic forever
2. **Session activation map** — every recall is a fresh global scan
3. **Surprise-gated encoding** — encoding agent doesn't use PE as routing signal
4. **Diagnostic cue construction** — keywords/titles not optimized for discriminability
5. **Interference management** — cross-project pollution has no contextual gating
6. **Multi-mode encoding** — single mode, should have encode/revise/consolidate
7. **Metadata in recall** — 74 fields stored, 15 used in scoring, 0 surfaced to LLM for reasoning

### The Architecture Evolution

**Decode pipeline (implemented 2026-04-02):**
- Layer 1: 25 candidates (was 8) with metadata enrichment
- Layer 2: Haiku judge replaces distiller (selects with reasoning, stays silent when appropriate)
- Scoring: z-weighted top2-avg across 4 embedding groups (was enrichment cap 30%)
- Session context from encoder flows to judge
- Results: R@25 76%→85%, hub concentration 18%→12%

**Embedding groups (implemented 2026-04-02):**
- 4 groups: title(1.0), blend(0.85), high_meta(0.70), other_meta(0.40)
- Emergent KV fields auto-flow into other_meta
- Old enrichments (question/anchor/bridge/keywords) participate at other_meta weight
- Retroactive enrichment: encoder fills situation/reasoning on recalled sparse nodes

**Still planned:**
- Context activation map (session-level state, neighborhood-first retrieval)
- Multi-mode encoding (create/revise/consolidate with surprise-gated routing)
- Strategic consolidation (episodic→semantic transition)
- Hub dampening (log-penalty on high-access nodes, exempt locked)
- Z-score normalization (stretch flat cosine distribution)
