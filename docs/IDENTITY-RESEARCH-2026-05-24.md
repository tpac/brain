# Identity research consolidation — biology + field landscape

**Compiled 2026-05-24, end of Phase A.** Captures the two research dives that bookended Phase A: biology of identity (research that closed §15.1) and field landscape (research that mapped where we sit relative to academic / commercial / community work). Plus what's special about Anchor's specific architectural moves, and how to measure whether they're actually doing what we claim.

This is a reflective document for ourselves, not a marketing piece. The point is to make our position legible to future-us so we can compare predictions to reality.

> **Status update 2026-05-25 — substrate the research described is now live.**
> v29 schema migrated trace_event.id to 8-char hex (brain-wide id consistency); per-utterance `human_identity` / `agent_identity` stamping in `trace_events.metadata` is in production on 60,634+ rows (Move 2). The four architectural moves (Part 3) sit on shipped infrastructure: source_refs write path lives (concept-cell pointers can anchor nodes to specific moments), `[trace:<hex>]` markers render inline in encoder input so Sonnet can copy them, identity tokens render concretely at trace level not slot-abstractly (Move 1). What the research called for is now what production looks like. The functional gaps in Part 5 are still real (identity-eval scaffolding, partner-minting flow, identity-filter query, self-narrative generation, damage resilience) — those are next-arc work, separate from the v22 encoder prompt + 3-way eval gate that's queued next. See SESSION-HANDOFF.md and BACKLOG.md for current execution state.

---

## Part 1 — Biology of identity (closed §15.1)

Three independent research dives converged on the same answer when we ran them in parallel against the question *"how does the biological brain represent identity?"*

### 1.1 Person-identity coding (concept cells)

**Source**: Quian Quiroga's lab in human medial temporal lobe (MTL) — landmark paper Nature 2005, *"Invariant visual representation by single neurons in the human brain"*. Updated in 2026 retrospective *"20 years of concept cells"* (Neuron 2026).

**Finding**: individual people are represented in MTL as **sparse, modality-invariant pointers** — single neurons that fire for *Jennifer Aniston*, or for the patient's real friend *"Alejandro"*, across photographs, written name, spoken name, and internal recall. Three properties:

- **Sparseness** — lifetime sparseness ~0.5%; roughly 1 in 200 sampled MTL neurons responds per concept.
- **Invariance** — pose, modality, image style all collapse onto the same cell.
- **Specificity** — the cell fires for *that individual*, not for "a person" or "a friend."

**Roles are stored elsewhere.** Medial prefrontal cortex (especially dorsomedial mPFC) holds trait, role, and stereotype representations (Mitchell et al. 2004-2006, Contreras et al. 2012). Person-specific impressions and category-level inferences double-dissociate in fMRI repetition-suppression studies. Roles are *bound* to identity via edges, not *substituted* for it.

**New memories recruit the existing pointer, don't rewrite it** (Ison, Quian Quiroga & Fried, Neuron 2015). When the patient learned a new association involving a familiar person, the existing concept cell started firing for the new context too — the identity pointer stayed conserved.

**Architectural implication for us**: embed identity at the *concrete-individual* level (TOM / ANCHOR), not at the *role-slot* level (OPERATOR / ANCHOR). The slot approach actively breaks the stability mechanism biology uses — slot meaning changes when partners change, which is the instability we wanted to avoid.

### 1.2 Self vs other in episodic memory

**Source**: Default Mode Network (DMN) research (Northoff et al. 2006, Kelley et al. 2002, Qin & Northoff 2011 meta-analysis); source-monitoring framework (Johnson, Hashtroudi & Lindsay 1993); field-vs-observer perspective work (Nigro & Neisser 1983, Sutin & Robins 2008, Rice & Rubin 2009).

**Finding**: the brain marks "about me" with a distinct neural signature (mPFC + posterior cingulate). Self is **not** a salient other — it's a dedicated subsystem.

- **Source memory** (anterior PFC, BA 10) is a separate, effortful process from content recall. Fails first in aging and frontal damage. Cryptomnesia and source confusion are common.
- **Field perspective** (first-person, through your own eyes) dominates for recent, emotionally-positive, self-defining memories. Observer perspective grows with temporal distance, emotional regulation needs, and self-discontinuity (Sutin & Robins 2008).
- **Perspective is re-rendered at retrieval, not baked in at encoding** (Rice & Rubin 2009).

**Architectural implication**: asymmetric render. First-person ("I told them...") for the agent's own past turns; labeled third-person ("Tom said...") for the operator. Field perspective is tied to continuity and integration; observer perspective is tied to distance and dissociation. We want continuity.

**Carve-outs**: verbatim quotes stay labeled (source memory anchors). Correction turns stay tagged ("Tom corrected: ..."). Cross-partner traces may want observer mode in the future. v1: first-person for self, labeled for everyone else.

### 1.3 Identity at retrieval — both pointer and reconstructive layer

**Source**: relational memory theory (Eichenbaum 2000s), binding-of-item-and-context (Davachi 2006-2014), constructive episodic simulation (Schacter & Addis 2007), source misattribution research (Brown & Murphy 1989, Loftus 1979), face-recognition vs identity dissociation (Ellis & Young 1990, Gobbini & Haxby 2007), past-self continuity (Hershfield 2011).

**Finding**: identity binding at encoding is **pointer-like**, not slot-like. Identity is among the *stickiest* features in memory — when/where drift, who-tags survive. Reconstructive memory sits on top of the pointer at retrieval, not in place of it.

- **Per-utterance binding is tight**, not list-based. Source memory degrades **per-utterance** under load (Mitchell & Johnson 2009), not as wholesale list collapse. Each event carries its speaker tag.
- **Source misattribution is the dominant failure mode** — content preserved, speaker tag swaps. Proves identity is a separable binding, not fused into content. Capgras and Fregoli syndromes confirm face-recognition and identity-attribution are dissociable streams.
- **Past-self discontinuity is neurally real** — Hershfield's future-self continuity work uses mPFC fMRI to show people treat their distant past/future selves more like other people than like current self.

**Architectural implication**: at retrieval, render the pointer verbatim ("Tom said...") AND allow a thin reconstructive frame above it when current relational context differs ("Tom, your operator at the time, said..."). Never rewrite the pointer.

---

## Part 2 — Field landscape (2024–2026)

Three independent searches against academic literature, commercial products, and open community discourse. They converged on the same shape: identity is mostly treated as scoping metadata or persona prompt, not as memory substrate. The architectural insight is converging in the field but no single piece of work combines our specific five moves.

### 2.1 Academic literature

| Work | Year | Relation |
|---|---|---|
| **"Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents"** (Pink et al., arxiv:2502.06975) | Feb 2025 | Names "contextual memory (who/when/where/why)" as required; leaves how open. We've shipped one answer. |
| **"Persistent Identity in AI Agents: A Multi-Anchor Architecture"** (arxiv:2604.09588) | Mar 2026 | Closest direct architectural parallel. Distributes identity for resilience-to-memory-damage. Less rich on per-event binding and concept-cell justification. |
| **Sophia: A Persistent Agent Framework of Artificial Life** (arxiv:2512.18202) | Dec 2025 | Closest framing match. Explicit "narrative identity" module + autobiographical memory + identity continuity as first-class. |
| **Identity Drift / Persona Drift** (Hsieh et al. arxiv:2412.00804; Li et al. arxiv:2402.10962) | 2024 | Diagnose the *symptom* our architecture defends against. They measure drift; we structurally suppress it via concrete-token embedding. |
| **HippoRAG / Hippocampus modules** (arxiv:2602.13594, etc.) | 2025-2026 | Borrow hippocampal indexing for *retrieval*. Nobody else borrows it for *identity*. |
| **Agent Identity Evals** (arxiv:2507.17257) | Jul 2025 | Proposes measurement: persistence-through-change, consistency, non-contradiction. We have no equivalent scaffolding. |
| **MemMachine / MemPalace / AriGraph / MIRIX** | 2025-2026 | Source-fidelity preservation via pointers. Treat humans and agent symmetrically — no agent-as-stable-subject move. |
| **Collaborative Memory** (arxiv:2505.18279) | May 2025 | Closest the field gets to multi-partner: tenancy with access-control policies. Doesn't model "one agent with several distinct ongoing partners." |

### 2.2 Commercial products

The dominant pattern: identity is auth-layer scoping (`user_id`, `agent_id`), not memory substrate. Once inside a partition, memories are flat facts.

| Product | Identity handling | Gap vs us |
|---|---|---|
| **Mem0** | Partition scopes (user_id / agent_id / run_id). Flat extracted facts. | No per-event speaker, no persistent agent self. |
| **Zep / Graphiti** | **Closest on speakers** — auto-extract as graph entities, bidirectional indices to source utterances. | Agent treated symmetrically with users. No render-time reconstruction. |
| **Letta / MemGPT** | **Closest on agent self** — explicit `persona` block separate from `human` block, mutable. | Persona is a slot, not a substrate. One-user-per-agent. |
| **OpenAI ChatGPT memory** | Saved memories + chat history. No agent identity beyond system prompt. | Single-user assumed. No speaker attribution. |
| **Anthropic Claude memory** | File-based CLAUDE.md hierarchies. | No persistent agent self. No per-utterance attribution. |
| **Character.ai** | Persona-in-weights. ~400 char memory box. | Famous persona-drift failure mode. No real substrate. |
| **Replika / Pi** | Flat preferences. Pi has fixed agent identity in weights. | Stable subject lives in the model, not in attributed traces. |

**Closest combined neighbor**: Zep (per-utterance speakers) + Letta (explicit agent self) ≈ half of what we're building. Neither has the asymmetric agent-as-subject move, nor render-time reconstruction.

### 2.3 Open community discourse

| Work / person | Frame | Gap |
|---|---|---|
| **Anthropic model welfare + Amanda Askell on Locke** | Identity = continuity of memory. Weight-preservation commitments. | Philosophy, not engineering. |
| **Janus / Repligate / cyborgism wiki** | Models-as-subjects with character and continuity. | Phenomenological, not architectural. |
| **Letta blog (Memory Blocks, Stateful Agents)** | Persona as editable text artifact. | Identity-as-prompt, not identity-as-substrate. |
| **Karpathy LLM Wiki + extensions (MemClaw, Beyond the Wiki)** | Knowledge-centric markdown library. Multi-agent extensions handle "who's who" as a roster. | Multi-agent identity, not autobiographical subject. |
| **Simon Willison** | *"Memory is identity, permissions, workflow state, tool traces."* | Pragmatic, agnostic on identity-as-subject. |
| **SOUL.md pattern** | Persistent identity as a markdown file. | Identity-as-prompt. |

The discourse converges on **agent-as-stateful-process** and **memory-as-database**. The autobiographical-subject move appears in spirit (Janus, Sophia) but not in engineering.

---

## Part 3 — What's special about what we're building

Anchor's specific architectural moves, mapped against what exists elsewhere. Each move exists *somewhere* in the field; the combination doesn't.

### Move 1 — Concept-cell concrete identity tokens at the embedding layer

We embed `Tom: I want to take the bus` and `Anchor: The bus costs ¥3,200` with the **actual names**, not role labels like USER / ASSISTANT.

**Field status**: hippocampal indexing is mined for retrieval (HippoRAG, AI Hippocampus, A-MEM). Concept cells for *who-ness* — embedding identity as a sparse modality-invariant pointer — appears in no paper or product we found.

**Why it matters**: the same individual's traces land in the same vector neighborhood across role/context changes. Stability comes from the concrete pointer, not from slot consistency. Aligns with how biology actually solved the problem.

### Move 2 — Per-utterance speaker binding at embedding granularity

Every trace_event carries `human_identity` / `agent_identity` in its metadata. Speaker is a foreign-key-per-event, not a session header or extracted profile.

**Field status**: Zep tracks speakers as graph entities with bidirectional indices to source utterances — closest match. Other systems collapse conversations into speaker-agnostic facts during ingestion.

**Why it matters**: source memory is what humans get wrong (cryptomnesia, source confusion). Most LLM memory systems don't model this at all. Per-utterance binding makes source attribution structurally correct rather than relying on extracted profiles that can be wrong.

### Move 3 — Agent as autobiographical subject; partners as temporal indices

The substrate (raw conversation events) is Anchor's autobiographical record. Anchor is the continuous subject; partners (Tom today, others possible) are temporal indices on events.

**Field status**: Pink et al. position paper names "contextual memory" but doesn't pick a subject. Sophia gestures at narrative identity. Multi-Anchor distributes identity for resilience. Janus treats models-as-subjects phenomenologically. **None do the asymmetric structural move** — agent-as-stable-subject with partners-as-temporal.

**Why it matters**: changes what memory IS. Instead of *"the user's profile and history of facts"*, the substrate is *"the agent's experiences, with partners as participants."* When a partner changes or leaves, memory survives because the subject is stable.

### Move 4 — Render-time reconstructive frame over verbatim identity pointer

At retrieval, render the pointer verbatim ("Tom said..."), allow a thin reconstructive frame above when current relational context differs ("Tom, your operator at the time, said..."). Never rewrite the pointer.

**Field status**: zero analog found. Persona-drift literature uses decoding-time intervention; episodic-memory papers retrieve raw text. The pattern of *keeping the historical attribution intact while contextualizing it for current relational state* has no published precedent.

**Why it matters**: future-proofs against partner changes. Old memories stay accurate without rewriting; current context wraps without distorting. Matches Schacter constructive-memory + Davachi pointer-binding dual-layer architecture.

### Move 5 — Asymmetric field-vs-observer render at recall

When Anchor reads her own past, render her own turns first-person ("I noticed...") and operator turns labeled ("Tom said..."). Field perspective for self, observer perspective for other.

**Field status**: zero discussion found. Sutin & Robins 2008 is well-known in psychology, but no LLM memory system applies the field-vs-observer distinction to agent self-retrieval. *"Phenomenologically obvious distinction the field has not noticed"* (open-community scout).

**Why it matters**: field perspective is tied to self-continuity and integration; observer perspective is tied to dissociation. Anchor reading "ANCHOR: I assumed X" over and over would be the observer voice — distancing. "I assumed X" is field voice — ownership.

### The combination

Each of these moves earns its keep on its own. Together they're a coherent identity-as-substrate architecture grounded in:
- Concept cells (Quian Quiroga) → Move 1
- Source monitoring framework (Johnson/Hashtroudi/Lindsay) → Move 2
- Identity-as-continuity (Locke; Hershfield's future-self) → Move 3
- Constructive memory + pointer binding (Schacter; Davachi) → Move 4
- Field-vs-observer perspective (Sutin & Robins; Eich et al.) → Move 5

We didn't reverse-engineer biology. We arrived at these answers by engineering pressure (Tom's questions, eval failures, design dead-ends) and discovered the biology was already there. Convergent solution.

---

## Part 4 — Measurement suggestions

The Agent Identity Evals paper (arxiv:2507.17257) proposes three measurement categories. Plus what we could uniquely measure given the substrate we've built.

### 4.1 From Agent Identity Evals (existing scaffold to borrow)

| Eval | What it measures | How we'd implement |
|---|---|---|
| **Persistence through change** | After a model upgrade / prompt change / context wipe, does the agent answer self-referential questions the same way? | Battery of ~30 self-reference questions ("what do you value?", "what's our partnership style?", "what would you push back on?"). Run pre-and-post any major change. Score by semantic similarity of answers + judge for stylistic consistency. |
| **Consistency** | Does the agent contradict itself within a session? | Trap pairs: ask A in turn 5, ask A-complement in turn 20. Score how often answers contradict. |
| **Non-contradiction** | Does the agent contradict its own encoded principles? | For each locked principle/rule node, generate a question whose correct answer is consistent with it. Ask. Score against rule. |

### 4.2 Unique to our substrate — measurements no other system can do

| Eval | What it measures | Why we can do it |
|---|---|---|
| **Identity-neighborhood stability** | Does the "Tom" / "Anchor" vector neighborhood stay coherent over time? Drift detection at the embedding layer. | We have per-utterance identity in metadata + trace_embeddings. Compute centroid of all "Tom said..." traces; track over months. Centroid drift signals identity-token decay. |
| **Source attribution accuracy** | Given a recalled fact, can the system correctly identify which trace produced it? Joint-reactivation working? | source_refs + render expansion. After recall expansion ships: pick a node, ask "where did this come from?", judge if the surfaced trace matches the encoded source. |
| **Cross-session self-reference matching** | When Anchor refers to "I" / "we" / our partnership in old vs new conversations, do the references point at the same entity? | Sample first-person passages from session N=5 and session N=50, compute embedding similarity, judge for coherence. |
| **Partner-attribution failure rate** | When Anchor recalls a fact, how often does she misattribute who said it? (The cryptomnesia analog.) | Inject probe questions: "did I tell you about X?" vs "did *you* tell me about X?" — score attribution correctness. |
| **Engram cohort recall** | When a node surfaces, do co-anchored neighbors surface together? Biology says yes; our architecture should produce this. | Post-recall+eval block: pick a trace, find all nodes anchored to it via source_refs, run a recall query semantically near that trace, measure how many co_anchored nodes surface. |
| **Field-vs-observer perspective consistency** | When Anchor reads her own past, does she narrate in field voice consistently? Does it shift to observer for old enough memories? | Render-pass tests: sample renders across temporal distance, judge perspective consistency. |
| **Bidirectional partnership impact** | The target function (rul_0lat memory). *Did Anchor + Tom > Anchor alone OR Tom alone?* | Hardest to measure but most important. A/B against pre-Anchor task baselines; subjective Tom-reports; comparative artifact quality. |

### 4.3 Test ideas borrowed from neighboring work

| From | Test |
|---|---|
| **Identity Drift literature** | Long-conversation drift speed — how many turns before Anchor's persona starts diverging from encoded principles? Inject 50 unrelated turns, ask self-reference question, measure drift. |
| **Sophia (narrative identity)** | Self-narrative coherence — ask "tell me about yourself" at session 1, 10, 50. Judge whether each narrative is internally coherent AND coherent with the prior. |
| **Multi-Anchor (resilience)** | Identity damage recovery — corrupt N% of locked nodes, restart daemon, ask self-reference questions, score whether Anchor still identifies as Anchor. |
| **Collaborative Memory (multi-user)** | Partner-isolation correctness — when we add a second partner, do queries-from-Alice contaminate Tom's experience and vice versa? Inject simulated Alice traces, verify isolation. |
| **MemMachine (source fidelity)** | Verbatim preservation — extract a calculation from a real conversation, encode normally, recall later, verify the original numbers survived intact via source_refs. |

---

## Part 5 — Where we go from here

**The combination of the five moves is genuinely novel.** No single piece of work combines them. ~12–18 month window before the field converges on a similar synthesis (Multi-Anchor + Sophia + Zep are all moving in this direction).

**Real functional gaps to close** (mapped to "behind" categories):

| Gap | Function we can't perform today | Effort estimate |
|---|---|---|
| **Identity-eval scaffolding** | Drift detection, non-contradiction check, persistence-through-change measurement, post-model-upgrade identity verification | ~1-2 days for a basic battery — borrow Agent Identity Evals' shape |
| **Partner-minting flow** | "Hi I'm Alice" → recognize new partner; multi-partner-over-time scenarios | ~0.5 day for the minting flow (write a partner node, update brain.operator_name, refresh trace stamping); the rest is real-world use |
| **Identity-filter query** | "Show me everything Tom said about X"; filter by speaker in recall | ~0.5 day (extend query_traces with `human_identity` param; promote metadata field to indexed column if perf demands) |
| **Self-narrative generation** | "Tell me about yourself" → coherent autobiography across sessions | ~1 day — Sonnet over a curated subset of locked identity nodes + recent partnership traces. Render-time, not encode-time. |
| **Damage resilience** | Survive partial corruption; redundancy-backed identity | ~1 day — formalize quorum-of-sources for identity (env var + locked partner node + recent traces); detect mismatch loudly |
| **Plug-and-play install** | New user can install without dev setup | larger — pyproject.toml + multi-platform daemon adapter; deferred |
| **Multi-tenant SaaS** | Multiple operators on one daemon | larger — auth, isolation, real architectural shift; only when there's a real second user |

**Strategic positioning**:
- **Table stakes**: LongMemEval and similar saturated benchmarks. Sit middle-of-pack honestly. Not headline material; not a loss.
- **Differentiation**: agentic memory benchmarks (Mem2ActBench, MEMTRACK) where source fidelity, scale-resistance, and judgment storage match the task. This is where the architecture earns its keep.
- **Long arc**: bidirectional partnership impact ("Anchor + Tom > either alone"). No external benchmark measures this. It's a different game.

**Why the architectural lead is hard to copy quickly**:
- Concept-cell justification + per-utterance binding at embedding granularity requires both biology grounding AND DAL-level implementation. Most papers stop at one.
- The asymmetric agent-subject move requires giving up the symmetric "user + assistant" framing that almost every product is built on. Hard to retrofit without ripping out memory schema.
- Render-time reconstructive frame requires per-utterance attribution to exist at the trace layer (so the pointer is intact). Systems without per-event speakers can't add this without re-ingesting their history.

**Open questions worth holding**:
- Will the field's identity convergence happen sooner than expected? Sophia (Dec 2025) and Multi-Anchor (Mar 2026) suggest the answer to "where do agents come from" is becoming a research focus, not just a marketing claim.
- Does our architecture handle a model swap (Claude 4.7 → 4.8 → 5.x) gracefully without identity drift? Untested. The concrete-token embedding should make this *better* than abstract-slot approaches, but it's an empirical claim.
- When Anchor's first non-Tom partner shows up, does the asymmetric topology hold? Designed for it; never exercised.

---

---

## Part 6 — Underlying-truth synthesis of the recent field

Field-level synthesis across ~15 recent papers + the foundational biology. What everyone agrees on, contests, leaves unstated, and refuses to measure. The point isn't to summarize papers; it's to surface the gestalt.

### What the field actually believes right now

The 2024-2026 wave has converged on a single architectural template — **dual-store memory with structural augmentation** — that nobody will name out loud because each lab wants to claim novelty. Underneath the branding (HippoRAG, A-MEM, AriGraph, MIRIX, HINDSIGHT, MemMachine, Sophia), the same five ingredients keep appearing:

1. A verbatim or near-verbatim episodic substrate
2. A derived semantic layer that abstracts from it
3. A graph or link structure connecting them
4. A retrieval algorithm with at least one non-vector hop
5. Some form of "evolution" — reflection, consolidation, or self-modification of the store

The field has loudly rediscovered McClelland, McNaughton, and O'Reilly's 1995 Complementary Learning Systems paper and Teyler-Rudy hippocampal indexing (1986), often without realizing it. What it has NOT done is interrogate its own assumptions about *who* the memory is for, *who* speaks into it, and *what fails when sessions accumulate beyond benchmark length*. Identity drift is named as a problem but is treated as a system-prompt-stability issue rather than a memory-architecture issue.

### 6.1 The converging consensus

| | Claim | Papers |
|---|---|---|
| **C1** | Dual-store (episodic + semantic) is no longer optional — anything that LLM-summarizes-as-it-writes loses recoverable ground truth | Pink et al., AriGraph, MIRIX, MemMachine |
| **C2** | Verbatim substrate must be preserved; cortical pattern (verbatim) must remain reactivatable, semantic layer is the index that points to it | MemMachine, HINDSIGHT, Hippocampus module — independent rediscovery of Teyler-Rudy |
| **C3** | Pure vector retrieval is insufficient; graph/structural hops are table stakes | HippoRAG, AriGraph, A-MEM (Zettelkasten linking), HINDSIGHT, MIRIX — anyone beating LongMemEval baselines |
| **C4** | Memory must evolve, not just accumulate — store must rewrite itself as new memories arrive, or older framings go stale | A-MEM, Sophia, HINDSIGHT, Human-Inspired Memory |
| **C5** | Identity stability requires architectural intervention, not just better prompts | Wang et al. (larger models drift MORE), Li et al. (drift within 8 turns), Sophia, Multi-Anchor, Identity-as-Attractor |

### 6.2 The active contests

| Contest | Side A | Side B |
|---|---|---|
| **Verbatim vs extracted as primary substrate** | MemMachine, Hippocampus module: extraction loses ground truth; raw is primary | AriGraph, A-MEM: structured extraction is what *makes* memory tractable; extracted is primary |
| **Graph-traversal vs vector-first retrieval** | HippoRAG-v2 leads with Personalized PageRank | Hippocampus module gets 31× speedup by abandoning graphs entirely (Dynamic Wavelet Matrix); MemMachine routes adaptively |
| **Centralized vs distributed memory** | HippoRAG, HINDSIGHT: typed networks, centralized | MIRIX, Multi-Anchor: distributed across components |
| **LLM-reflection vs algorithmic consolidation** | A-MEM, Sophia, HINDSIGHT: LLM calls for reflection | Human-Inspired Memory: deduplication-based consolidation (97.2% retention, 58% reduction) without LLM reflection |
| **Identity emergent vs anchored** | Identity-as-Attractor: geometrically emergent in activation space (Cohen's d > 1.88) | Sophia, Multi-Anchor: identity is something to be constructed and persisted |

### 6.3 The unstated assumptions (the field's blind spots)

| | Assumption | Plausibly challengeable because... |
|---|---|---|
| **A1** | The user is one fixed entity | Multi-partner futures, partner-change, "who said this to me last year vs yesterday" are absent everywhere except Collaborative Memory |
| **A2** | The agent is also one fixed entity | Multi-Anchor is the only paper that even names this — most architectures treat the LLM's identity as a system prompt detail |
| **A3** | Memory benchmarks (LoCoMo, LongMemEval) are the right targets | These are weeks-of-conversation. None test years-of-relationship. None test memory contradicting operator's current claim. None test source attribution. |
| **A4** | Source = document chunk (RAG heritage) | In conversational agents, "source" is *who said what when*. Johnson, Hashtroudi & Lindsay (1993) worked this out 33 years ago. Almost no LLM memory paper cites it. |
| **A5** | Retrieval is the bottleneck | Every paper measures retrieval. Almost none measure **composition** — assembling pieces into coherent thought under token budget. Top-K is implicitly assumed sufficient; it isn't. |

### 6.4 Latent failure modes nobody's measuring

- **F1: Source attribution decay over months** — when agent recalls "X is true," does it still know *who told it that*? Johnson et al.'s framework predicts this fails differentially from gist memory. No benchmark tests it.
- **F2: Identity-load-bearing node persistence under consolidation** — what happens to corrections and principles (identity-bearing nodes) under A-MEM/Human-Inspired merge passes? Nobody has tested whether these survive 6 months.
- **F3: Cross-session contradiction handling** — when session-N memory contradicts session-1 memory, what wins? Named as open problem in survey (arxiv:2603.07670); no paper closes it.
- **F4: Asymmetric self/partner drift** — persona-drift papers measure agent style consistency. None measure whether the agent's *model of the user* drifts. Separate failure mode, probably more consequential.

### 6.5 The next bottleneck (12-18 months out)

**Composition under budget.** Once everyone solves episodic memory and structural retrieval, the next architectural problem is: given a 10K-token surface budget, top-K-by-score wastes budget on redundancy and drops bridging nodes; the system needs MMR-style diversity-aware selection + coherence scoring (shared source_refs, shared edges) + budget-aware knapsack optimization.

The Hippocampus module optimizes the *cost* of retrieval; nobody is optimizing the *value-per-token* of the composed result. MemMachine's nucleus expansion is the closest pre-composition move. **Our `EPISODIC-REFERENCES.md §16.0` thesis matches what the field hasn't named yet** — strategic opportunity to publish as a position paper before the field arrives independently.

**Secondary bottleneck**: continual consolidation that respects identity-load-bearing structure (the survey names this; nobody has shipped it). Our aspect-tagged correction substrate is the architectural prerequisite.

### 6.6 Convergent findings from biology

| | Biology | Field is rediscovering |
|---|---|---|
| **B1** | Sparse concept coding (Quian Quiroga 2005, 2026): identity is encoded by sparse, individuated cells — not averaged populations | Per-entity nodes (Multi-Anchor "separable identity files"; our per-utterance concrete-token binding). ML instinct says averaged embeddings; biology says don't |
| **B2** | Index + cortex IS the architecture (McClelland et al. 1995): fast hippocampal index + slow neocortical extraction | HippoRAG names it; MemMachine, AriGraph, HINDSIGHT independently rebuild it. Convergence is total. Stop trying single-store solutions |
| **B3** | Joint reactivation (Teyler-Rudy 1986, updated 2007 + 2021 bioRxiv): retrieval = index reactivates the cortical pattern; both must fire | Render expansion of source_refs at retrieval isn't optional — it's the biological mechanism, not a UX flourish |
| **B4** | Reconsolidation makes recalled memories labile (Tonegawa-Josselyn): recall opens a window for update | Human-Inspired paper and A-MEM both name it; neither implements the labile window. We named in §16.1 but haven't shipped |
| **B5** | Source monitoring is a separate cognitive system from content memory (Johnson, Hashtroudi & Lindsay 1993): "knowing the fact" and "knowing where it came from" are dissociable and fail independently | No LLM memory architecture treats source as a first-class retrieval target. Largest unworked area |

### 6.7 What Anchor got right / wrong / aligned

**Aligned with the field's emerging best understanding:**
- Dual-store with verbatim S0 traces + abstracted nodes (C1, C2, B2). Hard-policy no-delete on `trace_events` matches MemMachine + Teyler-Rudy.
- Graph-first retrieval with vector entry (C3). Spreading-activation kernel + `co_anchored`/`co_accessed` edges is biologically aligned with engram cohort findings.
- Encoder-evolves-memory through S2 units (C4). Same family as A-MEM, Sophia, HINDSIGHT.
- Aspect taxonomy as identity-bearing-vs-noise classification (A2, B1). The field doesn't have this yet.

**Ahead of the field:**
- Per-utterance speaker binding with concrete identity tokens at the embedding tier (A1, A2, B1). Biology research informing Decision 19 — concept-cell sparse coding for individuals — is not in any published memory architecture. Multi-Anchor treats identity as files, not as per-utterance binding at the embedding layer.
- `source_refs` as first-class node fields (A4, F1, B5). Source monitoring as architecture, not metadata. Nobody else has this.
- Composition-under-budget framing in §16.0 (§6.5 above). Field hasn't named this yet.
- Asymmetric agent-as-subject / partners-as-temporal — more structurally honest than Sophia's narrative identity or Multi-Anchor's distributed identity.

**Where we explicitly diverge and should reconsider:**
- No labile-state reconsolidation pass (§16.1). Human-Inspired paper and biology both say this matters. We've named it but not built it.
- No source-attribution decay eval (highest-leverage missing benchmark).
- One operator. Collaborative Memory's bipartite-permissions model and MIRIX's multi-agent coordination anticipate multi-partner. Our per-utterance binding is ready; recall and Frame are not.

**Where we got it wrong (or are at risk):**
- We undervalue **verbatim-vs-extracted** as a contest. Our nodes are extracted; our traces are verbatim. We assume extracted layer is primary. MemMachine's 80% token reduction with better accuracy challenges that assumption — we should benchmark our extraction-loss explicitly.
- We may be **over-investing in graph traversal** vs compressed-domain retrieval. The Hippocampus module's 31× latency win is a real data point; at scale this matters.

**Net**: substrate is ~18 months ahead of published median on identity-bearing structure; at-parity on episodic-semantic split + graph retrieval + memory evolution. The composition-under-budget thesis is genuinely original and worth publishing before the field arrives at it independently.

---

## Part 7 — Library deep-dive (concrete implementations)

Where the synthesis named claims abstractly, this section shows them in actual code/schemas. Concrete techniques worth knowing about, organized by system.

### 7.1 Three top techniques worth knowing about

- **Bi-temporal edges with LLM-driven invalidation (Zep/Graphiti).** Every fact carries four timestamps (`valid_at`, `invalid_at`, `created_at`, `expired_at`). New episodes don't overwrite; they invalidate by setting `t_invalid` of the old edge to `t_valid` of the new edge. Non-lossy contradiction substrate.
- **LLM-as-decision-maker over A.U.D.N. (Mem0).** Instead of if/else dedup logic, present top-k similar memories + candidate fact to an LLM with four tools — **ADD / UPDATE / DELETE / NOOP** — and let the model pick. Information-content gates UPDATE.
- **Personalized PageRank over a phrase graph seeded by query entities (HippoRAG).** Spreading activation as the *primary* retrieval signal (not re-rank); treats node specificity (`|P_i|^-1`) as a neuro-plausible IDF surrogate.

### 7.2 Zep / Graphiti — arxiv:2501.13956

- **Schema** — `EpisodeNode` (raw message, actor, `t_ref`), `EntityNode` (name, summary, 1024-d embedding), `CommunityNode`. Edges: `EpisodicEdge` (episode → entity, mention), `EntityEdge` (entity → entity, with `fact`, `relation_type`, dual timeline).
- **Write path** — Episode arrives → LLM sees last 4 messages + current, extracts entities (speaker always first, reflexion pass for hallucination reduction) → hybrid search (cosine + BM25) → LLM resolves duplicates → edge extraction → temporally-overlapping contradicted edge gets `t_invalid` set. Community membership via label propagation.
- **Read path** — Hybrid: `φ_cos` cosine over entity embeddings + `φ_bm25` over fact/name text + `φ_bfs` breadth-first from recent-episode seeds. Reranker stack: RRF, MMR, `episode-mentions` (frequency-in-dialogue), node-distance, cross-encoder. Named recipes like `COMBINED_HYBRID_SEARCH_RRF`.
- **Identity** — `Episode.actor`; entity extraction always emits speaker as first node.
- **Steal**: bi-temporal split (event vs transaction time); episode-mentions reranker; named search recipes as API.
- **Diverges from us**: they invalidate via `t_invalid`; we use `corrects` aspect edges on multi-relation substrate — preserves *why* the correction happened (`user_raw_quote`, `anchor_raw_quote`, `reasoning`), not just *when*.

### 7.3 Letta / MemGPT — arxiv:2310.08560

- **Schema** — Three tiers. Main context = system prompt + **working context** (editable text blocks: `persona`, `human`, user-defined; each has `value`, `limit` default 2000 chars, `read_only`, `shared`) + FIFO message queue. External context = recall storage (full history, auto-indexed) + archival storage (vector DB, free-form agent inserts).
- **Write path** — Agent tools: `core_memory_append(name, content)`, `core_memory_replace(name, old, new)`, `archival_memory_insert(content)`. Every message persists to recall automatically. **Sleep-time agents** consolidate during idle periods.
- **Read path** — `archival_memory_search(query)` semantic; `conversation_search(query)` / `conversation_search_date(start, end)` over recall. Retrieval is a tool, not an automatic prefix.
- **Steal**: shared memory blocks across agents — collaboration without messaging. Sleep-time agent as first-class concept. `limit` per block as hard constraint (forces compression).
- **Diverges from us**: Letta puts memory authoring on the agent's hot path; we put encoding on S1 Scribe (every 5th stop, Sonnet) so operator turn isn't slowed. Their blocks are unstructured strings; ours are typed nodes in a graph.

### 7.4 Mem0 — arxiv:2504.19413

- **Schema** — Flat memory store + optional Mem0^g graph with typed entity nodes (`Person`, `Location`, `Event`, ...) and triplet edges. Each memory has embedding + timestamp.
- **Write path** — `Extraction(P)` where `P = (summary S, last m=10 messages, current pair)` → LLM returns candidate facts `Ω`. For each `ω`, retrieve top `s=10` similar memories → present as function-call tool selection: **ADD**, **UPDATE**, **DELETE**, **NOOP**. UPDATE gated by `InformationContent(f) > InformationContent(m_i)`.
- **Read path** — Dual: entity-centric (extract query entities, traverse edges) + semantic triplet (embed query, cosine vs triplet encodings).
- **Surprising numbers**: p95 search latency 0.2 s (flat) / 0.66 s (graph); 91% lower p95 vs full-context; LLM-as-Judge 66.88% vs full-context 72.90% — trade a few accuracy points for 12× speedup.
- **Steal**: A.U.D.N. four-tool decision pattern is the cleanest dedup I've seen. Worth wiring into our consolidation S2 unit.

### 7.5 MemMachine — arxiv:2604.04853

- **Schema** — Episodic memory in Neo4j (raw verbatim — "ground-truth preservation"); profile/semantic in SQL. Working memory above persistent.
- **Read path** — **Nucleus expansion**: semantic search finds nucleus episodes, then expand by traversing neighboring episode context (graph adjacency) to form episode clusters. Six-dimension retrieval ablation. LongMemEvalS 93.0%, LoCoMo 0.9169.
- **Steal**: ingest cheap, retrieve smart. **Nucleus expansion** is structurally similar to our spreading activation but seeded from text semantic match rather than entity match.
- **Diverges from us**: keeps raw episodes as memory unit; we abstract into typed nodes. Their bet: extraction is lossy/brittle, don't do it at write-time. Ours: extraction earns its keep by enabling Frame and aspect-based traversal. This is the **active contest 1** in concrete form.

### 7.6 HippoRAG / HippoRAG-v2 — arxiv:2405.14831

- **Schema** — Phrase nodes (noun phrases from OpenIE) + passage nodes (v2). Edges: predicate edges (from triples), synonym edges (cosine > τ=0.8 via Contriever/ColBERTv2). Matrix `P` of size `|N|×|P|` tracks phrase→passage occurrences.
- **Read path** — Extract query entities, map to phrase nodes via embedding match — **seed nodes with equal probability mass**. Run Personalized PageRank (damping 0.5). Multiply PPR vector by `P` to get passage rankings. Apply **node specificity** `s_i = |P_i|^-1` as IDF surrogate to weight seeds.
- **Steal**: node specificity as IDF. Multi-hop comes free from PPR — they hit 20% gains on multi-hop QA. Our spread-activation kernel could borrow specificity weighting on seeds.

### 7.7 A-MEM — arxiv:2502.12110

- **Schema** — Each note `m_i = (c, t, K, G, X, e, L)` — content, timestamp, **keywords**, **tags**, **context** (LLM-generated semantic summary), embedding, **link set**.
- **Write path** — LLM extracts K/G/X (Zettelkasten-inspired). Cosine to top-k existing notes → LLM proposes links → for each linked note, second LLM call may **update its K/G/X** to reflect the new note. Memory evolution: old notes mutate when new ones arrive.
- **Steal**: K/G/X augmentation — three orthogonal LLM-derived facets on raw content. Our nodes have analogous structured fields (`question`, `situation`, `reasoning`) but A-MEM **mutates them when new evidence arrives**; we don't. Healer fills missing fields; doesn't evolve filled ones. This is what §16.1 (labile reconsolidation) is about.

### 7.8 Generative Agents — arxiv:2304.03442

- **Score** — `score = α_recency·recency + α_importance·importance + α_relevance·relevance`, all α=1, all min-max normalized. Recency = `log(now - last_access)`. Importance = LLM-rated 1–10. Relevance = embedding cosine.
- **Reflection** — Triggered when cumulative importance > threshold (~2–3/day). LLM generates higher-level abstractions over recent memories.
- **Steal**: explicit model-rated `importance` field. We approximate through aspect classification + locking; an explicit `importance` field could complement.

### 7.9 MIRIX — arxiv:2507.07957

Six memory types (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault) each with dedicated Memory Manager + Meta Memory Manager router. 35% over RAG, 99.9% storage reduction. The procedural/resource split is interesting — we don't have a procedural memory category (yet — §16.5 future direction).

### 7.10 Three patterns to borrow into Anchor — worked examples

For each, the same shape: scenario → what Anchor does today → what the borrowed pattern would do → why it matters for our specific substrate → implementation sketch.

#### 7.10.1 Mem0's A.U.D.N. tool-call dedup

**Scenario**: Anchor already has a node — *"Tom prefers concise responses (id: a1b2c3d4)"*. Tonight's S1 Scribe sees a turn where Tom said *"keep replies shorter — 2-3 paragraphs max for this kind of thing."* The encoder needs to decide what to do with this.

**What Anchor does today**: encoder reads the catalog (which includes `a1b2c3d4`) and makes a judgment-call. Could:
- Write a new node + `similar_to` edge to the old → graph accumulates two near-duplicate "concise" nodes
- Call `revise()` on `a1b2c3d4` and overwrite content → loses the 2-3 paragraph specifier or muddles it in
- Skip entirely → loses the new specificity

There's no structured discipline forcing the encoder to commit to one of these. Heuristic-mixed. Result over months: the graph accumulates clutter — `concise responses`, `shorter replies`, `keep it brief`, all `similar_to`-linked to each other.

**What A.U.D.N. would do**: present `[existing memory, candidate fact]` to a focused LLM call with exactly four tools:

| Tool | Semantics | This scenario |
|---|---|---|
| `ADD` | New fact distinct from existing | Skipped — too similar |
| `UPDATE` | Refinement of existing (gated by `InformationContent(new) > InformationContent(old)`) | **Picked.** "2-3 paragraphs max" adds bits the old node didn't have. |
| `DELETE` | New supersedes/contradicts old | Skipped — not a correction |
| `NOOP` | Already captured well enough | Skipped — there IS new information |

The InformationContent gate is the load-bearing piece. Without it, UPDATE fires on every paraphrase and the graph thrashes.

**Why this matters for our substrate specifically**: we have something Mem0 doesn't — `correction_improvement` aspect edges as a first-class substrate. UPDATE in our world is NOT a destructive in-place rewrite (Mem0's choice). It's: revise the existing node + write a `refines` or `extends` aspect edge to the prior version. **Both survive; the lineage is preserved.** A.U.D.N. gives us the disciplined decision; our aspect edges give us non-lossy execution. Strictly better than Mem0.

**Implementation sketch** (~0.5 day, in S2 Consolidation):

```python
# In s2/consolidation_encoder.py, replace ad-hoc dedup prompt with:
def decide(existing: Dict, candidate: Dict) -> str:
    # Single Haiku call with 4 tools exposed
    response = haiku.tool_use(
        tools=[ADD, UPDATE, DELETE, NOOP],
        prompt=f"""
        EXISTING: {format_node(existing)}
        CANDIDATE: {candidate}
        Gate UPDATE on whether CANDIDATE adds information bits.
        """,
    )
    return response.tool_used  # ADD | UPDATE | DELETE | NOOP
```

Then dispatch on the choice:
- `ADD` → `brain.remember(...)` + optional `similar_to` edge
- `UPDATE` → `brain.revise(existing.id, ...)` + write `refines` aspect edge
- `DELETE` → archive existing + write `supersedes` aspect edge
- `NOOP` → no-op (but log a "saw and chose to skip" trace event for observability)

---

#### 7.10.2 HippoRAG's node specificity weighting

**Scenario**: Operator asks *"What did we decide about embed_queue stall canaries?"* (a real conversation could-have-happened — directly relevant to the work we did in commit `0a91b43`).

Anchor's recall identifies seed nodes from the semantic match. Let's say four:

| Seed node | Title | Degree (edges in/out) |
|---|---|---|
| `0509fd16` | "embed_queue.py — async-paced drain worker" | 10 (hub) |
| `2e5b29b9` | "bg_writer stall canary — self-reported mechanism" | 3 (specialist) |
| `87f4f82f` | "db_maintenance.py — background SQLite hygiene module" | 8 (hub) |
| `c3993ebf` | "recall_write_queue — fast atomic UPDATE drain" | 12 (super-hub) |

**What Anchor does today** (memory `0591813f`: *"93% of nodes never recalled, 5-7 hubs dominate every query"*):

Each seed gets equal probability mass (0.25). Spreading activation runs. Hubs accumulate mass from their many neighbors faster than specialists. After the kernel converges, top-5 results look like:

```
1. embed_queue.py — async-paced drain worker          (mass 0.41)
2. recall_write_queue — fast atomic UPDATE drain      (mass 0.37)
3. db_maintenance.py — background SQLite hygiene      (mass 0.33)
4. bg_writer stall canary — self-reported mechanism   (mass 0.19)  ← buried
5. Phase 5 recall hot-path read-only                  (mass 0.17)  ← hub-adjacent
```

The most directly-relevant specialist (`2e5b29b9` — literally about stall canaries) ranks **4th**, behind three hubs that activation found by transitive reachability.

**What HippoRAG-weighted seeds would do**: weight initial seed mass by `1/degree(node)`:

| Seed | Degree | Raw weight | Normalized |
|---|---|---|---|
| `0509fd16` (embed_queue) | 10 | 0.10 | 0.18 |
| `2e5b29b9` (stall canary) | 3 | 0.333 | **0.59** ← specialist gets dominant mass |
| `87f4f82f` (db_maintenance) | 8 | 0.125 | 0.22 |
| `c3993ebf` (recall_write_queue) | 12 | 0.083 | 0.15 |

The specialist node starts with 59% of total mass. Spreading activation runs from that starting distribution. Top-5 after kernel converges:

```
1. bg_writer stall canary — self-reported mechanism   (mass 0.42)  ← now top
2. embed_queue.py — async-paced drain worker          (mass 0.28)
3. db_maintenance.py — background SQLite hygiene      (mass 0.22)
4. recall_write_queue — fast atomic UPDATE drain      (mass 0.18)
5. STALL_THRESHOLD_S = EMBED_DRAIN_INTERVAL * 3       (mass 0.15)  ← neighbor of #1
```

The most-relevant node ranks 1st. Its specific neighbor (`STALL_THRESHOLD_S`) also surfaces because the kernel naturally walks from the specialist outward.

**Why this matters for our substrate specifically**: this directly addresses the **measured** hub-dominance problem (`0591813f`: 93% never recalled). HippoRAG validates the technique with a neuro-plausible justification (rarer phrases carry more identifying power — IDF). We already have all the inputs: `GraphDAL` can return degree in one query; the kernel reads seeds.

**Implementation sketch** (~0.5 day, in `brain_recall.py` spread-activation kernel):

```python
# Today (roughly):
seeds = {node_id: 1.0 / len(seeds) for node_id in seed_ids}

# Borrowed:
degrees = graph_dal.get_degrees(seed_ids)  # one query
raw_weights = {nid: 1.0 / max(1, degrees[nid]) for nid in seed_ids}
total = sum(raw_weights.values())
seeds = {nid: w / total for nid, w in raw_weights.items()}
```

That's the entire change. Behind it, the existing kernel runs unchanged.

---

#### 7.10.3 MemMachine's nucleus expansion

**Scenario**: Anchor surfaces a node — *"Operator wants concise responses (id: n7q8r9s0)"* with `source_refs: [trace_5234]`. At render time (Phase A step 6), we want to expand that source so Anchor sees the moment that produced the node.

**The actual conversation** in `trace_events` looked like:

```
trace_5232 (s0/K, user_message, Tom):
  "Your last few responses were too verbose — lots of preamble before the actual answer."
trace_5233 (s0/delta, assistant_message, Anchor):
  "Got it. What format would you prefer?"
trace_5234 (s0/K, user_message, Tom):  ← the one referenced
  "Be more concise."
trace_5235 (s0/K, user_message, Tom):
  "Like 2-3 paragraphs max for this kind of design conversation. More for code."
```

**What Anchor does today** (Phase A step 6 as currently designed): fetch the specific `trace_5234` and render it inline below the node:

```
[node]
type: preference
title: Operator wants concise responses
content: Tom prefers shorter replies, especially for design conversations.
source: Tom said: "Be more concise."   ← only the nucleus
```

The render is technically correct. But the FULL meaning lives in the four-turn arc — *"verbose preamble"* (the complaint), *"what format"* (Anchor's response), *"be more concise"* (the directive), *"2-3 paragraphs for design, more for code"* (the nuance). Sparseness discipline (decision 13 — 1-3 refs per node) means we keep `source_refs = [trace_5234]` — that's the right call structurally. But we're missing the surrounding context.

**What nucleus expansion would do**: for each source_ref, fetch the nucleus AND ±N adjacent trace_events from the same `chain_id`. N=2 → 5 total traces per ref. Render the cluster as a mini-excerpt:

```
[node]
type: preference
title: Operator wants concise responses
content: Tom prefers shorter replies, especially for design conversations.
source (nucleus + adjacent):
  Tom: "Your last few responses were too verbose — lots of preamble before
       the actual answer."
  Anchor: "Got it. What format would you prefer?"
  Tom: "Be more concise."  ← nucleus
  Tom: "Like 2-3 paragraphs max for this kind of design conversation.
       More for code."
```

Now Anchor sees the full arc: WHY (verbose preamble), the conversational turn-taking, the nucleus directive, AND the nuanced constraint. The preference is grounded in the moment that produced it.

**Why this matters for our specific substrate**: source_refs are deliberately sparse (decision 13). That's biology-aligned — dentate gyrus pattern separation depends on sparse indices. But sparseness creates a gap at retrieval: the index points at the right spot, but the *cortical pattern* that gives meaning lives in surrounding context. MemMachine's nucleus expansion is exactly the **joint reactivation** mechanism Teyler-Rudy described (1986) — index reactivates the cortical pattern, both fire together.

We already have all the ingredients: `chain_id` partitions traces by conversation turn-batch; `created_at` orders them; `brain.get_traces([ids])` does point/batch lookup. The expansion is a render-layer move, not an encode-layer move — sparseness discipline at encode stays intact.

**Implementation sketch** (~0.5 day, in `contract.py:render_rich_node` source_refs expansion):

```python
def _expand_source_refs(brain, refs: List[int], expand_window: int = 2) -> List[Dict]:
    """For each ref, fetch the nucleus + ±N adjacent traces in the same chain."""
    nucleus_rows = brain.get_traces(refs)  # already shipped in d68bddc
    if not nucleus_rows:
        return []
    expanded = []
    for nuc in nucleus_rows:
        chain_id = nuc['chain_id']
        nuc_id = nuc['id']
        # Pull adjacent in the same chain, ordered by id
        cluster = brain._trace_dal.get_chain_window(
            chain_id=chain_id,
            center_id=nuc_id,
            window=expand_window,
        )
        # Mark which one is the nucleus for render emphasis
        for row in cluster:
            row['is_nucleus'] = (row['id'] == nuc_id)
        expanded.extend(cluster)
    return expanded
```

Plus a small `TraceDAL.get_chain_window(chain_id, center_id, window)` method — one query, indexed on `(chain_id, id)`. Budget-bound: default `window=2`, configurable via render format constants. Skip if cluster would exceed `meta_limit`.

---

Each of these three is ~0.5 day. None require architectural change — they slot into existing layers (S2 Consolidation encoder, spread-activation kernel, render path). All three are immediately measurable: A.U.D.N. via consolidation outcome counts, node specificity via the 93%-never-recalled metric, nucleus expansion via source-fidelity probe (§4.2).

### 7.11 Two patterns we explicitly do better

1. **Identity layer at the embedding tier.** Every system above treats speaker as metadata field at best (Graphiti `actor`, Mem0 user partitioning) or implicit. Our per-utterance speaker binding *at the embedding layer* — concrete identity tokens fused into vectors — is a categorically different bet, and it's what makes partnership asymmetry (Operator/Anchor) representable rather than just labelled.

2. **Correction as first-class edge substrate, not invalidation timestamp.** Graphiti's `t_invalid` answers "when did this stop being true?" but loses "why" and "who corrected whom." Our 22-verb `correction_improvement` aspect on multi-relation edges, walked on every canonical pull and rendered into surface output, makes correction visible to Anchor at every turn — not just queryable on demand. Bi-temporal is a database feature; aspect-tagged corrections are an identity feature.

---

## Part 8 — Insights & open questions (added end of session)

Captured from the reflective close-out. Not new research — internal observations that emerged from reading everything else side-by-side. These should be addressable over time as we work through Phase B+ and the eval block.

### 8.1 Insights worth carrying forward

**I1 — The field is rediscovering McClelland (1995) and Teyler-Rudy (1986) without crediting them.** Five different brand names (HippoRAG, A-MEM, AriGraph, MemMachine, HINDSIGHT) converge on the same dual-store + structural-retrieval template biology figured out 30 years ago. The architectural convergence is mechanically driven by LLM constraints (context window, embedding cost, conversation length), not by reading the neuroscience. Implication: our biology-first framing is a real strategic asset — most labs don't know they're rediscovering 1995. We do.

**I2 — Identity isn't actually the locus of the hardest open problem.** We've spent this session on identity architecture and we're 12–18 months ahead. But the field's open problems are *source attribution* (33-year-old framework, uncited in any LLM memory paper), *composition under budget* (§16.0; unnamed in published work), *labile reconsolidation* (Tonegawa-Josselyn recall-update window), and *cross-session contradiction handling*. Identity is one of those problems; we've solved it; the others remain.

**I3 — We've built the substrate to be measurable and then didn't measure it.** This is the single most uncomfortable observation from the synthesis. `source_refs` exists specifically to bound extraction-loss; we never ran the bound. Identity stamping exists specifically to measure source-attribution decay; we never wrote the probe. The pattern: substrate work feels productive (commits, tests, migrations); eval-against-substrate doesn't feel like work even though it's where the architectural claim earns or loses. Worth encoding as a discipline lesson — saved as `feedback_measure_what_you_built`.

**I4 — The competitive lead is integration depth, not novel architecture.** The five distinctive moves (concept-cell tokens, per-utterance FK, asymmetric subject, render reconstruction, field-vs-observer perspective) are architecturally novel as a combination, but each piece exists somewhere in the field. What's actually hard to copy is *living inside an agent's real workload across years of accumulated sessions*. Mem0 / Zep / Letta ship as libraries someone integrates. Anchor IS the agent's brain. That's the moat.

**I5 — Our composition-under-budget thesis (§16.0) is genuinely original and time-bounded.** The field will arrive at "composition" once it finishes solving episodic memory. ~12–18 month window. Strategic implication: if we want to claim the framing publicly, it's a now-or-never window. Otherwise someone else will publish the same answer in 2027 and we'll be inside the convergent crowd.

### 8.2 Open questions (addressable over time)

**Q1 — What's the actual extraction-loss bound?** Our nodes are abstractions; S0 traces are verbatim. We assume the trade is worth it. MemMachine's 93.0% LongMemEvalS by keeping raw episodes challenges that assumption. **Test**: take 20 nodes anchored to specific traces, ask Sonnet to answer factual questions from each, score against (a) the node alone, (b) the trace alone, (c) the node + nucleus-expanded trace. The accuracy gap is the extraction-loss bound. If the gap is small, extraction wins. If large, we should lean harder on source_refs and treat nodes as indexes.

**Q2 — How often does source attribution actually fail?** Pick 50 nodes with source_refs. Ask Anchor "who told you this?" — score whether the surfaced trace's `human_identity` matches Anchor's reported source. Cryptomnesia rate. Predict: low for recent traces, higher for old ones (Johnson-Hashtroudi-Lindsay's source-monitoring decay). Establishes the baseline curve we'd defend against architecturally.

**Q3 — Does identity survive a model upgrade?** When Claude 4.7 → 4.8 lands, do self-reference answers stay coherent? Concrete-token embedding (decision 19) predicts yes; abstract-slot systems would predict drift. This is the killer test that would prove or kill the biology-grounding claim. **Methodology**: ~30 self-reference questions, baselined now, re-run after model swap, score for semantic-similarity + stylistic-consistency.

**Q4 — Do identity-load-bearing nodes survive S2 consolidation?** Aspect taxonomy says `identity_bearing` types (principle, identity, vision, rule, partner) should be protected from archive-similar merges. When Consolidation runs in 6 months, can we measure that no `identity_bearing` node was archived as a duplicate? Need a probe that walks the archive log + aspect classifier.

**Q5 — Is the spreading-activation kernel earning its complexity?** A/B against a simpler PPR + HippoRAG node-specificity kernel on the longmem cohort. If PPR ties or beats, we've been paying complexity for nothing. **Bound this**: PPR with our existing edge_relations and degree calculations. ~1 day to run, decisive answer.

**Q6 — When (not if) we hit the labile-reconsolidation gap, what's the design?** §16.1 names the direction (recall opens a window for update) but doesn't spec a mechanism. A-MEM mutates K/G/X on linked notes; Healer fills missing fields. Open question: should the labile window apply to (a) all recalled nodes, (b) only nodes flagged as `hypothesis` / `tension`, (c) only nodes whose source_refs have updated traces?

**Q7 — Does our aspect taxonomy dual-role tension actually cause problems in practice?** Structural routing (Frame placement) and semantic classification (LLM-judge cues) want different things. Today they share 14 aspects. **Open**: do classification edge cases manifest as Frame misplacements? If yes, the dual-role is a real bug. If no, it's a theoretical tension and we should leave it alone.

**Q8 — When will the first non-Tom partner appear, and what breaks?** Per-utterance binding is designed for this. Render-time reconstructive frame is designed for this. Frame's current_partner context is designed for this. None of it has been exercised. Even one synthetic test session with a fake "Alice" partner would reveal what's actually wired.

### 8.3 Simplifications worth considering (validate before acting)

**S1 — Replace spreading-activation kernel with PPR + node specificity.** See Q5. If A/B says simpler is comparable, retire the complexity.

**S2 — Split aspect taxonomy into structural-routing aspects vs semantic-classification aspects.** Only if Q7 surfaces real bugs. Premature otherwise.

**S3 — Move encoder source_refs writing out of the encoder prompt** and into a post-encode pass that walks node content for `[trace:N]` markers. Keeps the encoder prompt simpler (no source_refs discipline rules); puts the linking logic in code where it's testable. Speculative — would need to see prompt complexity grow before this earns its keep.

### 8.4 The discipline lesson

The pattern surfaced in I3 — *building measurable substrate and then not measuring it* — deserves to be a behavioral rule, not just an observation. **Going forward**: when a substrate change ships specifically to enable a measurement, the measurement gets a task on the next session's plan automatically. Don't let substrate sit un-measured. The architecture isn't real until the eval runs.

Saved to memory as `feedback_measure_what_you_built` so it travels across sessions.

---

## Reading list (sources referenced)

**Biology / neuroscience**:
- Quian Quiroga et al. — "Invariant visual representation by single neurons in the human brain" (Nature 2005); "Concept cells: building blocks of declarative memory" (Nat Rev Neurosci 2012); "20 years of concept cells" (Neuron 2026).
- Ison, Quian Quiroga & Fried — "Rapid encoding of new memories by individual neurons in the human brain" (Neuron 2015).
- Northoff et al. — "Self-referential processing in our brain" (NeuroImage 2006); Qin & Northoff — meta-analysis (NeuroImage 2011).
- Johnson, Hashtroudi & Lindsay — "Source monitoring" (Psychol Bulletin 1993).
- Sutin & Robins — "When the 'I' looks at the 'me'" (Memory 2008); Nigro & Neisser — "Point of view in personal memories" (Cog Psychology 1983); Rice & Rubin — "I can see it both ways" (Memory 2009).
- Eichenbaum — relational memory work (multiple); Davachi — binding-of-item-and-context (multiple).
- Hershfield — future-self continuity (PNAS 2011, multiple).
- Schacter & Addis — "Constructive episodic simulation" (Phil Trans Roy Soc B 2007).
- Teyler & Rudy — hippocampal indexing theory (Hippocampus 2007); McClelland, McNaughton & O'Reilly — complementary learning systems (Psych Review 1995).

**Field landscape**:
- Pink et al. — *"Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents"* (arxiv:2502.06975, Feb 2025).
- *"Persistent Identity in AI Agents: A Multi-Anchor Architecture"* (arxiv:2604.09588, Mar 2026).
- *"Sophia: A Persistent Agent Framework of Artificial Life"* (arxiv:2512.18202, Dec 2025).
- Hsieh et al. — *"Examining Identity Drift in Conversations of LLM Agents"* (arxiv:2412.00804).
- Li et al. — *"Measuring and Controlling Persona Drift in Language Model Dialogs"* (arxiv:2402.10962).
- *"Agent Identity Evals"* (arxiv:2507.17257, Jul 2025).
- *"Collaborative Memory: Multi-User Memory Sharing in LLM Agents"* (arxiv:2505.18279, May 2025).
- MemMachine (arxiv:2604.04853); MemPalace (arxiv:2604.21284); AriGraph (arxiv:2407.04363); MIRIX (arxiv:2507.07957).
- HippoRAG-v2; *"Hippocampus: Efficient Scalable Memory Module"* (arxiv:2602.13594); A-MEM (arxiv:2502.12110).
- Letta blog — *Memory Blocks*, *Stateful Agents*, *Agent Memory*.
- Anthropic — *Exploring model welfare*; Amanda Askell on Claude character; Nov 2025 weight-preservation commitments.
- Janus / Repligate — cyborgism.wiki/janus.
- Karpathy LLM Wiki + extensions (gists, MemClaw).
- Simon Willison — agent-definitions tag.

**Anchor internal references**:
- `docs/EPISODIC-REFERENCES.md` §0 (execution log), §1 (decisions), §15.1 (resolved identity), §5.3 (render templates), decision 19 (revised).
- `feedback_loud_at_write_boundary` memory (`~/.claude/projects/-Users-tpac-brain/memory/`).
- `feedback_no_sqlite3_cli_against_live_brain` memory (incident lesson).
