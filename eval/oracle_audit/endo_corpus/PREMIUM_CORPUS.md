# Premium Corpus — 10 endo cues, qualitatively examined (2026-06-18)

Ten cues (5 operator + 5 anchor) from the frozen endo gold corpus, each hand-examined by an
Opus agent: recall more candidates, read the S0 traces before/after, examine the gold node's
fields, verify/enrich the gold, and analyze **what would have surfaced it** against the theory
(flat embedding; two diseases — within-cluster smear vs cue-far; the levers). Selection +
dossiers: `endo_premium_select.py` → `premium_seeds.json`.

---

## SYNTHESIS — what would actually have surfaced them (the payoff)

The aggregate eval said *"every read-side lever loses to plain cosine; the embedding is the wall."*
True for **abstract relevance** — but the per-case dive found that **a large fraction of misses are
recoverable WITHOUT a new embedding**, via cheaper levers the aggregate hid. The two diseases have
**different** cheap fixes:

**The two HITS won on lexical/token overlap, not relevance.** Both `operator_msg_1566` (cue restates
"Haiku tools recall") and `anchor_turn_1120` (rare token `tool_result`) hit rank 1 via topic-cosine +
exact-token/FTS overlap — and both gold nodes were **same-session encode→recall echoes** (the gold was
encoded *from the cue's own turn*, a near-verbatim mirror). Lesson: the embedding works only when a
**distinctive shared token** is present (FTS-like exact match); some apparent successes are inflated by
the echo. Recurring artifact across the corpus: the "perfect" node is often a **post-cutoff** node
(the encoded distillation of the cue's conversation, e.g. `e3366969`, `3c33fe45`) correctly excluded —
the genuine pre-cutoff prior is the real gold.

**Disease A (within-cluster smear — buried 6–25):** the pool is RIGHT, the ordering is wrong. Two cheap
levers, available now:
- **FTS rare-token reweight** — the discriminating token is often *already in the cue* but drowned by
  topic-cosine: `co_accessed` (cue 0363), `PageRank`/`HippoRAG` (cue 0132). A lexical lane that surfaces
  rare-token matches over topical cosine recovers these.
- **Proposition/type-aware reranker** over the top-K — cue 0968 is textbook: the gold is the only
  `architecture` design-doc among ~8 near-tied liveness siblings; a type/proposition reranker promotes it.

**Disease B (cue-far — rank >25, ~53% of misses):** one dominant pattern — **the gold is the ANSWER, the
cue is the QUESTION**, and the answer's vocabulary ≠ the question's. In *every* far case, seeding recall
from the **next-move** (or the user's very next turn) ranks the gold **top 1–7**:
- 0611: next-move seed → golds at rank 1–6 (vs 26). 0387: seed from the user's "you can't revise an edge?"
  → golds rank 1–7 (vs 64). 1014: the answer *inverts* the question ("can you rebuild?" → "servers/ = restart
  only") → next-move seed ranks gold #2 (vs 34). 0475: cue = the *findings being reviewed*, gold = the
  *review method* — orthogonal spaces, rank 348 from cue, **#1 from the next-move**.
- This is the strongest vindication yet of recall-is-prediction: **the gold lives where the conversation is
  GOING, not where it IS.** But it's the same lever the followup arm tried and failed (3–5%) — because
  *analogical* prediction (followup of past analogs) is too noisy on flat embeddings. We know the next-move
  direction is the right seed; we lack a *realizable* predictor of it. That's the sharpened open problem.
- Reranking is **hopeless** for Disease B (the gold is below the candidate pool — nothing to re-rank).

**Two untested cheap levers the dive surfaced:**
- **Score the cue against the node's `question` field.** Cue 1272's best gold (`bcfd063b`) has a `question`
  field near-verbatim the cue, while its content/situation are farther. Recall scores content/title/situation
  but not `question` — yet `question` is literally "what does this node answer," often the best cue-match.
- **Encode-side gaps:** several gold are under-encoded — `a33b3d88` has *no keywords*; broaden situations to
  cover the operator-facing cue class. Cheap to fix at write-time.

**Refined lever ranking (per disease, cheapest-first):**
- Disease A → (1) FTS rare-token reweight, (2) type/proposition reranker, (3) better embedding.
- Disease B → (1) next-move/predictive seed [needs a realizable predictor], (2) `question`-field vector;
  reranker useless.
- Cross-cutting → populate `keywords`/`question`, add `question`-vector to scoring.

**Teacher-gold corrections found:** add `953ab4f7` (essential, cue 0611), `6ad62f8e` (cue 0387),
`bcfd063b` (cue 1272), `8a9a9393` (cue 0475) — several teacher picks were incomplete because the better
node wasn't in the nomic-generated candidate union (the gold-bias the bench stream flagged).

---

## THE 10 ENTRIES

### operator_msg_1566 (operator, factual, baseline rank 1 — HIT)
Gold `460b1cb1` ("Haiku tool set — three tools remain"). A vocabulary-dense factual cue against a
vocabulary-dense factual node — the cue restates the gold's whole topic ("tools haiku use… recall… nodes
and query"), so topic-cosine + literal token overlap ("Haiku","tools","recall") carry it. The rare regime
where flat-embedding topic-resolution is exactly enough. Reliable for cues that *name the artifact*; would
not carry the softer "see the nodes/query haiku used" (a capability, not a single node).
**Verdict:** factual cue restating a fact node's exact topic — topic-cosine + token overlap suffices.

### anchor_turn_1120 (anchor, design, baseline rank 1 — HIT)
Gold `9934b73a` ("tool_result asymmetry"). Hit via the rare compound token `tool_result` + a **same-session
encode→recall echo**: the gold was encoded *from this very turn* (shares "82% recall-echo poison",
`EAGER_TRACE_REF_TYPES` verbatim), so cue and gold are lexical mirrors. Reliable-but-partly-lucky: a
distinctive code identifier is a high-precision FTS/embedding anchor, but the near-verbatim alignment is the
echo. **Verdict:** design cue that is a near-verbatim lexical mirror of its gold; exact-token cosine+FTS on a
distinctive identifier puts it at rank 1.

### operator_msg_0363 (operator, factual, baseline rank 16 — buried)
Gold `d1d1a90c` ("co_accessed edges excluded at all degrees"). Disease B with A on top: the cue's surface is
the *correction* ("I didn't want it on main… didn't discuss lane meshing") — process language; "co_access"
appears once, the gold's dense vocabulary (traversal/excluded/Hebbian) zero times — so cosine anchors near the
lane-meshing cluster. Then 7 near-confusable co_accessed siblings smear it. **Lever:** FTS on the rare token
`co_accessed` (in the cue!) + situation-field cosine + access-count tiebreak within cluster. (The "perfect"
node `e3366969` is post-cutoff — the encode of this turn — correctly unreachable.)
**Verdict:** the cue carries "co_access" but cosine buried it under the placement topic; FTS5 on `co_accessed`
lifts the genuine prior.

### anchor_turn_0132 (anchor, compositional, baseline rank 6 — buried)
Gold `a33b3d88` ("spread kernel may be over-engineered vs pure PPR"). Textbook Disease A: a dozen cosine-near
spreading-activation nodes; the gold is dead-center topically but is a low-confidence (0.56), low-access,
**keyword-less** hypothesis competing with higher-degree community/mechanism twins. **Lever:** FTS on
`PageRank`/`HippoRAG` (the only node carrying them; cue uses them verbatim) + encode-side (add keywords) +
proposition reranker. (Post-cutoff "perfect" node `3c33fe45` = the next turn's encode.)
**Verdict:** the only prior naming PPR/HippoRAG sits keyword-less inside a dozen cosine-bunched spread nodes;
an FTS hit on the rare token separates it.

### operator_msg_1272 (operator, factual, baseline rank 18 — buried)
Gold `6303cc1b` ("max_messages=20") — but better gold found: `bcfd063b` ("encoder already flexible — 5 is
cadence not window") and `98133f0d` ("10 exchanges not 10 messages"), both missing from the candidate union.
Hybrid B+A: the cue is an under-specified yes/no *question* ("5 new / 10 context?") whose answer-node is keyed
by the *resolved* vocabulary. **Lever:** query/seed (next-move framing ranks gold #4 + pulls the better nodes);
**cue↔`question`-field cosine** (bcfd063b's `question` is near-verbatim the cue) — the cleanest field-level fix;
encode-side (broaden situation).
**Verdict:** cue-far question whose answer-node is keyed by resolved vocabulary; seed from the next-move or match
the cue against the `question` field.

### anchor_turn_0968 (anchor, action, baseline rank 7 — buried)
Gold `8ff5b19f` ("TRACE-NODE-RESOLUTION.md design"). Unambiguous Disease A: recall on the cue puts the gold
cluster at the very top — gold is highly cue-near but tied among 6–10 confusable liveness/archived siblings;
it's the lone `architecture` design-doc among `mechanism`/`principle`/`incident` twins. **Lever:** a
proposition/**type-aware reranker** over the (already-correct) top-15 — boost the design-doc when the cue asks
"what's the mandatory design." NOT a query/seed problem (next-move lands in the same cluster).
**Verdict:** the gold is the only design-doc among ~8 near-tied liveness siblings; a type/proposition reranker
over the right pool is the lever.

### operator_msg_0611 (operator, compositional, baseline rank 26 — far)
Gold `a4037bad` + `0f1ebbf6` + (missing) `953ab4f7` (Haiku 4096 cache floor). Disease B: the cue is a
*diagnose-the-timeout* mandate (symptom language); the gold is the *caching-threshold conclusion* three
reasoning hops downstream. Cue-recall returns the session's own fresh findings (same-session echoes).
**Lever:** seed from the next-move → golds at rank 1–6; FTS on `4096`/`cache_control`; episodic bridge on
"Haiku turn analysis" (this session resumes that deferred thread). Embedding alone won't fix (genuinely
different propositions). **Verdict:** the gold is the *answer* to the diagnosis, not the *words* of the
request; next-move seed surfaces all three at 1–6, and `953ab4f7` should be essential.

### anchor_turn_0387 (anchor, factual, baseline rank 64 — far)
Gold `a1364fc9` ("revise() is node-only / revise_edge exists"). Disease B: the cue (msg [23]) proposes a
connect+disconnect workaround in *seed-pack cleanup* vocabulary; the bridging phrase "revise an edge" appears
only in the *user's next turn* [24]. **Lever:** seed from the user's following turn ("you can't revise an
edge?") → gold cluster ranks 1–7 — i.e. the **predictive-episodic lever (seed from the anticipated next turn)
is the fix**. The better node `6ad62f8e` post-dates the cutoff; `a1364fc9` is the right call for what was
encodable. **Verdict:** the answer is invisible to the cue's cleanup vocabulary but ranks 1–7 when seeded from
the very next turn.

### operator_msg_1014 (operator, action, baseline rank 34 — far)
Gold `a3227733` ("deploy model: servers/ = restart only, not rebuild"). Disease B where **the answer inverts
the question**: cue asks "can you rebuild?", gold answers "you usually shouldn't." Cue-cosine bunches it with
merge/build-status siblings. **Lever:** query/seed — "do servers/ changes need rebuild or restart" ranks gold
#2; FTS on `servers/`. Reranker useless (gold below the top-25 pool). **Verdict:** an inversion the flat
embedder can't see; seeding from the next-move's reasoning ranks it #2.

### anchor_turn_0475 (anchor, procedural, baseline rank 348 — far, extreme)
Gold `2cc2be10` ("3-Opus-agent review method") + (missing) `8a9a9393`. The purest Disease B: cue = ~2,000 chars
of *partnership research content* (Lencioni/Gottman/Edmondson) with ZERO method vocabulary; gold = *the act of
reviewing*. Orthogonal regions → rank 348. **Lever:** next-move/predictive seed ("blind multi-Opus reviewers,
convergence") → gold #1; episodic pivot is strong (the two prior review *runs* are episodes — a recurring
*procedure* with episodic anchors even though the content differs). Reranker hopeless. **Verdict:** cue =
findings-being-reviewed, gold = review-method — orthogonal; rank 348 from cue, #1 from the next-move.
