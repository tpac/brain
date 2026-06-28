# Pilot Gold Corpus — Adjudication Questions (independent Opus review)

## OVERALL ASSESSMENT
The protocol is **sound and the blind+hindsight design is working** — across all 8 cards the judges genuinely classified by *reasoned helpfulness* over topic-proximity, and the hardest mechanic (the time-leak / same-session-echo discipline) was applied correctly and verifiably. I spot-checked ~30 nodes against the live brain: every cited `created_at` ≤ cutoff held, every "essential" node actually expresses what its card claims, and the dropped same-session echoes (0094, 0118) and future-leakage nodes (0387, 0118) are correctly identified. **The single biggest risk is not correctness but CALIBRATION INCONSISTENCY**: the essential/silver bar drifts across cards (0205 and 0131 promote "would have strengthened/grounded" nodes to essential; 0094 and 0387 hold a much stricter "absence actually hurt the move" bar and leave essential thin or empty). If left unreconciled, the gold's `essential` label means different things in different rows, which will bias any precision metric computed against it. Second risk: several cards lean heavily on `encode_gap`, and the gap-vs-recall-miss boundary is a judgment call that needs one explicit rule. The corpus is trustworthy as a *draft* pending the calibration decisions below.

---

## MUST-ANSWER (affects whether we trust / scale the gold)

**Q1 — [cross-card] — Is the bar for `essential` "absence materially hurt the move" (strict) or "presence would have strengthened/grounded the move" (loose)? Pick ONE.**
- This is the meta-question under half the others. Evidence of drift:
  - 0094: holds STRICT — drops 3 topically-perfect nodes, keeps 2; notes the kept ones are needed because their content was "not in front of the model."
  - 0387: holds STRICT — `essential:[]` because the decisive knowledge was discovered in-turn, not recallable.
  - 0205: holds LOOSE — promotes `681ae2dc`/`94738c5a` to essential while its own notes say "the stored knowledge legitimizes/strengthens the move rather than being strictly indispensable" and rates the card medium-confidence.
  - 0131: holds LOOSE — promotes `a33b3d88`/`ce4787f1` to essential for a move that was *primarily a web-literature validation*; the brain nodes would have "let the move say 'we already mapped this'" (strengthen), not unlocked it.
- options: A) Strict everywhere (absence hurts) — would demote 0205's two and likely 0131's two to silver. B) Loose everywhere (would-strengthen counts) — would force re-examining 0094/0387 for dropped silvers wrongly excluded. C) Keep a two-tier essential ("decisive" vs "strengthening") explicitly in the schema.
- **My lean: A (strict).** The whole point of a lens-independent oracle is "would its ABSENCE have hurt." "Would have strengthened" is exactly the topic-proximity trap re-entering through the back door. Re-grade 0205 and 0131 essentials as silver unless absence is defensible.

**Q2 — [anchor_turn_0205] — Should `681ae2dc` (file line limits) be essential, given its specific numbers are stale and contradict the moment?**
- nodes: `681ae2dc` (file line-limits list), `94738c5a` (stale-guardrails bump-with-approval).
- judge's call: both essential; "the discipline and bump-vs-split rule are the unlock, the stale number doesn't disqualify it."
- why contestable: I verified `681ae2dc` — it says `daemon_config.py: <100 lines` and was written 3mo ago; the moment's guard was `<160`→`175`. The node's *numbers* are wrong for the moment and the bump-vs-split heuristic it carries ("raising the limit is fine for a small feature") is also present in `94738c5a`. Two essentials for one decision that the judge itself calls non-indispensable looks like over-crediting.
- options: A) Demote `681ae2dc` to silver, keep `94738c5a` essential (it names `daemon_config_is_small` specifically + the bump-with-approval history). B) Keep both essential. C) Demote both to silver (the whole decision was a same-session echo + approval already granted).
- **My lean: A.** `94738c5a` is the on-point one (names the exact test, carries the blessed-pattern history); `681ae2dc` is redundant + stale and should be silver.

**Q3 — [anchor_turn_0205] — Is this cue even worthwhile, or is the decisive content a same-session echo + already-granted approval?**
- judge's call: worthwhile=true, medium confidence; notes "much of the immediate decision was a same-session echo... the operator had already approved ('yes :)')."
- why contestable: The two paths (bump vs compress), the dated-ledger pattern, the docstring-rationale tradeoff, AND Anchor's recommendation were ALL stated in the recall_moment; Tom said "yes :)". By the protocol's own STEP-1 same-session-echo test this is close to a CUT. The stored knowledge only adds prior-art *confidence*.
- options: A) Keep worthwhile=true (the bump-with-approval convention is genuine cross-session knowledge that de-risks the move). B) Flip to worthwhile=false (echo + pre-granted approval; a candidate to cut from the gold). C) Keep but down-weight to a low-value worthwhile.
- **My lean: A, narrowly.** The convention "size guards are stale thresholds you bump with approval, not real gates" is real cross-session knowledge that changes the move's *confidence* — but this is the weakest "worthwhile" of the 8 and should be tagged low-value so it doesn't anchor the bar.

**Q4 — [anchor_turn_0387] — Is `essential:[]` the right call for the content-graft time-leak, and is the handling of `a1364fc9` correct?**
- nodes: `a1364fc9` ("revise() is node-only") — current content says "As of commit 5c5204e (2026-06-07) edges CAN be revised"; `6ad62f8e`/`dedd977a`/`7b6b6301` (post-cutoff or recovery-reconstructed).
- judge's call: essential=[]; `a1364fc9` dropped because at the 16:42 cutoff its content was the OPPOSITE (it would have *confirmed* the connect+disconnect instinct, not corrected it); the three direct hits are future leakage.
- why contestable: I verified all four. `a1364fc9`'s live content does describe `revise_edge`/commit 5c5204e (grafted after cutoff); `6ad62f8e` is `src:recovery:trace_reconstruct` and marked "RESOLVED as of 5c5204e"; `dedd977a` is the decision from this very arc. The reasoning is correct — BUT it hinges on inferring `a1364fc9`'s *content-at-cutoff* from its commit references, since `revised_at` was not bumped when the body was edited (a known brain bug). That inference is sound but is the single most fragile judgment in the whole pilot.
- options: A) Accept essential=[] + the cutoff-state reasoning (this is the model behavior we WANT — punish content-grafted nodes). B) Treat `a1364fc9` as ineligible-by-policy (any node whose content provably changed after cutoff is dropped without trying to reconstruct its old state) and record essential=[] for that simpler reason. C) Reconstruct the actual pre-cutoff content from trace/episode history before finalizing.
- **My lean: B as the standing RULE, A's outcome for this card.** We can't trust `revised_at` (the graft didn't bump it), so "content references a post-cutoff commit ⇒ ineligible" is a cleaner, mechanical, defensible policy than asking each judge to reconstruct historical content. Same answer here, far more scalable. (See Q11 for the general policy.)

**Q5 — [anchor_turn_0387] — Are the 4 encode_gaps real, or is the knowledge encoded-but-unrecalled?**
- judge's call: 4 gaps (rename_relation exists; reclassify built for this; reclassify parked; not on MCP surface) — all "not encoded at cutoff."
- why contestable: I searched the brain with the cutoff filter for the reclassify-renames-in-place mechanism — NO pre-cutoff node expresses it; the only relative is the 2026-04 migration (`544bafaf`, correctly silvered). So the gaps are genuinely not-yet-encoded as of cutoff. They ARE encoded *now* (`6ad62f8e` etc., post-cutoff) — i.e. this exact turn's discovery became nodes ~3h later.
- options: A) Accept all 4 as encode_gaps (correct at cutoff). B) Collapse the 4 into 1 gap ("the reclassify/rename_relation edge-revise capability and its non-exposure") — 4 bullets over-counts one discovery. 
- **My lean: A on substance, B on form.** The gaps are real; but 4 near-identical bullets inflate the gap count and will distort any "gap rate" metric. Merge to 1-2.

**Q6 — [operator_msg_0094] — Were the 3 same-session-echo nodes correctly dropped, and is `f2573550` rightly NOT even silver?**
- nodes dropped: `7eb36c8e` (TTL redesign), `d27e6bdc` (letter no-TTL), `f2573550` (iso_after sibling) — all created `2026-06-05T17:04:54`, ~4 min before the 17:08:43 cutoff.
- judge's call: drop all three (not even silver) — they are the S1-Scribe encoding of THIS session's turns 2-5, content already verbatim in recent_context; crediting them rewards the lens.
- why contestable: I confirmed the exact timestamps via filter_nodes. `f2573550` literally names the `iso_after` tie-in the outcome implements — under a topical lens it's the single best hit, which is *precisely* why dropping it is the right but aggressive call. This is the cleanest demonstration of the protocol working; it deserves an explicit sign-off as the reference precedent.
- options: A) Endorse the drop as the canonical same-session-echo precedent (cite it in the protocol). B) `f2573550` should be silver (it crisply names the mechanism and its full rationale wasn't all in-context). 
- **My lean: A.** Same-session S1-Scribe echoes of the very plan being executed are the trap the gold exists to avoid. Endorse and make this the worked example in the protocol.

**Q7 — [operator_msg_0118] — Is `79811089` truly the lone essential, and was a stronger "recency=zero-weight" node wrongly silvered?**
- nodes: `79811089` (freshness/unified_score R@8 regression, essential); `a35ed242` ("recency carries zero weight in pool", silver); `951f3ac8` (RRF flagged in April, silver, cited by ID in-conversation).
- judge's call: one essential (`79811089`, the scar the operator is overriding, full content not in front of model); `a35ed242` silver; `951f3ac8` demoted to silver because cited by ID in-turn.
- why contestable: I verified `79811089` (created 2026-05-30, content = −10 R@8, line 1712, "freshness applied uniformly" cause) — genuinely essential and its detail is NOT in the truncated moment. BUT `a35ed242` directly contradicts the operator's load-bearing premise "we have recency there so we might be good" (it proves recency currently carries ZERO weight). A node that contradicts the operator's stated assumption arguably belongs in essential, not silver.
- options: A) Keep as-is (one essential). B) Promote `a35ed242` to essential too — it would have corrected the operator's "we might be good on recency" assumption that the doc encoded uncritically. C) The move deliberately de-scoped recency, so silver is right.
- **My lean: B.** The strongest hindsight signal is a node that would have CHANGED the move; `a35ed242` would have flagged that "recency already covered" is false. That is more decisive than `79811089` (which the move handled correctly anyway). At minimum this is a real essential-vs-silver fork.

**Q8 — [cross-card] — How do we treat hand-SEEDED identity/standard nodes (src:anchor:seed) as gold candidates?**
- nodes: `89bb94ed` ("don't declare victory on a single metric", `src:anchor:seed`, locked) silvered in 0475; same family as the partnership/identity seed cluster excluded in 0475.
- why contestable: `89bb94ed` is a hand-seeded generic principle that would plausibly surface for *any* eval/verification move. Crediting seeds as silver/essential risks rewarding the system for surfacing evergreen scaffolding rather than earned, situation-specific memory. 0475 silvered it; 0131 and others did not lean on seeds.
- options: A) Seeds are eligible like any node (helpfulness is helpfulness). B) Seeds are eligible only when situation-specific, never as generic-principle silver (they'd surface for everything). C) Tag seed-sourced credits separately so we can measure with/without.
- **My lean: B.** A locked evergreen principle surfacing for an eval move is closer to "always-on prior" than "recall win." Don't count generic seeds as gold; keep situation-specific seeds eligible.

**Q9 — [cross-card] — Is the same-session-echo rule being applied CONSISTENTLY across the 4 cards where it bites?**
- 0094: drops 3 echoes hard (strict). 0118: drops 2 echoes (`6a964255`,`703a9402`) hard (strict). 0387: prev turns were a *different* sub-arc (keywords), no echo issue. 0205: acknowledges heavy echo but still keeps 2 essentials (loose). 
- why contestable: 0094 and 0118 treat "content already in the moment ⇒ drop, not even silver." 0205 treats "content already in the moment + approval granted" but still credits essential. That is the same echo condition resolved two different ways.
- options: A) Apply 0094/0118's strict echo rule to 0205 (→ demote or cut). B) Accept that 0205's echo is about the *decision* while the credited nodes are the *cross-session convention* behind it (a real distinction). C) Add an explicit echo-test line to the protocol and re-run the 4 affected cards against it.
- **My lean: B + C.** 0205's distinction is legitimate (the convention is older than the session), but the inconsistency is real enough that the protocol needs one written echo-test so judges stop resolving it ad hoc.

**Q10 — [cross-card] — `encode_gap` vs `recall_miss`: several cards log as "gap" knowledge that the system arguably HAD in some form. One rule needed.**
- Examples: 0087 logs the self_message-session_id bug as an encode_gap (cutoff 15:54); I confirmed the encoded form (`24175c71`,`6f838cfe`,`57091d0c`) was created 17:04 — ~70 min LATER, so correct *at cutoff*. 0191 logs the per-turn-ping-vs-boot-fingerprint timing as a gap while crediting `0f51ee9e` (which has the persistence model but not the trigger map). 0475 logs "blindness/anti-anchoring" as a gap while the *parent* methodology (independent review) IS encoded.
- why contestable: All three are defensible as "the SPECIFIC claim wasn't encoded," but the granularity differs: 0087's gap is a whole missing node; 0475's gap is a missing *facet* of an encoded node. Measuring "encode-gap rate" across these is apples-to-oranges.
- options: A) Two gap subtypes: `missing_node` (no node covers the topic) vs `missing_facet` (topic encoded, the specific load-bearing detail isn't). B) Only `missing_node` counts as a gap; missing facets are just "silver node was imperfect." C) Leave as one bucket.
- **My lean: A.** The distinction is real and cheap, and it stops 0475-style facet-gaps from inflating the same metric as 0087-style true gaps. Without it the gap rate is uninterpretable.

---

## NICE-TO-ANSWER (refinements, lower stakes)

**Q11 — [protocol] — Adopt a mechanical "content-graft ⇒ ineligible" rule because `revised_at` is untrustworthy?**
- The 0387 case proved a node's body can be edited (to reference a post-cutoff commit) WITHOUT bumping `revised_at`. So cutoff-eligibility by `created_at`/`revised_at` alone is unsafe — a node can pass the timestamp gate while its *content* is post-cutoff.
- options: A) Add rule: "if a node's content references events/commits/dates after the cutoff, it is ineligible regardless of timestamps." B) Rely on per-judge reconstruction (status quo). C) Fix the `revised_at`-not-bumped bug upstream and re-derive eligibility.
- **My lean: A now, C eventually.** A is a cheap guardrail that makes the gold robust against the graft bug today; C is the real fix but out of scope for the pilot. This generalizes Q4.

**Q12 — [operator_msg_0131] — Is the cog-sci lens being under-credited as encode_gap while graph-CS gets essential, creating a lens imbalance?**
- judge's call: graph-CS mapping → essential (`a33b3d88`,`ce4787f1`); cog-sci grounding → silver (`788942af`) + the named sources (Collins & Loftus, ACT-R fan effect) logged as encode_gap.
- why contestable: The move's headline is "all THREE fields converged on degree-normalization." Crediting only the graph-CS leg as essential, with cog-sci as gap, encodes an asymmetry that reflects the brain's coverage, not the move's structure. Defensible (the brain genuinely has more graph-CS depth) but worth a conscious sign-off.
- options: A) Accept the asymmetry (it reflects real coverage). B) Demote graph-CS to silver too so the lenses are treated evenly (the move was web-validation across all three). 
- **My lean: A.** The asymmetry is honest — the brain really did pre-map the PPR/degree-normalization claim and did NOT hold the cog-sci citations. This is a correct *use* of hindsight, not a bug.

**Q13 — [anchor_turn_0087] — Is promoting the two cross-cutting conventions over the topically-nearest self-channel nodes the right essential call?**
- nodes: essential = `451e7a91` (close-the-class), `dec_bq3n` (config "parameters not decisions"); self-channel schema/type nodes (`549c43d9`,`8f7df29d`) silvered.
- judge's call: the engineering conventions are decisive because the outcome's two moves are a consolidation refactor + a config-driven TTL design; self-channel nodes are silver (Anchor had the live contract in front of it).
- why contestable: I verified both. `451e7a91` is genuinely the close-the-class principle Tom asked for ("consolidated, no spaghetti"); `dec_bq3n` is the canonical "make parameters not decisions" + the `boost_rule` example the prompt cites. Strong calls. The only quibble: `dec_bq3n` is a very high-access hub (access_count 2738) — is it essential because it's decisive, or because it's a hub that surfaces broadly? Its content IS exactly on-point, so likely decisive.
- options: A) Endorse both essentials. B) Demote `dec_bq3n` to silver (hub that would surface anyway; the config pattern is corroborated by silvers `8a9010f5`/`60dfd488`). 
- **My lean: A.** The prompt literally cites the `get_config('boost_rule',1.3)` pattern; `dec_bq3n` is that node. Decisive, not just a hub.

**Q14 — [anchor_turn_0475] — Is the "blindness/anti-anchoring" encode_gap a real gap or just a finer facet of encoded methodology?**
- judge's call: gap — "the catalog has parallel-multi-lens and independent-stateless-review nodes but none names the anti-anchoring blindness rule."
- why contestable: I searched; confirmed no node states "keep reviewers blind to the author's conclusion." But `05ed74e9`/`1d65f4fb`/`67064d68` collectively cover independent/blind review. This is a `missing_facet` (per Q10), not a `missing_node`.
- options: A) Keep as gap (the specific anti-anchoring discipline is genuinely unencoded). B) Reclassify as missing_facet under Q10's scheme. C) Drop the gap (parent methodology suffices).
- **My lean: B.** Real but it's a facet-gap, not a node-gap — tag it accordingly so it doesn't inflate the true-gap count.

**Q15 — [operator_msg_0191] — Single essential `0f51ee9e` — right, or should a recovery-cluster node join it?**
- judge's call: one essential (`0f51ee9e`, persistent-daemon-restart convention); recovery cluster (`c219791d` etc.) dropped as "an altitude above the one-line reassurance"; the exact trigger map is an encode_gap.
- why contestable: `0f51ee9e` (verified, locked, 3mo) carries "daemon loads modules at startup, code not picked up until restart" + the 3-levels model — genuinely load-bearing for the reassurance. The trigger-map gap (which events recompute the fingerprint) is correctly a gap. Clean, conservative call.
- options: A) Endorse single essential. B) Add a recovery node to essential (the "recovery only when unreachable" row of the outcome's table). 
- **My lean: A.** The recovery nodes are about crash-loop/circuit-breaker machinery, not the boot-vs-per-turn timing the move actually needed. Conservative single-essential is right.

**Q16 — [protocol] — Cards don't record WHICH lens found each node beyond `lens_tags`; can we verify the multi-method search requirement was met?**
- STEP 3 mandates wide, multi-method search (recall + recall_episodes + get_nodes + filter_nodes). Cards carry `lens_tags` per node (cos_cue/cos_outcome/fts/graph/browse) but no record of *negative* searches (what was tried and found nothing). For encode_gaps especially, we can't see whether the judge searched hard enough before declaring a gap.
- options: A) Require judges to log the queries they ran for each encode_gap (proof-of-search). B) Trust lens_tags + judge_notes (status quo). C) Spot-audit gaps adversarially (what this review did).
- **My lean: A for gaps only.** An encode_gap is an assertion of absence; absence claims should carry their search receipts. Essential/silver are self-justifying (the node is cited); gaps are not. Cheap insurance against false gaps.
