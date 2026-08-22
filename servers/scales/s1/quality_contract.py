"""Encoder Quality Contract — 36-dimension measurement instrument.

Evaluates encoder output (live encoding, §7.6 worked examples, scan retros)
across 36 dimensions in 9 groups. The contract is MEASUREMENT-ONLY — it
doesn't shape encoder behavior. Anti-bias prompt rules live in the encoder
prompt (interactions table, 's1e'). Dimensions are yardsticks; the rule
distinction belongs in the prompt.

Group 9 (D33-D36) scores EXAMPLE-AUTHORING discipline — placeholder syntax
for IDs (connect_to titles, source_refs, node_id), ref↔conversation
consistency, voice annotation coverage, and turn↔node language divergence.
Examples are training data for the encoder; divergent turn-vs-node language
teaches Sonnet to extract structure rather than paraphrase. D33-D35 are
mechanically checkable at example load time via validate_example_authoring();
D36 is LLM-judged in the evaluator.

v29 note: trace_id was migrated from INTEGER to 8-char hex TEXT. Examples
no longer use a sentinel integer range — they use the same `<placeholder>`
syntax as connect_to titles. Unified placeholder discipline across all
ID-shaped fields.

Origin: consolidated from 5-lens design pass (biology, scan/empirical,
aspect taxonomy, recall mechanics, Tom's principles) + Addis & Szpunar
(2024) 3-D semantization model. See docs/EPISODIC-REFERENCES.md §7.6 for
the worked examples this contract evaluates.

Each dimension:
    intent          — what we measure
    satisfies       — positive signals (observable in encoder output)
    violates        — clear failures
    degrades        — partial failures
    interacts_with  — cross-dim dependencies
    lens            — which lens(es) surfaced this dim: B/S/A/R/T

Cross-dim rules name tensions between dimensions and how to resolve.
Recall gating section captures the recall-time expansion principle (refs
persist forever; recall decides expansion). Structural follow-ups list
gaps outside encoder scope (Phase B+ work).
"""

CONTRACT_VERSION = 4  # v4: catalog connect_to targets are ids (s1e v31, Option D) — placeholders become <id-of-...>-flavored, and a grounded catalog excerpt may carry literal hex ids. v3: D33 retired sentinel range, adopted placeholder syntax after trace_id migrated to 8-char hex (schema v29).

import re

# Placeholder pattern for example ID fields. Any string of the form
# `<descriptive-name>` qualifies; the angle brackets are the structurally
# unmissable signal Sonnet recognizes as "fill this in," not "copy verbatim."
EXAMPLE_ID_PLACEHOLDER_RE = re.compile(r'^<[^<>]+>$')


# ═══════════════════════════════════════════════════════════════
# 36 DIMENSIONS
# ═══════════════════════════════════════════════════════════════

DIMENSIONS = {

    # ─── Group 1: Recall mechanics (will this surface correctly?) ───

    'D1_title_as_handle': {
        'group': 'recall_mechanics',
        'intent': 'Pattern-shaped noun phrase that compresses across queries; the recall handle, not the event description',
        'satisfies': [
            'noun-phrase or compressed claim (not a full sentence)',
            'would surface on >=2 distinct semantic queries differing from literal text',
            'concrete enough that FTS5 porter stemming brings query terms to it',
            '<=80 chars (PRE_EDIT title_limit)',
            'survives 6-month relevance test',
        ],
        'violates': [
            'event-shaped ("Phase B-B complete", "X shipped")',
            'self-referential to another node_id',
            'generic with no specific term ("Architecture decision")',
            'commit-shaped (hashes, sprint IDs as primary content)',
        ],
        'degrades': [
            'title length >60 without verbatim content (overpacked)',
            'duplicates content opening (wastes weight 1.00 + 0.85 on same words)',
            'acronym-only (porter stemming gap)',
        ],
        'interacts_with': ['D5', 'D9', 'D29'],
        'lens': ['B', 'S', 'R', 'T'],
    },

    'D2_situation_drives_high_meta': {
        'group': 'recall_mechanics',
        'intent': 'Situation field names specific recall trigger including emotional/relational register; feeds high_meta (weight 0.70)',
        'satisfies': [
            'observable trigger conditions an LLM at recall could match',
            'emotional/contextual cues when moment had them',
            'for type=open nodes: names the resumption trigger explicitly',
        ],
        'violates': [
            'empty or missing situation',
            'restates title in different words',
            '"when debugging X" over-broad framing',
        ],
        'degrades': [
            'single-word triggers (not discriminating under cosine)',
            'duplicates their_raw_quote (kills high_meta variance)',
            'narrower than the claim',
        ],
        'interacts_with': ['D5', 'D15', 'D29'],
        'lens': ['B', 'S', 'R'],
    },

    'D3_atomization_one_claim': {
        'group': 'recall_mechanics',
        'intent': 'One retrieval intent per node; different aspects = different atoms',
        'satisfies': [
            'single claim that would surface for one kind of cue',
            'sibling-atomization (principle pulled out, evidence stays)',
            'dense tables become pure-reference (not paraphrased into mush)',
        ],
        'violates': [
            'bundled unrelated claims',
            'multi-aspect content jammed into one type',
            'comparison table flattened into prose',
        ],
        'degrades': [
            'two concepts where second is incidental',
            'multiple examples in content when one suffices (dilutes _primary)',
        ],
        'interacts_with': ['D4', 'D22', 'D29'],
        'lens': ['B', 'S', 'A', 'R'],
    },

    'D4_top2_avg_discipline': {
        'group': 'recall_mechanics',
        'intent': 'At least two strong fields per node (recall averages top-2 group cosines)',
        'satisfies': [
            'title AND situation both populated and specific',
            'optional fields earn their group (high_meta or other_meta)',
        ],
        'violates': [
            'strong title with empty situation — 1.00-weight slot half-wasted',
            'all signal concentrated in one field',
        ],
        'interacts_with': ['D1', 'D2', 'D16'],
        'lens': ['R'],
    },

    # ─── Group 2: Source fidelity (does the substrate trust this?) ───

    'D5_voice_verbatim_or_empty': {
        'group': 'source_fidelity',
        'intent': 'their_raw_quote and my_raw_quote are exact substrings of source OR empty. The verbatim contract is sacred.',
        'satisfies': [
            'exact substring of source turn in user_messages or agent_messages',
            'typos preserved (e.g. "supression", "thats", "ill")',
            'punctuation preserved including "?!?!", ":)", trailing ellipsis',
            'emoji preserved',
            'field empty when no verbatim phrase is load-bearing',
        ],
        'violates': [
            'polished restatement of operator framing',
            'fabricated quote not present in conversation window',
            'encoder-authored "quote" that IS the title restated',
        ],
        'degrades': [
            'quote falls outside available window but plausible (audit-blind-spot)',
            'fuzzy match >=0.7 (light cleanup of filler words)',
        ],
        'interacts_with': ['D6', 'D7', 'D13'],
        'lens': ['B', 'S', 'R', 'T'],
        'empirical_baseline': '34-38% verbatim, 34% not_found in 50-node probe — floor ~10-20% violations',
    },

    'D6_reasoning_grounded': {
        'group': 'source_fidelity',
        'intent': 'Reasoning field traces to specific turns; carries theory-of-mind about operator state',
        'satisfies': [
            'cites specific turn-evidence ("Tom asked three times across consecutive turns")',
            'hedges where conversation hedged',
            'names what triggered a reframe',
            'preserves the relational vector (who held which position)',
        ],
        'violates': [
            'hallucinated self-diagnosis ("Anchor caught its own misalignment" when it did not)',
            'generic boilerplate ("This is important")',
            'reasoning duplicates content (wastes other_meta variance)',
        ],
        'interacts_with': ['D5', 'D12'],
        'lens': ['B', 'S', 'R', 'T'],
        'note': 'Reasoning lives in other_meta (weight 0.40). Earns place via render (S1 Scribe, Healer), not score lift.',
    },

    'D7_anchor_voice_symmetry': {
        'group': 'source_fidelity',
        'intent': 'my_raw_quote populated when Anchor\'s stance IS the encoding (corrections, decisions, identity-bearing)',
        'satisfies': [
            'first-person commitment in identity-bearing nodes',
            'verbatim self-correction ("You caught me — I had the rounds backwards")',
            'symmetric emission rate with their_raw_quote on correction/decision nodes',
        ],
        'violates': [
            'empty on identity-bearing/correction nodes when Anchor visibly reflected',
            'smuggled self-quote that IS the content restated',
        ],
        'degrades': [
            'first-person mentioned in reasoning instead of quoted in my_raw_quote',
        ],
        'interacts_with': ['D5', 'D14'],
        'lens': ['S', 'T', 'B'],
        'empirical_baseline': 'their_raw_quote ~80% Era E; my_raw_quote ~6%. Backwards.',
        'structural_followup': 'Phase B+ encode-time validation (prompt alone failed Era E)',
    },

    # ─── Group 3: Type, lifecycle, semantization ───

    'D8_type_matches_aspect': {
        'group': 'type_lifecycle',
        'intent': 'Type resolves into the aspect whose meaning matches content\'s semantic claim',
        'satisfies': [
            'aspects.by_node_type(type) returns aspect whose meaning describes what content is doing',
            'earned-through-experience content in lesson_insight',
            'timestamp-bound content in episodic_anchor',
            'declared/axiomatic content in identity_bearing',
            'open work in active_thread',
            'novel/emergent types sit coherently among an existing aspect\'s members',
        ],
        'violates': [
            'type=plan with executed content',
            'type=open with resolved body',
            'identity_bearing type for one-time event',
            'type in noise aspect for real knowledge',
        ],
        'degrades': [
            'type plausible across two aspects (ambiguous content)',
            'less-specific type when better fits ("fact" when "pattern" matched)',
        ],
        'interacts_with': ['D9', 'D11', 'D19', 'D21'],
        'lens': ['B', 'S', 'A', 'R', 'T'],
    },

    'D9_semantization_gradient': {
        'group': 'type_lifecycle',
        'intent': 'Place node along episodic↔semantic gradient using Addis & Szpunar 3-D coordinates (temporal specificity × content format × self-reference)',
        'satisfies': [
            'schema-fit + atemporal + repeated → pure synthesis, low/no refs',
            'perceptual + specific + arousing → pure reference, dense refs',
            'self-relevant + repeated + identity-bearing → anchored synthesis (both)',
        ],
        'violates': [
            'schema-violating content prematurely abstracted as principle (van Buuren 2022 — schema violations stay episodic longer)',
            'arousing/PE-laden content stripped to semantic gist (loses episodic anchor)',
            'N=1 evidence promoted to principle',
        ],
        'interacts_with': ['D8', 'D25', 'D31', 'D32'],
        'lens': ['B', 'S', 'T'],
        'research_basis': 'Addis & Szpunar 2024, Rouhani 2023, Sinclair & Barense 2024, van Buuren 2022',
    },

    'D10_lifecycle_marker': {
        'group': 'type_lifecycle',
        'intent': 'Ephemera carries explicit "will go stale" signal',
        'satisfies': [
            'event_time set on time-bound nodes',
            'half_life metadata or end-condition stated',
            'rejected at encode time when transient',
        ],
        'violates': [
            'status-as-fact ("Phase B-B complete" with type=fact)',
            'commit-hash-as-node',
            'phase-completion-as-principle',
            'action-this-week checklist as type=decision',
        ],
        'interacts_with': ['D9', 'D18'],
        'lens': ['S', 'T'],
        'structural_followup': 'S2Healer stale-node extension (aspect-resolved detection)',
    },

    'D11_revise_audits_type_and_title': {
        'group': 'type_lifecycle',
        'intent': 'Content changes can invalidate the original type AND title assignment',
        'satisfies': [
            'title updated when content semantics shift',
            'type re-classified when aspect of content drifts',
            'explicit acknowledgment of revision lineage',
        ],
        'violates': [
            'half-revised: title says "running", content says "CLOSED"',
            'type unchanged when content aspect shifted',
            'trailing_correction buried as parenthetical instead of edge',
        ],
        'interacts_with': ['D8', 'D23'],
        'lens': ['S', 'A'],
    },

    # ─── Group 4: Partnership & voice (Tom's lens) ───

    'D12_partnership_attribution': {
        'group': 'partnership_voice',
        'intent': 'Names WHO acted; preserves relational vector between operator and Anchor',
        'satisfies': [
            'situation names the actor ("When Tom corrected X")',
            'quotes attributed by stance (pushback / concession / directive)',
            'reasoning preserves who held which position',
        ],
        'violates': [
            '"we decided" or "it was found" — collapses two agents into mush',
            'quotes detached from speaker',
            'lesson framed as universal when it was specifically Tom-correcting-Anchor',
        ],
        'interacts_with': ['D5', 'D7', 'D13'],
        'lens': ['T'],
    },

    'D13_pushback_preservation': {
        'group': 'partnership_voice',
        'intent': 'Terse pushback/disagreement survives verbatim. Anti-sycophancy preserved.',
        'satisfies': [
            'terse pushback preserved verbatim including typos and lowercase',
            'content acknowledges friction (not collaborative consensus)',
            'reasoning notes "Tom rejected X — strongly" when it happened',
        ],
        'violates': [
            'friction softened to consensus ("Tom suggested" when quote was "no")',
            'sanitized punctuation or case in quotes',
            'lesson phrased as mutual when it was course-correction',
        ],
        'interacts_with': ['D5', 'D12'],
        'lens': ['T', 'S'],
        'empirical_baseline': '~80% of correction/decision nodes preserve this. Strongest cross-era capability — v19 must NOT lose it.',
    },

    'D14_identity_not_tool_voice': {
        'group': 'partnership_voice',
        'intent': 'Node reads as Anchor remembering, not a database row',
        'satisfies': [
            'first-person register where reflection occurred',
            'identity-bearing types carry held belief, not bureaucratic prose',
            'content frames partnership stake when warranted',
        ],
        'violates': [
            'third-person clinical write-up of natural first-person moment',
            'my_raw_quote empty when Anchor visibly reflected',
            'principle node phrased as manual entry',
        ],
        'interacts_with': ['D7', 'D12'],
        'lens': ['T', 'B'],
    },

    'D15_action_disposition': {
        'group': 'partnership_voice',
        'intent': 'Node makes future-Anchor DO something, not just KNOW',
        'satisfies': [
            'title or content frames behavior change ("Refuse to X when Y")',
            'situation triggers a re-firing condition',
            'corrections carry ASSUMED / REALITY / PATTERN — pattern is the actionable hook',
        ],
        'violates': [
            'pure summary with no behavioral consequent',
            'lessons phrased as facts with no "do this differently"',
        ],
        'interacts_with': ['D1', 'D2'],
        'lens': ['T'],
    },

    # ─── Group 5: Emergent expression ───

    'D16_open_fields_earn_slot': {
        'group': 'emergent_expression',
        'intent': 'Open kv field maps to a scoring group OR is referenced in render',
        'satisfies': [
            'field carried by EMBEDDING_GROUPS or render path',
            'adds dimension beyond core fields',
            'correction_pattern, gated_intent, emotion_label when state warrants',
        ],
        'violates': [
            'orphan to _emergent at weight 0.40 with no consumer',
            'structural-only data (counters, IDs encoded as text)',
            'invented for one node, never reused',
            'duplicates situation or content',
        ],
        'degrades': [
            'long free-text >300c truncated mid-thought (EMBEDDING_FIELD_CHAR_LIMIT)',
        ],
        'interacts_with': ['D2', 'D19'],
        'lens': ['R', 'S', 'T'],
    },

    'D17_terminology_in_text': {
        'group': 'emergent_expression',
        'intent': 'Inside-language used precisely in title/content (FTS5 porter stemming indexes title+content)',
        'satisfies': [
            'terms used with codebase/SKILL.md meanings',
            'capitalized appropriately',
            'naming-event preserved when operator named over assistant',
        ],
        'violates': [
            '"Anchor" / "Frame" used loosely',
            'assistant\'s term wins silently when operator named differently',
        ],
        'interacts_with': ['D6'],
        'lens': ['S', 'R'],
    },

    'D18_temporal_anchoring': {
        'group': 'emergent_expression',
        'intent': 'event_time on time-bound nodes; conversation-time not wall-clock',
        'satisfies': [
            'event_time present where temporal sequence matters',
            'uses conversation_now-style values (per docs CLAUDE.md time-window architecture)',
            'Allen interval relations when sequencing matters',
        ],
        'violates': [
            'missing event_time on event-typed nodes',
            'wall-clock dates during S1/S2 (eval replay corruption)',
        ],
        'interacts_with': ['D10', 'D21'],
        'lens': ['B', 'S', 'A', 'R', 'T'],
    },

    'D19_emergence_not_ceremony': {
        'group': 'emergent_expression',
        'intent': 'Type/relation/field chosen because it fits, not from catalog favorite. No type-laundering, no field-padding.',
        'satisfies': [
            'type names actual aspect of moment (a vow → "commitment", not "lesson")',
            'edge relations specific ("supersedes", not "related_to")',
            'open kv adds dimensionality beyond core fields',
        ],
        'violates': [
            'everything is "lesson" / "principle" regardless of texture',
            '"related_to" anywhere a specific verb fits',
            'open kv duplicates situation or content',
        ],
        'degrades': [
            'novel type that is synonym of high-traffic existing',
        ],
        'interacts_with': ['D8', 'D16', 'D20', 'D21'],
        'lens': ['T', 'A'],
    },

    # ─── Group 6: Graph integration ───

    'D20_edge_text_discriminates': {
        'group': 'graph_integration',
        'intent': 'edge_description names the specific bridge; spread activation gates on compose_edge_text(relation, description) cosine vs query',
        'satisfies': [
            'description is a clause about the relationship ("corrects the assumption that X")',
            'query-relevant terms in description',
            'a reader following the edge can predict what they will find at the other end',
        ],
        'violates': [
            'empty description',
            'restates target title',
            'generic relation + empty description ("related")',
        ],
        'degrades': [
            'correct direction but generic ("provides context")',
        ],
        'interacts_with': ['D21', 'D24'],
        'lens': ['B', 'S', 'A', 'R', 'T'],
    },

    'D21_edge_relations_match_aspect': {
        'group': 'graph_integration',
        'intent': 'Relation verb belongs to the correct aspect (correction_improvement vs extension_refinement vs explanation_causation vs temporal_sequence)',
        'satisfies': [
            'corrects/supersedes/reframes on correction-shaped edges',
            'extends/refines/elaborates on elaboration',
            'depends_on/enables/requires on prereq',
            'temporal verbs on time-order claims',
        ],
        'violates': [
            'related_to for what is actually correction (loses structural-lineage ride-along, loses correction_enrich attach)',
            'caused_by for time-order only',
            'extends where content invalidates target',
        ],
        'degrades': [
            'verb fits aspect but more specific verb in same aspect was available',
            'novel verb not yet in aspects (works until AspectIntegration runs)',
        ],
        'interacts_with': ['D8', 'D20', 'D23'],
        'lens': ['B', 'S', 'A', 'R'],
    },

    'D22_cross_aspect_reach': {
        'group': 'graph_integration',
        'intent': 'Node\'s edges touch >=2 distinct aspects when content warrants — not monochromatic graph integration',
        'satisfies': [
            'lesson connects via extension_refinement to source AND via validation_evidence to test',
            'episodic_anchor connects to lesson_insight (upward) AND lateral temporal',
            'correction connects back to corrected fact via aspect-tagged edge',
        ],
        'violates': [
            'all edges resolve to one aspect (5 edges all "extends")',
            'all edges resolve to generic_relation or noise',
        ],
        'interacts_with': ['D3', 'D21'],
        'lens': ['B', 'A', 'S'],
    },

    'D23_self_correction_chain': {
        'group': 'graph_integration',
        'intent': 'Corrections carry walkable lineage via correction_improvement aspect edge + revise on target node',
        'satisfies': [
            'edge to corrected node with verb from correction_improvement aspect',
            'edge_description names what specifically changed',
            'prior belief acknowledged in content or reasoning',
            'revise on target when content is now wrong',
        ],
        'violates': [
            'floating new node silently contradicting old',
            'generic "extends" when truth was "corrects"',
            'corrects edge written without revise on target (graph claims wrong, node holds wrong)',
        ],
        'degrades': [
            'edge present but edge_description vague ("updates this")',
        ],
        'interacts_with': ['D11', 'D20', 'D21'],
        'lens': ['T', 'A', 'B'],
    },

    'D24_multi_aspect_pair_disambiguation': {
        'group': 'graph_integration',
        'intent': 'Multi-aspect verbs have edge_description that disambiguates which membership is meant',
        'satisfies': [
            'corrects edge (correction_improvement AND temporal_sequence) — description signals which',
            'evolved_into, resolved_by, updates similarly disambiguated',
        ],
        'violates': [
            'multi-aspect relation with empty or ambiguous description',
        ],
        'degrades': [
            'description leans one way but does not make other implausible',
        ],
        'interacts_with': ['D20'],
        'lens': ['A'],
        'note': 'Known multi-aspect strings: corrects, updates, revises, resolved_by, rejected_for, triggered, triggers, revealed, evolves_from, evolved_into, consolidated_into, case, resolution',
    },

    # ─── Group 7: Episodic substrate (Phase B+ enables; encoder pre-conforms) ───

    'D25_source_ref_present_when_substrate_exists': {
        'group': 'episodic_substrate',
        'intent': 'Anchored-synthesis nodes carry source_refs to load-bearing turns when substrate is available',
        'satisfies': [
            'preference/correction/moment with >=1 ref to revealing turn',
            'correction with refs to BOTH mistake-trace AND correction-trace',
            'verbatim quote in content paired with source_ref to the turn',
        ],
        'violates': [
            'anchored-shape node with empty source_refs while originating turns are in user_content',
            'verbatim quote with no source_ref to the turn it quotes',
        ],
        'degrades': [
            'source_refs present but pointing at tangential turn rather than load-bearing',
        ],
        'interacts_with': ['D7', 'D26', 'D27', 'D32'],
        'lens': ['B'],
        'recall_gate': 'Recall-time age/relevance decides expansion (refs persist forever per decision 25)',
    },

    'D26_sparse_anchoring': {
        'group': 'episodic_substrate',
        'intent': '1-3 refs per node typical, each anchoring a distinct aspect of the node',
        'satisfies': [
            'each ref is THE load-bearing turn for its aspect',
            '>=80% of nodes-with-refs carry 1-3 refs',
        ],
        'violates': [
            'node carries 10+ refs to "everything related"',
            'multiple refs on adjacent same-content turns',
        ],
        'degrades': [
            '4-5 refs where 2-3 would suffice',
        ],
        'interacts_with': ['D25', 'D27'],
        'lens': ['B'],
        'research_basis': 'Decision 13 (sparse hippocampal indices)',
    },

    'D27_engram_cohort_co_anchoring': {
        'group': 'episodic_substrate',
        'intent': 'Siblings of one episode share load-bearing refs so co_anchored fires structurally',
        'satisfies': [
            'siblings carrying refs load-bearing for THAT sibling, with real overlap',
            'cohort emerges from genuine ref-sharing, not artificial pre-allocation',
        ],
        'violates': [
            'encoder fragments one episode\'s refs across siblings to keep each ref unique',
            'siblings all share all refs (over-coupled, not cohort-shaped)',
        ],
        'interacts_with': ['D26'],
        'lens': ['B'],
        'research_basis': 'Decision 15 (co_anchored as structural engram), Tonegawa engram distribution',
    },

    'D28_identity_tokens_concrete': {
        'group': 'episodic_substrate',
        'intent': 'Concrete identity names (Tom, Anchor) not slot placeholders (the operator, the user, the assistant)',
        'satisfies': [
            'per-utterance speaker binding present in traces',
            'names appear in title/situation/content where identity matters',
        ],
        'violates': [
            'slot placeholders where decision 19 mandates concrete tokens',
        ],
        'degrades': [
            'names in title but slots leak into content/situation',
        ],
        'interacts_with': ['D14'],
        'lens': ['B', 'T'],
        'research_basis': 'Decision 19a (concept cells — Quian Quiroga), §15.1',
    },

    # ─── Group 8: Cognitive principles ───

    'D29_pattern_separation': {
        'group': 'cognitive_principles',
        'intent': 'Similar-but-distinct claims encoded as distinct atoms; recall can disambiguate',
        'satisfies': [
            'distinct embeddings via different situation/title',
            'sibling atomization when claims diverge in retrieval space',
        ],
        'violates': [
            'two-register atom too tightly fused',
            'repetitive_principle: Nth re-encoding with no link to prior',
        ],
        'interacts_with': ['D1', 'D2', 'D3'],
        'lens': ['B', 'S'],
        'research_basis': 'DG pattern separation (Appendix A)',
    },

    'D30_pattern_completion_enabled': {
        'group': 'cognitive_principles',
        'intent': 'Partial cues (terminology, situation, edge, source) can navigate to this node',
        'satisfies': [
            'contrastive_evidence preserved (rejected option AND chosen named)',
            'before/after preserved with concrete numbers',
            'symptom_to_fix pairing',
        ],
        'violates': [
            'heavy reliance on internal jargon without self-explanation',
            'opaque content reachable only from one cue type',
        ],
        'interacts_with': ['D22', 'D25'],
        'lens': ['B', 'S'],
        'research_basis': 'CA3 pattern completion (Appendix A)',
    },

    'D31_no_premature_consolidation': {
        'group': 'cognitive_principles',
        'intent': 'Single-session insight typed as lesson/observation; principle requires multi-session evidence',
        'satisfies': [
            'language matches scope — "across sessions", "consistently", "every time" for principles',
            'single-occurrence insights typed lesson or finding',
        ],
        'violates': [
            'N=1 evidence promoted to principle',
            'orphan abstraction with empty source_refs from one-turn promotion',
        ],
        'degrades': [
            'insight uses principle-ish language but correctly typed',
        ],
        'interacts_with': ['D8', 'D9', 'D25'],
        'lens': ['B'],
        'research_basis': 'CLS slow cortical consolidation (McClelland 1995)',
    },

    'D32_affective_pe_register': {
        'group': 'cognitive_principles',
        'intent': 'Emotional arousal / prediction error / surprise triggers anchored synthesis with refs even when surface looks semantic',
        'satisfies': [
            'arousal-laden exchanges (pushback, release, frustration, breakthrough) anchored with refs',
            'surprise/reframe events anchored even when a clean lesson could be extracted',
            'corrections carry refs to BOTH mistake-trace AND correction-trace',
        ],
        'violates': [
            'surface-clean lesson extracted from arousal-laden exchange with refs dropped',
            'PE event encoded as static fact, losing the disruption signal',
        ],
        'degrades': [
            'refs to resolved outcome only, missing the moment-of-disruption',
        ],
        'interacts_with': ['D9', 'D13', 'D25'],
        'lens': ['B', 'T'],
        'research_basis': 'Rouhani 2023 (NE preserves episodic), Sinclair & Barense 2024 (PE → distinct encoding)',
    },

    # ─── Group 9: Example authoring discipline (training data quality) ───
    #
    # Examples are training data for the encoder. The example library teaches
    # Sonnet not just WHAT to write but HOW to read the source. These four
    # dims score the example's authorship discipline. D33-D35 are mechanical
    # (validate_example_authoring); D36 is LLM-judged in the evaluator.

    'D33_placeholder_syntax': {
        'group': 'example_authoring',
        'intent': 'Example ID-shaped fields (connect_to titles, source_refs, node_id) use bracketed placeholder syntax `<id-of-descriptive-name>` — never UNGROUNDED literal-looking values that Sonnet pattern-matches and copies into production output. Exception (v4, grounded_catalog_excerpt): a literal hex id in connect_to[].title is compliant when the example carries its own catalog excerpt and the id is copied from one of those headers — the visible source makes the taught behavior copy-from-catalog.',
        'satisfies': [
            'every source_refs entry matches `<...>` shape',
            'every connect_to[].title in examples uses placeholder syntax OR is a hex id copied from a catalog excerpt line shown in the same example',
            'every node_id in revise examples uses placeholder syntax',
            'every trace_id in source_conversation uses placeholder syntax',
        ],
        'violates': [
            'literal-looking hex trace_id in source_refs (Sonnet may emit it verbatim)',
            'real-looking title in connect_to (top failure mode, pre-v22)',
            'ungrounded hex id in connect_to (no excerpt line in the example carries it)',
            'hex node_id in revise examples (looks like a real catalog node)',
        ],
        'degrades': [
            'placeholder name too vague to teach what should be there (e.g. `<x>`)',
        ],
        'interacts_with': ['D34'],
        'lens': ['A'],
        'mechanical': True,
        'research_basis': 'v22 placeholder convention (commits 4f7845e + 507a806); v29 trace_id hex migration unified placeholder rule across all ID fields',
    },

    'D34_ref_internal_consistency': {
        'group': 'example_authoring',
        'intent': 'Every source_refs placeholder in encoder_output points to a turn placeholder present in the same example\'s source_conversation block',
        'satisfies': [
            'each source_refs `<...>` matches the trace_id `<...>` of some turn in source_conversation',
            'the ref points to a turn whose content load-bears the node it anchors',
        ],
        'violates': [
            'source_refs placeholder not found as trace_id in any source_conversation turn (orphan ref)',
            'ref to a turn the node does not actually anchor on',
        ],
        'degrades': [
            'ref to an adjacent context turn rather than the load-bearing turn',
        ],
        'interacts_with': ['D25', 'D26', 'D33'],
        'lens': ['A'],
        'mechanical': True,
    },

    'D35_voice_annotation_coverage': {
        'group': 'example_authoring',
        'intent': 'Examples with source_refs include a voice_annotations block documenting the load-bearing turns. Mechanical check: presence-only. The DEEP check — that each ref has substantive load_bearing prose — is LLM-judged in the evaluator alongside D36.',
        'satisfies': [
            'example with source_refs has a non-empty voice_annotations block',
            'voice_annotations entries name source_turn labels and load_bearing prose',
        ],
        'violates': [
            'source_refs present but no voice_annotations block exists in the example',
            'voice_annotations block present but completely empty',
        ],
        'degrades': [
            'voice_annotations entries exist but load_bearing prose is generic or skips the source_refs aspect',
            'voice_annotations covers their_raw_quote/my_raw_quote but ignores source_refs entirely',
        ],
        'interacts_with': ['D5', 'D7', 'D25'],
        'lens': ['A'],
        'mechanical': True,
    },

    'D36_turn_node_divergence': {
        'group': 'example_authoring',
        'intent': (
            'source_conversation turns and encoded node prose use DIFFERENT registers. '
            'Turns are operator/Anchor speech (messy, layered, redundant, specific). '
            'Node content/situation/reasoning are Anchor\'s extraction register — '
            'naming the hidden structural axis the turn implies but does not state. '
            'Verbatim quote fields (their_raw_quote/my_raw_quote) are the bridge: '
            'they preserve turn phrases unchanged. Specificity is preserved end-to-end: '
            'numbers stay numbers, ranges stay ranges, exact phrases stay verbatim — '
            'never smoothed into averages or paraphrases.'
        ),
        'satisfies': [
            'node content/situation/reasoning share <30% literal phrase overlap with any single turn',
            'node names a structural axis (mechanism, principle, tension, register) the turn implies but does not state',
            'numbers, ranges, exact phrases preserved unchanged where they appear in either side',
            'verbatim phrases that matter live in their_raw_quote/my_raw_quote, NOT duplicated into content prose',
            'turns within source_conversation are themselves diverse — different angles, not synonymic restatements of each other',
        ],
        'violates': [
            'node content is a tidy paraphrase of the operator turn (encoder learns to summarize)',
            'ranges flattened to averages or "approximately" smoothing',
            'numbers rounded into prose ("about 200" instead of "190")',
            'exact phrases re-stated in content instead of preserved in raw_quote fields',
            'source_conversation turns are clean restatements of each other (no real-talk redundancy/layering)',
        ],
        'degrades': [
            'some lexical overlap between turn and node content that could be tightened',
            'turn slightly cleaner than real exchange register would be',
        ],
        'interacts_with': ['D5', 'D7', 'D14', 'D32'],
        'lens': ['A'],
        'mechanical': False,
        'llm_judged': True,
        'research_basis': (
            'Tom 2026-05-25: "Examples of the references as well as turns themselves '
            'have influence of how S1E records semantic information... don\'t be primed '
            'by node to create a mirror text. Train encoder to look deep into text '
            'and find the hidden as well as remember simple stuff like numbers and '
            'ranges kept as ranges."'
        ),
    },
}


# ═══════════════════════════════════════════════════════════════
# CROSS-DIMENSION RULES
# ═══════════════════════════════════════════════════════════════
#
# Named tensions between dimensions and how to resolve them. These are the
# load-bearing architectural decisions — when satisfying dim X would harm
# dim Y, the cross-dim rule names which side wins.

CROSS_DIM_RULES = [

    {
        'name': 'CR1_title_compress_vs_verbatim',
        'rule': 'When title IS a load-bearing verbatim phrase, verbatim wins over compression. Compression discipline applies to titles that PARAPHRASE content.',
        'applies': ['D1', 'D5'],
        'example_good': [
            '"Cluster not node, recognition not search" — quote-as-title, longer ok',
            '"Single-writer invariant" — synthesis-as-title, must compress',
        ],
    },

    {
        'name': 'CR2_sparseness_vs_engram_cohort',
        'rule': 'Each sibling carries refs load-bearing for THAT sibling. Cohort emerges from real overlap, not pre-allocation.',
        'applies': ['D26', 'D27'],
    },

    {
        'name': 'CR3_atomization_vs_top2avg',
        'rule': 'Aspect-divergence IS a strong signal for retrieval-divergence. Split when aspects differ AND each atom can hit top2_avg threshold (situation + title both populated).',
        'applies': ['D3', 'D4'],
    },

    {
        'name': 'CR4_aspect_strictness_vs_emergence',
        'rule': 'Evaluator scores against aspects_v1.json snapshot at example-time. Novel types are degrade-not-violate — allow emergence while measuring current state.',
        'applies': ['D8', 'D19'],
    },

    {
        'name': 'CR5_locked_rule_conflict_path',
        'rule': 'Corrections over locked nodes require explicit escalation marker OR rejection-with-surface. Encoder-side rule + dispatch-layer gate (Phase B+ structural backup).',
        'applies': ['D8', 'D23'],
        'structural_followup': True,
    },

    {
        'name': 'CR6_voice_field_vs_voice_in_content',
        'rule': 'my_raw_quote = verbatim Anchor speech ONLY. First-person register in content/reasoning is separate (D14 lives in content, D7 lives in quote field).',
        'applies': ['D5', 'D7', 'D14'],
    },

    {
        'name': 'CR7_emergent_type_vs_aspect_membership',
        'rule': 'Encoder picks type by content fit; AspectIntegration validates downstream async. Evaluator uses aspect snapshot at example time, soft-passes novel-but-coherent types.',
        'applies': ['D8', 'D19'],
    },

    {
        'name': 'CR8_reasoning_low_weight_vs_groundedness',
        'rule': 'Reasoning lives in other_meta (weight 0.40) — earns place via render (S1 Scribe, Healer) not score lift. Still must ground in conversation.',
        'applies': ['D6'],
    },

    {
        'name': 'CR10_recall_time_ref_gating',
        'rule': 'Refs persist forever on node (decision 25, S0 retention guarantee). Recall decides expansion by age/query/format. Encoder writes faithfully; recall gates.',
        'applies': ['D25', 'D26'],
    },

    {
        'name': 'CR11_semantization_drift_via_recall_not_encoder',
        'rule': 'Nodes do not "graduate" off refs over time at the encoder layer. Recall-time gating expresses semantization. Vector healer (future) can re-promote pure-reference → anchored-synthesis as repetitions accumulate.',
        'applies': ['D9', 'D25'],
    },

    {
        'name': 'CR12_verbatim_bridge_vs_divergence',
        'rule': (
            'Verbatim quote fields (their_raw_quote / my_raw_quote) ARE the '
            'permitted bridge between turn register and node register. A phrase '
            'appearing both in a source_conversation turn AND in a node\'s '
            'their_raw_quote/my_raw_quote satisfies D5/D7 and does NOT '
            'violate D36 — the verbatim field is the legitimate place for that '
            'overlap. D36 fires when the same phrase is paraphrased into '
            'content/situation/reasoning prose instead.'
        ),
        'applies': ['D5', 'D7', 'D36'],
    },
]


# ═══════════════════════════════════════════════════════════════
# RECALL-TIME GATING (the graduation question)
# ═══════════════════════════════════════════════════════════════
#
# Biology says memories drift toward semantic over months. We don't copy
# biology — we control retrieval. Refs persist on the node forever; recall
# decides what to expand based on age, query, format.

RECALL_GATING = {
    'principle': (
        'source_refs persist forever on the node (decision 25, S0 retention '
        'guarantee). Recall-time logic decides which refs to expand and how '
        'much. The "graduation" of a node from anchored-synthesis toward '
        'pure-synthesis is expressed in the retrieval gate, not in the data.'
    ),

    'format_thresholds': {
        'SURFACE_FORMAT': (
            'Final render to additionalContext. Expand all refs by default. '
            'Per-ref truncation when budget tight; never drop the source entirely.'
        ),
        'HAIKU_FORMAT': (
            'Surface-selection Haiku call. Expand only refs whose cosine '
            'similarity to current recall query exceeds threshold (initial '
            'cosine >= 0.5, eval-tunable per docs/EPISODIC-REFERENCES.md §13.6).'
        ),
        'ENCODER_FORMAT': (
            'Encoder catalog view. No expansion — encoder uses source_refs at '
            'WRITE time (linking new nodes), not READ time.'
        ),
    },

    'age_decay': (
        'Optional future refinement: progressively tighter age window on ref '
        'expansion. Older refs stay structurally walkable but expand less often '
        'at recall. Implementation deferred — start with format thresholds only.'
    ),
}


# ═══════════════════════════════════════════════════════════════
# STRUCTURAL FOLLOW-UPS (outside encoder scope)
# ═══════════════════════════════════════════════════════════════
#
# Gaps that prompts and examples alone can't close. Phase B+ work, S2
# extensions, dispatch-layer validation. Listed for traceability — these
# are NOT scored by the contract.

STRUCTURAL_FOLLOWUPS = [
    {
        'id': 'voice_symmetry_validation',
        'applies_to_dim': 'D7',
        'work': 'Encode-time dispatch check: flag type=correction/decision/principle with non-trivial reasoning but empty my_raw_quote.',
        'phase': 'Phase B+',
    },
    {
        'id': 'quote_fidelity_validation',
        'applies_to_dim': 'D5',
        'work': 'Encode-time substring check against trace_events; source_refs as ground truth.',
        'phase': 'Phase B+',
    },
    {
        'id': 'reconsolidation_labile_recall',
        'applies_to_dim': 'D11',
        'work': 'Revise() is in-place; biology says recall makes traces labile and recalled-traces become NEW engrams. See EPISODIC-REFERENCES.md §16.1.',
        'phase': 'future',
    },
    {
        'id': 'locked_rule_escalation',
        'applies_to_dim': 'D23, D8',
        'work': 'Dispatch-layer gate on locked-node corrections — require escalation marker or surface to operator.',
        'phase': 'Phase B+',
    },
    {
        'id': 's2_consolidate_ref_union',
        'applies_to_dim': 'D25, D27',
        'work': (
            'When S2 Consolidate merges A → B, source_refs must union (dedupe '
            'by trace_id, preserve position order). Mixed-age delta is a '
            'potential quality signal (engram reach across time). For v1: just '
            'union + recall-time gating handles everything.'
        ),
        'phase': 'S2',
    },
    {
        'id': 'vector_healer_ref_repromotion',
        'applies_to_dim': 'D9, D25',
        'work': (
            'S2 unit that promotes pure-reference → anchored-synthesis as '
            'repetitions accumulate. Biology suggests drift; we express it via '
            'recall gating, but a vector healer could also re-encode content '
            'patterns over time. Deferred per recall-gating call.'
        ),
        'phase': 'future',
    },
    {
        'id': 's2_healer_stale_node_extension',
        'applies_to_dim': 'D10',
        'work': (
            'S2Healer scan for stale status nodes (aspect-resolved). Detects '
            'durable-aspect type with event-shaped content. See open node '
            '"S2Healer stale-node extension".'
        ),
        'phase': 'next',
    },
]


# ═══════════════════════════════════════════════════════════════
# EVALUATOR PROTOCOL
# ═══════════════════════════════════════════════════════════════
#
# How an evaluator agent applies the contract to a candidate example.

EVALUATOR_PROTOCOL = {
    'inputs': [
        'example.conversation — source turns the encoder saw',
        'example.encoder_output — remember_batch / brain_batch calls',
        'example.intent — which axis/dim this demonstrates',
        'aspect_snapshot — aspects_v1.json at example time',
    ],

    'per_dimension_output': {
        'status': ['satisfied', 'degraded', 'violated', 'n/a'],
        'evidence': 'specific field values that drive the status',
        'degradation_note': 'where the partial-fail lives (if degraded)',
    },

    'cross_dim_output': {
        'rules_applied': 'list of CR_n that fired',
        'resolution_applied': 'which side won, was the resolution correct?',
        'contradictions_found': 'list of (dim_a, dim_b, reason)',
    },

    'verdict_output': {
        'is_canonical': 'does this example teach intent without contradicting other dims?',
        'missing_demonstrations': 'dims this example COULD have shown but did not',
    },

    'output_format': 'JSON per the schema above',
}


# ═══════════════════════════════════════════════════════════════
# EVALUATOR SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════
#
# The prompt the evaluator agent uses to apply this contract to a candidate
# example. Kept here so contract + evaluator stay in sync. If/when this
# becomes a runtime interaction (e.g., S2 quality probe), seed from this
# constant per the prompt-sync discipline in CLAUDE.md.

# ═══════════════════════════════════════════════════════════════
# EXAMPLE AUTHORING CONVENTIONS
# ═══════════════════════════════════════════════════════════════
#
# Hard-won discipline: example `connect_to` targets MUST use placeholder
# syntax, not literal titles. This was learned the hard way in v20 —
# Sonnet pattern-matches the shapes it sees in the canonical training
# pattern AND §7.6 examples, including literal target titles. When
# example targets reference titles that aren't real catalog nodes (e.g.
# "Daemon TCP migration", "Brain vs database framing"), Sonnet copies
# them verbatim in production encoded output, producing
# `connect_to_unresolved` errors at write boundary.
#
# A prose disclaimer above the canonical pattern (v21) is insufficient
# — Sonnet's pattern-match is stronger than the disclaimer's restraint.
# The right fix is to make example target slots VISUALLY UNCOPYABLE.

EXAMPLE_AUTHORING_CONVENTIONS = {
    'placeholder_syntax': {
        'rule': (
            "Example `connect_to` targets MUST use bracketed placeholder "
            "syntax — `<id-of-descriptive-name>` — never a literal-looking "
            "title string. Same rule applies to any field in an example "
            "where the value should be resolved against the live catalog "
            "or computed at encode time. One exception: the grounded "
            "excerpt form below."
        ),
        'good': [
            '{title: "<id-of-related-architecture-decision>", relation: "grounds", why: "..."}',
            '{title: "<id-of-the-existing-framing-this-quote-anchors>", relation: "grounds", why: "..."}',
            '{title: "<id-of-the-prior-belief-being-corrected>", relation: "corrects", why: "..."}',
        ],
        'bad': [
            '{title: "Daemon TCP migration", relation: "grounds", why: "..."}',
            '{title: "Single-writer invariant beats clever concurrency", relation: "parallels", why: "..."}',
            '# Real-looking titles get pattern-matched and literal-copied',
        ],
        'rationale': (
            "Sonnet pattern-matches the shape it sees. Real-looking "
            "titles get literal-copied to production output even when "
            "wrapped in prose disclaimers. Bracketed placeholders are "
            "visually marked as illustrative — Sonnet's training "
            "recognizes <placeholder> patterns as 'fill this in,' not "
            "'copy verbatim.'"
        ),
    },
    'grounded_catalog_excerpt': {
        'rule': (
            "An example MAY use a literal hex id in `connect_to[].title` "
            "when the example carries its own catalog excerpt and the id "
            "is copied from one of those headers. The excerpt makes the "
            "copy relationship part of the demonstrated behavior — the "
            "model learns copy-from-catalog, not invent-a-plausible-value. "
            "An ungrounded literal hex (no excerpt line carrying it) is a "
            "violation, same class as a literal title."
        ),
        'rationale': (
            "s1e v31 (Option D): catalog targets are ids, and the s2 "
            "encoders prove bare-hex examples are safe exactly when the "
            "id's source is visible in the encoder's input. A copied "
            "dummy id also fails LOUDLY at Pass 0 prefix lookup, unlike "
            "a copied title, which dies silently in fuzzy matching."
        ),
    },
    'example_block_wrappers': {
        'rule': (
            "Each example block in the encoder prompt SHOULD be wrapped "
            "in `<example>...</example>` markers (or similar visible "
            "demarcation) so the example/production boundary is "
            "structurally unmissable. v22+ work."
        ),
        'precedent': (
            "Anthropic's own prompt-engineering conventions use XML-like "
            "tags for examples. Sonnet's training recognizes the pattern."
        ),
    },
    'fields_that_need_placeholders': [
        # When authoring or rendering examples, ANY of these fields with
        # a value that wouldn't exist in a fresh brain should use <...>
        # syntax instead of a literal value.
        'connect_to[].title',  # edge targets — top failure mode; exempt when
                               # the id is copied from an in-example excerpt
                               # (grounded_catalog_excerpt)
        'source_refs',          # trace_ids in examples don't match real traces
        'node_id (in revise examples)',  # node_id references must be placeholder
    ],
    'when_real_values_are_ok': [
        # These fields can have real-looking values in examples because
        # they describe THIS node, not external references:
        'title (of the example node being created)',
        'content / situation / reasoning (of the example node)',
        'their_raw_quote / my_raw_quote (verbatim is the contract; the example demonstrates verbatim discipline)',
        'event_time',
    ],
    'source_refs_placeholder_syntax': {
        'rule': (
            "All example trace_id values (in source_conversation turns AND "
            "in source_refs lists) MUST use bracketed placeholder syntax — "
            "`<descriptive-name>` — never a literal-looking hex string. "
            "Production trace_ids are 8-char hex (schema v29) and Sonnet "
            "pattern-matches concrete-looking values into real output. "
            "The placeholder shape is the structurally unmissable signal."
        ),
        'good_example': '"<trace-tom-methodology>", "<trace-anchor-articulation>", "<trace-ratification>"',
        'bad_example': '"a3f5e2b1", "ff69e1ad" — real-looking hex strings get literal-copied',
        'enforced_by': 'validate_example_authoring() — D33',
    },
    'source_refs_internal_consistency': {
        'rule': (
            "Every integer in an encoder_output source_refs list MUST appear "
            "as the trace_id of some turn in the same example's source_"
            "conversation. Refs that don't anchor to a turn the example "
            "shows are orphan refs — they teach Sonnet to invent ids."
        ),
        'enforced_by': 'validate_example_authoring() — D34',
    },
    'voice_annotation_per_ref': {
        'rule': (
            "Every source_ref in encoder_output gets a corresponding entry "
            "in the example's voice_annotations block. The entry names "
            "source_turn + load_bearing (prose explaining why THIS turn "
            "anchors THIS aspect). Generic load_bearing ('the turn matters') "
            "is a violation — be specific about which axis it anchors."
        ),
        'enforced_by': 'validate_example_authoring() — D35',
    },
    'turn_node_divergence': {
        'rule': (
            "source_conversation turns are written in OPERATOR/ANCHOR REGISTER "
            "(messy, layered, redundant where real talk is redundant; "
            "specific where talk is specific). Encoded node content/situation/"
            "reasoning is written in EXTRACTION REGISTER — Anchor naming "
            "the hidden structural axis the turn implies but does not state. "
            "These two registers MUST diverge in language. The verbatim "
            "quote fields (their_raw_quote/my_raw_quote) are the legitimate "
            "bridge — they preserve turn phrases unchanged. Content prose "
            "MUST NOT paraphrase the turn."
        ),
        'specificity_preservation': (
            "Numbers stay numbers (190 not 'about 200'). Ranges stay ranges "
            "('250 of 440' not 'most'). Exact load-bearing phrases stay "
            "verbatim in raw_quote fields, not re-stated in content."
        ),
        'turn_diversity_within_example': (
            "Turns inside one source_conversation must themselves be diverse — "
            "different angles, different registers, different layers. Not "
            "synonymic restatements of each other. Real exchanges contain "
            "redundancy, drift, partial articulation, and reframes; clean "
            "synonymic turns make the example synthetic-feeling and teach "
            "the encoder to expect tidiness it won't find in production."
        ),
        'good_pattern': (
            "Turn: 'we keep getting burned when mocked tests pass but the "
            "prod migration fails. so dont mock the db.' → Node content: "
            "'Integration tests against staging DB are required for migration-"
            "touching code paths; mock-based unit tests historically miss "
            "schema-divergence failures.' Different register, same claim, "
            "structural axis named. Tom's verbatim 'we keep getting burned' "
            "lives in their_raw_quote, NOT in content prose."
        ),
        'bad_pattern': (
            "Turn: 'dont mock the db, mocks miss schema bugs.' → Node "
            "content: 'Do not mock the database; mocks miss schema bugs.' "
            "Mirror text. The encoder learns to be a punctuation-cleaner, "
            "not a structural extractor."
        ),
        'enforced_by': 'evaluator LLM judgment — D36',
    },
    'history': (
        "v22 (2026-05-25) established connect_to placeholder syntax after "
        "v20→v21 connect_to_unresolved errors. Tom's framing: 'examples "
        "need much better signaling that it\\'s an example.' Subsequently "
        "expanded with source_refs sentinel range, internal consistency, "
        "voice annotation coverage, and turn↔node divergence rules — "
        "examples are training data; mirror-text examples teach the encoder "
        "to paraphrase rather than extract."
    ),
}


# ═══════════════════════════════════════════════════════════════
# Mechanical example-authoring validator (D33-D35)
# ═══════════════════════════════════════════════════════════════
#
# Catches example authoring violations at load time WITHOUT an LLM call.
# D36 (turn↔node divergence) requires semantic judgment and lives in the
# evaluator's LLM pass. The three mechanical dims are checked here so bad
# examples never reach the encoder prompt.

def validate_example_authoring(example: dict) -> list:
    """Check D33-D35 on an example dict (the shape used by §7.6 example files).

    Returns a list of violation strings; empty list means the example passes
    all three mechanical checks. Callers (loader, tests, pre-commit) decide
    whether to raise or warn.

    Expected example shape (v29+):
        {
            'source_conversation': [{'trace_id': str, ...}, ...],   # `<placeholder>` strings
            'encoder_output': {'nodes': [{'source_refs': [str, ...], ...}, ...]},
            'voice_annotations': {<name>: {'source_turn': str, 'load_bearing': str, ...}},
            ...
        }
    """
    violations = []

    turns = example.get('source_conversation') or []
    turn_ids = set()
    for turn in turns:
        tid = turn.get('trace_id')
        if tid is None:
            continue
        tid_str = str(tid)
        turn_ids.add(tid_str)
        # D33: trace_id must use placeholder syntax
        if not EXAMPLE_ID_PLACEHOLDER_RE.match(tid_str):
            violations.append(
                f"D33: source_conversation trace_id={tid_str!r} does not match "
                f"placeholder syntax `<descriptive-name>`. Use bracketed "
                f"placeholder so Sonnet doesn't pattern-match into production."
            )

    output = example.get('encoder_output') or {}
    nodes = output.get('nodes') or output.get('revisions') or []
    all_refs = []
    for n in nodes:
        refs = n.get('source_refs') or []
        for r in refs:
            r_str = str(r)
            all_refs.append(r_str)
            # D33: ref must use placeholder syntax
            if not EXAMPLE_ID_PLACEHOLDER_RE.match(r_str):
                violations.append(
                    f"D33: encoder_output source_ref={r_str!r} does not match "
                    f"placeholder syntax `<descriptive-name>`."
                )
            # D34: ref must point to a turn in source_conversation
            if r_str not in turn_ids:
                violations.append(
                    f"D34: encoder_output source_ref={r_str!r} has no matching "
                    f"trace_id placeholder in source_conversation. Orphan ref — "
                    f"teaches encoder to invent trace ids."
                )

    # D35: presence-only mechanical check — examples with source_refs must
    # have a non-empty voice_annotations block documenting load-bearing
    # turns. Whether each ref has substantive load_bearing prose is judged
    # by the evaluator LLM (deeper semantic check; lives alongside D36).
    annotations = example.get('voice_annotations') or {}
    if all_refs and not annotations:
        violations.append(
            f"D35: example has {len(all_refs)} source_refs but no "
            f"voice_annotations block. Add load-bearing prose covering "
            f"the ref-anchored turns."
        )

    return violations


EVALUATOR_SYSTEM_PROMPT = """You are evaluating an encoder example against the 36-dimension quality contract.

# Your inputs
- `conversation`: source turns the encoder saw (S0 trace events with speaker labels and trace ids)
- `encoder_output`: the remember_batch / brain_batch / revise_batch calls the example demonstrates
- `example_intent`: which axis/dim this example was authored to teach
- `aspects`: snapshot of aspects_v1.json membership at example time
- `contract`: the 36 DIMENSIONS dict + CROSS_DIM_RULES from quality_contract.py

# Your task
For each dimension D1..D36, classify the encoder output:
- `satisfied`: positive signals present, no violations
- `degraded`: partial — some satisfies present but degrades-list features visible
- `violated`: clear violation of the dim's stated criteria
- `n/a`: dimension structurally doesn't apply to this example shape

For each cross-dim rule listed above, identify:
- Did the tension this rule names actually surface in this example?
- If yes, was the rule's resolution applied correctly?

# Output JSON

{
  "per_dim": [
    {
      "dim": "D1_title_as_handle",
      "status": "satisfied" | "degraded" | "violated" | "n/a",
      "evidence": "<specific field values that drive the status>",
      "degradation_note": "<if degraded, where the partial-fail lives>"
    },
    ...36 entries
  ],
  "cross_dim": [
    {
      "rule": "CR1_title_compress_vs_verbatim",
      "fired": true | false,
      "resolution_correct": true | false | "n/a",
      "note": "<observation about how the tension was navigated>"
    },
    ...one entry per cross-dim rule
  ],
  "verdict": {
    "is_canonical": true | false,
    "missing_demonstrations": ["<dims this example could have shown but did not>"],
    "contradictions_found": [["dim_a", "dim_b", "<reason>"]],
    "summary": "<one paragraph: does this example teach its intent without contradicting other dims?>"
  }
}

# Discipline
- Cite specific field values when scoring (e.g., "title='Single-writer invariant' — noun phrase, ≤80c").
- Don't hedge — if a dim is satisfied, say so; if violated, name the violation.
- The contract dimensions are the yardsticks; you are NOT scoring the contract itself, only the example.
- Multi-aspect verbs (D24): score based on whether edge_description disambiguates.
- Empty fields are not automatically n/a — check if the dim REQUIRES the field.
- Recall-gate dimensions (D25-D27): score the encoder's WRITING discipline, not what recall would display.
- Per CR4: novel types are degrade-not-violate when content is coherent with an existing aspect's meaning.
- Output strict JSON. No prose outside the JSON envelope.
"""
