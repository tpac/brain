"""Code default for interaction `recall_query_expansion` — editing SYSTEM_PROMPT
here IS the deployment: every install without a deployed override follows on the
next daemon restart. Per-install override: register_interaction +
set_interaction_active; clear_interaction_override reverts to this default.
"""

SYSTEM_PROMPT = """Generate 2-3 alternate query phrasings that bridge LEXICAL GAPS — vocabulary differences between how the user asks and how the memory was originally stored. Don't paraphrase. Don't say the same thing in different words.

Each alternate MUST drop or replace at least one specific term from the original query, choosing one of these strategies:

1. STRIP the specific entity, keep the activity:
   "What did I bake for my uncle's birthday party?" → "what I baked for a birthday party"
   "Where did I attend study abroad?" → "country I studied in", "university I went to"

2. REPLACE the specific entity with a category or sibling entity (in case the memory is about a related entity):
   "uncle's birthday" → "family member's birthday", "niece's birthday"
   "feed" → "feed for chickens", "scratch grains for chickens"
   "Memrise" → "language learning apps with mnemonics", "apps for memorization"

3. BROADEN to the category the original is in:
   "siblings count" → "brothers and sisters family"
   "gym time" → "evening workout schedule"

The original query gets searched separately. Your alternates must reach memories the original would NOT.

Return ONLY a JSON array of 2-3 strings, no prose, no explanation.

Query: "{query}"
"""

# Interaction config default for the `recall_query_expansion` K. Live reads:
# `model`, `max_tokens` (brain_recall._expand_query_via_llm). Lives here with
# its template — the sole consumer; two keys don't earn a contract file.
RECALL_EXPANSION_INTERACTION_DEFAULT = {
    'model': 'claude-haiku-4-5',
    'max_tokens': 200,
}
