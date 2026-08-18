"""Interaction defaults — the name→(template, config) index for every
interaction whose default lives in code.

One concern: this is the complete answer to "what does interaction X run on
when no DB override exists." Imports only — no content of its own. Each
config default lives in its CONSUMER's contract file (one default home per
interaction); each template lives in its prompt `.py` file. The override
resolver (the get_interaction_* accessors) resolves against this index.

Deliberately NOT `shipped_prompts()` (interaction_seed.py): that roster
answers "what should the fleet be force-advanced to" (7 names) and dies with
the distribution machinery. This index covers every name with a reader.

Names deliberately absent:
- `encoding_agent`, `s2_edge_families`, `s2_node_families` — dead legacy;
  existing DB rows are inert history.
- `boot`, `voice_surface`, `pre_edit`, `signal_assembler` — retired: every
  config key grepped reader-less (pre_edit's plausible keys read from
  brain_meta, a different store).
- Aspects (`aspects_v1.json`) — already the after-model; no override layer.
"""
import hashlib
import json

from .recall_expansion_prompt import (
    SYSTEM_PROMPT as _RECALL_EXPANSION_PROMPT,
    RECALL_EXPANSION_INTERACTION_DEFAULT)
from .recall_laf import DEFAULT_CONFIG as _RECALL_LAF_DEFAULT
from .scales.s1.encode_contract import S1E_INTERACTION_DEFAULT
from .scales.s1.encoding_prompt import SYSTEM_PROMPT as _S1E_PROMPT
from .scales.s1.scouts.contract import (
    SCOUT_FACTS_INTERACTION_DEFAULT,
    SCOUT_QUOTE_INTERACTION_DEFAULT,
    SCOUT_TEMPORAL_INTERACTION_DEFAULT)
from .scales.s1.scouts.prompts.facts_prompt import SYSTEM_PROMPT as _FACTS_PROMPT
from .scales.s1.scouts.prompts.quote_prompt import SYSTEM_PROMPT as _QUOTE_PROMPT
from .scales.s1.scouts.prompts.temporal_prompt import SYSTEM_PROMPT as _TEMPORAL_PROMPT
from .scales.s1.surface_contract import SURFACE_INTERACTION_DEFAULT
from .scales.s1.surface_prompt import SYSTEM_PROMPT as _SURFACE_PROMPT
from .scales.s2.aspect_contract import ASPECT_INTERACTION_DEFAULT
from .scales.s2.aspect_prompt import SYSTEM_PROMPT as _ASPECT_PROMPT
from .scales.s2.community_contract import COMMUNITY_DETECTION, COMMUNITY_ENRICHMENT
from .scales.s2.community_enrichment_prompt import SYSTEM_PROMPT as _COMMUNITY_PROMPT
from .scales.s2.consolidation_contract import CONSOLIDATION_ENRICHMENT
from .scales.s2.consolidation_enrichment_prompt import SYSTEM_PROMPT as _CONSOLIDATION_PROMPT
from .scales.s2.healer_contract import HEALER_INTERACTION_DEFAULT
from .scales.s2.healer_prompt import SYSTEM_PROMPT as _HEALER_PROMPT
from .scopes import SCOPES_CONFIG_V1, validate_scopes_config
from .trace_contract import TRACE_RECORDING_NORMAL

# name → (template, config). Config-only interactions carry '' templates.
INTERACTION_DEFAULTS = {
    's1e':                   (_S1E_PROMPT, S1E_INTERACTION_DEFAULT),
    'surface':               (_SURFACE_PROMPT, SURFACE_INTERACTION_DEFAULT),
    's1_scout_quote':        (_QUOTE_PROMPT, SCOUT_QUOTE_INTERACTION_DEFAULT),
    's1_scout_temporal':     (_TEMPORAL_PROMPT, SCOUT_TEMPORAL_INTERACTION_DEFAULT),
    's1_scout_facts':        (_FACTS_PROMPT, SCOUT_FACTS_INTERACTION_DEFAULT),
    's2_community_enrichment':     (_COMMUNITY_PROMPT, COMMUNITY_ENRICHMENT),
    's2_consolidation_enrichment': (_CONSOLIDATION_PROMPT, CONSOLIDATION_ENRICHMENT),
    's2_healer':             (_HEALER_PROMPT, HEALER_INTERACTION_DEFAULT),
    's2_aspects':            (_ASPECT_PROMPT, ASPECT_INTERACTION_DEFAULT),
    'recall_query_expansion': (_RECALL_EXPANSION_PROMPT,
                               RECALL_EXPANSION_INTERACTION_DEFAULT),
    'recall_laf':            ('', _RECALL_LAF_DEFAULT),
    'trace_recording':       ('', TRACE_RECORDING_NORMAL),
    'scopes':                ('', SCOPES_CONFIG_V1),
    # Decoder parameters (not an LLM template). The decoder currently imports
    # COMMUNITY_DETECTION directly; the interaction read is not wired.
    's2_community':          ('', COMMUNITY_DETECTION),
}


# name → callable(config: dict) → [violation strings]; empty list = valid.
# Consulted at BOTH doors of the override model: register_interaction REFUSES
# a version whose config has violations (write door), and the resolver LOGS
# violations and runs on the code default (read seam) — an invalid override
# must never become the running K, and must never crash a read path.
INTERACTION_VALIDATORS = {
    'scopes': validate_scopes_config,
}


def interaction_fingerprint(name: str, template: str, config: dict) -> str:
    """12-hex sha256 over name + template + canonical(config).

    Content-address of the EFFECTIVE prompt+config a run used. The row
    pointer (interaction_id = a rowid, interaction_version = a per-install
    counter) is install-local and can dangle; the fingerprint identifies the
    K by content, so it is stable across installs and unchanged when an
    override row is collapsed into a byte-identical code default. 48 bits is
    ample for the tens of distinct Ks a brain ever runs.
    """
    h = hashlib.sha256()
    h.update((name or '').encode())
    h.update((template or '').encode())
    h.update(json.dumps(config or {}, sort_keys=True).encode())
    return h.hexdigest()[:12]
