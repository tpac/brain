"""Contract tests for servers/interaction_defaults.py — the name→(template,
config) index the override resolver serves.

The completeness test is what makes a raising resolver safe to ship (plan
step 4): every literal interaction name the runtime passes to the accessors
must have a registry entry, or a default-run on that name resolves to
nothing.
"""
import os
import re

from servers.interaction_defaults import INTERACTION_DEFAULTS

SERVERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'servers')

# LLM-prompt-backed names carry a real template; config-only names carry ''.
PROMPT_BACKED = {
    's1e', 'surface', 's1_scout_quote', 's1_scout_temporal', 's1_scout_facts',
    's2_community_enrichment', 's2_consolidation_enrichment', 's2_healer',
    's2_aspects', 'recall_query_expansion',
}


class TestRegistryShape:
    def test_every_entry_is_template_config_pair(self):
        for name, (template, config) in INTERACTION_DEFAULTS.items():
            assert isinstance(template, str), name
            assert isinstance(config, dict), name
            assert len(config) > 0, "%s config is empty" % name

    def test_prompt_backed_names_carry_real_templates(self):
        for name in PROMPT_BACKED:
            template, _ = INTERACTION_DEFAULTS[name]
            assert len(template) > 100, \
                "%s template suspiciously short (%d chars)" % (name, len(template))

    def test_config_only_names_carry_empty_templates(self):
        for name, (template, _) in INTERACTION_DEFAULTS.items():
            if name not in PROMPT_BACKED:
                assert template == '', \
                    "%s is config-only but carries a template" % name


class TestRegistryCompleteness:
    """Every literal name the runtime passes to get_interaction_prompt /
    get_interaction_config / get_interaction / get_interaction_stamp must be
    a registry key. A name outside the registry has no code default — under
    the override model a default-run on it resolves to nothing, silently."""

    _CALL_RE = re.compile(
        r"get_interaction(?:_prompt|_config|_stamp)?\(\s*['\"]([a-z0-9_]+)['\"]")

    def _literal_names_in_servers(self):
        names = set()
        for root, _dirs, files in os.walk(SERVERS_DIR):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(root, fn)
                with open(path, encoding='utf-8') as f:
                    names.update(self._CALL_RE.findall(f.read()))
        return names

    def test_every_runtime_literal_is_a_registry_key(self):
        literals = self._literal_names_in_servers()
        assert literals, "grep found no accessor literals — regex broken?"
        missing = literals - set(INTERACTION_DEFAULTS)
        assert not missing, \
            "accessor literals with no code default: %s" % sorted(missing)


class TestFingerprint:
    def test_fingerprint_is_stable_and_canonical(self):
        from servers.interaction_defaults import interaction_fingerprint
        a = interaction_fingerprint('n', 't', {'b': 1, 'a': 2})
        b = interaction_fingerprint('n', 't', {'a': 2, 'b': 1})
        assert a == b, "canonicalization must be key-order independent"
        assert len(a) == 12 and int(a, 16) >= 0
        assert interaction_fingerprint('n', 't', {'a': 1}) != a
