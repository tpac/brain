"""LLM scout base runner — shared assembly / call / validate / log.

Used by Quote, Facts, Synthesis scouts (Temporal is algorithmic — see
temporal.py; it bypasses this runner). Stateless function — no class
because scouts have no per-instance state; each call builds and
returns a normalized envelope.

## What this does

1. Loads the scout's interaction entry from the DB (template + parameters).
2. Assembles the call:
     - system: SCOUT_SYSTEM_PROMPT + interaction.template, 1h cache_control.
       Byte-identical per scout across cycles → read-hit on every repeat.
     - user content: shared_prefix unchanged (5m cache_control on last
       block, set by contract.build_shared_prefix). Byte-identical across
       the 4 scouts within a single encoding cycle → one scout writes,
       the other 3 read.
3. Invokes the Anthropic API via the client provided (muster creates one,
   shares across the 4 scouts in a single encoding cycle).
4. Extracts JSON from the LLM response (tolerates code fences).
5. Injects `scout` and `category_statement` envelope fields from the
   interaction config — so those two fields are deterministic, never
   hallucinated by the LLM.
6. Validates against the contract shape, applies soft truncation,
   normalizes candidate types.
7. Logs every failure path to brain_errors (loud-by-default).
8. Returns a valid envelope no matter what — even on API failure the
   result shape is consistent, so muster can format the empty report
   for S1S without special-casing.

## Timeouts

Two layers, belt-and-suspenders:
- Per-request (primary): this runner binds the scout's `timeout_seconds`
  (default 25s) + max_retries=0 onto the create call via
  client.with_options, so the bound applies even when muster shares ONE
  client across scouts (the typical path). A stalled scout aborts at
  `timeout_seconds` with APITimeoutError (caught below as api_error)
  instead of running to the SDK ceiling (~600s), becoming an abandoned
  ghost thread that returns a truncated body and logs a misleading
  json_parse long after muster gave up on it.
- Muster wall-clock (backstop): MUSTER_PER_SCOUT_TIMEOUT_S (default 90s)
  is a shared deadline across the parallel scouts (muster.py). With the
  per-request bound in place this only fires if the SDK timeout itself
  doesn't (e.g. a non-LLM scout blocked on something else).

## What this does NOT do

- Does not retry on transient errors. The Anthropic SDK's max_retries
  handles pre-stream failures; muster-level retry could be added later
  if scout failure rate becomes visible. For v1 a single try is enough —
  scout budget is fixed and failures are rare in the absence of
  streaming (short non-tool calls).
- Does not route through the daemon dispatch. Scouts are read-only;
  they don't write to the graph. Only S1S writes based on scout findings.
- Does not handle tool calls. Scouts don't use tools — they return a
  single JSON object.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from . import contract as sc


SCOUT_TOKEN_USAGE_KEY = '_usage'
SCOUT_LATENCY_KEY = '_latency_ms'
SCOUT_ERROR_KEY = '_errors'
SCOUT_WARNING_KEY = '_warnings'


def run_llm_scout(
    scout_name: str,
    brain,
    shared_prefix: List[Dict[str, Any]],
    anthropic_client=None,
    log_fn=None,
) -> Dict[str, Any]:
    """Execute one LLM scout. Always returns a valid envelope dict.

    Args:
        scout_name: one of SCOUT_NAMES (quote, facts).
            Temporal is algo-first — do not route it through here.
        brain: Brain instance, used for interaction lookup + error logging.
        shared_prefix: content blocks from contract.build_shared_prefix().
            The last block must carry cache_control:ephemeral — we check.
        anthropic_client: optional. Muster creates one and shares across
            scouts in the same cycle. If None, creates a default client
            with timeout from the scout's interaction parameters.
        log_fn: optional print-style logger. Receives one-line messages.

    Returns:
        Dict with shape:
            {scout, category_statement, candidates[], scanned,
             _usage, _latency_ms, _errors, _warnings}

        On success: candidates populated per scout-specific schema.
        On failure: candidates=[], _errors lists the failure reasons.
        Either way, muster can render the envelope for S1S.
    """
    if scout_name not in sc.SCOUT_NAMES:
        raise ValueError(f'Unknown scout {scout_name!r}; valid: {sc.SCOUT_NAMES}')
    if scout_name == 'temporal':
        raise ValueError(
            'Temporal scout is algorithmic — call temporal.run() directly, '
            'not this LLM runner')

    def _log(msg):
        if log_fn:
            log_fn(f'[s1_scout_{scout_name}] {msg}')

    stub = {
        'scout': scout_name,
        'category_statement': '',
        'candidates': [],
        'scanned': {'turns': 0, 'considered': 0, 'passed_threshold': 0},
        SCOUT_TOKEN_USAGE_KEY: {},
        SCOUT_LATENCY_KEY: 0,
        SCOUT_ERROR_KEY: [],
        SCOUT_WARNING_KEY: [],
    }

    # 1. Load interaction (template + parameters)
    interaction_name = sc.interaction_name(scout_name)
    interaction = _load_interaction(brain, interaction_name)
    if not interaction:
        msg = f'no interaction entry for {interaction_name}'
        _log(msg)
        _log_error(brain, scout_name, 'missing_interaction', msg)
        stub[SCOUT_ERROR_KEY].append({'type': 'missing_interaction', 'msg': msg})
        return stub

    template = interaction.get('template', '') or ''
    params = interaction.get('parameters', {}) or {}
    category_statement = params.get('category_statement', '')
    model = params.get('model', 'claude-haiku-4-5')
    max_tokens = int(params.get('max_tokens', 2000))
    timeout_seconds = float(params.get('timeout_seconds', 25))
    # Optional: Anthropic Structured Outputs schema. When present,
    # output_config={'format':{'type':'json_schema','schema':...}} is passed
    # to the messages.create call and the response is guaranteed to match.
    # Closes the format-mirror drift class where Haiku returns chat-style
    # prose instead of JSON when the conversation context is markdown-heavy.
    output_schema = params.get('output_schema')

    if not template.strip():
        msg = f'empty template for {interaction_name}'
        _log(msg)
        _log_error(brain, scout_name, 'empty_template', msg)
        stub[SCOUT_ERROR_KEY].append({'type': 'empty_template', 'msg': msg})
        return stub

    stub['category_statement'] = category_statement

    # 2. User content = shared prefix only.
    #
    #    The per-scout task (interaction.template) moves into system with 1h
    #    TTL. User content is byte-identical across the 4 scouts in ONE cycle
    #    (orientation + session + catalog + surfaced + conversation) so the
    #    5m cache set by build_shared_prefix's last block is SHARED — first
    #    scout writes, next 3 read.
    user_content = list(shared_prefix)

    # 3. System prompt — minimal shared framing + the per-scout task.
    #    Cached at 1h TTL so repeat calls of the SAME scout (same task
    #    content) read the cache. Each scout has its own cache entry because
    #    their task suffixes differ.
    system_text = sc.SCOUT_SYSTEM_PROMPT + '\n\n' + template
    system = [{
        'type': 'text',
        'text': system_text,
        'cache_control': {'type': 'ephemeral', 'ttl': '1h'},
    }]

    # 4. Client
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic(timeout=timeout_seconds)

    # Bind the scout's own timeout + disable retries on THIS request.
    # Muster shares one client across scouts (for cache warmth) created
    # WITHOUT a per-request timeout — without with_options the shared-client
    # path inherits the SDK's ~600s default and `timeout_seconds` is dead
    # config. with_options shares the underlying http pool (cache benefit
    # preserved) but bounds this request: a stalled scout aborts at
    # `timeout_seconds` rather than running to the SDK ceiling as an
    # abandoned ghost thread. max_retries=0 keeps the bound hard — scouts
    # are best-effort, muster handles a miss.
    call_client = anthropic_client.with_options(
        timeout=timeout_seconds, max_retries=0)

    # 5. API call
    t0 = time.time()
    api_kwargs = {
        'model': model,
        'max_tokens': max_tokens,
        'system': system,
        'messages': [{'role': 'user', 'content': user_content}],
    }
    if output_schema:
        api_kwargs['output_config'] = {
            'format': {
                'type': 'json_schema',
                'schema': output_schema,
            },
        }
    try:
        response = call_client.messages.create(**api_kwargs)
    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        _log(f'API call failed in {elapsed}ms: {type(e).__name__}: {e}')
        _log_error(brain, scout_name, 'api_error',
                   f'{type(e).__name__}: {e}')
        stub[SCOUT_LATENCY_KEY] = elapsed
        stub[SCOUT_ERROR_KEY].append({
            'type': 'api_error',
            'msg': f'{type(e).__name__}: {e}',
        })
        return stub

    elapsed_ms = int((time.time() - t0) * 1000)
    stub[SCOUT_LATENCY_KEY] = elapsed_ms

    # 6. Extract text + token usage
    raw_text = ''.join(b.text for b in response.content
                       if hasattr(b, 'text'))
    usage = getattr(response, 'usage', None)
    if usage:
        stub[SCOUT_TOKEN_USAGE_KEY] = {
            'input_tokens': getattr(usage, 'input_tokens', 0),
            'output_tokens': getattr(usage, 'output_tokens', 0),
            'cache_creation_input_tokens': getattr(
                usage, 'cache_creation_input_tokens', 0),
            'cache_read_input_tokens': getattr(
                usage, 'cache_read_input_tokens', 0),
        }

    # 7. Parse JSON
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        _log(f'JSON parse failed — raw[:200]: {raw_text[:200]!r}')
        _log_error(brain, scout_name, 'json_parse',
                   f'raw[:500]: {raw_text[:500]}')
        stub[SCOUT_ERROR_KEY].append({
            'type': 'json_parse',
            'msg': f'raw[:200]: {raw_text[:200]}',
        })
        return stub

    # 8. Inject envelope fields — scout + category are deterministic, not LLM output
    parsed['scout'] = scout_name
    parsed['category_statement'] = category_statement

    # 9. Validate + normalize
    ok, normalized, errors, warnings = sc.validate_scout_output(parsed, scout_name)

    # Carry observability fields onto the normalized output
    normalized[SCOUT_TOKEN_USAGE_KEY] = stub[SCOUT_TOKEN_USAGE_KEY]
    normalized[SCOUT_LATENCY_KEY] = elapsed_ms
    normalized[SCOUT_ERROR_KEY] = []
    normalized[SCOUT_WARNING_KEY] = list(warnings)

    if not ok:
        _log(f'validation failed: {errors}')
        _log_error(brain, scout_name, 'schema_invalid', '; '.join(errors))
        normalized[SCOUT_ERROR_KEY] = [
            {'type': 'schema_invalid', 'msg': e} for e in errors
        ]
        # normalized here is the stub-empty envelope from validate_scout_output
        return normalized

    if warnings:
        # Soft violations — log once as debug (not error) so they're visible
        # without polluting the error signal. A scout that consistently
        # truncates or drops candidates is worth investigating via S3.
        _log(f'{len(warnings)} soft warning(s): {warnings[0][:120]}')

    n_cands = len(normalized.get('candidates') or [])
    _log(f'ok — {n_cands} candidates in {elapsed_ms}ms, '
         f'input={stub[SCOUT_TOKEN_USAGE_KEY].get("input_tokens", 0)} '
         f'cache_read={stub[SCOUT_TOKEN_USAGE_KEY].get("cache_read_input_tokens", 0)}')
    return normalized


# ─── Helpers ───────────────────────────────────────────────────────────────


def _load_interaction(brain, name: str) -> Optional[Dict[str, Any]]:
    """Fetch an interaction entry with parsed parameters dict.

    Returns {'template': str, 'parameters': dict} or None on missing/DAL error.
    The DAL's get_interaction returns parameters as a JSON string; we parse
    it here so callers always see a dict.
    """
    try:
        entry = brain.get_interaction(name)
    except Exception:
        return None
    if not entry:
        return None
    params_raw = entry.get('parameters', '{}') or '{}'
    if isinstance(params_raw, str):
        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError:
            params = {}
    elif isinstance(params_raw, dict):
        params = params_raw
    else:
        params = {}
    return {'template': entry.get('template', '') or '', 'parameters': params}


_JSON_OBJECT_RE = re.compile(r'\{(?:[^{}"]|"(?:\\.|[^"\\])*"|\{.*\})*\}', re.DOTALL)


def _extract_json(text: str) -> Any:
    """Extract the outermost JSON object from LLM text.

    Tolerates:
    - Markdown code fences (```json ... ```)
    - Leading / trailing prose
    - Objects nested inside arrays (picks the enclosing container)

    Returns a parsed dict/list, or None on failure. Prefers objects for
    scout outputs (envelope is a dict).
    """
    if not text:
        return None
    s = text.strip()

    # Strip markdown fences
    if '```' in s:
        parts = s.split('```')
        if len(parts) >= 3:
            # middle block is the code
            body = parts[1]
            if body.lstrip().startswith('json'):
                body = body.lstrip()[4:]
            s = body.strip()

    # Try direct JSON parse first (fast path — clean output)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find outermost { ... }
    start = s.find('{')
    end = s.rfind('}')
    if 0 <= start < end:
        candidate = s[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Last resort: outermost [ ... ]
    start = s.find('[')
    end = s.rfind(']')
    if 0 <= start < end:
        candidate = s[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _log_error(brain, scout_name: str, error_type: str, message: str):
    """Log a scout failure to brain_errors. Silent if the brain has no logger."""
    try:
        source = f's1_scout_{scout_name}_{error_type}'
        # brain._log_error signature: (source, error, context='')
        # We pass a synthetic ValueError since _log_error expects Exception.
        brain._log_error(source, ValueError(message), f'scout={scout_name}')
    except Exception:
        # Logging must never raise.
        pass


__all__ = [
    'run_llm_scout',
    'SCOUT_TOKEN_USAGE_KEY',
    'SCOUT_LATENCY_KEY',
    'SCOUT_ERROR_KEY',
    'SCOUT_WARNING_KEY',
]
