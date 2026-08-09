# Test Standards

## Run all tests
```bash
python3 tests/run_all.py
```

## Structure
- `tests/test_*.py` — unit tests (no DB, no daemon, no API calls)
- `tests/integration/test_*.py` — integration tests (uses copy of brain DB, no daemon)
- `eval/brain_eval.py` — decode funnel + KPI evaluation (uses live or copy brain)

## Rules
1. Tests NEVER modify the live brain.db. Integration tests copy to /tmp first.
2. Each test file is independently runnable: `python3 tests/test_foo.py`
3. Tests must pass before any code ships to production.
4. New mechanisms need tests in the same PR.
5. The decode funnel baseline is stored in `eval/results/` — regression = fail.

## What we test

### Unit tests (fast, no DB)
- `test_pipeline_contract.py` — judge prompt building, output formatting, embedding groups
- `test_redistribution.py` — blend vectors, fidelity, bridge detection

### Integration tests (uses DB copy)
- `test_recall_pipeline.py` — full recall: cosine scan → z-weighted → fatigue → judge → graph expand
- `test_hebbian_flow.py` — judge-selected IDs → co_accessed edges created
- `test_metadata_kv.py` — remember → KV store → recall enrichment → judge sees metadata
- `test_encoding_groups.py` — remember → group vectors created (title, high_meta, other_meta)

### Regression tests (eval)
- `eval/brain_eval.py --mode decode` — R@25, hub concentration, MRR vs baseline
