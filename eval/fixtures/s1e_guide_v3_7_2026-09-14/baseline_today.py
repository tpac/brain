"""Same-day control for the V3.7 regression: v3_6_full re-run on the regression cell's six corpora and seed,
one process per repeat. The saved V3.6 brains were encoded on 2026-09-13; the regression arm on 2026-09-14 —
this run puts the baseline on the same day so a per-repeat difference reads as the carrier, not the day.
  ./dev python3 baseline_today.py --repeat N      (N in 1..3; launched three times in parallel)
"""
import argparse, importlib.util, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); import sequence  # noqa: E402
REG = ROOT / 'eval/results/s1e_v37_regression_v36_2026-09-14'
OUT = ROOT / 'eval/results/s1e_v37_regression_v36_baseline_today_2026-09-14'
spec = importlib.util.spec_from_file_location('_v37_arms', HERE / 'arms.py'); arms = importlib.util.module_from_spec(spec); spec.loader.exec_module(arms)
ap = argparse.ArgumentParser(); ap.add_argument('--repeat', type=int, required=True); a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
arm = arms.load_arm('v3_6_full')
for corpus in sorted(json.loads((REG / 'manifest.json').read_text())['corpora']):
    if not (OUT / corpus + '.json').exists() if False else not (OUT / (corpus + '.json')).exists():
        (OUT / (corpus + '.json')).write_text((REG / (corpus + '.json')).read_text())
    sequence.run_sequence(arm, 'v3_6_full', a.repeat, corpus, OUT, REG / 'seed_baseline')
print('BASELINE-TODAY REPEAT', a.repeat, 'COMPLETE', flush=True)
