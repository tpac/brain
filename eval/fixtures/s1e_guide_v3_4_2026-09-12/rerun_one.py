"""Rerun a single (arm, repeat, corpus) sequence of a pinned cell after a runner crash; the cell script is imported unchanged so check_pin still holds."""
import importlib.util, sys
from pathlib import Path
cell, arm, repeat, corpus = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(cell.parent))
spec = importlib.util.spec_from_file_location('_cell', cell); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.check_pin()
mod.sequence.run_sequence(mod.load_arm(arm), arm, repeat, corpus, mod.OUT, mod.OUT / 'seed_baseline')
print('RERUN COMPLETE', arm, repeat, corpus, flush=True)
