"""Approved Sonnet cell: six sequential isolated sequences, two windows each.
Invoked with ./dev env BRAIN_S1E_LISTS_PREAMBLE=1 python3 <this>.
A failed sequence stops the cell; folders are never overwritten.
"""
import json
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
OUT = WT / 'eval/results/s1e_guide_compression_2026-09-08'
ORDER = [('v2','repeat1'),('shapes','repeat1'),('episode','repeat1'),
         ('episode','repeat2'),('shapes','repeat2'),('v2','repeat2')]
for arm, repeat in ORDER:
    log = OUT / f'{arm}_{repeat}.log'
    if log.exists():
        raise SystemExit(f'Refusing to overwrite {log}')
    command = [sys.executable,str(WT/'eval/s1e_guide_v2_sequence_probe.py'),
               '--arm',arm,'--run-id',repeat,
               '--fixture',str(WT/'eval/fixtures/s1e_guide_compression_2026-09-08/contrasts.json'),
               '--out-dir',str(OUT),'--session-id','c03a9b72-compression-sequence']
    print('START',arm,repeat,flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command,cwd=WT,stdout=stream,stderr=subprocess.STDOUT)
    print('FINISH',arm,repeat,'exit',result.returncode,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
print('COMPRESSION CELL COMPLETE',flush=True)
