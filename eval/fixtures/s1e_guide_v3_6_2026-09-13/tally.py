"""Aggregate the hand-transcribed blind ratings through the sealed key. Reading aid, not a score."""
import json, pathlib, collections
here = pathlib.Path(__file__).parent
key = json.load(open(here.parent / 'key.json'))
t = json.load(open(here / 'tally.json'))
DIMS = ['1 Facts and concrete detail', '2 Decisions and arcs', '3 Revision and preservation',
        '4 Evidence, ownership, scope', '5 Voice and synthesis', '6 Recall usefulness', "7 Receiver's view", '8 Content and field quality']
ARMS = ['production_deployed', 'v3_4_titles', 'v3_5_titles']
per_dim = {a: [0]*8 for a in ARMS}
per_pack = collections.defaultdict(dict)
per_corpus = {a: collections.Counter() for a in ARMS}
for pack, ratings in t['packs'].items():
    corpus = pack.rsplit('_repeat', 1)[0]
    for letter, arm in key[pack].items():
        r = ratings[letter]
        assert len(r) == 8, pack
        per_pack[pack][arm] = sum(r)
        per_corpus[arm][corpus] += sum(r)
        for i, v in enumerate(r):
            per_dim[arm][i] += v
n = len(t['packs'])
print(f'{n} packs tallied (max per arm per pack 16; per dimension {2*n})\n')
print('| Dimension | Production | V3.4 | V3.5 |\n|---|---:|---:|---:|')
for i, d in enumerate(DIMS):
    print(f'| {d} | ' + ' | '.join(str(per_dim[a][i]) for a in ARMS) + ' |')
print(f'| Sum of {16*n} | ' + ' | '.join(str(sum(per_dim[a])) for a in ARMS) + ' |')
print('\nPer pack (production / V3.4 / V3.5):')
for pack in sorted(per_pack):
    print(f"  {pack:28} " + ' / '.join(f"{per_pack[pack].get(a, '-'):>2}" for a in ARMS))
print('\nPer corpus totals (production / V3.4 / V3.5):')
corpora = sorted({p.rsplit('_repeat', 1)[0] for p in t['packs']})
for c in corpora:
    print(f"  {c:20} " + ' / '.join(f"{per_corpus[a][c]:>2}" for a in ARMS))
