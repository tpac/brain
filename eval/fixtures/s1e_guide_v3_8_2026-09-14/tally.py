"""Aggregate the hand-transcribed blind ratings through the sealed key. Reading aid, not a score."""
import json, pathlib, collections
here = pathlib.Path(__file__).parent
key = json.load(open(here.parent / 'key.json'))
t = json.load(open(here / 'tally.json'))
DIMS = ['1 Facts and concrete detail', '2 Decisions and arcs', '3 Revision and preservation',
        '4 Evidence, ownership, scope', '5 Voice and synthesis', '6 Recall usefulness', "7 Receiver's view", '8 Content and field quality']
ORDER = ['v3_6_full', 'v3_7_advice', 'v3_7_quote']
ARMS = [a for a in ORDER if any(a in labels.values() for labels in key.values())]
per_dim = {a: [0]*8 for a in ARMS}
per_pack = collections.defaultdict(dict)
per_corpus = {a: collections.Counter() for a in ARMS}
packs_of = collections.Counter()
for pack, ratings in t['packs'].items():
    corpus = pack.rsplit('_repeat', 1)[0]
    for letter, arm in key[pack].items():
        r = ratings[letter]
        assert len(r) == 8, pack
        packs_of[arm] += 1
        per_pack[pack][arm] = sum(r)
        per_corpus[arm][corpus] += sum(r)
        for i, v in enumerate(r):
            per_dim[arm][i] += v
n = len(t['packs'])
print(f'{n} packs tallied (max per arm per pack 16; an arm absent from a pack scores nothing there)')
print('packs per arm: ' + ', '.join(f'{a} {packs_of[a]}' for a in ARMS) + '\n')
print('| Dimension | ' + ' | '.join(ARMS) + ' |\n|---|' + '---:|' * len(ARMS))
for i, d in enumerate(DIMS):
    print(f'| {d} | ' + ' | '.join(str(per_dim[a][i]) for a in ARMS) + ' |')
print('| Sum (max 16 × packs per arm) | ' + ' | '.join(f'{sum(per_dim[a])} of {16 * packs_of[a]}' for a in ARMS) + ' |')
print('\nPer pack (' + ' / '.join(ARMS) + '):')
for pack in sorted(per_pack):
    print(f"  {pack:28} " + ' / '.join(f"{per_pack[pack].get(a, '-'):>2}" for a in ARMS))
print('\nPer corpus totals (' + ' / '.join(ARMS) + '):')
corpora = sorted({p.rsplit('_repeat', 1)[0] for p in t['packs']})
for c in corpora:
    print(f"  {c:20} " + ' / '.join(f"{per_corpus[a][c]:>2}" for a in ARMS))
