"""What one arm wrote that another has no counterpart for — token-overlap view, no model calls.
For each (corpus, repeat) with both arms present: every node of arm A whose best token Jaccard
(title + content) against any node of arm B is below THRESH is 'uncovered'. Lists both directions,
totals by type/aspect. Usage: uncovered_nodes.py <root> [<root>...] --a production_deployed --b v3_4_titles --out FILE.md"""
import argparse, json, re, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
ASP = json.loads((ROOT / 'servers/scales/s2/aspects_v1.json').read_text())
TYPE_ASPECT = {t: n for n, s in ASP.items() if not n.startswith('_') and isinstance(s, dict) for t in s.get('node_types', [])}
THRESH = 0.22
def toks(*vals): return set(re.findall(r"[a-z0-9]{3,}", ' '.join(str(v or '') for v in vals).lower()))
def field(n, k): return n.get(k) if n.get(k) not in (None, '') else (n.get('_metadata') or {}).get(k)
def latest(root, arm):
    out = {}
    for na in root.glob(f'{arm}/repeat*/*/window*/nodes_after.json'):
        rep, corpus, w = na.parts[-4], na.parts[-3], int(na.parts[-2][6:])
        if corpus.endswith('crashed-window2'): continue
        if (rep, corpus) not in out or w > out[(rep, corpus)][0]: out[(rep, corpus)] = (w, na)
    return {k: {i: n for i, n in json.loads(v[1].read_text()).items() if isinstance(n, dict) and n.get('title')} for k, v in out.items()}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True); ap.add_argument('--out', required=True)
    x = ap.parse_args(); roots = [Path(r).resolve() for r in x.roots]
    A = {}; B = {}
    for r in roots: A.update(latest(r, x.a)); B.update(latest(r, x.b))
    L = [f'# Uncovered nodes — {x.a} vs {x.b} (token Jaccard on title+content < {THRESH})', '']
    tot = {x.a: collections.Counter(), x.b: collections.Counter()}; ntot = collections.Counter()
    for key in sorted(set(A) & set(B)):
        a, b = A[key], B[key]
        ta = {i: toks(field(n, 'title'), field(n, 'content')) for i, n in a.items()}; tb = {i: toks(field(n, 'title'), field(n, 'content')) for i, n in b.items()}
        def unc(src, srct, dstt):
            out = []
            for i, n in src.items():
                best = max((len(srct[i] & t) / len(srct[i] | t) for t in dstt.values()), default=0)
                if best < THRESH: out.append((best, i, n))
            return out
        ua, ub = unc(a, ta, tb), unc(b, tb, ta)
        ntot[x.a] += len(a); ntot[x.b] += len(b)
        L.append(f'## {key[1]} {key[0]} — {x.a} {len(a)} nodes, {x.b} {len(b)} nodes; uncovered {len(ua)} / {len(ub)}')
        for label, u, arm in ((f'{x.a} nodes with no {x.b} counterpart', ua, x.a), (f'{x.b} nodes with no {x.a} counterpart', ub, x.b)):
            L.append(f'- **{label}: {len(u)}**')
            for best, i, n in sorted(u):
                t = n.get('type'); tot[arm][f'{t} ({TYPE_ASPECT.get(t, "unmapped")})'] += 1
                L.append(f"    - [{t}] {field(n, 'title')} (best {best:.2f})")
        L.append('')
    L.insert(2, f'Totals — {x.a}: {ntot[x.a]} nodes, uncovered by type: {dict(tot[x.a].most_common())}  \n{x.b}: {ntot[x.b]} nodes, uncovered by type: {dict(tot[x.b].most_common())}\n')
    Path(x.out).write_text('\n'.join(L) + '\n'); print('\n'.join(L[:4]))
if __name__ == '__main__': main()
