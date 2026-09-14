"""Per-node content and field quality, judged against the source — the LLM half of the census.

For every FINAL node of every (arm, repeat, corpus) sequence, a Sonnet 4.6 judge reads the
whole source conversation (every window, both voices, with each window's clock) and the node
card (type, title, content, situation, question, reasoning, thought, quotes, event_time,
edges) and returns ONE strict JSON object per node:

  value       specific_knowledge | useful_synthesis | supporting_context | generic_advice | redundant | unsupported
              (QUALITY-RUBRIC marginal-value classes, eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08)
  fidelity    supported | overclaimed | weakened | fabricated_element | mixed   + evidence (memory phrase vs source phrase)
  status_ok   plan / leaning / proposal / decision / fact / completion kept at the status the source gives (semantic-fidelity rows)
  owner_ok    who said, proposed or did what is preserved (assistant advice not stored as the other side's fact)
  stale       a surface still asserts a value the source later replaced (only when the source changed something)
  fields      situation: trigger | restated_title | workflow_narration | overbroad | absent
              question:  real | formulaic | absent
              reasoning: basis | encoder_justification | generic | restated | absent
              thought:   hunch_or_connection | caveat | restatement | noise | absent
              quotes:    verbatim_loadbearing | verbatim_incidental | altered | fabricated | absent
              edges:     insightful | restating | generic | none
              event_time: correct | wrong | missing_where_supported | absent_ok
  defect      <= 20 words, the single most consequential problem, or "none"

Nodes are judged in batches of up to BATCH per call (same source), so the source is paid
for once per batch. Aggregates per arm: distributions of every class, and a reading queue
of nodes rated unsupported / fabricated_element / overclaimed / altered / stale. The judge is
an instrument for reading at scale, not a verdict: the blind reviewers and the author's
read decide; this says WHERE to read.

Usage: content_quality.py <root> [<root>...] --out <dir> --label <cell> [--arms a,b] [--limit-seqs N] [--max-nodes N]
"""
import argparse, json, re, sys, time, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
MODEL = 'claude-sonnet-4-6'
BATCH = 6
SYSTEM = """You audit memory nodes that an encoder wrote from a conversation. You see the WHOLE source conversation (every window in order, each with its clock; <other> is the encoder's partner, <me> is the encoder) and a batch of node cards. Judge each node ONLY against the source. Do not reward length, field count or confidence.

For each node return one JSON object with exactly these keys:
 "id": the node id as given
 "value": one of specific_knowledge | useful_synthesis | supporting_context | generic_advice | redundant | unsupported
   (specific_knowledge = a fact, decision, event, plan or preference the source supports, findable on its own; useful_synthesis = a supported reading, pattern or distinction beyond any one turn; supporting_context = true but only useful beside another node; generic_advice = the assistant's general guidance with no specific tie to this person; redundant = the same claim another node already holds; unsupported = the source does not support the main claim)
 "fidelity": one of supported | overclaimed | weakened | fabricated_element | mixed
   (overclaimed = certainty, completion, agreement or scope beyond what the source says; weakened = a clear statement turned into a possibility; fabricated_element = a date, number, name, rationale or quote the source does not contain; mixed = both directions)
 "fidelity_evidence": <= 30 words quoting the memory phrase and the source phrase that decide it, or "" when supported
 "status_ok": true/false — a plan stays a plan, a leaning a leaning, a proposal a proposal, a completion only where reported
 "owner_ok": true/false — who said, proposed or did what is preserved; the assistant's advice is not stored as the partner's fact or choice
 "stale": true/false — some surface of this node (title, situation, question, quote, edge) still asserts a value the source itself later replaced; false when nothing changed
 "fields": {"situation": trigger|restated_title|workflow_narration|overbroad|absent, "question": real|formulaic|absent, "reasoning": basis|encoder_justification|generic|restated|absent, "thought": hunch_or_connection|caveat|restatement|noise|absent, "quotes": verbatim_loadbearing|verbatim_incidental|altered|fabricated|absent, "edges": insightful|restating|generic|none, "event_time": correct|wrong|missing_where_supported|absent_ok}
   (situation trigger = a future moment in which this should surface; restated_title = the title again; workflow_narration = about the encoding, not the future. reasoning basis = source, strength, limits; encoder_justification = why it was worth storing; restated = the content again. quotes verbatim = exact words from the source; altered = words changed; fabricated = not in the source. edges insightful = the description says something neither node says alone; restating = repeats the titles or the verb. event_time correct = matches a date the source supports, resolved against that window's clock.)
 "defect": <= 20 words naming the single most consequential problem, or "none"

Reply with a JSON array of these objects, one per node, in the order given, and nothing else."""


def field(n, k):
    if k in n and n[k] not in (None, ''): return n[k]
    return (n.get('_metadata') or {}).get(k)


def edges_of(n):
    out = []
    for e in n.get('connections') or []:
        if e.get('direction') != 'outgoing': continue
        for r in e.get('relations') or [e]:
            if r.get('relation') in ('co_anchored', 'community_member'): continue
            out.append(f"{r.get('relation')} → “{e.get('title', '')}” — {r.get('description') or ''}")
    return out


def card(nid, n):
    lines = [f'id: {nid}', f"type: {n.get('type')}", f"title: {field(n, 'title')}", f"content: {field(n, 'content')}"]
    for k in ('situation', 'question', 'reasoning', 'thought', 'their_raw_quote', 'my_raw_quote', 'event_time', 'confidence', 'evolution_status'):
        v = field(n, k)
        if v not in (None, ''): lines.append(f'{k}: {v}')
    ed = edges_of(n)
    lines.append('edges: ' + ('; '.join(ed) if ed else 'none'))
    return '\n'.join(lines)


def source_block(root, corpus):
    fx = json.loads((Path(root) / (corpus + '.json')).read_text())
    parts = []
    for i, w in enumerate(fx['windows'], 1):
        parts.append(f'<window n="{i}" now="{w.get("now")}">')
        for t in w['turns']:
            parts.append(f"<other>{t.get('other', '')}</other>\n<me>{t.get('me', '')}</me>")
        parts.append('</window>')
    return '\n'.join(parts)


def final_nodes(root):
    root = Path(root)
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(('seed', 'process', 'blind', 'downstream', 'coverage', 'preflight', 'whole'))):
        for rep_dir in sorted(arm_dir.glob('repeat*')):
            for corpus_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir() and not p.name.endswith('.crashed-window2')):
                wins = sorted(corpus_dir.glob('window*/nodes_after.json'), key=lambda p: int(p.parent.name[6:]))
                if wins: yield arm_dir.name, int(rep_dir.name[6:]), corpus_dir.name, json.loads(wins[-1].read_text())


def call(client, user, max_tokens=4000):
    for attempt in range(4):
        try:
            r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=SYSTEM, messages=[{'role': 'user', 'content': user}])
            return ''.join(b.text for b in r.content if hasattr(b, 'text')), r.usage.input_tokens, r.usage.output_tokens
        except Exception:
            if attempt == 3: raise
            time.sleep(5 * (attempt + 1))


def parse(text, ids):
    m = re.search(r'\[.*\]', text, re.S)
    arr = json.loads(m.group() if m else text)
    by = {str(o.get('id')): o for o in arr if isinstance(o, dict)}
    return [by.get(i) for i in ids]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--label', default='cell')
    ap.add_argument('--arms'); ap.add_argument('--limit-seqs', type=int); ap.add_argument('--max-nodes', type=int)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import anthropic
    from isolated_brain import _load_env
    _load_env(); client = anthropic.Anthropic()
    store = out / f'content_quality_{a.label}.jsonl'
    done = set()
    if store.exists():
        for line in store.read_text().splitlines():
            if line.strip(): o = json.loads(line); done.add((o['arm'], o['repeat'], o['corpus'], o['id']))
    seqs = 0; usage = collections.Counter()
    with store.open('a') as fh:
        for root in a.roots:
            for arm, rep, corpus, nodes in final_nodes(root):
                if a.arms and arm not in a.arms.split(','): continue
                if a.limit_seqs and seqs >= a.limit_seqs: break
                seqs += 1
                ids = [i for i in sorted(nodes) if (arm, rep, corpus, i) not in done]
                if a.max_nodes: ids = ids[:a.max_nodes]
                if not ids: continue
                src = source_block(root, corpus)
                for b in range(0, len(ids), BATCH):
                    chunk = ids[b:b + BATCH]
                    user = '# Source conversation\n' + src + '\n\n# Nodes to judge\n\n' + '\n\n'.join(card(i, nodes[i]) for i in chunk)
                    text, tin, tout = call(client, user); usage['in'] += tin; usage['out'] += tout; usage['calls'] += 1
                    try: verdicts = parse(text, chunk)
                    except Exception as e:
                        verdicts = [None] * len(chunk); print('PARSE FAIL', arm, rep, corpus, chunk, str(e)[:80], file=sys.stderr)
                    for i, v in zip(chunk, verdicts):
                        rec = {'arm': arm, 'repeat': rep, 'corpus': corpus, 'id': i, 'type': nodes[i].get('type'), 'title': field(nodes[i], 'title'), 'verdict': v}
                        fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
                print(f'{arm} r{rep} {corpus}: {len(ids)} nodes judged (calls so far {usage["calls"]}, in {usage["in"]:,} out {usage["out"]:,})', flush=True)
    summarize(store, out, a.label)


def summarize(store, out, label):
    recs = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]
    per = collections.OrderedDict()
    keys = ['value', 'fidelity']; fkeys = ['situation', 'question', 'reasoning', 'thought', 'quotes', 'edges', 'event_time']
    for r in recs:
        v = r['verdict']
        s = per.setdefault(r['arm'], {'n': 0, 'unparsed': 0, 'status_ok': 0, 'owner_ok': 0, 'stale': 0, 'value': collections.Counter(), 'fidelity': collections.Counter(), **{k: collections.Counter() for k in fkeys}})
        s['n'] += 1
        if not v: s['unparsed'] += 1; continue
        for k in keys: s[k][str(v.get(k))] += 1
        s['status_ok'] += bool(v.get('status_ok')); s['owner_ok'] += bool(v.get('owner_ok')); s['stale'] += bool(v.get('stale'))
        for k in fkeys: s[k][str((v.get('fields') or {}).get(k))] += 1
    lines = [f'# Content and field quality (Sonnet 4.6 judge, per node against the source) — {label}', '', f'{len(recs)} node verdicts. An instrument for reading at scale, not a verdict; the queue below says where to read.', '']
    def dist(c, n): return ', '.join(f'{k} {v / n:.0%}' for k, v in c.most_common())
    lines += ['| Arm | nodes | value classes | fidelity | status kept | owner kept | stale surface |', '|---|---:|---|---|---:|---:|---:|']
    for arm, s in per.items():
        n = (s['n'] - s['unparsed']) or 1
        lines.append(f"| {arm} | {s['n']} | {dist(s['value'], n)} | {dist(s['fidelity'], n)} | {s['status_ok'] / n:.0%} | {s['owner_ok'] / n:.0%} | {s['stale'] / n:.0%} |")
    lines += ['', '| Arm | ' + ' | '.join(fkeys) + ' |', '|---|' + '---|' * len(fkeys)]
    for arm, s in per.items():
        n = (s['n'] - s['unparsed']) or 1
        lines.append(f'| {arm} | ' + ' | '.join(dist(s[k], n) for k in fkeys) + ' |')
    flagged = [r for r in recs if r['verdict'] and (r['verdict'].get('value') == 'unsupported' or r['verdict'].get('fidelity') in ('fabricated_element', 'overclaimed', 'mixed') or (r['verdict'].get('fields') or {}).get('quotes') in ('altered', 'fabricated') or r['verdict'].get('stale'))]
    lines += ['', f'## Queue — {len(flagged)} flagged nodes (unsupported, fabricated, overclaimed, altered quote or stale)', '']
    for r in flagged:
        v = r['verdict']
        lines.append(f"- **{r['arm']}** r{r['repeat']} {r['corpus']} `{r['id']}` [{r['type']}] {v.get('value')} / {v.get('fidelity')}{' / STALE' if v.get('stale') else ''} — “{str(r['title'])[:90]}” — {v.get('defect')} — {v.get('fidelity_evidence', '')}")
    (out / f'CONTENT-QUALITY-{label}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[4:8 + 2 * len(per)]))
    print(f'flagged {len(flagged)} of {len(recs)} → {out}')


if __name__ == '__main__':
    main()
