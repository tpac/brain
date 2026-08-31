#!/usr/bin/env python3
"""CONTROL-FAILS, FAITHFUL (stage 3+4) — LLM-judged gold, the version Tom designed.

Per control_corpus question:
  1. Build a candidate POOL (time-scoped): Control recall top-15 + keyword/fts top-15 + graph-walk of
     Control's top-5. Fetch each node's title + content snippet.
  2. JUDGE (Sonnet, structured output): which nodes are ESSENTIAL (answer requires them) vs HELPFUL
     (meaningfully add) vs ignore. Independent of recall RANK — judged on content.
  3. SCORE Control: are the essential/helpful nodes in Control's time-scoped top-5 / top-25?
     Control FAILS when essential gold is missing from its top-5 (and worse, top-25).

This is the test the deterministic proxies (presence saturated, rank tied, cosine-reach overcounted)
couldn't give: it judges NEED, not similarity. Time-the-brain: episodes scoped to their moment.
Writes gold back into control_corpus.json + dumps control_gold_result.json. Robust: per-question
try/except, incremental save, judge failure degrades to pool-only. Daemon-safe (IsolatedBrain).
Usage: ./dev python3 eval/oracle_audit/control_gold_judge.py
"""
import sys, json, re
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers import embedder                      # noqa: E402
import anthropic                                  # noqa: E402

CORPUS_PATH = f'{ROOT}/eval/oracle_audit/control_corpus.json'
CORPUS = json.load(open(CORPUS_PATH))
QS = CORPUS['queries']
TIMED = {'episode'}
MODEL = 'claude-sonnet-4-6'
SCHEMA = {'type': 'object', 'additionalProperties': False,
          'properties': {'essential': {'type': 'array', 'items': {'type': 'string'}},
                         'helpful': {'type': 'array', 'items': {'type': 'string'}}},
          'required': ['essential', 'helpful']}
SYS = ("You evaluate a memory system. Given a user's question/utterance and candidate memory nodes "
       "(id | type | title | snippet), classify which the system SHOULD surface:\n"
       "- essential: the IRREDUCIBLE CORE — ONLY nodes without which the answer is factually WRONG or "
       "fundamentally incomplete. Cap: typically 1-3, rarely more. When in doubt, it is NOT essential.\n"
       "- helpful: everything else that meaningfully adds context.\n"
       "Demote near-duplicates and cluster-padding to helpful — essential is the MINIMAL set a correct "
       "answer cannot omit. For a procedural cue ('let's commit'), essential = the 1-2 process/rule "
       "memories that MUST fire. Return ONLY ids from the provided list.")
client = anthropic.Anthropic(timeout=90)


def cutoff_for(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


with IsolatedBrain() as env:
    b = env.brain

    def fetch(n8):
        r = b.conn.execute("SELECT id,type,title,substr(content,1,260) FROM nodes WHERE id = ?",
                           (n8,)).fetchone()
        return (r[1], r[2], (r[3] or '')) if r else ('?', '?', '')

    def neighbors(n8):
        rows = b.conn.execute("SELECT target_id FROM edges WHERE source_id = ? "
                              "UNION SELECT source_id FROM edges WHERE target_id = ?",
                              (n8, n8)).fetchall()
        return [x[0][:8] for x in rows][:8]

    def judge(query, pool):
        lines = []
        for n8 in pool:
            t, title, snip = fetch(n8)
            lines.append("%s | %s | %s | %s" % (n8, t, title[:60], snip[:180].replace('\n', ' ')))
        msg = "USER UTTERANCE:\n%s\n\nCANDIDATE NODES:\n%s" % (query, '\n'.join(lines))
        resp = client.messages.create(
            model=MODEL, max_tokens=1200,
            output_config={'format': {'type': 'json_schema', 'schema': SCHEMA}},
            messages=[{'role': 'user', 'content': SYS + "\n\n" + msg}])
        txt = resp.content[0].text if resp.content else '{}'
        d = json.loads(txt)
        return [x[:8] for x in d.get('essential', [])], [x[:8] for x in d.get('helpful', [])]

    results = []
    for q in QS:
        try:
            cutoff = cutoff_for(q)
            elig = None
            if cutoff:
                elig = {x[0][:8] for x in b.conn.execute(
                    "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0", (cutoff,)).fetchall()}
            filt = {"created_at": {"lte": cutoff}} if cutoff else None
            try:
                out = b.recall(query=q['query'], limit=25, filter=filt)
            except Exception:
                out = b.recall(query=q['query'], limit=25)
            ctrl = [(r.get('id') or r.get('node_id'))[:8]
                    for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
            if elig is not None:
                ctrl = [n for n in ctrl if n in elig]
            top5, top25 = set(ctrl[:5]), set(ctrl[:25])

            fts = [n[:8] for n, _ in b._fts.search_scored(q['query'], 15)]
            graph = []
            for s in ctrl[:5]:
                graph += neighbors(s)
            pool = list(dict.fromkeys(ctrl[:15] + fts + graph))
            if elig is not None:
                pool = [n for n in pool if n in elig]
            pool = pool[:28]

            if q['id'] == 'EP5':            # preserve Tom's hand-fix — do NOT re-judge
                ess = [n[:8] for n in q.get('gold_essential', [])]
                helf = [n[:8] for n in q.get('gold_helpful', [])]
            else:
                ess, helf = judge(q['query'], pool)
                ess = [n for n in ess if n in set(pool)]
                helf = [n for n in helf if n in set(pool) and n not in set(ess)]
                q['gold_essential'], q['gold_helpful'] = ess, helf

            ess_miss5 = [n for n in ess if n not in top5]
            ess_miss25 = [n for n in ess if n not in top25]
            help_miss5 = [n for n in helf if n not in top5]
            rec = {'id': q['id'], 'mode': q['mode'], 'cutoff': cutoff, 'query': q['query'],
                   'essential': ess, 'helpful': helf,
                   'ess_in_top5': len(ess) - len(ess_miss5), 'ess_total': len(ess),
                   'ess_miss_top5': ess_miss5, 'ess_miss_top25': ess_miss25,
                   'help_in_top5': len(helf) - len(help_miss5), 'help_total': len(helf),
                   'control_fails': bool(ess_miss5)}
            results.append(rec)
            print("#%-4s %-8s cut=%s ess=%d(top5:%d top25-miss:%d) help=%d(top5:%d) FAIL=%s"
                  % (q['id'], q['mode'], (cutoff or 'today')[:10], len(ess), rec['ess_in_top5'],
                     len(ess_miss25), len(helf), rec['help_in_top5'], rec['control_fails']))
        except Exception as e:
            print("#%-4s ERROR %s" % (q['id'], e))
            results.append({'id': q['id'], 'error': str(e)})
        json.dump(results, open(f'{ROOT}/eval/oracle_audit/control_gold_result.json', 'w'), indent=2)
        json.dump(CORPUS, open(CORPUS_PATH, 'w'), indent=2)   # persist gold back into corpus

    ok = [r for r in results if 'control_fails' in r]
    fails = [r for r in ok if r['control_fails']]
    ess_q = [r for r in ok if r['ess_total'] > 0]
    print("\n=== FAITHFUL CONTROL-FAILS (essential gold NOT in Control top-5) ===")
    print("  questions with essential gold: %d/%d" % (len(ess_q), len(ok)))
    print("  CONTROL FAILS (>=1 essential missing from top-5): %d/%d" % (len(fails), len(ok)))
    tot_ess = sum(r['ess_total'] for r in ess_q)
    tot_in5 = sum(r['ess_in_top5'] for r in ess_q)
    tot_miss25 = sum(len(r['ess_miss_top25']) for r in ess_q)
    print("  essential nodes: %d total, %d in top-5 (%.0f%%), %d missing even from top-25"
          % (tot_ess, tot_in5, 100.0 * tot_in5 / max(tot_ess, 1), tot_miss25))
    from collections import Counter
    print("  fails by mode:", dict(Counter(r['mode'] for r in fails)))
    print("\nwrote control_gold_result.json + gold back into control_corpus.json")
