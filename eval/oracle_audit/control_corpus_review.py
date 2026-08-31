#!/usr/bin/env python3
"""CORPUS REVIEW SHEET — renders control_corpus.json gold as human-readable TITLES (not IDs) so a
human can eyeball "does this question's gold actually answer it?" Flags likely-over-marked essential
(>=5 nodes). Daemon-independent (IsolatedBrain copy — works even when live recall times out).
Writes control_corpus_review.md (full, both tiers) + prints a compact essential-only sheet.
Usage: ./dev python3 eval/oracle_audit/control_corpus_review.py
"""
import sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

C = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))
QS = C['queries']

with IsolatedBrain() as env:
    b = env.brain

    def t(n8):
        r = b.conn.execute("SELECT type,title FROM nodes WHERE id = ?", (n8,)).fetchone()
        return ("%s: %s" % (r[0][:4], r[1][:52])) if r else ("?? %s (NOT FOUND)" % n8)

    md = ["# Control corpus — gold review (titles)\n",
          "Per question: ESSENTIAL = can't-answer-without; HELPFUL = adds context. ⚠ = essential>=5 (scrutinize).\n"]
    print("=" * 78)
    for q in QS:
        ess = q.get('gold_essential', [])
        helf = q.get('gold_helpful', [])
        flag = ' ⚠OVER?' if len(ess) >= 5 else ''
        print("\n#%-4s [%s]%s  \"%s\"" % (q['id'], q['mode'], flag, q['query']))
        print("  ESSENTIAL(%d):" % len(ess))
        for n in ess:
            print("     - " + t(n))
        print("  helpful: %d nodes" % len(helf))
        md.append("\n### %s [%s]%s — \"%s\"\n" % (q['id'], q['mode'], flag, q['query']))
        md.append("**Essential (%d):**\n" % len(ess))
        for n in ess:
            md.append("- `%s` %s\n" % (n, t(n)))
        md.append("\n*Helpful (%d):* %s\n" % (len(helf), ", ".join("`%s` %s" % (n, t(n).split(': ', 1)[-1][:30]) for n in helf)))

    open(f'{ROOT}/eval/oracle_audit/control_corpus_review.md', 'w').write(''.join(md))
    print("\n" + "=" * 78)
    over = [q['id'] for q in QS if len(q.get('gold_essential', [])) >= 5]
    print("essential>=5 (scrutinize for over-marking): %s" % over)
    print("wrote control_corpus_review.md (full, both tiers)")
