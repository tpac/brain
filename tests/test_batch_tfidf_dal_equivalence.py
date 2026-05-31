"""Equivalence gate for the _batch_tfidf_scores DAL refactor (Phase 4 #6).

_batch_tfidf_scores is on the recall HOT PATH. The refactor routed its two
raw reads through the DAL (TfIdfDAL.get_doc_freq + get_tf_vectors_for) and must
be **behaviour-preserving**. This test pins the DAL-routed output to a reference
implementation of the original raw-SQL logic on identical data — the rigorous
"no scoring drift" gate (chosen over an LLM eval, which can't detect small drift).

Covers the load-bearing edge: an absent query term must default to df=1 (the
inline code did `df = row[0] if row else 1`; get_doc_freq returns 0, so the
adopted call uses `... or 1`).

Run: BRAIN_ALLOW_ANY_PYTHON=1 python3 -m pytest tests/test_batch_tfidf_dal_equivalence.py -v
"""
import math
import unittest

from tests.brain_test_base import BrainTestBase


def _reference_batch_tfidf(brain, query_terms, node_ids):
    """The ORIGINAL raw-SQL _batch_tfidf_scores logic, verbatim — the oracle
    the DAL-routed implementation must match exactly."""
    if not query_terms or not node_ids:
        return {}
    total_docs = brain._get_node_count()
    if total_docs == 0:
        return {}

    idf_map = {}
    for term in set(query_terms):
        row = brain.conn.execute(
            'SELECT count FROM doc_freq WHERE term = ?', (term,)).fetchone()
        df = row[0] if row else 1
        idf_map[term] = math.log((total_docs + 1) / (df + 1)) + 1

    query_vec = {}
    for term in query_terms:
        query_vec[term] = query_vec.get(term, 0) + 1
    q_max = max(query_vec.values()) if query_vec else 1
    for t in query_vec:
        query_vec[t] /= q_max

    query_norm_sq = 0
    for term, q_val in query_vec.items():
        idf = idf_map.get(term, 1)
        query_norm_sq += (q_val * idf) ** 2
    query_norm = math.sqrt(query_norm_sq)
    if query_norm == 0:
        return {}

    unique_terms = list(set(query_terms))
    tph = ','.join('?' * len(unique_terms))
    nph = ','.join('?' * len(node_ids))
    cur = brain.conn.execute(
        f'SELECT node_id, term, tf FROM node_vectors '
        f'WHERE term IN ({tph}) AND node_id IN ({nph})',
        unique_terms + node_ids)
    node_term_maps = {}
    for nid, term, tf in cur.fetchall():
        node_term_maps.setdefault(nid, {})[term] = tf

    scores = {}
    for nid in node_ids:
        ntm = node_term_maps.get(nid)
        if not ntm:
            scores[nid] = 0
            continue
        dot = 0
        doc_norm_sq = 0
        for term, tf_val in ntm.items():
            idf = idf_map.get(term, 1)
            d_val = tf_val * idf
            q_val = (query_vec.get(term, 0) or 0) * idf
            dot += q_val * d_val
            doc_norm_sq += d_val * d_val
        doc_norm = math.sqrt(doc_norm_sq)
        scores[nid] = dot / (query_norm * doc_norm) if doc_norm > 0 else 0
    return scores


class TestBatchTfidfDALEquivalence(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.nodes = [
            self.brain.remember(
                type='decision',
                title='Redis caching strategy for user session data',
                content='User sessions cached in Redis with a 30-minute TTL, '
                        'cache-aside pattern, invalidation on password change.'),
            self.brain.remember(
                type='decision',
                title='CDN caching rules for static assets',
                content='Static assets use Cache-Control max-age with content-hash '
                        'filenames; CDN purged on deployment.'),
            self.brain.remember(
                type='decision',
                title='OAuth2 implementation for third-party integrations',
                content='Third parties authenticate via OAuth2 code flow; access '
                        'tokens expire hourly, refresh tokens after thirty days.'),
        ]
        self.brain.save()
        self.ids = [n['id'] for n in self.nodes]

    def _assert_equivalent(self, query_terms):
        got = self.brain._batch_tfidf_scores(query_terms, self.ids)
        want = _reference_batch_tfidf(self.brain, query_terms, self.ids)
        self.assertEqual(set(got), set(want), 'same node-id key set')
        for nid in want:
            self.assertAlmostEqual(
                got[nid], want[nid], places=12,
                msg=f'score drift for {nid} on terms={query_terms}')

    def test_normal_query_matches_reference(self):
        terms = self.brain._tfidf_tokenize('redis caching session strategy')
        self._assert_equivalent(terms)

    def test_absent_term_defaults_df_1(self):
        # A term present nowhere in doc_freq must still score identically —
        # this is the `... or 1` default the refactor had to preserve.
        terms = self.brain._tfidf_tokenize('redis caching') + ['zzqxnonexistentterm']
        self._assert_equivalent(terms)

    def test_all_absent_terms(self):
        self._assert_equivalent(['zzqxnonexistentterm', 'anotherghostterm'])

    def test_empty_inputs(self):
        self.assertEqual(self.brain._batch_tfidf_scores([], self.ids), {})
        self.assertEqual(self.brain._batch_tfidf_scores(['redis'], []), {})


if __name__ == '__main__':
    unittest.main()
