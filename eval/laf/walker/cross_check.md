# cross_check — engine-as_of ↔ walker content lanes (§20.11 d)

engine commit: 91cb447
walker stamps: extract=v6-machineturns-noTSS-manifesthash embed=v2-qvec-incremental lanes=v3-qvec-j0|title,_primary,high_meta,other_meta,edge_context,question

| lane | rows | median |Δ| | p99 |Δ| | max |Δ| | worst node | verdict |
|---|---|---|---|---|---|---|
| v_title_op | 90081 | 0.00e+00 | 0.00e+00 | 0.00e+00 | — | AGREE |
| v_primary_op | 90081 | 0.00e+00 | 0.00e+00 | 0.00e+00 | — | AGREE |
| v_high_meta_op | 90081 | 0.00e+00 | 0.00e+00 | 2.98e-08 | 8f283c6e | AGREE |
| v_other_meta_op | 90081 | 0.00e+00 | 0.00e+00 | 0.00e+00 | — | AGREE |
| v_edge_context_op | 63854 | 0.00e+00 | 0.00e+00 | 0.00e+00 | — | AGREE |
| v_question_op | 90081 | 0.00e+00 | 0.00e+00 | 2.98e-08 | 8f283c6e | AGREE |
| sit_op | 90081 | 0.00e+00 | 0.00e+00 | 0.00e+00 | — | AGREE |
| idf_op | 90081 | 5.55e-17 | 3.88e-02 | 1.84e-01 | d58180c3 | AGREE |

## divergence-class counters (never folded into the stats)
- archived_since_build_nodes: 403
- cand_missing_engine_rows: 2827
- engine_rows: 7374
- engine_title_corpus: 7374
- rows_compared: 90081

**Overall: AGREE**
