# cross_check — engine-as_of ↔ walker content lanes (§20.11 d)

engine commit: e335b89
walker stamps: extract=v5-microturns-noTSS-manifesthash embed=v2-qvec-incremental lanes=v3-qvec-j0|title,_primary,high_meta,other_meta,edge_context,question

| lane | rows | median |Δ| | p99 |Δ| | max |Δ| | worst node | verdict |
|---|---|---|---|---|---|---|
| v_title_op | 105776 | 5.96e-08 | 1.79e-07 | 1.70e-02 | a98abd20 | AGREE (drift-explained: 1 nodes) |
| v_primary_op | 105776 | 5.96e-08 | 1.79e-07 | 4.10e-03 | a98abd20 | AGREE (drift-explained: 2 nodes) |
| v_high_meta_op | 105776 | 5.96e-08 | 1.79e-07 | 2.86e-02 | a98abd20 | AGREE (drift-explained: 1 nodes) |
| v_other_meta_op | 105776 | 5.96e-08 | 1.79e-07 | 2.98e-07 | c6fe1581 | AGREE |
| v_edge_context_op | 74350 | 5.96e-08 | 1.79e-07 | 2.98e-07 | 4042419a | AGREE |
| v_question_op | 105775 | 2.98e-08 | 1.19e-07 | 2.98e-07 | 23f4d46e | AGREE |
| sit_op | 105776 | 5.96e-08 | 1.79e-07 | 4.71e-02 | a98abd20 | AGREE (drift-explained: 1 nodes) |
| idf_op | 105776 | 0.00e+00 | 3.53e-02 | 1.84e-01 | d58180c3 | AGREE |

## divergence-class counters (never folded into the stats)
- archived_since_build_nodes: 434
- cand_missing_engine_rows: 3560
- engine_rows: 7365
- engine_title_corpus: 7365
- rows_compared: 105776
- walker_null_engine_ok: 1

## since-build drift nodes (excused from the verdict)
- 64c506c4: revised 2026-07-15T21:10:58.389048+00:00 (> last turn 2026-07-15T19:34:01.473928+00:00)
- a98abd20: revised 2026-07-15T20:53:06.098668+00:00 (> last turn 2026-07-15T19:34:01.473928+00:00)

**Overall: AGREE**
