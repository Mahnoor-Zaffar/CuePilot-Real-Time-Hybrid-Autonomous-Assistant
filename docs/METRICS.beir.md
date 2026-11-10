# BEIR subset — retrieval quality

Dataset: **scifact** (test split), 5000 documents, 100 queries, top-5. Embedder: `hash-feature-v1`.

| Mode | recall@k | 95% CI | MRR | 95% CI | hits |
|---|---|---|---|---|---|
| bm25 | 0.6700 | [0.5700, 0.7600] | 0.5590 | [0.4683, 0.6528] | 67/100 |
| dense | 0.5500 | [0.4500, 0.6500] | 0.4115 | [0.3208, 0.5008] | 55/100 |
| hybrid | 0.5800 | [0.4700, 0.6800] | 0.4823 | [0.3865, 0.5775] | 58/100 |

Hybrid vs best single: -9.00 pp.  Hybrid vs dense: +3.00 pp.  Hybrid vs BM25: -9.00 pp.

Paired bootstrap deltas (recall@k, pp):
- paired_hybrid_vs_dense: +3.00 pp [-1.00, +7.00]
- paired_hybrid_vs_bm25: -9.00 pp [-17.00, -1.00]

Notes: queries and documents are sampled (see --max-queries/--max-docs). Bootstrap CIs use 2000 resamples, seed 7. This is an external reference point, not a fit to the cuePilot corpus; the synthetic-corpus numbers in METRICS.md are the headline.
