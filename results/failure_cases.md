# Failure Cases

Generated from the full 15-config HotpotQA run on March 17, 2026 with:

```bash
python scripts/run_eval.py --config configs/experiment_grid.yaml
```

Run summary on 500 samples:

| slice | config | Recall@3 | MRR | Hit Rate |
|---|---|---:|---:|---:|
| best sparse baseline | `bm25_fixed_top3` | 0.635 | 0.8196 | 0.954 |
| best dense Recall@3 | `dense_paragraph_top3_all-MiniLM-L6-v2` | 0.726 | 0.8968 | 0.988 |
| best hybrid Recall@3 | `hybrid_paragraph_top3_all-MiniLM-L6-v2` | 0.742 | 0.8870 | 0.988 |

Main patterns from the full grid:

- Hybrid had the best average Recall@3 and Hit Rate, but dense still edged it slightly on average MRR.
- Paragraph chunking was the strongest average chunking strategy in the current 500-sample run.
- BM25 still had useful wins on easy lexical questions, so it stayed in the comparison instead of being treated like a throwaway baseline.

## Case 1: BM25 over-matched the surface phrase

Question:

> Where is the transportation company that owns Talbot based in?

Gold docs for `bm25_fixed_top3`:

- `NS VIRM`
- `Bombardier Inc.`

Observed retrieval from `bm25_fixed_top3`:

1. `Northern Transportation Company`
2. `Marquette Transportation Company`
3. `Shaver Transportation Company`

Why this failed:

- The query repeats the phrase "transportation company", so BM25 latched onto lexical lookalikes.
- The real bridge is `Talbot -> owner -> Bombardier Inc. -> Montreal`, which is relational, not just lexical.
- This is a good example of BM25 doing exactly what it is supposed to do, but still missing the multi-hop reasoning path.

## Case 2: Dense retrieval stayed in the right neighborhood but missed the exact page

Question:

> Which television series featured an actor who also performed in "The Young Ones"?

Gold docs for `dense_paragraph_top3_all-MiniLM-L6-v2`:

- `Ade Edmondson`
- `Bad News (band)`

Observed retrieval from `dense_paragraph_top3_all-MiniLM-L6-v2`:

1. `The Young Ones (TV series)`
2. `The Young Ones (video game)`
3. `Filthy Rich & Catflap`

Why this failed:

- The dense retriever stayed in the correct semantic neighborhood around `The Young Ones`.
- But topical neighbors still displaced the exact bridge docs the question needed.
- This is the kind of failure that looks reasonable at first glance, which is why per-query inspection matters.

## Case 3: Hybrid still struggled on an indirect bridge

Question:

> What year did the series on CBS, starring the actor who known for his role in "Rebel Without a Cause," air?

Gold docs for `hybrid_paragraph_top3_all-MiniLM-L6-v2`:

- `CBS Television Workshop`
- `James Dean`

Observed retrieval from `hybrid_paragraph_top3_all-MiniLM-L6-v2`:

1. `Jack Grinnage`
2. `Casey Braxton`
3. `Stewart Stern`

Why this failed:

- The retrieved pages stayed near the same TV and film neighborhood, but they missed the exact bridge from actor to series.
- This kind of question still needs the system to recover both entities, not just one thematic cluster.
- Fusion improved averages overall, but it still cannot fully solve every bridge-style miss.

## Main Takeaways

- BM25 fails hardest when the query shares strong surface words with many distractors.
- Dense retrieval improves average Recall@3, but semantically similar pages can still crowd out the exact evidence.
- Hybrid retrieval gave the best average Recall@3 and Hit Rate in the current 500-sample run, but it is not a universal fix.
- Paragraph chunking helped the strongest configs most in this run, which was not obvious before actually benchmarking it.
- Looking only at aggregate metrics would hide these failure patterns. The dashboard is useful because the mistakes are much easier to explain once you inspect the retrieved chunks.
