# Failure Cases

Generated from the standard HotpotQA run on March 17, 2026 with:

```bash
python scripts/run_eval.py --config configs/default.yaml
```

Run summary on 500 samples:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3` | 0.609 | 0.8059 | 0.944 |
| `dense_sentence_top3_minilm` | 0.664 | 0.8652 | 0.968 |
| `hybrid_sentence_top3_minilm` | 0.702 | 0.8648 | 0.992 |

## Case 1: BM25 over-matched the surface phrase

Question:

> Where is the transportation company that owns Talbot based in?

Gold docs:

- `NS VIRM`
- `Bombardier Inc.`

Observed retrieval from `bm25_sentence_top3`:

1. `Marquette Transportation Company`
2. `Shaver Transportation Company`
3. `Northern Transportation Company`

Why this failed:

- The query repeats the phrase "transportation company", so BM25 latched onto lexical lookalikes.
- The real bridge is `Talbot -> owner -> Bombardier Inc. -> Montreal`, which is relational, not just lexical.
- This is a good example of BM25 doing exactly what it is supposed to do, but still missing the multi-hop reasoning path.

## Case 2: Dense retrieval stayed in the right topic but the wrong year

Question:

> What is the stage name of the former pornographic actress born in 1981 that co-hosted the 26th AVN Awards?

Gold docs:

- `26th AVN Awards`
- `Belladonna (actress)`

Observed retrieval from `dense_sentence_top3_minilm`:

1. `16th AVN Awards`
2. `24th AVN Awards`
3. `20th AVN Awards`

Why this failed:

- The dense retriever stayed in the correct semantic neighborhood around AVN awards.
- But semantically similar neighboring award pages still displaced the exact gold page.
- This is the kind of failure that looks reasonable at first glance, which is why per-query inspection matters.

## Case 3: Hybrid still struggled on an indirect bridge

Question:

> What service is an Amtrak flagship that includes BWI Rail Station as one of its Amtrak intercity services?

Gold docs:

- `Acela Express`
- `BWI Rail Station`

Observed retrieval from `hybrid_sentence_top3_minilm`:

1. `BWI Marshall Airport Shuttle`
2. `BWI Business Partnership LINK Shuttle`
3. `Riverside – Downtown station`

Why this failed:

- The station entity pulled retrieval toward nearby transit pages and local shuttle services.
- The question needs the model to jump from the station page to the specific flagship service.
- Fusion improved averages overall, but it still cannot fully solve every bridge-style miss.

## Main Takeaways

- BM25 fails hardest when the query shares strong surface words with many distractors.
- Dense retrieval improves average Recall@3, but semantically similar pages can still crowd out the exact evidence.
- Hybrid retrieval gave the best average Recall@3 and Hit Rate in the current 500-sample run, but it is not a universal fix.
- Looking only at aggregate metrics would hide these failure patterns. The dashboard is useful because the mistakes are much easier to explain once you inspect the retrieved chunks.
