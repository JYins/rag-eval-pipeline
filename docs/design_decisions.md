# Design Decisions

This repo is intentionally small and explainable. I wanted a pipeline I could actually run and talk through in an interview, not a giant framework with too many abstractions.

## Why HotpotQA Comes First

HotpotQA gives me multi-hop questions and supporting facts out of the box. That means I can build the evaluation path first without doing manual labeling on day one.

This solves two problems:

- I can measure retrieval quality right away
- I can delay the Chinese sermon labeling work until the pipeline is stable

The sermon extension is still important, but it should come after the benchmark path is solid.

## Why the Standard Run Uses 500 Samples

`AGENTS.md` says the subset should stay in the `500–2000` range for local eval. That is the right trade-off.

- `20` samples are fine for debugging
- `500` samples already feel much more stable
- `1000+` samples can come later when I want stronger comparison confidence

So the repo now does both:

- `configs/default.yaml` uses `500` for the standard run
- `python scripts/run_eval.py --config configs/default.yaml --limit 20` is available for fast local debugging

## Why These Chunking Strategies

I picked 3 chunking strategies that are easy to explain and compare:

- fixed-size
- sentence-based
- paragraph-based

They are simple baselines, but that is the point. I do not want to jump into complicated semantic chunking before I have clear baseline numbers.

## Why BM25, Dense, and Hybrid

The retrieval path is layered in the most natural order:

1. BM25 for a lexical baseline
2. Dense retrieval for semantic matching
3. Hybrid retrieval for a simple fusion check

This is the simplest reasonable stack that still supports the resume claim directly.

## Why Only Two Embedding Models

The code supports:

- `all-MiniLM-L6-v2`
- `multi-qa-MiniLM-L6-cos-v1`

Two models are enough for a meaningful first comparison. More than that would make the repo wider before it becomes deeper.

## Why the Answer Metric Is Still a Proxy

Right now the answer-quality score is based on token overlap and keyword hit against the retrieved context. This is intentionally simple.

I kept it this way because:

- it is cheap to run on a laptop
- it tells me whether answer words are even present in the retrieved text
- it avoids pretending I already have a polished answer generator

RAGAS can come later, but I want the retrieval comparison stable first.

## Trade-Offs From the Current 500-Sample Run

Standard run:

```bash
python scripts/run_eval.py --config configs/default.yaml
```

Summary:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3` | 0.609 | 0.8059 | 0.944 |
| `dense_sentence_top3_minilm` | 0.664 | 0.8652 | 0.968 |
| `hybrid_sentence_top3_minilm` | 0.702 | 0.8648 | 0.992 |

What I take from this:

- BM25 is still a strong baseline. It wins many easy lexical questions and stays very competitive.
- Dense retrieval improves both average Recall@3 and MRR over BM25 in the current run.
- Hybrid retrieval gives the best Recall@3 and Hit Rate, so fusion seems to help coverage on multi-hop evidence.
- Dense and hybrid are almost tied on MRR, which suggests fusion helps more with coverage than with the very first hit.

One useful nuance from the per-query outputs:

- BM25 still wins many individual questions by a simple `(Recall@3, MRR)` tie-break count
- Hybrid still has the best average Recall@3

That is why I kept both aggregate metrics and the dashboard. Averages and single-query wins do not tell exactly the same story.

## What I Would Change Next

- expand the standard experiment grid so all 3 chunking strategies are compared directly
- run the second embedding model on the same 500-sample standard setup and document it carefully
- add CI before calling the repo complete
- add the Chinese sermon dataset only after the HotpotQA path is fully stable
