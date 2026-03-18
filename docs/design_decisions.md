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

## Why Only Two Main HotpotQA Embedding Models

The main HotpotQA comparison supports:

- `all-MiniLM-L6-v2`
- `multi-qa-MiniLM-L6-cos-v1`

Two models are enough for a meaningful first comparison. More than that would make the repo wider before it becomes deeper.

For the sermon path I added one multilingual option:

- `paraphrase-multilingual-MiniLM-L12-v2`

That is there because the sermon data is Chinese, so using only English-oriented models would be a weak extension.

## Why the Answer Metric Is Still a Proxy

Right now the answer-quality score is based on token overlap and keyword hit against the retrieved context. This is intentionally simple.

I kept it this way because:

- it is cheap to run on a laptop
- it tells me whether answer words are even present in the retrieved text
- it avoids pretending I already have a polished answer generator

RAGAS can come later, but I want the retrieval comparison stable first.

## Trade-Offs From the Current 500-Sample Full Grid

Grid run:

```bash
python scripts/run_eval.py --config configs/experiment_grid.yaml
```

Best configs:

| slice | config | Recall@3 | MRR | Hit Rate |
|---|---|---:|---:|---:|
| best sparse baseline | `bm25_fixed_top3` | 0.635 | 0.8196 | 0.954 |
| best dense Recall@3 | `dense_paragraph_top3_all-MiniLM-L6-v2` | 0.726 | 0.8968 | 0.988 |
| best dense MRR | `dense_paragraph_top3_multi-qa-MiniLM-L6-cos-v1` | 0.703 | 0.8972 | 0.986 |
| best hybrid Recall@3 | `hybrid_paragraph_top3_all-MiniLM-L6-v2` | 0.742 | 0.8870 | 0.988 |
| best hybrid Hit Rate | `hybrid_sentence_top3_all-MiniLM-L6-v2` | 0.702 | 0.8648 | 0.992 |

Average by retrieval mode:

| retrieval mode | avg Recall@3 | avg MRR | avg Hit Rate |
|---|---:|---:|---:|
| `bm25` | 0.6243 | 0.8138 | 0.9500 |
| `dense` | 0.6893 | 0.8809 | 0.9787 |
| `hybrid` | 0.7235 | 0.8791 | 0.9897 |

Average by chunking strategy:

| chunking | avg Recall@3 | avg MRR | avg Hit Rate |
|---|---:|---:|---:|
| `fixed` | 0.6968 | 0.8724 | 0.9784 |
| `sentence` | 0.6666 | 0.8512 | 0.9728 |
| `paragraph` | 0.7066 | 0.8768 | 0.9808 |

What I take from this:

- BM25 is still a respectable baseline, but fixed-size chunking helped it more than sentence chunking in the current setup.
- Dense retrieval clearly improves both average Recall@3 and MRR over BM25 across the grid.
- Hybrid retrieval gives the best average Recall@3 and Hit Rate, so fusion is helping coverage most consistently.
- Dense is still a tiny bit ahead of hybrid on average MRR, which suggests hybrid helps more with coverage than with always improving the first hit.
- Paragraph chunking ended up stronger than I expected. It had the best average Recall@3 and MRR across the 15 configs.
- The two English embedding models were very close: `all-MiniLM-L6-v2` had slightly stronger average Recall@3, while `multi-qa-MiniLM-L6-cos-v1` was a little better on average MRR and Hit Rate.

One useful nuance from the per-query outputs:

- BM25 still wins many individual questions by a simple `(Recall@3, MRR)` tie-break count
- Hybrid still has the best average Recall@3
- Some sentence-chunked hybrid runs have the best Hit Rate even though paragraph chunking is stronger on average

That is why I kept both aggregate metrics and the dashboard. Averages and single-query wins do not tell exactly the same story.

## Why the Sermon Path Starts With File Staging

The sermon extension is local, personal data. So before pretending there is a finished benchmark, I wanted the repo to do two honest things first:

- stage the real local transcript files into `data/raw/sermons/`
- create a labeling template in `data/eval/sermon_questions.csv`

This keeps the extension real. The code can already load the transcript files and run through the same experiment runner once the labels exist. I do not want to fabricate question-answer pairs just to say Phase B is complete.

## What I Would Change Next

- finish the sermon question labels now that the transcript files are staged into `data/raw/sermons/`
- run the same eval loop on the sermon config and compare where HotpotQA trends do or do not transfer
- add a more realistic answer metric later, either RAGAS or a small generated-answer path
- decide whether `chromadb` should stay as a dependency or be wired in for a second dense-storage backend
