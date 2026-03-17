# RAG Evaluation Pipeline

![CI status](https://img.shields.io/badge/CI-not%20set%20up%20yet-lightgrey)

A config-driven evaluation framework for RAG retrieval systems, benchmarked first on HotpotQA and later planned for Chinese sermon transcripts. I built this because I wanted something small but actually runnable, where I can compare chunking and retrieval choices clearly instead of just wiring up a chatbot and hoping the retrieval is good enough.

## Why I Built This

I wanted to understand what really changes retrieval quality in a RAG pipeline. Different chunk sizes, sentence splits, BM25 vs dense retrieval, and different embedding models all look like small choices, but they can change results a lot. This repo focuses on evaluation and engineering clarity, not packaging everything into a shiny chatbot.

The current Phase A path on HotpotQA is already runnable end to end: data loading, cleaning, chunking, sparse/dense/hybrid retrieval, metrics, result export, and a Streamlit dashboard. The sermon extension is still future work after the benchmark pipeline is fully cleaned up.

## What It Does

- Downloads and parses a local HotpotQA subset for retrieval experiments
- Cleans and normalizes text before chunking
- Supports 3 chunking strategies: fixed-size, sentence-based, paragraph-based
- Preserves chunk metadata like `doc_id`, `chunk_id`, and `source`
- Runs sparse retrieval with BM25
- Runs dense retrieval with `sentence-transformers` and FAISS
- Runs hybrid retrieval with simple weighted rank fusion
- Supports 2 embedding model choices
- Computes `Recall@1`, `Recall@3`, `Recall@5`, `MRR`, `Hit Rate`, and a simple answer overlap proxy
- Runs config-driven experiments and exports summary + per-query results
- Includes a Streamlit dashboard for cross-config inspection

## Project Structure

```text
rag-eval-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
├── configs/
│   ├── default.yaml
│   └── experiment_grid.yaml
├── data/
│   ├── raw/                    # downloaded at runtime
│   └── eval/
│       └── hotpotqa_subset.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── cleaning.py
│   ├── chunking.py
│   ├── indexing.py
│   ├── retriever_bm25.py
│   ├── retriever_dense.py
│   ├── retriever_hybrid.py
│   ├── eval_metrics.py
│   ├── answer_quality.py
│   ├── experiment_runner.py
│   └── utils.py
├── scripts/
│   ├── download_data.py
│   └── run_eval.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_metrics.py
├── results/
│   └── failure_cases.md
└── docs/
    ├── build_progress.md
    ├── dashboard_placeholder.md
    └── design_decisions.md
```

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download HotpotQA subset

```bash
python scripts/download_data.py
```

Expected output:

- a local subset file at `data/eval/hotpotqa_subset.json`

### 3. Run retrieval baselines

BM25 example:

```bash
python src/retriever_bm25.py --sample-index 0 --top-k 3 --strategy sentence --max-sentences 2
```

Dense example:

```bash
python src/retriever_dense.py --sample-index 0 --top-k 3 --strategy sentence --max-sentences 2 --model-name all-MiniLM-L6-v2
```

### 4. Run tests

```bash
pytest tests/test_chunking.py
pytest tests/test_retrieval.py
pytest tests/test_metrics.py
```

### 5. Run full eval

```bash
python scripts/run_eval.py --config configs/default.yaml
```

This is the standard run and now uses `500` HotpotQA samples by default.

For a quick debug run:

```bash
python scripts/run_eval.py --config configs/default.yaml --limit 20
```

### 6. View dashboard

```bash
streamlit run app/streamlit_app.py
```

## Configuration

The config files define settings like:

- chunking strategy
- embedding model
- retrieval mode
- `top_k`

Main files:

- [`configs/default.yaml`](/Users/yinshi/Documents/breadrag/configs/default.yaml)
- [`configs/experiment_grid.yaml`](/Users/yinshi/Documents/breadrag/configs/experiment_grid.yaml)

Current setup:

- `configs/default.yaml` is the standard `500`-sample run
- `configs/experiment_grid.yaml` expands a larger config set
- `--limit` can override dataset size for fast local debugging

## Metrics & Results

The main evaluation metrics in this repo are:

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`
- `Hit Rate`
- answer-quality proxy based on token overlap / keyword hit

Current standard-run artifacts:

- `results/metrics_summary.csv`
- `results/per_query_results.json`
- `results/failure_cases.md`

Current 500-sample comparison table:

| config | retrieval_mode | embedding_model | Recall@3 | MRR | Hit Rate |
|---|---|---|---:|---:|---:|
| `bm25_sentence_top3` | `bm25` | — | `0.609` | `0.8059` | `0.944` |
| `dense_sentence_top3_minilm` | `dense` | `all-MiniLM-L6-v2` | `0.664` | `0.8652` | `0.968` |
| `hybrid_sentence_top3_minilm` | `hybrid` | `all-MiniLM-L6-v2` | `0.702` | `0.8648` | `0.992` |

## Example Output

Current eval CLI example:

```text
saved metrics summary: results/metrics_summary.csv
saved per-query results: results/per_query_results.json
configs run:
- bm25_sentence_top3: Recall@3=0.6090 MRR=0.8059 HitRate=0.9440
- dense_sentence_top3_minilm: Recall@3=0.6640 MRR=0.8652 HitRate=0.9680
- hybrid_sentence_top3_minilm: Recall@3=0.7020 MRR=0.8648 HitRate=0.9920
```

Current dense retrieval example:

```text
question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
model: all-MiniLM-L6-v2
rank=1 score=0.7090 doc_id=Kiss and Tell (1945 film) support=True
Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer...
```

Dashboard screenshot placeholder:

- [`docs/dashboard_placeholder.md`](/Users/yinshi/Documents/breadrag/docs/dashboard_placeholder.md)

## Design Decisions

I kept the design simple on purpose.

- HotpotQA comes first because it already has supporting facts, so I can measure retrieval quality without manual labeling
- Fixed-size, sentence-based, and paragraph-based chunking are easy to explain and compare in an interview
- BM25 gives a strong sparse baseline with very little machinery
- Dense retrieval with FAISS is the next simplest reasonable step for semantic search
- Hybrid retrieval is simple enough to add, but still useful for checking multi-hop coverage
- The standard eval config now uses `500` samples, while `--limit` is reserved for debug runs only

More detail is in [`docs/design_decisions.md`](/Users/yinshi/Documents/breadrag/docs/design_decisions.md).

## Limitations

- The sermon transcript dataset has not started yet
- The default experiment currently compares retrieval modes on sentence-based chunking first; the wider chunking comparison still needs to be expanded in config
- The answer-quality score is still a cheap proxy based on token overlap, not a full generated-answer evaluation
- `chromadb` and `ragas` are listed in dependencies, but they are not wired into the current code path yet
- No GitHub Actions CI yet, so the badge is still a placeholder

## Future Work

- Expand the standard experiment grid to compare all 3 chunking strategies directly
- Run and document the second embedding model on the same 500-sample setup
- Add GitHub Actions CI and final cleanup
- Extend the same framework to Chinese sermon transcripts
- Add optional RAGAS-based answer evaluation
- Compare more embedding models if the simple baseline stays stable

## License

License is `TBD` for now. I will add an explicit license file before the repo reaches its final v1.0 state.
