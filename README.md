# RAG Evaluation Pipeline

[![CI](https://github.com/JYins/rag-eval-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/JYins/rag-eval-pipeline/actions/workflows/ci.yml)

A config-driven evaluation framework for RAG retrieval systems, benchmarked on HotpotQA and then extended with a local Chinese sermon transcript path. I built this because I wanted something small but actually runnable, where I can compare chunking and retrieval choices clearly instead of just wiring up a chatbot and hoping the retrieval is good enough.

## Why I Built This

I wanted to understand what really changes retrieval quality in a RAG pipeline. Different chunk sizes, sentence splits, BM25 vs dense retrieval, and different embedding models all look like small choices, but they can change results a lot. This repo focuses on evaluation and engineering clarity, not packaging everything into a shiny chatbot.

The HotpotQA path is already runnable end to end: data loading, cleaning, chunking, sparse/dense/hybrid retrieval, metrics, result export, and a Streamlit dashboard. The sermon extension now also has a real labeled question set over local transcripts, so the same eval runner can be reused on a second dataset instead of stopping at a template.

## What It Does

- Downloads and parses a local HotpotQA subset for retrieval experiments
- Cleans and normalizes text before chunking
- Supports 3 chunking strategies: fixed-size, sentence-based, paragraph-based
- Preserves chunk metadata like `doc_id`, `chunk_id`, and `source`
- Runs sparse retrieval with BM25
- Runs dense retrieval with `sentence-transformers` and FAISS
- Can switch dense indexing between FAISS and ChromaDB
- Runs hybrid retrieval with simple weighted rank fusion
- Supports 2 main HotpotQA embedding models and a second local sermon dataset path
- Computes `Recall@1`, `Recall@3`, `Recall@5`, `MRR`, `Hit Rate`, and a simple answer overlap proxy for both English and Chinese text
- Can optionally add a RAGAS id-based context recall metric in the eval output
- Runs config-driven experiments and exports summary + per-query results
- Includes a Streamlit dashboard for cross-config inspection
- Adds a second dataset path for Chinese sermon transcripts with local file staging and labeled eval questions

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
│   ├── experiment_grid.yaml
│   ├── sermon_chromadb_ragas.yaml
│   ├── sermon_doc_dedup.yaml
│   └── sermon.yaml
├── data/
│   ├── raw/                    # downloaded or staged at runtime
│   └── eval/
│       ├── hotpotqa_subset.json
│       └── sermon_questions.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_loader_sermon.py
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
│   ├── prepare_sermon_data.py
│   └── run_eval.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_loader_sermon.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_metrics.py
├── results/
│   └── failure_cases.md
└── docs/
    ├── build_progress.md
    ├── dashboard_placeholder.md
    ├── design_decisions.md
    └── sermon_failure_cases.md
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
pytest tests/test_data_loader.py
pytest tests/test_data_loader_sermon.py
pytest tests/test_chunking.py
pytest tests/test_retrieval.py
pytest tests/test_metrics.py
```

### 5. Run HotpotQA eval

```bash
python scripts/run_eval.py --config configs/default.yaml
```

This is the standard run and uses `500` HotpotQA samples by default.
The dense and hybrid configs will load `sentence-transformers` models, so the first run needs network access or a local Hugging Face cache.
If you only want a local sanity check in an offline environment, you can keep the run going and mark unavailable configs as skipped:

```bash
python scripts/run_eval.py --config configs/default.yaml --limit 20 --skip-unavailable
```

For a quick debug run:

```bash
python scripts/run_eval.py --config configs/default.yaml --limit 20
```

### 6. View dashboard

```bash
streamlit run app/streamlit_app.py
```

Use the sidebar `Dataset preset` switch to jump between the default HotpotQA artifacts and the sermon result files.
It now includes extra presets for the optional `Sermon (Doc Dedupe Study)`, `Sermon (Doc Penalty Study)`, `Sermon (Title-Aware Study)`, `Sermon (Metadata Rerank Study)`, `Sermon (Recommended Dense)`, and `Sermon (ChromaDB + RAGAS)` runs.

### 7. Prepare sermon transcripts

```bash
python scripts/prepare_sermon_data.py
```

This stages the local transcript `.docx` files into `data/raw/sermons/`, keeps the labeled question file at `data/eval/sermon_questions.csv`, and writes a local doc index to `data/eval/sermon_doc_index.csv` for checking doc ids.

### 8. Run sermon eval

```bash
python scripts/run_eval.py --config configs/sermon.yaml
```

This runs the current sermon comparison set on the labeled transcript questions and writes:

- `results/sermon_metrics_summary.csv`
- `results/sermon_per_query_results.json`

The current sermon config compares:

- BM25 sparse retrieval
- multilingual dense retrieval with `paraphrase-multilingual-MiniLM-L12-v2`
- multilingual hybrid retrieval

### 9. Run the optional ChromaDB + RAGAS smoke config

```bash
python scripts/run_eval.py --config configs/sermon_chromadb_ragas.yaml
```

This is a small real verification config for the optional stack:

- dense backend: `chromadb`
- optional metric: `ragas_context_recall`
- output files:
  - `results/sermon_chromadb_ragas_metrics.csv`
  - `results/sermon_chromadb_ragas_per_query.json`

### 10. Run the optional doc-dedupe study

```bash
python scripts/run_eval.py --config configs/sermon_doc_dedup.yaml
```

This keeps the main sermon baseline untouched and writes a separate study output for `dedupe_docs: true`:

- `results/sermon_doc_dedup_metrics.csv`
- `results/sermon_doc_dedup_per_query.json`

### 11. Run the optional doc-penalty study

```bash
python scripts/run_eval.py --config configs/sermon_doc_penalty.yaml
```

This is a softer rerank experiment that pushes repeated chunks from the same sermon down a bit without fully banning them:

- config knob: `doc_repeat_penalty: 2.0`
- output files:
  - `results/sermon_doc_penalty_metrics.csv`
  - `results/sermon_doc_penalty_per_query.json`

### 12. Run the optional title-aware study

```bash
python scripts/run_eval.py --config configs/sermon_title_aware.yaml
```

This keeps the sermon chunking the same, but prefixes each chunk with the sermon title before indexing:

- config knob: `chunking.include_title: true`
- output files:
  - `results/sermon_title_aware_metrics.csv`
  - `results/sermon_title_aware_per_query.json`

### 13. Run the optional metadata-rerank study

```bash
python scripts/run_eval.py --config configs/sermon_metadata_rerank.yaml
```

This keeps the title-aware dense retrieval path, then reranks candidates with small sermon-specific metadata hints:

- `title_hint_boost`: boosts title/day/lesson matches like `第六篇`, `Day6`, `最后一天`
- `opening_chunk_boost`: boosts early chunks when the query explicitly asks about the opening of a sermon
- output files:
  - `results/sermon_metadata_rerank_metrics.csv`
  - `results/sermon_metadata_rerank_per_query.json`

### 14. Run the recommended sermon dense config

```bash
python scripts/run_eval.py --config configs/sermon_dense_recommended.yaml
```

This is the current best sermon dense path in one file:

- title-aware chunks
- metadata rerank enabled
- output files:
  - `results/sermon_dense_recommended_metrics.csv`
  - `results/sermon_dense_recommended_per_query.json`

## Configuration

The config files define settings like:

- chunking strategy
- embedding model
- retrieval mode
- `top_k`

Main files:

- [`configs/default.yaml`](/Users/yinshi/Documents/breadrag/configs/default.yaml)
- [`configs/experiment_grid.yaml`](/Users/yinshi/Documents/breadrag/configs/experiment_grid.yaml)
- [`configs/sermon_chromadb_ragas.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_chromadb_ragas.yaml)
- [`configs/sermon_doc_dedup.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_doc_dedup.yaml)
- [`configs/sermon_dense_recommended.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_dense_recommended.yaml)
- [`configs/sermon_doc_penalty.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_doc_penalty.yaml)
- [`configs/sermon_metadata_rerank.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_metadata_rerank.yaml)
- [`configs/sermon_title_aware.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_title_aware.yaml)
- [`configs/sermon.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon.yaml)

Current setup:

- `configs/default.yaml` is the standard `500`-sample HotpotQA run
- `configs/experiment_grid.yaml` expands to compare 3 chunking strategies, 2 embedding models, and sparse/dense/hybrid retrieval on HotpotQA
- `configs/sermon.yaml` compares sparse, dense, and hybrid retrieval on the labeled sermon questions
- `--limit` can override dataset size for fast local debugging
- Dense configs can set `dense_backend: faiss` or `dense_backend: chromadb`
- Any experiment can set `answer_quality.use_ragas: true` to add the optional `ragas_context_recall` column
- Sermon study configs can set `dedupe_docs: true` or `doc_repeat_penalty: 2.0` for doc-level reranking experiments
- Sermon chunking configs can set `include_title: true` to prepend the sermon title into each chunk before retrieval
- Sermon study configs can also set `metadata_rerank` to do small post-retrieval boosts from title/day hints or early-chunk cues

Example dense config slice with the optional backends:

```yaml
- name: dense_sentence_top3_chromadb
  retrieval_mode: dense
  model_name: all-MiniLM-L6-v2
  dense_backend: chromadb
  top_k: 3
  answer_quality:
    use_ragas: true
  chunking:
    strategy: sentence
    max_sentences: 2
```

## Metrics & Results

The main evaluation metrics in this repo are:

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`
- `Hit Rate`
- answer-quality proxy based on token overlap / keyword hit
  For English it uses word overlap, and for Chinese it uses simple CJK character overlap so the sermon metrics stay readable.
- optional `ragas_context_recall` based on retrieved doc ids vs gold doc ids

Current run artifacts:

- `results/metrics_summary.csv`
- `results/per_query_results.json`
- `results/failure_cases.md`
- `results/sermon_metrics_summary.csv`
- `results/sermon_per_query_results.json`
- `results/sermon_doc_dedup_metrics.csv`
- `results/sermon_doc_dedup_per_query.json`
- `results/sermon_dense_recommended_metrics.csv`
- `results/sermon_dense_recommended_per_query.json`
- `results/sermon_doc_penalty_metrics.csv`
- `results/sermon_doc_penalty_per_query.json`
- `results/sermon_title_aware_metrics.csv`
- `results/sermon_title_aware_per_query.json`
- `results/sermon_metadata_rerank_metrics.csv`
- `results/sermon_metadata_rerank_per_query.json`
- `results/sermon_chromadb_ragas_metrics.csv`
- `results/sermon_chromadb_ragas_per_query.json`
- [`docs/sermon_failure_cases.md`](/Users/yinshi/Documents/breadrag/docs/sermon_failure_cases.md)

Current 500-sample full-grid highlights:

| config slice | config | Recall@3 | MRR | Hit Rate |
|---|---|---:|---:|---:|
| best sparse baseline | `bm25_fixed_top3` | `0.635` | `0.8196` | `0.954` |
| best dense Recall@3 | `dense_paragraph_top3_all-MiniLM-L6-v2` | `0.726` | `0.8968` | `0.988` |
| best dense MRR | `dense_paragraph_top3_multi-qa-MiniLM-L6-cos-v1` | `0.703` | `0.8972` | `0.986` |
| best hybrid Recall@3 | `hybrid_paragraph_top3_all-MiniLM-L6-v2` | `0.742` | `0.8870` | `0.988` |
| best Hit Rate | `hybrid_sentence_top3_all-MiniLM-L6-v2` | `0.702` | `0.8648` | `0.992` |

Average across the whole 15-config HotpotQA grid:

| retrieval mode | avg Recall@3 | avg MRR | avg Hit Rate |
|---|---:|---:|---:|
| `bm25` | `0.6243` | `0.8138` | `0.9500` |
| `dense` | `0.6893` | `0.8809` | `0.9787` |
| `hybrid` | `0.7235` | `0.8791` | `0.9897` |

Current sermon run highlights on the 28 labeled transcript questions:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3_sermon` | `0.2143` | `0.1667` | `0.2857` |
| `dense_sentence_top3_sermon_multilingual` | `0.7500` | `0.6589` | `0.8214` |
| `hybrid_sentence_top3_sermon_multilingual` | `0.5714` | `0.3571` | `0.6429` |

Optional ChromaDB + RAGAS smoke run:

| config | Recall@3 | MRR | Hit Rate | RAGAS Context Recall |
|---|---:|---:|---:|---:|
| `dense_sentence_top3_sermon_chromadb_ragas` | `0.7857` | `0.6071` | `0.7857` | `0.7857` |

Optional doc-dedupe study:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3_sermon_dedup` | `0.2143` | `0.1488` | `0.2143` |
| `dense_sentence_top3_sermon_multilingual_dedup` | `0.7857` | `0.6607` | `0.7857` |
| `hybrid_sentence_top3_sermon_multilingual_dedup` | `0.5714` | `0.3393` | `0.5714` |

Optional doc-penalty study:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3_sermon_doc_penalty` | `0.2143` | `0.1488` | `0.2143` |
| `dense_sentence_top3_sermon_multilingual_doc_penalty` | `0.7857` | `0.6607` | `0.7857` |
| `hybrid_sentence_top3_sermon_multilingual_doc_penalty` | `0.5714` | `0.3393` | `0.5714` |

This is why both doc-level reranking behaviors stay opt-in. Even after expanding the label set, the softer penalty still only ties hard dedupe on the dense path, while BM25 and hybrid do not improve, so I keep these as sermon-only studies instead of changing the main baseline.

Optional title-aware study:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3_sermon_title_aware` | `0.0714` | `0.0714` | `0.0714` |
| `dense_sentence_top3_sermon_multilingual_title_aware` | `0.8929` | `0.7292` | `0.9286` |
| `hybrid_sentence_top3_sermon_multilingual_title_aware` | `0.6429` | `0.3839` | `0.7500` |

This is the clearest sermon-specific gain so far. For multilingual dense retrieval, simply including the sermon title in each chunk helps the model anchor verse, sermon-series, and day-based questions much better. It is not a universal trick though, because BM25 gets much worse when the titles dominate token overlap.

Optional metadata-rerank study:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `dense_sentence_top3_sermon_multilingual_title_aware_base` | `0.8929` | `0.7292` | `0.9286` |
| `dense_sentence_top3_sermon_multilingual_metadata_rerank` | `0.9286` | `0.8214` | `0.9286` |

This is still the cleanest sermon retrieval path so far. The dense retriever does the main work, and the small rerank step still helps on day/title and opening-style questions, but the expanded label set makes it clear this is a strong local heuristic, not a solved benchmark.

- `sermon_004`: query asks about the opening of a sermon, so an early-chunk boost pulls the correct sermon to rank 1
- `sermon_021`: query asks about the last day of the seminar, so a day/title hint boosts `Day6` over semantically similar `Day4` / `Day5` candidates

Current recommended sermon dense config:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `dense_sentence_top3_sermon_multilingual_recommended` | `0.9286` | `0.8214` | `0.9286` |

I keep this as a separate recommended config instead of silently rewriting the shared sermon baseline. That way the repo still shows the honest cross-mode baseline, while also giving one direct command for the best current sermon dense path on the current 28-question set.

## Example Output

Current eval CLI example:

```text
saved metrics summary: results/metrics_summary.csv
saved per-query results: results/per_query_results.json
configs run:
- bm25_fixed_top3: Recall@3=0.6350 MRR=0.8196 HitRate=0.9540
- bm25_sentence_top3: Recall@3=0.6090 MRR=0.8059 HitRate=0.9440
- bm25_paragraph_top3: Recall@3=0.6290 MRR=0.8158 HitRate=0.9520
- dense_fixed_top3_all-MiniLM-L6-v2: Recall@3=0.7010 MRR=0.8812 HitRate=0.9780
- hybrid_paragraph_top3_all-MiniLM-L6-v2: Recall@3=0.7420 MRR=0.8870 HitRate=0.9880
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
- FAISS stays the default dense path because it is the lightest local baseline, but the same retriever can now swap to ChromaDB when I want to inspect a second vector store
- The standard eval config now uses `500` samples, while `--limit` is reserved for debug runs only
- In the full 15-config benchmark, paragraph chunking gave the strongest average retrieval quality while hybrid gave the best average coverage
- The sermon path stages real local transcripts first, then uses a small manually labeled eval set instead of pretending benchmark labels already exist
- On the sermon set, chunk granularity changes alone did not fix the remaining misses, but a simple title-aware chunk text did improve dense retrieval a lot
- On the sermon set, the last two dense misses were resolved by a very small metadata-aware rerank instead of a bigger model or a more complicated architecture

More detail is in [`docs/design_decisions.md`](/Users/yinshi/Documents/breadrag/docs/design_decisions.md).

## Limitations

- The sermon eval set is still small at 28 labeled questions, so the numbers are useful for iteration but not yet a stable benchmark
- The answer-quality score is still a cheap proxy based on token overlap, not a full generated-answer evaluation
- The optional ChromaDB and RAGAS paths are now verified with a real smoke config, but I have not folded them into the main published benchmark tables
- The doc-level reranking studies are useful for failure analysis, but even after the soft rerank tie-break fix they are still sermon-only experiments, not a new default path
- The title-aware chunk study is strong for dense sermon retrieval, but it hurts BM25 badly, so right now it is still an opt-in study instead of replacing the shared sermon baseline
- The metadata-rerank study is still the strongest local sermon path, but after expanding to 28 questions it no longer looks "perfect", which is a healthier signal than the earlier tiny-set result
- The recommended sermon dense config is genuinely the best current path, but it is still built from sermon-specific heuristics over a small label set
- Two of the new misses point at local transcript duplication / misnaming, so the sermon data itself still needs some cleanup before I treat these numbers as stable
- The first multilingual sermon run needs a Hugging Face download unless the model is already cached locally
- The current `ragas` / `langchain` stack emits a Python 3.14 warning during the optional run, even though the config finishes successfully

## Future Work

- Expand the sermon label set toward the original 20-50 question target with tighter coverage across more topics
- Clean or deduplicate the suspicious sermon source files before expanding the label set again, so retrieval misses are less confounded by duplicated content
- Add a side-by-side dataset comparison summary for HotpotQA vs sermon runs
- Tune the soft `doc_repeat_penalty` setting or try chunk-group reranking so repeated-sermon hits do not crowd out better alternatives
- Split the sermon path into retrieval-mode-specific configs if the title-aware dense variant keeps outperforming the one-size-fits-all baseline
- Decide whether the sermon metadata rerank should stay a study config or become the dense sermon default after the label set gets larger
- If the sermon label set grows, revisit whether `configs/sermon_dense_recommended.yaml` should replace the current sermon dense baseline
- Try a larger HotpotQA run in the `1000-2000` range once the local benchmark path feels stable
- Extend CI if needed with result-generation smoke checks after model caching is set up
- Add a stronger RAGAS answer metric beyond the current id-based context recall hook
- Compare more embedding models if the simple baseline stays stable

## License

License is `TBD` for now. I will add an explicit license file before the repo reaches its final v1.0 state.
