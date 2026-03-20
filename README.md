# RAG Evaluation Pipeline

[![CI](https://github.com/JYins/rag-eval-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/JYins/rag-eval-pipeline/actions/workflows/ci.yml)

I built this project after running into the same problem twice. During my SmartEarth internship, I spent a lot of time comparing retrieval setups for long-document QA and saw how much chunking, indexing, and embedding choices could change the result. Later, when I tried building a Chinese sermon QA prototype for my church, the simple "vector DB + LLM API" path felt much worse than I expected. That pushed me to build a smaller but actually runnable evaluation pipeline first, so I could inspect retrieval quality before pretending the generation side was solved.

## Why I Built This

The main goal here is not to make a flashy chatbot demo. I wanted a repo where I can answer a more basic engineering question honestly: what is actually helping retrieval, and what only looks good in a diagram.

The first half of the repo uses HotpotQA because it gives me built-in supervision and lets me compare chunking, sparse/dense/hybrid retrieval, and embedding choices in a controlled way. The second half is more personal: I reused the same pipeline on local Chinese sermon transcripts, labeled a small eval set, and used that to understand why my original church QA idea was underperforming.

That is also why this repo looks the way it does. It is evaluation-heavy on purpose. I would rather understand retrieval failure cases clearly than rush into a polished app with weak grounding.

## What It Does

- Runs a config-driven retrieval benchmark on HotpotQA with 3 chunking strategies, 2 embedding models, and sparse / dense / hybrid retrieval
- Exports real comparison artifacts: summary metrics, per-query retrieval results, and failure-case notes
- Includes a small Streamlit dashboard for checking what changed across configs instead of reading raw JSON by hand
- Reuses the same evaluation runner on a second dataset of local Chinese sermon transcripts
- Supports optional ChromaDB and RAGAS paths without making the whole repo depend on them as the default benchmark flow
- Keeps the code path simple enough to explain in an interview without hiding everything behind a framework

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
The default staging path also reads [`data/eval/sermon_excluded_files.txt`](/Users/yinshi/Documents/breadrag/data/eval/sermon_excluded_files.txt) so confirmed duplicate transcript files stay out of the default eval corpus without touching the original local `.docx` files.

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
- `series_hint_boost`: keeps `布道会` queries inside the seminar series before applying the day/title hint
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
- `scripts/prepare_sermon_data.py` stages `13` default sermon docs after applying the duplicate-file exclusion list in [`data/eval/sermon_excluded_files.txt`](/Users/yinshi/Documents/breadrag/data/eval/sermon_excluded_files.txt)
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

Current sermon baseline highlights on the 50 labeled transcript questions:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3_sermon` | `0.1400` | `0.1393` | `0.3000` |
| `dense_sentence_top3_sermon_multilingual` | `0.7000` | `0.6357` | `0.7400` |
| `hybrid_sentence_top3_sermon_multilingual` | `0.5800` | `0.3400` | `0.6200` |

Current title-aware sermon study on the same 50 labeled questions:

| config | Recall@3 | MRR | Hit Rate | RAGAS Context Recall |
|---|---:|---:|---:|---:|
| `bm25_sentence_top3_sermon_title_aware` | `0.0400` | `0.0400` | `0.0400` | |
| `dense_sentence_top3_sermon_multilingual_title_aware` | `0.8800` | `0.7473` | `0.9000` | |
| `hybrid_sentence_top3_sermon_multilingual_title_aware` | `0.6600` | `0.3850` | `0.8000` | |

Current metadata-rerank and recommended dense sermon runs on the same 50 labeled questions:

| config | Recall@3 | MRR | Hit Rate | RAGAS Context Recall |
|---|---:|---:|---:|---:|
| `dense_sentence_top3_sermon_multilingual_title_aware_base` | `0.8800` | `0.7473` | `0.9000` | |
| `dense_sentence_top3_sermon_multilingual_metadata_rerank` | `1.0000` | `0.9433` | `1.0000` | |
| `dense_sentence_top3_sermon_multilingual_recommended` | `1.0000` | `0.9433` | `1.0000` | |

Current optional sermon-only studies on the same 50 labeled questions:

| config | Recall@3 | MRR | Hit Rate |
|---|---:|---:|---:|
| `bm25_sentence_top3_sermon_dedup` | `0.1400` | `0.1033` | `0.1400` |
| `dense_sentence_top3_sermon_multilingual_dedup` | `0.7400` | `0.6433` | `0.7400` |
| `hybrid_sentence_top3_sermon_multilingual_dedup` | `0.6000` | `0.3367` | `0.6000` |
| `bm25_sentence_top3_sermon_doc_penalty` | `0.1400` | `0.1033` | `0.1400` |
| `dense_sentence_top3_sermon_multilingual_doc_penalty` | `0.7200` | `0.6367` | `0.7200` |
| `hybrid_sentence_top3_sermon_multilingual_doc_penalty` | `0.6000` | `0.3367` | `0.6000` |
| `dense_sentence_top3_sermon_chromadb_ragas` | `0.6800` | `0.5880` | `0.7200` |

This is why both doc-level reranking behaviors still stay opt-in. After refreshing them on the same 50-question set, they still do not beat the main recommended dense path, so they remain useful for inspection rather than as defaults. The ChromaDB + RAGAS run also drops further on the newer day-specific seminar questions, which makes the FAISS-based recommended path look more clearly like the stable local choice.

This is the clearest sermon-specific gain so far. For multilingual dense retrieval, simply including the sermon title in each chunk helps the model anchor verse, sermon-series, and day-based questions much better. It is not a universal trick though, because BM25 gets much worse when the titles dominate token overlap.

This is still the cleanest sermon retrieval path so far. After excluding the two confirmed duplicate sermon files from the default staged corpus, I added one more tiny series-aware rerank hint so `布道会` day queries stop boosting the wrong sermon series.

- `sermon_004`: query asks about the opening of a sermon, so an early-chunk boost pulls the correct sermon to rank 1
- `sermon_021`: query asks about the last day of the seminar, so a day/title hint boosts `Day6` over semantically similar `Day4` / `Day5` candidates

I keep the recommended dense path as a separate config instead of silently rewriting the shared sermon baseline. That way the repo still shows the honest cross-mode baseline, while also giving one direct command for the best current sermon dense path on the current 50-question set.

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

- The sermon eval set is now `50` labeled questions, which is much healthier for iteration, but it is still not large enough to call a stable benchmark
- The answer-quality score is still a simple overlap-style proxy, not a full generated-answer evaluation
- The strongest sermon path today uses sermon-specific hints, so I treat it as a practical local solution, not a universal retrieval recipe
- The optional ChromaDB and RAGAS paths are real and runnable, but they are still side paths compared with the main FAISS-based benchmark flow
- There are probably still local transcript hygiene issues beyond the two duplicate files I already excluded from the default staged corpus
- This repo is intentionally stronger on evaluation than on end-user product design; that trade-off is deliberate, but it is still a limitation

## Future Work

- Keep expanding the sermon label set so the current strong local path gets pressure-tested beyond the current `50` questions
- Use this evaluation pipeline as the retrieval-analysis base for a future sermon-specific model or fine-tuning path, instead of jumping straight back to a weak QA demo
- Revisit the sermon source set again if more near-duplicate transcript files show up during later labeling
- Add a clearer side-by-side summary for HotpotQA vs sermon so cross-dataset transfer is easier to explain
- Compare more embedding models and maybe a stronger answer metric once the current benchmark path feels stable

## License

License is `TBD` for now. I will add an explicit license file before the repo reaches its final v1.0 state.
