# Build Progress

## Current Step

- Current repo status: HotpotQA Phase A and the sermon extension are runnable, and the current sermon main path has now been pressure-tested on 35 labeled questions
- Next step: keep expanding the sermon label set from this cleaner 13-doc baseline instead of assuming the current full-hit result is already stable enough
- Small note this round: the slower sermon-only studies are now also refreshed on 35 labels after forcing offline model loading through the local cache
- Next phase: add another batch of discriminative sermon labels, especially around scripture-reference prompts and same-series day questions

## Last Step

- Added 7 new high-confidence sermon labels in [`data/eval/sermon_questions.csv`](/Users/yinshi/Documents/breadrag/data/eval/sermon_questions.csv), taking the local sermon eval set from 28 to 35 questions
- Re-ran the main sermon comparison configs plus the slower sermon-only study configs on the cleaned 13-doc staged corpus
- Verified the current main-path results on 35 questions:
  - baseline dense: Recall@3 `0.7429`, MRR `0.6652`, Hit Rate `0.8000`
  - title-aware dense: Recall@3 `0.9143`, MRR `0.7810`, Hit Rate `0.9143`
  - metadata-reranked dense / recommended dense: Recall@3 `1.0000`, MRR `0.9333`, Hit Rate `1.0000`
- Verified the current slower study results on the same 35-question set:
  - doc dedupe dense: Recall@3 `0.8000`, MRR `0.6762`, Hit Rate `0.8000`
  - doc penalty dense: Recall@3 `0.7714`, MRR `0.6667`, Hit Rate `0.7714`
  - ChromaDB + RAGAS dense: Recall@3 `0.7714`, MRR `0.6429`, Hit Rate `0.7714`, `ragas_context_recall=0.7714`
- Used `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for these reruns so the local Hugging Face cache could be reused even when the machine could not resolve `huggingface.co`
- Updated [`README.md`](/Users/yinshi/Documents/breadrag/README.md) and [`docs/sermon_failure_cases.md`](/Users/yinshi/Documents/breadrag/docs/sermon_failure_cases.md) so the sermon narrative now matches a fully refreshed 35-question checkpoint

- Added a small `series_hint_boost` path to [`src/experiment_runner.py`](/Users/yinshi/Documents/breadrag/src/experiment_runner.py) and wired it through [`configs/sermon_metadata_rerank.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_metadata_rerank.yaml) plus [`configs/sermon_dense_recommended.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_dense_recommended.yaml)
- Added regression coverage in [`tests/test_experiment_runner.py`](/Users/yinshi/Documents/breadrag/tests/test_experiment_runner.py) for series-aware reranking so `第四天布道会` does not get hijacked by `第四讲`
- Re-ran the metadata study and recommended dense config on the cleaned corpus: both now reach Recall@3 `1.0000`, and the recommended dense config is `MRR=0.9167`, `Hit Rate=1.0000` on the current 28-question set
- Verified locally with `32` passing targeted tests from [`tests/test_data_loader_sermon.py`](/Users/yinshi/Documents/breadrag/tests/test_data_loader_sermon.py), [`tests/test_experiment_runner.py`](/Users/yinshi/Documents/breadrag/tests/test_experiment_runner.py), and [`tests/test_streamlit_app.py`](/Users/yinshi/Documents/breadrag/tests/test_streamlit_app.py)

## Previous Step

- Added an explicit exclusion list in [`data/eval/sermon_excluded_files.txt`](/Users/yinshi/Documents/breadrag/data/eval/sermon_excluded_files.txt) for the confirmed duplicate `第5讲` and `第9讲` source files
- Updated [`scripts/prepare_sermon_data.py`](/Users/yinshi/Documents/breadrag/scripts/prepare_sermon_data.py) so default staging applies that exclusion list before creating symlinks, while leaving the original local `.docx` files untouched
- Added regression coverage in [`tests/test_data_loader_sermon.py`](/Users/yinshi/Documents/breadrag/tests/test_data_loader_sermon.py) for parsing the exclusion list and skipping blocked sermon files during staging
- Re-ran every sermon result artifact on the cleaned 13-doc staged corpus and verified the current best path on 28 questions: recommended dense Recall@3 `0.9643`, MRR `0.8571`, Hit Rate `0.9643`
- Verified locally with `30` passing targeted tests from [`tests/test_data_loader_sermon.py`](/Users/yinshi/Documents/breadrag/tests/test_data_loader_sermon.py), [`tests/test_experiment_runner.py`](/Users/yinshi/Documents/breadrag/tests/test_experiment_runner.py), and [`tests/test_streamlit_app.py`](/Users/yinshi/Documents/breadrag/tests/test_streamlit_app.py)

- Added 7 new high-confidence sermon labels in [`data/eval/sermon_questions.csv`](/Users/yinshi/Documents/breadrag/data/eval/sermon_questions.csv), focused on Day2-Day6 seminar passages that can be read directly from the local transcripts
- Raised the study config limits in [`configs/sermon_doc_dedup.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_doc_dedup.yaml), [`configs/sermon_doc_penalty.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_doc_penalty.yaml), [`configs/sermon_title_aware.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_title_aware.yaml), [`configs/sermon_metadata_rerank.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_metadata_rerank.yaml), [`configs/sermon_dense_recommended.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_dense_recommended.yaml), and [`configs/sermon_chromadb_ragas.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_chromadb_ragas.yaml) so all sermon presets run against the same expanded pool
- Re-ran every sermon result artifact and verified the current best path on 28 questions: recommended dense Recall@3 `0.9286`, MRR `0.8214`, Hit Rate `0.9286`
- Verified locally with `28` passing targeted tests from [`tests/test_data_loader_sermon.py`](/Users/yinshi/Documents/breadrag/tests/test_data_loader_sermon.py), [`tests/test_experiment_runner.py`](/Users/yinshi/Documents/breadrag/tests/test_experiment_runner.py), and [`tests/test_streamlit_app.py`](/Users/yinshi/Documents/breadrag/tests/test_streamlit_app.py)

- Added [`configs/sermon_dense_recommended.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_dense_recommended.yaml) as the one-command best sermon dense path
- Exposed it in [`app/streamlit_app.py`](/Users/yinshi/Documents/breadrag/app/streamlit_app.py) as `Sermon (Recommended Dense)` without changing the shared baseline preset
- Added regression coverage in [`tests/test_experiment_runner.py`](/Users/yinshi/Documents/breadrag/tests/test_experiment_runner.py) and [`tests/test_streamlit_app.py`](/Users/yinshi/Documents/breadrag/tests/test_streamlit_app.py) for the new recommended config and preset
- Verified the recommended dense run locally on the earlier 21-question set before this expansion pass

- Added [`configs/sermon_doc_dedup.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_doc_dedup.yaml) as a clean opt-in study for `dedupe_docs: true` instead of changing the main sermon baseline
- Verified the trade-off locally: doc dedupe helped the dense sermon run's Recall@3, but it hurt BM25 and hybrid hit rate, so it stays experimental
- Exposed the doc-dedupe study as another dashboard preset in [`app/streamlit_app.py`](/Users/yinshi/Documents/breadrag/app/streamlit_app.py)

- Installed local `chromadb` and `ragas` into the project venv and verified the optional path with [`configs/sermon_chromadb_ragas.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon_chromadb_ragas.yaml)
- Fixed [`src/indexing.py`](/Users/yinshi/Documents/breadrag/src/indexing.py) to batch Chroma inserts so larger chunk sets do not exceed ChromaDB's max batch size
- Added regression coverage in [`tests/test_retrieval.py`](/Users/yinshi/Documents/breadrag/tests/test_retrieval.py) and [`tests/test_experiment_runner.py`](/Users/yinshi/Documents/breadrag/tests/test_experiment_runner.py) for the optional config and Chroma batching
- Verified the real smoke run locally on the earlier 21-question set before this expansion pass

- Added `--skip-unavailable` to [`scripts/run_eval.py`](/Users/yinshi/Documents/breadrag/scripts/run_eval.py) and skip reporting in [`src/experiment_runner.py`](/Users/yinshi/Documents/breadrag/src/experiment_runner.py)
- Kept the default path fail-loud, but now offline debug runs can still export `metrics_summary.csv` with explicit `status=skipped` rows
- Added coverage in [`tests/test_experiment_runner.py`](/Users/yinshi/Documents/breadrag/tests/test_experiment_runner.py) for skipped experiment reporting
- Verified locally with `34` passing tests, clean `ruff`, and a debug eval run that writes BM25 results plus skipped dense/hybrid rows

## Earlier Step

- Added an optional `dense_backend` switch in [`src/retriever_dense.py`](/Users/yinshi/Documents/breadrag/src/retriever_dense.py) and [`src/retriever_hybrid.py`](/Users/yinshi/Documents/breadrag/src/retriever_hybrid.py) so dense search can use FAISS or ChromaDB
- Added optional `ragas_context_recall` scoring in [`src/answer_quality.py`](/Users/yinshi/Documents/breadrag/src/answer_quality.py) and [`src/experiment_runner.py`](/Users/yinshi/Documents/breadrag/src/experiment_runner.py)
- Updated [`README.md`](/Users/yinshi/Documents/breadrag/README.md) and [`docs/design_decisions.md`](/Users/yinshi/Documents/breadrag/docs/design_decisions.md) so the documented stack now matches the runnable code path
- Verified locally with `32` passing tests and clean `ruff` output

- Restored dense and hybrid sermon runs in [`configs/sermon.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon.yaml) with the multilingual MiniLM model
- Added a dataset preset switch in [`app/streamlit_app.py`](/Users/yinshi/Documents/breadrag/app/streamlit_app.py) so the dashboard can jump between HotpotQA and sermon artifacts
- Cached shared retrievers in [`src/experiment_runner.py`](/Users/yinshi/Documents/breadrag/src/experiment_runner.py) so repeated sermon docs do not rebuild indexes for every query
- Verified the first sermon dense baseline locally on the earlier 21-question set before later studies and relabeling

## History

### Commit 1

- Message: `init clean project skeleton`
- Built folder structure, placeholder files, `requirements.txt`, `.gitignore`

### Commit 2

- Message: `add hotpotqa data loader and cleaning`
- Added HotpotQA download script in [`scripts/download_data.py`](/Users/yinshi/Documents/breadrag/scripts/download_data.py)
- Added parsing helpers in [`src/data_loader.py`](/Users/yinshi/Documents/breadrag/src/data_loader.py)
- Added text cleaning in [`src/cleaning.py`](/Users/yinshi/Documents/breadrag/src/cleaning.py)
- Generated subset file: [`data/eval/hotpotqa_subset.json`](/Users/yinshi/Documents/breadrag/data/eval/hotpotqa_subset.json)

### Commit 3

- Message: `implement 3 chunking strategies`
- Added fixed-size, sentence-based, paragraph-based chunking
- Preserved chunk metadata for later retrieval and eval
- Verified with `pytest tests/test_chunking.py`

### Commit 4

- Message: `add bm25 retrieval baseline`
- Added BM25 retriever over chunked docs
- Added simple CLI demo for one query search
- Verified with `pytest tests/test_retrieval.py`

### Commit 5

- Message: `add dense retrieval with faiss`
- Added FAISS index helpers and dense retriever with 2 embedding model options
- Added runnable dense CLI path
- Extended retrieval tests to cover dense search path

### Commit 6

- Message: `add hybrid retrieval and eval metrics`
- Added hybrid retrieval with weighted rank fusion
- Added Recall@K, MRR, Hit Rate, and query scoring helpers
- Added answer-quality proxy with token overlap and keyword hit
- Added tests for metrics and hybrid rank fusion

### Commit 7

- Message: `add config-driven experiment runner`
- Added YAML-driven experiment loading with explicit experiments and grid expansion
- Added per-config summary export to `results/metrics_summary.csv`
- Added per-query export to `results/per_query_results.json`
- Added runnable eval CLI with default config

### Commit 8

- Message: `add streamlit dashboard`
- Added config comparison summary table
- Added per-query inspector with cross-config comparison
- Added failure-case review panel
- Added docs screenshot placeholder

### Commit 9

- Message: `clean README and add design docs`
- Cleaned README to match the actual runnable state of the repo
- Added design decisions with current retrieval trade-offs
- Added `results/failure_cases.md` based on the 500-sample standard run

### Commit 10

- Message: `add ci tests and final cleanup`
- Added CI workflow
- Improved dense model load error message
- Added missing data-loading test coverage
- Tightened final comparison config coverage

### Post-Commit 10 Cleanup

- Ran the full HotpotQA comparison space across all 15 configs
- Updated README, design docs, and failure-case notes so they match the real benchmark output
- Started Phase B ingestion with real local sermon transcript files
