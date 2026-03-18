# Build Progress

## Current Step

- Current repo status: Commit 10 in progress
- Next step: final verification + commit
- Next phase: stop after Phase A cleanup, do not start sermons yet

## Last Step

- Commit 10 work started
- Added GitHub Actions workflow in [`.github/workflows/ci.yml`](/Users/yinshi/Documents/breadrag/.github/workflows/ci.yml)
- Cleaned README CI status and documented dense model download requirement
- Improved dense model load failure message in [`src/retriever_dense.py`](/Users/yinshi/Documents/breadrag/src/retriever_dense.py)
- Verified tests with `./.venv/bin/pytest -q`
- Verified current environment cannot run dense eval without network or cached models

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
