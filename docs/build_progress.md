# Build Progress

## Current Step

- Current repo status: HotpotQA Phase A is documented through the full 15-config run
- Next step: fill `data/eval/sermon_questions.csv` with real labeled questions and run `configs/sermon.yaml`
- Next phase: Phase B eval on the staged sermon transcripts

## Last Step

- Ran the full 15-config HotpotQA grid on 500 samples and folded the results back into the docs
- Added [`src/data_loader_sermon.py`](/Users/yinshi/Documents/breadrag/src/data_loader_sermon.py) for local sermon transcript loading
- Added [`scripts/prepare_sermon_data.py`](/Users/yinshi/Documents/breadrag/scripts/prepare_sermon_data.py) to stage transcript `.docx` files into `data/raw/sermons/`
- Added [`configs/sermon.yaml`](/Users/yinshi/Documents/breadrag/configs/sermon.yaml) and a starter [`data/eval/sermon_questions.csv`](/Users/yinshi/Documents/breadrag/data/eval/sermon_questions.csv)
- Verified sermon staging against the real local transcript files

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
