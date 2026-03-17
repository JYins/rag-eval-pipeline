# Build Progress

## Current Step

- Current repo status: Commit 5 finished
- Next step: Commit 6
- Next phase: add hybrid retrieval and eval metrics

## Last Step

- Commit 5 finished
- Added FAISS helpers in [`src/indexing.py`](/Users/yinshi/Documents/breadrag/src/indexing.py)
- Added dense retrieval in [`src/retriever_dense.py`](/Users/yinshi/Documents/breadrag/src/retriever_dense.py)
- Extended retrieval coverage in [`tests/test_retrieval.py`](/Users/yinshi/Documents/breadrag/tests/test_retrieval.py)
- Verified with `pytest tests/test_retrieval.py`

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
