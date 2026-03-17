"""Download and prepare a small HotpotQA subset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import (  # noqa: E402
    RAW_FILE,
    SUBSET_FILE,
    build_subset,
    load_hotpot_raw,
    save_json,
)


DEFAULT_URL = (
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/"
    "hotpot_dev_distractor_v1.json?download=true"
)


def download_file(url: str, path: Path, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        print(f"use cached raw file: {path}", flush=True)
        return path

    print(f"download raw file from: {url}", flush=True)
    with urlopen(url, timeout=60) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)

    print(f"saved raw file: {path}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HotpotQA subset")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--raw-path", default=str(RAW_FILE))
    parser.add_argument("--output-path", default=str(SUBSET_FILE))
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_path)
    output_path = Path(args.output_path)

    download_file(args.url, raw_path, force=args.force_download)
    rows = load_hotpot_raw(raw_path)
    subset = build_subset(rows, limit=args.limit, seed=args.seed)
    save_json(subset, output_path)

    print(f"loaded rows: {len(rows)}", flush=True)
    print(f"saved subset rows: {len(subset)}", flush=True)
    print(f"subset file: {output_path}", flush=True)


if __name__ == "__main__":
    main()
