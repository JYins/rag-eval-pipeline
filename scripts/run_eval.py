# ruff: noqa: E402

"""CLI entry for running retrieval experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_runner import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval eval from YAML config")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_eval(args.config, limit_override=args.limit)

    print(f"saved metrics summary: {result['metrics_path']}")
    print(f"saved per-query results: {result['per_query_path']}")
    print("configs run:")
    for row in result["summary_rows"]:
        print(
            f"- {row['config_name']}: "
            f"Recall@3={row['recall@3']:.4f} "
            f"MRR={row['mrr']:.4f} "
            f"HitRate={row['hit_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
