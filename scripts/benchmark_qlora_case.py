"""Worker process for exactly one isolated QLoRA speed configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.benchmark_qlora_speed import run_case
from scripts.qlora_common import DATA_DIR, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--checkpointing", choices=("0", "1"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(DATA_DIR / "speed128.jsonl")
    result = run_case(
        args.name,
        args.micro_batch,
        args.accumulation,
        args.checkpointing == "1",
        rows,
        args.steps,
        args.warmup_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
