"""Evaluate every saved adapter checkpoint on dev, then emit paths for selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", choices=("dev", "repope"), default="dev")
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    # 防止与训练写在同一 Notebook 单元时，训练一结束便自动进入耗时评测。
    # 只有用户在独立单元显式传入 --start，才真正加载模型并评测 checkpoints。
    parser.add_argument("--start", action="store_true", help="Explicitly start checkpoint evaluation.")
    args = parser.parse_args()
    if not args.start:
        print("Checkpoint 评测未启动：请在独立单元确认后添加 --start。")
        return
    checkpoints = sorted(
        (path for path in args.run_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints: raise FileNotFoundError(f"No checkpoint-* directories under {args.run_dir}")
    script = PROJECT_DIR / "scripts" / "evaluate_qlora.py"; metric_paths = []
    # Outer bar answers "which checkpoint is being evaluated"; evaluate_qlora.py
    # supplies the nested per-sample bar and live dev Accuracy.
    for checkpoint in tqdm(checkpoints, desc=f"评估 {args.dataset} checkpoints", unit="checkpoint", dynamic_ncols=True):
        name = f"{args.run_name}_{checkpoint.name}"
        subprocess.run([sys.executable, str(script), "--dataset", args.dataset, "--adapter", str(checkpoint), "--run-name", name, "--max-visual-tokens", str(args.max_visual_tokens)], cwd=PROJECT_DIR, check=True)
        metric_paths.append(str(PROJECT_DIR / "results" / "qlora_evaluations" / f"{name}_{args.dataset}_{args.max_visual_tokens}vt_metrics.json"))
    output = args.run_dir / f"{args.dataset}_checkpoint_metrics.json"
    output.write_text(json.dumps({"run_dir": str(args.run_dir), "dataset": args.dataset, "metrics": metric_paths}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__": main()
