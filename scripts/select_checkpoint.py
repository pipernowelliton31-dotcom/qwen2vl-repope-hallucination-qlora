"""Apply the pre-registered RePOPE-style dev checkpoint rule without test leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Frozen E0 dev metrics JSON.")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True, help="One dev metrics JSON per checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8")); metrics = source["metrics"]
    return {"path": str(path), "precision": metrics["overall"]["precision"], "recall": metrics["overall"]["recall"], "f1": metrics["overall"]["f1"], "yes_ratio": metrics["overall"]["yes_ratio"], "true_yes_ratio": (metrics["overall"]["tp"] + metrics["overall"]["fn"]) / metrics["overall"]["total"], "adversarial_fpr": metrics["adversarial"]["false_positive_rate"]}


def choose(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if row["adversarial_fpr"] <= baseline["adversarial_fpr"] + .01 and row["precision"] >= baseline["precision"] - .01]
    if not eligible: return {"status": "no_eligible_checkpoint", "baseline": baseline, "candidates": candidates, "selected": None}
    # Recall is primary. Candidates within 0.2 percentage points use fixed tie-breakers.
    best_recall = max(row["recall"] for row in eligible)
    close = [row for row in eligible if best_recall - row["recall"] < .002]
    selected = sorted(close, key=lambda row: (-row["f1"], row["adversarial_fpr"], abs(row["yes_ratio"] - row["true_yes_ratio"]), row["path"]))[0]
    return {"status": "selected", "baseline": baseline, "candidates": candidates, "eligible": eligible, "selected": selected}


def main() -> None:
    args = parse_args(); result = choose(summary(args.baseline), [summary(path) for path in args.candidates])
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
