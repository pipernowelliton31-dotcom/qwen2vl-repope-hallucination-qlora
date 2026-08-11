"""Apply the pre-registered RePOPE-style dev checkpoint rule without test leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FPR_TOLERANCE = 0.01
PRECISION_TOLERANCE = 0.01
RECALL_TIE_WINDOW = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Frozen E0 dev metrics JSON.")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True, help="One dev metrics JSON per checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    metrics = source["metrics"]
    overall = metrics["overall"]
    return {
        "path": str(path),
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "yes_ratio": overall["yes_ratio"],
        "true_yes_ratio": (overall["tp"] + overall["fn"]) / overall["total"],
        "overall_fpr": overall["false_positive_rate"],
        "adversarial_fpr": metrics["adversarial"]["false_positive_rate"],
    }


def choose(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    constraints = {
        "overall_fpr_max": baseline["overall_fpr"] + FPR_TOLERANCE,
        "adversarial_fpr_max": baseline["adversarial_fpr"] + FPR_TOLERANCE,
        "precision_min": baseline["precision"] - PRECISION_TOLERANCE,
        "recall_tie_window": RECALL_TIE_WINDOW,
    }
    eligible = [
        row
        for row in candidates
        if row["overall_fpr"] <= constraints["overall_fpr_max"]
        and row["adversarial_fpr"] <= constraints["adversarial_fpr_max"]
        and row["precision"] >= constraints["precision_min"]
    ]
    result = {
        "protocol": "dev2k_fpr_constrained_v1",
        "constraints": constraints,
        "baseline": baseline,
        "candidates": candidates,
        "eligible": eligible,
    }
    if not eligible:
        return {**result, "status": "no_eligible_checkpoint", "selected": None}
    # Recall is primary. Candidates within 0.2 percentage points use fixed tie-breakers.
    best_recall = max(row["recall"] for row in eligible)
    close = [row for row in eligible if best_recall - row["recall"] < RECALL_TIE_WINDOW]
    selected = sorted(
        close,
        key=lambda row: (
            -row["f1"],
            row["overall_fpr"],
            row["adversarial_fpr"],
            abs(row["yes_ratio"] - row["true_yes_ratio"]),
            row["path"],
        ),
    )[0]
    return {**result, "status": "selected", "selected": selected}


def main() -> None:
    args = parse_args(); result = choose(summary(args.baseline), [summary(path) for path in args.candidates])
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
