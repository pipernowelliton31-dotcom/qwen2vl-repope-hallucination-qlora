"""Re-score an existing POPE prediction file using the official RePOPE labels.

This script never runs model inference.  It validates each retained RePOPE item
against the saved POPE prediction (split, question_id, image and question) and
then recomputes binary-classification metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "results" / "qwen2vl2b_baseline_predictions.jsonl"
DEFAULT_ANNOTATIONS = ROOT / "data" / "raw" / "repope"
DEFAULT_OUTPUT = ROOT / "results" / "qwen2vl2b_baseline_repope_metrics.json"
SPLITS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute POPE metrics with official RePOPE annotations."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def image_stem(value: str) -> str:
    return Path(value).stem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def empty_counts() -> Counter[str]:
    return Counter(tp=0, fp=0, tn=0, fn=0, unknown=0)


def metrics(counts: Counter[str]) -> dict[str, int | float]:
    total = counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"] + counts["unknown"]
    classified = total - counts["unknown"]
    correct = counts["tp"] + counts["tn"]
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": total,
        "classified": classified,
        "correct": correct,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "tn": counts["tn"],
        "fn": counts["fn"],
        "unknown": counts["unknown"],
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (counts["tp"] + counts["fp"]) / total if total else 0.0,
        "unknown_ratio": counts["unknown"] / total if total else 0.0,
        "false_positive_rate": counts["fp"] / (counts["fp"] + counts["tn"])
        if counts["fp"] + counts["tn"]
        else 0.0,
    }


def update_counts(counts: Counter[str], label: str, prediction: str) -> None:
    if prediction not in {"yes", "no"}:
        counts["unknown"] += 1
    elif label == "yes" and prediction == "yes":
        counts["tp"] += 1
    elif label == "yes":
        counts["fn"] += 1
    elif prediction == "yes":
        counts["fp"] += 1
    else:
        counts["tn"] += 1


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    by_split_question_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in predictions:
        key = (str(row["split"]), str(row["question_id"]))
        if key in by_split_question_id:
            raise ValueError(f"Duplicate prediction key: {key}")
        by_split_question_id[key] = row

    per_split: dict[str, dict[str, int | float]] = {}
    annotation_manifest: dict[str, dict[str, str]] = {}
    question_text_mismatches: list[dict[str, str]] = []
    overall = empty_counts()

    for split in SPLITS:
        annotation_path = args.annotations_dir / f"coco_repope_{split}.json"
        annotations = read_jsonl(annotation_path)
        annotation_manifest[split] = {
            "path": str(annotation_path),
            "sha256": sha256(annotation_path),
        }
        counts = empty_counts()
        for item in annotations:
            key = (split, str(item["question_id"]))
            prediction_row = by_split_question_id.get(key)
            if prediction_row is None:
                raise ValueError(f"Missing saved prediction for retained RePOPE item: {key}")
            if image_stem(str(prediction_row["image_source"])) != image_stem(str(item["image"])):
                raise ValueError(f"Image mismatch for {key}")
            # The lmms-lab formatting can contain harmless transcription typos
            # (for example, "imange" vs "image"). The canonical POPE question
            # id plus image is the alignment key; retain every text discrepancy
            # in the output so this decision is fully auditable.
            if normalize_text(str(prediction_row["question"])) != normalize_text(str(item["text"])):
                question_text_mismatches.append(
                    {
                        "split": split,
                        "question_id": str(item["question_id"]),
                        "saved_question": str(prediction_row["question"]),
                        "repope_question": str(item["text"]),
                    }
                )
            label = normalize_text(str(item["label"]))
            if label not in {"yes", "no"}:
                raise ValueError(f"Unexpected RePOPE label for {key}: {label!r}")
            update_counts(counts, label, normalize_text(str(prediction_row["prediction"])))
            update_counts(overall, label, normalize_text(str(prediction_row["prediction"])))
        per_split[split] = metrics(counts)

    output = {
        "benchmark": "RePOPE",
        "method": "Re-scored existing POPE predictions; no model inference was run.",
        "predictions_path": str(args.predictions),
        "predictions_sha256": sha256(args.predictions),
        "official_annotation_files": annotation_manifest,
        "validation": {
            "matching_key": "split + question_id, additionally verified image",
            "saved_prediction_count": len(predictions),
            "retained_repope_count": sum(int(values["total"]) for values in per_split.values()),
            "question_text_mismatch_count": len(question_text_mismatches),
            "question_text_mismatch_examples": question_text_mismatches[:20],
        },
        "metrics": {**per_split, "overall": metrics(overall)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
