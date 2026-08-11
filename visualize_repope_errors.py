"""Build an offline RePOPE error-review page from saved POPE predictions.

No model inference is run.  Existing predictions are relabeled with the
official RePOPE annotations, then representative wrong answers are exported
with their cached COCO images.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import DownloadConfig, load_dataset

from visualize_pope_errors import (
    DATASET_CONFIG,
    DATASET_ID,
    DEFAULT_CACHE_DIR,
    RESULTS_DIR,
    load_jsonl,
    page_html,
    safe_name,
    select_representative_errors,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = RESULTS_DIR / "qwen2vl2b_baseline_predictions.jsonl"
DEFAULT_METRICS = RESULTS_DIR / "qwen2vl2b_baseline_repope_metrics.json"
DEFAULT_ANNOTATIONS = PROJECT_DIR / "data" / "raw" / "repope"
SPLITS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline image review page for RePOPE errors."
    )
    parser.add_argument("--run-name", default="qwen2vl2b_baseline_repope")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--examples-per-split", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized(value: str) -> str:
    return " ".join(value.strip().lower().split())


def relabel_records(
    predictions: list[dict[str, Any]], annotations_dir: Path
) -> list[dict[str, Any]]:
    prediction_index = {
        (str(record["split"]), str(record["question_id"])): record
        for record in predictions
    }
    relabeled: list[dict[str, Any]] = []
    for split in SPLITS:
        for item in load_jsonl(annotations_dir / f"coco_repope_{split}.json"):
            key = (split, str(item["question_id"]))
            original = prediction_index.get(key)
            if original is None:
                raise ValueError(f"No saved prediction for RePOPE item: {key}")
            if Path(str(original["image_source"])).stem != Path(str(item["image"])).stem:
                raise ValueError(f"Image mismatch for RePOPE item: {key}")
            record = dict(original)
            record["original_pope_label"] = normalized(str(original["ground_truth"]))
            record["repope_label"] = normalized(str(item["label"]))
            record["ground_truth"] = record["repope_label"]
            record["label_changed"] = record["original_pope_label"] != record["repope_label"]
            record["correct"] = record["ground_truth"] == normalized(str(record["prediction"]))
            relabeled.append(record)
    return relabeled


def main() -> None:
    args = parse_args()
    run_name = safe_name(args.run_name)
    if args.examples_per_split < 1:
        raise ValueError("examples-per-split must be at least 1")

    original_metrics_path = RESULTS_DIR / "qwen2vl2b_baseline_metrics.json"
    original_metadata: dict[str, Any] = json.loads(original_metrics_path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = json.loads(args.metrics.read_text(encoding="utf-8"))
    metadata.update(
        {
            "model_path": original_metadata["model_path"],
            "quantization": original_metadata.get("quantization", "unknown quantization"),
            "compute_dtype": original_metadata.get("compute_dtype", "unknown dtype"),
        }
    )
    records = relabel_records(load_jsonl(args.predictions), args.annotations_dir)
    errors = [record for record in records if not record["correct"]]
    display_records = select_representative_errors(
        errors, args.examples_per_split, args.seed
    )

    assets_dir = RESULTS_DIR / f"{run_name}_review_assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)

    image_paths: list[str] = []
    pope = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        cache_dir=str(args.cache_dir),
        download_config=DownloadConfig(local_files_only=True),
    )
    for index, record in enumerate(display_records, 1):
        split = str(record["split"])
        dataset_index = int(record["dataset_index"])
        image = pope[split][dataset_index]["image"].convert("RGB")
        image_name = f"{index:02d}_{split}_{dataset_index}.jpg"
        image.save(assets_dir / image_name, quality=92)
        image_paths.append(f"{assets_dir.name}/{image_name}")

    output_path = RESULTS_DIR / f"{run_name}_review.html"
    output_path.write_text(
        page_html(
            run_name,
            metadata,
            records,
            display_records,
            image_paths,
            benchmark_label="RePOPE",
        ),
        encoding="utf-8",
    )
    print(f"Retained RePOPE records: {len(records)}; wrong answers: {len(errors)}")
    print(f"Exported {len(display_records)} examples ({args.examples_per_split} per split): {assets_dir}")
    print(f"Offline review page: {output_path}")


if __name__ == "__main__":
    main()
