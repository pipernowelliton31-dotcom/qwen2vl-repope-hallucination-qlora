"""Paired RePOPE positive-only recall test at a larger visual-token budget.

This intentionally evaluates only retained RePOPE `yes` items. It compares a
new 512-visual-token inference run with the existing 256-token baseline on the
same positive questions, so the reported change is a paired recall difference.
It is not a complete RePOPE evaluation and must not be used to report F1.

Example (NOT run automatically):
    python tools/analyze_repope_positive_512.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.evaluate_pope import MODEL_PATH, normalize_answer, run_single_inference
from scripts.evaluate_repope import image_stem, read_jsonl, sha256


RESULTS_DIR = PROJECT_DIR / "results"
ANNOTATIONS_DIR = PROJECT_DIR / "data" / "raw" / "repope"
BASELINE_PREDICTIONS = RESULTS_DIR / "qwen2vl2b_baseline_predictions.jsonl"
CACHE_DIR = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
DATASET_ID = "lmms-lab/POPE"
DATASET_CONFIG = "Full"
SPLITS = ("random", "popular", "adversarial")
PIXELS_PER_VISUAL_TOKEN = 28 * 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paired 512-visual-token RePOPE positive-only recall test."
    )
    parser.add_argument(
        "--run-name",
        default="qwen2vl2b_repope_random_positive_512",
        help="Prefix for output files. Existing outputs are protected by default.",
    )
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=512,
        help="Qwen2-VL image-token ceiling. 512 corresponds to max_pixels=401408.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Positive examples per split; 0 means all retained RePOPE positives.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["random"],
        help="RePOPE split(s) to test. Defaults to one low-cost group: random.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-predictions", type=Path, default=BASELINE_PREDICTIONS)
    parser.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output with the same run name.",
    )
    return parser.parse_args()


def load_positive_annotations(annotations_dir: Path) -> dict[str, list[dict[str, Any]]]:
    positives: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows = read_jsonl(annotations_dir / f"coco_repope_{split}.json")
        positive_rows = [
            item for item in rows if str(item["label"]).strip().lower() == "yes"
        ]
        positives[split] = positive_rows
    return positives


def choose_items(
    items: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(items):
        return items
    selected = random.Random(seed).sample(items, limit)
    return sorted(selected, key=lambda item: int(item["question_id"]))


def empty_positive_stats() -> Counter[str]:
    return Counter(total=0, detected=0, missed=0, unknown=0, baseline_detected=0)


def report_positive_stats(stats: Counter[str], transitions: Counter[str]) -> dict[str, int | float]:
    total = stats["total"]
    return {
        "positive_samples": total,
        "detected_yes": stats["detected"],
        "missed_yes": stats["missed"],
        "unknown": stats["unknown"],
        "recall_512": stats["detected"] / total if total else 0.0,
        "baseline_recall_256": stats["baseline_detected"] / total if total else 0.0,
        "recall_delta": (stats["detected"] - stats["baseline_detected"]) / total if total else 0.0,
        "fn_to_tp_improved": transitions["fn_to_tp"],
        "tp_to_fn_regressed": transitions["tp_to_fn"],
        "tp_to_tp_unchanged": transitions["tp_to_tp"],
        "fn_to_fn_unchanged": transitions["fn_to_fn"],
    }


def update_transition(transitions: Counter[str], baseline: str, current: str) -> None:
    baseline_yes = baseline == "yes"
    current_yes = current == "yes"
    if not baseline_yes and current_yes:
        transitions["fn_to_tp"] += 1
    elif baseline_yes and not current_yes:
        transitions["tp_to_fn"] += 1
    elif baseline_yes:
        transitions["tp_to_tp"] += 1
    else:
        transitions["fn_to_fn"] += 1


def main() -> None:
    args = parse_args()
    if args.max_visual_tokens < 1:
        raise ValueError("max-visual-tokens must be positive")
    if args.limit_per_split < 0:
        raise ValueError("limit-per-split cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU was detected.")

    run_name = "".join(char if char.isalnum() or char in "_.-" else "_" for char in args.run_name)
    predictions_path = RESULTS_DIR / f"{run_name}_predictions.jsonl"
    metrics_path = RESULTS_DIR / f"{run_name}_metrics.json"
    if not args.overwrite and (predictions_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"Output exists. Choose another --run-name or pass --overwrite: {run_name}")

    baseline_records = read_jsonl(args.baseline_predictions)
    baseline_by_key = {
        (str(row["split"]), str(row["question_id"])): row
        for row in baseline_records
    }
    positives = load_positive_annotations(args.annotations_dir)
    selected = {
        split: choose_items(positives[split], args.limit_per_split, args.seed + offset)
        for offset, split in enumerate(args.splits)
    }

    # RePOPE is the authority for labels; POPE supplies only cached images/questions.
    pope = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        cache_dir=str(args.cache_dir),
        download_config=DownloadConfig(local_files_only=True),
    )
    qid_to_index = {
        split: {str(row["question_id"]): index for index, row in enumerate(pope[split])}
        for split in args.splits
    }

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_pixels = args.max_visual_tokens * PIXELS_PER_VISUAL_TOKEN
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Visual-token ceiling: {args.max_visual_tokens} (max_pixels={max_pixels})")
    print("Scope: retained RePOPE positive items only; this reports recall, not full F1.")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, max_pixels=max_pixels
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, dict[str, int | float]] = {}
    all_stats = empty_positive_stats()
    all_transitions: Counter[str] = Counter()
    start_time = time.perf_counter()

    with predictions_path.open("w", encoding="utf-8") as output_file:
        for split in args.splits:
            stats = empty_positive_stats()
            transitions: Counter[str] = Counter()
            for item in tqdm(selected[split], desc=f"512-token positives · {split}", unit="sample"):
                question_id = str(item["question_id"])
                key = (split, question_id)
                baseline = baseline_by_key.get(key)
                if baseline is None:
                    raise ValueError(f"Saved baseline prediction missing for {key}")
                dataset_index = qid_to_index[split].get(question_id)
                if dataset_index is None:
                    raise ValueError(f"Cached POPE example missing for {key}")
                sample = pope[split][dataset_index]
                if image_stem(str(sample["image_source"])) != image_stem(str(item["image"])):
                    raise ValueError(f"Image mismatch for {key}")

                raw_answer, prediction, elapsed = run_single_inference(
                    model=model,
                    processor=processor,
                    image=sample["image"],
                    question=str(sample["question"]),
                    input_device=input_device,
                )
                baseline_prediction = normalize_answer(str(baseline["prediction"]))
                stats["total"] += 1
                all_stats["total"] += 1
                if prediction == "yes":
                    stats["detected"] += 1
                    all_stats["detected"] += 1
                else:
                    stats["missed"] += 1
                    all_stats["missed"] += 1
                if prediction == "unknown":
                    stats["unknown"] += 1
                    all_stats["unknown"] += 1
                if baseline_prediction == "yes":
                    stats["baseline_detected"] += 1
                    all_stats["baseline_detected"] += 1
                update_transition(transitions, baseline_prediction, prediction)
                update_transition(all_transitions, baseline_prediction, prediction)

                output_file.write(json.dumps({
                    "split": split,
                    "dataset_index": dataset_index,
                    "question_id": question_id,
                    "image_source": sample["image_source"],
                    "question": sample["question"],
                    "repope_label": "yes",
                    "baseline_prediction_256": baseline_prediction,
                    "prediction_512": prediction,
                    "raw_answer_512": raw_answer,
                    "transition": (
                        "fn_to_tp" if baseline_prediction != "yes" and prediction == "yes"
                        else "tp_to_fn" if baseline_prediction == "yes" and prediction != "yes"
                        else "tp_to_tp" if prediction == "yes" else "fn_to_fn"
                    ),
                    "elapsed_seconds": elapsed,
                }, ensure_ascii=False) + "\n")
                output_file.flush()
            per_split[split] = report_positive_stats(stats, transitions)

    output = {
        "benchmark": "RePOPE positive-only paired recall test",
        "scope_warning": "Only RePOPE yes samples were evaluated. Do not treat this file as full RePOPE Accuracy, Precision, or F1.",
        "model_path": MODEL_PATH,
        "baseline_predictions": str(args.baseline_predictions),
        "baseline_predictions_sha256": sha256(args.baseline_predictions),
        "annotations_dir": str(args.annotations_dir),
        "max_visual_tokens": args.max_visual_tokens,
        "max_pixels": max_pixels,
        "baseline_max_visual_tokens": 256,
        "quantization": "4-bit NF4 double quantization",
        "compute_dtype": str(compute_dtype),
        "limit_per_split": args.limit_per_split,
        "splits": args.splits,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - start_time,
        "metrics": {**per_split, "overall": report_positive_stats(all_stats, all_transitions)},
    }
    metrics_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved paired recall report: {metrics_path}")


if __name__ == "__main__":
    main()
