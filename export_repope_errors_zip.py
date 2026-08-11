"""Export every RePOPE error with its image and review metadata to a ZIP file."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from datasets import DownloadConfig, load_dataset

from visualize_pope_errors import DATASET_CONFIG, DATASET_ID, DEFAULT_CACHE_DIR, RESULTS_DIR, load_jsonl
from visualize_repope_errors import DEFAULT_ANNOTATIONS, DEFAULT_PREDICTIONS, relabel_records


DEFAULT_OUTPUT = RESULTS_DIR / "qwen2vl2b_baseline_repope_all_errors.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package all wrong Qwen2-VL answers under official RePOPE labels."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def error_type(record: dict[str, object]) -> str:
    if record["ground_truth"] == "yes":
        return "FN_false_negative"
    return "FP_false_positive"


def main() -> None:
    args = parse_args()
    records = relabel_records(load_jsonl(args.predictions), args.annotations_dir)
    errors = [record for record in records if not record["correct"]]
    errors.sort(key=lambda record: (str(record["split"]), int(record["question_id"])))
    counts = Counter(str(record["split"]) for record in errors)
    type_counts = Counter(error_type(record) for record in errors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pope = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        cache_dir=str(args.cache_dir),
        download_config=DownloadConfig(local_files_only=True),
    )

    manifest = []
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "archive_image_path", "split", "question_id", "image_source", "question",
            "repope_label", "original_pope_label", "label_changed", "prediction",
            "raw_answer", "error_type", "dataset_index", "elapsed_seconds",
        ],
    )
    writer.writeheader()

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for number, record in enumerate(errors, start=1):
            split = str(record["split"])
            question_id = int(record["question_id"])
            dataset_index = int(record["dataset_index"])
            image_path = f"images/{split}/{number:03d}_qid_{question_id}.jpg"
            image = pope[split][dataset_index]["image"].convert("RGB")
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG", quality=92)
            archive.writestr(image_path, image_buffer.getvalue())

            export_record = {
                "archive_image_path": image_path,
                "split": split,
                "question_id": question_id,
                "image_source": record["image_source"],
                "question": record["question"],
                "repope_label": record["repope_label"],
                "original_pope_label": record["original_pope_label"],
                "label_changed": record["label_changed"],
                "prediction": record["prediction"],
                "raw_answer": record["raw_answer"],
                "error_type": error_type(record),
                "dataset_index": dataset_index,
                "elapsed_seconds": record["elapsed_seconds"],
            }
            manifest.append(export_record)
            writer.writerow(export_record)

        readme = f"""# Qwen2-VL-2B-Instruct · RePOPE 错题包

本压缩包包含使用 RePOPE 官方修正标签重新计分后，模型答错的全部题目。

- 错题总数：{len(errors)}
- random：{counts['random']}
- popular：{counts['popular']}
- adversarial：{counts['adversarial']}
- FN（图中有该物，模型答 no）：{type_counts['FN_false_negative']}
- FP（图中无该物，模型答 yes）：{type_counts['FP_false_positive']}

## 内容

- `errors.csv`：适合用 Excel 打开筛选。
- `errors.jsonl`：同一元数据的机器可读版本，每行对应一道题。
- `images/<split>/`：每道错题对应的 COCO 图片。文件名中的 `qid` 对应 `question_id`。

字段说明：`repope_label` 是用于本次判断的修正真值；`original_pope_label` 保留原标签；`label_changed` 表示两者是否不同；`error_type` 为 FN 或 FP。
"""
        archive.writestr("README.md", readme.encode("utf-8"))
        archive.writestr("errors.csv", csv_buffer.getvalue().encode("utf-8-sig"))
        archive.writestr(
            "errors.jsonl",
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest).encode("utf-8"),
        )

    print(f"Packaged {len(errors)} RePOPE errors: {args.output}")
    print(f"ZIP size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
