"""复用已保存的 E0 3k 预测，对固定 2k dev 子集重算指标，不重复模型推理。"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.qlora_common import grouped_metrics, read_jsonl

DATASET = PROJECT_DIR / "data" / "processed" / "coco_dev_repope_style_2k.jsonl"
RESULTS = PROJECT_DIR / "results" / "qlora_evaluations"
SOURCE_PREDICTIONS = RESULTS / "e0_base_dev_256vt_predictions.jsonl"
OUTPUT_PREDICTIONS = RESULTS / "e0_base_2k_dev_256vt_predictions.jsonl"
OUTPUT_METRICS = RESULTS / "e0_base_2k_dev_256vt_metrics.json"


def key(row: dict) -> tuple[str, str, int, str]:
    return str(row["split"]), str(row["label"]), int(row["image_id"]), str(row["question"])


def main() -> None:
    subset = read_jsonl(DATASET)
    existing = read_jsonl(SOURCE_PREDICTIONS)
    buckets: dict[tuple[str, str, int, str], deque[dict]] = defaultdict(deque)
    for row in existing:
        buckets[key(row)].append(row)

    aligned: list[dict] = []
    for row in subset:
        matches = buckets[key(row)]
        if not matches:
            raise RuntimeError(f"E0 已保存预测无法与 2k dev 对齐：{key(row)}")
        aligned.append(matches.popleft())
    if len(aligned) != 2000:
        raise RuntimeError(f"对齐后应为 2000 条，实际为 {len(aligned)}")

    OUTPUT_PREDICTIONS.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned), encoding="utf-8"
    )
    payload = {
        "dataset": "dev",
        "dataset_sample_count": len(aligned),
        "dataset_path": str(DATASET),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "adapter": None,
        "max_visual_tokens": 256,
        "derived_from_predictions": str(SOURCE_PREDICTIONS),
        "metrics": grouped_metrics(aligned),
    }
    OUTPUT_METRICS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
