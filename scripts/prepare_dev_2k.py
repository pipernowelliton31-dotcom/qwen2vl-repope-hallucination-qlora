"""从完整 3k COCO dev 中构造固定、分层且可复现的 2k 评测子集。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
SOURCE = DATA_DIR / "coco_dev_repope_style.jsonl"
OUTPUT = DATA_DIR / "coco_dev_repope_style_2k.jsonl"
MANIFEST = DATA_DIR / "coco_dev_repope_style_2k_manifest.json"
SEED = 42

# 每个 split 内保持 yes/no 完全平衡；总计 2,000 条和 1,000 yes / 1,000 no。
# 2,000 不能被 3 个 split 整除，因此固定给 random 多 2 条。
QUOTAS = {
    ("random", "yes"): 334,
    ("random", "no"): 334,
    ("popular", "yes"): 333,
    ("popular", "no"): 333,
    ("adversarial", "yes"): 333,
    ("adversarial", "no"): 333,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = read_jsonl(SOURCE)
    if len(rows) != 3000:
        raise RuntimeError(f"完整 dev 应为 3000 条，实际为 {len(rows)}")

    selected: list[dict] = []
    for (split, label), quota in QUOTAS.items():
        group = [row for row in rows if row["split"] == split and row["label"] == label]
        # 每个分层使用独立固定随机流，避免源文件行序变化导致抽样变化。
        random.Random(f"{SEED}:{split}:{label}").shuffle(group)
        if len(group) < quota:
            raise RuntimeError(f"{split}/{label} 仅有 {len(group)} 条，无法抽取 {quota} 条")
        selected.extend(group[:quota])

    # 最后固定打散，所有 checkpoint 都读取完全相同的 2k 顺序。
    random.Random(SEED).shuffle(selected)
    counts = Counter(f"{row['split']}:{row['label']}" for row in selected)
    if len(selected) != 2000 or sum(row["label"] == "yes" for row in selected) != 1000:
        raise RuntimeError("2k dev 的总数或 yes/no 平衡检查失败")

    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    payload = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "seed": SEED,
        "sample_count": len(selected),
        "unique_image_count": len({row["image_id"] for row in selected}),
        "counts": dict(sorted(counts.items())),
        "source_sha256": sha256(SOURCE),
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
