import os
from pathlib import Path

from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets"))
SAMPLE_DIR = PROJECT_DIR / "data" / "samples"


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "lmms-lab/POPE",
        "Full",
        cache_dir=CACHE_DIR,
    )

    print(dataset)

    for split_name, split in dataset.items():
        print(f"\n{split_name}: {len(split)} 条")
        print(split.features)

        example = split[0]

        print("问题：", example["question"])
        print("答案：", example["answer"])
        print("类别：", example["category"])
        print("图像来源：", example["image_source"])

        output_path = SAMPLE_DIR / f"{split_name}_sample.jpg"
        example["image"].save(output_path)

        print("示例图片保存到：", output_path)


if __name__ == "__main__":
    main()
