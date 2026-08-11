from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("QWEN2VL_MODEL_PATH", "Qwen/Qwen2-VL-2B-Instruct")
DATASET_ID = "lmms-lab/POPE"
DATASET_CONFIG = "Full"

DEFAULT_SPLITS = ["random", "popular", "adversarial"]
RESULTS_DIR = PROJECT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2-VL-2B-Instruct on POPE."
    )

    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help=(
            "每个 split 最多评测多少条。"
            "0 表示评测该 split 的全部 3000 条。"
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="pope_baseline",
        help="结果文件名称，例如 smoke 或 qwen2vl2b_baseline。",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=DEFAULT_SPLITS,
        help="要评测的 POPE split。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Smoke test 抽样随机种子。",
    )

    return parser.parse_args()


def normalize_answer(text: str) -> str:
    """
    将模型输出解析为 yes、no 或 unknown。

    优先检查开头，避免句子里同时出现 yes 和 no 时误判。
    """
    normalized = text.strip().lower()

    if normalized.startswith("yes"):
        return "yes"

    if normalized.startswith("no"):
        return "no"

    match = re.search(r"\b(yes|no)\b", normalized)

    if match:
        return match.group(1)

    return "unknown"


def empty_stats() -> dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "predicted_yes": 0,
        "predicted_no": 0,
        "unknown": 0,
    }


def update_stats(
    stats: dict[str, int],
    ground_truth: str,
    prediction: str,
) -> None:
    stats["total"] += 1

    if prediction == ground_truth:
        stats["correct"] += 1

    if prediction == "yes":
        stats["predicted_yes"] += 1
    elif prediction == "no":
        stats["predicted_no"] += 1
    else:
        stats["unknown"] += 1

    # POPE 中将 yes 视为正类。
    if ground_truth == "yes":
        if prediction == "yes":
            stats["tp"] += 1
        else:
            # 回答 no 或无法解析，都视为漏检。
            stats["fn"] += 1
    else:
        if prediction == "yes":
            stats["fp"] += 1
        elif prediction == "no":
            stats["tn"] += 1


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(stats: dict[str, int]) -> dict[str, Any]:
    tp = stats["tp"]
    fp = stats["fp"]
    tn = stats["tn"]
    fn = stats["fn"]
    total = stats["total"]

    accuracy = safe_divide(stats["correct"], total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    yes_ratio = safe_divide(stats["predicted_yes"], total)
    unknown_ratio = safe_divide(stats["unknown"], total)

    return {
        **stats,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "unknown_ratio": unknown_ratio,
        "false_positive_rate": safe_divide(fp, fp + tn),
    }


def select_balanced_indices(
    dataset: Dataset,
    limit: int,
    seed: int,
) -> list[int]:
    """
    Smoke test 时尽量抽取数量相近的 yes/no 样本。

    正式评测 limit=0，直接使用全部数据。
    """
    if limit <= 0 or limit >= len(dataset):
        return list(range(len(dataset)))

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    yes_target = limit // 2
    no_target = limit - yes_target

    selected: list[int] = []
    yes_count = 0
    no_count = 0

    for index in indices:
        answer = str(dataset[index]["answer"]).strip().lower()

        if answer == "yes" and yes_count < yes_target:
            selected.append(index)
            yes_count += 1

        elif answer == "no" and no_count < no_target:
            selected.append(index)
            no_count += 1

        if len(selected) == limit:
            break

    return selected


def run_single_inference(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Any,
    question: str,
    input_device: torch.device,
) -> tuple[str, str, float]:
    """
    对一张图片和一个 POPE 问题执行推理。

    返回：
    1. 原始文本
    2. 标准化后的 yes/no/unknown
    3. 单条端到端推理耗时
    """
    image = image.convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": (
                        f"{question}\n"
                        'Answer using only "yes" or "no".'
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    start_time = time.perf_counter()

    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(input_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # 只保留新生成的 token，不保留原始输入。
    generated_only = generated_ids[
        :,
        inputs["input_ids"].shape[1] :
    ]

    raw_answer = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    elapsed = time.perf_counter() - start_time
    parsed_answer = normalize_answer(raw_answer)

    return raw_answer, parsed_answer, elapsed


def print_metrics(
    split_name: str,
    metrics: dict[str, Any],
) -> None:
    print(f"\n{split_name} 评测结果")
    print("-" * 55)
    print(f"样本数：       {metrics['total']}")
    print(f"Accuracy：     {metrics['accuracy']:.4f}")
    print(f"Precision：    {metrics['precision']:.4f}")
    print(f"Recall：       {metrics['recall']:.4f}")
    print(f"F1：           {metrics['f1']:.4f}")
    print(f"Yes Ratio：    {metrics['yes_ratio']:.4f}")
    print(f"Unknown Ratio：{metrics['unknown_ratio']:.4f}")
    print(f"TP / FP：      {metrics['tp']} / {metrics['fp']}")
    print(f"TN / FN：      {metrics['tn']} / {metrics['fn']}")


def main() -> None:
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = RESULTS_DIR / f"{args.run_name}_predictions.jsonl"
    metrics_path = RESULTS_DIR / f"{args.run_name}_metrics.json"

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 CUDA GPU。")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("GPU：", torch.cuda.get_device_name(0))
    print("计算精度：", compute_dtype)
    print("运行名称：", args.run_name)
    print("每组样本限制：", args.limit_per_split or "全部")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("\n[1/3] 加载 Processor")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        max_pixels=256 * 28 * 28,
    )

    print("[2/3] 以 4-bit 加载原始模型")

    model_load_start = time.perf_counter()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    model.eval()

    input_device = model.get_input_embeddings().weight.device
    model_load_time = time.perf_counter() - model_load_start

    print(f"模型加载完成：{model_load_time:.2f} 秒")
    print("模型输入设备：", input_device)

    print("[3/3] 加载 POPE 数据集")

    pope = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
    )

    all_results: dict[str, dict[str, Any]] = {}
    overall_stats = empty_stats()

    # 使用写入模式，避免误把不同配置结果混在一起。
    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as prediction_file:

        for split_offset, split_name in enumerate(args.splits):
            split_dataset = pope[split_name]

            indices = select_balanced_indices(
                split_dataset,
                args.limit_per_split,
                args.seed + split_offset,
            )

            split_stats = empty_stats()
            split_elapsed_total = 0.0

            progress = tqdm(
                indices,
                desc=f"Evaluating {split_name}",
                unit="sample",
            )

            for dataset_index in progress:
                sample = split_dataset[dataset_index]

                question = str(sample["question"])
                ground_truth = str(sample["answer"]).strip().lower()

                raw_answer, prediction, elapsed = run_single_inference(
                    model=model,
                    processor=processor,
                    image=sample["image"],
                    question=question,
                    input_device=input_device,
                )

                split_elapsed_total += elapsed

                update_stats(
                    split_stats,
                    ground_truth,
                    prediction,
                )

                update_stats(
                    overall_stats,
                    ground_truth,
                    prediction,
                )

                record = {
                    "split": split_name,
                    "dataset_index": dataset_index,
                    "id": sample["id"],
                    "question_id": sample["question_id"],
                    "image_source": sample["image_source"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "raw_answer": raw_answer,
                    "prediction": prediction,
                    "correct": prediction == ground_truth,
                    "elapsed_seconds": elapsed,
                }

                prediction_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # 每条立即写盘，意外中断时也能保留结果。
                prediction_file.flush()

                progress.set_postfix(
                    accuracy=(
                        split_stats["correct"]
                        / split_stats["total"]
                    ),
                    unknown=split_stats["unknown"],
                )

            split_metrics = calculate_metrics(split_stats)
            split_metrics["elapsed_seconds"] = split_elapsed_total
            split_metrics["average_seconds_per_sample"] = safe_divide(
                split_elapsed_total,
                split_stats["total"],
            )

            all_results[split_name] = split_metrics
            print_metrics(split_name, split_metrics)

    overall_metrics = calculate_metrics(overall_stats)
    all_results["overall"] = overall_metrics

    metadata = {
        "model_path": MODEL_PATH,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "splits": args.splits,
        "limit_per_split": args.limit_per_split,
        "seed": args.seed,
        "max_pixels": 256 * 28 * 28,
        "quantization": "4-bit NF4 double quantization",
        "compute_dtype": str(compute_dtype),
        "metrics": all_results,
    }

    metrics_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print_metrics("overall", overall_metrics)

    print("\n结果文件：")
    print("预测明细：", predictions_path)
    print("指标汇总：", metrics_path)

    print("\n显存统计：")
    print(
        "当前已分配："
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    print(
        "峰值已分配："
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )


if __name__ == "__main__":
    main()
