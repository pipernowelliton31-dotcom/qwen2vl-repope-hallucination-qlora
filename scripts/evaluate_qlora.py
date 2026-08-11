"""Evaluate a base model or adapter on leakage-free dev or official RePOPE labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.qlora_common import DATA_DIR, RESULTS_DIR, grouped_metrics, normalize_question, read_jsonl
from evaluate_repope import image_stem

REPOPE_DIR = PROJECT_DIR / "data" / "raw" / "repope"
CACHE_DIR = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline prompt/parser on local dev or official RePOPE.")
    parser.add_argument("--dataset", choices=("dev", "repope"), required=True)
    parser.add_argument("--adapter", type=Path, help="Saved PEFT adapter/checkpoint; omit for the frozen E0 base model.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--limit-per-split", type=int, default=0)
    return parser.parse_args()


def load_rows(dataset: str, cache_dir: Path) -> list[dict[str, Any]]:
    # checkpoint 选择统一使用预先固化的 2k 分层子集，确保所有实验完全可比。
    if dataset == "dev": return read_jsonl(DATA_DIR / "coco_dev_repope_style_2k.jsonl")
    from datasets import DownloadConfig, load_dataset
    pope = load_dataset("lmms-lab/POPE", "Full", cache_dir=str(cache_dir), download_config=DownloadConfig(local_files_only=True))
    annotations = {split: read_jsonl(REPOPE_DIR / f"coco_repope_{split}.json") for split in ("random", "popular", "adversarial")}
    rows: list[dict[str, Any]] = []
    for split, items in annotations.items():
        index = {str(sample["question_id"]): i for i, sample in enumerate(pope[split])}
        for item in items:
            dataset_index = index[str(item["question_id"])]
            sample = pope[split][dataset_index]
            if image_stem(str(sample["image_source"])) != image_stem(str(item["image"])):
                raise RuntimeError(f"RePOPE image alignment failed: {split}/{item['question_id']}")
            rows.append({"split": split, "label": str(item["label"]).lower(), "question": sample["question"], "image": sample["image"], "image_source": sample["image_source"], "question_id": item["question_id"]})
    return rows


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image
    from peft import PeftModel
    from tqdm.auto import tqdm
    from scripts.qlora_common import load_quantized_model

    # Load no fresh adapter: PeftModel reads target-module details from the saved adapter config.
    model, processor, _ = load_quantized_model("e1", False, args.max_visual_tokens, attach_lora=False)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval(); device = model.get_input_embeddings().weight.device
    rows = load_rows(args.dataset, args.cache_dir)
    output_rows: list[dict[str, Any]] = []
    limits: dict[str, int] = {split: 0 for split in ("random", "popular", "adversarial")}
    correct = 0
    progress = tqdm(rows, desc=f"评测 {args.dataset} · {args.max_visual_tokens}vt", unit="sample", dynamic_ncols=True)
    for row in progress:
        split = str(row["split"])
        if args.limit_per_split and limits[split] >= args.limit_per_split: continue
        image = row["image"] if "image" in row else Image.open(row["image_path"]).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(row["question"]) + '\nAnswer using only "yes" or "no".'}]}]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id)
        raw = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        prediction = __import__("evaluate_pope").normalize_answer(raw)
        correct += prediction == str(row["label"]).lower()
        output_rows.append({**{key: value for key, value in row.items() if key not in {"image"}}, "prediction": prediction, "raw_answer": raw, "elapsed_seconds": time.perf_counter() - started})
        limits[split] += 1
        # Dev/RePOPE is generation-based: Accuracy is the live meaningful signal;
        # final F1/Recall/FPR are printed after all categories are complete.
        progress.set_postfix(accuracy=f"{correct / len(output_rows):.3f}", done=len(output_rows))
    metrics = grouped_metrics(output_rows)
    out_dir = RESULTS_DIR / "qlora_evaluations"; out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.run_name}_{args.dataset}_{args.max_visual_tokens}vt"
    (out_dir / f"{prefix}_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in output_rows), encoding="utf-8")
    dataset_path = DATA_DIR / "coco_dev_repope_style_2k.jsonl" if args.dataset == "dev" else None
    payload = {
        "dataset": args.dataset,
        "dataset_sample_count": len(rows),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest() if dataset_path else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "max_visual_tokens": args.max_visual_tokens,
        "metrics": metrics,
    }
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
