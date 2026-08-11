"""Shared, commented building blocks for all QLoRA experiments.

The module deliberately keeps the training prompt, quantization, answer parser,
and metric implementation in one place.  This prevents an E1/E2/E3 comparison
from silently changing evaluation behavior.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluate_pope import normalize_answer  # Same parser as the established baseline.

CONFIG_PATH = PROJECT_DIR / "configs" / "qlora_experiments.json"
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "data" / "processed"
PIXELS_PER_VISUAL_TOKEN = 28 * 28


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def config() -> dict[str, Any]:
    cfg = read_json(CONFIG_PATH)
    model_override = os.environ.get("QWEN2VL_MODEL_PATH")
    if model_override:
        cfg["model_path"] = model_override
    return cfg


def metrics_from_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, int | float]:
    """Compute the exact POPE/RePOPE binary metrics from (label, prediction)."""
    counts: Counter[str] = Counter(tp=0, fp=0, tn=0, fn=0, unknown=0)
    for label, prediction in pairs:
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
    total = sum(counts.values())
    correct = counts["tp"] + counts["tn"]
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": total, "correct": correct, **dict(counts), "accuracy": correct / total if total else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "yes_ratio": (counts["tp"] + counts["fp"]) / total if total else 0.0,
        "unknown_ratio": counts["unknown"] / total if total else 0.0,
        "false_positive_rate": counts["fp"] / (counts["fp"] + counts["tn"])
        if counts["fp"] + counts["tn"] else 0.0,
    }


def grouped_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    rows = list(rows)
    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in ("random", "popular", "adversarial")}
    all_pairs: list[tuple[str, str]] = []
    for row in rows:
        pair = (str(row["label"]).lower(), str(row["prediction"]).lower())
        grouped[str(row["split"])].append(pair)
        all_pairs.append(pair)
    return {**{name: metrics_from_pairs(values) for name, values in grouped.items()}, "overall": metrics_from_pairs(all_pairs)}


def experiment_target_regex(experiment: str, include_merger: bool = False) -> str:
    """Return an exact PEFT regex; loose q_proj suffix matching is unsafe in VLMs."""
    attention = r"language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    if experiment == "e1":
        return rf"^model\.{attention}$"
    if experiment == "e2":
        return rf"^model\.({attention}|visual\.merger\.mlp\.(0|2))$"
    if experiment == "e4":
        all_linear = r"language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        suffix = rf"|visual\.merger\.mlp\.(0|2)" if include_merger else ""
        return rf"^model\.({all_linear}{suffix})$"
    raise ValueError(f"Unknown experiment: {experiment}")


def load_quantized_model(experiment: str, gradient_checkpointing: bool, max_visual_tokens: int = 256, attach_lora: bool = True, include_merger: bool = False):
    """Load Qwen2-VL, prepare frozen 4-bit weights, then attach only approved LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    cfg = config()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    processor = AutoProcessor.from_pretrained(cfg["model_path"], local_files_only=True, max_pixels=max_visual_tokens * PIXELS_PER_VISUAL_TOKEN)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model_path"], local_files_only=True, quantization_config=quant, torch_dtype=dtype,
        device_map="auto", attn_implementation="sdpa",
    )
    model.config.use_cache = False
    # Transformers 5 stores the decoder cache flag inside text_config for
    # multimodal models; setting only the top-level attribute is insufficient.
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if not attach_lora:
        return model, processor, dtype
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=experiment_target_regex(experiment, include_merger),
    )
    model = get_peft_model(model, lora_cfg)
    audit_trainable_parameters(model, experiment, include_merger)
    return model, processor, dtype


def audit_trainable_parameters(model: Any, experiment: str, include_merger: bool = False) -> list[str]:
    """Fail early if LoRA leaks into a frozen visual encoder or misses the merger in E2."""
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    forbidden = [name for name in names if ".visual." in name and ".visual.merger." not in name]
    if forbidden:
        raise RuntimeError(f"Visual encoder unexpectedly trainable: {forbidden[:5]}")
    merger = [name for name in names if ".visual.merger." in name]
    if (experiment == "e2" or include_merger) and not merger:
        raise RuntimeError("E2 requested Merger LoRA but no merger adapter parameters were found.")
    if experiment != "e2" and not include_merger and merger:
        raise RuntimeError(f"Only E2 may train merger adapters: {merger[:5]}")
    llm = [name for name in names if ".language_model." in name]
    if not llm:
        raise RuntimeError("No language-model LoRA parameters were found.")
    return names


def optimizer_groups(model: Any, experiment: str, include_merger: bool = False):
    """E2 uses a separate low LR only for Merger-LoRA; all other experiments have one group."""
    import torch
    cfg = config()
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if experiment != "e2" and not include_merger:
        return torch.optim.AdamW([parameter for _, parameter in named], lr=cfg["lora"]["llm_lr"])
    merger = [parameter for name, parameter in named if ".visual.merger." in name]
    llm = [parameter for name, parameter in named if ".visual.merger." not in name]
    return torch.optim.AdamW([
        {"params": llm, "lr": cfg["lora"]["llm_lr"]},
        {"params": merger, "lr": cfg["lora"]["merger_lr"]},
    ])


class CocoYesNoDataset:
    """Small map-style dataset; image files are opened lazily to keep host RAM stable."""
    def __init__(self, rows: list[dict[str, Any]]): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


class AnswerOnlyCollator:
    """Creates Qwen2-VL batches and masks every token except the assistant answer."""
    def __init__(self, processor: Any): self.processor = processor

    @staticmethod
    def _conversation(question: str, answer: str | None = None) -> list[dict[str, Any]]:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + '\nAnswer using only "yes" or "no".'}]}]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        from PIL import Image
        import torch
        encoded, labels, mm_token_types, pixels, grids = [], [], [], [], []
        for row in rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
                question, answer = str(row["question"]), str(row["label"])
                prefix = self.processor.apply_chat_template(self._conversation(question), tokenize=False, add_generation_prompt=True)
                full = self.processor.apply_chat_template(self._conversation(question, answer), tokenize=False, add_generation_prompt=False)
                prefix_ids = self.processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"][0]
                inputs = self.processor(text=[full], images=[image], return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            if len(prefix_ids) >= len(input_ids):
                raise RuntimeError("Assistant completion boundary could not be identified.")
            label_ids = input_ids.clone()
            label_ids[: len(prefix_ids)] = -100
            label_ids[input_ids == self.processor.tokenizer.pad_token_id] = -100
            encoded.append({"input_ids": input_ids, "attention_mask": inputs["attention_mask"][0]})
            labels.append(label_ids)
            # Transformers >=5 supplies this field for Qwen2-VL M-RoPE. It is
            # sequence-aligned and must be padded exactly like input_ids.
            if "mm_token_type_ids" not in inputs:
                raise RuntimeError("Processor did not return required mm_token_type_ids.")
            mm_token_types.append(inputs["mm_token_type_ids"][0])
            pixels.append(inputs["pixel_values"]); grids.append(inputs["image_grid_thw"])
        padded = self.processor.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        max_len = padded["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        padded_mm_types = torch.zeros((len(mm_token_types), max_len), dtype=torch.long)
        left_padding = self.processor.tokenizer.padding_side == "left"
        for index, (label_value, mm_value) in enumerate(zip(labels, mm_token_types)):
            offset = max_len - len(label_value) if left_padding else 0
            padded_labels[index, offset:offset + len(label_value)] = label_value
            padded_mm_types[index, offset:offset + len(mm_value)] = mm_value
        padded["labels"] = padded_labels
        padded["mm_token_type_ids"] = padded_mm_types
        padded["pixel_values"] = torch.cat(pixels, dim=0)
        padded["image_grid_thw"] = torch.cat(grids, dim=0)
        return padded

    def supervised_text(self, batch: dict[str, Any]) -> str:
        """Human-readable smoke-test assertion: it must decode to yes/no + EOS only."""
        ids = batch["labels"][0][batch["labels"][0] != -100].tolist()
        return self.processor.tokenizer.decode(ids, skip_special_tokens=False)


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
