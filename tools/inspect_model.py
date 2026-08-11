import os
from pathlib import Path

import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)


MODEL_PATH = Path(os.environ.get("QWEN2VL_MODEL_PATH", "Qwen/Qwen2-VL-2B-Instruct"))


def main() -> None:
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        os.fspath(MODEL_PATH),
        local_files_only=True,
        quantization_config=quant_config,
        dtype=dtype,
        device_map="auto",
    )

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"总参数量：{total_parameters:,}")
    print(f"当前可训练参数量：{trainable_parameters:,}")

    print("\n模型一级子模块：")
    for name, module in model.named_children():
        print(f"{name}: {type(module).__name__}")

    print("\n包含 visual / merger 的模块：")
    for name, module in model.named_modules():
        lower_name = name.lower()
        if "visual" in lower_name or "merger" in lower_name:
            print(name, "->", type(module).__name__)

    print("\n部分 LoRA 候选目标层：")
    count = 0

    for name, module in model.named_modules():
        if name.endswith(
            ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            print(name, "->", type(module).__name__)
            count += 1

            if count >= 30:
                break


if __name__ == "__main__":
    main()
