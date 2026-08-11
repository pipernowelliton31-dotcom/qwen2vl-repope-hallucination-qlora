import os
from pathlib import Path
import re
import time

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = os.environ.get("QWEN2VL_MODEL_PATH", "Qwen/Qwen2-VL-2B-Instruct")
IMAGE_PATH = Path(os.environ.get("QWEN2VL_TEST_IMAGE", PROJECT_DIR / "data" / "samples" / "test.jpg"))

# 这只是本地 POPE 风格连通性测试，不是正式 benchmark。
TEST_CASES = [
    ("Is there a car in the image?", "yes"),
    ("Is there water in the image?", "yes"),
    ("Is there a road in the image?", "yes"),
    ("Is there an airplane in the image?", "no"),
    ("Is there a refrigerator in the image?", "no"),
    ("Is there a cat in the image?", "no"),
]


def normalize_answer(text: str) -> str:
    """把模型输出统一解析为 yes、no 或 unknown。"""
    normalized = text.strip().lower()

    if normalized.startswith("yes"):
        return "yes"

    if normalized.startswith("no"):
        return "no"

    # 兼容模型输出 "The answer is yes." 等情况。
    match = re.search(r"\b(yes|no)\b", normalized)
    if match:
        return match.group(1)

    return "unknown"


def ask_question(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    image_uri: str,
    question: str,
    input_device: torch.device,
) -> tuple[str, str]:
    """针对同一张图片回答一个 yes/no 问题。"""

    # 为了让输出便于自动评测，明确限制只回答 yes 或 no。
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_uri,
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
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(input_device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
        )

    # generate() 的输出包含“原始输入 + 新生成内容”。
    # 这里删掉原始输入，只保留模型新生成的答案。
    generated_only = output_ids[
        :,
        inputs["input_ids"].shape[1] :
    ]

    raw_answer = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed_answer = normalize_answer(raw_answer)

    return raw_answer, parsed_answer


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"测试图片不存在：{IMAGE_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 CUDA 显卡。")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("GPU：", torch.cuda.get_device_name(0))
    print("计算精度：", compute_dtype)

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

    print("[2/3] 以 4-bit 加载模型")

    start_time = time.perf_counter()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    model.eval()

    load_time = time.perf_counter() - start_time
    input_device = model.get_input_embeddings().weight.device

    print(f"模型加载完成，耗时：{load_time:.2f} 秒")
    print("模型输入设备：", input_device)

    # 图片地址只生成一次，六个问题共同使用。
  # 图片地址只生成一次，六个问题共同使用。
    image_uri = str(IMAGE_PATH.resolve())

    print("\n[3/3] 开始 POPE 风格测试\n")

    correct_count = 0

    for index, (question, expected) in enumerate(TEST_CASES, start=1):
        start_time = time.perf_counter()

        raw_answer, parsed_answer = ask_question(
            model=model,
            processor=processor,
            image_uri=image_uri,
            question=question,
            input_device=input_device,
        )

        elapsed = time.perf_counter() - start_time
        is_correct = parsed_answer == expected

        if is_correct:
            correct_count += 1

        status = "正确" if is_correct else "错误"

        print(f"[{index}/{len(TEST_CASES)}] {question}")
        print(f"  原始输出：{raw_answer!r}")
        print(f"  解析结果：{parsed_answer}")
        print(f"  正确答案：{expected}")
        print(f"  判断：{status}")
        print(f"  耗时：{elapsed:.2f} 秒\n")

    accuracy = correct_count / len(TEST_CASES)

    print("=" * 50)
    print(f"正确数量：{correct_count}/{len(TEST_CASES)}")
    print(f"本地测试准确率：{accuracy:.2%}")

    print("\n显存统计：")
    print(
        "  当前已分配："
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    print(
        "  峰值已分配："
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )


if __name__ == "__main__":
    main()
