import os
from pathlib import Path
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

MAX_PIXELS = 256 * 28 * 28


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"测试图片不存在：{IMAGE_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 没有检测到 CUDA 显卡")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("GPU：", torch.cuda.get_device_name(0))
    print("计算精度：", compute_dtype)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("\n[1/4] 加载 Processor")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        max_pixels=MAX_PIXELS,
    )

    print("[2/4] 以 4-bit 加载模型")
    start_time = time.perf_counter()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map="auto",
    )

    model.eval()

    load_time = time.perf_counter() - start_time
    print(f"模型加载完成，耗时 {load_time:.2f} 秒")
    device_map = getattr(model, "hf_device_map", None)

    if device_map is not None:
        print("设备分配：", device_map)
    else:
        parameter_devices = sorted(
            {str(parameter.device) for parameter in model.parameters()}
        )
        print("未生成 hf_device_map")
        print("模型参数所在设备：", parameter_devices)

    image_uri = str(IMAGE_PATH.resolve())

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
                    "text": "Describe this image in one concise sentence.",
                },
            ],
        }
    ]

    print("\n[3/4] 处理图片和文本")

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

    print("\nProcessor 输出张量：")
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")

    inputs = inputs.to("cuda")

    torch.cuda.reset_peak_memory_stats()

    print("\n[4/4] 开始生成")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n模型回答：")
    print(answer)

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print("\n显存统计：")
    print(f"  当前已分配：{allocated:.2f} GB")
    print(f"  当前已保留：{reserved:.2f} GB")
    print(f"  峰值分配：{peak:.2f} GB")


if __name__ == "__main__":
    main()
