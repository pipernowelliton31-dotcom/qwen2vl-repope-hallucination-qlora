import os

from transformers import AutoProcessor


MODEL_PATH = os.environ.get("QWEN2VL_MODEL_PATH", "Qwen/Qwen2-VL-2B-Instruct")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    max_pixels=256 * 28 * 28,
)

print("Processor 加载成功")
print(type(processor))
print(processor)
