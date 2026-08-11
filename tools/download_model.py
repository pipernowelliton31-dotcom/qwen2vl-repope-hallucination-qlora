import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOCAL_DIR = Path(os.environ.get("QWEN2VL_MODEL_PATH", Path.cwd() / "models" / "Qwen2-VL-2B-Instruct"))


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"开始下载：{MODEL_ID}")
    print(f"保存位置：{LOCAL_DIR}")

    downloaded_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        max_workers=4,
    )

    print("\n下载完成。")
    print(f"模型目录：{downloaded_path}")


if __name__ == "__main__":
    main()
