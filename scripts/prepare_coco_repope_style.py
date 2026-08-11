"""Build leakage-free COCO-train yes/no data with POPE/RePOPE-style splits.

Run with --download once to fetch official COCO annotations and only the image
files referenced by the final JSONL rows. It never downloads or trains on POPE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from urllib.error import HTTPError, URLError
import zipfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.qlora_common import DATA_DIR, config, read_jsonl, write_jsonl

RAW = PROJECT_DIR / "data" / "raw" / "coco_train2017"
REPOPE = PROJECT_DIR / "data" / "raw" / "repope"
ANNOTATION_ZIP = RAW / "annotations_trainval2017.zip"
ANNOTATION_JSON = RAW / "annotations" / "instances_train2017.json"
IMAGE_DIR = RAW / "images"
# This environment uses a TLS-inspecting proxy with a hostname-mismatched
# certificate. The official COCO HTTP endpoint remains reachable here; ZIP
# integrity and byte-length checks below protect the downloaded artifacts.
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/train2017/{filename}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create COCO train/dev data without RePOPE image leakage.")
    parser.add_argument("--download", action="store_true", help="Download missing official COCO annotation ZIP and selected images.")
    parser.add_argument("--download-workers", type=int, default=24,
                        help="Concurrent COCO image downloads (default: 24).")
    parser.add_argument("--seed", type=int, default=config()["seed"])
    parser.add_argument("--force", action="store_true", help="Replace existing processed COCO JSONL files.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def publish_completed_download(temporary: Path, destination: Path) -> bool:
    """Publish a completed ``.part`` file, tolerating transient Windows file locks.

    Windows Defender, Explorer preview, and notebook-side file inspection can hold a
    just-written file briefly.  Keeping the valid ``.part`` file is safer than
    re-downloading hundreds of MiB, so a failed publication is explicitly
    recoverable on the next invocation.
    """
    for attempt in range(1, 16):
        try:
            temporary.replace(destination)
            return True
        except PermissionError as error:
            if attempt == 15:
                print(
                    f"下载已完整，但 {temporary.name} 仍被其他程序占用；将保留该文件供下次继续使用。\n"
                    f"请关闭正在查看该文件的程序后重试。详情：{error}"
                )
                return False
            print(f"等待 Windows 释放 {temporary.name}（{attempt}/15）…")
            time.sleep(1)
    return False  # Defensive; the loop always returns.


def download(url: str, destination: Path, *, show_byte_progress: bool = True) -> None:
    """Resumable stream download with progress and atomic final publication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 9):
        existing = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "qwen2vl-pope/1.0"}
        if existing: headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 8: raise RuntimeError(f"COCO server remained unavailable after 8 attempts: {error}") from error
            delay = min(5 * attempt, 30)
            print(f"COCO 服务器暂时不可用（{error}）；{delay}s 后第 {attempt + 1}/8 次重试，已下载内容会保留。")
            time.sleep(delay)
            continue
        with response:
            remaining = int(response.headers.get("Content-Length", 0)) or None
            partial_response = getattr(response, "status", response.getcode()) == 206
            # If a proxy ignores Range, restart safely rather than append duplicate bytes.
            if existing and not partial_response:
                print("服务器未接受断点续传，重新开始该文件下载。")
                existing = 0; temporary.unlink(missing_ok=True)
            total = existing + remaining if remaining else None
            mode = "ab" if existing else "wb"
            bar = tqdm(total=total, initial=existing, unit="B", unit_scale=True,
                       unit_divisor=1024, desc=f"下载 {destination.name}",
                       dynamic_ncols=True, disable=not show_byte_progress)
            with temporary.open(mode) as handle, bar:
                while block := response.read(1024 * 1024):
                    handle.write(block); bar.update(len(block))
        actual = temporary.stat().st_size
        if total is None or actual >= total:
            if publish_completed_download(temporary, destination):
                return
            raise RuntimeError(
                f"{destination.name} 已下载完成但无法改名；保留的文件为 {temporary}。"
            )
        print(f"下载提前中断（{actual / 1024**2:.1f} / {total / 1024**2:.1f} MiB），第 {attempt}/8 次从断点续传。")
    raise RuntimeError(f"Download failed after 8 resumable attempts: {url}")


def valid_zip(path: Path) -> bool:
    """Return whether a ZIP can be fully read, without trusting its file size."""
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def ensure_annotations(download_enabled: bool) -> None:
    if ANNOTATION_JSON.exists(): return
    temporary = ANNOTATION_ZIP.with_suffix(ANNOTATION_ZIP.suffix + ".part")
    archive_path: Path | None = ANNOTATION_ZIP if valid_zip(ANNOTATION_ZIP) else None
    # A prior interrupted transfer can leave a large-looking but invalid ZIP.
    if ANNOTATION_ZIP.exists() and archive_path is None:
        print(f"检测到未完成的 annotation ZIP，将从断点续传：{ANNOTATION_ZIP}")
        if not temporary.exists():
            ANNOTATION_ZIP.replace(temporary)
        else:
            # A separate partial file is more useful for resume than a known-invalid ZIP.
            ANNOTATION_ZIP.unlink(missing_ok=True)
    # A complete .part file may be left by a transient file lock after download.
    if archive_path is None and valid_zip(temporary):
        print("检测到已完整下载的 annotation .part 文件，尝试完成发布而不重复下载。")
        if publish_completed_download(temporary, ANNOTATION_ZIP):
            archive_path = ANNOTATION_ZIP
        else:
            archive_path = temporary
            print("将直接从有效的 .part 文件解压；下次执行会再次尝试改名。")
    if archive_path is None and not download_enabled:
        raise FileNotFoundError(f"Missing {ANNOTATION_JSON}. Re-run with --download.")
    if archive_path is None:
        download(COCO_ANNOTATIONS_URL, ANNOTATION_ZIP)
        archive_path = ANNOTATION_ZIP
    if not valid_zip(archive_path):
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("COCO annotation download is not a valid ZIP; please retry --download.")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extract("annotations/instances_train2017.json", RAW)


def repope_test_image_ids() -> set[int]:
    """Only image ids are read to block train leakage; no RePOPE labels/questions are used."""
    ids: set[int] = set()
    for split in ("random", "popular", "adversarial"):
        for row in read_jsonl(REPOPE / f"coco_repope_{split}.json"):
            ids.add(int(Path(str(row["image"])).stem.rsplit("_", 1)[1]))
    return ids


def load_coco() -> tuple[dict[int, str], dict[int, set[int]], dict[int, int], dict[int, Counter[int]]]:
    print(f"读取 COCO 标注（{ANNOTATION_JSON.stat().st_size / 1024**2:.1f} MiB）…", flush=True)
    raw = json.loads(ANNOTATION_JSON.read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in raw["categories"]}
    objects: dict[int, set[int]] = defaultdict(set)
    frequency: Counter[int] = Counter()
    for annotation in tqdm(raw["annotations"], desc="索引 COCO objects", unit="annotation",
                           dynamic_ncols=True, mininterval=2.0):
        objects[int(annotation["image_id"])].add(int(annotation["category_id"]))
        frequency[int(annotation["category_id"])] += 1
    # Count image-level co-occurrence; object instances must not inflate it.
    cooc: dict[int, Counter[int]] = defaultdict(Counter)
    for present in tqdm(objects.values(), desc="统计类别共现", unit="image",
                        dynamic_ncols=True, mininterval=2.0):
        for source in present:
            for target in present:
                if source != target: cooc[source][target] += 1
    return category_names, objects, dict(frequency), cooc


def choose_negative(strategy: str, present: set[int], categories: list[int], frequency: dict[int, int], cooc: dict[int, Counter[int]], rng: random.Random) -> int:
    absent = [category for category in categories if category not in present]
    if strategy == "random": return rng.choice(absent)
    if strategy == "popular": return rng.choices(absent, weights=[frequency[item] for item in absent], k=1)[0]
    if strategy == "adversarial":
        weights = [sum(cooc[source][candidate] for source in present) for candidate in absent]
        return rng.choices(absent, weights=weights if any(weights) else [frequency[item] for item in absent], k=1)[0]
    raise ValueError(strategy)


def question(category: str) -> str:
    return f"Is there a {category} in the image?"


def generate_rows(pool: list[int], counts: dict[str, int], category_names: dict[int, str], objects: dict[int, set[int]], frequency: dict[int, int], cooc: dict[int, Counter[int]], rng: random.Random, split: str) -> list[dict[str, Any]]:
    """Generate exact class quotas, rejecting duplicate image/category/label questions."""
    categories = sorted(category_names)
    target_order = [("positive", "yes", counts["positive"])] + [(kind, "no", counts[kind]) for kind in ("random", "popular", "adversarial")]
    rows: list[dict[str, Any]] = []; seen: set[tuple[int, int, str]] = set()
    valid_images = [image_id for image_id in pool if objects[image_id]]
    for kind, label, target in target_order:
        attempts = 0
        while sum(1 for row in rows if row["kind"] == kind) < target:
            attempts += 1
            if attempts > target * 200:
                raise RuntimeError(f"Could not satisfy {split}/{kind} quota; enlarge candidate image pool.")
            image_id = rng.choice(valid_images); present = objects[image_id]
            category = rng.choice(sorted(present)) if kind == "positive" else choose_negative(kind, present, categories, frequency, cooc, rng)
            key = (image_id, category, label)
            if key in seen: continue
            seen.add(key)
            filename = f"{image_id:012d}.jpg"
            rows.append({
                "split": split, "kind": kind, "label": label, "image_id": image_id,
                "object": category_names[category], "question": question(category_names[category]),
                "image_filename": filename, "image_path": str(IMAGE_DIR / filename),
            })
    rng.shuffle(rows)
    return rows


def generate_train_rows_with_exact_image_count(
    pool: list[int],
    counts: dict[str, int],
    category_names: dict[int, str],
    objects: dict[int, set[int]],
    frequency: dict[int, int],
    cooc: dict[int, Counter[int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build 16k questions while using every selected training image.

    For the configured 10k images and 16k questions, each image receives one
    mandatory question and 6k distinct images receive one additional question.
    Thus image frequency is controlled (one or two questions per image) rather
    than left to unconstrained sampling with replacement.
    """
    image_ids = list(pool)
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError("Training image pool contains duplicate image ids.")
    total = sum(int(value) for value in counts.values())
    positive_count = int(counts["positive"])
    negative_count = total - positive_count
    extra_count = total - len(image_ids)
    if not (0 <= positive_count <= len(image_ids)):
        raise RuntimeError("Positive quota must fit within the unique training-image pool.")
    if not (0 <= extra_count <= len(image_ids)):
        raise RuntimeError("Exact image coverage currently supports one or two questions per image.")
    if negative_count != sum(int(counts[k]) for k in ("random", "popular", "adversarial")):
        raise RuntimeError("Negative quota mismatch.")

    rng.shuffle(image_ids)
    positive_images = image_ids[:positive_count]
    # Images not used for the mandatory positives first receive a mandatory
    # negative; the extra 6k slots use distinct images sampled from the full pool.
    negative_images = image_ids[positive_count:] + rng.sample(image_ids, extra_count)
    if len(negative_images) != negative_count:
        raise RuntimeError("Image-slot construction does not match the negative quota.")
    negative_strategies = [
        kind
        for kind in ("random", "popular", "adversarial")
        for _ in range(int(counts[kind]))
    ]
    rng.shuffle(negative_strategies)

    categories = sorted(category_names)
    used: set[tuple[int, int, str]] = set()
    rows: list[dict[str, Any]] = []

    def append_row(image_id: int, category_id: int, kind: str, label: str) -> None:
        filename = f"{image_id:012d}.jpg"
        object_name = category_names[category_id]
        key = (image_id, category_id, label)
        if key in used:
            raise RuntimeError(f"Duplicate train question generated: {key}")
        used.add(key)
        rows.append({
            "split": "train", "kind": kind, "label": label, "image_id": image_id,
            "object": object_name, "question": question(object_name),
            "image_filename": filename, "image_path": str(IMAGE_DIR / filename),
        })

    for image_id in positive_images:
        append_row(image_id, rng.choice(sorted(objects[image_id])), "positive", "yes")
    for image_id, kind in zip(negative_images, negative_strategies):
        category_id = -1
        for _ in range(200):
            candidate = choose_negative(kind, objects[image_id], categories, frequency, cooc, rng)
            if (image_id, candidate, "no") not in used:
                category_id = candidate
                break
        if category_id < 0:
            raise RuntimeError(f"Could not create a unique train negative for image {image_id}.")
        append_row(image_id, category_id, kind, "no")

    rng.shuffle(rows)
    if len(rows) != total or len({row["image_id"] for row in rows}) != len(image_ids):
        raise RuntimeError("Exact training image/question quota was not satisfied.")
    return rows


def rebuild_negative_mix_on_same_images(
    base_rows: list[dict[str, Any]],
    counts: dict[str, int],
    category_names: dict[int, str],
    objects: dict[int, set[int]],
    frequency: dict[int, int],
    cooc: dict[int, Counter[int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Create E3 while preserving E1's exact image sequence and positive rows.

    E3 is intended to test only the negative-strategy mixture. Therefore every
    row keeps the E1 image assignment, all positive questions remain unchanged,
    and only negative objects/kinds are regenerated to meet the E3 quota.
    """
    positive_target = int(counts["positive"])
    negative_strategies = [
        kind
        for kind in ("random", "popular", "adversarial")
        for _ in range(int(counts[kind]))
    ]
    if sum(row["label"] == "yes" for row in base_rows) != positive_target:
        raise RuntimeError("E1 positive count cannot satisfy the requested E3 positive quota.")
    if sum(row["label"] == "no" for row in base_rows) != len(negative_strategies):
        raise RuntimeError("E1 negative slots cannot satisfy the requested E3 negative quota.")
    rng.shuffle(negative_strategies)
    categories = sorted(category_names)
    used: set[tuple[int, int, str]] = set()
    rebuilt: list[dict[str, Any]] = []
    strategy_index = 0

    for base in base_rows:
        image_id = int(base["image_id"])
        if base["label"] == "yes":
            row = dict(base)
            category_id = next(
                category for category, name in category_names.items()
                if name == row["object"]
            )
        else:
            kind = negative_strategies[strategy_index]
            strategy_index += 1
            present = objects[image_id]
            # Rejection sampling preserves the requested strategy while avoiding
            # duplicate image/object/label questions within E3.
            category_id = -1
            for _ in range(200):
                candidate = choose_negative(kind, present, categories, frequency, cooc, rng)
                if (image_id, candidate, "no") not in used:
                    category_id = candidate
                    break
            if category_id < 0:
                raise RuntimeError(f"Could not create a unique E3 negative for image {image_id}.")
            object_name = category_names[category_id]
            row = dict(base)
            row.update({"kind": kind, "label": "no", "object": object_name,
                        "question": question(object_name)})
        key = (image_id, category_id, str(row["label"]))
        if key in used:
            raise RuntimeError(f"Duplicate E3 question generated: {key}")
        used.add(key)
        rebuilt.append(row)

    if [row["image_id"] for row in rebuilt] != [row["image_id"] for row in base_rows]:
        raise RuntimeError("E3 image sequence differs from E1; experimental control violated.")
    if Counter(row["kind"] for row in rebuilt) != Counter({
        "positive": counts["positive"], "random": counts["random"],
        "popular": counts["popular"], "adversarial": counts["adversarial"],
    }):
        raise RuntimeError("E3 negative-strategy quota mismatch.")
    return rebuilt


def download_selected_images(rows: list[dict[str, Any]], enabled: bool, workers: int) -> None:
    missing = sorted({row["image_filename"] for row in rows if not Path(row["image_path"]).exists()})
    if missing and not enabled:
        raise FileNotFoundError(f"{len(missing)} COCO images are missing. Re-run with --download.")
    if not missing:
        print("所需 COCO 图片均已存在，跳过下载。")
        return
    if workers < 1:
        raise ValueError("--download-workers must be at least 1")
    print(f"需要下载 {len(missing)} 张 COCO 图片；并发数={workers}。已存在的图片已自动跳过。")

    def fetch(filename: str) -> str:
        download(COCO_IMAGE_URL.format(filename=filename), IMAGE_DIR / filename,
                 show_byte_progress=False)
        return filename

    # Small COCO JPEGs are latency-bound. Parallel requests remove the ~1–2 s
    # per-file handshake bottleneck while every file still uses an independent
    # .part path and atomic publication.
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="coco-download")
    futures = {executor.submit(fetch, filename): filename for filename in missing}
    try:
        with tqdm(total=len(missing), desc="下载 COCO 训练图片", unit="image",
                  dynamic_ncols=True, mininterval=1.0) as bar:
            for future in as_completed(futures):
                filename = future.result()
                bar.set_postfix_str(filename, refresh=False)
                bar.update(1)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def assert_disjoint(train: list[dict[str, Any]], dev: list[dict[str, Any]], blocked: set[int]) -> None:
    train_ids, dev_ids = {row["image_id"] for row in train}, {row["image_id"] for row in dev}
    if train_ids & dev_ids or train_ids & blocked or dev_ids & blocked:
        raise RuntimeError("Image leakage detected across train/dev/RePOPE test sets.")


def main() -> None:
    args = parse_args(); cfg = config(); ensure_annotations(args.download)
    out = {"e1": DATA_DIR / "coco_train_e1.jsonl", "e3": DATA_DIR / "coco_train_e3.jsonl", "dev": DATA_DIR / "coco_dev_repope_style.jsonl"}
    if any(path.exists() for path in out.values()) and not args.force:
        raise FileExistsError("Processed data exists; use --force only after intentionally replacing it.")
    names, objects, frequency, cooc = load_coco(); blocked = repope_test_image_ids()
    eligible = sorted(set(objects) - blocked); rng = random.Random(args.seed); rng.shuffle(eligible)
    # Reserve the dev candidate pool first, then select exactly the configured
    # number of leakage-free COCO images shared by E1–E4.
    dev_pool_size = 3000
    train_image_count = int(cfg["data"]["train_unique_images"])
    dev_pool = eligible[:dev_pool_size]
    train_pool = eligible[dev_pool_size:dev_pool_size + train_image_count]
    if len(train_pool) != train_image_count:
        raise RuntimeError("Not enough leakage-free COCO images for the configured training pool.")
    e1 = generate_train_rows_with_exact_image_count(
        train_pool, cfg["data"]["e1_counts"], names, objects, frequency, cooc,
        random.Random(args.seed),
    )
    e3 = rebuild_negative_mix_on_same_images(
        e1, cfg["data"]["e3_counts"], names, objects, frequency, cooc,
        random.Random(args.seed + 1),
    )
    dev_rows: list[dict[str, Any]] = []
    for offset, kind in enumerate(("random", "popular", "adversarial")):
        dev_rows.extend(generate_rows(dev_pool, {"positive": 500, "random": 500 if kind == "random" else 0, "popular": 500 if kind == "popular" else 0, "adversarial": 500 if kind == "adversarial" else 0}, names, objects, frequency, cooc, random.Random(args.seed + 10 + offset), kind))
    assert len(e1) == 16000 and len(e3) == 16000 and len(dev_rows) == 3000
    if len({row["image_id"] for row in e1}) != train_image_count:
        raise RuntimeError("Training set must contain exactly the configured number of unique images.")
    if [row["image_id"] for row in e1] != [row["image_id"] for row in e3]:
        raise RuntimeError("E1/E3 must use the exact same image sequence.")
    assert_disjoint(e1 + e3, dev_rows, blocked)
    download_selected_images(e1 + e3 + dev_rows, args.download, args.download_workers)
    write_jsonl(out["e1"], e1); write_jsonl(out["e3"], e3); write_jsonl(out["dev"], dev_rows)
    manifest = {"seed": args.seed, "source": "COCO train2017", "blocked_repope_image_count": len(blocked), "train_image_count": len({r['image_id'] for r in e1}), "dev_image_count": len({r['image_id'] for r in dev_rows}), "e1_e3_same_image_sequence": [r['image_id'] for r in e1] == [r['image_id'] for r in e3], "e1_e3_same_positive_rows": [r for r in e1 if r['label'] == 'yes'] == [r for r in e3 if r['label'] == 'yes'], "hashes": {name: sha256(path) for name, path in out.items()}, "counts": {"e1": Counter(row['kind'] for row in e1), "e3": Counter(row['kind'] for row in e3), "dev": Counter(f"{row['split']}:{row['label']}" for row in dev_rows)}}
    (DATA_DIR / "coco_repope_style_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__": main()
