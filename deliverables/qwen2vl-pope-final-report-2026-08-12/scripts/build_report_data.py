"""Build the compact, offline QLoRA final-report data and image package.

This script never runs model inference. It only reads frozen metrics/predictions,
selects a deterministic diagnostic set, and extracts matching images from the
already-cached local POPE dataset.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPORT_DIR.parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
EVAL_DIR = RESULTS_DIR / "qlora_evaluations"
RUNS_DIR = RESULTS_DIR / "qlora_runs"
DATA_DIR = PROJECT_DIR / "data" / "processed"
CACHE_DIR = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))

MODEL_FILES = {
    "E0": "e0_fresh_base_repope_256vt_predictions.jsonl",
    "E1": "e1_checkpoint-1000_repope_256vt_predictions.jsonl",
    "E2": "e2_checkpoint-1000_repope_256vt_predictions.jsonl",
    "E3": "e3_checkpoint-1000_repope_256vt_predictions.jsonl",
    "E4": "e4_checkpoint-1000_repope_256vt_predictions.jsonl",
}

MODEL_LABELS = {
    "E0": "Fresh base",
    "E1": "Attention LoRA",
    "E2": "+ Visual Merger",
    "E3": "+ Hard negatives",
    "E4": "Attention + MLP",
}

# Hand-audited picks from deterministic candidate sheets. The tuple order is
# the five fixed quota slots declared in select_error_cases().
CASE_QUESTION_IDS = {
    "E0": ("17", "165", "84", "174", "537"),
    "E1": ("50", "78", "1297", "413", "200"),
    "E2": ("130", "94", "735", "966", "192"),
    "E3": ("217", "289", "88", "133", "483"),
    "E4": ("195", "247", "228", "229", "627"),
}

CASE_DIAGNOSTICS = {
    ("E0", "17"): "密集交通场景中的真实 car 漏检",
    ("E0", "165"): "远距离人物与羊群背景下的 person 漏检",
    ("E0", "84"): "儿童室内场景中的 handbag 误报",
    ("E0", "174"): "反光车窗近景中的 cup 误报",
    ("E0", "537"): "镜中人物未被识别；五个模型共同失败",
    ("E1", "50"): "雪地救援场景中的 cell phone 误报",
    ("E1", "78"): "厨房备餐场景中的 bottle 误报",
    ("E1", "1297"): "室内人物场景中的 clock 漏检",
    ("E1", "413"): "婚礼照片中的 tie 漏检",
    ("E1", "200"): "风筝冲浪被误判为 skis；五个模型共同失败",
    ("E2", "130"): "床铺与宠物场景中的 clock 误报",
    ("E2", "94"): "婚礼切蛋糕场景中的 chair 误报",
    ("E2", "735"): "床头小型 clock 漏检",
    ("E2", "966"): "运动场远景中的 chair 漏检",
    ("E2", "192"): "滑板场景中的 bench 误报；五个模型共同失败",
    ("E3", "217"): "低机位运动场景中的 person 漏检",
    ("E3", "289"): "马术场景中的 chair 漏检",
    ("E3", "88"): "足球场景中的 truck 误报",
    ("E3", "133"): "多显示器工作区被误判为 tv",
    ("E3", "483"): "远景海滩中的 horse 漏检；五个模型共同失败",
    ("E4", "195"): "设备密集桌面中的 remote 漏检",
    ("E4", "247"): "街景中的 traffic light 漏检",
    ("E4", "228"): "食物近景中的 spoon 误报",
    ("E4", "229"): "笔记本工作区被误判为 tv",
    ("E4", "627"): "雪地人物身上的 backpack 漏检；五个模型共同失败",
}

DEV_FILES = [
    "e0_base_2k_dev_256vt_metrics.json",
    *[
        f"e{experiment}_checkpoint-{step}_dev_256vt_metrics.json"
        for experiment in range(1, 5)
        for step in (500, 1000, 1500, 2000)
    ],
]

REPOPE_FILES = [
    "e0_fresh_base_repope_256vt_metrics.json",
    "e1_checkpoint-1000_repope_256vt_metrics.json",
    "e2_checkpoint-1000_repope_256vt_metrics.json",
    "e3_checkpoint-1000_repope_256vt_metrics.json",
    "e3_checkpoint-1000_repope_512vt_metrics.json",
    "e3_exploratory_high_recall_checkpoint-2000_repope_256vt_metrics.json",
    "e4_checkpoint-1000_repope_256vt_metrics.json",
    "e4_exploratory_high_recall_checkpoint-2000_repope_256vt_metrics.json",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_payload(value):
    """Remove workstation-specific absolute paths from public JSON artifacts."""
    if isinstance(value, dict):
        return {key: portable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_payload(item) for item in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    project_prefix = str(PROJECT_DIR).replace("\\", "/").rstrip("/") + "/"
    if normalized.lower().startswith(project_prefix.lower()):
        return normalized[len(project_prefix):]
    if len(normalized) > 2 and normalized[1:3] == ":/":
        if normalized.endswith("Qwen2-VL-2B-Instruct"):
            return "Qwen/Qwen2-VL-2B-Instruct"
        return Path(normalized).name
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric_record(filename: str) -> dict:
    payload = read_json(EVAL_DIR / filename)
    stem = filename.removesuffix("_metrics.json")
    parts = stem.split("_")
    experiment = parts[0].upper()
    checkpoint = 0 if experiment == "E0" else int(next(p.split("-")[1] for p in parts if p.startswith("checkpoint-")))
    visual_tokens = int(next(p.removesuffix("vt") for p in parts if p.endswith("vt")))
    return {
        "id": stem,
        "experiment": experiment,
        "checkpoint": checkpoint,
        "visualTokens": visual_tokens,
        "exploratory": "exploratory" in stem,
        "dataset": payload["dataset"],
        "metrics": payload["metrics"],
    }


def prediction_maps() -> dict[str, dict[tuple[str, str], dict]]:
    maps: dict[str, dict[tuple[str, str], dict]] = {}
    for model, filename in MODEL_FILES.items():
        rows = read_jsonl(EVAL_DIR / filename)
        maps[model] = {(row["split"], str(row["question_id"])): row for row in rows}
    reference = set(maps["E0"])
    assert all(set(rows) == reference for rows in maps.values()), "RePOPE prediction keys are not aligned"
    return maps


def error_type(row: dict) -> str:
    return "FP" if row["label"] == "no" else "FN"


def select_error_cases(maps: dict[str, dict[tuple[str, str], dict]]) -> list[dict]:
    used_images: set[str] = set()
    selected: list[dict] = []
    dominant = {"E0": "FN", "E1": "FP", "E2": "FP", "E3": "FN", "E4": "FN"}

    def choose(model: str, split: str, kind: str, shared: bool, question_id: str) -> dict:
        candidates = []
        for key, row in maps[model].items():
            if row["prediction"] == row["label"] or row["split"] != split or error_type(row) != kind:
                continue
            all_wrong = all(maps[name][key]["prediction"] != row["label"] for name in MODEL_FILES)
            if all_wrong != shared or row["image_source"] in used_images:
                continue
            candidates.append(row)
        candidates = [row for row in candidates if str(row["question_id"]) == question_id]
        if not candidates:
            raise RuntimeError(f"No case for {model}/{split}/{kind}/shared={shared}")
        return candidates[0]

    for model in MODEL_FILES:
        if dominant[model] == "FN":
            slots = [
                ("random", "FN", False),
                ("popular", "FN", False),
                ("adversarial", "FP", False),
                ("popular", "FP", False),
                ("adversarial", "FN", True),
            ]
        else:
            slots = [
                ("random", "FP", False),
                ("popular", "FP", False),
                ("adversarial", "FN", False),
                ("popular", "FN", False),
                ("adversarial", "FP", True),
            ]
        for ordinal, ((split, kind, shared), question_id) in enumerate(zip(slots, CASE_QUESTION_IDS[model]), 1):
            row = choose(model, split, kind, shared, question_id)
            key = (row["split"], str(row["question_id"]))
            used_images.add(row["image_source"])
            answers = {name: maps[name][key]["prediction"] for name in MODEL_FILES}
            correct_models = [name for name, answer in answers.items() if answer == row["label"]]
            selected.append(
                {
                    "id": f"{model.lower()}-{ordinal:02d}",
                    "focusModel": model,
                    "focusLabel": MODEL_LABELS[model],
                    "split": row["split"],
                    "questionId": str(row["question_id"]),
                    "imageSource": row["image_source"],
                    "question": row["question"],
                    "truth": row["label"],
                    "errorType": kind,
                    "sharedFailure": shared,
                    "answers": answers,
                    "correctModels": correct_models,
                    "diagnostic": CASE_DIAGNOSTICS[(model, str(row["question_id"]))],
                    "image": f"assets/error-cases/{model.lower()}-{ordinal:02d}-{split}-{kind.lower()}-qid-{row['question_id']}.jpg",
                    "sourcePrediction": MODEL_FILES[model],
                }
            )
    return selected


def validate_cases(cases: list[dict], maps: dict[str, dict[tuple[str, str], dict]]) -> None:
    assert len(cases) == 25
    assert len({case["imageSource"] for case in cases}) == 25
    for model in MODEL_FILES:
        group = [case for case in cases if case["focusModel"] == model]
        assert len(group) == 5
        counts = Counter(case["errorType"] for case in group)
        assert counts["FP"] >= 2 and counts["FN"] >= 2
        assert {case["split"] for case in group} == {"random", "popular", "adversarial"}
        assert sum(case["sharedFailure"] for case in group) == 1
        for case in group:
            key = (case["split"], case["questionId"])
            assert case["answers"][model] != case["truth"]
            assert case["answers"] == {name: maps[name][key]["prediction"] for name in MODEL_FILES}
            assert case["sharedFailure"] or case["correctModels"]


def extract_case_images(cases: list[dict]) -> None:
    from datasets import DownloadConfig, load_dataset

    dataset = load_dataset(
        "lmms-lab/POPE",
        "Full",
        cache_dir=str(CACHE_DIR),
        download_config=DownloadConfig(local_files_only=True),
    )
    indexes = {
        split: {str(sample["question_id"]): index for index, sample in enumerate(dataset[split])}
        for split in ("random", "popular", "adversarial")
    }
    output_dir = REPORT_DIR / "assets" / "error-cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("*.jpg"):
        old_image.unlink()
    for case in cases:
        sample = dataset[case["split"]][indexes[case["split"]][case["questionId"]]]
        assert sample["image_source"] == case["imageSource"]
        image = sample["image"].convert("RGB")
        output = REPORT_DIR / case["image"]
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, "JPEG", quality=88, optimize=True, progressive=True)


def visual_token_transitions() -> dict:
    paths = {
        "from": EVAL_DIR / "e3_checkpoint-1000_repope_256vt_predictions.jsonl",
        "to": EVAL_DIR / "e3_checkpoint-1000_repope_512vt_predictions.jsonl",
    }
    rows_256 = {(r["split"], str(r["question_id"])): r for r in read_jsonl(paths["from"])}
    rows_512 = {(r["split"], str(r["question_id"])): r for r in read_jsonl(paths["to"])}
    transitions = Counter()
    by_split: dict[str, Counter] = {name: Counter() for name in ("random", "popular", "adversarial")}
    for key, before in rows_256.items():
        after = rows_512[key]
        label = before["label"]
        if before["prediction"] == after["prediction"]:
            continue
        name = (
            "FN_TO_TP" if label == "yes" and before["prediction"] == "no" else
            "TP_TO_FN" if label == "yes" else
            "FP_TO_TN" if before["prediction"] == "yes" else
            "TN_TO_FP"
        )
        transitions[name] += 1
        by_split[before["split"]][name] += 1
    return {"overall": dict(transitions), "bySplit": {name: dict(values) for name, values in by_split.items()}}


def copy_sources() -> None:
    metrics = [RESULTS_DIR / "qwen2vl2b_baseline_metrics.json", RESULTS_DIR / "qwen2vl2b_repope_random_positive_512_metrics.json", RESULTS_DIR / "qlora_speed_benchmark.json", RESULTS_DIR / "smoke_metrics.json"]
    metrics.extend(EVAL_DIR / filename for filename in DEV_FILES + REPOPE_FILES)
    for source in metrics:
        target = REPORT_DIR / "metrics" / source.name
        target.write_text(json.dumps(portable_payload(read_json(source)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config_sources = [
        PROJECT_DIR / "configs" / "qlora_experiments.json",
        RESULTS_DIR / "qlora_locked_training_config.json",
        *[RUNS_DIR / f"e{experiment}" / "run_manifest.json" for experiment in range(1, 5)],
        *[RUNS_DIR / f"e{experiment}" / "selection_dev2k.json" for experiment in range(1, 5)],
        RUNS_DIR / "smoke" / "run_manifest.json",
    ]
    for source in config_sources:
        prefix = source.parent.name + "_" if source.name in {"run_manifest.json", "selection_dev2k.json"} else ""
        target = REPORT_DIR / "config" / f"{prefix}{source.name}"
        target.write_text(json.dumps(portable_payload(read_json(source)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name in ("coco_repope_style_manifest.json", "coco_dev_repope_style_2k_manifest.json"):
        source = DATA_DIR / name
        target = REPORT_DIR / "manifests" / name
        target.write_text(json.dumps(portable_payload(read_json(source)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    maps = prediction_maps()
    cases = select_error_cases(maps)
    validate_cases(cases, maps)
    extract_case_images(cases)
    copy_sources()

    fresh = read_json(EVAL_DIR / "e0_fresh_base_repope_256vt_metrics.json")["metrics"]["overall"]
    assert (fresh["tp"], fresh["fp"], fresh["tn"], fresh["fn"]) == (3062, 187, 4459, 477)
    assert (fresh["tp"], fresh["fp"], fresh["tn"], fresh["fn"]) != (3056, 169, 4477, 483)

    pope = read_json(RESULTS_DIR / "qwen2vl2b_baseline_metrics.json")
    speed = read_json(RESULTS_DIR / "qlora_speed_benchmark.json")
    smoke = read_json(RUNS_DIR / "smoke" / "run_manifest.json")
    data_manifest = read_json(DATA_DIR / "coco_repope_style_manifest.json")
    dev_manifest = read_json(DATA_DIR / "coco_dev_repope_style_2k_manifest.json")

    report_data = {
        "meta": {
            "title": "用 QLoRA 缓解 Qwen2-VL 视觉幻觉：RePOPE 实验",
            "subtitle": "Qwen2-VL-2B object-existence hallucination",
            "date": "2026-08-12",
            "theme": "mist-blue",
            "freshBaselineOnly": True,
        },
        "models": MODEL_LABELS,
        "devCheckpoints": [metric_record(filename) for filename in DEV_FILES],
        "repopeRuns": [metric_record(filename) for filename in REPOPE_FILES],
        "originalPope": pope["metrics"],
        "ablations": {"visualTokens": visual_token_transitions()},
        "errorSummary": {
            model: {
                "total": sum(row["prediction"] != row["label"] for row in rows.values()),
                "fp": sum(row["prediction"] == "yes" and row["label"] == "no" for row in rows.values()),
                "fn": sum(row["prediction"] == "no" and row["label"] == "yes" for row in rows.values()),
            }
            for model, rows in maps.items()
        },
        "errorCases": cases,
        "engineering": {"speed": speed, "smoke": smoke},
        "protocol": {"data": data_manifest, "dev": dev_manifest},
    }

    report_data = portable_payload(report_data)
    (REPORT_DIR / "error_case_manifest.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = "window.QLORA_REPORT_DATA = " + json.dumps(report_data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (REPORT_DIR / "assets" / "report-data.js").write_text(payload, encoding="utf-8")
    print(f"Built {len(cases)} validated cases and canonical report data in {REPORT_DIR}")


if __name__ == "__main__":
    main()
