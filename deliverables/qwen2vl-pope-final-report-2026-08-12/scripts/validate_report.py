"""Validate the final report's canonical metrics, cases, and offline assets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPORT_DIR.parents[1]
EVAL_DIR = PROJECT_DIR / "results" / "qlora_evaluations"
MODEL_FILES = {
    "E0": "e0_fresh_base_repope_256vt_predictions.jsonl",
    "E1": "e1_checkpoint-1000_repope_256vt_predictions.jsonl",
    "E2": "e2_checkpoint-1000_repope_256vt_predictions.jsonl",
    "E3": "e3_checkpoint-1000_repope_256vt_predictions.jsonl",
    "E4": "e4_checkpoint-1000_repope_256vt_predictions.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    cases = json.loads((REPORT_DIR / "error_case_manifest.json").read_text(encoding="utf-8"))
    data_js = (REPORT_DIR / "assets" / "report-data.js").read_text(encoding="utf-8")
    prefix = "window.QLORA_REPORT_DATA = "
    assert data_js.startswith(prefix) and data_js.endswith(";\n")
    data = json.loads(data_js[len(prefix):-2])

    fresh = next(run for run in data["repopeRuns"] if run["id"] == "e0_fresh_base_repope_256vt")
    metric = fresh["metrics"]["overall"]
    assert (metric["tp"], metric["fp"], metric["tn"], metric["fn"]) == (3062, 187, 4459, 477)
    assert "qwen2vl2b_baseline_repope_metrics" not in data_js
    assert "3056/169/4477/483" not in data_js

    config = json.loads((REPORT_DIR / "config" / "qlora_experiments.json").read_text(encoding="utf-8"))
    data_config = config["data"]
    assert data_config["dev_source_size"] == 3000
    assert data_config["dev_selection_size"] == 2000
    assert data_config["dev_selection_quotas"] == {
        "random": {"yes": 334, "no": 334},
        "popular": {"yes": 333, "no": 333},
        "adversarial": {"yes": 333, "no": 333},
    }
    assert "dev_size" not in data_config and "dev_per_split" not in data_config

    selections = {
        experiment: json.loads(
            (REPORT_DIR / "config" / f"{experiment}_selection_dev2k.json").read_text(encoding="utf-8")
        )
        for experiment in ("e1", "e2", "e3", "e4")
    }
    for selection in selections.values():
        assert selection["protocol"] == "dev2k_fpr_constrained_v1"
        constraints = selection["constraints"]
        assert abs(constraints["overall_fpr_max"] - 0.031) < 1e-12
        assert abs(constraints["adversarial_fpr_max"] - 0.04603603603603604) < 1e-12
        assert abs(constraints["precision_min"] - 0.9670742358078602) < 1e-12
        for candidate in selection["eligible"]:
            assert candidate["overall_fpr"] <= constraints["overall_fpr_max"]
            assert candidate["adversarial_fpr"] <= constraints["adversarial_fpr_max"]
            assert candidate["precision"] >= constraints["precision_min"]
    assert selections["e1"]["status"] == selections["e2"]["status"] == "no_eligible_checkpoint"
    assert selections["e3"]["selected"]["path"].replace("\\", "/").endswith(
        "e3_checkpoint-1000_dev_256vt_metrics.json"
    )
    assert selections["e4"]["selected"]["path"].replace("\\", "/").endswith(
        "e4_checkpoint-1000_dev_256vt_metrics.json"
    )

    assert len(cases) == 25 and len(data["errorCases"]) == 25
    assert len({case["imageSource"] for case in cases}) == 25
    for model in MODEL_FILES:
        group = [case for case in cases if case["focusModel"] == model]
        assert len(group) == 5
        types = Counter(case["errorType"] for case in group)
        assert types["FP"] >= 2 and types["FN"] >= 2
        assert {case["split"] for case in group} == {"random", "popular", "adversarial"}
        assert sum(case["sharedFailure"] for case in group) == 1
        for case in group:
            assert case["answers"][model] != case["truth"]
            image = REPORT_DIR / case["image"]
            assert image.is_file() and image.stat().st_size > 0

    prediction_paths = {model: EVAL_DIR / filename for model, filename in MODEL_FILES.items()}
    deep_check = all(path.is_file() for path in prediction_paths.values())
    if deep_check:
        maps = {
            model: {(row["split"], str(row["question_id"])): row for row in read_jsonl(path)}
            for model, path in prediction_paths.items()
        }
        for case in cases:
            key = (case["split"], case["questionId"])
            expected = {name: rows[key]["prediction"] for name, rows in maps.items()}
            assert case["answers"] == expected

    for run in data["devCheckpoints"] + data["repopeRuns"]:
        for split, values in run["metrics"].items():
            assert values["tp"] + values["fp"] + values["tn"] + values["fn"] + values["unknown"] == values["total"], (run["id"], split)
            positives = values["tp"] + values["fn"]
            negatives = values["fp"] + values["tn"]
            expected_recall = values["tp"] / positives if positives else 0
            expected_fpr = values["fp"] / negatives if negatives else 0
            assert abs(values["recall"] - expected_recall) < 1e-12
            assert abs(values["false_positive_rate"] - expected_fpr) < 1e-12

    html = (REPORT_DIR / "QLoRA微调总报告.html").read_text(encoding="utf-8")
    for relative in ("assets/report.css", "assets/report-data.js", "assets/report.js"):
        assert relative in html and (REPORT_DIR / relative).is_file()
    assert not (REPORT_DIR / "metrics" / "qwen2vl2b_baseline_repope_metrics.json").exists()
    mode = "package + frozen predictions" if deep_check else "standalone package"
    print(
        f"PASS ({mode}): fresh baseline, dev protocol, checkpoint rule, "
        "25 cases, metrics, images, and offline assets are consistent."
    )


if __name__ == "__main__":
    main()
