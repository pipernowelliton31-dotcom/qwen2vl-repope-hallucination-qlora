"""Build an offline visual error-review page for a POPE evaluation run.

Example:
    python tools/visualize_pope_errors.py --run-name smoke
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import DownloadConfig, load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_CACHE_DIR = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
DATASET_ID = "lmms-lab/POPE"
DATASET_CONFIG = "Full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline HTML viewer for wrong POPE predictions."
    )
    parser.add_argument(
        "--run-name",
        default="smoke",
        help="Evaluation run name, for example smoke.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local Hugging Face datasets cache.",
    )
    parser.add_argument(
        "--examples-per-split",
        type=int,
        default=10,
        help="Number of representative wrong-answer images to show per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic error-example sampling.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    """Keep generated filenames predictable and inside the results directory."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not name:
        raise ValueError("run name must contain at least one letter or number")
    return name


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from error
    return records


def error_kind(record: dict[str, Any]) -> tuple[str, str]:
    truth = str(record["ground_truth"]).lower()
    prediction = str(record["prediction"]).lower()
    if truth == "yes" and prediction != "yes":
        return "fn", "漏检 · False Negative"
    if truth == "no" and prediction == "yes":
        return "fp", "误报 · False Positive"
    return "other", "无法解析或其他错误"


def select_representative_errors(
    errors: list[dict[str, Any]],
    examples_per_split: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically sample a readable cross-section of every split."""
    if examples_per_split < 1:
        raise ValueError("examples_per_split must be at least 1")

    selected: list[dict[str, Any]] = []
    split_order = ("random", "popular", "adversarial")
    for split_offset, split in enumerate(split_order):
        split_errors = [record for record in errors if record["split"] == split]
        sample_size = min(examples_per_split, len(split_errors))
        sampled = random.Random(seed + split_offset).sample(split_errors, sample_size)
        selected.extend(sorted(sampled, key=lambda record: int(record["dataset_index"])))
    return selected


def answer_badge(answer: str, kind: str) -> str:
    answer = html.escape(answer.upper())
    return f'<span class="answer {kind}">{answer}</span>'


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_card(label: str, value: str, detail: str, tone: str = "neutral") -> str:
    return f"""
        <article class="metric {tone}">
          <p>{html.escape(label)}</p>
          <strong>{html.escape(value)}</strong>
          <span>{html.escape(detail)}</span>
        </article>
    """


def split_rows(metrics: dict[str, dict[str, Any]]) -> str:
    rows: list[str] = []
    for split in ("random", "popular", "adversarial"):
        data = metrics.get(split)
        if not data:
            continue
        total = int(data["total"])
        correct = int(data["correct"])
        errors = total - correct
        width = (correct / total * 100) if total else 0
        rows.append(
            f"""
            <div class="split-row">
              <div class="split-name">{html.escape(split.title())}</div>
              <div class="bar" aria-label="{correct} correct out of {total}">
                <span style="width: {width:.2f}%"></span>
              </div>
              <div class="split-score">{correct} / {total}</div>
              <div class="split-errors">{errors} error{'s' if errors != 1 else ''}</div>
            </div>
            """
        )
    return "\n".join(rows)


def error_card(record: dict[str, Any], image_path: str, index: int) -> str:
    kind, label = error_kind(record)
    question = html.escape(str(record["question"]))
    raw_answer = html.escape(str(record["raw_answer"]).strip() or "(empty output)")
    image_source = html.escape(str(record["image_source"]))
    split = html.escape(str(record["split"]))
    elapsed = float(record["elapsed_seconds"])
    return f"""
        <article class="error-card" data-split="{split}" data-kind="{kind}">
          <button class="image-button" data-modal="modal-{index}" aria-label="Open image for: {question}">
            <img src="{html.escape(image_path)}" alt="POPE test image: {question}" loading="lazy">
            <span class="image-action">Click to inspect</span>
          </button>
          <div class="card-body">
            <div class="eyebrow-row">
              <span class="split-tag">{split.title()}</span>
              <span class="error-tag {kind}">{label}</span>
            </div>
            <h2>{question}</h2>
            <div class="answers">
              <div><small>Ground truth</small>{answer_badge(str(record['ground_truth']), 'truth')}</div>
              <div><small>Model output</small>{answer_badge(str(record['prediction']), 'wrong')}</div>
            </div>
            <dl>
              <div><dt>Raw answer</dt><dd>{raw_answer}</dd></div>
              <div><dt>Latency</dt><dd>{elapsed:.2f}s</dd></div>
              <div><dt>Image source</dt><dd>{image_source}</dd></div>
            </dl>
          </div>
        </article>
        <dialog id="modal-{index}">
          <button class="close" aria-label="Close image">×</button>
          <img src="{html.escape(image_path)}" alt="POPE test image: {question}">
          <div class="dialog-copy">
            <p class="dialog-label">{split.title()} · {label}</p>
            <h2>{question}</h2>
            <p>Ground truth: {answer_badge(str(record['ground_truth']), 'truth')}
              &nbsp; Model: {answer_badge(str(record['prediction']), 'wrong')}</p>
          </div>
        </dialog>
    """


def gallery_sections(
    display_records: list[dict[str, Any]],
    image_paths: list[str],
    all_split_counts: Counter[str],
) -> str:
    cards_by_split: dict[str, list[str]] = {}
    for index, (record, image_path) in enumerate(zip(display_records, image_paths), 1):
        cards_by_split.setdefault(str(record["split"]), []).append(
            error_card(record, image_path, index)
        )

    sections: list[str] = []
    for split in ("random", "popular", "adversarial"):
        cards = cards_by_split.get(split)
        if not cards:
            continue
        sections.append(
            f"""
            <section class="gallery-group" data-split="{split}">
              <div class="gallery-group-heading">
                <h3>{html.escape(split.title())}</h3>
                <p>{len(cards)} representative errors shown / {all_split_counts[split]} total errors</p>
              </div>
              <div class="error-grid">{"".join(cards)}</div>
            </section>
            """
        )
    return "\n".join(sections)


def page_html(
    run_name: str,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    display_records: list[dict[str, Any]],
    image_paths: list[str],
    benchmark_label: str = "POPE",
) -> str:
    metrics = metadata["metrics"]
    overall = metrics["overall"]
    errors = [record for record in records if not record.get("correct", False)]
    kind_counts = Counter(error_kind(record)[0] for record in errors)
    split_counts = Counter(str(record["split"]) for record in errors)
    galleries = gallery_sections(display_records, image_paths, split_counts)

    metrics_cards = "\n".join(
        [
            metric_card(
                "Accuracy",
                percentage(float(overall["accuracy"])),
                f"{overall['correct']} correct / {overall['total']} samples",
            ),
            metric_card(
                "Recall",
                percentage(float(overall["recall"])),
                f"{overall['tp']} detected / {overall['tp'] + overall['fn']} positive samples",
                "warning",
            ),
            metric_card(
                "Error pattern",
                f"{kind_counts['fn']} FN · {kind_counts['fp']} FP",
                "FN = object present but model answered no",
                "danger" if kind_counts["fn"] else "neutral",
            ),
        ]
    )

    split_buttons = "".join(
        f'<button class="filter" data-filter="{split}">{split.title()} ({sum(1 for record in display_records if record["split"] == split)} shown)</button>'
        for split in ("random", "popular", "adversarial")
        if split_counts[split]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(benchmark_label)} · {html.escape(run_name)} error review</title>
  <link rel="icon" href="data:,">
  <style>
    :root {{
      --ink: #17201f;
      --muted: #61706c;
      --canvas: #f3f1ec;
      --paper: #fffdfa;
      --line: #d9ded9;
      --green: #176a50;
      --green-pale: #ddf0e5;
      --red: #b33c32;
      --red-pale: #f8e3df;
      --amber: #9a611b;
      --amber-pale: #f8edcf;
      --shadow: 0 10px 25px rgba(26, 36, 33, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--canvas); font-family: Georgia, 'Noto Serif SC', serif; }}
    button {{ font: inherit; }}
    .shell {{ width: min(1400px, calc(100% - 48px)); margin: 0 auto; padding: 48px 0 72px; }}
    .masthead {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 32px; border-bottom: 1px solid var(--ink); padding-bottom: 28px; }}
    .kicker, .section-label, .eyebrow-row, .metric p, .metric span, .split-row, .controls, small, dt, .dialog-label {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .04em; }}
    .kicker {{ margin: 0 0 10px; color: var(--green); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    h1 {{ margin: 0; max-width: 760px; font-size: clamp(34px, 5vw, 68px); line-height: .96; letter-spacing: -.045em; }}
    .subtitle {{ max-width: 730px; margin: 18px 0 0; color: var(--muted); font-size: 17px; line-height: 1.55; }}
    .run-note {{ min-width: 230px; padding-top: 4px; color: var(--muted); font-size: 13px; line-height: 1.6; text-align: right; }}
    .run-note strong {{ display: block; color: var(--ink); font-size: 15px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; margin: 30px 0 48px; background: var(--line); border: 1px solid var(--line); }}
    .metric {{ min-height: 150px; padding: 22px; background: var(--paper); }}
    .metric p {{ margin: 0; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .metric strong {{ display: block; margin: 15px 0 8px; font-size: 34px; letter-spacing: -.04em; }}
    .metric span {{ color: var(--muted); font-size: 11px; line-height: 1.5; }}
    .metric.warning strong {{ color: var(--amber); }} .metric.danger strong {{ color: var(--red); }}
    .overview {{ display: grid; grid-template-columns: 1.45fr 1fr; gap: 48px; margin-bottom: 52px; }}
    .section-label {{ margin: 0 0 18px; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .split-table {{ border-top: 1px solid var(--line); }}
    .split-row {{ display: grid; grid-template-columns: 115px 1fr 64px 80px; align-items: center; gap: 14px; padding: 15px 0; border-bottom: 1px solid var(--line); font-size: 12px; }}
    .bar {{ height: 10px; overflow: hidden; background: #e5e8e3; }} .bar span {{ display: block; height: 100%; background: var(--green); }}
    .split-score {{ text-align: right; font-weight: 700; }} .split-errors {{ color: var(--red); text-align: right; }}
    .finding {{ align-self: end; padding: 26px; border-left: 4px solid var(--red); background: var(--paper); box-shadow: var(--shadow); }}
    .finding h2 {{ margin: 0 0 10px; font-size: 25px; line-height: 1.15; letter-spacing: -.025em; }}
    .finding p {{ margin: 0; color: var(--muted); font-size: 15px; line-height: 1.55; }}
    .review-heading {{ display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 20px; }}
    .review-heading h2 {{ margin: 0; font-size: 30px; letter-spacing: -.03em; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .filter {{ padding: 8px 10px; border: 1px solid var(--line); background: transparent; color: var(--muted); font-size: 11px; cursor: pointer; }}
    .filter:hover, .filter.active {{ border-color: var(--ink); color: var(--paper); background: var(--ink); }}
    .gallery-group {{ margin-top: 34px; }}
    .gallery-group-heading {{ display: flex; align-items: baseline; justify-content: space-between; gap: 20px; padding: 0 0 12px; border-bottom: 1px solid var(--ink); }}
    .gallery-group-heading h3 {{ margin: 0; font-size: 24px; letter-spacing: -.025em; }}
    .gallery-group-heading p {{ margin: 0; color: var(--muted); font: 11px ui-monospace, monospace; text-align: right; }}
    .error-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; margin-top: 18px; }}
    .error-card {{ overflow: hidden; background: var(--paper); border: 1px solid var(--line); box-shadow: var(--shadow); }}
    .image-button {{ position: relative; display: block; width: 100%; padding: 0; overflow: hidden; border: 0; background: #deded9; cursor: zoom-in; aspect-ratio: 4 / 3; }}
    .image-button img {{ display: block; width: 100%; height: 100%; object-fit: cover; transition: transform .25s ease; }}
    .image-button:hover img {{ transform: scale(1.03); }}
    .image-action {{ position: absolute; right: 10px; bottom: 10px; padding: 5px 8px; background: rgba(23, 32, 31, .82); color: white; font: 10px ui-monospace, monospace; }}
    .card-body {{ padding: 18px; }}
    .eyebrow-row {{ display: flex; flex-wrap: wrap; gap: 7px; font-size: 10px; text-transform: uppercase; }}
    .split-tag {{ color: var(--muted); }} .error-tag {{ padding: 3px 5px; }} .error-tag.fn {{ color: var(--red); background: var(--red-pale); }} .error-tag.fp {{ color: var(--amber); background: var(--amber-pale); }}
    .error-card h2 {{ min-height: 72px; margin: 13px 0 15px; font-size: 20px; line-height: 1.2; letter-spacing: -.02em; }}
    .answers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }}
    small {{ display: block; margin-bottom: 5px; color: var(--muted); font-size: 10px; text-transform: uppercase; }}
    .answer {{ display: inline-block; padding: 5px 8px; font: 700 12px ui-monospace, monospace; }} .answer.truth {{ color: var(--green); background: var(--green-pale); }} .answer.wrong {{ color: var(--red); background: var(--red-pale); }}
    dl {{ margin: 0; padding-top: 12px; border-top: 1px solid var(--line); }}
    dl div {{ display: grid; grid-template-columns: 94px 1fr; gap: 8px; padding: 4px 0; }} dt {{ color: var(--muted); font-size: 10px; }} dd {{ overflow-wrap: anywhere; margin: 0; font: 11px/1.35 ui-monospace, monospace; }}
    .empty {{ padding: 30px; background: var(--paper); border: 1px solid var(--line); color: var(--muted); }}
    dialog {{ width: min(1000px, calc(100% - 32px)); padding: 0; overflow: visible; border: 0; background: var(--paper); box-shadow: 0 30px 80px rgba(0, 0, 0, .38); }}
    dialog::backdrop {{ background: rgba(18, 25, 23, .72); }} dialog img {{ display: block; width: 100%; max-height: 70vh; object-fit: contain; background: #181d1b; }}
    .dialog-copy {{ padding: 20px 24px 24px; }} .dialog-copy h2 {{ margin: 5px 0 10px; font-size: 25px; }} .dialog-copy p {{ margin: 0; color: var(--muted); }} .dialog-label {{ color: var(--red) !important; font-size: 11px; text-transform: uppercase; }}
    .close {{ position: absolute; top: -14px; right: -14px; width: 36px; height: 36px; border: 0; border-radius: 50%; background: var(--ink); color: white; font-size: 25px; line-height: 1; cursor: pointer; }}
    @media (max-width: 900px) {{ .masthead, .overview {{ display: block; }} .run-note {{ margin-top: 22px; text-align: left; }} .metrics {{ grid-template-columns: 1fr; }} .finding {{ margin-top: 30px; }} .error-grid {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 620px) {{ .shell {{ width: min(100% - 28px, 1400px); padding-top: 28px; }} .error-grid {{ grid-template-columns: 1fr; }} .review-heading {{ display: block; }} .controls {{ margin-top: 16px; }} .split-row {{ grid-template-columns: 78px 1fr 48px; }} .split-errors {{ display: none; }} .gallery-group-heading {{ display: block; }} .gallery-group-heading p {{ margin-top: 6px; text-align: left; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="masthead">
      <div>
        <p class="kicker">{html.escape(benchmark_label)} · visual failure analysis</p>
        <h1>What the model missed.</h1>
        <p class="subtitle">A focused review of wrong answers from the <strong>{html.escape(run_name)}</strong> run. Images, questions and predictions stay together so that aggregate metrics lead directly to evidence.</p>
      </div>
      <div class="run-note"><strong>{html.escape(str(metadata.get('model_path', 'Model')).split('\\')[-1])}</strong>4-bit NF4 · {html.escape(str(metadata.get('compute_dtype', 'unknown dtype')))}<br>{int(overall['total'])} evaluated samples</div>
    </header>
    <section class="metrics" aria-label="Overall metrics">{metrics_cards}</section>
    <section class="overview">
      <div>
        <p class="section-label">Performance by {html.escape(benchmark_label)} split</p>
        <div class="split-table">{split_rows(metrics)}</div>
      </div>
      <aside class="finding">
        <p class="section-label">Observed pattern</p>
        <h2>{kind_counts['fn']} of {len(errors)} errors are misses.</h2>
        <p>False negatives mean an object was present but the model answered <strong>no</strong>; false positives mean it answered <strong>yes</strong> when absent. The gallery below samples both patterns from each split for visual review.</p>
      </aside>
    </section>
    <section>
      <div class="review-heading">
        <div><p class="section-label">Evidence</p><h2>Representative wrong-answer gallery</h2></div>
        <nav class="controls" aria-label="Filter error cards">
          <button class="filter active" data-filter="all">All ({len(display_records)} shown)</button>
          {split_buttons}
          <button class="filter" data-filter="fn">False negatives ({kind_counts['fn']} total)</button>
          <button class="filter" data-filter="fp">False positives ({kind_counts['fp']} total)</button>
        </nav>
      </div>
      <div id="error-gallery">{galleries or '<p class="empty">No wrong answers were found in this run.</p>'}</div>
    </section>
  </main>
  <script>
    const filters = document.querySelectorAll('.filter');
    const cards = document.querySelectorAll('.error-card');
    const groups = document.querySelectorAll('.gallery-group');
    filters.forEach((button) => button.addEventListener('click', () => {{
      const filter = button.dataset.filter;
      filters.forEach((item) => item.classList.toggle('active', item === button));
      cards.forEach((card) => {{
        const visible = filter === 'all' || card.dataset.split === filter || card.dataset.kind === filter;
        card.hidden = !visible;
      }});
      groups.forEach((group) => {{
        group.hidden = !Array.from(group.querySelectorAll('.error-card')).some((card) => !card.hidden);
      }});
    }}));
    document.querySelectorAll('[data-modal]').forEach((button) => button.addEventListener('click', () => {{
      document.getElementById(button.dataset.modal).showModal();
    }}));
    document.querySelectorAll('dialog').forEach((dialog) => {{
      dialog.querySelector('.close').addEventListener('click', () => dialog.close());
      dialog.addEventListener('click', (event) => {{ if (event.target === dialog) dialog.close(); }});
    }});
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    run_name = safe_name(args.run_name)
    predictions_path = RESULTS_DIR / f"{run_name}_predictions.jsonl"
    metrics_path = RESULTS_DIR / f"{run_name}_metrics.json"
    if not predictions_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "Expected evaluation results were not found:\n"
            f"  {predictions_path}\n  {metrics_path}"
        )

    records = load_jsonl(predictions_path)
    errors = [record for record in records if not record.get("correct", False)]
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    display_records = select_representative_errors(
        errors,
        args.examples_per_split,
        args.seed,
    )

    assets_dir = RESULTS_DIR / f"{run_name}_review_assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)

    image_paths: list[str] = []
    if display_records:
        print("Loading POPE from the local cache to export error images …")
        pope = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            cache_dir=str(args.cache_dir),
            download_config=DownloadConfig(local_files_only=True),
        )
        for index, record in enumerate(display_records, 1):
            split = str(record["split"])
            dataset_index = int(record["dataset_index"])
            image = pope[split][dataset_index]["image"].convert("RGB")
            image_name = f"{index:02d}_{split}_{dataset_index}.jpg"
            image.save(assets_dir / image_name, quality=92)
            image_paths.append(f"{assets_dir.name}/{image_name}")

    output_path = RESULTS_DIR / f"{run_name}_review.html"
    output_path.write_text(
        page_html(run_name, metadata, records, display_records, image_paths),
        encoding="utf-8",
    )
    print(f"Exported {len(display_records)} representative wrong-answer images to: {assets_dir}")
    print(f"Offline review page: {output_path}")


if __name__ == "__main__":
    main()
