"""Create an offline audit gallery for 256↔512 RePOPE positive transitions.

No inference is performed. The page shows every positive sample whose answer
changes between the 256- and 512-visual-token runs, together with both POPE and
RePOPE labels, so an annotator can independently inspect the image.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import DownloadConfig, load_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from tools.visualize_pope_errors import DATASET_CONFIG, DATASET_ID, DEFAULT_CACHE_DIR, RESULTS_DIR, load_jsonl
from tools.visualize_repope_errors import DEFAULT_ANNOTATIONS, DEFAULT_PREDICTIONS, relabel_records


DEFAULT_512_PREDICTIONS = RESULTS_DIR / "qwen2vl2b_repope_random_positive_512_predictions.jsonl"
SPLIT_ORDER = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image audit page for RePOPE 256-to-512 prediction changes."
    )
    parser.add_argument("--predictions-512", type=Path, default=DEFAULT_512_PREDICTIONS)
    parser.add_argument("--baseline-predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-name", default="qwen2vl2b_repope_random_256_vs_512_audit")
    return parser.parse_args()


def safe_name(value: str) -> str:
    name = "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)
    if not name.strip("._"):
        raise ValueError("run name must contain a letter or number")
    return name


def transition_label(transition: str) -> tuple[str, str]:
    if transition == "fn_to_tp":
        return "improved", "256 错 → 512 对"
    if transition == "tp_to_fn":
        return "regressed", "256 对 → 512 错"
    raise ValueError(f"Unexpected transition: {transition}")


def label_badge(value: str, style: str) -> str:
    return f'<span class="badge {style}">{html.escape(value.upper())}</span>'


def card(record: dict[str, Any], image_path: str, index: int) -> str:
    transition, transition_text = transition_label(str(record["transition"]))
    original_label = str(record["original_pope_label"])
    repope_label = str(record["repope_label"])
    label_note = "标签已由 RePOPE 修正" if record["label_changed"] else "原 POPE 与 RePOPE 标签一致"
    return f"""
      <article class="case {transition}" data-transition="{transition}" data-label-changed="{str(record['label_changed']).lower()}">
        <button class="image-button" data-modal="modal-{index}" aria-label="放大查看图片">
          <img src="{html.escape(image_path)}" alt="COCO 图像：{html.escape(str(record['question']))}" loading="lazy">
          <span>点击放大</span>
        </button>
        <div class="case-body">
          <div class="eyebrow"><span>{html.escape(str(record['split']).upper())}</span><b>{transition_text}</b></div>
          <h2>{html.escape(str(record['question']))}</h2>
          <div class="verdicts">
            <div><small>RePOPE 真值</small>{label_badge(repope_label, 'truth')}</div>
            <div><small>原 POPE 标签</small>{label_badge(original_label, 'old')}</div>
            <div><small>256 token</small>{label_badge(str(record['baseline_prediction_256']), 'before')}</div>
            <div><small>512 token</small>{label_badge(str(record['prediction_512']), 'after')}</div>
          </div>
          <p class="label-note">{html.escape(label_note)}</p>
          <dl>
            <div><dt>256 原始输出</dt><dd>{html.escape(str(record['baseline_raw_answer']))}</dd></div>
            <div><dt>512 原始输出</dt><dd>{html.escape(str(record['raw_answer_512']))}</dd></div>
            <div><dt>COCO 图像</dt><dd>{html.escape(str(record['image_source']))}</dd></div>
            <div><dt>题号</dt><dd>{html.escape(str(record['question_id']))}</dd></div>
          </dl>
        </div>
      </article>
      <dialog id="modal-{index}">
        <button class="close" aria-label="关闭">×</button>
        <img src="{html.escape(image_path)}" alt="COCO 图像">
        <div class="dialog-copy"><p>{transition_text} · {html.escape(str(record['split']))} · QID {html.escape(str(record['question_id']))}</p><h2>{html.escape(str(record['question']))}</h2><p>RePOPE: {label_badge(repope_label, 'truth')} &nbsp; 256: {label_badge(str(record['baseline_prediction_256']), 'before')} &nbsp; 512: {label_badge(str(record['prediction_512']), 'after')}</p></div>
      </dialog>
    """


def gallery(records: list[dict[str, Any]], image_paths: list[str], transition: str, index_start: int) -> str:
    title = "256 错 → 512 对" if transition == "fn_to_tp" else "256 对 → 512 错"
    description = "512 增加视觉 token 后检出了正样本。" if transition == "fn_to_tp" else "512 增加视觉 token 后反而漏检了正样本。"
    cards = [card(record, image_path, index) for index, (record, image_path) in enumerate(zip(records, image_paths), index_start) if record["transition"] == transition]
    return f'<section class="group" data-group="{transition}"><div class="group-head"><h2>{title}</h2><p>{len(cards)} 条 · {description}</p></div><div class="grid">{"".join(cards)}</div></section>'


def page_html(records: list[dict[str, Any]], image_paths: list[str]) -> str:
    counts = Counter(str(record["transition"]) for record in records)
    changed_counts = Counter(str(record["transition"]) for record in records if record["label_changed"])
    improvements = [record for record in records if record["transition"] == "fn_to_tp"]
    regressions = [record for record in records if record["transition"] == "tp_to_fn"]
    improvement_paths = [path for record, path in zip(records, image_paths) if record["transition"] == "fn_to_tp"]
    regression_paths = [path for record, path in zip(records, image_paths) if record["transition"] == "tp_to_fn"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RePOPE · 256 vs 512 人工审计</title><link rel="icon" href="data:,">
<style>
:root {{ --ink:#15201e; --muted:#64706c; --paper:#fffdf8; --canvas:#f1f0eb; --line:#d7dcd7; --good:#17694f; --good-bg:#dbefe3; --bad:#ae3d32; --bad-bg:#f7e2de; --old:#675f4d; --old-bg:#ede9dd; --amber:#8b5b19; --amber-bg:#f8ebca; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--canvas); color:var(--ink); font-family:Georgia,'Noto Serif SC',serif; }} button {{ font:inherit; }} .shell {{ width:min(1500px,calc(100% - 48px)); margin:auto; padding:46px 0 72px; }} .masthead {{ border-bottom:1px solid var(--ink); padding-bottom:28px; display:flex; justify-content:space-between; gap:30px; }} .kicker,.eyebrow,small,dt,.meta,.controls {{ font:11px ui-monospace,Consolas,monospace; letter-spacing:.045em; }} .kicker {{ color:var(--good); font-weight:bold; margin:0 0 10px; text-transform:uppercase; }} h1 {{ margin:0; max-width:850px; font-size:clamp(38px,5vw,72px); line-height:.96; letter-spacing:-.05em; }} .subtitle {{ max-width:780px; margin:18px 0 0; color:var(--muted); font-size:17px; line-height:1.6; }} .meta {{ min-width:210px; color:var(--muted); text-align:right; line-height:1.6; }} .meta strong {{ display:block; color:var(--ink); font-size:15px; }} .stats {{ margin:30px 0 48px; display:grid; grid-template-columns:repeat(3,1fr); border:1px solid var(--line); gap:1px; background:var(--line); }} .stat {{ padding:21px; min-height:135px; background:var(--paper); }} .stat p {{ margin:0; color:var(--muted); font:11px ui-monospace,monospace; text-transform:uppercase; }} .stat strong {{ display:block; margin:15px 0 7px; font-size:38px; letter-spacing:-.05em; }} .stat span {{ color:var(--muted); font:12px/1.5 ui-monospace,monospace; }} .stat.good strong {{ color:var(--good); }} .stat.bad strong {{ color:var(--bad); }} .note {{ margin:0 0 44px; padding:21px 24px; border-left:4px solid var(--amber); background:var(--paper); color:var(--muted); font-size:15px; line-height:1.6; }} .note strong {{ color:var(--ink); }} .controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:0 0 22px; }} .filter {{ padding:8px 11px; border:1px solid var(--line); color:var(--muted); background:transparent; cursor:pointer; }} .filter.active,.filter:hover {{ background:var(--ink); color:white; border-color:var(--ink); }} .group {{ margin:40px 0 58px; }} .group-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:20px; padding-bottom:13px; border-bottom:1px solid var(--ink); }} .group-head h2 {{ margin:0; font-size:28px; letter-spacing:-.03em; }} .group-head p {{ margin:0; color:var(--muted); font:11px ui-monospace,monospace; text-align:right; }} .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; margin-top:18px; }} .case {{ overflow:hidden; border:1px solid var(--line); background:var(--paper); box-shadow:0 8px 20px rgba(26,36,33,.07); }} .image-button {{ position:relative; display:block; width:100%; height:auto; padding:0; border:0; overflow:hidden; background:#ddd; cursor:zoom-in; aspect-ratio:4/3; }} .image-button img {{ display:block; width:100%; height:100%; object-fit:cover; transition:transform .2s; }} .image-button:hover img {{ transform:scale(1.025); }} .image-button span {{ position:absolute; right:10px; bottom:10px; padding:5px 8px; background:#15201ed9; color:white; font:10px ui-monospace,monospace; }} .case-body {{ padding:18px; }} .eyebrow {{ display:flex; justify-content:space-between; gap:8px; color:var(--muted); }} .eyebrow b {{ padding:3px 5px; font-weight:700; }} .improved .eyebrow b {{ color:var(--good); background:var(--good-bg); }} .regressed .eyebrow b {{ color:var(--bad); background:var(--bad-bg); }} .case h2 {{ min-height:71px; margin:14px 0 15px; font-size:20px; line-height:1.2; letter-spacing:-.02em; }} .verdicts {{ display:grid; grid-template-columns:1fr 1fr; gap:11px; }} small {{ display:block; margin-bottom:5px; color:var(--muted); text-transform:uppercase; }} .badge {{ display:inline-block; padding:5px 7px; font:700 12px ui-monospace,monospace; }} .truth,.after {{ color:var(--good); background:var(--good-bg); }} .before {{ color:var(--bad); background:var(--bad-bg); }} .old {{ color:var(--old); background:var(--old-bg); }} .label-note {{ margin:14px 0 0; padding:8px 10px; color:var(--amber); background:var(--amber-bg); font:11px/1.45 ui-monospace,monospace; }} dl {{ margin:14px 0 0; padding-top:11px; border-top:1px solid var(--line); }} dl div {{ display:grid; grid-template-columns:82px 1fr; gap:7px; padding:3px 0; }} dt {{ color:var(--muted); }} dd {{ overflow-wrap:anywhere; margin:0; font:11px/1.35 ui-monospace,monospace; }} dialog {{ width:min(1080px,calc(100% - 32px)); padding:0; overflow:visible; border:0; background:var(--paper); box-shadow:0 30px 90px #0008; }} dialog::backdrop {{ background:#101714c9; }} dialog img {{ display:block; max-height:70vh; width:100%; object-fit:contain; background:#101714; }} .dialog-copy {{ padding:20px 24px 25px; }} .dialog-copy p {{ color:var(--muted); }} .dialog-copy h2 {{ margin:8px 0; font-size:27px; }} .close {{ position:absolute; top:-15px; right:-15px; width:38px; height:38px; border:0; border-radius:50%; color:white; background:var(--ink); font-size:26px; cursor:pointer; }} @media(max-width:900px) {{ .masthead {{ display:block; }} .meta {{ margin-top:22px; text-align:left; }} .stats {{ grid-template-columns:1fr; }} .grid {{ grid-template-columns:repeat(2,1fr); }} }} @media(max-width:600px) {{ .shell {{ width:min(100% - 28px,1500px); padding-top:28px; }} .grid {{ grid-template-columns:1fr; }} .group-head {{ display:block; }} .group-head p {{ margin-top:6px; text-align:left; }} }}
</style></head><body><main class="shell"><header class="masthead"><div><p class="kicker">RePOPE · Paired resolution audit</p><h1>512 视觉 token 改变了哪些判断？</h1><p class="subtitle">这 58 条均为 RePOPE 保留的正样本，且模型在 256 与 512 token 设置间改变了答案。逐条看图与标签，判断变化是否有视觉依据，或仍存在标签与语义歧义。</p></div><div class="meta"><strong>Qwen2-VL-2B-Instruct</strong>256 → 512 visual tokens<br>random · positive-only</div></header><section class="stats"><article class="stat good"><p>256 错 → 512 对</p><strong>{counts['fn_to_tp']}</strong><span>其中 {changed_counts['fn_to_tp']} 条曾被 RePOPE 修正标签</span></article><article class="stat bad"><p>256 对 → 512 错</p><strong>{counts['tp_to_fn']}</strong><span>其中 {changed_counts['tp_to_fn']} 条曾被 RePOPE 修正标签</span></article><article class="stat"><p>净变化</p><strong>+{counts['fn_to_tp'] - counts['tp_to_fn']}</strong><span>同一批 1,159 个正样本中的净检出增量</span></article></section><p class="note"><strong>人工审核提示：</strong>先看图中目标是否确实可辨，再比对 RePOPE 与原 POPE 标签。只有 <em>RePOPE 真值</em> 用于本次指标；“原 POPE 标签”仅用于追踪是否发生过重标。页面中的“对/错”均以 RePOPE 真值为准。</p><nav class="controls"><button class="filter active" data-filter="all">全部 58 条</button><button class="filter" data-filter="improved">仅 256 错 → 512 对 ({counts['fn_to_tp']})</button><button class="filter" data-filter="regressed">仅 256 对 → 512 错 ({counts['tp_to_fn']})</button><button class="filter" data-filter="changed">仅标签被修正</button></nav>{gallery(improvements, improvement_paths, 'fn_to_tp', 1)}{gallery(regressions, regression_paths, 'tp_to_fn', len(improvements) + 1)}</main><script>const buttons=document.querySelectorAll('.filter'),cards=document.querySelectorAll('.case'),groups=document.querySelectorAll('.group');buttons.forEach(button=>button.addEventListener('click',()=>{{const f=button.dataset.filter;buttons.forEach(x=>x.classList.toggle('active',x===button));cards.forEach(card=>{{card.hidden=!(f==='all'||card.dataset.transition===f||(f==='changed'&&card.dataset.labelChanged==='true'));}});groups.forEach(group=>group.hidden=!Array.from(group.querySelectorAll('.case')).some(card=>!card.hidden));}}));document.querySelectorAll('[data-modal]').forEach(button=>button.addEventListener('click',()=>document.getElementById(button.dataset.modal).showModal()));document.querySelectorAll('dialog').forEach(dialog=>{{dialog.querySelector('.close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{{if(event.target===dialog)dialog.close();}});}});</script></body></html>"""


def main() -> None:
    args = parse_args()
    run_name = safe_name(args.run_name)
    changed_records = load_jsonl(args.predictions_512)
    baseline_records = relabel_records(load_jsonl(args.baseline_predictions), args.annotations_dir)
    baseline_by_key = {
        (str(record["split"]), str(record["question_id"])): record
        for record in baseline_records
    }
    audit_records: list[dict[str, Any]] = []
    for record in changed_records:
        transition = str(record["transition"])
        if transition not in {"fn_to_tp", "tp_to_fn"}:
            continue
        key = (str(record["split"]), str(record["question_id"]))
        baseline = baseline_by_key.get(key)
        if baseline is None:
            raise ValueError(f"Missing RePOPE baseline item: {key}")
        merged = dict(record)
        merged.update({
            "original_pope_label": baseline["original_pope_label"],
            "repope_label": baseline["repope_label"],
            "label_changed": baseline["label_changed"],
            "baseline_raw_answer": baseline["raw_answer"],
        })
        audit_records.append(merged)
    audit_records.sort(key=lambda record: (0 if record["transition"] == "fn_to_tp" else 1, int(record["question_id"])))
    if not audit_records:
        raise RuntimeError("No 256↔512 prediction changes were found.")

    assets_dir = RESULTS_DIR / f"{run_name}_assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)
    pope = load_dataset(DATASET_ID, DATASET_CONFIG, cache_dir=str(args.cache_dir), download_config=DownloadConfig(local_files_only=True))
    image_paths: list[str] = []
    for index, record in enumerate(audit_records, 1):
        split = str(record["split"])
        sample = pope[split][int(record["dataset_index"])]
        image = sample["image"].convert("RGB")
        filename = f"{index:02d}_{record['transition']}_{split}_qid_{record['question_id']}.jpg"
        image.save(assets_dir / filename, quality=92)
        image_paths.append(f"{assets_dir.name}/{filename}")

    output_path = RESULTS_DIR / f"{run_name}.html"
    output_path.write_text(page_html(audit_records, image_paths), encoding="utf-8")
    audit_path = RESULTS_DIR / f"{run_name}_records.jsonl"
    audit_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in audit_records), encoding="utf-8")
    print(f"Exported {len(audit_records)} paired transitions to: {output_path}")
    print(f"Images: {assets_dir}")
    print(f"Audit records: {audit_path}")


if __name__ == "__main__":
    main()
