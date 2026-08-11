# 用 QLoRA 缓解 Qwen2-VL 视觉幻觉：RePOPE 实验

本项目研究 Qwen2-VL-2B-Instruct 在物体存在性问答中的漏检与误报，并通过四组受控 QLoRA 实验比较 LoRA 注入范围、难负样本配比和视觉 token 预算对 Recall–FPR trade-off 的影响。

> 最终实验交付：2026-08-12
>
> 基准：POPE / RePOPE
>
> 正式 adapter：E3 checkpoint-1000 @256 visual tokens
>
> 最佳观测配置：E3 checkpoint-1000 @512 visual tokens（post-selection ablation）

## 最终报告

- [完整研究 README](deliverables/qwen2vl-pope-final-report-2026-08-12/README.md)
- [离线交互式报告](deliverables/qwen2vl-pope-final-report-2026-08-12/QLoRA微调总报告.html)
- [25 个错误案例清单](deliverables/qwen2vl-pope-final-report-2026-08-12/error_case_manifest.json)
- [最终交付目录](deliverables/qwen2vl-pope-final-report-2026-08-12/)

GitHub 不直接执行仓库中的本地 HTML。建议从 Releases 下载最终 ZIP，解压后双击 `QLoRA微调总报告.html`。

## 核心结论

1. Attention-only QLoRA 主要改变 yes/no calibration：Recall 明显提高，同时引入更多 false-positive hallucination。
2. Visual Merger LoRA 的边际作用很小；E2 基本复现 E1 的行为。
3. 增加 adversarial negatives 比继续扩大 LoRA target 更有效。E3 相对 E1 降低 overall 与 adversarial FPR。
4. 将 LoRA 扩展到 LLM MLP 没有形成更优的 Recall–FPR frontier。
5. 在冻结的 E3 adapter 上把视觉预算从 256 提升到 512 tokens，同时提高 Recall 并降低 FPR；这是 post-selection ablation，不是新的正式模型选择。
6. Dev 能预测配置的相对保守/激进顺序，但系统性低估 RePOPE 的 absolute FPR，不能替代独立 benchmark。

## Fresh RePOPE baseline

所有主图、排名、错误案例和增量计算只使用最新 fresh inference：

| N | Accuracy | Precision | Recall | F1 | FPR | TP / FP / TN / FN |
|---:|---:|---:|---:|---:|---:|---:|
| 8,185 | 91.8876% | 94.2444% | 86.5216% | 90.2180% | 4.0250% | 3062 / 187 / 4459 / 477 |

旧的“复用 POPE predictions 重算”baseline 已废弃，不进入最终主数据。

### Baseline protocol note

仓库保留两条用途不同的基础模型路径，不能混用：

- `scripts/evaluate_pope.py` 是原始 POPE baseline：只加载 raw 4-bit quantized base，不经过 PEFT 的 k-bit training preparation。
- `scripts/evaluate_qlora.py` 是正式 dev/RePOPE 比较入口：fresh E0 与所有 adapters 都经过相同的 quantized load + PEFT k-bit preparation；E0 仅不挂载 adapter。上表的 fresh baseline 来自这条 pipeline-matched 路径。

这一区分不会改写已冻结结果；它使 baseline 身份和复现入口显式化。所有 E0–E4 主比较必须使用第二条路径，不能把第一条路径或旧的 predictions 重评分结果代入增量计算。

## RePOPE 结果

除 E3@512 外均为 256 visual tokens。E3/E4-2000 是探索性 high-recall points。

| 配置 | Acc | Precision | Recall | F1 | FPR | TP / FP / TN / FN |
|---|---:|---:|---:|---:|---:|---:|
| E0 fresh base | 91.8876 | **94.2444** | 86.5216 | 90.2180 | **4.0250** | 3062 / 187 / 4459 / 477 |
| E1-1000 | 92.6329 | 89.9782 | 93.3597 | 91.6378 | 7.9208 | 3304 / 368 / 4278 / 235 |
| E2-1000 | 92.5596 | 89.9183 | 93.2467 | 91.5522 | 7.9638 | 3300 / 370 / 4276 / 239 |
| **E3-1000 · formal** | 92.6451 | 92.4054 | 90.4210 | 91.4025 | 5.6608 | 3200 / 263 / 4383 / 339 |
| **E3-1000 · 512vt** | **93.3903** | 93.0747 | 91.5230 | **92.2923** | 5.1873 | 3239 / 241 / 4405 / 300 |
| E3-2000 · exploratory | 92.0220 | 88.3369 | **93.9531** | 91.0585 | 9.4490 | 3325 / 439 / 4207 / 214 |
| E4-1000 | 92.3152 | 93.6413 | 88.2170 | 90.8482 | 4.5631 | 3122 / 212 / 4434 / 417 |
| E4-2000 · exploratory | 92.1075 | 88.7698 | 93.5858 | 91.1142 | 9.0185 | 3312 / 419 / 4227 / 227 |

E3@512 相对 fresh base：Accuracy **+1.5027 pp**、Recall **+5.0014 pp**、F1 **+2.0743 pp**、FPR **+1.1623 pp**；总错误从 664 降到 541（−18.52%）。因此它是最佳综合观测点，但不能表述为 hallucination 已低于 fresh base。

## 错误地图

最终网页为 E0–E4 五个统一 256vt 代表模型各展示 5 个 RePOPE 错误，共 25 张照片。每组满足：

- 至少 2 FP、2 FN，并覆盖 random、popular、adversarial；
- 4 个案例至少有另一个模型答对，1 个案例为五模型共同失败；
- 25 个 `image_source` 不重复；
- 卡片同时给出真值、焦点模型答案和 E0–E4 五模型答案条。

这些案例用于诊断错误模式，不是统计抽样，也不重新裁决 RePOPE 标签。

## 实验设计

- 训练来源：COCO train2017，10,000 张图片、16,000 个 yes/no 问题。
- Dev 来源池为 3,000 条；正式 checkpoint selection 固定使用其中 2,000 条分层子集：random yes/no 各 334，popular 与 adversarial yes/no 各 333。
- 防泄漏：训练、dev 与 RePOPE 按 image ID 隔离，零图片交集。
- 量化：4-bit NF4、double quantization、BF16 compute。
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`、学习率 `1e-4`。
- 训练：2 epochs、2,000 optimizer steps、effective batch 16。
- 监督：只监督 assistant 的 `yes/no + EOS`；推理使用 greedy decoding。
- E1：LLM attention q/k/v/o。
- E2：E1 + visual merger MLP 0/2。
- E3：E1 架构 + 更多 adversarial negatives。
- E4：E3 数据 + attention/MLP all-linear LoRA。

Checkpoint 硬约束相对 E0 dev baseline 固定为：overall FPR 不高于 `+1 pp`、adversarial FPR 不高于 `+1 pp`、Precision 不低于 `−1 pp`；通过约束后优先 Recall，差距不足 0.2 pp 时按 F1、FPR、calibration gap 和路径作确定性 tie-break。

## 仓库结构

```text
configs/                    可复现实验配置
scripts/                    数据、训练、评测与 checkpoint 选择
tools/                      检查、smoke test、可视化与导出工具
notebooks/                  无运行输出的实验流水线
deliverables/
  qwen2vl-pope-final-report-2026-08-12/
                             最终网页、指标、案例和研究 README
requirements.txt             核心 Python 依赖
THIRD_PARTY_NOTICES.md       模型、数据与基准来源
```

仓库不包含模型权重、LoRA adapters、optimizer 状态、完整 predictions、COCO 原始数据或 Hugging Face cache。

## 环境与路径

实验环境：Windows、Python 3.12、CUDA 13.2、PyTorch 2.13.0+cu132、Transformers 5.13.1、PEFT 0.20.0、bitsandbytes 0.50.0。

`requirements.txt` 中的 `torch==2.13.0` 锁定 Python API 版本；本实验实际使用的 CUDA wheel 是 `2.13.0+cu132`。请先按本机 CUDA/驱动从对应 PyTorch wheel channel 安装匹配构建，再安装其余依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:QWEN2VL_MODEL_PATH = "D:\models\Qwen2-VL-2B-Instruct"
$env:HF_DATASETS_CACHE = "D:\huggingface\datasets"
```

`QWEN2VL_MODEL_PATH` 可指向本地模型目录；未设置时使用 `Qwen/Qwen2-VL-2B-Instruct`。评测脚本默认 `local_files_only=True`，因此应提前下载模型和数据。

## 复现入口

```powershell
# 数据构建
python scripts/prepare_coco_repope_style.py --download --download-workers 24
python scripts/prepare_dev_2k.py

# 原始 POPE baseline；formal E0/E1–E4 比较请使用 evaluate_qlora.py
python scripts/evaluate_pope.py --run-name qwen2vl2b_baseline

# 工程测速与训练
python scripts/benchmark_qlora_speed.py --phase 1
python scripts/benchmark_qlora_speed.py --phase 2
python scripts/run_qlora.py --experiment e3

# Checkpoint dev 评测
python scripts/evaluate_checkpoints.py `
  --run-dir results/qlora_runs/e3 `
  --run-name e3 `
  --dataset dev `
  --start

# 按预注册的 overall/adversarial FPR 与 Precision 硬约束选择 checkpoint
python scripts/select_checkpoint.py `
  --baseline results/qlora_evaluations/e0_base_2k_dev_256vt_metrics.json `
  --candidates `
    results/qlora_evaluations/e3_checkpoint-500_dev_256vt_metrics.json `
    results/qlora_evaluations/e3_checkpoint-1000_dev_256vt_metrics.json `
    results/qlora_evaluations/e3_checkpoint-1500_dev_256vt_metrics.json `
    results/qlora_evaluations/e3_checkpoint-2000_dev_256vt_metrics.json `
  --output results/qlora_runs/e3/selection_dev2k.json

# 最终报告包完整性验证；公开仓库克隆后即可运行
python deliverables/qwen2vl-pope-final-report-2026-08-12/scripts/validate_report.py
```

若本地保留五份冻结 predictions，验证器会自动追加逐题答案一致性检查。报告数据重建脚本只读取冻结产物和本地数据缓存，不会重新运行模型推理。

## 研究边界

- 可以确认：QLoRA 能有效移动模型 operating point；hard negatives 能缓解 Attention LoRA 引入的误报；E3@512 是已测试配置中 Accuracy/F1 最高的点。
- 不能声称：微调模型 FPR 已低于 fresh base；E3@512 是无 post-selection bias 的正式选择；25 张精选案例代表全部错误分布。
- 后续优先方向：model-driven hard-example mining、多尺度视觉输入、logit-based calibration，而不是继续单纯扩大 LoRA target。

## 权利与引用

本仓库默认不授予代码的开源许可证。Qwen2-VL、COCO、POPE、RePOPE 及报告中的案例图片仍受各自条款约束，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
