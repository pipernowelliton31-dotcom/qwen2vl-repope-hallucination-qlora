# 用 QLoRA 缓解 Qwen2-VL 视觉幻觉：RePOPE 实验

> 最终实验交付｜2026-08-12
>
> 网页入口：双击 `QLoRA微调总报告.html`
>
> 研究问题：如何在提高真实物体识别 Recall 的同时，控制 false-positive object hallucination？

## 1. 最终结论

本项目对 Qwen2-VL-2B-Instruct 进行了四组 QLoRA 实验，并在固定的无泄漏 COCO-derived dev 与独立 RePOPE benchmark 上比较 Recall–FPR trade-off。

核心结论不是“微调全面提升了模型”，而是：

1. **Attention-only QLoRA 主要改变模型的 Yes/No calibration。** E1-1000 将 RePOPE Recall 提升到 93.36%，但 FPR 同时升到 7.92%，说明减少漏检伴随了额外误报。
2. **增加 Visual Merger LoRA 的边际作用很小。** E2 几乎复制 E1 的 dev 与 RePOPE 结果。
3. **针对错误模式调整训练数据，比继续扩大 LoRA target 更有效。** E3 增加 adversarial negatives 后，相对 E1 将 FPR 从 7.92% 降到 5.66%，adversarial FPR 从 12.09% 降到 8.62%。
4. **把 LoRA 扩展到 LLM MLP 没有形成更优的 Recall–FPR frontier。** E4-1000 更保守，继续训练恢复 Recall 时 FPR 又明显升高。
5. **增加视觉预算第一次在同一 adapter 上同时改善 Recall 和 FPR。** E3-1000 从 256 提升到 512 visual tokens 后，Recall +1.102 pp、FPR −0.474 pp、TP +39、FP −22，总错误减少 61 个。
6. **Dev 能预测相对 operating-point 顺序，但系统性低估 RePOPE 的 absolute FPR。** 因此 dev 适合 checkpoint selection，不能替代独立 benchmark。

正式 adapter 与最佳观测配置必须分开表述：

- **Formal adapter：** `E3 checkpoint-1000`，由预设的 dev-only、FPR-constrained selection protocol 选出。
- **Best observed configuration：** `E3 checkpoint-1000 @512 visual tokens`，属于模型选择完成后的 visual-token ablation。

## 2. Fresh RePOPE baseline

本报告所有 RePOPE baseline 比较均使用最新的 fresh inference：

```text
results/qlora_evaluations/e0_fresh_base_repope_256vt_metrics.json
```

| N | Accuracy | Precision | Recall | F1 | FPR | Adv. FPR | Yes Ratio | TP / FP / TN / FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,185 | 91.8876% | 94.2444% | 86.5216% | 90.2180% | 4.0250% | 5.9452% | 39.6946% | 3062 / 187 / 4459 / 477 |

旧阶段报告使用的 `qwen2vl2b_baseline_repope_metrics.json` 是对已保存 POPE predictions 的重新对齐计分，不是当前 RePOPE 文本上的 fresh inference。该口径已被替换，不参与本报告任何主图、排名、案例或增量计算。

## 3. RePOPE 完整结果

下表除 E3@512 外均使用 256 visual tokens。E3/E4-2000 是在正式 checkpoint 之外补充的 high-recall exploratory points。

| 配置 | Acc | Precision | Recall | F1 | FPR | Adv. FPR | Yes Ratio | TP / FP / TN / FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0 fresh base** | 91.8876 | **94.2444** | 86.5216 | 90.2180 | **4.0250** | **5.9452** | 39.6946 | 3062 / 187 / 4459 / 477 |
| E1-1000 | 92.6329 | 89.9782 | 93.3597 | 91.6378 | 7.9208 | 12.0908 | 44.8626 | 3304 / 368 / 4278 / 235 |
| E2-1000 | 92.5596 | 89.9183 | 93.2467 | 91.5522 | 7.9638 | 12.3580 | 44.8381 | 3300 / 370 / 4276 / 239 |
| **E3-1000 · formal** | 92.6451 | 92.4054 | 90.4210 | 91.4025 | 5.6608 | 8.6172 | 42.3091 | 3200 / 263 / 4383 / 339 |
| **E3-1000 · 512vt** | **93.3903** | 93.0747 | 91.5230 | **92.2923** | 5.1873 | 7.5484 | 42.5168 | 3239 / 241 / 4405 / 300 |
| E3-2000 · exploratory | 92.0220 | 88.3369 | **93.9531** | 91.0585 | 9.4490 | 14.8297 | 45.9866 | 3325 / 439 / 4207 / 214 |
| E4-1000 | 92.3152 | 93.6413 | 88.2170 | 90.8482 | 4.5631 | 6.8804 | 40.7330 | 3122 / 212 / 4434 / 417 |
| E4-2000 · exploratory | 92.1075 | 88.7698 | 93.5858 | 91.1142 | 9.0185 | 14.0949 | 45.5834 | 3312 / 419 / 4227 / 227 |

### E3@512 相对 fresh base

| 指标 | Fresh E0 · 256vt | E3-1000 · 512vt | 变化 |
|---|---:|---:|---:|
| Accuracy | 91.8876% | 93.3903% | **+1.5027 pp** |
| Recall | 86.5216% | 91.5230% | **+5.0014 pp** |
| F1 | 90.2180% | 92.2923% | **+2.0743 pp** |
| FPR | 4.0250% | 5.1873% | **+1.1623 pp** |
| 总错误 | 664 | 541 | **−123（−18.52%）** |

E3@512 是当前最有吸引力的综合点，但其 FPR 仍高于 fresh base，因此不能写成“微调后 hallucination 已低于基础模型”。

## 4. 固定 2k dev 的全部 checkpoint

所有 checkpoint 使用相同的 2,000 条分层 dev、相同顺序、256 visual tokens。dev SHA256：

```text
94c8c98a9a16efc691eb28239aa8c7e1fc545f92b9922211749c5a3ea1483b92
```

| 模型 | ckpt | Acc | Precision | Recall | F1 | FPR | Adv. FPR | Yes Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 Base | — | 93.70 | 97.71 | 89.50 | 93.42 | 2.10 | 3.60 | 45.80 |
| E1 Attention | 500 | 95.25 | 94.94 | 95.60 | 95.27 | 5.10 | 9.01 | 50.35 |
| E1 Attention | **1000** | **95.80** | 96.54 | 95.00 | **95.77** | 3.40 | 6.61 | 49.20 |
| E1 Attention | 1500 | 95.40 | 94.60 | 96.30 | 95.44 | 5.50 | 9.31 | 50.90 |
| E1 Attention | 2000 | 95.55 | 95.87 | 95.20 | 95.53 | 4.10 | 7.81 | 49.65 |
| E2 +Merger | 500 | 95.25 | 94.85 | 95.70 | 95.27 | 5.20 | 9.31 | 50.45 |
| E2 +Merger | **1000** | **95.80** | 96.45 | 95.10 | **95.77** | 3.50 | 6.61 | 49.30 |
| E2 +Merger | 1500 | 95.40 | 94.34 | 96.60 | 95.45 | 5.80 | 9.91 | 51.20 |
| E2 +Merger | 2000 | 95.70 | 95.88 | 95.50 | 95.69 | 4.10 | 7.21 | 49.80 |
| E3 Hard-neg | 500 | 95.20 | 94.93 | 95.50 | 95.21 | 5.10 | 9.01 | 50.30 |
| E3 Hard-neg | **1000** | 95.65 | **98.10** | 93.10 | 95.54 | **1.80** | **3.90** | 47.45 |
| E3 Hard-neg | 1500 | 95.45 | 95.04 | 95.90 | 95.47 | 5.00 | 8.71 | 50.45 |
| E3 Hard-neg | 2000 | **95.95** | 96.37 | 95.50 | **95.93** | 3.60 | 6.91 | 49.55 |
| E4 All-linear | 500 | 95.20 | 95.56 | 94.80 | 95.18 | 4.40 | 7.21 | 49.60 |
| E4 All-linear | **1000** | 94.75 | **98.80** | 90.60 | 94.52 | **1.10** | **2.10** | 45.85 |
| E4 All-linear | 1500 | 95.60 | 95.33 | 95.90 | **95.61** | 4.70 | 8.11 | 50.30 |
| E4 All-linear | 2000 | 95.60 | 96.15 | 95.00 | 95.57 | 3.80 | 6.61 | 49.40 |

E1/E2 的 checkpoint-1000 是人工平衡点，但未满足原始硬约束；E3-1000 与 E4-1000 是规则内 eligible points。最终选择 E3-1000，是因为它在保持较多 Recall 增益的同时，比 E1 明显减少 FP。

## 5. E0–E4 错误地图

网页为以下五个统一 256vt 代表模型各展示 5 张 RePOPE 错误照片：

| 模型 | 代表配置 | 总错误 | FP | FN | 精选案例 |
|---|---|---:|---:|---:|---:|
| E0 | fresh base | 664 | 187 | 477 | 5 |
| E1 | checkpoint-1000 | 603 | 368 | 235 | 5 |
| E2 | checkpoint-1000 | 609 | 370 | 239 | 5 |
| E3 | checkpoint-1000 | 602 | 263 | 339 | 5 |
| E4 | checkpoint-1000 | 629 | 212 | 417 | 5 |

案例选择约束：

- 每组至少 2 个 FP、2 个 FN，并覆盖 random、popular、adversarial；
- 第五例匹配模型的主要错误方向：E0/E3/E4 为 FN，E1/E2 为 FP；
- 每组 4 个案例至少有另一个模型答对，1 个案例为五模型共同失败；
- 25 个案例使用不同的 `image_source`；
- 卡片展示 RePOPE 真值和 E0–E4 五个模型对同一问题的答案。

完整清单见 `error_case_manifest.json`。这些是人工检查后的诊断性案例，不是统计代表性抽样，也不构成对 RePOPE 标签的重新裁决。

## 6. Visual tokens：256 → 512

同一 E3 checkpoint-1000、同一 8,185 条 RePOPE，只改变 inference-time visual-token budget：

| Overall | 256 vt | 512 vt | Δ |
|---|---:|---:|---:|
| Accuracy | 92.6451% | **93.3903%** | **+0.7453 pp** |
| Precision | 92.4054% | **93.0747%** | **+0.6693 pp** |
| Recall | 90.4210% | **91.5230%** | **+1.1020 pp** |
| F1 | 91.4025% | **92.2923%** | **+0.8899 pp** |
| FPR | 5.6608% | **5.1873%** | **−0.4735 pp** |
| Yes Ratio | 42.3091% | 42.5168% | +0.2077 pp |
| TP / FP / TN / FN | 3200 / 263 / 4383 / 339 | 3239 / 241 / 4405 / 300 | +39 / −22 / +22 / −39 |

逐条对齐后的 237 次预测变化：

```text
改善：90 FN→TP + 59 FP→TN = 149
退化：51 TP→FN + 37 TN→FP = 88
净改善：61
```

三个 split 均改善，adversarial 最明显：Recall +1.10 pp、FPR −1.07 pp、F1 +1.20 pp。

## 7. 数据与训练设计

### 数据

- 来源：COCO train2017。
- 测试图片黑名单：500 张 RePOPE 图片。
- 训练图片：10,000 张；训练问题：16,000。
- E1/E2：8,000 positive；random/popular/adversarial negatives 为 2,667/2,667/2,666。
- E3/E4：8,000 positive；random/popular/adversarial negatives 为 2,000/2,000/4,000。
- E1 与 E3 使用相同图片序列和相同 positive rows，只改变 negative composition。
- train、dev、RePOPE 按 image ID 隔离，零图片交集。

### QLoRA

| 参数 | 值 |
|---|---|
| Base model | Qwen2-VL-2B-Instruct |
| Quantization | 4-bit NF4 + double quantization |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Learning rate | 1e-4 |
| Training | 2 epochs / 2,000 optimizer steps |
| Batch | micro 2 × accumulation 8 = effective 16 |
| Supervision | assistant answer only：yes/no + EOS |
| Generation | greedy，最多 4 new tokens |

实验变量：

- E1：LLM self-attention q/k/v/o。
- E2：E1 + visual Merger MLP 0/2。
- E3：E1 架构 + 更多 adversarial negatives。
- E4：E3 数据 + attention 与 MLP gate/up/down all-linear LoRA。

## 8. 工程验证

速度基准锁定 B2：

| 配置 | Micro × Accum | Samples/s | Mean step | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| B1 | 1 × 16 | 1.397 | 11.46 s | 3.98 GiB |
| **B2** | **2 × 8** | **2.884** | **5.55 s** | **5.80 GiB** |

B2 吞吐约为 B1 的 2.07 倍。B4 因显存压力主动停止，不作为性能数据。

Smoke test 在 128 条数据上运行 8 optimizer steps，验证：

- answer-only labels 正确；
- LoRA 只注入批准的模块；
- loss 全部有限且存在非零梯度；
- LoRA 参数发生更新；
- step 1 后无显存增长趋势；
- adapter 保存、重载和固定样本推理成功。

## 9. 研究边界

可以确认：

- PEFT 是有效的行为干预，但主要体现为 operating-point shift；
- hard-negative supervision 能缓解 attention QLoRA 引入的误报；
- E3@512 是已测试配置中 Accuracy 与 F1 最高的点；
- dev 对相对保守/激进顺序有预测力，但低估 RePOPE FPR。

不能声称：

- fine-tuned 模型的 hallucination 已经低于 fresh base；
- E3@512 是未经 post-selection bias 的正式模型选择结果；
- 单个 256/512 消融已经证明更高分辨率修复了 grounding 机制；
- 25 张案例代表总体错误分布。

后续优先方向是 model-driven hard-example mining、更高质量的 hard positives/negatives、多尺度视觉输入，以及使用 logits 的 Recall–FPR calibration，而不是继续单纯扩大 LoRA target。

## 10. 交付目录

```text
QLoRA微调总报告.html     离线交互式总报告
README.md                完整研究结论与复现索引
PACKAGE_MANIFEST.md      文件清单与交付边界
error_case_manifest.json 25 个错误案例及五模型答案
assets/
  report.css
  report.js
  report-data.js
  error-cases/            25 张 RePOPE 图片
metrics/                  精简后的原始指标 JSON
config/                   训练配置、run 与 selection manifests
manifests/                数据构建与 2k dev manifest
scripts/build_report_data.py
```

报告包不包含模型权重、optimizer/scheduler 状态、完整 predictions JSONL、全量错误图片或大型 ZIP。

## 11. 重建数据与图片

如需从冻结实验文件重新生成 `report-data.js`、错误 manifest 和 25 张图片：

```powershell
cd deliverables\qwen2vl-pope-final-report-2026-08-12
python scripts\build_report_data.py
python scripts\validate_report.py
```

构建脚本只读取现有 metrics、predictions 与本地 Hugging Face cache，不运行模型推理。可用 `HF_DATASETS_CACHE` 指定缓存目录；若公开仓库中没有完整 predictions，验证器仍会执行独立交付包校验，并跳过逐题深度对照。
