# QLoRA 最终报告轻量交付包

> 整理日期：2026-08-12
>
> 打开方式：双击 `QLoRA微调总报告.html`

## 交付内容

- `QLoRA微调总报告.html`：离线长页报告，包含全部指标、结论图表和 E0–E4 错误地图。
- `README.md`：完整研究总结、全部核心指标、方法、局限和复现入口。
- `error_case_manifest.json`：25 个错误案例的来源、真值、五模型答案和诊断说明。
- `assets/`：网页样式、交互、规范化数据以及 25 张案例图片。
- `metrics/`：原始 POPE、fresh RePOPE、E0–E4 dev/RePOPE、视觉消融、测速与 Smoke 指标。
- `config/`：训练配置、E1–E4 run manifests、selection manifests 和 Smoke manifest。
- `manifests/`：训练/dev 数据构建摘要与哈希。
- `scripts/build_report_data.py`：从冻结实验结果重建网页数据和案例图片。
- `scripts/validate_report.py`：校验 fresh baseline、指标、错误案例和离线资源；存在冻结 predictions 时追加逐题检查。

## 口径锁定

RePOPE baseline 唯一来源：

```text
metrics/e0_fresh_base_repope_256vt_metrics.json
```

其混淆矩阵必须为：

```text
TP / FP / TN / FN = 3062 / 187 / 4459 / 477
```

旧的重评分 baseline 未复制到本交付包，也不进入网页数据。

## 错误案例约束

- E0–E4 各 5 张，共 25 张；
- 所有焦点模型均在对应问题上答错；
- 每组至少 2 FP、2 FN，并覆盖三个 RePOPE split；
- 每组一个五模型共同失败案例；
- 25 个 `image_source` 不重复；
- 所有答案逐条来自对应的 256vt predictions JSONL。

## 未包含

- 基础模型或 LoRA adapter 权重；
- optimizer、scheduler、RNG 与训练恢复状态；
- 完整 predictions JSONL；
- 全量错误图片、审计 ZIP 或 Hugging Face 数据缓存。
