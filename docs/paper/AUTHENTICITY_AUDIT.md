# 审查文档 · Authenticity Audit

**目的：** 证明本稿结果数字来自本仓库自行计算的流水线，而非抄录 *Nature Medicine* Reti-Pioneer 临床表。  
**日期：** 2026-09-15（full-deliverables 刷新）  
**仓库 HEAD（本轮交付前本地）：** 见文末验收写入的最终 SHA  
**公开快照：** https://github.com/Coucou2016/retinal-imaging-methods-public

---

## 1. 一句话结论

| 主张 | 判定 |
|------|------|
| 消融 AUROC / ECE / \(T\) / \(n\) | **本仓库 `run_ablations.py` 计算**，见 `results/ablation_summary.*` 与 `results/metrics_*.json` |
| Reti-Pioneer UKB AUROC 0.699–0.833 | **仅作背景引用**（Zhang et al. 2026），**未写入我们的结果表** |
| 临床主表（真实三骨干特征） | **待补充**；当前 `clinical_claim_allowed=false` |
| 图 1–4 | SciencePlots + Times New Roman，由 `scripts/plot_paper_figures.py` **从上述 CSV 重绘**（Fig1 含 E5 示意） |

---

## 2. SYNTHETIC vs 真实证据链

| 资产 | 路径 | 标记 | 含义 |
|------|------|------|------|
| ODIR 合成特征旗标 | `data/odir/SYNTHETIC_FEATURES.txt` | `Do not report AUROC…clinical` | 特征由标签确定性生成，非眼底像素 |
| BRSET 合成特征旗标 | `data/brset/SYNTHETIC_FEATURES.txt` | 同上 | 同上 |
| RFMiD 合成特征旗标 | `data/rfmid/SYNTHETIC_FEATURES.txt` | `clinical_claim_allowed: false` | RFMiD 影像可在本地，但当前缓存仍为合成骨干特征 |
| 消融总表 | `results/ablation_summary.csv` | 列 `disclaimer` 全行为 SYNTHETIC | 生成于 2026-09-15T02:45:26 |
| 评估 JSON | `results/metrics_{baseline,learnq,multitask,full}_odir_{val,to_brset}.json` | `disclaimer` 含 `clinical_claim_allowed=false unless alignment=direct` | 与 CSV 同行一致 |
| 运行元数据 | `results/ablation_runs/20260915*/**/run_meta.json` | `quality_router`, `masked_bce`, `dataset=odir` | 证明配置落入训练 |

**判定规则（代码）：** `scripts/evaluate.py` 写入 disclaimer；`reti_pioneer.label_map.features_clinical_claim_allowed` 在存在 `SYNTHETIC_FEATURES.txt` 时拒绝 clinical_ok；`--paper-mode` / `--clinical-tables` 要求 `alignment=direct`。

---

## 3. 数字出处（逐项）

### 3.1 消融矩阵（论文 Table 1 / 报告表 1）

命令（可复现）：

```text
python scripts/run_ablations.py --quick
```

| Arm | Eval | AUROC_D | ECE → Cal ECE | 源文件 |
|-----|------|---------|---------------|--------|
| baseline | odir_val | 0.381 | 0.363 → 0.289 | `metrics_baseline_odir_val.json` |
| baseline | odir_to_brset | 0.558 | 0.357 → 0.350 | `metrics_baseline_odir_to_brset.json` |
| learnq | odir_val | 0.762 | 0.319 → 0.400 | `metrics_learnq_odir_val.json` |
| learnq | odir_to_brset | 0.493 | 0.414 → 0.481 | `metrics_learnq_odir_to_brset.json` |
| multitask | odir_val | 0.500 (D) / 0.363 macro | 0.669 → 0.401 | `metrics_multitask_odir_val.json` |
| multitask | odir_to_brset | 0.563 (D) | 0.481 → 0.400 | `metrics_multitask_odir_to_brset.json` |
| full | odir_val | 0.438 (D) / 0.532 macro | 0.656 → 0.410 | `metrics_full_odir_val.json` |
| full | odir_to_brset | 0.516 (D) | 0.475 → 0.446 | `metrics_full_odir_to_brset.json` |

**样本：** ODIR val \(n=10\)；跨库 \(n=48\)。如此小 \(n\) 下出现个别头 AUROC=1.0 属于合成标签折统计噪声，**不是**临床性能。

### 3.2 配置与代码映射

| 论文概念 | 配置 / 代码 |
|----------|-------------|
| Fixed quality baseline | `quality_router: fixed`；`model/QualityAware.py` |
| Monotone router | `quality_router: monotone`；`QualityAware.monotone_weights`（softplus + cumsum 归一化） |
| Partial-label MTL | `masked_bce: true`；`utils/run.py::masked_bce_with_logits` |
| Released-code control | `ensemble: released_code`；独立病种循环 vs `multitask: true` |
| E5 backbone routing | `configs/ablation_e5.yaml` → `quality_gating: true`, `lambda_q: 0.1`；`model/quality_gate.py::QualityBackboneRouter` |
| Endpoint alignment | `reti_pioneer/label_map.py`；跨评 `clinical_claim_allowed` |
| 校准折不相交 | `split.cal_fraction` + `evaluate.py` `temperature_fit=calibration_idx` |

示例 run_meta（full / multitask ODIR）：

```json
{
  "learnable_q": true,
  "quality_router": "monotone",
  "ensemble": "released_code",
  "multitask": true,
  "masked_bce": true,
  "quality_gating": false,
  "dataset": "odir",
  "fast_mode": true
}
```

路径：`results/ablation_runs/20260915024410/multitask/y0/run_meta.json`。

---

## 4. 图件与交付物生成链

| 产物 | 脚本 | 输入 | 输出 | 样式 / 约束 |
|------|------|------|------|-------------|
| Fig 1 architecture | `plot_architecture_schematic` | 无性能数字 | `docs/paper_assets/fig1_architecture.{png,pdf}` | SciencePlots + Times New Roman；含 E5 橙框 |
| Fig 2 ablation bars | `plot_ablation_bars` | `results/ablation_summary.csv` | `fig2_ablation_bars.*` | 标题含 SYNTHETIC |
| Fig 3 calibration | `plot_calibration_effect` | 同上 | `fig3_calibration.*` | 同上 |
| Fig 4 cross-domain | `plot_cross_domain` | 同上 | `fig4_cross_domain.*` | 同上 |
| Base64 sidecar | `write_uri_sidecar` | PNG | `docs/paper_assets/embedded_png_uris.json` | 供 HTML 内嵌 |
| Paper md/html/pdf | `build_paper_report.py` | `manuscript.md` + URIs | `docs/paper/manuscript.{md,html,pdf}` | 论文无本机绝对路径；PDF 含 Fig1–4 |
| Report md/html/pdf | 同上 | CSV + URIs | `docs/report/report.{md,html,pdf}` | HTML：内联 CSS、Base64、无 CDN；可含路径 |

命令：

```text
pip install SciencePlots
python scripts/plot_paper_figures.py
python scripts/build_paper_report.py
```

---

## 5. 与 *Nature Medicine* 表的隔离检查

| 检查项 | 结果 |
|--------|------|
| 我们的结果表是否出现 0.833 / 0.832 / 0.787 / 0.740 / 0.736 / 0.699 作为“本方法 AUROC”？ | **否**。这些数仅在引言中作为 Reti-Pioneer **已发表**内部检验范围引用 |
| 我们的 AUROC_D 是否与上述集合重合？ | **否**（0.381–0.762 等，且 \(n\) 与终点不同） |
| 是否声称在 UKB / 医院队列上复现？ | **否**；明确 **待补充** |
| 合成结果是否标注？ | **是**（CSV disclaimer、图标题、论文 §5.1、报告红框） |

---

## 6. Git / 提交锚点（方法学相关）

| Commit | 主题 |
|--------|------|
| `f431a00` | monotone router、endpoint ontology、calibration split |
| `3584cb7` | threshold lock、bootstrap CI、E5 / λ_q |
| `cc70b8d` | YAML–code contract sync |
| `30ef160` | E5 backbone routing、MultiCohort、eval CIs |
| `0816710` | 方法稿 / 自包含报告 / 审查文档（上一轮公开 tip） |
| （本轮） | 学术语气重写、强化方法三模块、全图 PDF、full-deliverables ACCEPTANCE |

完整历史以 `git log` 为准；本文件不替代 ACCEPTANCE 清单。

---

## 7. 环境与阻塞（诚实记录）

| 项 | 状态 |
|----|------|
| CUDA / 三骨干特征提取 | **不可用** → 临床表待补充 |
| ODIR Kaggle | 无凭据 |
| BRSET PhysioNet | DUA / 凭据未完成 |
| RFMiD 影像 | 本地有（\(N=3200\)）；合成特征旗标仍在 → 不得 clinical claim |
| 论文稿主线 | `docs/paper/manuscript.md`（2026-09-15 full-deliverables：学术语气 + 强化 monotone / MTL / E5） |

---

## 8. 审阅人快速核验清单

1. 打开 `results/ablation_summary.csv`，确认 `disclaimer` 列全为 SYNTHETIC。  
2. 打开任一 `results/metrics_*_odir_val.json`，确认 `disclaimer` 与 `n`。  
3. 打开 `data/*/SYNTHETIC_FEATURES.txt`。  
4. 确认论文结果表 **没有** 把 0.833 等 UKB AUROC 写成本方法成绩。  
5. 确认 §5.2 / 报告临床表为 **待补充**。  
6. 运行 `python scripts/plot_paper_figures.py && python scripts/build_paper_report.py` 应可重生图与 HTML/PDF。  
7. 确认公开 tip SHA 与 `docs/chatgpt-runs/2026-09-15-full-deliverables/ACCEPTANCE.md` 一致。
