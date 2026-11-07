# 审查文档 · Authenticity Audit

**目的：** 证明本稿 Results 数字来自本仓库自行计算的流水线，而非抄录 *Nature Medicine* Reti-Pioneer 临床表，亦非标签派生 SYNTHETIC 特征。  
**日期：** 2026-09-17（real-results 刷新）  
**公开快照：** https://github.com/Coucou2016/retinal-imaging-methods-public

---

## 1. 一句话结论

| 主张 | 判定 |
|------|------|
| RFMiD E0–E5 AUROC / ECE / \(T\) / bootstrap CI | **本仓库** `build_real_rfmid_cache.py` → `run_real_rfmid_experiments.py` → `analyze_arms.py` |
| 校准诊断（logit 偏移、共享 Platt、事后校正阶梯） | `logit_offset_analysis.py`、`recalibration_holdout.py`、`recalibration_followup.py` |
| 工作点与阈值搬移 | `operating_point_analysis.py`、`threshold_transfer.py`、`clinical_utility.py` |
| 配对自助法分布（1000 次重采样密度） | `extra_evidence.py` → `stats/extra_bootstrap.npz`、`stats/extra_evidence.json` |
| 直观图（眼底质量分层、ROC、可靠性、DCA、消融柱、路由） | `rebuild_paper_figures.py`（SciencePlots + Times New Roman，600 dpi） |
| Reti-Pioneer UKB AUROC 0.699–0.833 | **仅作背景引用**，未写入我们的结果表 |
| ODIR / BRSET / UKB 临床主表 | **未包含**（本机无凭据；见 DATA_BLOCKERS） |
| 历史 SYNTHETIC ODIR 消融表 | **已移出主 Results**；仅留作 CI |

---

## 2. REAL vs SYNTHETIC 证据链

| 资产 | 路径 | 标记 | 含义 |
|------|------|------|------|
| RFMiD 特征出处 | `FEATURE_PROVENANCE.json` | `real_pixels: true` | 真实眼底像素；ImageNet 骨干替代 |
| SYNTHETIC 旗标 | `SYNTHETIC_FEATURES.txt` | **已删除** | 不再存在 |
| 骨干哈希（前 8MB SHA256 前缀） | `UKB_swin.npz` | `860db8257ec87300` | Swin-V2-B ImageNet |
| | `UKB_RETF.npz` | `37b95500eb975f58` | ViT-B→1024（RETFound gated） |
| | `UKB_vim.npz` | `4d62c0f2946365d0` | ViT-S/384（Vim 不可用） |
| 质量代理 | `UKB_mqd.npz` ql/qr | 1131 个唯一 soft-q | 非常数 default_quality |
| 结果总表 | `ablation_summary.csv` | disclaimer = REAL-PIXEL | 生成于 2026-09-17 |
| 评估 JSON | `metrics_E{0-5}_rfmid_{val,test}.json` | bootstrap + DeLong + dca_curves | calibrate on cal, score on split |

**诚实边界：** RETFound HF 仓库 gated（401）；Vision Mamba / mamba-ssm 在本机不可用。因此骨干为 ImageNet 替代，**不是** UKB Reti-Pioneer 三骨干复现，但是**真实像素**特征，可用于 RFMiD 眼底体征 Results。

---

## 3. 数字出处（RFMiD test）

### 可复现命令

```text
python build_real_rfmid_cache.py --cache-dir data/rfmid --batch-size 8
python run_real_rfmid_experiments.py --data-dir data/rfmid --epochs 15
python select_epochs.py --runs results/real_rfmid
python collect_arm_predictions.py --arms E0 E1 E2 E3 E4 E5
python collect_cal_predictions.py --arms E0 E1 E2 E3 E4 E5
python analyze_arms.py --arms E0 E1 E2 E3 E4 E5
python recalibrate_arms.py
python temperature_schemes.py
python logit_offset_analysis.py
python recalibration_holdout.py
python recalibration_followup.py
python extra_evidence.py
python export_manuscript_numbers.py
python rebuild_paper_figures.py
```

### Test 表（\(n=640\)，bootstrap \(B=1000\)，报告轮次按验证集选择）

Numbers below are read from `analysis_sel/manuscript_numbers.json`, the same file that `check_manuscript_numbers.py` uses to audit the manuscript. Any drift between table and paper fails that check.

| Arm | Macro AUROC | Bootstrap 95% CI | DeLong 95% CI | AUROC_DR | Mean per-head ECE | Brier | \(T\) (cal 64) |
|-----|-------------|------------------|---------------|----------|-------------------|-------|----------------|
| E0 | 0.932 | [0.908, 0.953] | [0.908, 0.956] | 0.932 | 0.031 | 0.075 | 1.099 |
| E1 | 0.933 | [0.912, 0.952] | [0.912, 0.955] | 0.933 | 0.025 | 0.079 | 1.325 |
| E2 | 0.896 | [0.881, 0.910] | [0.863, 0.928] | 0.926 | 0.136 | 0.098 | 1.216 |
| E3 | 0.906 | [0.890, 0.919] | [0.872, 0.940] | 0.934 | 0.149 | 0.101 | 1.261 |
| E4 | 0.898 | [0.882, 0.912] | [0.865, 0.930] | 0.931 | 0.122 | 0.097 | 1.382 |
| E5 | 0.916 | [0.903, 0.928] | [0.888, 0.944] | 0.938 | 0.122 | 0.087 | 1.132 |

Selected epochs: E0 12/15, E1 14/15, E2 15/15, E3 15/15, E4 18/23, E5 15/15. The earlier 12-epoch fixed-budget run is reported in §5.6 of the manuscript as a sensitivity check, not as the main result.

Run dirs：见 `run_manifest.json`。

### 配置映射

| 论文概念 | 配置 |
|----------|------|
| E0 fixed quality | `ablation_e0.yaml` |
| E1 monotone | `ablation_e1.yaml` |
| E2 MTL fixed-q | `ablation_e2.yaml` |
| E3 MTL + monotone | `ablation_e3.yaml` |
| E4 + cal protocol | `ablation_e4.yaml` |
| E5 backbone routing | `ablation_e5.yaml` (`quality_gating: true`) |

---

## 4. 图件生成链

All sixteen figures in the manuscript come from `rebuild_paper_figures.py`, which reads only the JSON/CSV artefacts under `analysis_sel/` and the RFMiD images under ``. Typography is SciencePlots with Times New Roman, and the PNGs are written at 600 dpi.

| 图 | 函数 | 输入 | 输出 |
|----|------|------|------|
| Fig 1 architecture | `fig1_architecture` | schematic | `fig1_architecture.*` |
| Fig 2 fundus strata | `fig5_fundus` | RFMiD images + soft-q | `fig5_fundus_quality_embed.*` |
| Fig 3 quality counts | `fig6b_pie` | `quality_strata.json` | `fig6b_quality_pie.*` |
| Fig 4 ablation bars | `fig2_ablation` | `headline.json` | `fig2_ablation_bars.*` |
| Fig 5 paired delta | `fig5_paired` | `extra_bootstrap.npz` | `fig10_paired_delta.*` |
| Fig 6 endpoint delta | `fig6_endpoint` | `extended.json` | `fig17_endpoint_delta.*` |
| Fig 7 ROC curves | `fig7_roc` | E5 test predictions | `fig8_roc_curves.*` |
| Fig 8 reliability | `fig8_reliability` | E3/E5 test predictions | `fig3_calibration.*` |
| Fig 9 per-head ECE | `fig9_ece` | `per_head.json` | `fig12_ece_by_endpoint.*` |
| Fig 10 correction ladder | `fig10_ladder` | `recalibration_holdout.json`, `recalibration_followup.json` | `fig16_recalibration_ladder.*` |
| Fig 11 calibration headline | `fig11_headline_calibrated` | `headline_calibrated.json` | `fig13_headline_calibrated.*` |
| Fig 12 threshold transfer | `fig12_transfer` | `threshold_transfer.json` | `fig14_threshold_transfer.*` |
| Fig 13 score shift | `fig13_shift` | E0/E5 test predictions | `fig15_score_shift.*` |
| Fig 14 decision curves | `fig14_dca` | `decision_curves.json` | `fig7_decision_curves.*` |
| Fig 15 quality strata | `fig15_strata` | `quality_strata.json` | `fig11_quality_strata.*` |
| Fig 16 router intervention | `fig16_router` | `router_intervention.json` | `fig18_router_intervention.*` |

Inline display equations are rendered to transparent PNGs by `render_equations.py` at 600 dpi and embedded by the Word exporter.

---

## 5. 与 *Nature Medicine* 表隔离

| 检查 | 结果 |
|------|------|
| 我们的结果表是否出现 0.833 / 0.832 / … 作为“本方法 AUROC”？ | **否** |
| 是否声称 UKB T2DM 复现？ | **否** |
| SYNTHETIC AUROC 是否仍在主 Results？ | **否**（已删除） |

---

## 6. 环境与阻塞

见 `DATA_BLOCKERS.md`（2026-09-17）：CUDA torch 不可用；ODIR/BRSET 无凭据；RETFound gated；Vim 不可用。
