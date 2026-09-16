# 审查文档 · Authenticity Audit

**目的：** 证明本稿 Results 数字来自本仓库自行计算的流水线，而非抄录 *Nature Medicine* Reti-Pioneer 临床表，亦非标签派生 SYNTHETIC 特征。  
**日期：** 2026-09-17（real-results 刷新）  
**公开快照：** https://github.com/Coucou2016/retinal-imaging-methods-public

---

## 1. 一句话结论

| 主张 | 判定 |
|------|------|
| RFMiD E0–E5 AUROC / ECE / \(T\) / bootstrap CI | **本仓库** `build_real_rfmid_cache.py` → `run_real_rfmid_experiments.py` → `evaluate.py` |
| 直观图（眼底质量分层、ROC、可靠性、DCA、消融柱、路由） | `scripts/plot_real_results.py`（SciencePlots + Times New Roman） |
| Reti-Pioneer UKB AUROC 0.699–0.833 | **仅作背景引用**，未写入我们的结果表 |
| ODIR / BRSET / UKB 临床主表 | **待补充**（见 DATA_BLOCKERS） |
| 历史 SYNTHETIC ODIR 消融表 | **已移出主 Results**；仅留作 CI |

---

## 2. REAL vs SYNTHETIC 证据链

| 资产 | 路径 | 标记 | 含义 |
|------|------|------|------|
| RFMiD 特征出处 | `data/rfmid/FEATURE_PROVENANCE.json` | `real_pixels: true` | 真实眼底像素；ImageNet 骨干替代 |
| SYNTHETIC 旗标 | `data/rfmid/SYNTHETIC_FEATURES.txt` | **已删除** | 不再存在 |
| 骨干哈希（前 8MB SHA256 前缀） | `UKB_swin.npz` | `860db8257ec87300` | Swin-V2-B ImageNet |
| | `UKB_RETF.npz` | `37b95500eb975f58` | ViT-B→1024（RETFound gated） |
| | `UKB_vim.npz` | `4d62c0f2946365d0` | ViT-S/384（Vim 不可用） |
| 质量代理 | `UKB_mqd.npz` ql/qr | 1131 个唯一 soft-q | 非常数 default_quality |
| 结果总表 | `results/real_rfmid/ablation_summary.csv` | disclaimer = REAL-PIXEL | 生成于 2026-09-17 |
| 评估 JSON | `results/real_rfmid/metrics_E{0-5}_rfmid_{val,test}.json` | bootstrap + DeLong + dca_curves | calibrate on cal, score on split |

**诚实边界：** RETFound HF 仓库 gated（401）；Vision Mamba / mamba-ssm 在本机不可用。因此骨干为 ImageNet 替代，**不是** UKB Reti-Pioneer 三骨干复现，但是**真实像素**特征，可用于 RFMiD 眼底体征 Results。

---

## 3. 数字出处（RFMiD test）

### 可复现命令

```text
python scripts/build_real_rfmid_cache.py --cache-dir data/rfmid --batch-size 8
python scripts/run_real_rfmid_experiments.py --data-dir data/rfmid --out-dir results/real_rfmid --epochs 12 --bootstrap 200
python scripts/plot_real_results.py
python scripts/build_paper_report.py
```

### Test 表（\(n=640\)，bootstrap \(B=200\)）

| Arm | Macro AUROC | AUROC_DR | ECE | Cal ECE | T | Bootstrap 95% CI |
|-----|-------------|----------|-----|---------|---|------------------|
| E0 | 0.925 | 0.925 | 0.026 | 0.031 | 1.06 | [0.902, 0.947] |
| E1 | 0.931 | 0.931 | 0.034 | 0.032 | 1.11 | [0.908, 0.949] |
| E2 | 0.886 | 0.910 | 0.170 | 0.176 | 1.06 | [0.872, 0.902] |
| E3 | 0.898 | 0.928 | 0.178 | 0.188 | 1.11 | [0.882, 0.914] |
| E4 | 0.898 | 0.928 | 0.178 | 0.188 | 1.11 | [0.882, 0.914] |
| E5 | 0.901 | 0.916 | 0.156 | 0.154 | 0.98 | [0.887, 0.914] |

Run dirs：见 `results/real_rfmid/run_manifest.json`。

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

| 图 | 脚本 | 输入 | 输出 |
|----|------|------|------|
| Fig1 architecture | `plot_paper_figures.py` | schematic | `fig1_architecture.*` |
| Fig2 ablation bars | `plot_real_results.py` | `ablation_summary.csv` | `fig2_ablation_bars.*` |
| Fig3 reliability | same | E5 test preds | `fig3_calibration.*` |
| Fig5 fundus quality | same | RFMiD images + ql | `fig5_fundus_quality.*` |
| Fig6 routing | same | QualityAware + router | `fig6_quality_routing.*` |
| Fig6b quality pie | same | ql argmax | `fig6b_quality_pie.*` |
| Fig7 DCA | same | metrics JSON dca_curves | `fig7_decision_curves.*` |
| Fig8 ROC | same | E5 test probs | `fig8_roc_curves.*` |
| Fig9 score hist | same | E5 DR scores | `fig9_score_hist.*` |

---

## 5. 与 *Nature Medicine* 表隔离

| 检查 | 结果 |
|------|------|
| 我们的结果表是否出现 0.833 / 0.832 / … 作为“本方法 AUROC”？ | **否** |
| 是否声称 UKB T2DM 复现？ | **否** |
| SYNTHETIC AUROC 是否仍在主 Results？ | **否**（已删除） |

---

## 6. 环境与阻塞

见 `docs/DATA_BLOCKERS.md`（2026-09-17）：CUDA torch 不可用；ODIR/BRSET 无凭据；RETFound gated；Vim 不可用。
