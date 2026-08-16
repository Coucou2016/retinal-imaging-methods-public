#!/usr/bin/env python3
"""Build manuscript HTML + self-contained research report (HTML/MD/PDF).

Figures are embedded as Base64 data URIs. Synthetic metrics are labeled SYNTHETIC.
"""
from __future__ import annotations

import csv
import html
import json
import re
from datetime import date
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "paper_assets"
PAPER = ROOT / "docs" / "paper"
REPORT = ROOT / "docs" / "report"
CSV_PATH = ROOT / "results" / "ablation_summary.csv"
URI_JSON = ASSETS / "embedded_png_uris.json"

FONT_EN = Path(r"C:\Windows\Fonts\times.ttf")
FONT_CJK = Path(r"C:\Windows\Fonts\simhei.ttf")


def load_rows() -> list[dict]:
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def load_uris() -> dict[str, str]:
    if URI_JSON.exists():
        return json.loads(URI_JSON.read_text(encoding="utf-8"))
    # regenerate
    import subprocess
    import sys

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "plot_paper_figures.py")])
    return json.loads(URI_JSON.read_text(encoding="utf-8"))


def fmt(x: str, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return x


def ablation_table_html(rows: list[dict]) -> str:
    head = (
        "<table><thead><tr>"
        "<th>Arm</th><th>Eval</th><th>AUROC (macro)</th><th>AUROC_D</th>"
        "<th>ECE</th><th>Cal ECE</th><th>NB@0.10</th><th>n</th><th>Note</th>"
        "</tr></thead><tbody>"
    )
    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(r['ablation'])}</td>"
            f"<td>{html.escape(r['eval'])}</td>"
            f"<td>{fmt(r['auroc'])}</td>"
            f"<td>{fmt(r['auroc_D'])}</td>"
            f"<td>{fmt(r['ece'])}</td>"
            f"<td>{fmt(r['cal_ece'])}</td>"
            f"<td>{fmt(r['nb@0.10'])}</td>"
            f"<td>{html.escape(r['n'])}</td>"
            "<td><strong>SYNTHETIC</strong></td>"
            "</tr>"
        )
    return head + "\n".join(body) + "</tbody></table>"


def ablation_table_md(rows: list[dict]) -> str:
    lines = [
        "| Arm | Eval | AUROC | AUROC_D | ECE | Cal ECE | NB@0.10 | n | Note |",
        "|-----|------|-------|---------|-----|---------|---------|---|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ablation']} | {r['eval']} | {fmt(r['auroc'])} | {fmt(r['auroc_D'])} | "
            f"{fmt(r['ece'])} | {fmt(r['cal_ece'])} | {fmt(r['nb@0.10'])} | {r['n']} | SYNTHETIC |"
        )
    return "\n".join(lines)


CSS = """
:root {
  --ink: #1a1a1a;
  --muted: #555;
  --line: #ccc;
  --accent: #1f4e79;
  --warn: #8b0000;
  --bg: #faf9f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Times New Roman", Times, "SimSun", serif;
  color: var(--ink);
  background: linear-gradient(180deg, #eef2f6 0%, var(--bg) 220px);
  line-height: 1.65;
  font-size: 16px;
}
.wrap { max-width: 920px; margin: 0 auto; padding: 32px 28px 80px; }
.cover {
  padding: 48px 36px;
  border: 1px solid var(--line);
  background: #fff;
  margin-bottom: 28px;
}
.cover h1 { font-size: 1.85rem; margin: 0 0 12px; color: var(--accent); }
.cover .meta { color: var(--muted); font-size: 0.95rem; }
.badge {
  display: inline-block;
  border: 1px solid var(--warn);
  color: var(--warn);
  padding: 2px 8px;
  font-size: 0.8rem;
  margin-right: 6px;
}
nav.toc { background: #fff; border: 1px solid var(--line); padding: 18px 22px; margin-bottom: 28px; }
nav.toc ol { margin: 8px 0 0; }
section {
  background: #fff;
  border: 1px solid var(--line);
  padding: 22px 26px 28px;
  margin-bottom: 22px;
}
h2 { color: var(--accent); border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { margin-top: 1.4em; }
figure { margin: 1.2em 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #e5e5e5; }
figcaption { text-align: left; font-size: 0.92rem; color: #333; margin-top: 10px; }
.longcap { background: #f6f8fb; border-left: 3px solid var(--accent); padding: 10px 14px; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 12px 0; }
th, td { border: 1px solid #bbb; padding: 6px 8px; text-align: left; }
th { background: #e8eef5; }
.warnbox {
  border: 1px solid var(--warn);
  background: #fff5f5;
  padding: 12px 14px;
  margin: 14px 0;
}
.pending { color: var(--warn); font-weight: bold; }
a { color: var(--accent); }
.abbr { color: var(--muted); font-size: 0.92rem; }
"""


def manuscript_md() -> str:
    return f"""# Quality-Adaptive Multi-Task Oculomics: Extending Reti-Pioneer with Learnable Fusion, Calibration, and Public-Cohort Validation

**Manuscript draft (methods paper)** · {date.today().isoformat()}  
**Target venues (realistic without UK Biobank):** *npj Digital Medicine*, *Medical Image Analysis*, *IEEE JBHI / TMI*, *Computers in Biology and Medicine*  
**Axes (nature-writing):** task=manuscript · paper_type=methods · language=en · journal=generic (Nature-family structure, not flagship *Nature Medicine* clinical claim)

> **One-sentence argument.** Reti-Pioneer showed that frozen foundation features plus quality-aware fusion can screen systemic disease from color fundus photographs (CFPs); we keep that skeleton, replace fixed quality weights with learnable quality routing, replace independent binary heads with a shared multi-task head, and evaluate calibration / decision-curve utility and cross-dataset generalization on public cohorts reviewers can download — without claiming UK Biobank-comparable area under the receiver operating characteristic curve (AUROC) from synthetic demo caches.

---

## Abstract

Endocrine and metabolic disease screening from retinal imaging (oculomics) remains limited by image-quality heterogeneity, single-task training, and poorly calibrated risk scores. Reti-Pioneer (Zhang et al., *Nature Medicine*, 2026) demonstrated that frozen vision foundation models with quality-aware bilinear fusion can screen six systemic conditions on UK Biobank (UKB) and hospital cohorts, but uses **fixed** quality routing, trains **independent** binary heads, and reports discrimination without treating calibration and decision-curve analysis (DCA) as primary utility. Here we propose a methods extension that (i) unfreezes quality routing (`learnable_q`), (ii) shares a multi-label head across correlated ocular/systemic labels, and (iii) reports expected calibration error (ECE), Brier score, and net benefit under temperature scaling. We specify patient-level evaluation on publicly downloadable Ocular Disease Intelligent Recognition (ODIR-5K), Brazilian Multilabel Ophthalmological Dataset (BRSET), and Retinal Fundus Multi-disease Image Dataset (RFMiD). **Results tables with clinical AUROCs are marked 待补充 until real images and foundation features replace synthetic caches.** Pipeline sanity metrics on synthetic features are reported only in the companion research report and are **not** manuscript claims.

**Keywords:** oculomics; fundus photography; quality-aware fusion; multi-task learning; calibration; domain generalization; Reti-Pioneer

---

## 1. Introduction

Color fundus photography is widely available in community eye care and can encode microvascular signatures of systemic disease. Foundation models such as RETFound (Zhou et al., *Nature*, 2023) provide transferable retinal representations, while quality assessment toolkits such as EyeQ support handling of imperfect acquisitions. Reti-Pioneer integrated frozen RETFound, Swin Transformer V2-B, and Vision Mamba-S with quality-aware fusion and metadata, reporting internal-test AUROCs of 0.699–0.833 across six diseases and demonstrating a primary-care silent trial.

Despite this progress, three methodological gaps remain actionable without UKB access. First, Reti-Pioneer freezes quality weights to good=1 / usable=0.5 / bad=0, which may be suboptimal when quality label distributions differ across cameras and sites. Second, training six independent binary models ignores label correlation (e.g., diabetes-related and hypertensive findings) that multi-label ocular datasets naturally provide. Third, screening deployment needs calibrated probabilities and decision-curve net benefit, not AUROC alone.

**Contributions.**

1. A Reti-Pioneer-compatible **learnable quality routing** module initialized at the paper’s fixed weights.
2. A **shared multi-task head** with joint binary cross-entropy (BCE) over K public labels, with per-class ensemble behavior retained as an ablation.
3. An evaluation protocol that elevates **temperature scaling, ECE, Brier, and DCA net benefit**, plus **patient-level** ODIR<->BRSET head-mapped transfer.
4. An open engineering stack (configs, ablation runner, public loaders) enabling independent reproduction once images are downloaded.

**Boundary.** We do not claim UKB T2DM AUROC 0.833 replication; we do not treat synthetic feature-cache AUROCs as clinical evidence; gout / osteoporosis / hyperlipidemia / thyroid remain UKB-only until public gold standards exist.

---

## 2. Related work

**Foundation models for CFP.** RETFound and related retinal foundation models reduce labeled-data needs for ocular and oculomic tasks. Reti-Pioneer ensembles multiple frozen backbones rather than proposing a new foundation model; our work inherits that stance.

**Quality-aware analysis.** EyeQ-style classifiers produce good/usable/bad probabilities. Prior screening studies show community CFPs are often degraded; Reti-Pioneer’s bilinear quality fusion is a strong baseline. Learnable routing is a bounded refinement, not a new quality taxonomy.

**Multi-label fundus datasets.** ODIR-5K, RFMiD, and BRSET provide multi-label ocular (and some systemic) annotations downloadable by reviewers. They are **not** UKB ICD endpoints; Methods must state label semantics explicitly.

**Calibration and decision curves.** Temperature scaling, ECE, and DCA are standard for clinical utility assessment. RETFound-enhanced community screening studies have used DCA; we adopt the same utility language for public oculomics baselines.

**Positioning.** This is a **methods paper on top of Reti-Pioneer**, not a competitor foundation model and not a Nature Medicine–scale clinical claim without biobank data.

---

## 3. Methods

### 3.1 Inputs

Paired or single CFPs, metadata (age, sex; weight/ethnicity padded when missing), and a 3-way quality probability vector. Patient IDs define folds so both eyes of one person stay in one split.

### 3.2 Frozen backbones and feature cache

Following Reti-Pioneer, we freeze RETFound / Swin V2-B / Vision Mamba-S and train on pre-extracted features (`fast_mode`). Public caches are built via `prepare_public_npz.py` then `extract_features.py`.

### 3.3 Quality routing

`QualityAware` implements bilinear fusion. With `learnable_q=False`, `q_fc` is frozen at (1, 0.5, 0). With `learnable_q=True`, the same initialization is used but `q_fc` is trainable. Optional supervised quality on BRSET focus/illumination/artifact labels is deferred (待补充).

### 3.4 Multi-task head

`ComplexModel` supports `num_classes=K`. Joint BCE (with optional `pos_weight`) shares the fused representation across labels. Independent single-disease training remains the Reti-Pioneer-clone control.

### 3.5 Ensemble

Train-time softmax soft voting and eval-time max across three backbone heads are retained; temperature-scaled mean is an alternative (`--ensemble temp_mean`).

### 3.6 Calibration and decision utility

Temperature T is fit on validation logits. We report AUROC / average precision (AP), ECE, Brier, sensitivity at high specificity, and net benefit at threshold 0.10 versus treat-all / treat-none.

### 3.7 Training protocol

Learning rate 1e-4 (config default), frozen backbones, patient-level split, seeds {{42,43,44}} for final tables (待补充 on real data). Ablation arms: baseline, learnable_q, multitask, full.

### 3.8 Label mapping (pre-registered)

| Head | ODIR | BRSET | Honesty |
|------|------|-------|---------|
| diabetes-related | D (ocular) | diabetes / DR-related mapping | Not UKB T2DM ICD |
| hypertension-ocular | H | hypertensive retinopathy | Ocular sign ≠ systemic HTN ICD |

---

## 4. Experiments (protocol)

- **Datasets:** ODIR-5K, BRSET, RFMiD (download status: 待补充 for full images).
- **Baselines:** linear probe on RETFound-only (待补充); Reti-Pioneer clone (fixed q, independent/single-task); each add-on.
- **Primary endpoint:** macro AUROC on frozen test split (real data).
- **Co-primary:** ECE and net benefit @ 10% after temperature scaling.
- **Transfer:** train ODIR, test BRSET overlapping heads; reverse.
- **Subgroups:** age tertiles, sex on BRSET (待补充).

**Honesty rule:** never place synthetic/demo AUROC in submission tables.

---

## 5. Results

**Clinical result tables: 待补充** (require real ODIR/BRSET/RFMiD pixels + foundation feature extraction).

For engineering verification only, a synthetic ablation matrix exists in `results/ablation_summary.csv` (see companion report). Those values are **pipeline sanity**, typically near chance on tiny synthetic caches, and must not be compared to Reti-Pioneer’s 0.699–0.833 UKB internal AUROCs.

**Expected qualitative story after real features (hypothesis, not claim):** modest AUROC gains from multi-task correlation; larger ECE reductions from temperature scaling; quality routing helps more when true quality labels exist (BRSET); cross-domain drop remains large and must be discussed.

---

## 6. Discussion

Public ocular+systemic labels enable reproducible methods claims that UKB-gated papers cannot always support. Transferring Reti-Pioneer’s skeleton to ODIR/BRSET forces explicit label semantics and domain-shift reporting. Fairness and calibration should be first-class, matching the authors’ own limitations discussion.

**Figure reading contract (for companion report figures; manuscript clinical panels 待补充).** Figure 1 is an information-flow schematic only (yellow = learnable quality fusion; green = multi-task + calibration/DCA). Figures 2–4 currently plot **SYNTHETIC** feature-cache metrics near chance with tiny *n*; they justify engineering readiness and the *protocol* of reporting ECE and cross-dataset drop, not clinical superiority over Reti-Pioneer’s UKB AUROCs.

**Honest novelty phrasing.** Prefer: “we unfreeze quality routing initialized at Reti-Pioneer’s fixed weights and evaluate jointly with multi-label heads and calibration/DCA on public cohorts.” Avoid: “we outperform Reti-Pioneer” or “we achieve AUROC X on T2DM” without the original UKB endpoint and cohort.

---

## 7. Limitations

No UKB / SEED access in this workspace; label mismatch vs ICD systemic endpoints; no prospective trial; Vision Mamba optional on Windows; EyeQ weights external; synthetic caches for CI only.

---

## 8. Reproducibility

Code, configs (`configs/ablation_*.yaml`), `scripts/run_ablations.py`, patient-level split, and `docs/REPRODUCIBILITY.md`. BRSET requires PhysioNet credentialing.

```powershell
python -m unittest discover -s tests -v
python scripts/plot_paper_figures.py
python scripts/run_ablations.py --quick
```

---

## References (seed list; expand on real submission)

1. Zhang et al. AI framework for multidisease detection via retinal imaging. *Nat Med* (2026). doi:10.1038/s41591-026-04359-w  
2. Zhou et al. A foundation model for generalizable disease detection from retinal images. *Nature* (2023). doi:10.1038/s41586-023-06555-x  
3. Fu et al. Evaluation of retinal image quality assessment systems (EyeQ). *MICCAI* workshops / related EyeQ resources.  
4. ODIR-5K Grand Challenge / dataset documentation.  
5. BRSET: Brazilian Multilabel Ophthalmological Dataset. *PLOS Digit Health* (doi:10.1371/journal.pdig.0000454); PhysioNet.  
6. RFMiD / RFMiD 2.0 multi-disease fundus datasets.  
7. Guo et al. On calibration of modern neural networks. *ICML* 2017 (temperature scaling).  
8. Vickers & Elkin. Decision curve analysis. *Med Decis Making* 2006.  
9. RETFound-enhanced community screening with DCA (e.g., PubMed 38693205).  
10. Official Reti-Pioneer code: https://github.com/lyhyl/Reti-Pioneer  

---

## Assumptions or missing inputs

- Real ODIR/BRSET/RFMiD downloads and GPU feature extraction not completed in this workspace.  
- ChatGPT Pro/Plus live consult was attempted but Cursor browser tabs failed to stick; literature outline above was independently verified via web search + `docs/PAPER_PLAN.md`.  
- No invented UKB results.
"""


def report_md(rows: list[dict], uris: dict[str, str]) -> str:
    table = ablation_table_md(rows)
    return f"""# 研究报告：基于 Reti-Pioneer 的可学习质量路由与多任务眼底筛查方法扩展

**日期：** {date.today().isoformat()}  
**项目路径：** `E:\\Projects\\20260522-retinal-imaging`  
**性质：** 方法学跟进 / 工程可复现报告（非临床试验报告）

<span class="badge">SYNTHETIC metrics ≠ clinical AUROC</span>

## 封面信息

| 项 | 内容 |
|----|------|
| 基线论文 | Zhang et al., Nat Med 2026, Reti-Pioneer, DOI 10.1038/s41591-026-04359-w |
| 本文定位 | Methods extension：learnable_q + multitask + calibration/DCA + 公开队列验证 |
| 目标期刊（无 UKB） | npj Digital Medicine / MedIA / IEEE JBHI·TMI / CIBM |
| 写作架构 | Nature-family methods 论证链（nature-writing skill） |
| 数据诚实性 | ODIR-D ≠ UKB T2DM；合成特征 AUROC 仅流水线自检 |

## 目录

1. 摘要  
2. 背景与问题  
3. 方法  
4. 实施过程  
5. 结果（含图/表详解）  
6. 讨论  
7. 结论  
8. 局限与待补充  
9. 缩略语  

---

## 1. 摘要

Reti-Pioneer 证明了冻结视觉基础模型特征 + 质量感知融合可用于系统性代谢病筛查，但其质量权重固定、疾病头相互独立，且公开可复核的校准/决策曲线证据不足。本仓库在保留 Reti-Pioneer 骨架的前提下，实现可学习质量路由（learnable quality routing）、共享多标签头（multi-task head），并把温度缩放（temperature scaling）、期望校准误差（ECE）、Brier 分数与决策曲线净收益（net benefit）纳入评估。公开队列采用 ODIR-5K、BRSET、RFMiD 的患者级划分与头映射。**当前数值结果来自合成特征缓存，仅用于流水线健全性检查，不得作为论文临床性能。** 真实图像与基础模型特征提取完成后，结果表标记为待补充处应整体替换。

---

## 2. 背景与问题

### 2.1 缩略语首次展开

- **CFP**（color fundus photograph）：彩色眼底照片。  
- **AUROC**（area under the receiver operating characteristic curve）：受试者工作特征曲线下面积。  
- **ECE**（expected calibration error）：期望校准误差。  
- **DCA**（decision curve analysis）：决策曲线分析。  
- **UKB**（UK Biobank）：英国生物银行（受控访问，本报告不声称已复现其结果）。  
- **ODIR**（Ocular Disease Intelligent Recognition）：眼病智能识别公开数据集。  
- **BRSET**：巴西多标签眼科数据集。  
- **RFMiD**：视网膜眼底多疾病图像数据集。  
- **BCE**（binary cross-entropy）：二元交叉熵。  

### 2.2 Reti-Pioneer 已完成与未完成

已完成：三骨干冻结集成、固定质量权重融合、六病种独立二分类、UKB+医院队列、外部验证与静默试验。  
作者自述缺口：临床广泛落地精度仍不足；5/10 年任务为固定切点而非生存分析；未覆盖心脑卒中/死亡等；残余混杂与族群失衡；需更大 RCT。  

### 2.3 本跟进可诚实声称的创新点

| ID | 创新 | 相对 Reti-Pioneer | 本仓库状态 |
|----|------|-------------------|------------|
| I | 可学习质量路由 | 解冻 `q_fc`，初始化仍为 1/0.5/0 | 已实现 |
| II | 共享多任务头 | 联合 K 类 BCE | 已实现 |
| III | 校准 + DCA 作为效用共主终点 | 温度缩放、ECE、Brier、NB@0.10 | 已实现 |
| IV | 患者级跨库泛化 | ODIR <-> BRSET 重叠头 | CLI 已实现；真实数待补充 |
| VI | 公平性切片 | 年龄/性别亚组 AUROC | 评估器支持；真实数待补充 |

**不可卖点：** 用合成 AUROC 对比论文 0.833；把 ODIR-D 叫作 UKB T2DM；无 UKB 时冲击 *Nature Medicine* 同级临床主张。

---

## 3. 方法

详见论文稿 `docs/paper/manuscript.md`。工程映射：

- `model/QualityAware.py`：`learnable_q`  
- `model/RetiPioneer.py`：`num_classes>1`  
- `scripts/train.py` / `evaluate.py` / `run_ablations.py`  
- `utils/calibration.py`  
- `dataset/odir.py`, `brset.py`, `rfmid.py`  
- `reti_pioneer/split.py`：患者级划分  

写作架构选择：**Nature-family methods paper**（问题→既有方法边界→新方法→公平消融→可复现→边界），而非旗舰 *Nature* 临床综述式叙事。文献锚点经独立检索核实：Reti-Pioneer (Nat Med 2026)、RETFound (Nature 2023)、BRSET (PLOS Digit Health)、温度缩放与 DCA 经典方法、以及社区筛查中的 DCA 应用。

---

## 4. 实施过程

1. 阅读 README、PAPER_PLAN、PUBLIC_DATA、既有 chatgpt-runs 与 `results/`。  
2. 安装 SciencePlots；脚本 `scripts/plot_paper_figures.py` 用 Times New Roman + science 样式重绘消融图（英文轴标签，中文说明放图注）。  
3. 按 nature-writing（methods）起草 `docs/paper/manuscript.md` 与 HTML。  
4. 生成自包含 `docs/report/report.html`（Base64 嵌图）。  
5. ChatGPT：Cursor 内置浏览器标签无法稳定创建/导航（反复 “No browser tab available”）；已用系统默认浏览器打开 chatgpt.com，并落盘 `TASK_BRIEF.md` 供用户粘贴；顾问结论以本地文献检索+ PAPER_PLAN 独立裁定。  

---

## 5. 结果

### 表 1. 合成消融矩阵（流水线自检）

{table}

**如何读表：**  
- `auroc`：多任务时为宏平均；单任务时为对应头。  
- `auroc_D`：ODIR 的 D 头（或映射后的糖尿病相关头），便于跨臂比较。  
- `ece` / `cal_ece`：温度缩放前后期望校准误差；**校准不改变排序鉴别力（AUROC）**，但可改善概率可靠性。  
- `nb@0.10`：阈值 0.10 处净收益，需与 treat-all 对照解读。  
- 样本量 `n` 极小（验证 12 / 跨库 48），曲线极不稳定。  

**结论（仅限合成自检）：** 数值多在机会水平附近波动；multitask/full 在合成缓存上宏 AUROC 更低，**不能**解释为真实方法失败或成功；温度缩放可降低 ECE（例如 full 臂 ODIR val raw ECE≈0.647 → cal ECE≈0.370）。**待补充：真实特征复跑后替换本表。**

### 图 1. 方法总览

![Figure 1](fig1_architecture.png)

**来龙去脉（问什么 / 怎么读 / 含义 / 结论）：**  
- **问什么：** 不改冻结骨干时，相对 Reti-Pioneer 的方法增量落在哪？  
- **怎么读：** 蓝框=共享骨架（输入与三骨干）；黄框=质量融合（固定 q vs learnable_q）；绿框=共享多任务头 + 温度缩放/DCA。  
- **含义：** 示意图只编码信息流与消融位置，**不含** AUROC/ECE。  
- **结论：** 可卖点集中在黄/绿；骨干不是新贡献。真实特征跑通后可在绿框旁标注主终点/共主终点，但仍勿在示意图写性能数字。  

### 图 2. 消融柱状图（SYNTHETIC）

![Figure 2](fig2_ablation_bars.png)

**来龙去脉：**  
- **问什么：** 四臂在鉴别力与校准误差上是否至少“跑通”？  
- **怎么读：** 左图 D 头 AUROC（深蓝 ODIR val / 浅蓝 ODIR→BRSET），虚线 0.5=随机；右图 ECE（橙 raw / 绿 calibrated）。  
- **含义：** n 极小（val=12，跨库=48）且特征合成时，柱高接近机会水平属预期；绿柱低于橙柱仅说明校准链路可压低 ECE。  
- **结论：** 流水线健全性证据，**禁止**与 Nat Med 内部 AUROC 0.699–0.833 比较。**待补充：** 真实特征复跑替换。  

### 图 3. 温度缩放对 ECE 的影响（SYNTHETIC）

![Figure 3](fig3_calibration.png)

**来龙去脉：**  
- **问什么：** 为何把校准写成共主终点？  
- **怎么读：** 圆点 raw ECE、方点 calibrated；标注 Delta≈raw−cal。  
- **含义：** 温度缩放不改变 AUROC，只调概率尖锐度；合成数据上 Delta 仍为正，支持协议层必须报告 ECE/Brier/NB。  
- **结论：** 方法学主张在“评估协议”，不等于临床筛查已可用。**待补充：** 可靠性图与 NB 曲线。  

### 图 4. 跨数据集落差（SYNTHETIC）

![Figure 4](fig4_cross_domain.png)

**来龙去脉：**  
- **问什么：** 只报源域是否过度乐观？  
- **怎么读：** 实线 ODIR val，虚线 ODIR→BRSET；红填充=域差距。  
- **含义：** 强制跨库叙事，把 domain shift 写进结果。  
- **结论：** 公开队列论文必须以患者级跨库协议约束卖点；本图数值为 SYNTHETIC。**待补充：** 真实跨库幅度与 RFMiD。

---

## 6. 讨论

1. **创新点可信边界：** I–III 有代码与消融协议支撑；临床增益必须等真实公开队列。  
2. **标签语义：** 公开数据适合方法学复核，不适合直接对标 Nat Med 六病种 AUROC。  
3. **ChatGPT 顾问角色：** 本次未能稳定完成对话；大纲以 PAPER_PLAN + 独立检索为准，不虚构顾问“已批准”表述。  

---

## 7. 结论

本仓库完成了面向公开队列的 Reti-Pioneer 方法学扩展工程与论文/报告初稿，并用 SciencePlots 重绘了合成消融图。下一步关键路径：下载真实 ODIR/BRSET/RFMiD → 提取基础模型特征 → 复跑消融与校准表 → 再将 待补充 替换为可投稿数字。

---

## 8. 局限与待补充

- 待补充：真实图像、EyeQ 权重、UKB 申请、三随机种子正式表、亚组公平性、生存分析。  
- Windows 上 Vision Mamba / WeasyPrint 系统库可能受限；PDF 使用 fpdf2 + 本地字体导出。  
- 公开 GitHub 快照仅含代码与文档（及合成消融 CSV/小图），不含患者影像、`.env`、ckpt 或大结果缓存。  
- ChatGPT 顾问建议须独立核实；不虚构临床 AUROC。  

---

## 9. 参考文献（精简）

同论文稿 References；完整条目见 `docs/paper/manuscript.md`。
"""


def md_images_to_embedded_html(md: str, uris: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        key = Path(src).name
        uri = uris.get(key)
        if not uri:
            return f"<p class='pending'>[缺失图片: {html.escape(src)}]</p>"
        return (
            f"<figure><img alt=\"{html.escape(alt)}\" src=\"{uri}\"/>"
            f"<figcaption class='longcap'><strong>{html.escape(alt)}</strong></figcaption></figure>"
        )

    # convert very lightly
    text = md
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, text)
    # headings
    lines_out = []
    for line in text.splitlines():
        if line.startswith("# "):
            lines_out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines_out.append(f"<h2 id=\"{html.escape(line[3:][:40])}\">{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines_out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("|") and "---" not in line:
            # leave tables to separate injection; keep as pre for md tables already converted in report path
            lines_out.append(line)
        elif line.strip() == "---":
            lines_out.append("<hr/>")
        elif line.startswith("> "):
            lines_out.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        elif line.strip() == "":
            lines_out.append("")
        else:
            # bold / code light
            s = html.escape(line)
            s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\\1</strong>", s)
            s = s.replace("\\\\1", r"\1")  # fix
            s = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", html.escape(line))
            s = re.sub(r"`([^`]+)`", lambda m: f"<code>{html.escape(m.group(1))}</code>", s)
            # Actually double-escaped — redo simply:
            raw = line
            esc = html.escape(raw)
            esc = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", raw)
            # mixed: escape then restore tags — simpler paragraph:
            p = html.escape(raw)
            p = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<strong>{html.escape(m.group(1))}</strong>", raw)
            # final approach
            def fmt_inline(t: str) -> str:
                parts = []
                i = 0
                while i < len(t):
                    if t.startswith("**", i):
                        j = t.find("**", i + 2)
                        if j != -1:
                            parts.append("<strong>" + html.escape(t[i + 2 : j]) + "</strong>")
                            i = j + 2
                            continue
                    if t[i] == "`":
                        j = t.find("`", i + 1)
                        if j != -1:
                            parts.append("<code>" + html.escape(t[i + 1 : j]) + "</code>")
                            i = j + 1
                            continue
                    parts.append(html.escape(t[i]))
                    i += 1
                return "".join(parts)

            if line.startswith("- "):
                lines_out.append(f"<li>{fmt_inline(line[2:])}</li>")
            else:
                lines_out.append(f"<p>{fmt_inline(line)}</p>")
    # wrap consecutive li
    html_body = "\n".join(lines_out)
    html_body = re.sub(r"(?:<li>.*?</li>\n?)+", lambda m: "<ul>" + m.group(0) + "</ul>", html_body)
    return html_body


def build_report_html(rows: list[dict], uris: dict[str, str]) -> str:
    fig_blocks = []
    captions = {
        "fig1_architecture.png": (
            "图 1 方法总览（来龙去脉）",
            "问什么：在不改动冻结骨干的前提下，方法增量落在何处？"
            "怎么读：左蓝框=输入（CFP/元数据/质量概率）；中蓝框=与 Reti-Pioneer 共享的冻结 RETFound/Swin/Vim；"
            "黄框=质量感知双线性融合（固定 q vs learnable_q）；绿框=共享多任务头 + 温度缩放/DCA。"
            "曲线/框含义：本图无性能数字，只编码信息流与消融位置。"
            "结论：相对基线论文，可卖点集中在黄/绿两框；蓝色骨干不是新贡献。"
            "待补充：真实特征跑通后可在绿框旁加“主终点/共主终点”标注，但仍勿在示意图上写 AUROC。",
        ),
        "fig2_ablation_bars.png": (
            "图 2 合成消融柱状图（来龙去脉）",
            "问什么：四臂（baseline / learnq / multitask / full）在鉴别力与校准误差上是否“跑通”？"
            "怎么读：左图 D 头 AUROC——深蓝=ODIR val，浅蓝=ODIR→BRSET；虚线 0.5=随机参考。"
            "右图 ECE——橙=原始，绿=温度缩放后。"
            "曲线含义：柱高接近 0.5 且 n 极小（val=12，跨库=48）时，臂间差异不可作方法优劣证据；"
            "右图若绿柱低于橙柱，仅说明校准链路对合成 logits 仍可压低 ECE。"
            "结论：流水线健全；禁止与 Nat Med 内部 AUROC 0.699–0.833 比较。"
            "待补充：真实 ODIR/BRSET 特征复跑后整图替换。",
        ),
        "fig3_calibration.png": (
            "图 3 温度缩放对 ECE 的影响（来龙去脉）",
            "问什么：把校准写成共主终点是否有方法学动机？"
            "怎么读：横轴消融臂，纵轴 ECE；圆点=raw，方点=calibrated；标注 Delta≈raw−cal。"
            "曲线含义：温度缩放不改变排序鉴别力（AUROC 不变），只调整概率尖锐度；"
            "合成数据上 Delta 仍为正，支持“协议层必须报告 ECE/Brier/NB”。"
            "结论：方法学主张成立于协议设计，不等于临床筛查已校准可用。"
            "待补充：真实验证集上的最优 T、可靠性图（reliability diagram）与 NB 曲线。",
        ),
        "fig4_cross_domain.png": (
            "图 4 跨数据集 AUROC 落差（来龙去脉）",
            "问什么：只报源域指标是否会过度乐观？"
            "怎么读：实线=ODIR val D 头，虚线=ODIR→BRSET 映射头；红填充=域差距。"
            "曲线含义：即使合成缓存也应强制画跨库落差，把 domain shift 写进结果叙事。"
            "结论：公开队列论文必须以跨库/患者级协议约束卖点；本图数值为 SYNTHETIC。"
            "待补充：真实跨库幅度、反向 BRSET→ODIR、以及 RFMiD 重叠头。",
        ),
    }
    for name, (title, longcap) in captions.items():
        uri = uris[name]
        fig_blocks.append(
            f"""
<figure id="{name}">
  <img src="{uri}" alt="{html.escape(title)}"/>
  <figcaption class="longcap">
    <strong>{html.escape(title)}</strong>（SYNTHETIC where metrics apply）<br/>
    {html.escape(longcap)}
  </figcaption>
</figure>
"""
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>研究报告 — Reti-Pioneer 方法扩展</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="wrap">
  <header class="cover">
    <div><span class="badge">SYNTHETIC AUROC ≠ 临床性能</span><span class="badge">无 UKB 结果</span></div>
    <h1>研究报告：Quality-Adaptive Multi-Task Oculomics<br/>（Reti-Pioneer 方法学扩展）</h1>
    <p class="meta">日期：{date.today().isoformat()} · 仓库：20260522-retinal-imaging · 写作技能：nature-writing（methods）</p>
    <p>本报告自包含（CSS 内联、图片 Base64）。图表中的性能数字来自合成特征缓存，仅流水线自检；论文可比临床 AUROC <span class="pending">待补充</span>。</p>
  </header>

  <nav class="toc">
    <strong>目录</strong>
    <ol>
      <li><a href="#abstract">摘要</a></li>
      <li><a href="#bg">背景</a></li>
      <li><a href="#methods">方法</a></li>
      <li><a href="#process">过程</a></li>
      <li><a href="#results">结果</a></li>
      <li><a href="#discussion">讨论</a></li>
      <li><a href="#conclusion">结论</a></li>
      <li><a href="#limits">局限</a></li>
    </ol>
  </nav>

  <section id="abstract">
    <h2>1. 摘要</h2>
    <p>Reti-Pioneer（Zhang 等，<em>Nature Medicine</em>，2026）展示了冻结基础模型特征与质量感知融合用于系统性代谢病筛查的可行性，但采用<strong>固定</strong>质量权重与<strong>独立</strong>二分类头，且公开可复核的校准/决策曲线协议不足。本工作在保留其骨架前提下实现：可学习质量路由（learnable quality routing）、共享多任务头（multi-task head），以及温度缩放（temperature scaling）、期望校准误差（ECE）、Brier 分数与决策曲线（DCA）净收益评估，并规划在 ODIR-5K、BRSET、RFMiD 等公开队列上做患者级验证。当前消融数字为<strong>SYNTHETIC</strong>，不得与论文内部检验 AUROC 0.699–0.833 比较。</p>
  </section>

  <section id="bg">
    <h2>2. 背景</h2>
    <p class="abbr">缩略语首次展开：CFP=彩色眼底照片；AUROC=受试者工作特征曲线下面积；ECE=期望校准误差；DCA=决策曲线分析；UKB=英国生物银行；BCE=二元交叉熵。</p>
    <p>眼底微血管改变可反映全身代谢与血管状态（oculomics）。RETFound（Zhou 等，<em>Nature</em>，2023）提供可迁移视网膜表征；Reti-Pioneer 进一步集成多骨干与质量感知融合，并在 UKB/医院数据上报告六病种筛查性能与静默试验。作者讨论指出精度、纵向建模、病种范围、混杂与前瞻验证仍有缺口。本跟进选择其中可用<strong>公开数据消融</strong>关闭的方法学缺口（质量路由、多任务、校准/效用、跨库泛化），而不是重复实现原论文。</p>
    <div class="warnbox">诚实边界：ODIR「Diabetes」主要为眼部证据/糖尿病视网膜病变相关标签，<strong>不是</strong> UKB ICD 式 2 型糖尿病；不得用合成 AUROC 宣称超越 0.833。</div>
  </section>

  <section id="methods">
    <h2>3. 方法</h2>
    <h3>3.1 写作与实验架构</h3>
    <p>采用 Nature-family <strong>methods paper</strong> 论证链：任务/问题 → 既有方法边界 → 提议方法 → 公平消融与效用指标 → 可复现证据 → 边界。目标期刊定位数字医学/医学影像方法刊物，而非无 UKB 时冲击旗舰 <em>Nature Medicine</em>。</p>
    <h3>3.2 模型改动</h3>
    <ul>
      <li><code>QualityAware(learnable_q=True)</code>：以 1/0.5/0 初始化并解冻 <code>q_fc</code>。</li>
      <li><code>ComplexModel(num_classes=K)</code>：共享融合 + 联合 BCE。</li>
      <li><code>evaluate.py --calibrate</code>：验证集拟合温度 T，报告 ECE/Brier/NB@0.10。</li>
      <li>患者级划分避免同一患者双眼泄漏到训练与测试。</li>
    </ul>
    <h3>3.3 消融臂</h3>
    <p>baseline（固定 q，单任务）→ learnq → multitask → full（learnq+multitask，评估含校准）。</p>
  </section>

  <section id="process">
    <h2>4. 过程</h2>
    <ol>
      <li>仓库基线核对；按用户授权准备<strong>公开</strong> GitHub 代码+文档快照（排除 data/ckpt/results 大缓存与密钥）。</li>
      <li>安装/确认 SciencePlots；运行 <code>scripts/plot_paper_figures.py</code>（Times New Roman + science 样式；中文说明放图注）重绘结果图。</li>
      <li>nature-writing（methods）升级论文稿与本报告；图注按「问什么 / 怎么读 / 含义 / 结论 / 待补充」加深。</li>
      <li>ChatGPT：粘贴文本简报 + 公开仓库 URL（启用 web search）；顾问建议经独立核实后才采纳。</li>
    </ol>
  </section>

  <section id="results">
    <h2>5. 结果</h2>
    <div class="warnbox">表 1 与图 2–4 中所有性能数字均为 <strong>SYNTHETIC FEATURE CACHE</strong> 流水线自检，样本极小。论文主表 <span class="pending">待补充</span>。不得与 UKB 临床 AUROC 比较。</div>
    <h3>表 1. 消融摘要</h3>
    {ablation_table_html(rows)}
    <p><strong>如何读表：</strong> <code>auroc_D</code> 便于跨臂对比糖尿病相关头；ECE 降而 AUROC 不变说明校准改善的是概率质量；NB@0.10 必须相对 treat-all 解释。合成结果接近随机，仅证明 train/eval/校准链路可运行。</p>
    <h3>图注阅读约定</h3>
    <p>每幅图按同一模板解读：<em>问什么 → 怎么读 → 曲线/框含义 → 结论 → 待补充</em>。图 1 无性能数字；图 2–4 凡涉及 AUROC/ECE 均标注 SYNTHETIC。</p>
    {''.join(fig_blocks)}
  </section>

  <section id="discussion">
    <h2>6. 讨论</h2>
    <p>方法学创新点 I–III 与仓库实现一致，具备可投稿叙事潜力；但<strong>证据强度</strong>目前停在工程与合成自检层。公开队列一旦接入，应优先报告：相对 Reti-Pioneer-clone 的同数据提升、校准改善、跨库落差与亚组公平性，而不是与 UKB 内部 AUROC 硬比。</p>
    <p>诚实措辞优先使用 “unfreeze quality routing initialized at fixed weights + multi-label + calibration/DCA on public cohorts”，避免在无 UKB 终点时声称 “outperform Reti-Pioneer”。</p>
  </section>

  <section id="conclusion">
    <h2>7. 结论</h2>
    <p>已形成可复核的方法扩展论文草稿、SciencePlots 图件与自包含研究报告，并准备公开代码+文档供外部顾问阅读。真实性能数字仍为 <span class="pending">待补充</span>；合成指标不得进入投稿主表。</p>
  </section>

  <section id="limits">
    <h2>8. 局限</h2>
    <ul>
      <li>无 UKB/SEED；无真实 ODIR/BRSET 像素特征（待补充）。</li>
      <li>公开快照不含私有数据、权重与大结果缓存；未上传任何患者级影像。</li>
      <li>ChatGPT 顾问输出需独立核实；浏览器自动化若失败则以 TASK_BRIEF 人工粘贴兜底。</li>
    </ul>
  </section>
</div>
</body>
</html>
"""


def build_manuscript_html(md_text: str, uris: dict[str, str]) -> str:
    # embed figures referenced by filename if any; manuscript mostly text
    body = []
    for line in md_text.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("|"):
            body.append(f"<pre>{html.escape(line)}</pre>")
        elif line.strip() == "---":
            body.append("<hr/>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        elif line.startswith("- ") or line.startswith("* "):
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip() == "":
            body.append("")
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    # insert architecture figure after Methods header area
    fig = (
        f"<figure><img src=\"{uris['fig1_architecture.png']}\" alt=\"Figure 1\"/>"
        f"<figcaption>Figure 1. Method overview (English labels; SYNTHETIC metrics elsewhere).</figcaption></figure>"
    )
    joined = "\n".join(body)
    joined = joined.replace("<h2>3. Methods</h2>", "<h2>3. Methods</h2>\n" + fig, 1)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Manuscript — Quality-Adaptive Multi-Task Oculomics</title>
<style>{CSS}</style>
</head>
<body><div class="wrap"><article>
{joined}
</article></div></body></html>
"""


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.set_text_color(100)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} · SYNTHETIC metrics · {date.today().isoformat()}", align="C")


def write_pdf(report_text: str, fig_paths: list[Path], out: Path) -> None:
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    use_cjk = FONT_CJK.exists()
    if use_cjk:
        pdf.add_font("CJK", fname=str(FONT_CJK))
        font = "CJK"
    else:
        font = "Helvetica"
    pdf.set_font(font, size=11)
    usable = pdf.epw

    def put(text: str, size: int = 11, lh: float = 5.5) -> None:
        pdf.set_font(font, size=size)
        # Skip markdown table separators / ultra-wide pipe rows → summarize
        if text.startswith("|") and text.count("|") > 4:
            text = re.sub(r"\s*\|\s*", " · ", text.strip("| "))
        if set(text.strip()) <= {"-", "|", " ", ":"}:
            return
        # Soft-wrap very long lines
        while text:
            pdf.multi_cell(usable, lh, text[:2000])
            text = text[2000:]

    plain = re.sub(r"!\[.*?\]\(.*?\)", "[Figure pages follow]", report_text)
    plain = plain.replace("**", "").replace("`", "")
    plain = (
        plain.replace("\u2194", "<->")
        .replace("\u2212", "-")
        .replace("\u0394", "Delta")
        .replace("\u2248", "~")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
    )
    for line in plain.splitlines():
        if line.startswith("# "):
            put(line[2:], size=16, lh=8)
            pdf.ln(2)
        elif line.startswith("## "):
            put(line[3:], size=13, lh=7)
            pdf.ln(1)
        elif line.startswith("### "):
            put(line[4:], size=12, lh=6)
        elif line.strip() == "":
            pdf.ln(2)
        else:
            put(line)

    for p in fig_paths:
        if not p.exists():
            continue
        pdf.add_page()
        put(p.stem, size=12, lh=6)
        pdf.image(str(p), w=min(180, usable))

    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))


def write_manuscript_pdf(md_text: str, fig_paths: list[Path], out: Path) -> None:
    write_pdf(md_text, fig_paths[:1], out)


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    uris = load_uris()

    ms = manuscript_md()
    (PAPER / "manuscript.md").write_text(ms, encoding="utf-8")
    (PAPER / "manuscript.html").write_text(build_manuscript_html(ms, uris), encoding="utf-8")

    rm = report_md(rows, uris)
    (REPORT / "report.md").write_text(rm, encoding="utf-8")
    (REPORT / "report.html").write_text(build_report_html(rows, uris), encoding="utf-8")

    figs = [
        ASSETS / "fig1_architecture.png",
        ASSETS / "fig2_ablation_bars.png",
        ASSETS / "fig3_calibration.png",
        ASSETS / "fig4_cross_domain.png",
    ]
    write_pdf(rm, figs, REPORT / "report.pdf")
    write_manuscript_pdf(ms, figs, PAPER / "manuscript.pdf")

    # also copy outline
    outline = PAPER / "OUTLINE.md"
    outline.write_text(
        """# Adopted writing architecture

**Emulate:** Nature-family *methods* article (argument: gap → method → fair ablation → calibration/utility → reproducibility → boundary).

**Do not emulate:** flagship *Nature Medicine* clinical discovery narrative requiring UKB-scale cohorts and prospective pilots as primary claims.

**Innovation claims (bounded):** learnable quality routing; shared multi-task head; calibration + DCA as co-primary; public ODIR/BRSET/RFMiD patient-level validation.

**Non-claims:** synthetic AUROC; beating 0.833 T2DM; ODIR-D as UKB T2DM.
""",
        encoding="utf-8",
    )
    print("Wrote manuscript + report HTML/MD/PDF")


if __name__ == "__main__":
    main()
