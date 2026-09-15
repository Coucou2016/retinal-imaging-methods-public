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
    """Fallback English MS template — keep aligned with ``docs/paper/manuscript.md``.

    ``main()`` prefers the on-disk manuscript when present; this body exists so a
    regenerating wipe cannot resurrect the pre-review title / single-task framing.
    """
    ms_path = PAPER / "manuscript.md"
    if ms_path.is_file() and ms_path.stat().st_size > 500:
        return ms_path.read_text(encoding="utf-8")
    return f"""# Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning

**Manuscript draft (methods paper)** · {date.today().isoformat()}  
**Axes:** task=manuscript · paper_type=methods · language=en · journal=generic (Nature-family structure; not a flagship *Nature Medicine* clinical claim)  
**Realistic venues without UK Biobank:** *npj Digital Medicine*, *Medical Image Analysis*, *IEEE JBHI / TMI*, *Computers in Biology and Medicine*

> **One-sentence argument.** Reti-Pioneer showed that frozen foundation features plus quality-aware fusion can screen systemic disease from color fundus photographs (CFPs); we keep that skeleton, replace fixed quality weights with a **monotone bounded quality router**, replace independent binary heads in the **released training code** with **partial-label multi-task learning**, and evaluate with an **endpoint-aware cross-cohort** protocol plus a calibration/decision-curve **evaluation framework** (not claimed as methodological novelty) on public cohorts reviewers can download — without inventing UK Biobank-comparable AUROCs from synthetic demo caches.

---

## Abstract

Endocrine and metabolic disease screening from retinal imaging (oculomics) remains limited by image-quality heterogeneity, single-task training loops, and poorly calibrated risk scores. Reti-Pioneer (Zhang et al., *Nature Medicine*, 2026) demonstrated that frozen vision foundation models with quality-aware bilinear fusion can screen six systemic conditions on UK Biobank (UKB) and hospital cohorts. The **published** article describes a multitask screening framework; the **released training code** (`main.py`) trains **per-disease independent binary loops**. Quality routing is fixed at good=1 / usable=0.5 / bad=0. Here we propose a methods extension that (i) learns **monotone bounded** quality weights (bad ≤ usable ≤ good in [0,1]) as the default learnable path, (ii) shares a multi-label head with **masked BCE** over labels present for each sample/dataset, and (iii) reports expected calibration error (ECE), Brier score, and **per-disease decision-curve analysis (DCA)** under temperature scaling fitted on a fold **disjoint** from evaluation. We specify patient-level **endpoint-aware cross-cohort evaluation** on ODIR-5K, BRSET, and RFMiD, with explicit alignment flags (`direct` / `partial` / `related_not_equivalent`).

**Honesty banner.** Clinical AUROC / ECE / decision-curve tables are marked **待补充** until real images and foundation features replace synthetic caches. Pipeline sanity metrics on synthetic features appear only in the companion research report and are **not** manuscript claims. Calibration and DCA are an **evaluation framework**, not novelty claims. We do not invent UKB or hospital AUROCs.

**Keywords:** oculomics; fundus photography; monotone quality routing; multi-task learning; calibration (evaluation); endpoint harmonization; Reti-Pioneer

---

## 1. Introduction

**Contributions.** (1) Monotone bounded quality router. (2) Shared multi-task head with masked BCE vs released-code independent loops. (3) Endpoint ontology + endpoint-aware cross-cohort evaluation. (4) Open engineering stack (E0–E5 configs, disjoint calibration, val→test frozen operating points, patient-level bootstrap CI).

**Boundary.** We do not claim UKB T2DM AUROC 0.833 replication; synthetic feature-cache AUROCs are not clinical evidence; `diabetes_related` mappings are exploratory only.

---

## 3. Methods (summary)

Monotone router (`quality_router=monotone`); optional E5 `quality_gating`; optional BRSET `lambda_q` soft-quality aux; masked BCE; temperature on calibration_ids ⊥ eval; Youden / sens@95%spec selected on val/cal then frozen on test; patient-level bootstrap AUROC CI; per-disease DCA (NB@0.10 illustrative only).

---

## 5. Results

**Clinical result tables: 待补充** (require real ODIR/BRSET/RFMiD pixels + foundation feature extraction).

---

## Assumptions or missing inputs

- Real ODIR/BRSET/RFMiD downloads and GPU feature extraction not completed in this workspace.
- No invented UKB results.

See the full draft at `docs/paper/manuscript.md`.
"""


def report_md(rows: list[dict], uris: dict[str, str]) -> str:
    table = ablation_table_md(rows)
    return f"""# 研究报告：基于 Reti-Pioneer 的单调质量路由与端点感知多任务学习扩展

**日期：** {date.today().isoformat()}  
**项目路径：** `E:/Projects/20260522-retinal-imaging`（本机；公开 clone 路径可变）  
**性质：** 方法学跟进 / 工程可复现报告（非临床试验报告；可含过程与本地路径）  
**公开代码：** https://github.com/Coucou2016/retinal-imaging-methods-public  
**论文主线：** `docs/paper/manuscript.md`  
**真实性审查：** `docs/paper/AUTHENTICITY_AUDIT.md`  
**工作标题：** Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning

- SYNTHETIC metrics ≠ clinical AUROC
- Calibration/DCA = evaluation framework (not novelty)
- Published multitask ≠ released independent loops
- clinical_claim_allowed: false on synthetic
- no invented UKB AUROC

## 封面信息

| 项 | 内容 |
|----|------|
| 基线论文 | Zhang et al., Nat Med 2026, Reti-Pioneer, DOI 10.1038/s41591-026-04359-w |
| 本文定位 | Methods extension：monotone quality router + masked multitask + endpoint-aware cross-cohort；校准/DCA 为**评估框架** |
| 目标期刊（无 UKB） | npj Digital Medicine / MedIA / IEEE JBHI·TMI / CIBM |
| 写作架构 | Nature-family methods 论证链（nature-writing） |
| 出图 | SciencePlots + Times New Roman（`scripts/plot_paper_figures.py`） |
| 数据诚实性 | ODIR-D ≠ UKB T2DM；`diabetes_related` 仅 exploratory；合成 AUROC 仅流水线自检 |

## 1. 摘要

Reti-Pioneer **发表文本**描述多任务筛查，但**发布训练代码**为疾病独立二分类循环，质量权重固定 1/0.5/0。本仓库实现单调有界质量路由、掩码多任务 BCE、endpoint-aware cross-cohort evaluation；可选 E5 质量条件骨干路由。校准/ECE/按病种 DCA 为**评估框架**（非新颖性主卖点）；NB@0.10 仅示意。**当前数值为 SYNTHETIC 流水线自检，不得作为临床性能。**

## 2. 术语表（首次展开）

| 缩写 | 全称 |
|------|------|
| CFP | Color fundus photograph，彩色眼底照片 |
| AUROC | Area under the ROC curve |
| ECE | Expected calibration error，期望校准误差 |
| DCA | Decision curve analysis，决策曲线分析 |
| BCE | Binary cross-entropy |
| MTL | Multi-task learning |
| UKB | UK Biobank |
| ODIR / BRSET / RFMiD | 公开眼底多标签数据集 |

## 3. 创新边界

| ID | 创新 | 状态 |
|----|------|------|
| I | 单调有界质量路由（`quality_router=monotone`） | 已实现 |
| II | 掩码多任务（对照 released independent loops） | 已实现 |
| III | 端点本体 + 跨队列 alignment 守卫 | 已实现 |
| IV | 校准/DCA 评估框架（cal ⊥ eval） | 已实现 |
| E5 | 质量条件门控 + 可选 λ_q | 已实现（`configs/ablation_e5.yaml`；默认关） |

## 4. 方法与代码映射

| 概念 | 路径 |
|------|------|
| Monotone router | `model/QualityAware.py` |
| Masked BCE | `utils/run.py::masked_bce_with_logits` |
| E5 gating | `model/quality_gate.py` + `configs/ablation_e5.yaml` |
| Endpoint map | `reti_pioneer/label_map.py` |
| 消融跑数 | `scripts/run_ablations.py` → `results/ablation_summary.csv` |
| 评估 | `scripts/evaluate.py`（disclaimer / paper_mode） |

## 5. 研究过程（本机）

1. 以 2026-09-13 审稿修订后的 `docs/paper/manuscript.md` 为论文主线（未另发现独立用户上传稿）。  
2. 安装 SciencePlots；`python scripts/plot_paper_figures.py` 重绘 Fig1–4。  
3. 数字仅取自 `results/ablation_summary.csv` 与 `results/metrics_*.json`（2026-09-15T02:45 批次）。  
4. `python scripts/build_paper_report.py` 生成论文/报告 HTML·MD·PDF。  
5. 撰写 `docs/paper/AUTHENTICITY_AUDIT.md` 证据链。

## 6. 结果（SYNTHETIC；本仓库计算）

{table}

**读表：** `auroc_D` 便于跨臂对比；ECE 变化而 AUROC 不变符合温度缩放语义；n=10/48 时禁止方法优劣结论。旗标见 `data/odir|brset|rfmid/SYNTHETIC_FEATURES.txt`。

<figure><img src="{uris.get('fig1_architecture.png','')}" alt="Fig1"/><figcaption>图 1. 方法概览（示意；无性能数字）。黄框=单调质量路由；绿框=部分标签 MTL + 校准/DCA 评估。</figcaption></figure>
<figure><img src="{uris.get('fig2_ablation_bars.png','')}" alt="Fig2"/><figcaption>图 2. 消融柱状图（SYNTHETIC）。问什么：四臂是否跑通？怎么读：左 AUROC_D，右 ECE；虚线 0.5=随机。结论：流水线健全，非临床主张。</figcaption></figure>
<figure><img src="{uris.get('fig3_calibration.png','')}" alt="Fig3"/><figcaption>图 3. 温度缩放与 ECE（SYNTHETIC）。问什么：校准链路是否可压低 ECE？结论：评估框架动机成立，≠临床已校准可用。</figcaption></figure>
<figure><img src="{uris.get('fig4_cross_domain.png','')}" alt="Fig4"/><figcaption>图 4. 跨队列落差（SYNTHETIC）。问什么：是否强制报告 domain drop？结论：协议层必须画跨库，数值不可作运输性估计。</figcaption></figure>

## 7. 讨论与局限

须区分 published multitask vs released independent loops；`diabetes_related` 不得进临床主表。待补充：真实 ODIR/BRSET 像素、GPU 三骨干特征、多 seed 临床表。RFMiD 影像本地有（N=3200）但仍带 SYNTHETIC 特征旗标。

## 8. 验收与审查

- 审稿修复验收：`docs/chatgpt-runs/2026-09-13-review-fixes/ACCEPTANCE.md`  
- 真实性审查：`docs/paper/AUTHENTICITY_AUDIT.md`
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
    <div><span class="badge">SYNTHETIC AUROC ≠ 临床性能</span><span class="badge">Calibration/DCA=评估框架</span><span class="badge">无 UKB 结果</span></div>
    <h1>研究报告：单调质量路由与端点感知多任务学习<br/>（Reti-Pioneer 方法学扩展）</h1>
    <p class="meta">日期：{date.today().isoformat()} · 本机路径：E:/Projects/20260522-retinal-imaging · nature-writing（methods）· SciencePlots</p>
    <p>本报告自包含（CSS 内联、图片 Base64、无 CDN）。性能数字来自<strong>本仓库自行计算</strong>的合成特征缓存流水线；论文临床 AUROC <span class="pending">待补充</span>。审查文档：<code>docs/paper/AUTHENTICITY_AUDIT.md</code>。</p>
  </header>

  <nav class="toc">
    <strong>目录</strong>
    <ol>
      <li><a href="#abstract">摘要</a></li>
      <li><a href="#bg">背景与目的</a></li>
      <li><a href="#methods">数据与方法</a></li>
      <li><a href="#process">研究过程</a></li>
      <li><a href="#results">结果展示</a></li>
      <li><a href="#discussion">分析与讨论</a></li>
      <li><a href="#conclusion">主要结论</a></li>
      <li><a href="#limits">不足与展望</a></li>
    </ol>
  </nav>

  <section id="abstract">
    <h2>1. 摘要</h2>
    <p>Reti-Pioneer（Zhang 等，<em>Nature Medicine</em>，2026）展示了冻结基础模型特征与质量感知融合用于系统性代谢病筛查的可行性。发表文本描述多任务筛查，但<strong>发布训练代码</strong>采用疾病独立二分类循环，质量权重固定为 good=1 / usable=0.5 / bad=0。本工作在保留冻结三骨干骨架前提下实现：（i）<strong>单调有界质量路由</strong>（bad ≤ usable ≤ good ∈ [0,1]）；（ii）共享头 + <strong>掩码 BCE</strong> 的部分标签多任务学习；（iii）端点感知跨队列评估与校准/DCA<strong>评估框架</strong>；可选 E5 质量条件骨干路由。当前消融数字为<strong>SYNTHETIC</strong>，不得与 Reti-Pioneer 内部检验 AUROC 0.699–0.833 比较。</p>
  </section>

  <section id="bg">
    <h2>2. 背景与目的</h2>
    <p class="abbr">缩略语：CFP=彩色眼底照片；AUROC=ROC 曲线下面积；ECE=期望校准误差；DCA=决策曲线分析；UKB=英国生物银行；BCE=二元交叉熵；MTL=多任务学习。</p>
    <p>眼底微血管改变可反映全身代谢与血管状态（oculomics）。RETFound（Zhou 等，<em>Nature</em>，2023）提供可迁移视网膜表征；Reti-Pioneer 进一步集成多骨干与质量感知融合。本跟进关闭可用公开数据推进的方法学缺口（质量路由约束、部分标签 MTL、端点对齐、校准协议），而不是在无 UKB 时复现旗舰临床表。</p>
    <div class="warnbox">诚实边界：ODIR「Diabetes」主要为眼部证据/糖尿病视网膜病变相关标签，<strong>不是</strong> UKB ICD 式 2 型糖尿病；合成 AUROC 不得宣称超越 0.833。真实性审查见 <code>docs/paper/AUTHENTICITY_AUDIT.md</code>。</div>
  </section>

  <section id="methods">
    <h2>3. 数据与方法</h2>
    <h3>3.1 写作与实验架构</h3>
    <p>采用 Nature-family <strong>methods paper</strong> 论证链：任务/问题 → 既有方法边界 → 提议方法 → 公平消融与效用指标 → 可复现证据 → 边界。论文（academic）与报告（可含路径/过程）分流；论文不写本机绝对路径。</p>
    <h3>3.2 模型改动（与代码一一对应）</h3>
    <ul>
      <li><code>QualityAware(quality_router='monotone')</code>：softplus 增量 + cumsum 归一化，强制 bad≤usable≤good 且 good=1（<code>model/QualityAware.py</code>）。</li>
      <li><code>masked_bce_with_logits</code>：仅监督 y≥0 的部分标签（<code>utils/run.py</code>）。</li>
      <li>可选 E5：<code>QualityBackboneRouter</code> 由 q 生成三骨干 softmax（<code>model/quality_gate.py</code>；<code>configs/ablation_e5.yaml</code>）。</li>
      <li><code>evaluate.py --calibrate</code>：温度 T 拟合于 calibration_idx ⊥ eval；报告 ECE/Brier/按病种 DCA；NB@0.10 示意。</li>
      <li>患者级划分，避免同一患者双眼泄漏。</li>
    </ul>
    <h3>3.3 消融臂与数据</h3>
    <p>baseline（固定 q，单任务）→ learnq（monotone）→ multitask → full（monotone+MTL+校准评估）。特征缓存：<code>data/odir|brset/SYNTHETIC_FEATURES.txt</code>；RFMiD 影像可本地存在但仍带合成特征旗标时 <code>clinical_claim_allowed: false</code>。</p>
  </section>

  <section id="process">
    <h2>4. 研究过程</h2>
    <ol>
      <li>以审稿修订后的 <code>docs/paper/manuscript.md</code> 为论文主线；Results 仅对齐本仓库计算的 SYNTHETIC 指标。</li>
      <li><code>pip install SciencePlots</code>；<code>python scripts/plot_paper_figures.py</code>（Times New Roman）重绘 Fig1–4，写入 <code>docs/paper_assets/</code>。</li>
      <li>核对 <code>results/ablation_summary.csv</code>（2026-09-15T02:45）与 <code>results/metrics_*.json</code> disclaimer。</li>
      <li><code>python scripts/build_paper_report.py</code> 产出论文/报告 HTML·MD·PDF；撰写 <code>AUTHENTICITY_AUDIT.md</code>。</li>
      <li>公开快照（无密钥/无患者影像）：https://github.com/Coucou2016/retinal-imaging-methods-public</li>
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
    <h2>6. 分析与讨论</h2>
    <p>方法学创新点（单调路由、掩码 MTL、端点感知跨评）与仓库实现一致；但<strong>证据强度</strong>目前停在工程与合成自检层。公开队列一旦接入真实三骨干特征，应优先报告：相对 released-code clone 的同数据提升、校准改善、跨库落差与亚组公平性，而不是与 UKB 内部 AUROC 硬比。</p>
    <p>诚实措辞：prefer “monotone bounded quality router + masked partial-label MTL + endpoint-aware evaluation”，avoid “outperform Reti-Pioneer” without UKB endpoints。</p>
    <p><strong>稿件风险：</strong> 临床 Results 仍为待补充；勿把 SYNTHETIC 图当临床主张；须写清 published multitask vs released independent loops；无质量监督时 monotone 权重可能贴近 (0,0.5,1)；DCA@0.10 仅示意。</p>
  </section>

  <section id="conclusion">
    <h2>7. 主要结论</h2>
    <p>已形成可复核的方法扩展论文草稿（主线：审稿修订后的 manuscript）、SciencePlots 图件、自包含研究报告与真实性审查文档。本仓库计算的 SYNTHETIC 消融表证明 train/eval/校准链路可运行；真实临床 AUROC 仍为 <span class="pending">待补充</span>，不得进入投稿主表。</p>
  </section>

  <section id="limits">
    <h2>8. 不足与展望</h2>
    <ul>
      <li>无 UKB/SEED；ODIR/BRSET 真实像素与 GPU 三骨干提取待补充；RFMiD 影像本地有但合成特征旗标仍在。</li>
      <li>公开快照不含私有数据、权重与患者级影像。</li>
      <li>展望：凭据齐备后重跑 E0–E5 多 seed，填充论文 §5.2 临床表，并刷新本报告图 2–4。</li>
    </ul>
  </section>
</div>
</body>
</html>
"""


def build_manuscript_html(md_text: str, uris: dict[str, str]) -> str:
    """Render manuscript.md to self-contained HTML (tables + embedded figures)."""

    def fmt_inline(t: str) -> str:
        parts: list[str] = []
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

    body: list[str] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("|")
            and set(lines[i + 1].replace("|", "").replace("-", "").replace(":", "").strip()) <= {""}
        ):
            rows_html: list[str] = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if all(set(c) <= {"-", ":", " "} for c in cells):
                    i += 1
                    continue
                tag = "th" if not rows_html else "td"
                rows_html.append(
                    "<tr>" + "".join(f"<{tag}>{fmt_inline(c)}</{tag}>" for c in cells) + "</tr>"
                )
                i += 1
            if rows_html:
                body.append(
                    "<table><thead>"
                    + rows_html[0]
                    + "</thead><tbody>"
                    + "".join(rows_html[1:])
                    + "</tbody></table>"
                )
            continue
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.strip() == "---":
            body.append("<hr/>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{fmt_inline(line[2:])}</blockquote>")
        elif line.startswith("- ") or line.startswith("* "):
            body.append(f"<li>{fmt_inline(line[2:])}</li>")
        elif line.startswith("```"):
            lang = line[3:].strip()
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            body.append(
                f"<pre><code class=\"{html.escape(lang)}\">"
                + html.escape("\n".join(code_lines))
                + "</code></pre>"
            )
        elif line.strip() == "":
            body.append("")
        else:
            body.append(f"<p>{fmt_inline(line)}</p>")
        i += 1

    fig1 = (
        f"<figure><img src=\"{uris['fig1_architecture.png']}\" alt=\"Figure 1\"/>"
        f"<figcaption>Figure 1. Method overview (English labels; SYNTHETIC metrics in Figures 2–4).</figcaption></figure>"
    )
    result_figs = ""
    for name, cap in [
        ("fig2_ablation_bars.png", "Figure 2. Ablation bars (SYNTHETIC; pipeline sanity)."),
        ("fig3_calibration.png", "Figure 3. Temperature scaling vs ECE (SYNTHETIC)."),
        ("fig4_cross_domain.png", "Figure 4. Cross-cohort AUROC_D drop (SYNTHETIC)."),
    ]:
        if name in uris:
            result_figs += (
                f"<figure><img src=\"{uris[name]}\" alt=\"{html.escape(cap)}\"/>"
                f"<figcaption>{html.escape(cap)}</figcaption></figure>"
            )
    joined = "\n".join(body)
    joined = re.sub(r"(?:<li>.*?</li>\n?)+", lambda m: "<ul>" + m.group(0) + "</ul>", joined)
    joined = joined.replace("<h2>3. Methods</h2>", "<h2>3. Methods</h2>\n" + fig1, 1)
    joined = joined.replace("<h2>5. Results</h2>", "<h2>5. Results</h2>\n" + result_figs, 1)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Manuscript — Monotone Quality Routing and Endpoint-Aware MTL</title>
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

    # Prefer on-disk manuscript.md as source of truth (do not clobber matured drafts).
    ms_path = PAPER / "manuscript.md"
    if ms_path.is_file() and ms_path.stat().st_size > 500:
        ms = ms_path.read_text(encoding="utf-8")
    else:
        ms = manuscript_md()
        ms_path.write_text(ms, encoding="utf-8")
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

**Emulate:** Nature-family *methods* article (gap → method → fair ablation → calibration/utility as **evaluation** → reproducibility → boundary).

**Title:** Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning

**Spine:** peer-review-aligned `docs/paper/manuscript.md` (post–2026-09-13 rewrite + 2026-09-15 Results alignment).

**Innovation claims (bounded):** monotone bounded quality routing; masked partial-label multitask vs released-code independent loops; endpoint ontology + endpoint-aware cross-cohort evaluation; optional E5 gating.

**Evaluation framework (not novelty):** temperature scaling, ECE, Brier, per-disease DCA (NB@0.10 illustrative only).

**Non-claims:** synthetic AUROC as clinical evidence; beating 0.833 T2DM; ODIR-D as UKB T2DM; `diabetes_related` as clinical head; calibration/DCA as methodological novelty.

## Section map

| Section | Job | Claim ceiling |
|---------|-----|---------------|
| Abstract / Intro | Gap vs fixed-q + released independent heads | Methods extension |
| Methods | monotone router (math), masked BCE, endpoint ontology, E5 optional, disjoint cal | Mechanism + protocol |
| Experiments | E0–E5 + endpoint-aware transfer | Real-data tables 待补充 |
| Results §5.1 | Our computed SYNTHETIC ablation table | Pipeline sanity only |
| Results §5.2 | Clinical real-feature tables | 待补充 |
| Discussion | Published vs released multitask; endpoint drift | Honest novelty only |

## Draft risks

1. SYNTHETIC figures mistaken for clinical evidence.
2. Published “multitask” vs released independent-head loops.
3. Endpoint drift (`diabetes_related` / ocular ≠ UKB ICD).
4. Monotone router without quality supervision may stay near (0, 0.5, 1).
5. Single DCA threshold — keep per-disease curves primary.

## Authenticity

See `docs/paper/AUTHENTICITY_AUDIT.md`.
""",
        encoding="utf-8",
    )
    print("Wrote manuscript + report HTML/MD/PDF")


if __name__ == "__main__":
    main()
