# 研究报告：基于 Reti-Pioneer 的可学习质量路由与多任务眼底筛查方法扩展

**日期：** 2026-08-16  
**项目路径：** `E:\Projects\20260522-retinal-imaging`  
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

| Arm | Eval | AUROC | AUROC_D | ECE | Cal ECE | NB@0.10 | n | Note |
|-----|------|-------|---------|-----|---------|---------|---|------|
| baseline | odir_val | 0.469 | 0.469 | 0.339 | 0.184 | 0.269 | 12 | SYNTHETIC |
| baseline | odir_to_brset | 0.499 | 0.499 | 0.372 | 0.261 | 0.093 | 48 | SYNTHETIC |
| learnq | odir_val | 0.469 | 0.469 | 0.339 | 0.184 | 0.269 | 12 | SYNTHETIC |
| learnq | odir_to_brset | 0.499 | 0.499 | 0.372 | 0.261 | 0.093 | 48 | SYNTHETIC |
| multitask | odir_val | 0.380 | 0.469 | 0.647 | 0.370 | 0.088 | 12 | SYNTHETIC |
| multitask | odir_to_brset | 0.415 | 0.361 | 0.480 | 0.338 | 0.057 | 48 | SYNTHETIC |
| full | odir_val | 0.380 | 0.469 | 0.647 | 0.370 | 0.088 | 12 | SYNTHETIC |
| full | odir_to_brset | 0.415 | 0.361 | 0.480 | 0.338 | 0.057 | 48 | SYNTHETIC |

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
