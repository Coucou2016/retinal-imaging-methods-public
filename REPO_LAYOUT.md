# 发布布局说明

本仓库的 GitHub 远端采用**无目录**布局。所有文件直接位于仓库根目录，这是远端在
`d653981`（"Flatten the entire repository to a folder-free root layout"）中做出的
决定，`FLAT_FILE_INDEX.md` 记录了每个扁平文件对应的原始嵌套路径。

本地工作区 `E:\Projects\20260522-retinal-imaging` 仍然保留嵌套结构。两者不一致，
因此本地 `main` 不能直接推送到远端。

## 两个分支

| 分支 | 位置 | 布局 | 用途 |
|---|---|---|---|
| `main` | 本地工作区 | 嵌套（`docs/`、`scripts/`、`results/`） | 日常开发、跑实验、生成稿件 |
| `flat-main` | `C:\Users\Administrator\flatwt` | 扁平（全部在根目录） | 与 GitHub 远端同步 |

`flat-main` 跟踪 `origin/main`。工作区通过 `git worktree` 挂载同一个 `.git`，
所以两个分支共享对象库，切换成本很低。

## 发布流程

扁平快照由 `tools/build_flat.py` 生成，不再手工复制。该脚本读取嵌套工作区，
写入 `docs/_flat_build/`，并同时改写路径、生成索引与清单。

1. 在本地工作区生成或更新稿件与统计（`docs/paper/manuscript_zh_v3.*`、
   `docs/paper_assets/*`、`results/real_rfmid/analysis_seed/*.json`）。

2. 构建扁平快照。

   ```powershell
   python tools\build_flat.py
   ```

   产物：353 个文件，约 43.7 MB，无路径改写告警。命名规则与排除项写在
   `FLAT_FILE_INDEX.md`，稿件相关如下。

   | 本地路径 | 扁平文件名 |
   |---|---|
   | `docs/paper/manuscript_zh_v2.md` | `manuscript_zh.md` |
   | `docs/paper/manuscript_zh_v3.docx` | `manuscript_zh.docx` |
   | `docs/paper/manuscript_zh_v3.pdf` | `manuscript_zh.pdf` |
   | `docs/paper/manuscript_zh_v3.html` | `manuscript_zh.html` |
   | `docs/paper_assets/figN_*.png` | `figN_*.png`（同名） |
   | `docs/paper_assets/equations/eqN.png` | `eqN.png`（同名） |
   | `results/real_rfmid/analysis_sel/stats/*.json` | `*.json`（同名，铺到根目录） |
   | `results/real_rfmid/analysis_seed/*.json` | `*.json`（同名，铺到根目录） |

   现在还会带上每臂预测张量（`arm_E*.npz`、`cal_E*.npz`）与四种子统计，
   审计脚本因此在扁平根目录也能直接复算。

3. 同步到 `flatwt`，提交并推送。

   ```powershell
   Copy-Item E:\Projects\20260522-retinal-imaging\docs\_flat_build\* C:\Users\Administrator\flatwt\ -Force
   cd C:\Users\Administrator\flatwt
   git add -A
   git commit -m "..."
   ```

4. 推送。本机到 `github.com:443` 的直连经常被重置，普通 `git push` 会失败：

   ```
   fatal: unable to access '...': Recv failure: Connection was reset
   ```

   REST API 可达，因此改用 Git Data API 上传对象并快进引用。

   ```powershell
   $env:GH_TOKEN = (gh auth token)
   python E:\Projects\20260522-retinal-imaging\tools\push_via_api.py <base-sha> <tip-sha> main
   ```

   脚本先确认远端等于 `base`，再上传 blob、tree、commit，最后快进 `main`，
   并打印远端与本地是否一致。

## 注意

- 远端不存在子目录。新增文件也必须放在根目录，否则与既有约定冲突。
- 扁平仓库里 `label_map.json` 与 `label_map.py` 同名不同后缀，两者可以共存，但
  新增 JSON 时要先查根目录有没有同名模块。
- 统计 JSON 铺平后可能与根目录的脚本同名。`build_flat.py` 遇到同名不同内容会
  直接报错；图的多份镜像例外，保留 `docs/paper_assets` 的正式版本。
- 审计与数值核对脚本已改写为扁平路径，可直接在 `flatwt` 运行：
  `check_manuscript_numbers.py`、`check_ordering.py`、`check_citation_order.py`、
  `audit_quotes.py`、`lint_manuscript.py`。全部通过后才提交。
- `scripts/scratch/` 与一次性探针脚本（`probe_*.py`、`_*.py`、
  `configs/_speedtest.yaml`）不进快照。
