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

1. 在本地工作区生成或更新稿件（`docs/paper/manuscript_zh_v3.*`、`docs/paper_assets/*`）。
2. 进入 `C:\Users\Administrator\flatwt`。
3. 把文件按扁平命名复制到根目录。命名规则见 `FLAT_FILE_INDEX.md`，稿件相关如下。

   | 本地路径 | 扁平文件名 |
   |---|---|
   | `docs/paper/manuscript_zh_v2.md` | `manuscript_zh.md` |
   | `docs/paper/manuscript_zh_v3.docx` | `manuscript_zh.docx` |
   | `docs/paper/manuscript_zh_v3.pdf` | `manuscript_zh.pdf` |
   | `docs/paper/manuscript_zh_v3.html` | `manuscript_zh.html` |
   | `docs/paper_assets/figN_*.png` | `figN_*.png`（同名） |
   | `results/real_rfmid/analysis_sel/stats/*.json` | `*.json`（同名，铺到根目录） |

4. `git add -A`，提交，`git push origin HEAD:main`。

## 注意

- 远端不存在子目录。新增文件也必须放在根目录，否则与既有约定冲突。
- 扁平仓库里 `label_map.json` 与 `label_map.py` 同名不同后缀，两者可以共存，但
  新增 JSON 时要先查根目录有没有同名模块。
- 统计 JSON 铺平后可能与根目录的脚本同名，复制前先比对。
- `verify_ref_authors.py` 与 `check_manuscript_numbers.py` 读取的是本地嵌套路径，
  在 `flatwt` 里直接运行需要改路径参数。当前只在本地工作区运行。
