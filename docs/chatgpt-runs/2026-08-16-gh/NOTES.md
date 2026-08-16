# NOTES — 2026-08-16-gh (public GitHub + paper upgrade)

## GitHub (authorized public snapshot)

| Field | Value |
|-------|-------|
| URL | https://github.com/Coucou2016/retinal-imaging-methods-public |
| Visibility | **PUBLIC** |
| Owner | Coucou2016 |
| Commits pushed | `1f72677` (snapshot), `f5ec6bd` (URL + captions/docs) |
| Purpose | Code + docs for ChatGPT / external reading |
| **Not uploaded** | `.env`, credentials, `data/` images/npz, `ckpt/`, heavy `results/`, `pretrained/`, `Reti-Pioneer-main/`, `.venv`, browser state, private UKB |

Local workspace previously had **no** `.git`. This run: `git init` → commit → push to **new/existing public** repo under user account (name collision on create; remote HEAD matches local snapshot).

**Status line for final report:** 已推送到新建公开仓库（用户本次授权用于 ChatGPT 阅读）.

## ChatGPT Pro/Plus

- Browser MCP (`cursor-ide-browser`) **unavailable** in this session (only `cursor-app-control` listed).
- Opened system default browser to `https://chatgpt.com/` and the public GitHub URL for the user.
- Paste-ready brief: [`TASK_BRIEF.md`](TASK_BRIEF.md) (text only; includes GitHub URL; no ZIP).
- Chat session URL: **none** (automation blocked; manual paste required).

## Independent verify (Cursor; do not invent clinical AUROC)

Adopted / confirmed without waiting on ChatGPT:

| Item | Verdict |
|------|---------|
| Reti-Pioneer Nat Med 2026 doi:10.1038/s41591-026-04359-w | Confirmed; internal AUROC 0.699–0.833 cited only as **baseline paper numbers**, not our results |
| Writing architecture | Nature-family **methods** paper (not flagship Nat Med clinical narrative) — keep |
| Novelty phrasing | “unfreeze q + multitask + calibration/DCA + public cohorts” — keep; no “outperform Reti-Pioneer” without same-cohort UKB |
| SYNTHETIC metrics | Remain labeled; not manuscript claims |
| Figure captions | Deepened 来龙去脉 (问什么/怎么读/含义/结论/待补充) in report builder |

Sound edits applied locally; ChatGPT free-text advice deferred until user pastes brief and returns answers.

## Local deliverables this run

- Regenerated SciencePlots + Times New Roman figures → `docs/paper_assets/`, mirrored `results/figures/`
- Rebuilt `docs/paper/manuscript.{md,html,pdf}` and self-contained `docs/report/report.{html,md,pdf}` (Base64 images; **0** external `http(s)` / CDN)
- `unittest`: 34 tests OK
- This folder: NOTES, TASK_BRIEF, ACCEPTANCE (§十九), BROWSER_FAILURE

## Acceptance §十九 pointer

See [`ACCEPTANCE.md`](ACCEPTANCE.md).
