# NOTES — 2026-08-16-gh-consult

## ChatGPT

| Field | Value |
|-------|-------|
| Chat URL | **none** (automation blocked) |
| Web search enabled by agent | **no** |
| Reply captured | **none** — do not invent |
| Paste brief | [`TASK_BRIEF.md`](TASK_BRIEF.md) (copy of `2026-08-16-gh/TASK_BRIEF.md`) |
| Failure log | [`BROWSER_FAILURE.md`](BROWSER_FAILURE.md) |

Public repo for manual ChatGPT reading: https://github.com/Coucou2016/retinal-imaging-methods-public

## Independent verify (no ChatGPT text)

Verified against Nature / PhysioNet / MICCAI sources + local `docs/PAPER_PLAN.md`:

| Claim | Verdict |
|-------|---------|
| Reti-Pioneer Nat Med 2026 doi:10.1038/s41591-026-04359-w; internal AUROC 0.699–0.833 | Confirmed (baseline paper only) |
| Research Briefing doi:10.1038/s41591-026-04424-4 | Confirmed; added to manuscript seed refs |
| EyeQ Fu et al. MICCAI 2019 doi:10.1007/978-3-030-32239-7_6 | Confirmed; replaced vague “workshops” wording |
| BRSET Nakayama et al. PLOS Digit Health e0000454 | Confirmed |
| Nature-family **methods** venue architecture (not flagship Nat Med clinical claim) | Keep |
| No invented clinical / UKB AUROC | Keep |

## Local polish applied (this turn; local-only, no git push)

Because ChatGPT was unreachable, applied **two** sound writing/docs improvements from acceptance residuals:

1. **Manuscript / OUTLINE:** expanded Nature-methods section map; added explicit **Draft risks** (SYNTHETIC misuse, multitask wording vs independent-head clone, label drift, unsupervised `q_fc`, DCA threshold); tightened EyeQ/BRSET/Research Briefing citations.  
2. **Chinese report:** updated ChatGPT failure status to this consult folder; added matching draft-risk bullets in Discussion.

Rebuilt via `python scripts/build_paper_report.py` → `docs/paper/*`, `docs/report/*` (report.html still Base64, no external CDN).

## Git

**No push this turn** (prefer local-only; ChatGPT already has prior public snapshot URL).
