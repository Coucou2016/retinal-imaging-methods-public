# ACCEPTANCE — 2026-09-15 full deliverables

**Status:** shipped; public tip confirmed.  
**Date:** 2026-09-15  
**Axes:** nature-writing / nature-polishing · task=manuscript · paper_type=methods · language=en · journal=generic

## What shipped

| Deliverable | Path | Notes |
|-------------|------|-------|
| Manuscript MD | `docs/paper/manuscript.md` | Academic rewrite; strengthened monotone / masked MTL / E5; Phrasebank-style English; no local absolute paths |
| Manuscript HTML | `docs/paper/manuscript.html` | Self-contained CSS; embedded figures |
| Manuscript PDF | `docs/paper/manuscript.pdf` | Full text + Figures 1–4 |
| Report MD | `docs/report/report.md` | Process/paths allowed; relative figure links; abbreviations expanded |
| Report HTML | `docs/report/report.html` | Inline CSS; Base64 images; **no CDN**; cover / TOC / abstract … / limitations; deep 来龙去脉 captions; CJK-capable fonts in CSS stack |
| Report PDF | `docs/report/report.pdf` | Chinese font (SimHei) when available + figure pages |
| Authenticity audit | `docs/paper/AUTHENTICITY_AUDIT.md` | Evidence chain: commands, configs, commits, `clinical_claim_allowed`, SYNTHETIC markers |
| Figures | `docs/paper_assets/fig{1..4}_*.{png,pdf}` | SciencePlots + Times New Roman (readable sizes); Fig1 includes E5 |
| URI sidecar | `docs/paper_assets/embedded_png_uris.json` | Base64 for HTML embeds |
| Build scripts | `scripts/plot_paper_figures.py`, `scripts/build_paper_report.py` | Extended: dual PDF embeds all figs; richer report |

## Results provenance (no invented clinical AUROCs)

- Source: `results/ablation_summary.csv` + `results/metrics_*.json` (batch `2026-09-15T02:45:26`)
- All rows: `disclaimer` = SYNTHETIC FEATURE CACHE; `n` = 10 (ODIR val) / 48 (ODIR→BRSET)
- Flags: `data/{odir,brset,rfmid}/SYNTHETIC_FEATURES.txt` → `clinical_claim_allowed: false`
- Clinical real-feature tables: **待补充** (no CUDA / no ODIR-Kaggle / BRSET DUA incomplete)
- Nature Medicine UKB AUROCs 0.699–0.833: **background citation only**, never as our results

## Commands used

```text
python scripts/plot_paper_figures.py
python scripts/build_paper_report.py
```

## SHA

| Ref | Value |
|-----|-------|
| Content commit | `1a71d64083e0bb18addc3ddfcf0c39912b0ad010` |
| Public tip (confirmed) | `8291781f58e8116adf7c1d815c9d3956e7f8cb50` |

## Push

- Remote: `https://github.com/Coucou2016/retinal-imaging-methods-public.git`
- HTTPS `git push` failed (`Recv failure: Connection was reset` on :443)
- Fallback: Git Data API non-force `PATCH refs/heads/main` via `_api_push.py` → **SUCCESS**
- Confirmed: `gh api .../commits/main` = `8291781f58e8116adf7c1d815c9d3956e7f8cb50`
- No force-push; no secrets; no patient images / UKB extracts

## Public URLs

- Tip: https://github.com/Coucou2016/retinal-imaging-methods-public/commit/8291781f58e8116adf7c1d815c9d3956e7f8cb50
- Repo: https://github.com/Coucou2016/retinal-imaging-methods-public
