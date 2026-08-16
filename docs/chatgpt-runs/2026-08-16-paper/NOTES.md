# Paper run — 2026-08-16

## ChatGPT Pro/Plus

- **Dialogues used:** 0 (attempted; not completed)
- **URLs:** none (no stable chat session)
- **What happened:**
  1. `browser_tabs` create returned a `viewId`, then the tab vanished before navigate.
  2. `browser_navigate` to `https://chatgpt.com/` repeatedly failed with `No browser tab available` / `Browser view not found`.
  3. Opened system default browser via `Start-Process https://chatgpt.com/` as fallback for the user.
  4. Persisted paste-ready brief: [`TASK_BRIEF.md`](TASK_BRIEF.md) (text only; no uploads).
- **If auth/rate-limit:** user should paste `TASK_BRIEF.md` manually with web search ON; Cursor continues local work without blocking.

## Independent literature / outline (Cursor)

Verified / adopted without waiting on ChatGPT:

| Source | Role |
|--------|------|
| Zhang et al., Nat Med 2026, doi:10.1038/s41591-026-04359-w | Baseline Reti-Pioneer |
| Zhou et al., Nature 2023 RETFound | Foundation features |
| BRSET PLOS Digit Health / PhysioNet | Public multi-label + quality metadata |
| ODIR-5K / RFMiD docs | Public multi-label ocular labels |
| Temperature scaling (Guo et al.); DCA (Vickers & Elkin); RETFound+DCA community screening | Calibration / utility framing |
| `docs/PAPER_PLAN.md` | Local innovation stack I–III (+ IV/VI) |

**Writing architecture adopted:** Nature-family **methods** paper (not flagship Nat Med clinical narrative). See `docs/paper/OUTLINE.md`.

**Novelty honesty:** learnable_q + multitask + calibration/DCA + public patient-level validation vs Reti-Pioneer clone on same public data; never synthetic AUROC as clinical claim; ODIR-D ≠ UKB T2DM.

## Local deliverables

| Path | Notes |
|------|-------|
| `docs/paper/manuscript.md` | Nature-skills methods draft |
| `docs/paper/manuscript.html` | HTML manuscript + embedded Fig.1 |
| `docs/paper/manuscript.pdf` | fpdf2 export |
| `docs/paper/OUTLINE.md` | Architecture choice |
| `docs/report/report.html` | Self-contained; Base64 figures; inline CSS |
| `docs/report/report.md` | Matching markdown |
| `docs/report/report.pdf` | Matching PDF + figure pages |
| `docs/paper_assets/fig*.png` | SciencePlots + Times New Roman |
| `scripts/plot_paper_figures.py` | Regenerable figures |
| `scripts/build_paper_report.py` | HTML/MD/PDF builder |
| `requirements.txt` | + SciencePlots, matplotlib, markdown, fpdf2 |

## Git

No `.git` in workspace (prior notes). **No commit / push / PR.** Existing local files preserved.
