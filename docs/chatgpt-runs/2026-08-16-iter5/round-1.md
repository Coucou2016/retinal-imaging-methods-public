# Round 1 — Lit survey + Nature-methods outline + novelty

**Date:** 2026-08-16  
**Mode:** ChatGPT live attempt blocked → **local structured review** (self + WebSearch). Paste brief: `PASTE_BRIEF_R1.md`.  
**Chat URL:** *unavailable* (`cursor-ide-browser` MCP absent; Playwright historically hits Cloudflare; `open_resource`/`Start-Process` opened system browser only — no agent-captured reply).  
**Public GitHub given to advisor protocol:** https://github.com/Coucou2016/retinal-imaging-methods-public

## ChatGPT attempt (required)

| Action | Result |
|--------|--------|
| MCP `browser_tabs` / navigate | Server **not available** (only `cursor-app-control`) |
| `open_resource` → chatgpt.com | Failed: unknown agent |
| `Start-Process https://chatgpt.com/` | Opened user browser (manual paste possible) |
| Playwright headless (this run / prior) | Cloudflare / lock hang risk; **no invented ChatGPT reply** |

## Independent literature (WebSearch verified)

| Topic | Citation | DOI / access | Verdict |
|-------|----------|--------------|---------|
| Baseline | Zhang et al. Reti-Pioneer, *Nat Med* 32:2494–2503 (2026) | 10.1038/s41591-026-04359-w | Confirmed AUROC ranges 0.699–0.833 internal |
| Foundation model | Zhou et al. RETFound, *Nature* 622:156–163 (2023) | 10.1038/s41586-023-06555-x | Confirmed |
| Quality | Fu et al. EyeQ / MCF-Net, MICCAI 2019 | 10.1007/978-3-030-32239-7_6 / arXiv:1907.05345 | Good/Usable/Reject taxonomy |
| Public cohort | Nakayama et al. BRSET, *PLOS Digit Health* 2024 | 10.1371/journal.pdig.0000454 | PhysioNet credentialed |
| Public cohort | Pachade et al. RFMiD, *Data* 2021 | 10.3390/data6020014 | 3200 images, 46 conditions |
| Calibration | Guo et al. temperature scaling, ICML 2017 | PMLR v70 | Standard post-hoc |
| Utility | Vickers & Elkin, DCA, *Med Decis Making* 2006 | PMID 17099194 | Net benefit language |
| Multi-label / DG | Quellec et al. population-independent multi-disease, *Sci Rep* 2023 | 10.1038/s41598-023-38610-y | Supports ODIR↔cross-site framing |

## Outline (Nature-family methods)

1. Abstract — problem, method deltas, public-protocol claim, **待补充** clinical tables  
2. Introduction — oculomics gap; three actionable gaps without UKB  
3. Related work — foundation models; quality; multi-label public sets; calibration/DCA  
4. Methods — inputs; frozen caches; learnable_q; multitask BCE; ensemble; temp scaling; patient split; label map  
5. Experiments protocol — ODIR/BRSET/RFMiD; ablations; primary AUROC + co-primary ECE/NB  
6. Results — **empty clinical tables** until real features; SYNTHETIC only in companion report  
7. Discussion — honesty, novelty phrasing, draft risks  
8. Limitations / Reproducibility / Data+Code availability  

## Novelty sentence (accepted)

> We unfreeze quality routing initialized at Reti-Pioneer’s fixed weights and evaluate jointly with multi-label heads and calibration/DCA on public cohorts reviewers can download.

**Rejected advice (pre-emptively):** claiming “outperform Reti-Pioneer UKB AUROC”; treating ODIR-D as T2DM ICD; putting synthetic AUROC in manuscript Results.

## Applied to manuscript

- Expanded Related work + Methods evaluation protocol.  
- Strengthened Abstract/Results honesty banners.  
- Added Sci Rep 2023 DG citation to reference list.
