# Round 2 — Methods section critique

**Date:** 2026-08-16  
**Mode:** Local structured critique (ChatGPT paste brief ready: `PASTE_BRIEF_R2.md`).  
**Chat URL:** unavailable (automation blocked; see round-1). Mid-run ChatGPT retry: `Start-Process` + prior Playwright path only.

## Critique (independent)

| Issue | Severity | Action |
|-------|----------|--------|
| Reti-Pioneer abstract “multitask” vs code independent heads | High (novelty straw-man) | Explicit Methods note: clone control = **independent binary heads as in released codebase** |
| Learnable_q without BRSET quality supervision | Medium | State collapse risk; defer supervised quality as 待补充 |
| Hyperparameters incomplete | Medium | List LR 1e-4, seeds {42,43,44}, frozen backbones, patient split, BCE±pos_weight, ensemble max / temp_mean |
| Hardware/software | Medium | Document CPU-only workspace; CUDA feature extract blocked → foundation features 待补充 |
| ECE binning / DCA threshold | Low–Med | State ECE bins (default in `utils/calibration.py`); NB@0.10 is illustrative until prevalence/cost calibrated |
| Label semantics table | High | Keep pre-registered map; forbid pooling diabetes heads without footnote |
| Failure modes | Medium | Add: quality vector missing → pad; Vim optional on Windows; synthetic cache exit path |

## Rejected “advice”

- Inventing AUROC placeholders “for reviewers to imagine”  
- Claiming learnable_q is a new quality taxonomy (it is not; EyeQ grades stay)

## Applied

- Methods § deepened (protocol box, software env, failure modes, straw-man clarification).  
- Explicit “Reti-Pioneer clone = independent heads” language.
