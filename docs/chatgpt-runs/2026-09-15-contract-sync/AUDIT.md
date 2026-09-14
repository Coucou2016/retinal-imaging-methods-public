# Contract sync audit — local vs remote HEAD

**Date:** 2026-09-15  
**Public repo:** https://github.com/Coucou2016/retinal-imaging-methods-public  
**Method:** `git fetch` + symbol/file comparison; ACCEPTANCE used only as a claim list.

## Git truth (pre-sync commit)

| Ref | SHA | Message |
|-----|-----|---------|
| Local `HEAD` (before this sync) | `2595a34bef068ad38ad16d2e762bcca73971eb50` | Note public snapshot push blocked by GitHub 443. |
| `origin/main` (after fetch) | `2595a34bef068ad38ad16d2e762bcca73971eb50` | identical |

**Verdict on reviewer claim that GitHub `main` still has only `Linear(3,1)` / `VALID_ENSEMBLES=("paper","mean","temp_mean")`:**  
**False for current remote HEAD.** At `2595a34`, GitHub already contains:

- `QualityRouter = Literal["fixed", "free_linear", "monotone"]` and monotone path in `model/QualityAware.py`
- `VALID_ENSEMBLES = ("released_code", "published_soft_vote", "mean", "temp_mean", "paper")` in `model/RetiPioneer.py`
- Endpoint ontology + `cal_fraction` / paper_mode cal guards from `f431a00` / `3584cb7`

Likely cause of the round-2 review mismatch: stale clone / ChatGPT snapshot older than `f431a00`, or reading root `ACCEPTANCE.md` (still frozen at iter-5 / SHA `5aab91c`) instead of code.

## Key-file matrix (pre → post this sync)

| File | Pre (`2595a34`) | Gap closed this run |
|------|-----------------|---------------------|
| `model/QualityAware.py` | monotone / free_linear / fixed | No change needed |
| `model/RetiPioneer.py` | released_code + wire quality_router / λ_q / gating | No change needed |
| `reti_pioneer/label_map.py` | `Endpoint` + ocular/systemic diabetes | **`EndpointSpec` alias; `hypertension_systemic`; provenance helpers** |
| `reti_pioneer/config.py` | load-only YAML | **Strict schema (forbid unknown keys) + `build_model_from_config`** |
| `reti_pioneer/split.py` | train/cal/val/test patient splits | Already present |
| `scripts/evaluate.py` | cal ⊥ eval; paper_mode refuse val-fit | **+ paper_mode refuses synthetic/stub features** |
| `scripts/train.py` | quality_router / masked_bce / λ_q | Already wired |
| `scripts/prepare_public_npz.py` | **BUG:** no-val path wiped official `test_idx` | **Carve val from train only; preserve test** |
| `scripts/extract_features.py` | cleared SYNTHETIC marker only | **Updates `label_map.json` provenance** |
| `pyproject.toml` | no local `../../wheel` paths | Confirmed clean |
| `configs/*` | ensembles / routers / E5 | Schema-validated; all build models |
| Root `ACCEPTANCE.md` | Stale iter-5 (`5aab91c`) | **Rewritten to this sync SHA** |
| `README.md` | Hard-coded `E:\Projects\...` | **`<repo-root>`** |

## Remaining honest blockers (not inventable)

| Item | Status |
|------|--------|
| ODIR download | No `~/.kaggle/kaggle.json` |
| BRSET download | No PhysioNet credentials |
| CUDA foundation extract | `torch.cuda.is_available() == False` |
| Clinical AUROC / Results tables | **待补充** (not invented) |

## Tests

```text
python -m unittest discover -s tests -v
# → 73 tests, OK (~279 s) after this sync
```

Hard-gates added in `tests/test_contract_sync.py`: all YAML parse+build, monotone bounds, ensemble rename, endpoint systemic/ocular split, official test never resplit, provenance gate, paper_mode val-calibrate refuse.

## Post-sync push

Local commits pushed to `origin/main`: `2595a34..47fa0fc`  
- `cc70b8d` Close P0 YAML-code contract gaps for reviewer sync.  
- `93170c6` / `69ef661` / `10afcf3` / `47fa0fc` ACCEPTANCE + audit bookkeeping  

**Public tip:** `47fa0fc03eae79d057ce0a10b3e8ddf4ad857d40`  
Remote `QualityAware.py` confirmed: `QualityRouter = Literal["fixed", "free_linear", "monotone"]`.
