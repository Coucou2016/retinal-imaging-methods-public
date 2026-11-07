#!/usr/bin/env python3
"""Reference database for the Chinese manuscript.

Every entry was checked against Crossref for title, journal, volume, pages and
year.  Keys are stable; numbering is assigned by first appearance in the text
by ``scripts/apply_citations.py``.
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


REFS: dict[str, str] = {
    "zhou2023": "Zhou Y, Chia MA, Wagner SK, et al. A foundation model for generalizable "
    "disease detection from retinal images. *Nature*. 2023;622:156-163. "
    "doi:10.1038/s41586-023-06555-x",
    "zhang2026": "Zhang X, Li Q, Liang Y, et al. AI framework for multidisease detection "
    "via retinal imaging. *Nature Medicine*. 2026;32:2494-2503. "
    "doi:10.1038/s41591-026-04359-w",
    "retipioneer": "Reti-Pioneer reference implementation. GitHub repository. "
    "https://github.com/lyhyl/Reti-Pioneer",
    "muller2021": "Müller D, Soto-Rey I, Kramer F. Multi-disease detection in retinal "
    "imaging based on ensembling heterogeneous deep learning models. *Studies in Health "
    "Technology and Informatics*. 2021;281:199-203. doi:10.3233/SHTI210537",
    "matta2023": "Matta S, Lamard M, Conze PH, et al. Towards population-independent, "
    "multi-disease detection in fundus photographs. *Scientific Reports*. 2023;13:11493. "
    "doi:10.1038/s41598-023-38610-y",
    "fu2019": "Fu H, Wang B, Shen J, et al. Evaluation of retinal image quality assessment "
    "networks in different color-spaces. In: *Medical Image Computing and Computer "
    "Assisted Intervention*. 2019:48-56. doi:10.1007/978-3-030-32239-7_6",
    "huang2024": "Huang C, Jiang Y, Yang X, et al. Enhancing retinal fundus image quality "
    "assessment with Swin-Transformer-based learning across multiple color-spaces. "
    "*Translational Vision Science & Technology*. 2024;13:8. doi:10.1167/tvst.13.4.8",
    "shi2022": "Shi C, Lee J, Wang G, et al. Assessment of image quality on color fundus "
    "retinal images using the automatic retinal image analysis. *Scientific Reports*. "
    "2022;12:10455. doi:10.1038/s41598-022-13919-2",
    "pachade2021": "Pachade S, Porwal P, Thulkar D, et al. Retinal Fundus Multi-Disease "
    "Image Dataset (RFMiD): a dataset for multi-disease detection research. *Data*. "
    "2021;6:14. doi:10.3390/data6020014",
    "nakayama2024": "Nakayama LF, Restrepo D, Matos J, et al. BRSET: a Brazilian multilabel "
    "ophthalmological dataset of retina fundus photos. *PLOS Digital Health*. "
    "2024;3:e0000454. doi:10.1371/journal.pdig.0000454",
    "odir2019": "ODIR-5K Grand Challenge dataset documentation. "
    "https://odir2019.grand-challenge.org/dataset/",
    "guo2017": "Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural "
    "networks. In: *International Conference on Machine Learning*. 2017:1321-1330.",
    "vancalster2019": "Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the "
    "Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230. "
    "doi:10.1186/s12916-019-1466-7",
    "naeini2015": "Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated "
    "probabilities using Bayesian binning. In: *Proceedings of the AAAI Conference on "
    "Artificial Intelligence*. 2015;29:2901-2907. doi:10.1609/aaai.v29i1.9602",
    "platt1999": "Platt J. Probabilities for SV machines. In: Smola AJ, Bartlett P, "
    "Schölkopf B, Schuurmans D, eds. *Advances in Large-Margin Classifiers*. Cambridge: "
    "MIT Press; 2000:61-74. doi:10.7551/mitpress/1113.003.0008",
    "zadrozny2002": "Zadrozny B, Elkan C. Transforming classifier scores into accurate "
    "multiclass probability estimates. In: *Proceedings of the Eighth ACM SIGKDD "
    "International Conference on Knowledge Discovery and Data Mining*. 2002:694-699. "
    "doi:10.1145/775047.775151",
    "niculescu2005": "Niculescu-Mizil A, Caruana R. Predicting good probabilities with "
    "supervised learning. In: *Proceedings of the 22nd International Conference on Machine "
    "Learning*. 2005:625-632. doi:10.1145/1102351.1102430",
    "vickers2006": "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for "
    "evaluating prediction models. *Medical Decision Making*. 2006;26:565-574. "
    "doi:10.1177/0272989X06295361",
    "zhangj2024": "Zhang J, Mao B, Wang X, et al. RETFound-enhanced community-based fundus "
    "disease screening: real-world evidence and decision curve analysis. *npj Digital "
    "Medicine*. 2024;7:108. doi:10.1038/s41746-024-01109-5",
    "brier1950": "Brier GW. Verification of forecasts expressed in terms of probability. "
    "*Monthly Weather Review*. 1950;78:1-3. "
    "doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2",
    "youden1950": "Youden WJ. Index for rating diagnostic tests. *Cancer*. 1950;3:32-35. "
    "doi:10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3",
    "delong1988": "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or "
    "more correlated receiver operating characteristic curves: a nonparametric approach. "
    "*Biometrics*. 1988;44:837-845. doi:10.2307/2531595",
    "efron1993": "Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. New York: "
    "Chapman and Hall; 1993. doi:10.1201/9780429246593",
    "collins2024": "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated "
    "guidance for reporting clinical prediction models that use regression or machine "
    "learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378",
}
