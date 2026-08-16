"""Disease and metadata definitions aligned with the Nature Medicine paper."""

DISEASE_NAMES = [
    "t2dm",
    "thyroid",
    "osteoporosis",
    "gout",
    "hyperlipemia",
    "hypertension",
]

PUBLIC_DATASETS = ("odir", "brset", "rfmid")

DISEASES = {
    "T2D": "t2dm",
    "Hypertension": "hypertension",
    "Hyperlipidemia": "hyperlipemia",
    "Gout": "gout",
    "Osteoporosis": "osteoporosis",
    "Thyroid": "thyroid",
}

CLINIC_VARIABLES = ["baselineage", "gender", "weight", "ethnicity"]

LONGITUDINAL_HORIZONS = [0, 5, 10]

# Paper-reported internal-test AUROCs (reference only)
PAPER_INTERNAL_AUROC = {
    "t2dm": 0.833,
    "gout": 0.832,
    "osteoporosis": 0.787,
    "hypertension": 0.740,
    "hyperlipemia": 0.736,
    "thyroid": 0.699,
}
