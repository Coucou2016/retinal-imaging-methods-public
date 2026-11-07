#!/usr/bin/env python3
"""Cross-check every number quoted in the Chinese manuscript against the stats files.

The manuscript is written by hand, so drift is inevitable unless it is checked.
Tables are located by a fragment of their caption rather than by number, so
inserting a table does not silently point the checker at the wrong block.

Exit code 1 when a mismatch exceeds the rounding tolerance.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "paper" / "manuscript_zh_v2.md"
NUM = ROOT / "results" / "real_rfmid" / "analysis_sel" / "manuscript_numbers.json"

TOL = 0.0015


def parse_number(cell: str) -> float | None:
    s = cell.strip().replace("−", "-").replace("+", "").replace("×", "e")
    try:
        return float(s)
    except ValueError:
        return None


def table_block(text: str, caption: str) -> list[list[str]] | None:
    """Rows of the markdown table whose caption contains ``caption``."""
    anchor = None
    for m in re.finditer(r"\*\*表\s*\d+\*\*[^\n]*", text):
        if caption in m.group(0):
            anchor = m
            break
    if anchor is None:
        return None
    rest = text[anchor.end():]
    rows: list[list[str]] = []
    for line in rest.splitlines():
        s = line.strip()
        if not s:
            if rows:
                break
            continue
        if not s.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def rows_by_arm(rows: list[list[str]]) -> dict[str, list[str]]:
    out = {}
    for r in rows[1:]:
        key = r[0]
        if re.fullmatch(r"E[0-5]", key):
            out[key] = r
    return out


def compare(problems: list[str], label: str, idx: int, got: float | None, want: float) -> None:
    if got is None:
        problems.append(f"{label} col {idx}: unparsable cell")
    elif abs(got - want) > TOL:
        problems.append(f"{label} col {idx}: manuscript {got} vs stats {round(want, 3)}")


def check_epoch(text, nums, problems):
    rows = rows_by_arm(table_block(text, "报告轮次") or [])
    for arm, row in nums["epoch_selection"].items():
        if arm not in rows:
            problems.append(f"epoch table: no row for {arm}")
            continue
        r = rows[arm]
        compare(problems, f"epoch {arm} n_epochs", 0, parse_number(r[1]), row["n_epochs"])
        compare(problems, f"epoch {arm} selected", 1, parse_number(r[2]), row["epoch"])
        compare(problems, f"epoch {arm} val", 2, parse_number(r[3]), row["val_macro_auroc"])


def check_main(text, nums, problems):
    rows = rows_by_arm(table_block(text, "RFMiD 测试集主要指标") or [])
    for arm, row in nums["arms"].items():
        if arm not in rows:
            problems.append(f"main table: no row for {arm}")
            continue
        r = rows[arm]
        # columns: arm | macro | lo 至 hi | DeLong lo 至 hi | DR | AP | ECE | Brier
        compare(problems, f"main {arm} macro", 0, parse_number(r[1]), row["macro_auroc"])
        m = re.match(r"([0-9.]+)\s*至\s*([0-9.]+)", r[2])
        if not m:
            problems.append(f"main {arm}: CI cell unparsable ({r[2]})")
        else:
            compare(problems, f"main {arm} ci_lo", 1, float(m.group(1)), row["macro_auroc_ci"][0])
            compare(problems, f"main {arm} ci_hi", 2, float(m.group(2)), row["macro_auroc_ci"][1])
        dl = re.match(r"([0-9.]+)\s*至\s*([0-9.]+)", r[3])
        if dl and "macro_auroc_delong" in row:
            compare(problems, f"main {arm} delong_lo", 3, float(dl.group(1)), row["macro_auroc_delong"][0])
            compare(problems, f"main {arm} delong_hi", 4, float(dl.group(2)), row["macro_auroc_delong"][1])
        compare(problems, f"main {arm} dr_auroc", 5, parse_number(r[4]), row["dr_auroc"])
        compare(problems, f"main {arm} ap", 6, parse_number(r[5]), row["macro_ap"])
        compare(problems, f"main {arm} ece", 7, parse_number(r[6]), row["mean_head_ece"])
        compare(problems, f"main {arm} brier", 8, parse_number(r[7]), row["pooled_brier"])


def check_paired(text, nums, problems):
    rows = table_block(text, "配对 AUROC 差值") or []
    mapping = {
        ("E1 减 E0", "DR"): ("E1_minus_E0", "DR"),
        ("E3 减 E2", "宏平均"): ("E3_minus_E2", "macro"),
        ("E3 减 E2", "糖尿病视网膜病变"): ("E3_minus_E2", "DR"),
        ("E4 减 E3", "宏平均"): ("E4_minus_E3", "macro"),
        ("E5 减 E3", "宏平均"): ("E5_minus_E3", "macro"),
        ("E5 减 E3", "糖尿病视网膜病变"): ("E5_minus_E3", "DR"),
    }
    found = {}
    for r in rows[1:]:
        found[(r[0], r[1])] = r
    for (label, endpoint), (key, which) in mapping.items():
        if key not in nums["paired"]:
            problems.append(f"paired: missing entry {key}")
            continue
        e = nums["paired"][key].get(which)
        if e is None:
            continue
        if (label, endpoint) not in found:
            problems.append(f"paired: no row for {label} / {endpoint}")
            continue
        r = found[(label, endpoint)]
        compare(problems, f"paired {label} {endpoint} delta", 0, parse_number(r[2]), e["delta"])
        m = re.match(r"([+\-−]?[0-9.]+)\s*至\s*([+\-−]?[0-9.]+)", r[3])
        if not m:
            problems.append(f"paired {label} {endpoint}: CI cell unparsable ({r[3]})")
        else:
            compare(problems, f"paired {label} {endpoint} lo", 1, parse_number(m.group(1)), e["ci_low"])
            compare(problems, f"paired {label} {endpoint} hi", 2, parse_number(m.group(2)), e["ci_high"])
        frac = re.search(r"([0-9.]+)", r[4])
        if frac:
            compare(problems, f"paired {label} {endpoint} share", 3, float(frac.group(1)), e["frac_positive"])


CN_EP = {
    "Disease_Risk": "任意病变",
    "DR": "糖尿病视网膜病变",
    "MH": "屈光介质混浊",
    "ODC": "视盘杯盘比异常",
    "TSLN": "豹纹状眼底",
    "DN": "玻璃膜疣",
    "ARMD": "年龄相关性黄斑变性",
    "MYA": "近视性改变",
}


def check_per_head(text, nums, problems):
    rows = {r[0]: r for r in (table_block(text, "在测试集上的分终点表现") or [])[1:]}
    for ep, row in nums["per_head"]["E5"].items():
        name = CN_EP[ep]
        if name not in rows:
            problems.append(f"per-head: no row for {ep}")
            continue
        r = rows[name]
        # endpoint | AUROC | lo 至 hi | AP | ECE | Brier | prevalence
        compare(problems, f"per-head {ep} auroc", 0, parse_number(r[1]), row["auroc"])
        compare(problems, f"per-head {ep} ap", 1, parse_number(r[3]), row["ap"])
        compare(problems, f"per-head {ep} ece", 2, parse_number(r[4]), row["ece"])
        compare(problems, f"per-head {ep} brier", 3, parse_number(r[5]), row["brier"])
        compare(problems, f"per-head {ep} prev", 4, parse_number(r[6]), row["prevalence"])


def check_calibration(text, nums, problems):
    rows = rows_by_arm(table_block(text, "校准指标") or [])
    for arm, row in nums["arms"].items():
        if arm not in rows:
            problems.append(f"calibration: no row for {arm}")
            continue
        r = rows[arm]
        # arm | T | cal raw | cal post | mean raw | mean post | worst | brier raw | brier post
        compare(problems, f"cal {arm} T", 0, parse_number(r[1]), row["temperature_cal64"])
        compare(problems, f"cal {arm} cal_raw", 1, parse_number(r[2]), row["ece_on_cal_raw"])
        compare(problems, f"cal {arm} cal_post", 2, parse_number(r[3]), row["ece_on_cal_post"])
        compare(problems, f"cal {arm} mean_raw", 3, parse_number(r[4]), row["mean_head_ece"])
        compare(problems, f"cal {arm} mean_post", 4, parse_number(r[5]), row["mean_head_ece_after_T"])
        compare(problems, f"cal {arm} worst", 5, parse_number(r[6]), row["worst_head_ece"])
        compare(problems, f"cal {arm} brier_raw", 6, parse_number(r[7]), row["pooled_brier"])
        compare(problems, f"cal {arm} brier_post", 7, parse_number(r[8]), row["brier_after_T"])


def check_youden(text, nums, problems):
    rows = rows_by_arm(table_block(text, "约登工作点") or [])
    op = nums["operating_points"]
    for arm in nums["arms"]:
        if arm not in op or "DR" not in op[arm]:
            continue
        if arm not in rows:
            problems.append(f"youden: no row for {arm}")
            continue
        r, dr = rows[arm], op[arm]["DR"]["youden"]
        compare(problems, f"youden {arm} thr", 0, parse_number(r[1]), dr["threshold"])
        compare(problems, f"youden {arm} val_sens", 1, parse_number(r[2]), dr["sensitivity"])
        compare(problems, f"youden {arm} val_spec", 2, parse_number(r[3]), dr["specificity"])
        compare(problems, f"youden {arm} test_sens", 3, parse_number(r[4]), dr["test_sensitivity"])
        compare(problems, f"youden {arm} test_spec", 4, parse_number(r[5]), dr["test_specificity"])


def check_s95(text, nums, problems):
    rows = rows_by_arm(table_block(text, "高特异度工作点") or [])
    op = nums["operating_points"]
    for arm in nums["arms"]:
        if arm not in op or "DR" not in op[arm]:
            continue
        if arm not in rows:
            problems.append(f"s95: no row for {arm}")
            continue
        r, s = rows[arm], op[arm]["DR"]["sens_at_95spec"]
        compare(problems, f"s95 {arm} thr", 0, parse_number(r[1]), s["threshold"])
        compare(problems, f"s95 {arm} val_sens", 1, parse_number(r[2]), s["sensitivity"])
        compare(problems, f"s95 {arm} test_sens", 2, parse_number(r[3]), s["test_sensitivity"])
        compare(problems, f"s95 {arm} test_spec", 3, parse_number(r[4]), s["test_specificity"])
        compare(problems, f"s95 {arm} oracle", 4, parse_number(r[5]),
                nums["operating_point_analysis"]["per_arm"][arm]["test_sensitivity_matched"])
        compare(problems, f"s95 {arm} gap", 5, parse_number(r[6]),
                nums["operating_point_analysis"]["per_arm"][arm]["test_sensitivity_matched"] - s["test_sensitivity"])


def check_strata(text, nums, problems):
    rows = rows_by_arm(table_block(text, "按软质量分层的宏平均 AUROC") or [])
    for arm, strata in nums["strata"].items():
        if "usable" not in strata:
            continue
        u = strata["usable"]["macro_auroc"]
        b = strata["bad"]["macro_auroc"]
        if arm not in rows:
            problems.append(f"strata: no row for {arm}")
            continue
        r = rows[arm]
        compare(problems, f"strata {arm} usable", 0, parse_number(r[1]), u)
        compare(problems, f"strata {arm} bad", 1, parse_number(r[2]), b)
        compare(problems, f"strata {arm} diff", 2, parse_number(r[3]), b - u)


def check_counts(text, nums, problems):
    q = nums["quality_counts"]
    for key, cn in (("good", "好"), ("usable", "可用"), ("bad", "差")):
        if f"{cn} {q[key]} 张" not in text:
            problems.append(f"quality count {key}={q[key]} not found")
    for ep, val in nums["prevalence"].items():
        s = f"{val:.3f}"
        if s not in text:
            problems.append(f"prevalence {ep}={s} not found verbatim")


def main() -> int:
    text = MD.read_text(encoding="utf-8")
    nums = json.loads(NUM.read_text(encoding="utf-8"))

    problems: list[str] = []
    for fn in (check_epoch, check_main, check_paired, check_per_head, check_calibration,
               check_youden, check_s95, check_strata, check_counts):
        fn(text, nums, problems)

    if not problems:
        print("OK: every checked number in the manuscript matches the statistics files.")
        return 0

    print(f"{len(problems)} mismatch(es):\n")
    for p in problems:
        print(" -", p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
