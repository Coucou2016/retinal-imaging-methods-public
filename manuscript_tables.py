#!/usr/bin/env python3
"""Generate the markdown bodies of every manuscript table from the stats files.

The manuscript is written by hand, but the tables are mechanical.  This module
is imported by ``sync_manuscript_tables.py`` (rewrite in place) and by
``check_manuscript_numbers.py`` (verify only), so the two can never disagree.

Each table is described by a function returning ``(caption_key, headers, rows)``
where ``caption_key`` is the string that follows ``**表 N**`` in the caption and
is used to locate the table in the manuscript.
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATS = ROOT
NUM = ROOT / "manuscript_numbers.json"

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
EP_CN = {
    "Disease_Risk": "任意病变",
    "DR": "糖尿病视网膜病变",
    "MH": "屈光介质混浊",
    "ODC": "视盘杯盘比异常",
    "TSLN": "豹纹状眼底",
    "DN": "玻璃膜疣",
    "ARMD": "年龄相关性黄斑变性",
    "MYA": "近视性改变",
}
EP_ORDER = ["Disease_Risk", "DR", "MH", "ODC", "TSLN", "DN", "ARMD", "MYA"]


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def numbers() -> dict:
    return json.loads(NUM.read_text(encoding="utf-8"))


def f3(x: float) -> str:
    """Three decimals with a real minus sign for display."""
    return f"{x:.3f}".replace("-", "\u2212")


def sgn(x: float, nd: int = 3) -> str:
    s = f"{x:+.{nd}f}"
    return s.replace("-", "\u2212")


def ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"{f3(lo)} 至 {f3(hi)}"


# --------------------------------------------------------------------- tables
def t_arms() -> tuple[str, list[str], list[list[str]]]:
    headers = ["实验臂", "质量路由", "头结构", "骨干路由", "检验的问题"]
    rows = [
        ["E0", "固定 1/0.5/0", "单头", "无", "公开代码基线"],
        ["E1", "单调有界", "单头", "无", "单任务下的路由增益"],
        ["E2", "固定 1/0.5/0", "八头共享，掩码损失", "无", "多任务本身的影响"],
        ["E3", "单调有界", "八头共享，掩码损失", "无", "两部件组合"],
        ["E4", "自由线性", "八头共享，掩码损失", "无", "单调约束是否必要"],
        ["E5", "单调有界", "八头共享，掩码损失", "质量条件 softmax", "骨干层级再分配"],
    ]
    return "实验臂配置", headers, rows


def t_epochs() -> tuple[str, list[str], list[list[str]]]:
    es = numbers()["epoch_selection"]
    headers = ["实验臂", "训练轮次", "选中轮次", "该轮验证宏平均 AUROC"]
    rows = [[a, str(es[a]["n_epochs"]), str(es[a]["epoch"]), f3(es[a]["val_macro_auroc"])] for a in ARMS]
    return "报告轮次", headers, rows


def t_main() -> tuple[str, list[str], list[list[str]]]:
    h = load("headline.json")
    headers = ["实验臂", "宏平均 AUROC", "自助法区间", "DeLong 区间", "DR AUROC", "宏平均 AP", "平均分病种 ECE", "Brier"]
    rows = []
    for a in ARMS:
        v = h[a]
        rows.append([
            a,
            f3(v["macro_auroc"]),
            ci(v["bootstrap"]["ci_low"], v["bootstrap"]["ci_high"]),
            ci(v["delong"]["ci_low"], v["delong"]["ci_high"]),
            f3(v["dr_auroc"]),
            f3(v["macro_ap"]),
            f3(v["mean_head_ece"]),
            f3(v["pooled_brier"]),
        ])
    return "RFMiD 测试集主要指标", headers, rows


def t_paired() -> tuple[str, list[str], list[list[str]]]:
    p = load("paired.json")
    spec = [
        ("E1 减 E0", "DR", "E1_minus_E0", "DR"),
        ("E3 减 E2", "宏平均", "E3_minus_E2", "macro"),
        ("E3 减 E2", "糖尿病视网膜病变", "E3_minus_E2", "DR"),
        ("E4 减 E3", "宏平均", "E4_minus_E3", "macro"),
        ("E5 减 E3", "宏平均", "E5_minus_E3", "macro"),
        ("E5 减 E3", "糖尿病视网膜病变", "E5_minus_E3", "DR"),
    ]
    headers = ["比较", "终点集合", "差值", "95% 区间", "差值大于零的比例"]
    rows = []
    for label, ep, key, which in spec:
        e = p[key][which]
        rows.append([label, ep, sgn(e["delta"]), f"{sgn(e['ci_low'])} 至 {sgn(e['ci_high'])}", f"{e['frac_positive']:.3f}"])
    return "配对 AUROC 差值", headers, rows


def t_perhead() -> tuple[str, list[str], list[list[str]]]:
    ph = load("per_head.json")["E5"]
    ci_map = load("extended.json")["per_endpoint"]["auroc"]["E5"]
    headers = ["终点", "AUROC", "95% 区间", "平均精度", "期望校准误差", "Brier", "患病率"]
    rows = []
    for ep in EP_ORDER:
        v = ph[ep]
        c = ci_map[ep]
        rows.append([
            EP_CN[ep], f3(v["auroc"]), ci(c["ci_low"], c["ci_high"]),
            f3(v["ap"]), f3(v["ece"]), f3(v["brier"]), f3(v["prevalence"]),
        ])
    vals = [ph[e]["auroc"] for e in EP_ORDER]
    aps = [ph[e]["ap"] for e in EP_ORDER]
    eces = [ph[e]["ece"] for e in EP_ORDER]
    briers = [ph[e]["brier"] for e in EP_ORDER]
    rows.append(["宏平均", f3(sum(vals) / 8), "—", f3(sum(aps) / 8), f3(sum(eces) / 8), f3(sum(briers) / 8), "—"])
    return "E5 在测试集上的分终点表现", headers, rows


def t_calibration() -> tuple[str, list[str], list[list[str]]]:
    arms = numbers()["arms"]
    headers = ["实验臂", "拟合温度", "校准集 ECE 前", "校准集 ECE 后", "测试集平均 ECE 前",
               "测试集平均 ECE 后", "最差终点 ECE", "Brier 前", "Brier 后"]
    rows = []
    for a in ARMS:
        r = arms[a]
        rows.append([
            a, f3(r["temperature_cal64"]), f3(r["ece_on_cal_raw"]), f3(r["ece_on_cal_post"]),
            f3(r["mean_head_ece"]), f3(r["mean_head_ece_after_T"]), f3(r["worst_head_ece"]),
            f3(r["pooled_brier"]), f3(r["brier_after_T"]),
        ])
    return "校准指标", headers, rows


def t_binning() -> tuple[str, list[str], list[list[str]]]:
    ae = load("extended.json")["adaptive_ece"]
    headers = ["实验臂", "等宽分箱 ECE", "等质量分箱 ECE", "两者之差"]
    rows = []
    for a in ARMS:
        w = ae[a]["mean_head_ece_equal_width"]
        m = ae[a]["mean_head_ece_equal_mass"]
        rows.append([a, f3(w), f3(m), sgn(m - w)])
    return "分箱方式敏感性", headers, rows


def t_logit_summary() -> tuple[str, list[str], list[list[str]]]:
    lo = load("logit_offset.json")["per_arm"]
    headers = ["实验臂", "端点数", "相关系数 r", "斜率", "平均绝对缺口", "最大绝对缺口", "平均 ECE"]
    rows = []
    for a in ["E2", "E3", "E4", "E5"]:
        s = lo[a]["summary"]
        rows.append([
            a, str(s["n_heads"]), f3(s["r_mean_logit_vs_logit_prevalence"]),
            f3(s["slope_mean_logit_on_logit_prevalence"]), f3(s["mean_abs_gap"]),
            f3(s["max_abs_gap"]), f3(s["mean_head_ece"]),
        ])
    return "logit 中心与 logit 流行率的关系", headers, rows


def t_logit_endpoint() -> tuple[str, list[str], list[list[str]]]:
    lo = load("logit_offset.json")["per_arm"]["E5"]["per_head"]
    co = load("calibration_offset.json")["per_arm"]["E5"]["per_head"]
    headers = ["终点", "患病率", "logit 流行率", "平均 logit", "缺口", "拟合偏移",
               "ECE 原始", "ECE 流行率偏移", "ECE Platt"]
    rows = []
    for ep in EP_ORDER:
        a = lo[ep]
        b = co[ep]
        rows.append([
            EP_CN[ep], f3(a["prevalence"]), sgn(a["logit_prevalence"], 2), sgn(a["mean_logit"], 2),
            sgn(a["gap"], 2), sgn(b["offset_fitted"], 2), f3(b["ece_none"]),
            f3(b["ece_offset_prev"]), f3(b["ece_platt"]),
        ])
    return "E5 的分终点 logit 偏移", headers, rows


def t_ladder() -> tuple[str, list[str], list[list[str]]]:
    rh = load("recalibration_holdout.json")["per_arm"]
    co = load("calibration_offset.json")["per_arm"]
    headers = ["实验臂", "原始", "温度", "流行率偏移", "拟合偏移", "Platt", "保序回归"]
    rows = []
    for a in ARMS:
        s = rh[a]["summary"]
        rows.append([
            a, f3(s["macro_ece_none"]), f3(s["macro_ece_temperature"]),
            f3(co[a]["summary"]["macro_ece_offset_prev"]), f3(co[a]["summary"]["macro_ece_offset_fit"]),
            f3(s["macro_ece_platt"]), f3(s["macro_ece_isotonic"]),
        ])
    return "事后校正阶梯", headers, rows


def t_youden() -> tuple[str, list[str], list[list[str]]]:
    op = numbers()["operating_points"]
    headers = ["实验臂", "阈值", "验证集灵敏度", "验证集特异度", "测试集灵敏度", "测试集特异度"]
    rows = []
    for a in ARMS:
        y = op[a]["DR"]["youden"]
        rows.append([a, f3(y["threshold"]), f3(y["sensitivity"]), f3(y["specificity"]),
                     f3(y["test_sensitivity"]), f3(y["test_specificity"])])
    return "约登工作点", headers, rows


def t_spec95() -> tuple[str, list[str], list[list[str]]]:
    op = numbers()["operating_points"]
    opa = numbers()["operating_point_analysis"]["per_arm"]
    headers = ["实验臂", "验证集阈值", "验证集灵敏度", "测试集灵敏度", "测试集特异度",
               "事后上界灵敏度", "搬移落差"]
    rows = []
    for a in ARMS:
        s = op[a]["DR"]["sens_at_95spec"]
        oracle = opa[a]["test_sensitivity_matched"]
        rows.append([a, f3(s["threshold"]), f3(s["sensitivity"]), f3(s["test_sensitivity"]),
                     f3(s["test_specificity"]), f3(oracle), f3(oracle - s["test_sensitivity"])])
    return "高特异度工作点", headers, rows


def pm(v: dict) -> str:
    """Mean with a between-seed standard deviation, both to three decimals."""
    sd = v.get("sd")
    return f3(v["mean"]) if sd is None else f"{f3(v['mean'])} ± {f3(sd)}"


def seed_tt() -> dict:
    p = ROOT / "threshold_transfer_replication.json"
    return json.loads(p.read_text(encoding="utf-8"))


def t_spec95_seed() -> tuple[str, list[str], list[list[str]]]:
    tt = seed_tt()["per_arm"]
    headers = ["实验臂", "冻结阈值灵敏度", "事后匹配灵敏度", "搬移落差", "超额特异度",
               "局部 ROC 斜率", "低假阳区 pAUC"]
    rows = []
    for a in ARMS:
        g = tt[a]["aggregate"]
        rows.append([
            a, pm(g["test_sensitivity_frozen"]), pm(g["test_sensitivity_matched"]),
            pm(g["transfer_loss"]), pm(g["excess_specificity"]),
            pm(g["local_roc_slope"]), pm(g["test_partial_auroc"]),
        ])
    return "高特异度工作点的四次重复", headers, rows


def t_utility_seed() -> tuple[str, list[str], list[list[str]]]:
    tt = seed_tt()["per_arm"]
    headers = ["实验臂", "PPV", "NPV", "每百人漏诊", "每百人误报"]
    rows = []
    for a in ARMS:
        g = tt[a]["aggregate"]
        rows.append([a, pm(g["ppv"]), pm(g["npv"]),
                     f"{g['missed_per_100']['mean']:.2f} ± {g['missed_per_100']['sd']:.2f}",
                     f"{g['false_alarm_per_100']['mean']:.2f} ± {g['false_alarm_per_100']['sd']:.2f}"])
    return "固定工作点上的临床效用四次重复", headers, rows


def t_utility() -> tuple[str, list[str], list[list[str]]]:
    cu = load("clinical_utility.json")
    headers = ["实验臂", "TP", "FP", "TN", "FN", "PPV", "NPV", "每百人漏诊", "每百人误报"]
    rows = []
    for a in ARMS:
        f = cu[a]["frozen"]
        rows.append([a, str(f["tp"]), str(f["fp"]), str(f["tn"]), str(f["fn"]),
                     f3(f["ppv"]), f3(f["npv"]), f"{f['missed_per_100_screened']:.2f}",
                     f"{f['false_alarms_per_100_screened']:.2f}"])
    return "固定工作点上的临床效用", headers, rows


def t_strata() -> tuple[str, list[str], list[list[str]]]:
    st = load("quality_strata.json")
    headers = ["实验臂", "可用层 AUROC", "差层 AUROC", "两层差值"]
    rows = []
    for a in ARMS:
        u = st[a]["usable"]["macro_auroc"]
        b = st[a]["bad"]["macro_auroc"]
        rows.append([a, f3(u), f3(b), sgn(b - u)])
    return "按软质量分层的宏平均 AUROC", headers, rows


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(x: float) -> str:
    """Format a small number as a x 10^-n with unicode exponents."""
    if x == 0:
        return "0"
    mant, exp = f"{abs(x):.1e}".split("e")
    if float(mant) == 10.0:
        mant, exp = "1.0", str(int(exp) + 1)
    return f"{mant}×10{str(int(exp)).translate(SUP)}"


def t_router() -> tuple[str, list[str], list[list[str]]]:
    co = load("router_coefficients.json")
    it = load("router_intervention.json")
    headers = ["实验臂", "好质量权重", "可用质量权重", "差质量权重", "最大偏离",
               "钳制后 AUROC 变化", "最大概率变化"]
    rows = []
    for a in ["E0", "E1", "E3", "E4", "E5"]:
        w = co[a]["normalised_to_good"]  # good, usable, bad
        e = it.get(a)
        rows.append([
            a, f3(w[0]), f3(w[1]), sgn(w[2], 4),
            f3(co[a]["max_abs_deviation"]),
            "—" if e is None else sci(e["delta"]),
            "—" if e is None else f3(e["max_abs_prob_change"]),
        ])
    return "路由器的实际取值与推理期干预", headers, rows


def t_robust() -> tuple[str, list[str], list[list[str]]]:
    rb = load("endpoint_robustness.json")
    headers = ["比较", "全终点差值", "逐一剔除后的最小差值", "逐一剔除后的最大差值", "符号是否保持"]
    rows = []
    for key, name in (
        ("E3_minus_E2", "E3 减 E2"),
        ("E4_minus_E3", "E4 减 E3"),
        ("E5_minus_E3", "E5 减 E3"),
    ):
        r = rb[key]
        rows.append([name, sgn(r["full_delta"]), sgn(r["min_delta"]), sgn(r["max_delta"]),
                     "保持" if r["sign_stable"] else "不保持"])
    return "逐步剔除单一终点后的宏平均差值", headers, rows


TABLES = [
    t_arms, t_epochs, t_main, t_paired, t_perhead, t_calibration, t_binning,
    t_logit_summary, t_logit_endpoint, t_ladder, t_youden, t_spec95,
    t_spec95_seed, t_utility, t_utility_seed, t_strata, t_router, t_robust,
]


def render(headers: list[str], rows: list[list[str]]) -> str:
    out = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("|" + "|".join(r) + "|")
    return "\n".join(out)


if __name__ == "__main__":
    for i, fn in enumerate(TABLES, 1):
        key, headers, rows = fn()
        print(f"===== 表 {i} {key}")
        print(render(headers, rows))
        print()
