#!/usr/bin/env python3
"""分临床时间窗分析 MAP 与脑氧/PSI：全部20秒配对点 + 项目同款 pyGAM。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s, te

plt = None


SEED = int(os.getenv("WINDOW_GAM_SEED", "20260820"))
N_BOOT = int(os.getenv("WINDOW_GAM_N_BOOT", "200"))
TENSOR_K = int(os.getenv("WINDOW_GAM_TENSOR_K", "6"))
COV_K = int(os.getenv("WINDOW_GAM_COV_K", "4"))
LAM_GRID = np.asarray([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
MAP_GRID = np.arange(45.0, 146.0, 1.0)
FIT_TIMEOUT_SECONDS = int(os.getenv("WINDOW_GAM_FIT_TIMEOUT", "180"))

WINDOWS = [
    {"key": "ward_preop", "label": "病房术前监测", "outcome": "sto2_mean", "outcome_label": "脑氧 StO2 (%)"},
    {"key": "preinduction_5_1", "label": "诱导前 −5～−1 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "postinduction_all", "label": "诱导后全程（总览）", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "induction_to_intubation", "label": "诱导至插管", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "intubation_to_surgery", "label": "插管至手术开始", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_0_60", "label": "手术后 0–60 min（总览）", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_0_30", "label": "手术后 0–30 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_0_5", "label": "手术后 0–5 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_5_10", "label": "手术后 5–10 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_10_15", "label": "手术后 10–15 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_15_20", "label": "手术后 15–20 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_20_25", "label": "手术后 20–25 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_25_30", "label": "手术后 25–30 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_30_60", "label": "手术后 30–60 min", "outcome": "psi_median", "outcome_label": "PSI"},
    {"key": "surgery_ge60", "label": "手术后 ≥60 min", "outcome": "psi_median", "outcome_label": "PSI"},
]

DRUG_SMOOTH = [
    "prop_bolus10_log", "prop_rate_log", "ket_bolus30_log",
    "midaz_bolus60_log", "dex_bolus60_log", "fent_bolus30_log",
]
DRUG_FACTOR = [
    "ket_infusion_active", "vasopressor_recent_bolus",
    "vasopressor_infusion_active",
]
STATIC_SMOOTH = ["age", "bmi", "preop_map"]
STATIC_FACTOR = [
    "sex", "asa", "procedure_complex", "pmh_heart_failure",
    "pmh_hypertension", "pmh_diabetes",
]


# ---------------------------------------------------------------------------
# 研究队列名单（Study roster）
# 名单是纳入分析的唯一依据：不在名单上的 ID 属于录入错误，会被报告而不是
# 悄悄分析；EXCLUDED_IDS 由研究者决定剔除。
# ---------------------------------------------------------------------------
EXCLUDED_IDS = frozenset({
    "IUMH2025121201", "IUMH2026020301", "IUMH2026031001",
})

COHORT_IDS = frozenset({
    "IUMH2025120901", "IUMH2025120902", "IUMH2025121501", "IUMH2025121801",
    "IUMH2025122901", "IUMH2025123001", "IUMH2025123101", "IUMH2025123102",
    "IUMH2026010501", "IUMH2026010601", "IUMH2026010602", "IUMH2026010901",
    "IUMH2026011301", "IUMH2026011302", "IUMH2026011501", "IUMH2026011502",
    "IUMH2026011601", "IUMH2026011602", "IUMH2026012201", "IUMH2026012301",
    "IUMH2026012302", "IUMH2026012701", "IUMH2026012702", "IUMH2026012901",
    "IUMH2026012902", "IUMH2026013001", "IUMH2026013002", "IUMH2026020501",
    "IUMH2026021001", "IUMH2026021002", "IUMH2026021901", "IUMH2026021902",
    "IUMH2026022001", "IUMH2026022401", "IUMH2026030301", "IUMH2026030302",
    "IUMH2026030501", "IUMH2026030502", "IUMH2026030401", "IUMH2026030602",
    "IUMH2026030901", "IUMH2026031002", "IUMH2026031701", "IUMH2026031702",
    "IUMH2026031801", "IUMH2026031901", "IUMH2026031902", "IUMH2026032001",
    "IUMH2026032002", "IUMH2026032401", "IUMH2026040201", "IUUH2026030601",
    "IUUH2026030602", "IUUH2026030603", "IUUH2026030901", "IUUH2026030902",
    "IUMH2026041701", "IUMH2026041702", "IUMH2026042101", "IUMH2026042102",
    "IUMH2026042301", "IUMH2026042401", "IUMH2026042701", "IUMH2026042702",
    "IUMH2026042801", "IUMH2026042802", "IUMH2026050501", "IUUH2026031001",
    "IUMH2026050502", "IUMH2026051201", "IUMH2026051202", "IUMH2026051203",
    "IUMH2026051901", "IUMH2026052001", "IUMH2026052002", "IUMH2026052101",
    "IUUH2026042801", "IUUH2026050601", "IUUH2026050602", "IUMH2026052201",
    "IUMH2026052202", "IUMH2026060101", "IUMH2026060102", "IUMH2026061601",
    "IUMH2026061701", "IUMH2026061702", "IUMH2026061901", "IUMH2026062301",
    "IUMH2026062302", "IUMH2026062501", "IUMH2026062502", "IUMH2026063001",
    "IUMH2026063002",
})

# 已知的 REDCap ID 录入错误 -> 正确的研究 ID。
REDCAP_ID_ALIASES = {
    "IUMH202601601": "IUMH2026011601",
    "IUMH2026010601-20260105": "IUMH2026010501",
}

_BAD_ALIASES = set(REDCAP_ID_ALIASES.values()) - COHORT_IDS
if _BAD_ALIASES:
    raise ValueError(f"REDCAP_ID_ALIASES 指向名单外的 ID: {sorted(_BAD_ALIASES)}")

# CI（心指数）是张量项 te(map, ci) 的一部分。缺失 CI 的时间点无法进入模型，
# 但它们仍然可以进入散点图。设为 False 时缺失 CI 的行只退出建模，不退出队列。
REQUIRE_CI = os.getenv("WINDOW_GAM_REQUIRE_CI", "1") != "0"


def normalize_subject_id(values: pd.Series) -> pd.Series:
    """统一 ID 格式；IUMH 与 IUUH 是不同医院，不做合并。"""
    return (
        values.astype("string").str.strip().str.upper()
        .str.replace(r"[\s_-]+", "", regex=True)
    )


def apply_id_aliases(values: pd.Series) -> pd.Series:
    aliases = {
        normalize_subject_id(pd.Series([bad])).iloc[0]: good
        for bad, good in REDCAP_ID_ALIASES.items()
    }
    return values.replace(aliases)


def find_redcap_column(redcap: pd.DataFrame, prefix: str) -> str | None:
    """按前缀查找 REDCap 列，容忍首尾空格与问卷文字改动。"""
    prefix = prefix.strip().lower()
    for column in redcap.columns:
        if str(column).strip().lower().startswith(prefix):
            return column
    return None


def reconcile_with_roster(points: pd.DataFrame) -> pd.DataFrame:
    """对照名单核对 ID，并在过滤前把每一类差异都打印出来。"""
    present = set(points["subject_id"].dropna().unique())
    excluded_present = sorted(present & EXCLUDED_IDS)
    off_roster = sorted(present - COHORT_IDS - EXCLUDED_IDS)
    missing = sorted(COHORT_IDS - present)

    log("\n" + "=" * 68)
    log("队列核对 ROSTER RECONCILIATION")
    log("=" * 68)
    log(f"  名单人数（已剔除 EXCLUDED_IDS） .............. {len(COHORT_IDS)}")
    log(f"  名单中出现在时序数据里的患者 ................ {len(present & COHORT_IDS)}")
    log(f"  名单中完全没有时序数据的患者 ................ {len(missing)}")
    log(f"  研究者剔除且出现在数据中的 ID ............... {len(excluded_present)}")
    log(f"  数据中存在但不在名单上的 ID ................. {len(off_roster)}")
    for subject_id in excluded_present:
        log(f"      [剔除] {subject_id}")
    for subject_id in off_roster:
        rows = int(points["subject_id"].eq(subject_id).sum())
        log(f"      [不在名单-疑似录入错误] {subject_id} ({rows} 行)")
    for subject_id in missing:
        log(f"      [无时序数据] {subject_id}")

    return points.loc[points["subject_id"].isin(COHORT_IDS)].copy()


def log(message: str) -> None:
    print(message, flush=True)


def setup_plotting():
    """拟合完成后才导入 Matplotlib，避免字体缓存阻塞模型。"""
    global plt
    if plt is not None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib import font_manager

    available = {item.name for item in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei", "SimHei", "DengXian",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    pyplot.rcParams["font.family"] = next(
        (name for name in candidates if name in available), "DejaVu Sans"
    )
    pyplot.rcParams["axes.unicode_minus"] = False
    plt = pyplot


@contextlib.contextmanager
def fit_timeout():
    """防止极少数病态 bootstrap 设计矩阵让单次 pyGAM 无限等待。"""
    # SIGALRM/ITIMER_REAL 仅在类 Unix 系统可用；Windows 直接拟合，
    # 由外层作业/进程监控负责兜底。
    if (
        FIT_TIMEOUT_SECONDS <= 0
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "ITIMER_REAL")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def handle_timeout(signum, frame):
        raise TimeoutError(f"单次 pyGAM 拟合超过 {FIT_TIMEOUT_SECONDS} 秒")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, FIT_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


WINDOW_REQUIRED_ANCHORS = {
    "ward_preop": [],
    "preinduction_5_1": ["seconds_from_induction"],
    "postinduction_all": ["seconds_from_induction"],
    "induction_to_intubation": ["seconds_from_induction", "seconds_from_intubation"],
    "intubation_to_surgery": ["seconds_from_intubation", "seconds_from_surgery_start"],
}


def window_mask(data: pd.DataFrame, key: str) -> pd.Series:
    needed = WINDOW_REQUIRED_ANCHORS.get(key, ["seconds_from_surgery_start"])
    unusable = [c for c in needed if c not in data.columns or data[c].isna().all()]
    if unusable:
        raise RuntimeError(
            f"窗口 {key} 需要的时间锚点列缺失或全为空：{unusable}。"
            "请改用包含这些列的时序文件，或跳过该窗口。"
        )

    ind = data["seconds_from_induction"]
    intu = data["seconds_from_intubation"]
    surg = data["seconds_from_surgery_start"]
    if key == "ward_preop":
        return (
            data["preop_start"].notna() & data["preop_end"].notna()
            & data["preop_start"].lt(data["preop_end"])
            & data["timepoint"].ge(data["preop_start"])
            & data["timepoint"].le(data["preop_end"])
        )
    if key == "preinduction_5_1":
        return ind.between(-300, -60, inclusive="both")
    if key == "postinduction_all":
        return ind.ge(0)
    if key == "induction_to_intubation":
        return ind.ge(0) & intu.lt(0)
    if key == "intubation_to_surgery":
        return intu.ge(0) & surg.lt(0)
    if key == "surgery_0_60":
        return surg.ge(0) & surg.lt(3600)
    if key == "surgery_ge60":
        return surg.ge(3600)
    bounds = key.removeprefix("surgery_").split("_")
    lower, upper = int(bounds[0]) * 60, int(bounds[1]) * 60
    return surg.ge(lower) & surg.lt(upper)


def time_in_window(data: pd.DataFrame, key: str) -> pd.Series:
    if key == "ward_preop":
        return (data["timepoint"] - data["preop_start"]).dt.total_seconds() / 60.0
    if key == "preinduction_5_1":
        return (data["seconds_from_induction"] + 300) / 60.0
    if key == "postinduction_all":
        return data["seconds_from_induction"] / 60.0
    if key == "induction_to_intubation":
        return data["seconds_from_induction"] / 60.0
    if key == "intubation_to_surgery":
        return data["seconds_from_intubation"] / 60.0
    offset = 3600 if key == "surgery_ge60" else 0
    if key.startswith("surgery_") and key not in {"surgery_0_60", "surgery_ge60"}:
        offset = int(key.removeprefix("surgery_").split("_")[0]) * 60
    return (data["seconds_from_surgery_start"] - offset) / 60.0


def _report_redcap_match(points: pd.DataFrame, how: str) -> None:
    """报告有多少患者拿到了 REDCap 术前时间，并列出没拿到的人。

    没有术前起止时间的患者会在 ward_preop 窗口整体消失，所以必须点名。
    """
    has_times = points.groupby("subject_id")["preop_start"].apply(
        lambda column: bool(column.notna().any())
    )
    matched = sorted(has_times[has_times].index)
    unmatched = sorted(has_times[~has_times].index)

    log(f"  REDCap 术前时间按{how}匹配成功：{len(matched)} 名患者")
    if unmatched:
        log(f"  时序数据里有、但 REDCap 无术前起止时间：{len(unmatched)} 名患者"
            "（ward_preop 窗口会缺席）")
        for subject_id in unmatched:
            log(f"      [无REDCap术前时间] {subject_id}")


def load_timepoint_data(timeseries_path: Path, covariate_path: Path, redcap_path: Path) -> pd.DataFrame:
    """保留每个HemoSphere 20秒时间点；分钟级药物协变量向下匹配。"""
    wanted = {
        "subject_id", "screening_id", "timestamp_local", "map", "ci",
        "sto2_mean", "brain_bilateral_usable", "psi_median", "artf_median",
        "eeg_qc_pass", "seconds_from_induction", "seconds_from_intubation",
        "seconds_from_surgery_start",
    }
    raw = pd.read_csv(timeseries_path, usecols=lambda c: c in wanted, low_memory=False)

    # usecols 是"有就读"，缺列不会报错，于是后面才在 window_mask 里
    # 抛 KeyError。这里先补齐列名，缺的列填 NaN 并明确提示。
    absent = sorted(wanted - set(raw.columns))
    if absent:
        log("\n注意：时序文件缺少以下列，已按缺失值处理：")
        for column in absent:
            log(f"      {column}")
            raw[column] = np.nan
        log("      依赖这些列的时间窗将没有数据，会被明确报错而不是静默出错。")

    raw["timepoint"] = pd.to_datetime(raw["timestamp_local"], errors="coerce")
    raw["minute"] = raw["timepoint"].dt.floor("min")
    numeric = [c for c in raw.columns if c not in {"subject_id", "screening_id", "timestamp_local", "timepoint", "minute"}]
    for col in numeric:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    points = raw.drop(columns="timestamp_local")

    cov = pd.read_csv(covariate_path, low_memory=False)
    cov["minute"] = pd.to_datetime(cov["minute"], errors="coerce")
    cov_cols = ["subject_id", "minute"] + DRUG_SMOOTH + DRUG_FACTOR
    cov_cols = [c for c in cov_cols if c in cov.columns]
    dynamic = cov[cov_cols].drop_duplicates(["subject_id", "minute"])
    points = points.merge(dynamic, on=["subject_id", "minute"], how="left")

    static_cols = ["subject_id", "procedure"] + STATIC_SMOOTH + [
        c for c in STATIC_FACTOR if c != "procedure_complex"
    ]
    static_cols = [c for c in static_cols if c in cov.columns]
    static = cov[static_cols].drop_duplicates("subject_id")
    points = points.merge(static, on="subject_id", how="left")
    points["procedure_complex"] = np.where(
        points.get("procedure", "").astype(str).str.contains("Complex|OPEN", case=False, regex=True),
        "complex_or_open", "straightforward_laparoscopic",
    )
    for col in DRUG_SMOOTH + DRUG_FACTOR:
        if col not in points:
            points[col] = 0.0
        points[col] = pd.to_numeric(points[col], errors="coerce").fillna(0.0)

    # ---- 统一研究 ID，并对照名单核对 --------------------------------- #
    points["subject_id_raw"] = points["subject_id"].astype("string")
    points["subject_id"] = apply_id_aliases(
        normalize_subject_id(points["subject_id"])
    )
    points = reconcile_with_roster(points)

    # ---- REDCap 术前监测起止时间 -------------------------------------- #
    # 整表读入：usecols 写死列名时，问卷文字或空格稍有变动就会直接报错。
    redcap = pd.read_csv(redcap_path, low_memory=False)
    start_col = find_redcap_column(redcap, "Time monitoring started in preop:")
    end_col = find_redcap_column(redcap, "Time that monitoring ended in preop:")
    if start_col is None or end_col is None:
        raise RuntimeError(
            "REDCap 中找不到术前监测起止时间列；ward_preop 窗口无法构建。"
        )

    # 只取需要的三列另建小表，避免往 740 列的宽表里反复插列。
    redcap = pd.DataFrame({
        "_start": pd.to_datetime(redcap[start_col], errors="coerce"),
        "_end": pd.to_datetime(redcap[end_col], errors="coerce"),
        **{column: redcap[column] for column in redcap.columns},
    })
    redcap["preop_start"] = redcap.pop("_start")
    redcap["preop_end"] = redcap.pop("_end")

    # 研究 ID 形如 IUMH2026030501，pd.to_numeric 会把它整列变成 NaN，
    # 于是合并不上、preop 时间全空、ward_preop 窗口一个点都没有。
    # 因此优先按研究 ID 合并，只有在确实是数字编号时才退回 screening_id。
    id_col = next(
        (
            column for column in redcap.columns
            if normalize_subject_id(redcap[column])
            .str.fullmatch(r"IU(?:MH|UH)\d+", na=False).any()
        ),
        None,
    )

    if id_col is not None:
        redcap["subject_id"] = apply_id_aliases(
            normalize_subject_id(redcap[id_col])
        )
        lookup = (
            redcap.loc[redcap["subject_id"].notna(),
                       ["subject_id", "preop_start", "preop_end"]]
            .sort_values(["preop_start", "preop_end"])
            .drop_duplicates("subject_id", keep="first")
        )
        points = points.merge(lookup, on="subject_id", how="left")
        _report_redcap_match(points, "研究 ID")
    elif "screening_id" in points.columns and "Screening ID" in redcap.columns:
        redcap["screening_id"] = pd.to_numeric(redcap["Screening ID"],
                                               errors="coerce")
        points["screening_id"] = pd.to_numeric(points["screening_id"],
                                               errors="coerce")
        lookup = (
            redcap.loc[redcap["screening_id"].notna(),
                       ["screening_id", "preop_start", "preop_end"]]
            .drop_duplicates("screening_id")
        )
        points = points.merge(lookup, on="screening_id", how="left")
        _report_redcap_match(points, "screening_id")
    else:
        raise RuntimeError("REDCap 与时序数据之间找不到可用的 ID 列。")

    return points


def encode_window(data: pd.DataFrame, spec: dict):
    d = data.loc[window_mask(data, spec["key"])].copy()
    outcome = spec["outcome"]
    d["outcome"] = pd.to_numeric(d[outcome], errors="coerce")
    d["time_in_window"] = time_in_window(d, spec["key"])
    # ---- 逐个筛选条件统计患者流失，避免"人不知道去哪了" ---------------- #
    stages = [
        ("窗口内有数据 in window", pd.Series(True, index=d.index)),
        ("ID/时间戳有效", d["subject_id"].notna() & d["timepoint"].notna()),
        ("MAP 20-160", d["map"].between(20, 160)),
    ]
    if REQUIRE_CI:
        stages.append(("CI 0.5-8（张量项必需）", d["ci"].between(0.5, 8)))
    if outcome == "sto2_mean":
        stages.append(("StO2 0-100", d["outcome"].between(0, 100)))
        stages.append(("双侧脑氧可用 bilateral==1",
                       d["brain_bilateral_usable"].eq(1)))
    else:
        stages.append(("PSI 0-100", d["outcome"].between(0, 100)))
        stages.append(("EEG 伪迹 artf<=20", d["artf_median"].le(20)))

    audit_rows = []
    keep = pd.Series(True, index=d.index)
    for label, condition in stages:
        before = int(d.loc[keep, "subject_id"].nunique())
        keep &= condition.fillna(False)
        after = int(d.loc[keep, "subject_id"].nunique())
        audit_rows.append({
            "window": spec["key"], "stage": label,
            "patients_before": before, "patients_after": after,
            "patients_lost": before - after,
            "rows_remaining": int(keep.sum()),
        })
    inclusion_audit = pd.DataFrame(audit_rows)

    log(f"\n  [{spec['label']}] 患者纳入流程：")
    for row in audit_rows:
        marker = "  <-- 流失" if row["patients_lost"] else ""
        log(f"      {row['stage']:<28} {row['patients_after']:>3} 名患者"
            f"（-{row['patients_lost']}）{marker}")

    d = d.loc[keep].copy()
    if d["subject_id"].nunique() < 20:
        raise RuntimeError(f"{spec['key']} 仅 {d['subject_id'].nunique()} 名患者，停止拟合")

    smooth = ["time_in_window"] + STATIC_SMOOTH
    factors = STATIC_FACTOR.copy()
    support = []
    if outcome != "sto2_mean":
        for col in DRUG_SMOOTH + DRUG_FACTOR:
            positive = d[col].fillna(0).gt(0)
            n_subjects = int(d.loc[positive, "subject_id"].nunique())
            n_rows = int(positive.sum())
            selected = n_subjects >= 3 and n_rows >= 10 and d[col].std() > 0
            support.append({"variable": col, "subjects_exposed": n_subjects, "rows_exposed": n_rows, "selected": selected})
            if selected:
                (factors if col in DRUG_FACTOR else smooth).append(col)

    for col in ["map", "ci", "outcome"] + smooth:
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d[col] = d[col].fillna(d[col].median()).astype(float)
    kept_factors = []
    encodings = {}
    for col in factors:
        if col in {"sex", "procedure_complex"}:
            cat = pd.Categorical(d[col].fillna("Missing"))
            d[col] = cat.codes.astype(float)
            encodings[col] = list(map(str, cat.categories))
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(float)
        if d[col].nunique() > 1:
            kept_factors.append(col)
    factors = kept_factors
    features = ["map", "ci"] + smooth + factors
    return (d.sort_values(["subject_id", "timepoint"]), features, smooth,
            factors, pd.DataFrame(support), encodings, inclusion_audit)


def build_terms(n_smooth: int, n_factor: int):
    terms = te(0, 1, n_splines=[TENSOR_K, TENSOR_K])
    for pos in range(2, 2 + n_smooth):
        terms += s(pos, n_splines=COV_K)
    for pos in range(2 + n_smooth, 2 + n_smooth + n_factor):
        terms += f(pos)
    return terms


def fit_gam(data: pd.DataFrame, features: list[str], smooth: list[str], factors: list[str], lam: float):
    with fit_timeout():
        return LinearGAM(
            build_terms(len(smooth), len(factors)),
            lam=lam,
            fit_intercept=True,
        ).fit(data[features].to_numpy(float), data["outcome"].to_numpy(float))


def tune_gam(data, features, smooth, factors):
    rows, best = [], None
    for lam in LAM_GRID:
        try:
            gam = fit_gam(data, features, smooth, factors, float(lam))
            gcv = float(gam.statistics_.get("GCV", np.inf))
            rows.append({"lambda": lam, "GCV": gcv, "ok": True})
            if best is None or gcv < best[0]:
                best = (gcv, float(lam), gam)
        except Exception as exc:
            rows.append({"lambda": lam, "GCV": np.nan, "ok": False, "error": str(exc)})
    if best is None:
        raise RuntimeError("所有 lambda 候选均拟合失败")
    return best[2], best[1], pd.DataFrame(rows)


def reference_values(data, features, factors):
    result = {}
    for col in features:
        mode = data[col].mode(dropna=True)
        result[col] = float(mode.iloc[0]) if col in factors else float(data[col].median())
    return result


def prediction_matrix(features, reference, grid):
    X = np.zeros((len(grid), len(features)), float)
    for j, col in enumerate(features):
        X[:, j] = reference[col]
    X[:, features.index("map")] = grid
    return X


def patient_bootstrap(data, rng):
    ids = data["subject_id"].drop_duplicates().to_numpy()
    draws = rng.choice(ids, size=len(ids), replace=True)
    pieces = []
    for draw_number, patient_id in enumerate(draws):
        part = data.loc[data["subject_id"].eq(patient_id)].copy()
        part["bootstrap_cluster"] = draw_number
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def row_bootstrap(data, rng):
    """直接从全部20秒记录中有放回抽取同样数量的行。"""
    take = rng.integers(0, len(data), size=len(data))
    return data.iloc[take].reset_index(drop=True)


def timepoint_scatter(data, spec):
    """每一行就是一名患者在一个20秒时间点的MAP–结局配对。"""
    return data[["subject_id", "timepoint", "map", "outcome"]].assign(
        window=spec["key"], window_label=spec["label"], outcome_name=spec["outcome"]
    )


def derivative_summary(grid, point, boots):
    point_d = np.gradient(point, grid)
    boot_d = np.vstack([np.gradient(row, grid) for row in boots])
    return pd.DataFrame({
        "map": grid, "derivative": point_d,
        "ci_low": np.percentile(boot_d, 2.5, axis=0),
        "ci_high": np.percentile(boot_d, 97.5, axis=0),
    })


def plot_single_window(output: Path, spec: dict, scatter: pd.DataFrame, curve: pd.DataFrame, patients: int, bootstrap_unit: str):
    """保存单窗口全部20秒散点图和调整后GAM图。"""
    setup_plotting()
    n_points = len(scatter)
    point_size = 7 if n_points < 5000 else 4
    alpha = 0.12 if n_points < 5000 else 0.05

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        scatter["map"], scatter["outcome"], s=point_size, alpha=alpha,
        color="#3F8F63", edgecolors="none", rasterized=True,
    )
    ax.set(
        xlabel="MAP (mmHg)", ylabel=spec["outcome_label"],
        title=f"{spec['label']}：全部20秒配对点",
    )
    ax.text(0.03, 0.97, f"N = {patients}名患者，n = {n_points:,}个20秒点", transform=ax.transAxes, va="top")
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(output / "scatter_all_20s_points.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "scatter_all_20s_points.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(curve["map"], curve["estimate"], color="#3366A6", lw=2.3)
    ax.fill_between(curve["map"], curve["ci_low"], curve["ci_high"], color="#3366A6", alpha=0.2)
    ax.set(
        xlabel="MAP (mmHg)", ylabel=f"调整后 {spec['outcome_label']}",
        title=f"{spec['label']}：MAP–{spec['outcome_label']} pyGAM（{'行级' if bootstrap_unit == 'row' else '患者级'} bootstrap 95% CI）",
    )
    ax.text(0.03, 0.97, f"N = {patients}名患者，n = {n_points:,}个20秒点", transform=ax.transAxes, va="top")
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(output / "gam_all_20s_points.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "gam_all_20s_points.pdf", bbox_inches="tight")
    plt.close(fig)


def fit_window(timeseries, covariates, redcap, spec, output_root, bootstrap_unit="patient"):
    output = output_root / spec["key"]
    output.mkdir(parents=True, exist_ok=True)
    all_points = load_timepoint_data(timeseries, covariates, redcap)
    data, features, smooth, factors, support, encodings, inclusion_audit = (
        encode_window(all_points, spec)
    )
    p01, p99 = data["map"].quantile([0.01, 0.99])
    grid = MAP_GRID[(MAP_GRID >= np.ceil(p01)) & (MAP_GRID <= np.floor(p99))]
    if len(grid) < 20:
        raise RuntimeError("MAP 有效支持范围过窄")

    log(f"{spec['label']}：{len(data)} 个20秒点，{data.subject_id.nunique()} 名患者，MAP支持 {grid.min():.0f}–{grid.max():.0f}")
    gam, chosen_lam, selection = tune_gam(data, features, smooth, factors)
    ref = reference_values(data, features, factors)
    Xp = prediction_matrix(features, ref, grid)
    point = gam.predict(Xp)

    rng = np.random.default_rng(SEED + WINDOWS.index(spec) * 1009)
    curves, failures = [], []
    attempt = 0
    max_attempts = N_BOOT + 20
    while len(curves) < N_BOOT and attempt < max_attempts:
        attempt += 1
        sample = row_bootstrap(data, rng) if bootstrap_unit == "row" else patient_bootstrap(data, rng)
        try:
            boot_gam = fit_gam(sample, features, smooth, factors, chosen_lam)
            curves.append(boot_gam.predict(Xp))
        except Exception as exc:
            failures.append({"attempt": attempt, "error": str(exc)})
        if attempt == 1 or attempt % 20 == 0 or len(curves) == N_BOOT:
            log(f"bootstrap 尝试 {attempt}/{max_attempts}，成功 {len(curves)}/{N_BOOT}")
    if not curves:
        raise RuntimeError(f"{bootstrap_unit} bootstrap 全部失败")
    boot = np.vstack(curves)

    curve = pd.DataFrame({
        "window": spec["key"], "window_label": spec["label"],
        "outcome_name": spec["outcome"], "map": grid,
        "estimate": point, "ci_low": np.percentile(boot, 2.5, axis=0),
        "ci_high": np.percentile(boot, 97.5, axis=0),
        "successful_bootstraps": len(boot), "bootstrap_unit": bootstrap_unit,
    })
    deriv = derivative_summary(grid, point, boot).assign(window=spec["key"], window_label=spec["label"])
    scatter = timepoint_scatter(data, spec)
    contrast_rows = []
    reference_map = 75.0
    if grid.min() <= reference_map <= grid.max():
        reference_point = float(np.interp(reference_map, grid, point))
        reference_boot = np.asarray([np.interp(reference_map, grid, row) for row in boot])
        for target_map in [65.0, 75.0, 85.0, 95.0]:
            if not grid.min() <= target_map <= grid.max():
                continue
            target_point = float(np.interp(target_map, grid, point))
            target_boot = np.asarray([np.interp(target_map, grid, row) for row in boot])
            difference = target_boot - reference_boot
            contrast_rows.append({
                "window": spec["key"], "window_label": spec["label"],
                "map": target_map, "reference_map": reference_map,
                "difference": target_point - reference_point,
                "ci_low": float(np.percentile(difference, 2.5)),
                "ci_high": float(np.percentile(difference, 97.5)),
                "bootstrap_unit": bootstrap_unit,
            })
    selection.to_csv(output / "model_selection.csv", index=False)
    support.to_csv(output / "covariate_support.csv", index=False)
    inclusion_audit.to_csv(output / "patient_inclusion_audit.csv", index=False)
    curve.to_csv(output / "adjusted_curve.csv", index=False)
    deriv.to_csv(output / "curve_derivative.csv", index=False)
    scatter.to_csv(output / "timepoint_scatter_20s.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output / "clinical_map_contrasts.csv", index=False)
    pd.DataFrame(boot, columns=[f"map_{int(x)}" for x in grid]).to_csv(
        output / "bootstrap_curves.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(failures).to_csv(output / "bootstrap_failures.csv", index=False)

    st = gam.statistics_
    summary = {
        "window": spec["key"], "window_label": spec["label"],
        "outcome": spec["outcome"], "analysis_unit": "patient_20s_timepoint",
        "rows": int(len(data)),
        "patients": int(data.subject_id.nunique()),
        "map_p01": float(p01), "map_p99": float(p99),
        "chosen_lambda": chosen_lam, "GCV": float(st.get("GCV", np.nan)),
        "AIC": float(st.get("AIC", np.nan)), "effective_dof": float(st.get("edof", np.nan)),
        "explained_deviance": float(st.get("pseudo_r2", {}).get("explained_deviance", np.nan)),
        "tensor_map_ci_p_value": float(st.get("p_values", [np.nan])[0]),
        "features": features, "factor_encodings": encodings,
        "reference_values": ref, "bootstraps_requested": N_BOOT,
        "bootstraps_successful": int(len(boot)),
        "bootstrap_unit": bootstrap_unit,
        "method": f"CO2-rSO2 project-compatible 20-second pyGAM with {bootstrap_unit} bootstrap",
        "caveat": (
            "行级bootstrap忽略同一患者内时间点相关性，置信区间仅用于趋势展示，不能作为严格患者总体推断。"
            if bootstrap_unit == "row" else
            "pyGAM 不含患者随机效应或 AR(1)；本分析用于分窗曲线形状，正式纵向推断仍以 GAMM 为主。"
        ),
    }
    (output / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{**summary, "features": json.dumps(features), "factor_encodings": json.dumps(encodings), "reference_values": json.dumps(ref)}]).to_csv(output / "model_summary.csv", index=False)
    plot_single_window(output, spec, scatter, curve, int(data.subject_id.nunique()), bootstrap_unit)


def panel_plot(items, value_file, output, kind):
    output_root = output.parent.parent
    first_summary = pd.read_csv(output_root / items[0]["key"] / "model_summary.csv")
    bootstrap_unit = first_summary.get("bootstrap_unit", pd.Series(["patient"])).iloc[0]
    bootstrap_label = "行级" if bootstrap_unit == "row" else "患者级"
    ncols = 3 if len(items) <= 6 else 4
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.7 * nrows), squeeze=False)
    for ax, spec in zip(axes.ravel(), items):
        frame = pd.read_csv(output_root / spec["key"] / value_file)
        if kind == "scatter":
            n_points = len(frame)
            ax.scatter(
                frame["map"], frame["outcome"],
                s=7 if n_points < 5000 else 4,
                alpha=0.12 if n_points < 5000 else 0.05,
                color="#4E9A62", edgecolors="none", rasterized=True,
            )
            ax.set_ylabel(spec["outcome_label"])
            summary = pd.read_csv(output_root / spec["key"] / "model_summary.csv").iloc[0]
            ax.text(
                0.04, 0.94,
                f"N = {int(summary.patients)}，n = {n_points:,}",
                transform=ax.transAxes, va="top", fontsize=9,
            )
        else:
            ax.plot(frame["map"], frame["estimate"], color="#3366A6", lw=2)
            ax.fill_between(frame["map"], frame["ci_low"], frame["ci_high"], color="#3366A6", alpha=0.2)
            ax.set_ylabel(f"调整后 {spec['outcome_label']}")
            ax.text(0.04, 0.94, f"N = {int(pd.read_csv(output_root / spec['key'] / 'model_summary.csv').patients.iloc[0])}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_title(spec["label"], fontsize=11)
        ax.set_xlabel("MAP (mmHg)")
        ax.grid(alpha=0.18)
    for ax in axes.ravel()[len(items):]:
        ax.set_visible(False)
    fig.suptitle(
        "全部20秒配对点散点图"
        if kind == "scatter" else
        f"分时间窗 MAP–结局 pyGAM 曲线（{bootstrap_label} bootstrap 95% CI）",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def assemble(output_root: Path):
    setup_plotting()
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summaries, curves, derivatives, scatters, contrasts = [], [], [], [], []
    for spec in WINDOWS:
        folder = output_root / spec["key"]
        summaries.append(pd.read_csv(folder / "model_summary.csv"))
        curves.append(pd.read_csv(folder / "adjusted_curve.csv"))
        derivatives.append(pd.read_csv(folder / "curve_derivative.csv"))
        scatters.append(pd.read_csv(folder / "timepoint_scatter_20s.csv"))
        contrasts.append(pd.read_csv(folder / "clinical_map_contrasts.csv"))
    pd.concat(summaries, ignore_index=True).to_csv(output_root / "all_window_model_summary.csv", index=False)
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(output_root / "all_adjusted_curves.csv", index=False)
    pd.concat(derivatives, ignore_index=True).to_csv(output_root / "all_curve_derivatives.csv", index=False)
    pd.concat(scatters, ignore_index=True).to_csv(output_root / "all_timepoint_scatter_20s.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(output_root / "all_clinical_map_contrasts.csv", index=False)

    main_keys = [
        "ward_preop", "preinduction_5_1", "postinduction_all", "induction_to_intubation",
        "intubation_to_surgery", "surgery_0_60", "surgery_ge60",
    ]
    detail_keys = [
        "surgery_0_5", "surgery_5_10", "surgery_10_15", "surgery_15_20",
        "surgery_20_25", "surgery_25_30", "surgery_30_60", "surgery_ge60",
    ]
    main = [next(x for x in WINDOWS if x["key"] == key) for key in main_keys]
    detail = [next(x for x in WINDOWS if x["key"] == key) for key in detail_keys]
    panel_plot(main, "timepoint_scatter_20s.csv", figures / "01_main_stages_scatter.png", "scatter")
    panel_plot(main, "adjusted_curve.csv", figures / "02_main_stages_gam.png", "gam")
    panel_plot(detail, "timepoint_scatter_20s.csv", figures / "03_surgery_windows_scatter.png", "scatter")
    panel_plot(detail, "adjusted_curve.csv", figures / "04_surgery_windows_gam.png", "gam")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(detail)))
    for color, spec in zip(colors, detail):
        frame = all_curves.loc[all_curves.window.eq(spec["key"])].copy()
        common = frame[frame["map"].between(60, 120)]
        if common.empty:
            continue
        at75 = np.interp(75, frame["map"], frame["estimate"])
        ax.plot(common["map"], common["estimate"] - at75, lw=2, color=color, label=spec["label"].replace("手术后 ", ""))
    ax.axhline(0, color="0.45", lw=1, ls=":")
    ax.axvline(75, color="0.45", lw=1, ls=":")
    ax.set(xlabel="MAP (mmHg)", ylabel="相对 MAP 75 mmHg 的调整后 PSI 差值", title="手术开始后 MAP–PSI 曲线形状随时间变化")
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(figures / "05_surgery_curve_evolution.png", dpi=220)
    fig.savefig(figures / "05_surgery_curve_evolution.pdf")
    plt.close(fig)
    log("汇总图和总表已生成。")


def assemble_coarse(output_root: Path):
    """仅汇总用户预先指定的三个粗时间窗。"""
    setup_plotting()
    coarse_keys = ["surgery_0_30", "surgery_30_60", "surgery_ge60"]
    coarse = [next(x for x in WINDOWS if x["key"] == key) for key in coarse_keys]
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summaries = pd.concat(
        [pd.read_csv(output_root / spec["key"] / "model_summary.csv") for spec in coarse],
        ignore_index=True,
    )
    curves = pd.concat(
        [pd.read_csv(output_root / spec["key"] / "adjusted_curve.csv") for spec in coarse],
        ignore_index=True,
    )
    contrasts = pd.concat(
        [pd.read_csv(output_root / spec["key"] / "clinical_map_contrasts.csv") for spec in coarse],
        ignore_index=True,
    )
    summaries.to_csv(output_root / "coarse_window_model_summary.csv", index=False)
    curves.to_csv(output_root / "coarse_adjusted_curves.csv", index=False)
    contrasts.to_csv(output_root / "coarse_clinical_map_contrasts.csv", index=False)

    panel_plot(coarse, "timepoint_scatter_20s.csv", figures / "01_coarse_windows_scatter.png", "scatter")
    panel_plot(coarse, "adjusted_curve.csv", figures / "02_coarse_windows_gam.png", "gam")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#0072B2", "#E69F00", "#009E73"]
    for color, spec in zip(colors, coarse):
        frame = curves.loc[curves.window.eq(spec["key"])].copy()
        common = frame[frame["map"].between(60, 120)]
        if common.empty or not frame["map"].min() <= 75 <= frame["map"].max():
            continue
        at75 = np.interp(75, frame["map"], frame["estimate"])
        ax.plot(
            common["map"], common["estimate"] - at75,
            lw=2.3, color=color, label=spec["label"].replace("手术后 ", ""),
        )
    ax.axhline(0, color="0.45", lw=1, ls=":")
    ax.axvline(75, color="0.45", lw=1, ls=":")
    ax.set(
        xlabel="MAP (mmHg)",
        ylabel="相对 MAP 75 mmHg 的调整后 PSI 差值",
        title="手术开始后三个粗时间窗的 MAP–PSI 曲线比较",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(figures / "03_coarse_curve_evolution.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "03_coarse_curve_evolution.pdf", bbox_inches="tight")
    plt.close(fig)
    log("三个粗时间窗的汇总图和总表已生成。")


def assemble_fine(output_root: Path):
    """汇总六个5分钟窗，并保留30–60和≥60分钟作为后续参照。"""
    setup_plotting()
    fine_keys = [
        "surgery_0_5", "surgery_5_10", "surgery_10_15",
        "surgery_15_20", "surgery_20_25", "surgery_25_30",
        "surgery_30_60", "surgery_ge60",
    ]
    fine = [next(x for x in WINDOWS if x["key"] == key) for key in fine_keys]
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summaries = pd.concat(
        [pd.read_csv(output_root / spec["key"] / "model_summary.csv") for spec in fine],
        ignore_index=True,
    )
    curves = pd.concat(
        [pd.read_csv(output_root / spec["key"] / "adjusted_curve.csv") for spec in fine],
        ignore_index=True,
    )
    contrasts = pd.concat(
        [pd.read_csv(output_root / spec["key"] / "clinical_map_contrasts.csv") for spec in fine],
        ignore_index=True,
    )
    summaries.to_csv(output_root / "fine_window_model_summary.csv", index=False)
    curves.to_csv(output_root / "fine_adjusted_curves.csv", index=False)
    contrasts.to_csv(output_root / "fine_clinical_map_contrasts.csv", index=False)

    panel_plot(fine, "timepoint_scatter_20s.csv", figures / "04_fine_windows_scatter.png", "scatter")
    panel_plot(fine, "adjusted_curve.csv", figures / "05_fine_windows_gam.png", "gam")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(fine)))
    for color, spec in zip(colors, fine):
        frame = curves.loc[curves.window.eq(spec["key"])].copy()
        common = frame[frame["map"].between(60, 120)]
        if common.empty or not frame["map"].min() <= 75 <= frame["map"].max():
            continue
        at75 = np.interp(75, frame["map"], frame["estimate"])
        ax.plot(
            common["map"], common["estimate"] - at75,
            lw=2, color=color, label=spec["label"].replace("手术后 ", ""),
        )
    ax.axhline(0, color="0.45", lw=1, ls=":")
    ax.axvline(75, color="0.45", lw=1, ls=":")
    ax.set(
        xlabel="MAP (mmHg)",
        ylabel="相对 MAP 75 mmHg 的调整后 PSI 差值",
        title="手术开始后 MAP–PSI 曲线随时间变化（5分钟细分）",
    )
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(figures / "06_fine_curve_evolution.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "06_fine_curve_evolution.pdf", bbox_inches="tight")
    plt.close(fig)
    log("5分钟细分窗口的汇总图和总表已生成。")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeseries", type=Path,
        default=Path("/N/project/Analgesia_BDproject/PR/data/"
                     "analysis_timeseries_20s.csv"),
        help="20秒对齐的多模态时序 CSV。")
    parser.add_argument(
        "--covariates", type=Path,
        help="分钟级药物 + 静态协变量 CSV（需含 subject_id 与 minute 两列）。"
             "没有默认值：该文件的位置因项目而异，必须显式给出。")
    parser.add_argument(
        "--redcap", type=Path,
        default=Path("/N/project/Analgesia_BDproject/PR/data/"
                     "6.16.26.FIXED-TYPOS-BDPostInductionHemod_"
                     "DATA_LABELS_2026-06-16_1657.csv"),
        help="REDCap 导出（含术前监测起止时间）。")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/N/project/Analgesia_BDproject/PR/figures/window_gam"),
        help="结果输出目录。")
    parser.add_argument("--window", choices=[x["key"] for x in WINDOWS])
    parser.add_argument("--window-index", type=int)
    parser.add_argument("--bootstrap-unit", choices=["patient", "row"], default="patient")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--assemble-coarse", action="store_true")
    parser.add_argument("--assemble-fine", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start-index", type=int, default=int(os.getenv("WINDOW_GAM_START_INDEX", "0")))
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    if args.assemble:
        assemble(args.output)
        return
    if args.assemble_coarse:
        assemble_coarse(args.output)
        return
    if args.assemble_fine:
        assemble_fine(args.output)
        return
    if not args.covariates:
        raise SystemExit(
            "拟合时必须提供 --covariates（分钟级药物/静态协变量 CSV）。\n"
            "  --timeseries、--redcap、--output 已有默认值，可省略。\n"
            "  例：python example.py --covariates /path/to/covariates.csv "
            "--window ward_preop"
        )
    missing_files = [
        str(path) for path in (args.timeseries, args.covariates, args.redcap)
        if not Path(path).is_file()
    ]
    if missing_files:
        raise SystemExit(
            "以下输入文件不存在，请检查路径：\n  " + "\n  ".join(missing_files)
        )
    if not args.window and args.window_index is None and not args.all:
        raise SystemExit(
            "必须指定要拟合的窗口：--window <名称>、--window-index <序号> "
            "或 --all（全部15个窗口）。\n"
            f"  可选窗口：{', '.join(x['key'] for x in WINDOWS)}"
        )
    if args.all:
        for spec in WINDOWS[args.start_index:]:
            fit_window(args.timeseries, args.covariates, args.redcap, spec, args.output, args.bootstrap_unit)
        assemble(args.output)
        return
    if args.window:
        spec = next(x for x in WINDOWS if x["key"] == args.window)
    elif args.window_index is not None:
        spec = WINDOWS[args.window_index]
    else:
        raise SystemExit("必须提供 --window 或 --window-index")
    fit_window(args.timeseries, args.covariates, args.redcap, spec, args.output, args.bootstrap_unit)


if __name__ == "__main__":
    main()
