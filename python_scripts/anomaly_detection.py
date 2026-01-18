from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Defaults

anomaly_features_default: Tuple[str, ...] = (
    "total_walk_time_sec",
    "total_steps",
    "mean_cadence",
    "step_interval_cv",
    "mean_load",
    "mean_peak_load",
    "pti",
    "mean_stance_pct",
    "mean_contact",
    "contact_var",
    "load_var",
    "temp_mean",
)

anomaly_min_features: Tuple[str, ...] = (
    "total_walk_time_sec",
    "total_steps",
    "mean_cadence",
    "step_interval_cv",
    "mean_load",
    "pti",
    "temp_mean",
)

clinical_thresholds: Dict[str, Optional[float]] = {
    "mean_cadence_low": 40.0,
    "mean_cadence_high": 150.0,

    "step_interval_cv_high": 0.50,

    "temp_mean_low": 25.0,

    "mean_contact_low": 0.40,
    "mean_contact_high": 1.30,

    "mean_stance_pct_low": 55.0,
    "mean_stance_pct_high": 65.0,

    "mean_peak_load_high": None,
    "mean_pti_high": None,
}

z_threds = 3.5
min_events = 6

reasons_rules = {
    "contact_time_sec": {
        "unit": "s",
        "normal_range": (0.4, 1.3),
        "template": (
            "Duration of foot on the ground: {value:.2f} {unit}, which was {direction}.\n"
            "Typical contact time range from 0.4 {unit} to 1.3 {unit}.\n"
            "- **Longer** duration: **slower or hesitant** walking. "
        ),
    },

    "pti": {
        "unit": "",
        "normal_range": "patient_baseline",
        "template": (
            "PTI (Pressure + duration of contact): {value:.2f} was {direction} "
            "than usual.\n"
            "- Your typical PTI ranged from {lo:.2f} to {hi:.2f}.\n"
            "- **High** values means slower walking due to **pain or fatigue** and shows **gait instability**.\n"
            "- **Lower** values leads to **healthy gait**."
        ),
    },

    "mean_load": {
        "unit": "",
        "normal_range": "patient_baseline",
        "template": (
            "The average pressure your foot experienced during a step: {value:.2f} was {direction} "
            "than typical.\n"
            "- Your typical average pressure ranged from {lo:.2f} to {hi:.2f}.\n"
            "- **High** pressure: comes from **long stance/contact time**.\n"
            "- **Low** pressure: **efficient walking** and **lighter total_steps** ."
        ),
    },

    "mean_peak_load": {
        "unit": "",
        "normal_range": "patient_baseline",
        "template": (
            "Your push-off pressure: {value:.2f} was {direction} "
            "compared to usual.\n"
            "- Your typical push-off pressure ranged from {lo:.2f} to {hi:.2f}.\n"
            "- **High** value:  **stronger** push-off\n"
            "- **Low** value:  **gentler** push-off"
        ),
    },
    
}

# helper functions
def _mad(x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad  

def robust_z(x: pd.Series):
    arr = x.to_numpy(dtype=float)
    med = np.nanmedian(arr)
    scale = _mad(arr)
    if not np.isfinite(scale) or scale <= 1e-12:
        return pd.Series(np.full_like(arr, np.nan, dtype=float), index=x.index)
    return (x - med) / scale

def _coerce_datetime(df: pd.DataFrame, time_col: str = "t") -> pd.DataFrame:
    df = df.copy()
    if time_col in df.columns and not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df

def _ensure_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "patient_id" not in df.columns:
        if "cid" in df.columns:
            df["patient_id"] = df["cid"]
        else:
            raise ValueError("Expected 'patient_id' or 'cid' column.")
    if "segment_id" not in df.columns:
        if "cid" in df.columns:
            df["segment_id"] = df["cid"]
        else:
            df["segment_id"] = df["patient_id"]
    return df

def _pick_feature_cols(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    return [c for c in features if c in df.columns]

def patient_baseline_range(
    series: pd.Series,
    low_q: float = 0.1,
    high_q: float = 0.9,
):
    series = series.dropna()
    if series.empty:
        return None, None
    return series.quantile(low_q), series.quantile(high_q)


def _parse_rule(rule: str):
    match = re.match (r"([a-zA-Z_]+)\s*([<>])\s*([\d\.]+)", rule)
    if not match:
        return None
    metric, op, threshold = match.groups()
    return metric, op, float(threshold)


def _classify_severity(value, normal_range):
    lo, hi = normal_range
    if lo is not None and value < lo*0.7:
        return "LOW"
    if lo is not None and value < lo:
        return "MODERATE"
    if hi is not None and value > hi*1.2:
        return "HIGH"
    if hi is not None and value > hi:
        return "MODERATE"
    return "MILD"

def _direction_values(value, normal_range):
    lo, hi = normal_range
    if lo is not None and value < lo:
        return "LOWER than typical"
    if hi is not None and value > hi:
        return "HIGHER than typical"
    return "within NORMAL range"

def _format_rules (rule: str, row: pd.Series):
    parsed = _parse_rule(rule)
    if parsed is None:
        return rule
    
    metric, _ , _ = parsed
    rule_def = reasons_rules.get(metric)

    if rule_def is None or metric not in row:
        return rule
    
    value = row[metric]

    if rule_def["normal_range"] == "patient_baseline":
        direction = row.get(f"{metric}_direction", "within normal range")
        lo = row.get(f"{metric}_lo")
        hi = row.get(f"{metric}_hi")
        severity = ""
    else:
        normal_range = rule_def["normal_range"]
        direction = _direction_values(value, normal_range)
        severity = _classify_severity(value, normal_range)
        lo, hi = normal_range

    return rule_def["template"].format(
        value = value,
        unit = rule_def["unit"],
        severity = severity,
        direction = direction,
        lo=lo,
        hi=hi,
    )

# Anomaly detection in events
@dataclass(frozen=True)
class AnomalyConf:
    features: Sequence[str] = anomaly_features_default
    z_threshold: float = z_threds
    eve_per_patient: int = min_events
    top_k_reasons: int = 3

def detect_eve_anom(
    segment_metrics: pd.DataFrame,
    config: Optional[AnomalyConf] = None,
) -> pd.DataFrame:
    
    if config is None:
        config = AnomalyConf()

    df = _ensure_id_cols(segment_metrics).copy()

    feat_cols = _pick_feature_cols(df, config.features)
    if len(feat_cols) == 0:
        feat_cols = _pick_feature_cols(df, anomaly_min_features)
    if len(feat_cols) == 0:
        raise ValueError("No usable feature columns found for anomaly detection.")

    z_cols = []
    for c in feat_cols:
        zc = f"z_{c}"
        z_cols.append(zc)
        df[zc] = df.groupby("patient_id")[c].transform(robust_z)

    abs_z = df[z_cols].abs()
    df["anomaly_score"] = abs_z.max(axis=1, skipna=True)

    seg_counts = df.groupby("patient_id")["segment_id"].transform("nunique")
    df["has_patient_baseline"] = seg_counts >= config.eve_per_patient
    df["is_segment_anomaly"] = (df["has_patient_baseline"]) & (df["anomaly_score"] >= config.z_threshold)

    def _reasons_row(row) -> str:
        pairs = []
        for c in feat_cols:
            zc = f"z_{c}"
            zval = row.get(zc, np.nan)
            if np.isfinite(zval):
                pairs.append((abs(zval), c, float(zval)))
        if not pairs:
            return ""
        pairs.sort(reverse=True, key=lambda t: t[0])
        top = pairs[: config.top_k_reasons]
        return "; ".join([f"{c} z={zval:+.2f}" for _, c, zval in top])

    df["anomaly_reasons"] = df.apply(
        lambda r: _reasons_row(r) if r["is_segment_anomaly"] else "",
        axis=1
    )
    
    # "pressure load" columns range
    for metric in ["pti", "mean_load", "mean_peak_load"]:
        if metric in df.columns:
            df[f"{metric}_lo"] = (
                df.groupby("patient_id")[metric]
                .transform(lambda s: patient_baseline_range(s)[0])
            )
            df[f"{metric}_hi"] = (
                df.groupby("patient_id")[metric]
                .transform(lambda s: patient_baseline_range(s)[1])
            )

            df[f"{metric}_direction"] = np.where(
                df[metric] < df[f"{metric}_lo"],
                "LOWER",
                np.where(
                    df[metric] > df[f"{metric}_hi"],
                    "HIGHER",
                    "within NORMAL range",
                ),
            )

    anom_reasons_exp = []

    for _, row in df.iterrows():
        if not row["anomaly_reasons"]:
            anom_reasons_exp.append("")
            continue

        anom_explanations = []
        for rule in row["anomaly_reasons"].split(";"):
            metric = rule.split()[0] 
            rule = f"{metric} > 0"
            if metric in {"pti", "mean_load", "mean_peak_load"}:
                direction = row.get(f"{metric}_direction", "within NORMAL range")
            else:
                rule_def= reasons_rules.get(metric)
                if rule_def is None:
                    continue
                direction = _direction_values(row[metric], rule_def["normal_range"])

            if direction == "within NORMAL range":
                continue
            anom_explanations.append(_format_rules(rule, row))

        anom_reasons_exp.append("||".join(anom_explanations))

    df["anom_reason_exp"] = anom_reasons_exp
    
    return df


# Threshold-based clinical flags
@dataclass(frozen=True)
class ThreadsConf:
    thresholds: Optional[Dict[str, Optional[float]]] = None

    def __post_init__(self):
        if self.thresholds is None:
            object.__setattr__(self, "thresholds", clinical_thresholds.copy())

def apply_clinical_flags(
    segment_metrics: pd.DataFrame,
    config: Optional[ThreadsConf] = None,
):
    
    if config is None:
        config = ThreadsConf()

    df = _ensure_id_cols(segment_metrics).copy()
    thr = config.thresholds or {}

    reasons: List[List[str]] = [[] for _ in range(len(df))]

    def _add_reason(mask: pd.Series, msg: str):
        idxs = np.where(mask.fillna(False).to_numpy())[0]
        for i in idxs:
            reasons[i].append(msg)

    # cadence
    if "mean_cadence" in df.columns:
        lo = thr.get("mean_cadence_low")
        hi = thr.get("mean_cadence_high")
        if lo is not None:
            _add_reason(df["mean_cadence"] < lo, f"mean_cadence<{lo:g}")
        if hi is not None:
            _add_reason(df["mean_cadence"] > hi, f"mean_cadence>{hi:g}")

    # step interval variability
    if "step_interval_cv" in df.columns:
        cv_hi = thr.get("step_interval_cv_high")
        if cv_hi is not None:
            _add_reason(df["step_interval_cv"] > cv_hi, f"step_interval_cv>{cv_hi:g}")

    # minimum walking dose
    if "total_walk_time_sec" in df.columns:
        d_lo = thr.get("total_walk_time_sec_low")
        if d_lo is not None:
            _add_reason(df["total_walk_time_sec"] < d_lo, f"total_walk_time_sec<{d_lo:g}")
    if "total_steps" in df.columns:
        s_lo = thr.get("total_steps_low")
        if s_lo is not None:
            _add_reason(df["total_steps"] < s_lo, f"total_steps<{s_lo:g}")

    # wear temperature
    if "temp_mean" in df.columns:
        t_lo = thr.get("temp_mean_low")
        if t_lo is not None:
            _add_reason(df["temp_mean"] < t_lo, f"temp_mean<{t_lo:g} (not worn?)")

    # contact time
    if "mean_contact" in df.columns:
        ct_lo = thr.get("mean_contact_low")
        ct_hi = thr.get("mean_contact_high")
        if ct_lo is not None:
            _add_reason(df["mean_contact"] < ct_lo, f"mean_contact<{ct_lo:g}s")
        if ct_hi is not None:
            _add_reason(df["mean_contact"] > ct_hi, f"mean_contact>{ct_hi:g}s")

    # stance %
    if "mean_stance_pct" in df.columns:
        st_lo = thr.get("mean_stance_pct_low")
        st_hi = thr.get("mean_stance_pct_high")
        if st_lo is not None:
            _add_reason(df["mean_stance_pct"] < st_lo, f"mean_stance_pct<{st_lo:g}%")
        if st_hi is not None:
            _add_reason(df["mean_stance_pct"] > st_hi, f"mean_stance_pct>{st_hi:g}%")

    if "mean_peak_load" in df.columns:
        pk_hi = thr.get("mean_peak_load_high")
        if pk_hi is not None:
            _add_reason(df["mean_peak_load"] > pk_hi, f"mean_peak_load>{pk_hi:g}")
    if "mean_pti" in df.columns:
        mean_pti_hi = thr.get("mean_pti_high")
        if mean_pti_hi is not None:
            _add_reason(df["mean_pti"] > mean_pti_hi, f"mean_pti>{mean_pti_hi:g}")

    df["clinical_reasons"] = ["; ".join(r) for r in reasons]
    df["clinical_flag"] = df["clinical_reasons"].str.len() > 0
    
    return df
