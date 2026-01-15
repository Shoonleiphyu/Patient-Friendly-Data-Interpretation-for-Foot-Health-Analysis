from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Defaults

anomaly_features_default: Tuple[str, ...] = (
    "dur_sec",
    "steps",
    "cadence_spm",
    "step_interval_cv",
    "mean_load",
    "peak_load",
    "pti",
    "stance_pct",
    "mean_contact",
    "contact_var",
    "load_var",
    "temp_mean",
)

anomaly_min_features: Tuple[str, ...] = (
    "dur_sec",
    "steps",
    "cadence_spm",
    "step_interval_cv",
    "mean_load",
    "pti",
    "temp_mean",
)

clinical_thresholds: Dict[str, Optional[float]] = {
    "cadence_spm_low": 40.0,
    "cadence_spm_high": 150.0,

    "step_interval_cv_high": 0.50,

    "temp_mean_low": 25.0,

    "mean_contact_low": 0.40,
    "mean_contact_high": 1.30,

    "stance_pct_low": 55.0,
    "stance_pct_high": 65.0,

    "peak_load_high": None,
    "pti_high": None,
}

Z_THRESHOLDS = 3.5
MIN_EVENTS_PER_PATIENT = 6

reasons_rules = {
    "cadence_spm": {
        "unit": "step/mins",
        "normal_range": (90, 120),
        "slow_td": 40,
        "template": (
            "Walking speed (cadence) was ({value:.0f} {unit}) and {direction}. "
            "Normal adult walking speed (cadence) typically ranges from 90 to 120 {unit} and 40 {unit} considered slow walking and over 150 {unit} is running. "
            "**Slow** speed indicates **fatigue, or short walking bouts.**"
        ),

    },

    "step_interval_cv": {
        "unit": "",
        "normal_range": (0.0, 0.5),
        "template": (
            "Step rhythm variability: {value:.2f}, "
            "considered {severity}.\n"
            "**High** variability shows **irregular** walking rhythm. "
            ),
    },

    "mean_contact": {
        "unit": "s",
        "normal_range": (0.4, 1.3),
        "template": (
            "Your average foot contact time was ({value:.2f} {unit}), which was {direction}. "
            "Typical contact time ranges from 0.4{unit} to 1.3{unit}. "
            "**High** pressure: reduce **push-off efficiency** and indicates **pain or fatigue**.\n\n"
            "**Low** pressure: **efficient** push-off and balance. "
            ),
    },

    "stance_pct": {
        "unit": "%",
        "normal_range": (55, 65),
        "template": (
            "Your stance phase occupied {value:.2f} {unit} of the walking cycle, "
            "which is considered {severity}. "
            "Normal stance phase typically ranges from 55 to 65 {unit} of the gait cycle. "
            ),
    },

    "temp_mean": {
        "unit": "°C",
        "normal_range": (25, 30),
        "template": (
            "Your average foot tmeperature was {value:.1f} {unit}, "
            "which is considered {severity}. "
            "Lower temperature indicate reduced skin contact or device may not have been worn properly."
            ),
    },

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
            "- **Low** pressure: **efficient walking** and **lighter steps** ."
        ),
    },

    "peak_load": {
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
class SegmentAnomalyConfig:
    features: Sequence[str] = anomaly_features_default
    z_threshold: float = Z_THRESHOLDS
    min_segments_per_patient: int = MIN_EVENTS_PER_PATIENT
    top_k_reasons: int = 3

def detect_segment_anomalies(
    segment_metrics: pd.DataFrame,
    config: Optional[SegmentAnomalyConfig] = None,
) -> pd.DataFrame:
    
    if config is None:
        config = SegmentAnomalyConfig()

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
    df["has_patient_baseline"] = seg_counts >= config.min_segments_per_patient
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

    df["anomaly_reasons"] = df.apply(_reasons_row, axis=1)
    
    # "pressure load" columns range
    for metric in ["pti", "mean_load", "peak_load"]:
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
            if metric in {"pti", "mean_load", "peak_load"}:
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
class ClinicalThresholdConfig:
    thresholds: Optional[Dict[str, Optional[float]]] = None

    def __post_init__(self):
        if self.thresholds is None:
            object.__setattr__(self, "thresholds", clinical_thresholds.copy())

def apply_clinical_flags(
    segment_metrics: pd.DataFrame,
    config: Optional[ClinicalThresholdConfig] = None,
) -> pd.DataFrame:
    
    if config is None:
        config = ClinicalThresholdConfig()

    df = _ensure_id_cols(segment_metrics).copy()
    thr = config.thresholds or {}

    reasons: List[List[str]] = [[] for _ in range(len(df))]

    def _add_reason(mask: pd.Series, msg: str):
        idxs = np.where(mask.fillna(False).to_numpy())[0]
        for i in idxs:
            reasons[i].append(msg)

    # cadence
    if "cadence_spm" in df.columns:
        lo = thr.get("cadence_spm_low")
        hi = thr.get("cadence_spm_high")
        if lo is not None:
            _add_reason(df["cadence_spm"] < lo, f"cadence_spm<{lo:g}")
        if hi is not None:
            _add_reason(df["cadence_spm"] > hi, f"cadence_spm>{hi:g}")

    # step interval variability
    if "step_interval_cv" in df.columns:
        cv_hi = thr.get("step_interval_cv_high")
        if cv_hi is not None:
            _add_reason(df["step_interval_cv"] > cv_hi, f"step_interval_cv>{cv_hi:g}")

    # minimum walking dose
    if "dur_sec" in df.columns:
        d_lo = thr.get("dur_sec_low")
        if d_lo is not None:
            _add_reason(df["dur_sec"] < d_lo, f"dur_sec<{d_lo:g}")
    if "steps" in df.columns:
        s_lo = thr.get("steps_low")
        if s_lo is not None:
            _add_reason(df["steps"] < s_lo, f"steps<{s_lo:g}")

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
    if "stance_pct" in df.columns:
        st_lo = thr.get("stance_pct_low")
        st_hi = thr.get("stance_pct_high")
        if st_lo is not None:
            _add_reason(df["stance_pct"] < st_lo, f"stance_pct<{st_lo:g}%")
        if st_hi is not None:
            _add_reason(df["stance_pct"] > st_hi, f"stance_pct>{st_hi:g}%")

    if "peak_load" in df.columns:
        pk_hi = thr.get("peak_load_high")
        if pk_hi is not None:
            _add_reason(df["peak_load"] > pk_hi, f"peak_load>{pk_hi:g}")
    if "pti" in df.columns:
        pti_hi = thr.get("pti_high")
        if pti_hi is not None:
            _add_reason(df["pti"] > pti_hi, f"pti>{pti_hi:g}")

    df["clinical_reasons"] = ["; ".join(r) for r in reasons]
    df["clinical_flag"] = df["clinical_reasons"].str.len() > 0
    
    reasons_exp = []

    for _, row in df.iterrows():
        if not row["clinical_reasons"]:
            reasons_exp.append("")
            continue

        explanations = []
        for rule in row["clinical_reasons"].split(";"):
            rule = rule.strip()
            explanations.append(_format_rules(rule,row))

        reasons_exp.append("||".join(explanations))

    df["reason_exp"] = reasons_exp
    
    return df



def run_anomaly_and_clinical(
    segment_metrics: pd.DataFrame,
    anomaly_config: Optional[SegmentAnomalyConfig] = None,
    clinical_config: Optional[ClinicalThresholdConfig] = None,
):
    df = detect_segment_anomalies(segment_metrics, config=anomaly_config)
    df = apply_clinical_flags(df, config=clinical_config)
    return df
