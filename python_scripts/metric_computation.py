import numpy as np
import pandas as pd


def compute_segment_metrics(events_df: pd.DataFrame):
    rows = []

    events_df = events_df.sort_values("start_time")

    for (pid, seg_id), g in events_df.groupby(["patient_id", "segment_id"]):
        steps = len(g)

        seg_duration = (g["end_time"].max() - g["start_time"].min()).total_seconds()

        cadence_spm = (
            60.0 * steps / seg_duration
            if seg_duration > 0 else np.nan
        )

        step_intervals = g["start_time"].diff().dt.total_seconds().dropna()

        total_stance_time = g["contact_time_sec"].sum()
        total_cycle_time = step_intervals.sum()
        stance_pct = (
            100.0 * total_stance_time / total_cycle_time
            if total_cycle_time > 0 else np.nan
        )

        rows.append({
            "patient_id": pid,
            "segment_id": seg_id,

            "steps": steps,
            "dur_sec": seg_duration,

            "cadence_spm": cadence_spm,

            "mean_contact": g["contact_time_sec"].mean(),
            "contact_var": g["contact_time_sec"].std(),

            "step_interval_mean": step_intervals.mean(),
            "step_interval_cv": (
                step_intervals.std() / step_intervals.mean()
                if step_intervals.mean() and step_intervals.mean() > 0
                else np.nan
            ),
            "stance_pct": stance_pct,

            "mean_pti": g["pti"].mean(),
            "mean_load": g["mean_load"].mean(),
            "peak_load": g["peak_load"].max(),

            "temp_mean": g["temp_mean"].mean(),
            "temp_max": g["temp_max"].max(),
            "temp_min": g["temp_min"].min(),
        })

    return pd.DataFrame(rows)


def compute_patient_metrics(segment_df: pd.DataFrame):
    rows = []

    for pid, g in segment_df.groupby("patient_id"):
        total_steps = g["steps"].sum()

        stance_pct = (
            (g["stance_pct"] * g["steps"]).sum() / total_steps
            if total_steps > 0 and "stance_pct" in g.columns
            else np.nan
        )

        rows.append({
            "patient_id": pid,

            "total_steps": g["steps"].sum(),
            "total_walk_time_sec": g["dur_sec"].sum(),

            "mean_cadence": np.average(g["cadence_spm"],weights=g["dur_sec"]),
            "cadence_var": g["cadence_spm"].std(),
            
            "mean_stance_pct": stance_pct,

            "step_interval_mean": g["step_interval_mean"].mean(),
            "step_interval_cv": g["step_interval_cv"].mean(),

            "mean_contact": np.average(g["mean_contact"],weights=g["steps"]),
            "contact_var": g["mean_contact"].std(),

            "mean_peak_load": g["peak_load"].mean(),
            "max_peak_load": g["peak_load"].max(),

            "mean_temp": g["temp_mean"].mean(),
            "max_temp": g["temp_max"].max(),
        })

    return pd.DataFrame(rows)
