import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HYST_HIGH = 0.6
HYST_LOW  = 0.4



# Event detection
def detect_step_peaks(load, fs, min_step_interval=0.4):
    min_distance = int(min_step_interval * fs)

    if min_distance < 1:
        min_distance = 1

    peaks, _ = find_peaks(
        load,
        distance=min_distance,
        prominence=np.std(load) * 0.3
    )
    return peaks

# Event-level metrics
def compute_event_metrics(
    g: pd.DataFrame,
    sensors,
    temp_cols,
    time_col="t",
    min_contact=0.2,
    max_contact=1.5,
):
    events = []

    t = g[time_col].to_numpy().astype("datetime64[ns]")
    t_sec = g[time_col].astype("int64").to_numpy() / 1e9
    load = g[sensors].sum(axis=1).to_numpy(dtype=float)

    if len(t_sec) < 2:
        return pd.DataFrame(events)

    fs = 1 / np.median(np.diff(t_sec))

    peaks = detect_step_peaks(load, fs)

    if len(peaks) < 2:
        return pd.DataFrame(events)

    for event_id in range(len(peaks) - 1):
        h = peaks[event_id]
        c_offe = peaks[event_id + 1]

        contact_time = t_sec[c_offe] - t_sec[h]

        if contact_time < min_contact or contact_time > max_contact:
            continue

        seg = g.iloc[h:c_offe]
        dt = np.diff(t_sec[h:c_offe], prepend=t_sec[h])

        pti = float(np.sum(load[h:c_offe] * dt))

        events.append({
            "patient_id": g["patient_id"].iloc[0],
            "segment_id": g["segment_id"].iloc[0],
            "event_id": event_id,

            "start_time": t[h],
            "end_time": t[c_offe],

            "contact_time_sec": contact_time,

            "mean_load": float(np.nanmean(load[h:c_offe])),
            "peak_load": float(np.nanmax(load[h:c_offe])),
            "load_var": float(np.nanvar(load[h:c_offe])),

            "pti": pti,

            "temp_mean": seg[temp_cols].mean().mean(),
            "temp_max": seg[temp_cols].max().max(),
            "temp_min": seg[temp_cols].min().min(),
            "temp_std": seg[temp_cols].stack().std(),
        })

    return pd.DataFrame(events)



def compute_all_events(clean_df, sensors, temp_cols):
    out = []

    for (pid, seg_id), g in clean_df.groupby(["patient_id", "segment_id"]):
        duration = (g["t"].max() - g["t"].min()).total_seconds()
        if duration < 3.0:
            continue

        ev = compute_event_metrics(
            g,
            sensors=sensors,
            temp_cols=temp_cols
        )

        if not ev.empty:
            out.append(ev)

    if len(out) == 0:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)

