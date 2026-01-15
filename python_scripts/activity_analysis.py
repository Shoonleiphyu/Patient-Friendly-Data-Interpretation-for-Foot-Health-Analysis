import numpy as np
import pandas as pd

HYST_HIGH = 0.6
HYST_LOW  = 0.4



# Event detection
def hysteresis_mask(sig, hi=HYST_HIGH, lo=HYST_LOW):
    smin, smax = np.nanmin(sig), np.nanmax(sig)
    rng = max(smax - smin, 1e-6)

    hi_val = smin + hi * rng
    lo_val = smin + lo * rng

    mask = np.zeros(len(sig), dtype=bool)
    on = False
    for i, v in enumerate(sig):
        if not on and v >= hi_val:
            on = True
        elif on and v <= lo_val:
            on = False
        mask[i] = on
    return mask


def stance_events(mask):
    diff = np.diff(mask.astype(int))
    c_on = np.where(diff == 1)[0] + 1
    c_off = np.where(diff == -1)[0] + 1
    return c_on, c_off


def detect_events(df, sensors):
    load = df[sensors].sum(axis=1).to_numpy(dtype=float)
    mask = hysteresis_mask(load)
    c_on, c_off = stance_events(mask)
    return c_on, c_off



# Event-level metrics

def compute_event_metrics(
    g: pd.DataFrame,
    c_on,
    c_off,
    sensors,
    temp_cols,
    time_col="t"
):
    events = []

    t = g[time_col].to_numpy().astype("datetime64[ns]")
    load = g[sensors].sum(axis=1).to_numpy(dtype=float)

    j = 0
    for event_id, h in enumerate(c_on):
        while j < len(c_off) and c_off[j] <= h:
            j += 1
        if j >= len(c_off):
            break

        c_offe = c_off[j]
        j += 1

        if c_offe <= h:
            continue

        seg = g.iloc[h:c_offe]

        # contact time
        MIN_CONTACT_SEC = 0.2
        contact_time = (t[c_offe] - t[h]).astype("timedelta64[ns]").astype(float) / 1e9
        if contact_time <= MIN_CONTACT_SEC:
            continue

        # PTI
        t_sec = seg[time_col].astype("int64").to_numpy() / 1e9
        dt = np.diff(t_sec, prepend=t_sec[0])
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
        if (g["t"].max() - g["t"].min()).total_seconds() < 3.0:
            continue

        c_on, c_off = detect_events(g, sensors)

        if len(c_on) == 0 or len(c_off) == 0:
            continue

        ev = compute_event_metrics(
            g,
            c_on,
            c_off,
            sensors=sensors,
            temp_cols=temp_cols
        )

        if not ev.empty:
            out.append(ev)

    return pd.concat(out, ignore_index=True)
