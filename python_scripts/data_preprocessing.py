import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from .config import RAW_P_SENSORS, P_RENAME_MAP, FINAL_P_SENSORS, FINAL_T_SENSORS, RAW_T_SENSORS, T_RENAME_MAP

# Structural / temporal cleaning
def structural_clean(df: pd.DataFrame, dt_min: float = 0.0, split_gaps: float = 1.0):

    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df = df.dropna(subset=["cid", "t"])
    df = df.sort_values(['cid', 't']).copy()
    df = df.drop_duplicates(subset=['cid', 't'])


    df['dt'] = df.groupby('cid')['t'].diff().dt.total_seconds()

    df = df[(df['dt'].isna()) | (df['dt'] > dt_min)] # remove negative

    gap_dec = df['dt'] > split_gaps
    df['sub_id'] = gap_dec.groupby(df['cid']).cumsum()

    df['patient_id'] = df['cid']
    df['segment_id'] = df['sub_id']
    df = df.drop(columns=['sub_id'])
    return df

# Savitzky-Golay Filteration
def _clean_records(g: pd.DataFrame, sensors: list[str]):
    g = g.sort_values('t').copy()
    g[sensors] = g[sensors].interpolate(limit_direction='both')

    for c in sensors:
        arr = g[c].to_numpy()
        win = min(31, len(arr)//2*2 - 1) if len(arr) >= 31 else len(arr)
        if win >= 7:
            g[c + "_f"] = savgol_filter(arr, window_length=win, polyorder=2)
        else:
            g[c + "_f"] = arr
    return g

def signal_clean(df: pd.DataFrame, sensors: list[str]):
    return df.groupby('cid', group_keys=False).apply(lambda g: _clean_records(g, sensors))

# Outlier removing
def remove_outliers(df: pd.DataFrame, cols: list[str], z_thresh: float = 4.0):
    df = df.copy()
    for c in cols:
        z = (df[c] - df[c].mean()) / df[c].std(ddof=0)
        df.loc[z.abs() > z_thresh, c] = np.nan
    df[cols] = df[cols].interpolate(limit_direction='both')
    return df

# Normalisation
def normalize_signals(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    df = df.copy()
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


# Rename cols
def apply_rename(df: pd.DataFrame) -> pd.DataFrame:
    # Check pressure sensors
    missing_p = [c for c in RAW_P_SENSORS if c not in df.columns]
    if missing_p:
        raise ValueError(f"Expected raw pressure cols not found: {missing_p}")

    # Check temperature sensors
    missing_t = [c for c in RAW_T_SENSORS if c not in df.columns]
    if missing_t:
        raise ValueError(f"Expected raw temperature cols not found: {missing_t}")

    # Rename all sensors
    rename_map = {}
    rename_map.update(P_RENAME_MAP)
    rename_map.update(T_RENAME_MAP)

    return df.rename(columns=rename_map)



# Missing data handler
def ensure_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def add_total_load(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["total_load"] = df[sensors].sum(axis=1)
    return df


# detecting temperature
def detect_wear_by_temperature(
    df: pd.DataFrame,
    temp_cols=FINAL_T_SENSORS,
    wear_threshold=25.0
):
    df = df.copy()

    # Mean temperature across sensors
    df["temp_mean"] = df[temp_cols].mean(axis=1)

    # Binary wear flag
    df["is_worn"] = df["temp_mean"] > wear_threshold

    return df

