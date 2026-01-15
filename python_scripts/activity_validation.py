import pandas as pd


def validate_walking_records(
    df: pd.DataFrame,
    min_steps: int = 3,
    min_duration: float = 5.0,
    min_temp: float = 25.0,
):

    required_cols = [
        "steps",
        "dur_sec",
        "cadence_spm",
        "step_interval_cv",
        "temp_mean"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df["is_walking_records"] = (
        (df["steps"] >= min_steps) &
        (df["dur_sec"] >= min_duration) &
        (df["temp_mean"] >= min_temp)
    )

    return df

