import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- DASHBOARD TITLE ---
st.set_page_config(
    page_title="My Walking Health Data",
    layout="wide"
)


# --- LOAD DATA ---
@st.cache_data
def load_patient_data():
    return pd.read_csv("data/metrics/patient_metrics.csv")

@st.cache_data
def load_anomaly():
    return pd.read_csv("data/anomaly/event_anomalies.csv")

@st.cache_data
def load_td():
    return pd.read_csv("data/anomaly/metric_flags.csv")

df = load_patient_data()
ano_df = load_anomaly()
td_df = load_td()

# --- APP TITLE ---
st.title("Walking Session Summary Dashboard")

st.write(
    "This dashboard gives you a simple overview of your walking patterns based on your recent activity."
)


# --- patients SELECTION ---
patient_ids = df["patient_id"].unique()

# dropdown selectbox
selected_patients = st.selectbox(
    "Select patient",
    patient_ids
)

patients = df[df["patient_id"] == selected_patients].iloc[0]

ano = ano_df[ano_df["patient_id"] == selected_patients]
td = td_df[td_df["patient_id"] == selected_patients]

# --- DEVICE STATUS ---
if patients["mean_temp"] > 25:
    st.markdown("🟢 Device was worn properly during this patients.")
else:
    st.markdown("🟡 Device may not have been worn properly.")

st.write(
f"(*Average temperature recorded: **{round(patients['mean_temp'], 1)} °C***)"
)

st.divider()


# --- METRICS DURING SESSION ---
st.subheader("Metrics Summary")
st.write(
    "This section summarises how your walking looked overall during this session."
)

if patients["total_steps"] < 30:
    st.info(
        "ℹ️ This summary is based on a short walking session. "
        "Longer walks provide more reliable insights."
    )


# Helper functions
def status_and_note(value, lo, hi, unit="", lower_note="", higher_note=""):
    if value < lo:
        return "🔴 Low", lower_note
    elif value > hi:
        return "🟠 High", higher_note
    else:
        return "🟢 Normal", "\nWithin the expected range for healthy walking."

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "**Steps Taken**",
            int(patients["total_steps"])
        )

    with col2:
        st.metric(
            "**Walking Time**",
            f"{patients['total_walk_time_sec']:.1f} sec"
        )

col1, col2 = st.columns(2)
   
with col1:
    cadence_status, cadence_note = status_and_note(
        patients["mean_cadence"],
        lo=90,
        hi=120,
        unit="steps/min",
        lower_note="\n⚠️ Slow speed indicates fatigue, caution, or short walking bouts.",
        higher_note=""
    )
    if patients["mean_cadence"] < 10:
        st.markdown("**Walking Speed (Cadence)**")
        st.write(
            "*Walking activity was too brief or intermittent to estimate a reliable walking speed.*"
        )
    else:
        st.metric(
            "**Walking Speed (Cadence)**",
            f"{patients['mean_cadence']:.0f} steps/min",
        )
        st.markdown(f"{cadence_status}")
        st.caption(
            "Normal adult cadence: **90–120 steps/min**.\n\n"
            "Slow Walking: Below **40 steps/min**.\n"
            f"{cadence_note}"
        )

with col2:
    contact_status, contact_note = status_and_note(
        patients["mean_contact"],
        lo=0.4,
        hi=1.3,
        unit="sec",
        lower_note="Shorter duration shows faster walking.",
        higher_note="\n⚠️ Longer contact time possibly means you walked slower or cautious."
    )

    st.metric(
        "**Foot Contact Duration**",
        f"{patients['mean_contact']:.2f} sec",
    )
    st.markdown(f"{contact_status}")
    st.caption(
        "Normal contact time: **0.4–1.3 sec**.\n"
        f"{contact_note}"
    )

st.divider()

# --- STEP RHYTHM ANOMALY ---

st.subheader("Walking Rhythm")
st.caption("Walking rhythm reflects how evenly timed your steps are. Variability more than 0.6 shows irregular walking rhythm.")

cv = patients["step_interval_cv"]

st.header(f"{cv:.2f}")

if pd.isna(cv):
    st.info("Not enough walking data to assess step rhythm.")
elif cv < 0.6:
    st.markdown(
        "🟢 Stable, with consistent timing between steps."
    )
else:
    st.write(
        f"🟠 Outside of the typical variability of 0.6\n"
        "\nThis means the time between steps was less consistent, which can possibly occur because of "
        "fatigue, changes in balance, or uneven pacing."
    )

st.divider()

# --- ANOMALY DETECTION ---
st.subheader("Irregularities in Walking Pattern")
st.caption("These irregularities happened during some steps you made which deviated from your usual walking pattern.")

n_ano = int(ano["is_segment_anomaly"].sum())

with st.container(border=True):
    if n_ano == 0:
        st.success("No concerning walking patterns detected")
    else:
        st.warning("Some walking events(steps) showed unsual patterns.")

        st.subheader("Details of irregular walking pattern")

        ano_txt = (
        ano.loc[ano["is_segment_anomaly"], "anom_reason_exp"].dropna().astype(str).str.split("||", regex = False)
        .explode().str.strip().unique()
        )

        if len(ano_txt) > 0:
            for reason in ano_txt:
                with st.container(border=True):
                    st.write(f"{reason}")

st.divider()



st.write(
    "This information is for personal awareness only and is not a medical diagnosis."
)
