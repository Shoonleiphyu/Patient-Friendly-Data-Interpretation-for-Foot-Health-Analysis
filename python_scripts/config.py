# --- Pressure Sensors ---

# Raw pressure sensor names
RAW_P_SENSORS = ['pData', 'pData_2', 'pData_3', 'pData_4']

# Rename columns
FINAL_P_SENSORS = ['p_BigToe', 'p_1st-metatarsal', 'p_3rd-metatarsal','p_5th-metatarsal(side)']

# Rename mapping
P_RENAME_MAP = dict(zip(RAW_P_SENSORS, FINAL_P_SENSORS))

# --- Temperature Sensors ---

# Raw temperature sensor names
RAW_T_SENSORS = ['tData', 'tData_2', 'tData_3', 'tData_4']

# Rename columns
FINAL_T_SENSORS = ['t_BigToe', 't_1st-metatarsal', 't_3rd-metatarsal','t_5th-metatarsal(side)']

T_RENAME_MAP = dict(zip(RAW_T_SENSORS, FINAL_T_SENSORS))

# File paths
DB_PATH = '../data/sensor_data.db'
RAW_TABLE = 'capture_data'
CLEAN_PATH = '../data/capture_clean.csv'

# Data cleaning thresholds
DT_MIN = 0.0     # minimum interval between samples (sec)
DT_MAX = 1.0     # maximum interval between samples (sec)
Z_THRESH = 4     # outlier threshold (z-score)

