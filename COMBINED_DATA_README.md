# Combined Multi-Sensor Data

Combined EMG, fNIRS, and Franka robot data from the robot-assisted therapy experiment. Designed for easy sharing and analysis — load one file, get all three streams aligned.

## What's Inside

| File | Description |
|------|-------------|
| `combined_100hz.parquet` | All streams resampled to 100 Hz on a shared time grid |
| `data_packet/emg_full.parquet` | EMG at native resolution (~1926 Hz) |
| `data_packet/robot_full.parquet` | Robot telemetry at native resolution |
| `data_packet/fnirs_full.parquet` | fNIRS at native resolution (~10 Hz) |
| `data_packet/epoch_index.csv` | Metadata for each epoch (participant, condition, task) |

## Quick Start

```python
import polars as pl
import matplotlib.pyplot as plt

# Load the combined file
df = pl.read_parquet("combined_100hz.parquet")

# Pick one epoch
epoch = df.filter(
    (pl.col("run_id") == "sam_robot_1") & (pl.col("task_instance") == 1)
)

# Plot fNIRS, EMG, and robot on stacked subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# fNIRS HbO (pick one channel)
axes[0].plot(epoch["time_sec"], epoch["S1_D1_hbo"])
axes[0].set_ylabel("HbO (a.u.)")
axes[0].set_title("fNIRS")

# EMG (pick one channel)
axes[1].plot(epoch["time_sec"], epoch["Avanti Sensor 1 (82703) | EMG 1 (mV)"])
axes[1].set_ylabel("EMG (mV)")
axes[1].set_title("EMG")

# Robot end-effector position (parse the string column)
import json
ee = epoch["Franka_ee"].drop_nulls().to_list()
ee_parsed = [json.loads(x) for x in ee]
t_ee = epoch.filter(pl.col("Franka_ee").is_not_null())["time_sec"].to_list()
axes[2].plot(t_ee, [p[3] for p in ee_parsed], label="x")
axes[2].plot(t_ee, [p[7] for p in ee_parsed], label="y")
axes[2].plot(t_ee, [p[11] for p in ee_parsed], label="z")
axes[2].set_ylabel("Position (m)")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Franka End-Effector")
axes[2].legend()

plt.tight_layout()
plt.savefig("sample_epoch.png", dpi=150)
plt.show()
```

## Time Grid

Each epoch spans **−5 to +15 seconds** relative to task onset, sampled at **100 Hz** (2001 time points).

| Stream | Original Rate | Grid Range | Notes |
|--------|--------------|------------|-------|
| fNIRS | ~10 Hz | −5 to 15s | Interpolated up to 100 Hz |
| EMG | ~1926 Hz | -5 to 15s | Nearest-neighbor downsampled |
| Robot | Variable | -5 to 15s | Nearest-neighbor matched |

All data streams include a -5 baseline portion.

## Resampling Details

All streams are aligned to the same 100 Hz time grid using **nearest-neighbor matching** (`join_asof`).

- **fNIRS**: Originally ~10 Hz. Resampled to 100 Hz via nearest-neighbor. The small interpolation error is acceptable for the slow-changing hemodynamic signal. This also resolves minor floating-point `sfreq` differences between SNiRF files that prevented direct concatenation.
- **EMG**: Originally ~1926 Hz. Downsampled to 100 Hz (Nyquist = 50 Hz), which preserves the meaningful frequency content for gesture/muscle analysis (most motor unit power is below 50 Hz).
- **Robot**: Variable rate CSV. Matched to 100 Hz via nearest-neighbor. The `Franka_ee`, `Franka_q`, `Franka_dq`, and `Franka_tau_J` columns are string-encoded arrays that travel with their timestamp row.

## Null Values

| Null Type | Columns Affected | Cause |
|-----------|-----------------|-------|
| **Baseline** | EMG, Robot (−5 to 0s) | No data before task onset — expected |
| **Sensor dropout** | EMG (~0.6% of task window) | Trigno sensor disconnection during recording |
| **Missing stream** | Robot columns (norobot sessions) | Robot was recording but not actively assisting |

The baseline nulls are **25% of rows** (500 / 2001 time points per epoch). This is correct — EMG and robot data only exist from 0 to 15s.

## Schema

### Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | String | Session identifier (e.g. `sam_robot_1`) |
| `task_instance` | Int64 | Epoch number (1–30) |
| `time_sec` | Float64 | Time relative to task onset (−5.0 to 15.0) |
| `participant` | String | Participant name |
| `is_robot` | Boolean | True if robot was actively assisting |
| `task` | String | Task ID (e.g. `task_6`) |

### fNIRS Channels (52 columns)

HbO and HbR for 26 source-detector pairs. Column naming: `S{source}_D{detector}_hbo` / `_hbr`.

Example: `S1_D1_hbo`, `S3_D10_hbr`

### EMG Channels (22 columns)

Three Trigno sensors, each with EMG, accelerometer, and gyroscope:

| Sensor | Channels |
|--------|----------|
| Avanti Sensor 1 (82703) | EMG 1 (mV), ACC X/Y/Z (G), GYRO X/Y/Z (deg/s) |
| Avanti Sensor 2 (82529) | EMG 1 (mV), ACC X/Y/Z (G), GYRO X/Y/Z (deg/s) |
| Duo Sensor 3 (78042) | EMG 1/2 (mV), ACC X/Y/Z (G), GYRO X/Y/Z (deg/s) |

### Robot Columns (6 columns)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Float64 | Absolute timestamp from robot recording |
| `expression_happy_index` | Int64 | Facial expression classifier output |
| `Franka_ee` | String | End-effector pose as JSON array `[x, y, z, ...]` |
| `Franka_q` | String | Joint positions as JSON array |
| `Franka_dq` | String | Joint velocities as JSON array |
| `Franka_tau_J` | String | Joint torques as JSON array |

Parse Franka columns with `json.loads()`:
```python
import json
ee_values = json.loads(epoch_row["Franka_ee"])  # → list of floats
```

## Epoch Structure

- **30 epochs** per session
- Each epoch covers the **15-second task window** (0 to 15s)
- `task_instance` indexes from 1 to 30
- fNIRS includes the 5-second pre-task baseline (−5 to 0s)

## Sessions

The dataset contains **10 complete experiment sessions** across 5 participants, each with a robot-assisted and non-robot-assisted condition:

| Session | Participant | Condition |
|---------|------------|-----------|
| sam_robot_1 | Sam | Robot |
| sam_robot_2 | Sam | Robot |
| sam_norobot_1 | Sam | No Robot |
| sam_norobot_2 | Sam | No Robot |
| caroline_norobot_1 | Caroline | No Robot |
| clarence_robot_1 | Clarence | Robot |
| clarence_norobot_1 | Clarence | No Robot |
| farhir_robot_1 | Farhir | Robot |
| jiang_norobot_1 | Jiang | No Robot |
| ronald_robot_4 | Ronald | Robot |

## Data Packet (Full Resolution)

Use the data packet when you need the original sampling rates:

```python
emg = pl.read_parquet("data_packet/emg_full.parquet")
robot = pl.read_parquet("data_packet/robot_full.parquet")
fnirs = pl.read_parquet("data_packet/fnirs_full.parquet")
index = pl.read_csv("data_packet/epoch_index.csv")

# Filter to a specific session
session_emg = emg.filter(pl.col("run_id") == "sam_robot_1")
```

The `epoch_index.csv` contains one row per epoch with metadata (run_id, task_instance, participant, is_robot, task_id). Use it to look up which epochs belong to which session/condition.
