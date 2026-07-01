# WildPPG Wrist Raw Data Placement

The raw WildPPG dataset is not bundled in this repository.

Week 6 uses a configurable manifest so the loader does not depend on one hard-coded archive layout. Place the wrist subset files under:

```text
data/raw/WildPPG-wrist/
```

Add:

```text
data/raw/WildPPG-wrist/wildppg_wrist_manifest.csv
```

Required columns:

```text
participant_id,ppg_path,ppg_col,reference_path
```

Common optional columns:

```text
session_id,activity,
ppg_time_col,ppg_sampling_hz,
acc_path,acc_time_col,acc_x_col,acc_y_col,acc_z_col,acc_sampling_hz,
reference_time_col,ecg_col,rr_col,reference_sampling_hz,
time_unit
```

Rules:

- Paths are relative to `data/raw/WildPPG-wrist/` unless absolute.
- `time_unit` supports `ms`, `s`, `us`, and `ns`; default is `ms`.
- If a stream has no timestamp column, provide its sampling rate.
- Reference rows can provide either ECG samples (`ecg_col`) or RR intervals (`rr_col`).

Example:

```csv
participant_id,session_id,activity,ppg_path,ppg_col,ppg_sampling_hz,acc_path,acc_x_col,acc_y_col,acc_z_col,acc_sampling_hz,reference_path,ecg_col,reference_sampling_hz,time_unit
W01,outdoor_walk,walking,W01/wrist_ppg.csv,ppg,64,W01/wrist_acc.csv,x,y,z,32,W01/ecg.csv,ecg,700,ms
```
