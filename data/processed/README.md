# Processed data

This folder stores derived artifacts generated from raw data.

- `windows/`: persisted window tables in JSON Lines format (`*.jsonl.gz`)
- `labels/`: scalar window-level labels and split assignments (`*.csv`)
- `*_manifest.json`: dataset settings, participant split, and artifact paths for one processed export

Current exports record whether Galaxy Watch PPG was inverted at load time, the canonical PPG source column, and the label-generation rule. The default target rule is beat-interval instantaneous HR from IBI (`60000 / rr_interval_ms`) with a median target over each 10-second window and a minimum of 2 valid beat intervals per window. ECG references use the same target definition after per-window R-peak detection and RR-interval construction.
