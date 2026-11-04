"""Native WildPPG participant MAT support and bounded-memory window export.

Only the wrist green PPG/accelerometer and synchronized sternum lead-I ECG
are selected. The cleaned 8-second HR supplied by the quick-start distribution
is never used. Continuous signals are stored once; overlapping windows are
represented by a small CSV index rather than repeated waveform JSON arrays.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .labels import LabelGenerationConfig
from src.utils.submission_protocol import sha256, write_json


KNOWN_WRIST_INTERRUPTION_SAMPLES = {
    # Official raw MAT notes and bundled heuristic baseline; Python [start,end).
    "k2s": [(3786939, 4194811)],
    "l38": [(4337572, 4545544)],
}


def wfdb_detector():
    """Resolve the pinned reference detector without changing source-model code."""
    import sys
    bundled = Path(__file__).resolve().parents[2]/".deps/wfdb"
    if bundled.is_dir() and str(bundled) not in sys.path:
        sys.path.insert(0, str(bundled))
    try:
        import wfdb
        from wfdb import processing
    except ImportError as error:
        raise ImportError("Native WildPPG reference extraction requires wfdb==4.3.0; install pinned external dependencies") from error
    if wfdb.__version__ != "4.3.0":
        raise ValueError("Native reference detector is frozen to WFDB4.3.0; use a new protocol for another version")
    return processing.xqrs_detect


def native_paths(root: str | Path) -> dict[str, Path]:
    paths = sorted(Path(root).glob("WildPPG_Part_*.mat"))
    return {p.stem.removeprefix("WildPPG_Part_"): p for p in paths}


def _native_channel(body: dict, name: str) -> tuple[np.ndarray, float, str]:
    channel = body.get(name)
    if not isinstance(channel, dict) or not {"v", "fs"} <= set(channel):
        raise ValueError(f"Missing native WildPPG channel: {name}")
    values = np.asarray(channel["v"])
    if values.ndim != 1:
        raise ValueError(f"Native channel must be a vector: {name}")
    fs = float(channel["fs"])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid native sampling rate: {name}")
    return values, fs, str(channel.get("descr", ""))


def read_native_signals(path: str | Path) -> dict:
    """Select required channels while preserving native numeric values and fs."""
    path = Path(path)
    data = loadmat(path, variable_names=["id", "notes", "wrist", "sternum"], simplify_cells=True)
    ident = str(data.get("id", path.stem.removeprefix("WildPPG_Part_")))
    if ident != path.stem.removeprefix("WildPPG_Part_"):
        raise ValueError("Native participant identifier disagrees with its filename")
    if not isinstance(data.get("wrist"), dict) or not isinstance(data.get("sternum"), dict):
        raise ValueError("Native WildPPG must contain wrist and sternum structs")
    ppg, ppg_fs, ppg_description = _native_channel(data["wrist"], "ppg_g")
    ecg, ecg_fs, ecg_description = _native_channel(data["sternum"], "ecg")
    axes = [_native_channel(data["wrist"], axis) for axis in ("acc_x", "acc_y", "acc_z")]
    if len({len(a[0]) for a in axes}) != 1 or len({a[1] for a in axes}) != 1:
        raise ValueError("Wrist acceleration axes must share lengths and clocks")
    acc = np.column_stack([a[0] for a in axes])
    notes = data.get("notes", "")
    notes = str(notes) if not isinstance(notes, np.ndarray) or notes.size else ""
    result = {"participant_id": ident, "notes": notes, "ppg": ppg, "ppg_sampling_hz": ppg_fs,
              "acc": acc, "acc_sampling_hz": axes[0][1], "ecg": ecg, "ecg_sampling_hz": ecg_fs,
              "ppg_description": ppg_description, "ecg_description": ecg_description,
              "clock_policy": "official synchronized native sample-index axis; common start0; no per-sensor reset"}
    del data, axes
    return result


def derive_native_rr(ecg: np.ndarray, fs: float, config: LabelGenerationConfig,
                     core_seconds: float = 300, context_seconds: float = 5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Detect QRS in bounded contextual blocks and use the shared RR validity rule.

    Each peak is owned by exactly one core block; overlap is for filtering only.
    Nonfinite ECG samples are never bridged by an interval. No RR smoothing or
    replacement, quotient filtering, or precomputed dataset HR is performed.
    """
    signal = np.asarray(ecg, dtype=float)
    core = max(1, int(round(core_seconds * fs)))
    context = max(1, int(round(context_seconds * fs)))
    peaks = []
    detector = wfdb_detector()
    audit = {"core_seconds": core_seconds, "context_seconds": context_seconds,
             "detector": "wfdb4.3.0.processing.xqrs_detect; default untuned configuration", "failed_core_blocks": 0,
             "nonfinite_ecg_samples": int((~np.isfinite(signal)).sum())}
    finite = np.isfinite(signal)
    # Prefix invalid counts permit an exact no-gap-bridging interval check.
    invalid_prefix = np.concatenate(([0], np.cumsum(~finite, dtype=np.int64)))
    for start in range(0, len(signal), core):
        end = min(start+core, len(signal))
        left, right = max(0, start-context), min(len(signal), end+context)
        block = signal[left:right]
        # Process finite runs separately; do not remove missing samples or shift time.
        mask = np.isfinite(block)
        borders = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
        runs = zip(np.flatnonzero(borders == 1), np.flatnonzero(borders == -1))
        core_peaks = []
        for a, b in runs:
            if b-a < max(3, int(2*fs)):
                continue
            detected = detector(block[a:b], fs=fs, verbose=False) + left+a
            core_peaks.extend(detected[(detected >= start) & (detected < end)].tolist())
        if not core_peaks:
            audit["failed_core_blocks"] += 1
        peaks.extend(core_peaks)
    peaks = np.unique(np.asarray(peaks, dtype=np.int64))
    audit["ecg_detected_peak_count"] = len(peaks)
    if len(peaks) < 2:
        return np.array([], float), np.array([], float), audit
    intervals = np.diff(peaks) * (1000.0/fs)
    no_gap = (invalid_prefix[peaks[1:]+1] - invalid_prefix[peaks[:-1]]) == 0
    physiological = np.isfinite(intervals) & (intervals >= 60000/config.max_hr_bpm) & (intervals <= 60000/config.min_hr_bpm)
    keep = no_gap & physiological
    audit["rr_gap_rejections"] = int((~no_gap).sum())
    audit["rr_physiological_rejections"] = int((~physiological).sum())
    audit["valid_rr_intervals"] = int(keep.sum())
    return peaks[1:][keep] * (1000.0/fs), intervals[keep], audit


def index_native_windows(signals: dict, rr_times: np.ndarray, rr_ms: np.ndarray,
                         config: LabelGenerationConfig, window_seconds=10., stride_seconds=2.,
                         min_ppg_coverage=.8) -> tuple[pd.DataFrame, dict]:
    ident = signals["participant_id"]
    ppg, fs = signals["ppg"], signals["ppg_sampling_hz"]
    duration = min(len(ppg)/fs, len(signals["ecg"])/signals["ecg_sampling_hz"],
                   len(signals["acc"])/signals["acc_sampling_hz"])
    rows, dropped = [], {"known_wrist_interruption": 0, "insufficient_ppg_coverage": 0,
                          "insufficient_valid_beat_intervals": 0}
    candidates = 0
    # Window boundary arithmetic is exact in seconds; the native fs can be fractional.
    for i in range(max(0, int(np.floor((duration-window_seconds)/stride_seconds))+1)):
        candidates += 1
        start_ms, end_ms = i*stride_seconds*1000, (i*stride_seconds+window_seconds)*1000
        left, right = int(round(start_ms*fs/1000)), int(round(end_ms*fs/1000))
        if any(left < b and right > a for a,b in KNOWN_WRIST_INTERRUPTION_SAMPLES.get(ident, [])):
            dropped["known_wrist_interruption"] += 1
            continue
        count = int(np.isfinite(ppg[left:right]).sum())
        if count < max(1, int(np.ceil(window_seconds*fs*min_ppg_coverage))):
            dropped["insufficient_ppg_coverage"] += 1
            continue
        a,b = np.searchsorted(rr_times, [start_ms,end_ms], side="left")
        if b-a < config.min_valid_beats:
            dropped["insufficient_valid_beat_intervals"] += 1
            continue
        session = f"{ident}:full_recording#1"
        rows.append({"participant_id": ident, "session_id": session, "session_name": "full_recording",
            "activity": "full_recording", "window_index": i, "window_start_ms": int(start_ms),
            "window_end_ms": int(end_ms), "window_length_s": float(window_seconds), "stride_s": float(stride_seconds),
            "ppg_sampling_hz": fs, "ppg_sample_count": count, "reference_sample_count": int(b-a),
            "valid_beat_count": int(b-a), "label_hr_bpm": float(np.median(60000.0/rr_ms[a:b])),
            "label_method": config.method, "label_aggregation": config.aggregation, "reference_source": "ecg",
            "ppg_inverted": False, "ppg_canonical_source": "native_mat:wrist.ppg_g.v",
            "ppg_inversion_count": 0, "wear_location": "wrist"})
    return pd.DataFrame(rows), {"candidate_windows": candidates, "accepted_windows": len(rows),
        "dropped_windows": dropped, "duration_seconds": duration, "native_notes": signals["notes"],
        "interruption_policy": "official wrist exclusions for k2s/l38; an0 exclusion applies to ankle per official baseline"}


def export_native_dataset(dataset_root: str | Path, output_root: str | Path, participant_ids=None,
                          window_seconds=10., stride_seconds=2., min_valid_beats=2,
                          min_hr_bpm=35., max_hr_bpm=220., min_ppg_coverage=.8,
                          test_size=.2, random_state=42) -> Path:
    """Export one participant at a time, resuming only verified identical inputs."""
    root, out = Path(dataset_root).resolve(), Path(output_root).resolve()
    paths = native_paths(root)
    selected = participant_ids or sorted(paths)
    if not selected or set(selected)-set(paths):
        raise FileNotFoundError("Native WildPPG participant MAT files are missing")
    if window_seconds != 10 or stride_seconds != 2:
        raise ValueError("The corrected native protocol requires 10-second windows and 2-second stride")
    config = LabelGenerationConfig(min_valid_beats=min_valid_beats, min_hr_bpm=min_hr_bpm, max_hr_bpm=max_hr_bpm)
    for directory in [out, out/"signals", out/"indices", out/"manifests"]:
        directory.mkdir(parents=True, exist_ok=True)
    # Seeded participant assignment; no labels or windows enter split generation.
    import random
    shuffled = list(selected)
    random.Random(random_state).shuffle(shuffled)
    n_test = max(1, min(len(selected)-1, int(np.ceil(len(selected)*test_size)))) if len(selected)>1 else 0
    test = sorted(shuffled[:n_test]); train = sorted(set(selected)-set(test))
    protocol = {"window_seconds": window_seconds, "stride_seconds": stride_seconds,
                "label_generation": config.to_dict(), "min_ppg_coverage": min_ppg_coverage,
                "native_channel": "wrist.ppg_g", "source_code_sha256": sha256(__file__),
                "label_code_sha256": sha256(Path(__file__).with_name("labels.py")),
                "reference_detector": "WFDB4.3.0 default XQRS; native_reference_v2",
                "detector_source_sha256": sha256(Path(wfdb_detector().__code__.co_filename)),
                "reference_QC_policy": "prespecified raw ECG epochs; no model prediction errors used in detector selection"}
    records, frames = {}, []
    for ident in selected:
        print(f"Native WildPPG {ident}: load/validate/export", flush=True)
        path = paths[ident]
        inputs = {"native_mat_sha256": sha256(path), "protocol": protocol}
        receipt_path = out/"manifests"/f"{ident}.json"
        archive_path, index_path = out/"signals"/f"{ident}.npz", out/"indices"/f"{ident}.csv"
        if receipt_path.exists():
            record = json.loads(receipt_path.read_text())
            if record["inputs"] != inputs or sha256(archive_path) != record["archive_sha256"] or sha256(index_path) != record["index_sha256"]:
                raise ValueError("Native export inputs/cache changed; select a fresh output directory")
            frame = pd.read_csv(index_path)
        else:
            signals = read_native_signals(path)
            rr_times, rr_ms, rr_audit = derive_native_rr(signals["ecg"], signals["ecg_sampling_hz"], config)
            frame, window_audit = index_native_windows(signals, rr_times, rr_ms, config,
                window_seconds, stride_seconds, min_ppg_coverage)
            np.savez(archive_path, ppg=signals["ppg"], acc=signals["acc"],
                     ppg_sampling_hz=signals["ppg_sampling_hz"], acc_sampling_hz=signals["acc_sampling_hz"],
                     ecg_sampling_hz=signals["ecg_sampling_hz"], rr_timestamps_ms=rr_times, rr_interval_ms=rr_ms)
            frame.to_csv(index_path, index=False)
            record = {"inputs": inputs, "path": archive_path.relative_to(out).as_posix(),
                "index_path": index_path.relative_to(out).as_posix(), "archive_sha256": sha256(archive_path),
                "index_sha256": sha256(index_path), "ecg_audit": rr_audit, "window_audit": window_audit,
                "ppg_sampling_hz": signals["ppg_sampling_hz"], "acc_sampling_hz": signals["acc_sampling_hz"],
                "ecg_sampling_hz": signals["ecg_sampling_hz"], "ppg_description": signals["ppg_description"],
                "clock_policy": signals["clock_policy"], "raw_source": str(path), "native_notes": signals["notes"]}
            write_json(receipt_path, record)
            del signals, rr_times, rr_ms
            gc.collect()
        records[ident] = record
        frame["split"] = "test" if ident in test else "train"
        frame["window_uid"] = frame.participant_id+"|"+frame.session_id+"|"+frame.window_start_ms.astype(str)+"|"+frame.window_end_ms.astype(str)
        frames.append(frame)
    windows = pd.concat(frames, ignore_index=True)
    windows_path = out/"wildppg_wrist_windows.csv"
    windows.to_csv(windows_path, index=False)
    manifest = {"artifact_name": "wildppg_wrist_ecg_w10_s2_beat_interval_instant_hr_median",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "WildPPG-wrist",
        "dataset_root": str(root), "canonical_schema": {"version": "canonical_ppg_v1"},
        "reference_source": "ecg", "ppg_inverted": False, "ppg_canonical_source": "native_mat:wrist.ppg_g.v",
        "window_seconds": window_seconds, "stride_seconds": stride_seconds,
        "label_method": config.method, "label_aggregation": config.aggregation,
        "min_valid_beats": config.min_valid_beats, "min_reference_samples": 1,
        "label_generation": {**config.to_dict(), "ecg_processing": "WFDB4.3.0 default XQRS; 300s core/5s context; intervals assigned to second beat",
            "no_dataset_precomputed_hr": True, "window_audit": {ident:r["window_audit"] for ident,r in records.items()}},
        "window_storage": "continuous_npz_index_v1", "windows_path": windows_path.name, "labels_path": windows_path.name,
        "participant_archives": records, "num_windows": len(windows), "num_participants": len(selected),
        "num_train_windows": int(windows.split.eq("train").sum()), "num_test_windows": int(windows.split.eq("test").sum()),
        "train_participants": train, "test_participants": test, "random_state": random_state,
        "test_size": test_size, "split_name": "wildppg_wrist_seeded_participant_holdout",
        "signal_polarity_policy": "native wrist green PPG preserved; Galaxy inversion not applied",
        "activity_policy": "No activity labels supplied in native MAT; full_recording is an explicit unknown-activity placeholder",
        "reference_limitations": "fex/kjd/p5d contain dataset notes warning of noisy ECG; physiological RR bounds alone do not establish reference accuracy"}
    target = out/"wildppg_wrist_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json"
    write_json(target, manifest)
    return target


def iter_native_windows(manifest: dict, chunk_size=2048):
    """Yield standard signal-window DataFrames, loading one continuous archive at a time."""
    base = Path(manifest["manifest_path"]).parent
    metadata = pd.read_csv(manifest["windows_path"])
    for ident, frame in metadata.groupby("participant_id", sort=False):
        record = manifest["participant_archives"][ident]
        with np.load(base/record["path"]) as archive:
            ppg = archive["ppg"]; fs = float(archive["ppg_sampling_hz"])
            rr_times, rr_ms = archive["rr_timestamps_ms"], archive["rr_interval_ms"]
            for start in range(0,len(frame),chunk_size):
                rows = []
                for row in frame.iloc[start:start+chunk_size].to_dict("records"):
                    a,b = int(round(row["window_start_ms"]*fs/1000)), int(round(row["window_end_ms"]*fs/1000))
                    values = ppg[a:b]; timestamps = np.arange(a,b,dtype=float)*(1000/fs)
                    keep = np.isfinite(values)
                    ra,rb = np.searchsorted(rr_times,[row["window_start_ms"],row["window_end_ms"]],side="left")
                    row.update(ppg_values=values[keep], ppg_raw_values=values[keep], ppg_timestamps_ms=timestamps[keep],
                        reference_timestamps_ms=rr_times[ra:rb], reference_rr_interval_ms_values=rr_ms[ra:rb],
                        reference_hr_bpm_values=60000/rr_ms[ra:rb])
                    rows.append(row)
                yield pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,default=Path("data/raw/WildPPG-wrist"))
    parser.add_argument("--output-root",type=Path,default=Path("data/processed/external_2026_11_01/WildPPG-wrist_v2"))
    parser.add_argument("--participants",nargs="*")
    args=parser.parse_args()
    print(export_native_dataset(args.dataset_root,args.output_root,args.participants))


if __name__=="__main__":
    main()
