"""Memory-bounded real PPG-DaLiA export using continuous arrays and window indices.

The official release's protocol-0 pickle is large. Each participant is decoded
once, the unused channels are released, and only native wrist arrays and ECG
beat annotations are retained. The precomputed smoothed ``label`` field is never
used. Numerical window targets follow labels.LabelGenerationConfig without
expanding overlapping waveforms into thousands of Python lists.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import build_artifact_name
from .canonical import canonical_schema_description
from .labels import LabelGenerationConfig, build_instant_hr_from_ecg
from .ppgdalia_loader import (
    ACTIVITY_HZ, ACTIVITY_LABELS, CHEST_ECG_HZ, DATASET_NAME, WRIST_ACC_HZ,
    WRIST_BVP_HZ, discover_ppgdalia_participant_paths, read_ppgdalia_pickle, validated_ppgdalia_rpeaks,
)

STORAGE_VERSION = "continuous_npz_index_v1"
COLUMNS = ["window_uid", "window_key", "split", "participant_id", "session_id", "session_name",
    "activity", "window_index", "window_start_ms", "window_end_ms", "window_length_s", "stride_s",
    "ppg_sampling_hz", "ppg_sample_count", "reference_sample_count", "valid_beat_count", "label_hr_bpm",
    "label_method", "label_aggregation", "reference_source", "ppg_inverted", "ppg_inversion_count",
    "ppg_canonical_source"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def participant_arrays(payload, config: LabelGenerationConfig):
    """Extract native arrays; supplied R peaks are zero-based ECG sample indices."""
    wrist, chest = payload.get("signal", {}).get("wrist", {}), payload.get("signal", {}).get("chest", {})
    ppg = np.asarray(wrist.get("BVP"), dtype=np.float64).reshape(-1)
    acc = np.asarray(wrist.get("ACC"), dtype=np.float64)
    ecg = np.asarray(chest.get("ECG"), dtype=np.float64).reshape(-1)
    activity = np.asarray(payload.get("activity", []), dtype=float).reshape(-1)
    if not len(ppg) or not len(ecg) or acc.ndim != 2 or acc.shape[1] != 3 or not len(acc):
        raise ValueError("PPG-DaLiA requires nonempty wrist BVP/ACC and chest ECG")
    lengths = {"ppg_samples": len(ppg), "acc_samples": len(acc), "ecg_samples": len(ecg), "activity_samples": len(activity)}
    audit = {**lengths, "precomputed_HR_label_used": False, "raw_ppg_polarity_preserved": True}
    supplied = payload.get("rpeaks")
    if supplied is not None and np.asarray(supplied).size >= 2:
        peaks = validated_ppgdalia_rpeaks(supplied,len(ecg),audit)
        # Preserve sub-millisecond precision from the sampling index. There is
        # no use of PPG-derived peaks or the dataset's smoothed HR label field.
        rr = np.diff(peaks).astype(float) * (1000.0 / CHEST_ECG_HZ)
        beat_timestamps = peaks[1:].astype(float) * (1000.0 / CHEST_ECG_HZ)
        audit.update(reference_method="official_supplied_ECG_rpeak_indices", rpeak_index_origin=0,
                     detected_peak_count=len(peaks), reference_detection_failure=False)
    else:
        # This branch supports recordings without official annotations. It is
        # independent of any PPG predictor and uses the shared ECG detector.
        reference_audit = {}
        reference = build_instant_hr_from_ecg(pd.DataFrame({
            "timestamp_ms": np.arange(len(ecg), dtype=float) * (1000.0 / CHEST_ECG_HZ), "ecg_uv": ecg}),
            config=config, audit=reference_audit)
        rr = reference.rr_interval_ms.to_numpy(float)
        beat_timestamps = reference.timestamp_ms.to_numpy(float)
        peaks = np.asarray([], dtype=np.int64)
        audit.update(reference_method="shared_ECG_detector", ECG_detection_audit=reference_audit,
                     reference_detection_failure=reference.empty)
    audit["raw_reference_intervals"] = len(rr)
    return {"ppg": ppg, "acc": acc, "activity": activity, "rpeaks": peaks,
            "rr_timestamps_ms": beat_timestamps, "rr_interval_ms": rr,
            "ppg_sampling_hz": np.asarray(WRIST_BVP_HZ), "acc_sampling_hz": np.asarray(WRIST_ACC_HZ),
            "ecg_sampling_hz": np.asarray(CHEST_ECG_HZ), "activity_sampling_hz": np.asarray(ACTIVITY_HZ)}, audit


def activity_intervals(activity, duration_ms):
    """Return disjoint activity intervals on the official common zero-time axis."""
    activity = np.asarray(activity, dtype=float).reshape(-1)
    if not len(activity):
        return [("full_recording", 0, int(duration_ms))]
    if not np.isfinite(activity).all() or np.any(activity != np.floor(activity)):
        raise ValueError("Activity annotations must contain finite integer codes")
    values = activity.astype(int)
    boundaries = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    result = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        start = int(round(a * 1000 / ACTIVITY_HZ))
        end = min(int(duration_ms), int(round(b * 1000 / ACTIVITY_HZ)))
        if end > start:
            result.append((ACTIVITY_LABELS.get(int(values[a]), f"activity_{values[a]}"), start, end))
    return result


def iter_windows(arrays, participant, split, config, window_seconds=10., stride_seconds=2., min_ppg_coverage=.8, audit=None):
    """Yield scalar windows with exact shared RR validation and no interpolation."""
    if config.method != "beat_interval_instant_hr" or config.aggregation not in {"mean", "median"}:
        raise ValueError("Compact PPG-DaLiA export requires beat-interval targets")
    if config.rr_relative_deviation_threshold is not None:
        raise ValueError("Compact export does not enable optional window-relative RR rejection")
    if window_seconds <= 0 or stride_seconds <= 0 or not 0 < min_ppg_coverage <= 1:
        raise ValueError("Invalid window, stride or coverage settings")
    audit = audit if audit is not None else {}
    audit.update(candidate_windows=0, accepted_windows=0, dropped_windows={}, sessions=[])
    ppg = arrays["ppg"]
    times, rr = arrays["rr_timestamps_ms"], arrays["rr_interval_ms"]
    if len(times) != len(rr) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Reference timestamps must be finite, increasing and aligned with intervals")
    valid = np.isfinite(rr) & (rr >= 60000/config.max_hr_bpm) & (rr <= 60000/config.min_hr_bpm)
    valid_times, valid_rr = times[valid], rr[valid]
    audit["physiologic_or_nonfinite_rr_rejections_unique"] = int((~valid).sum())
    audit["valid_reference_intervals"] = int(valid.sum())
    duration_ms = int(min(len(ppg)/WRIST_BVP_HZ, len(arrays["acc"])/WRIST_ACC_HZ) * 1000)
    window_ms, stride_ms = int(window_seconds*1000), int(stride_seconds*1000)
    if not window_ms or not stride_ms:
        raise ValueError("Window and stride must be at least one millisecond")
    minimum_samples = math.ceil(window_seconds * WRIST_BVP_HZ * min_ppg_coverage)
    aggregate = np.median if config.aggregation == "median" else np.mean
    counts = {}
    for name, session_start, session_end in activity_intervals(arrays["activity"], duration_ms):
        counts[name] = counts.get(name, 0) + 1
        session = f"{participant}:{name}#{counts[name]}"
        record = {"session_id": session, "start_ms": session_start, "end_ms": session_end,
                  "candidate_windows": 0, "accepted_windows": 0}
        audit["sessions"].append(record)
        for index, start in enumerate(range(session_start, session_end-window_ms+1, stride_ms)):
            end = start+window_ms
            audit["candidate_windows"] += 1
            record["candidate_windows"] += 1
            left, right = int(math.ceil(start*WRIST_BVP_HZ/1000)), int(math.ceil(end*WRIST_BVP_HZ/1000))
            sample_count = int(np.isfinite(ppg[left:right]).sum())
            if sample_count < minimum_samples:
                reason = "insufficient_ppg_coverage"
            elif not np.isfinite(ppg[left:right]).all():
                # Preserve the native clock instead of connecting over gaps.
                reason = "nonfinite_ppg_samples"
            else:
                a, b = np.searchsorted(valid_times, [start, end], side="left")
                reason = "insufficient_valid_beat_intervals" if b-a < config.min_valid_beats else None
            if reason:
                audit["dropped_windows"][reason] = audit["dropped_windows"].get(reason, 0)+1
                continue
            raw_a, raw_b = np.searchsorted(times, [start, end], side="left")
            target = float(aggregate(60000./valid_rr[a:b]))
            audit["accepted_windows"] += 1
            record["accepted_windows"] += 1
            yield {"window_uid": f"{session}:w{index}", "window_key": f"{participant}|{session}|{start}|{end}",
                "split": split, "participant_id": participant, "session_id": session, "session_name": name,
                "activity": name, "window_index": index, "window_start_ms": start, "window_end_ms": end,
                "window_length_s": window_seconds, "stride_s": stride_seconds, "ppg_sampling_hz": WRIST_BVP_HZ,
                "ppg_sample_count": sample_count, "reference_sample_count": int(raw_b-raw_a),
                "valid_beat_count": int(b-a), "label_hr_bpm": target, "label_method": config.method,
                "label_aggregation": config.aggregation, "reference_source": "ecg", "ppg_inverted": False,
                "ppg_inversion_count": 0, "ppg_canonical_source": "wrist/BVP"}


def export_compact(dataset_root, output_root, participant_ids=None, *, window_seconds=10., stride_seconds=2.,
                   label_aggregation="median", min_valid_beats=2, min_ppg_coverage=.8,
                   min_hr_bpm=35., max_hr_bpm=220., test_size=.2, random_state=42):
    """Export one participant at a time and retain resumable verified receipts."""
    root, out = Path(dataset_root).resolve(), Path(output_root).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Derived compact data must be separate from the raw dataset")
    paths = {p.participant_id: p.pickle_path for p in discover_ppgdalia_participant_paths(root)}
    participants = list(participant_ids) if participant_ids else list(paths)
    if len(participants) < 2 or not set(participants) <= set(paths):
        raise ValueError("Compact export requires at least two discovered participants")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0,1)")
    shuffled = sorted(participants)
    random.Random(random_state).shuffle(shuffled)
    test = sorted(shuffled[:max(1,int(round(len(shuffled)*test_size)))])
    train = sorted(set(participants)-set(test))
    config = LabelGenerationConfig(aggregation=label_aggregation, min_valid_beats=min_valid_beats,
                                    min_hr_bpm=min_hr_bpm, max_hr_bpm=max_hr_bpm)
    settings = {"window_seconds": window_seconds, "stride_seconds": stride_seconds,
                "min_ppg_coverage": min_ppg_coverage, "label_generation": config.to_dict(),
                "participants": participants, "train_participants": train, "test_participants": test}
    out.mkdir(parents=True, exist_ok=True)
    for name in ["participants", "manifests", "labels", "windows"]:
        (out/name).mkdir(exist_ok=True)
    protocol = out/"manifests/export_protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != settings:
        raise ValueError("Compact export protocol changed; use a new output root")
    _json(protocol, settings)
    participant_archives, receipts = {}, []
    for participant in participants:
        path = paths[participant]
        archive = out/f"participants/{participant}.npz"
        csv_path = out/f"windows/{participant}.csv"
        receipt_path = out/f"manifests/{participant}.json"
        source_hash = sha256_file(path)
        inputs = {"source_pickle_sha256": source_hash, "settings": settings,
                  "export_code_sha256": sha256_file(Path(__file__)),
                  "loader_code_sha256": sha256_file(Path(__file__).with_name("ppgdalia_loader.py")),
                  "label_rules_code_sha256": sha256_file(Path(__file__).with_name("labels.py"))}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["inputs"] != inputs or receipt["archive_sha256"] != sha256_file(archive) or receipt["windows_sha256"] != sha256_file(csv_path):
                raise ValueError(f"Changed source/export/cache for {participant}; use a new output root")
            print(f"{participant}: verified completed compact export ({receipt['num_windows']} windows)", flush=True)
        else:
            print(f"{participant}: decoding {path.name}, {path.stat().st_size:,} bytes", flush=True)
            payload = read_ppgdalia_pickle(path)
            arrays, payload_audit = participant_arrays(payload, config)
            del payload
            gc.collect()
            # Uncompressed NPZ avoids expensive decoding on every model batch.
            np.savez(archive, **arrays)
            window_audit = {}
            count = 0
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                for row in iter_windows(arrays, participant, "test" if participant in test else "train", config,
                                        window_seconds, stride_seconds, min_ppg_coverage, window_audit):
                    writer.writerow(row)
                    count += 1
            if not count:
                raise ValueError(f"No valid PPG-DaLiA windows for {participant}")
            receipt = {"participant_id": participant, "inputs": inputs, "source_pickle": str(path),
                       "num_windows": count, "array_metadata": payload_audit, "window_audit": window_audit,
                       "archive_sha256": sha256_file(archive), "windows_sha256": sha256_file(csv_path)}
            _json(receipt_path, receipt)
            del arrays
            gc.collect()
            print(f"{participant}: compact export completed ({count} windows)", flush=True)
        receipts.append(receipt)
        participant_archives[participant] = {"path": archive.relative_to(out).as_posix(),
            "sha256": receipt["archive_sha256"], "ppg_sampling_hz": WRIST_BVP_HZ,
            "acc_sampling_hz": WRIST_ACC_HZ, "clock": "synchronized_relative_zero_ms",
            "ppg_key": "ppg", "acc_key": "acc", "timestamp_policy": "sample_index / sampling_hz"}
    artifact = build_artifact_name("ecg", window_seconds, stride_seconds, config.method, config.aggregation, DATASET_NAME)
    windows_path = out/f"windows/{artifact}_windows.csv"
    with windows_path.open("w", newline="", encoding="utf-8") as combined:
        writer = csv.DictWriter(combined, fieldnames=COLUMNS)
        writer.writeheader()
        for participant in participants:
            with (out/f"windows/{participant}.csv").open(newline="", encoding="utf-8") as handle:
                writer.writerows(csv.DictReader(handle))
    total = sum(r["num_windows"] for r in receipts)
    train_n = sum(r["num_windows"] for r in receipts if r["participant_id"] in train)
    manifest = {"artifact_name": artifact, "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": DATASET_NAME,
        "dataset_root": Path(os.path.relpath(root,out)).as_posix(), "canonical_schema": canonical_schema_description().to_dict(),
        "reference_source": "ecg", "ppg_inverted": False, "ppg_canonical_source": "wrist/BVP",
        "window_seconds": window_seconds, "stride_seconds": stride_seconds, "label_method": config.method,
        "label_aggregation": config.aggregation, "min_valid_beats": min_valid_beats, "min_reference_samples": 1,
        "label_generation": {**config.to_dict(), "ECG_reference": "official supplied zero-based 700-Hz ECG R peaks; shared detector fallback only if absent",
            "rpeak_annotation_duplicate_policy": "remove consecutive identical sample-index repeats; audit counts; surrounding nonzero RR unchanged; reject decreasing/out-of-range indices",
            "precomputed_dataset_HR_labels_used": False, "timestamp_precision": "float64 native sampling index; no beat-time rounding",
            "window_audit": {"participants": {r["participant_id"]:r["window_audit"] for r in receipts}},
            "polarity": {"rule":"ppg = ppg_raw", "inversion_count":0}},
        "test_size": test_size, "random_state": random_state, "num_windows": total, "num_train_windows": train_n,
        "num_test_windows": total-train_n, "num_participants": len(participants), "train_participants":train,
        "test_participants":test, "split_config_path":None, "split_name":f"ppgdalia_prespecified_subject_holdout_seed{random_state}",
        "validation_strategy":{"type":"participant_grouped_inside_training_only"}, "validation_folds":None,
        "window_storage": STORAGE_VERSION, "windows_path":windows_path.relative_to(out).as_posix(),
        "labels_path":windows_path.relative_to(out).as_posix(), "windows_sha256":sha256_file(windows_path),
        "participant_archives": participant_archives,
        "raw_input_sha256": {r["participant_id"]:r["inputs"]["source_pickle_sha256"] for r in receipts},
        "native_acceleration_unit": "native_release_values; verify supplied documentation before physical-unit interpretation"}
    manifest_path = out/f"{artifact}_manifest.json"
    _json(manifest_path, manifest)
    print(f"PPG-DaLiA compact manifest: {manifest_path} ({total} windows)",flush=True)
    return manifest
