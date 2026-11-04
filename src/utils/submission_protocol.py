"""Fail-closed contracts for the participant-independent submission experiment.

These checks are used by the real fitting path, as well as the negative tests.
Reference beat counts and expert errors are deliberately absent from FEATURES.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MOTION = ["acc_norm_mean", "acc_norm_std", "acc_dominant_frequency_hz",
          "acc_cadence_band_power_fraction"]
QUALITY = ["ppg_amplitude_range", "ppg_clipping_rate", "ppg_flatline_rate",
           "ppg_autocorr_peak_strength", "ppg_spectral_entropy",
           "ppg_spectral_peak_sharpness", "peak_spectral_disagreement_bpm"]
FEATURES = {"motion": MOTION, "quality": QUALITY, "motion_quality": MOTION + QUALITY}
BASELINES = ["peak_based", "spectral", "pulseppg_linear", "pulseppg_ridge",
             "pulseppg_random_forest", "papagei_linear", "papagei_ridge"]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                    encoding="utf-8")


def window_keys(frame: pd.DataFrame) -> pd.Series:
    return (frame.participant_id.astype(str) + "|" + frame.session_id.astype(str) + "|"
            + frame.window_start_ms.astype("int64").astype(str) + "|"
            + frame.window_end_ms.astype("int64").astype(str))


def validate_split(metadata: pd.DataFrame, train: list[str], test: list[str]) -> None:
    if not train or not test or set(train) & set(test):
        raise ValueError("Training and test participant sets must be nonempty and disjoint")
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError("Duplicate participants in split configuration")
    if metadata.participant_id.isna().any():
        raise ValueError("Missing participant identifiers")
    observed_train = set(metadata.loc[metadata.split == "train", "participant_id"])
    observed_test = set(metadata.loc[metadata.split == "test", "participant_id"])
    if observed_train != set(train) or observed_test != set(test):
        raise ValueError("Metadata participant membership disagrees with fixed split")
    if not metadata.split.isin(["train", "test"]).all():
        raise ValueError("Unknown split labels")
    if metadata.groupby("participant_id").split.nunique().max() != 1:
        raise ValueError("One participant crosses the train/test window boundary")
    if metadata.window_key.duplicated().any():
        raise ValueError("Duplicated window identifiers")


def assert_fit_scope(groups, allowed_train, heldout=()) -> None:
    observed = set(np.asarray(groups).astype(str))
    if not observed or not observed <= set(allowed_train):
        raise ValueError("Label leakage: fit contains non-training/test participants")
    if observed & set(heldout):
        raise ValueError("Label leakage: validation participant enters fit")


def grouped_splits(groups, n_splits: int):
    groups = np.asarray(groups)
    n = min(n_splits, len(np.unique(groups)))
    if n < 2:
        raise ValueError("At least two distinct participants required for grouped CV")
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups))


def fit_expert(x, y, groups, kind, seed, allowed_train, heldout=(), inner_folds=3):
    """Nested tuning: every scaler is inside the search pipeline, never prefit."""
    assert_fit_scope(groups, allowed_train, heldout)
    if len(x) != len(y) or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Expert training requires aligned finite features/targets")
    folds = grouped_splits(groups, inner_folds)
    if kind == "linear":
        estimator = make_pipeline(StandardScaler(), LinearRegression())
        grid = None
    elif kind == "ridge":
        estimator = make_pipeline(StandardScaler(), Ridge())
        grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif kind == "random_forest":
        estimator = make_pipeline(RandomForestRegressor(n_estimators=100, max_depth=16,
            max_features="sqrt", random_state=seed, n_jobs=1))
        grid = {"randomforestregressor__min_samples_leaf": [1, 5]}
    else:
        raise ValueError(f"Unknown expert {kind}")
    search = (GridSearchCV(estimator, grid, scoring="neg_mean_absolute_error", cv=folds,
                          n_jobs=1, refit=True, error_score="raise") if grid else estimator)
    search.fit(x, y)
    fitted = search.best_estimator_ if grid else search
    record = {"fit_participants": sorted(set(groups)), "heldout_participants": sorted(set(heldout)),
              "inner_folds": [{"train_participants": sorted(set(np.asarray(groups)[a])),
                               "validation_participants": sorted(set(np.asarray(groups)[b]))}
                              for a, b in folds],
              "best_params": search.best_params_ if grid else {},
              "preprocessing_fit_scope": "each_inner_training_fold_pipeline",
              "selection_scoring": "negative_window_MAE_with_grouped_CV"}
    return fitted, record


def validate_oof(oof: pd.DataFrame, provenance: list[dict], allowed_train, test) -> None:
    if set(oof.participant_id) != set(allowed_train) or set(oof.participant_id) & set(test):
        raise ValueError("OOF router data contains test participants or misses training participants")
    if oof.window_key.duplicated().any():
        raise ValueError("OOF rows must be unique")
    if oof.groupby("participant_id").fold.nunique().ne(1).any():
        raise ValueError("A participant must occur in exactly one OOF validation fold")
    fold_records = {int(r["fold"]): r for r in provenance}
    if len(fold_records) != len(provenance) or set(oof.fold) != set(fold_records):
        raise ValueError("OOF fold records must be unique and complete")
    for fold, rows in oof.groupby("fold"):
        if int(fold) not in fold_records:
            raise ValueError("OOF provenance missing")
        record = fold_records[int(fold)]
        fit, validation = set(record["train_participants"]), set(record["validation_participants"])
        if fit & validation or (fit | validation) != set(allowed_train):
            raise ValueError("OOF expert fit/validation provenance overlaps or is incomplete")
        if set(rows.participant_id) != validation:
            raise ValueError("OOF row participant not held out by its expert")
        if set(record.get("experts", {})) != set(BASELINES[2:]):
            raise ValueError("OOF provenance must identify every fitted expert")
        for expert in record.get("experts", {}).values():
            if set(expert["fit_participants"]) != fit:
                raise ValueError("OOF expert trained on an excluded participant")
            for inner in expert.get("inner_folds", []):
                a, b = set(inner["train_participants"]), set(inner["validation_participants"])
                if a & b or (a | b) != fit:
                    raise ValueError("Inner CV leakage or missing participant")


def fit_gate(frame, target, columns, allowed_train, seed=42):
    assert_fit_scope(frame.participant_id, allowed_train)
    if not set(columns) <= set(MOTION + QUALITY):
        raise ValueError("Forbidden router feature: label, reference, error or unsupported field")
    if not np.isin(target, [0, 1]).all() or len(frame) != len(target):
        raise ValueError("Invalid gate target")
    classifier = (LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
                  if len(np.unique(target)) == 2 else DummyClassifier(strategy="prior"))
    gate = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), classifier)
    gate.fit(frame[columns], target)
    return gate


def gate_probability(gate, frame, columns):
    p = gate.predict_proba(frame[columns])
    classes = gate.steps[-1][1].classes_
    return p[:, list(classes).index(1)] if 1 in classes else np.zeros(len(frame))


def route(classical, foundation, p, soft=False):
    classical, foundation, p = map(lambda a: np.asarray(a, dtype=float), (classical, foundation, p))
    if classical.shape != foundation.shape or p.shape != classical.shape:
        raise ValueError("Routing prediction arrays must align")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Gate probabilities must be finite in [0,1]")
    return ((1-p)*classical + p*foundation if soft else np.where(p >= .5, foundation, classical))


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    valid = np.isfinite(y) & np.isfinite(prediction)
    if not valid.any():
        raise ValueError("No finite pairs for metrics")
    e = np.abs(y[valid] - prediction[valid])
    return {"MAE": float(e.mean()), "RMSE": float(np.sqrt(np.mean(e**2))),
            "median_absolute_error": float(np.median(e)), "p95_absolute_error": float(np.quantile(e, .95)),
            "catastrophic_error_rate_20bpm": float(np.mean(e > 20)), "n_windows": int(len(e))}


def select_experts(oof, allowed_train, candidates=BASELINES):
    """Preselect using participant-balanced training OOF MAE on common valid rows."""
    assert_fit_scope(oof.participant_id, allowed_train)
    if "split" in oof and not oof.split.eq("train").all():
        raise ValueError("Label leakage: expert selection accepts training OOF rows only")
    usable = np.isfinite(oof[candidates].to_numpy(float)).all(axis=1) & np.isfinite(oof.y_true)
    common = oof.loc[usable]
    if common.empty:
        raise ValueError("No common finite training OOF expert predictions")
    scores = {}
    for name in candidates:
        error = (common[name] - common.y_true).abs()
        scores[name] = float(error.groupby(common.participant_id).mean().mean())
    classical = min([n for n in candidates if n in ["peak_based", "spectral"]], key=lambda n: (scores[n], n))
    foundation = min([n for n in candidates if n.startswith(("pulseppg", "papagei"))], key=lambda n: (scores[n], n))
    best = min(candidates, key=lambda n: (scores[n], n))
    return {"selection_source": "training_participants_oof_only",
            "selection_protocol": "nested_GroupKFold_experts_then_participant_mean_OOF_MAE",
            "selection_metric": "participant_balanced_MAE", "scores": scores,
            "best_classical": classical, "best_foundation": foundation, "best_single": best,
            "primary_router": "motion_quality_soft", "primary_router_source": "a_priori_config",
            "common_training_windows": len(common), "training_participants": sorted(set(common.participant_id))}


def apply_regime_bins(values, edges):
    """Training quantiles may coincide: tied thresholds put ties in the low bin."""
    edges = np.asarray(edges, dtype=float)
    if len(edges) != 2 or not np.isfinite(edges).all() or edges[0] > edges[1]:
        raise ValueError("Two finite ordered training regime thresholds required")
    values = np.asarray(values, dtype=float)
    labels = np.where(values <= edges[0], "low", np.where(values <= edges[1], "mid", "high"))
    return np.where(np.isfinite(values), labels, "missing")
