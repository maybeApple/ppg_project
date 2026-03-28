"""Train regressors on extracted embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.regression.evaluate import evaluate_prediction_frame


@dataclass(slots=True)
class RegressionArtifacts:
    """Paths and metadata for one regressor training run."""

    model_name: str
    regressor: str
    output_dir: str
    predictions_path: str
    metrics_path: str
    estimator_path: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a JSON-serializable dictionary."""

        return {
            "model_name": self.model_name,
            "regressor": self.regressor,
            "output_dir": self.output_dir,
            "predictions_path": self.predictions_path,
            "metrics_path": self.metrics_path,
            "estimator_path": self.estimator_path,
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument(
        "--regressor",
        choices=["linear", "ridge", "random_forest", "gradient_boosting"],
        default="ridge",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_feature_bundle(feature_manifest_path: str | Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Load exported embeddings, metadata, and manifest JSON."""

    manifest_path = Path(feature_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = np.load(_resolve_manifest_path(manifest_path, manifest["features_path"]))
    metadata = pd.read_csv(_resolve_manifest_path(manifest_path, manifest["metadata_path"]))
    if len(features) != len(metadata):
        raise ValueError(
            f"Feature rows ({len(features)}) do not match metadata rows ({len(metadata)})."
        )
    return features, metadata, manifest


def split_train_test(
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Split features strictly using the saved participant-level split labels."""

    if "split" not in metadata.columns:
        raise KeyError("Metadata must contain a `split` column.")

    train_mask = metadata["split"] == "train"
    test_mask = metadata["split"] == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Both train and test rows must be present in the metadata split column.")

    train_metadata = metadata.loc[train_mask].reset_index(drop=True)
    test_metadata = metadata.loc[test_mask].reset_index(drop=True)
    train_features = features[train_mask.to_numpy()]
    test_features = features[test_mask.to_numpy()]
    train_labels = train_metadata["label_hr_bpm"].to_numpy(dtype=float, copy=True)
    return train_features, test_features, train_metadata, test_metadata, train_labels


def build_regressor_search(
    regressor_name: str,
    random_state: int,
    n_splits: int,
) -> GridSearchCV | Pipeline:
    """Construct a regressor or tuned search object."""

    if regressor_name == "linear":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )

    if regressor_name == "ridge":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge()),
            ]
        )
        return GridSearchCV(
            estimator=pipeline,
            param_grid={"regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
            cv=GroupKFold(n_splits=n_splits),
            scoring="neg_mean_squared_error",
            n_jobs=1,
            refit=True,
        )

    if regressor_name == "random_forest":
        pipeline = Pipeline(
            steps=[
                (
                    "regressor",
                    RandomForestRegressor(
                        random_state=random_state,
                        n_jobs=1,
                        max_features="sqrt",
                    ),
                )
            ]
        )
        return GridSearchCV(
            estimator=pipeline,
            param_grid={
                "regressor__n_estimators": [100],
                "regressor__max_depth": [16],
                "regressor__min_samples_leaf": [1, 5],
            },
            cv=GroupKFold(n_splits=n_splits),
            scoring="neg_mean_squared_error",
            n_jobs=1,
            refit=True,
        )

    if regressor_name == "gradient_boosting":
        pipeline = Pipeline(
            steps=[
                (
                    "regressor",
                    GradientBoostingRegressor(random_state=random_state),
                )
            ]
        )
        return GridSearchCV(
            estimator=pipeline,
            param_grid={
                "regressor__n_estimators": [100],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [2],
            },
            cv=GroupKFold(n_splits=n_splits),
            scoring="neg_mean_squared_error",
            n_jobs=1,
            refit=True,
        )

    raise ValueError(f"Unsupported regressor: {regressor_name}")


def train_and_predict(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    regressor_name: str,
    random_state: int,
) -> tuple[np.ndarray, object, dict[str, object]]:
    """Train a regressor and return test predictions plus fit metadata."""

    n_groups = int(pd.Series(train_groups).nunique())
    n_splits = min(5, n_groups)
    if n_splits < 2:
        raise ValueError("At least two distinct training participants are required for grouped validation.")

    estimator = build_regressor_search(
        regressor_name=regressor_name,
        random_state=random_state,
        n_splits=n_splits,
    )

    if isinstance(estimator, GridSearchCV):
        estimator.fit(train_features, train_labels, groups=train_groups)
        fitted_estimator = estimator.best_estimator_
        fit_metadata = {
            "cv_best_score": float(estimator.best_score_),
            "best_params": estimator.best_params_,
            "cv_n_splits": n_splits,
            "cv_fold_assignments": describe_group_kfold_assignments(train_groups, n_splits),
        }
    else:
        estimator.fit(train_features, train_labels)
        fitted_estimator = estimator
        fit_metadata = {
            "cv_n_splits": n_splits,
            "cv_fold_assignments": describe_group_kfold_assignments(train_groups, n_splits),
        }

    predictions = fitted_estimator.predict(test_features)
    return predictions.astype(float, copy=False), fitted_estimator, fit_metadata


def default_output_dir(feature_manifest: dict[str, object], regressor_name: str) -> Path:
    """Build a default output directory grouped by feature extractor and date."""

    model_name = str(feature_manifest["model_name"])
    return (
        Path("experiments")
        / f"{model_name}_results"
        / datetime.now().strftime("%Y-%m-%d")
        / f"regression_{regressor_name}"
    )


def save_regression_outputs(
    model_name: str,
    regressor_name: str,
    output_dir: str | Path,
    predictions: pd.DataFrame,
    estimator: object,
    metrics: dict[str, object],
) -> RegressionArtifacts:
    """Persist predictions, metrics, and the fitted estimator."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = target_dir / f"{model_name}_{regressor_name}_predictions.csv"
    metrics_path = target_dir / f"{model_name}_{regressor_name}_metrics.json"
    estimator_path = target_dir / f"{model_name}_{regressor_name}_estimator.joblib"

    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(estimator, estimator_path)

    return RegressionArtifacts(
        model_name=model_name,
        regressor=regressor_name,
        output_dir=str(target_dir),
        predictions_path=str(predictions_path),
        metrics_path=str(metrics_path),
        estimator_path=str(estimator_path),
    )


def _resolve_manifest_path(manifest_path: Path, stored_path: str | Path) -> Path:
    """Resolve a path stored inside a feature manifest."""

    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def describe_group_kfold_assignments(train_groups: np.ndarray, n_splits: int) -> list[dict[str, object]]:
    """Describe the deterministic participant-level GroupKFold assignments."""

    dummy_features = np.zeros((len(train_groups), 1), dtype=np.float32)
    assignments: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=n_splits)
    group_series = pd.Series(train_groups)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(dummy_features, groups=train_groups), start=1):
        assignments.append(
            {
                "fold": fold_idx,
                "train_participants": sorted(group_series.iloc[train_idx].dropna().unique().tolist()),
                "validation_participants": sorted(group_series.iloc[val_idx].dropna().unique().tolist()),
                "num_train_windows": int(len(train_idx)),
                "num_validation_windows": int(len(val_idx)),
            }
        )
    return assignments


def save_run_log(
    output_dir: str | Path,
    model_name: str,
    regressor_name: str,
    argv: list[str],
    feature_manifest_path: str,
    metrics: dict[str, object],
    artifacts: RegressionArtifacts,
) -> Path:
    """Persist a lightweight run log for reproducibility."""

    log_path = Path(output_dir) / f"{model_name}_{regressor_name}_run_log.json"
    payload = {
        "module": "src.regression.train_regressor",
        "argv": argv,
        "feature_manifest_path": feature_manifest_path,
        "artifacts": artifacts.to_dict(),
        "metrics": metrics,
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return log_path


def main() -> None:
    """Train a regressor on extracted embeddings and evaluate it on held-out participants."""

    args = parse_args()
    features, metadata, feature_manifest = load_feature_bundle(args.feature_manifest)
    train_features, test_features, train_metadata, test_metadata, train_labels = split_train_test(
        features=features,
        metadata=metadata,
    )

    test_predictions, fitted_estimator, fit_metadata = train_and_predict(
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        train_groups=train_metadata["participant_id"].to_numpy(copy=True),
        regressor_name=args.regressor,
        random_state=args.random_state,
    )

    prediction_frame = test_metadata.copy()
    prediction_frame["predicted_hr_bpm"] = test_predictions
    summary = evaluate_prediction_frame(prediction_frame)

    output_dir = args.output_dir or default_output_dir(feature_manifest, args.regressor)
    metrics = {
        "model_name": feature_manifest["model_name"],
        "regressor": args.regressor,
        "random_state": args.random_state,
        "num_train_windows": int(len(train_metadata)),
        "num_test_windows": int(len(test_metadata)),
        "train_participants": sorted(train_metadata["participant_id"].unique().tolist()),
        "test_participants": sorted(test_metadata["participant_id"].unique().tolist()),
        "embedding_dim": int(features.shape[1]),
        "feature_manifest_path": str(args.feature_manifest),
        **fit_metadata,
        **summary.to_dict(),
    }
    artifacts = save_regression_outputs(
        model_name=str(feature_manifest["model_name"]),
        regressor_name=args.regressor,
        output_dir=output_dir,
        predictions=prediction_frame,
        estimator=fitted_estimator,
        metrics=metrics,
    )
    run_log_path = save_run_log(
        output_dir=output_dir,
        model_name=str(feature_manifest["model_name"]),
        regressor_name=args.regressor,
        argv=sys.argv,
        feature_manifest_path=str(args.feature_manifest),
        metrics=metrics,
        artifacts=artifacts,
    )

    print(
        json.dumps(
            {**artifacts.to_dict(), **summary.to_dict(), "run_log_path": str(run_log_path)},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
