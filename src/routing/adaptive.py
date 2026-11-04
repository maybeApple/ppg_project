"""Small two-expert routers trained to reflect estimation error, not accuracy alone.

The archived submission gate remains untouched. This module offers development
alternatives with an explicit training-only API. Every scaler, imputer, regret
cap and objective is fitted solely on the rows passed to ``fit``. The caller is
responsible for providing honest out-of-fold expert predictions from its own
training participants, and for holding validation participants out end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.utils.submission_protocol import MOTION, QUALITY


METHODS = ("legacy_logistic", "balanced_logistic", "regret_logistic", "direct_soft")


def _vector(values, frame: pd.DataFrame, name: str, *, numeric: bool = True):
    if isinstance(values, pd.Series) and not values.index.equals(frame.index):
        raise ValueError(f"{name} index must exactly match the feature frame")
    array = np.asarray(values, dtype=float if numeric else object)
    if array.ndim != 1 or len(array) != len(frame):
        raise ValueError(f"{name} must be a one-dimensional aligned vector")
    if numeric and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _participant_weights(groups: np.ndarray) -> np.ndarray:
    if pd.isna(groups).any() or any(not str(group).strip() for group in groups):
        raise ValueError("Participant groups must be present and nonempty")
    keys = np.asarray([str(group) for group in groups])
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


class AdaptiveRouter(BaseEstimator):
    """Logistic or direct-error convex blending of two supplied expert estimates.

    ``columns`` must be a nonempty subset of the original inference feature
    whitelist. ``legacy_logistic`` reproduces the original unweighted gate;
    ``balanced_logistic`` gives equal total weight to each participant;
    ``regret_logistic`` also weights each decision by its capped absolute error
    gap. Classification uses foundation-wins labels, with ties favoring the
    classical expert.

    ``direct_soft`` minimizes participant-balanced mean
    ``sqrt((blend-y)**2 + epsilon**2) - epsilon`` plus
    ``regularization / 2 * ||coef||**2``. The intercept is unpenalized, and
    ``blend = classical + sigmoid(X @ coef + intercept) * (foundation-classical)``.
    This optimizes the actual soft estimator rather than a binary winner proxy.
    Coefficients/intercept are bounded to [-8,8]/[-12,12]. A constant blend and a
    balanced-logistic start are considered deterministically. Optimization
    failure is recorded and falls back to the training-optimal constant blend.
    The direct objective is nonconvex and provides no global optimum guarantee.
    """

    def __init__(self, columns, method="balanced_logistic", C=1.0,
                 regularization=0.02, smooth_epsilon_bpm=0.5,
                 regret_cap_quantile=0.95, max_iter=500, seed=42):
        self.columns = columns
        self.method = method
        self.C = C
        self.regularization = regularization
        self.smooth_epsilon_bpm = smooth_epsilon_bpm
        self.regret_cap_quantile = regret_cap_quantile
        self.max_iter = max_iter
        self.seed = seed

    def _validate_parameters(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown routing method: {self.method}")
        if isinstance(self.columns, str) or self.columns is None:
            raise ValueError("Router columns must be an explicit nonempty feature list")
        columns = list(self.columns)
        if not columns:
            raise ValueError("Router columns must be an explicit nonempty feature list")
        if len(set(columns)) != len(columns) or not set(columns) <= set(MOTION + QUALITY):
            raise ValueError("Forbidden or duplicate router feature: labels, errors and identities are not inputs")
        for name in ["C", "smooth_epsilon_bpm"]:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if not np.isfinite(self.regret_cap_quantile) or not 0 < self.regret_cap_quantile <= 1:
            raise ValueError("regret_cap_quantile must lie in (0,1]")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        return columns

    def _features(self, frame: pd.DataFrame):
        if (not isinstance(frame, pd.DataFrame) or frame.empty
                or not frame.index.is_unique or not frame.columns.is_unique):
            raise ValueError("Features require a nonempty DataFrame with unique row/column identities")
        if set(self.columns_) - set(frame):
            raise ValueError("Required inference features are missing")
        values = frame[self.columns_].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("Infinite inference features are invalid; NaN denotes missing observations")
        return values

    def _classifier(self, matrix, target, weights):
        if weights is not None:
            supported = weights > 0
            matrix, target, weights = matrix[supported], target[supported], weights[supported]
        classifier = (LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000,
                                        random_state=self.seed)
                      if len(np.unique(target)) == 2 else DummyClassifier(strategy="prior"))
        classifier.fit(matrix, target, sample_weight=weights)
        return classifier

    def fit(self, frame, y, classical, foundation, groups):
        self.columns_ = self._validate_parameters()
        matrix = self._features(frame)
        y = _vector(y, frame, "y")
        classical = _vector(classical, frame, "classical")
        foundation = _vector(foundation, frame, "foundation")
        groups = _vector(groups, frame, "groups", numeric=False)
        balanced = _participant_weights(groups)
        # Clear fitted attributes when the same estimator is fitted again.
        for name in ["constant_probability_", "classifier_", "coef_", "intercept_"]:
            self.__dict__.pop(name, None)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        matrix = self.imputer_.fit_transform(matrix)
        self.scaler_ = StandardScaler()
        if self.method == "legacy_logistic":
            matrix = self.scaler_.fit_transform(matrix)
        else:
            matrix = self.scaler_.fit(matrix, sample_weight=balanced).transform(matrix)
        self.n_features_in_ = len(self.columns_)
        self.feature_names_in_ = np.asarray(self.columns_, dtype=object)
        error_c, error_f = np.abs(classical-y), np.abs(foundation-y)
        target = (error_f < error_c).astype(int)
        self.diagnostics_ = {"method": self.method, "fit_rows": len(frame),
            "fit_participants": len(np.unique(groups.astype(str))),
            "feature_columns": list(self.columns_), "preprocessing_scope": "fit_rows_only",
            "weight_scope": "window_uniform" if self.method == "legacy_logistic" else "equal_participant_total",
            "foundation_wins_fraction": float(target.mean()), "ties_favor_classical": True}
        if self.method == "direct_soft":
            self._fit_direct(matrix, y, classical, foundation, target, balanced)
        else:
            weights = None if self.method == "legacy_logistic" else balanced * len(frame)
            if self.method == "regret_logistic":
                regret = np.abs(error_c-error_f)
                positive = regret > 0
                if positive.any():
                    # Participant-weighted cap prevents long recordings dominating its scale.
                    order = np.argsort(regret, kind="stable")
                    cap = float(np.interp(self.regret_cap_quantile,
                        np.cumsum(balanced[order]), regret[order]))
                    cap = max(cap, np.finfo(float).eps)
                    weights = balanced * np.minimum(regret, cap)
                    weights *= len(frame) / weights.sum()
                else:
                    cap = 0.0
                self.diagnostics_.update({"regret_cap_bpm": cap,
                    "regret_cap_quantile": self.regret_cap_quantile,
                    "regret_weight_source": "fit_expert_error_gap_only"})
            self.classifier_ = self._classifier(matrix, target, weights)
            self.diagnostics_["optimizer_status"] = "classification_fit_complete"
            if hasattr(self.classifier_, "coef_"):
                self.coef_ = self.classifier_.coef_.copy()
                self.intercept_ = self.classifier_.intercept_.copy()
        return self

    def _fit_direct(self, matrix, y, classical, foundation, target, weights):
        delta = foundation-classical
        if not np.any(delta):
            self.constant_probability_ = 0.0
            self.diagnostics_.update({"optimizer_status": "identical_experts_classical_fallback",
                "constant_probability": 0.0, "fallback_used": False})
            return
        epsilon = float(self.smooth_epsilon_bpm)
        def objective(parameters):
            p = expit(matrix @ parameters[:-1] + parameters[-1])
            residual = classical + p * delta - y
            denominator = np.hypot(residual, epsilon)
            loss = float(np.dot(weights, denominator-epsilon)
                         + self.regularization/2 * np.dot(parameters[:-1], parameters[:-1]))
            contribution = weights * residual / denominator * delta * p * (1-p)
            gradient = np.r_[matrix.T @ contribution + self.regularization*parameters[:-1],
                             contribution.sum()]
            return loss, gradient
        bound = float(expit(-12))
        constant = minimize_scalar(lambda p: float(np.dot(weights,
            np.hypot(classical+p*delta-y, epsilon)-epsilon)),
            bounds=(bound, 1-bound), method="bounded", options={"xatol": 1e-10})
        probability = float(np.clip(constant.x, bound, 1-bound))
        initial = np.r_[np.zeros(matrix.shape[1]), logit(probability)]
        baseline_loss = objective(initial)[0]
        starts = [initial]
        if len(np.unique(target)) == 2:
            warm = self._classifier(matrix, target, weights * len(matrix))
            starts.append(np.r_[np.clip(warm.coef_[0], -8, 8), np.clip(warm.intercept_[0], -12, 12)])
        results = []
        for start in starts:
            result = minimize(objective, start, jac=True, method="L-BFGS-B",
                bounds=[(-8, 8)]*matrix.shape[1]+[(-12, 12)],
                options={"maxiter": int(self.max_iter), "ftol": 1e-10, "gtol": 1e-6})
            results.append(result)
        valid = [result for result in results if result.success and np.isfinite(result.fun)
                 and np.isfinite(result.x).all() and result.fun <= baseline_loss + 1e-10]
        if valid:
            best = min(valid, key=lambda result: result.fun)
            self.coef_, self.intercept_ = best.x[:-1][None, :], best.x[-1:]
            status, final_loss = "direct_soft_fit_converged", float(best.fun)
        else:
            self.constant_probability_ = probability
            status, final_loss = "constant_blend_fallback_no_converged_improvement", baseline_loss
        self.diagnostics_.update({"optimizer_status": status, "fallback_used": not bool(valid),
            "constant_probability": probability, "constant_objective": baseline_loss,
            "fitted_objective": final_loss, "regularization": self.regularization,
            "smooth_epsilon_bpm": epsilon, "objective": "participant_mean_smoothed_MAE_plus_coefficient_L2",
            "optimization_runs": [{"success": bool(r.success), "iterations": int(r.nit),
                "objective": float(r.fun) if np.isfinite(r.fun) else None,
                "message": str(r.message)} for r in results]})

    def predict_probability(self, frame):
        check_is_fitted(self, "n_features_in_")
        matrix = self.scaler_.transform(self.imputer_.transform(self._features(frame)))
        if hasattr(self, "constant_probability_"):
            return np.full(len(frame), self.constant_probability_)
        if self.method == "direct_soft":
            return expit(matrix @ self.coef_[0] + self.intercept_[0])
        probability = self.classifier_.predict_proba(matrix)
        classes = self.classifier_.classes_.tolist()
        return probability[:, classes.index(1)] if 1 in classes else np.zeros(len(frame))

    def predict(self, frame, classical, foundation, soft=True):
        classical = _vector(classical, frame, "classical")
        foundation = _vector(foundation, frame, "foundation")
        probability = self.predict_probability(frame)
        return (classical + probability*(foundation-classical) if soft
                else np.where(probability >= .5, foundation, classical))
