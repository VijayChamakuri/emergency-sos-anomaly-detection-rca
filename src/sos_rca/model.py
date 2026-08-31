"""Leakage-safe supervised anomaly scoring and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "region",
    "environment",
    "signal_dbm",
    "satellite_count",
    "handoff_latency_ms",
    "weather_severity",
    "retry_count",
    "hour",
    "day_of_week",
]
NUMERIC = [
    "signal_dbm",
    "satellite_count",
    "handoff_latency_ms",
    "weather_severity",
    "retry_count",
    "hour",
    "day_of_week",
]
CATEGORICAL = ["region", "environment"]


@dataclass
class ModelResult:
    pipeline: Pipeline
    threshold: float
    metrics: dict
    scored_test: pd.DataFrame


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"], utc=True)
    result["hour"] = timestamp.dt.hour
    result["day_of_week"] = timestamp.dt.dayofweek
    return result


def train_temporal_model(frame: pd.DataFrame, precision_target: float = 0.90) -> ModelResult:
    """Train, calibrate, and test on consecutive 60/20/20 percent time windows.

    Calibration adds a five-point precision margin to reduce threshold fragility
    under modest temporal drift.
    """
    data = prepare_features(frame).sort_values("timestamp").reset_index(drop=True)
    train_end, validation_end = int(len(data) * 0.60), int(len(data) * 0.80)
    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]
    partitions = (train, validation, test)
    if any(part["is_injected_anomaly"].nunique() < 2 for part in partitions):
        raise ValueError("all temporal partitions must contain positive and negative labels")

    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=160, max_leaf_nodes=24, l2_regularization=1.0, random_state=42
    )
    pipeline = Pipeline([("features", preprocessing), ("model", classifier)])
    pipeline.fit(train[FEATURES], train["is_injected_anomaly"])
    validation_probability = pipeline.predict_proba(validation[FEATURES])[:, 1]
    precision, _, thresholds = precision_recall_curve(
        validation["is_injected_anomaly"], validation_probability
    )
    calibration_target = min(0.99, precision_target + 0.05)
    eligible = np.flatnonzero(precision[:-1] >= calibration_target)
    threshold = float(thresholds[eligible[0]]) if len(eligible) else 0.5
    probability = pipeline.predict_proba(test[FEATURES])[:, 1]
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["is_injected_anomaly"], predicted, labels=[0, 1]).ravel()
    metrics = {
        "average_precision": float(average_precision_score(test["is_injected_anomaly"], probability)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "threshold": threshold,
        "requested_precision": precision_target,
        "validation_precision_target": calibration_target,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }
    scored = test.copy()
    scored["anomaly_probability"] = probability
    scored["predicted_anomaly"] = predicted
    return ModelResult(pipeline, threshold, metrics, scored)
