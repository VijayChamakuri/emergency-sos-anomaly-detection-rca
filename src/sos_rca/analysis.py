"""Statistical anomaly detection, root cause ranking, and experiment analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from statsmodels.tsa.seasonal import STL


def daily_anomalies(frame: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["timestamp"], utc=True).dt.floor("D")
    daily = data.groupby("date", observed=True).agg(
        calls=("event_id", "size"), completion_rate=("call_completed", "mean")
    )
    daily["failure_rate"] = 1 - daily["completion_rate"]
    period = 7 if len(daily) >= 21 else max(2, len(daily) // 3)
    fitted = STL(daily["failure_rate"], period=period, robust=True).fit()
    daily["trend"] = fitted.trend
    daily["seasonal"] = fitted.seasonal
    daily["residual"] = fitted.resid
    scale = daily["residual"].std(ddof=0)
    daily["residual_zscore"] = daily["residual"] / scale if scale else 0.0
    daily["is_statistical_anomaly"] = daily["residual_zscore"].abs() >= z_threshold
    return daily.reset_index()


def root_cause_segments(scored: pd.DataFrame) -> pd.DataFrame:
    overall = scored["predicted_anomaly"].mean()
    grouped = scored.groupby(["region", "environment"], observed=True).agg(
        events=("event_id", "size"),
        anomaly_rate=("predicted_anomaly", "mean"),
        completion_rate=("call_completed", "mean"),
        median_handoff_ms=("handoff_latency_ms", "median"),
        mean_signal_dbm=("signal_dbm", "mean"),
    )
    grouped["anomaly_lift"] = grouped["anomaly_rate"] / max(overall, 1e-9)
    grouped["impact_score"] = grouped["events"] * grouped["anomaly_rate"] * grouped["anomaly_lift"]
    return grouped.reset_index().sort_values("impact_score", ascending=False)


def simulate_handoff_experiment(rows: int = 24_000, seed: int = 7) -> dict:
    """Simulate a randomized intervention and calculate a two-proportion z-test."""
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, 2, rows)
    baseline_rate = 0.78
    treatment_rate = baseline_rate + 0.082
    probability = np.where(assignment == 1, treatment_rate, baseline_rate)
    completed = rng.random(rows) < probability
    control = completed[assignment == 0]
    treatment = completed[assignment == 1]
    p1, p0 = treatment.mean(), control.mean()
    pooled = completed.mean()
    se = np.sqrt(pooled * (1 - pooled) * (1 / len(treatment) + 1 / len(control)))
    z = (p1 - p0) / se
    p_value = 2 * norm.sf(abs(z))
    table = np.array([[treatment.sum(), len(treatment) - treatment.sum()], [control.sum(), len(control) - control.sum()]])
    chi2, chi_p, _, _ = chi2_contingency(table, correction=False)
    return {
        "control_n": int(len(control)),
        "treatment_n": int(len(treatment)),
        "control_completion_rate": float(p0),
        "treatment_completion_rate": float(p1),
        "absolute_lift": float(p1 - p0),
        "relative_lift": float((p1 - p0) / p0),
        "z_statistic": float(z),
        "p_value": float(p_value),
        "chi_square": float(chi2),
        "chi_square_p_value": float(chi_p),
        "is_simulated": True,
    }
