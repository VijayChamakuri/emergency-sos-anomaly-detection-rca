"""Deterministic synthetic telemetry for development and demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = np.array(["Northeast", "Southeast", "Midwest", "Southwest", "West"])
ENVIRONMENTS = np.array(["urban", "suburban", "rural"])
FAILURE_REASONS = np.array(["none", "handoff_timeout", "no_satellite", "network_drop"])


def generate_events(rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Create realistic, labeled SOS telemetry with an injected rural handoff issue."""
    if rows < 1_000:
        raise ValueError("rows must be at least 1,000")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    minute = rng.integers(0, 180 * 24 * 60, rows)
    timestamp = start + pd.to_timedelta(minute, unit="m")
    region = rng.choice(REGIONS, rows, p=[0.18, 0.22, 0.20, 0.16, 0.24])
    environment = rng.choice(ENVIRONMENTS, rows, p=[0.48, 0.30, 0.22])
    rural = environment == "rural"
    west = region == "West"
    hour = pd.DatetimeIndex(timestamp).hour.to_numpy()
    peak = ((hour >= 17) & (hour <= 21)).astype(float)

    signal_dbm = rng.normal(-101, 8, rows) - rural * 9 - west * 2
    satellite_count = np.clip(rng.poisson(6, rows) - rural * 2, 0, None)
    handoff_ms = np.clip(rng.normal(1150, 260, rows) + rural * 480 + peak * 120, 150, None)
    weather_severity = rng.beta(1.4, 5.0, rows)
    retry_count = rng.poisson(0.25 + rural * 0.22, rows)

    issue_start = pd.Timestamp("2025-04-10", tz="UTC")
    issue_end = pd.Timestamp("2025-07-01", tz="UTC")
    incident = rural & west & (timestamp >= issue_start) & (timestamp < issue_end)
    handoff_ms += incident * rng.normal(950, 160, rows)

    failure_logit = (
        -4.2
        + 0.014 * (-signal_dbm - 95)
        + 0.00125 * (handoff_ms - 1000)
        + 0.26 * retry_count
        + 0.9 * weather_severity
        + 1.9 * incident
    )
    failure_probability = 1 / (1 + np.exp(-failure_logit))
    completed = rng.random(rows) >= failure_probability
    anomaly = incident | ((handoff_ms > 2450) & rural)

    reason = np.full(rows, "none", dtype=object)
    failed = ~completed
    reason[failed & (handoff_ms > 2100)] = "handoff_timeout"
    reason[failed & (satellite_count <= 2)] = "no_satellite"
    reason[failed & (reason == "none")] = "network_drop"

    frame = pd.DataFrame(
        {
            "event_id": [f"SOS-{i:09d}" for i in range(rows)],
            "timestamp": timestamp,
            "region": region,
            "environment": environment,
            "signal_dbm": signal_dbm.round(2),
            "satellite_count": satellite_count.astype(int),
            "handoff_latency_ms": handoff_ms.round(1),
            "weather_severity": weather_severity.round(4),
            "retry_count": retry_count.astype(int),
            "call_completed": completed.astype(int),
            "failure_reason": reason,
            "is_injected_anomaly": anomaly.astype(int),
        }
    )
    return frame.sort_values("timestamp").reset_index(drop=True)
