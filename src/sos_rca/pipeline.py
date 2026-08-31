"""End-to-end project pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .analysis import daily_anomalies, root_cause_segments, simulate_handoff_experiment
from .data import generate_events
from .model import train_temporal_model
from .reporting import write_dashboard, write_engineering_report


def run_pipeline(output: Path, rows: int = 100_000, seed: int = 42) -> dict:
    data_dir = output / "data"
    model_dir = output / "models"
    report_dir = output / "reports"
    for path in (data_dir, model_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    events = generate_events(rows=rows, seed=seed)
    result = train_temporal_model(events)
    daily = daily_anomalies(events)
    segments = root_cause_segments(result.scored_test)
    experiment = simulate_handoff_experiment()
    summary = {
        "data": {"rows": rows, "seed": seed, "is_synthetic": True},
        "model": result.metrics,
        "experiment": experiment,
        "top_segment": segments.iloc[0].to_dict(),
        "statistical_anomaly_days": int(daily["is_statistical_anomaly"].sum()),
    }

    events.to_csv(data_dir / "sos_events.csv", index=False)
    result.scored_test.to_csv(report_dir / "scored_events.csv", index=False)
    daily.to_csv(report_dir / "daily_anomalies.csv", index=False)
    segments.to_csv(report_dir / "root_cause_segments.csv", index=False)
    joblib.dump({"pipeline": result.pipeline, "threshold": result.threshold}, model_dir / "anomaly_model.joblib")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_engineering_report(report_dir / "engineering_report.md", summary, segments)
    write_dashboard(report_dir / "executive_dashboard.html", summary, daily, segments)
    return summary
