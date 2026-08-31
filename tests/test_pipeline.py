import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sos_rca.analysis import daily_anomalies, root_cause_segments, simulate_handoff_experiment
from sos_rca.data import generate_events
from sos_rca.model import FEATURES, train_temporal_model
from sos_rca.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_generation_is_deterministic_and_valid(self):
        first = generate_events(5_000, seed=11)
        second = generate_events(5_000, seed=11)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["event_id"].is_unique)
        self.assertTrue(first["call_completed"].isin([0, 1]).all())
        self.assertTrue(first["timestamp"].is_monotonic_increasing)

    def test_model_excludes_outcome_and_post_outcome_fields(self):
        self.assertNotIn("call_completed", FEATURES)
        self.assertNotIn("failure_reason", FEATURES)
        self.assertNotIn("is_injected_anomaly", FEATURES)

    def test_temporal_model_and_rca_recover_injected_segment(self):
        events = generate_events(35_000, seed=42)
        result = train_temporal_model(events, precision_target=0.85)
        self.assertGreaterEqual(result.metrics["precision"], 0.85)
        self.assertEqual(result.metrics["train_rows"], result.metrics["test_rows"] * 3)
        segments = root_cause_segments(result.scored_test)
        self.assertEqual(segments.iloc[0]["environment"], "rural")
        self.assertEqual(segments.iloc[0]["region"], "West")

    def test_statistical_and_experiment_outputs_are_computed(self):
        events = generate_events(12_000, seed=4)
        daily = daily_anomalies(events)
        self.assertTrue(daily["residual_zscore"].notna().all())
        experiment = simulate_handoff_experiment(rows=10_000, seed=7)
        self.assertTrue(0.05 < experiment["absolute_lift"] < 0.11)
        self.assertLess(experiment["p_value"], 0.05)

    def test_end_to_end_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary = run_pipeline(output, rows=12_000, seed=42)
            self.assertTrue(summary["data"]["is_synthetic"])
            self.assertTrue((output / "reports" / "executive_dashboard.html").exists())
            self.assertTrue((output / "reports" / "engineering_report.md").exists())
            self.assertTrue((output / "models" / "anomaly_model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
