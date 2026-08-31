# Data dictionary

All fields are synthetic. No device, customer, carrier, or emergency-service records are used.

| Field | Type | Meaning |
|---|---|---|
| `event_id` | string | Unique synthetic call attempt identifier |
| `timestamp` | UTC datetime | Synthetic event time |
| `region` | category | Coarse synthetic geographic region |
| `environment` | category | Urban, suburban, or rural context |
| `signal_dbm` | float | Simulated received signal strength |
| `satellite_count` | integer | Simulated visible satellite count |
| `handoff_latency_ms` | float | Simulated terrestrial-to-satellite handoff latency |
| `weather_severity` | float | Normalized synthetic weather severity from 0 to 1 |
| `retry_count` | integer | Connection retries known at scoring time |
| `call_completed` | binary | Simulated operational outcome, excluded from model features |
| `failure_reason` | category | Simulated post-outcome diagnosis, excluded from model features |
| `is_injected_anomaly` | binary | Ground truth for evaluation only, excluded from features |
