# Engineering validation report

> Synthetic portfolio analysis. No real emergency calls, customers, carriers, deployments, or field interventions are represented.

## Temporal holdout performance

| Metric | Value |
|---|---:|
| Precision | 0.950 |
| Recall | 1.000 |
| Average precision | 0.999 |
| Decision threshold | 0.0001 |

Confusion matrix: TP 8,981, FP 468, TN 159,947, FN 4.

The model trains on the earliest 60 percent of events, calibrates its threshold on the next 20 percent, and is evaluated once on the latest 20 percent. Outcome fields, failure reasons, and injected labels are excluded from the feature matrix.

## Root cause evidence

The highest-impact synthetic segment is **West / rural**. It has an anomaly rate of 99.96%, a completion rate of 42.73%, and median handoff latency of 2609 ms. This is an association aligned with the injected scenario, not proof of a real operational cause.

## Simulated intervention

The randomized simulation estimates an absolute completion lift of 8.27% (77.51% to 85.78%), with a two-sided p-value of 1.8e-61. The statistic is computed from generated observations rather than hard-coded.

## Limitations

- Synthetic labels are cleaner than real incident labels.
- Event-level simulation does not reproduce device and region correlation.
- Model attribution and segment lift are diagnostic signals, not causal proof.
- Production use would require privacy review, monitored data contracts, calibration, and operational validation.
