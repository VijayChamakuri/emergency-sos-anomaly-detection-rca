# Architecture and decision record

## Goal

Provide a small, auditable batch pipeline that demonstrates anomaly detection, root cause prioritization, and experiment analysis without implying access to emergency-service data.

## Key decisions

1. **Synthetic-first evidence:** every generated artifact carries a disclosure. This prevents portfolio metrics from being mistaken for operational results.
2. **Chronological evaluation:** random row splitting would leak a persistent incident pattern across partitions. Consecutive train, validation, and test windows better approximate daily deployment.
3. **Locked threshold:** operating precision is calibrated on validation data, with a stability margin, and evaluated once on test data.
4. **Dual detection:** supervised event scoring and unsupervised daily residuals answer different questions and provide complementary evidence.
5. **Dependency-light reporting:** HTML and SVG are generated directly, so the core project does not require a dashboard server.

## Production extensions

- Immutable columnar storage and schema contracts.
- Device and region grouping with purged temporal splits.
- Calibrated probabilities, bootstrap confidence intervals, and drift alarms.
- Geospatial clustering on privacy-preserving cells.
- Human-reviewed incident labels and explicit feedback loops.
- Orchestrated daily scoring with idempotency, monitoring, and alert suppression.
