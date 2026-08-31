# Model card

## Intended use

Demonstrate a leakage-aware anomaly detection and root cause analysis workflow for a portfolio project.

## Non-intended use

This model must not be used for emergency-service, safety, customer, network, or production decisions. It has not been trained on real calls or validated with operational data.

## Model and validation

The pipeline uses a histogram gradient-boosted classifier with standardized numeric features and one-hot encoded categorical features. Consecutive time windows allocate 60 percent to training, 20 percent to threshold calibration, and 20 percent to final testing. The validation target includes a five percentage point precision margin to reduce degradation under modest temporal drift. The threshold is locked before test evaluation. Post-outcome fields are excluded.

## Limitations

Synthetic labels are deterministic in structure, geographic units are coarse, and repeated device correlation is not represented. Production evaluation would also need time gaps between partitions, drift monitoring, confidence intervals, and validation against independently adjudicated incidents.

The injected label is partly defined by rural context and extreme handoff latency, and both are available model inputs. This makes part of the synthetic target directly reconstructible and explains the unusually high average precision. The score demonstrates pipeline mechanics, not expected production performance.
