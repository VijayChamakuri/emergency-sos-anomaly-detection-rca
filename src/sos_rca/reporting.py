"""Generate self-contained reporting artifacts without optional UI dependencies."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def write_engineering_report(path: Path, summary: dict, segments: pd.DataFrame) -> None:
    model = summary["model"]
    exp = summary["experiment"]
    top = segments.iloc[0]
    text = f"""# Engineering validation report

> Synthetic portfolio analysis. No real emergency calls, customers, carriers, deployments, or field interventions are represented.

## Temporal holdout performance

| Metric | Value |
|---|---:|
| Precision | {model['precision']:.3f} |
| Recall | {model['recall']:.3f} |
| Average precision | {model['average_precision']:.3f} |
| Decision threshold | {model['threshold']:.4f} |

Confusion matrix: TP {model['true_positive']:,}, FP {model['false_positive']:,}, TN {model['true_negative']:,}, FN {model['false_negative']:,}.

The model trains on the earliest 60 percent of events, calibrates its threshold on the next 20 percent, and is evaluated once on the latest 20 percent. Outcome fields, failure reasons, and injected labels are excluded from the feature matrix.

## Root cause evidence

The highest-impact synthetic segment is **{top['region']} / {top['environment']}**. It has an anomaly rate of {top['anomaly_rate']:.2%}, a completion rate of {top['completion_rate']:.2%}, and median handoff latency of {top['median_handoff_ms']:.0f} ms. This is an association aligned with the injected scenario, not proof of a real operational cause.

## Simulated intervention

The randomized simulation estimates an absolute completion lift of {exp['absolute_lift']:.2%} ({exp['control_completion_rate']:.2%} to {exp['treatment_completion_rate']:.2%}), with a two-sided p-value of {exp['p_value']:.3g}. The statistic is computed from generated observations rather than hard-coded.

## Limitations

- Synthetic labels are cleaner than real incident labels.
- Event-level simulation does not reproduce device and region correlation.
- Model attribution and segment lift are diagnostic signals, not causal proof.
- Production use would require privacy review, monitored data contracts, calibration, and operational validation.
"""
    path.write_text(text)


def _sparkline(values: list[float], width: int = 900, height: int = 210) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    points = []
    for i, value in enumerate(values):
        x = 10 + i * (width - 20) / max(len(values) - 1, 1)
        y = 10 + (high - value) * (height - 20) / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _point(values: list[float], index: int, width: int = 900, height: int = 210) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    x = 10 + index * (width - 20) / max(len(values) - 1, 1)
    y = 10 + (high - values[index]) * (height - 20) / span
    return x, y


def write_dashboard(path: Path, summary: dict, daily: pd.DataFrame, segments: pd.DataFrame) -> None:
    model, exp = summary["model"], summary["experiment"]
    series = (daily["completion_rate"] * 100).tolist()
    anomaly_marks = "".join(
        f'<circle cx="{_point(series, i)[0]:.1f}" cy="{_point(series, i)[1]:.1f}" r="5" fill="#b6412e"><title>Statistical anomaly: {daily.iloc[i]["date"]:%Y-%m-%d}</title></circle>'
        for i in range(len(daily))
        if bool(daily.iloc[i]["is_statistical_anomaly"])
    )
    first_date = pd.to_datetime(daily["date"].iloc[0]).strftime("%b %d, %Y")
    last_date = pd.to_datetime(daily["date"].iloc[-1]).strftime("%b %d, %Y")
    rows = "".join(
        f"<tr><td>{html.escape(str(row.region))}</td><td>{html.escape(str(row.environment))}</td>"
        f"<td>{row.events:,}</td><td>{row.anomaly_rate:.2%}</td><td>{row.completion_rate:.2%}</td>"
        f"<td>{row.median_handoff_ms:,.0f} ms</td></tr>"
        for row in segments.head(8).itertuples()
    )
    payload = html.escape(json.dumps(summary, default=str))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emergency SOS Reliability Intelligence</title>
<style>
:root{{--navy:#10263f;--ink:#182431;--muted:#607080;--teal:#0f7c7b;--amber:#b66a00;--paper:#f3f5f6;--card:#fff;--line:#dce2e6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}
header{{background:var(--navy);color:white;padding:28px max(24px,calc((100% - 1180px)/2))}}header p{{color:#cbd9e6;margin:4px 0 0}}
main{{max-width:1180px;margin:auto;padding:24px}}.notice{{background:#fff6df;border:1px solid #e6c26b;border-radius:10px;padding:14px 16px;color:#694000}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}}.card,.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;box-shadow:0 2px 8px #10263f0a}}
.label{{color:var(--muted);font-size:13px;font-weight:650;text-transform:uppercase;letter-spacing:.05em}}.value{{font-size:31px;font-weight:750;margin-top:5px;color:var(--navy)}}
.two{{display:grid;grid-template-columns:1.45fr 1fr;gap:16px;margin:16px 0}}h2{{font-size:19px;margin:0 0 4px}}.sub{{color:var(--muted);margin:0 0 16px}}
svg{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:11px;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-size:12px;text-transform:uppercase}}
.pill{{display:inline-block;background:#e4f3f1;color:#075f5e;padding:4px 9px;border-radius:99px;font-weight:650;font-size:12px}}footer{{color:var(--muted);padding:20px 0}}
@media(max-width:820px){{.grid{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}}}@media(max-width:480px){{.grid{{grid-template-columns:1fr}}main{{padding:14px}}}}
</style></head><body>
<header><span class="pill">MODEL MONITORING</span><h1>Emergency SOS Reliability Intelligence</h1><p>Daily anomaly detection, geographic risk, and root cause evidence</p></header>
<main><div class="notice"><strong>Synthetic portfolio dataset.</strong> Metrics and findings are simulated and do not represent real emergency calls or interventions.</div>
<section class="grid" aria-label="Key performance indicators">
<div class="card"><div class="label">Synthetic events</div><div class="value">{summary['data']['rows']:,}</div></div>
<div class="card"><div class="label">Model precision</div><div class="value">{model['precision']:.1%}</div></div>
<div class="card"><div class="label">Model recall</div><div class="value">{model['recall']:.1%}</div></div>
<div class="card"><div class="label">Simulated lift</div><div class="value">+{exp['absolute_lift']:.1%}</div></div></section>
<section class="two"><div class="panel"><h2>Daily call completion</h2><p class="sub">Temporal signal across the generated observation window</p>
<svg viewBox="0 0 900 260" role="img" aria-label="Line chart of daily call completion rate with statistical anomaly markers"><line x1="10" y1="210" x2="890" y2="210" stroke="#cbd4da"/><polyline points="{_sparkline(series)}" fill="none" stroke="#0f7c7b" stroke-width="4" stroke-linejoin="round"/>{anomaly_marks}<text x="10" y="235" fill="#607080" font-size="13">{first_date}</text><text x="890" y="235" fill="#607080" font-size="13" text-anchor="end">{last_date}</text><text x="10" y="20" fill="#607080" font-size="13">{max(series):.1f}%</text><text x="10" y="205" fill="#607080" font-size="13">{min(series):.1f}%</text><circle cx="710" cy="18" r="5" fill="#b6412e"/><text x="722" y="22" fill="#607080" font-size="13">STL residual anomaly</text></svg></div>
<div class="panel"><h2>Validated signal</h2><p class="sub">Evidence chain from detection to intervention</p><p><strong>Highest-impact segment</strong><br>{html.escape(str(summary['top_segment']['region']))} / {html.escape(str(summary['top_segment']['environment']))}</p><p><strong>Temporal anomaly days</strong><br>{summary['statistical_anomaly_days']}</p><p><strong>Simulated experiment</strong><br>p = {exp['p_value']:.3g}, n = {exp['control_n'] + exp['treatment_n']:,}</p></div></section>
<section class="panel"><h2>Priority investigation queue</h2><p class="sub">Ranked by volume-adjusted anomaly impact. Driver labels remain hypotheses.</p><div style="overflow:auto"><table><thead><tr><th>Region</th><th>Rurality</th><th>Events</th><th>Anomaly rate</th><th>Completion</th><th>Median handoff</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<footer>Generated from a deterministic pipeline. Embedded summary checksum source: <span hidden>{payload}</span>See the engineering report for methodology and limitations.</footer></main></body></html>"""
    path.write_text(document)
