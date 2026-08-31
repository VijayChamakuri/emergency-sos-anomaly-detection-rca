"""Generate source-backed README visualizations from pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
OUTPUT = ROOT / "docs" / "images"

NAVY = "#10263f"
TEAL = "#0f7c7b"
RED = "#b6412e"
AMBER = "#d58a16"
SLATE = "#607080"
LIGHT = "#eef2f3"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d9e0e4",
            "axes.labelcolor": SLATE,
            "axes.titlecolor": NAVY,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overview(summary: dict, daily: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 7.6), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=[0.8, 2.4])
    fig.suptitle("Emergency SOS reliability intelligence", fontsize=23, color=NAVY, weight="bold")
    fig.text(
        0.5,
        0.935,
        f"{summary['data']['rows']:,} synthetic events | Chronological holdout | Reproducible portfolio case study",
        ha="center",
        color=SLATE,
        fontsize=11,
    )

    cards = [
        ("EVENTS", f"{summary['data']['rows'] / 1000:.0f}K", "deterministic synthetic calls"),
        ("TEST PRECISION", f"{summary['model']['precision']:.1%}", "locked temporal holdout"),
        ("TEST RECALL", f"{summary['model']['recall']:.1%}", "injected anomaly recovery"),
        ("SIMULATED LIFT", f"+{summary['experiment']['absolute_lift']:.1%}", "completion-rate points"),
    ]
    for i, (label, value, note) in enumerate(cards):
        ax = fig.add_subplot(grid[0, i])
        ax.set_axis_off()
        ax.text(0.04, 0.82, label, transform=ax.transAxes, color=SLATE, fontsize=9, weight="bold")
        ax.text(0.04, 0.38, value, transform=ax.transAxes, color=NAVY, fontsize=25, weight="bold")
        ax.text(0.04, 0.10, note, transform=ax.transAxes, color=SLATE, fontsize=9)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#d9e0e4", linewidth=1.2))

    ax = fig.add_subplot(grid[1, :])
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    ax.plot(daily["date"], daily["completion_rate"], color=TEAL, linewidth=2.6, label="Daily completion rate")
    flagged = daily[daily["is_statistical_anomaly"].astype(bool)]
    ax.scatter(flagged["date"], flagged["completion_rate"], color=RED, edgecolor="white", linewidth=0.8, s=55, zorder=3, label="STL residual anomaly")
    ax.axvspan(pd.Timestamp("2025-04-10", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"), color=AMBER, alpha=0.12, label="Injected handoff incident")
    ax.set_title("Rural handoff incident appears in daily reliability signals", loc="left", pad=14)
    ax.set_ylabel("Call completion rate")
    ax.set_xlabel("Synthetic event date")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color=LIGHT, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="lower left")
    fig.text(0.01, 0.005, "Source: deterministic synthetic pipeline, seed 42. Not real emergency-service data.", color=SLATE, fontsize=9)
    save(fig, "project-overview.png")


def root_cause(segments: pd.DataFrame) -> None:
    data = segments.copy()
    data["segment"] = data["region"] + " / " + data["environment"]
    data = data.sort_values("impact_score", ascending=False).head(8).sort_values("impact_score")
    colors = [RED if segment == "West / rural" else "#7f909d" for segment in data["segment"]]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    y = np.arange(len(data))
    values = data["impact_score"].clip(lower=1)
    ax.hlines(y, 1, values, color="#d5dde2", linewidth=2)
    ax.scatter(values, y, color=colors, s=95, zorder=3, edgecolor="white", linewidth=0.8)
    for position, value, lift in zip(y, values, data["anomaly_lift"]):
        ax.annotate(f"{lift:.1f}x lift", (value, position), xytext=(8, 0), textcoords="offset points", va="center", color=SLATE, fontsize=10)
    ax.set_yticks(y, data["segment"])
    ax.set_xscale("log")
    ax.set_title("West / rural dominates the root-cause priority queue", loc="left", fontsize=17, pad=12)
    ax.text(0, 1.01, "Impact combines affected volume, anomaly rate, and anomaly lift", transform=ax.transAxes, color=SLATE)
    ax.set_xlabel("Volume-adjusted anomaly impact score (log scale)")
    ax.grid(axis="x", color=LIGHT)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    fig.text(0.01, 0.01, "Diagnostic ranking from synthetic test-window predictions. Association is not causal proof.", color=SLATE, fontsize=9)
    save(fig, "root-cause-ranking.png")


def confusion_matrix(summary: dict) -> None:
    metrics = summary["model"]
    matrix = np.array([[metrics["true_negative"], metrics["false_positive"]], [metrics["false_negative"], metrics["true_positive"]]])
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(np.log10(matrix + 1), cmap="Blues", vmin=0)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            color = "white" if image.norm(np.log10(value + 1)) > 0.58 else NAVY
            ax.text(column, row - 0.05, f"{value:,}", ha="center", va="center", fontsize=22, weight="bold", color=color)
            ax.text(column, row + 0.18, labels[row][column], ha="center", va="center", fontsize=10, color=color)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted anomaly"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual anomaly"])
    ax.set_title("Locked test threshold preserves high precision and recall", fontsize=16, pad=16)
    ax.set_xlabel(f"Precision {metrics['precision']:.1%} | Recall {metrics['recall']:.1%} | AP {metrics['average_precision']:.1%}", labelpad=14)
    ax.tick_params(length=0)
    fig.text(0.01, 0.01, "Chronological 20% holdout, 169,400 synthetic events.", color=SLATE, fontsize=9)
    save(fig, "confusion-matrix.png")


def main() -> None:
    style()
    summary = json.loads((REPORTS / "summary.json").read_text())
    daily = pd.read_csv(REPORTS / "daily_anomalies.csv")
    segments = pd.read_csv(REPORTS / "root_cause_segments.csv")
    overview(summary, daily)
    root_cause(segments)
    confusion_matrix(summary)


if __name__ == "__main__":
    main()
