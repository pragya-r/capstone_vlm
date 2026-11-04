"""
Generate a live VLM inference-latency figure from live.jsonl.

Example:
python make_live_latency_figure.py \
    --input runs/20261102_181322/live.jsonl \
    --output figures \
    --fmt png \
    --no-title
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {i}") from exc
    if not rows:
        raise ValueError("No records found.")
    return pd.json_normalize(rows)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 600,
    })


def make_figure(df: pd.DataFrame, output_dir: Path, fmt: str, show_title: bool) -> None:
    required = {"frame_id", "decision.latency_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values("frame_id").copy()
    frame = pd.to_numeric(df["frame_id"], errors="coerce")
    latency = pd.to_numeric(df["decision.latency_s"], errors="coerce")

    valid = frame.notna() & latency.notna()
    frame = frame[valid].to_numpy(dtype=float)
    latency = latency[valid].to_numpy(dtype=float)

    if len(latency) == 0:
        raise ValueError("No valid latency values found.")

    mean = float(np.mean(latency))
    median = float(np.median(latency))
    p95 = float(np.quantile(latency, 0.95))
    minimum = float(np.min(latency))
    maximum = float(np.max(latency))
    max_idx = int(np.argmax(latency))
    max_frame = int(frame[max_idx])

    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.2], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharey=ax)

    ax.plot(
        frame,
        latency,
        marker="o",
        markersize=3,
        linewidth=0.9,
        label="Inference latency",
    )
    ax.axhline(
        median,
        linestyle="--",
        linewidth=1.0,
        label=f"Median = {median:.2f} s",
    )
    ax.annotate(
        f"Max {maximum:.2f} s",
        (frame[max_idx], latency[max_idx]),
        xytext=(6, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
    )

    ax.set_xlabel("Decision-cycle frame")
    ax.set_ylabel("Inference latency (s)")
    ax.set_xlim(frame.min() - 1, frame.max() + 1)

    pad = max(0.25, 0.08 * (maximum - minimum))
    ax.set_ylim(max(0, minimum - pad), maximum + pad)

    ax.grid(axis="y", linewidth=0.5, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False)

    axh.hist(
        latency,
        bins=10,
        orientation="horizontal",
        edgecolor="white",
        linewidth=0.5,
    )
    axh.set_xlabel("Count")
    axh.tick_params(axis="y", labelleft=False, left=False)
    axh.grid(False)

    if show_title:
        fig.suptitle("Live VLM Inference Latency", y=1.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"figure_live_inference_latency.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("\nLive inference latency summary")
    print(f"Decision cycles: {len(latency)}")
    print(f"Mean:   {mean:.2f} s")
    print(f"Median: {median:.2f} s")
    print(f"P95:    {p95:.2f} s")
    print(f"Min:    {minimum:.2f} s")
    print(f"Max:    {maximum:.2f} s at frame {max_frame}")

    if len(latency) > 1:
        remaining = np.delete(latency, max_idx)
        print(
            f"Range excluding max: "
            f"{remaining.min():.2f}–{remaining.max():.2f} s"
        )

    print(f"Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=Path("figures"))
    parser.add_argument(
        "--fmt",
        choices=["png", "pdf", "svg", "eps"],
        default="png",
    )
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()

    set_style()
    df = load_jsonl(args.input)
    make_figure(df, args.output, args.fmt, not args.no_title)


if __name__ == "__main__":
    main()
