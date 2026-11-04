"""
Generate final capstone result figures from the controlled and live JSONL logs.

Example
-------
python make_capstone_figures.py \
    --controlled results/final/final_experiment.jsonl \
    --live runs/20261102_181322/live.jsonl \
    --output figures \
    --fmt pdf \
    --no-titles

Outputs
-------
figure_4_1_action_agreement.<fmt>
figure_4_2_controlled_latency.<fmt>
figure_4_3_live_motion_recovery.<fmt>
figure_4_4_live_actions.<fmt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FIG_SIZE_SINGLE = (3.5, 2.9)
FIG_SIZE_POLICY = (3.5, 4.1)
FIG_SIZE_WIDE = (7.16, 2.9)

MODE_ORDER = [
    "current_only",
    "raw_history",
    "structured_temporal",
]

MODE_LABELS = {
    "current_only": "Current state",
    "raw_history": "Raw history",
    "structured_temporal": "Structured temporal",
}

MODE_COLORS = {
    "current_only": "#0000FF",         # blue
    "raw_history": "#FF0000",          # red
    "structured_temporal": "#00A000",  # green
}

MODE_LS = {
    "current_only": "-",
    "raw_history": "--",
    "structured_temporal": "-.",
}

SCENARIO_ORDER = [
    "healthy_no_enemy",
    "healthy_enemy",
    "critical_health_no_enemy",
    "critical_health_enemy",
    "no_ammo_no_enemy",
    "no_ammo_enemy",
    "enemy_long_absent",
    "enemy_recently_disappeared",
    "health_recovers",
    "ammo_recovers",
]

SCENARIO_LABELS = {
    "healthy_no_enemy": "Healthy, no enemy",
    "healthy_enemy": "Healthy, enemy",
    "critical_health_no_enemy": "Critical health, no enemy",
    "critical_health_enemy": "Critical health, enemy",
    "no_ammo_no_enemy": "No ammo, no enemy",
    "no_ammo_enemy": "No ammo, enemy",
    "enemy_long_absent": "Enemy absent throughout",
    "enemy_recently_disappeared": "Enemy recently disappeared",
    "health_recovers": "Health recovers",
    "ammo_recovers": "Ammo recovers",
}

INK = "#222222"
GREY = "#8A8A8A"
RED = "#C0392B"


def set_style() -> None:
    """Apply compact IEEE-style plotting defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",

        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",

        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "legend.fontsize": 8,
        "legend.frameon": False,

        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,

        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "grid.linestyle": ":",
        "axes.axisbelow": True,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame."""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {n}"
                ) from exc

    if not records:
        raise ValueError(f"No records found in {path}")

    return pd.DataFrame(records)


def flatten_live(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the nested decision object in the retained live-run log."""
    if "decision" not in df.columns:
        raise ValueError(
            "Live JSONL must contain a top-level 'decision' object."
        )

    decision = pd.json_normalize(
        df["decision"]
    ).add_prefix("decision.")

    return pd.concat(
        [
            df.drop(columns=["decision"]).reset_index(drop=True),
            decision.reset_index(drop=True),
        ],
        axis=1,
    )


def save_chart(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    fmt: str,
) -> None:
    """Save one figure in the requested format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{stem}.{fmt}"

    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)

    print(f"Saved -> {path}")


def figure_action_agreement(
    controlled: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    show_title: bool,
) -> None:

    required = {"scenario", "mode", "correct"}
    missing = required - set(controlled.columns)

    if missing:
        raise ValueError(
            f"Controlled JSONL missing columns: {sorted(missing)}"
        )

    pct = (
        controlled
        .groupby(["scenario", "mode"])["correct"]
        .mean()
        .unstack()
        .reindex(
            index=SCENARIO_ORDER,
            columns=MODE_ORDER,
        )
        * 100.0
    )

    overall = (
        controlled
        .groupby("mode")["correct"]
        .mean()
        .reindex(MODE_ORDER)
        * 100.0
    )

    if pct.isna().any().any():
        missing_cells = np.argwhere(pct.isna().to_numpy())
        raise ValueError(
            "Missing scenario/mode combinations in controlled results: "
            f"{missing_cells.tolist()}"
        )

    markers = {
        "current_only": "o",
        "raw_history": "s",
        "structured_temporal": "^",
    }

    offsets = {
        "current_only": -0.28,
        "raw_history": 0.0,
        "structured_temporal": 0.28,
    }

    nrow = len(SCENARIO_ORDER)

    fig, ax = plt.subplots(
        figsize=FIG_SIZE_POLICY
    )

    ax.grid(False)

    for g in range(0, nrow, 2):
        if (g // 2) % 2 == 0:
            ax.axhspan(
                g - 0.5,
                g + 1.5,
                color="#F1F1F1",
                lw=0,
                zorder=0,
            )

    ax.axhline(
        nrow - 0.5,
        color=INK,
        linewidth=0.6,
    )

    # 100% reference line.
    ax.axvline(
        100,
        color=GREY,
        linewidth=0.5,
        linestyle=":",
        zorder=1,
    )

    for i, scenario in enumerate(SCENARIO_ORDER):
        values = pct.loc[scenario]

        ax.plot(
            [values.min(), values.max()],
            [i, i],
            color="#B5B5B5",
            linewidth=1.0,
            zorder=1,
        )

        for mode in MODE_ORDER:
            value = float(values[mode])
            y = i + offsets[mode]

            ax.scatter(
                value,
                y,
                s=22,
                marker=markers[mode],
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidths=0.6,
                zorder=3,
            )

    overall_y = nrow + 0.15

    for mode in MODE_ORDER:
        value = float(overall[mode])
        y = overall_y + offsets[mode]

        ax.scatter(
            value,
            y,
            s=34,
            marker=markers[mode],
            color=MODE_COLORS[mode],
            edgecolor=INK,
            linewidths=0.6,
            zorder=3,
        )

    labels = (
        [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
        + ["Overall"]
    )

    ax.set_yticks(range(nrow + 1))
    ax.set_yticklabels(labels)
    ax.get_yticklabels()[-1].set_fontweight("bold")

    ax.set_ylim(nrow + 0.9, -0.6)
    ax.set_xlim(-6, 112)
    ax.set_xticks([0, 25, 50, 75, 100])

    ax.set_xlabel("Action agreement (%)")

    ax.grid(
        axis="x",
        visible=True,
    )

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.spines["left"].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            marker=markers[mode],
            color=MODE_COLORS[mode],
            linestyle="",
            markersize=6,
            markeredgecolor="white",
            label=MODE_LABELS[mode],
        )
        for mode in MODE_ORDER
    ]

    ax.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.0,
        handletextpad=0.3,
        fontsize=7.0,
    )

    if show_title:
        ax.set_title(
            "Action Agreement by Scenario",
            pad=22,
        )

    save_chart(
        fig,
        output_dir,
        "figure_4_1_action_agreement",
        fmt,
    )


def figure_controlled_latency(
    controlled: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    show_title: bool,
) -> None:

    required = {"mode", "latency_s"}
    missing = required - set(controlled.columns)

    if missing:
        raise ValueError(
            f"Controlled JSONL missing columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(
        figsize=FIG_SIZE_SINGLE
    )

    xmax = float(
        controlled["latency_s"].max()
    )

    for mode in MODE_ORDER:
        values = np.sort(
            controlled.loc[
                controlled["mode"] == mode,
                "latency_s",
            ].to_numpy(dtype=float)
        )

        if len(values) == 0:
            raise ValueError(
                f"No rows found for mode '{mode}'."
            )

        y = (
            np.arange(1, len(values) + 1)
            / len(values)
        )

        p95 = float(
            np.quantile(values, 0.95)
        )

        n_over_two = int(
            (values > 2.0).sum()
        )

        ax.step(
            values,
            y,
            where="post",
            color=MODE_COLORS[mode],
            linestyle=MODE_LS[mode],
            label=MODE_LABELS[mode],
        )

        print(
            f"{MODE_LABELS[mode]}: "
            f"median={np.median(values):.3f}s "
            f"P95={p95:.3f}s "
            f">2s={n_over_two}/{len(values)}"
        )

    ax.axvline(
        2.0,
        color=GREY,
        linestyle=":",
        linewidth=1.0,
    )

    ax.text(
        2.04,
        0.02,
        "2 s",
        color=GREY,
        fontsize=7.5,
        va="bottom",
    )

    ax.set_xlabel("Inference latency (s)")
    ax.set_ylabel("Cumulative fraction of calls")

    ax.set_xlim(
        0.7,
        xmax * 1.03,
    )
    ax.set_ylim(0, 1.02)

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.12),
        fontsize=7.5,
        handlelength=2.2,
        borderaxespad=0.2,
    )

    if show_title:
        ax.set_title(
            "Controlled Inference Latency"
        )

    save_chart(
        fig,
        output_dir,
        "figure_4_2_controlled_latency",
        fmt,
    )


def figure_live_motion(
    live: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    show_title: bool,
    threshold: float = 5.0,
) -> None:

    required = {
        "frame_id",
        "motion_diff",
        "stuck_override",
    }

    missing = required - set(live.columns)

    if missing:
        raise ValueError(
            f"Live JSONL missing columns: {sorted(missing)}"
        )

    live = (
        live
        .sort_values("frame_id")
        .copy()
    )

    live["stuck_override"] = (
        live["stuck_override"]
        .fillna(False)
        .astype(bool)
    )

    motion = live.dropna(
        subset=["motion_diff"]
    )

    triggered = motion[
        motion["stuck_override"]
    ]

    untriggered = motion[
        ~motion["stuck_override"]
    ]

    if motion.empty:
        raise ValueError(
            "No non-null motion_diff values found in live log."
        )

    fig, ax = plt.subplots(
        figsize=FIG_SIZE_WIDE
    )

    ymax = (
        float(motion["motion_diff"].max())
        * 1.12
    )

    ax.axhspan(
        0,
        threshold,
        color=RED,
        alpha=0.07,
        lw=0,
    )

    enemy_col = "decision.enemy_visible"

    if enemy_col in live.columns:
        flag = (
            live[enemy_col]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        frame_ids = (
            live["frame_id"]
            .to_numpy()
        )

        start = None

        for i, is_enemy in enumerate(flag):
            if is_enemy and start is None:
                start = frame_ids[i]

            if (
                (not is_enemy or i == len(flag) - 1)
                and start is not None
            ):
                end = (
                    frame_ids[i]
                    if is_enemy
                    else frame_ids[i - 1]
                )

                ax.axvspan(
                    start - 0.5,
                    end + 0.5,
                    color="#56B4E9",
                    alpha=0.18,
                    lw=0,
                )

                start = None

    series = (
        live
        .set_index("frame_id")["motion_diff"]
        .astype(float)
    )

    ax.plot(
        series.index,
        series.values,
        color=GREY,
        linewidth=0.9,
        zorder=2,
    )

    ax.scatter(
        untriggered["frame_id"],
        untriggered["motion_diff"],
        s=16,
        color="#0072B2",
        zorder=3,
    )

    ax.scatter(
        triggered["frame_id"],
        triggered["motion_diff"],
        s=48,
        marker="X",
        color=RED,
        edgecolor="white",
        linewidths=0.5,
        zorder=4,
    )

    ax.axhline(
        threshold,
        color=RED,
        linestyle="--",
        linewidth=1.0,
    )

    ax.text(
        live["frame_id"].min() - 0.8,
        threshold + 0.6,
        f"threshold = {threshold:g}",
        color=RED,
        fontsize=7.5,
        va="bottom",
        ha="left",
    )

    for _, row in triggered.iterrows():
        ax.annotate(
            f'{int(row["frame_id"])}',
            (
                row["frame_id"],
                row["motion_diff"],
            ),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=RED,
        )

    ax.set_xlim(
        live["frame_id"].min() - 1,
        live["frame_id"].max() + 1,
    )

    ax.set_ylim(
        0,
        ymax,
    )

    ax.set_xlabel(
        "Decision-cycle frame"
    )
    ax.set_ylabel(
        "Mean abs. pixel diff."
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    handles = [
        Line2D(
            [],
            [],
            color="#0072B2",
            marker="o",
            linestyle="",
            markersize=4,
            label="Motion difference",
        ),
        Line2D(
            [],
            [],
            color=RED,
            marker="X",
            linestyle="",
            markersize=6,
            label="ESCAPE substituted",
        ),
        Patch(
            facecolor="#56B4E9",
            alpha=0.3,
            label="VLM-reported enemy visible",
        ),
        Patch(
            facecolor=RED,
            alpha=0.12,
            label="Below threshold",
        ),
    ]

    ax.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=7.5,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    if show_title:
        ax.set_title(
            "Live Motion Difference and Recovery",
            pad=22,
        )

    save_chart(
        fig,
        output_dir,
        "figure_4_3_live_motion_recovery",
        fmt,
    )

    if not triggered.empty:
        print(
            "Triggered motion differences:",
            [
                round(float(v), 2)
                for v in triggered["motion_diff"]
            ],
        )

    if not untriggered.empty:
        print(
            "Untriggered range: "
            f"{untriggered['motion_diff'].min():.2f} - "
            f"{untriggered['motion_diff'].max():.2f}"
        )


ACTION_ORDER = ["MOVE", "ENGAGE", "ESCAPE", "SEARCH", "RESPAWN"]


def figure_live_actions(
    live: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    show_title: bool,
) -> None:

    required = {"decision.action", "executed_action"}
    missing = required - set(live.columns)

    if missing:
        raise ValueError(
            f"Live JSONL missing columns: {sorted(missing)}"
        )

    def counts(col: str) -> np.ndarray:
        s = live[col].astype(str).str.upper()
        return np.array(
            [(s == action).sum() for action in ACTION_ORDER]
        )

    selected = counts("decision.action")
    executed = counts("executed_action")

    x = np.arange(len(ACTION_ORDER))
    w = 0.38

    fig, ax = plt.subplots(
        figsize=FIG_SIZE_SINGLE
    )

    bars_sel = ax.bar(
        x - w / 2,
        selected,
        w,
        color=MODE_COLORS["current_only"],
        label="Selected by model",
    )

    bars_exe = ax.bar(
        x + w / 2,
        executed,
        w,
        color=MODE_COLORS["structured_temporal"],
        label="Executed",
    )

    for bars in (bars_sel, bars_exe):
        for bar in bars:
            ax.annotate(
                f"{int(bar.get_height())}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [action.capitalize() for action in ACTION_ORDER],
        rotation=30,
        ha="right",
    )

    ax.set_ylabel("Decision cycles")

    ymax = max(
        int(selected.max()),
        int(executed.max()),
    )
    ax.set_ylim(
        0,
        max(1, ymax * 1.15),
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.legend(
        loc="upper right"
    )

    if show_title:
        ax.set_title(
            "Selected vs. Executed Actions"
        )

    save_chart(
        fig,
        output_dir,
        "figure_4_4_live_actions",
        fmt,
    )


def print_controlled_summary(
    controlled: pd.DataFrame,
) -> None:

    print(
        "\nControlled experiment\n"
        + "-" * 60
    )

    for mode in MODE_ORDER:
        subset = controlled[
            controlled["mode"] == mode
        ]

        k = int(
            subset["correct"].sum()
        )
        n = len(subset)

        values = subset[
            "latency_s"
        ].to_numpy(dtype=float)

        print(
            f"{MODE_LABELS[mode]:<20} "
            f"{k}/{n} ({k / n:.0%}) "
            f"median={np.median(values):.2f}s "
            f"P95={np.quantile(values, 0.95):.2f}s"
        )


def print_live_summary(
    live: pd.DataFrame,
) -> None:

    print(
        "\nLive experiment\n"
        + "-" * 60
    )

    if "decision.action" in live.columns:
        print(
            "Model actions:",
            live[
                "decision.action"
            ].value_counts().to_dict(),
        )

    if "executed_action" in live.columns:
        print(
            "Executed:",
            live[
                "executed_action"
            ].value_counts().to_dict(),
        )

    if "decision.latency_s" in live.columns:
        values = live[
            "decision.latency_s"
        ].to_numpy(dtype=float)

        print(
            f"Latency mean={values.mean():.2f}s "
            f"median={np.median(values):.2f}s "
            f"min={values.min():.2f}s "
            f"max={values.max():.2f}s"
        )

    if (
        "decision.action" in live.columns
        and "decision.rule_action" in live.columns
    ):
        agreement = (
            live["decision.action"]
            == live["decision.rule_action"]
        )

        print(
            "rule_action agreement:",
            f"{int(agreement.sum())}/{len(agreement)}",
        )


def main(
    controlled_path: Path,
    live_path: Path,
    output_dir: Path,
    fmt: str,
    show_title: bool,
) -> None:

    set_style()

    controlled = load_jsonl(
        controlled_path
    )

    live = flatten_live(
        load_jsonl(live_path)
    )

    print_controlled_summary(
        controlled
    )

    print_live_summary(
        live
    )

    figure_action_agreement(
        controlled,
        output_dir,
        fmt,
        show_title,
    )

    figure_controlled_latency(
        controlled,
        output_dir,
        fmt,
        show_title,
    )

    figure_live_motion(
        live,
        output_dir,
        fmt,
        show_title,
        threshold=5.0,
    )

    figure_live_actions(
        live,
        output_dir,
        fmt,
        show_title,
    )

    print(
        "\nDone. Figures written to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate final capstone result figures."
        )
    )

    parser.add_argument(
        "--controlled",
        type=Path,
        required=True,
        help="Path to final_experiment.jsonl",
    )

    parser.add_argument(
        "--live",
        type=Path,
        required=True,
        help="Path to retained live.jsonl",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("figures"),
        help="Output directory.",
    )

    parser.add_argument(
        "--fmt",
        default="pdf",
        choices=[
            "pdf",
            "svg",
            "png",
            "eps",
        ],
        help="Output figure format (default: pdf).",
    )

    parser.add_argument(
        "--no-titles",
        action="store_true",
        help=(
            "Omit plot titles when captions in the report "
            "already provide the figure title."
        ),
    )

    args = parser.parse_args()

    main(
        args.controlled,
        args.live,
        args.output,
        args.fmt,
        not args.no_titles,
    )
