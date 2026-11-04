"""
Generate the controlled-experiment summary table directly from final_experiment.jsonl.

Example
-------
python make_controlled_summary_table.py \
    --input results/final/final_experiment.jsonl \
    --output results/tables

Outputs
-------
controlled_summary_table.csv
controlled_summary_table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODE_ORDER = [
    "current_only",
    "raw_history",
    "structured_temporal",
]

MODE_LABELS = {
    "current_only": "Current-state-only",
    "raw_history": "Raw history",
    "structured_temporal": "Structured temporal",
}

HISTORY_REQUIRED_SCENARIO = "enemy_recently_disappeared"


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

    if not records:
        raise ValueError(f"No records found in {path}")

    return pd.DataFrame(records)


def validate(df: pd.DataFrame) -> None:
    required = {
        "mode",
        "scenario",
        "action",
        "expected",
        "correct",
        "latency_s",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input JSONL is missing required columns: {sorted(missing)}"
        )


def action_consistency(mode_df: pd.DataFrame) -> float:
    scenario_scores = []

    for _, group in mode_df.groupby("scenario"):
        counts = group["action"].astype(str).value_counts()

        if counts.empty:
            continue

        modal_count = int(counts.iloc[0])
        scenario_scores.append(modal_count / len(group))

    if not scenario_scores:
        return float("nan")

    return float(np.mean(scenario_scores))


def build_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_numeric = []
    rows_formatted = []

    for mode in MODE_ORDER:
        sub = df[df["mode"] == mode].copy()

        if sub.empty:
            raise ValueError(f"No rows found for mode: {mode}")

        all_n = len(sub)
        all_correct = int(sub["correct"].astype(bool).sum())
        agreement_all = all_correct / all_n

        restricted = sub[
            sub["scenario"] != HISTORY_REQUIRED_SCENARIO
        ].copy()

        restricted_n = len(restricted)
        restricted_correct = int(
            restricted["correct"].astype(bool).sum()
        )
        agreement_restricted = restricted_correct / restricted_n

        consistency = action_consistency(sub)

        latency = sub["latency_s"].astype(float).to_numpy()
        median_latency = float(np.median(latency))
        p95_latency = float(np.quantile(latency, 0.95))

        rows_numeric.append({
            "Configuration": MODE_LABELS[mode],
            "Agreement: all scenarios": agreement_all,
            "Agreement: excluding history-required scenario": agreement_restricted,
            "Mean action consistency": consistency,
            "Median latency (s)": median_latency,
            "95th-percentile latency (s)": p95_latency,
            "n_all": all_n,
            "n_restricted": restricted_n,
        })

        rows_formatted.append({
            "Configuration": MODE_LABELS[mode],
            f"Agreement: all scenarios, n = {all_n}":
                f"{agreement_all * 100:.1f}%",
            (
                "Agreement: excluding the history-required scenario, "
                f"n = {restricted_n}"
            ):
                f"{agreement_restricted * 100:.1f}%",
            "Mean action consistency":
                f"{consistency * 100:.0f}%",
            "Median latency (s)":
                f"{median_latency:.2f}",
            "95th-percentile latency (s)":
                f"{p95_latency:.2f}",
        })

    numeric = pd.DataFrame(rows_numeric)
    formatted = pd.DataFrame(rows_formatted)

    return numeric, formatted


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    path.write_text(
        df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def main(input_path: Path, output_dir: Path) -> None:
    df = load_jsonl(input_path)
    validate(df)

    numeric, formatted = build_summary(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "controlled_summary_table.csv"
    md_path = output_dir / "controlled_summary_table.md"

    formatted.to_csv(csv_path, index=False)
    write_markdown(formatted, md_path)

    print("\nControlled experiment summary\n")
    print(formatted.to_markdown(index=False))

    print("\nRaw calculated values")
    print("-" * 60)

    for _, row in numeric.iterrows():
        print(
            f"{row['Configuration']}: "
            f"all={row['Agreement: all scenarios']:.3f}, "
            f"restricted="
            f"{row['Agreement: excluding history-required scenario']:.3f}, "
            f"consistency={row['Mean action consistency']:.3f}, "
            f"median={row['Median latency (s)']:.6f}s, "
            f"P95={row['95th-percentile latency (s)']:.6f}s"
        )

    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate the controlled-experiment summary table "
            "from final_experiment.jsonl."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to final_experiment.jsonl",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tables"),
        help="Directory for generated table files.",
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output,
    )
