"""Final controlled temporal-context experiment.

Compares:
1. current_only
2. raw_history
3. structured_temporal

All other experimental conditions are held constant.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from game_state import EnemyDetection, GameState, WeaponType
from llm_decision import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    TEMPERATURE,
    get_decision,
)
from temporal_memory import TemporalMemory


MODES = [
    "current_only",
    "raw_history",
    "structured_temporal",
]

HISTORY_FRAMES = 5


def enemy(
    x: float = 0.5,
    y: float = 0.45,
    confidence: float = 0.9,
) -> EnemyDetection:

    return EnemyDetection(
        bbox=(600, 300, 700, 500),
        confidence=confidence,
        center_frac=(x, y),
    )


def make_state(
    frame_id: int,
    health: int,
    armor: int,
    ammo: int,
    enemy_visible: bool,
    enemy_x: float = 0.5,
) -> GameState:

    enemies = []

    if enemy_visible:
        enemies.append(
            enemy(x=enemy_x)
        )

    return GameState(
        frame_id=frame_id,
        timestamp=frame_id / 10.0,
        health=health,
        armor=armor,
        ammo=ammo,
        weapon=WeaponType.MACHINEGUN,
        enemies=enemies,
    )


def healthy_no_enemy():
    return [
        make_state(
            i,
            health=100,
            armor=50,
            ammo=30,
            enemy_visible=False,
        )
        for i in range(5)
    ]


def healthy_enemy():
    return [
        make_state(
            i,
            health=100,
            armor=50,
            ammo=30,
            enemy_visible=True,
        )
        for i in range(5)
    ]


def critical_health_enemy():
    return [
        make_state(
            i,
            health=15,
            armor=0,
            ammo=20,
            enemy_visible=True,
        )
        for i in range(5)
    ]


def critical_health_no_enemy():
    return [
        make_state(
            i,
            health=15,
            armor=0,
            ammo=20,
            enemy_visible=False,
        )
        for i in range(5)
    ]


def no_ammo_enemy():
    return [
        make_state(
            i,
            health=100,
            armor=50,
            ammo=0,
            enemy_visible=True,
        )
        for i in range(5)
    ]


def no_ammo_no_enemy():
    return [
        make_state(
            i,
            health=100,
            armor=50,
            ammo=0,
            enemy_visible=False,
        )
        for i in range(5)
    ]


def enemy_recently_disappeared():
    states = []

    # Enemy visible in first four frames.
    for i in range(4):
        states.append(
            make_state(
                i,
                health=100,
                armor=50,
                ammo=30,
                enemy_visible=True,
                enemy_x=0.42 + (0.02 * i),
            )
        )

    states.append(
        make_state(
            4,
            health=100,
            armor=50,
            ammo=30,
            enemy_visible=False,
        )
    )

    return states


def enemy_long_absent():
    return [
        make_state(
            i,
            health=100,
            armor=50,
            ammo=30,
            enemy_visible=False,
        )
        for i in range(5)
    ]


def health_recovers():
    health_values = [
        15,
        15,
        40,
        70,
        100,
    ]

    return [
        make_state(
            i,
            health=health,
            armor=50,
            ammo=30,
            enemy_visible=True,
        )
        for i, health in enumerate(health_values)
    ]


def ammo_recovers():
    ammo_values = [
        0,
        0,
        5,
        15,
        30,
    ]

    return [
        make_state(
            i,
            health=100,
            armor=50,
            ammo=ammo,
            enemy_visible=True,
        )
        for i, ammo in enumerate(ammo_values)
    ]


SCENARIOS = [
    (
        "healthy_no_enemy",
        healthy_no_enemy,
        "SEARCH",
    ),
    (
        "healthy_enemy",
        healthy_enemy,
        "ENGAGE",
    ),
    (
        "critical_health_enemy",
        critical_health_enemy,
        "RETREAT",
    ),
    (
        "critical_health_no_enemy",
        critical_health_no_enemy,
        "SEARCH",
    ),
    (
        "no_ammo_enemy",
        no_ammo_enemy,
        "SWITCH_WEAPON",
    ),
    (
        "no_ammo_no_enemy",
        no_ammo_no_enemy,
        "SWITCH_WEAPON",
    ),
    (
        "enemy_recently_disappeared",
        enemy_recently_disappeared,
        "TRACK",
    ),
    (
        "enemy_long_absent",
        enemy_long_absent,
        "SEARCH",
    ),
    (
        "health_recovers",
        health_recovers,
        "ENGAGE",
    ),
    (
        "ammo_recovers",
        ammo_recovers,
        "ENGAGE",
    ),
]


def build_memory(states) -> TemporalMemory:
    memory = TemporalMemory(
        window_size=HISTORY_FRAMES
    )

    for state in states:
        memory.push(state)

    return memory

def warm_up_model():
    print("Warming up Ollama model...")

    states = healthy_no_enemy()
    memory = build_memory(states)

    decision = get_decision(
        memory,
        model=DEFAULT_MODEL,
        mode="current_only",
        n_frames=HISTORY_FRAMES,
    )

    print(
        f"Warm-up complete "
        f"({decision.latency_s:.2f}s, "
        f"action={decision.action.value})"
    )


def run_experiment(
    repeats: int,
    out_path: Path,
    limit_scenarios: int | None = None,
):

    scenarios = SCENARIOS

    if limit_scenarios is not None:
        scenarios = scenarios[:limit_scenarios]

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    warm_up_model()

    results = []

    with out_path.open("w") as output_file:

        for mode in MODES:

            print("\n" + "=" * 70)
            print("MODE:", mode)
            print("=" * 70)

            for (
                scenario_name,
                state_fn,
                expected,
            ) in scenarios:

                actions = []

                for rep in range(repeats):
                    states = state_fn()
                    memory = build_memory(states)

                    decision = get_decision(
                        memory,
                        model=DEFAULT_MODEL,
                        mode=mode,
                        n_frames=HISTORY_FRAMES,
                    )

                    actual = decision.action.value
                    correct = actual == expected

                    actions.append(actual)

                    record = {
                        "input_states": [
                            state.to_prompt_dict()
                            for state in states
                        ],
                        "experiment_version": (
                            "final_temporal_comparison_v1"
                        ),
                        "prompt_version": PROMPT_VERSION,
                        "model": DEFAULT_MODEL,
                        "temperature": TEMPERATURE,
                        "history_frames": HISTORY_FRAMES,
                        "mode": mode,
                        "scenario": scenario_name,
                        "rep": rep,
                        "expected": expected,
                        "action": actual,
                        "correct": correct,
                        **asdict(decision),
                    }

                    record["action"] = actual

                    results.append(record)

                    output_file.write(
                        json.dumps(record) + "\n"
                    )

                    output_file.flush()

                counts = Counter(actions)

                correct_count = sum(
                    action == expected
                    for action in actions
                )

                print(
                    f"{scenario_name:30s} "
                    f"expected={expected:14s} "
                    f"correct={correct_count}/{repeats} "
                    f"actions={dict(counts)}"
                )

    print_summary(results)

    print(
        f"\nSaved {len(results)} decisions to:"
    )
    print(out_path)


def print_summary(results):

    print("\n")
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    by_mode = defaultdict(list)

    for result in results:
        by_mode[result["mode"]].append(result)

    for mode in MODES:

        mode_results = by_mode.get(mode, [])

        if not mode_results:
            continue

        total = len(mode_results)

        correct = sum(
            r["correct"]
            for r in mode_results
        )

        parsed = sum(
            r["parse_ok"]
            for r in mode_results
        )

        latencies = [
            r["latency_s"]
            for r in mode_results
        ]

        median_latency = statistics.median(
            latencies
        )

        scenario_actions = defaultdict(list)

        for r in mode_results:
            scenario_actions[
                r["scenario"]
            ].append(r["action"])

        consistency_scores = []

        for actions in scenario_actions.values():

            most_common_count = Counter(
                actions
            ).most_common(1)[0][1]

            consistency_scores.append(
                most_common_count / len(actions)
            )

        mean_consistency = statistics.mean(
            consistency_scores
        )

        print(f"\n{mode}")
        print(
            f"  policy agreement: "
            f"{correct}/{total} "
            f"({correct / total:.1%})"
        )
        print(
            f"  parse success: "
            f"{parsed}/{total} "
            f"({parsed / total:.1%})"
        )
        print(
            f"  mean action consistency: "
            f"{mean_consistency:.1%}"
        )
        print(
            f"  median latency: "
            f"{median_latency:.2f}s"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--out",
        default=(
            "results/final/"
            "final_experiment.jsonl"
        ),
    )

    args = parser.parse_args()

    run_experiment(
        repeats=args.repeats,
        out_path=Path(args.out),
        limit_scenarios=args.limit_scenarios,
    )


if __name__ == "__main__":
    main()