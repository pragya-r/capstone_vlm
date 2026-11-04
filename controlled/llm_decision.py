"""High-level OpenArena decision stage using a local Ollama LLM."""

import json
import time
from dataclasses import dataclass
from enum import Enum

import requests

from temporal_memory import TemporalMemory


OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_VERSION = "final_temporal_comparison_v1"
DEFAULT_MODEL = "llama3.2:3b"
TEMPERATURE = 0.2

VALID_MODES = {
    "current_only",
    "raw_history",
    "structured_temporal",
}


class Action(str, Enum):
    ENGAGE = "ENGAGE"
    RETREAT = "RETREAT"
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    SWITCH_WEAPON = "SWITCH_WEAPON"
    MOVE = "MOVE"


@dataclass
class Decision:
    action: Action
    reasoning: str
    raw_response: str
    latency_s: float
    parse_ok: bool


SYSTEM_PROMPT = """You are the high-level decision-making module for a bot in OpenArena.

You receive structured game-state information derived from gameplay.

Use the CURRENT state as the primary basis for the decision.
If temporal information is supplied, use it only as supporting context.

Choose exactly one action:
ENGAGE, RETREAT, SEARCH, TRACK, SWITCH_WEAPON, MOVE.

Decision module — apply these rules in order:

1. If ammunition is 0, choose SWITCH_WEAPON.
2. Otherwise, if an enemy is currently visible and health is below 25,
   choose RETREAT.
3. Otherwise, if an enemy is currently visible and health is 25 or higher,
   choose ENGAGE.
4. Otherwise, if no enemy is currently visible but an enemy was recently
   visible according to the supplied temporal information, choose TRACK.
5. Otherwise, choose SEARCH.

MOVE is reserved for cases where the supplied state explicitly requires
repositioning and none of the rules above applies.

Important:
- Use the CURRENT state as the primary source of health, ammunition and
  current enemy visibility.
- Historical states must not override newer values.
- Do not invent enemies or state values that are not present.

Respond with ONLY a JSON object:
{"action": "<ONE_OF_THE_ABOVE>", "reasoning": "<one sentence>"}
"""


def _structured_context(states) -> dict:
    """Derive temporal features using exactly the supplied state window."""

    current = states[-1]
    first = states[0]

    health_delta = None
    if first.health is not None and current.health is not None:
        health_delta = current.health - first.health

    ammo_delta = None
    if first.ammo is not None and current.ammo is not None:
        ammo_delta = current.ammo - first.ammo

    enemy_seen_in_prior_frames = any(
        len(state.enemies) > 0
        for state in states[:-1]
    )

    if current.enemies:
        frames_since_enemy_seen = 0
    else:
        frames_since_enemy_seen = None

        for offset, state in enumerate(
            reversed(states[:-1]),
            start=1,
        ):
            if state.enemies:
                frames_since_enemy_seen = offset
                break

    last_known_enemy = None

    for state in reversed(states):
        if state.enemies:
            enemy = state.enemies[0]

            last_known_enemy = {
                "track_id": enemy.track_id,
                "bearing_x": round(enemy.center_frac[0], 3),
                "bearing_y": round(enemy.center_frac[1], 3),
                "confidence": round(enemy.confidence, 2),
            }
            break

    enemy_persistence = 0
    enemy_bearing_delta = None

    if current.enemies:
        enemy = current.enemies[0]
        track_id = enemy.track_id

        for state in reversed(states):
            if track_id is not None:
                found = any(
                    candidate.track_id == track_id
                    for candidate in state.enemies
                )
            else:
                found = bool(state.enemies)

            if found:
                enemy_persistence += 1
            else:
                break

        if track_id is not None:
            positions = [
                (state.timestamp, candidate.center_frac)
                for state in states
                for candidate in state.enemies
                if candidate.track_id == track_id
            ]

            if len(positions) >= 2:
                (t0, p0) = positions[0]
                (t1, p1) = positions[-1]

                dt = t1 - t0

                if dt > 0:
                    enemy_bearing_delta = (
                        round((p1[0] - p0[0]) / dt, 3),
                        round((p1[1] - p0[1]) / dt, 3),
                    )

    return {
        "frames_observed": len(states),
        "health_delta": health_delta,
        "ammo_delta": ammo_delta,
        "enemy_seen_in_prior_frames": enemy_seen_in_prior_frames,
        "frames_since_enemy_seen": frames_since_enemy_seen,
        "enemy_persistence": enemy_persistence,
        "enemy_bearing_delta": enemy_bearing_delta,
        "last_known_enemy": last_known_enemy,
    }


def _build_prompt(
    memory: TemporalMemory,
    mode: str = "structured_temporal",
    n_frames: int = 5,
) -> str:

    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Expected one of {sorted(VALID_MODES)}"
        )

    states = list(memory.history)[-n_frames:]

    if not states:
        return "No game state available."

    current = states[-1]

    if mode == "current_only":
        prompt_data = {
            "current_state": current.to_prompt_dict(),
        }

    elif mode == "raw_history":
        prompt_data = {
            "current_state": current.to_prompt_dict(),
            "prior_states_oldest_to_newest": [
                state.to_prompt_dict()
                for state in states[:-1]
            ],
        }

    else:
        prompt_data = {
            "current_state": current.to_prompt_dict(),
            "temporal_context": _structured_context(states),
        }

    return (
        f"Input representation: {mode}\n"
        "Use the current state as the primary basis for the decision.\n\n"
        + json.dumps(prompt_data, indent=2)
    )


def _call_ollama(
    prompt: str,
    model: str,
) -> tuple[str, float]:

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": TEMPERATURE,
        },
    }

    start = time.monotonic()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    latency = time.monotonic() - start

    return response.json()["response"], latency


def get_decision(
    memory: TemporalMemory,
    model: str = DEFAULT_MODEL,
    mode: str = "structured_temporal",
    n_frames: int = 5,
) -> Decision:

    prompt = _build_prompt(
        memory,
        mode=mode,
        n_frames=n_frames,
    )

    raw, latency = _call_ollama(
        prompt,
        model,
    )

    try:
        parsed = json.loads(raw)

        action = Action(parsed["action"])
        reasoning = str(parsed.get("reasoning", ""))

        parse_ok = True

    except (json.JSONDecodeError, KeyError, ValueError):

        latest = (
            memory.history[-1]
            if memory.history
            else None
        )

        action = (
            Action.SEARCH
            if not (latest and latest.enemies)
            else Action.TRACK
        )

        reasoning = "fallback: malformed LLM output"
        parse_ok = False

    return Decision(
        action=action,
        reasoning=reasoning,
        raw_response=raw,
        latency_s=latency,
        parse_ok=parse_ok,
    )
