import base64
import json
import time
from dataclasses import dataclass

import cv2
import requests

from config import (
    KEEP_ALIVE,
    MAX_IMAGE_SIDE,
    MODEL,
    NUM_CTX,
    OLLAMA_URL,
    REQUEST_TIMEOUT_S,
)


SYSTEM_PROMPT = """
You are the visual perception and high-level decision module for an OpenArena bot.

You receive ONE current gameplay screenshot.

Choose exactly one action:

ENGAGE
MOVE
SEARCH
RESPAWN

Decision policy, in strict priority order:

1. If the player appears dead or on a death/respawn screen:
   choose RESPAWN.

2. Otherwise, if an opponent/player character is visibly present:
   choose ENGAGE.
   Set enemy_horizontal to left, center, or right.
   Do not choose MOVE or SEARCH when an enemy is visibly present.

3. Otherwise, if there is a CLEAR navigable route:
   choose MOVE.
   Set action_direction to:
   - forward if the route directly ahead is clearly open;
   - left if the clearer route is left;
   - right if the clearer route is right.

4. Otherwise, if the route is blocked OR uncertain:
   choose SEARCH.
   SEARCH means rotate the camera to inspect the environment.
   It does NOT mean walk forward.
   Set action_direction to left or right.

Navigation rules:
- Only classify path_ahead="open" when traversable floor or space clearly
  extends forward from the player's position.
- If a wall, pillar, large vertical surface, or close geometry occupies
  the centre foreground, classify it as "blocked".
- If you cannot confidently tell whether forward movement is safe,
  classify it as "uncertain".
- When uncertain, choose SEARCH rather than MOVE.

Important:
- Do not invent enemies.
- Do not infer hidden enemies.
- A weapon model, HUD, corpse, decoration, statue, or wall texture
  is not an enemy.
- enemy_visible must describe a currently visible opponent character.
- Keep reasoning to one short sentence.

Return ONLY valid JSON with exactly these fields:

{
  "enemy_visible": false,
  "enemy_horizontal": "left|center|right|none",
  "path_ahead": "open|blocked|uncertain",
  "open_left": false,
  "open_right": false,
  "dead": false,
  "action": "ENGAGE|MOVE|SEARCH|RESPAWN",
  "action_direction": "left|center|right|forward|none",
  "reasoning": "short sentence"
}
""".strip()


@dataclass
class QwenDecision:
    enemy_visible: bool
    enemy_horizontal: str
    path_ahead: str
    open_left: bool
    open_right: bool
    dead: bool
    action: str
    action_direction: str
    rule_action: str
    reasoning: str
    latency_s: float
    parse_ok: bool
    raw_response: str


VALID_ACTIONS = {
    "ENGAGE",
    "MOVE",
    "SEARCH",
    "RESPAWN",
}

VALID_HORIZONTAL = {
    "left",
    "center",
    "right",
    "none",
}

VALID_PATH = {
    "open",
    "blocked",
    "uncertain",
}

VALID_DIRECTION = {
    "left",
    "center",
    "right",
    "forward",
    "none",
}


def _encode_frame(frame_rgb) -> str:
    h, w = frame_rgb.shape[:2]
    scale = MAX_IMAGE_SIDE / max(h, w)

    if scale < 1.0:
        frame_rgb = cv2.resize(
            frame_rgb,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    frame_bgr = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR,
    )

    ok, buffer = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
    )

    if not ok:
        raise RuntimeError(
            "Could not encode screenshot as JPEG."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


def _safe_choice(
    value,
    valid,
    fallback,
):
    value = str(value).strip().lower()

    if value in valid:
        return value

    return fallback


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "enemy_visible": {"type": "boolean"},
        "enemy_horizontal": {
            "type": "string",
            "enum": sorted(VALID_HORIZONTAL),
        },
        "path_ahead": {
            "type": "string",
            "enum": sorted(VALID_PATH),
        },
        "open_left": {"type": "boolean"},
        "open_right": {"type": "boolean"},
        "dead": {"type": "boolean"},
        "action": {
            "type": "string",
            "enum": sorted(VALID_ACTIONS),
        },
        "action_direction": {
            "type": "string",
            "enum": sorted(VALID_DIRECTION),
        },
        "reasoning": {"type": "string"},
    },
    "required": [
        "enemy_visible",
        "enemy_horizontal",
        "path_ahead",
        "open_left",
        "open_right",
        "dead",
        "action",
        "action_direction",
        "reasoning",
    ],
}


def _derive_action(
    enemy_visible,
    dead,
    path_ahead,
    open_left,
    open_right,
    enemy_horizontal,
    model_direction,
):
    if dead:
        return "RESPAWN", "none"

    if enemy_visible:
        return "ENGAGE", enemy_horizontal

    if path_ahead == "open":
        return "MOVE", "forward"

    if open_left and not open_right:
        return "SEARCH", "left"

    if open_right and not open_left:
        return "SEARCH", "right"

    if model_direction in {"left", "right"}:
        return "SEARCH", model_direction

    return "SEARCH", "right"


def analyse_frame(frame_rgb) -> QwenDecision:
    image_b64 = _encode_frame(frame_rgb)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": SYSTEM_PROMPT,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.0,
            "num_ctx": NUM_CTX,
        },
    }

    start = time.monotonic()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )

    response.raise_for_status()

    latency = time.monotonic() - start

    raw = response.json()["message"]["content"]

    try:
        parsed = json.loads(raw)

        enemy_visible = bool(
            parsed.get(
                "enemy_visible",
                False,
            )
        )

        dead = bool(
            parsed.get(
                "dead",
                False,
            )
        )

        enemy_horizontal = _safe_choice(
            parsed.get(
                "enemy_horizontal",
                "none",
            ),
            VALID_HORIZONTAL,
            "none",
        )

        path_ahead = _safe_choice(
            parsed.get(
                "path_ahead",
                "uncertain",
            ),
            VALID_PATH,
            "uncertain",
        )

        action = str(
            parsed.get(
                "action",
                "SEARCH",
            )
        ).strip().upper()

        if action not in VALID_ACTIONS:
            action = "SEARCH"

        direction = _safe_choice(
            parsed.get(
                "action_direction",
                "none",
            ),
            VALID_DIRECTION,
            "none",
        )

        if enemy_visible and enemy_horizontal == "none":
            enemy_horizontal = "center"

        open_left = bool(parsed.get("open_left", False))
        open_right = bool(parsed.get("open_right", False))

        rule_action, _ = _derive_action(
            enemy_visible,
            dead,
            path_ahead,
            open_left,
            open_right,
            enemy_horizontal,
            direction,
        )

        if action == "MOVE" and direction not in {
            "forward",
            "left",
            "right",
        }:
            direction = "forward"

        return QwenDecision(
            enemy_visible=enemy_visible,
            enemy_horizontal=enemy_horizontal,
            path_ahead=path_ahead,
            open_left=open_left,
            open_right=open_right,
            dead=dead,
            action=action,
            action_direction=direction,
            rule_action=rule_action,
            reasoning=str(
                parsed.get(
                    "reasoning",
                    "",
                )
            ),
            latency_s=latency,
            parse_ok=True,
            raw_response=raw,
        )

    except Exception:
        return QwenDecision(
            enemy_visible=False,
            enemy_horizontal="none",
            path_ahead="uncertain",
            open_left=False,
            open_right=False,
            dead=False,
            action="SEARCH",
            action_direction="right",
            rule_action="SEARCH",
            reasoning="Fallback after invalid model output.",
            latency_s=latency,
            parse_ok=False,
            raw_response=raw,
        )