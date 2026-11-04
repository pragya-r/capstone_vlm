import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from capture import ScreenCapture
from config import (
    GAME_TIMESCALE,
    PAUSE_TOGGLE_DELAY_S,
    POST_ACTION_SETTLE_S,
    SAVE_FRAME_STRIDE,
    START_DELAY_S,
    STUCK_DIFF_THRESHOLD,
)
from controller import GameController
from qwen_agent import analyse_frame


RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("runs") / RUN_ID
FRAME_DIR = RUN_DIR / "frames"
LOG_PATH = RUN_DIR / "live.jsonl"


def save_frame(
    frame,
    frame_id,
):
    FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        FRAME_DIR
        / f"frame_{frame_id:06d}.jpg"
    )

    Image.fromarray(frame).save(
        path,
        quality=85,
    )

    return path


def _signature(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY,
    )

    small = cv2.resize(
        gray,
        (96, 54),
        interpolation=cv2.INTER_AREA,
    )

    return small.astype(np.float32)


def main():
    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = ScreenCapture()
    controller = GameController()

    frame_id = 0

    prev_sig = None
    prev_moved = False
    escape_count = 0

    print(f"Run ID: {RUN_ID}")
    print(f"Log: {LOG_PATH}")
    print(f"Frames: {FRAME_DIR}")
    print(
        f"OpenArena timescale should be: "
        f"{GAME_TIMESCALE}"
    )
    print()
    print(
        "IMPORTANT: this version assumes Esc "
        "really pauses the local match."
    )
    print()
    print(
        f"Starting in {START_DELAY_S} seconds."
    )
    print("Switch to OpenArena now.")
    print(
        "Press Ctrl+C in the terminal to stop."
    )

    time.sleep(START_DELAY_S)

    try:
        with LOG_PATH.open(
            "w",
            encoding="utf-8",
        ) as log_file:

            while True:

                frame = capture.grab()
                capture_time = time.monotonic()

                sig = _signature(frame)
                motion_diff = None
                stuck = False

                if prev_sig is not None and prev_moved:
                    motion_diff = float(
                        np.mean(np.abs(sig - prev_sig))
                    )
                    stuck = (
                        motion_diff < STUCK_DIFF_THRESHOLD
                    )

                frame_path = None

                if frame_id % SAVE_FRAME_STRIDE == 0:
                    frame_path = str(
                        save_frame(
                            frame,
                            frame_id,
                        )
                    )

                print()
                print(
                    f"[{frame_id:05d}] "
                    f"captured gameplay frame"
                )

                controller.toggle_pause()

                time.sleep(
                    PAUSE_TOGGLE_DELAY_S
                )

                print(
                    "    GAME PAUSED -> "
                    "Qwen thinking"
                )


                decision = analyse_frame(
                    frame
                )

                print(
                    f"    QWEN -> "
                    f"{decision.action}"
                )

                print(
                    f"    enemy="
                    f"{decision.enemy_visible} "
                    f"enemy_pos="
                    f"{decision.enemy_horizontal} "
                    f"path="
                    f"{decision.path_ahead} "
                    f"dead="
                    f"{decision.dead}"
                )

                print(
                    f"    direction="
                    f"{decision.action_direction}"
                )

                print(
                    f"    reason="
                    f"{decision.reasoning}"
                )

                print(
                    f"    latency="
                    f"{decision.latency_s:.2f}s"
                )

                action = decision.action

                if action == "ENGAGE":
                    direction = (
                        decision.enemy_horizontal
                    )
                else:
                    direction = (
                        decision.action_direction
                    )

                stuck_override = False

                if motion_diff is not None:
                    print(
                        f"    motion_diff={motion_diff:.2f}"
                    )

                if stuck and action == "MOVE":
                    action = "ESCAPE"
                    stuck_override = True
                    escape_count += 1

                    if (
                        decision.open_left
                        and not decision.open_right
                    ):
                        direction = "left"
                    elif (
                        decision.open_right
                        and not decision.open_left
                    ):
                        direction = "right"
                    else:
                        direction = (
                            "left"
                            if escape_count % 2
                            else "right"
                        )

                    print(
                        f"    STUCK -> ESCAPE {direction}"
                    )

                controller.set_action(
                    action,
                    direction=direction,
                )


                controller.toggle_pause()

                time.sleep(
                    PAUSE_TOGGLE_DELAY_S
                )

                print(
                    "    GAME RESUMED -> "
                    "executing action"
                )

                action_start = time.monotonic()


                while (
                    not controller.action_finished
                ):
                    controller.execute()
                    time.sleep(0.01)

                action_duration = (
                    time.monotonic()
                    - action_start
                )

                print(
                    f"    ACTION COMPLETE "
                    f"({action_duration:.2f}s)"
                )

                record = {
                    "frame_id": frame_id,
                    "capture_time": capture_time,
                    "game_timescale": (
                        GAME_TIMESCALE
                    ),
                    "saved_frame": frame_path,
                    "decision": (
                        asdict(decision)
                    ),
                    "executed_action": action,
                    "stuck_override": stuck_override,
                    "motion_diff": motion_diff,
                    "controller_direction": (
                        direction
                    ),
                    "action_duration_s": (
                        action_duration
                    ),
                }

                log_file.write(
                    json.dumps(record)
                    + "\n"
                )

                log_file.flush()


                time.sleep(
                    POST_ACTION_SETTLE_S
                )

                prev_sig = sig
                prev_moved = action in {"MOVE", "ESCAPE"}

                frame_id += 1

    except KeyboardInterrupt:
        print()
        print("Stopping.")

    finally:
        controller.close()
        capture.close()

        print(
            f"Saved run to: {RUN_DIR}"
        )


if __name__ == "__main__":
    main()