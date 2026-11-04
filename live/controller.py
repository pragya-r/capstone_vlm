import time

from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button


SEARCH_TURN_S = 0.35

MOVE_FORWARD_S = 0.35
MOVE_TURN_S = 0.35


ESCAPE_TURN_S = 1.0

ENGAGE_TURN_S = 0.30
ENGAGE_FIRE_INTERVAL_S = 0.12
ENGAGE_BURST_S = 0.80

RESPAWN_COOLDOWN_S = 0.50


class GameController:
    def __init__(self):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()

        self.held_keys = {}

        self.current_action = None
        self.current_direction = "none"

        self.phase = 0
        self.next_phase_at = 0.0

        self.action_finished = True

        self.engage_until = 0.0
        self.next_fire_at = 0.0


    def toggle_pause(self):
        self.keyboard.press(Key.esc)
        self.keyboard.release(Key.esc)


    def update(self):
        now = time.monotonic()

        for key, release_at in list(self.held_keys.items()):
            if now >= release_at:
                try:
                    self.keyboard.release(key)
                finally:
                    del self.held_keys[key]

    def _hold(self, key, duration):
        if duration <= 0:
            return

        now = time.monotonic()

        if key not in self.held_keys:
            self.keyboard.press(key)

        self.held_keys[key] = now + duration

    def _release_all(self):
        for key in list(self.held_keys):
            try:
                self.keyboard.release(key)
            except Exception:
                pass

        self.held_keys.clear()

    def close(self):
        self._release_all()


    def set_action(self, action, direction="none"):
        self._release_all()

        self.current_action = str(action).upper()
        self.current_direction = str(direction).lower()

        self.phase = 0
        self.next_phase_at = 0.0

        self.action_finished = False

        self.engage_until = 0.0
        self.next_fire_at = 0.0

        print(
            f"    CONTROLLER: start "
            f"{self.current_action} "
            f"direction={self.current_direction}"
        )


    def execute(self):
        self.update()

        if self.action_finished:
            return

        if self.current_action == "SEARCH":
            self._search()

        elif self.current_action == "MOVE":
            self._move()

        elif self.current_action == "ENGAGE":
            self._engage()

        elif self.current_action == "RESPAWN":
            self._respawn()

        elif self.current_action == "ESCAPE":
            self._escape()

        else:
            self._finish()

    def _finish(self):
        self._release_all()

        print(
            f"    CONTROLLER: finished "
            f"{self.current_action}"
        )

        self.action_finished = True

    def _search(self):
        now = time.monotonic()

        if now < self.next_phase_at:
            return

        if self.phase == 0:
            direction = self.current_direction

            if direction not in {"left", "right"}:
                direction = "right"

            key = Key.left if direction == "left" else Key.right

            self._hold(
                key,
                SEARCH_TURN_S,
            )

            self.next_phase_at = now + SEARCH_TURN_S
            self.phase = 1
            return

        self._finish()

    def _move(self):
        now = time.monotonic()

        if now < self.next_phase_at:
            return


        if self.current_direction == "forward":
            if self.phase == 0:
                self._hold(
                    "w",
                    MOVE_FORWARD_S,
                )

                self.next_phase_at = now + MOVE_FORWARD_S
                self.phase = 1
                return

            self._finish()
            return


        if self.phase == 0:
            direction = self.current_direction

            if direction not in {"left", "right"}:
                direction = "right"

            key = Key.left if direction == "left" else Key.right

            self._hold(
                key,
                MOVE_TURN_S,
            )

            self.next_phase_at = now + MOVE_TURN_S
            self.phase = 1
            return

        if self.phase == 1:
            self._hold(
                "w",
                MOVE_FORWARD_S,
            )

            self.next_phase_at = now + MOVE_FORWARD_S
            self.phase = 2
            return

        self._finish()


    def _engage(self):
        now = time.monotonic()

        if self.phase == 0:
            if self.current_direction == "left":
                self._hold(
                    Key.left,
                    ENGAGE_TURN_S,
                )
                self.next_phase_at = now + ENGAGE_TURN_S

            elif self.current_direction == "right":
                self._hold(
                    Key.right,
                    ENGAGE_TURN_S,
                )
                self.next_phase_at = now + ENGAGE_TURN_S

            else:
                self.next_phase_at = now

            self.phase = 1
            return

        if now < self.next_phase_at:
            return

        if self.phase == 1:
            self.engage_until = now + ENGAGE_BURST_S
            self.next_fire_at = now
            self.phase = 2

        if self.phase == 2:
            if now >= self.engage_until:
                self._finish()
                return

            if now >= self.next_fire_at:
                self.mouse.click(
                    Button.left,
                    1,
                )

                self.next_fire_at = (
                    now + ENGAGE_FIRE_INTERVAL_S
                )


    def _respawn(self):
        now = time.monotonic()

        if self.phase == 0:
            self.mouse.click(
                Button.left,
                1,
            )

            self.next_phase_at = (
                now + RESPAWN_COOLDOWN_S
            )

            self.phase = 1
            return

        if now >= self.next_phase_at:
            self._finish()


    def _escape(self):
        now = time.monotonic()

        if now < self.next_phase_at:
            return

        if self.phase == 0:
            direction = self.current_direction

            if direction not in {"left", "right"}:
                direction = "right"

            key = Key.left if direction == "left" else Key.right

            self._hold(
                key,
                ESCAPE_TURN_S,
            )

            self.next_phase_at = now + ESCAPE_TURN_S
            self.phase = 1
            return

        if self.phase == 1:
            self._hold(
                "w",
                MOVE_FORWARD_S,
            )

            self.next_phase_at = now + MOVE_FORWARD_S
            self.phase = 2
            return

        self._finish()