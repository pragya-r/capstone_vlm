"""TemporalMemory: sliding window over GameState history.

Responsibilities:
  - hold last N states
  - assign stable track_ids to enemy detections across frames (nearest-
    neighbour in bearing-space, no re-ID model)
  - derive simple temporal features (enemy persistence, closing/opening
    bearing) for inclusion in the LLM prompt
"""

from collections import deque

from game_state import EnemyDetection, GameState

MAX_MATCH_DIST = 0.08


class TemporalMemory:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.history: deque[GameState] = deque(maxlen=window_size)
        self._next_track_id = 0
        self._last_tracks: dict[int, tuple[float, float]] = {}

    def _assign_tracks(self, enemies: list[EnemyDetection]) -> None:
        unmatched = list(self._last_tracks.items())
        new_tracks: dict[int, tuple[float, float]] = {}

        for enemy in enemies:
            best_id, best_dist = None, MAX_MATCH_DIST
            for track_id, center in unmatched:
                dist = ((enemy.center_frac[0] - center[0]) ** 2
                        + (enemy.center_frac[1] - center[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = track_id, dist
            if best_id is not None:
                enemy.track_id = best_id
                unmatched = [(tid, c) for tid, c in unmatched if tid != best_id]
            else:
                enemy.track_id = self._next_track_id
                self._next_track_id += 1
            new_tracks[enemy.track_id] = enemy.center_frac

        self._last_tracks = new_tracks

    def push(self, state: GameState) -> None:
        self._assign_tracks(state.enemies)
        self.history.append(state)

    def enemy_persistence(self, track_id: int) -> int:
        count = 0
        for state in reversed(self.history):
            if any(e.track_id == track_id for e in state.enemies):
                count += 1
            else:
                break
        return count

    def bearing_delta(self, track_id: int) -> tuple[float, float] | None:
        positions = [
            (s.timestamp, e.center_frac)
            for s in self.history
            for e in s.enemies
            if e.track_id == track_id
        ]
        if len(positions) < 2:
            return None
        (t0, p0), (t1, p1) = positions[0], positions[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)

    def summary_for_prompt(self, n: int = 5) -> list[dict]:
        return [s.to_prompt_dict() for s in list(self.history)[-n:]]