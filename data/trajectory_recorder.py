import json
from collections import Counter
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class TrajectoryRecorder:
    """Capture observations and actions emitted through an environment step."""

    def __init__(self, env, phase_provider=None):
        self.env = env
        self.phase_provider = phase_provider
        self.observations = []
        self.actions = []
        self.phases = []
        self._original_step = None

    def start(self, initial_observation):
        if self._original_step is not None:
            raise RuntimeError("Recorder is already active")
        self.observations = [self._copy_observation(initial_observation)]
        self.actions = []
        self.phases = []
        self._original_step = self.env.step

        def recorded_step(action):
            phase = self._current_phase()
            result = self._original_step(action)
            observation = result[0]
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
            self.phases.append(phase)
            self.observations.append(self._copy_observation(observation))
            return result

        self.env.step = recorded_step

    def stop(self):
        if self._original_step is not None:
            self.env.step = self._original_step
            self._original_step = None

    def _copy_observation(self, observation):
        return {
            key: np.asarray(value).copy()
            for key, value in observation.items()
            if isinstance(value, (np.ndarray, np.generic, int, float, bool))
        }

    def _current_phase(self):
        if self.phase_provider is None:
            return "unlabeled"
        phase = self.phase_provider()
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase_provider must return a non-empty string")
        return phase

    def save(self, path, metadata, result_info):
        if self._original_step is not None:
            raise RuntimeError("Stop the recorder before saving")
        if len(self.observations) != len(self.actions) + 1:
            raise RuntimeError("Trajectory must contain T actions and T+1 observations")
        if len(self.phases) != len(self.actions):
            raise RuntimeError("Trajectory must contain one phase label per action")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        common_keys = set(self.observations[0])
        for observation in self.observations[1:]:
            common_keys.intersection_update(observation)

        payload = {
            "actions": np.stack(self.actions).astype(np.float32),
            "phases": np.asarray(self.phases, dtype=np.str_),
            "metadata_json": np.asarray(
                json.dumps(metadata, default=_json_default, sort_keys=True)
            ),
            "result_json": np.asarray(
                json.dumps(result_info, default=_json_default, sort_keys=True)
            ),
        }
        for key in sorted(common_keys):
            values = [observation[key] for observation in self.observations]
            try:
                payload[f"obs__{key}"] = np.stack(values)
            except ValueError:
                continue

        np.savez_compressed(path, **payload)
        return {
            "path": str(path),
            "steps": len(self.actions),
            "phase_counts": dict(sorted(Counter(self.phases).items())),
            "observation_keys": sorted(common_keys),
        }

    def __enter__(self):
        raise RuntimeError("Call start(initial_observation) explicitly")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
