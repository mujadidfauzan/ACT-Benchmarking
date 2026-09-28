import json
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

    def __init__(self, env):
        self.env = env
        self.observations = []
        self.actions = []
        self._original_step = None

    def start(self, initial_observation):
        if self._original_step is not None:
            raise RuntimeError("Recorder is already active")
        self.observations = [self._copy_observation(initial_observation)]
        self.actions = []
        self._original_step = self.env.step

        def recorded_step(action):
            result = self._original_step(action)
            observation = result[0]
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
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

    def save(self, path, metadata, result_info):
        if self._original_step is not None:
            raise RuntimeError("Stop the recorder before saving")
        if len(self.observations) != len(self.actions) + 1:
            raise RuntimeError("Trajectory must contain T actions and T+1 observations")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        common_keys = set(self.observations[0])
        for observation in self.observations[1:]:
            common_keys.intersection_update(observation)

        payload = {
            "actions": np.stack(self.actions).astype(np.float32),
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
            "observation_keys": sorted(common_keys),
        }

    def __enter__(self):
        raise RuntimeError("Call start(initial_observation) explicitly")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
