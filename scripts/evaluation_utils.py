"""Camera preprocessing, video recording, and logging for policy evaluation."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=json_default, sort_keys=True) + "\n")


def wrap_text(text, maximum_characters=36):
    words = text.split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join((*current, word))
        if current and len(candidate) > maximum_characters:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


class EvaluationVideoRecorder:
    """Write policy-view RGB frames with a readable target overlay."""

    def __init__(self, path, first_frame, label, task, episode, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = np.asarray(first_frame)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Unexpected video frame shape: {frame.shape}")
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.path), fourcc, float(fps), (width, height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.path}")
        self.label_lines = wrap_text(label)
        self.task = task
        self.episode = int(episode)

    def write(self, rgb_frame, step):
        frame = np.asarray(rgb_frame).copy()
        overlay_height = 28 + 19 * len(self.label_lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
        cv2.putText(
            frame,
            f"{self.task.upper()} | Episode {self.episode + 1} | Step {step}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for line_index, line in enumerate(self.label_lines):
            cv2.putText(
                frame,
                line,
                (8, 38 + line_index * 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self):
        self.close()


def image_tensor(image, image_size, device):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unexpected agentview image shape: {image.shape}")
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    image = torch.from_numpy(image).float().div_(255.0).unsqueeze(0).to(device)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(
            image,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return image


def policy_observation(
    observation, image_size, device, proprio_keys, camera_names
):
    camera_inputs = {
        "image": image_tensor(
            observation["agentview_image"], image_size, device
        )
    }
    if "robot0_eye_in_hand" in camera_names:
        camera_inputs["eye_in_hand_image"] = image_tensor(
            observation["robot0_eye_in_hand_image"], image_size, device
        )
    proprio_parts = [
        np.asarray(
            observation[
                key.removeprefix("obs__")
                if key.startswith("obs__")
                else key
            ],
            dtype=np.float32,
        ).reshape(-1)
        for key in proprio_keys
    ]
    proprio = torch.from_numpy(np.concatenate(proprio_parts))
    return camera_inputs, proprio.unsqueeze(0).to(device)


def compose_camera_frame(observation, camera_names):
    frames = []
    for camera_name in camera_names:
        frame = np.asarray(observation[f"{camera_name}_image"]).copy()
        cv2.putText(
            frame,
            camera_name,
            (8, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)
    return np.concatenate(frames, axis=1)


def entity_positions(env):
    return {
        "objects": {
            object_id: env.get_object_position(object_id)
            for object_id in env.object_ids
        }
    }
