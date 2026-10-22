"""Prepare an episode-level 90/10 split and BC statistics after collection."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from data.episode_io import HDF5_SUFFIXES, open_episode, read_json
from environments.pick_config import validate_pick_settings


STATE_KEYS = ("obs__robot0_joint_pos", "obs__robot0_eef_pos", "obs__robot0_gripper_qpos")
PROPRIO_KEYS = STATE_KEYS + ("obs__robot0_eef_quat",)
CAMERA_KEYS = {"agentview": "obs__agentview_image", "robot0_eye_in_hand": "obs__robot0_eye_in_hand_image"}


def default_training_dir(dataset):
    return Path("data/splits") / f"{Path(dataset).name}_bc"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def suspicious_camera_frames(images):
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("Camera frames must be uint8 [T, H, W, 3]")
    if min(images.shape[1:3]) < 2:
        raise ValueError("Camera images are too small")
    change = np.abs(np.diff(images.astype(np.int16), axis=2)).mean(axis=(1, 2, 3))
    black = (images == 0).mean(axis=(1, 2, 3))
    return np.flatnonzero((change > 20) | (black > .95))


def dataset_entries(dataset):
    dataset = Path(dataset)
    entries = []
    for task in ("pick", "place", "stack"):
        directory = dataset / task
        if list(directory.glob("episode_*.npz")):
            raise ValueError(f"Legacy NPZ demos in {directory}; convert with python -m data.episode_io into a new dataset")
        disk_names = {path.name for path in directory.glob("episode_*") if path.suffix in HDF5_SUFFIXES}
        manifest = directory / "manifest.jsonl"
        if not manifest.exists():
            if disk_names:
                raise ValueError(f"Unmanifested episodes in {directory}")
            continue
        names = set()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            name = value["path"]
            if Path(name).name != name or Path(name).suffix not in HDF5_SUFFIXES or name in names:
                raise ValueError(f"Invalid/duplicate episode path in {manifest}: {name}")
            names.add(name)
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append({**value, "task": task, "path": f"{task}/{name}"})
        orphans = disk_names - names
        if orphans:
            raise ValueError(f"Unmanifested episodes in {directory}: {sorted(orphans)}")
    return sorted(entries, key=lambda value: value["path"])


def source_digest(dataset, entries):
    values = []
    for entry in entries:
        digest = hashlib.sha256()
        with (Path(dataset) / entry["path"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        values.append({**entry, "sha256": digest.hexdigest()})
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def write_split(path, entries):
    content = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries).encode("utf-8")
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def prepare_bc_training(dataset, output=None, *, seed=123, val_fraction=.1):
    dataset = Path(dataset)
    output = default_training_dir(dataset) if output is None else Path(output)
    entries = dataset_entries(dataset)
    if len(entries) < 2:
        raise ValueError("At least two successful demonstrations are needed for train/validation")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    if n_val >= len(shuffled):
        raise ValueError("Validation fraction leaves no training episodes")
    val, train = shuffled[:n_val], shuffled[n_val:]
    signatures = set()
    orientation = set()
    environment_configs = {}
    place_colors = set()
    for entry in entries:
        with open_episode(dataset / entry["path"]) as archive:
            if entry["task"] == "place":
                metadata = entry.get("metadata", {})
                color = metadata.get("fixed_target_color")
                objects = [v for v in metadata.get("objects", []) if v["id"] == metadata.get("target_object")]
                trays = [v for v in metadata.get("receptacles", []) if v["id"] == metadata.get("target_receptacle")]
                if color is None or len(objects) != 1 or len(trays) != 1 or objects[0]["color"] != color or trays[0]["color"] != color:
                    raise ValueError("Place BC requires a fixed matching-color object/receptacle pair")
                stored = read_json(archive["metadata_json"], "metadata_json")
                if stored != metadata:
                    raise ValueError("Place metadata differs between manifest and episode")
                if sum(v["color"] == color for v in metadata["objects"]) != 1 or sum(v["color"] == color for v in metadata["receptacles"]) != 1:
                    raise ValueError("Ambiguous fixed-color Place targets")
                place_colors.add(color)
            actions = archive["actions"][:]
            if actions.shape != (entry["steps"], 7) or not len(actions) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions: {entry['path']}")
            names = tuple(name for name, key in CAMERA_KEYS.items() if key in archive)
            if not names or names[0] != "agentview":
                raise ValueError(f"Missing agentview RGB in {entry['path']}")
            sizes = []
            for name in names:
                frames = archive[CAMERA_KEYS[name]]
                if not frames.ndim or len(frames) != len(actions) + 1:
                    raise ValueError(f"Invalid/corrupted {name} frames in {entry['path']}")
                for start in range(0, len(frames), 32):
                    if len(suspicious_camera_frames(frames[start:start + 32])):
                        raise ValueError(f"Invalid/corrupted {name} frames in {entry['path']}")
                sizes.append(tuple(frames.shape[1:3]))
            signatures.add((names, tuple(sizes)))
            declared = entry.get("metadata", {}).get("control_orientation")
            control_orientation = bool(declared) if declared is not None else bool(np.any(np.abs(actions[:, 3:6]) > 1e-8))
            mode = entry.get("metadata", {}).get("rotation_mode", "align_yaw" if control_orientation else "none")
            if mode not in ("none", "align_yaw") or (mode != "none") != control_orientation:
                raise ValueError(f"Inconsistent rotation mode in {entry['path']}")
            if not control_orientation and np.any(actions[:, 3:6] != 0):
                raise ValueError(f"Nonzero rotational action in rotation-mode none: {entry['path']}")
            orientation.add(control_orientation)
            contract = entry.get("metadata", {}).get("collection_config", {})
            if contract.get("rotation_mode", mode) != mode:
                raise ValueError(f"Collection rotation mode differs from episode metadata: {entry['path']}")
            if contract.get("yaw_range_deg") is not None:
                validate_pick_settings(mode, contract["yaw_range_deg"])
                stored_yaw = contract.get("environment_kwargs", {}).get("yaw_range")
                if stored_yaw is not None and not np.allclose(stored_yaw, np.deg2rad(contract["yaw_range_deg"]), rtol=0, atol=1e-10):
                    raise ValueError(f"Yaw metadata differs from environment settings: {entry['path']}")
            previous = environment_configs.setdefault(entry["task"], contract)
            if previous != contract:
                raise ValueError("Dataset mixes collection/controller settings")
            state_keys = PROPRIO_KEYS
            for key in state_keys:
                value = archive[key]
                if value.ndim != 2 or len(value) != len(actions) + 1 or not np.isfinite(value[:]).all():
                    raise ValueError(f"Invalid state observations for {key} in {entry['path']}")
    if len(signatures) != 1 or len(orientation) != 1:
        raise ValueError("Dataset mixes camera/resolution/orientation configurations; use separate datasets")
    if len(place_colors) > 1:
        raise ValueError("Dataset mixes fixed Place target colors")
    camera_names, camera_sizes = signatures.pop()
    if len(set(camera_sizes)) != 1 or camera_sizes[0][0] != camera_sizes[0][1]:
        raise ValueError("Demo collection expects square images of equal size for every camera")
    predict_orientation = orientation.pop()
    keys = PROPRIO_KEYS
    state_sum = state_sq = motion_sum = motion_sq = None
    count = 0
    for entry in train:
        with open_episode(dataset / entry["path"]) as archive:
            steps = int(entry["steps"])
            parts = []
            for key in keys:
                value = archive[key]
                if value.ndim != 2 or len(value) != steps + 1:
                    raise ValueError(f"Invalid state alignment for {key} in {entry['path']}")
                parts.append(value[:steps])
            state = np.concatenate(parts, axis=1).astype(np.float64)
            motion = archive["actions"][:, :6 if predict_orientation else 3].astype(np.float64)
            if not np.isfinite(state).all():
                raise ValueError(f"Nonfinite state in {entry['path']}")
            if state_sum is None:
                state_sum = np.zeros(state.shape[1]); state_sq = state_sum.copy()
                motion_sum = np.zeros(motion.shape[1]); motion_sq = motion_sum.copy()
            state_sum += state.sum(0)
            state_sq += (state ** 2).sum(0)
            motion_sum += motion.sum(0)
            motion_sq += (motion ** 2).sum(0)
            count += steps
    state_mean, motion_mean = state_sum / count, motion_sum / count
    state_std = np.maximum(np.sqrt(np.maximum(state_sq / count - state_mean ** 2, 0)), 1e-3)
    motion_std = np.maximum(np.sqrt(np.maximum(motion_sq / count - motion_mean ** 2, 0)), 1e-2)
    # Gripper stays binary; unused rotation is exactly zero in position-only mode.
    action_mean, action_std = np.zeros(7), np.ones(7)
    action_mean[:len(motion_mean)], action_std[:len(motion_std)] = motion_mean, motion_std
    output.mkdir(parents=True, exist_ok=True)
    train_hash = write_split(output / "train.jsonl", train)
    val_hash = write_split(output / "val.jsonl", val)
    statistics = {"normalization_type": "visual_bc", "split_sha256": train_hash,
                  "proprio": {"keys": list(keys), "mean": state_mean.tolist(), "std": state_std.tolist()},
                  "action": {"mean": action_mean.tolist(), "std": action_std.tolist()},
                  "training_timesteps": count}
    save_json(output / "normalization.json", statistics)
    target_colors = {e.get("metadata", {}).get("target_color") for e in entries if e["task"] == "pick"} | place_colors
    config = {"format_version": 1, "method": "visual_bc", "storage_format": "hdf5", "seed": seed,
              "rotation_mode": "align_yaw" if predict_orientation else "none",
              "val_fraction": val_fraction, "source_digest": source_digest(dataset, entries),
              "split_hashes": {"train": train_hash, "val": val_hash},
              "camera_names": list(camera_names), "camera_sizes": [list(size) for size in camera_sizes],
              "image_size": 128, "image_convention": "opencv", "predict_orientation": predict_orientation,
              "proprio_keys": list(keys), "proprio_dim": len(state_mean), "normalization": statistics,
              "tasks": sorted({e["task"] for e in entries}),
              "environment_configs": environment_configs,
              "fixed_target_color": next(iter(target_colors)) if len(target_colors) == 1 else None,
              "train_episodes": len(train), "val_episodes": len(val),
              "scene_counts": sorted({(int(e.get("num_objects", 3)), int(e.get("num_receptacles") or 0)) for e in entries})}
    save_json(output / "training.json", config)
    return config


def load_bc_training(dataset, output=None):
    output = default_training_dir(dataset) if output is None else Path(output)
    path = output / "training.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing automatic training artifacts: {path}; collect demonstrations first")
    config = json.loads(path.read_text(encoding="utf-8"))
    if json.loads((output / "normalization.json").read_text(encoding="utf-8")) != config["normalization"]:
        raise ValueError("Prepared normalization file changed")
    if config["source_digest"] != source_digest(dataset, dataset_entries(dataset)):
        raise ValueError("Dataset changed after preparation; finish/re-run collection to refresh training artifacts")
    for split in ("train", "val"):
        if hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest() != config["split_hashes"][split]:
            raise ValueError(f"Prepared {split} split changed")
    return config, output
