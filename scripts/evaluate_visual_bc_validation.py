"""Compare visual BC and image BC-RNN on reconstructed validation scenes.

Writes videos, trajectory/joint PNGs, per-step JSONL/CSV/NPZ, and gripper
events. Demonstration actions never enter the policy. Reconstruction is not
an exact replay of the original physics: absent initial velocities are zero.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from data.bc_preparation import load_bc_training
from data.bc_preparation import save_json as save_plain_json
from data.episode_io import open_episode, read_json
from data.manipulation_dataset import read_split_manifest
from models.robomimic_bc_rnn.policy import RawPolicyInputs
from scripts.evaluate_robomimic_bc_rnn import load_policy as load_bcrnn_policy
from scripts.evaluate_visual_bc import build_environment, evaluate_episode
from scripts.evaluate_visual_bc import load_policy as load_visual_policy
from scripts.evaluation_utils import compose_camera_frame, json_default
from scripts.train_visual_bc import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-results",
        type=Path,
        help="Reuse completed models from a previous matching validation evaluation",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--episodes", type=int, help="First N split episodes; default all"
    )
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        help="Dataset episode IDs within this split",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Same environment reset sequence for every checkpoint",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--rnn-reset-interval",
        type=int,
        default=0,
        help="A/B/C GRU reset interval; D always uses its checkpoint sequence length",
    )
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=1e-4,
        help="Initial position tolerance, meters",
    )
    parser.add_argument(
        "--orientation-tolerance",
        type=float,
        default=1e-3,
        help="Initial quaternion angle tolerance, radians",
    )
    parser.add_argument(
        "--joint-tolerance",
        type=float,
        default=1e-5,
        help="Initial joint position tolerance, radians",
    )
    args = parser.parse_args()
    if args.max_steps <= 0 or (args.episodes is not None and args.episodes <= 0):
        parser.error("Episode and step limits must be positive")
    if args.rnn_reset_interval < 0 or any(
        not np.isfinite(v) or v <= 0
        for v in (
            args.video_fps,
            args.position_tolerance,
            args.orientation_tolerance,
            args.joint_tolerance,
        )
    ):
        parser.error("Invalid reset interval, FPS, or reconstruction tolerances")
    return args


def checkpoint_specs(values):
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path:
            raise ValueError(
                "Use --checkpoints A=path/to/best_rollout.pt B=path/to/best_rollout.pt"
            )
        if name in result:
            raise ValueError(f"Duplicate checkpoint name: {name}")
        result[name] = Path(path)
    return result


def save_json(path, value):
    save_plain_json(path, json.loads(json.dumps(value, default=json_default)))


def load_validation_policy(path, device, gru_reset_interval):
    # Our training checkpoints contain NumPy RNG state, not only tensors.
    # Match the existing evaluators' full-checkpoint loading for these files.
    probe = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    policy_type = probe.get("policy_type")
    del probe
    if policy_type == "visual_bc":
        args = SimpleNamespace(checkpoint=path, normalization=None, image_size=None)
        model, normalizer, checkpoint, image_size, color = load_visual_policy(
            args, device
        )
        return model, normalizer, checkpoint, image_size, color, gru_reset_interval
    if policy_type == "robomimic_bc_rnn":
        model, checkpoint = load_bcrnn_policy(path, device)
        # Pass source-size RGB to the model; resize/crop belongs to D's own
        # preprocessor. Robot state and actions remain raw controller values.
        return (
            model,
            RawPolicyInputs(),
            checkpoint,
            model.observation_image_size,
            checkpoint["data_config"]["fixed_target_color"],
            model.sequence_length,
        )
    if policy_type == "act":
        from scripts.evaluate_act import load_policy as load_act_policy

        model, normalizer, checkpoint = load_act_policy(path, device)
        return (model, normalizer, checkpoint, model.observation_image_size,
                checkpoint["data_config"]["fixed_target_color"], 0)
    raise ValueError(f"Unsupported policy_type: {policy_type}")


def gripper_mode(command):
    return "close" if command > 0 else "open" if command < 0 else "neutral"


def quaternion_error(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(
        2
        * np.arccos(
            np.clip(abs(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))), 0, 1)
        )
    )


def load_scene(dataset, entry):
    path = (dataset / entry["path"]).resolve()
    if not path.is_relative_to(dataset.resolve()):
        raise ValueError("Split episode path escapes dataset")
    with open_episode(path) as episode:
        metadata = read_json(episode["metadata_json"], "metadata_json")
        if (
            metadata.get("task") != "place"
            or len(metadata.get("objects", [])) != 1
            or len(metadata.get("receptacles", [])) != 1
        ):
            raise ValueError(
                "Validation reconstruction currently requires Place with one cube and one tray"
            )
        ids = [metadata["objects"][0]["id"], metadata["receptacles"][0]["id"]]
        required = [
            "robot0_joint_pos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        required += [
            f"{entity}_{suffix}" for entity in ids for suffix in ("pos", "quat")
        ]
        initial = {}
        for key in required + ["robot0_joint_vel", "robot0_gripper_qvel"]:
            stored = f"obs__{key}"
            if stored not in episode:
                if key in required:
                    raise ValueError(f"{path}: missing {stored}")
                continue
            value = np.asarray(episode[stored][0], dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: nonfinite {stored}")
            if key.endswith("quat") and (
                value.shape != (4,)
                or not np.isclose(np.linalg.norm(value), 1, atol=1e-4)
            ):
                raise ValueError(f"{path}: invalid quaternion {stored}")
            initial[key] = value
        if initial["robot0_joint_pos"].shape != (7,) or initial[
            "robot0_gripper_qpos"
        ].shape != (2,):
            raise ValueError("Expected seven Panda joints and two gripper joints")
        # Numeric expert paths are plot references only. No RGB sequence or
        # expert actions are supplied to the policy during simulator rollout.
        expert = {
            key: episode[f"obs__{key}"][:]
            for key in (
                "robot0_joint_pos",
                "robot0_eef_pos",
                "robot0_gripper_qpos",
                f"{ids[0]}_pos",
            )
        }
        frames = {
            key.removeprefix("obs__"): episode[key][0]
            for key in episode
            if key.endswith("_image") and key.startswith("obs__")
        }
    return dict(
        entry=entry,
        path=str(path),
        metadata=metadata,
        initial=initial,
        expert=expert,
        frames=frames,
    )


def state_hash(env):
    digest = hashlib.sha256()
    for value in (
        env.sim.data.qpos,
        env.sim.data.qvel,
        env.sim.data.ctrl,
        env.sim.data.act,
        np.asarray([env.sim.data.time]),
        env.sim.model.geom_rgba,
    ):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def restore_scene(env, scene, args):
    initial, metadata = scene["initial"], scene["metadata"]
    robot = env.robots[0]
    arm = robot.arms[0]
    env.sim.data.qvel[:] = 0
    env.sim.data.ctrl[:] = 0
    env.sim.data.act[:] = 0
    env.sim.data.qacc_warmstart[:] = 0
    robot.set_robot_joint_positions(initial["robot0_joint_pos"])
    robot.set_gripper_joint_positions(initial["robot0_gripper_qpos"], gripper_arm=arm)
    if "robot0_joint_vel" in initial:
        env.sim.data.qvel[robot._ref_joint_vel_indexes] = initial["robot0_joint_vel"]
    if "robot0_gripper_qvel" in initial:
        env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes[arm]] = initial[
            "robot0_gripper_qvel"
        ]
    for entity in env.objects + env.receptacles:
        pos, quat = initial[f"{entity.name}_pos"], initial[f"{entity.name}_quat"]
        env.sim.data.set_joint_qpos(entity.joints[0], np.r_[pos, quat[[3, 0, 1, 2]]])
        env.sim.data.set_joint_qvel(entity.joints[0], np.zeros(6))
    env.object_id_to_color = {e["id"]: e["color"] for e in metadata["objects"]}
    env.receptacle_id_to_color = {e["id"]: e["color"] for e in metadata["receptacles"]}
    env.object_color_to_id = {v: k for k, v in env.object_id_to_color.items()}
    env.receptacle_color_to_id = {v: k for k, v in env.receptacle_id_to_color.items()}
    env.target_object_id, env.target_receptacle_id = (
        metadata["target_object"],
        metadata["target_receptacle"],
    )
    env.cube_color, env.receptacle_color = (
        metadata["target_color"],
        metadata["target_receptacle_color"],
    )
    env.instruction_metadata = {
        key: metadata[key]
        for key in ("instruction", "instruction_split", "instruction_template_id")
        if key in metadata
    }
    env._apply_colors()
    env.sim.forward()
    # Recreate controller references after teleporting, so its first goal is
    # based on the reconstructed pose rather than the random reset pose.
    robot._load_controller()
    robot.composite_controller.update_state()
    robot.composite_controller.reset()
    robot.gripper[arm].current_action[:] = 0
    env._obs_cache.clear()
    for observable in env._observables.values():
        observable.reset()
    observation = env._get_observations(force_update=True)
    errors = {}
    for key, target in initial.items():
        if key not in observation:
            continue
        actual = observation[key]
        error = (
            quaternion_error(actual, target)
            if key.endswith("quat")
            else float(np.max(abs(actual - target)))
        )
        errors[key] = error
        tolerance = (
            args.orientation_tolerance
            if key.endswith("quat")
            else (
                args.joint_tolerance
                if key == "robot0_joint_pos"
                else args.position_tolerance
            )
        )
        if error > tolerance:
            raise ValueError(
                f"Initial reconstruction mismatch {key}: {error:.6g} > {tolerance:.6g}"
            )
    return observation, dict(
        errors=errors,
        state_sha256=state_hash(env),
        reconstruction="observed_initial_pose_with_recorded_robot_velocities_when_available",
        zeroed_unrecorded_velocities=True,
        exact_original_physics_replay=False,
    )


class ValidationEpisode:
    """Shared evaluator adapter: reconstruct on reset and capture each step."""

    def __init__(
        self, env, scene, args, expected_hash=None, video_path=None, name="policy"
    ):
        self.env, self.scene, self.args = env, scene, args
        self.expected_hash = expected_hash
        self.states, self.events, self.actions = [], [], []
        self.previous_gripper = -1.0
        self.success_streak = 0
        self.video_path, self.name, self.video = video_path, name, None

    def __getattr__(self, key):
        return getattr(self.env, key)

    def get_task_metadata(self):
        return {**self.scene["metadata"], "scene_origin": "reconstructed_dataset_split"}

    def reset(self):
        self.env.reset()
        observation, self.reconstruction = restore_scene(
            self.env, self.scene, self.args
        )
        if (
            self.expected_hash is not None
            and self.reconstruction["state_sha256"] != self.expected_hash
        ):
            raise ValueError(
                "Checkpoints do not start from identical reconstructed simulator states"
            )
        self.initial_objects = {
            key: self.env.get_object_position(key).copy() for key in self.env.object_ids
        }
        self.initial_observation = observation
        self.states.append(self.snapshot(observation, None))
        self.write_video(observation, None)
        return observation

    def write_video(self, observation, action):
        if self.video_path is None:
            return
        rgb = compose_camera_frame(observation, tuple(self.env.camera_names))
        bgr = cv2.cvtColor(
            cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_RGB2BGR,
        )
        frame = cv2.copyMakeBorder(
            bgr, 0, 96, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
        )
        if self.video is None:
            self.video = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.args.video_fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not self.video.isOpened():
                raise RuntimeError(f"Cannot open video writer: {self.video_path}")
        state = self.states[-1]
        command = (
            "initial"
            if action is None
            else f"{gripper_mode(action[6]).upper()} ({action[6]:+.3f})"
        )
        joints = state["joint_pos"]
        lines = [
            f'{self.name} | demo {self.scene["entry"]["episode"]} | step {state["step"]} | t={state["time_s"]:.2f}s',
            f'Grip command: {command} | two-finger contact: {state["two_finger_contact"]}',
            "J1-4 (rad): " + "  ".join(f"{v:+.3f}" for v in joints[:4]),
            "J5-7 (rad): " + "  ".join(f"{v:+.3f}" for v in joints[4:]),
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (8, bgr.shape[0] + 19 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        self.video.write(frame)

    def close_video(self):
        if self.video is not None:
            self.video.release()
            self.video = None

    def snapshot(self, obs, action):
        robot, sim = self.env.robots[0], self.env.sim
        arm = robot.arms[0]
        target = self.scene["metadata"]["target_object"]
        tray = self.scene["metadata"]["target_receptacle"]
        obj = next(o for o in self.env.objects if o.name == target)
        placement = self.env.placement_status(
            obs["robot0_eef_pos"], self.initial_objects
        )
        placement = {
            **placement,
            "gripper_open": bool(action is not None and action[6] < 0),
        }
        return dict(
            step=len(self.actions),
            time_s=float(sim.data.time),
            joint_pos=np.asarray(obs["robot0_joint_pos"]).copy(),
            joint_vel=np.asarray(sim.data.qvel[robot._ref_joint_vel_indexes]).copy(),
            gripper_qpos=np.asarray(obs["robot0_gripper_qpos"]).copy(),
            gripper_qvel=np.asarray(
                sim.data.qvel[robot._ref_gripper_joint_vel_indexes[arm]]
            ).copy(),
            eef_pos=np.asarray(obs["robot0_eef_pos"]).copy(),
            eef_quat=np.asarray(obs["robot0_eef_quat"]).copy(),
            cube_pos=self.env.get_object_position(target).copy(),
            cube_quat=self.env.get_object_orientation(target).copy(),
            tray_pos=self.env.get_receptacle_position(tray).copy(),
            tray_quat=self.env.get_receptacle_orientation(tray).copy(),
            two_finger_contact=bool(self.env._check_grasp(robot.gripper[arm], obj)),
            placement=placement,
            sim_qpos=sim.data.qpos.copy(),
            sim_qvel=sim.data.qvel.copy(),
            sim_ctrl=sim.data.ctrl.copy(),
            sim_act=sim.data.act.copy(),
        )

    def step(self, action):
        result = self.env.step(action)
        self.actions.append(np.asarray(action).copy())
        state = self.snapshot(result[0], action)
        before = self.states[-1]
        self.states.append(state)
        names = []
        if gripper_mode(action[6]) != gripper_mode(self.previous_gripper):
            names.append(f"{gripper_mode(action[6])}_command")
        self.previous_gripper = float(action[6])
        if state["two_finger_contact"] != before["two_finger_contact"]:
            names.append(
                "two_finger_contact_on"
                if state["two_finger_contact"]
                else "two_finger_contact_off"
            )
        z0 = self.states[0]["cube_pos"][2]
        if before["cube_pos"][2] - z0 < 0.02 <= state["cube_pos"][2] - z0:
            names.append("lift_above_initial_2cm")
        if (
            state["placement"]["inside_target"]
            and not before["placement"]["inside_target"]
        ):
            names.append("entered_target_tray")
        achieved = state["placement"]["success"] and state["placement"]["gripper_open"]
        self.success_streak = self.success_streak + 1 if achieved else 0
        if self.success_streak == 20:
            names.append("success_held_20_steps")
        for name in names:
            self.events.append(
                dict(
                    event=name,
                    step=state["step"],
                    command_time_s=before["time_s"],
                    observation_time_s=state["time_s"],
                    gripper_action=float(action[6]),
                    joint_pos_before=before["joint_pos"],
                    joint_pos_after=state["joint_pos"],
                    gripper_qpos_before=before["gripper_qpos"],
                    gripper_qpos_after=state["gripper_qpos"],
                    eef_pos_before=before["eef_pos"],
                    eef_pos_after=state["eef_pos"],
                    cube_pos_after=state["cube_pos"],
                    two_finger_contact=state["two_finger_contact"],
                )
            )
        self.write_video(result[0], action)
        return result


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")


def save_numeric(directory, episode):
    trajectory = directory / "trajectory.jsonl"
    policy = [
        json.loads(line) for line in trajectory.read_text().splitlines() if line.strip()
    ]
    policy[0].update(
        source_episode=episode.scene["entry"],
        reconstruction=episode.reconstruction,
        initial_state=episode.states[0],
        units={
            "joint_pos": "rad",
            "joint_vel": "rad/s",
            "gripper_qpos": "m",
            "positions": "m",
            "time": "simulation_seconds",
        },
    )
    policy[0]["policy_type"] = getattr(episode, "policy_type", "visual_bc")
    policy[0]["prediction_space"] = (
        "raw_controller"
        if policy[0]["policy_type"] == "robomimic_bc_rnn"
        else "normalized_motion_continuous_gripper"
        if policy[0]["policy_type"] == "act"
        else "normalized_motion_binary_gripper"
    )
    for i, row in enumerate(policy[1:], 1):
        row["robot_state_before"], row["robot_state_after"] = (
            episode.states[i - 1],
            episode.states[i],
        )
    write_jsonl(trajectory, policy)
    write_jsonl(directory / "gripper_events.jsonl", episode.events)
    data = {
        key: np.stack([s[key] for s in episode.states])
        for key in (
            "joint_pos",
            "joint_vel",
            "gripper_qpos",
            "gripper_qvel",
            "eef_pos",
            "eef_quat",
            "cube_pos",
            "cube_quat",
            "tray_pos",
            "tray_quat",
            "sim_qpos",
            "sim_qvel",
            "sim_ctrl",
            "sim_act",
        )
    }
    predictions = np.array([r["normalized_action"] for r in policy[1:]])
    data.update(
        time_s=np.array([s["time_s"] for s in episode.states]),
        actions=np.asarray(episode.actions),
        raw_actions=np.array([r["raw_action"] for r in policy[1:]]),
        predicted_actions=predictions,
        gripper_logits=np.array(
            [
                r["gripper_logit"] if r["gripper_logit"] is not None else np.nan
                for r in policy[1:]
            ],
            dtype=np.float64,
        ),
        two_finger_contact=np.array([s["two_finger_contact"] for s in episode.states]),
    )
    if policy[0]["policy_type"] in ("visual_bc", "act"):
        data["normalized_actions"] = predictions
    np.savez_compressed(directory / "trajectory.npz", **data)
    events_by_step = {}
    for event in episode.events:
        events_by_step.setdefault(event["step"], []).append(event["event"])
    rows = []
    for i, s in enumerate(episode.states):
        row = {
            "step": i,
            "time_s": s["time_s"],
            "events": ";".join(events_by_step.get(i, [])),
            "two_finger_contact": s["two_finger_contact"],
            **{
                k: s["placement"][k]
                for k in ("inside_target", "released", "gripper_open", "success")
            },
        }
        row["success_condition"] = row.pop("success") and row["gripper_open"]
        for key in (
            "joint_pos",
            "joint_vel",
            "gripper_qpos",
            "gripper_qvel",
            "eef_pos",
            "eef_quat",
            "cube_pos",
            "cube_quat",
            "tray_pos",
            "tray_quat",
        ):
            row.update({f"{key}_{j}": float(v) for j, v in enumerate(s[key])})
        row.update(
            {
                f"action_{j}": float(episode.actions[i - 1][j]) if i else None
                for j in range(7)
            }
        )
        row["gripper_logit"] = policy[i]["gripper_logit"] if i else None
        rows.append(row)
    with (directory / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return data


def plot_episode(directory, episode, data, result, name):
    import matplotlib.pyplot as plt

    t, eef, cube = data["time_s"], data["eef_pos"], data["cube_pos"]
    expert = episode.scene["expert"]
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    ax = fig.add_subplot(231, projection="3d")
    ax.plot(*eef.T, label="Policy EEF")
    ax.plot(*cube.T, label="Cube")
    ax.plot(*expert["robot0_eef_pos"].T, "--", alpha=0.5, label="Expert EEF")
    ax.scatter(*data["tray_pos"][0], marker="s", label="Initial tray")
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="World trajectories")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(232)
    ax.plot(eef[:, 0], eef[:, 1], label="Policy EEF")
    ax.plot(cube[:, 0], cube[:, 1], label="Cube")
    ax.plot(
        expert["robot0_eef_pos"][:, 0],
        expert["robot0_eef_pos"][:, 1],
        "--",
        alpha=0.5,
        label="Expert EEF",
    )
    ax.scatter(*data["tray_pos"][0, :2], marker="s", label="Initial tray")
    for event in episode.events:
        if event["event"] in ("close_command", "open_command"):
            pos = event["eef_pos_before"]
            ax.scatter(
                pos[0],
                pos[1],
                marker="x" if event["event"] == "close_command" else "o",
                c="black",
                s=35,
            )
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="XY; x=close, o=open command")
    ax.axis("equal")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(233)
    ax.plot(t, eef[:, 2], label="EEF Z")
    ax.plot(t, cube[:, 2], label="Cube Z")
    ax.set(xlabel="Time (s)", ylabel="Z (m)", title="Height")
    ax.legend(fontsize=8)
    for slot, key, title, unit in (
        (234, "joint_pos", "Arm joints", "rad"),
        (235, "joint_vel", "Joint velocities", "rad/s"),
    ):
        ax = fig.add_subplot(slot)
        for j in range(7):
            ax.plot(t, data[key][:, j], label=f"J{j + 1}", lw=0.9)
        ax.set(xlabel="Time (s)", ylabel=unit, title=title)
        ax.legend(ncol=4, fontsize=7)
    ax = fig.add_subplot(236)
    aperture = abs(data["gripper_qpos"][:, 0] - data["gripper_qpos"][:, 1])
    ax.plot(t, aperture, label="Finger separation", color="#3979b7")
    ax.fill_between(
        t,
        0,
        max(float(aperture.max()), 0.001),
        where=data["two_finger_contact"],
        alpha=0.15,
        color="green",
        label="Two-finger contact",
    )
    other = ax.twinx()
    other.step(
        t[1:],
        data["actions"][:, 6],
        where="post",
        color="#d65f3e",
        label="Gripper command",
    )
    other.set(ylim=(-1.2, 1.2), ylabel="Command: -1 open / +1 close")
    ax.set(xlabel="Time (s)", ylabel="Separation (m)", title="Gripper")
    ax.legend(loc="upper left", fontsize=7)
    for axis in fig.axes:
        if getattr(axis, "name", "") != "3d":
            axis.grid(alpha=0.2)
    fig.suptitle(
        f'{name} | demo {episode.scene["entry"]["episode"]} | success={result["success"]} | {result.get("failure_reason") or "complete"}'
    )
    fig.savefig(directory / "trajectory.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(7, 2, figsize=(12, 16), layout="constrained", sharex=True)
    expert_time = np.arange(len(expert["robot0_joint_pos"])) / episode.env.control_freq
    for j in range(7):
        axes[j, 0].plot(t, data["joint_pos"][:, j], label="Policy")
        axes[j, 0].plot(
            expert_time,
            expert["robot0_joint_pos"][:, j],
            "--",
            alpha=0.5,
            label="Expert",
        )
        axes[j, 1].plot(t, data["joint_vel"][:, j])
        axes[j, 0].set(ylabel=f"J{j + 1} (rad)")
        axes[j, 1].set(ylabel=f"dJ{j + 1} (rad/s)")
        for axis in axes[j]:
            for event in episode.events:
                if event["event"] in ("close_command", "open_command"):
                    axis.axvline(
                        event["command_time_s"],
                        color="red" if event["event"] == "close_command" else "green",
                        alpha=0.45,
                        lw=0.8,
                    )
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Joint positions; red=close, green=open command")
    axes[0, 1].set_title("Joint velocities")
    for axis in axes[-1]:
        axis.set_xlabel("Time (s); expert and policy are not phase-aligned")
    fig.savefig(directory / "joints.png", dpi=120)
    plt.close(fig)


def initial_picture(path, episode, cameras):
    demo = compose_camera_frame(episode.scene["frames"], cameras)
    actual = compose_camera_frame(episode.initial_observation, cameras)
    panels = []
    for label, rgb in (
        ("Recorded initial observation", demo),
        ("Reconstructed initial observation", actual),
    ):
        bgr = cv2.cvtColor(
            cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_RGB2BGR,
        )
        bgr = cv2.copyMakeBorder(
            bgr, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
        )
        cv2.putText(
            bgr,
            label,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(bgr)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=0)):
        raise RuntimeError(f"Could not write {path}")


def write_episode_summary(output, results, entries=None, model_names=None):
    """Readable, incremental outcome table; missing results stay PENDING."""
    names = list(results) if model_names is None else list(model_names)
    if entries is None:
        first = next(iter(results.values()))
        entries = [
            {"path": r["source_path"], "episode": r["source_episode"]} for r in first
        ]
    lookup = {
        name: {r["source_path"]: r for r in results.get(name, [])} for name in names
    }
    flat_rows, nested_rows, markdown_rows = [], [], []
    for entry in entries:
        flat = dict(episode=int(entry["episode"]), source_path=entry["path"])
        nested = {**flat, "models": {}}
        cells = [str(entry["episode"])]
        for name in names:
            r = lookup[name].get(entry["path"])
            status = (
                "PENDING" if r is None else "SUCCESS" if r["success"] else "FAILURE"
            )
            item = dict(
                status=status,
                success=r["success"] if r is not None else None,
                failure_reason=(r.get("failure_reason") if r is not None else None),
                steps=r["steps"] if r is not None else None,
            )
            nested["models"][name] = item
            flat.update({f"{name}_{key}": value for key, value in item.items()})
            text = (
                status
                if r is None
                else (
                    f'SUCCESS ({r["steps"]})'
                    if r["success"]
                    else f'FAIL: {r.get("failure_reason") or "unspecified"} ({r["steps"]})'
                )
            )
            cells.append(text.replace("|", "\\|"))
        flat_rows.append(flat)
        nested_rows.append(nested)
        markdown_rows.append("| " + " | ".join(cells) + " |")
    with (output / "episode_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    save_json(output / "episode_summary.json", nested_rows)
    lines = [
        "# Results per validation episode",
        "",
        "Cells show status / failure reason and executed step count.",
        "",
    ]
    for name in names:
        rows = results.get(name, [])
        lines.append(
            f'- {name}: {sum(r["success"] for r in rows)}/{len(rows)} completed episodes successful; '
            f"{len(entries) - len(rows)} pending."
        )
    lines += [
        "",
        "| Episode | " + " | ".join(names) + " |",
        "| " + " | ".join(["---"] * (len(names) + 1)) + " |",
    ]
    lines.extend(markdown_rows)
    (output / "episode_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    for name in names:
        folder = output / name
        folder.mkdir(exist_ok=True)
        with (folder / "episode_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode",
                    "source_path",
                    "status",
                    "success",
                    "failure_reason",
                    "steps",
                ],
            )
            writer.writeheader()
            writer.writerows(
                dict(
                    episode=r["episode"],
                    source_path=r["source_path"],
                    **r["models"][name],
                )
                for r in nested_rows
            )


def load_reused_results(root, entries, args, prepared):
    config = json.loads((root / "config.json").read_text())
    if any(
        config.get(key) != getattr(args, key) for key in ("split", "max_steps", "seed")
    ):
        raise ValueError(
            "Reused results must have the same split, max_steps, and reconstruction seed"
        )
    results, hashes, summaries = {}, {}, {}
    for path in sorted(root.glob("*/episodes.jsonl")):
        name = path.parent.name
        summary = json.loads(path.with_name("summary.json").read_text())
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        indexed = {r["source_path"]: r for r in rows}
        if len(indexed) != len(rows) or summary["attempted"] != len(rows):
            raise ValueError(f"{name}: incomplete or duplicated reused episode results")
        if (
            summary.get("scene_origin") != "reconstructed_dataset_split"
            or summary.get("success_hold_steps") != 20
        ):
            raise ValueError(
                f"{name}: incompatible reconstruction or success criterion"
            )
        selected = []
        for entry in entries:
            r = indexed.get(entry["path"])
            if (
                r is None
                or r["source_episode"] != int(entry["episode"])
                or r.get("split") != args.split
            ):
                raise ValueError(
                    f"{name}: selected split episode missing from reused results: {entry['path']}"
                )
            if (
                r["metadata"].get("collection_config")
                != prepared["environment_configs"]["place"]
            ):
                raise ValueError(f"{name}: reused collection configuration differs")
            digest = r["reconstruction"]["state_sha256"]
            if hashes.setdefault(entry["path"], digest) != digest:
                raise ValueError(
                    "Reused models have inconsistent initial simulator states"
                )
            selected.append({**r, "reused_from": str(path.parent)})
        results[name] = selected
        summaries[name] = {
            **summary,
            "attempted": len(selected),
            "successful": sum(r["success"] for r in selected),
            "success_rate": sum(r["success"] for r in selected) / len(selected),
            "failure_counts": dict(
                Counter(r["failure_reason"] for r in selected if not r["success"])
            ),
            "reused_from": str(path.parent),
        }
    if not results:
        raise ValueError("No completed model results found in --reuse-results")
    return results, hashes, summaries


def compare_results(output, results):
    import matplotlib.pyplot as plt

    names = list(results)
    paths = [r["source_path"] for r in results[names[0]]]
    table = []
    for i, path in enumerate(paths):
        row = {"source_path": path, "episode": results[names[0]][i]["source_episode"]}
        for name in names:
            r = results[name][i]
            if r["source_path"] != path:
                raise ValueError("Unpaired comparison")
            row.update(
                {
                    f"{name}_success": r["success"],
                    f"{name}_failure": r["failure_reason"],
                    f"{name}_steps": r["steps"],
                }
            )
        table.append(row)
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    rates = {
        name: sum(r["success"] for r in rr) / len(rr) for name, rr in results.items()
    }
    paired = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            paired[f"{a}_vs_{b}"] = dict(
                both_success=sum(
                    x["success"] and y["success"]
                    for x, y in zip(results[a], results[b])
                ),
                a_only=sum(
                    x["success"] and not y["success"]
                    for x, y in zip(results[a], results[b])
                ),
                b_only=sum(
                    y["success"] and not x["success"]
                    for x, y in zip(results[a], results[b])
                ),
                both_fail=sum(
                    not x["success"] and not y["success"]
                    for x, y in zip(results[a], results[b])
                ),
            )
    save_json(
        output / "comparison.json",
        dict(
            episodes=len(paths),
            success_rates=rates,
            paired_counts=paired,
            scene_origin="reconstructed_dataset_split",
            exact_original_physics_replay=False,
        ),
    )
    write_episode_summary(output, results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), layout="constrained")
    bars = axes[0].bar(names, [rates[n] * 100 for n in names])
    axes[0].bar_label(
        bars,
        labels=[f'{sum(r["success"] for r in results[n])}/{len(paths)}' for n in names],
        padding=4,
    )
    axes[0].set(
        ylim=(0, 110), ylabel="Success (%)", title="Reconstructed split rollout"
    )
    matrix = np.array([[int(r["success"]) for r in results[n]] for n in names]).T
    axes[1].imshow(
        matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest"
    )
    axes[1].set(
        xticks=range(len(names)),
        xticklabels=names,
        ylabel="Same source episode index",
        title="Green=success; red=failure",
    )
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    specs = checkpoint_specs(args.checkpoints)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Choose an empty --output-dir; previous evaluation artifacts are never overwritten"
        )
    # Fail before long rollouts if a checkpoint copy is truncated (notably C).
    for name, path in specs.items():
        if not path.is_file() or not zipfile.is_zipfile(path):
            raise ValueError(
                f"{name}: missing/incomplete checkpoint {path}; copy the complete PyTorch checkpoint first"
            )
    prepared, directory = load_bc_training(args.dataset, args.training_dir)
    for name, path in specs.items():
        probe = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if probe.get("policy_type") not in ("visual_bc", "robomimic_bc_rnn", "act"):
            raise ValueError(
                "This script accepts visual_bc, robomimic_bc_rnn, and ACT checkpoints"
            )
        if probe["data_config"] != prepared:
            raise ValueError(
                f"{name}: dataset/split/normalization differs from checkpoint"
            )
        del probe
    if prepared["tasks"] != ["place"] or prepared["scene_counts"] != [[1, 1]]:
        raise ValueError("Expected fixed-target Place with one cube and one tray")
    entries = read_split_manifest(directory / f"{args.split}.jsonl")
    if args.episode_ids:
        wanted = set(args.episode_ids)
        if len(wanted) != len(args.episode_ids) or not wanted <= {
            int(e["episode"]) for e in entries
        }:
            raise ValueError(
                "Episode IDs must be unique and belong to the selected split"
            )
        entries = [e for e in entries if int(e["episode"]) in wanted]
    if args.episodes is not None:
        if args.episodes > len(entries):
            raise ValueError("--episodes exceeds selected split size")
        entries = entries[: args.episodes]
    results, reference_hashes, reused_summaries = {}, {}, {}
    if args.reuse_results is not None:
        results, reference_hashes, reused_summaries = load_reused_results(
            args.reuse_results, entries, args, prepared
        )
        if set(results) & set(specs):
            raise ValueError("New checkpoint names must differ from reused model names")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(args.output_dir.resolve() / ".matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    device = resolve_device(args.device)
    for name, summary in reused_summaries.items():
        folder = args.output_dir / name
        folder.mkdir()
        save_json(folder / "summary.json", summary)
        write_jsonl(folder / "episodes.jsonl", results[name])
        print(
            f'Reusing {name}: {summary["successful"]}/{summary["attempted"]} successful',
            flush=True,
        )
    all_names = [*results, *specs]
    write_episode_summary(args.output_dir, results, entries, all_names)
    for name, path in specs.items():
        set_seed(args.seed)
        model, normalizer, checkpoint, image_size, color, reset_interval = (
            load_validation_policy(path, device, args.rnn_reset_interval)
        )
        policy_type = checkpoint["policy_type"]
        model_info = dict(
            policy_type=policy_type,
            checkpoint_epoch=(
                checkpoint["epoch"] if policy_type == "visual_bc" else None
            ),
            checkpoint_updates=checkpoint.get("updates"),
            rnn_reset_interval=reset_interval,
        )
        if policy_type == "act":
            model_info.update(chunk_size=model.chunk_size, execution_steps=model.execution_steps,
                              temporal_ensemble=model.temporal_ensemble,
                              ensemble_decay=model.ensemble_decay, latent_at_inference="zero")
        env = build_environment(
            args.seed, 1, args.render, model, prepared, args.max_steps
        )
        model_dir = args.output_dir / name
        model_dir.mkdir(exist_ok=True)
        results[name] = []
        try:
            (model_dir / "model.xml").write_text(env.model.get_xml(), encoding="utf-8")
            for index, entry in enumerate(entries):
                scene = load_scene(args.dataset, entry)
                folder = model_dir / f'episode_{int(entry["episode"]):06d}'
                folder.mkdir()
                episode = ValidationEpisode(
                    env,
                    scene,
                    args,
                    reference_hashes.get(entry["path"]),
                    folder / "rollout.mp4",
                    name,
                )
                episode.policy_type = policy_type
                try:
                    result = evaluate_episode(
                        episode,
                        model,
                        normalizer,
                        image_size,
                        device,
                        args.max_steps,
                        20,
                        args.render,
                        int(entry["episode"]),
                        None,
                        args.video_fps,
                        color,
                        folder / "trajectory.jsonl",
                        rnn_reset_interval=reset_interval,
                    )
                finally:
                    episode.close_video()
                reference_hashes.setdefault(
                    entry["path"], episode.reconstruction["state_sha256"]
                )
                data = save_numeric(folder, episode)
                initial_picture(
                    folder / "initial_comparison.png", episode, model.camera_names
                )
                plot_episode(folder, episode, data, result, name)
                result.update(
                    model=name,
                    checkpoint=str(path),
                    **model_info,
                    source_path=entry["path"],
                    source_episode=int(entry["episode"]),
                    split=args.split,
                    reconstruction=episode.reconstruction,
                    events=episode.events,
                    video_path=str(folder / "rollout.mp4"),
                )
                save_json(folder / "summary.json", result)
                results[name].append(result)
                write_jsonl(model_dir / "episodes.jsonl", results[name])
                write_episode_summary(args.output_dir, results, entries, all_names)
                print(
                    f'{name} {index + 1}/{len(entries)} demo={entry["episode"]}: success={result["success"]} failure={result["failure_reason"]}',
                    flush=True,
                )
        finally:
            env.close()
        rr = results[name]
        save_json(
            model_dir / "summary.json",
            dict(
                model=name,
                checkpoint=str(path),
                **model_info,
                split=args.split,
                attempted=len(rr),
                successful=sum(r["success"] for r in rr),
                success_rate=sum(r["success"] for r in rr) / len(rr),
                failure_counts=dict(
                    Counter(r["failure_reason"] for r in rr if not r["success"])
                ),
                max_steps=args.max_steps,
                success_hold_steps=20,
                scene_origin="reconstructed_dataset_split",
                exact_original_physics_replay=False,
            ),
        )
        del model, normalizer, checkpoint
    compare_results(args.output_dir, results)
    print(f"Comparison and all trajectories saved: {args.output_dir}")


if __name__ == "__main__":
    main()
