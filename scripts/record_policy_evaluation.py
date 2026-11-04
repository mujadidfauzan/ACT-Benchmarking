"""Record a selected seeded evaluation episode in native high resolution.

Policy RGB stays at checkpoint resolution. ACT dots visualize integrated,
clipped OSC translation commands, not a physics rollout or model XYZ output.
Episode indices are zero-based; all preceding episodes are evaluated normally.
"""

import argparse
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from scripts.evaluate_visual_bc import build_environment, evaluate_episode
from scripts.train_visual_bc import resolve_device

CAMERAS = ("agentview", "robot0_eye_in_hand", "frontview")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--episode", type=int, required=True, help="Zero-based evaluation episode index within the seed")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS),
                        choices=("agentview", "robot0_eye_in_hand", "frontview", "sideview", "birdview"),
                        help="Documentation cameras only; frontview shows the robot and table")
    parser.add_argument("--fps", type=float, default=20., help="Playback FPS; task control stays 20 Hz")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execution-steps", type=int)
    parser.add_argument("--temporal-ensemble", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ensemble-decay", type=float)
    parser.add_argument("--rnn-reset-interval", type=int, default=0, help="A/B/C only; BC-RNN uses its stored horizon")
    parser.add_argument("--trajectory-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-points", type=int, default=32, help="Maximum future offsets displayed")
    parser.add_argument("--encoder", choices=("auto", "ffmpeg", "opencv"), default="auto")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H264 quality, lower is higher quality")
    parser.add_argument("--reference-evaluation", type=Path,
                        help="Optional original evaluation folder: verifies seed/episode, horizon and initial scene")
    args = parser.parse_args()
    if args.episode < 0 or min(args.max_steps, args.width, args.height, args.trajectory_points) <= 0:
        parser.error("Episode must be nonnegative; sizes and step limits must be positive")
    if args.width % 2 or args.height % 2 or args.rnn_reset_interval < 0:
        parser.error("Video width/height must be even; RNN interval must be nonnegative")
    if len(set(args.cameras)) != len(args.cameras):
        parser.error("--cameras must not contain duplicates")
    if not math.isfinite(args.fps) or args.fps <= 0 or not 0 <= args.crf <= 51:
        parser.error("Need positive finite FPS and CRF in [0,51]")
    return args


def load_policy(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    kind = checkpoint["policy_type"]
    if kind in ("act", "act_lerobot"):
        if kind == "act_lerobot":
            from scripts.evaluate_act_lerobot import load_policy as load
        else:
            from scripts.evaluate_act import load_policy as load
        model, normalizer, checkpoint = load(args.checkpoint, device, execution_steps=args.execution_steps,
            temporal_ensemble=args.temporal_ensemble, ensemble_decay=args.ensemble_decay)
        reset = 0
    elif kind == "visual_bc":
        from scripts.evaluate_visual_bc import load_policy as load
        model, normalizer, checkpoint, _, _ = load(SimpleNamespace(checkpoint=args.checkpoint, normalization=None, image_size=None), device)
        reset = args.rnn_reset_interval
    elif kind == "robomimic_bc_rnn":
        from scripts.evaluate_robomimic_bc_rnn import load_policy as load
        from models.robomimic_bc_rnn.policy import RawPolicyInputs
        model, checkpoint = load(args.checkpoint, device)
        normalizer, reset = RawPolicyInputs(), model.sequence_length
    else:
        raise ValueError(f"Unsupported documentation policy: {kind}")
    if kind not in ("act", "act_lerobot") and any(v is not None for v in
            (args.execution_steps, args.temporal_ensemble, args.ensemble_decay)):
        raise ValueError("Chunk execution options require an ACT checkpoint")
    return model, normalizer, checkpoint, reset


class Video:
    def __init__(self, path, args):
        self.path, self.process, self.writer = path, None, None
        ffmpeg = shutil.which("ffmpeg")
        if args.encoder == "ffmpeg" and not ffmpeg:
            raise RuntimeError("ffmpeg is unavailable; use --encoder opencv or install ffmpeg")
        if ffmpeg and args.encoder != "opencv":
            command = [ffmpeg, "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
                "-threads", "2", "-filter_threads", "1",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self.encoder = "H264/libx264"
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                          (args.width, args.height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {path}")
            self.encoder = "OpenCV/mp4v"

    def write(self, rgb):
        if self.process:
            try:
                self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            except BrokenPipeError as error:
                message = self.process.stderr.read().decode(errors="replace")
                raise RuntimeError(f"Video encoding failed: {message}") from error
        else:
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer:
            self.writer.release()
        if self.process:
            self.process.stdin.close()
            message = self.process.stderr.read().decode(errors="replace")
            code = self.process.wait()
            self.process.stderr.close()
            if code:
                raise RuntimeError(f"Encoding {self.path} failed: {message}")


def integrated_command_path(controller, raw_actions, origin, action_spec):
    """Controller goal estimate, assuming each prior translation is achieved."""
    if controller.input_type != "delta" or controller.input_ref_frame != "world":
        raise ValueError("ACT command-path overlay requires world-frame OSC delta control")
    low, high = action_spec
    actions = np.clip(np.asarray(raw_actions), low, high)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Expected finite raw action chunk [K,7]")
    lo, hi = np.asarray(controller.input_min)[:3], np.asarray(controller.input_max)[:3]
    outlo, outhi = np.asarray(controller.output_min)[:3], np.asarray(controller.output_max)[:3]
    delta = (np.clip(actions[:, :3], lo, hi)-(hi+lo)/2) * (abs(outhi-outlo)/abs(hi-lo)) + (outhi+outlo)/2
    return np.concatenate((np.asarray(origin)[None], np.asarray(origin)[None]+np.cumsum(delta, axis=0)))


def project_points(points, transform, width, height):
    """World points to top-left-origin pixels; reject behind-camera points."""
    homogeneous = np.concatenate((np.asarray(points), np.ones((len(points), 1))), axis=1)
    projected = homogeneous @ transform.T
    depth = projected[:, 2]
    pixels = np.zeros((len(points), 2))
    np.divide(projected[:, :2], depth[:, None], out=pixels, where=abs(depth[:, None]) > 1e-9)
    visible = (depth > 1e-6) & np.isfinite(pixels).all(1)
    visible &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels, visible


def draw_path(frame, points, transform, *, predicted):
    pixels, visible = project_points(points, transform, frame.shape[1], frame.shape[0])
    previous = None
    for i, (pixel, valid) in enumerate(zip(pixels, visible)):
        if not valid:
            previous = None
            continue
        point = tuple(np.rint(pixel).astype(int))
        color = (255, int(210*(1-i/max(len(points)-1,1))), 60) if predicted else (70, 225, 255)
        if previous is not None:
            cv2.line(frame, previous, point, color, 2, cv2.LINE_AA)
        if predicted:
            cv2.circle(frame, point, 5 if i else 7, color, -1, cv2.LINE_AA)
        previous = point


class Recorder:
    def __init__(self, env, model, normalizer, args):
        self.env, self.model, self.normalizer, self.args = env, model, normalizer, args
        self.videos, self.trail = {}, []
        self.annotate = model.policy_type in ("act", "act_lerobot")
        self.observation, self.step = None, 0

    def capture(self, output=None, final=None):
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        eef = np.asarray(self.observation["robot0_eef_pos"])
        self.trail.append(eef.copy())
        raw_chunk, path, offset = None, None, None
        if self.model.policy_type in ("act", "act_lerobot") and output is not None:
            state = output.get("hidden")
            if state is not None and state.predictions:
                start, chunk = state.predictions[-1]
                offset = state.time-1-start
                with torch.inference_mode():
                    raw_chunk = self.normalizer.denormalize_action(chunk)[0].cpu().numpy()
                robot = self.env.robots[0]
                controller = robot.composite_controller.part_controllers[robot.arms[0]]
                remaining = raw_chunk[offset:offset+self.args.trajectory_points]
                path = integrated_command_path(controller, remaining, eef, self.env.action_spec)
        for camera in self.args.cameras:
            # MjSim.render returns bottom-up RGB; OpenCV image convention is
            # top-left. High-res renders never become model observations.
            frame = self.env.sim.render(width=self.args.width, height=self.args.height,
                                        camera_name=camera)[::-1].copy()
            if camera not in self.videos:
                self.videos[camera] = [Video(self.args.output_dir/f"{camera}.mp4",self.args)]
                if self.annotate:
                    self.videos[camera].append(Video(self.args.output_dir/f"{camera}_annotated.mp4",self.args))
            self.videos[camera][0].write(frame)
            if not self.annotate:
                continue
            annotated = frame.copy()
            transform = get_camera_transform_matrix(self.env.sim, camera, self.args.height, self.args.width)
            if self.args.trajectory_overlay:
                draw_path(annotated, np.asarray(self.trail), transform, predicted=False)
                if path is not None:
                    draw_path(annotated, path, transform, predicted=True)
            self.videos[camera][1].write(annotated)

    def close(self):
        errors = []
        for videos in self.videos.values():
            for video in videos:
                try:
                    video.close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]


class RecordingEnvironment:
    def __init__(self, env, recorder):
        self.env, self.recorder = env, recorder

    def __getattr__(self, key):
        return getattr(self.env, key)

    def reset(self):
        observation = self.env.reset()
        self.recorder.observation, self.recorder.step = observation, 0
        return observation

    def step(self, action):
        result = self.env.step(action)
        self.recorder.observation = result[0]
        self.recorder.step += 1
        return result


class RecordingPolicy:
    def __init__(self, model, recorder):
        self.model, self.recorder = model, recorder

    def __getattr__(self, key):
        return getattr(self.model, key)

    def step(self, *args, **kwargs):
        output = self.model.step(*args, **kwargs)
        self.recorder.capture(output)
        return output


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    reference = None
    if args.reference_evaluation:
        summary = json.loads((args.reference_evaluation/"summary.json").read_text())
        if summary["max_steps"] != args.max_steps:
            raise ValueError("--max-steps differs from reference evaluation")
        task = summary["tasks"][0]["task"]
        for line in (args.reference_evaluation/f"{task}_attempts.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["seed"] == args.seed and row["episode"] == args.episode:
                reference = row
        if reference is None:
            raise ValueError("Requested seed/episode not found in reference evaluation")
    device = resolve_device(args.device)
    model, normalizer, checkpoint, reset = load_policy(args,device)
    if args.reference_evaluation:
        for key, value in (("checkpoint_updates",checkpoint.get("updates")),
                           ("execution_steps",getattr(model,"execution_steps",None)),
                           ("temporal_ensemble",getattr(model,"temporal_ensemble",None)),
                           ("rnn_reset_interval",reset)):
            if key in summary and summary[key] != value:
                raise ValueError(f"{key} differs from reference evaluation")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    prepared = checkpoint["data_config"]
    env = build_environment(args.seed,max(v[0] for v in prepared["scene_counts"]),False,model,prepared,args.max_steps)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    recorder = None
    try:
        image_size = getattr(model,"observation_image_size",128)
        for episode in range(args.episode+1):
            selected = episode==args.episode
            if selected:
                recorder = Recorder(env,model,normalizer,args)
                active_env, active_model = RecordingEnvironment(env,recorder), RecordingPolicy(model,recorder)
            else:
                active_env, active_model = env, model
            result = evaluate_episode(active_env,active_model,normalizer,image_size,device,args.max_steps,20,
                False,episode,None,args.fps,prepared["fixed_target_color"],
                None,rnn_reset_interval=reset)
            print(f"{'Recording' if selected else 'Replaying preceding episode'} {episode}: "
                  f"success={result['success']} failure={result.get('failure_reason')}",flush=True)
        recorder.capture(final=result)
        if reference:
            for group in ("objects","receptacles"):
                for name,position in reference["initial_positions"][group].items():
                    if not np.allclose(position,result["initial_positions"][group][name],rtol=0,atol=1e-6):
                        raise ValueError("Initial scene differs from reference evaluation")
            print(f"Initial scene matches reference; original success={reference['success']}, "
                  f"recorded success={result['success']}",flush=True)
    finally:
        try:
            if recorder:
                recorder.close()
        finally:
            env.close()
    print(f"Documentation videos saved: {args.output_dir}")


if __name__ == "__main__":
    main()
