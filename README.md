# SimMT-ACT

Panda manipulation in MuJoCo/robosuite: scripted demonstrations, behavior cloning,
and evaluation for Pick, Pick-and-Place, and Stack with variable object counts.

## Current BC Workflow

The current model maps RGB and robot proprioception to one action per timestep:

1. Collect successful demonstrations and reject corrupted camera frames.
2. Automatically prepare a seeded 90/10 episode split, train-only statistics,
   and camera/state/controller configuration when collection finishes.
3. Train complete globally shuffled epochs, without task/phase balancing.
4. Evaluate held-out simulator seeds and report per-seed success rates and failures.

```bash
python -m scripts.record_demonstrations --task pick --num-demos 100 \
  --output data/demos/pick_red_v01 --fixed-target-color red \
  --camera-names agentview robot0_eye_in_hand
python -m scripts.train_visual_bc --dataset data/demos/pick_red_v01 \
  --epochs 100 --batch-size 128 --learning-rate 0.0001 \
  --output-dir results/visual_bc/pick_red_v01
python -m scripts.evaluate_visual_bc \
  --checkpoint results/visual_bc/pick_red_v01/best.pt \
  --episodes 20 --num-seeds 5 --device cpu \
  --output-dir results/visual_bc_evaluation/pick_red_v01
```

No separate split/normalization command is required. Training artifacts go to
`data/splits/<dataset-name>_bc/`; checkpoints embed normalization and input config.
`--num-demos` targets the total successful episodes for a task. The legacy
`--episodes` recorder flag still means additional episodes.

## HDF5 Storage

Each successful episode is written atomically as `episode_000000.hdf5` before
its manifest entry. Array names remain `actions`, `phases`, `obs__...`,
`metadata_json`, and `result_json`. Numeric observations have T+1 frames for T
actions; strings use UTF-8. RGB is losslessly compressed with gzip level 4 and
one frame per chunk. Frame and GRU loaders read only the requested timestep or
window, with up to 16 cached file handles per dataset instance/process.

Old NPZ datasets are not mixed with HDF5 recordings. Optional migration creates
a new dataset and automatically prepares its split/statistics, without changing
the source:

```bash
python -m data.episode_io --source data/demos/pick_red_v01 \
  --output data/demos/pick_red_v01_hdf5
```

Use the new dataset path for training and a new run directory. Migration changes
dataset fingerprints, so an old training checkpoint cannot resume on the
converted dataset. Existing self-contained checkpoints remain usable for
evaluation because the model architecture and normalization are unchanged.

## Architecture And Objective

- Independent trainable CNNs for agentview and optional eye-in-hand RGB.
- Four Conv/ReLU layers, adaptive pooling 4x4, projection to 256D per camera.
- Direct normalized robot joint positions, EEF position, and gripper state;
  EEF quaternion stays included in all new experiments (16D total proprio).
- Frame policy: concatenation -> MLP 256 -> 128 -> motion and gripper logit.
- Optional `--temporal`: RGB/proprio/previous gripper -> GRU 256 -> MLP 128
  -> motion and gripper logit, trained on masked episode-local windows.
- Loss: XYZ MSE + optional rotation MSE + 0.2 x binary gripper BCEWithLogits.
- AdamW, LR 0.0001, batch size 128, no scheduler/clipping/dropout/LayerNorm.

Widths/depth live in `models/visual_bc.py`, not CLI. `VisualBCPolicy` uses
`VisualEncoder` from `models/visual_encoder.py`. Training deliberately requires
fixed-target Pick or matching-color Place, so each scene has one unambiguous target.
For Place, collection uses a fixed object/tray color (default red to red), with
randomized IDs and positions. Training, frame/GRU evaluation, and periodic
rollouts select the task from the checkpoint. See section 7 of either platform guide.

The scene geometry, variable counts, stable object IDs, and instruction metadata
remain project-specific. Pick collection uses a state machine and world-frame
orientation commands. `--random-yaw` enables initial cube yaw within +/-45
degrees (default); `--no-random-yaw` fixes initial yaw at zero.
`--rotation-mode none` disables rotational commands and orientation gates,
while `align_yaw` retains yaw-following control. Position ranges are unchanged.
These settings are saved and automatically restored during evaluation. GRU
evaluation carries recurrent state within each episode and resets it on reset.
Evaluation requires a 0.18 m target lift held for 20 steps
plus the environment success predicate. Place/Stack keep their task experts.

## Layout

```text
benchmarks/            Expert reliability gates
configs/               Semantic split definitions
controllers/           Panda controller
data/                  Recorder, preparation, dataset, normalization
  demos/               Generated trajectories (ignored)
  splits/              Automatically generated BC artifacts (ignored)
diagnostics/           Camera checks
docs/                  WINDOWS.md and LINUX.md
environments/          Tasks, scene configuration, and metadata descriptions
experts/               Task experts and primitives
models/                VisualBCPolicy and VisualEncoder
scripts/               Collection, training, evaluation and audits
tests/                 Regression, artifact-dependent smoke, simulation checks
results/               Reports, checkpoints and videos (ignored)
```

## Supported Paths

The maintained entry points are `scripts.record_demonstrations`,
`scripts.train_visual_bc`, and `scripts.evaluate_visual_bc`. Splits and statistics
are generated automatically during collection; audit tools remain optional.

Language-conditioned models, embedding caches, ResNet/spatial-softmax variants,
legacy history models, custom training samplers, and manual split/statistics commands have been
removed. Environment task descriptions remain metadata only; they are not model
inputs. Pick/Place/Stack environments and collection experts are still available,
but the current model trainer supports only fixed-target Pick and matching-color Place, not Stack.

Generated demos, splits, reports, and checkpoints are local artifacts ignored by
Git. Start by collecting a new dataset; no historical pilot data is needed by
code or offline tests. Checkpoints from removed models require their original
code version and cannot be resumed with this model.

## Guides

- [Windows CMD workflow and six camera/orientation experiments](docs/WINDOWS.md)
- [Linux Bash workflow and six camera/orientation experiments](docs/LINUX.md)

Both guides include frame and temporal-GRU training commands. A collected dataset
supports both; use separate output directories rather than switching architecture
on resume. Collection rejects appending different or missing yaw/rotation metadata.

## Verification

```bash
python -m tests
git diff --check
```

Offline regression tests use synthetic data and mocked simulation. Production
collection, full training, simulator expert gates, and Windows GPU execution are
separate, explicitly run workflows in the platform guides.
