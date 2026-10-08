# Spatial-Softmax Vision-BC Runbook

This experiment is a controlled Pick baseline: target is always the red cube; other cubes remain visual distractors. It uses no language and no temporal history.

## Architecture

Two RGB views are used: `agentview` and `robot0_eye_in_hand`. A frozen ImageNet ResNet18 produces each view's stride-16 layer-3 feature map. Each camera has its own trainable `1x1 Conv -> GroupNorm -> ReLU -> spatial softmax` module with 64 channels. Each channel produces an expected `(x, y)` coordinate, so each view yields 128 values.

The policy input is:

```text
agentview spatial coordinates  128D
eye-in-hand spatial coordinates 128D
EEF position + quaternion + gripper qpos 9D
---------------------------------------------
fusion input                   265D
```

The action head is `Linear(265, 1024) -> ReLU -> Linear(1024, 1024) -> ReLU -> Linear(1024, 7)`. LayerNorm and dropout are included after each hidden linear layer. The output is a normalized 7D action. Training learning rate is `1e-4`.

## 1. Verify Dataset Loading

Run this from the repository root. `--no-vertical-flip` is correct for this OpenCV-convention dataset.

```bat
python -m scripts.test_dataset_loader ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --proprio-preset eef_gripper ^
  --camera-names agentview robot0_eye_in_hand ^
  --batch-size 4 ^
  --num-workers 0 ^
  --no-vertical-flip
```

Expected: `Proprio: (4, 9)` and both RGB camera tensors are present.

## 2. Compute The 9D Normalization File

This reads only the train split. It must be a new file because prior 23D normalization files are incompatible.

```bat
python -m scripts.compute_normalization ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --proprio-preset eef_gripper ^
  --image-convention opencv ^
  --output data/splits/pick_fixed_red_v01_iid/normalization_eef_gripper_9d.json
```

If rerunning intentionally, append `--overwrite`.

## 3. Mechanical Tiny-Overfit Gate

The loss should collapse on one fixed batch before spending GPU time on full training.

```bat
python -m scripts.tiny_overfit_visual_bc ^
  --architecture spatial_softmax ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_eef_gripper_9d.json ^
  --camera-names agentview robot0_eye_in_hand ^
  --spatial-channels 64 ^
  --hidden-dim 1024 ^
  --image-size 128 ^
  --steps 300 ^
  --learning-rate 1e-3 ^
  --no-vertical-flip ^
  --pretrained-visual ^
  --output-dir results/tiny_overfit/spatial_softmax_pick_red
```

Expected: `Visual-BC mechanical tiny overfit: PASS`.

## 4. Train With Closed-Loop Evaluation Every 20 Epochs

This training run has 400 phase-balanced batches per epoch (`5 phases x 80`). At epochs 20, 40, 60, and 80 it evaluates the in-memory policy on 20 deterministic Pick scenes and stores each success rate.

```bat
python -m scripts.train_spatial_softmax_visual_bc ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --train-split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --val-split data/splits/pick_fixed_red_v01_iid/val.jsonl ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_eef_gripper_9d.json ^
  --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand ^
  --spatial-channels 64 ^
  --hidden-dim 1024 ^
  --dropout 0.1 ^
  --phase-balanced ^
  --batches-per-phase 80 ^
  --epochs 80 ^
  --batch-size 32 ^
  --image-size 128 ^
  --learning-rate 0.0001 ^
  --weight-decay 0.0001 ^
  --gradient-clip 1.0 ^
  --num-workers 0 ^
  --seed 42 ^
  --pretrained-visual ^
  --no-vertical-flip ^
  --rollout-eval-every 20 ^
  --rollout-eval-episodes 20 ^
  --rollout-eval-seed-offset 0 ^
  --device cuda ^
  --output-dir results/spatial_softmax_visual_bc/pick_fixed_red_v01_seed42
```

The first pretrained run may download ResNet18 weights. Use `--device cpu` only for a short smoke run, and set `--rollout-eval-every 0` when simulation evaluation is not available.

## 5. Inspect And Evaluate The Best Closed-Loop Checkpoint

Training outputs:

- `best_val.pt`: lowest validation MSE.
- `best_rollout.pt`: highest periodic closed-loop success rate.
- `rollout_evaluations.json`: 20-episode results at each scheduled epoch.
- `epoch_020.pt`, etc.: checkpoint at each rollout evaluation.

Evaluate `best_rollout.pt` on a different deterministic seed range, record videos, and log controls:

```bat
python -m scripts.evaluate_visual_bc ^
  --checkpoint results/spatial_softmax_visual_bc/pick_fixed_red_v01_seed42/best_rollout.pt ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_eef_gripper_9d.json ^
  --episodes 20 ^
  --seed-offset 10000 ^
  --image-size 128 ^
  --max-steps 400 ^
  --success-hold-steps 5 ^
  --record-video ^
  --log-rollout ^
  --device cuda ^
  --output-dir results/spatial_softmax_visual_bc_evaluation/pick_red_test20
```

`seed-offset 0` is the periodic development measurement. Keep it separate from `seed-offset 10000` so the later evaluation is not merely a replay of the checkpoint-selection scenes.
