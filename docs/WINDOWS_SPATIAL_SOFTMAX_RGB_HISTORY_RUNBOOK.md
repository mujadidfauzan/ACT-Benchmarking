# Windows: Spatial-Softmax BC With RGB History

Run these commands in Windows CMD from the repository root with the Conda environment active. Reuse `pick_fixed_red_v01`, its IID split, and the existing 9D normalizer. No recording, language cache, or new split is needed.

Update the remote repository with the committed changes before using the new flags:

```bat
git pull
set MUJOCO_GL=wgl
```

## Architecture

- Eight observations, ordered oldest to newest: `t-7, ..., t`.
- Both camera histories: `agentview` and `robot0_eye_in_hand`.
- Shared frozen ResNet18 layer3; each camera retains its own spatial-softmax adapter. Every frame produces 64 XY keypoints (128D) per camera.
- Proprio history: eight normalized EEF position/quaternion + gripper states (9D each).
- Action history: seven previously executed actions, `t-7, ..., t-1`. The target `action_t` is never an input.
- Concatenation: 2048 visual + 72 proprio + 49 action + 15 validity masks = 2184D.
- Head: `2184 -> 1024 -> 1024 -> 7`, with LayerNorm/ReLU/dropout on hidden layers. The output is one normalized action, not an action chunk.

Early-episode histories are left-padded with zeros. `True` masks identify real observations/actions; padded images are excluded from the visual encoder. Each evaluation episode starts with empty buffers. Frames are already OpenCV-oriented, so no vertical flip.

This is a frame-stack MLP, not a recurrent model. It may help motion and phase transitions, but does not guarantee better visual generalization. Train a fresh model; do not resume a no-history checkpoint into this architecture.

## 1. Loader Check

```bat
python -m scripts.test_dataset_loader ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --proprio-preset eef_gripper ^
  --camera-names agentview robot0_eye_in_hand ^
  --history-size 8 ^
  --rgb-history ^
  --batch-size 4 ^
  --num-workers 0 ^
  --no-vertical-flip
```

Expected BC shapes: each RGB history `(4, 8, 3, 256, 256)`, proprio history `(4, 8, 9)`, action history `(4, 7, 7)`. ACT remains unchanged by this BC-only flag.

Optional offline regression tests (synthetic data, CPU, no model downloads or real simulator rollouts):

```bat
python -m scripts.test_spatial_softmax_history
```

## 2. Train A Fresh History Run

Keep the same settings as the no-phase-balanced comparison: 400 optimizer batches per epoch, batch size 32, seed 42, learning rate 0.0001. Only temporal inputs/model width and the output folder change.

```bat
python -m scripts.train_spatial_softmax_visual_bc ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --train-split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --val-split data/splits/pick_fixed_red_v01_iid/val.jsonl ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_eef_gripper_9d.json ^
  --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand ^
  --history-size 8 ^
  --rgb-history ^
  --spatial-channels 64 ^
  --hidden-dim 1024 ^
  --dropout 0.1 ^
  --no-phase-balanced ^
  --batches-per-task 400 ^
  --epochs 200 ^
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
  --output-dir results/spatial_softmax_visual_bc/pick_fixed_red_rgb_h8_uniform_seed42
```

Startup must show `History size: 8; RGB history: True`. This is not phase-balanced sampling; phase labels remain diagnostic only. Each sampled target timestep brings its own contiguous history, even though batches are shuffled.

Eight-frame RGB encoding costs substantially more compute/memory than a single frame. If CUDA runs out of memory, reduce `--batch-size` to 16 or 8; note that this also changes the number of target samples per epoch at 400 batches. Do not compare those runs as if training budgets were identical.

For a state/action-history-only ablation, omit `--rgb-history` and use a separate output folder. For the old no-history architecture, omit both history flags.

To resume this history run, repeat exactly the same training command and append:

```bat
  --resume results/spatial_softmax_visual_bc/pick_fixed_red_rgb_h8_uniform_seed42/latest.pt
```

Add `^` to the preceding output-dir line when appending this flag in CMD. `--epochs 200` means the final epoch is 200, not 200 additional epochs.

## 3. Evaluate On Separate Scenes

Every 20 epochs the trainer computes success rate on the fixed development seed range and writes `rollout_evaluations.json`. `best_rollout.pt` is selected by development success; `best_val.pt` by validation MSE.

Evaluate the selected checkpoint on a different seed range:

```bat
python -m scripts.evaluate_visual_bc ^
  --checkpoint results/spatial_softmax_visual_bc/pick_fixed_red_rgb_h8_uniform_seed42/best_rollout.pt ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_eef_gripper_9d.json ^
  --episodes 20 ^
  --seed-offset 10000 ^
  --image-size 128 ^
  --max-steps 400 ^
  --success-hold-steps 5 ^
  --record-video ^
  --log-rollout ^
  --device cuda ^
  --output-dir results/spatial_softmax_visual_bc_evaluation/pick_fixed_red_rgb_h8_test20
```

Evaluation reads history settings from the checkpoint automatically; do not add training history flags to the evaluation command. Summary fields include `history_size: 8` and `rgb_history: true`. Step logs include valid RGB/proprio/action counts, starting with 1/1/0.

Use the same independent evaluation seeds for the no-history baseline. Compare success rate and failure reasons, not validation loss alone. These are independently sampled simulator scenes, not exact replay of the dataset's test episodes.
