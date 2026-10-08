# Windows Visual-BC Fixed-Red Pick Runbook

This experiment trains Pick with no language input. The red cube is always the
target, while its object ID, position, yaw, and non-red distractors remain
random. Commands use Anaconda Prompt / CMD.

## 1. Update and Configure

```bat
conda activate E:\Fauzan\conda-envs\sim-mt-act
cd /d E:\Fauzan\ACT-Benchmarking
git pull
set MUJOCO_GL=wgl
set HF_HUB_DISABLE_XET=1
```

Verify CUDA:

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. Record a 20-Episode Debug Dataset

Use a new directory so it cannot mix with language-conditioned datasets.

```bat
python -m scripts.record_demonstrations ^
  --task pick ^
  --episodes 20 ^
  --output data/demos/pick_fixed_red_v01_debug ^
  --fixed-target-color red ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 5107 ^
  --max-attempts 80
```

Audit both cameras and require red to be the target in every episode:

```bat
python -m scripts.audit_dataset ^
  --dataset data/demos/pick_fixed_red_v01_debug ^
  --tasks pick ^
  --mode full ^
  --check-duplicates ^
  --camera-names agentview robot0_eye_in_hand ^
  --expected-pick-target-color red ^
  --output results/dataset_audit/pick_fixed_red_v01_debug
```

Continue only after `Audit status: PASS (0 errors)`.

## 3. Record 100 Production Demonstrations

```bat
python -m scripts.record_demonstrations ^
  --task pick ^
  --episodes 100 ^
  --output data/demos/pick_fixed_red_v01 ^
  --fixed-target-color red ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 5207 ^
  --max-attempts 400
```

Verify the episode count:

```bat
dir /b data\demos\pick_fixed_red_v01\pick\episode_*.npz | find /c /v ""
```

Expected: `100`.

Run the production audit:

```bat
python -m scripts.audit_dataset ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --tasks pick ^
  --mode full ^
  --check-duplicates ^
  --camera-names agentview robot0_eye_in_hand ^
  --expected-pick-target-color red ^
  --output results/dataset_audit/pick_fixed_red_v01_full
```

## 4. Create the IID Split

```bat
python -m scripts.create_dataset_splits ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --output data/splits/pick_fixed_red_v01_iid ^
  --tasks pick ^
  --protocol iid ^
  --seed 42
```

Expected episode counts: `80 train / 10 val / 10 test`.

## 5. Compute 23D Normalization

```bat
python -m scripts.compute_normalization ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --output data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --image-convention opencv
```

Validate it:

```bat
python -m scripts.test_normalization ^
  --statistics data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --dataset data/demos/pick_fixed_red_v01
```

The output must report `Proprio shape: (23,)`.

## 6. Test the Dataset Loader

```bat
python -m scripts.test_dataset_loader ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --batch-size 4 ^
  --num-workers 0 ^
  --camera-names agentview robot0_eye_in_hand ^
  --no-vertical-flip
```

Confirm that both images and 23D proprioception are present.

## 7. Run the Mechanical Tiny Overfit

```bat
python -m scripts.tiny_overfit_visual_bc ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --visual-fusion spatial_attention ^
  --camera-names agentview robot0_eye_in_hand ^
  --no-vertical-flip ^
  --steps 300 ^
  --batch-size 4 ^
  --image-size 128 ^
  --output-dir results/tiny_overfit/pick_fixed_red_visual_bc
```

Continue only if it reports `Visual-BC mechanical tiny overfit: PASS`.

## 8. Train Visual-BC on the GPU

The ResNet18 backbone remains frozen for the entire run. The learned spatial
attention, proprio encoder, fusion MLP, and action head are trained.

```bat
python -m scripts.train_visual_bc ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --train-split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --val-split data/splits/pick_fixed_red_v01_iid/val.jsonl ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --fixed-target-color red ^
  --visual-fusion spatial_attention ^
  --camera-names agentview robot0_eye_in_hand ^
  --phase-balanced ^
  --output-dir results/visual_bc/pick_fixed_red_v01_seed42 ^
  --epochs 30 ^
  --batch-size 64 ^
  --image-size 224 ^
  --learning-rate 1e-4 ^
  --weight-decay 1e-4 ^
  --gradient-clip 1.0 ^
  --num-workers 4 ^
  --cache-size 8 ^
  --seed 42 ^
  --pretrained-visual ^
  --no-vertical-flip ^
  --device cuda
```

If Windows DataLoader workers fail, rerun with `--num-workers 0`.

## 9. Evaluate the Best Checkpoint

```bat
python -m scripts.evaluate_visual_bc ^
  --checkpoint results/visual_bc/pick_fixed_red_v01_seed42/best.pt ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --episodes 20 ^
  --device cuda ^
  --record-video ^
  --log-rollout ^
  --output-dir results/visual_bc_evaluation/pick_fixed_red_v01_20
```

Each `rollouts/pick_episode_XXX.jsonl` stores per-step EEF-to-target distance,
target lift, normalized/raw/executed actions, clipping state, and gripper
command. Use a new output directory for every evaluation run.

## 10. Resume an Interrupted Run

Repeat the exact training command and append:

```bat
--resume results/visual_bc/pick_fixed_red_v01_seed42/latest.pt
```

Do not resume from a Language-BC checkpoint.

## 11. History-Visual-BC Experiment

This branch reuses the same demonstrations, split, and normalization. It adds
the current and seven previous proprio states plus the seven previously
executed actions. Current RGB remains the only visual history.

History training uses the phase-balanced sampler. Pick has five recorded
phases, so `--batches-per-phase 80` produces 400 optimizer batches per epoch.
Do not combine `--phase-balanced` with `--batches-per-task`; those options
select different samplers.

Test history-window alignment:

```bat
python -m scripts.test_dataset_loader ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --batch-size 4 ^
  --history-size 8 ^
  --num-workers 0 ^
  --camera-names agentview robot0_eye_in_hand ^
  --no-vertical-flip
```

Run the history tiny-overfit gate:

```bat
python -m scripts.tiny_overfit_visual_bc ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --history-size 8 ^
  --visual-fusion spatial_attention ^
  --camera-names agentview robot0_eye_in_hand ^
  --no-vertical-flip ^
  --steps 300 ^
  --batch-size 4 ^
  --image-size 128 ^
  --output-dir results/tiny_overfit/pick_fixed_red_history_visual_bc
```

Train with the frozen ResNet18 backbone:

```bat
python -m scripts.train_history_visual_bc ^
  --dataset data/demos/pick_fixed_red_v01 ^
  --train-split data/splits/pick_fixed_red_v01_iid/train.jsonl ^
  --val-split data/splits/pick_fixed_red_v01_iid/val.jsonl ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --fixed-target-color red ^
  --history-size 8 ^
  --visual-fusion spatial_attention ^
  --camera-names agentview robot0_eye_in_hand ^
  --phase-balanced ^
  --batches-per-phase 80 ^
  --output-dir results/history_visual_bc/pick_fixed_red_h8_seed42 ^
  --epochs 80 ^
  --batch-size 32 ^
  --image-size 128 ^
  --learning-rate 1e-4 ^
  --weight-decay 1e-4 ^
  --gradient-clip 1.0 ^
  --num-workers 0 ^
  --cache-size 8 ^
  --seed 42 ^
  --pretrained-visual ^
  --no-vertical-flip ^
  --device cuda
```

Evaluate five logged rollouts first:

```bat
python -m scripts.evaluate_visual_bc ^
  --checkpoint results/history_visual_bc/pick_fixed_red_h8_seed42/best.pt ^
  --normalization data/splits/pick_fixed_red_v01_iid/normalization_23d.json ^
  --episodes 5 ^
  --device cuda ^
  --record-video ^
  --log-rollout ^
  --output-dir results/history_visual_bc_evaluation/pick_fixed_red_h8_smoke5
```

The evaluator obtains history size from the checkpoint. At rollout time it uses
the actions actually sent to the environment, matching deployment conditions.
