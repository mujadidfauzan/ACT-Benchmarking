# Windows: Variable Object Counts

Run from the repository root in Windows CMD with Conda active. Training/model inputs do not change: both RGB images and robot proprio have fixed dimensions even when the scene contains different numbers of objects.

```bat
git pull
set MUJOCO_GL=wgl
```

Supported new dataset configurations:

| Task | Objects | Receptacles |
| --- | --- | --- |
| Pick | 1, 2, 3 | None |
| Place | 1, 2, 3 | 1, 2, 3, independently |
| Stack | 2, 3 | None |

Legacy fixed-count commands with 4 objects remain supported, but all examples below cap counts at 3. `--num-cubes` and `--num-objects` are aliases. `--object-counts` overrides the fixed object count; `--receptacle-counts` overrides the fixed receptacle count for Place.

The physical scene is rebuilt when its count changes; absent objects are not merely hidden. One environment stays open at a time. A new scene model gets a deterministic, different seed so rebuilding does not replay the same layout. Existing clearance, position, and yaw randomization rules remain unchanged.

## 1. Expert Gate Before Production

Benchmark counts are balanced over attempts, including failures. `summary.json` contains `by_scene_counts`; the gate must pass in every count group, not just overall. Use separate commands for each task because Stack cannot accept one object.

```bat
python -m benchmarks.expert_reliability ^
  --task pick ^
  --episodes 60 ^
  --object-counts 1 2 3 ^
  --run-name pick_counts_1_2_3 ^
  --enforce-gates
```

```bat
python -m benchmarks.expert_reliability ^
  --task place ^
  --episodes 180 ^
  --object-counts 1 2 3 ^
  --receptacle-counts 1 2 3 ^
  --run-name place_counts_1_2_3 ^
  --enforce-gates
```

```bat
python -m benchmarks.expert_reliability ^
  --task stack ^
  --episodes 40 ^
  --object-counts 2 3 ^
  --run-name stack_counts_2_3 ^
  --enforce-gates
```

These give 20 attempts per count group. For the fixed-red Pick experiment, append `--fixed-target-color red` to the Pick benchmark. Run names must be new; existing benchmark output directories are not overwritten.

Optional offline code tests, without real simulation or expert rollouts:

```bat
python -m scripts.test_variable_scene_counts
```

## 2. Collect The Multi-Task Dataset

The following Pick command varies target colors and is intended for language-conditioned training, not fixed-red vision-only BC.

```bat
python -m scripts.record_demonstrations ^
  --task pick ^
  --episodes 300 ^
  --object-counts 1 2 3 ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 1007 ^
  --max-attempts 1200 ^
  --output data/demos/variable_counts_v01
```

```bat
python -m scripts.record_demonstrations ^
  --task place ^
  --episodes 450 ^
  --object-counts 1 2 3 ^
  --receptacle-counts 1 2 3 ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 2007 ^
  --max-attempts 1800 ^
  --output data/demos/variable_counts_v01
```

```bat
python -m scripts.record_demonstrations ^
  --task stack ^
  --episodes 300 ^
  --object-counts 2 3 ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 3007 ^
  --max-attempts 1200 ^
  --output data/demos/variable_counts_v01
```

Successful-episode quotas:

- Pick: 100 each for 1/2/3 objects.
- Place: 50 for each of the nine object/receptacle count pairs.
- Stack: 150 each for 2/3 objects.

Failures are logged but not saved to NPZ. The failed configuration is retried with a fresh reset until its next success or the overall attempt limit; it is not replaced by an easier configuration. Partial runs raise an error and write a collection summary showing missing quotas.

Each invocation requests **additional** successful episodes, not the final dataset total. Rerunning a 300-episode command in an existing dataset adds another 300. There is no automatic quota-aware resume across separate invocations. Use a fresh directory for a clean experiment.

Each task directory contains:

- `episode_*.npz`: existing RGB, proprio, action, phase, metadata, and result payload.
- `manifest.jsonl` and `attempts.jsonl`: explicit object/receptacle counts.
- `collection_*.json`: requested versus saved count distributions for that invocation.

Count quotas do not guarantee equal color frequencies, stack orders, or task/phase timestep frequencies. Audit those separately. Metadata contains the actual scene entities and generates the instruction; experts continue to use privileged IDs.

## 3. Audit And Split

```bat
python -m scripts.audit_dataset ^
  --dataset data/demos/variable_counts_v01 ^
  --camera-names agentview robot0_eye_in_hand ^
  --mode full ^
  --check-duplicates ^
  --output results/dataset_audit/variable_counts_v01
```

The semantic statistics include `object_counts`, `receptacle_counts`, and `scene_count_pairs`. The audit validates declared counts against entity lists; older metadata without count fields still works.

For a pipeline sanity experiment, split within each count group:

```bat
python -m scripts.create_dataset_splits ^
  --dataset data/demos/variable_counts_v01 ^
  --tasks pick place stack ^
  --protocol iid ^
  --stratify-counts ^
  --seed 42 ^
  --output data/splits/variable_counts_v01_iid
```

At 80/10/10, each Pick group becomes 80/10/10, each Place pair 40/5/5, and each Stack group 120/15/15. Split summaries retain count distributions. Episode paths never cross splits.

`--stratify-counts` is IID-only. This is not a language/compositional generalization benchmark. For semantic/language holdouts, use the existing protocols without this flag and inspect their count distributions. Regenerate normalization and language caches for every new split; do not reuse files from the previous dataset.

For Language-BC preparation/training, follow [the existing runbook](VAST_AI_TRAINING_RUNBOOK.md), using these new dataset/split paths and OpenCV images without a vertical flip. Place and Stack with changing goals require language conditioning; the vision-only fixed-red trainer must not consume the variable-color Pick dataset above.

## 4. Separate Fixed-Red Pick Ablation

To compare data coverage with the current vision-only baseline, keep the target red and collect a separate dataset:

```bat
python -m scripts.record_demonstrations ^
  --task pick ^
  --episodes 300 ^
  --object-counts 1 2 3 ^
  --fixed-target-color red ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 4007 ^
  --max-attempts 1200 ^
  --output data/demos/pick_fixed_red_counts_v02
```

```bat
python -m scripts.audit_dataset ^
  --dataset data/demos/pick_fixed_red_counts_v02 ^
  --tasks pick ^
  --camera-names agentview robot0_eye_in_hand ^
  --expected-pick-target-color red ^
  --mode full ^
  --check-duplicates ^
  --output results/dataset_audit/pick_fixed_red_counts_v02
```

```bat
python -m scripts.create_dataset_splits ^
  --dataset data/demos/pick_fixed_red_counts_v02 ^
  --tasks pick ^
  --protocol iid ^
  --stratify-counts ^
  --seed 42 ^
  --output data/splits/pick_fixed_red_counts_v02_iid
```

```bat
python -m scripts.compute_normalization ^
  --dataset data/demos/pick_fixed_red_counts_v02 ^
  --split data/splits/pick_fixed_red_counts_v02_iid/train.jsonl ^
  --proprio-preset eef_gripper ^
  --image-convention opencv ^
  --output data/splits/pick_fixed_red_counts_v02_iid/normalization_eef_gripper_9d.json
```

Keep the same architecture as the experiment you are comparing against. For example, no-history spatial-softmax BC:

```bat
python -m scripts.train_spatial_softmax_visual_bc ^
  --dataset data/demos/pick_fixed_red_counts_v02 ^
  --train-split data/splits/pick_fixed_red_counts_v02_iid/train.jsonl ^
  --val-split data/splits/pick_fixed_red_counts_v02_iid/val.jsonl ^
  --normalization data/splits/pick_fixed_red_counts_v02_iid/normalization_eef_gripper_9d.json ^
  --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand ^
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
  --device cuda ^
  --output-dir results/spatial_softmax_visual_bc/pick_red_counts_v02_uniform_seed42
```

For the existing RGB-history architecture, append `--history-size 8 --rgb-history` and use a distinct output directory. Start a fresh training run for the new dataset/normalizer rather than resuming an old experiment.

Periodic trainer evaluation remains on 3-object scenes, preserving the old development benchmark. After checkpoint selection, evaluate each object count independently. This CMD loop runs 20 episodes each with 1, 2, and 3 objects:

```bat
for %N in (1 2 3) do python -m scripts.evaluate_visual_bc ^
  --checkpoint results/spatial_softmax_visual_bc/pick_red_counts_v02_uniform_seed42/best_rollout.pt ^
  --normalization data/splits/pick_fixed_red_counts_v02_iid/normalization_eef_gripper_9d.json ^
  --num-objects %N ^
  --episodes 20 ^
  --seed-offset 10000 ^
  --record-video ^
  --log-rollout ^
  --device cuda ^
  --output-dir results/spatial_softmax_visual_bc_evaluation/pick_red_counts_v02_n%N_test20
```

Use `%N` when pasting into CMD; use `%%N` inside a saved `.bat` file. Each summary/log records `num_objects`. Use the same seeds and counts when evaluating the previous checkpoint. Report each count's success rate; an overall increase driven only by one-object scenes is not evidence that three-object performance improved.

For Language-BC, the same fixed-count evaluation is supported with `--task pick --num-objects 1`, `--task place --num-objects 2 --num-receptacles 3`, or `--task stack --num-objects 2`, plus the normal checkpoint/cache arguments from the language runbook.

## 5. Multi-Task Spatial-Softmax Language-BC

This mode is implemented directly in `models/language_bc.py`. Use the multi-task dataset `variable_counts_v01` with Pick, Place, and Stack, not the Pick-only fixed-red dataset. Complete collection, audit, and the multi-task split in sections 1-3 first.

The architecture with two cameras, 64 spatial channels, and the 9D EEF/gripper preset is:

```text
Agentview: frozen shared ResNet18 layer3 -> camera adapter -> spatial softmax -> 128D
Wrist:     frozen shared ResNet18 layer3 -> camera adapter -> spatial softmax -> 128D
Proprio:   normalized EEF position/quaternion + gripper, directly concatenated -> 9D
Language:  frozen MiniLM cache -> trainable projection / token attention pooler -> 128D
Fusion:    393 -> H1 -> H2 -> 7D normalized action
```

Each hidden layer uses LayerNorm, ReLU, and dropout. Spatial keypoints are extracted independently of language; language conditions the final action through concatenation. This is late fusion, not language-guided visual attention. There is no temporal history in this model.

Select this mode explicitly with `--visual-fusion spatial_softmax`. Existing `global` and `spatial_attention` modes remain available, and the CLI default remains `spatial_attention` for compatibility.

### Prepare Normalization And Language Caches

```bat
python -m scripts.compute_normalization ^
  --dataset data/demos/variable_counts_v01 ^
  --split data/splits/variable_counts_v01_iid/train.jsonl ^
  --proprio-preset eef_gripper ^
  --image-convention opencv ^
  --output data/splits/variable_counts_v01_iid/normalization_eef_gripper_9d.json
```

Prepare the cache for the selected language mode. These commands may download MiniLM if it is not already cached; they are for you to run.

```bat
set HF_HUB_DISABLE_XET=1
python -m scripts.prepare_language_embeddings ^
  --dataset data/demos/variable_counts_v01 ^
  --split-dir data/splits/variable_counts_v01_iid ^
  --output data/splits/variable_counts_v01_iid/language_embeddings.npz ^
  --report-dir results/language_audit/variable_counts_v01_pooled
```

For token mode:

```bat
python -m scripts.prepare_language_token_embeddings ^
  --dataset data/demos/variable_counts_v01 ^
  --split-dir data/splits/variable_counts_v01_iid ^
  --output data/splits/variable_counts_v01_iid/language_token_embeddings.npz ^
  --report-dir results/language_audit/variable_counts_v01_tokens
```

### Configure Hidden Widths

Choose one form, not both:

```text
--hidden-dims 1024 1024    two equal hidden layers
--hidden-dims 512 256      two different hidden layers
--hidden-dim 512           shorthand for 512 512
```

`--fusion-hidden-dims` is an alias for `--hidden-dims`. If omitted, spatial-softmax mode uses 1024/1024; legacy modes use 256/128. `--spatial-channels 64` controls keypoint count, not hidden MLP width. Changing it changes fusion input size. Both hidden dimensions must be positive; spatial channels must be a positive multiple of 8.

### Train

```bat
python -m scripts.train_language_bc ^
  --dataset data/demos/variable_counts_v01 ^
  --train-split data/splits/variable_counts_v01_iid/train.jsonl ^
  --val-split data/splits/variable_counts_v01_iid/val.jsonl ^
  --normalization data/splits/variable_counts_v01_iid/normalization_eef_gripper_9d.json ^
  --pooled-cache data/splits/variable_counts_v01_iid/language_embeddings.npz ^
  --language-mode pooled ^
  --visual-fusion spatial_softmax ^
  --camera-names agentview robot0_eye_in_hand ^
  --spatial-channels 64 ^
  --hidden-dims 1024 1024 ^
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
  --device cuda ^
  --output-dir results/language_bc/variable_counts_v01_spatial_pooled_h1024_seed42
```

Startup should show `Fusion: 393 -> 1024 -> 1024 -> 7`. A different proprio normalizer changes the input size. One shared policy learns all three tasks; there are no task-specific action heads.

400 batches per task means **1200 training batches per epoch** for three tasks. It does not mean 400 total. For token mode, replace `--language-mode pooled` with `--language-mode token_attention`, replace the pooled-cache argument with `--token-cache data/splits/variable_counts_v01_iid/language_token_embeddings.npz`, and use a new output directory.

The language trainer still selects `best.pt` by validation loss and saves `latest.pt`. Unlike the spatial-softmax vision-only trainer, it does not currently provide periodic rollout evaluation flags. Do not copy `--rollout-eval-every` into this command. `--unfreeze-layer4-epoch` is rejected in spatial-softmax mode because this mode does not use layer4.

To resume, repeat the same architecture and command settings and append `--resume <output-dir>/latest.pt`. Hidden widths, spatial channels, camera configuration, and dropout are saved in the checkpoint and checked on resume. Changing widths is a new experiment, not a resume; use a new output directory.

### Offline Test And Final Rollout

This regression test uses synthetic data, no downloads, and no real simulator rollouts:

```bat
python -m scripts.test_language_bc_spatial_softmax
```

For final simulation evaluation, the evaluator reads hidden widths and spatial-channel settings from the checkpoint automatically:

```bat
python -m scripts.evaluate_language_bc ^
  --checkpoint results/language_bc/variable_counts_v01_spatial_pooled_h1024_seed42/best.pt ^
  --normalization data/splits/variable_counts_v01_iid/normalization_eef_gripper_9d.json ^
  --pooled-cache data/splits/variable_counts_v01_iid/language_embeddings.npz ^
  --task pick ^
  --num-objects 3 ^
  --episodes 20 ^
  --seed-offset 10000 ^
  --record-video ^
  --device cuda ^
  --output-dir results/language_bc_evaluation/variable_counts_v01_spatial_pooled_pick_n3
```

Run separate commands/output directories for other object counts and tasks. For Place also set `--num-receptacles`; Stack requires at least two objects. Include a token-cache argument when evaluating token mode. Language mode itself is read from the checkpoint, not an evaluation CLI flag. A strict cache lookup on newly generated instructions can miss combinations not present in the recorded dataset; the default evaluator falls back to the frozen encoder, which may require its model files locally.

## 6. Fine-Tune ResNet Layer3 In Vision-Only BC

These options apply to `scripts.train_spatial_softmax_visual_bc`, including its optional RGB-history model. They are not currently exposed by `scripts.train_language_bc`.

Spatial-softmax mode reads ResNet layer3 and does not use layer4. This experiment unfreezes only layer3; the stem, layer1, layer2, and unused layer4 stay frozen. Layer3 convolution and BatchNorm affine parameters are trained, but **all backbone BatchNorm running statistics remain fixed**, even when the policy is in training mode. Camera adapters and the action MLP continue learning.

Starting settings, not a guarantee of improvement:

| Setting | Value |
| --- | --- |
| Frozen warm-up | Epochs 1-20 |
| Start training layer3 | Epoch 21 |
| Head learning rate | 0.0001 |
| Layer3 learning rate | 0.00001 |

Fine-tuning can adapt visual features to the robot scene, but it can also increase overfitting on a small dataset. Compare independent rollout success rates against the frozen-backbone baseline, not training loss alone.

### New ImageNet-Initialized Run

Keep dataset, split, seed, sampling, and architecture unchanged so this run isolates the backbone training change. Use a new output directory:

```bat
python -m scripts.train_spatial_softmax_visual_bc ^
  --dataset data/demos/pick_fixed_red_counts_v02 ^
  --train-split data/splits/pick_fixed_red_counts_v02_iid/train.jsonl ^
  --val-split data/splits/pick_fixed_red_counts_v02_iid/val.jsonl ^
  --normalization data/splits/pick_fixed_red_counts_v02_iid/normalization_eef_gripper_9d.json ^
  --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand ^
  --spatial-channels 64 ^
  --hidden-dim 1024 ^
  --dropout 0.1 ^
  --no-phase-balanced ^
  --batches-per-task 400 ^
  --epochs 200 ^
  --batch-size 32 ^
  --image-size 128 ^
  --learning-rate 0.0001 ^
  --backbone-learning-rate 0.00001 ^
  --unfreeze-layer3-epoch 21 ^
  --weight-decay 0.0001 ^
  --gradient-clip 1.0 ^
  --num-workers 0 ^
  --seed 42 ^
  --pretrained-visual ^
  --no-vertical-flip ^
  --rollout-eval-every 20 ^
  --rollout-eval-episodes 20 ^
  --device cuda ^
  --output-dir results/spatial_softmax_visual_bc/pick_red_counts_v02_layer3_ft_seed42
```

At epoch 21 the log prints `Unfroze ResNet layer3 at epoch 21; LR=1e-05`. `metrics.json` records `layer3_trainable` and the learning rates for the head/backbone groups per epoch. Without `--unfreeze-layer3-epoch`, the old fully frozen behavior remains unchanged.

### Start From Existing Trained Weights

To fine-tune an existing model as a separate experiment, repeat the command above with these replacements:

- Set `--unfreeze-layer3-epoch 1` to train layer3 immediately.
- Add `--init-checkpoint results/spatial_softmax_visual_bc/pick_red_counts_v02_uniform_seed42/best_val.pt` (or your selected `best_rollout.pt`).
- Set `--epochs 40` for a shorter initial experiment, and use a distinct output directory such as `results/spatial_softmax_visual_bc/pick_red_counts_v02_layer3_from_best_seed42`.

`--init-checkpoint` loads **model weights only**. It starts at epoch 1 with fresh optimizer state and fresh checkpoint-selection metrics. Use the same architecture, dataset split, and normalization as the source model. Changing normalization changes the meaning of the model's normalized inputs/actions.

When loading `--init-checkpoint` or `--resume`, pretrained ImageNet initialization is skipped because the checkpoint already contains backbone weights. No additional ResNet weight download is needed for these paths.

### Resume A Fine-Tuning Run

Repeat the fine-tuning command with the same unfreeze epoch and append `--resume <fine-tuning-output>/latest.pt`. A fine-tuned checkpoint restores both optimizer groups and layer3's trainable state. A checkpoint saved during warm-up still unfreezes at the configured future epoch. CLI `--learning-rate` and `--backbone-learning-rate` values are applied after optimizer restoration.

`--resume` continues epoch numbers, optimizer state, and previous best metrics; `--init-checkpoint` starts a new experiment. The flags are mutually exclusive. A legacy frozen checkpoint can be resumed with a newly supplied unfreeze epoch; if the threshold is already past, layer3 starts learning at the first resumed epoch. Prefer weights-only initialization into a new directory for a clean ablation.

### Verify And Evaluate

Offline regression test (synthetic trajectories; no real simulator or pretrained download):

```bat
python -m scripts.test_layer3_finetuning
```

Use the same `scripts.evaluate_visual_bc` command from section 4 with the fine-tuned checkpoint path and a new evaluation output directory. The evaluator loads the updated ResNet weights automatically; it does not need an unfreeze flag. Fine-tuning increases training memory/compute; reducing batch size may be necessary, but changes the sample budget at a fixed 400 batches per epoch.
