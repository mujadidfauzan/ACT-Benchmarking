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
