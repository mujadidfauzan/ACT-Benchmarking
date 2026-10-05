# SimMT-ACT

## Language-Conditioned Multi-Task Robot Manipulation with Action Chunking Transformers in Simulation

SimMT-ACT is a simulation-based robot learning project focused on **language-conditioned multi-task manipulation** using imitation learning and action chunking.

For a complete clean-instance workflow covering Vast.ai setup, dataset
production, multi-task training, per-task training, artifact download, and
closed-loop evaluation, see
[Vast.ai Dataset and Training Runbook](docs/VAST_AI_TRAINING_RUNBOOK.md).

The project adapts ideas from **Action Chunking Transformer (ACT)** and **MT-ACT / RoboAgent** into a custom MuJoCo and robosuite simulation environment with a Franka Panda robot.

The main objective is to build a complete robot learning pipeline:

```text
Simulation Environment
        ↓
Scripted Expert Demonstrations
        ↓
Imitation Learning Policies
        ↓
Benchmarking
        ↓
Generalization Evaluation
```

This project focuses on reproducing and adapting modern robot learning approaches rather than proposing a new learning algorithm.

---

# Project Goals

This project aims to:

- Build a multi-task robot manipulation environment using MuJoCo and robosuite.
- Generate scalable demonstrations using scripted expert controllers.
- Reproduce ACT-style action chunking for robot manipulation.
- Extend policies into language-conditioned multi-task manipulation.
- Compare different imitation learning architectures fairly.
- Evaluate policy robustness under different object, spatial, and language variations.
- Build a reproducible benchmark for simulation-based robot learning.

---

# Environment

## Simulation Stack

| Component        | Description                    |
| ---------------- | ------------------------------ |
| Simulator        | MuJoCo                         |
| Framework        | robosuite                      |
| Robot            | Franka Emika Panda             |
| Learning Setting | Imitation Learning             |
| Data Source      | Scripted Expert Demonstrations |

---

## Observation Modalities

The policy receives:

```text
RGB Images
+
Robot Proprioception
+
Language Instruction
```

The expert controller additionally has access to privileged simulator information:

```text
Object position
Object orientation
Target position
Target orientation
```

However, these privileged states are **only used for demonstration generation** and are not provided to the learned policy.

---

# Manipulation Tasks

The project focuses on three manipulation tasks.

---

# 1. Pick

Language-grounded selection from a scene containing three cubes. One cube is the instructed target and the remaining cubes are distractors.

Example instruction:

```text
"Pick up the red cube."
```

Task objective:

```text
Locate object
      ↓
Approach
      ↓
Grasp
      ↓
Lift object
```

---

# 2. Pick-and-Place

Object-to-container manipulation task.

The environment contains:

- Three cube objects
- Three target trays/receptacles
- One randomly selected source-target pair per episode
- Non-target objects and trays as distractors

Example instructions:

```text
"Place the red cube into the yellow container."

"Put the blue cylinder inside the green tray."
```

Object attributes:

- Shape
- Color
- Position
- Orientation

Target attributes:

- Container color
- Position
- Orientation

The task requires the robot to understand:

```text
Object identity
        +
Target identity
        +
Spatial relationship
```

---

# 3. Multi-Object Stack

Long-horizon object-object relational manipulation.

The primary task uses three cubes. A four-cube variant is reserved as a
long-horizon evaluation extension beyond the main training scope.

Example instructions:

```text
"Stack the green cube on the blue cube, and stack the blue cube on the red cube."

"Build a stack with the red cube at the bottom, the blue cube in the middle,
and the green cube on top."
```

Stack order is always represented from bottom to top:

```text
stack_order = ["red", "blue", "green"]

       green
        ■
       blue
        ■
        red
        ■
---------------- table
```

The robot must understand:

```text
Object identity
      +
Stack order
      +
Intermediate targets
      +
Stack stability
      +
Long-horizon error accumulation
```

The stack is constructed bottom-up. For
`[red, blue, green]`, the robot first places blue on red and then places green
on blue. It does not attempt to grasp and transport a pre-built sub-stack.

Primary training scope:

- Three cubes
- Identical cube geometry
- Unique colors sampled from red, blue, green, and yellow
- Random non-overlapping positions
- Bounded random yaw
- Random bottom-to-top color order
- Position and height alignment; cube yaw is not part of the success criterion

Evaluation extension:

- Four cubes
- Longer episode horizon
- Higher final stack
- Increased collision and stability risk
- Generalization beyond the three-object training horizon

Each adjacent pair in the final stack must satisfy:

```text
XY distance(top, support) < 0.02 m

Z(top) ≈ Z(support) + cube_height
```

The stack must remain valid after a settling window. A cube that initially
lands correctly but later falls is counted as a failure.

---

# Stack Development Plan

Stacking is implemented incrementally before demonstration recording:

```text
1. Build StackEnv with three cubes
        ↓
2. Validate ten randomized resets
        ↓
3. Generalize object access by object name
        ↓
4. Generalize PickExpert with object_name
        ↓
5. Compute stack destination and reuse transport_object()
        ↓
6. Implement StackExpert as an orchestrator
        ↓
7. Validate three-object stack success
        ↓
8. Add four-object evaluation extension
        ↓
9. Record demonstrations after all experts are stable
```

`StackEnv` contains a Panda robot, a table, and three cubes by default. A
fourth cube can be enabled for evaluation. Every reset randomizes cube colors,
positions, yaw within `[-π/4, π/4]`, and stack order while keeping all cubes
reachable, separated, and fully supported by the table.

Simulator object IDs and observation keys remain stable across resets even when
colors change:

```text
cube_0_pos
cube_0_quat
cube_1_pos
cube_1_quat
cube_2_pos
cube_2_quat
```

Color is semantic metadata, not part of the fixed observation schema. Because
colors are unique within an episode, the environment maintains a resolver such
as `color_to_object_id` and exposes named object state:

```text
get_object_position(object_name)
get_object_orientation(object_name)
resolve_object_id(color)
```

The ten-reset environment test validates:

- Stable observation keys and shapes
- Unique sampled colors
- Valid bottom-to-top stack order
- Position and yaw variation across resets
- No initial cube overlap
- Full support within table boundaries
- Minimum safe separation and robot reachability bounds

`StackExpert` is an orchestrator built from reusable skills:

```text
for each object above the bottom object:
    pick(object_name)
    place_on_object(target_object_name)
    wait_for_settling()
    verify_adjacent_pair()
```

---

# Demonstration Dataset

Demonstrations are generated using scripted expert controllers.

Demonstration recording starts only after PickExpert, PlaceExpert, and
StackExpert have passed their task-level success evaluations. Environment and
expert validation are completed before dataset collection so failed controller
behavior is not silently included in the imitation dataset.

The expert pipeline:

```text
Sample Task
      ↓
Randomize Scene
      ↓
Generate Language Instruction
      ↓
Execute Expert Controller
      ↓
Verify Success
      ↓
Store Trajectory
```

## Expert Reliability Gate

Before pilot dataset production, run 20 reproducible attempts for every expert:

```bash
python -m benchmarks.expert_reliability \
  --task all \
  --episodes 20 \
  --run-name production_gate_20 \
  --enforce-gates
```

The benchmark uses non-overlapping seed ranges: Pick starts at `1000`, Place
at `2000`, and Stack at `3000`. Results are written under
`results/expert_reliability/<run-name>/`:

```text
pick_attempts.jsonl
place_attempts.jsonl
stack_attempts.jsonl
summary.json
```

Every attempt is logged, including failures. Logs contain the seed, initial
scene poses, target IDs and poses, stack order where applicable, expert stage,
normalized failure reason, instruction template, semantic split, and number of
steps. The initial gates are 95% Pick, 90% Place, and 85% Stack.

After changing environment or expert behavior, use a fresh seed range with
`--seed-offset`. For example, `--seed-offset 20` evaluates Pick on
`1020-1039`, Place on `2020-2039`, and Stack on `3020-3039`.

Initial-scene clearance is pair-aware: cube-cube, cube-tray, and tray-tray
pairs use progressively wider minimum distances. Stack uses additional
cube-cube clearance because the gripper repeatedly approaches objects beside
a growing tower. Unsafe dense scenes are rejected during reset instead of
being silently filtered after expert execution.

Language templates are selected from a controlled template bank. Use
`--instruction-split held_out` for a separate language-variation benchmark;
production training demonstrations use the default `train` split. Semantic
combination assignments live in `configs/dataset_splits.json` and remain
independent from the language-template split.

---

## Recording Demonstrations

The recorder stores only expert episodes that pass task verification. Failed
attempts are rejected and retried up to `--max-attempts`.

Record RGB, state, action, language, and task metadata:

```bash
MUJOCO_GL=egl python -m scripts.record_demonstrations \
  --task pick --episodes 100 --camera-obs

MUJOCO_GL=egl python -m scripts.record_demonstrations \
  --task place --episodes 100 --camera-obs

MUJOCO_GL=egl python -m scripts.record_demonstrations \
  --task stack --episodes 100 --camera-obs
```

Use `--render` for an interactive viewer. Omit `--camera-obs` only for a
smaller state/action debugging dataset. Output is written to
`data/demos/<task>/episode_XXXXXX.npz` with a `manifest.jsonl` index.

`manifest.jsonl` contains successful demonstrations only. `attempts.jsonl`
contains every production attempt, including scene-sampling failures, expert
failures, unexpected exceptions, and successful episode indices. Place scene
metadata also records sampler attempts, internal sampler errors, and clearance
rejections, allowing reset rejection rates to be audited separately from
expert reliability.

Each episode contains `T` actions and `T+1` observations. Observation arrays
use the `obs__<key>` prefix, while `metadata_json` contains the instruction and
privileged target IDs. Privileged object poses may be retained for analysis,
but policy loaders must expose only RGB, proprioception, and language.

## Dataset Splits and Loaders

Build the semantic-aware pilot split after the dataset passes audit:

```bash
python -m scripts.create_dataset_splits \
  --dataset data/demos/pilot_v01 \
  --output data/splits/pilot_v01_semantic \
  --protocol semantic
```

Place and Stack episodes marked `compositional_test` are assigned only to the
test split. Tasks without held-out semantic episodes, currently Pick, use an
IID 80/10/10 fallback. The builder writes `train.jsonl`, `val.jsonl`,
`test.jsonl`, and `summary.json`. Existing outputs require `--overwrite` to be
replaced.

The PyTorch dataset supports one-step BC and padded ACT action chunks. The
legacy `pilot_v01` RGB observations use the OpenGL image convention, so enable
`vertical_flip=True` specifically for this dataset:

```python
from data.manipulation_dataset import ManipulationDataset, create_dataloader

train_bc = ManipulationDataset(
    dataset_root="data/demos/pilot_v01",
    split_manifest="data/splits/pilot_v01_semantic/train.jsonl",
    mode="bc",
    vertical_flip=True,
)
train_loader = create_dataloader(train_bc, batch_size=32, seed=42)

train_act = ManipulationDataset(
    dataset_root="data/demos/pilot_v01",
    split_manifest="data/splits/pilot_v01_semantic/train.jsonl",
    mode="act",
    chunk_size=32,
    vertical_flip=True,
)
act_loader = create_dataloader(train_act, batch_size=32, seed=42)
```

For multi-task training, enable equal task weighting:

```python
train_loader = create_dataloader(
    train_act,
    batch_size=32,
    task_balanced=True,
    seed=42,
)
```

The balanced sampler assigns the same number of batches to Pick, Place, and
Stack, and switches tasks in short blocks to retain efficient episode caching.
Validation and test loaders should use `task_balanced=False` so metrics cover
every recorded timestep exactly once.

Compute normalization statistics only from the training split:

```bash
python -m scripts.compute_normalization \
  --dataset data/demos/pilot_v01 \
  --split data/splits/pilot_v01_semantic/train.jsonl \
  --output data/splits/pilot_v01_semantic/normalization.json
```

The output stores raw timestep-weighted statistics, per-task statistics, and
equal-task statistics. Multi-task models use `task_balanced_proprio` and
`task_balanced_action` so normalization matches the balanced training
distribution. It also records the split hash and legacy image-flip requirement.

Load the equal-task statistics for training and inference:

```python
from data.normalization import PolicyNormalizer

normalizer = PolicyNormalizer.from_json(
    "data/splits/pilot_v01_semantic/normalization.json"
).to(device)
normalizer.validate_split(
    "data/splits/pilot_v01_semantic/train.jsonl"
)

normalized_proprio = normalizer.normalize_proprio(batch["proprio"].to(device))
normalized_target = normalizer.normalize_action(batch["action"].to(device))

# Convert a normalized policy prediction back to the environment action scale.
environment_action = normalizer.denormalize_action(predicted_action)
```

The normalizer stores statistics as non-trainable PyTorch buffers, so they move
between CPU and GPU and are included in a model checkpoint. It accepts both
single actions shaped `[7]` and action chunks shaped `[..., 7]`. Run its smoke
test with `python -m scripts.test_normalization`.

## Frozen Language Embeddings

Language-BC and Language-ACT use the pooled 384-dimensional output from frozen
`sentence-transformers/all-MiniLM-L6-v2`. Install the local CPU dependencies:

```bash
pip install -r requirements-ml-cpu.txt
```

Build one embedding per unique instruction and audit semantic separation:

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.prepare_language_embeddings
```

This writes:

```text
data/splits/pilot_v01_semantic/language_embeddings.npz
results/language_audit/pilot_v01_minilm/audit_report.{json,md}
```

The cache covers instructions in train, validation, and test without fitting
on them: MiniLM is pretrained and frozen. The artifact records the exact model
revision and contains unit-normalized vectors. Validate it with:

```bash
python -m scripts.test_language_encoder --verify-live-encoder
```

The pilot audit raises a warning for Stack: some instructions with different
or reversed orders have very high pooled-embedding cosine similarity. Pooled
MiniLM remains the lightweight language baseline, while the main MT-ACT model
will use token-level language features to preserve object order and relations.

The token-aware language baseline keeps MiniLM frozen but caches every token
embedding before sentence pooling:

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.prepare_language_token_embeddings
```

This creates:

```text
data/splits/pilot_v01_semantic/language_token_embeddings.npz
results/language_audit/pilot_v01_minilm_tokens/audit_report.{json,md}
```

The cache contains padded token embeddings, input IDs, token strings, and
attention masks. A trainable attention pooler projects each 384D token to a
128D feature space, assigns masked attention weights, and produces one language
feature. MiniLM remains frozen; only the projection and attention scorer are
optimized with the manipulation policy.

After building the cache, validate masking, gradients, and checkpoint loading:

```bash
python -m scripts.test_token_language
```

The token audit verifies that canonical Stack orders from episode metadata map
to distinct token sequences and color-token positions. It does not by itself
prove learned order understanding. That is tested later by overfitting two
instructions containing the same colors in reversed bottom-to-top order.

BC samples contain `image`, `proprio`, and `action`. ACT samples contain
`image`, `proprio`, `actions`, and `action_padding_mask`, where `True` marks
padded chunk positions. Both modes also return instruction, task, episode
path, and timestep. The image tensor is float RGB in `CHW` layout and `[0, 1]`;
proprioception concatenates joint position, joint velocity, and gripper qpos.
The episode-aware batch sampler keeps compressed NPZ access practical while
still shuffling episodes and timesteps each epoch.

---

## Dataset Variations

Pick displays three uniquely colored cubes and chooses one instructed target per reset. Place displays three uniquely colored cubes and three uniquely colored trays; object and tray palettes are sampled independently, so a matching source-target color pair is valid. Stack uses unique colors within an episode to keep its requested order unambiguous.

### Object Attributes

- Shape
- Color
- Position
- Orientation

### Target Attributes

- Container color
- Container position
- Orientation

### Scene Conditions

- Object placement
- Initial robot state
- Distractor objects (optional)

### Stack Attributes

- Number of cubes
- Unique cube colors
- Bottom-to-top stack order
- Initial cube positions
- Bounded cube yaw
- Three-object training horizon or four-object evaluation horizon

---

## Stored Trajectory Information

Each episode contains:

```text
RGB observations

Robot joint states

End-effector pose

Gripper state

Actions

Language instruction

Task metadata

Success label
```

Example metadata:

```json
{
  "task": "place",

  "object": {
    "shape": "cube",
    "color": "red"
  },

  "target": {
    "type": "container",
    "color": "yellow"
  },

  "instruction": "Place the red cube into the yellow container."
}
```

Example stack metadata:

```json
{
  "task": "stack",
  "num_objects": 3,
  "objects": [
    {"id": "cube_0", "color": "red"},
    {"id": "cube_1", "color": "blue"},
    {"id": "cube_2", "color": "green"}
  ],
  "stack_order": ["red", "blue", "green"],
  "instruction": "Build a stack with the red cube at the bottom, the blue cube in the middle, and the green cube on top."
}
```

For stack episodes, `stack_order` is always stored from bottom to top and is
the canonical task specification used by both the instruction generator and
the scripted expert.

---

# Models

The project evaluates several imitation learning policies.

---

# 1. Behavioral Cloning (BC)

Baseline single-step imitation learning model.

Architecture:

```text
Vision
+
Robot State
+
Language (optional)
        ↓
      Policy
        ↓
    Single Action
```

Used as a baseline to evaluate the effect of action chunking.

---

# 2. Action Chunking Transformer (ACT)

Based on:

> Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, RSS 2023

Instead of predicting one action:

```text
a_t
```

ACT predicts an action sequence:

```text
a_t, a_t+1, ..., a_t+k
```

Benefits:

- Reduces effective decision horizon.
- Reduces compounding error.
- Produces smoother manipulation trajectories.

Temporal ensembling is used to combine overlapping action predictions.

---

# 3. Language-Conditioned BC

Multi-task behavioral cloning with language input.

```text
RGB
+
Robot State
+
Language Embedding

        ↓

      Policy

        ↓

   Single Action
```

Used to evaluate whether language conditioning improves multi-task manipulation.

---

# 4. Language-Conditioned ACT

ACT with language conditioning.

```text
Vision Features
+
Robot State
+
Language Embedding

        ↓

 Transformer

        ↓

 Action Chunk
```

This isolates the contribution of action chunking while maintaining equivalent task information.

---

# 5. MT-ACT

Main multi-task model inspired by RoboAgent / MT-ACT.

Language information conditions visual representations before action prediction.

Architecture:

```text
Language Instruction

        ↓

Language Encoder

        ↓

Task Representation

        ↓

Feature Conditioning (FiLM)

        ↓

ACT Transformer

        ↓

Action Chunk
```

---

# Benchmark Design

The project contains two evaluation tracks.

---

# Single-Task Benchmark

Purpose:

Evaluate the effect of action chunking without task ambiguity.

Models:

- BC
- ACT

Task:

- Pick

---

# Multi-Task Language Benchmark

Purpose:

Evaluate language-conditioned multi-task manipulation.

Models:

- Language-BC
- Language-ACT
- MT-ACT

Tasks:

- Pick
- Pick-and-Place
- Three-object Stack
- Four-object Stack as a long-horizon generalization evaluation

All models receive:

```text
RGB
+
Robot State
+
Language Instruction
```

This ensures fair comparison.

---

# Evaluation

The evaluation focuses on:

1. Task performance
2. Generalization capability
3. Computational efficiency

---

# Main Metric

## Task Success Rate (SR)

Primary evaluation metric:

\[
SR =
\frac{
N_{successful\ episodes}
}
{
N_{total\ episodes}
}
\]

A successful episode requires satisfying task-specific constraints.

Examples:

### Pick-and-Place

Success:

```text
Object is placed inside the correct container.
```

### Stack

Success requires every adjacent pair in the requested bottom-to-top order to
remain aligned after settling:

```text
For each (top, support) pair:
    XY distance(top, support) < 0.02 m
    Z error relative to one cube height is within tolerance

All cubes remain stable for the settling window.
```

Partial stacks are failures even when the correctly placed prefix remains
stable.

---

# Additional Metrics

## Completion Time

Number of environment steps required to finish the task.

Measures manipulation efficiency.

---

## Final Goal Distance

Measures final object error relative to the target.

Examples:

Pick-and-Place:

```text
Distance(object, container center)
```

Stack:

```text
Maximum and mean XY / Z error across all adjacent stack pairs
```

---

## Failure Analysis

Failure cases are categorized into:

- Grasp failure
- Object drop
- Wrong object selection
- Wrong target selection
- Placement misalignment
- Wrong stack order
- Partial stack
- Stack instability after settling
- Timeout

---

## Computational Metrics

Reported metrics:

- Training time
- Peak VRAM usage
- Inference latency

---

# Generalization Evaluation

---

## 1. In-Distribution Evaluation

Evaluate known tasks with unseen random seeds.

Variation:

- Object position
- Object orientation
- Initial robot state

Goal:

Measure normal task execution robustness.

---

## 2. Spatial Generalization

Tests unseen spatial configurations.

Example:

Training:

```text
Objects sampled from normal workspace
```

Testing:

```text
Objects placed in unseen positions
```

---

## 3. Object Attribute Generalization

Tests semantic understanding.

Training:

```text
Red cube → Yellow container

Blue cylinder → Green container
```

Testing:

```text
Red cylinder → Green container
```

The model must generalize across:

- Color
- Shape
- Target identity

---

## 4. Language Variation

Tests robustness to different instructions.

Training:

```text
"Place the red cube into the yellow container."
```

Testing:

```text
"Put the red block inside the yellow tray."
```

The task remains identical while language changes.

---

## 5. Novel Compositional Combination

Tests whether the model can combine learned concepts.

Training:

```text
Red cube → Yellow container

Blue cylinder → Green container
```

Testing:

```text
Red cylinder → Green container
```

---

## 6. Relational and Horizon Generalization (Stack)

Tests object-object reasoning, unseen color orders, and execution beyond the
training horizon.

Three-object training examples:

```text
[red, blue, green]
[green, yellow, red]
```

Three-object compositional test:

```text
[blue, red, green]
```

Four-object horizon test:

```text
[yellow, blue, red, green]
```

The model must understand:

```text
Bottom-to-top object order
        +
Repeated pick-and-place relations
        +
Intermediate stack state
        +
Execution beyond the three-object training horizon
```

The four-object evaluation supports the claim that the policy is tested on
multi-step manipulation beyond its primary training horizon. It is reported
separately from the three-object in-distribution success rate.

---

## 7. Visual Domain Shift (Optional)

Tests robustness against visual changes.

Variations:

- Lighting
- Camera viewpoint
- Object texture
- Background

---

# Ablation Studies

---

## 1. Action Chunk Size

Evaluate different action horizons:

```text
k = 1, 10, 20, 40
```

Goal:

Study the effect of action chunking on:

- Smoothness
- Stability
- Success rate

---

## 2. Demonstration Efficiency

Train models using different dataset sizes:

```text
25%
50%
100%
```

Goal:

Measure data efficiency.

---

## 3. Language Conditioning Ablation

Compare:

```text
Without language conditioning

vs

Language-conditioned policy
```

Goal:

Measure the contribution of language information.

---

## 4. Architecture Ablation

Compare:

```text
BC

ACT

MT-ACT
```

Goal:

Analyze the contribution of:

- Action chunking
- Transformer policy
- Multi-task conditioning

---

## 5. Domain Randomization Ablation (Optional)

Compare:

```text
MT-ACT

vs

MT-ACT + Domain Randomization
```

---

## 6. CVAE Ablation (Optional)

Because scripted demonstrations are relatively deterministic:

Compare:

```text
ACT + CVAE

vs

ACT without CVAE
```

---

# Project Timeline

## Week 1 — Simulation & Expert Controllers

Completed foundation:

- Setup MuJoCo and robosuite
- Configure Franka Panda
- Build Pick and Pick-and-Place environments
- Implement and evaluate PickExpert and PlaceExpert
- Extract reusable grasp, transport, and release primitives

Current Stack implementation order:

1. Build `StackEnv` with three cubes and optional fourth cube
2. Add a ten-reset environment test
3. Add generic named-object state access
4. Refactor PickExpert to accept `object_name`
5. Compute object-on-object destination poses using `transport_object()`
6. Implement StackExpert as a bottom-up orchestrator
7. Evaluate three-object expert success
8. Evaluate the four-object extension

Target:

```text
Validated environments and reliable scripted experts for every task
```

No demonstration recording begins until this target is reached.

---

## Week 2 — Dataset Generation & BC/ACT

Tasks:

- Build dataset recorder
- Generate demonstrations
- Implement dataset loader
- Train BC baseline
- Implement ACT
- Evaluate single-task benchmark

Target:

```text
BC vs ACT comparison
```

---

## Week 3 — Language Conditioning & MT-ACT

Tasks:

- Generate language instructions
- Integrate text encoder
- Implement:
  - Language-BC
  - Language-ACT
  - MT-ACT

Target:

```text
Multi-task language benchmark
```

---

## Week 4 — Benchmarking & Documentation

Tasks:

- Run generalization experiments
- Perform ablation studies
- Generate:
  - Training curves
  - Benchmark tables
  - Demo videos
  - Failure analysis

Target:

```text
Portfolio-ready robot learning project
```

---

# Repository Structure

```text
sim-mt-act/
├── configs/
├── controllers/
│   └── panda_controller.py
├── environments/
│   ├── pick_env.py             # random-color Lift task
│   ├── place_env.py            # random object/target colors
│   └── stack_env.py              # 3/4-cube stack task
├── experts/
│   ├── primitives/
│   │   ├── grasp.py
│   │   ├── release.py
│   │   ├── result.py
│   │   └── transport.py
│   ├── pick_expert.py
│   ├── place_expert.py
│   └── stack_expert.py           # bottom-up stack orchestrator
├── data/
├── models/
├── training/
├── benchmarks/
├── scripts/
│   ├── test_pick_expert.py
│   ├── test_place_expert.py
│   ├── test_stack_env.py         # random reset validation
│   └── test_stack_expert.py      # StackExpert evaluation
├── results/
└── README.md
```

---

# Final Deliverables

The completed project should contain:

- Custom MuJoCo/robosuite manipulation environment
- Scripted Pick, Place, and three-object Stack experts
- Four-object long-horizon Stack evaluation
- Demonstration generation pipeline created after expert validation
- Behavioral Cloning baseline
- ACT implementation
- Language-conditioned policies
- MT-ACT implementation
- Quantitative benchmark results
- Generalization evaluation
- Ablation studies
- Training visualization
- Failure analysis
- Demonstration videos

---

# References

Main methods and ideas are inspired by:

- **ACT**
  _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_
  RSS 2023

- **RoboAgent / MT-ACT**
  _Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking_
  ICRA 2024

- **VIMA**
  _General Robot Manipulation with Multimodal Prompts_
  ICML 2023

- **BAKU**
  _An Efficient Transformer for Multi-Task Policy Learning_
  NeurIPS 2024

---

# Status

```text
[x] Simulation setup

[x] Franka Panda environment

[x] Pick Expert

[x] Place Expert

[x] Reusable grasp / transport / release primitives

[x] Stack environment with three cubes

[x] Stack environment ten-reset test

[x] Generic named-object state API

[x] Pick Expert object_name support

[x] Stack destination via transport primitive

[x] Three-object Stack Expert

[ ] Four-object Stack evaluation extension

[x] Demonstration recorder with success filtering and RGB support

[ ] BC

[ ] ACT

[ ] Language-BC

[ ] Language-ACT

[ ] MT-ACT

[ ] Benchmarking

[ ] Final documentation
```
