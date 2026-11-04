# Benchmarking Visuomotor Models for Robotic Manipulation

**Scene 1 — All Models Succeed**

https://github.com/user-attachments/assets/1dff3e0a-3334-4f0c-9ce5-751efd00504f

**Scene 2 — Contrasting Outcomes**

https://github.com/user-attachments/assets/1bf43f87-f7f1-411c-8a0d-22e479f00f16

This project provides a simulation benchmark comparing several behavior cloning models for Franka Panda manipulation in MuJoCo/robosuite. Five policy configurations are trained on a shared expert demonstration dataset and evaluated on matched scenes to examine task success, completion time, and failure modes. The project provides an integrated workflow for demonstration collection, policy training, closed-loop evaluation, and trajectory analysis.

The intended tasks are **Pick-and-Place** and **Cube Stacking**. The results reported here concern Pick-and-Place. Stacking environments and scripted experts are available, while integration into the training benchmark remains future work. **Language-conditioned manipulation** is also planned; instructions are currently stored as metadata and are not policy inputs.

## Policy Architectures

All policies receive RGB observations and robot proprioception and produce seven-dimensional operational-space delta commands: XYZ translation, rotation, and gripper control, executed at **20 Hz**.

| Model | Visual representation                                       | Temporal model / output                        | Training objective                        |
| ----- | ----------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------- |
| **A** | Four-layer CNN + 4×4 average pooling                        | GRU-256; single-step action                    | Normalized motion MSE + 0.2 × gripper BCE |
| **B** | Four-layer CNN + Spatial Softmax                            | GRU-256; single-step action                    | Same as A                                 |
| **C** | ResNet18 + Spatial Softmax                                  | GRU-256; single-step action                    | Same as A                                 |
| **D** | robomimic-style ResNet18 + 32 learned spatial keypoints     | Two-layer LSTM-1000; deterministic action head | Continuous 7D action MSE                  |
| **E** | ImageNet-initialized ResNet18 + Transformer encoder–decoder | CVAE; 32-action chunks                         | Masked action L1 + 10 × KL divergence     |

**A–C** use independent camera encoders, 256-dimensional features per camera, 16-dimensional proprioception, and the previous gripper action. Training uses 16-step windows with stride 8. The visual backbones are trained from scratch.

**D** implements a deterministic image BC-RNN recipe inspired by [robomimic](https://github.com/ARISE-Initiative/robomimic). Images are resized to 84×84 and cropped to 76×76, with random crops during training and center crops during evaluation. Each camera uses a separately trained ResNet18 and a 64-dimensional keypoint representation. The policy receives 9-dimensional EEF/gripper proprioception and uses a continuous `tanh` action head. Training sequences and the evaluation recurrent-state reset interval are both 10 steps.

**E** adapts the [LeRobot ACT core](https://github.com/huggingface/lerobot/blob/cc2fbf08976c6edd14edb001616ae79f3121de96/src/lerobot/policies/act/modeling_act.py), pinned to commit `cc2fbf08976c6edd14edb001616ae79f3121de96`. It uses a shared ResNet18 with frozen BatchNorm statistics, 16-dimensional proprioception, hidden dimension 512, eight attention heads, four encoder layers, one decoder layer, and a 32-dimensional latent. The reported evaluation uses a zero latent and replans every step, executing the first predicted action without temporal ensembling. ACT is adapted to the project's end-effector controller rather than the original ALOHA joint-control interface.

## Experimental Protocol

The dataset contains **1,000 successful scripted demonstrations**, split by episode into **900 training and 100 validation episodes**. Scenes contain one red cube and one red tray on opposite sides of the workspace. Their positions and orientations are randomized, with an initial XY center-distance limit of **0.20 m**. Observations include `agentview` and wrist-camera RGB at 128×128. Normalization statistics, where used, are computed exclusively from the training split. All policies read the native HDF5 episodes without conversion to external dataset formats.

Final evaluation comprises **100 episodes per model**, using five environment seeds (2000–2004) and 20 episodes per seed. All models use an **800-step horizon**, equivalent to 40 seconds of simulated control. Recorded initial cube, tray, and EEF positions match across models for each seed–episode pair. These are randomized evaluation scenes, distinct from rollouts reconstructed from the validation dataset.

Success requires valid placement in the target tray, the evaluator's release condition, an open-gripper command, and stable success for **20 consecutive steps**. Release is approximated by an EEF–object distance greater than 8 cm. Thus, visible placement alone may not satisfy the benchmark criterion.

Reported policies use **training seed 123** and checkpoints selected by periodic rollout success during training. Selected checkpoints occur at different training stages. The comparison evaluates complete policy configurations, including preprocessing, proprioception, loss, and pretraining, rather than isolating architecture alone.

## Results and Analysis

| Model                      | Successful episodes | Success rate | Across-seed SD (percentage points) | Median steps, successful episodes |
| -------------------------- | ------------------: | -----------: | ---------------------------------: | --------------------------------: |
| A — CNN Average + GRU      |              61/100 |          61% |                               10.8 |                               300 |
| B — CNN Spatial + GRU      |              56/100 |          56% |                               10.8 |                               249 |
| C — ResNet18 Spatial + GRU |              86/100 |          86% |                               12.4 |                               228 |
| D — BC-RNN                 |              80/100 |          80% |                                9.4 |                             190.5 |
| E — ACT                    |              88/100 |          88% |                                5.7 |                             235.5 |

SD is the sample standard deviation of success rates across five evaluation seeds. It describes scene-dependent variation for these checkpoints, not variability across independently trained policies. Completion steps are conditional on success and should be interpreted together with success rate.

**Key Takeaways.** Under the shared 800-step evaluation horizon, ACT achieves the highest observed success rate (88%), closely followed by ResNet18-GRU (86%). BC-RNN has the lowest median completion time among successful episodes. Failure to lift the object is the dominant failure mode across all five models.

**Success and completion time.** ACT achieves the highest observed success rate, while C provides a strong balance between success and completion time. D has the lowest median completion time among its successful episodes. The two-episode difference between C and ACT is insufficient to establish general superiority: both succeed on 76 scenes, C alone on 10, and ACT alone on 12.

**Failure modes.** Failure to lift the object dominates across configurations. This category indicates that the object did not reach the lift threshold; it does not identify whether alignment, grasp timing, or contact mechanics caused the failure.

| Model | No lift | Placement timeout | Not released / open-command criterion unmet |
| ----- | ------: | ----------------: | ------------------------------------------: |
| A     |      31 |                 8 |                                           0 |
| B     |      34 |                10 |                                           0 |
| C     |      11 |                 2 |                                           1 |
| D     |      14 |                 6 |                                           0 |
| E     |      10 |                 2 |                                           0 |

**Interpretation and limitations.** The stronger performance of C relative to A/B supports the ResNet-based configuration in this setting, without establishing a causal explanation for the gain. Validation loss remains a diagnostic measure: prediction error on expert observations does not directly measure closed-loop task completion, and losses with different objectives or normalization are not numerically comparable. Results are limited to one training seed, one scene distribution, and scripted demonstrations. They do not establish robustness across training runs, stacking tasks, language instructions, or physical robots.

## Project Structure

```text
benchmarks/                 Expert reliability checks
configs/                    Scene and semantic configuration
controllers/                Panda control interface
environments/               Pick, Place, and Stack tasks
experts/                    Scripted demonstration policies
data/                       HDF5 I/O, recording, splits, and normalization
models/
  visual_encoder.py         CNN / ResNet18 and spatial pooling
  visual_bc.py              Models A–C
  robomimic_bc_rnn/          Model D
  act_lerobot/              Model E and attributed reference core
  act/                      Earlier ACT implementation and chunk utilities
  low_dim_bc_rnn/            Diagnostic low-dimensional baseline
scripts/                    Collection, training, evaluation, and visualization
diagnostics/                Camera checks
tests/                      Regression and simulation checks
```

## Usage

Run from the repository root with Python 3.10 and a compatible `torch` / `torchvision` GPU build. Install dependencies and enable headless rendering:

```bash
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
```

MuJoCo rendering requires EGL / OpenGL system libraries. On Ubuntu, install `libegl1 libgl1 libglfw3`; FFmpeg is optional for H.264 documentation videos.

### 1. Record Demonstrations

```bash
python -m scripts.record_demonstrations \
  --task place --num-demos 1000 \
  --num-objects 1 --num-receptacles 1 --fixed-target-color red \
  --placement-layout opposite_sides --max-object-receptacle-distance 0.20 \
  --camera-names agentview robot0_eye_in_hand --image-size 128 \
  --seed 42 --max-attempts 5000 \
  --output data/demos/place_red_1cube_1tray \
  --training-dir data/splits/place_red_1cube_1tray
```

Collection automatically prepares the 90/10 episode split, train-only statistics, and camera/controller configuration. All models read the native HDF5 dataset directly. Repeat the same command to complete interrupted collection with unchanged settings.

### 2. Train A–C

Example for A; change the backbone, pooling, and output name using the table for B or C.

| Model | Backbone   | Pooling           | Output run name                          |
| ----- | ---------- | ----------------- | ---------------------------------------- |
| A     | `cnn`      | `avg`             | `place_red_1000_d20_A_avg_gru_s123`      |
| B     | `cnn`      | `spatial_softmax` | `place_red_1000_d20_B_spatial_gru_s123`  |
| C     | `resnet18` | `spatial_softmax` | `place_red_1000_d20_C_resnet18_gru_s123` |

```bash
python -m scripts.train_visual_bc \
  --dataset data/demos/place_red_1cube_1tray \
  --training-dir data/splits/place_red_1cube_1tray \
  --output-dir results/visual_bc/place_red_1000_d20_A_avg_gru_s123 \
  --temporal --visual-backbone cnn --visual-pooling avg \
  --sequence-length 16 --sequence-stride 8 \
  --epochs 100 --batch-size 8 --seed 123 --early-stopping-patience 0 \
  --rollout-eval-every 10 --rollout-eval-episodes 10 \
  --rollout-eval-num-seeds 1 --rollout-eval-seed 1000 \
  --rollout-eval-max-steps 300 --device cuda
```

### 3. Train D: BC-RNN

```bash
python -m scripts.train_robomimic_bc_rnn \
  --dataset data/demos/place_red_1cube_1tray \
  --training-dir data/splits/place_red_1cube_1tray \
  --output-dir results/robomimic_bc_rnn/place_red_1000_d20_D_bcrnn_s123 \
  --max-updates 324700 --batch-size 8 --seed 123 \
  --rollout-eval-every 32470 --rollout-eval-episodes 10 \
  --rollout-eval-num-seeds 1 --rollout-eval-seed 1000 \
  --rollout-eval-max-steps 300 --device cuda
```

### 4. Train E: ACT

Download and verify the pretrained backbone once:

```bash
mkdir -p checkpoints/pretrained
curl -fL --retry 3 https://download.pytorch.org/models/resnet18-f37072fd.pth \
  -o checkpoints/pretrained/resnet18-f37072fd.pth
echo "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec  checkpoints/pretrained/resnet18-f37072fd.pth" | sha256sum -c -
```

```bash
python -m scripts.train_act_lerobot \
  --dataset data/demos/place_red_1cube_1tray \
  --training-dir data/splits/place_red_1cube_1tray \
  --output-dir results/act_lerobot/place_red_1000_d20_ACT_LR_s123 \
  --backbone-weights checkpoints/pretrained/resnet18-f37072fd.pth \
  --batch-size 8 --chunk-size 32 --seed 123 \
  --max-updates 324700 --checkpoint-every 500 --log-every 50 \
  --rollout-eval-every 32470 --rollout-eval-episodes 10 \
  --rollout-eval-num-seeds 1 --rollout-eval-seed 1000 \
  --rollout-eval-max-steps 300 --device cuda
```

These commands illustrate a 100-epoch A–C schedule and a nominally matched 324,700-update budget for D/E (3,247 A–C updates per epoch on this dataset). They do not imply equal dataset passes, computation, or identical training histories for the reported checkpoints. Training rollout intervals use epochs for A–C and updates for D/E.

### 5. Evaluate

Example for E: **5 seeds × 20 episodes, 800 steps maximum**. For A–D, replace the evaluator and checkpoint using the table and choose a separate output directory.

| Model     | Evaluator module                    | Checkpoint directory              |
| --------- | ----------------------------------- | --------------------------------- |
| A / B / C | `scripts.evaluate_visual_bc`        | `results/visual_bc/<run>/`        |
| D         | `scripts.evaluate_robomimic_bc_rnn` | `results/robomimic_bc_rnn/<run>/` |
| E         | `scripts.evaluate_act_lerobot`      | `results/act_lerobot/<run>/`      |

```bash
python -m scripts.evaluate_act_lerobot \
  --checkpoint results/act_lerobot/place_red_1000_d20_ACT_LR_s123/best_rollout.pt \
  --output-dir results/act_lerobot_evaluation/place_red_1000_d20_ACT_LR_s123_h800 \
  --episodes 20 --num-seeds 5 --seed 2000 --max-steps 800 \
  --device cuda --record-video --log-rollout
```

Use `latest.pt` to resume training, `best.pt` for the best validation-loss checkpoint, and `best_rollout.pt` for the best training-rollout checkpoint. Evaluation checkpoints embed the policy weights and input configuration. Use a new output directory for each evaluation.

Additional tools: [validation-scene comparison](scripts/evaluate_visual_bc_validation.py) and [high-resolution video recording](scripts/record_policy_evaluation.py). Run `python -m <module> --help` for options.

### Verification

```bash
python -m tests
git diff --check
```

Datasets, pretrained weights, checkpoints, and evaluation artifacts are excluded from Git and must be generated or obtained separately.

## References and Citation

### robomimic / BC-RNN

Model D is inspired by:

- [robomimic](https://github.com/ARISE-Initiative/robomimic)
- _What Matters in Learning from Offline Human Demonstrations for Robot Manipulation_

```bibtex
@inproceedings{robomimic2021,
  title     = {What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author    = {Mandlekar, Ajay and Xu, Danfei and Wong, Josiah and Nasiriany, Soroush and Wang, Chen and Kulkarni, Rohun and Fei-Fei, Li and Savarese, Silvio and Zhu, Yuke and Mart{\'i}n-Mart{\'i}n, Roberto},
  booktitle = {Conference on Robot Learning},
  year      = {2021}
}
```

### Action Chunking Transformer

Model E builds on:

- [ACT](https://github.com/tonyzhaozh/act)
- _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_
- [LeRobot ACT implementation used by this project](https://github.com/huggingface/lerobot/blob/cc2fbf08976c6edd14edb001616ae79f3121de96/src/lerobot/policies/act/modeling_act.py)

```bibtex
@article{zhao2023learning,
  title   = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author  = {Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal = {arXiv preprint arXiv:2304.13705},
  year    = {2023}
}
```
