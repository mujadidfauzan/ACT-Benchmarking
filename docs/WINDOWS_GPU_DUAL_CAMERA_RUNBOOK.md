# Windows GPU Dual-Camera Training Runbook

This runbook starts from a clean native Windows machine and trains the current
Pick-only Language-BC policy with:

- `agentview`
- `robot0_eye_in_hand`
- 23D proprioception, including EEF pose
- frozen pooled MiniLM
- language-conditioned spatial attention
- phase-balanced sampling

Commands below use **Anaconda Prompt / CMD**, not PowerShell. CMD uses `^` for
line continuation and `set NAME=value` for environment variables.

## 0. Publish the Latest Pipeline

Before cloning or pulling on Windows, commit and push the current dual-camera
changes from the development machine:

```bash
git status
git add -u
git add docs/WINDOWS_GPU_DUAL_CAMERA_RUNBOOK.md
git commit -m "add dual-camera spatial language BC pipeline"
git push
```

Review `git status` before committing; do not include generated datasets,
checkpoints, or unrelated local files.

## 1. Clone the Repository

```bat
cd /d E:\Fauzan
git clone <REPOSITORY_URL> ACT-Benchmarking
cd /d E:\Fauzan\ACT-Benchmarking
```

For an existing clone:

```bat
cd /d E:\Fauzan\ACT-Benchmarking
git pull
```

## 2. Create the Conda Environment

```bat
conda create -p E:\Fauzan\conda-envs\sim-mt-act python=3.10 -y
conda activate E:\Fauzan\conda-envs\sim-mt-act
python -m pip install --upgrade pip setuptools wheel
```

Do not install `requirements-ml-cpu.txt` on this machine because it installs a
CPU-only PyTorch build.

Install CUDA-enabled PyTorch. CUDA 12.8 wheels are suitable for the current
Windows setup:

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bat
python -m pip install numpy==1.26.4 scipy mujoco==3.3.7 robosuite==1.5.2 opencv-python glfw h5py pynput qpsolvers quadprog mink PyYAML sentence-transformers==6.1.0
```

Check dependency consistency:

```bat
python -m pip check
```

## 3. Configure the Current CMD Session

Run these commands whenever a new CMD session is opened:

```bat
conda activate E:\Fauzan\conda-envs\sim-mt-act
cd /d E:\Fauzan\ACT-Benchmarking
set MUJOCO_GL=wgl
set HF_HUB_DISABLE_XET=1
```

## 4. Verify CUDA and Imports

```bat
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

`CUDA available` must be `True`.

Verify the project imports:

```bat
python -c "import mujoco, robosuite, sentence_transformers; from models.language_bc import LanguageBCPolicy; print('Project imports: PASS')"
```

## 5. Run the Pick Expert Gate

```bat
python -m benchmarks.expert_reliability ^
  --task pick ^
  --episodes 20 ^
  --run-name pilot_v04_pick_gate_20 ^
  --enforce-gates
```

Continue only if the expert passes its reliability gate.

## 6. Record a 20-Episode Dual-Camera Debug Dataset

```bat
python -m scripts.record_demonstrations ^
  --task pick ^
  --episodes 20 ^
  --output data/demos/pilot_v04_debug ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 4007 ^
  --max-attempts 80
```

Audit both camera streams:

```bat
python -m scripts.audit_dataset ^
  --dataset data/demos/pilot_v04_debug ^
  --tasks pick ^
  --mode full ^
  --check-duplicates ^
  --camera-names agentview robot0_eye_in_hand ^
  --output results/dataset_audit/pilot_v04_debug
```

Continue only when the result is:

```text
Audit status: PASS (0 errors)
```

## 7. Record the 100-Episode Production Dataset

Use a fresh directory. Do not mix debug episodes into production.

```bat
python -m scripts.record_demonstrations ^
  --task pick ^
  --episodes 100 ^
  --output data/demos/pilot_v04 ^
  --camera-obs ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 4107 ^
  --max-attempts 400
```

Verify the count:

```bat
dir /b data\demos\pilot_v04\pick\episode_*.npz | find /c /v ""
```

Expected result: `100`.

Audit production data:

```bat
python -m scripts.audit_dataset ^
  --dataset data/demos/pilot_v04 ^
  --tasks pick ^
  --mode full ^
  --check-duplicates ^
  --camera-names agentview robot0_eye_in_hand ^
  --output results/dataset_audit/pilot_v04_full
```

## 8. Create the Pick-Only IID Split

```bat
python -m scripts.create_dataset_splits ^
  --dataset data/demos/pilot_v04 ^
  --output data/splits/pilot_v04_pick_iid ^
  --tasks pick ^
  --protocol iid ^
  --seed 42
```

The expected episode split is `80 train / 10 val / 10 test`.

## 9. Prepare Pooled MiniLM Embeddings

```bat
python -m scripts.prepare_language_embeddings ^
  --dataset data/demos/pilot_v04 ^
  --split-dir data/splits/pilot_v04_pick_iid ^
  --output data/splits/pilot_v04_pick_iid/language_embeddings.npz ^
  --report-dir results/language_audit/pilot_v04_pick_pooled ^
  --device cpu
```

This step downloads MiniLM only when it is absent from the Hugging Face cache.

## 10. Compute 23D Normalization

```bat
python -m scripts.compute_normalization ^
  --dataset data/demos/pilot_v04 ^
  --split data/splits/pilot_v04_pick_iid/train.jsonl ^
  --output data/splits/pilot_v04_pick_iid/normalization_23d.json ^
  --image-convention opencv
```

Validate it:

```bat
python -m scripts.test_normalization ^
  --statistics data/splits/pilot_v04_pick_iid/normalization_23d.json ^
  --split data/splits/pilot_v04_pick_iid/train.jsonl ^
  --dataset data/demos/pilot_v04
```

The output must include `Proprio shape: (23,)`.

## 11. Smoke-Test the Dual-Camera Pipeline

Dataset loader:

```bat
python -m scripts.test_dataset_loader ^
  --dataset data/demos/pilot_v04 ^
  --split data/splits/pilot_v04_pick_iid/train.jsonl ^
  --batch-size 4 ^
  --num-workers 0 ^
  --camera-names agentview robot0_eye_in_hand ^
  --no-vertical-flip
```

The authoritative model gate for this pooled-only run is a tiny overfit on real
dual-camera samples. It checks the complete loader, normalization, two visual
inputs, language cache, forward pass, backward pass, and checkpoint path without
requiring an unused token cache:

```bat
python -m scripts.tiny_overfit_language_bc ^
  --dataset data/demos/pilot_v04 ^
  --split data/splits/pilot_v04_pick_iid/train.jsonl ^
  --normalization data/splits/pilot_v04_pick_iid/normalization_23d.json ^
  --pooled-cache data/splits/pilot_v04_pick_iid/language_embeddings.npz ^
  --language-mode pooled ^
  --visual-fusion spatial_attention ^
  --camera-names agentview robot0_eye_in_hand ^
  --no-vertical-flip ^
  --steps 300 ^
  --batch-size 4 ^
  --image-size 128 ^
  --output-dir results/tiny_overfit/pilot_v04_pick_dual_pooled
```

Continue only if it reports `PASS`.

## 12. Train on the GPU

```bat
python -m scripts.train_language_bc ^
  --dataset data/demos/pilot_v04 ^
  --train-split data/splits/pilot_v04_pick_iid/train.jsonl ^
  --val-split data/splits/pilot_v04_pick_iid/val.jsonl ^
  --normalization data/splits/pilot_v04_pick_iid/normalization_23d.json ^
  --pooled-cache data/splits/pilot_v04_pick_iid/language_embeddings.npz ^
  --language-mode pooled ^
  --visual-fusion spatial_attention ^
  --camera-names agentview robot0_eye_in_hand ^
  --phase-balanced ^
  --output-dir results/language_bc/pilot_v04_pick_dual_pooled_seed42 ^
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
  --unfreeze-layer4-epoch 8 ^
  --no-vertical-flip ^
  --device cuda
```

If Windows DataLoader workers fail or consume too much RAM, rerun with:

```bat
--num-workers 0
```

Do not resume from a `pilot_v03` checkpoint. Its image inputs and model shape
do not match the dual-camera policy.

## 13. Evaluate 20 Closed-Loop Episodes

```bat
python -m scripts.evaluate_language_bc ^
  --checkpoint results/language_bc/pilot_v04_pick_dual_pooled_seed42/best.pt ^
  --normalization data/splits/pilot_v04_pick_iid/normalization_23d.json ^
  --pooled-cache data/splits/pilot_v04_pick_iid/language_embeddings.npz ^
  --task pick ^
  --episodes 20 ^
  --instruction-split train ^
  --strict-language-cache ^
  --device cuda ^
  --record-video ^
  --output-dir results/language_bc_evaluation/pilot_v04_pick_dual_pooled_20
```

The evaluator reads the two camera names from the checkpoint. Videos contain
`agentview` on the left and `robot0_eye_in_hand` on the right.

## 14. Resume an Interrupted Training Run

Repeat the exact training command and append:

```bat
--resume results/language_bc/pilot_v04_pick_dual_pooled_seed42/latest.pt
```

The split, normalization, language mode, camera list, visual fusion, and image
size must remain unchanged.
