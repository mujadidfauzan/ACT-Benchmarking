# Vast.ai Dataset and Training Runbook

This runbook reproduces the pipeline from a clean Vast.ai instance through
dataset generation, Language-BC training, per-task diagnosis, and checkpoint
evaluation.

The commands assume the repository is located at:

```text
/root/ACT-Benchmarking
```

The new dataset is named `pilot_v02`. It is recorded with robosuite's OpenCV
image convention and therefore **must not be vertically flipped**.

```text
pilot_v01 (legacy) -> OpenGL images -> vertical flip required
pilot_v02 (new)    -> OpenCV images -> no vertical flip
```

## 1. Vast.ai Instance

Recommended minimum:

- NVIDIA GPU with at least 12 GB VRAM
- 8 CPU cores
- 16 GB RAM
- 40 GB disk, preferably 60 GB
- A PyTorch CUDA image

Use a persistent volume or download all artifacts before destroying the
instance.

## 2. Repository and Environment

```bash
git clone <REPOSITORY_URL> ACT-Benchmarking
cd ACT-Benchmarking

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Do not install `requirements-ml-cpu.txt` on Vast.ai. It pins a CPU-only build
of PyTorch.

Keep the CUDA-enabled PyTorch supplied by the image and install the simulation
and language dependencies:

```bash
pip install \
  numpy==1.26.4 \
  scipy \
  mujoco==3.3.7 \
  robosuite==1.5.2 \
  opencv-python-headless \
  glfw h5py pynput qpsolvers quadprog mink PyYAML \
  sentence-transformers==6.1.0
```

Install a torchvision version compatible with the installed PyTorch. For the
tested Vast.ai environment using `torch==2.14.1+cu130`:

```bash
pip install torchvision==0.29.1 --no-deps
```

Verify the installation:

```bash
python - <<'PY'
import torch
import torchvision
from torchvision.models import resnet18

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("ResNet18: PASS")
PY
```

`torch.cuda.is_available()` must be `True`.

## 3. Expert Reliability Gate

Run the scripted experts before producing a new dataset:

```bash
MUJOCO_GL=egl python -m benchmarks.expert_reliability \
  --task all \
  --episodes 20 \
  --run-name vast_production_gate_20 \
  --enforce-gates
```

The command must complete without failing the configured Pick, Place, or Stack
gate.

## 4. Record `pilot_v02`

```bash
mkdir -p data/demos/pilot_v02
```

Pick:

```bash
MUJOCO_GL=egl python -m scripts.record_demonstrations \
  --task pick \
  --episodes 100 \
  --output data/demos/pilot_v02 \
  --camera-obs \
  --seed 1007 \
  --max-attempts 400
```

Place:

```bash
MUJOCO_GL=egl python -m scripts.record_demonstrations \
  --task place \
  --episodes 100 \
  --output data/demos/pilot_v02 \
  --camera-obs \
  --seed 2007 \
  --max-attempts 400
```

Stack:

```bash
MUJOCO_GL=egl python -m scripts.record_demonstrations \
  --task stack \
  --episodes 100 \
  --output data/demos/pilot_v02 \
  --camera-obs \
  --seed 3007 \
  --max-attempts 400
```

These three commands may run concurrently in three terminals because each task
writes to a different directory. Never run two recorders for the same task and
output directory concurrently.

Use `tmux` for long jobs:

```bash
tmux new -s record_pick
```

Detach with `Ctrl+B`, then `D`, and reconnect with:

```bash
tmux attach -t record_pick
```

Verify the episode count:

```bash
for task in pick place stack; do
  printf '%s: ' "$task"
  find "data/demos/pilot_v02/$task" -name 'episode_*.npz' | wc -l
done
```

Expected result: 100 episodes for every task, 300 total.

## 5. Audit the Dataset

```bash
python -m scripts.audit_dataset \
  --dataset data/demos/pilot_v02 \
  --mode full \
  --check-duplicates \
  --output results/dataset_audit/pilot_v02_full
```

Continue only when the audit reports:

```text
Audit status: PASS (0 errors)
```

## 6. Build the Multi-Task Semantic Split

```bash
python -m scripts.create_dataset_splits \
  --dataset data/demos/pilot_v02 \
  --output data/splits/pilot_v02_semantic \
  --protocol semantic \
  --seed 42
```

Inspect the result:

```bash
cat data/splits/pilot_v02_semantic/summary.json
```

## 7. Multi-Task Normalization

Statistics must come from the training split only:

```bash
python -m scripts.compute_normalization \
  --dataset data/demos/pilot_v02 \
  --split data/splits/pilot_v02_semantic/train.jsonl \
  --output data/splits/pilot_v02_semantic/normalization.json \
  --image-convention opencv
```

## 8. Frozen MiniLM Caches

Pooled sentence embeddings:

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.prepare_language_embeddings \
  --dataset data/demos/pilot_v02 \
  --split-dir data/splits/pilot_v02_semantic \
  --output data/splits/pilot_v02_semantic/language_embeddings.npz \
  --report-dir results/language_audit/pilot_v02_minilm \
  --device cpu
```

Token-level embeddings:

```bash
HF_HUB_DISABLE_XET=1 python -m scripts.prepare_language_token_embeddings \
  --dataset data/demos/pilot_v02 \
  --split-dir data/splits/pilot_v02_semantic \
  --output data/splits/pilot_v02_semantic/language_token_embeddings.npz \
  --report-dir results/language_audit/pilot_v02_minilm_tokens \
  --device cpu
```

Validate both caches:

```bash
python -m scripts.test_language_encoder \
  --cache data/splits/pilot_v02_semantic/language_embeddings.npz \
  --split-dir data/splits/pilot_v02_semantic

python -m scripts.test_token_language \
  --cache data/splits/pilot_v02_semantic/language_token_embeddings.npz
```

## 9. Multi-Task Language-BC

### 9.1 Pooled MiniLM

```bash
python -m scripts.train_language_bc \
  --dataset data/demos/pilot_v02 \
  --train-split data/splits/pilot_v02_semantic/train.jsonl \
  --val-split data/splits/pilot_v02_semantic/val.jsonl \
  --normalization data/splits/pilot_v02_semantic/normalization.json \
  --pooled-cache data/splits/pilot_v02_semantic/language_embeddings.npz \
  --token-cache data/splits/pilot_v02_semantic/language_token_embeddings.npz \
  --language-mode pooled \
  --output-dir results/language_bc/pilot_v02_pooled_seed42 \
  --epochs 30 \
  --batch-size 64 \
  --image-size 128 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --gradient-clip 1.0 \
  --num-workers 4 \
  --cache-size 2 \
  --seed 42 \
  --pretrained-visual \
  --unfreeze-layer4-epoch 8 \
  --no-vertical-flip \
  --device cuda
```

### 9.2 Token-Attention MiniLM

Run the same configuration with a separate output directory:

```bash
python -m scripts.train_language_bc \
  --dataset data/demos/pilot_v02 \
  --train-split data/splits/pilot_v02_semantic/train.jsonl \
  --val-split data/splits/pilot_v02_semantic/val.jsonl \
  --normalization data/splits/pilot_v02_semantic/normalization.json \
  --pooled-cache data/splits/pilot_v02_semantic/language_embeddings.npz \
  --token-cache data/splits/pilot_v02_semantic/language_token_embeddings.npz \
  --language-mode token_attention \
  --output-dir results/language_bc/pilot_v02_token_seed42 \
  --epochs 30 \
  --batch-size 64 \
  --image-size 128 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --gradient-clip 1.0 \
  --num-workers 4 \
  --cache-size 2 \
  --seed 42 \
  --pretrained-visual \
  --unfreeze-layer4-epoch 8 \
  --no-vertical-flip \
  --device cuda
```

The two jobs may run concurrently if CPU, storage bandwidth, and VRAM are
sufficient. Timing from concurrent runs must not be treated as a clean speed
benchmark.

### 9.3 Resume Training

Repeat the original command and append:

```bash
--resume results/language_bc/<RUN_NAME>/latest.pt
```

Keep all architecture, split, normalization, image, and optimizer arguments
unchanged. Use `best.pt` for evaluation and `latest.pt` only for resuming.

## 10. Build Per-Task Splits

Per-task training diagnoses whether failure comes from multi-task interference
or from one-step BC itself. Start with Pick. Continue to Place and Stack only
after interpreting Pick-only rollout results.

The following command creates a task-specific subset from the existing
semantic split. Set `TASK` to `pick`, `place`, or `stack`:

```bash
TASK=pick python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

task = os.environ["TASK"]
source = Path("data/splits/pilot_v02_semantic")
target = Path(f"data/splits/pilot_v02_{task}")
target.mkdir(parents=True, exist_ok=True)

summary = {"task": task, "splits": {}}
for split in ("train", "val", "test"):
    entries = [
        json.loads(line)
        for line in (source / f"{split}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    entries = [entry for entry in entries if entry["task"] == task]
    with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    summary["splits"][split] = {
        "episodes": len(entries),
        "steps": sum(entry["steps"] for entry in entries),
        "templates": dict(sorted(Counter(
            entry["instruction_template_id"] for entry in entries
        ).items())),
    }

(target / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY
```

For Pick, the expected split is 80 train, 10 validation, and 10 test episodes.
Place and Stack counts follow the semantic held-out split and may differ.

## 11. Per-Task Normalization

Set the task first:

```bash
TASK=pick
```

Then compute statistics only from that task's training episodes:

```bash
python -m scripts.compute_normalization \
  --dataset data/demos/pilot_v02 \
  --split "data/splits/pilot_v02_${TASK}/train.jsonl" \
  --output "data/splits/pilot_v02_${TASK}/normalization.json" \
  --image-convention opencv
```

Do not reuse the multi-task normalization for a per-task experiment.

## 12. Per-Task Language-BC Training

Start with pooled Pick-only:

```bash
TASK=pick

python -m scripts.train_language_bc \
  --dataset data/demos/pilot_v02 \
  --train-split "data/splits/pilot_v02_${TASK}/train.jsonl" \
  --val-split "data/splits/pilot_v02_${TASK}/val.jsonl" \
  --normalization "data/splits/pilot_v02_${TASK}/normalization.json" \
  --pooled-cache data/splits/pilot_v02_semantic/language_embeddings.npz \
  --token-cache data/splits/pilot_v02_semantic/language_token_embeddings.npz \
  --language-mode pooled \
  --output-dir "results/language_bc/pilot_v02_${TASK}_pooled_seed42" \
  --epochs 30 \
  --batch-size 64 \
  --image-size 128 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --gradient-clip 1.0 \
  --num-workers 4 \
  --cache-size 2 \
  --seed 42 \
  --pretrained-visual \
  --unfreeze-layer4-epoch 8 \
  --no-vertical-flip \
  --device cuda
```

After Pick, replace `TASK=pick` with `TASK=place` or `TASK=stack` only when the
experiment is justified. Do not compare per-task and multi-task validation MSE
directly because they use different normalization statistics.

## 13. Download Artifacts Before Stopping Vast.ai

Run these commands from the laptop, not from Vast.ai:

```bash
rsync -avP -e "ssh -p <PORT>" \
  root@<HOST>:/root/ACT-Benchmarking/results/language_bc/ \
  results/language_bc/
```

```bash
rsync -avP -e "ssh -p <PORT>" \
  root@<HOST>:/root/ACT-Benchmarking/data/splits/ \
  data/splits/
```

Required evaluation artifacts are:

- `best.pt`
- `config.json`
- `metrics.json`
- the matching `normalization.json`
- pooled and token language caches

The demonstration `.npz` files are not required for simulator rollout on the
laptop.

## 14. Closed-Loop Evaluation

Multi-task token model example:

```bash
MUJOCO_GL=egl python -m scripts.evaluate_language_bc \
  --checkpoint results/language_bc/pilot_v02_token_seed42/best.pt \
  --normalization data/splits/pilot_v02_semantic/normalization.json \
  --pooled-cache data/splits/pilot_v02_semantic/language_embeddings.npz \
  --token-cache data/splits/pilot_v02_semantic/language_token_embeddings.npz \
  --task all \
  --episodes 20 \
  --record-video \
  --output-dir results/language_bc_evaluation/token_20 \
  --device cpu
```

Pick-only example:

```bash
MUJOCO_GL=egl python -m scripts.evaluate_language_bc \
  --checkpoint results/language_bc/pilot_v02_pick_pooled_seed42/best.pt \
  --normalization data/splits/pilot_v02_pick/normalization.json \
  --pooled-cache data/splits/pilot_v02_semantic/language_embeddings.npz \
  --token-cache data/splits/pilot_v02_semantic/language_token_embeddings.npz \
  --task pick \
  --episodes 20 \
  --record-video \
  --output-dir results/language_bc_evaluation/pick_only_pooled_20 \
  --device cpu
```

Every output directory contains:

```text
summary.json
pick_attempts.jsonl / place_attempts.jsonl / stack_attempts.jsonl
videos/*.mp4 (when --record-video is enabled)
```

Use a fresh output directory for every evaluation. Compare success rate,
failure reasons, and `mean_action_clip_fraction`; validation MSE alone is not a
closed-loop task metric.

## 15. Experiment Decision Rule

Interpret Pick-only first:

```text
Pick-only succeeds, multi-task fails
    -> multi-task interference is a major bottleneck

Pick-only also fails
    -> one-step BC, action averaging, and closed-loop drift are the likely
       bottlenecks; proceed to Language-ACT rather than training every task
```

Keep dataset version, split hash, normalization file, seed, image convention,
checkpoint epoch, and language mode in every experiment report.
