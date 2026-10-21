# Linux: Visual Behavior Cloning

Jalankan dari root repo dalam Bash. Model Visual-BC menerima RGB dan proprio
robot untuk memprediksi satu action. Storage episode memakai HDF5, dan training saat ini
khusus fixed-target Pick.

## 1. Setup

Python 3.10, Git, dan library renderer harus tersedia. Pada Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv libegl1 libgl1 libglfw3 tmux
git clone https://github.com/mujadidfauzan/ACT-Benchmarking.git
cd ACT-Benchmarking
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); x=torch.randn(128,128,device='cuda'); print((x@x).mean().item())"
python -m tests
```

Pilih wheel melalui [selector PyTorch](https://pytorch.org/get-started/locally/)
bila hardware berbeda. Untuk CPU, ganti index dengan
`https://download.pytorch.org/whl/cpu` dan lewati tes tensor CUDA. Jangan install
requirements CPU pada environment GPU. Bila Python 3.10 belum tersedia, Conda
dapat menggantikan pembuatan venv. ROS tidak dibutuhkan.

EGL untuk rendering offscreen server GPU. GUI memakai backend glfw dan display
yang tersedia; jangan memakai `--render` tanpa dukungan display. Training CPU
tidak membutuhkan renderer; simulasi CPU headless memerlukan backend/library
seperti OSMesa yang terpasang sesuai mesin.

## 2. Gate Expert

```bash
python -m benchmarks.expert_reliability \
  --task pick --episodes 60 --object-counts 1 2 3 \
  --fixed-target-color red --run-name pick_gate --enforce-gates
```

Pick memakai state machine: approach 0.12 m, align 0.08 m, descend 0.005 m,
close hold 15 langkah, lalu lift 0.20 m. Target harus naik minimal 0.18 m
selama 20 langkah berurutan dan lolos `env._check_success()`. Delta orientasi
berada pada world frame dan mengikuti yaw objek. Place/Stack tetap memakai expert
task-specific masing-masing. Pick mendukung `--rotation-mode none`: command
rotasi nol pada semua fase dan expert tidak menunggu alignment.
Gunakan run name baru; gate 95% Pick diperiksa per count group, bukan sampler
training. Tambahkan `--render` hanya pada mesin dengan viewer yang tersedia.

## 3. Collect Dan Persiapan Otomatis

```bash
DATASET=data/demos/pick_red_v01
RUN=results/visual_bc/pick_red_v01_seed123
python -m scripts.record_demonstrations \
  --task pick --num-demos 300 --output "$DATASET" \
  --object-counts 1 2 3 --fixed-target-color red \
  --camera-names agentview robot0_eye_in_hand \
  --seed 42 --max-attempts 1200
```

Gunakan dataset baru untuk controller/metode baru. RGB otomatis aktif, 128x128,
OpenCV tanpa vertical flip. Untuk satu kamera gunakan `--camera-names agentview`.
Dataset pilot, split, dan cache eksperimen lama sudah dibersihkan dari workspace.
Collection mengisi kembali `data/demos/` dan `data/splits/`; artifact hasil
generate tidak disertakan dalam Git.
Pick secara default memakai target merah. Pertahankan satu warna target untuk
training Visual-BC; opsi collection `--random-target` tidak didukung trainer.
Jangan merekam paralel ke dataset yang sama karena persiapan membaca seluruh
manifest setelah collect. Objek 1-3 dapat bervariasi; ini variasi scene, bukan
balancing task/phase saat training.

`--num-demos` adalah target total sukses per task. Ulangi command yang sama
untuk melengkapi recording terputus. Opsi lama `--episodes N` masih menambah N
episode; jangan menggabungkan keduanya. Demo gagal dan frame korup ditolak.
HDF5 ditulis atomik sebelum manifest; orphan akibat interupsi perlu diperiksa
dan dikarantina, bukan dimasukkan ke manifest secara sembarang.

Collection menghasilkan episode T actions/T+1 observations di DATASET/pick,
serta manifest sukses dan attempts untuk seluruh percobaan. Setelah selesai,
otomatis ditulis di `data/splits/pick_red_v01_bc/`:

- `train.jsonl` dan `val.jsonl`: split episode 90/10, seed 123.
- `normalization.json`: statistik **train-only**, tanpa task balancing.
- `training.json`: input state, kamera, orientation mode, reset/controller,
  checksum sumber, dan konfigurasi yang dibutuhkan trainer.

Tidak perlu command create-splits atau compute-normalization. Minimal dua
episode sukses dibutuhkan. Lokasi dapat diubah dengan `--training-dir PATH`.
Data lama yang lengkap dapat dipersiapkan otomatis oleh trainer bila artifact
belum ada, tetapi gunakan recording baru untuk mencocokkan expert/controller.

### Penyimpanan HDF5 Dan Dataset Lama

Satu episode disimpan dalam `episode_000000.hdf5`, dengan array `actions`,
`phases`, `obs__...`, `metadata_json`, dan `result_json`. RGB memakai gzip level 4
dan chunk satu frame. Loader membaca satu timestep atau window GRU, bukan semua
RGB episode. Cache dibatasi 16 file handle per dataset/proses dan tidak dibagikan
antar-worker. Pemeriksaan kamera memakai blok frame terbatas; statistik tetap
train-only.

NPZ lama harus dikonversi ke dataset baru sebelum training. Sumber tidak
dimodifikasi, split/normalisasi dibuat otomatis, dan output harus belum ada:

```bash
python -m data.episode_io \
  --source data/demos/pick_red_v01 \
  --output data/demos/pick_red_v01_hdf5 \
  --training-dir data/splits/pick_red_v01_hdf5_bc
```

Lalu gunakan dataset `_hdf5` untuk training di run baru. Fingerprint berbeda,
sehingga checkpoint training lama tidak bisa di-resume pada hasil konversi.
Checkpoint mandiri lama tetap bisa dievaluasi karena arsitektur tidak berubah.
Jangan mencampur NPZ dan HDF5 dalam direktori collection. Command enam eksperimen
tetap sama; file baru otomatis HDF5. Converter ini khusus schema NPZ project ini,
bukan importer HDF5 dari repo lain.

Audit opsional:

```bash
python -m scripts.audit_dataset \
  --dataset "$DATASET" --tasks pick --mode full --check-duplicates \
  --camera-names agentview robot0_eye_in_hand \
  --output results/dataset_audit/pick_red_v01
python -m scripts.audit_pose_distribution \
  --dataset "$DATASET" --split-dir data/splits/pick_red_v01_bc \
  --output results/pose_distribution/pick_red_v01
```

Audit pose menerima train/val bila tidak ada test. Baseline ini memang 90/10,
bukan 80/10/10. Evaluasi final membuat scene baru, bukan replay episode val.

### Enam Eksperimen Kamera Dan Orientasi

Posisi sampling tetap x `[-0.10, 0.10]` m, y `[-0.20, 0.20]` m. Yaw memakai
toggle tanpa input angka dan berlaku pada semua kubus. `--random-yaw` (default)
berarti uniform -45 sampai +45 derajat; `--no-random-yaw` berarti tetap nol.
Untuk mengisolasi kamera/orientasi,
gunakan tiga objek, target merah, seed collection sama, dan enam dataset baru.

Tanpa eye-in-hand:

```bash
python -m scripts.record_demonstrations \
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red \
  --camera-names agentview --no-random-yaw --rotation-mode none \
  --seed 42 --max-attempts 1200 --output data/demos/pick_ext_yaw0_none
python -m scripts.record_demonstrations \
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red \
  --camera-names agentview --random-yaw --rotation-mode none \
  --seed 42 --max-attempts 1200 --output data/demos/pick_ext_yaw45_none
python -m scripts.record_demonstrations \
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red \
  --camera-names agentview --random-yaw --rotation-mode align_yaw \
  --seed 42 --max-attempts 1200 --output data/demos/pick_ext_yaw45_align
```

Dengan eye-in-hand:

```bash
python -m scripts.record_demonstrations \
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red \
  --camera-names agentview robot0_eye_in_hand --no-random-yaw --rotation-mode none \
  --seed 42 --max-attempts 1200 --output data/demos/pick_dual_yaw0_none
python -m scripts.record_demonstrations \
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red \
  --camera-names agentview robot0_eye_in_hand --random-yaw --rotation-mode none \
  --seed 42 --max-attempts 1200 --output data/demos/pick_dual_yaw45_none
python -m scripts.record_demonstrations \
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red \
  --camera-names agentview robot0_eye_in_hand --random-yaw --rotation-mode align_yaw \
  --seed 42 --max-attempts 1200 --output data/demos/pick_dual_yaw45_align
```

Jalankan gate ketiga konfigurasi fisik sebelum collect besar:

```bash
python -m benchmarks.expert_reliability --task pick --episodes 60 --num-objects 3 --fixed-target-color red --no-random-yaw --rotation-mode none --run-name pick_yaw0_none_gate --enforce-gates
python -m benchmarks.expert_reliability --task pick --episodes 60 --num-objects 3 --fixed-target-color red --random-yaw --rotation-mode none --run-name pick_yaw45_none_gate --enforce-gates
python -m benchmarks.expert_reliability --task pick --episodes 60 --num-objects 3 --fixed-target-color red --random-yaw --rotation-mode align_yaw --run-name pick_yaw45_align_gate --enforce-gates
```

`none` menjamin command rotasi nol, bukan orientasi fisik EEF selalu konstan:
joint reset noise 3 derajat dan dinamika tetap dipertahankan. Grasp tanpa
rotasi pada kubus miring mungkin lebih sering gagal. Filter sukses bisa
menyebabkan distribusi scene berbeda antar-mode meskipun seed sama.

Metadata episode dan `training.json` menyimpan `rotation_mode`,
`control_orientation`, `collection_config.yaw_range_deg`, dan environment aktual.
Rentang angka dihitung otomatis dan tetap disimpan untuk reproduksi, bukan
diinput lewat CLI. Append dengan kamera/yaw/mode berbeda ditolak. Dataset lama tanpa metadata
kontrak ini tidak dapat dilanjutkan; gunakan direktori baru.

## 4. Train

```bash
python -m scripts.train_visual_bc \
  --dataset "$DATASET" --output-dir "$RUN" \
  --epochs 100 --batch-size 128 --learning-rate 0.0001 --device cuda
```

Setiap kamera mempunyai CNN **independen**: Conv 32/64/128/128 + ReLU, pool 4x4,
flatten, Linear 256 + ReLU. Bukan shared encoder. Proprio: joint position 7D,
EEF XYZ 3D, gripper qpos 2D, dan quaternion EEF 4D = 16D pada semua mode.
Fitur kamera dan proprio langsung di-concat, lalu head 256 ReLU -> 128 ReLU ->
normalized motion dan gripper logit. Ukuran/depth di kode
`models/visual_bc.py`, bukan CLI.

```text
loss = XYZ MSE + rotation MSE (bila aktif) + 0.2 x gripper BCEWithLogits
```

Tidak ada LayerNorm, dropout, pretrained, phase/task balancing,
scheduler, clipping gradient, atau batch cap. AdamW memakai LR 0.0001 dan
weight decay default 0.01. Satu epoch melewati seluruh
train timestep sekali dengan global shuffle. Batch bisa menggabungkan episode
berbeda; batch terakhir disimpan. Batch 128 valid meski episode pendek.

Proprio std dibatasi minimum 0.001, motion std minimum 0.01. Gripper merupakan
label biner 0/1; inference memakai threshold logit -0.5 lalu
menghasilkan -1/+1. `align_yaw` belajar motion 6D, `none` hanya XYZ; output
rotasi mode `none` tetap nol setelah denormalisasi. Proprio kedua mode sama 16D.

Log memisahkan XYZ, rotation, dan gripper loss. `best.pt` dipilih dari val loss;
`latest.pt` berisi optimizer/progress. Tidak ada rollout periodik: training
offline lalu evaluation. Trainer ini khusus fixed-target Pick dan menolak
dataset dengan target ambigu.

CPU smoke: pada data kecil gunakan `--device cpu --epochs 2 --batch-size 4` dan
RUN baru. Itu bukan pengukuran final. Tidak perlu melatih CPU lagi setelah
smoke pipeline sudah lolos hanya untuk mengulangi validasi mekanis.

### Temporal History Dengan GRU

Dataset yang sama mendukung frame dan GRU tanpa recording ulang. Jalur temporal
mengikuti `../behavior_cloning`, dengan input RGB setiap kamera, proprio, dan
command gripper **sebelumnya**:

```text
CNN tiap kamera per timestep -> concat RGB + proprio 16D + previous gripper 1D
GRU 1 layer, hidden 256 -> Linear 128 -> ReLU -> motion + gripper logit
```

```bash
DATASET=data/demos/pick_dual_yaw45_align
RUN=results/visual_bc/pick_dual_yaw45_align_gru
python -m tests.smoke.test_dataset_loader --dataset "$DATASET" --temporal --sequence-length 16 --sequence-stride 8
python -m scripts.train_visual_bc \
  --dataset "$DATASET" --output-dir "$RUN" --temporal \
  --sequence-length 16 --sequence-stride 8 --batch-size 8 \
  --epochs 100 --learning-rate 0.0001 --seed 123 --device cuda
```

Batch 8 = delapan window, masing-masing sampai 16 timestep; bukan delapan frame.
Window tidak melewati episode dan loss mengabaikan padding kanan. Training
memulai hidden nol per window, dengan previous gripper dari action expert t-1,
bukan action t. Pada awal episode previous gripper = -1/open. Evaluasi membawa
hidden antar-langkah dan memakai gripper yang benar-benar dieksekusi pada
langkah sebelumnya, lalu me-reset keduanya setiap episode.

Untuk menjalankan semua eksperimen berurutan:

```bash
EXPERIMENTS="pick_ext_yaw0_none pick_ext_yaw45_none pick_ext_yaw45_align pick_dual_yaw0_none pick_dual_yaw45_none pick_dual_yaw45_align"
for E in $EXPERIMENTS; do
python -m scripts.train_visual_bc --dataset "data/demos/$E" --epochs 100 --batch-size 128 --learning-rate 0.0001 --seed 123 --device cuda --output-dir "results/visual_bc/${E}_frame"
done
for E in $EXPERIMENTS; do
python -m scripts.train_visual_bc --dataset "data/demos/$E" --temporal --sequence-length 16 --sequence-stride 8 --epochs 100 --batch-size 8 --learning-rate 0.0001 --seed 123 --device cuda --output-dir "results/visual_bc/${E}_gru"
done
```

Frame dan GRU harus memakai output berbeda. Window 16/stride 8 bertumpuk, jadi
jumlah sample/optimizer update per epoch tidak identik dengan frame. Catat
budget training dan bandingkan success rate pada seed/count/horizon yang sama.
Loss total `none` dan `align_yaw` tidak langsung sebanding karena rotation loss.

### Resume

```bash
python -m scripts.train_visual_bc \
  --dataset "$DATASET" --output-dir "$RUN" \
  --epochs 200 --learning-rate 0.0001 --device cuda \
  --resume "$RUN/latest.pt"
```

Epoch 200 adalah target total. Output harus sama dengan folder checkpoint.
Arsitektur/data/split/statistik harus cocok. Menambah demo memerlukan run baru;
jangan diam-diam melanjutkan optimizer lama dengan statistik yang berubah.
Checkpoint membawa normalisasi dan konfigurasi kamera/reset untuk inference.
Untuk GRU, arahkan RUN ke `_gru`. Resume otomatis membaca mode temporal dan
panjang/stride window checkpoint. Tanpa `--batch-size`, default-nya 128 frame
atau 8 window; beri flag itu bila run awal memakai nilai lain. Mengubah window
atau melanjutkan frame sebagai GRU ditolak.

## 5. Evaluate

```bash
python -m scripts.evaluate_visual_bc \
  --checkpoint "$RUN/best.pt" --episodes 20 --num-seeds 5 --seed 1000 \
  --num-objects 3 --device cuda --record-video --log-rollout \
  --output-dir results/visual_bc_evaluation/pick_red_v01_n3
```

Lima seed x 20 episode = 100 attempts. Smoke: `--num-seeds 1 --episodes 5`.
Ulangi jumlah objek 1 dan 2 ke output berbeda. Tanpa `--num-objects`, evaluator
memakai count maksimum pada data. Horizon default 500; target harus naik
0.18 m dan bertahan 20 langkah. Mean dan sample SD dihitung antar-seed.

Tidak perlu argumen normalisasi/history/yaw: bobot, kamera, mode frame/GRU,
yaw/rotasi, statistik, dan controller/reset dipulihkan dari checkpoint. Gunakan
RUN `_frame` atau `_gru` sesuai model yang diuji. Video menampilkan target dan merekam seluruh
episode jika flag aktif. `summary.json` dan `pick_attempts.jsonl` meliputi
success rate dan semua kegagalan/exception. Rollout logs berisi action, clipping,
EEF, dan target. Evaluasi CPU: ganti `--device cpu` setelah menyalin `best.pt`.
GUI hanya dengan `--render` dan display yang tersedia. Loss lama MSE-7D tidak
sebanding langsung dengan objective baru.

## 6. Vast.ai, SSH, Dan Utilitas

```bash
tmux new -s train
```

Di dalam tmux aktifkan environment lagi, set DATASET/RUN dan `MUJOCO_GL=egl`,
lalu jalankan training. Detach Ctrl-b lalu d; attach dengan `tmux attach -t train`.
Unduh artifact dan verifikasi sebelum mematikan instance. Dari mesin lokal:

```bash
HOST=USERNAME@SERVER_IP
PORT=22
REMOTE_RUN=/absolute/path/to/ACT-Benchmarking/results/visual_bc/RUN
mkdir -p results/downloaded
scp -P "$PORT" "$HOST:$REMOTE_RUN/best.pt" results/downloaded/
scp -P "$PORT" "$HOST:$REMOTE_RUN/latest.pt" results/downloaded/
scp -P "$PORT" "$HOST:$REMOTE_RUN/config.json" results/downloaded/
scp -P "$PORT" "$HOST:$REMOTE_RUN/metrics.json" results/downloaded/
```

Checkpoint baru cukup untuk inference, dataset tidak wajib dipindah. Simpan
config/metrics dan artifact split untuk reproduksi. Bandingkan `sha256sum` di
kedua mesin. Muat hanya checkpoint yang dipercaya; file training berisi metadata
Python selain tensor.

```bash
python -m diagnostics.check_cameras
python -m tests.smoke.test_dataset_loader --dataset "$DATASET"
python -m tests.simulation.test_pick_env --resets 10
python -m tests
```

Smoke loader membaca artifact otomatis dan memeriksa input train/val tanpa
training. Model language, embedding, ResNet, spatial softmax, dan sampler
training lama sudah dihapus. History aktif sekarang adalah GRU di
`VisualBCPolicy`, bukan model history lama. Deskripsi instruksi tetap tersimpan sebagai
metadata di `environments/instructions.py`, bukan input policy. Split dan
normalisasi dibuat otomatis saat collection, tanpa command persiapan manual.

Gunakan dataset baru mengikuti alur di atas. Checkpoint model lama tidak dapat
di-resume dengan model visual ini.
