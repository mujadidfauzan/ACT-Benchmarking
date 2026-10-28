# Linux: Visual Behavior Cloning

Jalankan dari root repo dalam Bash. Model Visual-BC menerima RGB dan proprio
robot untuk memprediksi satu action. Storage episode memakai HDF5, dan training saat ini
mendukung fixed-target Pick dan matching-color Place (bagian 7).

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
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
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

`audit_dataset` juga mencetak total action step (T, bukan T+1 observation),
mean/median/min/max/P95 panjang episode, dan menghasilkan `episode_lengths.csv`.
Jika split default tersedia, laporan menyertakan jumlah episode/step train-val
dan estimasi update per putaran dataset untuk A/B/C serta BC-RNN. Untuk split
khusus gunakan `--split-dir <direktori>`. Asumsi estimasi default adalah batch 8,
stride A/B/C 8, accumulation 1; ubah dengan `--batch-size`, `--sequence-stride`,
dan `--gradient-accumulation-steps`. BC-RNN memakai stride 1 dan drop-last,
sehingga estimasi jumlah updatenya berbeda. Flag ini hanya menghitung estimasi;
tidak mengubah data atau konfigurasi training. Dengan `--limit-per-task`, total
audit hanya mencakup subset tersebut, sedangkan ringkasan split tetap penuh.

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
`latest.pt` berisi optimizer/progress. Default rollout setiap 10 epoch:
`--rollout-eval-every 10 --rollout-eval-episodes 10 --rollout-eval-num-seeds 1
--rollout-eval-seed 1000` (10 percobaan per evaluasi dengan seed tetap).
`best_rollout.pt` dipilih dari success rate tertinggi; nilai seri tidak mengganti
checkpoint. Laporan ada di `rollouts/epoch_XXXX/` dan `rollout_evaluations.json`.
Tidak merekam video otomatis. `--early-stopping-patience 7` menghentikan training
setelah 7 epoch berturut-turut tanpa val loss lebih rendah, termasuk sebelum
rollout pertama. Counter tersimpan saat resume; naikkan patience atau gunakan 0
untuk melanjutkan run yang sudah early stop. `--rollout-eval-every 0` mematikan
rollout. Gunakan seed final berbeda dari seed rollout pemilihan model.
Trainer mendukung fixed-target Pick atau matching-color Place dan menolak target ambigu.
`--rollout-eval-max-steps N` mengatur batas episode rollout; default Pick 500,
Place 3000. Gunakan batas yang sama pada evaluasi final dengan `--max-steps N`.
Batas tersimpan di checkpoint dan dipulihkan saat resume; menggantinya memerlukan
run baru agar success rate pemilihan checkpoint tetap sebanding.

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

Untuk menguji pengaruh panjang memori pada checkpoint yang sudah ada, tambahkan
`--rnn-reset-interval 16` saat evaluasi. Hidden state dikosongkan sebelum action
1, 17, 33, dan seterusnya, sementara previous gripper tetap berisi command yang
benar-benar dieksekusi pada langkah sebelumnya. Default `0` mempertahankan
memori sepanjang episode. Opsi ini hanya menerima checkpoint GRU dan tidak
mengubah bobot atau melatih ulang model.

Contoh checkpoint A, dengan scene dan horizon yang sama dengan evaluasi lama:

```bash
MUJOCO_GL=egl python -u -m scripts.evaluate_visual_bc \
  --checkpoint results/visual_bc/place_red_1cube_1tray_v01_gru/best_rollout.pt \
  --num-objects 1 --num-receptacles 1 \
  --episodes 20 --num-seeds 5 --seed 2000 \
  --max-steps 300 --rnn-reset-interval 16 --device cuda \
  --record-video --log-rollout \
  --output-dir results/visual_bc_evaluation/place_red_1cube_1tray_v01_gru_reset16_h300
```

`summary.json` dan setiap attempt mencatat `rnn_reset_interval`. Dengan
`--log-rollout`, setiap langkah juga mencatat `hidden_state_reset`,
`previous_gripper_input`, dan `gripper_logit` sebelum threshold. Bandingkan
dengan hasil default pada seed, jumlah objek/tray, dan horizon yang identik;
reset berkala merupakan hipotesis eksperimen, belum tentu meningkatkan success.
Rollout otomatis selama training tetap memakai default memori sepanjang episode.

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

### Eksperimen B: CNN Dengan Spatial Softmax

Gunakan dataset dan split yang sama, tanpa recording ulang. Opsi
`--visual-pooling spatial_softmax` mempertahankan empat convolution layer,
tetapi mengganti pool 4x4 dengan expected koordinat (x, y) per channel pada
rentang [-1, 1], temperature tetap 1.0. Ada 128 channel sehingga pooling
menghasilkan 256 koordinat, diproyeksikan kembali ke 256 fitur per kamera.
GRU, proprio, action, dan objective tetap sama. Baseline memakai
`--visual-pooling avg` (default).

Contoh Place 1 kubus merah dan 1 tray merah, history 16, batch 64 dari awal,
dengan batas rollout dan evaluasi 300 langkah:

```bash
python -m scripts.train_visual_bc \
  --dataset data/demos/place_red_1cube_1tray_v01 \
  --temporal --sequence-length 16 --sequence-stride 8 \
  --visual-pooling spatial_softmax \
  --epochs 100 --batch-size 64 --learning-rate 0.0001 \
  --seed 123 --device cuda --early-stopping-patience 15 \
  --rollout-eval-every 10 --rollout-eval-episodes 10 \
  --rollout-eval-num-seeds 1 --rollout-eval-seed 1000 \
  --rollout-eval-max-steps 300 \
  --output-dir results/visual_bc/place_red_1cube_1tray_v01_gru_spatial

python -m scripts.evaluate_visual_bc \
  --checkpoint results/visual_bc/place_red_1cube_1tray_v01_gru_spatial/best_rollout.pt \
  --num-objects 1 --num-receptacles 1 \
  --episodes 20 --num-seeds 5 --seed 2000 \
  --max-steps 300 --device cuda \
  --output-dir results/visual_bc_evaluation/place_red_1cube_1tray_v01_gru_spatial_h300
```

Gunakan output baru, tanpa resume checkpoint average pooling. Pooling dipulihkan
otomatis dari checkpoint saat resume dan evaluasi. Untuk perbandingan A/B,
latih baseline dari awal dengan budget yang sama, `--visual-pooling avg`, dan
output terpisah. `summary.json` mencatat `max_steps` agar horizon bisa diperiksa.

### Eksperimen C: ResNet18 Dengan Spatial Softmax

Gunakan dataset dan split eksperimen B. `--visual-backbone resnet18` mengganti
empat convolution layer dengan feature backbone ResNet18 torchvision, tanpa
global average pooling dan classifier. Ada 512 channel sehingga Spatial Softmax
menghasilkan 1024 koordinat, diproyeksikan ke 256 fitur per kamera. Setiap kamera
mempunyai encoder independen. RGB tetap 128x128 dalam [0, 1]. ResNet dilatih
dari nol (`weights=None`), tanpa unduhan pretrained atau normalisasi ImageNet.
Lihat [API ResNet18 torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Pasang pasangan torch dan torchvision dari index CPU/CUDA yang sama, seperti
setup di atas. ResNet18 lebih berat daripada CNN A/B. Contoh memakai batch fisik
8 dan `--gradient-accumulation-steps 8`: 64 window per full optimizer update.
Gradien dirata-ratakan berdasarkan valid timestep, termasuk padding GRU dan
grup terakhir yang lebih kecil. BatchNorm tetap memakai batch fisik 8 sehingga
ini tidak identik dengan batch fisik 64. Metrik `batches` menghitung batch fisik;
`optimizer_steps` mencatat jumlah update aktual.

```bash
python -m scripts.train_visual_bc \
  --dataset data/demos/place_red_1cube_1tray_v01 \
  --visual-backbone resnet18 --visual-pooling spatial_softmax \
  --temporal --sequence-length 16 --sequence-stride 8 \
  --epochs 100 --batch-size 8 --gradient-accumulation-steps 8 \
  --learning-rate 0.0001 --seed 123 --device cuda \
  --early-stopping-patience 15 \
  --rollout-eval-every 10 --rollout-eval-episodes 10 \
  --rollout-eval-num-seeds 1 --rollout-eval-seed 1000 \
  --rollout-eval-max-steps 300 \
  --output-dir results/visual_bc/place_red_1cube_1tray_v01_gru_resnet18

python -m scripts.evaluate_visual_bc \
  --checkpoint results/visual_bc/place_red_1cube_1tray_v01_gru_resnet18/best_rollout.pt \
  --num-objects 1 --num-receptacles 1 \
  --episodes 20 --num-seeds 5 --seed 2000 \
  --max-steps 300 --device cuda \
  --output-dir results/visual_bc_evaluation/place_red_1cube_1tray_v01_gru_resnet18_h300
```

Mulai run baru; checkpoint CNN tidak dapat di-resume menjadi ResNet18. Backbone
dan pooling dipulihkan dari checkpoint pada resume/evaluasi. Akumulasi gradien
dipulihkan saat resume; cantumkan batch fisik yang dipakai run tersebut.
Jika batch fisik 8 kehabisan VRAM, gunakan batch 4 dengan akumulasi 16.

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

### Baseline baru: deterministic BC-RNN ala robomimic

Baseline ini membaca dataset HDF5 dan split yang sama secara langsung, tanpa
konversi. Entry point terpisah adalah `scripts.train_robomimic_bc_rnn` dan
`scripts.evaluate_robomimic_bc_rnn`. ResNet18 + learned keypoints + LSTM,
continuous 7D action MSE, dan Adam mengikuti resep rujukan; batch default 8
menyesuaikan hardware. Lihat [implementasi arsitektur](../models/robomimic_bc_rnn/policy.py)
dan command train + eval di bawah.

```bash
python -m scripts.train_robomimic_bc_rnn \
  --dataset data/demos/place_red_1cube_1tray_v01 \
  --batch-size 8 --max-updates 60000 --device cuda \
  --output-dir results/robomimic_bc_rnn/place_red_1cube_1tray_v01
python -m scripts.evaluate_robomimic_bc_rnn \
  --checkpoint results/robomimic_bc_rnn/place_red_1cube_1tray_v01/best_rollout.pt \
  --num-objects 1 --num-receptacles 1 \
  --episodes 20 --num-seeds 5 --seed 2000 --max-steps 300 --device cuda \
  --record-video --log-rollout \
  --output-dir results/robomimic_bc_rnn_evaluation/place_red_1cube_1tray_v01_h300
```

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
training. Model language, embedding, dan sampler
training lama sudah dihapus. History aktif sekarang adalah GRU di
`VisualBCPolicy`, bukan model history lama. Deskripsi instruksi tetap tersimpan sebagai
metadata di `environments/instructions.py`, bukan input policy. Split dan
normalisasi dibuat otomatis saat collection, tanpa command persiapan manual.

Gunakan dataset baru mengikuti alur di atas. Checkpoint model lama tidak dapat
di-resume dengan model visual ini.

## 7. Place Tanpa Language: Merah ke Merah

Dataset baru: kubus merah ke tray merah, warna lain distractor, ID dan posisi
diacak. Instruksi tersimpan sebagai metadata saja. Place memakai yaw acak
+/-45 derajat dengan action rotasi aktif, bukan flag ablation yaw/rotasi Pick.

```bash
python -m scripts.record_demonstrations \
  --task place --num-demos 300 \
  --num-objects 3 --num-receptacles 3 \
  --fixed-target-color red \
  --camera-names agentview robot0_eye_in_hand \
  --seed 42 --max-attempts 1500 \
  --output data/demos/place_red_v01

python -m scripts.train_visual_bc \
  --dataset data/demos/place_red_v01 \
  --epochs 100 --batch-size 128 --learning-rate 0.0001 \
  --rollout-eval-every 10 --rollout-eval-episodes 10 \
  --rollout-eval-num-seeds 1 --rollout-eval-seed 1000 \
  --early-stopping-patience 7 --device cuda \
  --output-dir results/visual_bc/place_red_v01_frame

python -m scripts.evaluate_visual_bc \
  --checkpoint results/visual_bc/place_red_v01_frame/best_rollout.pt \
  --episodes 20 --num-seeds 5 --seed 2000 \
  --num-objects 3 --num-receptacles 3 \
  --device cuda --record-video --log-rollout \
  --output-dir results/visual_bc_evaluation/place_red_v01_final
```

Gunakan konfigurasi MUJOCO_GL yang sudah dipilih pada setup Linux di atas.
HDF5, split train/val dan normalisasi dibuat otomatis setelah collection.
Untuk GRU tambahkan `--temporal --sequence-length 16 --sequence-stride 8`,
ubah batch size ke 8 dan output ke `results/visual_bc/place_red_v01_gru`.
Resume: tambahkan `--resume results/visual_bc/place_red_v01_frame/latest.pt`
pada training dengan output sama dan epochs total lebih besar. Jangan resume
checkpoint Pick sebagai Place. Bila early stopping terjadi sebelum rollout
pertama dan `best_rollout.pt` belum ada, evaluasi `best.pt`.

Evaluator memilih task dari checkpoint, dengan horizon Place default 3000.
Sukses memerlukan kubus muat dalam tray (orientasi diperhitungkan), tinggi
sesuai tray, EEF menjauh >8 cm, action gripper open, dan distractor bergeser
tidak lebih dari 4 cm selama 20 langkah berurutan. Expert memakai pemeriksaan
fisik yang sama setelah release/retreat. Kegagalan dicatat di
`place_attempts.jsonl`: no_lift, wrong_object, wrong_receptacle, not_released,
distractor_displacement, place_timeout. Evaluasi final memakai seed berbeda
dari rollout pemilihan model.

Variasi scene: gunakan `--object-counts 1 2 3 --receptacle-counts 1 2 3` sebagai
pengganti count tunggal saat collection. Evaluasi tiap kombinasi dengan output
terpisah. Rollout otomatis memakai jumlah maksimum di dataset, bukan semua
kombinasi sekaligus.

### Place dengan kubus dan tray di sisi berlawanan

Gunakan `--placement-layout opposite_sides` saat collection untuk mengacak dua
arah: kubus di Y negatif dan tray di Y positif, atau sebaliknya. Kanan/kiri
di sini relatif robot (sumbu Y dunia); robot menghadap +X. Aturan berlaku pada
posisi awal setiap episode. X tetap diacak dalam rentang -0,16 sampai +0,16 m,
dan tiap entitas mendapatkan yaw independen dalam rentang -45 sampai +45 derajat.
Sampler mempertimbangkan ukuran benda agar seluruh benda berada di setengah
meja masing-masing, serta tetap memeriksa jarak minimum dan overlap. Posisi
kubus dan tray diacak secara independen; keduanya tidak dipasang sebagai pose
cermin. Untuk scene multiobjek, semua kubus berada di satu sisi dan semua tray
di sisi lain.

```bash
python -m scripts.record_demonstrations \
  --task place --num-demos 1000 \
  --num-objects 1 --num-receptacles 1 --fixed-target-color red \
  --placement-layout opposite_sides \
  --camera-names agentview robot0_eye_in_hand --image-size 128 \
  --seed 42 --max-attempts 5000 \
  --output data/demos/place_red_1cube_1tray_opposite_v01
```

Pola tersimpan dalam `collection_config.environment_kwargs.placement_layout`.
Train memakai dataset baru ini; evaluator otomatis memulihkan layout dari
checkpoint, termasuk untuk visual BC, image BC-RNN dan low-dim BC-RNN. Summary
evaluasi mencatat `placement_layout`. Dataset lama tanpa field tersebut tetap
memakai layout `uniform`. Collection menolak append dengan layout berbeda
ke folder dataset yang sama. Expert, durasi fase dan kriteria sukses tetap
mengikuti konfigurasi sebelumnya. Pemilihan arah acak per reset tidak menjamin
jumlah demo sukses persis 50:50 pada kedua arah.

Untuk membatasi jarak awal antarpusat kubus dan tray di bidang XY, tambahkan
`--max-object-receptacle-distance 0.30`. Dengan minimum default, jaraknya
menjadi 0,12–0,30 m. Batas maksimum berlaku pada setiap pasangan kubus/tray;
sampling mengulang hingga batas minimum, maksimum, dan layout terpenuhi.
Posisi dan yaw tetap acak, tetapi batas jarak menghubungkan pilihan kedua pose.
Ini tidak menjamin episode selesai dalam 300 step.

```bash
MUJOCO_GL=egl python -u -m scripts.record_demonstrations \
  --task place --num-demos 1000 \
  --num-objects 1 --num-receptacles 1 --fixed-target-color red \
  --placement-layout opposite_sides --max-object-receptacle-distance 0.30 \
  --camera-names agentview robot0_eye_in_hand --image-size 128 \
  --seed 42 --max-attempts 5000 \
  --output data/demos/place_red_1cube_1tray_opposite_d30_v01 \
  --training-dir data/splits/place_red_1cube_1tray_opposite_d30_v01_bc
```

Batas disimpan sebagai `maximum_object_receptacle_distance` dalam
`collection_config.environment_kwargs`, otomatis dipulihkan semua evaluator,
dan dicatat dalam summary evaluasi. Collection menolak append jika batasnya
berbeda. Tanpa opsi tersebut, dataset dan checkpoint lama tetap tanpa batas
maksimum. Gunakan folder dataset baru untuk eksperimen jarak ini.

## 8. Baseline Low-Dim BC-RNN

Untuk diagnosis persepsi pada Place satu kubus / satu tray, gunakan
[implementasi low-dim BC-RNN](../models/low_dim_bc_rnn/policy.py). Model memakai state
robot, kubus dan tray (23D) dari HDF5 yang sama, tanpa input gambar. Script baru
`scripts.train_low_dim_bc_rnn` dan `scripts.evaluate_low_dim_bc_rnn` memakai
split yang sudah tersedia, dengan direktori hasil terpisah. Parameter train,
eval dan resume tersedia melalui `--help` pada masing-masing script.
