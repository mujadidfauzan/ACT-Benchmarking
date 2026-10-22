# Windows: Visual Behavior Cloning

Jalankan dari root repo dalam **CMD** dengan Conda aktif. Model Visual-BC menerima
RGB dan proprio robot untuk memprediksi satu action. Environment Pick/Place/Stack
tetap dipertahankan. Episode disimpan sebagai HDF5; training mendukung fixed-target
Pick dan Place dengan objek/tray target berwarna sama (bagian 7).

## 1. Setup

Git, Conda, dan driver NVIDIA harus tersedia. Contoh berikut memakai wheel CUDA
12.8; gunakan [selector PyTorch](https://pytorch.org/get-started/locally/) untuk
mesin lain. Jangan menjalankan install CPU dan CUDA pada environment yang sama.

```bat
git clone https://github.com/mujadidfauzan/ACT-Benchmarking.git
cd ACT-Benchmarking
conda create -n sim-mt-act python=3.10 -y
conda activate sim-mt-act
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
set MUJOCO_GL=wgl
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); x=torch.randn(128,128,device='cuda'); print((x@x).mean().item())"
python -m tests
```

CPU: ganti index instalasi PyTorch dengan `https://download.pytorch.org/whl/cpu`.
Lewati command tensor CUDA pada CPU. Jangan memakai `requirements-ml-cpu.txt`
di environment GPU. Gunakan environment/TEMP/cache di drive yang cukup ruang.
Tidak perlu WSL atau ROS. `$env:...` adalah sintaks PowerShell, bukan CMD.

## 2. Gate Expert

Expert Pick menggunakan state machine: approach, align,
descend, grasp hold 15 langkah, lalu lift 0.20 m. Sukses memerlukan target naik
setidaknya 0.18 m selama 20 langkah berturut-turut dan lolos `env._check_success()`.
Orientasi mengikuti yaw objek dalam world frame. Place/Stack tetap memakai
expert task-specific project. Pick juga mendukung `--rotation-mode none`:
ketiga command rotasi tepat nol dan expert tidak menunggu alignment.

```bat
python -m benchmarks.expert_reliability ^
  --task pick --episodes 60 --object-counts 1 2 3 ^
  --fixed-target-color red --run-name pick_gate --enforce-gates
```

Tambahkan `--render` untuk viewer bila tersedia. Nama run harus baru. Gate
tetap menghitung setiap attempt; ini pemeriksaan expert, bukan balancing training.

## 3. Collect Sekaligus Persiapan Training

Gunakan direktori baru agar tidak mencampur kamera/controller/data lama.
Dataset pilot, split, dan cache eksperimen lama sudah dibersihkan dari workspace.
Folder `data/demos/` dan `data/splits/` diisi kembali oleh collection; artifact
hasil generate tidak disertakan dalam Git.
Contoh fixed-red Pick dengan 1-3 objek dan dua kamera:

```bat
set DATASET=data/demos/pick_red_v01
set RUN=results/visual_bc/pick_red_v01_seed123
python -m scripts.record_demonstrations ^
  --task pick --num-demos 300 --output %DATASET% ^
  --object-counts 1 2 3 --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 42 --max-attempts 1200
```

RGB otomatis aktif dan direkam pada 128x128 dengan konvensi OpenCV. Untuk satu
kamera, gunakan `--camera-names agentview`. Jangan jalankan dua recorder pada
dataset yang sama secara bersamaan.
Pick default-nya target merah. Pertahankan satu warna target untuk training
Visual-BC. Opsi collection `--random-target` tidak didukung trainer ini.

`--num-demos 300` berarti **target total sukses untuk task tersebut**. Mengulang
command setelah terputus hanya melengkapi kekurangan. Opsi lama `--episodes N`
masih berarti **menambah N**, dan tidak dapat digabung dengan `--num-demos`.
Percobaan gagal dan frame kamera korup tidak masuk manifest sukses.

Setelah collect selesai, otomatis tersedia:

```text
data/demos/pick_red_v01/pick/
  episode_000000.hdf5 ...     T actions, T+1 observations
  manifest.jsonl             Episode sukses
  attempts.jsonl             Semua percobaan
data/splits/pick_red_v01_bc/
  train.jsonl                Episode train, sekitar 90%
  val.jsonl                  Episode validation, sekitar 10%
  normalization.json         Statistik train-only, bukan task-balanced
  training.json              Kamera, proprio, action, fingerprint dan konfigurasi
```

Tidak perlu command create-splits atau compute-normalization. Default split
memakai seed 123, tanpa stratifikasi fase/count/task. Minimal dua demo sukses
dibutuhkan. `--training-dir PATH` dapat mengubah lokasi artifact otomatis.
Orphan HDF5 akibat interupsi harus diperiksa dan dikarantina sebelum persiapan;
jangan asal menambahkan file ke manifest.

### Penyimpanan HDF5 Dan Dataset Lama

Satu episode = satu `.hdf5`, ditulis atomik sebelum manifest. Array tetap memakai
nama `actions`, `phases`, `obs__...`, `metadata_json`, dan `result_json`.
RGB memakai gzip level 4 dengan chunk satu frame; loader frame/GRU hanya membaca
timestep/window yang diperlukan. Cache menyimpan maksimum 16 file handle per
dataset/proses, bukan seluruh gambar episode. Handle tidak dibagikan antar-worker.
Normalisasi tetap train-only, dan pemeriksaan kamera membaca blok frame terbatas.

NPZ lama tidak dapat langsung dipakai atau dicampur dengan collection HDF5.
Konversi opsional berikut tidak mengubah sumber dan otomatis membuat split serta
normalisasi baru. Kedua folder output harus baru:

```bat
python -m data.episode_io ^
  --source data/demos/pick_red_v01 ^
  --output data/demos/pick_red_v01_hdf5 ^
  --training-dir data/splits/pick_red_v01_hdf5_bc
```

Setelah konversi, gunakan `--dataset data/demos/pick_red_v01_hdf5` dan run training
baru. Fingerprint berubah, jadi jangan resume checkpoint training NPZ pada data
hasil konversi. Checkpoint mandiri lama tetap dapat dievaluasi: arsitektur model
tidak berubah. Command enam eksperimen di bawah tetap sama, hanya format file
yang otomatis menjadi HDF5. Konversi ini untuk schema NPZ project ini, bukan
import bebas dari HDF5 repo lain.

Opsional, audit tidak diperlukan untuk menghasilkan artifact training:

```bat
python -m scripts.audit_dataset ^
  --dataset %DATASET% --tasks pick --mode full --check-duplicates ^
  --camera-names agentview robot0_eye_in_hand ^
  --output results/dataset_audit/pick_red_v01
python -m scripts.audit_pose_distribution ^
  --dataset %DATASET% --split-dir data/splits/pick_red_v01_bc ^
  --output results/pose_distribution/pick_red_v01
```

Audit pose otomatis memakai train/val bila split tidak memiliki test.

### Enam Eksperimen Kamera Dan Orientasi

Posisi sampling tetap x `[-0.10, 0.10]` m dan y `[-0.20, 0.20]` m.
Yaw memakai toggle tanpa input angka dan berlaku pada semua kubus, bukan hanya
target. `--random-yaw` (default) berarti uniform -45 sampai +45 derajat;
`--no-random-yaw` berarti yaw awal tetap nol.
Untuk mengisolasi kamera/orientasi, semua command berikut memakai tiga objek,
target merah, dan seed collection yang sama. Gunakan direktori baru.

Tanpa eye-in-hand:

```bat
python -m scripts.record_demonstrations ^
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red ^
  --camera-names agentview --no-random-yaw --rotation-mode none ^
  --seed 42 --max-attempts 1200 --output data/demos/pick_ext_yaw0_none
python -m scripts.record_demonstrations ^
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red ^
  --camera-names agentview --random-yaw --rotation-mode none ^
  --seed 42 --max-attempts 1200 --output data/demos/pick_ext_yaw45_none
python -m scripts.record_demonstrations ^
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red ^
  --camera-names agentview --random-yaw --rotation-mode align_yaw ^
  --seed 42 --max-attempts 1200 --output data/demos/pick_ext_yaw45_align
```

Dengan eye-in-hand:

```bat
python -m scripts.record_demonstrations ^
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand --no-random-yaw --rotation-mode none ^
  --seed 42 --max-attempts 1200 --output data/demos/pick_dual_yaw0_none
python -m scripts.record_demonstrations ^
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand --random-yaw --rotation-mode none ^
  --seed 42 --max-attempts 1200 --output data/demos/pick_dual_yaw45_none
python -m scripts.record_demonstrations ^
  --task pick --num-demos 300 --num-objects 3 --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand --random-yaw --rotation-mode align_yaw ^
  --seed 42 --max-attempts 1200 --output data/demos/pick_dual_yaw45_align
```

Sebelum collect besar, periksa expert pada ketiga konfigurasi fisiknya:

```bat
python -m benchmarks.expert_reliability --task pick --episodes 60 --num-objects 3 --fixed-target-color red --no-random-yaw --rotation-mode none --run-name pick_yaw0_none_gate --enforce-gates
python -m benchmarks.expert_reliability --task pick --episodes 60 --num-objects 3 --fixed-target-color red --random-yaw --rotation-mode none --run-name pick_yaw45_none_gate --enforce-gates
python -m benchmarks.expert_reliability --task pick --episodes 60 --num-objects 3 --fixed-target-color red --random-yaw --rotation-mode align_yaw --run-name pick_yaw45_align_gate --enforce-gates
```

`none` bukan jaminan orientasi fisik EEF benar-benar konstan: joint reset noise
3 derajat dan dinamika tetap ada. Yang dijamin adalah command rotasi nol.
Expert tanpa rotasi pada kubus miring bisa lebih sering gagal; gate simulator
tetap perlu dijalankan. Filter demo sukses juga dapat membedakan distribusi
scene antar-mode meskipun seed sama.

Metadata episode dan `training.json` menyimpan `rotation_mode`,
`control_orientation`, serta `collection_config.yaw_range_deg` dan konfigurasi
environment aktual. Rentang angka dihitung otomatis dan tetap disimpan dalam
metadata untuk reproduksi, bukan diinput lewat CLI. Append dengan
mode/yaw/kamera berbeda ditolak. Dataset lama
tanpa kontrak metadata ini tidak dapat dilanjutkan; buat dataset baru.

## 4. Train

```bat
python -m scripts.train_visual_bc ^
  --dataset %DATASET% --output-dir %RUN% ^
  --epochs 100 --batch-size 128 --learning-rate 0.0001 --device cuda
```

Arsitektur Visual-BC:

```text
Per kamera: Conv(32) -> Conv(64) -> Conv(128) -> Conv(128)
            ReLU setiap Conv -> pool 4x4 -> Linear 256 -> ReLU
Dua kamera: CNN terpisah, bukan shared weights
Proprio: joint_pos 7 + EEF XYZ 3 + gripper qpos 2 + EEF quaternion 4 = 16D
Concat: 256 x jumlah kamera + proprio -> 256 ReLU -> 128 ReLU
Output: normalized XYZ, optional normalized rotation, gripper logit
```

Ukuran/depth diatur dalam `models/visual_bc.py`, bukan CLI. Tidak ada
pretrained, LayerNorm, dropout, scheduler, gradient clipping, balancing,
atau batas batch buatan pada trainer ini. AdamW memakai LR 0.0001 dan weight decay
default 0.01. Epoch melewati **seluruh timestep train sekali**
dengan shuffle global; batch dapat berisi episode berbeda dan batch terakhir
tidak dibuang. Karena itu batch 128 tetap valid walau episode lebih pendek.

```text
loss = XYZ MSE + rotation MSE (bila aktif) + 0.2 x gripper BCEWithLogits
```

Proprio std minimum 0.001; motion std minimum 0.01. Gripper memakai label 0/1,
bukan z-score/regresi. Inference memakai threshold logit -0.5
menjadi action gripper -1/+1. Mode `align_yaw` belajar motion 6D dan gripper;
mode `none` belajar XYZ dan gripper, dengan action rotasi nol juga setelah
denormalisasi. Proprio tetap 16D pada kedua mode, agar input tidak ikut berubah.

Log mencatat loss total, XYZ, rotation, dan gripper secara terpisah. `best.pt`
dipilih dari val loss; `latest.pt` menyimpan optimizer dan progress.
Default rollout setiap 10 epoch: `--rollout-eval-every 10`,
`--rollout-eval-episodes 20 --rollout-eval-num-seeds 5 --rollout-eval-seed 1000`
(100 percobaan per evaluasi, seed tetap). `best_rollout.pt` menyimpan model dengan
success rate tertinggi; nilai seri tidak mengganti checkpoint. Laporan ada di
`rollouts/epoch_XXXX/` dan `rollout_evaluations.json`, tanpa video otomatis.
`--early-stopping-patience 7` menghentikan training setelah 7 epoch berturut-turut
tanpa val loss lebih rendah. Bisa berhenti sebelum rollout pertama.
Counter disimpan saat resume; naikkan patience atau gunakan 0 untuk melanjutkan
run yang sudah early stop. `--rollout-eval-every 0` mematikan rollout.
Gunakan seed evaluasi final yang berbeda dari seed rollout pemilihan model.

CPU smoke: gunakan `--device cpu --epochs 2 --batch-size 4` pada dataset kecil
dan RUN baru. Trainer ini menolak
dataset selain fixed-target Pick atau matching-color Place agar target tidak ambigu.

### Temporal History Dengan GRU

Dataset yang sama bisa dilatih tanpa history atau dengan GRU, tanpa collect
ulang. History meliputi RGB dari setiap kamera, proprio, dan command gripper
**sebelumnya**, mengikuti alur GRU di `../behavior_cloning`:

```text
Per timestep: CNN tiap kamera -> concat fitur RGB + proprio 16D + previous grip 1D
GRU 1 layer, hidden 256 -> Linear 128 -> ReLU -> motion + gripper logit
```

```bat
set DATASET=data/demos/pick_dual_yaw45_align
set RUN=results/visual_bc/pick_dual_yaw45_align_gru
python -m tests.smoke.test_dataset_loader --dataset %DATASET% --temporal --sequence-length 16 --sequence-stride 8
python -m scripts.train_visual_bc ^
  --dataset %DATASET% --output-dir %RUN% --temporal ^
  --sequence-length 16 --sequence-stride 8 --batch-size 8 ^
  --epochs 100 --learning-rate 0.0001 --seed 123 --device cuda
```

Batch 8 berarti delapan **window**, masing-masing sampai 16 timestep. Window
tidak menyeberangi episode; padding kanan tidak dihitung dalam loss. Saat
training, hidden mulai nol pada setiap window dan previous gripper berasal dari
action expert t-1 (bukan action t). Pada awal episode nilainya -1/open.
Evaluasi membawa hidden antar-langkah dan memakai gripper yang dieksekusi pada
langkah sebelumnya; hidden dan previous gripper di-reset tiap episode.

Untuk melatih semua enam dataset, loop berikut dijalankan di **CMD interaktif**:

```bat
for %E in (pick_ext_yaw0_none pick_ext_yaw45_none pick_ext_yaw45_align pick_dual_yaw0_none pick_dual_yaw45_none pick_dual_yaw45_align) do python -m scripts.train_visual_bc --dataset data/demos/%E --epochs 100 --batch-size 128 --learning-rate 0.0001 --seed 123 --device cuda --output-dir results/visual_bc/%E_frame
for %E in (pick_ext_yaw0_none pick_ext_yaw45_none pick_ext_yaw45_align pick_dual_yaw0_none pick_dual_yaw45_none pick_dual_yaw45_align) do python -m scripts.train_visual_bc --dataset data/demos/%E --temporal --sequence-length 16 --sequence-stride 8 --epochs 100 --batch-size 8 --learning-rate 0.0001 --seed 123 --device cuda --output-dir results/visual_bc/%E_gru
```

Di file `.bat`, ganti `%E` menjadi `%%E`. Frame dan GRU memakai run terpisah.
Stride 8/window 16 membuat timestep interior muncul pada window bertumpuk;
epoch GRU bukan jumlah sample/optimizer update yang identik dengan epoch frame.
Bandingkan success rate dengan seed/count/horizon sama, serta catat budget
training. Total loss `none` dan `align_yaw` juga berbeda karena rotation loss.

### Resume

```bat
python -m scripts.train_visual_bc ^
  --dataset %DATASET% --output-dir %RUN% ^
  --epochs 200 --learning-rate 0.0001 --device cuda ^
  --resume %RUN%/latest.pt
```

Epoch 200 adalah batas akhir, bukan tambahan 200. Dataset/split/normalisasi dan
arsitektur harus tetap cocok; penambahan demo membentuk eksperimen baru.
Checkpoint menyimpan statistik dan konfigurasi input sehingga bisa dievaluasi
di komputer lain tanpa command normalisasi terpisah.
Untuk resume GRU, arahkan RUN ke folder `_gru`; mode temporal dan panjang/stride
window otomatis diambil dari checkpoint. Tanpa flag batch size, default-nya
128 frame atau 8 window; beri `--batch-size` bila run awal memakai nilai lain.
Jangan mengubah panjang/stride atau melanjutkan checkpoint frame sebagai GRU.

## 5. Evaluate

```bat
python -m scripts.evaluate_visual_bc ^
  --checkpoint %RUN%/best.pt --episodes 20 --num-seeds 5 --seed 1000 ^
  --num-objects 3 --device cuda --record-video --log-rollout ^
  --output-dir results/visual_bc_evaluation/pick_red_v01_n3
```

Ini 5 seed x 20 episode = 100 percobaan, bukan total 20. Untuk smoke gunakan
`--num-seeds 1 --episodes 5`. Ulangi count 1 dan 2 dengan output baru. Default
count berasal dari jumlah maksimum objek di data bila tidak diberikan. Default
horizon 500 dan hold 20, target harus benar-benar naik 0.18 m. Seed evaluasi
berbeda dari seed collection; scene bukan replay train/val.

Evaluator membaca kamera, arsitektur frame/GRU, yaw, mode rotasi, normalisasi,
dan reset/controller dari checkpoint. Tidak perlu `--normalization` atau flag
history/yaw pada evaluator. Gunakan RUN `_frame` atau `_gru` yang hendak diuji.
Video merekam semua episode ketika
flag aktif dan menampilkan target. `summary.json` berisi overall success rate,
mean/sample SD antar-seed, dan failure counts; `pick_attempts.jsonl` mencatat
semua kegagalan termasuk exception. `--render` hanya untuk sesi GUI.

Untuk CPU laptop, salin `best.pt` lalu ganti ke `--device cpu`. Checkpoint baru
mandiri untuk inference; simpan juga config/metrics untuk reproduksi. Jangan
membandingkan loss MSE-7D lama langsung dengan loss BCE+motion baru.

## 6. Utilitas

```bat
python -m diagnostics.check_cameras
python -m tests.smoke.test_dataset_loader --dataset %DATASET%
python -m tests.simulation.test_pick_env --resets 10
python -m tests
```

Camera export berada di `results/diagnostics/cameras/`. Tests offline tidak
mengunduh bobot atau menjalankan simulator nyata. Smoke loader membaca artifact
otomatis dan memeriksa input train/val tanpa training.

Model language, embedding, ResNet, spatial softmax, dan sampler training lama
sudah dihapus. History yang didukung adalah GRU pada `VisualBCPolicy`, bukan
model history lama. Deskripsi instruksi tetap tersimpan sebagai metadata di
`environments/instructions.py`, bukan input policy. Split dan normalisasi hanya
dibuat otomatis saat collection, tanpa command persiapan manual.

Gunakan dataset baru mengikuti alur di atas. Checkpoint model lama tidak dapat
di-resume dengan model visual ini.

Pada SSH Windows, CMD tidak memiliki tmux. Gunakan sesi desktop/Task Scheduler
yang tetap hidup dan working directory root repo; jangan menganggap proses
bertahan setelah SSH ditutup. WinError 1455/MemoryError memerlukan pemeriksaan
RAM, pagefile, dan disk, bukan otomatis berarti GPU tidak kompatibel.

## 7. Place Tanpa Language: Merah ke Merah

Gunakan dataset/run baru, bukan resume checkpoint Pick. Kubus merah dan tray
merah selalu ada dan menjadi target; warna lain distractor. ID dan posisi
diacak. Deskripsi instruksi hanya metadata, bukan input model.
Place saat ini memakai yaw acak +/-45 derajat dan action rotasi aktif;
flag ablation yaw/rotasi Pick tidak berlaku untuk Place.

```bat
python -m scripts.record_demonstrations ^
  --task place --num-demos 300 ^
  --num-objects 3 --num-receptacles 3 ^
  --fixed-target-color red ^
  --camera-names agentview robot0_eye_in_hand ^
  --seed 42 --max-attempts 1500 ^
  --output data/demos/place_red_v01

python -m scripts.train_visual_bc ^
  --dataset data/demos/place_red_v01 ^
  --epochs 100 --batch-size 128 --learning-rate 0.0001 ^
  --rollout-eval-every 10 --rollout-eval-episodes 10 ^
  --rollout-eval-num-seeds 1 --rollout-eval-seed 1000 ^
  --early-stopping-patience 7 --device cuda ^
  --output-dir results/visual_bc/place_red_v01_frame

python -m scripts.evaluate_visual_bc ^
  --checkpoint results/visual_bc/place_red_v01_frame/best_rollout.pt ^
  --episodes 20 --num-seeds 5 --seed 2000 ^
  --num-objects 3 --num-receptacles 3 ^
  --device cuda --record-video --log-rollout ^
  --output-dir results/visual_bc_evaluation/place_red_v01_final
```

HDF5, train/val split dan normalisasi dibuat otomatis setelah collection.
Untuk GRU, tambahkan `--temporal --sequence-length 16 --sequence-stride 8`,
ubah batch size menjadi 8, dan output menjadi `results/visual_bc/place_red_v01_gru`.
Untuk resume tambahkan `--resume results/visual_bc/place_red_v01_frame/latest.pt`
pada command training dengan output yang sama dan epochs total lebih besar.
Early stopping dapat terjadi sebelum rollout pertama; bila `best_rollout.pt`
belum ada, evaluasi `best.pt`. Seed final berbeda dari seed pemilihan model.

Evaluator otomatis memilih Place dari checkpoint; horizon default 3000 langkah.
Sukses membutuhkan kubus muat dalam tray (termasuk orientasi), tinggi sesuai
tray, EEF menjauh lebih dari 8 cm, action gripper open, dan distractor tidak
bergeser lebih dari 4 cm, selama 20 langkah berurutan. Kriteria fisik yang sama
dipakai verifikasi expert (setelah release/retreat). Bukan hanya jarak XY atau lift.
Laporan `place_attempts.jsonl` mencatat `no_lift`, `wrong_object`,
`wrong_receptacle`, `not_released`, `distractor_displacement`, atau `place_timeout`.
Untuk variasi jumlah scene, gunakan `--object-counts 1 2 3` dan
`--receptacle-counts 1 2 3` sebagai pengganti count tunggal saat collection;
evaluasi tiap kombinasi dengan output terpisah. Rollout otomatis memakai
jumlah maksimum yang tercatat, bukan mengevaluasi semua kombinasi.
