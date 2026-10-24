# Deterministic image BC-RNN

Implementasi lokal yang mengikuti resep **image BC-RNN deterministic** robomimic,
dengan pembaca dataset HDF5 dan simulator proyek ini. Tidak memerlukan paket
`robomimic`, konversi dataset, atau download pretrained weights. Dependensi
torch/torchvision yang sudah dipakai eksperimen C cukup.

Ini adaptasi resep untuk task kita, bukan reproduksi persis hasil paper atau
checkpoint yang kompatibel dengan paket robomimic.

## Model dan objective

Setiap kamera memiliki encoder independen: RGB `[0,1]` -> resize 84x84 -> satu
random crop 76x76 per frame saat training (center crop saat validation/eval)
-> ResNet18 dari awal dengan BatchNorm -> Conv1x1 512 ke 32 keypoint maps
-> spatial softmax temperature 1.0 -> 64 koordinat -> Linear64 -> ReLU.
Batch dan waktu diratakan saat encoding, sesuai pemrosesan sequence image.

Fitur kamera digabung dengan proprio **9D** dalam urutan EEF position (3),
EEF quaternion (4), gripper qpos (2), tanpa standardisasi. Tidak ada joint
position, privileged object poses, atau previous commanded gripper sebagai
input policy. Default LSTM dua layer, hidden 1000, sequence/horizon 10,
diikuti Linear7 dan tanh. Tidak ada hidden MLP setelah LSTM atau GMM.

Action target adalah command controller asli dalam `[-1,1]`:
delta XYZ (3), delta rotasi (3), dan gripper (1). Tidak memakai normalizer A,
BCE, threshold gripper, atau pembobotan loss per bagian. Loss = MSE seluruh
7 komponen action; diagnostik XYZ/rotation/gripper dilaporkan terpisah.
Optimizer Adam LR 1e-4, weight decay 0, tanpa gradient clipping atau scheduler.
Perintah gripper kontinu dikirim langsung ke controller yang sama.

Referensi implementasi dan konfigurasi:

- [Image BC-RNN dan opsi deterministic MG](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/generate_paper_configs.py)
- [VisualCore dan CropRandomizer](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/obs_core.py)
- [Feature activation ObservationEncoder](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/obs_nets.py)
- [RNNActorNetwork, tanh output](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/policy_nets.py)
- [Loss dan pengelolaan recurrent state](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/algo/bc.py)

## Dataset asli

`BCRNNSequenceDataset` membaca split manifest dan episode yang sama dengan A/B/C.
Setiap timestep action menjadi awal satu window (stride 1), tidak menyeberangi
episode. Pasangan training adalah `obs[t] -> action[t]`. Window akhir mengulang
observasi dan action terakhir sampai panjang 10; padding ikut MSE sebagaimana
default padded SequenceDataset + deterministic BC-RNN rujukan. Tidak memakai
mask loss milik trainer A. RGB dibaca sebagai uint8 agar hemat memori dan baru
diubah ke float/di-resize di model.

`training.json` lama tetap dipakai untuk memverifikasi source digest, split,
kamera, dan environment. Statistik normalisasi di dalamnya tidak diterapkan
ke policy ini. Jika split belum tersedia, trainer membuat metadata split dengan
pipeline existing; berkas demo tidak dikonversi atau ditulis ulang.

## Training dan checkpoint

Anggaran memakai **optimizer updates**, bukan putaran penuh dataset. Default
300000 update mengacu pada 600 x 500 update resep image paper. Mulai dengan
60000 untuk pemeriksaan awal. Batch default 8 adalah adaptasi VRAM terhadap
referensi 16; tidak ada akumulasi. `--max-updates` adalah total termasuk resume.

Checkpoint/validation setiap 500 update. Default validation mengukur 50 batch
window acak yang tetap (seed terpisah); `--validation-batches 0` mengevaluasi
seluruh window validation. Tidak ada early stopping berdasarkan loss.
Rollout default setiap 10000 update, 10 episode x 5 seed (1000..1004), horizon
300, menggunakan factory/controller dan hold-success 20 langkah proyek ini.
`--rollout-eval-every 0` menonaktifkan rollout otomatis.

- `latest.pt`: checkpoint terakhir, termasuk optimizer, RNG CPU/CUDA, urutan
  window dan cursor sampler untuk resume tanpa mengulang shuffle.
- `best.pt`: validation MSE terendah.
- `best_rollout.pt`: rollout validation success tertinggi (tie mempertahankan
  checkpoint sebelumnya). Hanya tersedia setelah rollout pertama.
- `config.json`, `metrics.json`, `rollout_evaluations.json`: konfigurasi,
  kurva terhadap updates, dan riwayat pemilihan checkpoint.

Resume memakai `--resume <run>/latest.pt --max-updates <total-baru>` dan output
directory yang sama. Hyperparameter disimpan/dipulihkan; perubahan batch,
arsitektur, dataset/split, atau protokol rollout pada resume ditolak.
Kesetaraan numerik resume diuji pada CPU; perpindahan device tidak menjamin
hasil bitwise identik.

## Linux: train lalu eval

Jalankan dari root repository, setelah memperbarui kode di PC remote. Kamera
dibaca otomatis dari dataset. Output berikut untuk eksperimen baru:

```bash
export MUJOCO_GL=egl && \
python -u -m scripts.train_robomimic_bc_rnn \
  --dataset data/demos/place_red_1cube_1tray_v01 \
  --batch-size 8 --max-updates 60000 --device cuda \
  --output-dir results/robomimic_bc_rnn/place_red_1cube_1tray_v01 && \
python -u -m scripts.evaluate_robomimic_bc_rnn \
  --checkpoint results/robomimic_bc_rnn/place_red_1cube_1tray_v01/best_rollout.pt \
  --num-objects 1 --num-receptacles 1 \
  --episodes 20 --num-seeds 5 --seed 2000 --max-steps 300 --device cuda \
  --record-video --log-rollout \
  --output-dir results/robomimic_bc_rnn_evaluation/place_red_1cube_1tray_v01_h300
```

Evaluator memulihkan seluruh konfigurasi checkpoint. LSTM di-reset tiap 10
action dan di awal episode. Tidak memakai opsi reset GRU atau checkpoint A/B/C.
Log action memakai controller units; field legacy `normalized_action` identik
dengan `raw_action`. `gripper_logit` dan `previous_gripper_input` bernilai null
karena policy tidak memakai kedua konsep tersebut. Summary menyimpan jumlah
update, preprocessing, image/crop size, dan horizon reset yang sebenarnya.

Tes offline: `.venv/bin/python -m tests` (Linux) atau `python -m tests` di env
aktif. Tes mencakup encoder ResNet asli dengan backward, kesetaraan streaming
dan sequence, raw MSE, alignment/padding, checkpoint, resume, serta integrasi
evaluator tanpa menjalankan simulator fisik atau training panjang.
