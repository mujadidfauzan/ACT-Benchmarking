"""HDF5 persistence, partial reads, worker isolation, and legacy migration."""

import json
import pickle
import subprocess
import sys
from unittest.mock import patch
import unittest

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.bc_preparation import dataset_entries, load_bc_training
from data.episode_io import convert_dataset, open_episode, read_json, write_episode
from data.manipulation_dataset import CAMERA_OBSERVATION_KEYS, EpisodeSequenceDataset, ManipulationDataset
from tests.regression import test_visual_bc as fixtures


class HDF5StorageTests(unittest.TestCase):
    setUp = fixtures.VisualBCTests.setUp
    prepare = fixtures.VisualBCTests.prepare

    def loader(self, cls=ManipulationDataset, **kwargs):
        config = self.prepare()
        return cls(self.dataset, self.prepared / "train.jsonl", proprio_keys=config["proprio_keys"],
                   camera_keys=tuple(CAMERA_OBSERVATION_KEYS.values()), **kwargs)

    def legacy_dataset(self):
        root = self.root / "legacy"
        (root / "pick").mkdir(parents=True)
        entries = []
        for entry in self.entries:
            name = entry["path"].replace(".hdf5", ".npz")
            with open_episode(self.dataset / "pick" / entry["path"]) as episode:
                arrays = {key: np.asarray(value.asstr()[()], dtype=str) if h5py.check_string_dtype(value.dtype)
                          else value[()] for key, value in episode.items()}
            np.savez_compressed(root / "pick" / name, **arrays)
            entries.append({**entry, "path": name})
        (root / "pick/manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        (root / "pick/attempts.jsonl").write_text('{"success": true}\n')
        return root

    def test_round_trip_rgb_chunks_utf8_and_overwrite_protection(self):
        path = self.root / "round_trip.hdf5"
        rgb = np.arange(3 * 32 * 32 * 3, dtype=np.uint8).reshape(3, 32, 32, 3)
        payload = {"actions": np.zeros((2, 7), np.float32), "obs__agentview_image": rgb,
                   "phases": np.asarray(["approach", "close"]),
                   "metadata_json": np.asarray(json.dumps({"label": "merah \u2192 angkat"}, ensure_ascii=False)),
                   "result_json": np.asarray('{"success": true}')}
        write_episode(path, payload)
        with open_episode(path) as episode:
            self.assertEqual(episode["obs__agentview_image"].chunks, (1, 32, 32, 3))
            self.assertEqual(episode["obs__agentview_image"].compression, "gzip")
            np.testing.assert_array_equal(episode["obs__agentview_image"][:], rgb)
            self.assertEqual(episode["phases"].asstr()[:].tolist(), ["approach", "close"])
            self.assertEqual(read_json(episode["metadata_json"], "metadata_json")["label"], "merah \u2192 angkat")
            self.assertEqual(read_json(episode["result_json"], "result_json"), {"success": True})
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_episode(path, payload)
        self.assertEqual(path.read_bytes(), original)

    def test_failed_save_leaves_no_published_or_temporary_episode(self):
        path = self.root / "broken.hdf5"
        with self.assertRaises(TypeError):
            write_episode(path, {"actions": np.zeros((2, 7)), "invalid": np.asarray([object()])})
        self.assertFalse(path.exists())
        self.assertFalse(list(self.root.glob(".broken.hdf5.*.tmp")))
        with self.assertRaises(ValueError):
            write_episode(self.root / "wrong.npz", {})

    def test_frame_and_sequence_read_only_requested_rgb_slices(self):
        frame = self.loader()
        sequence = self.loader(EpisodeSequenceDataset, sequence_length=4, sequence_stride=2)
        self.addCleanup(frame.close)
        self.addCleanup(sequence.close)
        reads = []
        original = h5py.Dataset.__getitem__

        def traced(dataset, selection, *args, **kwargs):
            if dataset.name.endswith("_image"):
                reads.append(selection)
            return original(dataset, selection, *args, **kwargs)

        with patch.object(h5py.Dataset, "__getitem__", traced):
            frame[0]
            self.assertEqual(reads, [0, 0])
            reads.clear()
            window = sequence[0]
            valid = int(window["valid_mask"].sum())
            self.assertEqual(reads, [slice(0, valid), slice(0, valid)])

    def test_handle_eviction_close_and_pickling(self):
        dataset = self.loader(cache_size=1)
        self.addCleanup(dataset.close)
        first_index = next(index for index, (episode, _) in enumerate(dataset._sample_index) if episode == 1)
        dataset[0]
        first = dataset._cache[0]
        dataset[first_index]
        self.assertFalse(first.id.valid)
        self.assertEqual(len(dataset._cache), 1)
        second = dataset._cache[1]
        restored = pickle.loads(pickle.dumps(dataset))
        self.addCleanup(restored.close)
        self.assertEqual(len(restored._cache), 0)
        torch.testing.assert_close(dataset[0]["image"], restored[0]["image"])
        dataset.close()
        self.assertFalse(second.id.valid)
        uncached = self.loader(cache_size=0)
        uncached[0]
        self.assertFalse(uncached._cache)

    def test_spawn_worker_can_read_after_parent_opened_an_episode(self):
        dataset = self.loader(cache_size=2)
        self.addCleanup(dataset.close)
        expected = dataset[0]["image"]
        loader = DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn")
        iterator = iter(loader)
        try:
            batch = next(iterator)
            torch.testing.assert_close(batch["image"][0], expected)
        finally:
            del iterator

    def test_conversion_preserves_arrays_source_and_train_statistics(self):
        original_config = self.prepare()
        source = self.legacy_dataset()
        originals = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
        output, prepared = self.root / "converted", self.root / "converted_training"
        config = convert_dataset(source, output, prepared)
        loaded, _ = load_bc_training(output, prepared)
        self.assertEqual(loaded, config)
        self.assertEqual(config["storage_format"], "hdf5")
        self.assertEqual(config["normalization"]["proprio"], original_config["normalization"]["proprio"])
        self.assertEqual(config["normalization"]["action"], original_config["normalization"]["action"])
        self.assertEqual((output / "pick/attempts.jsonl").read_bytes(), originals[source / "pick/attempts.jsonl"])
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)
        for entry in dataset_entries(output):
            with np.load(source / entry["path"].replace(".hdf5", ".npz"), allow_pickle=False) as legacy, open_episode(output / entry["path"]) as episode:
                for key in legacy.files:
                    if legacy[key].dtype.kind in {"U", "S"}:
                        np.testing.assert_array_equal(episode[key].asstr()[()], legacy[key])
                    else:
                        np.testing.assert_array_equal(episode[key][()], legacy[key])

    def test_conversion_refuses_existing_targets_and_invalid_data_without_publication(self):
        source = self.legacy_dataset()
        output, prepared = self.root / "converted", self.root / "converted_training"
        prepared.mkdir()
        with self.assertRaises(FileExistsError):
            convert_dataset(source, output, prepared)
        self.assertFalse(output.exists())
        prepared.rmdir()
        with self.assertRaises(ValueError):
            convert_dataset(source, source / "nested", prepared)
        with self.assertRaises(ValueError):
            convert_dataset(source, output, source / "artifacts")
        path = source / "pick" / self.entries[0]["path"].replace(".hdf5", ".npz")
        path.write_bytes(b"interrupted NPZ file")
        with self.assertRaises((ValueError, OSError)):
            convert_dataset(source, output, prepared)
        self.assertFalse(output.exists())
        self.assertFalse(prepared.exists())

    def test_legacy_or_mixed_datasets_are_not_silently_used(self):
        source = self.legacy_dataset()
        with self.assertRaisesRegex(ValueError, "Legacy NPZ"):
            dataset_entries(source)
        np.savez(self.dataset / "pick/episode_999999.npz", actions=np.zeros((1, 7)))
        with self.assertRaisesRegex(ValueError, "Legacy NPZ"):
            self.prepare()

    def test_conversion_cli_prepares_artifacts(self):
        source = self.legacy_dataset()
        output, prepared = self.root / "cli_converted", self.root / "cli_training"
        result = subprocess.run([sys.executable, "-m", "data.episode_io", "--source", str(source),
                                 "--output", str(output), "--training-dir", str(prepared)],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Converted 10 episodes", result.stdout)
        self.assertEqual(load_bc_training(output, prepared)[0]["storage_format"], "hdf5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
