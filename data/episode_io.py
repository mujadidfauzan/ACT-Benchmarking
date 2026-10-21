"""Atomic HDF5 episode storage and explicit, non-destructive NPZ migration."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np


EPISODE_SUFFIX = ".hdf5"
HDF5_SUFFIXES = {".hdf5", ".h5"}
FORMAT_NAME = "visual_bc_episode"
FORMAT_VERSION = 1


def open_episode(path):
    path = Path(path)
    if path.suffix not in HDF5_SUFFIXES:
        raise ValueError(f"Expected an HDF5 episode: {path}; convert old NPZ demos with python -m data.episode_io")
    handle = h5py.File(path, "r")
    if handle.attrs.get("format") != FORMAT_NAME or handle.attrs.get("format_version") != FORMAT_VERSION:
        handle.close()
        raise ValueError(f"Unsupported episode format/version: {path}")
    return handle


def read_json(dataset, field):
    if dataset.shape != () or h5py.check_string_dtype(dataset.dtype) is None:
        raise ValueError(f"{field} must contain a scalar JSON string")
    return json.loads(dataset.asstr()[()])


def write_episode(path, arrays):
    """Keep existing array names, with frame-sized RGB chunks and UTF-8 strings."""
    path = Path(path)
    if path.suffix not in HDF5_SUFFIXES:
        raise ValueError(f"Episode extension must be .hdf5 or .h5: {path}")
    if path.exists():
        raise FileExistsError(f"Episode already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with h5py.File(temporary_path, "w") as handle:
            handle.attrs.update(format=FORMAT_NAME, format_version=FORMAT_VERSION)
            for key, value in arrays.items():
                value = np.asarray(value)
                if value.dtype.kind in {"U", "S"}:
                    text = value.astype(str).astype(object)
                    handle.create_dataset(key, data=text, dtype=h5py.string_dtype("utf-8"))
                else:
                    options = {}
                    if value.ndim and value.size:
                        options = dict(compression="gzip", compression_opts=4, shuffle=True)
                        if key.endswith("_image") and value.ndim == 4:
                            options["chunks"] = (1, *value.shape[1:])
                    handle.create_dataset(key, data=value, **options)
        if path.exists():
            raise FileExistsError(f"Episode already exists: {path}")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_dataset(source, output, training_dir=None):
    """Convert manifest-listed NPZ episodes to a new dataset; never edit source."""
    from data.bc_preparation import default_training_dir, prepare_bc_training

    source, output = Path(source).resolve(), Path(output).resolve()
    target = (default_training_dir(output) if training_dir is None else Path(training_dir)).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Choose a new output dataset: {output}")
    if source == output or source in output.parents:
        raise ValueError("Conversion output must be outside the source dataset")
    if target == source or source in target.parents:
        raise ValueError("Training artifacts must be outside the source dataset")
    if target == output:
        raise ValueError("training-dir must differ from the output dataset directory")
    if target.exists():
        raise FileExistsError(f"Choose a new training-dir: {target}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.converting-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for task in ("pick", "place", "stack"):
            directory = source / task
            manifest = directory / "manifest.jsonl"
            disk_names = {path.name for path in directory.glob("episode_*.npz")}
            if not manifest.is_file():
                if disk_names:
                    raise ValueError(f"Unmanifested NPZ episodes in {directory}")
                continue
            entries, names = [], set()
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                name = entry["path"]
                if Path(name).name != name or Path(name).suffix != ".npz" or name in names:
                    raise ValueError(f"Invalid/duplicate NPZ manifest path: {name}")
                names.add(name)
                converted = Path(name).with_suffix(EPISODE_SUFFIX).name
                with np.load(directory / name, allow_pickle=False) as archive:
                    write_episode(staging / task / converted, {key: archive[key] for key in archive.files})
                entries.append({**entry, "path": converted})
                count += 1
            if disk_names != names:
                raise ValueError(f"NPZ files and manifest differ in {directory}")
            destination = staging / task
            destination.mkdir(exist_ok=True)
            (destination / "manifest.jsonl").write_text(
                "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
            for extra in (*directory.glob("collection_*.json"), directory / "attempts.jsonl"):
                if extra.is_file():
                    shutil.copy2(extra, destination / extra.name)
        if count < 2:
            raise ValueError("At least two successful demos are needed for conversion and train/validation")
        # Validate the converted dataset before publishing; original recordings stay untouched.
        prepared = Path(temporary) / "prepared"
        prepare_bc_training(staging, prepared)
        staging.rename(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prepared, target)
    return json.loads((target / "training.json").read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert legacy NPZ demonstrations into a new HDF5 dataset")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = convert_dataset(args.source, args.output, args.training_dir)
    print(f"Converted {config['train_episodes'] + config['val_episodes']} episodes to {args.output}")
    print("Train/validation and normalization are ready; source NPZ files were not modified")


if __name__ == "__main__":
    main()
