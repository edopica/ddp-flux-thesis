"""Shared numpy-only helpers for the static-scaling full-dataset reference.

The definitions here mirror ``full_dataset.jl`` byte for byte: C-order hashes,
the frozen transform specification, manifest-declared file hashes, the frozen
bundle traversal order, epoch ids, rank shards, and the update-1 reference.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

import dataset_adapter as da

TRANSFORM_SPEC = (
    b"pad=4|crop=64|top_left_zero_based|flip_when=1|uint8_to_float32|"
    b"div255|sub_mean|div_std|layout=WHCN"
)

REQUIRED_FILES = (
    "train_images.npy",
    "train_labels.npy",
    "permutations.npy",
    "augmentation.npy",
    "stats.json",
    "dataset_manifest.json",
    "schedule_manifest.json",
)
OPTIONAL_FILES = ("initial_bundle.npz", "initial_bundle.manifest.json")
LARGE_FILES = ("train_images.npy", "train_labels.npy", "val_images.npy")
DEFAULT_WORLDS = (1, 2, 4)


def sha256_c_order(arr) -> str:
    return da.sha256_c_order(arr)


def _manifest_file_records(path, source):
    records = {}
    if not os.path.isfile(path):
        return records
    document = da.read_json(path)
    entries = document.get("files")
    if not isinstance(entries, list):
        return records
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("path")
        if not isinstance(name, str) or not name:
            continue
        record = records.setdefault(name, {})
        record["source"] = source
        for key in ("bytes", "sha256", "dtype", "shape"):
            if key in entry:
                record[key] = entry[key]
    return records


def _declared_files(prep_dir):
    declared = {}
    declared.update(_manifest_file_records(
        os.path.join(prep_dir, "dataset_manifest.json"), "dataset_manifest.json"))
    declared.update(_manifest_file_records(
        os.path.join(prep_dir, "schedule_manifest.json"), "schedule_manifest.json"))
    return declared


def dataset_file_hashes(prep_dir, verify_large=False):
    """Return ``name -> {bytes, sha256, source}`` for one prepared dataset.

    Small files are always hashed and compared with manifest declarations.
    Large image arrays use the manifest digest unless ``verify_large`` is true;
    their sizes are always checked.
    """
    prep_dir = os.path.abspath(prep_dir)
    if not os.path.isdir(prep_dir):
        raise FileNotFoundError("dataset directory not found: %s" % prep_dir)
    declared = _declared_files(prep_dir)

    names = list(REQUIRED_FILES)
    for name in OPTIONAL_FILES:
        if os.path.isfile(os.path.join(prep_dir, name)) and name not in names:
            names.append(name)
    for name in sorted(declared):
        if name not in names and os.path.isfile(os.path.join(prep_dir, name)):
            names.append(name)

    hashes = {}
    for name in names:
        path = os.path.join(prep_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError("dataset file not found: %s" % path)
        size = os.path.getsize(path)
        declaration = declared.get(name, {})
        declared_sha = declaration.get("sha256")
        large = name in LARGE_FILES
        computed = ""
        if (not large) or verify_large or not declared_sha:
            computed = da.sha256_file(path)
        if declared_sha is not None and computed and computed != declared_sha:
            raise ValueError(
                "dataset file hash mismatch for %s: actual=%s declared=%s"
                % (name, computed, declared_sha))
        if "bytes" in declaration and int(declaration["bytes"]) != size:
            raise ValueError(
                "dataset file size mismatch for %s: actual=%d declared=%d"
                % (name, size, int(declaration["bytes"])))
        digest = computed or declared_sha
        if not digest:
            digest = da.sha256_file(path)
        hashes[name] = {
            "bytes": int(size),
            "sha256": str(digest),
            "source": str(declaration.get("source", "actual-only")),
        }
    return hashes


def transform_spec_sha256(mean32, std32) -> str:
    mean = np.asarray(mean32, dtype="<f4").reshape(-1)
    std = np.asarray(std32, dtype="<f4").reshape(-1)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("mean32 and std32 must each have 3 entries")
    digest = hashlib.sha256()
    digest.update(TRANSFORM_SPEC)
    digest.update(mean.tobytes(order="C"))
    digest.update(std.tobytes(order="C"))
    return digest.hexdigest()


def _bundle_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def bundle_info(prep_dir, bundle_path=None):
    """Verify the frozen bundle against its manifest and return its identities."""
    prep_dir = os.path.abspath(prep_dir)
    manifest_path = os.path.join(prep_dir, "initial_bundle.manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("bundle manifest not found: %s" % manifest_path)
    manifest = da.read_json(manifest_path)
    declared_hash = manifest.get("bundle_hash")
    if not isinstance(declared_hash, str) or not declared_hash:
        raise ValueError("initial_bundle.manifest.json has no bundle_hash")
    path = os.path.abspath(bundle_path) if bundle_path else os.path.join(prep_dir, "initial_bundle.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError("bundle not found: %s" % path)

    names = [spec[0] for spec in da.bundle_specs()]
    arrays = _bundle_arrays(path)
    missing = sorted(set(names) - set(arrays))
    extra = sorted(set(arrays) - set(names))
    if missing or extra:
        raise ValueError("bundle name-set mismatch: missing=%s extra=%s" % (missing[:5], extra[:5]))
    recomputed = da.bundle_hash_of_arrays([arrays[name] for name in names])
    if recomputed != declared_hash:
        raise AssertionError(
            "bundle hash mismatch: recomputed=%s declared=%s" % (recomputed, declared_hash))
    return {
        "bundle_hash": recomputed,
        "bundle_file_sha256": da.sha256_file(path),
        "array_count": len(names),
        "element_count": int(sum(int(arrays[name].size) for name in names)),
        "bundle_path": path,
        "manifest_path": manifest_path,
    }


def load_epoch_arrays(prep_dir, epoch):
    """Open one epoch with read-only memory maps; labels and stats are loaded."""
    prep_dir = os.path.abspath(prep_dir)
    if not os.path.isdir(prep_dir):
        raise FileNotFoundError("dataset directory not found: %s" % prep_dir)
    images = da.open_prepared_train_images(prep_dir)
    labels = np.load(os.path.join(prep_dir, "train_labels.npy"), mmap_mode="r")
    permutations = da.open_prepared_permutations(prep_dir)
    augmentation = da.open_prepared_augmentation(prep_dir)
    mean32, std32, stats = da.load_stats(prep_dir)

    if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
        raise ValueError("train_images.npy must be (N, 3, 64, 64), got %s" % (images.shape,))
    if images.dtype != np.uint8:
        raise ValueError("train_images.npy must be uint8, got %s" % images.dtype)
    if permutations.ndim != 2:
        raise ValueError("permutations.npy must be 2-D, got %s" % (permutations.shape,))
    expected_aug = (permutations.shape[0], permutations.shape[1], 3)
    if augmentation.shape != expected_aug or augmentation.dtype != np.uint8:
        raise ValueError("augmentation.npy must be %s uint8, got %s %s"
                         % (expected_aug, augmentation.shape, augmentation.dtype))
    if labels.shape != (permutations.shape[1],) or labels.dtype != np.int64:
        raise ValueError("train_labels.npy must be (%d,) int64, got %s %s"
                         % (permutations.shape[1], labels.shape, labels.dtype))
    if images.shape[0] != permutations.shape[1]:
        raise ValueError("image count %d does not match permutation width %d"
                         % (images.shape[0], permutations.shape[1]))
    epoch = int(epoch)
    if not 0 <= epoch < permutations.shape[0]:
        raise ValueError("epoch %d outside 0..%d" % (epoch, permutations.shape[0] - 1))

    return {
        "dataset_dir": prep_dir,
        "epoch": epoch,
        "train_images": images,
        "train_labels": labels,
        "permutations": permutations,
        "augmentation": augmentation,
        "epoch_permutation": np.asarray(permutations[epoch]),
        "epoch_augmentation": np.asarray(augmentation[epoch]),
        "mean32": mean32,
        "std32": std32,
        "stats": stats,
    }


def epoch_ids(arrays, batch):
    ids = da.epoch_sample_ids(np.asarray(arrays["epoch_permutation"]), int(batch))
    return np.asarray(ids, dtype=np.int64)


def global_batch_ids(arrays, batch, update):
    batch = int(batch)
    update = int(update)
    ids = epoch_ids(arrays, batch)
    updates = len(ids) // batch
    if not 1 <= update <= updates:
        raise ValueError("update %d outside 1..%d" % (update, updates))
    start = (update - 1) * batch
    return np.asarray(ids[start:start + batch], dtype=np.int64)


def local_batch(arrays, batch, update, rank0, world):
    """Return ``(x_nchw_float32, labels_int64, shard_ids, global_ids)``."""
    world = int(world)
    rank0 = int(rank0)
    gids = global_batch_ids(arrays, batch, update)
    sids = np.asarray(da.rank_shard(gids, rank0, world), dtype=np.int64)
    images = np.asarray(arrays["train_images"][sids])
    aug = np.asarray(arrays["epoch_augmentation"][sids])
    x = da.replay_transform(images, aug, arrays["mean32"], arrays["std32"])
    labels = np.asarray(arrays["train_labels"][sids], dtype=np.int64)
    return (
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(labels, dtype=np.int64),
        np.ascontiguousarray(sids, dtype=np.int64),
        np.ascontiguousarray(gids, dtype=np.int64),
    )


def _rank_reference(perm, batch, world):
    if int(world) == 1:
        return np.asarray(da.rank_sequence(perm, batch, 0, 1), dtype=np.int64)
    rows = [np.asarray(da.rank_sequence(perm, batch, rank, world), dtype=np.int64)
            for rank in range(int(world))]
    return np.stack(rows, axis=0)


def build_reference(arrays, epoch, batch, worlds=DEFAULT_WORLDS):
    """Build the exact NPZ payload shared with the Julia reference."""
    batch = int(batch)
    epoch = int(epoch)
    if int(arrays.get("epoch", epoch)) != epoch:
        raise ValueError("arrays were opened for epoch %r, not %d" % (arrays.get("epoch"), epoch))
    ids = epoch_ids(arrays, batch)
    if len(ids) == 0 or len(ids) % batch != 0:
        raise ValueError("effective epoch length %d is not a positive multiple of %d"
                         % (len(ids), batch))
    if batch % 4 != 0:
        raise ValueError("batch %d is not divisible by 4" % batch)
    for world in worlds:
        if int(world) not in DEFAULT_WORLDS:
            raise ValueError("worlds entries must be 1, 2, or 4, got %r" % (world,))

    gids = global_batch_ids(arrays, batch, 1)
    x, labels, sids, gids_from_local = local_batch(arrays, batch, 1, 0, 1)
    if not np.array_equal(gids, gids_from_local):
        raise AssertionError("update-1 global ids disagree with local_batch")
    if not np.array_equal(sids, gids):
        raise AssertionError("world-1 shard ids disagree with global ids")

    perm = np.asarray(arrays["epoch_permutation"])
    payload = {
        "epoch_ids": np.asarray(ids, dtype=np.int64),
        "rank_ids_R1": _rank_reference(perm, batch, 1),
        "rank_ids_R2": _rank_reference(perm, batch, 2).astype(np.int64, copy=False),
        "rank_ids_R4": _rank_reference(perm, batch, 4).astype(np.int64, copy=False),
        "update1_global_ids": np.asarray(gids, dtype=np.int64),
        "update1_labels": labels,
        "update1_transformed_nchw": x,
        "stats_mean": np.asarray(arrays["mean32"], dtype=np.float32),
        "stats_std": np.asarray(arrays["std32"], dtype=np.float32),
    }
    return payload
