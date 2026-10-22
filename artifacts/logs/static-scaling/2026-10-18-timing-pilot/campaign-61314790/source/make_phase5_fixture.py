#!/usr/bin/env python3
"""Build the Phase 5 memory-pilot fixture from the prepared Tiny ImageNet data.

The fixture contains two complete global batches taken from the frozen epoch
schedule (default epoch 0). Both frameworks read the same saved arrays, so the
memory pilot exercises real decoded/transformed samples and the selected-batch
correctness check can reuse the same directory as a Phase 4 fixture.

Outputs in ``--out-dir``:

  data_fixture.npz          train_transformed_ref (2, B, 3, 64, 64) float32
                            train_labels (B,) int64, normalization constants,
                            decoded uint8 inputs, augmentation records, ids
  schedule_fixture.npz      byte copy of the canonical Phase 2 schedule fixture
  initial_bundle.npz        byte copy of the canonical initial bundle
  fixture_manifest.json     file and per-array C-order SHA-256 entries

Reproduction (Leonardo compute node):

  $VENV/bin/python scripts/static_scaling/make_phase5_fixture.py \
    --prep-dir $WORK/epicazio/static-scaling-prepared \
    --schedule-source artifacts/baselines/static-scaling/2026-10-16-phase2 \
    --bundle-source artifacts/baselines/static-scaling/2026-10-16-phase2 \
    --batch-size 512 --epoch 0 --out-dir $OUT/fixture-b512 --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

import dataset_adapter as da


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prep-dir", required=True, help="Prepared dataset directory ($PREP)")
    p.add_argument("--out-dir", required=True, help="Output fixture directory")
    p.add_argument("--schedule-source", required=True,
                   help="Directory containing the canonical schedule_fixture.npz")
    p.add_argument("--bundle-source", required=True,
                   help="Directory containing the canonical initial_bundle.npz")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Global batch size (512 or 256; must divide 99840)")
    p.add_argument("--epoch", type=int, default=0, help="Frozen schedule epoch")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--source-state-sha256", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def load_prepared(prep_dir: str):
    for name in ("train_images.npy", "train_labels.npy", "permutations.npy",
                 "augmentation.npy", "stats.json"):
        path = os.path.join(prep_dir, name)
        if not os.path.isfile(path):
            raise SystemExit("missing prepared file: %s" % path)
    images = da.open_prepared_train_images(prep_dir)
    labels = np.load(os.path.join(prep_dir, "train_labels.npy"))
    perms = da.open_prepared_permutations(prep_dir)
    augs = da.open_prepared_augmentation(prep_dir)
    mean32, std32, stats = da.load_stats(prep_dir)
    return images, labels, perms, augs, mean32, std32, stats


def build_batch(images, labels, augs, epoch, ids, mean32, std32):
    uint8 = np.asarray(images[np.asarray(ids)])
    triples = np.asarray(augs[epoch][np.asarray(ids)])
    transformed = da.replay_transform(uint8, triples, mean32, std32)
    return uint8, triples, transformed, np.asarray(labels[np.asarray(ids)], dtype=np.int64)


def array_entry(npz_name, name, arr):
    return {
        "name": npz_name + "::" + name,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "layout": "nchw_float32_normalized" if arr.dtype == np.float32 else str(arr.dtype),
        "sha256": da.sha256_c_order(arr),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    batch = int(args.batch_size)
    if batch <= 0:
        raise SystemExit("--batch-size must be positive")
    if da.SAMPLES_PER_EPOCH % batch != 0:
        raise SystemExit("batch %d does not divide %d" % (batch, da.SAMPLES_PER_EPOCH))
    if args.epoch < 0 or args.epoch >= da.EPOCHS:
        raise SystemExit("--epoch %d outside 0..%d" % (args.epoch, da.EPOCHS - 1))

    out_dir = os.path.abspath(args.out_dir)
    schedule_src = os.path.join(os.path.abspath(args.schedule_source), "schedule_fixture.npz")
    bundle_src = os.path.join(os.path.abspath(args.bundle_source), "initial_bundle.npz")
    if not os.path.isfile(schedule_src):
        raise SystemExit("missing schedule fixture: %s" % schedule_src)
    if not os.path.isfile(bundle_src):
        raise SystemExit("missing initial bundle: %s" % bundle_src)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "fixture_manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        raise SystemExit("refusing to overwrite %s (use --force)" % manifest_path)

    images, labels, perms, augs, mean32, std32, stats = load_prepared(args.prep_dir)
    perm = np.asarray(perms[args.epoch], dtype=np.int64)
    if perm.shape != (da.N_TRAIN,):
        raise SystemExit("epoch %d permutation has shape %s" % (args.epoch, perm.shape))
    ids = perm[:batch]
    epochs = [int(args.epoch), int((args.epoch + 1) % da.EPOCHS)]
    u0, t0, x0, y0 = build_batch(images, labels, augs, epochs[0], ids, mean32, std32)
    u1, t1, x1, y1 = build_batch(images, labels, augs, epochs[1], ids, mean32, std32)
    if not np.array_equal(y0, y1):
        raise SystemExit("the two pilot rows have different label vectors")

    data_path = os.path.join(out_dir, "data_fixture.npz")
    np.savez(
        data_path,
        train_transformed_ref=np.stack([x0, x1], axis=0),
        train_labels=y0,
        decoded_train_images=u0,
        augmentation=np.stack([t0, t1], axis=0),
        sample_ids=np.stack([ids, ids], axis=0),
        normalization_mean=np.asarray(mean32, dtype=np.float32),
        normalization_std=np.asarray(std32, dtype=np.float32),
    )

    schedule_dst = os.path.join(out_dir, "schedule_fixture.npz")
    bundle_dst = os.path.join(out_dir, "initial_bundle.npz")
    shutil.copyfile(schedule_src, schedule_dst)
    shutil.copyfile(bundle_src, bundle_dst)

    with np.load(data_path) as archive:
        arrays = [
            array_entry("data_fixture.npz", name, np.asarray(archive[name]))
            for name in archive.files
        ]
    schedule_arrays = []
    with np.load(schedule_dst) as archive:
        schedule_arrays = [
            array_entry("schedule_fixture.npz", name, np.asarray(archive[name]))
            for name in archive.files
        ]

    files = []
    for path in (data_path, schedule_dst, bundle_dst):
        files.append({
            "path": os.path.basename(path),
            "bytes": os.path.getsize(path),
            "sha256": da.sha256_file(path),
        })

    manifest = {
        "producer": "scripts/static_scaling/make_phase5_fixture.py",
        "created": da.utc_now(),
        "synthetic": False,
        **da.source_info(args.source_commit, args.source_state_sha256),
        "purpose": "Phase 5 memory pilot and selected-batch correctness fixture",
        "schema": "flat arrays list; name = '<npz file>::<array name>'; "
                  "sha256 = sha256 of arr.tobytes(order=\"C\"); files carry the npz byte hash",
        "batch_size": batch,
        "epoch": int(args.epoch),
        "prep_dir": os.path.abspath(args.prep_dir),
        "schedule_source": os.path.abspath(args.schedule_source),
        "bundle_source": os.path.abspath(args.bundle_source),
        "batch_epochs": epochs,
        "sample_ids": {
            "count": int(batch),
            "first": int(ids[0]),
            "last": int(ids[-1]),
            "sha256": da.sha256_c_order(ids),
        },
        "normalization": {
            "mean": [float(x) for x in np.asarray(mean32)],
            "std": [float(x) for x in np.asarray(std32)],
            "stats_sha256": da.sha256_file(os.path.join(args.prep_dir, "stats.json")),
            "stats_pixel_count_per_channel": stats.get("pixel_count_per_channel"),
        },
        "files": files,
        "arrays": arrays + schedule_arrays,
    }
    da.write_json(manifest_path, manifest)
    print("PHASE5_FIXTURE DATA batch=%d epoch=%d arrays=%d dir=%s"
          % (batch, args.epoch, len(arrays), out_dir))
    print("PHASE5_FIXTURE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
