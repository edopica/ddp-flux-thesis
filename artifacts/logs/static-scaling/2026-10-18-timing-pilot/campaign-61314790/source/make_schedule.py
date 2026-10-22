"""
Phase 2 static-scaling schedule generator (numpy only).

Writes the frozen global sample schedule into the prepared dataset directory:
  permutations.npy   int64 (EPOCHS, 100000): row e = permutation of 0..99999
  augmentation.npy   uint8 (EPOCHS, 100000, 3): (crop_top, crop_left, flip)
  schedule_manifest.json

Frozen RNG scheme (recorded 2026-10-16; specification.md 3.3):
  ROOT_SCHEDULE = 20261016
  ss_perm = numpy.random.SeedSequence([20261016, 1, epoch]); permutation =
            Generator(PCG64(ss_perm)).permutation(100000)
  ss_aug  = numpy.random.SeedSequence([20261016, 2, epoch]); per id ascending:
            crop_top = gen.integers(0, 9); crop_left = gen.integers(0, 9);
            flip = gen.integers(0, 2)   (three scalar draws per id)
  Epochs are independent: extending the schedule with more epochs never changes
  an earlier epoch's rows (extension rule recorded in the manifest).

Schedule invariants (specification.md 2): effective samples per epoch 99840,
the same final 160 permutation entries dropped in every configuration; updates
per epoch 195 at batch 512 and 390 at batch 256.

Refuses to overwrite an existing schedule_manifest.json unless --force.

Reproduction (Leonardo compute node; numpy only):
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/make_schedule.py \\
    --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared

Deterministic stdout lines: SCHEDULE <key>=<value>.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import dataset_adapter as da


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the frozen epoch permutations and augmentation records")
    p.add_argument("--out-dir", required=True, help="prepared dataset directory ($PREP)")
    p.add_argument("--epochs", type=int, default=da.EPOCHS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--source-state-sha256", default=None)
    args = p.parse_args(argv)

    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "schedule_manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        raise SystemExit("refusing to overwrite %s (use --force)" % manifest_path)

    perms = np.empty((args.epochs, da.N_TRAIN), dtype=np.int64)
    augs = np.empty((args.epochs, da.N_TRAIN, 3), dtype=np.uint8)
    for e in range(args.epochs):
        perm = da.epoch_permutation(e)
        sorted_row = np.sort(perm)
        if not np.array_equal(sorted_row, np.arange(da.N_TRAIN, dtype=np.int64)):
            raise SystemExit("epoch %d permutation is not a permutation of 0..%d" % (e, da.N_TRAIN - 1))
        perms[e] = perm
        augs[e] = da.epoch_augmentation(e)
        if (e + 1) % 5 == 0 or e == args.epochs - 1:
            print("SCHEDULE generated_epochs=%d" % (e + 1), flush=True)

    n_eff = da.epoch_sample_ids(perms[0], da.EPOCH_BATCH)
    checks = {
        "samples_per_epoch": int(len(n_eff)),
        "tail_size": int(da.N_TRAIN - len(n_eff)),
        "updates_per_epoch_b512": int(len(n_eff) // da.EPOCH_BATCH),
        "updates_per_epoch_b256": int(len(n_eff) // da.FALLBACK_BATCH),
    }
    if checks["samples_per_epoch"] != da.SAMPLES_PER_EPOCH or checks["tail_size"] != da.TAIL_SIZE:
        raise SystemExit("tail arithmetic drifted: %s" % checks)

    # Determinism self-check: a fresh generator must reproduce epoch 0 bitwise.
    if not np.array_equal(da.epoch_permutation(0), perms[0]):
        raise SystemExit("epoch 0 permutation is not reproducible from the recorded scheme")
    if not np.array_equal(da.epoch_augmentation(0), augs[0]):
        raise SystemExit("epoch 0 augmentation is not reproducible from the recorded scheme")

    perm_path = os.path.join(args.out_dir, "permutations.npy")
    aug_path = os.path.join(args.out_dir, "augmentation.npy")
    np.save(perm_path, perms)
    np.save(aug_path, augs)

    manifest = {
        "producer": "scripts/static_scaling/make_schedule.py",
        "created": da.utc_now(),
        "synthetic": False,
        **da.source_info(args.source_commit, args.source_state_sha256),
        "generator": {
            "library": "numpy",
            "version": np.__version__,
            "bit_generator": "PCG64",
            "seeding": "numpy.random.SeedSequence([ROOT_SCHEDULE, domain, epoch])",
        },
        "root_seed": da.ROOT_SCHEDULE,
        "domains": {"1": "permutations", "2": "augmentation"},
        "scheme": (
            "permutation: Generator(PCG64(SeedSequence([20261016, 1, epoch]))).permutation(100000); "
            "augmentation: Generator(PCG64(SeedSequence([20261016, 2, epoch]))), then for id in 0..99999 "
            "ascending three scalar draws: crop_top = gen.integers(0, 9), crop_left = gen.integers(0, 9), "
            "flip = gen.integers(0, 2)"
        ),
        "epochs": args.epochs,
        "extension_rule": (
            "an epoch beyond the stored range is generated independently with the same "
            "SeedSequence([20261016, DOMAIN, epoch]); adding a later epoch never changes the stream "
            "or the rows of an earlier epoch"
        ),
        "schedule_invariants": checks,
        "files": [
            {"path": "permutations.npy", "dtype": "int64", "shape": [args.epochs, da.N_TRAIN],
             "layout": "epoch_major_sample_id", "sha256": da.sha256_file(perm_path)},
            {"path": "augmentation.npy", "dtype": "uint8", "shape": [args.epochs, da.N_TRAIN, 3],
             "layout": "epoch_sample_top_left_flip", "sha256": da.sha256_file(aug_path)},
        ],
    }
    da.write_json(manifest_path, manifest)
    print("SCHEDULE wrote %s epochs=%d" % (manifest_path, args.epochs))
    print("SCHEDULE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
