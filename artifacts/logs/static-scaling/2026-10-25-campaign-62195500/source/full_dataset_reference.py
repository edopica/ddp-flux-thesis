#!/usr/bin/env python3
"""Build the frozen full-dataset reference NPZ/JSON (numpy only)."""

from __future__ import annotations

import argparse
import os

import numpy as np

import dataset_adapter as da
import full_dataset_common as fdc

SCHEMA = "static-full-dataset-reference-v1"


def parse_worlds(value):
    worlds = [int(part) for part in str(value).split(",") if part.strip()]
    if not worlds:
        raise argparse.ArgumentTypeError("--worlds must not be empty")
    for world in worlds:
        if world not in (1, 2, 4):
            raise argparse.ArgumentTypeError("--worlds entries must be 1, 2, or 4")
    return worlds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the full-dataset static-scaling reference NPZ/JSON.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--worlds", type=parse_worlds, default=parse_worlds("1,2,4"))
    parser.add_argument("--verify-large-hashes", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _rank_concatenation_matches(rank_ids, ids, batch, world):
    if world == 1:
        rows = [rank_ids]
    else:
        rows = [rank_ids[rank] for rank in range(world)]
    chunk = batch // world
    if chunk * world != batch:
        return False
    updates = len(ids) // batch
    for update in range(updates):
        pieces = [rows[rank][update * chunk:(update + 1) * chunk] for rank in range(world)]
        joined = np.concatenate(pieces) if pieces else np.asarray([], dtype=np.int64)
        if not np.array_equal(joined, ids[update * batch:(update + 1) * batch]):
            return False
    return True


def build_metadata(arrays, reference, args):
    ids = reference["epoch_ids"]
    gids = reference["update1_global_ids"]
    x = reference["update1_transformed_nchw"]
    mean32 = np.asarray(arrays["mean32"], dtype=np.float32)
    std32 = np.asarray(arrays["std32"], dtype=np.float32)
    perm = np.asarray(arrays["epoch_permutation"])
    n = int(len(perm))
    n_eff = int(len(ids))
    tail = int(n - n_eff)
    updates = int(n_eff // args.batch_size)

    missing = set(range(n)) - set(int(value) for value in ids)
    expected_tail = set(int(value) for value in perm[n_eff:])
    missing_sorted = np.asarray(sorted(missing), dtype=np.int64)
    tail_ids = np.asarray(perm[n_eff:], dtype=np.int64)
    ranks_concat_ok = {
        "R1": _rank_concatenation_matches(reference["rank_ids_R1"], ids, args.batch_size, 1),
        "R2": _rank_concatenation_matches(reference["rank_ids_R2"], ids, args.batch_size, 2),
        "R4": _rank_concatenation_matches(reference["rank_ids_R4"], ids, args.batch_size, 4),
    }
    files = fdc.dataset_file_hashes(args.dataset_dir, verify_large=args.verify_large_hashes)
    bundle = fdc.bundle_info(args.dataset_dir)
    spec_sha = fdc.transform_spec_sha256(mean32, std32)
    sample_sha = da.sha256_c_order(ids)
    update1_sha = da.sha256_c_order(gids)
    transform_sha = da.sha256_c_order(x)

    return {
        "schema": SCHEMA,
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "epoch": int(args.epoch),
        "batch_size": int(args.batch_size),
        "n_samples": n,
        "n_epochs": int(np.asarray(arrays["permutations"]).shape[0]),
        "n_eff": n_eff,
        "tail_size": tail,
        "updates_per_epoch": updates,
        "worlds": [int(world) for world in args.worlds],
        "sample_order_sha256": sample_sha,
        "update1_order_sha256": update1_sha,
        "transform_spec_sha256": spec_sha,
        "transform_sample_sha256": transform_sha,
        "bundle_hash": bundle["bundle_hash"],
        "bundle_file_sha256": bundle["bundle_file_sha256"],
        "dataset_files": files,
        "stats_mean": [float(value) for value in mean32],
        "stats_std": [float(value) for value in std32],
        "counts": {
            "epoch_ids": int(len(ids)),
            "epoch_ids_unique": int(len(np.unique(ids))),
            "update1_ids": int(len(gids)),
            "update1_ids_unique": int(len(np.unique(gids))),
            "rank_ids_R1": int(len(reference["rank_ids_R1"])),
            "rank_ids_R2": [int(value) for value in reference["rank_ids_R2"].shape],
            "rank_ids_R4": [int(value) for value in reference["rank_ids_R4"].shape],
            "tail_ids": int(len(tail_ids)),
            "missing_ids": int(len(missing_sorted)),
        },
        "checks": {
            "epoch_ids_unique": bool(len(np.unique(ids)) == len(ids)),
            "epoch_ids_in_range": bool(np.all((ids >= 0) & (ids < n))),
            "missing_equals_final_tail": bool(missing == expected_tail),
            "update1_matches_first_batch": bool(np.array_equal(gids, ids[:args.batch_size])),
            "rank_concatenation_R1": bool(ranks_concat_ok["R1"]),
            "rank_concatenation_R2": bool(ranks_concat_ok["R2"]),
            "rank_concatenation_R4": bool(ranks_concat_ok["R4"]),
            "transform_sample_shape": list(x.shape),
            "transform_sample_dtype": str(x.dtype),
            "stats_mean_dtype": str(mean32.dtype),
            "stats_std_dtype": str(std32.dtype),
        },
        "missing_ids_sha256": da.sha256_c_order(missing_sorted),
        "tail_ids_sha256": da.sha256_c_order(tail_ids),
        "bundle_array_count": int(bundle["array_count"]),
        "bundle_element_count": int(bundle["element_count"]),
    }


def main(argv=None):
    args = parse_args(argv)
    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(dataset_dir):
        raise SystemExit("dataset directory not found: %s" % dataset_dir)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.batch_size % 4 != 0:
        raise SystemExit("--batch-size must be divisible by 4")

    npz_path = os.path.join(out_dir, "full_dataset_reference.npz")
    json_path = os.path.join(out_dir, "full_dataset_reference.json")
    if not args.force and (os.path.exists(npz_path) or os.path.exists(json_path)):
        raise SystemExit("output exists; pass --force to overwrite: %s" % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    arrays = fdc.load_epoch_arrays(dataset_dir, args.epoch)
    reference = fdc.build_reference(arrays, args.epoch, args.batch_size, args.worlds)
    np.savez(npz_path, **reference)

    metadata = build_metadata(arrays, reference, args)
    da.write_json(json_path, metadata)
    print("FULL_DATASET_REFERENCE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
