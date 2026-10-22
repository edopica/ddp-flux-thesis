#!/usr/bin/env python3
"""Independent numpy checker for the static-scaling full-dataset reference."""

from __future__ import annotations

import argparse
import os

import numpy as np

import dataset_adapter as da
import full_dataset_common as fdc

SCHEMA = "static-full-dataset-python-check-v1"
REFERENCE_SCHEMAS = (
    "static-full-dataset-reference-v1",
    "synthetic-full-dataset-reference-v1",
)
REQUIRED_NPZ_KEYS = (
    "epoch_ids",
    "rank_ids_R1",
    "rank_ids_R2",
    "rank_ids_R4",
    "update1_global_ids",
    "update1_labels",
    "update1_transformed_nchw",
    "stats_mean",
    "stats_std",
)
REQUIRED_JSON_KEYS = (
    "schema",
    "dataset_dir",
    "epoch",
    "batch_size",
    "n_eff",
    "tail_size",
    "updates_per_epoch",
    "sample_order_sha256",
    "update1_order_sha256",
    "transform_spec_sha256",
    "transform_sample_sha256",
    "bundle_hash",
    "bundle_file_sha256",
    "dataset_files",
    "stats_mean",
    "stats_std",
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


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
        description="Check the full-dataset static-scaling reference independently.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--worlds", type=parse_worlds, default=parse_worlds("1,2,4"))
    parser.add_argument("--strict-contract", nargs="?", const=True, default=True,
                        type=_parse_bool)
    parser.add_argument("--julia-json", default=None)
    return parser.parse_args(argv)


def _updates_per_epoch(reference, batch):
    value = reference.get("updates_per_epoch")
    if isinstance(value, dict):
        for key in (str(batch), str(int(batch))):
            if key in value:
                return int(value[key])
        raise ValueError("updates_per_epoch has no entry for batch %d" % batch)
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError("updates_per_epoch is missing or invalid")


def _rank_concatenation_matches(rows, ids, batch, world):
    chunk = batch // world
    if chunk * world != batch:
        return False
    updates = len(ids) // batch
    for update in range(updates):
        pieces = [rows[rank][update * chunk:(update + 1) * chunk] for rank in range(world)]
        if not np.array_equal(np.concatenate(pieces), ids[update * batch:(update + 1) * batch]):
            return False
    return True


def _check_rank_sequences(arrays, ids, batch, world, ref_arrays, record):
    key = "rank_ids_R%d" % world
    if key not in ref_arrays:
        record("rank_sequence.%s.present" % key, False, "missing from reference NPZ")
        return
    n_eff = int(len(ids))
    if world <= 0 or n_eff % world != 0 or batch % world != 0:
        record("rank_sequence.R%d.divisibility" % world, False,
               "n_eff=%d batch=%d world=%d" % (n_eff, batch, world))
        return
    ref_sequence = np.asarray(ref_arrays[key])
    if world == 1:
        record("rank_sequence.R1.shape", ref_sequence.shape == (n_eff,),
               "ref=%s actual=%s" % (ref_sequence.shape, (n_eff,)))
        record("rank_sequence.R1.dtype", ref_sequence.dtype == np.int64,
               "ref=%s" % (ref_sequence.dtype,))
        ref_rows = [ref_sequence]
    else:
        want = (world, n_eff // world)
        record("rank_sequence.R%d.shape" % world, ref_sequence.shape == want,
               "ref=%s actual=%s" % (ref_sequence.shape, want))
        record("rank_sequence.R%d.dtype" % world, ref_sequence.dtype == np.int64,
               "ref=%s" % (ref_sequence.dtype,))
        if ref_sequence.shape != want:
            return
        ref_rows = [ref_sequence[rank] for rank in range(world)]

    perm = np.asarray(arrays["epoch_permutation"])
    expected_rows = [np.asarray(da.rank_sequence(perm, batch, rank, world), dtype=np.int64)
                     for rank in range(world)]
    for rank in range(world):
        record("rank_sequence.R%d.rank%d" % (world, rank),
               np.array_equal(ref_rows[rank], expected_rows[rank]),
               "first_ref=%r first_expected=%r"
               % (int(ref_rows[rank][0]) if len(ref_rows[rank]) else None,
                  int(expected_rows[rank][0]) if len(expected_rows[rank]) else None))

    record("rank_sequence.R%d.concatenation" % world,
           _rank_concatenation_matches(ref_rows, ids, batch, world),
           "per-update rank order")
    flat = np.concatenate(ref_rows)
    record("rank_sequence.R%d.length" % world, len(flat) == n_eff,
           "length=%d expected=%d" % (len(flat), n_eff))
    record("rank_sequence.R%d.union" % world,
           set(int(value) for value in flat) == set(int(value) for value in ids))
    record("rank_sequence.R%d.unique" % world, len(np.unique(flat)) == len(flat))
    disjoint = True
    for left in range(world):
        for right in range(left + 1, world):
            if np.intersect1d(ref_rows[left], ref_rows[right]).size != 0:
                disjoint = False
    record("rank_sequence.R%d.disjoint" % world, disjoint)


def _check_update1(arrays, worlds, ref_arrays, batch, record):
    ref_gids = np.asarray(ref_arrays["update1_global_ids"])
    ref_labels = np.asarray(ref_arrays["update1_labels"])
    ref_x = np.asarray(ref_arrays["update1_transformed_nchw"])
    gids = fdc.global_batch_ids(arrays, batch, 1)
    x, labels, sids, local_gids = fdc.local_batch(arrays, batch, 1, 0, 1)
    ids = fdc.epoch_ids(arrays, batch)

    record("update1.global_ids.shape", ref_gids.shape == (batch,),
           "ref=%s actual=%s" % (ref_gids.shape, (batch,)))
    record("update1.global_ids.dtype", ref_gids.dtype == np.int64,
           "ref=%s" % (ref_gids.dtype,))
    record("update1.global_ids.values", np.array_equal(ref_gids, gids))
    record("update1.global_ids.first_batch", np.array_equal(ref_gids, ids[:batch]))
    record("update1.global_ids.local_agreement",
           np.array_equal(local_gids, gids) and np.array_equal(sids, gids))
    record("update1.labels.shape", ref_labels.shape == (batch,),
           "ref=%s actual=%s" % (ref_labels.shape, (batch,)))
    record("update1.labels.dtype", ref_labels.dtype == np.int64,
           "ref=%s" % (ref_labels.dtype,))
    record("update1.labels.values", np.array_equal(ref_labels, labels))
    record("update1.labels.range", bool(np.all((ref_labels >= 0) & (ref_labels < 200))))
    record("update1.transformed.shape", ref_x.shape == (batch, 3, 64, 64),
           "ref=%s actual=%s" % (ref_x.shape, (batch, 3, 64, 64)))
    record("update1.transformed.dtype", ref_x.dtype == np.float32,
           "ref=%s" % (ref_x.dtype,))
    if ref_x.shape == x.shape:
        diff = float(np.max(np.abs(ref_x.astype(np.float64) - x.astype(np.float64))))
    else:
        diff = float("inf")
    record("update1.transformed.max_abs_diff", diff <= 1e-6, "max_abs_diff=%r" % diff)

    for world in worlds:
        if batch % int(world) != 0:
            continue
        pieces = []
        sid_pieces = []
        for rank in range(int(world)):
            local_x, local_y, local_sids, _ = fdc.local_batch(arrays, batch, 1, rank, world)
            pieces.append(local_x)
            sid_pieces.append(local_sids)
            record("update1.R%d.rank%d.labels" % (world, rank),
                   np.all(local_y == labels[rank * (batch // world):(rank + 1) * (batch // world)]))
        joined = np.concatenate(pieces, axis=0)
        joined_ids = np.concatenate(sid_pieces)
        if joined.shape == ref_x.shape:
            world_diff = float(np.max(np.abs(joined.astype(np.float64)
                                             - ref_x.astype(np.float64))))
        else:
            world_diff = float("inf")
        record("update1.R%d.transformed.max_abs_diff" % world, world_diff <= 1e-6,
               "max_abs_diff=%r" % world_diff)
        record("update1.R%d.shard_ids" % world, np.array_equal(joined_ids, ref_gids))


def _check_onehot(labels, record):
    record("onehot.labels_in_range", bool(np.all((labels >= 0) & (labels < 200))))
    if not np.all((labels >= 0) & (labels < 200)):
        return
    eye = np.eye(200, dtype=np.float32)[labels]
    record("onehot.shape", eye.shape == (len(labels), 200), "actual=%s" % (eye.shape,))
    record("onehot.row_sums", bool(np.all(np.sum(eye, axis=1) == np.float32(1.0))))
    record("onehot.row_argmax", bool(np.all(np.argmax(eye, axis=1) == labels)))
    flux_matrix = eye.T
    record("onehot.flux.shape", flux_matrix.shape == (200, len(labels)),
           "actual=%s" % (flux_matrix.shape,))
    record("onehot.flux.column_sums",
           bool(np.all(np.sum(flux_matrix, axis=0) == np.float32(1.0))))
    record("onehot.flux.row_argmax",
           bool(np.all(np.argmax(flux_matrix, axis=0) == labels)))
    width = min(4, len(labels))
    explicit = np.zeros((200, width), dtype=np.float32)
    for column in range(width):
        explicit[int(labels[column]), column] = np.float32(1.0)
    record("onehot.flux.explicit", np.array_equal(flux_matrix[:, :width], explicit))


def _check_dataset_files(dataset_dir, reference, record):
    files = reference.get("dataset_files")
    if not isinstance(files, dict) or not files:
        record("dataset_files.present", False)
        return {}
    try:
        declared = fdc.dataset_file_hashes(dataset_dir, verify_large=False)
        record("dataset_files.manifest", True, "files=%d" % len(declared))
    except Exception as exc:
        declared = {}
        record("dataset_files.manifest", False, "%s: %s" % (type(exc).__name__, exc))

    actual = {}
    for name in sorted(files):
        entry = files[name]
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            record("dataset_file.%s.present" % name, False, "missing %s" % path)
            continue
        size = int(os.path.getsize(path))
        digest = da.sha256_file(path)
        actual[name] = {"bytes": size, "sha256": digest}
        expected_bytes = int(entry["bytes"]) if "bytes" in entry else None
        expected_sha = str(entry["sha256"]) if "sha256" in entry else None
        record("dataset_file.%s.bytes" % name, expected_bytes == size,
               "actual=%d ref=%r" % (size, expected_bytes))
        record("dataset_file.%s.sha256" % name, expected_sha == digest,
               "actual=%s ref=%r" % (digest, expected_sha))
        if name in declared:
            record("dataset_file.%s.manifest_sha256" % name,
                   declared[name]["sha256"] == expected_sha,
                   "manifest=%s ref=%r" % (declared[name]["sha256"], expected_sha))
    for name in fdc.REQUIRED_FILES:
        record("dataset_file.%s.required" % name,
               name in files and os.path.isfile(os.path.join(dataset_dir, name)))
    return actual


def _check_julia(path, summary, record):
    if not path:
        return None
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        record("julia.present", False, "missing %s" % path)
        return {"path": path}
    document = da.read_json(path)
    marker = document.get("pass")
    passed = (_parse_bool(document.get("pass_checks")) if marker is None
              else _parse_bool(marker))
    record("julia.pass", passed, "pass=%r checks_failed=%r"
           % (marker, document.get("checks_failed")))
    record("julia.checks_failed", int(document.get("checks_failed", -1)) == 0,
           "checks_failed=%r" % (document.get("checks_failed"),))
    pairs = (
        ("julia.sample_order_sha256", document.get("sample_order_sha256"),
         summary["sample_order_sha256"]),
        ("julia.update1_order_sha256", document.get("update1_order_sha256"),
         summary["update1_order_sha256"]),
        ("julia.transform_spec_sha256", document.get("transform_spec_sha256"),
         summary["transform_spec_sha256"]),
        ("julia.update1_transformed_sha256", document.get("update1_transformed_sha256"),
         summary["transform_sample_sha256"]),
    )
    for name, actual, expected in pairs:
        record(name, actual == expected, "julia=%r python=%r" % (actual, expected))
    bundle = document.get("bundle") or {}
    record("julia.bundle_hash", bundle.get("bundle_hash") == summary["bundle_hash"],
           "julia=%r python=%r" % (bundle.get("bundle_hash"), summary["bundle_hash"]))
    record("julia.bundle_file_sha256",
           bundle.get("bundle_file_sha256") == summary["bundle_file_sha256"],
           "julia=%r python=%r"
           % (bundle.get("bundle_file_sha256"), summary["bundle_file_sha256"]))
    return {"path": path, "pass": bool(passed)}


def run_checks(args, record, summary):
    dataset_dir = os.path.abspath(args.dataset_dir)
    reference_dir = os.path.abspath(args.reference_dir)
    npz_path = os.path.join(reference_dir, "full_dataset_reference.npz")
    json_path = os.path.join(reference_dir, "full_dataset_reference.json")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError("reference NPZ not found: %s" % npz_path)
    if not os.path.isfile(json_path):
        raise FileNotFoundError("reference JSON not found: %s" % json_path)

    with np.load(npz_path, allow_pickle=False) as archive:
        ref_arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    reference = da.read_json(json_path)
    arrays = fdc.load_epoch_arrays(dataset_dir, args.epoch)

    for key in REQUIRED_NPZ_KEYS:
        record("reference.npz.%s" % key, key in ref_arrays,
               "missing" if key not in ref_arrays else "present")
    for key in REQUIRED_JSON_KEYS:
        record("reference.json.%s" % key, key in reference,
               "missing" if key not in reference else "present")
    record("reference.schema", reference.get("schema") in REFERENCE_SCHEMAS,
           "schema=%r" % (reference.get("schema"),))
    record("reference.dataset_dir",
           os.path.abspath(str(reference.get("dataset_dir", ""))) == dataset_dir,
           "ref=%r actual=%s" % (reference.get("dataset_dir"), dataset_dir))

    perm = np.asarray(arrays["epoch_permutation"])
    n = int(len(perm))
    ids = fdc.epoch_ids(arrays, args.batch_size)
    n_eff = int(len(ids))
    tail = int(n - n_eff)
    updates = int(n_eff // args.batch_size)
    gids = fdc.global_batch_ids(arrays, args.batch_size, 1)
    mean32 = np.asarray(arrays["mean32"], dtype=np.float32)
    std32 = np.asarray(arrays["std32"], dtype=np.float32)

    record("reference.epoch", int(reference.get("epoch", -1)) == args.epoch,
           "ref=%r actual=%d" % (reference.get("epoch"), args.epoch))
    record("reference.batch_size", int(reference.get("batch_size", -1)) == args.batch_size,
           "ref=%r actual=%d" % (reference.get("batch_size"), args.batch_size))
    record("reference.n_eff", int(reference.get("n_eff", -1)) == n_eff,
           "ref=%r actual=%d" % (reference.get("n_eff"), n_eff))
    record("reference.tail_size", int(reference.get("tail_size", -1)) == tail,
           "ref=%r actual=%d" % (reference.get("tail_size"), tail))
    try:
        ref_updates = _updates_per_epoch(reference, args.batch_size)
    except Exception as exc:
        ref_updates = -1
        record("reference.updates_per_epoch", False, "%s: %s" % (type(exc).__name__, exc))
    else:
        record("reference.updates_per_epoch", ref_updates == updates,
               "ref=%d actual=%d" % (ref_updates, updates))

    ref_ids = np.asarray(ref_arrays["epoch_ids"])
    record("epoch_ids.shape", ref_ids.shape == (n_eff,),
           "ref=%s actual=%s" % (ref_ids.shape, (n_eff,)))
    record("epoch_ids.dtype", ref_ids.dtype == np.int64, "ref=%s" % (ref_ids.dtype,))
    record("epoch_ids.values", np.array_equal(ref_ids, ids))
    record("epoch_ids.unique", len(np.unique(ref_ids)) == len(ref_ids),
           "unique=%d total=%d" % (len(np.unique(ref_ids)), len(ref_ids)))
    record("epoch_ids.range", bool(np.all((ref_ids >= 0) & (ref_ids < n))))
    missing = set(range(n)) - set(int(value) for value in ref_ids)
    expected_tail = set(int(value) for value in perm[n_eff:])
    record("epoch_ids.missing_equals_final_tail", missing == expected_tail,
           "missing=%d tail=%d" % (len(missing), len(expected_tail)))

    if args.strict_contract:
        if n == 100000:
            record("contract.n_eff", n_eff == 99840, "n_eff=%d" % n_eff)
            record("contract.tail_size", tail == 160, "tail=%d" % tail)
            record("contract.unique", len(np.unique(ids)) == len(ids))
            record("contract.range", bool(np.all((ids >= 0) & (ids < 100000))))
            for batch in (512, 256):
                batch_ids = fdc.epoch_ids(arrays, batch)
                want = 195 if batch == 512 else 390
                record("contract.updates.b%d" % batch, len(batch_ids) // batch == want,
                       "updates=%d expected=%d" % (len(batch_ids) // batch, want))
                record("contract.length.b%d" % batch, len(batch_ids) == 99840,
                       "length=%d" % len(batch_ids))
        else:
            record("contract.n_eff_multiple", n_eff % args.batch_size == 0,
                   "n_eff=%d batch=%d" % (n_eff, args.batch_size))
            record("contract.unique", len(np.unique(ids)) == len(ids))
            record("contract.range", bool(np.all((ids >= 0) & (ids < n))))
            record("contract.tail_size", tail == n - n_eff)
    else:
        record("contract.n_eff_multiple", n_eff % args.batch_size == 0,
               "n_eff=%d batch=%d" % (n_eff, args.batch_size))
        record("contract.unique", len(np.unique(ids)) == len(ids))
        record("contract.range", bool(np.all((ids >= 0) & (ids < n))))
        record("contract.tail_size", tail == n - n_eff)

    sample_sha = da.sha256_c_order(ref_ids)
    update1_sha = da.sha256_c_order(gids)
    spec_sha = fdc.transform_spec_sha256(mean32, std32)
    ref_x = np.asarray(ref_arrays["update1_transformed_nchw"])
    transform_sha = da.sha256_c_order(ref_x)
    record("sample_order_sha256",
           sample_sha == reference.get("sample_order_sha256") == da.sha256_c_order(ids),
           "npz=%s json=%r expected=%s"
           % (sample_sha, reference.get("sample_order_sha256"), da.sha256_c_order(ids)))
    record("update1_order_sha256",
           update1_sha == reference.get("update1_order_sha256"),
           "actual=%s ref=%r" % (update1_sha, reference.get("update1_order_sha256")))
    record("transform_spec_sha256",
           spec_sha == reference.get("transform_spec_sha256"),
           "actual=%s ref=%r" % (spec_sha, reference.get("transform_spec_sha256")))
    record("transform_sample_sha256",
           transform_sha == reference.get("transform_sample_sha256"),
           "actual=%s ref=%r" % (transform_sha, reference.get("transform_sample_sha256")))

    ref_mean = np.asarray(ref_arrays["stats_mean"])
    ref_std = np.asarray(ref_arrays["stats_std"])
    json_mean = np.asarray(reference.get("stats_mean"), dtype=np.float32)
    json_std = np.asarray(reference.get("stats_std"), dtype=np.float32)
    record("stats_mean.shape", ref_mean.shape == (3,), "ref=%s" % (ref_mean.shape,))
    record("stats_std.shape", ref_std.shape == (3,), "ref=%s" % (ref_std.shape,))
    record("stats_mean.dtype", ref_mean.dtype == np.float32, "ref=%s" % (ref_mean.dtype,))
    record("stats_std.dtype", ref_std.dtype == np.float32, "ref=%s" % (ref_std.dtype,))
    record("stats_mean.npz_vs_dataset", np.array_equal(ref_mean, mean32),
           "npz=%r dataset=%r" % (ref_mean, mean32))
    record("stats_std.npz_vs_dataset", np.array_equal(ref_std, std32),
           "npz=%r dataset=%r" % (ref_std, std32))
    record("stats_mean.json_vs_dataset", np.array_equal(json_mean, mean32),
           "json=%r dataset=%r" % (json_mean, mean32))
    record("stats_std.json_vs_dataset", np.array_equal(json_std, std32),
           "json=%r dataset=%r" % (json_std, std32))
    record("stats_mean.json_vs_npz", np.array_equal(json_mean, ref_mean))
    record("stats_std.json_vs_npz", np.array_equal(json_std, ref_std))

    for world in args.worlds:
        _check_rank_sequences(arrays, ids, args.batch_size, world, ref_arrays, record)

    _check_update1(arrays, args.worlds, ref_arrays, args.batch_size, record)
    _check_onehot(np.asarray(ref_arrays["update1_labels"]), record)

    try:
        bundle = fdc.bundle_info(dataset_dir)
        record("bundle.bundle_hash", bundle["bundle_hash"] == reference.get("bundle_hash"),
               "actual=%s ref=%r" % (bundle["bundle_hash"], reference.get("bundle_hash")))
        record("bundle.file_sha256",
               bundle["bundle_file_sha256"] == reference.get("bundle_file_sha256"),
               "actual=%s ref=%r"
               % (bundle["bundle_file_sha256"], reference.get("bundle_file_sha256")))
        record("bundle.array_count", bundle["array_count"] == 62,
               "actual=%d" % bundle["array_count"])
        record("bundle.element_count", bundle["element_count"] == 11271432,
               "actual=%d" % bundle["element_count"])
    except Exception as exc:
        bundle = {}
        record("bundle.bundle_hash", False, "%s: %s" % (type(exc).__name__, exc))

    actual_files = _check_dataset_files(dataset_dir, reference, record)

    summary.update({
        "n_samples": n,
        "n_eff": n_eff,
        "tail_size": tail,
        "updates_per_epoch": updates,
        "sample_order_sha256": sample_sha,
        "update1_order_sha256": update1_sha,
        "transform_spec_sha256": spec_sha,
        "transform_sample_sha256": transform_sha,
        "bundle_hash": bundle.get("bundle_hash"),
        "bundle_file_sha256": bundle.get("bundle_file_sha256"),
        "bundle": bundle,
        "dataset_files": actual_files,
        "stats_mean": [float(value) for value in mean32],
        "stats_std": [float(value) for value in std32],
    })
    summary["julia"] = _check_julia(args.julia_json, summary, record)


def main(argv=None):
    args = parse_args(argv)
    results = []

    def record(name, ok, detail=""):
        results.append({"name": str(name), "pass": bool(ok), "detail": str(detail)})
        return bool(ok)

    summary = {}
    try:
        run_checks(args, record, summary)
    except Exception as exc:
        record("checker.exception", False, "%s: %s" % (type(exc).__name__, exc))

    failed = sum(1 for result in results if not result["pass"])
    payload = {
        "schema": SCHEMA,
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "reference_dir": os.path.abspath(args.reference_dir),
        "epoch": int(args.epoch),
        "batch_size": int(args.batch_size),
        "worlds": [int(world) for world in args.worlds],
        "strict_contract": bool(args.strict_contract),
        "julia_json": os.path.abspath(args.julia_json) if args.julia_json else None,
        "n_samples": summary.get("n_samples"),
        "n_eff": summary.get("n_eff"),
        "tail_size": summary.get("tail_size"),
        "updates_per_epoch": summary.get("updates_per_epoch"),
        "checks_total": len(results),
        "checks_failed": failed,
        "pass": failed == 0,
        "checks": results,
        "hashes": {
            "sample_order_sha256": summary.get("sample_order_sha256"),
            "update1_order_sha256": summary.get("update1_order_sha256"),
            "transform_spec_sha256": summary.get("transform_spec_sha256"),
            "transform_sample_sha256": summary.get("transform_sample_sha256"),
            "bundle_hash": summary.get("bundle_hash"),
            "bundle_file_sha256": summary.get("bundle_file_sha256"),
        },
        "bundle": summary.get("bundle"),
        "dataset_files": summary.get("dataset_files"),
        "stats_mean": summary.get("stats_mean"),
        "stats_std": summary.get("stats_std"),
        "julia": summary.get("julia"),
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    da.write_json(out_path, payload)

    if failed:
        print("FULL_DATASET_PYTHON_CHECK FAIL checks=%d" % len(results))
        return 1
    print("FULL_DATASET_PYTHON_CHECK PASS checks=%d" % len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
