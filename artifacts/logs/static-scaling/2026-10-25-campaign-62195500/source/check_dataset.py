"""
Phase 2 Python acceptance driver for the static-scaling prepared dataset and
shared fixtures (numpy + optional PIL-free raw-dataset checks; runs on Leonardo).

Checks (each prints `CHECK <name> PASS|FAIL <detail>`; SKIP lines never fail):
  0. prepared dataset and schedule manifests: presence, required file sets
     (dataset: train_images.npy, train_labels.npy, val_images.npy,
     val_labels.npy, class_wnids.txt, stats.json; extra entries only for
     permutations.npy/augmentation.npy. schedule: exactly permutations.npy and
     augmentation.npy), and sha256/byte-size recomputation of every listed
     entry. A failure gates every prep-dependent semantic section;
  1. data counts, label histograms (500/class train, 50/class val), class
     coverage, and raw annotation mapping when --dataset-root is available;
  2. stats recomputation vs stats.json (max abs delta <= 2e-7) and
     pixel_count_per_channel == 100000*4096;
  3. permutations.npy: every row is a permutation of 0..99999;
  4. augmentation ranges (top/left 0..8, flip 0..1) and bitwise determinism of
     epochs 0/1 regenerated from the recorded scheme, including epoch
     independence after generating a later epoch, plus bitwise regeneration of
     all 50 stored epochs;
  5. schedule reconstruction at batch 512 and 256 for R = 1, 2, 4: shard sizes
     and order equal perm[0][:99840], 195/390 updates, the same 160-image tail,
     and the rank-independence property of the transform (transform of the
     global batch sliced to rank r equals transform of the rank shard);
  6. initial bundle: 62 arrays, exact names/shapes/dtypes, 11,271,432 float32
     elements, GN ones/zeros (all 20 weight and 20 bias arrays), per-array
     sha256 vs manifest, bundle_hash recomputed with the recorded rule;
  7. fixture integrity: every array in data/schedule/linear/kernel fixtures
     matches fixture_manifest.json (dtype, shape, sha256) and the npz byte
     hashes match;
  8. data_fixture recomputed from $PREP via dataset_adapter, bitwise;
  9. schedule_fixture recomputed from $PREP via dataset_adapter, bitwise.

Output ends with `PHASE2_DATASET_CHECK PASS` or
`PHASE2_DATASET_CHECK FAIL <names>`; exit 0/1.

Reproduction (Leonardo, inside or outside an allocation):
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/check_dataset.py \\
    --prep-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/initial_bundle.npz \\
    --bundle-manifest /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/initial_bundle.manifest.json

Local synthetic integration run (no torch/PIL):
  python3 scripts/static_scaling/make_fixtures.py --synthetic --synthetic-prep --out-dir /tmp/opencode/phase2-full
  python3 scripts/static_scaling/check_dataset.py --prep-dir /tmp/opencode/phase2-full --fixture-dir /tmp/opencode/phase2-full
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import dataset_adapter as da

DATASET_ROOT_DEFAULT = "/leonardo_work/IscrC_UNMASKED/epicazio/datasets/tiny-imagenet-200"
STATS_ABS_TOL = 2.0e-7
BUNDLE_HASH_DEFINITION = "sha256 over concat of per-array raw C-order bytes in manifest order"
DATA_FIXTURE_CONTRACT = {
    "decoded_train_images": ("uint8", (12, 3, 64, 64)),
    "train_labels": ("int64", (12,)),
    "augmentation": ("uint8", (2, 12, 3)),
    "train_transformed_ref": ("float32", (2, 12, 3, 64, 64)),
    "decoded_val_images": ("uint8", (8, 3, 64, 64)),
    "val_labels": ("int64", (8,)),
    "val_transformed_ref": ("float32", (8, 3, 64, 64)),
    "normalization_mean": ("float32", (3,)),
    "normalization_std": ("float32", (3,)),
}
SCHEDULE_FIXTURE_CONTRACT = {
    "epoch0_permutation": ("int64", (100000,)),
    "augmentation_epochs01": ("uint8", (2, 100000, 3)),
    "rank_ids_R1": ("int64", (99840,)),
    "rank_ids_R2": ("int64", (2, 49920)),
    "rank_ids_R4": ("int64", (4, 24960)),
}


class CheckLog:
    def __init__(self):
        self.failed = []

    def check(self, name: str, ok: bool, detail: str = ""):
        print("CHECK %s %s%s" % (name, "PASS" if ok else "FAIL", (" " + detail) if detail else ""),
              flush=True)
        if not ok:
            self.failed.append(name)
        return ok

    def skip(self, name: str, detail: str):
        print("CHECK %s SKIP %s" % (name, detail), flush=True)


def _fmt_float(x) -> str:
    return "%.9g" % float(x)


def _files(directory):
    return sorted(n for n in os.listdir(directory) if n.lower().endswith(".jpeg"))


def _shortlist(items, limit=5):
    items = list(items)
    return [str(x) for x in items[:limit]] if items else "none"


# ---------------------------------------------------------------------------
# 0. prepared dataset and schedule manifests
# ---------------------------------------------------------------------------

PREP_DATASET_REQUIRED = ("train_images.npy", "train_labels.npy", "val_images.npy",
                         "val_labels.npy", "class_wnids.txt", "stats.json")
PREP_DATASET_ALLOWED_EXTRA = ("permutations.npy", "augmentation.npy")
PREP_SCHEDULE_REQUIRED = ("permutations.npy", "augmentation.npy")


def _check_prep_manifest(log: CheckLog, present_name: str, file_set_name: str, sha_name: str,
                         manifest_path: str, prep_dir: str, required, allowed_extra=()) -> bool:
    required = tuple(required)
    if not os.path.exists(manifest_path):
        log.check(present_name, False, "not found: %s" % manifest_path)
        log.skip(file_set_name, "skipped: manifest missing")
        log.skip(sha_name, "skipped: manifest missing")
        return False
    try:
        manifest = da.read_json(manifest_path)
    except (OSError, ValueError) as err:
        log.check(present_name, False, "unreadable: %s" % err)
        log.skip(file_set_name, "skipped: manifest unreadable")
        log.skip(sha_name, "skipped: manifest unreadable")
        return False
    log.check(present_name, True, str(manifest_path))

    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        log.check(file_set_name, False, "'files' is not a list")
        log.skip(sha_name, "skipped: no file list")
        return False
    listed = [e.get("path") for e in files if isinstance(e, dict)]
    missing = sorted(set(required) - set(listed))
    extra = sorted(set(listed) - set(required) - set(allowed_extra))
    file_set_ok = not missing and not extra and len(listed) == len(files)
    log.check(file_set_name, bool(file_set_ok),
              "required=%d listed=%d missing=%s extra=%s"
              % (len(required), len(files), _shortlist(missing), _shortlist(extra)))

    mismatches = []
    for e in files:
        path = e.get("path") if isinstance(e, dict) else None
        if not isinstance(path, str):
            mismatches.append("<malformed-entry>")
            continue
        full = os.path.join(prep_dir, path)
        if not os.path.exists(full):
            mismatches.append(path + ":missing")
            continue
        try:
            digest = da.sha256_file(full)
        except OSError:
            mismatches.append(path + ":unreadable")
            continue
        recorded = e.get("sha256")
        if not isinstance(recorded, str) or digest != recorded.lower():
            mismatches.append(path + ":sha256")
            continue
        if e.get("bytes") is not None:
            try:
                want_bytes = int(e["bytes"])
            except (TypeError, ValueError):
                mismatches.append(path + ":bytes-meta")
                continue
            if os.path.getsize(full) != want_bytes:
                mismatches.append(path + ":bytes")
    log.check(sha_name, bool(not mismatches),
              "checked=%d mismatches=%s" % (len(files), _shortlist(mismatches)))
    return bool(file_set_ok and not mismatches)


def check_prep_manifests(log: CheckLog, prep_dir: str) -> bool:
    """Enforce dataset_manifest.json and schedule_manifest.json before semantics.

    Returns True iff all six prep checks PASS (a missing manifest fails its
    present check and SKIPs file_set + files_sha256_bytes).
    """
    dataset_ok = _check_prep_manifest(
        log, "prep.dataset_manifest_present", "prep.dataset_file_set",
        "prep.dataset_files_sha256_bytes",
        os.path.join(prep_dir, "dataset_manifest.json"),
        prep_dir, PREP_DATASET_REQUIRED, PREP_DATASET_ALLOWED_EXTRA)
    schedule_ok = _check_prep_manifest(
        log, "prep.schedule_manifest_present", "prep.schedule_file_set",
        "prep.schedule_files_sha256_bytes",
        os.path.join(prep_dir, "schedule_manifest.json"),
        prep_dir, PREP_SCHEDULE_REQUIRED)
    return bool(dataset_ok and schedule_ok)


# ---------------------------------------------------------------------------
# 1. counts, histograms, coverage, annotation mapping
# ---------------------------------------------------------------------------

def check_counts(log: CheckLog, prep_dir: str, dataset_root: str):
    try:
        train_images = da.open_prepared_train_images(prep_dir)
        val_images = da.open_prepared_val_images(prep_dir)
        train_labels = np.load(os.path.join(prep_dir, "train_labels.npy"), mmap_mode="r")
        val_labels = np.load(os.path.join(prep_dir, "val_labels.npy"), mmap_mode="r")
    except FileNotFoundError as err:
        log.check("data.files_present", False, str(err))
        return
    ok = (train_images.shape == (da.N_TRAIN, 3, 64, 64) and train_images.dtype == np.uint8
          and val_images.shape == (da.N_VAL, 3, 64, 64) and val_images.dtype == np.uint8
          and train_labels.shape == (da.N_TRAIN,) and train_labels.dtype == np.int64
          and val_labels.shape == (da.N_VAL,) and val_labels.dtype == np.int64)
    log.check("data.counts",
              ok,
              "train=%s/%s val=%s/%s labels=%s/%s"
              % (train_labels.shape, train_images.shape, val_labels.shape, val_images.shape,
                 train_labels.dtype, val_labels.dtype))

    tl = np.asarray(train_labels)
    vl = np.asarray(val_labels)
    in_range = bool(np.all((tl >= 0) & (tl < da.N_CLASSES)) and np.all((vl >= 0) & (vl < da.N_CLASSES)))
    log.check("data.label_range_0_199", in_range,
              "train_min=%d train_max=%d val_min=%d val_max=%d"
              % (tl.min(), tl.max(), vl.min(), vl.max()))
    if in_range:
        th = np.bincount(tl, minlength=da.N_CLASSES)
        vh = np.bincount(vl, minlength=da.N_CLASSES)
        log.check("data.train_histogram_500_per_class", bool(np.all(th == 500)),
                  "min=%d max=%d classes=%d" % (th.min(), th.max(), len(th)))
        log.check("data.val_histogram_50_per_class", bool(np.all(vh == 50)),
                  "min=%d max=%d classes=%d" % (vh.min(), vh.max(), len(vh)))
        cov = np.unique(np.concatenate([tl, vl]))
        log.check("data.class_coverage_200", bool(len(cov) == da.N_CLASSES and cov.min() == 0 and cov.max() == 199),
                  "unique=%d" % len(cov))

    if not dataset_root or not os.path.isdir(dataset_root):
        log.skip("data.annotation_mapping", "--dataset-root not available: %s" % dataset_root)
        return
    raw_wnids = os.path.join(dataset_root, "wnids.txt")
    prep_wnids = os.path.join(prep_dir, "class_wnids.txt")
    same = os.path.exists(prep_wnids) and open(raw_wnids, "rb").read() == open(prep_wnids, "rb").read()
    log.check("data.class_wnids_copy", bool(same), "byte-identical wnids.txt" if same else "differs")

    wnid_list = [line.strip() for line in open(raw_wnids, encoding="utf-8") if line.strip()]
    counts_ok = True
    blocks_ok = True
    pos = 0
    for c, wnid in enumerate(wnid_list):
        n = len(_files(os.path.join(dataset_root, "train", wnid, "images")))
        if n != 500:
            counts_ok = False
        if not np.all(np.asarray(train_labels[pos:pos + n]) == c):
            blocks_ok = False
        pos += n
    log.check("data.train_listing_500_per_class", bool(counts_ok and pos == da.N_TRAIN),
              "sum=%d" % pos)
    log.check("data.train_label_blocks_by_wnid_order", bool(blocks_ok),
              "labels follow wnids.txt order with sorted files")

    annot_path = os.path.join(dataset_root, "val", "val_annotations.txt")
    class_index = {w: i for i, w in enumerate(wnid_list)}
    annot = {}
    for line in open(annot_path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1].strip() in class_index:
            annot[parts[0].strip()] = class_index[parts[1].strip()]
    val_files = _files(os.path.join(dataset_root, "val", "images"))
    expected = np.asarray([annot[f] for f in val_files], dtype=np.int64)
    ok = (len(val_files) == da.N_VAL and np.array_equal(expected, np.asarray(val_labels)))
    bad = 0 if ok else int(np.sum(expected != np.asarray(val_labels)))
    log.check("data.annotation_mapping", bool(ok),
              "val_files=%d annotations=%d mismatches=%d" % (len(val_files), len(annot), bad))


# ---------------------------------------------------------------------------
# 2. stats recomputation
# ---------------------------------------------------------------------------

def recompute_stats_counts(train_images) -> tuple:
    """Exact per-value histograms; float64 mean/std from integer counts."""
    counts = np.zeros((3, 256), dtype=np.int64)
    n = train_images.shape[0]
    for start in range(0, n, 2000):
        chunk = np.asarray(train_images[start:min(n, start + 2000)])
        for c in range(3):
            counts[c] += np.bincount(chunk[:, c].ravel(), minlength=256)
    values = np.arange(256, dtype=np.float64) / 255.0
    pixel_count = int(counts[0].sum())
    s = (counts * values).sum(axis=1)
    sq = (counts * (values * values)).sum(axis=1)
    mean64 = s / float(pixel_count)
    var64 = sq / float(pixel_count) - mean64 * mean64
    std64 = np.sqrt(np.maximum(var64, 0.0))
    return mean64, std64, pixel_count


def check_stats(log: CheckLog, prep_dir: str):
    try:
        mean32, std32, stats = da.load_stats(prep_dir)
        train_images = da.open_prepared_train_images(prep_dir)
    except FileNotFoundError as err:
        log.check("stats.recompute", False, str(err))
        return None, None, None
    mean64, std64, pixel_count = recompute_stats_counts(train_images)
    jmean = np.asarray(stats["mean"], dtype=np.float64)
    jstd = np.asarray(stats["std"], dtype=np.float64)
    dmean = float(np.max(np.abs(mean64 - jmean)))
    dstd = float(np.max(np.abs(std64 - jstd)))
    log.check("stats.recompute_mean_std_vs_json", bool(dmean <= STATS_ABS_TOL and dstd <= STATS_ABS_TOL),
              "max_abs_delta_mean=%s max_abs_delta_std=%s tol=%s"
              % (_fmt_float(dmean), _fmt_float(dstd), _fmt_float(STATS_ABS_TOL)))
    expected_pixels = da.N_TRAIN * 64 * 64
    declared = int(stats.get("pixel_count_per_channel", -1))
    log.check("stats.pixel_count_per_channel",
              bool(declared == expected_pixels and pixel_count == expected_pixels),
              "json=%d recomputed=%d expected=%d" % (declared, pixel_count, expected_pixels))
    return mean32, std32, stats


# ---------------------------------------------------------------------------
# 3-4. permutations and augmentation
# ---------------------------------------------------------------------------

def check_permutation_augmentation(log: CheckLog, prep_dir: str):
    try:
        perms = da.open_prepared_permutations(prep_dir)
        augs = da.open_prepared_augmentation(prep_dir)
    except FileNotFoundError as err:
        log.check("schedule.permutations_valid", False, str(err))
        return None
    ok_shape = perms.shape == (da.EPOCHS, da.N_TRAIN) and perms.dtype == np.int64
    log.check("schedule.permutations_shape", bool(ok_shape),
              "shape=%s dtype=%s" % (perms.shape, perms.dtype))
    target = np.arange(da.N_TRAIN, dtype=np.int64)
    bad_rows = [e for e in range(perms.shape[0]) if not np.array_equal(np.sort(perms[e]), target)]
    log.check("schedule.permutations_valid", bool(ok_shape and not bad_rows),
              "checked_rows=%d bad_rows=%s" % (perms.shape[0], bad_rows[:5] if bad_rows else "none"))

    ok_shape = augs.shape == (da.EPOCHS, da.N_TRAIN, 3) and augs.dtype == np.uint8
    top = np.asarray(augs[:, :, 0])
    left = np.asarray(augs[:, :, 1])
    flip = np.asarray(augs[:, :, 2])
    ok_range = bool(ok_shape and top.max() <= 8 and left.max() <= 8 and flip.max() <= 1)
    log.check("schedule.augmentation_ranges", bool(ok_range),
              "shape=%s dtype=%s top_max=%d left_max=%d flip_max=%d"
              % (augs.shape, augs.dtype, top.max(), left.max(), flip.max()))

    e0_perm_ok = np.array_equal(da.epoch_permutation(0), np.asarray(perms[0]))
    e1_perm_ok = np.array_equal(da.epoch_permutation(1), np.asarray(perms[1]))
    e0_aug_ok = np.array_equal(da.epoch_augmentation(0), np.asarray(augs[0]))
    e1_aug_ok = np.array_equal(da.epoch_augmentation(1), np.asarray(augs[1]))
    log.check("schedule.regenerated_epochs_0_1_bitwise",
              bool(e0_perm_ok and e1_perm_ok and e0_aug_ok and e1_aug_ok),
              "perm0=%s perm1=%s aug0=%s aug1=%s" % (e0_perm_ok, e1_perm_ok, e0_aug_ok, e1_aug_ok))

    # Epoch independence: burn a later epoch on a fresh generator first.
    last = perms.shape[0] - 1
    last_perm = da.epoch_permutation(last)
    last_aug = da.epoch_augmentation(last)
    e0_ok = np.array_equal(da.epoch_permutation(0), np.asarray(perms[0])) and \
        np.array_equal(da.epoch_augmentation(0), np.asarray(augs[0]))
    last_ok = np.array_equal(last_perm, np.asarray(perms[last])) and \
        np.array_equal(last_aug, np.asarray(augs[last]))
    log.check("schedule.epoch_independence", bool(e0_ok and last_ok),
              "epoch 0 unchanged after regenerating epoch %d last_epoch_matches_stored=%s"
              % (last, last_ok))

    # Full semantic regeneration: every stored epoch must reproduce bitwise.
    t0 = time.perf_counter()
    n_stored = min(perms.shape[0], augs.shape[0])
    bad_epochs = list(range(n_stored, da.EPOCHS))
    for e in range(n_stored):
        if not np.array_equal(da.epoch_permutation(e), np.asarray(perms[e])) or \
                not np.array_equal(da.epoch_augmentation(e), np.asarray(augs[e])):
            bad_epochs.append(e)
    elapsed = time.perf_counter() - t0
    log.check("schedule.regenerated_all_epochs_bitwise", bool(not bad_epochs),
              "bad_epochs=%s %d epochs in %.2fs" % (_shortlist(bad_epochs), da.EPOCHS, elapsed))
    return np.asarray(perms[0])


# ---------------------------------------------------------------------------
# 5. schedule reconstruction, tail, rank independence
# ---------------------------------------------------------------------------

def check_schedule_reconstruction(log: CheckLog, prep_dir: str, perm0: np.ndarray,
                                  mean32, std32):
    n_eff = da.SAMPLES_PER_EPOCH
    for batch in (da.EPOCH_BATCH, da.FALLBACK_BATCH):
        ids = da.epoch_sample_ids(perm0, batch)
        ok = len(ids) == n_eff and np.array_equal(ids, perm0[:n_eff])
        updates = len(ids) // batch
        log.check("schedule.epoch_ids_b%d" % batch, bool(ok),
                  "samples=%d updates=%d" % (len(ids), updates))
        log.check("schedule.updates_b%d" % batch, bool(updates == (195 if batch == 512 else 390)),
                  "updates=%d" % updates)

    tail = np.asarray(perm0[n_eff:])
    ids256 = da.epoch_sample_ids(perm0, da.FALLBACK_BATCH)
    same_prefix = bool(np.array_equal(ids256, perm0[:n_eff]))
    log.check("schedule.tail_160_same_set",
              bool(len(tail) == da.TAIL_SIZE and same_prefix and len(set(tail.tolist())) == da.TAIL_SIZE),
              "tail_size=%d same_set=%s" % (len(tail), same_prefix))

    for batch in (da.EPOCH_BATCH, da.FALLBACK_BATCH):
        ids = da.epoch_sample_ids(perm0, batch)
        for world in (1, 2, 4):
            shards = [da.rank_sequence(perm0, batch, r, world) for r in range(world)]
            per = batch // world
            recon = np.empty(n_eff, dtype=np.int64)
            k = 0
            for start in range(0, n_eff, batch):
                for r in range(world):
                    recon[start + r * per:start + (r + 1) * per] = shards[r][k:k + per]
                k += per
            sizes_ok = all(len(s) == n_eff // world for s in shards)
            ok = sizes_ok and np.array_equal(recon, ids)
            log.check("schedule.shards_b%d_R%d" % (batch, world), bool(ok),
                      "sizes=%s order_matches_epoch_ids=%s" % ([len(s) for s in shards], np.array_equal(recon, ids)))

    try:
        train_images = da.open_prepared_train_images(prep_dir)
        augmentation = da.open_prepared_augmentation(prep_dir)
    except FileNotFoundError as err:
        log.check("schedule.rank_independence_transform_R2", False, str(err))
        log.check("schedule.rank_independence_transform_R4", False, str(err))
        return
    ids = da.epoch_sample_ids(perm0, da.EPOCH_BATCH)
    batch_ids = ids[:da.EPOCH_BATCH]
    images = np.ascontiguousarray(train_images[batch_ids])
    triples = np.ascontiguousarray(np.asarray(augmentation[0])[batch_ids])
    full = da.replay_transform(images, triples, mean32, std32)
    for world in (2, 4):
        per = da.EPOCH_BATCH // world
        ok = True
        for r in range(world):
            shard = da.replay_transform(images[r * per:(r + 1) * per],
                                        triples[r * per:(r + 1) * per], mean32, std32)
            if not np.array_equal(full[r * per:(r + 1) * per], shard):
                ok = False
        log.check("schedule.rank_independence_transform_R%d" % world, bool(ok),
                  "transform(global)[rank r] == transform(shard r) bitwise")


# ---------------------------------------------------------------------------
# 6. initial bundle
# ---------------------------------------------------------------------------

def check_bundle(log: CheckLog, bundle_path: str, manifest_path: str):
    specs = da.bundle_specs()
    spec_names = [s[0] for s in specs]
    if not os.path.exists(bundle_path):
        log.check("bundle.file_present", False, "not found: %s" % bundle_path)
        return
    with np.load(bundle_path) as z:
        names = sorted(z.files)
        missing = sorted(set(spec_names) - set(names))
        extra = sorted(set(names) - set(spec_names))
        log.check("bundle.name_set", bool(len(names) == 62 and not missing and not extra),
                  "n=%d missing=%s extra=%s" % (len(names), missing[:5] or "none", extra[:5] or "none"))

        bad_shape = []
        bad_dtype = []
        for name, shape, _layout, _kind in specs:
            if name not in z.files:
                continue
            arr = z[name]
            if tuple(arr.shape) != tuple(shape):
                bad_shape.append(name)
            if arr.dtype != np.float32:
                bad_dtype.append(name)
        log.check("bundle.shapes", not bad_shape, "mismatches=%s" % (bad_shape[:5] or "none"))
        log.check("bundle.dtypes_float32", not bad_dtype, "mismatches=%s" % (bad_dtype[:5] or "none"))

        total = sum(int(np.prod(z[n].shape)) for n in z.files if n in spec_names)
        log.check("bundle.element_count", bool(total == 11271432), "total=%d want=11271432" % total)

        gn_w = [n for n in spec_names
                if n.endswith((".gn.weight", ".gn1.weight", ".gn2.weight", ".proj_gn.weight"))]
        gn_b = [n for n in spec_names
                if n.endswith((".gn.bias", ".gn1.bias", ".gn2.bias", ".proj_gn.bias"))]
        okw = len(gn_w) == 20 and all(n in z.files and np.all(z[n] == np.float32(1.0)) for n in gn_w)
        okb = len(gn_b) == 20 and all(n in z.files and np.all(z[n] == np.float32(0.0)) for n in gn_b)
        log.check("bundle.gn_initial_values", bool(okw and okb),
                  "weights%s (%d arrays) biases%s (%d arrays)"
                  % ("==1" if okw else "!=1", len(gn_w), "==0" if okb else "!=0", len(gn_b)))
        ok = "classifier.weight" in z.files and "classifier.bias" in z.files and \
            z["classifier.weight"].shape == (200, 512) and z["classifier.bias"].shape == (200,)
        log.check("bundle.classifier_shapes", bool(ok),
                  "weight=%s bias=%s" % (z["classifier.weight"].shape if "classifier.weight" in z.files else None,
                                         z["classifier.bias"].shape if "classifier.bias" in z.files else None))

        if not os.path.exists(manifest_path):
            log.check("bundle.manifest_sha256", False, "manifest not found: %s" % manifest_path)
            log.check("bundle.bundle_hash", False, "manifest not found")
            return
        manifest = da.read_json(manifest_path)
        entries = manifest.get("arrays")
        if not isinstance(entries, list):
            log.check("bundle.manifest_entries", False, "manifest 'arrays' is not a list")
            log.check("bundle.manifest_sha256", False, "cannot check")
            log.check("bundle.bundle_hash", False, "cannot check")
            return
        entry_names = [e.get("name") for e in entries]
        order_ok = entry_names == spec_names
        entries_ok = len(entries) == 62 and order_ok
        log.check("bundle.manifest_entries", bool(entries_ok),
                  "n_entries=%d order_matches_traversal=%s" % (len(entries), order_ok))
        mismatch = []
        meta_bad = []
        for e in entries:
            if e.get("name") not in z.files:
                continue
            arr = z[e["name"]]
            if da.sha256_c_order(arr) != e.get("sha256"):
                mismatch.append(e["name"])
            if e.get("dtype") != "float32" or [int(x) for x in e.get("shape", [])] != [int(x) for x in arr.shape]:
                meta_bad.append(e["name"])
        log.check("bundle.manifest_sha256", bool(not mismatch and not meta_bad),
                  "checked=%d sha_mismatch=%s meta_mismatch=%s" % (len(entries), mismatch[:5] or "none",
                                                                   meta_bad[:5] or "none"))

        definition = manifest.get("bundle_hash_definition")
        recorded = manifest.get("bundle_hash")
        if definition != BUNDLE_HASH_DEFINITION:
            log.check("bundle.bundle_hash", False,
                      "unexpected bundle_hash_definition: %r" % definition)
        else:
            order = [e["name"] for e in entries]
            recomputed = da.bundle_hash_of_arrays([z[n] for n in order])
            log.check("bundle.bundle_hash", bool(recomputed == recorded),
                      "recomputed=%s recorded=%s rule=\"concat C-order bytes, manifest order\""
                      % (recomputed, recorded))


# ---------------------------------------------------------------------------
# 7-9. fixtures
# ---------------------------------------------------------------------------

def _load_npz(path):
    with np.load(path) as z:
        return {n: z[n] for n in z.files}


def check_fixture_integrity(log: CheckLog, fixture_dir: str):
    manifest_path = os.path.join(fixture_dir, "fixture_manifest.json")
    if not os.path.exists(manifest_path):
        log.check("fixtures.manifest_present", False, "not found: %s" % manifest_path)
        return None
    manifest = da.read_json(manifest_path)
    log.check("fixtures.manifest_present", True, "synthetic=%s" % manifest.get("synthetic"))
    files_ok = True
    for f in manifest.get("files", []):
        path = os.path.join(fixture_dir, f["path"])
        if not os.path.exists(path) or da.sha256_file(path) != f["sha256"]:
            files_ok = False
    log.check("fixtures.file_sha256", bool(files_ok),
              "files=%d" % len(manifest.get("files", [])))
    by_file = {}
    for e in manifest.get("arrays", []):
        if "::" not in e.get("name", ""):
            by_file.setdefault("__bad__", []).append(e)
            continue
        fname, aname = e["name"].split("::", 1)
        by_file.setdefault(fname, []).append((aname, e))
    for fname, flist in by_file.items():
        path = os.path.join(fixture_dir, fname)
        if not os.path.exists(path):
            log.check("fixtures.%s" % fname, False, "missing file")
            continue
        got = _load_npz(path)
        want = sorted(a for a, _ in flist)
        extra = sorted(set(got) - set(want))
        missing = sorted(set(want) - set(got))
        bad = []
        for aname, e in flist:
            if aname not in got:
                continue
            arr = got[aname]
            if str(arr.dtype) != e.get("dtype") or [int(x) for x in arr.shape] != e.get("shape"):
                bad.append(aname + ":dtype/shape")
            elif da.sha256_c_order(arr) != e.get("sha256"):
                bad.append(aname + ":sha256")
        log.check("fixtures.%s" % fname,
                  bool(not extra and not missing and not bad),
                  "arrays=%d extra=%s missing=%s bad=%s"
                  % (len(flist), extra[:3] or "none", missing[:3] or "none", bad[:3] or "none"))
    # D3/D4 must exist even if the manifest were empty
    for fname in ("linear_fixture.npz", "kernel_fixture.npz"):
        if not os.path.exists(os.path.join(fixture_dir, fname)):
            log.check("fixtures.%s" % fname, False, "missing file")
    return manifest


def _contract_ok(arr, dtype, shape):
    return str(arr.dtype) == dtype and tuple(arr.shape) == shape


def check_data_fixture_recompute(log: CheckLog, prep_dir: str, fixture_dir: str, mean32, std32):
    path = os.path.join(fixture_dir, "data_fixture.npz")
    if not os.path.exists(path):
        log.check("data_fixture.file", False, "missing %s" % path)
        return
    fx = _load_npz(path)
    contract_bad = [n for n, (dt, sh) in DATA_FIXTURE_CONTRACT.items()
                    if n not in fx or not _contract_ok(fx[n], dt, sh)]
    log.check("data_fixture.contract", not contract_bad, "bad=%s" % (contract_bad or "none"))
    if contract_bad:
        return
    train_ids = np.asarray(da.TRAIN_FIXTURE_IDS, dtype=np.int64)
    val_ids = np.asarray(da.VAL_FIXTURE_IDS, dtype=np.int64)
    train_images = da.open_prepared_train_images(prep_dir)
    val_images = da.open_prepared_val_images(prep_dir)
    train_labels = np.load(os.path.join(prep_dir, "train_labels.npy"))
    val_labels = np.load(os.path.join(prep_dir, "val_labels.npy"))
    augmentation = da.open_prepared_augmentation(prep_dir)

    decoded_train = np.ascontiguousarray(train_images[train_ids])
    log.check("data_fixture.decoded_train_images",
              bool(np.array_equal(decoded_train, fx["decoded_train_images"])),
              "ids=%s" % list(da.TRAIN_FIXTURE_IDS))
    log.check("data_fixture.train_labels",
              bool(np.array_equal(train_labels[train_ids], fx["train_labels"])),
              "")
    aug = np.ascontiguousarray(np.asarray(augmentation[0:2])[:, train_ids, :])
    log.check("data_fixture.augmentation",
              bool(np.array_equal(aug, fx["augmentation"])),
              "epochs=[0,1]")
    refs = np.empty((2, len(train_ids), 3, 64, 64), dtype=np.float32)
    for e in range(2):
        refs[e] = da.replay_transform(decoded_train, aug[e], mean32, std32)
    log.check("data_fixture.train_transformed_ref",
              bool(np.array_equal(refs, fx["train_transformed_ref"])),
              "bitwise replay from PREP")
    decoded_val = np.ascontiguousarray(val_images[val_ids])
    log.check("data_fixture.decoded_val_images",
              bool(np.array_equal(decoded_val, fx["decoded_val_images"])),
              "ids=%s" % list(da.VAL_FIXTURE_IDS))
    log.check("data_fixture.val_labels",
              bool(np.array_equal(val_labels[val_ids], fx["val_labels"])),
              "")
    val_ref = da.replay_transform_validation(decoded_val, mean32, std32)
    log.check("data_fixture.val_transformed_ref",
              bool(np.array_equal(val_ref, fx["val_transformed_ref"])),
              "bitwise replay from PREP")
    log.check("data_fixture.normalization_constants",
              bool(np.array_equal(mean32, fx["normalization_mean"]) and
                   np.array_equal(std32, fx["normalization_std"])),
              "mean/std equal stats.json float32")


def check_schedule_fixture_recompute(log: CheckLog, prep_dir: str, fixture_dir: str, perm0):
    path = os.path.join(fixture_dir, "schedule_fixture.npz")
    if not os.path.exists(path):
        log.check("schedule_fixture.file", False, "missing %s" % path)
        return
    fx = _load_npz(path)
    contract_bad = [n for n, (dt, sh) in SCHEDULE_FIXTURE_CONTRACT.items()
                    if n not in fx or not _contract_ok(fx[n], dt, sh)]
    log.check("schedule_fixture.contract", not contract_bad, "bad=%s" % (contract_bad or "none"))
    if contract_bad:
        return
    log.check("schedule_fixture.epoch0_permutation",
              bool(np.array_equal(np.asarray(perm0), fx["epoch0_permutation"])), "")
    augmentation = da.open_prepared_augmentation(prep_dir)
    aug01 = np.ascontiguousarray(np.asarray(augmentation[0:2]))
    log.check("schedule_fixture.augmentation_epochs01",
              bool(np.array_equal(aug01, fx["augmentation_epochs01"])), "")
    for world, key in ((1, "rank_ids_R1"), (2, "rank_ids_R2"), (4, "rank_ids_R4")):
        seqs = [da.rank_sequence(perm0, da.EPOCH_BATCH, r, world) for r in range(world)]
        ref = np.ascontiguousarray(seqs[0]) if world == 1 else np.stack(seqs)
        log.check("schedule_fixture.%s" % key, bool(np.array_equal(ref, fx[key])),
                  "batch=%d world=%d" % (da.EPOCH_BATCH, world))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checks(log: CheckLog, opts) -> None:
    prep_dir = opts.prep_dir
    fixture_dir = opts.fixture_dir
    for p in (prep_dir, fixture_dir):
        if not os.path.isdir(p):
            log.check("directories.exist", False, "not a directory: %s" % p)
            return

    if not check_prep_manifests(log, prep_dir):
        log.skip("prep.semantic_checks", "skipped: prep manifest integrity failed")
        check_bundle(log, opts.bundle, opts.bundle_manifest)
        check_fixture_integrity(log, fixture_dir)
        log.skip("data_fixture.recompute", "skipped: prep manifest integrity failed")
        log.skip("schedule_fixture.recompute", "skipped: prep manifest integrity failed")
        return

    check_counts(log, prep_dir, opts.dataset_root)
    mean32, std32, _stats = check_stats(log, prep_dir)
    perm0 = check_permutation_augmentation(log, prep_dir)
    if mean32 is None or perm0 is None:
        return
    check_schedule_reconstruction(log, prep_dir, perm0, mean32, std32)
    check_bundle(log, opts.bundle, opts.bundle_manifest)
    check_fixture_integrity(log, fixture_dir)
    check_data_fixture_recompute(log, prep_dir, fixture_dir, mean32, std32)
    check_schedule_fixture_recompute(log, prep_dir, fixture_dir, perm0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 2 Python acceptance driver (prepared dataset + fixtures)")
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--fixture-dir", required=True)
    p.add_argument("--bundle", default=None, help="default: <fixture-dir>/initial_bundle.npz")
    p.add_argument("--bundle-manifest", default=None, help="default: <fixture-dir>/initial_bundle.manifest.json")
    p.add_argument("--dataset-root", default=DATASET_ROOT_DEFAULT,
                   help="raw tiny-imagenet-200 root for the annotation-mapping check; SKIP if absent")
    args = p.parse_args(argv)
    if args.bundle is None:
        args.bundle = os.path.join(args.fixture_dir, "initial_bundle.npz")
    if args.bundle_manifest is None:
        args.bundle_manifest = os.path.join(args.fixture_dir, "initial_bundle.manifest.json")

    log = CheckLog()
    try:
        run_checks(log, args)
    except Exception as err:  # noqa: BLE001 - acceptance driver must report, not crash
        import traceback
        log.check("unhandled_exception", False, "%s: %s" % (type(err).__name__, err))
        traceback.print_exc()
    if log.failed:
        print("PHASE2_DATASET_CHECK FAIL %s" % ",".join(log.failed))
        return 1
    print("PHASE2_DATASET_CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
