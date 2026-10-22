"""
Phase 2 static-scaling shared fixture generator (real torch mode / synthetic mode).

Real mode (Leonardo compute node, pinned venv with torch) writes into --out-dir:
  data_fixture.npz      D1 decoded/transform reference fixtures
  schedule_fixture.npz  D2 epoch-0 schedule and rank-shard references
  linear_fixture.npz    D3 rectangular linear forward/gradient references (torch autograd)
  kernel_fixture.npz    D4 asymmetric cross-correlation forward/gradient references (torch autograd)
  fixture_manifest.json D5 integrity manifest

Synthetic mode (--synthetic; numpy only, for local integration of the Julia
checker) writes contract-shaped fixtures with procedure-generated values and
fixture_manifest.json marked "synthetic": true. It also writes
initial_bundle.npz + initial_bundle.manifest.json (procedural weights, also
marked synthetic). With --synthetic-prep it additionally writes a full
contract-shaped PREP directory (uint8 images, permutations, augmentation,
stats.json, manifests) so that check_dataset.py can be exercised locally.
Real Leonardo runs must never use --synthetic.

Torch fixture values (real mode): per random source array i (see seed_rule in
the manifest) torch.Generator(device="cpu").manual_seed(20261016+900+i) with
torch.randn(shape); y and all gradients come from torch autograd of sum(y).
Linear: y = W @ x + b with W (3,7), b (3,), x (7,5). Kernel case A: weight
(3,2,2,3), x (2,2,5,6), stride (1,1), padding (0,0), y (2,3,4,4); case B:
weight (2,3,2,3), x (1,3,9,7), stride (2,2), padding (1,1), y (1,2,5,4). Both
kernels and both inputs are asserted nonzero and asymmetric.

fixture_manifest.json schema (v1, one flat list):
  producer, created, synthetic, source_commit, source_state_sha256, generator,
  generator_version, schema, seed_rule (real mode),
  files:  [{"path", "bytes", "sha256"}]
  arrays: [{"name": "<npz file>::<array name>", "dtype", "shape", "layout",
            "sha256"}] ; sha256 = hashlib.sha256(arr.tobytes(order="C"))
An independent checker verifies every array's dtype/shape/sha256 and the set
equality per file (check_dataset.py check 7).

Reproduction (real, Leonardo compute node):
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/make_fixtures.py \\
    --prep-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared \\
    --out-dir artifacts/baselines/static-scaling/2026-10-16-phase2

Reproduction (synthetic, local, no torch/PIL):
  python3 scripts/static_scaling/make_fixtures.py --synthetic \\
    --out-dir /tmp/opencode/phase2-synth

Deterministic stdout lines: FIXTURES <key>=<value>.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import dataset_adapter as da

FIXTURE_MANIFEST_NAME = "fixture_manifest.json"
SCHEMA_TEXT = (
    "flat arrays list; name = '<npz file>::<array name>'; sha256 = "
    "sha256 of arr.tobytes(order=\"C\"); files entry carries the npz byte hash"
)
REAL_SEED_RULE = (
    "torch.Generator(device='cpu').manual_seed(20261016 + 900 + i) per random source array; "
    "index i: linear.weight=0, linear.bias=1, linear.x=2, kernel.a_weight=3, kernel.a_x=4, "
    "kernel.b_weight=5, kernel.b_x=6; values are torch.randn(shape, generator=g, dtype=torch.float32); "
    "y and gradients are torch autograd results of sum(y)"
)
SYNTHETIC_SEED_RULE = (
    "synthetic mode: numpy.random.default_rng(20261016) drives the procedural fixture values in a "
    "fixed documented order (linear weights/bias/x, kernel weights/inputs) and "
    "default_rng(20261016 + i) per bundle array i"
)

LAYOUT = {
    "decoded_train_images": "nchw_uint8",
    "train_labels": "int64_vector",
    "augmentation": "epoch_sample_top_left_flip_uint8",
    "train_transformed_ref": "nchw_float32_normalized",
    "decoded_val_images": "nchw_uint8",
    "val_labels": "int64_vector",
    "val_transformed_ref": "nchw_float32_normalized",
    "normalization_mean": "float32_channel_constants",
    "normalization_std": "float32_channel_constants",
    "epoch0_permutation": "int64_vector",
    "augmentation_epochs01": "epoch_sample_top_left_flip_uint8",
    "rank_ids_R1": "int64_rank_sequence",
    "rank_ids_R2": "int64_rank_sequences",
    "rank_ids_R4": "int64_rank_sequences",
    "weight": "linear_out_in",
    "bias": "linear_bias",
    "x": "linear_input_in_batch",
    "y": "linear_output_out_batch",
    "x_grad": "linear_input_in_batch",
    "w_grad": "linear_out_in",
    "b_grad": "linear_bias",
    "a_weight": "conv_kernel_out_in_kh_kw",
    "a_x": "nchw_float32",
    "a_y": "nchw_float32",
    "a_x_grad": "nchw_float32",
    "a_w_grad": "conv_kernel_out_in_kh_kw",
    "a_stride": "int64_pair",
    "a_padding": "int64_pair",
    "b_weight": "conv_kernel_out_in_kh_kw",
    "b_x": "nchw_float32",
    "b_y": "nchw_float32",
    "b_x_grad": "nchw_float32",
    "b_w_grad": "conv_kernel_out_in_kh_kw",
    "b_stride": "int64_pair",
    "b_padding": "int64_pair",
}


# ---------------------------------------------------------------------------
# Real-mode builders (D1, D2 read $PREP; D3, D4 use torch autograd)
# ---------------------------------------------------------------------------

def build_data_fixture(prep_dir: str) -> dict:
    mean32, std32, _ = da.load_stats(prep_dir)
    train_images = da.open_prepared_train_images(prep_dir)
    val_images = da.open_prepared_val_images(prep_dir)
    train_labels = np.load(os.path.join(prep_dir, "train_labels.npy"))
    val_labels = np.load(os.path.join(prep_dir, "val_labels.npy"))
    augmentation = da.open_prepared_augmentation(prep_dir)

    train_ids = np.asarray(da.TRAIN_FIXTURE_IDS, dtype=np.int64)
    val_ids = np.asarray(da.VAL_FIXTURE_IDS, dtype=np.int64)
    decoded_train = np.ascontiguousarray(train_images[train_ids])
    decoded_val = np.ascontiguousarray(val_images[val_ids])
    aug = np.ascontiguousarray(np.asarray(augmentation[0:2])[:, train_ids, :])

    train_refs = np.empty((2, len(train_ids), 3, 64, 64), dtype=np.float32)
    for e in range(2):
        train_refs[e] = da.replay_transform(decoded_train, aug[e], mean32, std32)
    val_ref = da.replay_transform_validation(decoded_val, mean32, std32)

    return {
        "decoded_train_images": decoded_train,
        "train_labels": np.ascontiguousarray(train_labels[train_ids]).astype(np.int64, copy=False),
        "augmentation": aug,
        "train_transformed_ref": train_refs,
        "decoded_val_images": decoded_val,
        "val_labels": np.ascontiguousarray(val_labels[val_ids]).astype(np.int64, copy=False),
        "val_transformed_ref": val_ref,
        "normalization_mean": np.ascontiguousarray(mean32),
        "normalization_std": np.ascontiguousarray(std32),
    }


def build_schedule_fixture(prep_dir: str) -> dict:
    permutations = da.open_prepared_permutations(prep_dir)
    augmentation = da.open_prepared_augmentation(prep_dir)
    perm0 = np.ascontiguousarray(permutations[0]).astype(np.int64, copy=False)
    aug01 = np.ascontiguousarray(np.asarray(augmentation[0:2]))
    rank_ids = {}
    for r_world in (1, 2, 4):
        seqs = [da.rank_sequence(perm0, da.EPOCH_BATCH, r, r_world) for r in range(r_world)]
        rank_ids[r_world] = (np.ascontiguousarray(seqs[0]).astype(np.int64, copy=False)
                             if r_world == 1 else np.stack(seqs).astype(np.int64, copy=False))
    return {
        "epoch0_permutation": perm0,
        "augmentation_epochs01": aug01,
        "rank_ids_R1": rank_ids[1],
        "rank_ids_R2": rank_ids[2],
        "rank_ids_R4": rank_ids[4],
    }


def _np(t) -> np.ndarray:
    return np.ascontiguousarray(t.detach().numpy())


def _assert_nonzero(arrays, label):
    for name, arr in arrays.items():
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        if not np.all(arr != 0):
            raise SystemExit("%s fixture %s contains an exact zero" % (label, name))


def _assert_asymmetric_kernel(w: np.ndarray, name: str):
    if not np.any(w != 0):
        raise SystemExit("kernel %s is all zero" % name)
    variants = [np.flip(w, 2), np.flip(w, 3), np.flip(np.flip(w, 2), 3)]
    if w.shape[2] == w.shape[3]:
        variants.append(np.swapaxes(w, 2, 3))
    if any(np.array_equal(w, v) for v in variants):
        raise SystemExit("kernel %s is symmetric; the orientation canary would not fire" % name)


def _assert_asymmetric_input(x: np.ndarray, name: str):
    if not np.any(x != 0):
        raise SystemExit("input %s is all zero" % name)
    for axis in (2, 3):
        if np.array_equal(x, np.flip(x, axis)):
            if x.shape[2] == 1 or x.shape[3] == 1:
                continue
            raise SystemExit("input %s symmetric along axis %d" % (name, axis))


def build_linear_fixture_torch(seed_base: int = da.ROOT_SCHEDULE + 900) -> dict:
    import torch

    def gen(i):
        return torch.Generator(device="cpu").manual_seed(seed_base + i)

    w = torch.randn((3, 7), generator=gen(0), dtype=torch.float32).requires_grad_(True)
    b = torch.randn((3,), generator=gen(1), dtype=torch.float32).requires_grad_(True)
    x = torch.randn((7, 5), generator=gen(2), dtype=torch.float32).requires_grad_(True)
    y = w @ x + b[:, None]
    y.sum().backward()
    arrays = {
        "weight": _np(w), "bias": _np(b), "x": _np(x), "y": _np(y),
        "x_grad": _np(x.grad), "w_grad": _np(w.grad), "b_grad": _np(b.grad),
    }
    if arrays["y"].shape != (3, 5):
        raise SystemExit("linear fixture y shape drift: %s" % (arrays["y"].shape,))
    _assert_nonzero(arrays, "linear")
    return arrays


def build_kernel_fixture_torch(seed_base: int = da.ROOT_SCHEDULE + 900) -> dict:
    import torch
    import torch.nn.functional as F

    def gen(i):
        return torch.Generator(device="cpu").manual_seed(seed_base + i)

    def case(tag, idx, w_shape, x_shape, stride, padding, y_shape):
        w = torch.randn(w_shape, generator=gen(idx), dtype=torch.float32).requires_grad_(True)
        x = torch.randn(x_shape, generator=gen(idx + 1), dtype=torch.float32).requires_grad_(True)
        y = F.conv2d(x, w, None, stride=tuple(stride), padding=tuple(padding))
        if tuple(y.shape) != tuple(y_shape):
            raise SystemExit("kernel case %s y shape %s, expected %s" % (tag, tuple(y.shape), y_shape))
        y.sum().backward()
        out = {
            "%s_weight" % tag: _np(w),
            "%s_x" % tag: _np(x),
            "%s_y" % tag: _np(y),
            "%s_x_grad" % tag: _np(x.grad),
            "%s_w_grad" % tag: _np(w.grad),
            "%s_stride" % tag: np.asarray(stride, dtype=np.int64),
            "%s_padding" % tag: np.asarray(padding, dtype=np.int64),
        }
        _assert_asymmetric_kernel(out["%s_weight" % tag], "%s_weight" % tag)
        _assert_asymmetric_input(out["%s_x" % tag], "%s_x" % tag)
        _assert_nonzero(out, "kernel_%s" % tag)
        return out

    arrays = {}
    arrays.update(case("a", 3, (3, 2, 2, 3), (2, 2, 5, 6), (1, 1), (0, 0), (2, 3, 4, 4)))
    arrays.update(case("b", 5, (2, 3, 2, 3), (1, 3, 9, 7), (2, 2), (1, 1), (1, 2, 5, 4)))
    return arrays


# ---------------------------------------------------------------------------
# Synthetic numpy references (cross-correlation = torch conv2d semantics)
# ---------------------------------------------------------------------------

def numpy_conv2d(x: np.ndarray, w: np.ndarray, stride, padding) -> np.ndarray:
    n, _, h, wd = x.shape
    o, _, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (wd + 2 * pw - kw) // sw + 1
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    y = np.zeros((n, o, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            y[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return y


def numpy_conv2d_grads(x: np.ndarray, w: np.ndarray, stride, padding):
    """Gradients of sum(conv2d(x, w)); manual accumulation in float32."""
    n, _, h, wd = x.shape
    o, _, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (wd + 2 * pw - kw) // sw + 1
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    dy = np.ones((n, o, oh, ow), dtype=np.float32)
    dw = np.zeros_like(w)
    dxp = np.zeros_like(xp)
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            dw += np.einsum("ncuv,no->ocuv", patch, dy[:, :, i, j])
            dxp[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw] += np.einsum(
                "ocuv,no->ncuv", w, dy[:, :, i, j])
    dx = dxp[:, :, ph:ph + h, pw:pw + wd]
    return dx, dw


def build_linear_fixture_numpy(rng: np.random.Generator) -> dict:
    w = rng.standard_normal((3, 7)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    x = rng.standard_normal((7, 5)).astype(np.float32)
    y = (w @ x + b[:, None]).astype(np.float32)
    dy = np.ones_like(y)
    arrays = {
        "weight": w, "bias": b, "x": x, "y": y,
        "x_grad": (w.T @ dy).astype(np.float32),
        "w_grad": (dy @ x.T).astype(np.float32),
        "b_grad": dy.sum(axis=1).astype(np.float32),
    }
    _assert_nonzero(arrays, "linear")
    return arrays


def build_kernel_fixture_numpy(rng: np.random.Generator) -> dict:
    def case(tag, w_shape, x_shape, stride, padding, y_shape):
        w = rng.standard_normal(w_shape).astype(np.float32)
        x = rng.standard_normal(x_shape).astype(np.float32)
        y = numpy_conv2d(x, w, stride, padding)
        if tuple(y.shape) != tuple(y_shape):
            raise SystemExit("synthetic kernel %s y shape %s, expected %s" % (tag, y.shape, y_shape))
        dx, dw = numpy_conv2d_grads(x, w, stride, padding)
        out = {
            "%s_weight" % tag: w,
            "%s_x" % tag: x,
            "%s_y" % tag: y,
            "%s_x_grad" % tag: dx,
            "%s_w_grad" % tag: dw,
            "%s_stride" % tag: np.asarray(stride, dtype=np.int64),
            "%s_padding" % tag: np.asarray(padding, dtype=np.int64),
        }
        _assert_asymmetric_kernel(w, "%s_weight" % tag)
        _assert_asymmetric_input(x, "%s_x" % tag)
        _assert_nonzero(out, "kernel_%s" % tag)
        return out

    arrays = {}
    arrays.update(case("a", (3, 2, 2, 3), (2, 2, 5, 6), (1, 1), (0, 0), (2, 3, 4, 4)))
    arrays.update(case("b", (2, 3, 2, 3), (1, 3, 9, 7), (2, 2), (1, 1), (1, 2, 5, 4)))
    return arrays


# ---------------------------------------------------------------------------
# Fixture writing and manifest
# ---------------------------------------------------------------------------

def write_fixture_set(out_dir: str, file_arrays, meta: dict):
    entries = []
    files_meta = []
    for fname, arrays in file_arrays:
        path = os.path.join(out_dir, fname)
        np.savez(path, **arrays)
        files_meta.append({"path": fname, "bytes": os.path.getsize(path),
                           "sha256": da.sha256_file(path)})
        for aname, arr in arrays.items():
            entries.append({
                "name": fname + "::" + aname,
                "dtype": str(arr.dtype),
                "shape": [int(x) for x in arr.shape],
                "layout": LAYOUT.get(aname, ""),
                "sha256": da.sha256_c_order(arr),
            })
        print("FIXTURES wrote %s arrays=%d" % (path, len(arrays)), flush=True)

    manifest = {
        "producer": "scripts/static_scaling/make_fixtures.py",
        "created": da.utc_now(),
        "synthetic": meta["synthetic"],
        **da.source_info(meta.get("source_commit"), meta.get("source_state_sha256")),
        "generator": meta["generator"],
        "generator_version": meta["generator_version"],
        "schema": SCHEMA_TEXT,
        "seed_rule": meta["seed_rule"],
        "files": files_meta,
        "arrays": entries,
    }
    manifest_path = os.path.join(out_dir, FIXTURE_MANIFEST_NAME)
    da.write_json(manifest_path, manifest)
    print("FIXTURES wrote %s arrays_total=%d" % (manifest_path, len(entries)), flush=True)

    # Verify by re-reading every archive: name set, dtype, shape, sha256.
    by_file = {}
    for entry in entries:
        by_file.setdefault(entry["name"].split("::")[0], []).append(entry)
    for fname, flist in by_file.items():
        with np.load(os.path.join(out_dir, fname)) as z:
            got = sorted(z.files)
            want = sorted(e["name"].split("::", 1)[1] for e in flist)
            if got != want:
                raise SystemExit("%s array set mismatch after write: %s vs %s" % (fname, got, want))
            for entry in flist:
                aname = entry["name"].split("::", 1)[1]
                arr = z[aname]
                if str(arr.dtype) != entry["dtype"] or [int(x) for x in arr.shape] != entry["shape"]:
                    raise SystemExit("%s::%s dtype/shape mismatch after write" % (fname, aname))
                if da.sha256_c_order(arr) != entry["sha256"]:
                    raise SystemExit("%s::%s sha256 mismatch after write" % (fname, aname))
    print("FIXTURES verified files=%d arrays=%d sha256=true" % (len(files_meta), len(entries)), flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Synthetic PREP / bundle helpers (numpy only)
# ---------------------------------------------------------------------------

def _fill_synthetic_images(memmap_path, shape, base):
    images = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.uint8, shape=shape)
    n = shape[0]
    chunk = 2000
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        idx = np.arange(start, stop, dtype=np.int64)[:, None, None, None]
        images[start:stop] = base[None] ^ (idx % 256).astype(np.uint8)
    images.flush()
    del images


def write_synthetic_prep(out_dir: str) -> None:
    """Contract-shaped synthetic PREP (marked synthetic in its manifests)."""
    rng = np.random.default_rng(da.ROOT_SCHEDULE)
    base_train = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)
    base_val = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)
    _fill_synthetic_images(os.path.join(out_dir, "train_images.npy"), (da.N_TRAIN, 3, 64, 64), base_train)
    _fill_synthetic_images(os.path.join(out_dir, "val_images.npy"), (da.N_VAL, 3, 64, 64), base_val)
    train_labels = ((np.arange(da.N_TRAIN, dtype=np.int64) // 500) % da.N_CLASSES).astype(np.int64)
    val_labels = ((np.arange(da.N_VAL, dtype=np.int64) // 50) % da.N_CLASSES).astype(np.int64)
    np.save(os.path.join(out_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(out_dir, "val_labels.npy"), val_labels)
    with open(os.path.join(out_dir, "class_wnids.txt"), "w", encoding="utf-8") as f:
        for c in range(da.N_CLASSES):
            f.write("synthetic-wnid-%03d\n" % c)

    train_images = np.load(os.path.join(out_dir, "train_images.npy"), mmap_mode="r")
    sum64 = np.zeros(3, dtype=np.float64)
    sumsq64 = np.zeros(3, dtype=np.float64)
    for start in range(0, da.N_TRAIN, 2000):
        stop = min(da.N_TRAIN, start + 2000)
        x64 = train_images[start:stop].astype(np.float64) / 255.0
        sum64 += x64.sum(axis=(0, 2, 3))
        sumsq64 += (x64 * x64).sum(axis=(0, 2, 3))
    pixel_count = da.N_TRAIN * 64 * 64
    mean64 = sum64 / float(pixel_count)
    var64 = sumsq64 / float(pixel_count) - mean64 * mean64
    mean32 = mean64.astype(np.float32)
    std32 = np.sqrt(np.maximum(var64, 0.0)).astype(np.float32)
    stats = {
        "mean": [float(x) for x in mean32],
        "std": [float(x) for x in std32],
        "pixel_count_per_channel": pixel_count,
        "dtype": "float32",
        "method": "synthetic procedural images; same OD-5 float64 sequential reduction as the real builder",
        "reduction": "per-chunk float64 sums in canonical order",
        "synthetic": True,
    }
    da.write_json(os.path.join(out_dir, "stats.json"), stats)

    perms = np.empty((da.EPOCHS, da.N_TRAIN), dtype=np.int64)
    augs = np.empty((da.EPOCHS, da.N_TRAIN, 3), dtype=np.uint8)
    for e in range(da.EPOCHS):
        perms[e] = da.epoch_permutation(e)
        augs[e] = da.epoch_augmentation(e)
    np.save(os.path.join(out_dir, "permutations.npy"), perms)
    np.save(os.path.join(out_dir, "augmentation.npy"), augs)

    files = [{"path": n, "bytes": os.path.getsize(os.path.join(out_dir, n)),
              "sha256": da.sha256_file(os.path.join(out_dir, n))}
             for n in ("train_images.npy", "train_labels.npy", "val_images.npy",
                       "val_labels.npy", "class_wnids.txt", "stats.json",
                       "permutations.npy", "augmentation.npy")]
    da.write_json(os.path.join(out_dir, "dataset_manifest.json"), {
        "producer": "scripts/static_scaling/make_fixtures.py --synthetic",
        "created": da.utc_now(), "synthetic": True,
        **da.source_info(),
        "generator": {"library": "numpy", "version": np.__version__},
        "note": "synthetic procedure-generated PREP for local integration only; never use for Leonardo runs",
        "files": files,
    })
    da.write_json(os.path.join(out_dir, "schedule_manifest.json"), {
        "producer": "scripts/static_scaling/make_fixtures.py --synthetic",
        "created": da.utc_now(), "synthetic": True,
        **da.source_info(),
        "generator": {"library": "numpy", "version": np.__version__, "bit_generator": "PCG64"},
        "root_seed": da.ROOT_SCHEDULE,
        "scheme": "same SeedSequence scheme as make_schedule.py (real frozen scheme)",
        "epochs": da.EPOCHS,
        "extension_rule": "same as make_schedule.py; earlier epochs never change",
        "files": [f for f in files if f["path"].endswith(("permutations.npy", "augmentation.npy"))],
    })
    print("FIXTURES synthetic_prep wrote %s" % out_dir, flush=True)


def build_synthetic_bundle(out_dir: str):
    """Procedural 62-array bundle (no torch); same contract, marked synthetic."""
    names = []
    payload = {}
    for i, (name, shape, _layout, kind) in enumerate(da.bundle_specs()):
        if kind == "ones":
            arr = np.ones(shape, dtype=np.float32)
        elif kind == "zeros":
            arr = np.zeros(shape, dtype=np.float32)
        else:
            arr = np.random.default_rng(da.ROOT_SCHEDULE + i).standard_normal(shape).astype(np.float32)
        names.append(name)
        payload[name] = np.ascontiguousarray(arr)
    total = sum(int(a.size) for a in payload.values())
    if len(payload) != 62 or total != 11271432:
        raise SystemExit("synthetic bundle shape drift: arrays=%d elements=%d" % (len(payload), total))
    npz_path = os.path.join(out_dir, "initial_bundle.npz")
    np.savez(npz_path, **payload)
    entries = []
    for name, shape, layout, _kind in da.bundle_specs():
        entries.append({"name": name, "dtype": "float32", "shape": [int(x) for x in shape],
                        "layout": layout, "sha256": da.sha256_c_order(payload[name])})
    bundle_hash = da.bundle_hash_of_arrays([payload[e["name"]] for e in entries])
    da.write_json(os.path.join(out_dir, "initial_bundle.manifest.json"), {
        "producer": "scripts/static_scaling/make_fixtures.py --synthetic",
        "created": da.utc_now(), "synthetic": True,
        **da.source_info(),
        "generator": "numpy procedural (synthetic mode; torch not used)",
        "generator_version": np.__version__,
        "root_seed": da.ROOT_SCHEDULE,
        "per_array_seed_rule": SYNTHETIC_SEED_RULE,
        "recipe": "procedural: ones/zeros for GN constants, default_rng(20261016+i).standard_normal(shape) for weights",
        "arrays": entries,
        "bundle_hash": bundle_hash,
        "bundle_hash_definition": "sha256 over concat of per-array raw C-order bytes in manifest order",
    })
    with np.load(npz_path) as z:
        reread = da.bundle_hash_of_arrays([z[e["name"]] for e in entries])
    if reread != bundle_hash:
        raise SystemExit("synthetic bundle_hash mismatch after re-read")
    print("FIXTURES wrote %s arrays=62 bundle_hash=%s" % (npz_path, bundle_hash), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _lean_synthetic_data_schedule():
    rng = np.random.default_rng(da.ROOT_SCHEDULE)
    train_ids = np.asarray(da.TRAIN_FIXTURE_IDS, dtype=np.int64)
    val_ids = np.asarray(da.VAL_FIXTURE_IDS, dtype=np.int64)
    decoded_train = rng.integers(0, 256, size=(len(train_ids), 3, 64, 64), dtype=np.uint8)
    decoded_val = rng.integers(0, 256, size=(len(val_ids), 3, 64, 64), dtype=np.uint8)
    train_labels = ((train_ids // 500) % da.N_CLASSES).astype(np.int64)
    val_labels = ((val_ids // 50) % da.N_CLASSES).astype(np.int64)
    mean32 = rng.uniform(0.3, 0.6, size=3).astype(np.float32)
    std32 = rng.uniform(0.2, 0.4, size=3).astype(np.float32)
    perm0 = da.epoch_permutation(0)
    aug01 = np.stack([da.epoch_augmentation(0), da.epoch_augmentation(1)])
    aug = np.ascontiguousarray(aug01[:, train_ids, :])

    train_refs = np.empty((2, len(train_ids), 3, 64, 64), dtype=np.float32)
    for e in range(2):
        train_refs[e] = da.replay_transform(decoded_train, aug[e], mean32, std32)
    val_ref = da.replay_transform_validation(decoded_val, mean32, std32)

    data = {
        "decoded_train_images": decoded_train,
        "train_labels": train_labels,
        "augmentation": aug,
        "train_transformed_ref": train_refs,
        "decoded_val_images": decoded_val,
        "val_labels": val_labels,
        "val_transformed_ref": val_ref,
        "normalization_mean": np.ascontiguousarray(mean32),
        "normalization_std": np.ascontiguousarray(std32),
    }
    rank_ids = {}
    for r_world in (1, 2, 4):
        seqs = [da.rank_sequence(perm0, da.EPOCH_BATCH, r, r_world) for r in range(r_world)]
        rank_ids[r_world] = (np.ascontiguousarray(seqs[0]).astype(np.int64, copy=False)
                             if r_world == 1 else np.stack(seqs).astype(np.int64, copy=False))
    schedule = {
        "epoch0_permutation": perm0,
        "augmentation_epochs01": aug01,
        "rank_ids_R1": rank_ids[1],
        "rank_ids_R2": rank_ids[2],
        "rank_ids_R4": rank_ids[4],
    }
    return data, schedule


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the Phase 2 shared correctness fixtures")
    p.add_argument("--prep-dir", default=None, help="prepared dataset directory (real mode)")
    p.add_argument("--out-dir", required=True, help="fixture output directory")
    p.add_argument("--synthetic", action="store_true", help="numpy-only contract-shaped integration fixtures")
    p.add_argument("--synthetic-prep", action="store_true",
                   help="synthetic mode only: also write a contract-shaped PREP for local check_dataset runs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--source-state-sha256", default=None)
    args = p.parse_args(argv)

    if args.synthetic and args.prep_dir:
        raise SystemExit("--synthetic and --prep-dir are mutually exclusive")
    if not args.synthetic and not args.prep_dir:
        raise SystemExit("real mode needs --prep-dir (or use --synthetic)")
    if args.synthetic_prep and not args.synthetic:
        raise SystemExit("--synthetic-prep requires --synthetic")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, FIXTURE_MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        raise SystemExit("refusing to overwrite %s (use --force)" % manifest_path)

    if args.synthetic:
        if args.synthetic_prep:
            write_synthetic_prep(args.out_dir)
            data = build_data_fixture(args.out_dir)
            schedule = build_schedule_fixture(args.out_dir)
        else:
            data, schedule = _lean_synthetic_data_schedule()
        build_synthetic_bundle(args.out_dir)
        rng_fix = np.random.default_rng(da.ROOT_SCHEDULE)
        linear = build_linear_fixture_numpy(rng_fix)
        kernel = build_kernel_fixture_numpy(rng_fix)
        meta = {
            "synthetic": True,
            "generator": "numpy manual references (synthetic mode; torch not used)",
            "generator_version": np.__version__,
            "seed_rule": SYNTHETIC_SEED_RULE,
            "source_commit": args.source_commit,
            "source_state_sha256": args.source_state_sha256,
        }
    else:
        data = build_data_fixture(args.prep_dir)
        schedule = build_schedule_fixture(args.prep_dir)
        linear = build_linear_fixture_torch()
        kernel = build_kernel_fixture_torch()
        import torch
        meta = {
            "synthetic": False,
            "generator": "torch autograd (matmul, torch.nn.functional.conv2d) on CPU float32 tensors",
            "generator_version": torch.__version__,
            "seed_rule": REAL_SEED_RULE,
            "source_commit": args.source_commit,
            "source_state_sha256": args.source_state_sha256,
        }

    file_arrays = [
        ("data_fixture.npz", data),
        ("schedule_fixture.npz", schedule),
        ("linear_fixture.npz", linear),
        ("kernel_fixture.npz", kernel),
    ]
    write_fixture_set(args.out_dir, file_arrays, meta)
    print("FIXTURES DONE synthetic=%s" % args.synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
