"""
Phase 2 static-scaling shared dataset adapter (numpy only).

Frozen contract implemented here (experiments/static-scaling/specification.md
sections 2, 3, 8; wiki/devlog/03-static-scaling/2026-10-16-od1-od5-od6-recorded.md):

  * Schedule RNG: ss = SeedSequence([ROOT_SCHEDULE, DOMAIN, epoch]); DOMAIN 1
    permutations, DOMAIN 2 augmentation; gen = Generator(PCG64(ss)).
    Permutation: gen.permutation(N_TRAIN). Augmentation: per id ascending three
    scalar draws gen.integers(0, 9), gen.integers(0, 9), gen.integers(0, 2)
    (crop_top, crop_left, flip). Epoch streams are independent: generating more
    epochs never changes an earlier epoch.
  * Epoch ids: the incomplete tail is dropped so that len(ids) is a multiple of
    the global batch (100000 -> 99840 for batch 512 and 256; the same final 160
    saved entries of the permutation are dropped in every configuration).
  * Rank sharding: rank r (0-based) takes contiguous chunk r of every global
    batch; global batches are formed from the saved permutation before sharding
    (semantically identical to replay.jl rank_shard/rank_sequence, which are
    1-based).
  * Training transform: zero-pad 4 pixels per side, 64x64 crop at zero-based
    (top, left), reverse the W axis when flip == 1; then
    u2 = uint8.astype(float32); t = (u2 / float32(255) - mean32) / std32.
    Validation: float32 conversion and normalization only. This mirrors
    replay.jl replay_transform exactly (padding zeros are normalized).
  * Hashes: sha256 over raw numpy C-order bytes, arr.tobytes(order="C").
  * Bundle traversal order: the 62 trainable arrays of the frozen small-image
    ResNet-18 (same order as check_fixtures_data.jl expected_bundle_shapes).

Reproduction:
  python3 scripts/static_scaling/selftest_local.py
  python3 scripts/static_scaling/dataset_adapter.py --prep-dir <PREP> \
      --epoch 0 --batch-size 512 --ranks 4 --emit-npz /tmp/rank_ids.npz

CLI output is deterministic: one DATASET_ADAPTER line per reported fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Frozen contract constants
# ---------------------------------------------------------------------------

ROOT_SCHEDULE = 20261016
PERM_DOMAIN = 1
AUG_DOMAIN = 2
EPOCHS = 50
N_TRAIN = 100000
N_VAL = 10000
N_CLASSES = 200
EPOCH_BATCH = 512
FALLBACK_BATCH = 256
SAMPLES_PER_EPOCH = 99840
TAIL_SIZE = 160

TRAIN_FIXTURE_IDS = (0, 1, 2, 63, 64, 199, 200, 999, 5000, 49999, 99998, 99999)
VAL_FIXTURE_IDS = (0, 1, 2, 5000, 9999, 27, 4108, 7777)

FIXTURE_FILES = (
    "data_fixture.npz",
    "schedule_fixture.npz",
    "linear_fixture.npz",
    "kernel_fixture.npz",
)


# ---------------------------------------------------------------------------
# Time / provenance helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_info(source_commit=None, source_state_sha256=None) -> dict:
    """Source pin recorded in every manifest.

    SOURCE_COMMIT is the THESIS git HEAD; SOURCE_STATE_SHA256 is sha256 of the
    raw `git status --porcelain` output (e3b0c442... for a clean tree). The
    launchers export both; unset values are recorded as "unknown".
    """
    commit = source_commit or os.environ.get("SOURCE_COMMIT") or "unknown"
    state = source_state_sha256 or os.environ.get("SOURCE_STATE_SHA256") or "unknown"
    return {"source_commit": commit, "source_state_sha256": state}


# ---------------------------------------------------------------------------
# Hashing (C-order bytes; the frozen integrity rule)
# ---------------------------------------------------------------------------

def sha256_c_order(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bundle_hash_of_arrays(arrays) -> str:
    """sha256 over the concatenation of per-array raw C-order bytes in order."""
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Frozen model traversal order (62 arrays)
# ---------------------------------------------------------------------------

def bundle_specs():
    """[(name, shape, layout, kind)] in the frozen traversal order.

    layout: "conv_kernel_out_in_kh_kw" for 4-D convolution weights,
    "linear_out_in" for classifier.weight, "vector" for 1-D arrays.
    kind: initialization family used by make_initial_bundle.py.
    """
    spec = [
        ("stem.conv.weight", (64, 3, 3, 3), "conv_kernel_out_in_kh_kw", "kaiming_normal"),
        ("stem.gn.weight", (64,), "vector", "ones"),
        ("stem.gn.bias", (64,), "vector", "zeros"),
    ]
    widths = (64, 128, 256, 512)
    prev = 64
    for s in (1, 2, 3, 4):
        ch = widths[s - 1]
        for b in (0, 1):
            cin = prev if b == 0 else ch
            pre = "stage%d.block%d" % (s, b)
            spec.append((pre + ".conv1.weight", (ch, cin, 3, 3), "conv_kernel_out_in_kh_kw", "kaiming_normal"))
            spec.append((pre + ".conv2.weight", (ch, ch, 3, 3), "conv_kernel_out_in_kh_kw", "kaiming_normal"))
            spec.append((pre + ".gn1.weight", (ch,), "vector", "ones"))
            spec.append((pre + ".gn1.bias", (ch,), "vector", "zeros"))
            spec.append((pre + ".gn2.weight", (ch,), "vector", "ones"))
            spec.append((pre + ".gn2.bias", (ch,), "vector", "zeros"))
        if s >= 2:
            spec.append(("stage%d.block0.proj.weight" % s, (ch, ch // 2, 1, 1), "conv_kernel_out_in_kh_kw", "kaiming_normal"))
            spec.append(("stage%d.block0.proj_gn.weight" % s, (ch,), "vector", "ones"))
            spec.append(("stage%d.block0.proj_gn.bias" % s, (ch,), "vector", "zeros"))
        prev = ch
    spec.append(("classifier.weight", (200, 512), "linear_out_in", "kaiming_uniform"))
    spec.append(("classifier.bias", (200,), "vector", "uniform_bias"))
    return spec


def bundle_total_elements() -> int:
    return sum(int(np.prod(shape)) for _, shape, _, _ in bundle_specs())


# ---------------------------------------------------------------------------
# Schedule RNG (frozen scheme)
# ---------------------------------------------------------------------------

def schedule_generator(epoch: int, domain: int) -> np.random.Generator:
    if epoch < 0:
        raise ValueError("epoch must be >= 0, got %d" % epoch)
    if domain not in (PERM_DOMAIN, AUG_DOMAIN):
        raise ValueError("domain must be 1 (permutation) or 2 (augmentation), got %r" % domain)
    ss = np.random.SeedSequence([ROOT_SCHEDULE, domain, epoch])
    return np.random.Generator(np.random.PCG64(ss))


def epoch_permutation(epoch: int, n: int = N_TRAIN) -> np.ndarray:
    return schedule_generator(epoch, PERM_DOMAIN).permutation(n).astype(np.int64, copy=False)


def epoch_augmentation(epoch: int, n: int = N_TRAIN) -> np.ndarray:
    """(n, 3) uint8 (crop_top, crop_left, flip); per id ascending scalar draws."""
    g = schedule_generator(epoch, AUG_DOMAIN)
    out = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        out[i, 0] = g.integers(0, 9)
        out[i, 1] = g.integers(0, 9)
        out[i, 2] = g.integers(0, 2)
    return out


# ---------------------------------------------------------------------------
# Epoch ids and rank sharding (mirror replay.jl)
# ---------------------------------------------------------------------------

def epoch_sample_ids(perm, batch: int) -> np.ndarray:
    if batch <= 0:
        raise ValueError("batch must be positive, got %d" % batch)
    n_eff = (len(perm) // batch) * batch
    return perm[:n_eff]


def rank_shard(batch_ids, rank: int, world: int) -> np.ndarray:
    if world < 1:
        raise ValueError("world must be >= 1, got %d" % world)
    if not 0 <= rank < world:
        raise ValueError("rank %d outside 0..%d" % (rank, world - 1))
    if len(batch_ids) % world != 0:
        raise ValueError("batch length %d not divisible by world %d" % (len(batch_ids), world))
    chunk = len(batch_ids) // world
    return batch_ids[rank * chunk:(rank + 1) * chunk]


def rank_sequence(perm, batch: int, rank: int, world: int) -> np.ndarray:
    """Rank shard of every global batch, concatenated (batch 512 gives 99840/world)."""
    ids = epoch_sample_ids(perm, batch)
    if batch % world != 0:
        raise ValueError("batch %d not divisible by world %d" % (batch, world))
    n_batches = len(ids) // batch
    chunk = batch // world
    out = np.empty(n_batches * chunk, dtype=ids.dtype)
    pos = 0
    for k in range(n_batches):
        gb = ids[k * batch:(k + 1) * batch]
        out[pos:pos + chunk] = gb[rank * chunk:(rank + 1) * chunk]
        pos += chunk
    return out


def iter_epoch_batches(perm, batch: int):
    """Yield global batch id arrays from the permutation BEFORE sharding."""
    ids = epoch_sample_ids(perm, batch)
    for start in range(0, len(ids), batch):
        yield ids[start:start + batch]


def iter_rank_batches(perm, batch: int, rank: int, world: int):
    """Yield this rank's contiguous shard of every global batch."""
    for gb in iter_epoch_batches(perm, batch):
        yield rank_shard(gb, rank, world)


def iter_validation_batches(images, labels, batch: int, rank: int = 0, world: int = 1):
    """Yield (uint8 NCHW batch, int64 labels) for validation; no augmentation.

    The global validation order is 0..N_VAL-1; every batch is split into equal
    contiguous rank shards exactly like training batches.
    """
    n = images.shape[0]
    if n % batch != 0:
        raise ValueError("validation count %d not divisible by batch %d" % (n, batch))
    for start in range(0, n, batch):
        xy = np.asarray(images[start:start + batch])
        yy = np.asarray(labels[start:start + batch])
        yield rank_shard(xy, rank, world), rank_shard(yy, rank, world)


# ---------------------------------------------------------------------------
# Frozen transforms (mirror replay.jl)
# ---------------------------------------------------------------------------

def _check_triples(triples, n):
    t = np.asarray(triples)
    if t.shape != (n, 3):
        raise ValueError("triples shape %s, want (%d, 3)" % (t.shape, n))
    if not (np.all(t[:, 0] <= 8) and np.all(t[:, 1] <= 8) and np.all(t[:, 2] <= 1)):
        raise ValueError("triples out of range: top/left in 0..8, flip in 0..1")
    return t


def replay_transform(nchw_uint8_batch, triples, mean32, std32) -> np.ndarray:
    """Training transform: pad4, crop (top,left), W-flip, float32 normalize.

    nchw_uint8_batch: (N, 3, 64, 64) uint8; triples: (N, 3) uint8;
    mean32/std32: (3,) float32 or convertible. Returns (N, 3, 64, 64) float32
    with the exact operation order u2 = x.astype(float32);
    t = (u2 / float32(255) - mean32) / std32.
    """
    x = np.asarray(nchw_uint8_batch)
    if x.dtype != np.uint8:
        raise ValueError("expected uint8 images, got %s" % x.dtype)
    if x.ndim != 4 or x.shape[1:] != (3, 64, 64):
        raise ValueError("expected (N, 3, 64, 64), got %s" % (x.shape,))
    n, c, h, w = x.shape
    t = _check_triples(triples, n)
    mean32 = np.asarray(mean32, dtype=np.float32).reshape(1, c, 1, 1)
    std32 = np.asarray(std32, dtype=np.float32).reshape(1, c, 1, 1)
    padded = np.zeros((n, c, h + 8, w + 8), dtype=np.uint8)
    padded[:, :, 4:4 + h, 4:4 + w] = x
    out = np.empty((n, c, h, w), dtype=np.uint8)
    for i in range(n):
        top = int(t[i, 0])
        left = int(t[i, 1])
        flip = int(t[i, 2])
        window = padded[i, :, top:top + h, left:left + w]
        if flip == 1:
            window = window[:, :, ::-1]
        out[i] = window
    u2 = out.astype(np.float32)
    return (u2 / np.float32(255.0) - mean32) / std32


def replay_transform_validation(nchw_uint8_batch, mean32, std32) -> np.ndarray:
    """Validation transform: float32 /255 then normalize; no pad, crop, flip."""
    x = np.asarray(nchw_uint8_batch)
    if x.dtype != np.uint8:
        raise ValueError("expected uint8 images, got %s" % x.dtype)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("expected (N, 3, H, W), got %s" % (x.shape,))
    c = x.shape[1]
    mean32 = np.asarray(mean32, dtype=np.float32).reshape(1, c, 1, 1)
    std32 = np.asarray(std32, dtype=np.float32).reshape(1, c, 1, 1)
    u2 = x.astype(np.float32)
    return (u2 / np.float32(255.0) - mean32) / std32


# ---------------------------------------------------------------------------
# Prepared-dataset access
# ---------------------------------------------------------------------------

def load_stats(prep_dir: str):
    """Return (mean32 (3,), std32 (3,), stats dict) from <prep_dir>/stats.json."""
    stats = read_json(os.path.join(prep_dir, "stats.json"))
    mean32 = np.asarray(stats["mean"], dtype=np.float32)
    std32 = np.asarray(stats["std"], dtype=np.float32)
    if mean32.shape != (3,) or std32.shape != (3,):
        raise ValueError("stats.json mean/std must each have 3 entries")
    return mean32, std32, stats


def open_prepared_train_images(prep_dir: str):
    """Memory-mapped uint8 NCHW train images; never load 1.2 GB eagerly."""
    return np.load(os.path.join(prep_dir, "train_images.npy"), mmap_mode="r")


def open_prepared_val_images(prep_dir: str):
    return np.load(os.path.join(prep_dir, "val_images.npy"), mmap_mode="r")


def open_prepared_permutations(prep_dir: str):
    return np.load(os.path.join(prep_dir, "permutations.npy"), mmap_mode="r")


def open_prepared_augmentation(prep_dir: str):
    return np.load(os.path.join(prep_dir, "augmentation.npy"), mmap_mode="r")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 2 schedule/sharding adapter for the static-scaling prepared dataset")
    p.add_argument("--prep-dir", required=True, help="prepared dataset directory (permutations.npy, augmentation.npy)")
    p.add_argument("--epoch", type=int, default=0, help="epoch index (0-based)")
    p.add_argument("--batch-size", type=int, default=EPOCH_BATCH)
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--emit-npz", default="", help="optional output .npz with permutation/augmentation/ids/rank ids")
    args = p.parse_args(argv)

    if args.ranks < 1:
        raise SystemExit("--ranks must be >= 1")
    if args.batch_size % args.ranks != 0:
        raise SystemExit("--batch-size must be divisible by --ranks")

    perm = np.load(os.path.join(args.prep_dir, "permutations.npy"), mmap_mode="r")[args.epoch]
    aug = np.load(os.path.join(args.prep_dir, "augmentation.npy"), mmap_mode="r")[args.epoch]
    ids = epoch_sample_ids(perm, args.batch_size)
    updates = len(ids) // args.batch_size
    print("DATASET_ADAPTER epoch=%d batch=%d ranks=%d samples_per_epoch=%d updates_per_epoch=%d"
          % (args.epoch, args.batch_size, args.ranks, len(ids), updates))
    rank_ids = []
    for r in range(args.ranks):
        seq = rank_sequence(perm, args.batch_size, r, args.ranks)
        rank_ids.append(seq)
        print("DATASET_ADAPTER rank=%d shard_size=%d first=%d last=%d"
              % (r, len(seq), int(seq[0]), int(seq[-1])))
    if args.emit_npz:
        payload = {
            "permutation": np.asarray(perm, dtype=np.int64),
            "augmentation": np.asarray(aug, dtype=np.uint8),
            "ids": np.asarray(ids, dtype=np.int64),
        }
        for r, seq in enumerate(rank_ids):
            payload["rank_ids_R%d" % (r + 1)] = seq
        np.savez(args.emit_npz, **payload)
        print("DATASET_ADAPTER wrote %s arrays=%d" % (args.emit_npz, len(payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
