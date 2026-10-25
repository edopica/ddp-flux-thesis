"""
Phase 2 local self-test (numpy only; no torch, no PIL, no PREP needed).

Exercises the frozen contract implemented in dataset_adapter.py:
  * SeedSequence([20261016, DOMAIN, epoch]) scheme, including the property that
    generating more epochs does not change earlier epochs;
  * rank sharding and reconstruction at R = 1, 2, 4 (0-based ranks);
  * the training transform against a naive independent pixel-loop reference,
    including flip and corner crops, and the validation transform;
  * tail arithmetic 100000 -> 99840 with 160 dropped, 195/390 updates;
  * manifest hash roundtrip: npz -> per-array C-order sha256 -> bundle_hash
    (concatenation rule), verified by re-reading the archive.

Prints one `SELFTEST <name> ok` line per group and `SELFTEST PASS`; exits
nonzero on the first failure.

Reproduction:
  python3 scripts/static_scaling/selftest_local.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

import dataset_adapter as da


def test_schedule_scheme():
    n = 997
    perm0 = da.epoch_permutation(0, n)
    assert perm0.dtype == np.int64 and perm0.shape == (n,)
    assert np.array_equal(np.sort(perm0), np.arange(n))
    aug0 = da.epoch_augmentation(0, n)
    assert aug0.shape == (n, 3) and aug0.dtype == np.uint8
    assert aug0[:, 0].max() <= 8 and aug0[:, 1].max() <= 8 and aug0[:, 2].max() <= 1

    # More epochs must not change earlier rows (fresh generators, later epochs
    # drawn first).
    for e in (3, 1, 5):
        da.epoch_permutation(e, n)
        da.epoch_augmentation(e, n)
    assert np.array_equal(da.epoch_permutation(0, n), perm0)
    assert np.array_equal(da.epoch_augmentation(0, n), aug0)
    print("SELFTEST schedule_scheme ok epochs=0..5 n=%d" % n)


def test_shards():
    for n_perm, batch in ((96, 12), (480, 40)):
        perm = np.random.default_rng(7).permutation(n_perm)
        ids = da.epoch_sample_ids(perm, batch)
        assert len(ids) % batch == 0
        for world in (1, 2, 4):
            shards = [da.rank_sequence(perm, batch, r, world) for r in range(world)]
            per = batch // world
            recon = np.empty(len(ids), dtype=perm.dtype)
            k = 0
            for start in range(0, len(ids), batch):
                for r in range(world):
                    recon[start + r * per:start + (r + 1) * per] = shards[r][k:k + per]
                k += per
            assert np.array_equal(recon, ids), "reconstruction failed at world=%d" % world
            for r in range(world):
                assert len(shards[r]) == len(ids) // world
                for start in range(0, len(ids), batch):
                    ref = ids[start:start + batch][r * per:(r + 1) * per]
                    got = shards[r][(start // batch) * per:(start // batch + 1) * per]
                    assert np.array_equal(ref, got)
    # error paths
    try:
        da.rank_shard(np.arange(10), 0, 3)
        raise AssertionError("rank_shard accepted a non-divisible world")
    except ValueError:
        pass
    try:
        da.epoch_sample_ids(np.arange(10), 0)
        raise AssertionError("epoch_sample_ids accepted batch 0")
    except ValueError:
        pass
    print("SELFTEST shards ok reconstruction R=1,2,4")


def naive_transform(x, triples, mean32, std32):
    """Independent pixel-loop reference (same semantics as replay.jl)."""
    n, c, h, w = x.shape
    out = np.zeros((n, c, h, w), dtype=np.float32)
    for i in range(n):
        top = int(triples[i, 0])
        left = int(triples[i, 1])
        flip = int(triples[i, 2])
        for cc in range(c):
            m = np.float32(mean32[cc])
            s = np.float32(std32[cc])
            for hh in range(h):
                src_h = hh + top - 4
                for ww in range(w):
                    dst_w = (w - 1 - ww) if flip == 1 else ww
                    src_w = dst_w + left - 4
                    if 0 <= src_h < h and 0 <= src_w < w:
                        u = np.float32(x[i, cc, src_h, src_w])
                    else:
                        u = np.float32(0.0)
                    out[i, cc, hh, ww] = (u / np.float32(255.0) - m) / s
    return out


def test_transform():
    rng = np.random.default_rng(11)
    x = rng.integers(0, 256, size=(5, 3, 64, 64), dtype=np.uint8)
    triples = np.asarray([
        [0, 0, 0], [8, 8, 0], [0, 8, 1], [8, 0, 1], [4, 4, 0],
    ], dtype=np.uint8)
    mean32 = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)
    std32 = np.asarray([0.2, 0.25, 0.3], dtype=np.float32)
    got = da.replay_transform(x, triples, mean32, std32)
    want = naive_transform(x, triples, mean32, std32)
    assert got.dtype == np.float32 and got.shape == (5, 3, 64, 64)
    assert np.array_equal(got, want), "transform differs from the naive reference"
    # explicit flip identity: crop then reverse W
    no_flip = da.replay_transform(x[:1], np.asarray([[3, 7, 0]], dtype=np.uint8), mean32, std32)
    flipped = da.replay_transform(x[:1], np.asarray([[3, 7, 1]], dtype=np.uint8), mean32, std32)
    assert np.array_equal(flipped, no_flip[:, :, :, ::-1])
    # validation transform == identity crop under padding 4 (mirrors the Julia check)
    val = da.replay_transform_validation(x[:2], mean32, std32)
    identity = da.replay_transform(x[:2], np.asarray([[4, 4, 0], [4, 4, 0]], dtype=np.uint8), mean32, std32)
    assert np.array_equal(val, identity)
    # error paths
    try:
        da.replay_transform(x, np.asarray([[9, 0, 0]] * 5, dtype=np.uint8), mean32, std32)
        raise AssertionError("replay_transform accepted crop_top=9")
    except ValueError:
        pass
    try:
        da.replay_transform(x, np.asarray([[0, 0]], dtype=np.uint8), mean32, std32)
        raise AssertionError("replay_transform accepted a wrong triples shape")
    except ValueError:
        pass
    print("SELFTEST transform ok naive_reference_bitwise=true flips_and_corners=true")


def test_tail_arithmetic():
    perm = np.arange(da.N_TRAIN, dtype=np.int64)
    ids512 = da.epoch_sample_ids(perm, da.EPOCH_BATCH)
    ids256 = da.epoch_sample_ids(perm, da.FALLBACK_BATCH)
    assert len(ids512) == 99840 and len(ids256) == 99840
    assert da.N_TRAIN - len(ids512) == da.TAIL_SIZE == 160
    assert len(ids512) // da.EPOCH_BATCH == 195
    assert len(ids256) // da.FALLBACK_BATCH == 390
    assert np.array_equal(ids512, ids256)
    assert len(set(perm[99840:].tolist())) == 160
    print("SELFTEST tail_arithmetic ok 100000->99840 dropped=160 updates=195/390")


def test_manifest_roundtrip():
    tmp = tempfile.mkdtemp(prefix="phase2-selftest-")
    arrays = {
        "u": np.arange(6, dtype=np.uint8).reshape(2, 3),
        "l": np.asarray([0, 5, 9], dtype=np.int64),
        "f": np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(2, 3),
    }
    npz_path = os.path.join(tmp, "tiny_fixture.npz")
    np.savez(npz_path, **arrays)
    entries = [{"name": "tiny_fixture.npz::" + k, "dtype": str(v.dtype),
                "shape": [int(x) for x in v.shape], "sha256": da.sha256_c_order(v)}
               for k, v in arrays.items()]
    manifest = {"files": [{"path": "tiny_fixture.npz", "sha256": da.sha256_file(npz_path)}],
                "arrays": entries}
    manifest_path = os.path.join(tmp, "fixture_manifest.json")
    da.write_json(manifest_path, manifest)
    loaded = da.read_json(manifest_path)
    with np.load(npz_path) as z:
        for e in loaded["arrays"]:
            name = e["name"].split("::", 1)[1]
            assert str(z[name].dtype) == e["dtype"]
            assert [int(x) for x in z[name].shape] == e["shape"]
            assert da.sha256_c_order(z[name]) == e["sha256"]
    # 62-array bundle-hash roundtrip over the real spec shapes
    specs = da.bundle_specs()
    assert len(specs) == 62 and da.bundle_total_elements() == 11271432
    payload = {}
    for name, shape, _layout, kind in specs:
        if kind == "ones":
            payload[name] = np.ones(shape, dtype=np.float32)
        elif kind == "zeros":
            payload[name] = np.zeros(shape, dtype=np.float32)
        else:
            payload[name] = np.zeros(shape, dtype=np.float32)
    bundle_path = os.path.join(tmp, "bundle.npz")
    np.savez(bundle_path, **payload)
    order = [s[0] for s in specs]
    h_in_memory = da.bundle_hash_of_arrays([payload[n] for n in order])
    with np.load(bundle_path) as z:
        h_reread = da.bundle_hash_of_arrays([z[n] for n in order])
    assert h_in_memory == h_reread, "bundle_hash is not stable across an npz roundtrip"
    print("SELFTEST manifest_roundtrip ok arrays=3 bundle_arrays=62 bundle_hash=%s" % h_reread[:16])


def main() -> int:
    try:
        test_schedule_scheme()
        test_shards()
        test_transform()
        test_tail_arithmetic()
        test_manifest_roundtrip()
    except Exception:
        traceback.print_exc()
        print("SELFTEST FAIL")
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
