"""
Phase 2 static-scaling canonical initial-parameter bundle generator (torch).

Writes:
  initial_bundle.npz           62 float32 arrays, uncompressed np.savez
  initial_bundle.manifest.json JSON object with the frozen fields below

Frozen recipe (OD-1, 2026-10-16; specification.md 4.1):
  * convolution weights (20 arrays, incl. 1x1 projections):
      torch.nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu",
                                    generator=g) on a CPU float32 tensor
  * GroupNorm gamma = 1, beta = 0 (40 arrays)
  * classifier.weight: torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=g)
  * classifier.bias:   bound = 1/sqrt(512); uniform_(b, -bound, bound, generator=g)
  * per array i (0-based index in the frozen traversal list):
      g = torch.Generator(device="cpu").manual_seed(20261016 + i)
    GroupNorm constants still occupy their index and consume no draw.
  * total: exactly 11,271,432 float32 elements across 62 arrays.

Manifest schema (JSON object):
  producer, created, source_commit, source_state_sha256, generator,
  generator_version, root_seed, per_array_seed_rule, recipe, synthetic,
  layout_note,
  arrays:       LIST of exactly 62 entries in traversal order:
                {"name", "dtype": "float32", "shape": [ints],
                 "layout", "sha256"}
  bundle_hash:  sha256 over the concatenation of per-array raw C-order bytes,
                in manifest/traversal order (see bundle_hash_definition)
  bundle_hash_definition: "sha256 over concat of per-array raw C-order bytes in
                manifest order"

The file is verified after writing by re-reading the .npz and recomputing every
per-array sha256 and the bundle hash.

Reproduction (Leonardo compute node; pinned venv with torch):
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/make_initial_bundle.py \\
    --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared

Deterministic stdout lines: BUNDLE <key>=<value>.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import dataset_adapter as da

BUNDLE_NAME = "initial_bundle.npz"
MANIFEST_NAME = "initial_bundle.manifest.json"
RECIPE_TEXT = (
    "conv/proj weights: kaiming_normal_(mode='fan_out', nonlinearity='relu'); "
    "GN gamma=1, beta=0; classifier.weight: kaiming_uniform_(a=sqrt(5)); "
    "classifier.bias: uniform_(-1/sqrt(512), 1/sqrt(512)); all on CPU float32 "
    "tensors with per-array torch Generator seed 20261016+i"
)
SEED_RULE_TEXT = (
    "i = 0-based index of the array in the traversal list; for every array "
    "g = torch.Generator(device='cpu').manual_seed(20261016 + i); GroupNorm "
    "constants consume their index and draw nothing"
)


def build_arrays(torch):
    """Return (names, arrays) per the frozen recipe; asserts element total."""
    keys = ["kaiming_normal", "ones", "zeros", "kaiming_uniform", "uniform_bias"]
    names = []
    arrays = []
    for i, (name, shape, _layout, kind) in enumerate(da.bundle_specs()):
        if kind not in keys:
            raise SystemExit("unknown initialization kind %r" % kind)
        g = torch.Generator(device="cpu").manual_seed(da.ROOT_SCHEDULE + i)
        if kind == "ones":
            t = torch.ones(shape, dtype=torch.float32)
        elif kind == "zeros":
            t = torch.zeros(shape, dtype=torch.float32)
        elif kind == "kaiming_normal":
            t = torch.empty(shape, dtype=torch.float32)
            torch.nn.init.kaiming_normal_(t, mode="fan_out", nonlinearity="relu", generator=g)
        elif kind == "kaiming_uniform":
            t = torch.empty(shape, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5), generator=g)
        else:  # uniform_bias
            bound = 1.0 / math.sqrt(512.0)
            t = torch.empty(shape, dtype=torch.float32)
            torch.nn.init.uniform_(t, -bound, bound, generator=g)
        names.append(name)
        arrays.append(np.ascontiguousarray(t.detach().numpy()))
    total = sum(int(a.size) for a in arrays)
    if len(arrays) != 62 or total != 11271432:
        raise SystemExit("bundle shape drift: arrays=%d elements=%d" % (len(arrays), total))
    return names, arrays


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the canonical initial-parameter bundle with torch initializers")
    p.add_argument("--out-dir", required=True, help="prepared dataset directory ($PREP)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--source-state-sha256", default=None)
    args = p.parse_args(argv)

    import torch  # lazy: keeps pure-numpy modules importable locally

    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, BUNDLE_NAME)
    manifest_path = os.path.join(args.out_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        raise SystemExit("refusing to overwrite %s (use --force)" % manifest_path)

    torch.manual_seed(0)  # irrelevant to the recipe (all draws use explicit generators)
    names, arrays = build_arrays(torch)
    payload = {n: a for n, a in zip(names, arrays)}
    np.savez(npz_path, **payload)
    print("BUNDLE wrote %s arrays=%d" % (npz_path, len(names)), flush=True)

    specs = da.bundle_specs()
    entries = []
    for (name, shape, layout, _kind), arr in zip(specs, arrays):
        if name != name or tuple(arr.shape) != tuple(shape):
            raise SystemExit("internal error: array %s shape mismatch" % name)
        entries.append({
            "name": name,
            "dtype": "float32",
            "shape": [int(x) for x in shape],
            "layout": layout,
            "sha256": da.sha256_c_order(arr),
        })
    bundle_hash = da.bundle_hash_of_arrays([payload[e["name"]] for e in entries])

    manifest = {
        "producer": "scripts/static_scaling/make_initial_bundle.py",
        "created": da.utc_now(),
        "synthetic": False,
        **da.source_info(args.source_commit, args.source_state_sha256),
        "generator": "torch.nn.init on CPU float32 tensors",
        "generator_version": torch.__version__,
        "root_seed": da.ROOT_SCHEDULE,
        "per_array_seed_rule": SEED_RULE_TEXT,
        "recipe": RECIPE_TEXT,
        "layout_note": (
            "layout values: conv_kernel_out_in_kh_kw for 4-D convolution weights, "
            "linear_out_in for classifier.weight, vector for 1-D arrays; all arrays C-order"
        ),
        "arrays": entries,
        "bundle_hash": bundle_hash,
        "bundle_hash_definition": "sha256 over concat of per-array raw C-order bytes in manifest order",
    }
    da.write_json(manifest_path, manifest)
    print("BUNDLE elements=%d bundle_hash=%s" % (sum(int(a.size) for a in arrays), bundle_hash), flush=True)

    # Verify by re-reading the archive: bitwise equality, per-array sha256,
    # bundle hash from the stored bytes.
    with np.load(npz_path) as z:
        if sorted(z.files) != sorted(names):
            raise SystemExit("npz name set mismatch after write: %d vs %d" % (len(z.files), len(names)))
        for entry in entries:
            arr = z[entry["name"]]
            if arr.dtype != np.float32 or list(arr.shape) != entry["shape"]:
                raise SystemExit("npz %s dtype/shape mismatch" % entry["name"])
            if da.sha256_c_order(arr) != entry["sha256"]:
                raise SystemExit("npz %s sha256 mismatch after write" % entry["name"])
            if not np.array_equal(arr, payload[entry["name"]]):
                raise SystemExit("npz %s is not bitwise equal to the in-memory array" % entry["name"])
        reread_hash = da.bundle_hash_of_arrays([z[e["name"]] for e in entries])
    if reread_hash != bundle_hash:
        raise SystemExit("bundle_hash mismatch after re-read: %s vs %s" % (reread_hash, bundle_hash))

    print("BUNDLE verified name_set=true per_array_sha256=true bundle_hash=true torch=%s numpy=%s"
          % (torch.__version__, np.__version__))
    print("BUNDLE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
