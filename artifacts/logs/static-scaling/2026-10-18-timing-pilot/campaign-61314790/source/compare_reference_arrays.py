#!/usr/bin/env python3
"""Read-only comparison of two Phase 3 static-scaling reference NPZ files.

This diagnostic applies the declared elementwise acceptance rule

    |a - b| <= atol + rtol * max(|a|, |b|)

with the declared defaults atol=1e-5 and rtol=1e-4, and reports the same
per-leaf relative L2 categories that `check_phase3_flux.jl` uses.  It never
changes the rule and never writes to either input; the optional `--json-out`
summary is the only output file.

`--ref` is the denominator for per-leaf relative L2, so a Flux-vs-PyTorch
comparison should pass the PyTorch reference as `--ref`.  `--bundle` is
required only for the `param_increment.u3` category.

Reproduction:
  python3 scripts/static_scaling/compare_reference_arrays.py \
    --ref phase3_reference_gpu.npz \
    --cand phase3_reference_cpu.npz \
    --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
    --atol 1e-5 --rtol 1e-4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

EPS_FLOOR = 1.0e-12

CATEGORY_RULES = (
    ("eta", lambda name: name.startswith("eta.")),
    ("loss", lambda name: name.startswith("loss.")),
    ("logits", lambda name: name.startswith("logits.")),
    ("activations", lambda name: name.startswith("act.")),
    ("grads.u1", lambda name: name.startswith("grad.u1.")),
    ("grads.u3", lambda name: name.startswith("grad.u3.")),
    ("params.u3", lambda name: name.startswith("param.u3.")),
    ("m.u3", lambda name: name.startswith("m.u3.")),
    ("v.u3", lambda name: name.startswith("v.u3.")),
    ("step", lambda name: name.startswith("step.")),
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def category_names(names, rule) -> list:
    return sorted(name for name in names if rule(name))


def elementwise_stats(ref: np.ndarray, cand: np.ndarray, atol: float, rtol: float) -> dict:
    ref = np.asarray(ref)
    cand = np.asarray(cand)
    if ref.shape != cand.shape or ref.dtype != cand.dtype:
        return {
            "shape_mismatch": ref.shape != cand.shape,
            "dtype_mismatch": ref.dtype != cand.dtype,
            "elements": int(ref.size),
            "max_abs": math.inf,
            "max_rel": math.inf,
            "violations": int(ref.size),
            "max_ratio": math.inf,
            "worst_index": None,
            "nonfinite": int(np.count_nonzero(~np.isfinite(ref)) + np.count_nonzero(~np.isfinite(cand))),
        }
    r = ref.astype(np.float64, copy=False).ravel()
    c = cand.astype(np.float64, copy=False).ravel()
    if r.size == 0:
        return {
            "shape_mismatch": False,
            "dtype_mismatch": False,
            "elements": 0,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "violations": 0,
            "max_ratio": 0.0,
            "worst_index": None,
            "nonfinite": 0,
        }
    nonfinite = ~np.isfinite(r) | ~np.isfinite(c)
    d = np.abs(r - c)
    m = np.maximum(np.abs(r), np.abs(c))
    rel = np.divide(d, m, out=np.zeros_like(d), where=m > 0.0)
    tol = atol + rtol * m
    ratio = np.divide(d, tol, out=np.full_like(d, np.inf), where=tol > 0.0)
    violations = int(np.count_nonzero(d > tol))
    if np.any(nonfinite):
        d = d.copy()
        rel = rel.copy()
        ratio = ratio.copy()
        d[nonfinite] = math.inf
        rel[nonfinite] = math.inf
        ratio[nonfinite] = math.inf
        violations = int(ref.size)
    worst_index = int(np.argmax(ratio)) if ratio.size else None
    return {
        "shape_mismatch": False,
        "dtype_mismatch": False,
        "elements": int(r.size),
        "max_abs": float(np.max(d)),
        "max_rel": float(np.max(rel)),
        "violations": violations,
        "max_ratio": float(np.max(ratio)),
        "worst_index": worst_index,
        "nonfinite": int(np.count_nonzero(nonfinite)),
    }


def l2_stats(ref: np.ndarray, cand: np.ndarray) -> dict:
    ref = np.asarray(ref)
    cand = np.asarray(cand)
    if ref.shape != cand.shape:
        return {"ref_l2": float("nan"), "cand_l2": float("nan"),
                "diff_l2": float("nan"), "rel_l2": None}
    r = ref.astype(np.float64, copy=False).ravel()
    c = cand.astype(np.float64, copy=False).ravel()
    ref_l2 = float(np.sqrt(np.dot(r, r)))
    cand_l2 = float(np.sqrt(np.dot(c, c)))
    diff = r - c
    diff_l2 = float(np.sqrt(np.dot(diff, diff)))
    rel_l2 = diff_l2 / ref_l2 if ref_l2 > EPS_FLOOR else None
    return {"ref_l2": ref_l2, "cand_l2": cand_l2,
            "diff_l2": diff_l2, "rel_l2": rel_l2}


def compare_category(category, pairs, atol, rtol, worst_n, print_leaves):
    total_elements = 0
    max_abs = 0.0
    max_rel = 0.0
    max_ratio = 0.0
    violations = 0
    worst_array = "-"
    leaves = []
    agg_ref = 0.0
    agg_cand = 0.0
    agg_diff_sq = 0.0
    zero_ref_nonzero = 0
    nonzero_ref_zero = 0
    below_floor = 0
    missing = 0
    for name, ref, cand in pairs:
        if ref is None or cand is None:
            missing += 1
            print("COMPARE_LEAF_MISSING category=%s leaf=%s ref_present=%s cand_present=%s"
                  % (category, name, ref is not None, cand is not None))
            continue
        stats = elementwise_stats(ref, cand, atol, rtol)
        l2 = l2_stats(ref, cand)
        total_elements += stats["elements"]
        max_abs = max(max_abs, stats["max_abs"])
        max_rel = max(max_rel, stats["max_rel"])
        violations += stats["violations"]
        if stats["max_ratio"] > max_ratio:
            max_ratio = stats["max_ratio"]
            worst_array = name
        if l2["rel_l2"] is not None:
            agg_ref += l2["ref_l2"] ** 2
            agg_cand += l2["cand_l2"] ** 2
            agg_diff_sq += l2["diff_l2"] ** 2
            leaves.append((name, l2))
            if print_leaves:
                print("COMPARE_LEAF_L2 category=%s leaf=%s ref_l2=%.6e cand_l2=%.6e "
                      "diff_l2=%.6e rel_l2=%.6e"
                      % (category, name, l2["ref_l2"], l2["cand_l2"],
                         l2["diff_l2"], l2["rel_l2"]))
        else:
            below_floor += 1
            if l2["ref_l2"] == 0.0 or l2["cand_l2"] == 0.0:
                if l2["ref_l2"] == 0.0 and l2["cand_l2"] > 0.0:
                    zero_ref_nonzero += 1
                elif l2["ref_l2"] > 0.0 and l2["cand_l2"] == 0.0:
                    nonzero_ref_zero += 1
            if print_leaves:
                print("COMPARE_LEAF_L2 category=%s leaf=%s ref_l2=%.6e cand_l2=%.6e "
                      "diff_l2=%.6e rel_l2=NA"
                      % (category, name, l2["ref_l2"], l2["cand_l2"], l2["diff_l2"]))
        if stats["shape_mismatch"] or stats["dtype_mismatch"]:
            print("COMPARE_LEAF_MISMATCH category=%s leaf=%s shape_mismatch=%s "
                  "dtype_mismatch=%s ref_shape=%s cand_shape=%s"
                  % (category, name, stats["shape_mismatch"], stats["dtype_mismatch"],
                     tuple(ref.shape), tuple(cand.shape)))
    agg_ref_l2 = math.sqrt(agg_ref)
    agg_cand_l2 = math.sqrt(agg_cand)
    agg_diff_l2 = math.sqrt(agg_diff_sq)
    agg_rel = agg_diff_l2 / agg_ref_l2 if agg_ref_l2 > EPS_FLOOR else None
    worst_leaf = "-"
    max_leaf_rel = None
    leaves.sort(key=lambda item: (-item[1]["rel_l2"], item[0]))
    if leaves:
        max_leaf_rel = leaves[0][1]["rel_l2"]
        worst_leaf = leaves[0][0]
    print("COMPARE_CATEGORY category=%s arrays=%d leaves_l2=%d missing=%d elements=%d "
          "max_abs=%.6e max_rel=%.6e violations=%d max_ratio=%.6e worst_array=%s "
          "aggregate_ref_l2=%.6e aggregate_cand_l2=%.6e aggregate_diff_l2=%.6e "
          "aggregate_rel_l2=%s"
          % (category, len(pairs), len(leaves), missing, total_elements,
             max_abs, max_rel, violations, max_ratio, worst_array,
             agg_ref_l2, agg_cand_l2, agg_diff_l2,
             "NA" if agg_rel is None else "%.6e" % agg_rel))
    print("COMPARE_LEAF_L2_SUMMARY category=%s leaves=%d max_rel_l2=%s worst_leaf=%s "
          "zero_ref_nonzero_candidate=%d nonzero_ref_zero_candidate=%d below_floor=%d"
          % (category, len(pairs),
             "NA" if max_leaf_rel is None else "%.6e" % max_leaf_rel,
             worst_leaf, zero_ref_nonzero, nonzero_ref_zero, below_floor))
    for rank, (name, l2) in enumerate(leaves[:worst_n], start=1):
        print("COMPARE_LEAF_L2_WORST category=%s rank=%d leaf=%s rel_l2=%.6e"
              % (category, rank, name, l2["rel_l2"]))
    return {
        "category": category,
        "arrays": len(pairs),
        "missing": missing,
        "elements": total_elements,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "violations": violations,
        "max_ratio": max_ratio,
        "worst_array": worst_array,
        "aggregate_rel_l2": agg_rel,
        "max_leaf_rel_l2": max_leaf_rel,
        "worst_leaf": worst_leaf,
        "zero_ref_nonzero_candidate": zero_ref_nonzero,
        "nonzero_ref_zero_candidate": nonzero_ref_zero,
        "below_floor": below_floor,
        "worst_leaves": [{"rank": rank, "leaf": name, "rel_l2": l2["rel_l2"]}
                         for rank, (name, l2) in enumerate(leaves[:worst_n], start=1)],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two Phase 3 reference NPZ files with the declared elementwise rule")
    parser.add_argument("--ref", required=True, help="reference NPZ (relative-L2 denominator)")
    parser.add_argument("--cand", required=True, help="candidate NPZ")
    parser.add_argument("--bundle", default=None,
                        help="initial_bundle.npz; enables param_increment.u3")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument("--no-leaves", action="store_true",
                        help="print only summaries and worst leaves")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for label, path in (("ref", args.ref), ("cand", args.cand)):
        if not os.path.isfile(path):
            print("COMPARE_ERROR %s not found: %s" % (label, path))
            return 2
    if args.bundle is not None and not os.path.isfile(args.bundle):
        print("COMPARE_ERROR bundle not found: %s" % args.bundle)
        return 2

    ref = load_npz(args.ref)
    cand = load_npz(args.cand)
    ref_hash = sha256_file(args.ref)
    cand_hash = sha256_file(args.cand)
    print("COMPARE_CONFIG rule=|a-b| <= atol + rtol*max(|a|,|b|) atol=%r rtol=%r"
          % (args.atol, args.rtol))
    print("COMPARE_INPUT role=ref path=%s bytes=%d sha256=%s arrays=%d"
          % (args.ref, os.path.getsize(args.ref), ref_hash, len(ref)))
    print("COMPARE_INPUT role=cand path=%s bytes=%d sha256=%s arrays=%d"
          % (args.cand, os.path.getsize(args.cand), cand_hash, len(cand)))

    ref_names = set(ref)
    cand_names = set(cand)
    print("COMPARE_ARRAY_SET ref_arrays=%d cand_arrays=%d missing_in_cand=%s extra_in_cand=%s"
          % (len(ref), len(cand), sorted(ref_names - cand_names),
             sorted(cand_names - ref_names)))

    bundle = load_npz(args.bundle) if args.bundle is not None else None
    categories = []
    total_violations = 0
    total_max_abs = 0.0
    for category, rule in CATEGORY_RULES:
        names = category_names(ref_names & cand_names, rule)
        pairs = [(name, ref.get(name), cand.get(name)) for name in names]
        result = compare_category(category, pairs, args.atol, args.rtol,
                                  args.worst, not args.no_leaves)
        categories.append(result)
        total_violations += result["violations"]
        total_max_abs = max(total_max_abs, result["max_abs"])

    if bundle is not None:
        pair_names = sorted(name for name in ref_names & cand_names
                            if name.startswith("param.u3."))
        pairs = []
        for key in pair_names:
            leaf = key[len("param.u3."):]
            if leaf not in bundle:
                pairs.append(("delta.u3." + leaf, None, None))
                continue
            pairs.append(("delta.u3." + leaf,
                          np.asarray(ref[key]) - np.asarray(bundle[leaf]),
                          np.asarray(cand[key]) - np.asarray(bundle[leaf])))
        result = compare_category("param_increment.u3", pairs, args.atol, args.rtol,
                                  args.worst, not args.no_leaves)
        categories.append(result)
        total_violations += result["violations"]
        total_max_abs = max(total_max_abs, result["max_abs"])
        print("COMPARE_BUNDLE path=%s sha256=%s arrays=%d"
              % (args.bundle, sha256_file(args.bundle), len(bundle)))

    passed = (total_violations == 0
              and ref_names == cand_names)
    print("COMPARE_TOTAL categories=%d violations=%d max_abs=%.6e"
          % (len(categories), total_violations, total_max_abs))
    print("COMPARE_RESULT %s" % ("PASS" if passed else "FAIL"))

    if args.json_out:
        summary = {
            "rule": "abs(a-b) <= atol + rtol * max(abs(a), abs(b))",
            "atol": args.atol,
            "rtol": args.rtol,
            "ref": {"path": args.ref, "sha256": ref_hash, "arrays": len(ref)},
            "cand": {"path": args.cand, "sha256": cand_hash, "arrays": len(cand)},
            "missing_in_cand": sorted(ref_names - cand_names),
            "extra_in_cand": sorted(cand_names - ref_names),
            "categories": categories,
            "total_violations": total_violations,
            "max_abs": total_max_abs,
            "result": "PASS" if passed else "FAIL",
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
