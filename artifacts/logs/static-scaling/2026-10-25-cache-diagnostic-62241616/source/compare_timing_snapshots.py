#!/usr/bin/env python3
"""Describe 300-update state differences without an equivalence policy.

This reader consumes the ``numerics`` mode of ``run-static-timing.sh``:

  snapshots/<framework>-w<world>/state.npz
  <framework>-w<world>-r1.json

It checks structural completeness, array shapes, and finiteness. It reports
within-framework and cross-framework differences. It does not apply a
numerical-equivalence acceptance rule.

Reproduction::

  python3 scripts/static_scaling/compare_timing_snapshots.py \\
    artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-<job>
  python3 scripts/static_scaling/compare_timing_snapshots.py --self-test
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


FRAMEWORKS = ("flux", "pytorch")
WORLDS = (1, 2, 4)
CATEGORIES = ("param", "m", "v", "beta_power")
EXPECTED_UPDATES = 300


def _fail(message):
    raise AssertionError(message)


def _load_state(root, framework, world):
    snapshot = Path(root) / "snapshots" / f"{framework}-w{world}" / "state.npz"
    metadata = Path(root) / f"{framework}-w{world}-r1.json"
    if not snapshot.is_file():
        _fail(f"missing snapshot: {snapshot}")
    if not metadata.is_file():
        _fail(f"missing metadata: {metadata}")
    with np.load(snapshot) as archive:
        arrays = {name: archive[name] for name in archive.files}
    record = json.loads(metadata.read_text())
    if record.get("framework") != framework or record.get("world_size") != world:
        _fail(f"metadata identity mismatch: {metadata}")
    if record.get("updates") != EXPECTED_UPDATES:
        _fail(f"unexpected update count: {metadata}")
    if "step" in arrays:
        step = int(np.asarray(arrays["step"]).reshape(-1)[0])
        if step != EXPECTED_UPDATES:
            _fail(f"optimizer step mismatch: {snapshot}")
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number):
            _fail(f"non-numeric array {name}: {snapshot}")
        if name != "step" and not np.all(np.isfinite(array)):
            _fail(f"non-finite array {name}: {snapshot}")
    return arrays


def _category(name):
    return name.split(".", 1)[0]


def _compare(reference, candidate):
    if set(reference) != set(candidate):
        _fail(f"array sets differ: missing={sorted(set(reference) - set(candidate))[:5]} extra={sorted(set(candidate) - set(reference))[:5]}")
    rows = []
    all_diff = []
    all_ref = []
    for category in CATEGORIES:
        keys = sorted(
            name for name in reference
            if name in candidate and _category(name) == category
        )
        if not keys:
            continue
        max_abs = 0.0
        exact = 0
        diff_chunks = []
        ref_chunks = []
        worst_leaf = ""
        worst_rel = -1.0
        for name in keys:
            ref = np.asarray(reference[name], dtype=np.float64)
            got = np.asarray(candidate[name], dtype=np.float64)
            if ref.shape != got.shape:
                _fail(f"shape mismatch for {name}: {ref.shape} != {got.shape}")
            diff = got - ref
            leaf_max = float(np.max(np.abs(diff))) if diff.size else 0.0
            denom = float(np.linalg.norm(ref))
            leaf_rel = float(np.linalg.norm(diff) / denom) if denom > 0 else (0.0 if leaf_max == 0 else float("inf"))
            if leaf_rel > worst_rel or (leaf_rel == worst_rel and leaf_max > max_abs):
                worst_rel = leaf_rel
                worst_leaf = name
            max_abs = max(max_abs, leaf_max)
            exact += int(np.array_equal(ref, got))
            diff_chunks.append(diff.reshape(-1))
            ref_chunks.append(ref.reshape(-1))
        diff_flat = np.concatenate(diff_chunks)
        ref_flat = np.concatenate(ref_chunks)
        denom = float(np.linalg.norm(ref_flat))
        rel_l2 = float(np.linalg.norm(diff_flat) / denom) if denom > 0 else (0.0 if max_abs == 0 else float("inf"))
        rows.append({
            "category": category,
            "arrays": len(keys),
            "exact_arrays": exact,
            "max_abs": max_abs,
            "rel_l2": rel_l2,
            "worst_leaf": worst_leaf,
            "worst_leaf_rel_l2": worst_rel,
        })
        all_diff.append(diff_flat)
        all_ref.append(ref_flat)
    if not rows:
        _fail("no comparable arrays")
    diff_flat = np.concatenate(all_diff)
    ref_flat = np.concatenate(all_ref)
    denom = float(np.linalg.norm(ref_flat))
    return {
        "categories": rows,
        "max_abs": max(float(row["max_abs"]) for row in rows),
        "rel_l2": float(np.linalg.norm(diff_flat) / denom) if denom > 0 else 0.0,
        "exact": all(row["exact_arrays"] == row["arrays"] for row in rows),
    }


def _comparison(label, reference, candidate):
    return {"label": label, **_compare(reference, candidate)}


def analyse(root, out_path=None):
    root = Path(root)
    states = {(framework, world): _load_state(root, framework, world) for framework in FRAMEWORKS for world in WORLDS}
    flux_categories = {_category(name) for name in states["flux", 1] if name != "step"}
    torch_categories = {_category(name) for name in states["pytorch", 1] if name != "step"}
    if flux_categories != {"param", "m", "v", "beta_power"}:
        _fail(f"unexpected Flux categories: {sorted(flux_categories)}")
    if torch_categories != {"param", "m", "v"}:
        _fail(f"unexpected PyTorch categories: {sorted(torch_categories)}")
    flux_keys = {name for name in states["flux", 1] if _category(name) in {"param", "m", "v"}}
    torch_keys = {name for name in states["pytorch", 1] if name != "step"}
    if flux_keys != torch_keys:
        _fail(f"cross-framework array sets differ: flux_only={sorted(flux_keys - torch_keys)[:5]} torch_only={sorted(torch_keys - flux_keys)[:5]}")
    comparisons = []
    for framework in FRAMEWORKS:
        for world in (2, 4):
            comparisons.append(_comparison(f"{framework}-w{world}-vs-w1", states[framework, 1], states[framework, world]))
    for world in WORLDS:
        flux_cross = {name: value for name, value in states["flux", world].items() if _category(name) in {"param", "m", "v"}}
        torch_cross = {name: value for name, value in states["pytorch", world].items() if name != "step"}
        comparisons.append(_comparison(f"flux-vs-pytorch-w{world}", torch_cross, flux_cross))
    report = {
        "schema": "static-timing-numerics-report-v1",
        "claim": "descriptive only; no numerical-equivalence acceptance rule applied",
        "root": str(root),
        "updates": EXPECTED_UPDATES,
        "records": [
            {"framework": framework, "world_size": world, "arrays": len(states[framework, world])}
            for framework in FRAMEWORKS for world in WORLDS
        ],
        "comparisons": comparisons,
    }
    if out_path is None:
        out_path = root / "numerics-report.json"
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print_report(report):
    print("TIMING_NUMERICS updates=300 policy=descriptive")
    print("comparison category arrays exact max_abs rel_l2 worst_leaf")
    for comparison in report["comparisons"]:
        for row in comparison["categories"]:
            print(
                comparison["label"], row["category"], row["arrays"], row["exact_arrays"],
                f"{row['max_abs']:.9g} {row['rel_l2']:.9g} {row['worst_leaf']}",
            )
    print(f"TIMING_NUMERICS_REPORT COMPLETE report={report['root']}/numerics-report.json")
    print("NUMERICAL_EQUIVALENCE_CLAIM none")


def _write_synthetic(root):
    snapshots = Path(root) / "snapshots"
    names = ("param.a", "param.b", "m.a", "m.b", "v.a", "v.b")
    for framework in FRAMEWORKS:
        for world in WORLDS:
            target = snapshots / f"{framework}-w{world}"
            target.mkdir(parents=True)
            arrays = {name: np.arange(6, dtype=np.float32).reshape(2, 3) + idx for idx, name in enumerate(names)}
            if framework == "flux":
                arrays["beta_power.a"] = np.array([0.9, 0.999], dtype=np.float32)
                arrays["beta_power.b"] = np.array([0.9, 0.999], dtype=np.float32)
            else:
                arrays["step"] = np.array(EXPECTED_UPDATES, dtype=np.int64)
            np.savez(target / "state.npz", **arrays)
            record = {"framework": framework, "world_size": world, "updates": EXPECTED_UPDATES}
            (Path(root) / f"{framework}-w{world}-r1.json").write_text(json.dumps(record) + "\n")


def run_self_tests():
    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "numerics-1"
        _write_synthetic(base)
        report = analyse(base)
        checks += 1
        for comparison in report["comparisons"]:
            if comparison["max_abs"] != 0 or not comparison["exact"]:
                _fail(f"self-test comparison is not exact: {comparison['label']}")
        mutations = []

        def add(name, mutate):
            mutations.append((name, mutate))

        def mutate_npz(root, framework, world, fn):
            path = Path(root) / "snapshots" / f"{framework}-w{world}" / "state.npz"
            with np.load(path) as archive:
                arrays = {name: archive[name] for name in archive.files}
            fn(arrays)
            np.savez(path, **arrays)

        add("nonfinite", lambda root: mutate_npz(root, "flux", 2, lambda a: a.__setitem__("param.a", np.full_like(a["param.a"], np.nan))))
        add("missing-array", lambda root: mutate_npz(root, "pytorch", 4, lambda a: a.pop("m.a")))
        add("wrong-updates", lambda root: (Path(root) / "pytorch-w2-r1.json").write_text(json.dumps({"framework": "pytorch", "world_size": 2, "updates": 3}) + "\n"))
        add("missing-snapshot", lambda root: (Path(root) / "snapshots" / "flux-w4" / "state.npz").unlink())
        add("shape", lambda root: mutate_npz(root, "pytorch", 2, lambda a: a.__setitem__("v.b", np.zeros((3, 2), dtype=np.float32))))
        add("step", lambda root: mutate_npz(root, "pytorch", 1, lambda a: a.__setitem__("step", np.array(299, dtype=np.int64))))

        for name, mutate in mutations:
            copy = Path(tmp) / f"mut-{name}"
            shutil.copytree(base, copy)
            mutate(copy)
            try:
                analyse(copy)
            except Exception:
                checks += 1
            else:
                _fail(f"self-test mutation was accepted: {name}")
    print(f"TIMING_NUMERICS_SELFTEST PASS checks={checks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    report = analyse(args.root, args.out)
    _print_report(report)


if __name__ == "__main__":
    main()
