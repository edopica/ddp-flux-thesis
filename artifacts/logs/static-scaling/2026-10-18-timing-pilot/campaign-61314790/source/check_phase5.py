#!/usr/bin/env python3
"""Phase 5 selected-batch numerical checker.

Consumes the DDP artifacts produced by ``phase4_flux.jl --mode ddp`` and
``phase4_torch.py --mode ddp`` on the Phase 5 fixture. The checks are the
accepted Phase 4 numerical rule, finite ``abs(a - b) < 1e-3`` with no relative
allowance, plus exact structural, shard, replica, and step checks.

The checker does not load activations, so a selected global batch of 512 does
not create multi-gigabyte reference archives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

UPDATES = ("u1", "u2", "u3")
ATOL = 1.0e-3
FRAMEWORKS = ("flux", "pytorch")


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


class Results:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, max_abs=0.0, violations=0, detail=""):
        self.rows.append({
            "name": name,
            "ok": bool(ok),
            "max_abs": float(max_abs),
            "violations": int(violations),
            "detail": detail,
        })
        print("PHASE5_CATEGORY %s %s max_abs=%.9g violations=%d %s"
              % (name, "PASS" if ok else "FAIL", max_abs, violations, detail))

    def ok(self):
        return all(row["ok"] for row in self.rows)


def exact(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, int(a.size + b.size), "shape %s != %s" % (a.shape, b.shape)
    bad = np.count_nonzero(a != b)
    return bad == 0, int(bad), ""


def abs_rule(a, b, atol=ATOL):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, math.inf, int(a.size + b.size), "shape %s != %s" % (a.shape, b.shape)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return False, math.inf, int(a.size + b.size), "nonfinite"
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    violations = int(np.count_nonzero(~(diff < atol)))
    return violations == 0, float(diff.max()) if diff.size else 0.0, violations, ""


def load_config(root, framework, world, errors):
    directory = os.path.join(root, "ddp", framework, "w%d" % world)
    label = "%s.w%d" % (framework, world)
    summary_path = os.path.join(directory, "summary.json")
    if not os.path.isfile(summary_path):
        errors.append("%s:summary_missing" % label)
        return None
    summary = load_json(summary_path)
    ranks = []
    for rank in range(world):
        npz_path = os.path.join(directory, "rank%d.npz" % rank)
        json_path = os.path.join(directory, "rank%d.json" % rank)
        if not os.path.isfile(npz_path) or not os.path.isfile(json_path):
            errors.append("%s:rank%d_missing" % (label, rank))
            continue
        ranks.append({
            "rank": rank,
            "npz": load_npz(npz_path),
            "json": load_json(json_path),
            "npz_path": npz_path,
        })
    if len(ranks) != world:
        return None
    return {"label": label, "framework": framework, "world": world,
            "summary": summary, "ranks": ranks, "directory": directory}


def flux_syncgrad(config, name, update):
    arrays = [rank["npz"]["localgrad.%s.%s" % (update, name)] for rank in config["ranks"]]
    total = arrays[0].astype(np.float64)
    for array in arrays[1:]:
        total = total + array.astype(np.float64)
    return (total / float(config["world"])).astype(np.float32)


def torch_syncgrad(config, name, update):
    values = [rank["npz"]["syncgrad.%s.%s" % (update, name)] for rank in config["ranks"]]
    reference = values[0]
    for value in values[1:]:
        if not np.array_equal(reference, value):
            raise ValueError("PyTorch rank gradients differ for %s.%s" % (update, name))
    return reference


def check_integrity(results, config, data, batch, bundle_names):
    label = config["label"]
    summary = config["summary"]
    problems = []
    if int(summary.get("global_batch_size", -1)) != batch:
        problems.append("global_batch_size")
    if int(summary.get("local_batch_size", -1)) != batch // config["world"]:
        problems.append("local_batch_size")
    if int(summary.get("updates", -1)) != 3:
        problems.append("updates")
    train_ref = data["train_transformed_ref"][0]
    labels = data["train_labels"]
    chunk = batch // config["world"]
    for record in config["ranks"]:
        rank = record["rank"]
        npz = record["npz"]
        meta = record["json"]
        if int(np.asarray(npz["meta.rank"]).reshape(-1)[0]) != rank:
            problems.append("rank%d:meta_rank" % rank)
        if int(np.asarray(npz["meta.world_size"]).reshape(-1)[0]) != config["world"]:
            problems.append("rank%d:meta_world" % rank)
        expected_ids = np.arange(rank * chunk, (rank + 1) * chunk)
        expected_input = np.ascontiguousarray(train_ref[expected_ids])
        expected_labels = np.ascontiguousarray(labels[expected_ids])
        for update in ("u1", "u3"):
            ok, bad, detail = exact(npz["input." + update], expected_input)
            if not ok:
                problems.append("rank%d:input.%s:%s:%d" % (rank, update, detail, bad))
            ok, bad, detail = exact(npz["label." + update], expected_labels)
            if not ok:
                problems.append("rank%d:label.%s:%s:%d" % (rank, update, detail, bad))
        if int(meta.get("global_batch_size", -1)) != batch:
            problems.append("rank%d:json_global_batch" % rank)
        if int(meta.get("local_batch_size", -1)) != chunk:
            problems.append("rank%d:json_local_batch" % rank)
        if int(np.asarray(npz["step.u3"]).reshape(-1)[0]) != 3:
            problems.append("rank%d:step" % rank)
        for name in bundle_names:
            for prefix in ("param.u3", "m.u3", "v.u3"):
                if prefix + "." + name not in npz:
                    problems.append("rank%d:missing:%s.%s" % (rank, prefix, name))
            if config["framework"] == "flux":
                if "localgrad.u1." + name not in npz:
                    problems.append("rank%d:missing:localgrad.%s" % (rank, name))
            else:
                if "syncgrad.u1." + name not in npz:
                    problems.append("rank%d:missing:syncgrad.%s" % (rank, name))
    results.add("integrity." + label, not problems,
               0.0, len(problems), "problems=%s" % (";".join(problems[:6]) or "none"))


def check_replicas(results, config):
    label = config["label"]
    hashes = [record["json"].get("state_hashes") for record in config["ranks"]]
    ok = all(value is not None for value in hashes) and all(value == hashes[0] for value in hashes)
    results.add("replicas." + label, ok, 0.0, 0 if ok else 1,
                "state_hashes_equal=%s" % ok)


def check_cross_framework(results, flux, torch, bundle_names, name):
    categories = []
    worst = 0.0
    violations = 0
    for update in UPDATES:
        for param in bundle_names:
            a = flux_syncgrad(flux, param, update)
            b = torch_syncgrad(torch, param, update)
            ok, diff, bad, detail = abs_rule(a, b)
            worst = max(worst, diff)
            violations += bad
            if not ok:
                categories.append("%s.%s:%s" % (update, param, detail or bad))
    label = "syncgrad.cross." + name
    results.add(label, not categories, worst, violations,
                "problems=%s" % (";".join(categories[:6]) or "none"))


def check_cross_state(results, flux, torch, bundle_names, name):
    categories = []
    worst = 0.0
    violations = 0
    for prefix in ("param.u3", "m.u3", "v.u3"):
        for param in bundle_names:
            a = flux["ranks"][0]["npz"][prefix + "." + param]
            b = torch["ranks"][0]["npz"][prefix + "." + param]
            ok, diff, bad, detail = abs_rule(a, b)
            worst = max(worst, diff)
            violations += bad
            if not ok:
                categories.append("%s.%s:%s" % (prefix, param, detail or bad))
    results.add("state.cross." + name, not categories, worst, violations,
                "problems=%s" % (";".join(categories[:6]) or "none"))


def check_cross_logits_loss(results, flux, torch, name):
    problems = []
    worst = 0.0
    violations = 0
    for key in ("logits.u1", "logits.u3", "loss.u1", "loss.u2", "loss.u3"):
        a = flux["ranks"][0]["npz"][key]
        b = torch["ranks"][0]["npz"][key]
        ok, diff, bad, detail = abs_rule(a, b)
        worst = max(worst, diff)
        violations += bad
        if not ok:
            problems.append("%s:%s" % (key, detail or bad))
    results.add("logits_loss.cross." + name, not problems, worst, violations,
                "problems=%s" % (";".join(problems[:6]) or "none"))


def check_same_framework(results, reference, other, bundle_names, name):
    problems = []
    worst = 0.0
    violations = 0
    for update in UPDATES:
        for param in bundle_names:
            if reference["framework"] == "flux":
                a = flux_syncgrad(reference, param, update)
                b = flux_syncgrad(other, param, update)
            else:
                a = torch_syncgrad(reference, param, update)
                b = torch_syncgrad(other, param, update)
            ok, diff, bad, detail = abs_rule(a, b)
            worst = max(worst, diff)
            violations += bad
            if not ok:
                problems.append("%s.%s:%s" % (update, param, detail or bad))
    for prefix in ("param.u3", "m.u3", "v.u3"):
        for param in bundle_names:
            a = reference["ranks"][0]["npz"][prefix + "." + param]
            b = other["ranks"][0]["npz"][prefix + "." + param]
            ok, diff, bad, detail = abs_rule(a, b)
            worst = max(worst, diff)
            violations += bad
            if not ok:
                problems.append("%s.%s:%s" % (prefix, param, detail or bad))
    results.add("samefw." + name, not problems, worst, violations,
                "problems=%s" % (";".join(problems[:6]) or "none"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--ddp-root", required=True,
                        help="Directory containing <framework>/w<W>/ artifacts")
    parser.add_argument("--world-sizes", default="1,2,4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    fixture_dir = os.path.abspath(args.fixture_dir)
    data_path = os.path.join(fixture_dir, "data_fixture.npz")
    bundle_path = os.path.join(fixture_dir, "initial_bundle.npz")
    data = load_npz(data_path)
    with np.load(bundle_path, allow_pickle=False) as archive:
        bundle_names = list(archive.files)
    batch = int(data["train_transformed_ref"].shape[1])
    world_sizes = [int(item) for item in args.world_sizes.split(",") if item]
    if any(world not in (1, 2, 4) for world in world_sizes):
        raise SystemExit("--world-sizes must contain only 1, 2, 4")

    results = Results()
    errors = []
    configs = {}
    for framework in FRAMEWORKS:
        for world in world_sizes:
            config = load_config(args.ddp_root, framework, world, errors)
            if config is not None:
                configs[(framework, world)] = config
    if errors:
        results.add("artifacts.complete", False, 0.0, len(errors),
                    "problems=%s" % (";".join(errors[:6])))

    for (framework, world), config in configs.items():
        check_integrity(results, config, data, batch, bundle_names)
        check_replicas(results, config)

    for world in world_sizes:
        flux = configs.get(("flux", world))
        torch = configs.get(("pytorch", world))
        if flux is None or torch is None:
            results.add("cross.artifacts.w%d" % world, False, 0.0, 1, "missing")
            continue
        check_cross_framework(results, flux, torch, bundle_names, "w%d" % world)
        check_cross_state(results, flux, torch, bundle_names, "w%d" % world)
        check_cross_logits_loss(results, flux, torch, "w%d" % world)

    for framework in FRAMEWORKS:
        reference = configs.get((framework, 1))
        if reference is None:
            continue
        for world in world_sizes:
            if world == 1:
                continue
            other = configs.get((framework, world))
            if other is None:
                results.add("samefw.%s.w%d" % (framework, world), False, 0.0, 1, "missing")
                continue
            check_same_framework(results, reference, other, bundle_names,
                                 "%s.w%d" % (framework, world))

    def fixture_identity(record_json):
        value = record_json.get("fixture_sha256")
        if isinstance(value, dict):
            return value.get("data_fixture_sha256")
        value = record_json.get("fixture")
        if isinstance(value, dict):
            return value.get("data_fixture_sha256")
        return None

    def bundle_identity(record_json):
        return record_json.get("bundle_array_hash") or record_json.get("bundle_hash")

    identities = []
    for config in configs.values():
        for record in config["ranks"]:
            identities.append((config["framework"], config["world"],
                               fixture_identity(record["json"]),
                               bundle_identity(record["json"]),
                               int(record["json"].get("global_batch_size", -1))))
    ok = len(identities) > 0 and all(
        item[2] is not None and item[3] is not None
        and item[2] == identities[0][2] and item[3] == identities[0][3]
        and item[4] == identities[0][4] for item in identities)
    results.add("workload.identity", ok, 0.0, 0 if ok else 1,
                "fixture_bundle_batch_equal=%s" % ok)

    report = {
        "schema": "phase5-numerical-check-v1",
        "fixture_dir": fixture_dir,
        "ddp_root": os.path.abspath(args.ddp_root),
        "global_batch": batch,
        "world_sizes": world_sizes,
        "atol": ATOL,
        "categories": results.rows,
        "pass": results.ok(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print("PHASE5_NUMERICAL_CHECK %s categories=%d"
          % ("PASS" if results.ok() else "FAIL", len(results.rows)))
    return 0 if results.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
