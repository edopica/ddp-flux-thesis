#!/usr/bin/env python3
"""Artifact-only checker and batch selector for the Phase 5 memory pilot.

Each configuration directory contains ``rank0.json`` ... and ``summary.json``.
The checker verifies workload identity, required counters, explicit
unavailable-counter status, and the recorded 80% device-memory margin.

Examples::

  python3 scripts/static_scaling/check_memory.py \
    --batch-root 512=/evidence/memory-b512 --out /evidence/decision-b512.json

  python3 scripts/static_scaling/check_memory.py \
    --batch-root 512=/evidence/memory-b512 \
    --batch-root 256=/evidence/memory-b256 --select \
    --out /evidence/batch-decision.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

FRAMEWORKS = ("flux", "pytorch")
WORLDS = (1, 2, 4)
REQUIRED_RANK_KEYS = (
    "global_batch", "world_size", "local_batch", "warmup", "updates",
    "fixture_sha256", "bundle_sha256", "workload", "staging_bytes",
    "model_bytes", "device_total_bytes", "device_process_peak_bytes",
    "allocator_live_peak_bytes", "allocator_reserved_peak_bytes", "phases",
    "counter_status", "margin", "pass_checks",
)
PHASE_KEYS = (
    "forward_live_bytes", "backward_live_bytes", "optimizer_live_bytes",
)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def as_int(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def evaluate_config(root, framework, world, errors):
    directory = os.path.join(root, "%s-w%d" % (framework, world))
    label = "%s.w%d" % (framework, world)
    summary_path = os.path.join(directory, "summary.json")
    if not os.path.isfile(summary_path):
        errors.append("%s:summary_missing" % label)
        return None
    try:
        summary = load_json(summary_path)
    except Exception as exc:
        errors.append("%s:summary_json:%s" % (label, type(exc).__name__))
        return None
    if summary.get("framework") != framework or as_int(summary.get("world_size")) != world:
        errors.append("%s:summary_identity" % label)
    rank_files = summary.get("rank_files")
    if not isinstance(rank_files, list) or len(rank_files) != world:
        errors.append("%s:rank_file_set" % label)
        rank_files = ["rank%d.json" % r for r in range(world)]
    ranks = []
    for name in rank_files:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            errors.append("%s:%s_missing" % (label, name))
            continue
        try:
            record = load_json(path)
        except Exception as exc:
            errors.append("%s:%s_json:%s" % (label, name, type(exc).__name__))
            continue
        missing = [key for key in REQUIRED_RANK_KEYS if key not in record]
        if missing:
            errors.append("%s:%s_missing_keys:%s" % (label, name, ",".join(missing)))
            continue
        if not record.get("pass_checks"):
            errors.append("%s:%s_pass_checks_false" % (label, name))
        phases = record.get("phases") or {}
        missing_phases = [key for key in PHASE_KEYS if as_int(phases.get(key)) is None]
        if missing_phases:
            errors.append("%s:%s_missing_phase:%s"
                          % (label, name, ",".join(missing_phases)))
        if world > 1:
            status = (record.get("counter_status") or {}).get("communication")
            if status in (None, "unavailable"):
                errors.append("%s:%s_communication_status_missing" % (label, name))
            if framework == "flux" and as_int(phases.get("sync_live_bytes")) is None:
                errors.append("%s:%s_sync_counter_missing" % (label, name))
        for key in ("staging_bytes", "model_bytes", "device_total_bytes",
                    "allocator_live_peak_bytes", "allocator_reserved_peak_bytes"):
            if as_int(record.get(key)) is None:
                errors.append("%s:%s_null_counter:%s" % (label, name, key))
        ranks.append(record)

    if len(ranks) != world:
        return None
    identity = {
        "global_batch": as_int(summary.get("global_batch")),
        "warmup": as_int(summary.get("warmup")),
        "updates": as_int(summary.get("updates")),
    }
    for record in ranks:
        for key, value in identity.items():
            if as_int(record.get(key)) != value:
                errors.append("%s:identity_%s" % (label, key))
    fixture_hashes = {record.get("fixture_sha256") for record in ranks}
    bundle_hashes = {record.get("bundle_sha256") for record in ranks}
    if len(fixture_hashes) != 1 or len(bundle_hashes) != 1:
        errors.append("%s:input_hash_mismatch" % label)

    process = [as_int(r.get("device_process_peak_bytes")) for r in ranks]
    reserved = [as_int(r.get("allocator_reserved_peak_bytes")) for r in ranks]
    total_values = {as_int(r.get("device_total_bytes")) for r in ranks}
    if len(total_values) != 1 or None in total_values:
        errors.append("%s:device_total_mismatch" % label)
        total = None
    else:
        total = total_values.pop()
    complete = all(value is not None for value in process)
    peak_process = max(process) if complete else None
    peak_reserved = max(reserved) if all(value is not None for value in reserved) else None
    limit = int(0.8 * total) if total is not None else None
    fits = (peak_process is not None and limit is not None and peak_process <= limit)
    status = "PASS" if fits else ("INCOMPLETE" if not complete else "FAIL")
    return {
        "label": label,
        "framework": framework,
        "world_size": world,
        "global_batch": identity["global_batch"],
        "local_batch": as_int(ranks[0].get("local_batch")),
        "warmup": identity["warmup"],
        "updates": identity["updates"],
        "fixture_sha256": sorted(fixture_hashes)[0] if len(fixture_hashes) == 1 else None,
        "bundle_sha256": sorted(bundle_hashes)[0] if len(bundle_hashes) == 1 else None,
        "device_total_bytes": total,
        "limit_bytes": limit,
        "peak_process_bytes": peak_process,
        "peak_reserved_bytes": peak_reserved,
        "staging_bytes": max(as_int(r.get("staging_bytes")) or 0 for r in ranks),
        "model_bytes": max(as_int(r.get("model_bytes")) or 0 for r in ranks),
        "margin_bytes": (total - peak_process) if complete and total is not None else None,
        "status": status,
        "rank_process_bytes": process,
        "rank_reserved_bytes": reserved,
        "counter_status": [r.get("counter_status") for r in ranks],
    }


def evaluate_batch(batch, root, errors):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        errors.append("batch%d:root_missing:%s" % (batch, root))
        return {"global_batch": batch, "root": root, "status": "INCOMPLETE",
                "configs": [], "errors": ["root_missing"]}
    local_errors = []
    configs = []
    for framework in FRAMEWORKS:
        for world in WORLDS:
            config = evaluate_config(root, framework, world, local_errors)
            if config is not None:
                configs.append(config)
    if configs:
        batches = {config["global_batch"] for config in configs}
        if batches != {batch}:
            local_errors.append("batch%d:global_batch_mismatch:%s"
                                % (batch, sorted(batches)))
        warmups = {config["warmup"] for config in configs}
        updates = {config["updates"] for config in configs}
        if len(warmups) != 1:
            local_errors.append("batch%d:warmup_mismatch" % batch)
        if len(updates) != 1:
            local_errors.append("batch%d:updates_mismatch" % batch)
        fixtures = {config["fixture_sha256"] for config in configs}
        bundles = {config["bundle_sha256"] for config in configs}
        if len(fixtures) != 1 or None in fixtures:
            local_errors.append("batch%d:fixture_identity_mismatch" % batch)
        if len(bundles) != 1 or None in bundles:
            local_errors.append("batch%d:bundle_identity_mismatch" % batch)
    statuses = [config["status"] for config in configs]
    complete = len(configs) == len(FRAMEWORKS) * len(WORLDS) and not local_errors
    passed = complete and all(status == "PASS" for status in statuses)
    status = "PASS" if passed else ("INCOMPLETE" if not complete else "FAIL")
    errors.extend(local_errors)
    return {
        "global_batch": batch,
        "root": root,
        "status": status,
        "complete": complete,
        "configs": configs,
        "errors": local_errors,
    }


def print_batch(result):
    print("MEMORY_BATCH %d %s root=%s"
          % (result["global_batch"], result["status"], result["root"]))
    print("framework W status peak_process_MiB limit_MiB margin_MiB "
          "peak_reserved_MiB staging_MiB")
    for config in result["configs"]:
        process = config["peak_process_bytes"]
        limit = config["limit_bytes"]
        margin = config["margin_bytes"]
        reserved = config["peak_reserved_bytes"]
        staging = config.get("staging_bytes")
        print("%s %d %s %s %s %s %s %s" % (
            config["framework"], config["world_size"], config["status"],
            "NA" if process is None else "%.1f" % (process / 2**20),
            "NA" if limit is None else "%.1f" % (limit / 2**20),
            "NA" if margin is None else "%.1f" % (margin / 2**20),
            "NA" if reserved is None else "%.1f" % (reserved / 2**20),
            "NA" if staging is None else "%.1f" % (staging / 2**20),
        ))
    for error in result["errors"]:
        print("MEMORY_ERROR " + error)


def parse_batch_root(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected B=DIR, got %r" % value)
    batch_text, root = value.split("=", 1)
    try:
        batch = int(batch_text)
    except ValueError:
        raise argparse.ArgumentTypeError("batch must be an integer: %r" % batch_text)
    if batch not in (512, 256):
        raise argparse.ArgumentTypeError("batch must be 512 or 256: %d" % batch)
    return batch, root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", action="append", type=parse_batch_root,
                        required=True, metavar="B=DIR")
    parser.add_argument("--select", action="store_true",
                        help="Choose 512 if it passes, otherwise 256")
    parser.add_argument("--out", required=True, help="Decision JSON path")
    args = parser.parse_args(argv)

    roots = {}
    for batch, root in args.batch_root:
        if batch in roots:
            raise SystemExit("duplicate --batch-root for %d" % batch)
        roots[batch] = root

    errors = []
    results = {}
    for batch in sorted(roots, reverse=True):
        result = evaluate_batch(batch, roots[batch], errors)
        results[batch] = result
        print_batch(result)

    decision = {
        "schema": "static-memory-batch-decision-v1",
        "batches": results,
        "select_requested": bool(args.select),
    }
    if args.select:
        selected = None
        reason = None
        for batch in (512, 256):
            result = results.get(batch)
            if result is not None and result["status"] == "PASS":
                selected = batch
                reason = "batch %d passes the 80%% margin in all six configurations" % batch
                break
        if selected is None:
            reason = "neither candidate batch passes or is complete"
        decision["selected_global_batch"] = selected
        decision["reason"] = reason
        print("MEMORY_SELECTED %s" % ("none" if selected is None else selected))
    else:
        selected = None
        for batch, result in results.items():
            if result["status"] == "PASS":
                selected = batch
        decision["selected_global_batch"] = selected
        decision["reason"] = "single-batch evaluation"

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=False)
        handle.write("\n")

    if errors:
        for error in errors:
            print("MEMORY_CHECK_ERROR " + error)
    if not args.select:
        ok = all(result["status"] == "PASS" for result in results.values())
    else:
        ok = decision["selected_global_batch"] is not None
    print("MEMORY_CHECK %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
