#!/usr/bin/env python3
"""Independently validate the bounded W=1 cache campaign and report point deltas.

This checker does not certify live-tensor equivalence, continuous peaks, a leak
bound, or throughput. Reuses the existing full-workload identity and trace
checks; cache snapshots have their own lifecycle and counter-semantics checks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import check_full_campaign as full

PHASES = ("initialized", "post_warmup", "post_reset_gc", "post_training",
          "post_gc", "post_reclaim")
METRICS = ("allocator_used_bytes", "allocator_reserved_bytes", "process_bytes",
           "device_used_bytes", "total_bytes")
FRAMEWORKS = ("flux-current", "pytorch")
QUALIFICATIONS = [
    "Snapshot counters are synchronized point samples, not within-step peaks.",
    "Process memory includes allocator cache, libraries and runtime overhead.",
    "The process-memory drop after cache release measures observed reclaimability at that boundary; the remainder is not pure live-tensor memory.",
    "Framework allocator counters retain their loaded API semantics; cross-framework values are not assumed equivalent.",
    "Short W=1 runs diagnose the baseline footprint; they do not establish absence of leaks or quantify multi-rank DDP overhead.",
    "Diagnostic timing includes instrumentation and is not throughput evidence.",
    "State hashing between post_training and explicit GC may itself trigger ordinary collection; the observed GC-stage drop includes any such collection.",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def byte_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_snapshots(record, registration):
    diagnostic = record.get("cache_diagnostic")
    require(isinstance(diagnostic, dict), "missing cache_diagnostic")
    require(diagnostic.get("schema") == "static-cache-diagnostic-v1", "cache schema mismatch")
    require(diagnostic.get("pid") == registration["pid"], "cache PID differs from rank registration")
    require(diagnostic.get("device_uuid") == registration["gpu_uuid"], "cache GPU differs from rank registration")
    for field in ("retention_policy", "peak_policy"):
        require(isinstance(diagnostic.get(field), str) and diagnostic[field].strip(), "missing " + field)
    retention = diagnostic["retention_policy"].lower()
    require(all(word in retention for word in ("model", "optimizer", "dataset", "post_training", "post_gc", "post_reclaim")), "retention policy does not cover persistent state and release stages")
    semantics = diagnostic.get("counter_semantics")
    require(isinstance(semantics, dict), "missing counter_semantics")
    for metric in METRICS:
        require(isinstance(semantics.get(metric), str) and semantics[metric].strip(), "missing counter semantics for " + metric)
    snapshots = diagnostic.get("snapshots")
    require(isinstance(snapshots, list) and tuple(row.get("phase") for row in snapshots) == PHASES, "cache phases missing, duplicated or reordered")
    previous_end = -math.inf
    total = full._record_total_bytes(record)
    for row in snapshots:
        start, end = row.get("epoch"), row.get("end_epoch")
        require(full._finite_number(start) and full._finite_number(end) and previous_end <= start <= end, "snapshot timestamps overlap or are invalid")
        previous_end = end
        status = row.get("counter_status")
        require(isinstance(status, dict), "missing counter_status")
        for metric in METRICS:
            value, state = row.get(metric), status.get(metric)
            require(isinstance(state, str), "missing counter status for " + metric)
            if state == "measured":
                require(byte_count(value), "invalid measured byte counter " + metric)
                require(value <= total, "counter exceeds device total: " + metric)
            else:
                require(metric.startswith("allocator_"), "required process/device counter unavailable")
                require(state.startswith("unavailable:") and state.split(":", 1)[1].strip(), "counter unavailability lacks reason")
                require(value is None, "unavailable counter must be null")
        require(row["total_bytes"] == total, "snapshot device total differs")
        if row["allocator_used_bytes"] is not None and row["allocator_reserved_bytes"] is not None:
            require(row["allocator_used_bytes"] <= row["allocator_reserved_bytes"], "allocator used exceeds reserved")
    windows = full._record_training_windows(record, 1)
    start, end = windows[0]
    require(snapshots[2]["end_epoch"] <= start, "reset snapshot overlaps training window")
    require(snapshots[3]["epoch"] >= end, "post-training snapshot precedes synchronized training end")
    evidence = diagnostic.get("runtime_api_evidence")
    require(isinstance(evidence, dict), "missing runtime API evidence")
    for metric in METRICS[:2]:
        require(isinstance(evidence.get(metric), dict) and evidence[metric].get("api"), "missing loaded allocator API evidence")
        if any(row["counter_status"][metric] == "measured" for row in snapshots):
            require(isinstance(evidence[metric].get("doc"), str) and evidence[metric]["doc"].strip(), "measured allocator counter lacks loaded API documentation")
    invariant = diagnostic.get("state_invariant")
    hashes = record.get("hashes") if record["framework"] == "flux" else record["ranks"][0]["hashes"]
    require(isinstance(invariant, dict) and invariant.get("passed") is True
            and invariant.get("before") == hashes and invariant.get("after") == hashes,
            "cache release state invariant missing or changed")
    by_phase = {row["phase"]: row for row in snapshots}
    return {
        "pid": registration["pid"], "gpu_uuid": registration["gpu_uuid"],
        "counter_semantics": semantics,
        "snapshots": snapshots,
        "process_gc_drop_bytes": by_phase["post_training"]["process_bytes"] - by_phase["post_gc"]["process_bytes"],
        "process_cache_release_drop_bytes": by_phase["post_gc"]["process_bytes"] - by_phase["post_reclaim"]["process_bytes"],
    }


def check_campaign(root):
    root = Path(root)
    manifest = full._load_json(root / "manifest.json")
    expected = {"schema": "static-cache-campaign-v1", "repeats": 2,
                "frameworks": list(FRAMEWORKS), "world_size": 1,
                "global_batch": 256, "warmup": 20, "updates": 20,
                "cpus_per_task": 4, "epoch": 0}
    for key, value in expected.items():
        require(manifest.get(key) == value, "manifest mismatch: " + key)
    source_files = manifest.get("source_files")
    require(isinstance(source_files, dict) and source_files, "missing source_files")
    for relative, digest in source_files.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve() / "source"), "source path escapes snapshot")
        require(full._is_hex64(digest) and path.is_file() and full._sha256(path) == digest, "source snapshot hash mismatch: " + relative)
    names = {Path(relative).name for relative in source_files}
    require(full.REQUIRED_SOURCE_FILES <= names, "source snapshot lacks required workload source files")
    require({"check_cache_diagnostic.py", "cache_diagnostic.jl", "cache_diagnostic.py"} <= names, "source snapshot lacks diagnostic sources")
    identity = full._parse_pairs((root / "source-identity.txt").read_text())
    require(manifest.get("source_identity") == identity, "manifest source identity differs")
    for key in ("THESIS_STATE_SHA256", "CURRENT_STATE_SHA256", "CURRENT_SOURCE_TREE_SHA256"):
        require(full._is_hex64(identity.get(key)), "invalid source identity: " + key)
    require(identity.get("CURRENT_COMMIT") == full.CURRENT_COMMIT, "unreviewed current Flux commit")
    require(identity.get("CURRENT_STATE_SHA256") == full.CURRENT_STATE_SHA256, "unreviewed current Flux state")
    require(identity.get("CURRENT_SOURCE_TREE_SHA256") == full.CURRENT_SOURCE_TREE_SHA256, "unreviewed current Flux source tree")
    dataset = full._check_dataset_identity(root, {})
    resources = full._load_json(root / "resources.json")
    require(resources.get("schema") == "static-cache-resources-v1", "resource schema mismatch")
    require(resources.get("cpus_per_task") == 4 and resources.get("ntasks") == 1, "actual resource policy differs from four CPUs and one task")
    require(resources.get("job") == manifest.get("job") and resources.get("node") == manifest.get("node")
            and bool(resources.get("job")) and bool(resources.get("node")), "manifest resource job/node mismatch")
    runtime = full._load_json(root / "runtime-python.json")
    for field in ("python", "executable", "torch", "cuda", "cudnn", "torch_path"):
        require(runtime.get(field) not in (None, ""), "missing loaded Python runtime evidence: " + field)
    expected_labels = {"%s-w1-r%d" % (framework, repeat) for framework in FRAMEWORKS for repeat in (1, 2)}
    require({p.stem for p in (root / "records").glob("*.json")} == expected_labels, "record set incomplete or contains extras")
    records, rows, pids = [], [], set()
    for framework in FRAMEWORKS:
        for repeat in (1, 2):
            label = "%s-w1-r%d" % (framework, repeat)
            record = full._check_record(root / "records" / (label + ".json"), framework, 1, 20, 20, 0, 256, identity, dataset, False)
            require(record.get("job") == resources["job"] and record.get("node") == resources["node"], "record job/node differs from allocation")
            if framework == "pytorch":
                require(record["source"]["torch_version"] == runtime["torch"], "loaded torch version differs")
                require(runtime["python"].startswith(record["source"]["python_version"]), "loaded Python version differs")
            else:
                require(bool(record["source"].get("julia_version")), "missing loaded Julia version")
            sample_path = root / "samples" / (label + ".csv")
            registration = full._load_rank_registration(sample_path, 1)[0]
            require(registration["pid"] not in pids, "PID reused across purported fresh-process runs")
            pids.add(registration["pid"])
            row = check_snapshots(record, registration)
            row.update(framework=framework, repeat=repeat)
            row["sampled_memory"] = full._summarize_gpu_samples(sample_path, 1, full._record_training_windows(record, 1), require_density=True)
            rows.append(row)
            records.append(record)
    full._check_record_identity(records)
    return {"schema": "static-cache-diagnostic-report-v1", "status": "PASS",
            "runs": rows, "qualifications": QUALIFICATIONS,
            "NUMERICAL_EQUIVALENCE_CLAIM": "none"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = check_campaign(args.root)
    except (AssertionError, OSError, ValueError, TypeError, KeyError) as error:
        print("CACHE_DIAGNOSTIC_CHECK FAIL: " + str(error))
        return 1
    (args.root / "CACHE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report["runs"]:
        print("CACHE_DIAGNOSTIC %s repeat=%d GC_drop_MiB=%.3f cache_release_drop_MiB=%.3f" %
              (row["framework"], row["repeat"], row["process_gc_drop_bytes"] / 1024**2,
               row["process_cache_release_drop_bytes"] / 1024**2))
    print("CACHE_DIAGNOSTIC_CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
