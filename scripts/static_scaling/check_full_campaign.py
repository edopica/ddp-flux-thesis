#!/usr/bin/env python3
"""Preflight and campaign artifact checker for the full-dataset benchmark.

The checker has two subcommands.

``preflight``
  Reads one candidate directory per batch. Each candidate contains one
  successful ``static-timing-full-dataset-v1`` record per framework and world,
  plus the ``nvidia-smi`` process samples collected while that run executed.
  It applies the recorded OD-8 criterion: the peak per-process device memory
  must not exceed 80% of the device total. A missing sample or counter is
  ``INCOMPLETE`` and cannot pass. It writes the selected batch decision.

``campaign``
  Verifies the selected configuration, actual source identity, loaded runtime
  versions, device mapping, resource policy, complete repeat set, timing
  arithmetic, and the matching untimed state diagnostics required by OD-14.
  It also requires every timing repeat's timestamped ``nvidia-smi`` trace and
  per-rank PID/GPU registration, clips samples to each rank's recorded
  training window, and reports sampled training, startup, and whole-process
  peaks with a simultaneous-sum aggregate footprint. It writes ``REPORT.json``
  and prints the timing, memory, and diagnostic summaries. Finite differences
  in the diagnostics receive no pass/fail threshold.

Reproduction::

  python3 scripts/static_scaling/check_full_campaign.py preflight \\
    --root OUT --candidate 512=OUT/preflight/b512 \\
    --updates 20 --warmup 20 --select --out OUT/preflight/decision.json
  python3 scripts/static_scaling/check_full_campaign.py campaign --root OUT
  python3 scripts/static_scaling/check_full_campaign.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np


SCHEMA = "static-timing-full-dataset-v1"
PREFLIGHT_SCHEMA = "static-full-preflight-decision-v1"
PROVENANCE_SCHEMA = "static-full-campaign-provenance-v1"
CONFIG_SCHEMA = "static-full-campaign-config-v1"
REPORT_SCHEMA = "static-full-campaign-report-v1"
RUNTIME_SCHEMA = "static-full-runtime-v1"
RESOURCES_SCHEMA = "static-full-resources-v1"
DATASET_SCHEMA = "static-full-dataset-identity-v1"

THESIS_COMMIT = "660e363e6de5b6e4b398e542d79b6ce1adb381da"
CURRENT_COMMIT = "f7b4520c4bd307c67bbc5ac92a1c68493eccff6e"
CURRENT_STATE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CURRENT_SOURCE_TREE_SHA256 = "83da2929ab7ebd66dbceb83452c8e55eae5e0444c8e96866cc2a25b513c3eb0d"
HISTORICAL_COMMIT = "8add0e42b7d9388ff365335152126e5b1e9de657"
HISTORICAL_STATE_SHA256 = "4af8409c2a2ed1c661e8ae7742877f1ee9789d00dc7e404365ecefd98ab0d34c"
HISTORICAL_SOURCE_TREE_SHA256 = "ad114ed385f7cc78e8409fa8e1f09d38133555f7fd58d5d9aca6a846bcf5e757"

FRAMEWORKS = ("flux-current", "flux-historical", "pytorch")
WORLDS = (1, 2, 4)
BATCHES = (512, 256)
N_EFF = 99840
TAIL = 160
TARGET_SAMPLE_INTERVAL_SECONDS = 0.25
MAX_SAMPLE_GAP_SECONDS = 5.0
SAMPLE_COVERAGE_SLACK_SECONDS = 5.0
MIN_TRAINING_SAMPLES = 2
MIN_PREFLIGHT_TRAINING_SAMPLES = 1
MIN_TRAINING_SAMPLE_FRACTION = 0.25
CAMPAIGN_CPUS = 4
MEMORY_QUALIFICATIONS = (
    "Reported peaks are sampled nvidia-smi per-process used-memory values at a "
    "%.2f s target interval, not continuous high-water marks." % TARGET_SAMPLE_INTERVAL_SECONDS,
    "GPU process usage includes framework allocator caches and runtime/"
    "communication overhead; it is not a pure tensor footprint.",
    "Training-window peaks use per-rank wall-clock windows recorded at the "
    "synchronized start and end of the measured updates; warmup, reset, final "
    "hashing, snapshots, and teardown are outside the window.",
    "Campaign coverage requires at least max(%d, ceil(window_seconds)) "
    "training samples per rank inside its training window, plus simultaneous "
    "samples spanning the common window within a %.1f s slack and no "
    "simultaneous gap over %.1f s; the short preflight requires one "
    "training-window sample per rank after whole-process coverage. A trace "
    "that stops early is rejected."
    % (MIN_TRAINING_SAMPLES, SAMPLE_COVERAGE_SLACK_SECONDS,
       MAX_SAMPLE_GAP_SECONDS),
    "Sample timestamps must be strictly increasing. Duplicate or out-of-order "
    "TIME blocks are rejected, and only distinct instants contribute to "
    "sample density.",
    "Startup peaks and whole-process peaks are reported separately so "
    "initialization is not attributed to training.",
    "Aggregate device footprint sums simultaneous per-rank samples; it is never "
    "the sum of independent rank maxima.",
    "Framework allocator counters are supplementary and retain their own "
    "semantics; they are not substituted for sampled process usage.",
)
LARGE_FILES = {"train_images.npy", "train_labels.npy", "val_images.npy"}
REQUIRED_DATASET_FILES = (
    "train_images.npy",
    "train_labels.npy",
    "permutations.npy",
    "augmentation.npy",
    "stats.json",
    "dataset_manifest.json",
    "schedule_manifest.json",
    "initial_bundle.npz",
    "initial_bundle.manifest.json",
)
REQUIRED_SOURCE_FILES = {
    "timing_flux.jl",
    "timing_torch.py",
    "full_dataset.jl",
    "full_dataset_timing.jl",
    "full_dataset_common.py",
    "check_full_campaign.py",
    "check_full_dataset.py",
    "precision.jl",
    "phase4_common.jl",
    "model_flux.jl",
    "model_torch.py",
    "dataset_adapter.py",
    "replay.jl",
}
REQUIRED_DRIVER_LINES = (
    "FULL_CAMPAIGN_DRIVER START",
    "FLUX_PRECOMPILE_OK",
    "FULL_CAMPAIGN_preflight PASS",
    "FULL_CAMPAIGN_timing PASS",
    "FULL_CAMPAIGN_diagnostics PASS",
)

FLUX_PRECISION = "FP32 strict FMA, TF32 off"
FLUX_GUARD = "presence + nonfinite loss/gradient rejection inside timed window"
FLUX_STATIC_GUARD = "no presence or nonfinite distributed validation"
SINGLE_GUARD = "not applicable: single-GPU baseline"
PYTORCH_PRECISION = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RECORD_NAME = re.compile(r"^(flux-current|flux-historical|pytorch)-w(\d+)-r(\d+)\.json$")
DIAGNOSTIC_NAME = re.compile(r"^(flux-current|flux-historical|pytorch)-w(\d+)\.json$")
SOURCE_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def expected_updates(batch):
    if batch == 512:
        return 195
    if batch == 256:
        return 390
    raise ValueError("batch must be 512 or 256")


def _fail(message):
    raise AssertionError(message)


def _require(condition, message):
    if not condition:
        _fail(message)


def _is_hex64(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(document):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path):
    path = Path(path)
    _require(path.is_file(), "missing file: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail("%s: invalid JSON: %s" % (path, exc))


def _parse_pairs(text):
    result = {}
    for raw in text.splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _framework_kind(framework):
    return "flux" if framework.startswith("flux") else "pytorch"


def _flux_variant(framework):
    if framework == "flux-current":
        return "flux-current"
    if framework == "flux-historical":
        return "flux-historical"
    raise ValueError(framework)


def _normalise_dataset_hashes(record):
    raw = record.get("dataset_hashes")
    _require(isinstance(raw, dict), "dataset_hashes is not an object")
    hashes = {}
    for name, info in raw.items():
        if isinstance(info, dict):
            digest = info.get("sha256")
        else:
            digest = info
        if isinstance(digest, str):
            hashes[name] = digest
    return hashes


def _record_total_bytes(record):
    if record.get("framework") == "flux":
        memory = record.get("memory")
        _require(isinstance(memory, dict), "missing Flux memory record")
        total = memory.get("device_total_bytes")
        _require(isinstance(total, int) and total > 0, "invalid Flux device total")
        return total
    ranks = record.get("ranks")
    _require(isinstance(ranks, list) and ranks, "missing PyTorch ranks")
    totals = set()
    for rank in ranks:
        memory = rank.get("memory")
        _require(isinstance(memory, dict), "missing PyTorch rank memory")
        total = memory.get("mem_get_info_total")
        _require(isinstance(total, int) and total > 0, "invalid PyTorch device total")
        totals.add(total)
    _require(len(totals) == 1, "PyTorch ranks report different device totals")
    return totals.pop()


def _load_rank_registration(sample_path, world):
    """Load the per-rank PID/GPU registration written by each runner."""
    sample_path = Path(sample_path)
    rank_dir = sample_path.with_suffix(".ranks")
    ranks = [_load_json(rank_dir / ("rank-%d.json" % rank)) for rank in range(world)]
    _require([row.get("rank") for row in ranks] == list(range(world)),
             "memory rank mismatch")
    for row in ranks:
        pid = row.get("pid")
        _require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0,
                 "memory rank %r has an invalid PID" % row.get("rank"))
        _require(isinstance(row.get("gpu_uuid"), str) and row["gpu_uuid"],
                 "memory rank %r has an invalid GPU UUID" % row.get("rank"))
    _require(len({row["pid"] for row in ranks}) == world, "duplicate memory PID")
    _require(len({row["gpu_uuid"] for row in ranks}) == world, "duplicate memory GPU")
    return ranks


def _read_gpu_samples(path, ranks):
    """Parse timestamped ``nvidia-smi`` process samples into per-stamp batches.

    Every ``TIME`` block must carry a strictly later timestamp than the
    previous block. This rejects duplicate or out-of-order stamps, which would
    otherwise inflate the sample count and hide a stalled or stepped clock.
    """
    path = Path(path)
    _require(path.is_file(), "missing GPU samples")
    expected = {(row["pid"], row["gpu_uuid"]) for row in ranks}
    by_pid = {row["pid"]: row for row in ranks}
    batches = []
    current_stamp = None
    previous_stamp = None
    current = None
    samples = 0

    def flush():
        nonlocal current_stamp, current
        if current_stamp is not None:
            batches.append((current_stamp, current or {}))
        current_stamp = None
        current = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("TIME,"):
            flush()
            try:
                stamp = float(line.split(",", 1)[1])
            except ValueError:
                continue
            if not math.isfinite(stamp):
                continue
            _require(previous_stamp is None or stamp > previous_stamp,
                     "GPU sample timestamps must be strictly increasing; "
                     "got %r after %r" % (stamp, previous_stamp))
            previous_stamp = stamp
            current_stamp = stamp
            current = {}
            continue
        if current_stamp is None:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            uuid = parts[1]
            mib = int(parts[2])
        except ValueError:
            continue
        key = (pid, uuid)
        if pid < 0 or mib < 0 or key not in expected:
            continue
        current[pid] = max(current.get(pid, 0), mib * 1024 * 1024)
        samples += 1
    flush()
    _require(batches, "GPU sample trace is empty")
    return by_pid, batches, samples


def _sample_times_and_gaps(batches):
    stamps = [stamp for stamp, _ in batches]
    _require(all(later > earlier for earlier, later in zip(stamps, stamps[1:])),
             "GPU sample timestamps are not strictly increasing")
    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    return stamps, gaps


def _required_training_samples(start_epoch, end_epoch):
    """Minimum campaign training-window samples for a rank.

    The 0.25 s target is ideal; the acceptance floor is one sample per second
    (a quarter of the target rate) with an absolute floor of two samples.
    """
    duration = max(0.0, float(end_epoch) - float(start_epoch))
    return max(MIN_TRAINING_SAMPLES,
               int(math.ceil(duration * MIN_TRAINING_SAMPLE_FRACTION
                             / TARGET_SAMPLE_INTERVAL_SECONDS)))


def _summarize_gpu_samples(sample_path, world, windows, *,
                           require_density=False):
    """Summarize per-rank and simultaneous aggregate sampled GPU usage.

    ``windows`` maps rank index to ``(start_epoch, end_epoch)``. Training peaks
    and medians use only samples inside each rank's window; whole-process and
    startup peaks are kept separate so initialization is not reported as
    training. The aggregate is the maximum of the sum of simultaneous per-rank
    samples, never the sum of independent per-rank maxima. Its timestamps must
    themselves span the common window and contain no gap over the limit when
    campaign density is required; a dense burst followed by a long uncovered
    tail is rejected even if the sample count is high. Campaign repeats pass
    ``require_density=True``; the short preflight requires only whole-process
    coverage plus one training-window sample per rank.
    """
    ranks = _load_rank_registration(sample_path, world)
    by_pid, batches, samples = _read_gpu_samples(sample_path, ranks)
    _require(set(windows) == set(range(world)),
             "training windows do not cover every rank")
    _, trace_gaps = _sample_times_and_gaps(batches)
    trace_gap_max = max(trace_gaps) if trace_gaps else None
    trace_gap_median = statistics.median(trace_gaps) if trace_gaps else None

    entries = []
    rank_gap_max = None
    for row in ranks:
        rank = row["rank"]
        pid = row["pid"]
        start, end = windows[rank]
        start = float(start)
        end = float(end)
        whole_samples = [(stamp, memory[pid]) for stamp, memory in batches
                         if pid in memory]
        whole = [value for _, value in whole_samples]
        _require(len(whole) >= 2,
                 "missing repeated GPU samples for rank %d PID %d" % (rank, pid))
        first_sample_epoch = min(stamp for stamp, _ in whole_samples)
        last_sample_epoch = max(stamp for stamp, _ in whole_samples)
        _require(first_sample_epoch <= start + SAMPLE_COVERAGE_SLACK_SECONDS,
                 "GPU samples start %.2f s after rank %d's training window"
                 % (first_sample_epoch - start, rank))
        _require(last_sample_epoch >= end - SAMPLE_COVERAGE_SLACK_SECONDS,
                 "GPU samples end %.2f s before rank %d's training window"
                 % (end - last_sample_epoch, rank))
        rank_stamps = sorted({stamp for stamp, _ in whole_samples})
        rank_gaps = [later - earlier
                     for earlier, later in zip(rank_stamps, rank_stamps[1:])]
        rank_gap = max(rank_gaps) if rank_gaps else None
        if rank_gap is not None:
            rank_gap_max = (rank_gap if rank_gap_max is None
                            else max(rank_gap_max, rank_gap))
        training = [memory[pid] for stamp, memory in batches
                    if start <= stamp <= end and pid in memory]
        required_training = (_required_training_samples(start, end)
                             if require_density
                             else MIN_PREFLIGHT_TRAINING_SAMPLES)
        _require(len(training) >= required_training,
                 "rank %d has %d training-window samples; need at least %d"
                 % (rank, len(training), required_training))
        startup = [memory[pid] for stamp, memory in batches
                   if stamp < start and pid in memory]
        entries.append({
            "rank": rank,
            "pid": pid,
            "gpu_uuid": row["gpu_uuid"],
            "whole_process_peak_bytes": max(whole),
            "startup_peak_bytes": max(startup) if startup else None,
            "training_peak_bytes": max(training),
            "training_median_bytes": statistics.median(training),
            "training_samples": len(training),
            "required_training_samples": required_training,
            "sample_count": len(whole),
            "max_gap_seconds": rank_gap,
            "first_sample_epoch": first_sample_epoch,
            "last_sample_epoch": last_sample_epoch,
        })
    common_start = max(start for start, _ in windows.values())
    common_end = min(end for _, end in windows.values())
    _require(common_end > common_start,
             "per-rank training windows do not overlap")
    aggregate_samples = []
    for stamp, memory in batches:
        if not common_start <= stamp <= common_end:
            continue
        if not all(pid in memory for pid in by_pid):
            continue
        aggregate_samples.append(
            (stamp, sum(memory[pid] for pid in by_pid)))
    aggregate_stamps = [stamp for stamp, _ in aggregate_samples]
    aggregate_values = [value for _, value in aggregate_samples]
    aggregate_gaps = [later - earlier
                      for earlier, later in zip(aggregate_stamps,
                                                aggregate_stamps[1:])]
    aggregate_gap_max = max(aggregate_gaps) if aggregate_gaps else None
    required_aggregate = (_required_training_samples(common_start, common_end)
                          if require_density
                          else MIN_PREFLIGHT_TRAINING_SAMPLES)
    _require(len(aggregate_values) >= required_aggregate,
             "only %d simultaneous samples cover every rank inside the "
             "training window; need at least %d"
             % (len(aggregate_values), required_aggregate))
    aggregate_first = aggregate_stamps[0]
    aggregate_last = aggregate_stamps[-1]
    if require_density:
        _require(aggregate_first <= common_start + SAMPLE_COVERAGE_SLACK_SECONDS,
                 "simultaneous GPU samples start %.2f s after the common "
                 "training window" % (aggregate_first - common_start))
        _require(aggregate_last >= common_end - SAMPLE_COVERAGE_SLACK_SECONDS,
                 "simultaneous GPU samples end %.2f s before the common "
                 "training window" % (common_end - aggregate_last))

    gap_values = [value for value in
                  (trace_gap_max, rank_gap_max, aggregate_gap_max)
                  if value is not None]
    _require(gap_values, "GPU samples contain no comparable timestamp gaps")
    gap_max = max(gap_values)
    _require(gap_max <= MAX_SAMPLE_GAP_SECONDS,
             "GPU sampling gap of %.2f s exceeds the %.2f s limit "
             "(trace=%s, rank=%s, simultaneous=%s)"
             % (gap_max, MAX_SAMPLE_GAP_SECONDS,
                trace_gap_max, rank_gap_max, aggregate_gap_max))

    return {
        "ranks": entries,
        "whole_process_peak_bytes": max(entry["whole_process_peak_bytes"]
                                        for entry in entries),
        "training_peak_bytes": max(entry["training_peak_bytes"]
                                   for entry in entries),
        "aggregate_training_peak_bytes": max(aggregate_values),
        "aggregate_training_samples": len(aggregate_values),
        "required_aggregate_samples": required_aggregate,
        "aggregate_gap_max_seconds": aggregate_gap_max,
        "aggregate_first_sample_epoch": aggregate_first,
        "aggregate_last_sample_epoch": aggregate_last,
        "aggregate_window_start_epoch": common_start,
        "aggregate_window_end_epoch": common_end,
        "sample_count": samples,
        "gap_median_seconds": trace_gap_median,
        "trace_gap_max_seconds": trace_gap_max,
        "rank_gap_max_seconds": rank_gap_max,
        "gap_max_seconds": gap_max,
        "trace_start_epoch": min(stamp for stamp, _ in batches),
        "trace_end_epoch": max(stamp for stamp, _ in batches),
    }


def _record_training_windows(record, world):
    """Return rank -> (start_epoch, end_epoch) from a timing record."""
    if record.get("framework") == "flux":
        raw = record.get("rank_training_windows")
        start_key, end_key = "start_epoch", "end_epoch"
    else:
        raw = record.get("ranks")
        start_key, end_key = "training_start_epoch", "training_end_epoch"
    _require(isinstance(raw, list) and len(raw) == world,
             "record training windows mismatch")
    windows = {}
    for entry in raw:
        _require(isinstance(entry, dict), "record training window is not an object")
        rank = entry.get("rank")
        _require(isinstance(rank, int) and not isinstance(rank, bool)
                 and 0 <= rank < world,
                 "record training window rank is invalid")
        _require(rank not in windows, "duplicate record training window rank")
        start = entry.get(start_key)
        end = entry.get(end_key)
        _require(_finite_number(start) and _finite_number(end) and end > start,
                 "record training window is invalid for rank %d" % rank)
        windows[rank] = (float(start), float(end))
    _require(set(windows) == set(range(world)),
             "record training windows do not cover every rank")
    return windows


def _source_relative(raw):
    if "/source/" in raw:
        return raw.split("/source/", 1)[1]
    if raw.startswith("source/"):
        return raw[len("source/"):]
    return None


def _check_source_snapshot(root):
    root = Path(root)
    manifest = root / "source-sha256.txt"
    source_dir = root / "source"
    _require(source_dir.is_dir(), "missing source snapshot: %s" % source_dir)
    _require(manifest.is_file(), "missing source manifest: %s" % manifest)
    listed = {}
    for raw in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        match = SOURCE_LINE.fullmatch(raw)
        _require(match is not None, "malformed source manifest line: %r" % raw)
        digest, remote = match.groups()
        relative = _source_relative(remote)
        _require(relative is not None, "source path is not inside source/: %s" % remote)
        target = source_dir / relative
        _require(target.is_file(), "source snapshot file is missing: %s" % target)
        _require(_sha256(target) == digest, "source snapshot hash mismatch: %s" % target)
        listed[relative] = digest
    basenames = {Path(name).name for name in listed}
    missing = REQUIRED_SOURCE_FILES - basenames
    _require(not missing, "source snapshot is missing required files: %s" % sorted(missing))
    return listed


def _check_source_identity(root):
    path = Path(root) / "source-identity.txt"
    _require(path.is_file(), "missing source identity: %s" % path)
    identity = _parse_pairs(path.read_text(encoding="utf-8"))
    required = (
        "THESIS_COMMIT", "THESIS_COMMIT_SOURCE", "THESIS_STATE_SHA256",
        "THESIS_STATE_RECIPE",
        "CURRENT_COMMIT", "CURRENT_STATE_SHA256", "CURRENT_SOURCE_TREE_SHA256",
        "CURRENT_COMMIT_SOURCE", "CURRENT_STATE_SOURCE",
        "HISTORICAL_COMMIT", "HISTORICAL_STATE_SHA256",
        "HISTORICAL_SOURCE_TREE_SHA256", "HISTORICAL_COMMIT_SOURCE",
        "HISTORICAL_STATE_SOURCE",
    )
    for key in required:
        _require(key in identity and identity[key], "source identity is missing %s" % key)
    for key in ("THESIS_STATE_SHA256", "CURRENT_STATE_SHA256",
                "CURRENT_SOURCE_TREE_SHA256", "HISTORICAL_STATE_SHA256",
                "HISTORICAL_SOURCE_TREE_SHA256"):
        _require(_is_hex64(identity[key]), "source identity %s is not a SHA-256" % key)
    _require(identity["CURRENT_COMMIT"] != identity["HISTORICAL_COMMIT"],
             "current and historical Flux commits must differ")
    _require(identity["CURRENT_SOURCE_TREE_SHA256"]
             != identity["HISTORICAL_SOURCE_TREE_SHA256"],
             "current and historical Flux source trees must differ")
    expected = {
        "THESIS_COMMIT": THESIS_COMMIT,
        "CURRENT_COMMIT": CURRENT_COMMIT,
        "CURRENT_STATE_SHA256": CURRENT_STATE_SHA256,
        "CURRENT_SOURCE_TREE_SHA256": CURRENT_SOURCE_TREE_SHA256,
        "HISTORICAL_COMMIT": HISTORICAL_COMMIT,
        "HISTORICAL_STATE_SHA256": HISTORICAL_STATE_SHA256,
        "HISTORICAL_SOURCE_TREE_SHA256": HISTORICAL_SOURCE_TREE_SHA256,
    }
    for key, value in expected.items():
        _require(identity[key] == value,
                 "source identity %s does not match the accepted source" % key)
    return identity


def _check_runtime(root):
    root = Path(root)
    text_path = root / "runtime-env.txt"
    _require(text_path.is_file(), "missing runtime-env.txt")
    text = text_path.read_text(encoding="utf-8", errors="replace")
    _require("RUNTIME_ENV_CAPTURE_FAILED" not in text,
             "runtime capture failed")
    for marker in ("RUNTIME_JULIA", "RUNTIME_PYTHON", "RUNTIME_GPU"):
        _require(marker in text, "runtime-env.txt is missing %s" % marker)
    runtime_path = root / "runtime.json"
    runtime = _load_json(runtime_path)
    _require(runtime.get("schema") == RUNTIME_SCHEMA, "runtime.json schema mismatch")
    _require(runtime.get("julia_version"), "runtime.json has no Julia version")
    variants = runtime.get("variants")
    _require(isinstance(variants, dict), "runtime.json variants is not an object")
    for variant in ("flux-current", "flux-historical"):
        values = variants.get(variant)
        _require(isinstance(values, dict), "runtime.json is missing %s" % variant)
        for key in ("julia", "flux", "cuda", "nccl", "optimisers", "functors"):
            _require(values.get(key) not in (None, "", "unknown", "unavailable"),
                     "runtime.json %s.%s is unavailable" % (variant, key))
    python = runtime.get("python")
    _require(isinstance(python, dict), "runtime.json python is not an object")
    for key in ("python", "torch", "numpy", "cuda", "cudnn", "executable"):
        _require(python.get(key) not in (None, "", "unknown", "unavailable"),
                 "runtime.json python.%s is unavailable" % key)
    gpus = runtime.get("gpus")
    _require(isinstance(gpus, list) and gpus, "runtime.json has no GPU mapping")
    for gpu in gpus:
        _require(isinstance(gpu, dict), "runtime.json GPU entry is not an object")
        _require(isinstance(gpu.get("index"), int), "runtime.json GPU index is invalid")
        _require(isinstance(gpu.get("total_bytes"), int) and gpu["total_bytes"] > 0,
                 "runtime.json GPU total is invalid")
        _require(gpu.get("uuid"), "runtime.json GPU UUID is missing")
    _require(runtime.get("driver_version"), "runtime.json has no driver version")
    return runtime


def _check_resources(root, job, node):
    root = Path(root)
    text_path = root / "resources.txt"
    _require(text_path.is_file(), "missing resources.txt")
    resources_path = root / "resources.json"
    resources = _load_json(resources_path)
    _require(resources.get("schema") == RESOURCES_SCHEMA, "resources.json schema mismatch")
    _require(resources.get("job") == job, "resources.json job mismatch")
    _require(resources.get("node") == node, "resources.json node mismatch")
    for key in ("partition", "qos", "account", "gres", "time_limit"):
        _require(resources.get(key) not in (None, ""), "resources.json is missing %s" % key)
    _require(isinstance(resources.get("ntasks"), int) and resources["ntasks"] >= 1,
             "resources.json ntasks is invalid")
    _require(isinstance(resources.get("cpus_per_task"), int)
             and resources["cpus_per_task"] >= 1,
             "resources.json cpus_per_task is invalid")
    _require(isinstance(resources.get("gpu_mapping"), list) and resources["gpu_mapping"],
             "resources.json has no GPU mapping")
    return resources


def _check_launcher(root, provenance):
    path = Path(root) / "launcher-sha256.txt"
    _require(path.is_file(), "missing launcher-sha256.txt")
    listed = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SOURCE_LINE.fullmatch(raw.strip())
        if match is not None:
            listed[Path(match.group(2)).name] = match.group(1)
    for key in ("launcher", "checker"):
        entry = provenance.get(key)
        _require(isinstance(entry, dict), "provenance is missing %s identity" % key)
        digest = entry.get("sha256")
        _require(_is_hex64(digest), "provenance %s hash is invalid" % key)
        name = entry.get("name")
        _require(isinstance(name, str) and name in listed,
                 "provenance %s name is not in launcher-sha256.txt" % key)
        _require(listed[name] == digest, "provenance %s hash does not match" % key)
    return listed


def _check_execution_policy(policy):
    _require(isinstance(policy, dict), "execution_policy is not an object")
    flux = policy.get("flux")
    _require(isinstance(flux, dict), "execution_policy.flux is missing")
    _require(flux.get("precision") == FLUX_PRECISION, "Flux precision policy mismatch")
    _require(flux.get("cudnn_deterministic") is False,
             "Flux must record non-deterministic algorithm selection")
    _require(flux.get("nonfinite_guard") == FLUX_GUARD,
             "Flux guard policy mismatch")
    pytorch = policy.get("pytorch")
    _require(isinstance(pytorch, dict), "execution_policy.pytorch is missing")
    for key, value in PYTORCH_PRECISION.items():
        _require(pytorch.get(key) == value, "PyTorch policy %s mismatch" % key)
    _require(pytorch.get("nonfinite_guard") == "outside timed loop",
             "PyTorch guard policy mismatch")


def _check_provenance(root, config, identity):
    path = Path(root) / "provenance.json"
    provenance = _load_json(path)
    _require(provenance.get("schema") == PROVENANCE_SCHEMA,
             "provenance schema mismatch")
    _require(provenance.get("source_identity") == identity,
             "provenance source identity differs from source-identity.txt")
    _require(provenance.get("campaign_config") == config,
             "provenance campaign config differs from campaign-config.json")
    _require(provenance.get("NUMERICAL_EQUIVALENCE_CLAIM") == "none",
             "provenance must retain NUMERICAL_EQUIVALENCE_CLAIM none")
    qualifications = provenance.get("qualifications")
    _require(isinstance(qualifications, list) and qualifications,
             "provenance qualifications must be a non-empty list")
    _check_execution_policy(provenance.get("execution_policy"))
    for key in ("source_snapshot_manifest_sha256", "runtime_env_sha256",
                "runtime_json_sha256", "dataset_identity_sha256",
                "resources_json_sha256", "preflight_decision_sha256"):
        _require(_is_hex64(provenance.get(key)), "provenance %s is invalid" % key)
    _require(provenance["source_snapshot_manifest_sha256"]
             == _sha256(Path(root) / "source-sha256.txt"),
             "source snapshot manifest hash mismatch")
    _require(provenance["runtime_env_sha256"]
             == _sha256(Path(root) / "runtime-env.txt"),
             "runtime-env.txt hash mismatch")
    _require(provenance["runtime_json_sha256"]
             == _sha256(Path(root) / "runtime.json"),
             "runtime.json hash mismatch")
    _require(provenance["dataset_identity_sha256"]
             == _sha256(Path(root) / "dataset-identity.json"),
             "dataset-identity.json hash mismatch")
    _require(provenance["resources_json_sha256"]
             == _sha256(Path(root) / "resources.json"),
             "resources.json hash mismatch")
    _require(provenance["preflight_decision_sha256"]
             == _sha256(Path(root) / "preflight" / "decision.json"),
             "preflight decision hash mismatch")
    return provenance


def _check_dataset_identity(root, config):
    path = Path(root) / "dataset-identity.json"
    dataset = _load_json(path)
    _require(dataset.get("schema") == DATASET_SCHEMA,
             "dataset identity schema mismatch")
    files = dataset.get("files")
    _require(isinstance(files, dict), "dataset identity files is not an object")
    for name in REQUIRED_DATASET_FILES:
        _require(name in files, "dataset identity is missing %s" % name)
        _require(_is_hex64(files[name].get("sha256")),
                 "dataset identity %s hash is invalid" % name)
    bundle = dataset.get("bundle")
    _require(isinstance(bundle, dict), "dataset identity bundle is missing")
    _require(_is_hex64(bundle.get("bundle_hash")), "bundle hash is invalid")
    _require(_is_hex64(bundle.get("bundle_file_sha256")), "bundle file hash is invalid")
    _require(_is_hex64(dataset.get("transform_spec_sha256")),
             "transform spec hash is invalid")
    workloads = dataset.get("workloads")
    _require(isinstance(workloads, dict), "dataset identity workloads is missing")
    for batch in BATCHES:
        entry = workloads.get(str(batch))
        _require(isinstance(entry, dict), "dataset identity is missing batch %d" % batch)
        _require(entry.get("updates_per_epoch") == expected_updates(batch),
                 "dataset identity batch %d update count mismatch" % batch)
        _require(entry.get("n_eff") == N_EFF, "dataset identity n_eff mismatch")
        _require(entry.get("tail_size") == TAIL, "dataset identity tail mismatch")
        for key in ("sample_order_sha256", "update1_order_sha256"):
            _require(_is_hex64(entry.get(key)), "dataset identity %s is invalid" % key)
    return dataset


def _check_config(root):
    config = _load_json(Path(root) / "campaign-config.json")
    _require(config.get("schema") == CONFIG_SCHEMA, "campaign config schema mismatch")
    batch = config.get("batch")
    _require(batch in BATCHES, "campaign batch must be 512 or 256")
    _require(config.get("updates") == expected_updates(batch),
             "campaign update count does not match the selected batch")
    _require(isinstance(config.get("epoch"), int) and 0 <= config["epoch"] <= 49,
             "campaign epoch is invalid")
    _require(isinstance(config.get("warmup"), int) and 2 <= config["warmup"] <= 100,
             "campaign warmup is invalid")
    _require(isinstance(config.get("repeats"), int) and config["repeats"] >= 1,
             "campaign repeats is invalid")
    _require(config.get("worlds") == list(WORLDS), "campaign worlds mismatch")
    _require(config.get("frameworks") == list(FRAMEWORKS),
             "campaign frameworks mismatch")
    _require(isinstance(config.get("preflight_updates"), int)
             and config["preflight_updates"] >= 1,
             "campaign preflight update count is invalid")
    _require(isinstance(config.get("preflight_warmup"), int)
             and config["preflight_warmup"] >= 2,
             "campaign preflight warmup is invalid")
    return config


def _check_preflight_decision(root, config, identity, dataset):
    path = Path(root) / "preflight" / "decision.json"
    decision = _load_json(path)
    _require(decision.get("schema") == PREFLIGHT_SCHEMA,
             "preflight decision schema mismatch")
    _require(decision.get("selected_global_batch") == config["batch"],
             "campaign batch differs from the preflight decision")
    _require(decision.get("updates") == expected_updates(config["batch"]),
             "preflight decision update count mismatch")
    _require(decision.get("preflight_updates") == config["preflight_updates"],
             "preflight decision sustained-update count mismatch")
    _require(decision.get("preflight_warmup") == config["preflight_warmup"],
             "preflight decision warmup mismatch")
    _require(decision.get("source_identity") == identity,
             "preflight decision source identity differs")
    _require(decision.get("dataset_identity_sha256")
             == _sha256(Path(root) / "dataset-identity.json"),
             "preflight decision dataset identity differs")
    candidates = decision.get("candidates")
    _require(isinstance(candidates, dict), "preflight decision candidates is missing")
    selected = candidates.get(str(config["batch"]))
    _require(isinstance(selected, dict), "preflight decision lacks the selected batch")
    _require(selected.get("status") == "PASS",
             "selected preflight batch is not PASS")
    _require(selected.get("batch") == config["batch"],
             "preflight candidate batch mismatch")
    return decision


def _check_policy_fields(record, framework, world):
    if framework.startswith("flux"):
        _require(record.get("precision") == FLUX_PRECISION, "Flux precision mismatch")
        if world == 1:
            expected = ("single_gpu", "single_gpu", SINGLE_GUARD)
        else:
            expected = {
                "flux-current": ("guarded", "guarded_ddp", FLUX_GUARD),
                "flux-historical": ("static", "historical_static_ddp",
                                    FLUX_STATIC_GUARD),
            }[framework]
        _require(record.get("distributed_api") == expected[0],
                 "Flux distributed API mismatch")
        _require(record.get("config_kind") == expected[1],
                 "Flux config kind mismatch")
        _require(record.get("guard_policy") == expected[2],
                 "Flux guard policy mismatch")
    else:
        _require(record.get("precision") == PYTORCH_PRECISION,
                 "PyTorch precision mismatch")
    loader = record.get("loader_policy")
    _require(isinstance(loader, str) and "mmap" in loader,
             "loader policy does not record mmap")
    timing = record.get("timing_policy")
    _require(isinstance(timing, str) and "timed region includes" in timing,
             "timing policy is missing the timed-region description")
    for token in ("forward", "backward", "optimizer step"):
        _require(token in timing, "timing policy does not mention %s" % token)
    warmup = record.get("warmup_policy")
    _require(isinstance(warmup, str) and "reset" in warmup,
             "warmup policy does not record the reset")
    _require("update 1" in warmup, "warmup policy does not record update 1 restart")


def _check_record(path, framework, world, expected_updates_value, expected_warmup,
                  expected_epoch, batch, identity, dataset, snapshot_expected):
    record = _load_json(path)
    kind = _framework_kind(framework)
    _require(record.get("schema") == SCHEMA, "record schema mismatch")
    _require(record.get("framework") == kind, "record framework mismatch")
    _require(record.get("world_size") == world, "record world_size mismatch")
    _require(record.get("epoch") == expected_epoch, "record epoch mismatch")
    _require(record.get("batch_size") == batch, "record batch_size mismatch")
    _require(record.get("global_batch") == batch, "record global_batch mismatch")
    _require(record.get("updates") == expected_updates_value,
             "record update count mismatch")
    _require(record.get("warmup") == expected_warmup, "record warmup mismatch")
    _require(_is_hex64(record.get("eta_sha256")),
             "record eta_sha256 is not a SHA-256")
    _require(record.get("updates_per_epoch") == expected_updates(batch),
             "record updates_per_epoch mismatch")
    _require(record.get("n_eff") == N_EFF, "record n_eff mismatch")
    _require(record.get("tail_size") == TAIL, "record tail_size mismatch")
    _require(record.get("snapshot") is snapshot_expected, "record snapshot flag mismatch")
    _require(record.get("timing") == "full-dataset-end-to-end",
             "record timing view mismatch")
    _require(record.get("workload") == "full-dataset-epoch",
             "record workload mismatch")
    _require(record.get("pass_checks") is True, "record pass_checks is not true")
    _require(record.get("loss_finite") is True, "record loss_finite is not true")
    _require(_finite_number(record.get("final_loss")), "record final_loss is invalid")
    _require(bool(record.get("job")), "record job is missing")
    _require(bool(record.get("node")), "record node is missing")
    _check_policy_fields(record, framework, world)

    seconds = record.get("seconds")
    _require(_finite_number(seconds) and seconds > 0, "record seconds is invalid")
    if kind == "flux":
        durations = record.get("rank_seconds")
        _require(isinstance(durations, list) and len(durations) == world,
                 "record rank_seconds mismatch")
    else:
        ranks = record.get("ranks")
        _require(isinstance(ranks, list) and len(ranks) == world,
                 "record ranks mismatch")
        _require(sorted(rank.get("rank") for rank in ranks) == list(range(world)),
                 "record rank set mismatch")
        _require(len({rank.get("device") for rank in ranks}) == world,
                 "record device mapping is not one-to-one")
        durations = [rank.get("seconds") for rank in ranks]
    _require(all(_finite_number(value) and value > 0 for value in durations),
             "record rank duration is invalid")
    _require(math.isclose(max(durations), seconds, rel_tol=1e-12),
             "record seconds is not the maximum rank duration")
    _require(math.isclose(record.get("images_per_second"),
                          batch * expected_updates_value / seconds, rel_tol=1e-12),
             "record images_per_second is inconsistent")
    _require(math.isclose(record.get("update_seconds"),
                          seconds / expected_updates_value, rel_tol=1e-12),
             "record update_seconds is inconsistent")
    duration_by_rank = (dict(enumerate(durations)) if kind == "flux"
                        else {rank["rank"]: rank["seconds"] for rank in ranks})
    windows = _record_training_windows(record, world)
    for rank, (window_start, window_end) in windows.items():
        duration = duration_by_rank[rank]
        _require(abs((window_end - window_start) - duration)
                 <= max(0.02 * duration, 1.0),
                 "record training window for rank %d does not match its "
                 "measured duration" % rank)

    source = record.get("source")
    _require(isinstance(source, dict), "record source is missing")
    _require(source.get("thesis_commit") == identity["THESIS_COMMIT"],
             "record thesis commit differs")
    _require(source.get("thesis_state_sha256") == identity["THESIS_STATE_SHA256"],
             "record thesis state differs")
    if kind == "flux":
        variant = _flux_variant(framework)
        prefix = "CURRENT" if variant == "flux-current" else "HISTORICAL"
        _require(record.get("flux_commit") == identity[prefix + "_COMMIT"],
                 "record Flux commit differs")
        _require(record.get("flux_state_sha256") == identity[prefix + "_STATE_SHA256"],
                 "record Flux state differs")
        _require(record.get("source", {}).get("flux_commit")
                 == identity[prefix + "_COMMIT"],
                 "record nested Flux commit differs")
        _require(record.get("source", {}).get("flux_state_sha256")
                 == identity[prefix + "_STATE_SHA256"],
                 "record nested Flux state differs")
        _require(isinstance(record.get("device"), int) and record["device"] >= 0,
                 "record Flux device is invalid")
    else:
        _require(source.get("python_version") not in (None, ""),
                 "record Python version is missing")
        _require(source.get("torch_version") not in (None, ""),
                 "record torch version is missing")
        _require(source.get("numpy_version") not in (None, ""),
                 "record numpy version is missing")

    hashes = _normalise_dataset_hashes(record)
    for name in REQUIRED_DATASET_FILES:
        _require(name in hashes, "record dataset hash is missing %s" % name)
        _require(hashes[name] == dataset["files"][name]["sha256"],
                 "record dataset hash differs for %s" % name)
    _require(record.get("bundle_hash") == dataset["bundle"]["bundle_hash"],
             "record bundle hash differs")
    _require(record.get("bundle_file_sha256")
             == dataset["bundle"]["bundle_file_sha256"],
             "record bundle file hash differs")
    workload = dataset["workloads"][str(batch)]
    _require(record.get("sample_order_sha256") == workload["sample_order_sha256"],
             "record sample order hash differs")
    _require(record.get("update1_order_sha256") == workload["update1_order_sha256"],
             "record update-1 order hash differs")
    _require(record.get("transform_spec_sha256") == dataset["transform_spec_sha256"],
             "record transform spec hash differs")

    if kind == "flux":
        keys = record.get("hashes")
        _require(isinstance(keys, dict)
                 and set(keys) == {"param", "m", "v", "beta_power"},
                 "record Flux state hash keys mismatch")
        for digest in keys.values():
            _require(_is_hex64(digest), "record Flux state hash is invalid")
        rank_hashes = record.get("rank_hashes")
        _require(isinstance(rank_hashes, dict)
                 and set(rank_hashes) == {"param", "m", "v", "beta_power"},
                 "record rank_hashes keys mismatch")
        for key, values in rank_hashes.items():
            _require(isinstance(values, list) and len(values) == world,
                     "record rank_hashes length mismatch for %s" % key)
            _require(len(set(values)) == 1, "record replicas differ for %s" % key)
            _require(values[0] == keys[key],
                     "record rank_hashes value differs for %s" % key)
    else:
        ranks = record["ranks"]
        first = ranks[0].get("hashes")
        _require(isinstance(first, dict)
                 and set(first) == {"param", "m", "v", "step"},
                 "record PyTorch state hash keys mismatch")
        _require(first.get("step") == expected_updates_value,
                 "record optimizer step mismatch")
        for key in ("param", "m", "v"):
            _require(_is_hex64(first[key]), "record PyTorch state hash is invalid")
        _require(all(rank.get("hashes") == first for rank in ranks),
                 "record PyTorch replicas differ")
    _record_total_bytes(record)
    return record


def _check_record_identity(records):
    if not records:
        return
    reference = records[0]
    keys = (
        "job", "node", "epoch", "batch_size", "global_batch", "updates",
        "warmup", "updates_per_epoch", "n_eff", "tail_size", "eta_sha256",
        "sample_order_sha256", "update1_order_sha256", "transform_spec_sha256",
        "bundle_hash", "bundle_file_sha256",
    )
    for record in records[1:]:
        for key in keys:
            _require(record.get(key) == reference.get(key),
                     "record %s differs across the campaign" % key)
        _require(record.get("source", {}).get("thesis_commit")
                 == reference.get("source", {}).get("thesis_commit"),
                 "record thesis commit differs across the campaign")
        _require(_normalise_dataset_hashes(record)
                 == _normalise_dataset_hashes(reference),
                 "record dataset hashes differ across the campaign")


def _check_runtime_versions(records, runtime):
    for key, record in records.items():
        framework = key[0]
        world = key[1]
        label = "%s-w%d" % (framework, world)
        if len(key) > 2:
            label += "-r%d" % key[2]
        source = record.get("source")
        _require(isinstance(source, dict), "%s record source is missing" % label)
        if framework.startswith("flux"):
            variant = _flux_variant(framework)
            variants = runtime.get("variants")
            _require(isinstance(variants, dict)
                     and isinstance(variants.get(variant), dict),
                     "runtime.json is missing variants.%s" % variant)
            expected = variants[variant].get("julia")
            actual = source.get("julia_version")
            _require(actual == expected,
                     "%s source.julia_version %r does not match "
                     "runtime variants.%s.julia %r"
                     % (label, actual, variant, expected))
        else:
            python = runtime.get("python")
            _require(isinstance(python, dict),
                     "runtime.json python is not an object")
            for field, runtime_key in (("python_version", "python"),
                                       ("torch_version", "torch"),
                                       ("numpy_version", "numpy")):
                expected = python.get(runtime_key)
                actual = source.get(field)
                _require(actual == expected,
                         "%s source.%s %r does not match runtime python.%s %r"
                         % (label, field, actual, runtime_key, expected))


def _check_record_files(directory, expected, label):
    directory = Path(directory)
    _require(directory.is_dir(), "missing %s record directory: %s"
             % (label, directory))
    actual = {path.name for path in directory.glob("*.json") if path.is_file()}
    missing = expected - actual
    unexpected = actual - expected
    _require(not missing, "%s records are missing: %s"
             % (label, sorted(missing)))
    _require(not unexpected, "unexpected %s records: %s"
             % (label, sorted(unexpected)))


def _evaluate_preflight_candidate(batch, root, updates, warmup, epoch,
                                  identity, dataset):
    root = Path(root)
    errors = []
    rows = []
    records = []
    for framework in FRAMEWORKS:
        for world in WORLDS:
            label = "%s.w%d" % (framework, world)
            record_path = root / "records" / ("%s-w%d.json" % (framework, world))
            sample_path = root / "samples" / ("%s-w%d.csv" % (framework, world))
            row = {
                "framework": framework,
                "world_size": world,
                "status": "INCOMPLETE",
                "peak_process_bytes": None,
                "training_peak_process_bytes": None,
                "aggregate_training_peak_bytes": None,
                "max_sample_gap_seconds": None,
                "limit_bytes": None,
                "margin_bytes": None,
                "sample_count": 0,
                "process_peaks": [],
            }
            if not record_path.is_file():
                errors.append("%s:record_missing" % label)
                rows.append(row)
                continue
            try:
                record = _check_record(
                    record_path, framework, world, updates, warmup, epoch,
                    batch, identity, dataset, False)
                records.append(record)
                total = _record_total_bytes(record)
                windows = _record_training_windows(record, world)
                row["device_total_bytes"] = total
                row["limit_bytes"] = int(0.8 * total)
                memory = (record.get("memory") if record.get("framework") == "flux"
                          else record["ranks"][0]["memory"])
                row["allocator_record"] = memory
            except AssertionError as exc:
                errors.append("%s:%s" % (label, exc))
                rows.append(row)
                continue
            try:
                summary = _summarize_gpu_samples(sample_path, world, windows)
            except AssertionError as exc:
                errors.append("%s:%s" % (label, exc))
                rows.append(row)
                continue
            row["sample_count"] = summary["sample_count"]
            row["process_peaks"] = summary["ranks"]
            row["training_peak_process_bytes"] = summary["training_peak_bytes"]
            row["aggregate_training_peak_bytes"] = summary["aggregate_training_peak_bytes"]
            row["max_sample_gap_seconds"] = summary["gap_max_seconds"]
            peak = summary["whole_process_peak_bytes"]
            row["peak_process_bytes"] = peak
            row["margin_bytes"] = row["device_total_bytes"] - peak
            if peak > row["limit_bytes"]:
                errors.append("%s:over_memory_limit:%d>%d"
                              % (label, peak, row["limit_bytes"]))
                row["status"] = "FAIL"
            else:
                row["status"] = "PASS"
            rows.append(row)
    _check_record_identity(records)
    complete = len(records) == len(FRAMEWORKS) * len(WORLDS) and not errors
    passed = complete and all(row["status"] == "PASS" for row in rows)
    return {
        "batch": batch,
        "root": str(root),
        "status": "PASS" if passed else ("INCOMPLETE" if not complete else "FAIL"),
        "complete": complete,
        "rows": rows,
        "errors": errors,
    }


def _print_preflight(result):
    print("PREFLIGHT_BATCH %d %s root=%s"
          % (result["batch"], result["status"], result["root"]))
    print("framework W status peak_process_MiB training_peak_MiB "
          "limit_MiB margin_MiB samples train_samples")
    for row in result["rows"]:
        peak = row["peak_process_bytes"]
        training_peak = row.get("training_peak_process_bytes")
        limit = row["limit_bytes"]
        margin = row["margin_bytes"]
        training_samples = min((entry.get("training_samples", 0)
                                for entry in row.get("process_peaks", [])),
                               default=0)
        print("%s %d %s %s %s %s %s %d %d" % (
            row["framework"], row["world_size"], row["status"],
            "NA" if peak is None else "%.1f" % (peak / 2**20),
            "NA" if training_peak is None else "%.1f" % (training_peak / 2**20),
            "NA" if limit is None else "%.1f" % (limit / 2**20),
            "NA" if margin is None else "%.1f" % (margin / 2**20),
            row["sample_count"], training_samples,
        ))
    for error in result["errors"]:
        print("PREFLIGHT_ERROR " + error)


def _preflight(args):
    identity = _check_source_identity(args.root)
    dataset = _check_dataset_identity(args.root, None)
    runtime = _check_runtime(args.root)
    resources = _load_json(Path(args.root) / "resources.json")
    candidates = {}
    errors = []
    for batch, candidate_root in args.candidate:
        result = _evaluate_preflight_candidate(
            batch, candidate_root, args.updates, args.warmup, args.epoch,
            identity, dataset)
        candidates[str(batch)] = result
        errors.extend(result["errors"])
        _print_preflight(result)
    selected = None
    reason = "no candidate passes the OD-8 80% peak-process memory criterion"
    for batch in BATCHES:
        result = candidates.get(str(batch))
        if result is not None and result["status"] == "PASS":
            selected = batch
            reason = "batch %d is PASS in all nine framework/world configurations" % batch
            break
    decision = {
        "schema": PREFLIGHT_SCHEMA,
        "selected_global_batch": selected,
        "updates": expected_updates(selected) if selected is not None else None,
        "preflight_updates": args.updates,
        "preflight_warmup": args.warmup,
        "epoch": args.epoch,
        "source_identity": identity,
        "runtime": runtime,
        "resources": resources,
        "dataset_identity_sha256": _sha256(Path(args.root) / "dataset-identity.json"),
        "candidates": candidates,
        "errors": errors,
        "reason": reason,
        "NUMERICAL_EQUIVALENCE_CLAIM": "none",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print("PREFLIGHT_SELECTED %s" % (selected if selected is not None else "none"))
    print("PREFLIGHT_DECISION_WRITTEN %s" % out)
    return 0 if selected is not None else 1


def _check_ready(args):
    """Fail before timing, including when a separate job reuses a decision."""
    root = Path(args.root)
    config = _check_config(root)
    _require(config["batch"] == 256, "this campaign is pinned to batch 256")
    identity = _check_source_identity(root)
    dataset = _check_dataset_identity(root, config)
    _check_source_snapshot(root)
    runtime = _check_runtime(root)
    provenance = _check_provenance(root, config, identity)
    _check_launcher(root, provenance)
    decision = _check_preflight_decision(root, config, identity, dataset)
    _require(decision.get("epoch") == config["epoch"], "preflight epoch differs")
    old_runtime = decision.get("runtime", {})
    for key in ("variants", "python", "driver_version", "julia_version"):
        _require(old_runtime.get(key) == runtime.get(key), "preflight runtime differs: " + key)
    hardware = lambda doc: sorted((gpu["name"], gpu["total_bytes"])
                                  for gpu in doc.get("gpus", []))
    _require(len(runtime["gpus"]) == 4 and hardware(old_runtime) == hardware(runtime),
             "preflight GPU hardware differs")
    resources = _load_json(root / "resources.json")
    _check_resources(root, resources["job"], resources["node"])
    for key in ("ntasks", "cpus_per_task"):
        _require(decision.get("resources", {}).get(key) == resources[key],
                 "preflight resources differ: " + key)
    _require(resources["ntasks"] == 1
             and resources["cpus_per_task"] == CAMPAIGN_CPUS,
             "campaign requires one task and %d CPUs" % CAMPAIGN_CPUS)
    selected = decision["candidates"]["256"]
    rows = selected.get("rows", [])
    _require(len(rows) == 9 and {(r["framework"], r["world_size"]) for r in rows}
             == {(f, w) for f in FRAMEWORKS for w in WORLDS}, "incomplete preflight matrix")
    for row in rows:
        _require(row["status"] == "PASS" and row["peak_process_bytes"] <= row["limit_bytes"],
                 "preflight memory did not pass")
        _require(_finite_number(row.get("training_peak_process_bytes")),
                 "preflight row lacks a training-window peak")
        _require(row["training_peak_process_bytes"] <= row["limit_bytes"],
                 "preflight training-window peak did not pass")
        peaks = row.get("process_peaks", [])
        _require(len(peaks) == row["world_size"],
                 "preflight lacks per-rank sample coverage")
        _require(all(p.get("training_samples", 0) >= MIN_PREFLIGHT_TRAINING_SAMPLES
                     for p in peaks),
                 "preflight lacks per-rank training-window sample coverage")
    print("FULL_CAMPAIGN_READY PASS batch=256 updates=390")
    return 0


def _load_snapshot(root, framework, world):
    path = Path(root) / "diagnostics" / "snapshots" / ("%s-w%d" % (framework, world)) / "state.npz"
    _require(path.is_file(), "missing diagnostic snapshot: %s" % path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    _require(arrays, "diagnostic snapshot is empty: %s" % path)
    for name, array in arrays.items():
        _require(np.issubdtype(array.dtype, np.number),
                 "diagnostic array is not numeric: %s" % name)
        _require(np.all(np.isfinite(array)),
                 "diagnostic array is not finite: %s" % name)
    return arrays


def _expected_beta_power(updates):
    b1 = np.float32(0.9)
    b2 = np.float32(0.999)
    for _ in range(updates):
        b1 = np.float32(b1 * np.float32(0.9))
        b2 = np.float32(b2 * np.float32(0.999))
    return np.array([b1, b2], dtype=np.float32)


def _check_snapshot_structure(framework, arrays, updates):
    kind = _framework_kind(framework)
    param_names = sorted(name[len("param."):] for name in arrays
                         if name.startswith("param."))
    _require(param_names, "snapshot has no parameters: %s" % framework)
    for category in ("m", "v"):
        names = sorted(name[len(category) + 1:] for name in arrays
                       if name.startswith(category + "."))
        _require(names == param_names,
                 "snapshot %s arrays do not match the parameter set" % category)
    _require("step" in arrays, "snapshot has no step array")
    _require(int(np.asarray(arrays["step"]).reshape(-1)[0]) == updates,
             "snapshot step does not match the update count")
    if kind == "flux":
        names = sorted(name[len("beta_power."):] for name in arrays
                       if name.startswith("beta_power."))
        _require(names == param_names,
                 "snapshot beta_power arrays do not match the parameter set")
        expected = _expected_beta_power(updates)
        for name in param_names:
            _require(np.array_equal(arrays["beta_power." + name], expected),
                     "snapshot beta_power differs for %s" % name)
        allowed = {"param", "m", "v", "beta_power", "step"}
    else:
        allowed = {"param", "m", "v", "step"}
    categories = {name.split(".", 1)[0] for name in arrays}
    extra = categories - allowed
    _require(not extra, "snapshot has unexpected categories: %s" % sorted(extra))
    return param_names


def _compare_states(reference, candidate, categories, label):
    rows = []
    diff_chunks = []
    ref_chunks = []
    for category in categories:
        keys = sorted(
            name for name in reference
            if name.startswith(category + ".") and name in candidate
        )
        _require(keys, "no %s arrays to compare in %s" % (category, label))
        category_diff = []
        category_ref = []
        max_abs = 0.0
        exact = 0
        worst_leaf = ""
        worst_rel = -1.0
        worst_leaf_max = 0.0
        for name in keys:
            ref = np.asarray(reference[name], dtype=np.float64)
            got = np.asarray(candidate[name], dtype=np.float64)
            _require(ref.shape == got.shape,
                     "shape mismatch for %s in %s" % (name, label))
            diff = got - ref
            leaf_max = float(np.max(np.abs(diff))) if diff.size else 0.0
            denom = float(np.linalg.norm(ref))
            leaf_rel = (float(np.linalg.norm(diff)) / denom if denom > 0
                        else (0.0 if leaf_max == 0 else float("inf")))
            if leaf_rel > worst_rel or (leaf_rel == worst_rel
                                        and leaf_max > worst_leaf_max):
                worst_rel = leaf_rel
                worst_leaf = name
                worst_leaf_max = leaf_max
            max_abs = max(max_abs, leaf_max)
            exact += int(np.array_equal(ref, got))
            category_diff.append(diff.reshape(-1))
            category_ref.append(ref.reshape(-1))
        diff_flat = np.concatenate(category_diff)
        ref_flat = np.concatenate(category_ref)
        denom = float(np.linalg.norm(ref_flat))
        rel_l2 = (float(np.linalg.norm(diff_flat)) / denom if denom > 0
                  else (0.0 if max_abs == 0 else float("inf")))
        rows.append({
            "category": category,
            "arrays": len(keys),
            "exact_arrays": exact,
            "max_abs": max_abs,
            "rel_l2": rel_l2,
            "worst_leaf": worst_leaf,
            "worst_leaf_rel_l2": worst_rel,
            "worst_leaf_max_abs": worst_leaf_max,
        })
        diff_chunks.append(diff_flat)
        ref_chunks.append(ref_flat)
    all_diff = np.concatenate(diff_chunks)
    all_ref = np.concatenate(ref_chunks)
    denom = float(np.linalg.norm(all_ref))
    overall = {
        "label": label,
        "categories": rows,
        "max_abs": max(row["max_abs"] for row in rows),
        "rel_l2": (float(np.linalg.norm(all_diff)) / denom if denom > 0
                   else 0.0),
        "exact": all(row["exact_arrays"] == row["arrays"] for row in rows),
    }
    return overall


def _check_diagnostics(root, config, identity, dataset, timing_records, runtime):
    root = Path(root)
    updates = config["updates"]
    _check_record_files(
        root / "diagnostics" / "records",
        {"%s-w%d.json" % (framework, world)
         for framework in FRAMEWORKS for world in WORLDS},
        "diagnostic")
    records = {}
    for framework in FRAMEWORKS:
        for world in WORLDS:
            path = root / "diagnostics" / "records" / ("%s-w%d.json" % (framework, world))
            record = _check_record(
                path, framework, world, updates, config["warmup"],
                config["epoch"], config["batch"], identity, dataset, True)
            records[framework, world] = record
    _check_record_identity(list(records.values()))
    _check_runtime_versions(records, runtime)
    _require(len({record.get("node") for record in records.values()}) == 1,
             "diagnostic node differs")

    timing_identity = {
        (framework, world): {
            "sample_order_sha256": timing_records[framework, world].get("sample_order_sha256"),
            "update1_order_sha256": timing_records[framework, world].get("update1_order_sha256"),
            "bundle_hash": timing_records[framework, world].get("bundle_hash"),
            "eta_sha256": timing_records[framework, world].get("eta_sha256"),
        }
        for framework in FRAMEWORKS for world in WORLDS
    }
    for key, record in records.items():
        for field, value in timing_identity[key].items():
            _require(record.get(field) == value,
                     "diagnostic %s differs from timing %s" % (key, field))

    states = {}
    for framework in FRAMEWORKS:
        for world in WORLDS:
            arrays = _load_snapshot(root, framework, world)
            _check_snapshot_structure(framework, arrays, updates)
            states[framework, world] = arrays

    reference_names = {
        framework: sorted(name[len("param."):] for name in states[framework, 1]
                          if name.startswith("param."))
        for framework in FRAMEWORKS
    }
    _require(reference_names["flux-current"] == reference_names["flux-historical"],
             "Flux variants have different parameter sets")
    _require(reference_names["flux-current"] == reference_names["pytorch"],
             "Flux and PyTorch have different parameter sets")

    comparisons = []
    for framework in FRAMEWORKS:
        categories = ("param", "m", "v", "beta_power") if framework.startswith("flux") else ("param", "m", "v")
        for world in (2, 4):
            comparisons.append(_compare_states(
                states[framework, 1], states[framework, world], categories,
                "%s-w%d-vs-w1" % (framework, world)))
    for framework in ("flux-current", "flux-historical"):
        for world in WORLDS:
            comparisons.append(_compare_states(
                states["pytorch", world], states[framework, world],
                ("param", "m", "v"), "%s-vs-pytorch-w%d" % (framework, world)))
    for world in WORLDS:
        comparisons.append(_compare_states(
            states["flux-historical", world], states["flux-current", world],
            ("param", "m", "v", "beta_power"),
            "flux-current-vs-flux-historical-w%d" % world))

    return {
        "updates": updates,
        "records": {
            "%s-w%d" % key: {
                "job": value.get("job"),
                "node": value.get("node"),
                "rank_hashes": value.get("rank_hashes") if value.get("framework") == "flux" else [rank.get("hashes") for rank in value.get("ranks", [])],
                "step": updates,
            }
            for key, value in records.items()
        },
        "comparisons": comparisons,
        "relative_l2_denominator": (
            "L2 norm of the concatenated reference arrays for each comparison "
            "category; per-leaf denominator is the leaf L2 norm. A zero "
            "denominator yields 0 when the difference norm is also zero and "
            "infinity otherwise."
        ),
        "NUMERICAL_EQUIVALENCE_CLAIM": "none",
        "threshold": None,
    }


def _timing_table(records, config):
    batch = config["batch"]
    updates = config["updates"]
    rows = []
    values = {}
    for framework in FRAMEWORKS:
        for world in WORLDS:
            samples = [records[framework, world, repeat]["seconds"]
                       for repeat in range(1, config["repeats"] + 1)]
            values[framework, world] = samples
    for framework in FRAMEWORKS:
        baseline = statistics.median(values[framework, 1])
        for world in WORLDS:
            samples = values[framework, world]
            median = statistics.median(samples)
            speedup = baseline / median if median > 0 else float("inf")
            rows.append({
                "framework": framework,
                "world_size": world,
                "repeats": len(samples),
                "median_s": median,
                "min_s": min(samples),
                "max_s": max(samples),
                "spread_s": max(samples) - min(samples),
                "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "images_per_second": batch * updates / median,
                "update_seconds": median / updates,
                "speedup": speedup,
                "efficiency": speedup / world,
            })
    return rows


def _check_memory(root, config, records):
    """Validate per-repeat GPU sampling and summarize sampled memory.

    Every repeat must have a parseable trace, complete per-rank PID/GPU
    registration, and enough samples inside each rank's recorded training
    window. The simultaneous sample stream must also span the common window
    without an over-limit internal gap. Missing coverage is an error, not a
    zero. Reported peaks are sampled values, not continuous high-water marks.
    """
    root = Path(root)
    rows = []
    totals = set()
    for framework in FRAMEWORKS:
        for world in WORLDS:
            per_repeat = []
            for repeat in range(1, config["repeats"] + 1):
                record = records[framework, world, repeat]
                total = _record_total_bytes(record)
                totals.add(total)
                windows = _record_training_windows(record, world)
                sample_path = root / "samples" / ("%s-w%d-r%d.csv"
                                                  % (framework, world, repeat))
                summary = _summarize_gpu_samples(sample_path, world, windows,
                                                 require_density=True)
                per_repeat.append({
                    "repeat": repeat,
                    "worst_rank_peak_bytes": summary["training_peak_bytes"],
                    "training_peak_bytes": summary["training_peak_bytes"],
                    "aggregate_training_peak_bytes":
                        summary["aggregate_training_peak_bytes"],
                    "aggregate_training_samples":
                        summary["aggregate_training_samples"],
                    "required_aggregate_samples":
                        summary["required_aggregate_samples"],
                    "aggregate_gap_max_seconds":
                        summary["aggregate_gap_max_seconds"],
                    "aggregate_first_sample_epoch":
                        summary["aggregate_first_sample_epoch"],
                    "aggregate_last_sample_epoch":
                        summary["aggregate_last_sample_epoch"],
                    "aggregate_window_start_epoch":
                        summary["aggregate_window_start_epoch"],
                    "aggregate_window_end_epoch":
                        summary["aggregate_window_end_epoch"],
                    "whole_process_peak_bytes": summary["whole_process_peak_bytes"],
                    "max_sample_gap_seconds": summary["gap_max_seconds"],
                    "trace_gap_max_seconds": summary["trace_gap_max_seconds"],
                    "rank_gap_max_seconds": summary["rank_gap_max_seconds"],
                    "median_sample_gap_seconds": summary["gap_median_seconds"],
                    "sample_count": summary["sample_count"],
                    "ranks": summary["ranks"],
                })
            rank_entries = {}
            for repeat_summary in per_repeat:
                for entry in repeat_summary["ranks"]:
                    rank_entries.setdefault(entry["rank"], []).append(entry)
            per_rank = []
            for rank in sorted(rank_entries):
                entries = rank_entries[rank]
                peaks = [entry["training_peak_bytes"] for entry in entries]
                medians = [entry["training_median_bytes"] for entry in entries]
                startup = [entry["startup_peak_bytes"] for entry in entries
                           if entry["startup_peak_bytes"] is not None]
                per_rank.append({
                    "rank": rank,
                    "pids": [entry["pid"] for entry in entries],
                    "gpu_uuids": [entry["gpu_uuid"] for entry in entries],
                    "training_peak_bytes": max(peaks),
                    "training_peak_median_bytes": int(round(statistics.median(peaks))),
                    "training_median_bytes": int(round(statistics.median(medians))),
                    "training_samples_min": min(entry["training_samples"]
                                                for entry in entries),
                    "startup_peak_bytes": max(startup) if startup else None,
                    "whole_process_peak_bytes": max(entry["whole_process_peak_bytes"]
                                                    for entry in entries),
                })
            run_peaks = [summary["worst_rank_peak_bytes"] for summary in per_repeat]
            aggregate = [summary["aggregate_training_peak_bytes"]
                         for summary in per_repeat]
            startup_peaks = [entry["startup_peak_bytes"] for entry in per_rank
                             if entry["startup_peak_bytes"] is not None]
            worst = max(per_rank, key=lambda entry: entry["training_peak_bytes"])
            limit = int(0.8 * total)
            rows.append({
                "framework": framework,
                "world_size": world,
                "repeats": len(per_repeat),
                "device_total_bytes": total,
                "limit_bytes": limit,
                "training_peak_worst_rank_bytes": worst["training_peak_bytes"],
                "training_peak_worst_rank": worst["rank"],
                "training_peak_median_bytes": int(round(statistics.median(run_peaks))),
                "training_peak_min_bytes": min(run_peaks),
                "training_peak_max_bytes": max(run_peaks),
                "training_peak_spread_bytes": max(run_peaks) - min(run_peaks),
                "whole_process_peak_worst_rank_bytes": max(
                    entry["whole_process_peak_bytes"] for entry in per_rank),
                "startup_peak_worst_rank_bytes": (max(startup_peaks)
                                                  if startup_peaks else None),
                "aggregate_training_peak_median_bytes":
                    int(round(statistics.median(aggregate))),
                "aggregate_training_peak_min_bytes": min(aggregate),
                "aggregate_training_peak_max_bytes": max(aggregate),
                "aggregate_training_samples_min": min(
                    summary["aggregate_training_samples"]
                    for summary in per_repeat),
                "aggregate_gap_max_seconds": max(
                    (summary["aggregate_gap_max_seconds"]
                     for summary in per_repeat), default=None),
                "margin_bytes": limit - worst["training_peak_bytes"],
                "max_sample_gap_seconds": max(summary["max_sample_gap_seconds"]
                                              for summary in per_repeat),
                "training_samples_min": min(entry["training_samples_min"]
                                            for entry in per_rank),
                "per_rank": per_rank,
                "per_repeat": per_repeat,
            })
    _require(len(totals) == 1, "campaign device totals differ across records")
    return {
        "target_sample_interval_seconds": TARGET_SAMPLE_INTERVAL_SECONDS,
        "max_sample_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
        "min_training_samples_per_rank": MIN_TRAINING_SAMPLES,
        "criterion": (
            "The preflight OD-8 gate still uses the whole-process sampled peak "
            "at or below 80% of the device total. Campaign rows report "
            "training-window and whole-process sampled peaks without adding a "
            "new acceptance threshold."
        ),
        "aggregate_policy": (
            "sum of simultaneous per-rank samples at the same sample "
            "timestamp; never the sum of independent rank maxima"
        ),
        "qualifications": list(MEMORY_QUALIFICATIONS),
        "rows": rows,
    }


def _print_memory(memory):
    print("FULL_MEMORY target_interval_s=%.2f min_training_samples_per_rank=%d"
          % (memory["target_sample_interval_seconds"],
             memory["min_training_samples_per_rank"]))
    print("framework W worst_rank_MiB rank median_MiB min_MiB max_MiB "
          "whole_peak_MiB startup_peak_MiB agg_median_MiB agg_min_MiB "
          "agg_max_MiB margin_MiB gap_max_s train_samples")
    for row in memory["rows"]:
        startup = row["startup_peak_worst_rank_bytes"]
        print("%s %d %.1f %d %.1f %.1f %.1f %.1f %s %.1f %.1f %.1f %.1f %.2f %d" % (
            row["framework"], row["world_size"],
            row["training_peak_worst_rank_bytes"] / 2**20,
            row["training_peak_worst_rank"],
            row["training_peak_median_bytes"] / 2**20,
            row["training_peak_min_bytes"] / 2**20,
            row["training_peak_max_bytes"] / 2**20,
            row["whole_process_peak_worst_rank_bytes"] / 2**20,
            "NA" if startup is None else "%.1f" % (startup / 2**20),
            row["aggregate_training_peak_median_bytes"] / 2**20,
            row["aggregate_training_peak_min_bytes"] / 2**20,
            row["aggregate_training_peak_max_bytes"] / 2**20,
            row["margin_bytes"] / 2**20,
            row["max_sample_gap_seconds"],
            row["training_samples_min"],
        ))
    for row in memory["rows"]:
        for rank in row["per_rank"]:
            startup = rank["startup_peak_bytes"]
            print("MEMORY_RANK framework=%s W=%d rank=%d peak_MiB=%.1f "
                  "median_MiB=%.1f startup_MiB=%s whole_MiB=%.1f "
                  "train_samples_min=%d"
                  % (row["framework"], row["world_size"], rank["rank"],
                     rank["training_peak_bytes"] / 2**20,
                     rank["training_median_bytes"] / 2**20,
                     "NA" if startup is None else "%.1f" % (startup / 2**20),
                     rank["whole_process_peak_bytes"] / 2**20,
                     rank["training_samples_min"]))


def _check_campaign(args):
    root = Path(args.root)
    config = _check_config(root)
    identity = _check_source_identity(root)
    dataset = _check_dataset_identity(root, config)
    _check_source_snapshot(root)
    runtime = _check_runtime(root)
    provenance = _check_provenance(root, config, identity)
    _check_launcher(root, provenance)
    decision = _check_preflight_decision(root, config, identity, dataset)

    _check_record_files(
        root / "records",
        {"%s-w%d-r%d.json" % (framework, world, repeat)
         for framework in FRAMEWORKS for world in WORLDS
         for repeat in range(1, config["repeats"] + 1)},
        "timing")
    first_record = None
    records = {}
    for framework in FRAMEWORKS:
        for world in WORLDS:
            for repeat in range(1, config["repeats"] + 1):
                path = root / "records" / ("%s-w%d-r%d.json"
                                           % (framework, world, repeat))
                record = _check_record(
                    path, framework, world, config["updates"], config["warmup"],
                    config["epoch"], config["batch"], identity, dataset, False)
                records[framework, world, repeat] = record
                if first_record is None:
                    first_record = record
                else:
                    _require(record.get("job") == first_record.get("job"),
                             "record job differs within the campaign")
                    _require(record.get("node") == first_record.get("node"),
                             "record node differs within the campaign")
    _check_record_identity(list(records.values()))
    _check_runtime_versions(records, runtime)
    _check_resources(root, first_record["job"], first_record["node"])
    driver = root / "driver.log"
    _require(driver.is_file(), "missing driver.log")
    driver_text = driver.read_text(encoding="utf-8", errors="replace")
    for marker in REQUIRED_DRIVER_LINES:
        _require(marker in driver_text, "driver.log is missing %r" % marker)

    timing = _timing_table(records, config)
    memory = _check_memory(root, config, records)
    diagnostics = _check_diagnostics(
        root, config, identity, dataset,
        {(framework, world): records[framework, world, 1]
         for framework in FRAMEWORKS for world in WORLDS},
        runtime)

    report = {
        "schema": REPORT_SCHEMA,
        "config": config,
        "source_identity": identity,
        "dataset_identity_sha256": _sha256(root / "dataset-identity.json"),
        "runtime": runtime,
        "resources": _load_json(root / "resources.json"),
        "preflight": {
            "selected_global_batch": decision["selected_global_batch"],
            "reason": decision["reason"],
            "decision_sha256": _sha256(root / "preflight" / "decision.json"),
        },
        "timing": timing,
        "memory": memory,
        "diagnostics": diagnostics,
        "qualifications": [
            "Descriptive throughput and strong scaling only.",
            "Flux uses strict FMA candidate selection without a determinism filter.",
            "PyTorch uses deterministic cuDNN, benchmark disabled, and highest matmul precision.",
            "Flux DDP includes its non-finite guard inside the timed window; PyTorch does not.",
            "The state diagnostics are descriptive and apply no numerical-equivalence threshold.",
            "NUMERICAL_EQUIVALENCE_CLAIM none.",
        ],
        "pass": True,
    }
    (root / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("FULL_TIMING batch=%d updates=%d warmup=%d repeats=%d"
          % (config["batch"], config["updates"], config["warmup"],
             config["repeats"]))
    print("framework W median_s min_s max_s spread_s stdev_s images/s speedup efficiency")
    for row in timing:
        print("%s %d %.6f %.6f %.6f %.6f %.6f %.3f %.4f %.4f" % (
            row["framework"], row["world_size"], row["median_s"],
            row["min_s"], row["max_s"], row["spread_s"], row["stdev_s"],
            row["images_per_second"], row["speedup"], row["efficiency"]))
    _print_memory(memory)
    print("FULL_DIAGNOSTICS updates=%d policy=descriptive" % config["updates"])
    print("comparison category arrays exact max_abs rel_l2 worst_leaf")
    for comparison in diagnostics["comparisons"]:
        for row in comparison["categories"]:
            print("%s %s %d %d %.9g %.9g %s" % (
                comparison["label"], row["category"], row["arrays"],
                row["exact_arrays"], row["max_abs"], row["rel_l2"],
                row["worst_leaf"]))
    print("NUMERICAL_EQUIVALENCE_CLAIM none")
    print("FULL_CAMPAIGN_ARTIFACT_CHECK PASS")
    return 0


def _write_gpu_trace(sample_path, samples, world, pid_base=100):
    """Write a synthetic timestamped ``nvidia-smi`` trace and rank registration."""
    sample_path = Path(sample_path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for stamp, values in samples:
        lines.append("TIME,%.9f" % stamp)
        for rank, mib in enumerate(values):
            if mib is None:
                continue
            lines.append("%d, GPU-%d, %d" % (pid_base + rank, rank, mib))
    sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rank_dir = sample_path.with_suffix(".ranks")
    rank_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(world):
        (rank_dir / ("rank-%d.json" % rank)).write_text(json.dumps(
            {"rank": rank, "pid": pid_base + rank, "gpu_uuid": "GPU-%d" % rank}),
            encoding="utf-8")


def _synthetic_trace(world):
    samples = [(999.0, [900] * world)]
    for index in range(21):  # 1000.0 .. 1010.0 every 0.5 s
        stamp = 1000.0 + 0.5 * index
        bump = 10 if 1005.0 <= stamp < 1006.0 else 5
        samples.append((stamp, [1000 + 100 * rank + bump
                                for rank in range(world)]))
    samples.append((1011.0, [950] * world))
    return samples


def _write_synthetic(root, repeats=2, batch=512):
    import numpy as np

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    updates = expected_updates(batch)
    identity = {
        "THESIS_COMMIT": THESIS_COMMIT,
        "THESIS_COMMIT_SOURCE": "git",
        "THESIS_STATE_SHA256": hashlib.sha256(b"synthetic-thesis-state").hexdigest(),
        "THESIS_STATE_RECIPE": "sha256 of the synthetic thesis tree",
        "CURRENT_COMMIT": CURRENT_COMMIT,
        "CURRENT_STATE_SHA256": CURRENT_STATE_SHA256,
        "CURRENT_SOURCE_TREE_SHA256": CURRENT_SOURCE_TREE_SHA256,
        "CURRENT_COMMIT_SOURCE": "git",
        "CURRENT_STATE_SOURCE": "git",
        "HISTORICAL_COMMIT": HISTORICAL_COMMIT,
        "HISTORICAL_STATE_SHA256": HISTORICAL_STATE_SHA256,
        "HISTORICAL_SOURCE_TREE_SHA256": HISTORICAL_SOURCE_TREE_SHA256,
        "HISTORICAL_COMMIT_SOURCE": "expected",
        "HISTORICAL_STATE_SOURCE": "expected",
    }
    (root / "source-identity.txt").write_text(
        "\n".join("%s=%s" % item for item in identity.items()) + "\n")
    config = {
        "schema": CONFIG_SCHEMA,
        "batch": batch,
        "epoch": 0,
        "updates": updates,
        "warmup": 20,
        "repeats": repeats,
        "preflight_updates": 2,
        "preflight_warmup": 2,
        "worlds": list(WORLDS),
        "frameworks": list(FRAMEWORKS),
    }
    (root / "campaign-config.json").write_text(json.dumps(config, indent=2) + "\n")

    source = root / "source"
    source.mkdir(parents=True)
    files = {name: name + "\n" for name in REQUIRED_SOURCE_FILES}
    files["launcher/run-static-full-campaign.sh"] = "launcher\n"
    for relative, text in files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    manifest_lines = []
    for target in sorted(source.rglob("*")):
        if target.is_file():
            relative = str(target.relative_to(source))
            manifest_lines.append("%s  source/%s" % (_sha256(target), relative))
    (root / "source-sha256.txt").write_text("\n".join(manifest_lines) + "\n")

    dataset_files = {name: {"bytes": 1, "sha256": hashlib.sha256(name.encode()).hexdigest(),
                            "source": "synthetic"}
                     for name in REQUIRED_DATASET_FILES}
    dataset = {
        "schema": DATASET_SCHEMA,
        "dataset_dir": "/synthetic",
        "files": dataset_files,
        "bundle": {
            "bundle_hash": hashlib.sha256(b"bundle").hexdigest(),
            "bundle_file_sha256": hashlib.sha256(b"bundle-file").hexdigest(),
        },
        "epoch": 0,
        "workloads": {},
        "transform_spec_sha256": hashlib.sha256(b"transform").hexdigest(),
    }
    for candidate in BATCHES:
        dataset["workloads"][str(candidate)] = {
            "n_eff": N_EFF,
            "tail_size": TAIL,
            "updates_per_epoch": expected_updates(candidate),
            "sample_order_sha256": hashlib.sha256(b"sample-%d" % candidate).hexdigest(),
            "update1_order_sha256": hashlib.sha256(b"update1-%d" % candidate).hexdigest(),
        }
    (root / "dataset-identity.json").write_text(json.dumps(dataset, indent=2) + "\n")

    runtime = {
        "schema": RUNTIME_SCHEMA,
        "julia_version": "1.12.6",
        "variants": {
            "flux-current": {
                "variant": "flux-current", "julia": "1.12.6", "flux": "0.16.11",
                "cuda": "6.2.2", "nccl": "0.1.1", "mpi": "0.20.23",
                "optimisers": "0.4.6", "functors": "0.5.2",
            },
            "flux-historical": {
                "variant": "flux-historical", "julia": "1.12.6", "flux": "0.16.11",
                "cuda": "6.2.2", "nccl": "0.1.1", "mpi": "0.20.23",
                "optimisers": "0.4.6", "functors": "0.5.2",
            },
        },
        "python": {
            "executable": "/synthetic/python", "python": "3.11.17",
            "torch": "2.9.1+cu126", "numpy": "2.4.6", "cuda": "12.6",
            "cudnn": "90102",
        },
        "gpus": [{"index": index, "name": "A100", "uuid": "GPU-%d" % index,
                  "total_bytes": 64 * 1024**3} for index in range(4)],
        "driver_version": "535.274.02",
    }
    (root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (root / "runtime-env.txt").write_text(
        "RUNTIME_JULIA variant=flux-current julia=1.12.6 flux=0.16.11\n"
        "RUNTIME_JULIA variant=flux-historical julia=1.12.6 flux=0.16.11\n"
        "RUNTIME_PYTHON python=3.11.17 torch=2.9.1+cu126\n"
        "RUNTIME_GPU index=0 name=A100 total=68719476736\n")
    resources = {
        "schema": RESOURCES_SCHEMA,
        "job": "12345",
        "node": "lrdn0001.leonardo.local",
        "partition": "boost_usr_prod",
        "qos": "boost_qos_lprod",
        "account": "iscrc_unmasked",
        "ntasks": 1,
        "cpus_per_task": CAMPAIGN_CPUS,
        "gres": "gpu:4",
        "time_limit": "04:00:00",
        "cuda_visible_devices": "0,1,2,3",
        "gpu_mapping": runtime["gpus"],
    }
    (root / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    (root / "resources.txt").write_text("SLURM_JOB_ID=12345\n")
    (root / "launcher-sha256.txt").write_text(
        "%s  /synthetic/launcher/run-static-full-campaign.sh\n"
        "%s  /synthetic/check_full_campaign.py\n"
        % (_sha256(source / "launcher" / "run-static-full-campaign.sh"),
           _sha256(source / "check_full_campaign.py")))
    (root / "driver.log").write_text("\n".join(REQUIRED_DRIVER_LINES) + "\n")

    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "selected_global_batch": batch,
        "updates": updates,
        "preflight_updates": 2,
        "preflight_warmup": 2,
        "epoch": 0,
        "source_identity": identity,
        "dataset_identity_sha256": _sha256(root / "dataset-identity.json"),
        "candidates": {
            str(batch): {"batch": batch, "status": "PASS", "errors": []},
        },
        "errors": [],
        "reason": "synthetic",
        "NUMERICAL_EQUIVALENCE_CLAIM": "none",
    }
    (root / "preflight").mkdir()
    (root / "preflight" / "decision.json").write_text(
        json.dumps(preflight, indent=2) + "\n")

    def base_record(framework, world, snapshot, count, batch_value=batch,
                    warmup_value=20):
        kind = _framework_kind(framework)
        record = {
            "schema": SCHEMA,
            "framework": kind,
            "world_size": world,
            "epoch": 0,
            "batch_size": batch_value,
            "global_batch": batch_value,
            "updates": count,
            "warmup": warmup_value,
            "updates_per_epoch": expected_updates(batch_value),
            "n_eff": N_EFF,
            "tail_size": TAIL,
            "snapshot": snapshot,
            "timing": "full-dataset-end-to-end",
            "workload": "full-dataset-epoch",
            "pass_checks": True,
            "loss_finite": True,
            "final_loss": 1.0,
            "eta_sha256": hashlib.sha256(b"eta-%d" % count).hexdigest(),
            "job": "12345",
            "node": resources["node"],
            "seconds": 10.0,
            "images_per_second": batch_value * count / 10.0,
            "update_seconds": 10.0 / count,
            "dataset_hashes": {name: {"sha256": info["sha256"]}
                               for name, info in dataset_files.items()},
            "bundle_hash": dataset["bundle"]["bundle_hash"],
            "bundle_file_sha256": dataset["bundle"]["bundle_file_sha256"],
            "sample_order_sha256": dataset["workloads"][str(batch_value)]["sample_order_sha256"],
            "update1_order_sha256": dataset["workloads"][str(batch_value)]["update1_order_sha256"],
            "transform_spec_sha256": dataset["transform_spec_sha256"],
            "loader_policy": "mmap opened once per process; synchronous",
            "timing_policy": "timed region includes batch assembly, forward, backward, and optimizer step",
            "warmup_policy": "discarded warmup updates; model and optimizer reset; measured region restarts at update 1",
            "source": {
                "thesis_commit": identity["THESIS_COMMIT"],
                "thesis_state_sha256": identity["THESIS_STATE_SHA256"],
                "flux_commit": "",
                "flux_state_sha256": "",
            },
        }
        if kind == "flux":
            prefix = "CURRENT" if framework == "flux-current" else "HISTORICAL"
            record.update({
                "precision": FLUX_PRECISION,
                "distributed_api": "single_gpu" if world == 1 else ("guarded" if framework == "flux-current" else "static"),
                "config_kind": "single_gpu" if world == 1 else ("guarded_ddp" if framework == "flux-current" else "historical_static_ddp"),
                "guard_policy": SINGLE_GUARD if world == 1 else (FLUX_GUARD if framework == "flux-current" else FLUX_STATIC_GUARD),
                "rank_seconds": [10.0] * world,
                "rank_training_windows": [
                    {"rank": rank, "start_epoch": 1000.0,
                     "end_epoch": 1010.0, "seconds": 10.0}
                    for rank in range(world)],
                "hashes": {key: hashlib.sha256((framework + key).encode()).hexdigest()
                           for key in ("param", "m", "v", "beta_power")},
                "device": 0,
                "memory": {"device_total_bytes": 64 * 1024**3},
                "source": {"thesis_commit": identity["THESIS_COMMIT"],
                           "thesis_state_sha256": identity["THESIS_STATE_SHA256"],
                           "flux_commit": identity[prefix + "_COMMIT"],
                           "flux_state_sha256": identity[prefix + "_STATE_SHA256"],
                           "julia_version":
                               runtime["variants"][_flux_variant(framework)]["julia"]},
            })
            record["flux_commit"] = identity[prefix + "_COMMIT"]
            record["flux_state_sha256"] = identity[prefix + "_STATE_SHA256"]
            record["rank_hashes"] = {
                key: [value] * world for key, value in record["hashes"].items()}
        else:
            record["precision"] = dict(PYTORCH_PRECISION)
            record["ranks"] = [{
                "rank": rank,
                "seconds": 10.0,
                "device": (rank + 1) % 4 if world > 1 else 0,
                "training_start_epoch": 1000.0,
                "training_end_epoch": 1010.0,
                "hashes": {key: hashlib.sha256(("pytorch" + key).encode()).hexdigest()
                           for key in ("param", "m", "v")} | {"step": count},
                "memory": {"mem_get_info_total": 64 * 1024**3},
            } for rank in range(world)]
            record["source"].update({
                "python_version": "3.11.17",
                "torch_version": "2.9.1+cu126",
                "numpy_version": "2.4.6",
                "host": resources["node"],
            })
        return record

    (root / "records").mkdir()
    (root / "samples").mkdir()
    for framework in FRAMEWORKS:
        for world in WORLDS:
            for repeat in range(1, repeats + 1):
                record = base_record(framework, world, False, updates)
                label = "%s-w%d-r%d" % (framework, world, repeat)
                (root / "records" / (label + ".json")).write_text(
                    json.dumps(record, indent=2) + "\n")
                _write_gpu_trace(root / "samples" / (label + ".csv"),
                                 _synthetic_trace(world), world)

    (root / "diagnostics").mkdir()
    (root / "diagnostics" / "records").mkdir()
    (root / "diagnostics" / "snapshots").mkdir()
    arrays = {
        "param.a": np.arange(4, dtype=np.float32),
        "param.b": np.arange(4, 8, dtype=np.float32),
        "m.a": np.arange(4, dtype=np.float32) + 1,
        "m.b": np.arange(4, 8, dtype=np.float32) + 1,
        "v.a": np.arange(4, dtype=np.float32) + 2,
        "v.b": np.arange(4, 8, dtype=np.float32) + 2,
        "step": np.array(updates, dtype=np.int64),
    }
    for framework in FRAMEWORKS:
        for world in WORLDS:
            record = base_record(framework, world, True, updates)
            (root / "diagnostics" / "records" / ("%s-w%d.json"
                                                 % (framework, world))).write_text(
                json.dumps(record, indent=2) + "\n")
            snapshot = dict(arrays)
            if framework.startswith("flux"):
                snapshot["beta_power.a"] = _expected_beta_power(updates)
                snapshot["beta_power.b"] = _expected_beta_power(updates)
            target = root / "diagnostics" / "snapshots" / ("%s-w%d" % (framework, world))
            target.mkdir()
            np.savez(target / "state.npz", **snapshot)

    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "source_identity": identity,
        "campaign_config": config,
        "source_snapshot_manifest_sha256": _sha256(root / "source-sha256.txt"),
        "runtime_env_sha256": _sha256(root / "runtime-env.txt"),
        "runtime_json_sha256": _sha256(root / "runtime.json"),
        "dataset_identity_sha256": _sha256(root / "dataset-identity.json"),
        "resources_json_sha256": _sha256(root / "resources.json"),
        "preflight_decision_sha256": _sha256(root / "preflight" / "decision.json"),
        "launcher": {
            "name": "run-static-full-campaign.sh",
            "path": "/synthetic/launcher/run-static-full-campaign.sh",
            "sha256": _sha256(source / "launcher" / "run-static-full-campaign.sh"),
        },
        "checker": {
            "name": "check_full_campaign.py",
            "path": "/synthetic/check_full_campaign.py",
            "sha256": _sha256(source / "check_full_campaign.py"),
        },
        "execution_policy": {
            "flux": {
                "precision": FLUX_PRECISION,
                "cudnn_deterministic": False,
                "nonfinite_guard": FLUX_GUARD,
            },
            "pytorch": dict(PYTORCH_PRECISION, nonfinite_guard="outside timed loop"),
        },
        "qualifications": ["synthetic"],
        "NUMERICAL_EQUIVALENCE_CLAIM": "none",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return base_record


def run_self_tests():
    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "campaign-12345"
        base_record = _write_synthetic(base)
        _check_campaign(argparse.Namespace(root=str(base)))
        checks += 1

        preflight_base = Path(tmp) / "preflight-case"
        shutil.copytree(base, preflight_base)
        for batch in BATCHES:
            candidate = preflight_base / "preflight" / ("b%d" % batch)
            (candidate / "records").mkdir(parents=True)
            (candidate / "samples").mkdir(parents=True)
            for framework in FRAMEWORKS:
                for world in WORLDS:
                    label = "%s-w%d" % (framework, world)
                    record = base_record(framework, world, False, 2,
                                         batch_value=batch, warmup_value=2)
                    (candidate / "records" / (label + ".json")).write_text(
                        json.dumps(record, indent=2) + "\n")
                    _write_gpu_trace(candidate / "samples" / (label + ".csv"),
                                     _synthetic_trace(world), world, pid_base=1234)

        def run_preflight():
            return _preflight(argparse.Namespace(
                root=str(preflight_base),
                candidate=[(batch, str(preflight_base / "preflight" / ("b%d" % batch)))
                           for batch in BATCHES],
                updates=2,
                warmup=2,
                epoch=0,
                select=True,
                out=str(preflight_base / "preflight" / "decision.json"),
            ))

        _require(run_preflight() == 0, "preflight self-test did not select a batch")
        decision = _load_json(preflight_base / "preflight" / "decision.json")
        _require(decision["selected_global_batch"] == 512,
                 "preflight self-test did not select 512")
        checks += 1

        for framework in FRAMEWORKS:
            for world in WORLDS:
                _write_gpu_trace(
                    preflight_base / "preflight" / "b512" / "samples"
                    / ("%s-w%d.csv" % (framework, world)),
                    [(1002.0, [60000] * world), (1006.0, [60000] * world)],
                    world, pid_base=1234)
        _require(run_preflight() == 0, "preflight fallback did not select 256")
        decision = _load_json(preflight_base / "preflight" / "decision.json")
        _require(decision["selected_global_batch"] == 256,
                 "preflight self-test did not fall back to 256")
        checks += 1

        def reject(name, mutate):
            nonlocal checks
            copy = Path(tmp) / ("mut-" + name)
            shutil.copytree(base, copy)
            mutate(copy)
            try:
                _check_campaign(argparse.Namespace(root=str(copy)))
            except Exception:
                checks += 1
            else:
                _fail("self-test mutation was accepted: %s" % name)

        def mutate_json(root, relative, fn):
            path = Path(root) / relative
            document = json.loads(path.read_text())
            fn(document)
            path.write_text(json.dumps(document, indent=2) + "\n")

        def mutate_identity(root, key, value):
            path = Path(root) / "source-identity.txt"
            document = _parse_pairs(path.read_text())
            document[key] = value
            path.write_text("\n".join("%s=%s" % item
                                      for item in document.items()) + "\n")

        reject("missing-record", lambda root: (
            root / "records" / "flux-current-w2-r2.json").unlink())
        reject("wrong-updates", lambda root: mutate_json(
            root, "records/pytorch-w4-r1.json",
            lambda record: record.__setitem__("updates", 194)))
        reject("wrong-source", lambda root: mutate_json(
            root, "records/flux-historical-w2-r1.json",
            lambda record: record["source"].__setitem__(
                "flux_commit", "9" * 40)))
        reject("replica-hash", lambda root: mutate_json(
            root, "records/flux-current-w4-r1.json",
            lambda record: record["rank_hashes"]["param"].__setitem__(1, "0" * 64)))
        reject("nonfinite-snapshot", lambda root: _mutate_snapshot(
            root, "flux-current", 2, lambda arrays:
            arrays.__setitem__("param.a", np.full_like(arrays["param.a"], np.nan))))
        reject("wrong-step", lambda root: _mutate_snapshot(
            root, "pytorch", 4, lambda arrays:
            arrays.__setitem__("step", np.array(194, dtype=np.int64))))
        reject("missing-runtime", lambda root: (
            root / "runtime-env.txt").write_text("RUNTIME_ENV_CAPTURE_FAILED\n"))
        reject("preflight-batch", lambda root: mutate_json(
            root, "preflight/decision.json",
            lambda decision: decision.__setitem__("selected_global_batch", 256)))
        reject("provenance-claim", lambda root: mutate_json(
            root, "provenance.json",
            lambda document: document.__setitem__(
                "NUMERICAL_EQUIVALENCE_CLAIM", "thresholded")))
        reject("eta-mismatch", lambda root: mutate_json(
            root, "records/pytorch-w2-r1.json",
            lambda record: record.__setitem__("eta_sha256", "0" * 64)))
        reject("wrong-current-commit", lambda root: mutate_identity(
            root, "CURRENT_COMMIT", "9" * 40))
        reject("wrong-current-state", lambda root: mutate_identity(
            root, "CURRENT_STATE_SHA256", "9" * 64))
        reject("wrong-current-tree", lambda root: mutate_identity(
            root, "CURRENT_SOURCE_TREE_SHA256", "9" * 64))
        reject("duplicate-flux-commit", lambda root: mutate_identity(
            root, "HISTORICAL_COMMIT", CURRENT_COMMIT))
        reject("duplicate-flux-tree", lambda root: mutate_identity(
            root, "HISTORICAL_SOURCE_TREE_SHA256", CURRENT_SOURCE_TREE_SHA256))
        reject("runtime-julia-version", lambda root: mutate_json(
            root, "records/flux-current-w2-r1.json",
            lambda record: record["source"].__setitem__(
                "julia_version", "0.0.0")))
        reject("runtime-torch-version", lambda root: mutate_json(
            root, "records/pytorch-w2-r1.json",
            lambda record: record["source"].__setitem__(
                "torch_version", "0.0.0+none")))
        reject("extra-timing-record", lambda root: shutil.copyfile(
            root / "records" / "pytorch-w1-r1.json",
            root / "records" / "pytorch-w1-r99.json"))
        reject("missing-campaign-sample", lambda root: (
            root / "samples" / "flux-current-w2-r2.csv").unlink())
        reject("missing-rank-registration", lambda root: (
            root / "samples" / "pytorch-w4-r2.ranks" / "rank-3.json").unlink())
        reject("no-training-sample", lambda root: _write_gpu_trace(
            root / "samples" / "flux-current-w2-r1.csv",
            [(996.0, [1000, 1100]), (998.0, [1000, 1100])], 2))
        reject("sampling-gap", lambda root: _write_gpu_trace(
            root / "samples" / "pytorch-w2-r1.csv",
            [(1000.0 + 0.5 * index, [1000, 1100]) for index in range(9)]
            + [(1009.5, [1000, 1100])], 2))
        reject("truncated-trace", lambda root: _write_gpu_trace(
            root / "samples" / "pytorch-w4-r1.csv",
            [(1000.0 + 0.5 * index, [1000, 1100, 1200, 1300])
             for index in range(10)], 4))
        reject("sparse-trace", lambda root: _write_gpu_trace(
            root / "samples" / "flux-historical-w2-r2.csv",
            [(1001.0, [1000, 1100]), (1005.0, [1000, 1100]),
             (1009.0, [1000, 1100])], 2))
        reject("missing-training-window", lambda root: mutate_json(
            root, "records/flux-current-w2-r1.json",
            lambda record: record.pop("rank_training_windows")))
        reject("window-order", lambda root: mutate_json(
            root, "records/pytorch-w4-r1.json",
            lambda record: record["ranks"][1].__setitem__(
                "training_end_epoch", 1000.0)))
        reject("window-duration", lambda root: mutate_json(
            root, "records/flux-current-w4-r1.json",
            lambda record: record["rank_training_windows"][2].__setitem__(
                "start_epoch", 900.0)))
        reject("duplicate-sample-pid", lambda root: mutate_json(
            root, "samples/pytorch-w4-r1.ranks/rank-3.json",
            lambda document: document.__setitem__("pid", 100)))
        reject("duplicate-sample-timestamp", lambda root: _write_gpu_trace(
            root / "samples" / "flux-historical-w2-r1.csv",
            [(stamp, [1000, 1100]) for stamp in (1001.0, 1005.0, 1009.0)
             for _ in range(4)], 2))
        reject("aggregate-start-coverage", lambda root: _write_gpu_trace(
            root / "samples" / "flux-current-w2-r1.csv",
            [(1000.0 + 0.5 * index, [1000, 1100]) for index in range(10)]
            + [(1005.0 + 0.5 * index,
                [1000, None] if index % 2 == 0 else [None, 1100])
               for index in range(11)], 2))
        reject("aggregate-middle-gap", lambda root: _write_gpu_trace(
            root / "samples" / "pytorch-w2-r1.csv",
            [(1000.0 + 0.5 * index, [1000, 1100]) for index in range(5)]
            + [(1002.5 + 0.5 * index,
                [1000, None] if index % 2 == 0 else [None, 1100])
               for index in range(11)]
            + [(1008.0 + 0.5 * index, [1000, 1100])
               for index in range(5)], 2))

        report = _load_json(base / "REPORT.json")
        _require(len(report["memory"]["rows"]) == 9,
                 "campaign report is missing the nine memory rows")
        row = report["memory"]["rows"][0]
        _require(row["training_peak_worst_rank_bytes"] > 0,
                 "campaign memory row has no training peak")
        _require(row["aggregate_training_samples_min"] >= MIN_TRAINING_SAMPLES,
                 "campaign memory row has no simultaneous coverage")
        _require(row["aggregate_gap_max_seconds"] is not None,
                 "campaign memory row is missing the simultaneous gap")
        checks += 1

        with tempfile.TemporaryDirectory() as trace_tmp:
            path = Path(trace_tmp) / "staggered.csv"
            _write_gpu_trace(path, [
                (1002.0, [1000, 500]),
                (1004.0, [500, 1000]),
                (1006.0, [700, 700]),
            ], 2)
            summary = _summarize_gpu_samples(
                path, 2, {0: (1000.0, 1010.0), 1: (1000.0, 1010.0)})
            _require(summary["aggregate_training_peak_bytes"] == 1500 * 2**20,
                     "aggregate sample peak is not the simultaneous sum")
            _require(summary["aggregate_training_peak_bytes"]
                     < sum(entry["training_peak_bytes"]
                           for entry in summary["ranks"]),
                     "aggregate sample peak must not sum independent rank maxima")
            checks += 1

        with tempfile.TemporaryDirectory() as trace_tmp:
            path = Path(trace_tmp) / "early-burst-20s.csv"
            samples = [(1000.0 + 0.25 * index, [1000, 1100])
                       for index in range(20)]
            samples += [(1005.0 + 0.25 * index,
                         [1000, None] if index % 2 == 0 else [None, 1100])
                        for index in range(61)]
            _write_gpu_trace(path, samples, 2)
            try:
                _summarize_gpu_samples(
                    path, 2, {0: (1000.0, 1020.0), 1: (1000.0, 1020.0)},
                    require_density=True)
            except AssertionError as exc:
                _require("simultaneous GPU samples end" in str(exc),
                         "20 s early-burst trace failed for the wrong reason: %s"
                         % exc)
                checks += 1
            else:
                _fail("20 s early-burst aggregate was accepted")
    expected_checks = 37
    _require(checks == expected_checks,
             "self-test check count is %d, expected %d"
             % (checks, expected_checks))
    print("FULL_CAMPAIGN_SELFTEST PASS checks=%d" % checks)


def _mutate_snapshot(root, framework, world, fn):
    path = (Path(root) / "diagnostics" / "snapshots"
            / ("%s-w%d" % (framework, world)) / "state.npz")
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    fn(arrays)
    np.savez(path, **arrays)


def _parse_candidate(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected B=DIR, got %r" % value)
    batch_text, root = value.split("=", 1)
    try:
        batch = int(batch_text)
    except ValueError:
        raise argparse.ArgumentTypeError("batch must be an integer: %r" % batch_text)
    if batch not in BATCHES:
        raise argparse.ArgumentTypeError("batch must be 512 or 256")
    return batch, root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--root", required=True)
    preflight.add_argument("--candidate", action="append", type=_parse_candidate,
                           required=True, metavar="B=DIR")
    preflight.add_argument("--updates", type=int, default=20)
    preflight.add_argument("--warmup", type=int, default=20)
    preflight.add_argument("--epoch", type=int, default=0)
    preflight.add_argument("--select", action="store_true")
    preflight.add_argument("--out", required=True)

    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--root", required=True)
    ready = subparsers.add_parser("ready")
    ready.add_argument("--root", required=True)

    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_tests()
        return 0
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "campaign":
        return _check_campaign(args)
    if args.command == "ready":
        return _check_ready(args)
    parser.error("a subcommand or --self-test is required")


if __name__ == "__main__":
    sys.exit(main())
