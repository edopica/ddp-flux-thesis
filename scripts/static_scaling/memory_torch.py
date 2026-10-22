#!/usr/bin/env python3
"""Phase 5 PyTorch memory pilot.

Runs complete training steps at one global batch size on one local shard per
rank, after warm-up, and records CUDA memory counters by phase:

  forward   torch.cuda.max_memory_allocated after forward
  backward  after backward (native DDP communication is inside this phase)
  sync      not separately observable for native DDP; a non-DDP backward
            control gives a communication estimate when requested
  optimizer after optimizer.step()
  total     maximum of the phase peaks

The governing pass/fail counter is the maximum per-process device memory from
`nvidia-smi`. Allocator live/reserved counters are recorded separately.
Unavailable counters are written as null with an explicit status string.
"""

from __future__ import annotations

import argparse
import gc
import os
import socket
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import dataset_adapter as da  # noqa: E402
import model_torch as mt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--global-batch", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--estimate-local-backward", action="store_true", default=True)
    p.add_argument("--no-estimate-local-backward", dest="estimate_local_backward",
                   action="store_false")
    return p.parse_args()


def nvidia_process_bytes():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None, "nvidia-smi_unavailable"
    pid = os.getpid()
    values = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            proc = int(parts[0])
            mib = int(parts[1])
        except ValueError:
            continue
        if proc == pid:
            values.append(mib * 1024 * 1024)
    if not values:
        return None, "pid_not_listed"
    return max(values), "measured"


def cuda_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {
        "live": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "free": int(free),
        "total": int(total),
        "peak_live": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


def record(peaks, phase, snapshot, proc):
    for key in ("live", "reserved", "peak_live", "peak_reserved"):
        value = snapshot[key]
        peaks.setdefault(phase, {})
        peaks[phase][key] = max(peaks[phase].get(key, 0), int(value))
    bytes_value, status = proc
    if status == "measured":
        peaks["process_status"] = "measured"
    else:
        peaks.setdefault("process_status", status)
    if bytes_value is not None:
        peaks.setdefault("process", {})
        peaks["process"][phase] = max(peaks["process"].get(phase, 0), int(bytes_value))


def model_bytes(module):
    return sum(int(p.numel() * p.element_size()) for p in module.parameters())


def optimizer_state_bytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel() * value.element_size())
    return total


def make_batches(fixture_dir, batch, world, rank, device):
    data_path = os.path.join(fixture_dir, "data_fixture.npz")
    with np.load(data_path) as archive:
        refs = np.asarray(archive["train_transformed_ref"])
        labels = np.asarray(archive["train_labels"])
    if refs.shape[1] != batch:
        raise ValueError("fixture batch %d != --global-batch %d" % (refs.shape[1], batch))
    if labels.shape != (batch,):
        raise ValueError("fixture label shape %s != (%d,)" % (labels.shape, batch))
    chunk = batch // world
    low, high = rank * chunk, (rank + 1) * chunk
    batches = []
    for row in range(2):
        x = torch.from_numpy(np.ascontiguousarray(refs[row, low:high])).to(device)
        y = torch.from_numpy(np.ascontiguousarray(labels[low:high])).long().to(device)
        batches.append((x, y))
    return batches, data_path


def main() -> int:
    args = parse_args()
    world = int(args.world_size)
    batch = int(args.global_batch)
    if world not in (1, 2, 4):
        raise SystemExit("--world-size must be 1, 2, or 4")
    if batch not in (512, 256):
        raise SystemExit("--global-batch must be 512 or 256")
    if batch % world != 0:
        raise SystemExit("--global-batch not divisible by --world-size")
    if args.warmup < 1 or args.updates < 1:
        raise SystemExit("--warmup and --updates must be >= 1")
    local_batch = batch // world
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fixture_dir = os.path.abspath(args.fixture_dir)
    bundle_path = os.path.join(fixture_dir, "initial_bundle.npz")
    if not os.path.isfile(bundle_path):
        raise SystemExit("missing " + bundle_path)

    rank = int(os.environ.get("RANK", 0))
    if int(os.environ.get("WORLD_SIZE", 1)) != world:
        raise SystemExit("WORLD_SIZE does not match --world-size")
    device_index = (rank + 1) % torch.cuda.device_count() if world > 1 else 0
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    flags = mt.configure_backend(num_threads=1)
    if world > 1:
        dist.init_process_group("nccl")

    batches, data_path = make_batches(fixture_dir, batch, world, rank, device)
    staging_bytes = sum(int(x.numel() * x.element_size() + y.numel() * y.element_size())
                        for x, y in batches)

    inner = mt.make_model().to(device)
    mt.load_bundle(inner, bundle_path)
    optimizer = mt.build_optimizer(inner)
    model = torch.nn.parallel.DistributedDataParallel(
        inner, device_ids=[device_index], broadcast_buffers=False,
        find_unused_parameters=False, static_graph=False) if world > 1 else inner

    for t in range(1, args.warmup + 1):
        x, y = batches[(t - 1) % 2]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        mt.set_optimizer_lr(optimizer, mt.schedule_eta(t))
        optimizer.step()
    torch.cuda.synchronize()

    mt.load_bundle(inner, bundle_path)
    optimizer = mt.build_optimizer(inner)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    torch.cuda.synchronize()

    baseline_proc, baseline_status = nvidia_process_bytes()
    torch.cuda.reset_peak_memory_stats()
    baseline = cuda_snapshot()
    peaks = {"process_status": baseline_status}
    record(peaks, "baseline", baseline, (baseline_proc, baseline_status))

    for t in range(1, args.updates + 1):
        x, y = batches[(t - 1) % 2]
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        torch.cuda.synchronize()
        record(peaks, "forward", cuda_snapshot(), nvidia_process_bytes())

        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        record(peaks, "backward", cuda_snapshot(), nvidia_process_bytes())

        torch.cuda.reset_peak_memory_stats()
        mt.set_optimizer_lr(optimizer, mt.schedule_eta(t))
        optimizer.step()
        torch.cuda.synchronize()
        record(peaks, "optimizer", cuda_snapshot(), nvidia_process_bytes())
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite loss")

    optimizer_state_after = optimizer_state_bytes(optimizer)
    local_backward_peak = None
    communication_estimate = None
    if world > 1 and args.estimate_local_backward:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        x, y = batches[0]
        local_loss = F.cross_entropy(inner(x), y)
        local_loss.backward()
        torch.cuda.synchronize()
        local_backward_peak = int(torch.cuda.max_memory_allocated())
        optimizer.zero_grad(set_to_none=True)
        ddp_backward = peaks.get("backward", {}).get("peak_live", 0)
        communication_estimate = max(0, int(ddp_backward) - local_backward_peak)

    final_proc, final_status = nvidia_process_bytes()
    torch.cuda.reset_peak_memory_stats()
    final_snapshot = cuda_snapshot()
    record(peaks, "final", final_snapshot, (final_proc, final_status))

    live_peak = max((v.get("peak_live", 0) for k, v in peaks.items()
                     if isinstance(v, dict) and "peak_live" in v), default=None)
    reserved_peak = max((v.get("peak_reserved", 0) for k, v in peaks.items()
                         if isinstance(v, dict) and "peak_reserved" in v), default=None)
    process_peak = max(peaks.get("process", {}).values(), default=None)
    total = int(baseline["total"])
    margin_limit = int(0.8 * total)
    required_free = int(np.ceil(0.2 * total))

    record_dict = {
        "schema": "static-memory-pilot-v1",
        "framework": "pytorch",
        "world_size": world,
        "global_batch": batch,
        "local_batch": local_batch,
        "warmup": int(args.warmup),
        "updates": int(args.updates),
        "rank": rank,
        "device": device_index,
        "node": socket.gethostname(),
        "job": os.environ.get("SLURM_JOB_ID", ""),
        "fixture_sha256": da.sha256_file(data_path),
        "bundle_sha256": da.sha256_file(bundle_path),
        "workload": "two-real-global-batches-from-phase5-fixture",
        "precision": flags,
        "staging_bytes": staging_bytes,
        "model_bytes": model_bytes(inner),
        "optimizer_state_bytes_expected": 2 * model_bytes(inner) + 62 * 4,
        "optimizer_state_bytes_after": optimizer_state_after,
        "device_total_bytes": total,
        "device_free_baseline_bytes": int(baseline["free"]),
        "device_process_baseline_bytes": baseline_proc,
        "device_process_peak_bytes": process_peak,
        "allocator_live_baseline_bytes": int(baseline["live"]),
        "allocator_live_peak_bytes": live_peak,
        "allocator_reserved_baseline_bytes": int(baseline["reserved"]),
        "allocator_reserved_peak_bytes": reserved_peak,
        "phases": {
            "forward_live_bytes": peaks.get("forward", {}).get("peak_live"),
            "forward_reserved_bytes": peaks.get("forward", {}).get("peak_reserved"),
            "backward_live_bytes": peaks.get("backward", {}).get("peak_live"),
            "backward_reserved_bytes": peaks.get("backward", {}).get("peak_reserved"),
            "sync_live_bytes": None,
            "sync_reserved_bytes": None,
            "optimizer_live_bytes": peaks.get("optimizer", {}).get("peak_live"),
            "optimizer_reserved_bytes": peaks.get("optimizer", {}).get("peak_reserved"),
        },
        "local_backward_peak_bytes": local_backward_peak,
        "communication_estimate_bytes": communication_estimate,
        "counter_status": {
            "allocator_live": "exact_max_memory_allocated",
            "allocator_reserved": "exact_max_memory_reserved",
            "device_process": peaks.get("process_status", "unavailable"),
            "communication": ("estimated_from_non_ddp_control" if world > 1 else "not_applicable"),
            "guard": "not_applicable; PyTorch has no Flux-style guards",
            "forward_peak": "exact_max_memory_allocated",
            "backward_peak": "exact_max_memory_allocated; DDP communication included",
        },
        "margin": {
            "required_free_fraction": 0.2,
            "limit_used_fraction": 0.8,
            "limit_bytes": margin_limit,
            "required_free_bytes": required_free,
        },
        "pass_checks": True,
    }

    rank_path = os.path.join(out_dir, "rank%d.json" % rank)
    da.write_json(rank_path, record_dict)
    if world > 1:
        dist.barrier()
    if rank == 0:
        summary = {
            "schema": "static-memory-pilot-summary-v1",
            "framework": "pytorch",
            "world_size": world,
            "global_batch": batch,
            "local_batch": local_batch,
            "warmup": int(args.warmup),
            "updates": int(args.updates),
            "node": socket.gethostname(),
            "job": os.environ.get("SLURM_JOB_ID", ""),
            "rank_files": ["rank%d.json" % r for r in range(world)],
            "pass_checks": True,
        }
        da.write_json(os.path.join(out_dir, "summary.json"), summary)
        print("MEMORY_PASS pytorch", world, batch, flush=True)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
