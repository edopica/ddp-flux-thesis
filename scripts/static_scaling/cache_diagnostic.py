"""Opt-in W=1 synchronized snapshots for the full-dataset PyTorch runner."""
import os
import subprocess
import time


def diagnostic_init(torch, device_uuid):
    return dict(
        schema="static-cache-diagnostic-v1", pid=os.getpid(),
        device_uuid=device_uuid, cuda_version=str(torch.version.cuda),
        runtime_api_evidence={
            "allocator_used_bytes": dict(api="torch.cuda.memory_allocated()",
                                         doc=torch.cuda.memory_allocated.__doc__),
            "allocator_reserved_bytes": dict(api="torch.cuda.memory_reserved()",
                                             doc=torch.cuda.memory_reserved.__doc__)},
        retention_policy="At post_training/post_gc/post_reclaim the same model, optimizer, dataset maps, final loss and parameter .grad buffers are retained. Per-step x/y do not escape steps(). Raw post_training is captured before state hashing; model/optimizer copies to CPU for invariant hashing occur between post_training and explicit GC, and again after post_reclaim.",
        counter_semantics={
            "allocator_used_bytes": "current tensor bytes managed by the PyTorch allocator; excludes runtime/library allocations",
            "allocator_reserved_bytes": "current total allocator-managed bytes INCLUDING allocated tensor bytes",
            "process_bytes": "nvidia-smi per-process resident GPU memory, MiB converted to bytes",
            "device_used_bytes": "device total minus free; includes other processes and runtime allocations",
            "total_bytes": "device total bytes"},
        peak_policy="snapshots are synchronized point samples, not within-step high-water marks; external nvidia-smi trace remains sampled process footprint",
        snapshots=[])


def process_bytes():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], universal_newlines=True)
    values = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) < 2:
            continue
        try:
            pid, mib = int(fields[0].strip()), int(fields[1].strip())
        except ValueError:
            continue
        if pid == os.getpid():
            values.append(mib * 1024**2)
    if not values:
        raise RuntimeError("cache diagnostic: nvidia-smi did not report this process")
    return max(values)


def snapshot(diagnostic, torch, phase):
    torch.cuda.synchronize()
    epoch = time.time()
    free, total = torch.cuda.mem_get_info()
    counters = dict(allocator_used_bytes=int(torch.cuda.memory_allocated()),
                    allocator_reserved_bytes=int(torch.cuda.memory_reserved()),
                    process_bytes=process_bytes(), device_used_bytes=int(total-free),
                    total_bytes=int(total))
    diagnostic["snapshots"].append(dict(
        phase=phase, epoch=epoch, end_epoch=time.time(),
        counter_status={key: "measured" for key in counters}, **counters))
    print("CACHE_SNAPSHOT", phase, "process_MiB=", counters["process_bytes"]/1024**2,
          "allocator_used_bytes=", counters["allocator_used_bytes"],
          "allocator_reserved_bytes=", counters["allocator_reserved_bytes"], flush=True)
