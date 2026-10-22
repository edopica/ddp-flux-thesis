"""Bounded GPU training-only pilot on the two saved Phase 2 batches."""
import argparse
import os
import time
import socket

import numpy as np
import torch
import torch.distributed as dist
import dataset_adapter as da
import model_torch as mt
from phase4_torch import compute_state_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    a = p.parse_args()
    w = a.world_size
    rank = int(os.environ.get("RANK", 0))
    assert w in (1, 2, 4) and 1 <= a.updates <= 9750 and 2 <= a.warmup <= 100
    assert int(os.environ.get("WORLD_SIZE", 1)) == w
    index = (rank + 1) % torch.cuda.device_count() if w > 1 else 0
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    flags = mt.configure_backend(num_threads=1)
    if w > 1:
        dist.init_process_group("nccl")
    fixture = os.path.join(a.fixture_dir, "data_fixture.npz")
    bundle = os.path.join(a.fixture_dir, "initial_bundle.npz")
    with np.load(fixture) as f:
        lo, hi = rank * (12 // w), (rank + 1) * (12 // w)
        batches = [(torch.from_numpy(f["train_transformed_ref"][i, lo:hi].copy()).to(device),
                    torch.from_numpy(f["train_labels"][lo:hi].copy()).long().to(device))
                   for i in range(2)]

    def setup():
        inner = mt.make_model().to(device)
        mt.load_bundle(inner, bundle)
        opt = mt.build_optimizer(inner)
        model = torch.nn.parallel.DistributedDataParallel(
            inner, device_ids=[index], broadcast_buffers=False,
            find_unused_parameters=False, static_graph=False) if w > 1 else inner
        return inner, model, opt

    def steps(model, opt, count):
        for t in range(1, count + 1):
            x, y = batches[(t - 1) % 2]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            mt.set_optimizer_lr(opt, mt.schedule_eta(t))
            opt.step()
        return loss

    inner, model, opt = setup()
    steps(model, opt, a.warmup)
    torch.cuda.synchronize()
    # Preserve warmed DDP buckets while restoring all mathematical state.
    mt.load_bundle(inner, bundle)
    opt = mt.build_optimizer(inner)
    torch.cuda.synchronize()
    if w > 1:
        dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = steps(model, opt, a.updates)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert torch.isfinite(loss).item()
    hashes = compute_state_hashes(inner, opt, a.updates)
    records = [None] * w
    local = dict(rank=rank, seconds=elapsed, hashes=hashes, device=index)
    if w > 1:
        dist.all_gather_object(records, local)
    else:
        records = [local]
    assert all(r["hashes"] == records[0]["hashes"] for r in records)
    if rank == 0:
        seconds = max(r["seconds"] for r in records)
        da.write_json(a.out, dict(schema="static-timing-pilot-v1", framework="pytorch",
            world_size=w, global_batch=12, updates=a.updates, warmup=a.warmup,
            seconds=seconds, images_per_second=12*a.updates/seconds,
            update_seconds=seconds/a.updates, ranks=records, node=socket.gethostname(),
            fixture_sha256=da.sha256_file(fixture), bundle_sha256=da.sha256_file(bundle),
            workload="alternating-two-fixture-batches", timing="training-only", precision=flags,
            job=os.environ.get("SLURM_JOB_ID"), pass_checks=True))
        print("TIMING_PASS pytorch", w, seconds, flush=True)
    if w > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
