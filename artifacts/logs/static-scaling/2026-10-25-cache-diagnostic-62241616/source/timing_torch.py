"""Fixture and full-dataset PyTorch timing paths for static scaling."""
import argparse
import gc
import hashlib
import math
import os
import platform
import resource
import socket
import time

import numpy as np

import dataset_adapter as da
import full_dataset_common as fdc
import cache_diagnostic as cache_diag


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-dir", default=None)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--bundle", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--snapshot-dir")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--source-state-sha256", default=None)
    p.add_argument("--flux-commit", default=None)
    p.add_argument("--flux-state-sha256", default=None)
    p.add_argument("--cache-diagnostic", choices=("true", "false"), default="false")
    return p.parse_args(argv)


def torch_stack():
    import torch
    import torch.distributed as dist
    import model_torch as mt
    from phase4_torch import compute_state_hashes, optimizer_leaf_state_np, param_cpu_np
    return torch, dist, mt, compute_state_hashes, optimizer_leaf_state_np, param_cpu_np


def gpu_uuid_string(device_properties):
    value = str(device_properties.uuid)
    return value if value.startswith("GPU-") else "GPU-" + value


def memory_record(torch):
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
        "memory_allocated": int(torch.cuda.memory_allocated()),
        "memory_reserved": int(torch.cuda.memory_reserved()),
        "mem_get_info_free": int(free_bytes),
        "mem_get_info_total": int(total_bytes),
        "host_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }


def run_fixture(a):
    if a.fixture_dir is None:
        raise SystemExit("--fixture-dir is required when --dataset-dir is not set")
    torch, dist, mt, compute_state_hashes, optimizer_leaf_state_np, param_cpu_np = torch_stack()
    updates = 300 if a.updates is None else a.updates
    w = a.world_size
    rank = int(os.environ.get("RANK", 0))
    assert w in (1, 2, 4) and 1 <= updates <= 9750 and 2 <= a.warmup <= 100
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
    loss = steps(model, opt, updates)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert torch.isfinite(loss).item()
    hashes = compute_state_hashes(inner, opt, updates)
    records = [None] * w
    local = dict(rank=rank, seconds=elapsed, hashes=hashes, device=index)
    if w > 1:
        dist.all_gather_object(records, local)
    else:
        records = [local]
    assert all(r["hashes"] == records[0]["hashes"] for r in records)
    if a.snapshot_dir and rank == 0:
        os.makedirs(a.snapshot_dir, exist_ok=True)
        arrays = {}
        for name in mt.BUNDLE_NAMES:
            arrays[f"param.{name}"] = param_cpu_np(inner, name)
            m_arr, v_arr, step_int = optimizer_leaf_state_np(inner, opt, name)
            if step_int != updates:
                raise AssertionError(f"Optimizer step mismatch for {name}: got {step_int}, expected {updates}")
            arrays[f"m.{name}"] = m_arr
            arrays[f"v.{name}"] = v_arr
        arrays["step"] = np.array(updates, dtype=np.int64)
        np.savez(os.path.join(a.snapshot_dir, "state.npz"), **arrays)
    if rank == 0:
        seconds = max(r["seconds"] for r in records)
        da.write_json(a.out, dict(schema="static-timing-pilot-v1", framework="pytorch",
            world_size=w, global_batch=12, updates=updates, warmup=a.warmup,
            seconds=seconds, images_per_second=12*updates/seconds,
            update_seconds=seconds/updates, ranks=records, node=socket.gethostname(),
            fixture_sha256=da.sha256_file(fixture), bundle_sha256=da.sha256_file(bundle),
            workload="alternating-two-fixture-batches", timing="training-only", precision=flags,
            job=os.environ.get("SLURM_JOB_ID"), snapshot=bool(a.snapshot_dir), pass_checks=True))
        print("TIMING_PASS pytorch", w, seconds, flush=True)
    if w > 1:
        dist.destroy_process_group()


def run_full_dataset(a):
    torch, dist, mt, compute_state_hashes, optimizer_leaf_state_np, param_cpu_np = torch_stack()
    updates = 195 if a.updates is None else a.updates
    batch = a.batch_size
    w = a.world_size
    cache_enabled = a.cache_diagnostic == "true"
    if cache_enabled and w != 1:
        raise ValueError("--cache-diagnostic requires --world-size 1")
    rank = int(os.environ.get("RANK", 0))
    assert w in (1, 2, 4)
    assert batch in (512, 256)
    assert 0 <= a.epoch <= 49
    assert 2 <= a.warmup <= 100
    assert int(os.environ.get("WORLD_SIZE", 1)) == w
    index = (rank + 1) % torch.cuda.device_count() if w > 1 else 0
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    flags = mt.configure_backend(num_threads=1)
    if w > 1:
        dist.init_process_group("nccl")

    dataset_dir = os.path.abspath(a.dataset_dir)
    if os.environ.get("FULL_PREFLIGHT_RANK_DIR"):
        da.write_json(os.path.join(os.environ["FULL_PREFLIGHT_RANK_DIR"],
                                   "rank-%d.json" % rank),
                      dict(rank=rank, pid=os.getpid(),
                           gpu_uuid=gpu_uuid_string(
                               torch.cuda.get_device_properties(index))))
    arrays = fdc.load_epoch_arrays(dataset_dir, a.epoch)
    bundle_path = os.path.abspath(a.bundle) if a.bundle else os.path.join(dataset_dir, "initial_bundle.npz")
    if not os.path.isfile(bundle_path):
        raise FileNotFoundError("bundle not found: %s" % bundle_path)
    dataset_hashes = fdc.dataset_file_hashes(dataset_dir, verify_large=False)
    bundle_meta = fdc.bundle_info(dataset_dir, bundle_path=bundle_path)
    ids = fdc.epoch_ids(arrays, batch)
    updates_per_epoch = int(len(ids) // batch)
    def full_eta(t):
        return mt.schedule_eta(t, warmup=5 * updates_per_epoch,
                               total=50 * updates_per_epoch)
    assert 1 <= updates <= updates_per_epoch
    if batch % w != 0:
        raise AssertionError("batch %d is not divisible by world size %d" % (batch, w))
    eta_values = np.asarray(
        [full_eta(t) for t in range(1, updates + 1)], dtype="<f4")
    eta_sha256 = hashlib.sha256(eta_values.tobytes(order="C")).hexdigest()
    sample_order_sha256 = da.sha256_c_order(ids)
    update1_order_sha256 = da.sha256_c_order(fdc.global_batch_ids(arrays, batch, 1))
    transform_spec_sha256 = fdc.transform_spec_sha256(arrays["mean32"], arrays["std32"])

    def setup():
        inner = mt.make_model().to(device)
        mt.load_bundle(inner, bundle_path)
        opt = mt.build_optimizer(inner)
        model = torch.nn.parallel.DistributedDataParallel(
            inner, device_ids=[index], broadcast_buffers=False,
            find_unused_parameters=False, static_graph=False) if w > 1 else inner
        return inner, model, opt

    def batch_for(t):
        x_np, y_np, _, _ = fdc.local_batch(arrays, batch, t, rank, w)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.int64)
        return x, y

    def steps(model, opt, count):
        for t in range(1, count + 1):
            x, y = batch_for(t)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            mt.set_optimizer_lr(opt, full_eta(t))
            opt.step()
        return loss

    inner, model, opt = setup()
    diagnostic = cache_diag.diagnostic_init(
        torch, gpu_uuid_string(torch.cuda.get_device_properties(index))) if cache_enabled else None
    if cache_enabled:
        cache_diag.snapshot(diagnostic, torch, "initialized")
    steps(model, opt, a.warmup)
    torch.cuda.synchronize()
    if cache_enabled:
        cache_diag.snapshot(diagnostic, torch, "post_warmup")
    mt.load_bundle(inner, bundle_path)
    opt = mt.build_optimizer(inner)
    torch.cuda.synchronize()
    if cache_enabled:
        gc.collect()
        cache_diag.snapshot(diagnostic, torch, "post_reset_gc")
    if w > 1:
        dist.barrier()
    torch.cuda.synchronize()
    # Wall-clock epoch timestamps bracket the same synchronized boundaries as
    # perf_counter so the external nvidia-smi sampler can be clipped to the
    # post-warmup training window. Final hashing, snapshots, and teardown stay
    # outside this window.
    start_epoch = time.time()
    start = time.perf_counter()
    loss = steps(model, opt, updates)
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = time.perf_counter() - start
    loss_value = float(loss.detach().cpu().item())
    assert math.isfinite(loss_value)
    if cache_enabled:
        cache_diag.snapshot(diagnostic, torch, "post_training")
        invariant_before = compute_state_hashes(inner, opt, updates)
        gc.collect()
        cache_diag.snapshot(diagnostic, torch, "post_gc")
        torch.cuda.empty_cache()
        cache_diag.snapshot(diagnostic, torch, "post_reclaim")
    hashes = compute_state_hashes(inner, opt, updates)
    if cache_enabled:
        if hashes != invariant_before:
            raise AssertionError("cache release changed model or optimizer state")
        diagnostic["state_invariant"] = dict(before=invariant_before, after=hashes, passed=True)
    local = dict(rank=rank, seconds=elapsed, hashes=hashes, device=index,
                 training_start_epoch=start_epoch,
                 training_end_epoch=end_epoch,
                 memory=memory_record(torch))
    records = [None] * w
    if w > 1:
        dist.all_gather_object(records, local)
    else:
        records = [local]
    assert all(r["hashes"] == records[0]["hashes"] for r in records)
    if a.snapshot_dir and rank == 0:
        os.makedirs(a.snapshot_dir, exist_ok=True)
        arrays_out = {}
        for name in mt.BUNDLE_NAMES:
            arrays_out[f"param.{name}"] = param_cpu_np(inner, name)
            m_arr, v_arr, step_int = optimizer_leaf_state_np(inner, opt, name)
            if step_int != updates:
                raise AssertionError(f"Optimizer step mismatch for {name}: got {step_int}, expected {updates}")
            arrays_out[f"m.{name}"] = m_arr
            arrays_out[f"v.{name}"] = v_arr
        arrays_out["step"] = np.array(updates, dtype=np.int64)
        np.savez(os.path.join(a.snapshot_dir, "state.npz"), **arrays_out)
    source = dict(
        thesis_commit=a.source_commit or os.environ.get("SOURCE_COMMIT", ""),
        thesis_state_sha256=a.source_state_sha256 or os.environ.get("SOURCE_STATE_SHA256", ""),
        flux_commit=a.flux_commit or os.environ.get("FLUX_COMMIT", ""),
        flux_state_sha256=a.flux_state_sha256 or os.environ.get("FLUX_STATE_SHA256", ""),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        numpy_version=str(np.__version__),
        host=socket.gethostname(),
    )
    if rank == 0:
        seconds = max(r["seconds"] for r in records)
        da.write_json(a.out, dict(
            schema="static-timing-full-dataset-v1", framework="pytorch",
            world_size=w, global_batch=batch, batch_size=batch, epoch=a.epoch,
            updates=updates, warmup=a.warmup, updates_per_epoch=updates_per_epoch,
            n_eff=int(len(ids)), tail_size=int(len(arrays["epoch_permutation"]) - len(ids)),
            seconds=seconds, images_per_second=batch*updates/seconds,
            update_seconds=seconds/updates, ranks=records,
            node=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"),
            dataset_dir=dataset_dir, dataset_hashes=dataset_hashes,
            sample_order_sha256=sample_order_sha256,
            update1_order_sha256=update1_order_sha256,
            transform_spec_sha256=transform_spec_sha256,
            bundle_hash=bundle_meta["bundle_hash"],
            bundle_file_sha256=bundle_meta["bundle_file_sha256"],
            loader_policy="mmap opened once per process before timing; synchronous numpy batch assembly and transform; no DataLoader workers",
            timing="full-dataset-end-to-end",
            timing_policy="timed region includes batch assembly, transform, H2D, forward, backward, and optimizer step",
            training_window_basis="per-rank wall-clock epoch seconds recorded at the synchronized start and after the synchronized end of the measured updates; warmup, reset, final hashing, snapshots, and teardown are outside the window",
            warmup_policy="%d discarded updates at t=1..%d; model and optimizer reset from the bundle; measured region restarts at update 1 and data at batch 1"
            % (a.warmup, a.warmup),
            workload="full-dataset-epoch", precision=flags,
            cpu_policy="torch.set_num_threads(1) via mt.configure_backend; CPU batch assembly only",
            source=source, snapshot=bool(a.snapshot_dir), pass_checks=True,
            eta_sha256=eta_sha256, final_loss=loss_value, loss_finite=True,
            **({"cache_diagnostic": diagnostic} if cache_enabled else {})))
        print("TIMING_PASS pytorch-full", w, seconds, flush=True)
    if w > 1:
        dist.destroy_process_group()


def main(argv=None):
    a = parse_args(argv)
    if a.cache_diagnostic == "true" and not a.dataset_dir:
        raise ValueError("--cache-diagnostic requires --dataset-dir")
    if a.dataset_dir:
        run_full_dataset(a)
    else:
        run_fixture(a)


if __name__ == "__main__":
    main()
