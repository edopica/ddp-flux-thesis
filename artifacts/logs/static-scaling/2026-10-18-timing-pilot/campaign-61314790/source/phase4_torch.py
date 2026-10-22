#!/usr/bin/env python3
"""Phase 4 PyTorch single-process reference, shard-reference, split-control, and DDP driver."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import dataset_adapter as da  # noqa: E402
import model_torch as mt  # noqa: E402

UPDATES: Tuple[str, str, str] = ("u1", "u2", "u3")
GLOBAL_BATCH_SIZE: int = 12
NUM_CLASSES: int = 200
INPUT_HW: Tuple[int, int] = (64, 64)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 PyTorch reference, shard-reference, split-control, and DDP runner."
    )
    parser.add_argument(
        "--fixture-dir",
        default="artifacts/fixtures/static-scaling/current",
        help="Path to Phase 2 static-scaling fixture directory.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional override path to init_bundle.npz (defaults to <fixture-dir>/init_bundle.npz).",
    )
    parser.add_argument(
        "--mode",
        choices=("reference", "shard-reference", "ddp", "split-control"),
        required=True,
        help="Execution mode.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device type.",
    )
    parser.add_argument(
        "--backend",
        choices=("gloo", "nccl"),
        default="gloo",
        help="torch.distributed communication backend for --mode ddp.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="World size (1, 2, or 4).",
    )
    parser.add_argument(
        "--global-batch",
        type=int,
        default=GLOBAL_BATCH_SIZE,
        help="Global batch size; defaults to the Phase 4 fixture size (12).",
    )
    parser.add_argument(
        "--out-dir",
        "--out",
        dest="out_dir",
        required=True,
        help="Root output directory.",
    )
    parser.add_argument(
        "--rank-out",
        default=None,
        help="Optional explicit output directory override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files if present.",
    )
    parser.add_argument(
        "--source-commit",
        default="",
        help="Thesis repository git commit SHA.",
    )
    parser.add_argument(
        "--source-state-sha256",
        default="",
        help="Thesis repository working tree state SHA-256.",
    )
    parser.add_argument(
        "--flux-commit",
        default="",
        help="Flux repository git commit SHA.",
    )
    parser.add_argument(
        "--flux-state-sha256",
        default="",
        help="Flux repository working tree state SHA-256.",
    )
    return parser.parse_args(argv)


def resolve_bundle_path(args: argparse.Namespace) -> str:
    if args.bundle:
        return os.path.abspath(args.bundle)
    return os.path.abspath(os.path.join(args.fixture_dir, "init_bundle.npz"))


def load_fixture_data(
    fixture_dir: str,
    expected_batch: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    data_path = os.path.abspath(os.path.join(fixture_dir, "data_fixture.npz"))
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Missing data fixture: {data_path}")

    with np.load(data_path, allow_pickle=False) as archive:
        refs = np.array(archive["train_transformed_ref"], dtype=np.float32, copy=True)
        labels = np.array(archive["train_labels"], dtype=np.int64, copy=True)

    if refs.ndim != 5 or refs.shape[0] != 2 or refs.shape[2:] != (3,) + INPUT_HW:
        raise ValueError(f"Unexpected train_transformed_ref shape: {refs.shape}")
    batch = int(refs.shape[1])
    if expected_batch is not None and batch != int(expected_batch):
        raise ValueError(
            f"Fixture batch {batch} != --global-batch {int(expected_batch)}"
        )
    if labels.shape != (batch,):
        raise ValueError(f"Unexpected train_labels shape: {labels.shape}")

    update_inputs = {
        "u1": np.ascontiguousarray(refs[0], dtype=np.float32),
        "u2": np.ascontiguousarray(refs[1], dtype=np.float32),
        "u3": np.ascontiguousarray(refs[0], dtype=np.float32),
    }
    update_labels = {
        "u1": np.ascontiguousarray(labels, dtype=np.int64),
        "u2": np.ascontiguousarray(labels, dtype=np.int64),
        "u3": np.ascontiguousarray(labels, dtype=np.int64),
    }

    perm_path = os.path.abspath(os.path.join(fixture_dir, "permutation_seed20260412.npz"))
    fixture_hashes = {
        "data_fixture_path": data_path,
        "data_fixture_sha256": da.sha256_file(data_path),
        "permutation_path": perm_path if os.path.isfile(perm_path) else "",
        "permutation_sha256": da.sha256_file(perm_path) if os.path.isfile(perm_path) else "",
    }
    return update_inputs, update_labels, fixture_hashes


def shard_batch(
    x_global: np.ndarray,
    y_global: np.ndarray,
    rank: int,
    world: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if world not in (1, 2, 4):
        raise ValueError(f"Unsupported world_size={world}; expected 1, 2, or 4")
    if not (0 <= rank < world):
        raise ValueError(f"Invalid rank={rank} for world_size={world}")
    batch = int(x_global.shape[0])
    if int(y_global.shape[0]) != batch:
        raise ValueError(
            f"Input/label batch mismatch: {batch} != {int(y_global.shape[0])}"
        )
    if batch % world != 0:
        raise ValueError(f"Global batch {batch} not divisible by world={world}")
    chunk = batch // world
    ids = np.arange(rank * chunk, (rank + 1) * chunk, dtype=np.int64)
    x_local = np.ascontiguousarray(x_global[ids], dtype=np.float32)
    y_local = np.ascontiguousarray(y_global[ids], dtype=np.int64)
    return ids, x_local, y_local


def param_cpu_np(model: torch.nn.Module, name: str) -> np.ndarray:
    named = dict(model.named_parameters())
    arr = named[name].detach().cpu().numpy().astype(np.float32, copy=True)
    return np.ascontiguousarray(arr, dtype=np.float32)


def grad_cpu_np(model: torch.nn.Module, name: str) -> np.ndarray:
    named = dict(model.named_parameters())
    grad = named[name].grad
    if grad is None:
        raise RuntimeError(f"Missing gradient for parameter {name}")
    arr = grad.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.ascontiguousarray(arr, dtype=np.float32)


def optimizer_leaf_state_np(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    named = dict(model.named_parameters())
    param = named[name]
    if param not in optimizer.state:
        raise RuntimeError(f"Missing optimizer state for parameter {name}")
    st = optimizer.state[param]
    exp_avg = np.ascontiguousarray(
        st["exp_avg"].detach().cpu().numpy().astype(np.float32, copy=True),
        dtype=np.float32,
    )
    exp_avg_sq = np.ascontiguousarray(
        st["exp_avg_sq"].detach().cpu().numpy().astype(np.float32, copy=True),
        dtype=np.float32,
    )
    step_val = st["step"]
    step_int = int(step_val.item()) if torch.is_tensor(step_val) else int(step_val)
    return exp_avg, exp_avg_sq, step_int


def compute_state_hashes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_step: int,
) -> Dict[str, object]:
    param_arrays: List[np.ndarray] = []
    m_arrays: List[np.ndarray] = []
    v_arrays: List[np.ndarray] = []
    for name in mt.BUNDLE_NAMES:
        param_arrays.append(param_cpu_np(model, name))
        m_arr, v_arr, step_int = optimizer_leaf_state_np(model, optimizer, name)
        if step_int != int(expected_step):
            raise AssertionError(
                f"Optimizer step mismatch for {name}: got {step_int}, expected {expected_step}"
            )
        m_arrays.append(m_arr)
        v_arrays.append(v_arr)
    return {
        "param": da.bundle_hash_of_arrays(param_arrays),
        "m": da.bundle_hash_of_arrays(m_arrays),
        "v": da.bundle_hash_of_arrays(v_arrays),
        "step": int(expected_step),
    }


def provenance_dict(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "source_commit": args.source_commit,
        "source_state_sha256": args.source_state_sha256,
        "flux_commit": args.flux_commit,
        "flux_state_sha256": args.flux_state_sha256,
    }


def run_reference_or_shard_reference(args: argparse.Namespace) -> None:
    world = int(args.world_size)
    if args.mode == "reference" and world != 1:
        raise ValueError(f"--mode reference requires --world-size 1, got {world}")
    if world not in (1, 2, 4):
        raise ValueError(f"Unsupported --world-size {world}; expected 1, 2, or 4")

    flags = mt.configure_backend(num_threads=1)
    dev = mt.resolve_device(args.device)
    bundle_path = resolve_bundle_path(args)
    update_inputs, update_labels, fixture_hashes = load_fixture_data(args.fixture_dir, args.global_batch)

    if args.rank_out:
        target_dir = os.path.abspath(args.rank_out)
    elif args.mode == "reference":
        target_dir = os.path.abspath(os.path.join(args.out_dir, "reference", "pytorch", args.device))
    else:
        target_dir = os.path.abspath(
            os.path.join(args.out_dir, "shard_reference", "pytorch", args.device, f"w{world}")
        )
    os.makedirs(target_dir, exist_ok=True)

    npz_path = os.path.join(target_dir, "phase4_reference.npz")
    json_path = os.path.join(target_dir, "phase4_reference.json")
    manifest_path = os.path.join(target_dir, "phase4_reference_manifest.json")

    model = mt.make_model().to(dev)
    info = mt.load_bundle(model, bundle_path)
    init_param_hash = da.bundle_hash_of_arrays([param_cpu_np(model, n) for n in mt.BUNDLE_NAMES])
    optimizer = mt.build_optimizer(model)
    named_params = dict(model.named_parameters())

    arrays: Dict[str, np.ndarray] = {}
    etas: Dict[str, float] = {}
    losses: Dict[str, float] = {}
    shard_losses: Dict[str, List[float]] = {}
    state_hashes: Dict[str, Dict[str, object]] = {}

    for t, tag in enumerate(UPDATES, start=1):
        x_global = update_inputs[tag]
        y_global = update_labels[tag]
        eta = float(mt.schedule_eta(t))
        etas[tag] = eta

        arrays[f"input.{tag}"] = np.ascontiguousarray(x_global, dtype=np.float32)
        arrays[f"label.{tag}"] = np.ascontiguousarray(y_global, dtype=np.int64)
        arrays[f"eta.{tag}"] = np.array([eta], dtype=np.float64)

        logits_shards: List[np.ndarray] = []
        act_shards: Dict[str, List[np.ndarray]] = {name: [] for name in mt.ACTIVATION_NAMES}
        grad_accum: Dict[str, torch.Tensor] = {}
        loss_terms: List[torch.Tensor] = []
        loss_floats: List[float] = []

        for r in range(world):
            _, x_local, y_local = shard_batch(x_global, y_global, r, world)
            x_dev = torch.from_numpy(x_local).to(device=dev, dtype=torch.float32)
            y_dev = torch.from_numpy(y_local).to(device=dev, dtype=torch.int64)

            optimizer.zero_grad(set_to_none=True)
            if tag == "u1":
                logits_dev, acts_dev = mt.forward_with_activations(model, x_dev)
                for act_name in mt.ACTIVATION_NAMES:
                    act_np = np.ascontiguousarray(
                        acts_dev[act_name].detach().cpu().numpy().astype(np.float32, copy=True),
                        dtype=np.float32,
                    )
                    act_shards[act_name].append(act_np)
            else:
                logits_dev = model(x_dev)

            loss_dev = F.cross_entropy(logits_dev, y_dev, reduction="mean")
            loss_dev.backward()

            logits_np = np.ascontiguousarray(
                logits_dev.detach().cpu().numpy().astype(np.float32, copy=True),
                dtype=np.float32,
            )
            logits_shards.append(logits_np)
            loss_terms.append(loss_dev.detach())
            loss_floats.append(float(loss_dev.detach().cpu().item()))

            for name in mt.BUNDLE_NAMES:
                g = named_params[name].grad
                if g is None:
                    raise RuntimeError(f"Missing gradient on shard {r} for {name}")
                g_detached = g.detach().clone()
                if r == 0:
                    grad_accum[name] = g_detached
                else:
                    grad_accum[name] = grad_accum[name] + g_detached

        if world == 1:
            loss_mean_tensor = loss_terms[0]
        else:
            loss_sum = loss_terms[0]
            for lt in loss_terms[1:]:
                loss_sum = loss_sum + lt
            loss_mean_tensor = loss_sum / float(world)

        loss_val = float(loss_mean_tensor.cpu().item())
        losses[tag] = loss_val
        shard_losses[tag] = loss_floats
        arrays[f"loss.{tag}"] = np.array([loss_val], dtype=np.float32)
        arrays[f"logits.{tag}"] = np.ascontiguousarray(
            np.concatenate(logits_shards, axis=0), dtype=np.float32
        )

        if tag == "u1":
            for act_name in mt.ACTIVATION_NAMES:
                act_key = act_name if act_name.startswith("act.") else f"act.{act_name}"
                arrays[act_key] = np.ascontiguousarray(
                    np.concatenate(act_shards[act_name], axis=0), dtype=np.float32
                )

        for name in mt.BUNDLE_NAMES:
            g_avg = grad_accum[name] if world == 1 else (grad_accum[name] / float(world))
            named_params[name].grad = g_avg.clone()
            if tag in ("u1", "u3"):
                arrays[f"grad.{tag}.{name}"] = np.ascontiguousarray(
                    g_avg.detach().cpu().numpy().astype(np.float32, copy=True),
                    dtype=np.float32,
                )

        mt.set_optimizer_lr(optimizer, eta)
        optimizer.step()
        state_hashes[tag] = compute_state_hashes(model, optimizer, t)

    for name in mt.BUNDLE_NAMES:
        arrays[f"param.u3.{name}"] = param_cpu_np(model, name)
        m_arr, v_arr, _ = optimizer_leaf_state_np(model, optimizer, name)
        arrays[f"m.u3.{name}"] = m_arr
        arrays[f"v.u3.{name}"] = v_arr

    arrays["step.u3"] = np.array([3], dtype=np.int64)

    if len(arrays) != 337:
        raise AssertionError(f"Expected 337 reference arrays, got {len(arrays)}")

    np.savez(npz_path, **arrays)
    npz_sha256 = da.sha256_file(npz_path)

    meta = {
        "framework": "pytorch",
        "mode": args.mode,
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "world_size": world,
        "updates": 3,
        "global_batch_size": int(args.global_batch),
        "local_batch_size": int(args.global_batch) // world,
        "parameter_count": mt.PARAMETER_COUNT,
        "array_count": mt.ARRAY_COUNT,
        "reference_array_count": len(arrays),
        "bundle_path": bundle_path,
        "bundle_hash": info["bundle_hash"],
        "init_param_hash": init_param_hash,
        "etas": etas,
        "losses": losses,
        "shard_losses": shard_losses,
        "state_hashes": state_hashes,
        "final_param_hash": state_hashes["u3"]["param"],
        "final_m_hash": state_hashes["u3"]["m"],
        "final_v_hash": state_hashes["u3"]["v"],
        "final_step": 3,
        "backend_flags": flags,
        "fixture": fixture_hashes,
        "npz_path": npz_path,
        "npz_sha256": npz_sha256,
        **provenance_dict(args),
        "created_at": da.utc_now(),
    }
    da.write_json(json_path, meta)

    manifest_arrays = {}
    for key, arr in arrays.items():
        manifest_arrays[key] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256_c": da.sha256_c_order(arr),
        }
    manifest = {
        "framework": "pytorch",
        "mode": args.mode,
        "device": args.device,
        "world_size": world,
        "npz_path": npz_path,
        "npz_sha256": npz_sha256,
        "array_count": len(arrays),
        "arrays": manifest_arrays,
        **provenance_dict(args),
        "created_at": da.utc_now(),
    }
    da.write_json(manifest_path, manifest)


def run_split_control(args: argparse.Namespace) -> None:
    if int(args.world_size) != 1:
        raise ValueError(f"--mode split-control requires --world-size 1, got {args.world_size}")

    flags = mt.configure_backend(num_threads=1)
    dev = mt.resolve_device(args.device)
    bundle_path = resolve_bundle_path(args)
    update_inputs, update_labels, fixture_hashes = load_fixture_data(args.fixture_dir, args.global_batch)

    if args.rank_out:
        target_dir = os.path.abspath(args.rank_out)
    else:
        target_dir = os.path.abspath(
            os.path.join(args.out_dir, "split", "pytorch", args.device)
        )
    os.makedirs(target_dir, exist_ok=True)

    npz_path = os.path.join(target_dir, "split_control.npz")
    json_path = os.path.join(target_dir, "split_control.json")

    model = mt.make_model().to(dev)
    info = mt.load_bundle(model, bundle_path)
    optimizer = mt.build_optimizer(model)

    arrays: Dict[str, np.ndarray] = {}
    summary_updates: Dict[str, Dict[str, float]] = {}

    for t, tag in enumerate(UPDATES, start=1):
        x_global = update_inputs[tag]
        y_global = update_labels[tag]
        eta = float(mt.schedule_eta(t))

        if tag in ("u1", "u3"):
            arrays[f"input.{tag}"] = np.ascontiguousarray(x_global, dtype=np.float32)
            arrays[f"label.{tag}"] = np.ascontiguousarray(y_global, dtype=np.int64)
            arrays[f"eta.{tag}"] = np.array([eta], dtype=np.float64)

            _, x_h1, y_h1 = shard_batch(x_global, y_global, 0, 2)
            _, x_h2, y_h2 = shard_batch(x_global, y_global, 1, 2)

            optimizer.zero_grad(set_to_none=True)
            loss_h1 = F.cross_entropy(
                model(torch.from_numpy(x_h1).to(device=dev, dtype=torch.float32)),
                torch.from_numpy(y_h1).to(device=dev, dtype=torch.int64),
                reduction="mean",
            )
            loss_h1.backward()
            g_h1 = {name: grad_cpu_np(model, name) for name in mt.BUNDLE_NAMES}

            optimizer.zero_grad(set_to_none=True)
            loss_h2 = F.cross_entropy(
                model(torch.from_numpy(x_h2).to(device=dev, dtype=torch.float32)),
                torch.from_numpy(y_h2).to(device=dev, dtype=torch.int64),
                reduction="mean",
            )
            loss_h2.backward()
            g_h2 = {name: grad_cpu_np(model, name) for name in mt.BUNDLE_NAMES}

            optimizer.zero_grad(set_to_none=True)
            loss_full = F.cross_entropy(
                model(torch.from_numpy(x_global).to(device=dev, dtype=torch.float32)),
                torch.from_numpy(y_global).to(device=dev, dtype=torch.int64),
                reduction="mean",
            )
            loss_full.backward()
            g_full = {name: grad_cpu_np(model, name) for name in mt.BUNDLE_NAMES}

            max_full_vs_halfmean = 0.0
            max_full_vs_half1 = 0.0
            max_full_vs_half2 = 0.0

            for name in mt.BUNDLE_NAMES:
                halfavg = np.ascontiguousarray(
                    (0.5 * (g_h1[name] + g_h2[name])).astype(np.float32),
                    dtype=np.float32,
                )
                arrays[f"grad.{tag}.full.{name}"] = g_full[name]
                arrays[f"grad.{tag}.half1.{name}"] = g_h1[name]
                arrays[f"grad.{tag}.half2.{name}"] = g_h2[name]
                arrays[f"grad.{tag}.halfavg.{name}"] = halfavg

                f64 = g_full[name].astype(np.float64)
                h1_64 = g_h1[name].astype(np.float64)
                h2_64 = g_h2[name].astype(np.float64)
                hm_64 = 0.5 * (h1_64 + h2_64)
                max_full_vs_halfmean = max(
                    max_full_vs_halfmean, float(np.max(np.abs(f64 - hm_64)))
                )
                max_full_vs_half1 = max(
                    max_full_vs_half1, float(np.max(np.abs(f64 - h1_64)))
                )
                max_full_vs_half2 = max(
                    max_full_vs_half2, float(np.max(np.abs(f64 - h2_64)))
                )

            summary_updates[tag] = {
                "loss_full": float(loss_full.detach().cpu().item()),
                "loss_half1": float(loss_h1.detach().cpu().item()),
                "loss_half2": float(loss_h2.detach().cpu().item()),
                "max_abs_full_vs_halfmean": max_full_vs_halfmean,
                "max_abs_full_vs_half1": max_full_vs_half1,
                "max_abs_full_vs_half2": max_full_vs_half2,
            }
        else:
            optimizer.zero_grad(set_to_none=True)
            loss_u2 = F.cross_entropy(
                model(torch.from_numpy(x_global).to(device=dev, dtype=torch.float32)),
                torch.from_numpy(y_global).to(device=dev, dtype=torch.int64),
                reduction="mean",
            )
            loss_u2.backward()

        mt.set_optimizer_lr(optimizer, eta)
        optimizer.step()

    np.savez(npz_path, **arrays)
    meta = {
        "framework": "pytorch",
        "mode": "split-control",
        "device": args.device,
        "world_size": 1,
        "bundle_path": bundle_path,
        "bundle_hash": info["bundle_hash"],
        "updates": summary_updates,
        "backend_flags": flags,
        "fixture": fixture_hashes,
        "npz_path": npz_path,
        "npz_sha256": da.sha256_file(npz_path),
        **provenance_dict(args),
        "created_at": da.utc_now(),
    }
    da.write_json(json_path, meta)


def run_ddp(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if world != int(args.world_size):
        raise ValueError(
            f"WORLD_SIZE env ({world}) != --world-size ({args.world_size})"
        )
    if world not in (1, 2, 4):
        raise ValueError(f"Unsupported world_size={world}; expected 1, 2, or 4")

    if args.device == "cuda":
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError("CUDA requested for DDP, but torch.cuda.device_count() == 0")
        device_index = 0 if world == 1 else (rank + 1) % count
        torch.cuda.set_device(device_index)
        dev = torch.device("cuda", device_index)
    else:
        device_index = 0
        dev = torch.device("cpu")

    flags = mt.configure_backend(num_threads=1)

    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        rank=rank,
        world_size=world,
    )
    try:
        bundle_path = resolve_bundle_path(args)
        update_inputs, update_labels, fixture_hashes = load_fixture_data(args.fixture_dir, args.global_batch)

        inner_model = mt.make_model().to(dev)
        info = mt.load_bundle(inner_model, bundle_path)
        init_param_hash = da.bundle_hash_of_arrays(
            [param_cpu_np(inner_model, n) for n in mt.BUNDLE_NAMES]
        )

        gathered_bundle_hashes: List[Optional[str]] = [None for _ in range(world)]
        dist.all_gather_object(gathered_bundle_hashes, info["bundle_hash"])
        if any(h != gathered_bundle_hashes[0] for h in gathered_bundle_hashes):
            raise RuntimeError(
                f"Rank bundle_hash mismatch across world={world}: {gathered_bundle_hashes}"
            )

        optimizer = mt.build_optimizer(inner_model)

        ddp_model = DistributedDataParallel(
            inner_model,
            device_ids=[device_index] if args.device == "cuda" else None,
            output_device=device_index if args.device == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=False,
        )

        arrays: Dict[str, np.ndarray] = {
            "meta.rank": np.array(rank, dtype=np.int64),
            "meta.world_size": np.array(world, dtype=np.int64),
            "meta.device_index": np.array(device_index, dtype=np.int64),
            "meta.updates": np.array(3, dtype=np.int64),
        }

        etas: Dict[str, float] = {}
        losses: Dict[str, float] = {}
        state_hashes: Dict[str, Dict[str, object]] = {}
        shard_ids_list: List[int] = []

        for t, tag in enumerate(UPDATES, start=1):
            x_global = update_inputs[tag]
            y_global = update_labels[tag]
            ids, x_local, y_local = shard_batch(x_global, y_global, rank, world)
            if t == 1:
                shard_ids_list = [int(i) for i in ids.tolist()]

            if tag in ("u1", "u3"):
                arrays[f"input.{tag}"] = x_local
                arrays[f"label.{tag}"] = y_local

            x_local_dev = torch.from_numpy(x_local).to(device=dev, dtype=torch.float32)
            y_local_dev = torch.from_numpy(y_local).to(device=dev, dtype=torch.int64)

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(x_local_dev)
            loss = F.cross_entropy(logits, y_local_dev)
            loss.backward()

            loss_val = float(loss.detach().cpu().item())
            eta = float(mt.schedule_eta(t))
            etas[tag] = eta
            losses[tag] = loss_val

            arrays[f"eta.{tag}"] = np.array([eta], dtype=np.float64)
            arrays[f"loss.{tag}"] = np.array([loss_val], dtype=np.float32)
            if tag in ("u1", "u3"):
                arrays[f"logits.{tag}"] = np.ascontiguousarray(
                    logits.detach().cpu().numpy().astype(np.float32, copy=True),
                    dtype=np.float32,
                )

            for name in mt.BUNDLE_NAMES:
                arrays[f"syncgrad.{tag}.{name}"] = grad_cpu_np(inner_model, name)

            mt.set_optimizer_lr(optimizer, eta)
            optimizer.step()
            state_hashes[tag] = compute_state_hashes(inner_model, optimizer, t)

        for name in mt.BUNDLE_NAMES:
            arrays[f"param.u3.{name}"] = param_cpu_np(inner_model, name)
            m_arr, v_arr, _ = optimizer_leaf_state_np(inner_model, optimizer, name)
            arrays[f"m.u3.{name}"] = m_arr
            arrays[f"v.u3.{name}"] = v_arr

        arrays["step.u3"] = np.array([3], dtype=np.int64)

        rank_dir = os.path.abspath(
            args.rank_out or os.path.join(args.out_dir, "ddp", "pytorch", f"w{world}")
        )
        os.makedirs(rank_dir, exist_ok=True)

        rank_npz_path = os.path.join(rank_dir, f"rank{rank}.npz")
        rank_json_path = os.path.join(rank_dir, f"rank{rank}.json")

        np.savez(rank_npz_path, **arrays)
        rank_npz_sha256 = da.sha256_file(rank_npz_path)

        rank_arrays_meta = {}
        for key, arr in arrays.items():
            digest = da.sha256_c_order(arr)
            rank_arrays_meta[key] = {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "sha256": digest,
                "sha256_c": digest,
            }

        rank_meta = {
            "framework": "pytorch",
            "mode": "ddp",
            "backend": args.backend,
            "device": args.device,
            "device_index": int(device_index),
            "cuda_device_name": (
                torch.cuda.get_device_name(device_index) if args.device == "cuda" else "cpu"
            ),
            "rank": int(rank),
            "local_rank": int(local_rank),
            "world_size": int(world),
            "updates": 3,
            "global_batch_size": int(args.global_batch),
            "local_batch_size": int(args.global_batch) // world,
            "shard_sample_ids": shard_ids_list,
            "shard_ids": {
                "u1": shard_ids_list,
                "u2": shard_ids_list,
                "u3": shard_ids_list,
            },
            "parameter_count": mt.PARAMETER_COUNT,
            "array_count": mt.ARRAY_COUNT,
            "bundle_path": bundle_path,
            "bundle_hash": info["bundle_hash"],
            "init_param_hash": init_param_hash,
            "etas": etas,
            "losses": losses,
            "state_hashes": state_hashes,
            "final_param_hash": state_hashes["u3"]["param"],
            "final_m_hash": state_hashes["u3"]["m"],
            "final_v_hash": state_hashes["u3"]["v"],
            "final_step": 3,
            "backend_flags": flags,
            "ddp_config": {
                "broadcast_buffers": False,
                "find_unused_parameters": False,
                "static_graph": False,
            },
            "fixture": fixture_hashes,
            "npz_path": rank_npz_path,
            "npz_sha256": rank_npz_sha256,
            "arrays": rank_arrays_meta,
            **provenance_dict(args),
            "created_at": da.utc_now(),
        }
        da.write_json(rank_json_path, rank_meta)

        dist.barrier()

        gathered_metas: List[Optional[Dict[str, object]]] = [None for _ in range(world)]
        dist.all_gather_object(gathered_metas, rank_meta)

        if rank == 0:
            ordered_metas = sorted(
                [m for m in gathered_metas if m is not None],
                key=lambda item: int(item["rank"]),  # type: ignore[arg-type]
            )
            r0_hashes = ordered_metas[0]["state_hashes"]
            all_equal = all(m["state_hashes"] == r0_hashes for m in ordered_metas)
            summary = {
                "framework": "pytorch",
                "mode": "ddp",
                "backend": args.backend,
                "device": args.device,
                "world_size": int(world),
                "updates": 3,
                "global_batch_size": int(args.global_batch),
                "local_batch_size": int(args.global_batch) // world,
                "parameter_count": mt.PARAMETER_COUNT,
                "array_count": mt.ARRAY_COUNT,
                "bundle_path": bundle_path,
                "bundle_hash": info["bundle_hash"],
                "init_param_hash": init_param_hash,
                "all_ranks_bundle_hash_equal": True,
                "all_ranks_state_hashes_equal": bool(all_equal),
                "state_hashes_by_rank": {
                    str(m["rank"]): m["state_hashes"] for m in ordered_metas
                },
                "ranks": ordered_metas,
                "backend_flags": flags,
                "fixture": fixture_hashes,
                **provenance_dict(args),
                "created_at": da.utc_now(),
            }
            summary_path = os.path.join(rank_dir, "summary.json")
            da.write_json(summary_path, summary)

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode in ("reference", "shard-reference"):
        run_reference_or_shard_reference(args)
    elif args.mode == "split-control":
        run_split_control(args)
    elif args.mode == "ddp":
        run_ddp(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
