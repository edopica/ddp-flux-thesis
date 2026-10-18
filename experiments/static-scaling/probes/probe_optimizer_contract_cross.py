#!/usr/bin/env python3
"""Cross-framework optimizer-contract probe: PyTorch side.

Runs on CPU on a Leonardo compute node with the pinned benchmark venv.  It
asserts the actual training-path AdamW group configuration on the full
11,271,432-parameter model without running 11M-parameter updates, then runs
controlled small-leaf update sequences with nonzero moments.

The probe-specific comparison rule is documented in
``probe_optimizer_contract_cross.jl`` and enforced by
``compare_optimizer_contract.py``:

    |a - b| <= 1e-7 + 5e-7 * max(|a|, |b|)

This is a sensitivity probe, not the Phase 3 acceptance rule.  The library
optimizer is used for every measured update; the Float64 arithmetic oracle
lives only in the comparator.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

PROBE_DIR = Path(__file__).resolve().parent
THESIS = PROBE_DIR.parents[2]
SCRIPTS = THESIS / "scripts" / "static_scaling"
sys.path.insert(0, str(SCRIPTS))

import model_torch as mt  # noqa: E402

PROBE_ETA = 1.0e-2
PROBE_BETA = (0.9, 0.999)
PROBE_EPS = 1.0e-8
PROBE_WD = 1.0e-4


def run_case(name: str, x0_decay, grads_decay, x0_nodecay, grads_nodecay):
    """Run one two-group AdamW sequence and return both leaves' records."""
    p_decay = torch.nn.Parameter(torch.tensor(x0_decay, dtype=torch.float32))
    p_nodecay = torch.nn.Parameter(torch.tensor(x0_nodecay, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p_decay], "weight_decay": PROBE_WD},
            {"params": [p_nodecay], "weight_decay": 0.0},
        ],
        lr=PROBE_ETA,
        betas=PROBE_BETA,
        eps=PROBE_EPS,
    )
    assert len(optimizer.param_groups) == 2
    assert [g["weight_decay"] for g in optimizer.param_groups] == [PROBE_WD, 0.0]
    for group in optimizer.param_groups:
        assert group["lr"] == PROBE_ETA
        assert tuple(group["betas"]) == PROBE_BETA
        assert group["eps"] == PROBE_EPS

    params = {"decay": p_decay, "nodecay": p_nodecay}
    grads = {"decay": grads_decay, "nodecay": grads_nodecay}
    x_steps = {"decay": [], "nodecay": []}
    m_before = {}
    v_before = {}
    m_final = {}
    v_final = {}
    step_final = {}

    nsteps = len(grads_decay)
    assert nsteps == len(grads_nodecay) and nsteps >= 2
    for i in range(nsteps):
        for leaf in ("decay", "nodecay"):
            params[leaf].grad = torch.tensor(grads[leaf][i], dtype=torch.float32)
        if i == nsteps - 1:
            for leaf in ("decay", "nodecay"):
                state = optimizer.state[params[leaf]]
                assert "exp_avg" in state and "exp_avg_sq" in state, \
                    "moments must be nonzero before the measured update"
                m_before[leaf] = state["exp_avg"].detach().numpy().astype(np.float64)
                v_before[leaf] = state["exp_avg_sq"].detach().numpy().astype(np.float64)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for leaf in ("decay", "nodecay"):
            x_steps[leaf].append(
                params[leaf].detach().numpy().astype(np.float64).tolist())

    for leaf in ("decay", "nodecay"):
        state = optimizer.state[params[leaf]]
        m_final[leaf] = state["exp_avg"].detach().numpy().astype(np.float64)
        v_final[leaf] = state["exp_avg_sq"].detach().numpy().astype(np.float64)
        step_final[leaf] = int(state["step"].item())

    out = {"name": name, "lambda": {"decay": PROBE_WD, "nodecay": 0.0}}
    for leaf in ("decay", "nodecay"):
        out[leaf] = {
            "name": f"{name}.{leaf}",
            "lambda": PROBE_WD if leaf == "decay" else 0.0,
            "x0": np.asarray(x0_decay if leaf == "decay" else x0_nodecay,
                             dtype=np.float64).tolist(),
            "grads": [np.asarray(g, dtype=np.float64).tolist()
                      for g in (grads_decay if leaf == "decay" else grads_nodecay)],
            "x_steps": x_steps[leaf],
            "x_final": x_steps[leaf][-1],
            "m_before_measured": m_before[leaf].tolist(),
            "v_before_measured": v_before[leaf].tolist(),
            "m_final": m_final[leaf].tolist(),
            "v_final": v_final[leaf].tolist(),
            "step_final": step_final[leaf],
            "all_nonzero_m_before_measured": bool(np.all(m_before[leaf] != 0.0)),
            "all_nonzero_v_before_measured": bool(np.all(v_before[leaf] != 0.0)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="initial_bundle.npz")
    parser.add_argument("--out", required=True, help="output probe_torch.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    bundle = Path(args.bundle).resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)

    # Actual training-path model and optimizer configuration.
    model = mt.make_model()
    load_info = mt.load_bundle(model, str(bundle))
    decay, no_decay = mt.parameter_groups(model)
    optimizer = mt.build_optimizer(model, lr=mt.ETA_MAX)
    assert len(optimizer.param_groups) == 2
    assert len(optimizer.param_groups[0]["params"]) == 21
    assert len(optimizer.param_groups[1]["params"]) == 41
    assert optimizer.param_groups[0]["weight_decay"] == mt.WEIGHT_DECAY
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    training_path = {
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "leaf_count": int(len(list(model.parameters()))),
        "decay_leaf_count": int(len(decay)),
        "nodecay_leaf_count": int(len(no_decay)),
        "decay_lambda_values": [float(optimizer.param_groups[0]["weight_decay"])],
        "nodecay_lambda_values": [float(optimizer.param_groups[1]["weight_decay"])],
        "decay_leaf_names_head": [name for name, _ in decay[:5]],
        "nodecay_leaf_names_head": [name for name, _ in no_decay[:5]],
        "rule_meta_ok": all(
            tuple(g["betas"]) == mt.BETAS
            and g["eps"] == mt.EPS
            and g["lr"] == mt.ETA_MAX
            and g["amsgrad"] is False
            and g["maximize"] is False
            for g in optimizer.param_groups),
        "eta_max": float(mt.ETA_MAX),
        "beta": [float(b) for b in mt.BETAS],
        "epsilon": float(mt.EPS),
        "weight_decay": float(mt.WEIGHT_DECAY),
        "bundle_array_count": int(load_info["array_count"]),
        "bundle_element_count": int(load_info["element_count"]),
    }

    schedule = {
        "eta_max": float(mt.ETA_MAX),
        "W": int(mt.WARMUP_UPDATES),
        "T": int(mt.TOTAL_UPDATES),
        "t1": float(mt.schedule_eta(1)),
        "tW": float(mt.schedule_eta(mt.WARMUP_UPDATES)),
        "tW1": float(mt.schedule_eta(mt.WARMUP_UPDATES + 1)),
        "tT": float(mt.schedule_eta(mt.TOTAL_UPDATES)),
    }
    for key, t in (("reject_t0", 0), ("reject_tTplus1", mt.TOTAL_UPDATES + 1)):
        try:
            mt.schedule_eta(t)
            schedule[key] = False
        except RuntimeError:
            schedule[key] = True
        schedule[key + "_type"] = "RuntimeError"
    schedule["rejection_ok"] = (schedule["reject_t0"]
                                and schedule["reject_tTplus1"])

    cases = {
        "main": run_case(
            "main",
            [1.0, -0.5, 0.25],
            [[0.1, -0.2, 0.05], [-0.3, 0.4, -0.15], [0.2, 0.15, -0.25]],
            [2.0, -1.5, 0.75],
            [[0.05, -0.1, 0.2], [-0.4, 0.3, 0.1], [0.15, -0.2, 0.05]],
        ),
        "tiny": run_case(
            "tiny",
            [1.0, -1.0, 0.5],
            [[1.0e-9, -1.0e-12, 1.0e-9],
             [-2.0e-9, 1.0e-12, -1.0e-9],
             [1.5e-9, -2.0e-12, 0.5e-9]],
            [2.0, -1.5, 0.75],
            [[-1.0e-9, 2.0e-12, -1.0e-9],
             [1.0e-9, -1.0e-12, 2.0e-9],
             [-1.5e-9, 2.0e-12, -0.5e-9]],
        ),
    }

    result = {
        "framework": "pytorch",
        "probe_eta": PROBE_ETA,
        "probe_beta": list(PROBE_BETA),
        "probe_epsilon": PROBE_EPS,
        "probe_weight_decay": PROBE_WD,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "training_path": training_path,
        "schedule": schedule,
        "cases": cases,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    npz = {}
    for case_name, case in cases.items():
        for leaf in ("decay", "nodecay"):
            prefix = f"{case_name}_{leaf}"
            npz[f"{prefix}_x0"] = np.asarray(case[leaf]["x0"], dtype=np.float64)
            npz[f"{prefix}_x_final"] = np.asarray(case[leaf]["x_final"], dtype=np.float64)
            npz[f"{prefix}_m_final"] = np.asarray(case[leaf]["m_final"], dtype=np.float64)
            npz[f"{prefix}_v_final"] = np.asarray(case[leaf]["v_final"], dtype=np.float64)
    arrays_path = out_path.with_name(out_path.stem + "_arrays.npz")
    np.savez(arrays_path, **npz)

    print("PROBE_TORCH training_path "
          f"leaves={training_path['leaf_count']} decay={training_path['decay_leaf_count']} "
          f"nodecay={training_path['nodecay_leaf_count']} rule_meta_ok={training_path['rule_meta_ok']}")
    print("PROBE_TORCH schedule "
          f"t1={schedule['t1']} tW={schedule['tW']} tW1={schedule['tW1']} tT={schedule['tT']} "
          f"rejection_ok={schedule['rejection_ok']}")
    for case_name, case in cases.items():
        for leaf in ("decay", "nodecay"):
            c = case[leaf]
            print(f"PROBE_TORCH case={case_name} leaf={leaf} lambda={c['lambda']} "
                  f"m_nonzero={c['all_nonzero_m_before_measured']} "
                  f"v_nonzero={c['all_nonzero_v_before_measured']} x_final={c['x_final']}")
    print(f"PROBE_TORCH WROTE {out_path} {arrays_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
