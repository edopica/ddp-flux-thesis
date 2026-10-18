#!/usr/bin/env python3
"""Compare the Flux and PyTorch optimizer-contract probes against an oracle.

The comparator loads ``probe_flux.json`` and ``probe_torch.json``, rebuilds
the AdamW trajectory with an independent Float64 arithmetic oracle, checks the
library updates, runs the required negative controls, and checks the schedule
at ``1``, ``W``, ``W+1``, and ``T``.

Probe-specific rule (not the Phase 3 acceptance rule):

    |a - b| <= 1e-7 + 5e-7 * max(|a|, |b|)

Justification: Float32 has ~1.19e-7 relative spacing near |x| = 1.  The
three-update library-vs-oracle calibration leaves at most ~0.17 of this bound
(about 6x margin for library rounding).  A wrong decay applied to the no-decay
leaf moves x by ``eta*lambda*|x| = 1e-6*|x|``; the probe's no-decay leaf has
|x| >= 2, so the control exceeds the bound by at least ~1.8x.  Wrong epsilon
placement changes the tiny-gradient update by ~1e-3 and is rejected by orders
of magnitude.

Exit status is 0 only when every positive check passes and every negative
control is rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ATOL = 1.0e-7
RTOL = 5.0e-7
# Supplementary moment check only.  The library carries beta as Float32, so
# (1 - beta2) differs from the Float64 oracle by ~1.29e-5 relative; the
# observed moment residual stays below that scale.  The primary bound on x
# remains ATOL/RTOL above.
MOMENT_RTOL = 5.0e-5
SCHEDULE_ATOL = 1.0e-15
SCHEDULE_RTOL = 1.0e-9

EXPECTED_BETA = (0.9, 0.999)
EXPECTED_EPS = 1.0e-8
EXPECTED_WD = 1.0e-4
EXPECTED_ETA = 1.0e-2
ETA_MAX = 1.0e-3
W = 975
T = 9750
PARAMETER_COUNT = 11271432
ARRAY_COUNT = 62
DECAY_COUNT = 21
NODECAY_COUNT = 41


class Checker:
    def __init__(self) -> None:
        self.failures = []
        self.checks = 0
        self.negative = 0
        self.rejected = 0
        self.metrics = {}

    def record(self, ok: bool, name: str, detail: str) -> bool:
        self.checks += 1
        status = "PASS" if ok else "FAIL"
        print(f"CASE {name} {status} {detail}")
        if not ok:
            self.failures.append(name)
        return ok

    def negative_control(self, name: str, candidate, reference) -> bool:
        m = compare_arrays(candidate, reference, ATOL, RTOL)
        ok = m["violations"] > 0
        self.negative += 1
        if ok:
            self.rejected += 1
        status = "REJECTED" if ok else "NOT_REJECTED"
        print(
            f"NEGATIVE_CONTROL {name} {status} "
            f"max_abs={m['max_abs']:.6e} bound_max={m['bound_max']:.6e} "
            f"ratio={m['max_ratio']:.3f} violations={m['violations']}"
        )
        if not ok:
            self.failures.append(f"negative:{name}")
        return ok


def compare_arrays(a, b, atol: float, rtol: float) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    bound = atol + rtol * np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bound > 0.0, diff / bound, np.where(diff == 0.0, 0.0, np.inf))
    return {
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "bound_max": float(np.max(bound)) if bound.size else 0.0,
        "max_ratio": float(np.max(ratio)) if ratio.size else 0.0,
        "violations": int(np.sum(diff > bound)),
    }


def oracle(
    x0,
    grads,
    lambda_: float,
    *,
    eta: float = EXPECTED_ETA,
    betas=EXPECTED_BETA,
    eps: float = EXPECTED_EPS,
    bias_correction: bool = True,
    eps_inside_sqrt: bool = False,
) -> dict:
    x = np.asarray(x0, dtype=np.float64).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    x_steps = []
    m_steps = []
    v_steps = []
    for t, g in enumerate(grads, start=1):
        g = np.asarray(g, dtype=np.float64)
        m = betas[0] * m + (1.0 - betas[0]) * g
        v = betas[1] * v + (1.0 - betas[1]) * g * g
        if bias_correction:
            mhat = m / (1.0 - betas[0] ** t)
            vhat = v / (1.0 - betas[1] ** t)
        else:
            mhat = m
            vhat = v
        if eps_inside_sqrt:
            dx = mhat / np.sqrt(vhat + eps)
        else:
            dx = mhat / (np.sqrt(vhat) + eps)
        x = x - eta * (dx + lambda_ * x)
        x_steps.append(x.copy())
        m_steps.append(m.copy())
        v_steps.append(v.copy())
    return {"x_steps": x_steps, "x_final": x, "m_steps": m_steps, "v_steps": v_steps}


def beta_power(beta, completed_updates: int) -> np.ndarray:
    p = np.asarray(beta, dtype=np.float32)
    b = np.asarray(beta, dtype=np.float32)
    for _ in range(completed_updates):
        p = (p * b).astype(np.float32)
    return p


def schedule_eta(t: int) -> float:
    if not 1 <= t <= T:
        raise ValueError(f"schedule index {t} outside 1..{T}")
    if t <= W:
        return ETA_MAX * t / W
    return ETA_MAX * 0.5 * (1.0 + math.cos(math.pi * (t - W) / (T - W)))


def check_schedule(checker: Checker, framework: str, sched: dict) -> None:
    expected = {
        1: schedule_eta(1),
        W: schedule_eta(W),
        W + 1: schedule_eta(W + 1),
        T: schedule_eta(T),
    }
    keyed = {"t1": 1, "tW": W, "tW1": W + 1, "tT": T}
    for key, t in keyed.items():
        value = float(sched[key])
        ref = expected[t]
        diff = abs(value - ref)
        bound = SCHEDULE_ATOL + SCHEDULE_RTOL * max(1.0, abs(ref))
        checker.record(
            diff <= bound,
            f"schedule.{framework}.t{t}",
            f"value={value:.17g} oracle={ref:.17g} absdiff={diff:.3e} bound={bound:.3e}",
        )
    checker.record(
        bool(sched.get("rejection_ok")) and sched.get("reject_t0") and sched.get("reject_tTplus1"),
        f"schedule.{framework}.reject_out_of_range",
        f"t0={sched.get('reject_t0')} tTplus1={sched.get('reject_tTplus1')}",
    )
    # Negative control: warm-up formula used one step too long at W+1.
    correct = expected[W + 1]
    wrong = ETA_MAX * (W + 1) / W
    bound = ATOL + RTOL * max(abs(correct), abs(wrong))
    checker.negative_control(
        f"schedule.{framework}.warmup_at_W_plus_1",
        np.array([correct]),
        np.array([wrong]),
    )
    checker.record(
        abs(correct - wrong) > bound,
        f"schedule.{framework}.off_by_one_delta",
        f"delta={abs(correct - wrong):.3e} bound={bound:.3e}",
    )


def check_training_path(checker: Checker, framework: str, path: dict) -> None:
    checker.record(
        int(path["parameter_count"]) == PARAMETER_COUNT,
        f"training_path.{framework}.parameter_count",
        f"value={path['parameter_count']} expected={PARAMETER_COUNT}",
    )
    checker.record(
        int(path["leaf_count"]) == ARRAY_COUNT,
        f"training_path.{framework}.leaf_count",
        f"value={path['leaf_count']} expected={ARRAY_COUNT}",
    )
    checker.record(
        int(path["decay_leaf_count"]) == DECAY_COUNT
        and int(path["nodecay_leaf_count"]) == NODECAY_COUNT,
        f"training_path.{framework}.group_counts",
        f"decay={path['decay_leaf_count']} nodecay={path['nodecay_leaf_count']} "
        f"expected={DECAY_COUNT}/{NODECAY_COUNT}",
    )
    checker.record(
        [float(v) for v in path["decay_lambda_values"]] == [EXPECTED_WD]
        and [float(v) for v in path["nodecay_lambda_values"]] == [0.0],
        f"training_path.{framework}.lambda_values",
        f"decay={path['decay_lambda_values']} nodecay={path['nodecay_lambda_values']}",
    )
    checker.record(
        bool(path["rule_meta_ok"]),
        f"training_path.{framework}.rule_meta",
        f"eta_max={path['eta_max']} beta={path['beta']} eps={path['epsilon']} wd={path['weight_decay']}",
    )
    checker.record(
        [float(b) for b in path["beta"]] == list(EXPECTED_BETA)
        and float(path["epsilon"]) == EXPECTED_EPS
        and float(path["weight_decay"]) == EXPECTED_WD,
        f"training_path.{framework}.hyperparameters",
        f"beta={path['beta']} eps={path['epsilon']} wd={path['weight_decay']}",
    )


def check_leaf(
    checker: Checker,
    framework: str,
    case_name: str,
    leaf: str,
    record: dict,
    *,
    cross: dict | None = None,
) -> None:
    lambda_ = float(record["lambda"])
    ref = oracle(record["x0"], record["grads"], lambda_)
    name = f"{case_name}.{leaf}"

    m = compare_arrays(record["x_final"], ref["x_final"], ATOL, RTOL)
    checker.record(
        m["violations"] == 0,
        f"optimizer.{framework}.{name}.x_final",
        f"max_abs={m['max_abs']:.6e} bound_max={m['bound_max']:.6e} "
        f"ratio={m['max_ratio']:.3f} violations={m['violations']}",
    )
    checker.metrics[f"{framework}.{name}.x_final"] = m

    m_steps = compare_arrays(record["x_steps"], ref["x_steps"], ATOL, RTOL)
    checker.record(
        m_steps["violations"] == 0,
        f"optimizer.{framework}.{name}.x_steps",
        f"max_abs={m_steps['max_abs']:.6e} ratio={m_steps['max_ratio']:.3f} "
        f"violations={m_steps['violations']}",
    )

    n = len(record["grads"])
    m_before = compare_arrays(
        record["m_before_measured"], ref["m_steps"][n - 2], 0.0, MOMENT_RTOL)
    v_before = compare_arrays(
        record["v_before_measured"], ref["v_steps"][n - 2], 0.0, MOMENT_RTOL)
    checker.record(
        m_before["violations"] == 0 and v_before["violations"] == 0,
        f"optimizer.{framework}.{name}.moments_before_measured",
        f"m_ratio={m_before['max_ratio']:.3f} v_ratio={v_before['max_ratio']:.3f} "
        f"m_viol={m_before['violations']} v_viol={v_before['violations']}",
    )
    checker.record(
        bool(record["all_nonzero_m_before_measured"])
        and bool(record["all_nonzero_v_before_measured"]),
        f"optimizer.{framework}.{name}.nonzero_moments",
        f"m_nonzero={record['all_nonzero_m_before_measured']} "
        f"v_nonzero={record['all_nonzero_v_before_measured']}",
    )

    if framework == "flux":
        bp_before = np.asarray(record["beta_power_before_measured"], dtype=np.float64)
        bp_final = np.asarray(record["beta_power_final"], dtype=np.float64)
        bp_ref_before = beta_power(EXPECTED_BETA, n - 1)
        bp_ref_final = beta_power(EXPECTED_BETA, n)
        ok = (np.array_equal(bp_before.astype(np.float32), bp_ref_before)
              and np.array_equal(bp_final.astype(np.float32), bp_ref_final))
        checker.record(
            ok,
            f"optimizer.flux.{name}.beta_power",
            f"before={bp_before.tolist()} expected={bp_ref_before.tolist()} "
            f"final={bp_final.tolist()} expected_final={bp_ref_final.tolist()}",
        )

    if cross is not None:
        mc = compare_arrays(record["x_final"], cross["x_final"], ATOL, RTOL)
        checker.record(
            mc["violations"] == 0,
            f"cross.{name}.flux_vs_pytorch.x_final",
            f"max_abs={mc['max_abs']:.6e} ratio={mc['max_ratio']:.3f} "
            f"violations={mc['violations']}",
        )
        checker.metrics[f"cross.{name}.flux_vs_pytorch.x_final"] = mc


def check_negative_controls(
    checker: Checker,
    framework: str,
    record: dict,
    *,
    wrong_decay: bool,
    wrong_epsilon: bool,
) -> None:
    name = record["name"]
    x0 = record["x0"]
    grads = record["grads"]
    lambda_ = float(record["lambda"])
    x_final = record["x_final"]

    checker.negative_control(
        f"optimizer.{framework}.{name}.omitted_update",
        np.asarray(x_final),
        np.asarray(x0),
    )
    if wrong_decay:
        checker.negative_control(
            f"optimizer.{framework}.{name}.wrong_decay_scale_1e-3",
            np.asarray(x_final),
            oracle(x0, grads, 1.0e-3)["x_final"],
        )
        # The correct decay term must itself be visible at the probe's lr.
        checker.negative_control(
            f"optimizer.{framework}.{name}.omitted_decay",
            np.asarray(x_final),
            oracle(x0, grads, 0.0)["x_final"],
        )
    if wrong_epsilon:
        checker.negative_control(
            f"optimizer.{framework}.{name}.wrong_epsilon_placement",
            np.asarray(x_final),
            oracle(x0, grads, lambda_, eps_inside_sqrt=True)["x_final"],
        )
    checker.negative_control(
        f"optimizer.{framework}.{name}.wrong_bias_correction",
        np.asarray(x_final),
        oracle(x0, grads, lambda_, bias_correction=False)["x_final"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux", required=True)
    parser.add_argument("--torch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.flux, encoding="utf-8") as f:
        flux = json.load(f)
    with open(args.torch, encoding="utf-8") as f:
        torch_result = json.load(f)

    checker = Checker()
    probes = {"flux": flux, "pytorch": torch_result}
    for framework, probe in probes.items():
        checker.record(
            probe.get("framework") == framework,
            f"probe.{framework}.identity",
            f"framework={probe.get('framework')}",
        )
        checker.record(
            abs(float(probe["probe_eta"]) - EXPECTED_ETA) == 0.0
            and [float(b) for b in probe["probe_beta"]] == list(EXPECTED_BETA)
            and float(probe["probe_epsilon"]) == EXPECTED_EPS
            and float(probe["probe_weight_decay"]) == EXPECTED_WD,
            f"probe.{framework}.sensitivity_settings",
            f"eta={probe['probe_eta']} beta={probe['probe_beta']} "
            f"eps={probe['probe_epsilon']} wd={probe['probe_weight_decay']}",
        )
        check_training_path(checker, framework, probe["training_path"])
        check_schedule(checker, framework, probe["schedule"])

    checker.record(
        bool(flux.get("legacy_probe_included")),
        "legacy_probe.flux.included",
        "existing constructor and beta-power @assert checks ran in-process",
    )

    for case_name in ("main", "tiny"):
        flux_case = flux["cases"][case_name]
        torch_case = torch_result["cases"][case_name]
        for leaf in ("decay", "nodecay"):
            check_leaf(
                checker, "flux", case_name, leaf, flux_case[leaf],
                cross=torch_case[leaf])
            check_leaf(checker, "pytorch", case_name, leaf, torch_case[leaf])

    # Required negative controls: omitted update, wrong decay scale, decay on
    # the no-decay leaf, wrong epsilon placement, and wrong bias correction.
    for framework, probe in probes.items():
        main = probe["cases"]["main"]
        tiny = probe["cases"]["tiny"]
        check_negative_controls(
            checker, framework, main["decay"],
            wrong_decay=True, wrong_epsilon=False)
        check_negative_controls(
            checker, framework, main["nodecay"],
            wrong_decay=False, wrong_epsilon=False)
        check_negative_controls(
            checker, framework, tiny["decay"],
            wrong_decay=True, wrong_epsilon=True)
        check_negative_controls(
            checker, framework, tiny["nodecay"],
            wrong_decay=False, wrong_epsilon=True)
        # Explicit decay-on-no-decay control: compare the measured no-decay
        # update with an oracle that incorrectly applies lambda to it.
        for case in (main, tiny):
            checker.negative_control(
                f"optimizer.{framework}.{case['nodecay']['name']}.decay_on_nodecay",
                np.asarray(case["nodecay"]["x_final"]),
                oracle(
                    case["nodecay"]["x0"],
                    case["nodecay"]["grads"],
                    EXPECTED_WD,
                )["x_final"],
            )

    # The schedule off-by-one control runs once per framework inside
    # check_schedule.

    result = {
        "pass": not checker.failures,
        "tolerance": {
            "atol": ATOL,
            "rtol": RTOL,
            "moment_rtol": MOMENT_RTOL,
            "schedule_atol": SCHEDULE_ATOL,
            "schedule_rtol": SCHEDULE_RTOL,
            "note": "probe-specific; not the Phase 3 acceptance rule",
        },
        "checks": checker.checks,
        "negative_controls": checker.negative,
        "rejected": checker.rejected,
        "failures": checker.failures,
        "metrics": checker.metrics,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"SUMMARY checks={checker.checks} negative_controls={checker.negative} "
        f"rejected={checker.rejected} failures={len(checker.failures)}"
    )
    if checker.failures:
        for name in checker.failures:
            print(f"FAILED {name}")
        print("PHASE3_OPTIMIZER_PROBE FAIL")
        return 1
    print("PHASE3_OPTIMIZER_PROBE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
