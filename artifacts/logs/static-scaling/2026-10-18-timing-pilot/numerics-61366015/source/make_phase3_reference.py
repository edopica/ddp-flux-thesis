"""
Phase 3 static-scaling PyTorch reference generator.

Produces, in --out-dir:
  phase3_reference.npz
  phase3_reference_manifest.json

The reference is the matched single-GPU PyTorch training track for the frozen
Phase 2 fixtures:
  * model: frozen small-image ResNet-18 with GroupNorm-32, 11,271,432
    parameters in 62 arrays loaded explicitly from initial_bundle.npz;
  * optimizer: torch.optim.AdamW with two parameter groups (21 decay with
    weight_decay=1e-4, 41 no-decay with weight_decay=0.0), betas=(0.9, 0.999),
    eps=1e-8, amsgrad=False;
  * schedule: warm-up eta_t = eta_max * t / W with W=975, eta_max=1e-3;
    Phase 3 uses t=1,2,3;
  * updates: u1 = train_transformed_ref[0], u2 = [1], u3 = [0]; labels are the
    shared int64 0-based train_labels; loss is mean cross-entropy on raw
    logits;
  * arrays: inputs, labels, eta, loss, logits, the 11 u1 activations, all 62
    u1 and u3 gradients, all 62 u3 parameters/moments, and step.u3.

Every array is float32 in canonical PyTorch layout (NCHW images, conv
(out,in,kh,kw), linear (out,in), C-order) unless the contract names a scalar
float64/int64. Array sha256 = sha256 of np.ascontiguousarray(a).tobytes("C").

The generator self-checks and prints ``CHECK <name> PASS|FAIL <detail>`` lines,
then ``PHASE3_REFERENCE PASS|FAIL ...``. Exit code 0 only on PASS. Artifacts are
not written when a pre-write check fails.

Reproduction (Leonardo compute node, pinned venv):
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/make_phase3_reference.py \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --out-dir artifacts/baselines/static-scaling/2026-10-17-phase3 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

try:
    import dataset_adapter as da
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dataset_adapter as da

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = REPO_ROOT / "artifacts" / "baselines" / "static-scaling" / "2026-10-16-phase2"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "baselines" / "static-scaling" / "2026-10-17-phase3"
REFERENCE_NAME = "phase3_reference.npz"
MANIFEST_NAME = "phase3_reference_manifest.json"
SCHEMA = (
    "phase3-reference-v1; arrays entries use raw npz key names; dtype/shape; "
    "sha256 = sha256 of np.ascontiguousarray(a).tobytes(order='C'); "
    "canonical PyTorch layouts (NCHW images, conv (out,in,kh,kw), linear (out,in))"
)
UPDATE_TAGS = ("u1", "u2", "u3")
UPDATE_EPOCHS = {"u1": 0, "u2": 1, "u3": 0}
EXPECTED_FIXTURE_SHAPE = (2, 12, 3, 64, 64)
EXPECTED_LABEL_SHAPE = (12,)
ACTIVATION_SHAPES = {
    "stem.gn": (12, 64, 64, 64),
    "stem": (12, 64, 64, 64),
    "stage1.block0": (12, 64, 64, 64),
    "stage1.block1": (12, 64, 64, 64),
    "stage2.block0": (12, 128, 32, 32),
    "stage2.block1": (12, 128, 32, 32),
    "stage3.block0": (12, 256, 16, 16),
    "stage3.block1": (12, 256, 16, 16),
    "stage4.block0": (12, 512, 8, 8),
    "stage4.block1": (12, 512, 8, 8),
    "pooled": (12, 512),
}


class CheckLog:
    def __init__(self) -> None:
        self.failed = []
        self.total = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.total += 1
        print("CHECK %s %s%s" % (name, "PASS" if ok else "FAIL",
                                 (" " + detail) if detail else ""), flush=True)
        if not ok:
            self.failed.append(name)

    @property
    def ok(self) -> bool:
        return not self.failed


def _to_numpy(tensor) -> np.ndarray:
    arr = np.ascontiguousarray(tensor.detach().to(device="cpu").numpy())
    if arr.dtype != np.float32:
        raise RuntimeError("expected float32 tensor, got %s" % (arr.dtype,))
    return arr


def _scalar_int(value) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _provenance(commit_arg, state_arg, commit_env: str, state_env: str):
    commit = commit_arg or os.environ.get(commit_env) or "unknown"
    state = state_arg or os.environ.get(state_env) or "unknown"
    return commit, state


def _file_entry(path: str, name: str = None) -> dict:
    return {
        "path": name or os.path.basename(path),
        "bytes": int(os.path.getsize(path)),
        "sha256": da.sha256_file(path),
    }


def _layout_for(name: str, spec_layouts: dict) -> str:
    if name.startswith("input."):
        return "nchw_float32_normalized"
    if name.startswith("label."):
        return "int64_vector"
    if name.startswith("eta."):
        return "scalar_float64"
    if name.startswith("loss."):
        return "scalar_float32"
    if name.startswith("step."):
        return "scalar_int64"
    if name.startswith("logits."):
        return "logits_batch_classes"
    if name.startswith("act."):
        return "features_batch_channels" if name == "act.pooled" else "nchw_float32"
    for prefix in ("grad.", "param.", "m.", "v."):
        if name.startswith(prefix):
            param = name.split(".", 2)[2]
            if param not in spec_layouts:
                raise RuntimeError("unknown parameter name in array %r" % (name,))
            return spec_layouts[param]
    raise RuntimeError("no layout rule for array %r" % (name,))


def _load_data_fixture(path: str):
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("train_transformed_ref", "train_labels") if name not in archive.files]
        if missing:
            raise RuntimeError("data fixture missing arrays: %s" % (missing,))
        refs = np.ascontiguousarray(archive["train_transformed_ref"])
        labels = np.ascontiguousarray(archive["train_labels"])
    return refs, labels


def _optimizer_steps(optimizer, model) -> dict:
    steps = {}
    for name, param in model.named_parameters():
        state = optimizer.state.get(param)
        if state is None or "step" not in state:
            continue
        steps[name] = _scalar_int(state["step"])
    return steps


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate the Phase 3 PyTorch reference for the static-scaling fixtures")
    parser.add_argument("--fixture-dir", default=str(DEFAULT_FIXTURE_DIR),
                        help="directory containing data_fixture.npz and initial_bundle.npz")
    parser.add_argument("--bundle", default=None,
                        help="initial parameter bundle (default: <fixture-dir>/initial_bundle.npz)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="output directory for phase3_reference.npz and its manifest")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--updates", type=int, default=3,
                        help="number of optimizer updates (Phase 3 requires exactly 3)")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--source-state-sha256", default=None)
    parser.add_argument("--flux-commit", default=None)
    parser.add_argument("--flux-state-sha256", default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing artifacts")
    return parser.parse_args(argv)


def _run(args, log: CheckLog) -> int:
    if int(args.updates) != 3:
        raise RuntimeError("Phase 3 reference requires --updates 3 (got %d); the artifact "
                           "names u1/u2/u3 and step.u3=3 are frozen" % int(args.updates))

    fixture_dir = os.path.abspath(args.fixture_dir)
    bundle_path = os.path.abspath(args.bundle or os.path.join(fixture_dir, "initial_bundle.npz"))
    data_path = os.path.join(fixture_dir, "data_fixture.npz")
    fixture_manifest_path = os.path.join(fixture_dir, "fixture_manifest.json")
    out_dir = os.path.abspath(args.out_dir)
    npz_path = os.path.join(out_dir, REFERENCE_NAME)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)

    if not args.force:
        for path in (npz_path, manifest_path):
            if os.path.exists(path):
                raise RuntimeError("refusing to overwrite %s (use --force)" % path)
    for path in (data_path, bundle_path, fixture_manifest_path):
        if not os.path.exists(path):
            raise RuntimeError("required fixture file not found: %s" % path)

    import torch
    try:
        import model_torch as mt
    except ModuleNotFoundError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import model_torch as mt

    device = mt.resolve_device(args.device)
    flags = mt.configure_backend(num_threads=1)
    tf32_disabled = (
        flags["cuda_matmul_allow_tf32"] is False
        and flags["cudnn_allow_tf32"] is False
        and flags["float32_matmul_precision"] == "highest"
    )
    log.check("backend.tf32_disabled", tf32_disabled,
              "cuda_matmul_allow_tf32=%r cudnn_allow_tf32=%r float32_matmul_precision=%r"
              % (flags["cuda_matmul_allow_tf32"], flags["cudnn_allow_tf32"],
                 flags["float32_matmul_precision"]))
    log.check("backend.cudnn_deterministic",
              flags["cudnn_deterministic"] is True and flags["cudnn_benchmark"] is False,
              "deterministic=%r benchmark=%r" % (flags["cudnn_deterministic"], flags["cudnn_benchmark"]))
    if device.type == "cuda":
        log.check("backend.cuda_available", torch.cuda.is_available() and torch.cuda.device_count() >= 1,
                  "device=%s name=%r capability=%s"
                  % (device, torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)))
    else:
        log.check("backend.cuda_available", True, "device=cpu")

    model = mt.make_model()
    named = list(model.named_parameters())
    parameter_count = sum(int(p.numel()) for _, p in named)
    log.check("model.parameter_count", parameter_count == mt.PARAMETER_COUNT,
              "count=%d want=%d" % (parameter_count, mt.PARAMETER_COUNT))
    log.check("model.array_count", len(named) == mt.ARRAY_COUNT,
              "count=%d want=%d" % (len(named), mt.ARRAY_COUNT))
    log.check("model.parameter_names", set(name for name, _ in named) == set(mt.BUNDLE_NAMES),
              "names=%d bundle_names=%d" % (len(set(name for name, _ in named)), len(mt.BUNDLE_NAMES)))
    model = model.to(device)
    model.train()

    info = mt.load_bundle(model, bundle_path)
    log.check("bundle.names_shapes",
              info["array_count"] == mt.ARRAY_COUNT
              and info["element_count"] == mt.PARAMETER_COUNT
              and set(info["names"]) == set(mt.BUNDLE_NAMES),
              "arrays=%d elements=%d" % (info["array_count"], info["element_count"]))
    log.check("bundle.load_exact", not info["mismatched"] and info["max_abs_diff"] == 0.0,
              "mismatched=%d max_abs_diff=%r" % (len(info["mismatched"]), info["max_abs_diff"]))
    log.check("model.device", all(p.device == device for p in model.parameters()), "device=%s" % device)

    decay, no_decay = mt.parameter_groups(model)
    decay_names = [name for name, _ in decay]
    no_decay_names = [name for name, _ in no_decay]
    log.check("optimizer.param_groups", len(decay) == 21 and len(no_decay) == 41,
              "decay=%d no_decay=%d" % (len(decay), len(no_decay)))
    log.check("optimizer.group_membership",
              len([n for n in decay_names if n.endswith(".weight") and n != "classifier.weight"]) == 20
              and "classifier.weight" in decay_names
              and len([n for n in no_decay_names if n.endswith((".gn.weight", ".gn1.weight",
                                                                ".gn2.weight", ".proj_gn.weight"))]) == 20
              and len([n for n in no_decay_names if n.endswith((".gn.bias", ".gn1.bias",
                                                                ".gn2.bias", ".proj_gn.bias"))]) == 20
              and "classifier.bias" in no_decay_names,
              "decay=20 conv weights + classifier.weight; no_decay=20 GN weight + 20 GN bias + classifier.bias")

    optimizer = mt.build_optimizer(model)
    group0, group1 = optimizer.param_groups
    log.check("optimizer.adamw_config",
              group0["weight_decay"] == 1.0e-4 and group1["weight_decay"] == 0.0
              and group0["betas"] == (0.9, 0.999) and group1["betas"] == (0.9, 0.999)
              and group0["eps"] == 1.0e-8 and group1["eps"] == 1.0e-8
              and group0["amsgrad"] is False and group1["amsgrad"] is False,
              "betas=(0.9, 0.999) eps=1e-08 weight_decay=(1e-04, 0.0) amsgrad=False")
    log.check("optimizer.initial_zero_state",
              len(optimizer.state) == 0 and all(p.grad is None for p in model.parameters()),
              "state_entries=%d grads_none=%s"
              % (len(optimizer.state), all(p.grad is None for p in model.parameters())))

    etas = {tag: mt.schedule_eta(t) for t, tag in zip((1, 2, 3), UPDATE_TAGS)}
    expected_etas = {tag: float(mt.ETA_MAX) * t / mt.WARMUP_UPDATES
                     for t, tag in zip((1, 2, 3), UPDATE_TAGS)}
    log.check("schedule.warmup_eta", all(etas[tag] == expected_etas[tag] for tag in UPDATE_TAGS),
              "W=%d T=%d eta.u1=%r eta.u2=%r eta.u3=%r"
              % (mt.WARMUP_UPDATES, mt.TOTAL_UPDATES, etas["u1"], etas["u2"], etas["u3"]))

    with open(fixture_manifest_path, encoding="utf-8") as handle:
        fixture_manifest = json.load(handle)
    log.check("fixture.non_synthetic", fixture_manifest.get("synthetic") is False,
              "synthetic=%r" % (fixture_manifest.get("synthetic"),))
    if fixture_manifest.get("synthetic") is not False:
        raise RuntimeError("refusing to build a reference from synthetic fixtures")
    recorded = {entry.get("path"): entry.get("sha256")
                for entry in fixture_manifest.get("files", [])}
    data_sha = da.sha256_file(data_path)
    log.check("fixture.manifest_sha256", recorded.get("data_fixture.npz") == data_sha,
              "data_fixture.npz=%s recorded=%s" % (data_sha, recorded.get("data_fixture.npz")))
    if recorded.get("data_fixture.npz") != data_sha:
        raise RuntimeError("data_fixture.npz does not match fixture_manifest.json")

    refs, labels = _load_data_fixture(data_path)
    log.check("fixture.data_contract",
              refs.dtype == np.float32 and refs.shape == EXPECTED_FIXTURE_SHAPE
              and labels.dtype == np.int64 and labels.shape == EXPECTED_LABEL_SHAPE,
              "refs=%s/%s labels=%s/%s"
              % (refs.dtype, tuple(refs.shape), labels.dtype, tuple(labels.shape)))
    log.check("fixture.labels_range", int(labels.min()) >= 0 and int(labels.max()) < 200,
              "min=%d max=%d" % (int(labels.min()), int(labels.max())))
    log.check("fixture.update_protocol",
              [UPDATE_EPOCHS[tag] for tag in UPDATE_TAGS] == [0, 1, 0],
              "u1=train_transformed_ref[0] u2=train_transformed_ref[1] u3=train_transformed_ref[0]")

    spec_layouts = {name: layout for name, _shape, layout, _kind in da.bundle_specs()}
    arrays = {}
    step_numbers = {}
    last_steps = {}
    for t, tag in enumerate(UPDATE_TAGS, start=1):
        epoch = UPDATE_EPOCHS[tag]
        x_np = np.ascontiguousarray(refs[epoch])
        y_np = np.ascontiguousarray(labels)
        capture_activations = tag == "u1"
        capture_grads = tag in ("u1", "u3")
        result = mt.train_step(
            model, optimizer, x_np, y_np, t,
            capture_activations=capture_activations,
            capture_grads=capture_grads,
        )
        arrays["input." + tag] = x_np
        arrays["label." + tag] = y_np
        arrays["eta." + tag] = np.asarray(result["eta"], dtype=np.float64)
        arrays["loss." + tag] = np.asarray(result["loss"].item(), dtype=np.float32)
        arrays["logits." + tag] = _to_numpy(result["logits"])
        if capture_activations:
            log.check("activations.%s.keys" % tag,
                      set(result["activations"]) == set(mt.ACTIVATION_NAMES),
                      "keys=%d names=%d" % (len(result["activations"]), len(mt.ACTIVATION_NAMES)))
            wrong_shape = [key for key in mt.ACTIVATION_NAMES
                           if tuple(result["activations"][key].shape) != ACTIVATION_SHAPES[key]]
            log.check("activations.%s.shapes" % tag, not wrong_shape,
                      "mismatches=%s" % (wrong_shape[:3] or "none"))
            for key in mt.ACTIVATION_NAMES:
                arrays["act." + key] = _to_numpy(result["activations"][key])
        if capture_grads:
            grads = result["grads"]
            log.check("grads.%s.all_present" % tag,
                      set(grads) == set(mt.BUNDLE_NAMES),
                      "arrays=%d want=%d" % (len(grads), len(mt.BUNDLE_NAMES)))
            for name in mt.BUNDLE_NAMES:
                arrays["grad.%s.%s" % (tag, name)] = _to_numpy(grads[name])
        steps = _optimizer_steps(optimizer, model)
        last_steps = steps
        ok = len(steps) == mt.ARRAY_COUNT and all(value == t for value in steps.values())
        log.check("optimizer.step.%s" % tag, ok,
                  "state_step=%s params=%d" % (sorted(set(steps.values())), len(steps)))
        step_numbers[tag] = t
        if not ok:
            raise RuntimeError("optimizer step counter mismatch after %s" % tag)

    final_step_values = sorted(set(last_steps.values()))
    log.check("optimizer.final_step",
              len(last_steps) == mt.ARRAY_COUNT and final_step_values == [3],
              "state_step=%s params=%d" % (final_step_values, len(last_steps)))
    final_step = int(final_step_values[0])
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name in mt.BUNDLE_NAMES:
            param = named_params[name]
            state = optimizer.state.get(param)
            if state is None or "exp_avg" not in state or "exp_avg_sq" not in state:
                raise RuntimeError("missing optimizer moments for %s" % name)
            arrays["param.u3." + name] = _to_numpy(param)
            arrays["m.u3." + name] = _to_numpy(state["exp_avg"])
            arrays["v.u3." + name] = _to_numpy(state["exp_avg_sq"])
    arrays["step.u3"] = np.asarray(np.int64(final_step), dtype=np.int64)
    log.check("inputs.u1_equals_u3",
              np.array_equal(arrays["input.u1"], arrays["input.u3"])
              and np.array_equal(arrays["label.u1"], arrays["label.u3"]),
              "u1/u3 share train_transformed_ref[0] and train_labels")
    log.check("state.u3_arrays",
              len(arrays) == 15 + len(mt.ACTIVATION_NAMES) + 62 + 62 + 3 * 62 + 1,
              "arrays=%d" % len(arrays))

    ordered = {}
    for tag in UPDATE_TAGS:
        for prefix in ("input", "label", "eta", "loss", "logits"):
            ordered["%s.%s" % (prefix, tag)] = arrays["%s.%s" % (prefix, tag)]
    for key in mt.ACTIVATION_NAMES:
        ordered["act." + key] = arrays["act." + key]
    for tag in ("u1", "u3"):
        for name in mt.BUNDLE_NAMES:
            ordered["grad.%s.%s" % (tag, name)] = arrays["grad.%s.%s" % (tag, name)]
    for prefix in ("param", "m", "v"):
        for name in mt.BUNDLE_NAMES:
            ordered["%s.u3.%s" % (prefix, name)] = arrays["%s.u3.%s" % (prefix, name)]
    ordered["step.u3"] = arrays["step.u3"]
    _require(len(ordered) == len(arrays), "array reordering dropped or duplicated entries")
    arrays = ordered

    arrays_meta = []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arrays[name] = arr
        arrays_meta.append({
            "name": name,
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "layout": _layout_for(name, spec_layouts),
            "sha256": da.sha256_c_order(arr),
        })

    if not log.ok:
        return 1

    os.makedirs(out_dir, exist_ok=True)
    np.savez(npz_path, **arrays)
    with np.load(npz_path, allow_pickle=False) as archive:
        stored = set(archive.files)
        bad = []
        if stored != set(arrays):
            bad.append("name_set")
        for meta in arrays_meta:
            if meta["name"] not in archive.files:
                bad.append(meta["name"] + ":missing")
                continue
            arr = archive[meta["name"]]
            if (str(arr.dtype) != meta["dtype"]
                    or [int(x) for x in arr.shape] != meta["shape"]
                    or da.sha256_c_order(arr) != meta["sha256"]):
                bad.append(meta["name"] + ":metadata")
        if bad:
            raise RuntimeError("reference npz failed re-read verification: %s" % (bad[:5],))
    log.check("reference.arrays", True,
              "arrays=%d dtype_shape_sha256=true" % len(arrays_meta))

    source_commit, source_state = _provenance(
        args.source_commit, args.source_state_sha256, "SOURCE_COMMIT", "SOURCE_STATE_SHA256")
    flux_commit, flux_state = _provenance(
        args.flux_commit, args.flux_state_sha256, "FLUX_COMMIT", "FLUX_STATE_SHA256")

    fixture_files = []
    fixture_sha256 = {}
    for path in (data_path, bundle_path, fixture_manifest_path):
        entry = _file_entry(path)
        fixture_files.append(entry)
        fixture_sha256[entry["path"]] = entry["sha256"]
    bundle_file_sha256 = da.sha256_file(bundle_path)
    with open(fixture_manifest_path, encoding="utf-8") as handle:
        fixture_manifest = json.load(handle)
    for entry in fixture_manifest.get("files", []):
        fixture_sha256.setdefault(entry.get("path"), entry.get("sha256"))

    runtime = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "cuda_capability": (list(torch.cuda.get_device_capability(device))
                            if device.type == "cuda" else None),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__,
        "num_threads": int(torch.get_num_threads()),
    }
    schedule = {
        "eta_max": float(mt.ETA_MAX),
        "betas": [float(x) for x in mt.BETAS],
        "eps": float(mt.EPS),
        "warmup_updates": int(mt.WARMUP_UPDATES),
        "total_updates": int(mt.TOTAL_UPDATES),
        "rule": "eta_t = eta_max * t / W for t <= W; cosine from eta_max to 0 for W < t <= T",
        "eta": {tag: float(etas[tag]) for tag in UPDATE_TAGS},
        "updates": list(range(1, 4)),
    }
    optimizer_meta = {
        "name": "torch.optim.AdamW",
        "lr": float(mt.ETA_MAX),
        "betas": [float(x) for x in mt.BETAS],
        "eps": float(mt.EPS),
        "weight_decay": float(mt.WEIGHT_DECAY),
        "amsgrad": False,
        "maximize": False,
        "bias_correction": True,
        "gradient_clipping": None,
        "group_count": 2,
        "decay_group": {
            "weight_decay": float(mt.WEIGHT_DECAY),
            "parameter_count": len(decay_names),
            "parameters": decay_names,
        },
        "no_decay_group": {
            "weight_decay": 0.0,
            "parameter_count": len(no_decay_names),
            "parameters": no_decay_names,
        },
    }
    model_meta = {
        "name": "Tiny ImageNet ResNet-18, small-image stem, GroupNorm-32",
        "parameter_count": int(parameter_count),
        "parameter_array_count": len(named),
        "parameter_names": list(mt.BUNDLE_NAMES),
        "input_layout": "NCHW",
        "activation_names": list(mt.ACTIVATION_NAMES),
    }
    training_meta = {
        "updates": 3,
        "batch_size": int(labels.shape[0]),
        "tags": list(UPDATE_TAGS),
        "inputs": {tag: "train_transformed_ref[%d]" % UPDATE_EPOCHS[tag] for tag in UPDATE_TAGS},
        "labels": "train_labels (int64, 0-based)",
        "loss": "mean cross-entropy on raw logits (torch.nn.functional.cross_entropy)",
        "optimizer_step_numbers": {tag: int(step_numbers[tag]) for tag in UPDATE_TAGS},
        "final_step": int(final_step),
        "captured_activations": "u1",
        "captured_gradients": ["u1", "u3"],
        "captured_optimizer_state": "u3",
    }

    manifest = {
        "producer": "scripts/static_scaling/make_phase3_reference.py",
        "created": da.utc_now(),
        "synthetic": False,
        "schema": SCHEMA,
        "source_commit": source_commit,
        "source_state_sha256": source_state,
        "flux_commit": flux_commit,
        "flux_state_sha256": flux_state,
        "command": list(sys.argv),
        "fixture_dir": fixture_dir,
        "bundle_path": bundle_path,
        "fixture_files": fixture_files,
        "fixture_sha256": fixture_sha256,
        "data_fixture_sha256": fixture_sha256.get("data_fixture.npz"),
        "fixture_manifest_sha256": fixture_sha256.get("fixture_manifest.json"),
        "initial_bundle_sha256": bundle_file_sha256,
        "bundle_sha256": bundle_file_sha256,
        "bundle_file_sha256": bundle_file_sha256,
        "bundle_array_hash": info["bundle_hash"],
        "bundle_hash": info["bundle_hash"],
        "bundle_hash_definition": "sha256 over concat of per-array raw C-order bytes in manifest order",
        "bundle_array_hash_definition": "sha256 over concat of per-array raw C-order bytes in bundle order",
        "model_parameter_count": int(parameter_count),
        "model_array_count": len(named),
        "optimizer_step_numbers": {tag: int(step_numbers[tag]) for tag in UPDATE_TAGS},
        "warmup_updates": int(mt.WARMUP_UPDATES),
        "total_updates": int(mt.TOTAL_UPDATES),
        "eta_max": float(mt.ETA_MAX),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "runtime": runtime,
        "tf32": flags,
        "optimizer": optimizer_meta,
        "schedule": schedule,
        "model": model_meta,
        "training": training_meta,
        "reference_file": {
            "path": REFERENCE_NAME,
            "bytes": int(os.path.getsize(npz_path)),
            "sha256": da.sha256_file(npz_path),
            "array_count": len(arrays_meta),
        },
        "arrays": arrays_meta,
    }
    da.write_json(manifest_path, manifest)
    with open(manifest_path, encoding="utf-8") as handle:
        written_manifest = json.load(handle)
    log.check("reference.manifest",
              len(written_manifest.get("arrays", [])) == len(arrays_meta)
              and written_manifest.get("reference_file", {}).get("sha256") == manifest["reference_file"]["sha256"],
              "path=%s arrays=%d" % (manifest_path, len(arrays_meta)))
    log.check("reference.write", True, "npz=%s manifest=%s" % (npz_path, manifest_path))
    return 0 if log.ok else 1


def main(argv=None) -> int:
    args = _parse_args(argv)
    log = CheckLog()
    exit_code = 1
    try:
        exit_code = _run(args, log)
    except Exception as exc:
        traceback.print_exc()
        log.check("generator.exception", False, "%s: %s" % (type(exc).__name__, exc))
        exit_code = 1
    ok = log.ok and exit_code == 0
    if ok:
        print("PHASE3_REFERENCE PASS checks=%d failed=0" % log.total, flush=True)
        return 0
    print("PHASE3_REFERENCE FAIL checks=%d checks_failed=%d failed=%s"
          % (log.total, len(log.failed), ",".join(log.failed)), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
