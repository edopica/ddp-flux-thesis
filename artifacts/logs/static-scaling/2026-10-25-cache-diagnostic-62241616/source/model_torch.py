"""
Phase 3 static-scaling PyTorch model and single-GPU training primitives.

Implements the frozen OD-1 architecture from experiments/static-scaling/
specification.md sections 4-6:
  * small-image ResNet-18: 3x3 stride-1 stem, four stages with two BasicBlocks
    each, widths 64/128/256/512, GroupNorm(32, eps=1e-5) everywhere,
    1x1 stride-2 projections (with their own GroupNorm) at stage 2/3/4 block 0,
    global spatial mean, biased 200-way classifier;
  * 11,271,432 trainable elements in exactly 62 arrays whose names equal the
    frozen bundle names (for example ``stage2.block0.proj_gn.weight``);
  * explicit loading of every array from ``initial_bundle.npz`` with dtype,
    shape, name-set, and exact-equality verification;
  * two AdamW parameter groups (21 decay, 41 no-decay) and the frozen warm-up
    schedule.

Canonical layout is PyTorch-native: images NCHW, conv kernels
(out, in, kh, kw), linear weights (out, in), C-order; GroupNorm arrays are
(C,). The module is torch-only; importing it requires torch.

Reused by ``make_phase3_reference.py`` and later phases. No file writes and no
default-initialized training happens here.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dataset_adapter as da
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dataset_adapter as da

GROUP_COUNT = 32
GN_EPS = 1.0e-5
NUM_CLASSES = 200
CLASSIFIER_IN = 512

ETA_MAX = 1.0e-3
BETAS = (0.9, 0.999)
EPS = 1.0e-8
WEIGHT_DECAY = 1.0e-4
WARMUP_UPDATES = 975
TOTAL_UPDATES = 9750

PARAMETER_COUNT = 11271432
ARRAY_COUNT = 62

BUNDLE_SPECS = da.bundle_specs()
BUNDLE_NAMES = tuple(spec[0] for spec in BUNDLE_SPECS)
BUNDLE_SHAPES = {spec[0]: tuple(int(x) for x in spec[1]) for spec in BUNDLE_SPECS}
BUNDLE_LAYOUTS = {spec[0]: spec[2] for spec in BUNDLE_SPECS}
BUNDLE_ELEMENTS = sum(int(np.prod(shape)) for shape in BUNDLE_SHAPES.values())

ACTIVATION_NAMES = (
    "stem.gn",
    "stem",
    "stage1.block0",
    "stage1.block1",
    "stage2.block0",
    "stage2.block1",
    "stage3.block0",
    "stage3.block1",
    "stage4.block0",
    "stage4.block1",
    "pooled",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class Stem(nn.Module):
    """Bias-free stride-1 3x3 convolution followed by affine GroupNorm(32)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn = nn.GroupNorm(GROUP_COUNT, 64, eps=GN_EPS, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


class BasicBlock(nn.Module):
    """conv3x3 -> GN -> ReLU -> conv3x3 -> GN -> (+) identity -> ReLU."""

    def __init__(self, cin: int, cout: int, stride: int = 1, projection: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(GROUP_COUNT, cout, eps=GN_EPS, affine=True)
        self.gn2 = nn.GroupNorm(GROUP_COUNT, cout, eps=GN_EPS, affine=True)
        if projection:
            self.proj = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)
            self.proj_gn = nn.GroupNorm(GROUP_COUNT, cout, eps=GN_EPS, affine=True)
        else:
            self.proj = None
            self.proj_gn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        if self.proj is None:
            identity = x
        else:
            identity = self.proj_gn(self.proj(x))
        return F.relu(y + identity)


class Stage(nn.Module):
    """Two BasicBlocks; block 0 may carry the stride-2 projection."""

    def __init__(self, cin: int, cout: int, stride: int = 1, projection: bool = False) -> None:
        super().__init__()
        self.block0 = BasicBlock(cin, cout, stride=stride, projection=projection)
        self.block1 = BasicBlock(cout, cout, stride=1, projection=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block1(self.block0(x))


class SmallResNet(nn.Module):
    """Frozen small-image ResNet-18 with GroupNorm-32."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = Stem()
        self.stage1 = Stage(64, 64, stride=1, projection=False)
        self.stage2 = Stage(64, 128, stride=2, projection=True)
        self.stage3 = Stage(128, 256, stride=2, projection=True)
        self.stage4 = Stage(256, 512, stride=2, projection=True)
        self.classifier = nn.Linear(CLASSIFIER_IN, NUM_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        pooled = torch.mean(h, dim=(2, 3))
        return self.classifier(pooled)


def make_model() -> SmallResNet:
    """Return the frozen architecture with framework-default initial values.

    Callers must replace every array with ``load_bundle`` before training;
    the default initializer is never the source of Phase 3 parameters.
    """
    model = SmallResNet()
    named = list(model.named_parameters())
    names = [name for name, _ in named]
    _require(len(named) == ARRAY_COUNT, "model has %d parameter arrays, want %d" % (len(named), ARRAY_COUNT))
    _require(set(names) == set(BUNDLE_NAMES), "model parameter names differ from the frozen bundle names")
    _require(len({name for name, _ in named}) == ARRAY_COUNT, "model parameter names are not unique")
    bad_shapes = [name for name, param in named
                  if tuple(int(x) for x in param.shape) != BUNDLE_SHAPES[name]]
    _require(not bad_shapes, "model parameter shapes differ from the frozen bundle: %s" % (bad_shapes[:5],))
    total = sum(int(p.numel()) for _, p in named)
    _require(total == PARAMETER_COUNT, "model has %d parameters, want %d" % (total, PARAMETER_COUNT))
    _require(all(p.dtype == torch.float32 for _, p in named), "model parameters are not all float32")
    _require(all(p.requires_grad for _, p in named), "model parameters are not all trainable")
    return model


def load_bundle(model: nn.Module, bundle_path: str) -> dict:
    """Load every array from an ``initial_bundle.npz`` and verify exactness.

    Verifies the archive name set, dtype, shape, the expected 62 names and
    11,271,432 elements, and bitwise equality after loading. Raises
    ``RuntimeError`` on any mismatch, so a partially loaded model is never
    returned.
    """
    with np.load(bundle_path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    info = load_bundle_arrays(model, arrays)
    info["path"] = os.path.abspath(bundle_path)
    return info


def load_bundle_arrays(model: nn.Module, arrays: dict) -> dict:
    """Array-level implementation of ``load_bundle`` (testable without a file)."""
    named = dict(model.named_parameters())
    expected = set(BUNDLE_NAMES)
    actual = set(arrays)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(not missing and not extra,
             "bundle name-set mismatch: missing=%s extra=%s" % (missing[:5], extra[:5]))
    _require(set(named) == expected,
             "model parameter name-set mismatch: missing=%s extra=%s"
             % (sorted(expected - set(named))[:5], sorted(set(named) - expected)[:5]))

    bad_dtype = []
    bad_shape = []
    bad_model_shape = []
    for name in BUNDLE_NAMES:
        arr = arrays[name]
        shape = BUNDLE_SHAPES[name]
        if arr.dtype != np.float32:
            bad_dtype.append("%s:%s" % (name, arr.dtype))
        if tuple(int(x) for x in arr.shape) != shape:
            bad_shape.append("%s:%s" % (name, tuple(arr.shape)))
        if tuple(int(x) for x in named[name].shape) != shape:
            bad_model_shape.append("%s:%s" % (name, tuple(named[name].shape)))
    _require(not bad_dtype, "bundle dtype mismatch: %s" % (bad_dtype[:5],))
    _require(not bad_shape, "bundle shape mismatch: %s" % (bad_shape[:5],))
    _require(not bad_model_shape, "model shape mismatch: %s" % (bad_model_shape[:5],))
    total = sum(int(arrays[name].size) for name in BUNDLE_NAMES)
    _require(total == PARAMETER_COUNT, "bundle has %d elements, want %d" % (total, PARAMETER_COUNT))

    mismatched = []
    max_abs_diff = 0.0
    with torch.no_grad():
        for name in BUNDLE_NAMES:
            param = named[name]
            source = torch.from_numpy(arrays[name]).to(device=param.device, dtype=torch.float32)
            param.copy_(source)
            if not torch.equal(param, source):
                mismatched.append(name)
            diff = float((param - source).abs().max().item())
            max_abs_diff = max(max_abs_diff, diff)
    _require(not mismatched, "bundle load was not exact: %s" % (mismatched[:5],))
    _require(max_abs_diff == 0.0, "bundle load max_abs_diff=%r, want 0.0" % (max_abs_diff,))

    return {
        "array_count": len(BUNDLE_NAMES),
        "element_count": total,
        "names": list(BUNDLE_NAMES),
        "shapes": {name: list(BUNDLE_SHAPES[name]) for name in BUNDLE_NAMES},
        "max_abs_diff": max_abs_diff,
        "mismatched": mismatched,
        "bundle_hash": da.bundle_hash_of_arrays([arrays[name] for name in BUNDLE_NAMES]),
    }


def parameter_groups(model: nn.Module):
    """Return ``(decay, no_decay)`` as ordered ``[(name, Parameter), ...]``.

    Decay is the 20 convolution weights plus ``classifier.weight`` (21 arrays);
    no-decay is the 40 GroupNorm gamma/beta arrays plus ``classifier.bias``
    (41 arrays). Membership is derived from the frozen bundle layouts and is
    asserted here, not inferred from library defaults.
    """
    named = dict(model.named_parameters())
    _require(set(named) == set(BUNDLE_NAMES),
             "model parameter names differ from the frozen bundle names")
    decay = []
    no_decay = []
    for name in BUNDLE_NAMES:
        param = named[name]
        layout = BUNDLE_LAYOUTS[name]
        if layout == "conv_kernel_out_in_kh_kw" or name == "classifier.weight":
            decay.append((name, param))
        elif layout == "vector":
            no_decay.append((name, param))
        else:
            raise RuntimeError("unclassified parameter %s with layout %r" % (name, layout))

    conv_names = [name for name in BUNDLE_NAMES if BUNDLE_LAYOUTS[name] == "conv_kernel_out_in_kh_kw"]
    vector_names = [name for name in BUNDLE_NAMES if BUNDLE_LAYOUTS[name] == "vector"]
    _require(len(decay) == 21, "decay group has %d arrays, want 21" % len(decay))
    _require(len(no_decay) == 41, "no-decay group has %d arrays, want 41" % len(no_decay))
    _require(set(name for name, _ in decay) == set(conv_names) | {"classifier.weight"},
             "decay group membership mismatch")
    _require(set(name for name, _ in no_decay) == set(vector_names),
             "no-decay group membership mismatch")
    _require(len(conv_names) == 20, "expected 20 convolution weights, found %d" % len(conv_names))
    _require(len(vector_names) == 41, "expected 41 vector arrays, found %d" % len(vector_names))
    _require(all(p.ndim == 4 for name, p in decay if name != "classifier.weight"),
             "a decayed convolution weight is not 4-D")
    _require(all(p.ndim == 2 for name, p in decay if name == "classifier.weight"),
             "classifier.weight is not 2-D")
    _require(all(p.ndim == 1 for _, p in no_decay), "a no-decay parameter is not 1-D")
    return decay, no_decay


def build_optimizer(model: nn.Module, lr: float = ETA_MAX, weight_decay: float = WEIGHT_DECAY,
                    betas=BETAS, eps: float = EPS) -> torch.optim.AdamW:
    """Build the frozen two-group AdamW configuration and assert it exactly."""
    decay, no_decay = parameter_groups(model)
    decay_params = [p for _, p in decay]
    no_decay_params = [p for _, p in no_decay]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(lr),
        betas=tuple(float(b) for b in betas),
        eps=float(eps),
    )
    _require(len(optimizer.param_groups) == 2, "optimizer must have exactly two parameter groups")
    _require([id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in decay_params],
             "decay parameter group order/content changed")
    _require([id(p) for p in optimizer.param_groups[1]["params"]] == [id(p) for p in no_decay_params],
             "no-decay parameter group order/content changed")
    _require(len(optimizer.param_groups[0]["params"]) == 21,
             "decay group has %d parameters, want 21" % len(optimizer.param_groups[0]["params"]))
    _require(len(optimizer.param_groups[1]["params"]) == 41,
             "no-decay group has %d parameters, want 41" % len(optimizer.param_groups[1]["params"]))
    for group in optimizer.param_groups:
        _require(group["betas"] == tuple(float(b) for b in betas), "betas mismatch: %r" % (group["betas"],))
        _require(group["eps"] == float(eps), "eps mismatch: %r" % (group["eps"],))
        _require(group["lr"] == float(lr), "lr mismatch: %r" % (group["lr"],))
        _require(group["amsgrad"] is False, "amsgrad must be false")
        _require(group["maximize"] is False, "maximize must be false")
    _require(optimizer.param_groups[0]["weight_decay"] == float(weight_decay),
             "decay group weight_decay=%r, want %r"
             % (optimizer.param_groups[0]["weight_decay"], float(weight_decay)))
    _require(optimizer.param_groups[1]["weight_decay"] == 0.0,
             "no-decay group weight_decay=%r, want 0.0" % (optimizer.param_groups[1]["weight_decay"],))
    return optimizer


def schedule_eta(t: int, eta_max: float = ETA_MAX, warmup: int = WARMUP_UPDATES,
                 total: int = TOTAL_UPDATES) -> float:
    """Frozen warm-up/cosine learning rate at global update index ``t`` (1-based)."""
    t = int(t)
    warmup = int(warmup)
    total = int(total)
    _require(1 <= t <= total, "update index t=%d outside 1..%d" % (t, total))
    _require(1 <= warmup < total, "warmup=%d outside 1..%d-1" % (warmup, total))
    if t <= warmup:
        return float(eta_max) * t / warmup
    return float(eta_max) * 0.5 * (1.0 + math.cos(math.pi * (t - warmup) / (total - warmup)))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, eta: float) -> float:
    """Set the learning rate on every parameter group and assert the value."""
    eta = float(eta)
    for group in optimizer.param_groups:
        group["lr"] = eta
    for group in optimizer.param_groups:
        _require(group["lr"] == eta, "learning rate did not stick: %r != %r" % (group["lr"], eta))
    return eta


def configure_backend(num_threads: int = 1) -> dict:
    """Disable TF32, force deterministic cuDNN, and assert/return the flags."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(num_threads))
    flags = backend_flags()
    expected = {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, want in expected.items():
        _require(flags[key] == want, "%s=%r, want %r" % (key, flags[key], want))
    return flags


def backend_flags() -> dict:
    """Current numerics-relevant backend flags."""
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def resolve_device(name: str) -> torch.device:
    """Resolve ``cuda`` to device 0 and fail loudly when CUDA is unavailable."""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is not available "
                "(torch.cuda.is_available() == False)"
            )
        if torch.cuda.device_count() < 1:
            raise RuntimeError("--device cuda requested but no CUDA device is visible")
        return torch.device("cuda", 0)
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("unsupported device %r (expected 'cuda' or 'cpu')" % (name,))


def forward_with_activations(model: nn.Module, x: torch.Tensor):
    """Return ``(logits, activations)`` with the frozen Phase 3 activation keys.

    Keys: ``stem.gn`` (after GroupNorm, before ReLU), ``stem`` (after ReLU),
    ``stage{1..4}.block{0,1}`` (block outputs after the residual ReLU), and
    ``pooled`` (global spatial mean, before the classifier).
    """
    activations = {}
    h = model.stem.conv(x)
    h = model.stem.gn(h)
    activations["stem.gn"] = h
    h = F.relu(h)
    activations["stem"] = h
    for s in (1, 2, 3, 4):
        stage = getattr(model, "stage%d" % s)
        for b in (0, 1):
            h = getattr(stage, "block%d" % b)(h)
            activations["stage%d.block%d" % (s, b)] = h
    pooled = torch.mean(h, dim=(2, 3))
    activations["pooled"] = pooled
    logits = model.classifier(pooled)
    return logits, activations


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, x, y, t: int,
               capture_activations: bool = False, capture_grads: bool = False) -> dict:
    """Run one frozen update: zero -> forward -> loss -> backward -> eta -> step.

    Returns a dict with detached ``logits``/``loss``, ``eta``, and optionally
    detached ``activations`` and cloned ``grads`` (captured after backward and
    before ``optimizer.step()``). Gradients are cloned so they remain valid
    after the next ``zero_grad``.
    """
    _require(isinstance(model, nn.Module), "model must be an nn.Module")
    params = list(model.parameters())
    _require(len(params) > 0, "model has no parameters")
    device = params[0].device
    _require(all(p.device == device for p in params), "model parameters span multiple devices")

    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.int64, device=device)
    _require(x.ndim == 4, "input must be NCHW, got shape %s" % (tuple(x.shape),))
    _require(y.ndim == 1 and int(y.shape[0]) == int(x.shape[0]),
             "labels must be 1-D with one entry per sample")

    optimizer.zero_grad(set_to_none=True)
    if capture_activations:
        logits, activations = forward_with_activations(model, x)
    else:
        logits = model(x)
        activations = None
    loss = F.cross_entropy(logits, y)
    _require(bool(torch.isfinite(loss).item()), "non-finite loss: %r" % (loss.item(),))
    loss.backward()

    grads = None
    if capture_grads:
        grads = {}
        for name, param in model.named_parameters():
            _require(param.grad is not None, "missing gradient for %s" % name)
            grads[name] = param.grad.detach().clone()

    eta = schedule_eta(t)
    set_optimizer_lr(optimizer, eta)
    optimizer.step()

    return {
        "logits": logits.detach(),
        "loss": loss.detach(),
        "eta": eta,
        "activations": (None if activations is None
                        else {name: value.detach() for name, value in activations.items()}),
        "grads": grads,
        "update": int(t),
    }
