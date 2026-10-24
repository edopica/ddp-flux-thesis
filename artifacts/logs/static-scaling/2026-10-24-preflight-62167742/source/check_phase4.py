#!/usr/bin/env python3
"""
Independent Phase 4 distributed-correctness checker (numpy + stdlib only).

This checker consumes the static-graph Phase 4 artifacts described in
``temp/phase4-implementation-contract.md``:

  reference/<framework>/<device>/phase4_reference.npz + .json
  ddp/<framework>/w<world>/rank<rank>.npz + .json
  ddp/<framework>/w<world>/summary.json
  split/<framework>/<device>/split_control.npz + .json

It does not import torch, Julia, or any project module.  Model shapes are
read from the canonical initial bundle, so a reduced synthetic bundle can be
used to exercise the checker end to end.

Main checks (one ``PHASE4_CATEGORY <name> PASS|FAIL`` line each):

  * fixture/bundle/reference/DDP file, manifest, hash, array-set integrity;
  * Phase 2 shard replay and the schedule fixture rank sequences;
  * rank replica state hashes and bitwise-identical u3 state arrays;
  * Flux offline synchronized gradient ``sum_r(localgrad_r) / world``;
  * PyTorch rank-0 DDP synchronized gradient and replica identity;
  * an independent float64 AdamW oracle (exactly-once update check);
  * same-framework DDP-versus-reference comparisons;
  * Flux-versus-PyTorch DDP and Flux-DDP-versus-PyTorch-reference checks;
  * Flux collective protocol (guarded 64-entry by default, or the historical
    static 62-entry protocol) and PyTorch DDP configuration;
  * split-control diagnostics (reported, not acceptance).

The elementwise rule is always
``|a - b| <= atol + rtol * max(|a|, |b|)``.
Integer counters and SHA-256 values are compared exactly.

Reproduction examples::

  python3 scripts/static_scaling/check_phase4.py --self-test
  python3 scripts/static_scaling/check_phase4.py \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \\
    --pytorch-reference <OUT>/reference/pytorch/cpu/phase4_reference.npz \\
    --pytorch-reference-manifest <OUT>/reference/pytorch/cpu/phase4_reference_manifest.json \\
    --flux-reference <OUT>/reference/flux/cpu/phase4_reference.npz \\
    --flux-reference-manifest <OUT>/reference/flux/cpu/phase4_reference_manifest.json \\
    --ddp-root <OUT>/ddp --out-dir <OUT>/check --device cpu \\
    --world-sizes 2,4 --frameworks flux,pytorch

Exit code is 0 only when the final line is ``PHASE4_CHECK PASS``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Frozen Phase 2/3/4 constants
# ---------------------------------------------------------------------------

UPDATES = ("u1", "u2", "u3")
INPUT_UPDATE_INDEX = {"u1": 0, "u2": 1, "u3": 0}
EXPECTED_UPDATES = 3
BATCH = 12
SCHEDULE_BATCH = 512
SCHEDULE_SAMPLES = 99840
SCHEDULE_STEPS = 195
WARMUP_UPDATES = 975
TOTAL_UPDATES = 9750
ETA_MAX = 1.0e-3
BETAS = (0.9, 0.999)
EPS = 1.0e-8
WEIGHT_DECAY = 1.0e-4
DEFAULT_WORLD_SIZES = "2,4"
FLUX_COLLECTIVE_PROTOCOLS = ("guarded", "static")
DEFAULT_FLUX_COLLECTIVE_PROTOCOL = "guarded"
STATIC_FLUX_GRADIENT_ENTRIES = 62
GUARDED_FLUX_TRACE_ENTRIES = STATIC_FLUX_GRADIENT_ENTRIES + 2

_MISSING = object()
_HEX = set("0123456789abcdef")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def fmt_float(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return repr(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return "%.9g" % number


def shortlist(items, limit: int = 6) -> str:
    items = [str(item) for item in items]
    if not items:
        return "none"
    shown = ";".join(items[:limit])
    if len(items) > limit:
        shown += ";+%dmore" % (len(items) - limit)
    return shown


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_c_order(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def bundle_hash_of_arrays(arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def load_npz(path: str) -> dict:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _norm_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _first_key(container, keys):
    if not isinstance(container, dict):
        return _MISSING
    normalised = {_norm_key(key): key for key in container}
    for wanted in keys:
        key = normalised.get(_norm_key(wanted))
        if key is not None:
            return container[key]
    return _MISSING


def _first_value(container, keys, default=None):
    value = _first_key(container, keys)
    return default if value is _MISSING else value


def _as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            match = re.search(r"(\d+)\s*$", text)
            return int(match.group(1)) if match else None
    if isinstance(value, np.ndarray) and value.size == 1:
        return _as_int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _as_int(value[0])
    return None


def _is_hash(value) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and set(text) <= _HEX


def _is_dtype_name(value, expected: str) -> bool:
    if value is None:
        return False
    text = str(value).lower()
    aliases = {
        "float32": {"float32", "f32", "float", "single", "torch.float32",
                    "numpy.float32", "<f4", "float32le"},
        "float64": {"float64", "f64", "double", "torch.float64",
                    "numpy.float64", "<f8"},
        "int64": {"int64", "i64", "long", "torch.int64", "numpy.int64",
                  "<i8", "integer"},
    }
    wanted = aliases.get(expected, {expected})
    if text in wanted:
        return True
    if expected == "int64" and text.startswith(("int", "uint")):
        return True
    return False


# ---------------------------------------------------------------------------
# Comparison primitives
# ---------------------------------------------------------------------------

@dataclass
class CompareResult:
    ok: bool
    max_abs: float
    violations: int
    checked: int
    detail: str = ""

    def merge(self, other: "CompareResult") -> "CompareResult":
        return CompareResult(
            ok=self.ok and other.ok,
            max_abs=max(self.max_abs, other.max_abs),
            violations=self.violations + other.violations,
            checked=self.checked + other.checked,
            detail=";".join(part for part in (self.detail, other.detail)
                            if part),
        )


FP32_ABSOLUTE_LIMIT = None


def compare_arrays(actual, expected, atol: float, rtol: float, *, fp32=False) -> CompareResult:
    if actual is None or expected is None:
        return CompareResult(False, math.inf, 1, 0, "missing")
    try:
        a = np.asarray(actual)
        b = np.asarray(expected)
    except Exception as exc:  # pragma: no cover - defensive
        return CompareResult(False, math.inf, 1, 0,
                             "%s:%s" % (type(exc).__name__, exc))
    if a.shape != b.shape:
        return CompareResult(
            False, math.inf, 1, 0,
            "shape%s_vs_%s" % (tuple(a.shape), tuple(b.shape)))
    if a.size == 0:
        return CompareResult(True, 0.0, 0, 0, "empty")
    try:
        af = np.asarray(a, dtype=np.float64)
        bf = np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return CompareResult(False, math.inf, 1, 0,
                             "%s:%s" % (type(exc).__name__, exc))
    finite = np.isfinite(af) & np.isfinite(bf)
    diff = np.abs(af - bf)
    bound = atol + rtol * np.maximum(np.abs(af), np.abs(bf))
    bad = (~finite) | (diff > bound)
    if FP32_ABSOLUTE_LIMIT is not None and (atol != 0 or rtol != 0) and (fp32 or a.dtype == np.float32 or b.dtype == np.float32):
        # User-approved numerical rule. Exact structural comparisons stay exact.
        bad = (~finite) | (diff >= FP32_ABSOLUTE_LIMIT)
    violations = int(np.count_nonzero(bad))
    max_abs = float(np.max(diff)) if bool(np.all(finite)) else math.inf
    detail = "n=%d" % int(a.size)
    return CompareResult(violations == 0, max_abs, violations,
                         int(a.size), detail)


def compare_exact(actual, expected) -> CompareResult:
    if actual is None or expected is None:
        return CompareResult(False, math.inf, 1, 0, "missing")
    a = np.asarray(actual)
    b = np.asarray(expected)
    if a.shape != b.shape:
        return CompareResult(False, math.inf, 1, 0,
                             "shape%s_vs_%s" % (tuple(a.shape),
                                                tuple(b.shape)))
    if a.size == 0:
        return CompareResult(True, 0.0, 0, 0, "empty")
    if a.dtype.kind in "iu" and b.dtype.kind in "iu":
        different = a != b
        max_abs = 0.0
        if np.any(different):
            max_abs = float(np.max(np.abs(a.astype(np.float64) -
                                         b.astype(np.float64))))
        return CompareResult(not bool(np.any(different)), max_abs,
                             int(np.count_nonzero(different)), int(a.size),
                             "n=%d" % int(a.size))
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    different = af != bf
    max_abs = float(np.max(np.abs(af - bf))) if np.any(different) else 0.0
    return CompareResult(not bool(np.any(different)), max_abs,
                         int(np.count_nonzero(different)), int(a.size),
                         "n=%d" % int(a.size))


def merge_results(results) -> CompareResult:
    merged = CompareResult(True, 0.0, 0, 0, "")
    problems = []
    checked = 0
    for result in results:
        merged = merged.merge(result)
        checked += 1
        if result.detail and not result.detail.startswith("n=") and (
                not result.ok):
            problems.append(result.detail)
    merged.detail = "arrays=%d" % checked
    if problems:
        merged.detail += " " + shortlist(problems)
    return merged


# ---------------------------------------------------------------------------
# Canonical bundle/model schema
# ---------------------------------------------------------------------------

def canonical_bundle_names():
    names = ["stem.conv.weight", "stem.gn.weight", "stem.gn.bias"]
    for stage in range(1, 5):
        for block in (0, 1):
            prefix = "stage%d.block%d" % (stage, block)
            names.extend([
                prefix + ".conv1.weight",
                prefix + ".conv2.weight",
                prefix + ".gn1.weight",
                prefix + ".gn1.bias",
                prefix + ".gn2.weight",
                prefix + ".gn2.bias",
            ])
        if stage >= 2:
            prefix = "stage%d.block0" % stage
            names.extend([
                prefix + ".proj.weight",
                prefix + ".proj_gn.weight",
                prefix + ".proj_gn.bias",
            ])
    names.extend(["classifier.weight", "classifier.bias"])
    return names


def is_group_norm_name(name: str) -> bool:
    for suffix in (".gn.weight", ".gn.bias", ".gn1.weight", ".gn1.bias",
                   ".gn2.weight", ".gn2.bias", ".proj_gn.weight",
                   ".proj_gn.bias"):
        if name.endswith(suffix):
            return True
    return False


def is_decay_name(name: str) -> bool:
    return name.endswith(".weight") and not is_group_norm_name(name)


def expected_reference_arrays(bundle_shapes, batch, input_hw, n_classes):
    expected = {}
    for update in UPDATES:
        expected["input." + update] = ("float32", (batch, 3) + tuple(input_hw))
        expected["label." + update] = ("int64", (batch,))
        expected["eta." + update] = ("float64", (1,))
        expected["loss." + update] = ("float32", (1,))
        expected["logits." + update] = ("float32", (batch, n_classes))

    stem_width = bundle_shapes["stem.conv.weight"][0]
    stage_widths = [
        bundle_shapes["stage%d.block0.conv2.weight" % stage][0]
        for stage in range(1, 5)
    ]
    height, width = int(input_hw[0]), int(input_hw[1])
    expected["act.stem.gn"] = ("float32",
                               (batch, stem_width, height, width))
    expected["act.stem"] = ("float32", (batch, stem_width, height, width))
    for stage in range(1, 5):
        scale = 2 ** (stage - 1)
        shape = (batch, stage_widths[stage - 1],
                 max(1, height // scale), max(1, width // scale))
        for block in (0, 1):
            expected["act.stage%d.block%d" % (stage, block)] = (
                "float32", shape)
    expected["act.pooled"] = ("float32", (batch, stage_widths[-1]))

    for update in ("u1", "u3"):
        for name in bundle_shapes:
            expected["grad.%s.%s" % (update, name)] = (
                "float32", tuple(bundle_shapes[name]))
    for prefix in ("param.u3.", "m.u3.", "v.u3."):
        for name in bundle_shapes:
            expected[prefix + name] = ("float32", tuple(bundle_shapes[name]))
    expected["step.u3"] = ("int64", (1,))
    return expected


def expected_ddp_arrays(framework, world, batch, input_hw, n_classes,
                        bundle_shapes):
    local = batch // world
    expected = {
        "meta.rank": ("int", ()),
        "meta.world_size": ("int", ()),
        "meta.device_index": ("int", ()),
        "meta.updates": ("int", ()),
        "input.u1": ("float32", (local, 3) + tuple(input_hw)),
        "input.u3": ("float32", (local, 3) + tuple(input_hw)),
        "label.u1": ("int64", (local,)),
        "label.u3": ("int64", (local,)),
    }
    for update in UPDATES:
        expected["eta." + update] = ("float64", (1,))
        expected["loss." + update] = ("float32", (1,))
    expected["logits.u1"] = ("float32", (local, n_classes))
    expected["logits.u3"] = ("float32", (local, n_classes))
    gradient_prefix = "localgrad" if framework == "flux" else "syncgrad"
    for update in UPDATES:
        for name in bundle_shapes:
            expected["%s.%s.%s" % (gradient_prefix, update, name)] = (
                "float32", tuple(bundle_shapes[name]))
    for prefix in ("param.u3.", "m.u3.", "v.u3."):
        for name in bundle_shapes:
            expected[prefix + name] = ("float32", tuple(bundle_shapes[name]))
    if framework == "flux":
        for name in bundle_shapes:
            expected["beta_power.u3." + name] = ("float32", (2,))
    else:
        expected["step.u3"] = ("int64", (1,))
    return expected


# ---------------------------------------------------------------------------
# Schedule and sharding (independent replay of the Phase 2 rule)
# ---------------------------------------------------------------------------

def schedule_eta(update_index: int) -> float:
    if update_index < 1 or update_index > TOTAL_UPDATES:
        raise ValueError("schedule index outside 1..T: %r" % (update_index,))
    if update_index <= WARMUP_UPDATES:
        return ETA_MAX * update_index / WARMUP_UPDATES
    return ETA_MAX * 0.5 * (1.0 + math.cos(
        math.pi * (update_index - WARMUP_UPDATES) /
        (TOTAL_UPDATES - WARMUP_UPDATES)))


def epoch_sample_ids(permutation, batch: int = SCHEDULE_BATCH):
    array = np.asarray(permutation)
    if batch <= 0:
        raise ValueError("batch must be positive")
    n_effective = (len(array) // batch) * batch
    return array[:n_effective]


def rank_shard(batch_ids, rank: int, world: int):
    array = np.asarray(batch_ids)
    if world < 1 or not 0 <= rank < world:
        raise ValueError("invalid rank/world")
    if len(array) % world != 0:
        raise ValueError("batch length not divisible by world")
    chunk = len(array) // world
    return array[rank * chunk:(rank + 1) * chunk]


def rank_sequence(permutation, batch: int, rank: int, world: int):
    ids = epoch_sample_ids(permutation, batch)
    if batch % world != 0:
        raise ValueError("batch not divisible by world")
    chunk = batch // world
    n_batches = len(ids) // batch
    out = np.empty(n_batches * chunk, dtype=ids.dtype)
    position = 0
    for index in range(n_batches):
        global_batch = ids[index * batch:(index + 1) * batch]
        out[position:position + chunk] = global_batch[
            rank * chunk:(rank + 1) * chunk]
        position += chunk
    return out


def beta_power_expected(beta, completed_updates: int):
    power = np.asarray(beta, dtype=np.float32)
    base = np.asarray(beta, dtype=np.float32)
    for _ in range(int(completed_updates)):
        power = (power * base).astype(np.float32)
    return power


# ---------------------------------------------------------------------------
# Independent AdamW oracle
# ---------------------------------------------------------------------------

def adamw_oracle(x0_by_name, gradients, etas, bundle_names,
                 betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY):
    """Float64 decoupled AdamW oracle.

    ``gradients[update][name]`` holds the synchronized float64 gradient.
    Per-leaf lambda is 1e-4 for the 21 convolution/classifier weights and 0
    for the other 41 leaves.  This mirrors the frozen Phase 3 optimizer
    contract, not a framework implementation.
    """
    x = {name: np.asarray(x0_by_name[name], dtype=np.float64).copy()
         for name in bundle_names}
    m = {name: np.zeros_like(x[name]) for name in bundle_names}
    v = {name: np.zeros_like(x[name]) for name in bundle_names}
    final = {}
    for index, update in enumerate(UPDATES, start=1):
        eta = float(etas[update])
        for name in bundle_names:
            gradient = np.asarray(gradients[update][name], dtype=np.float64)
            if gradient.shape != x[name].shape:
                raise ValueError("gradient shape mismatch for %s: %s vs %s" %
                                 (name, gradient.shape, x[name].shape))
            lambda_ = weight_decay if is_decay_name(name) else 0.0
            m[name] = betas[0] * m[name] + (1.0 - betas[0]) * gradient
            v[name] = betas[1] * v[name] + (1.0 - betas[1]) * gradient * gradient
            m_hat = m[name] / (1.0 - betas[0] ** index)
            v_hat = v[name] / (1.0 - betas[1] ** index)
            dx = m_hat / (np.sqrt(v_hat) + eps)
            x[name] = x[name] - eta * (dx + lambda_ * x[name])
        final[update] = {
            "param": {name: x[name].copy() for name in bundle_names},
            "m": {name: m[name].copy() for name in bundle_names},
            "v": {name: v[name].copy() for name in bundle_names},
        }
    return final


# ---------------------------------------------------------------------------
# Category log
# ---------------------------------------------------------------------------

class CheckLog:
    def __init__(self):
        self.categories = []
        self.failed = []
        self.diagnostics = {}

    def category(self, name: str, ok: bool, max_abs=0.0, violations: int = 0,
                 detail: str = "", required: bool = True) -> bool:
        status = "PASS" if ok else "FAIL"
        line = "PHASE4_CATEGORY %s %s max_abs=%s violations=%d" % (
            name, status, fmt_float(max_abs), int(violations))
        if detail:
            line += " " + detail
        print(line, flush=True)
        self.categories.append({
            "name": name,
            "status": status,
            "max_abs": float(max_abs),
            "violations": int(violations),
            "detail": detail,
            "required": bool(required),
        })
        if required and not ok:
            self.failed.append(name)
        return bool(ok)

    def from_result(self, name: str, result: CompareResult, detail: str = "",
                    required: bool = True) -> bool:
        text = result.detail
        if detail:
            text = (detail + " " + text).strip()
        return self.category(name, result.ok, result.max_abs,
                             result.violations, text, required=required)

    def diagnostic(self, name: str, value) -> None:
        self.diagnostics[name] = value


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def file_entries(manifest):
    entries = []
    if not isinstance(manifest, dict):
        return entries
    keys = ("files", "fixture_files", "reference_files", "artifact_files",
            "npz_file", "reference_file", "file")
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and isinstance(
                        entry.get("path"), str):
                    entries.append((entry["path"], entry.get("sha256"),
                                    entry.get("bytes")))
        elif isinstance(value, dict):
            path = value.get("path")
            if isinstance(path, str):
                entries.append((path, value.get("sha256"), value.get("bytes")))
            else:
                for path, meta in value.items():
                    if isinstance(meta, str):
                        entries.append((path, meta, None))
                    elif isinstance(meta, dict):
                        entries.append((path, meta.get("sha256"),
                                        meta.get("bytes")))
    for key in ("fixture_sha256", "fixture_hashes", "file_sha256", "hashes"):
        value = manifest.get(key)
        if not isinstance(value, dict):
            continue
        for path, meta in value.items():
            if isinstance(meta, str):
                entries.append((path, meta, None))
            elif isinstance(meta, dict):
                entries.append((path, meta.get("sha256"),
                                meta.get("bytes")))
    return entries


def find_recorded_sha(manifest, filename: str, extra_keys=()):
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]
    for path, digest, size in file_entries(manifest):
        if os.path.basename(path) == basename and isinstance(digest, str):
            return digest.lower(), size, path
    if not isinstance(manifest, dict):
        return None, None, None
    keys = list(extra_keys) + [
        stem + "_sha256",
        stem.replace(".", "_") + "_sha256",
        basename + "_sha256",
    ]
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and _is_hash(value):
            return value.lower(), None, key
    reference_entry = manifest.get("reference_file")
    if isinstance(reference_entry, dict):
        digest = reference_entry.get("sha256")
        path = reference_entry.get("path")
        if isinstance(digest, str) and (
                not isinstance(path, str) or
                os.path.basename(path) == basename):
            return digest.lower(), reference_entry.get("bytes"), "reference_file"
    return None, None, None


def extract_array_entries(manifest):
    if not isinstance(manifest, dict):
        return None
    for key in ("arrays", "npz_arrays", "array_metadata", "array_entries"):
        value = manifest.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            entries = []
            for name, meta in value.items():
                if isinstance(meta, dict):
                    entry = dict(meta)
                    entry.setdefault("name", name)
                    entries.append(entry)
            return entries
    return None


def artifact_array_name(name):
    if isinstance(name, str) and "::" in name:
        return name.split("::", 1)[1]
    return name


def normalise_array_entry(entry):
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    digest = entry.get("sha256") or entry.get("sha256_c")
    if isinstance(shape, list):
        shape = tuple(shape)
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "sha256": digest.lower() if isinstance(digest, str) else None,
    }


def check_recorded_file_hash(manifest, path, extra_keys=()):
    if not os.path.isfile(path):
        return ["missing_file:" + os.path.basename(path)], None
    recorded, size, source = find_recorded_sha(
        manifest, os.path.basename(path), extra_keys=extra_keys)
    if recorded is None:
        return ["no_recorded_sha256:" + os.path.basename(path)], None
    problems = []
    actual = sha256_file(path)
    if recorded != actual:
        problems.append("sha256:" + os.path.basename(path))
    if size is not None:
        try:
            if int(size) != os.path.getsize(path):
                problems.append("bytes:" + os.path.basename(path))
        except (TypeError, ValueError):
            problems.append("bad_bytes_meta:" + os.path.basename(path))
    return problems, source


# ---------------------------------------------------------------------------
# Trace parsing / validation
# ---------------------------------------------------------------------------

def _update_from_tag(value):
    text = _norm_key(value)
    if not text:
        return None
    if text in ("u1", "1", "t1", "update1", "optimizerstep1", "step1"):
        return "u1"
    if text in ("u2", "2", "t2", "update2", "optimizerstep2", "step2"):
        return "u2"
    if text in ("u3", "3", "t3", "update3", "optimizerstep3", "step3"):
        return "u3"
    for update in UPDATES:
        if update in text:
            return update
    return None


def _looks_like_entry(value) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {_norm_key(key) for key in value}
    return bool(keys & {
        "size", "count", "n", "numel", "elements", "shape", "dims",
        "op", "operation", "reduce", "reduceop", "type", "buffertype",
        "kind", "dtype", "eltype",
    })


def _trace_mapping_from_object(value, depth=0):
    if depth > 6:
        return None
    if isinstance(value, dict):
        direct = {}
        for key, item in value.items():
            update = _update_from_tag(key)
            if update and isinstance(item, list) and (
                    not item or _looks_like_entry(item[0])):
                direct[update] = item
            elif update and isinstance(item, dict):
                for nested_key in ("collectives", "collective_trace",
                                   "trace", "entries", "records",
                                   "allreduces"):
                    nested = item.get(nested_key)
                    if isinstance(nested, list) and (
                            not nested or _looks_like_entry(nested[0])):
                        direct[update] = nested
                        break
        if len(direct) >= 2:
            return direct
        for key in ("collectives", "collective_trace", "trace", "traces",
                    "protocol", "collective_protocol", "entries",
                    "allreduces", "records"):
            if key in value:
                found = _trace_mapping_from_object(value[key], depth + 1)
                if found:
                    return found
        for item in value.values():
            if isinstance(item, (dict, list)):
                found = _trace_mapping_from_object(item, depth + 1)
                if found:
                    return found
        return None
    if isinstance(value, list):
        if len(value) == 3 and all(isinstance(item, list) for item in value):
            if all(not item or _looks_like_entry(item[0]) for item in value):
                return dict(zip(UPDATES, value))
        if len(value) in (3 * GUARDED_FLUX_TRACE_ENTRIES,
                          3 * STATIC_FLUX_GRADIENT_ENTRIES) and (
                not value or _looks_like_entry(value[0])):
            per_update = len(value) // 3
            return {
                update: value[index * per_update:(index + 1) * per_update]
                for index, update in enumerate(UPDATES)
            }
        per_update = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            tag = None
            for key in ("update", "tag", "name", "label", "t", "step"):
                if key in item:
                    tag = _update_from_tag(item[key])
                    if tag:
                        break
            if tag is None:
                continue
            entries = None
            for key in ("collectives", "collective_trace", "trace",
                        "entries", "records", "allreduces"):
                if key in item and isinstance(item[key], list):
                    entries = item[key]
                    break
            if entries is not None:
                per_update[tag] = entries
        if len(per_update) >= 2:
            return per_update
    return None


def _trace_digest_mapping(value, depth=0):
    if depth > 6:
        return None
    if isinstance(value, dict):
        direct = {}
        for key, item in value.items():
            update = _update_from_tag(key)
            if update and isinstance(item, dict) and (
                    "count" in item or "sha256" in item):
                direct[update] = {
                    "count": _as_int(item.get("count")),
                    "sha256": (str(item.get("sha256")).lower()
                               if _is_hash(item.get("sha256")) else None),
                }
        if direct:
            return direct
        for key in ("collective_trace", "trace", "collectives", "protocol"):
            if key in value:
                found = _trace_digest_mapping(value[key], depth + 1)
                if found:
                    return found
        for item in value.values():
            if isinstance(item, (dict, list)):
                found = _trace_digest_mapping(item, depth + 1)
                if found:
                    return found
    elif isinstance(value, list):
        for item in value:
            found = _trace_digest_mapping(item, depth + 1)
            if found:
                return found
    return None


def extract_traces(*records):
    for record in records:
        if isinstance(record, dict):
            found = _trace_mapping_from_object(record)
            if found:
                return found
    return None


def _int_tuple(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
    return None


def normalise_trace_entry(entry):
    if not isinstance(entry, dict):
        return None
    size = _first_value(entry, ("size", "count", "n", "numel", "elements"))
    shape = _first_value(entry, ("shape", "dims", "buffer_shape"))
    shape_values = None
    if isinstance(size, (list, tuple, np.ndarray)):
        shape_values = _int_tuple(size)
        product = 1
        for item in size:
            product *= int(item)
        size = product
    elif size is None and isinstance(shape, (list, tuple)):
        shape_values = _int_tuple(shape)
        product = 1
        for item in shape:
            product *= int(item)
        size = product
    size = _as_int(size)
    canonical_size = _int_tuple(_first_value(
        entry, ("canonical_size", "canonical_shape")))
    layout = _first_value(entry, ("layout", "parameter_layout"))
    dtype = _first_value(entry, ("dtype", "eltype", "data_type"))
    operation = _first_value(entry, ("op", "operation", "reduce",
                                     "reduce_op", "reduction"))
    avg_flag = _first_value(entry, ("avg", "average", "is_avg"))
    if operation is None and isinstance(avg_flag, bool):
        operation = "avg" if avg_flag else "sum"
    buffer_type = _first_value(entry, ("type", "buftype", "buffer_type",
                                       "location", "target", "buffer_kind"))
    if dtype is None and isinstance(buffer_type, str):
        text = buffer_type.lower()
        if "float32" in text:
            dtype = "float32"
        elif "float64" in text:
            dtype = "float64"
    kind = _first_value(entry, ("kind", "role", "category"))
    name = _first_value(entry, ("name", "buffer_name", "leaf",
                                "parameter", "key"))
    is_device = _first_value(entry, ("is_device", "device_buffer",
                                     "on_device"))
    device_value = _first_value(entry, ("device", "buffer_device"))

    def classify_type():
        if isinstance(is_device, bool):
            return "device" if is_device else "host"
        if isinstance(device_value, bool):
            return "device" if device_value else "host"
        if isinstance(kind, str) and kind.lower() in ("presence", "vote"):
            # The presence flags and validity vote are host buffers by
            # contract, even if the gradient buffers are device arrays.
            return "host"
        for candidate in (buffer_type, device_value, kind):
            if candidate is None:
                continue
            text = str(candidate).lower()
            if any(token in text for token in ("cuarray", "device", "cuda",
                                               "gpu")):
                return "device"
            if any(token in text for token in ("host", "cpu", "pinned",
                                               "vector", "array", "matrix",
                                               "float32", "float64",
                                               "subarray")):
                return "host"
        return None

    return {
        "size": size,
        "shape": shape_values,
        "canonical_size": canonical_size,
        "layout": str(layout).lower() if layout is not None else None,
        "kind": str(kind).lower() if kind is not None else None,
        "type": classify_type(),
        "dtype": dtype,
        "op": str(operation).lower() if operation is not None else None,
        "name": str(name) if name is not None else None,
        "raw": entry,
    }


def parameter_layout(name: str):
    if name.endswith((".conv.weight", ".conv1.weight", ".conv2.weight",
                     ".proj.weight")):
        return "conv_kernel_out_in_kh_kw"
    if name == "classifier.weight":
        return "linear_out_in"
    return "vector"


def expected_flux_gradient_entry(name, shape, gradient_type):
    canonical_size = tuple(int(item) for item in shape)
    layout = parameter_layout(name)
    flux_shape = (tuple(reversed(canonical_size))
                  if layout == "conv_kernel_out_in_kh_kw"
                  else canonical_size)
    return {
        "size": int(np.prod(canonical_size)),
        "flux_shape": flux_shape,
        "canonical_size": canonical_size,
        "layout": layout,
        "type": gradient_type,
        "dtype": "float32",
        "op": "avg",
        "name": name,
        "kind": "gradient",
    }


def expected_flux_trace(bundle_names, bundle_shapes, device: str,
                        protocol: str = DEFAULT_FLUX_COLLECTIVE_PROTOCOL):
    if protocol not in FLUX_COLLECTIVE_PROTOCOLS:
        raise ValueError("unknown Flux collective protocol %r" % (protocol,))
    gradient_type = "device" if device == "cuda" else "host"
    gradients = [
        expected_flux_gradient_entry(name, bundle_shapes[name],
                                     gradient_type)
        for name in bundle_names
    ]
    if protocol == "static":
        return gradients
    presence = {"size": len(bundle_names), "type": "host",
                "dtype": "float32", "op": "avg", "name": None,
                "kind": "presence", "flux_shape": (len(bundle_names),),
                "canonical_size": None, "layout": None}
    vote = {"size": 1, "type": "host", "dtype": "float32",
            "op": "avg", "name": None, "kind": "vote",
            "flux_shape": (1,), "canonical_size": None, "layout": None}
    return [presence] + gradients + [vote]


def _op_ok(value) -> bool:
    if value is None:
        return False
    text = str(value).lower()
    return ("avg" in text or "mean" in text or
            text in ("allreduce_avg", "avg!"))


def validate_flux_trace(entries, bundle_names, bundle_shapes, device: str,
                        protocol: str = DEFAULT_FLUX_COLLECTIVE_PROTOCOL):
    expected = expected_flux_trace(bundle_names, bundle_shapes, device,
                                   protocol=protocol)
    violations = 0
    details = []
    if not isinstance(entries, list):
        return 1, "trace_not_list", []
    if len(entries) != len(expected):
        violations += 1
        details.append("count=%d_want_%d" % (len(entries), len(expected)))
    normalised = []
    for index in range(min(len(entries), len(expected))):
        actual = normalise_trace_entry(entries[index])
        want = expected[index]
        if actual is None:
            violations += 1
            details.append("[%d]:malformed" % index)
            normalised.append(None)
            continue
        normalised.append(actual)
        if actual["size"] != want["size"]:
            violations += 1
            details.append("[%d]:size=%s_want_%d" %
                           (index, actual["size"], want["size"]))
        if actual["type"] != want["type"]:
            violations += 1
            details.append("[%d]:type=%s_want_%s" %
                           (index, actual["type"], want["type"]))
        if actual["dtype"] is not None and not _is_dtype_name(
                actual["dtype"], "float32"):
            violations += 1
            details.append("[%d]:dtype=%s" % (index, actual["dtype"]))
        if not _op_ok(actual["op"]):
            violations += 1
            details.append("[%d]:op=%s" % (index, actual["op"]))
        if protocol == "static":
            if actual["kind"] != "gradient":
                violations += 1
                details.append("[%d]:kind=%s_want=gradient" %
                               (index, actual["kind"]))
            if want["name"] is not None and actual["name"] != want["name"]:
                violations += 1
                details.append("[%d]:name=%s_want=%s" %
                               (index, actual["name"], want["name"]))
        elif want["name"] is not None and actual["name"] is not None:
            if actual["name"] != want["name"]:
                violations += 1
                details.append("[%d]:name=%s_want=%s" %
                               (index, actual["name"], want["name"]))
        want_canonical = want.get("canonical_size")
        want_layout = want.get("layout")
        if actual["shape"] is not None and want_canonical is not None:
            actual_canonical = (tuple(reversed(actual["shape"]))
                                if want_layout == "conv_kernel_out_in_kh_kw"
                                else actual["shape"])
            if actual_canonical != want_canonical:
                violations += 1
                details.append("[%d]:canonical_size=%s_want=%s" %
                               (index, actual_canonical, want_canonical))
        if actual["canonical_size"] is not None and want_canonical is not None:
            if actual["canonical_size"] != want_canonical:
                violations += 1
                details.append("[%d]:recorded_canonical_size=%s_want=%s" %
                               (index, actual["canonical_size"],
                                want_canonical))
        if actual["layout"] is not None and want_layout is not None:
            if actual["layout"] != str(want_layout).lower():
                violations += 1
                details.append("[%d]:layout=%s_want=%s" %
                               (index, actual["layout"], want_layout))
    signature = []
    for entry in normalised:
        if entry is None:
            signature.append(None)
        else:
            signature.append((entry["size"], entry["type"], entry["op"],
                              str(entry["dtype"]).lower()
                              if entry["dtype"] is not None else None))
    return violations, shortlist(details), signature


# ---------------------------------------------------------------------------
# DDP artifact loading
# ---------------------------------------------------------------------------

def resolve_ddp_dir(root: str, framework: str, world: int):
    candidates = [
        os.path.join(root, framework, "w%d" % world),
        os.path.join(root, "ddp", framework, "w%d" % world),
        os.path.join(root, "w%d" % world, framework),
        os.path.join(root, "ddp", "w%d" % world, framework),
        os.path.join(root, "w%d" % world),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "rank0.npz")):
            return candidate
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def load_ddp_run(root: str, framework: str, world: int):
    directory = resolve_ddp_dir(root, framework, world)
    run = {
        "framework": framework,
        "world": world,
        "dir": directory,
        "ranks": [],
        "summary": None,
        "summary_path": None,
        "errors": [],
    }
    if directory is None:
        run["errors"].append("ddp_dir_missing")
        return run
    for rank in range(world):
        npz_path = os.path.join(directory, "rank%d.npz" % rank)
        json_path = os.path.join(directory, "rank%d.json" % rank)
        record = {
            "rank": rank,
            "npz_path": npz_path,
            "json_path": json_path,
            "npz": None,
            "json": None,
        }
        if os.path.isfile(npz_path):
            try:
                record["npz"] = load_npz(npz_path)
            except Exception as exc:
                run["errors"].append("rank%d.npz:%s:%s" %
                                     (rank, type(exc).__name__, exc))
        else:
            run["errors"].append("rank%d.npz_missing" % rank)
        if os.path.isfile(json_path):
            try:
                record["json"] = read_json(json_path)
            except Exception as exc:
                run["errors"].append("rank%d.json:%s:%s" %
                                     (rank, type(exc).__name__, exc))
        else:
            run["errors"].append("rank%d.json_missing" % rank)
        run["ranks"].append(record)

    summary_path = os.path.join(directory, "summary.json")
    run["summary_path"] = summary_path
    if os.path.isfile(summary_path):
        try:
            run["summary"] = read_json(summary_path)
        except Exception as exc:
            run["errors"].append("summary.json:%s:%s" %
                                 (type(exc).__name__, exc))
    else:
        run["errors"].append("summary.json_missing")
    return run


def _metadata_container(record):
    if not isinstance(record, dict):
        return {}
    for key in ("meta", "metadata", "run_metadata", "run"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return record


def extract_meta_value(record, field_name):
    aliases = {
        "rank": ("rank", "global_rank", "rank_id", "process_index"),
        "world_size": ("world_size", "world", "worldsize", "nranks",
                       "num_ranks"),
        "device_index": ("device_index", "local_rank", "local_device_index",
                         "gpu_index", "device"),
        "updates": ("updates", "num_updates", "update_count", "nupdates",
                    "n_updates", "steps"),
        "framework": ("framework", "name"),
        "backend": ("backend", "communication_backend", "comm_backend"),
        "device": ("device", "device_name", "device_type"),
    }.get(field_name, (field_name,))
    containers = []
    if isinstance(record, dict):
        containers.append(_metadata_container(record))
        containers.append(record)
        for key in ("ddp", "ddp_config", "ddp_configuration",
                    "distributed", "distributed_config", "torch_ddp",
                    "backend_configuration", "config", "runtime",
                    "environment"):
            value = record.get(key)
            if isinstance(value, dict):
                containers.append(value)
    for container in containers:
        value = _first_value(container, aliases, _MISSING)
        if value is not _MISSING:
            return value
    return _MISSING


def _state_from_key(key):
    text = _norm_key(key)
    if text in ("param", "params", "parameters", "parameter", "weights",
                "stateparam"):
        return "param"
    if text in ("m", "ema", "expavg", "expavgsq", "firstmoment", "moment1",
                "momentum"):
        return "m" if text != "expavgsq" else "v"
    if text in ("v", "expavgsq", "secondmoment", "moment2", "velocity"):
        return "v"
    if text in ("betapower", "betapowers", "beta", "betapowerpair"):
        return "beta_power"
    if text in ("step", "steps", "stepcounter", "completedupdates"):
        return "step"
    return None


def _state_from_flat_key(key):
    text = _norm_key(key)
    if "betapower" in text:
        return "beta_power"
    if any(token in text for token in ("expavgsq", "secondmoment", "moment2",
                                       "velocity")):
        return "v"
    if any(token in text for token in ("expavg", "firstmoment", "moment1",
                                       "momentum")):
        return "m"
    if any(token in text for token in ("stepcounter", "completedupdates")):
        return "step"
    if "step" in text:
        return "step"
    if "param" in text:
        return "param"
    return None


def _merge_state_maps(target, source):
    for update, states in source.items():
        if update not in UPDATES:
            continue
        bucket = target.setdefault(update, {})
        for state, value in states.items():
            bucket.setdefault(state, value)


def parse_state_hash_mapping(value):
    result = {}

    def add(update, state, digest):
        if update not in UPDATES or state is None:
            return
        if _is_hash(digest):
            result.setdefault(update, {})[state] = digest.lower()
        elif state == "step" and isinstance(digest, int) and not isinstance(
                digest, bool):
            result.setdefault(update, {})[state] = "int:%d" % digest

    if isinstance(value, dict):
        # Update-first dictionaries: {"u1": {"param": hash, ...}, ...}
        for key, item in value.items():
            update = _update_from_tag(key)
            if update is None:
                continue
            if isinstance(item, dict):
                for key2, item2 in item.items():
                    state = _state_from_key(key2)
                    if state is None:
                        state = _state_from_flat_key(key2)
                    if state is None:
                        state = _state_from_key(key)
                    add(update, state, item2)
            elif isinstance(item, list) and len(item) == 3 and all(
                    _is_hash(entry) for entry in item):
                for state, digest in zip(("param", "m", "v"), item):
                    add(update, state, digest)
        # State-first dictionaries: {"param": {"u1": hash, ...}, ...}
        for key, item in value.items():
            state = _state_from_key(key)
            if state is None:
                continue
            if isinstance(item, dict):
                for key2, item2 in item.items():
                    update = _update_from_tag(key2)
                    add(update, state, item2)
            elif isinstance(item, (list, tuple)):
                if len(item) == 3 and all(
                        _is_hash(entry) or (
                            state == "step" and isinstance(entry, int))
                        for entry in item):
                    for update, digest in zip(UPDATES, item):
                        add(update, state, digest)
        # Flat keys: {"param.u1.sha256": ..., "u1_m_sha256": ...}
        for key, item in value.items():
            state = _state_from_key(key)
            if state is None:
                state = _state_from_flat_key(key)
            update = _update_from_tag(key)
            if state is not None and update is not None:
                add(update, state, item)
    elif isinstance(value, list):
        for index, item in enumerate(value[:3]):
            update = UPDATES[index]
            if isinstance(item, dict):
                tag = None
                for key in ("update", "tag", "name", "label", "t", "step"):
                    if key in item:
                        tag = _update_from_tag(item[key])
                        if tag:
                            break
                if tag:
                    update = tag
                _merge_state_maps(result, parse_state_hash_mapping(item))
    return result


def extract_state_hashes(record):
    result = {}
    if not isinstance(record, dict):
        return result
    for key in ("state_hashes", "update_state_hashes", "state_sha256",
                "update_hashes", "hashes", "optimizer_state_hashes"):
        if key in record:
            _merge_state_maps(result, parse_state_hash_mapping(record[key]))
    for key in ("updates", "update_records", "records", "states",
                "optimizer_states"):
        if key in record and isinstance(record[key], (list, dict)):
            _merge_state_maps(result, parse_state_hash_mapping(record[key]))
    _merge_state_maps(result, parse_state_hash_mapping(record))
    return result


def required_state_names(framework: str):
    if framework == "flux":
        return ("param", "m", "v", "beta_power")
    return ("param", "m", "v", "step")


def extract_npz_record(record):
    """Return (file_sha256, array_entries) recorded in a rank JSON."""
    if not isinstance(record, dict):
        return None, None
    containers = []
    for key in ("npz", "npz_file", "artifact", "file", "output"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append(value)
    containers.append(record)
    file_sha = None
    for container in containers:
        for key in ("sha256", "npz_sha256", "npz_file_sha256",
                    "file_sha256", "sha256_file", "artifact_sha256"):
            value = container.get(key)
            if _is_hash(value):
                file_sha = value.lower()
                break
        if file_sha:
            break
    entries = None
    for container in containers:
        for key in ("arrays", "array_metadata", "npz_arrays",
                    "array_entries", "npz_metadata", "array_sha256"):
            value = container.get(key)
            if isinstance(value, list):
                entries = value
                break
            if isinstance(value, dict):
                entries = []
                for name, meta in value.items():
                    if isinstance(meta, dict):
                        entry = dict(meta)
                        entry.setdefault("name", name)
                        entries.append(entry)
                break
        if entries is not None:
            break
    return file_sha, entries


def extract_shard_ids(record):
    if not isinstance(record, dict):
        return {}
    result = {}
    containers = [record]
    for key in ("shards", "shard_ids", "inputs", "data"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key, value in container.items():
            update = _update_from_tag(key)
            if update and isinstance(value, (list, tuple, np.ndarray)):
                array = np.asarray(value)
                if array.ndim == 1 and array.dtype.kind in "iu":
                    array = array.astype(np.int64)
                    if "0based" in _norm_key(key):
                        result[update] = array
                    else:
                        result.setdefault(update, array)
            if _norm_key(key) in ("shardids", "shardsampleids", "sampleids",
                                  "inputids", "rowids", "indices", "ids"):
                if isinstance(value, dict):
                    for key2, item in value.items():
                        update2 = _update_from_tag(key2)
                        if update2 and isinstance(
                                item, (list, tuple, np.ndarray)):
                            array = np.asarray(item)
                            if array.ndim == 1 and array.dtype.kind in "iu":
                                result.setdefault(update2,
                                                  array.astype(np.int64))
                elif isinstance(value, (list, tuple, np.ndarray)):
                    array = np.asarray(value)
                    if array.ndim == 1 and array.dtype.kind in "iu":
                        result.setdefault("u1", array.astype(np.int64))
                        result.setdefault("u3", array.astype(np.int64))
    return result


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def check_fixture_integrity(log, fixture_dir):
    manifest_path = os.path.join(fixture_dir, "fixture_manifest.json")
    data_path = os.path.join(fixture_dir, "data_fixture.npz")
    schedule_path = os.path.join(fixture_dir, "schedule_fixture.npz")
    problems = []
    manifest = None
    if not os.path.isfile(manifest_path):
        problems.append("fixture_manifest.json_missing")
    else:
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            problems.append("fixture_manifest.json:%s" % type(exc).__name__)
    if not os.path.isfile(data_path):
        problems.append("data_fixture.npz_missing")
    if not os.path.isfile(schedule_path):
        problems.append("schedule_fixture.npz_missing")

    fixture_checked = 0
    if isinstance(manifest, dict):
        listed = file_entries(manifest)
        if not listed:
            problems.append("fixture_manifest_no_files")
        for path, digest, size in listed:
            basename = os.path.basename(path)
            local = os.path.join(fixture_dir, basename)
            if not os.path.isfile(local):
                problems.append("fixture_file_missing:" + basename)
                continue
            if isinstance(digest, str):
                fixture_checked += 1
                if sha256_file(local) != digest.lower():
                    problems.append("fixture_sha256:" + basename)
            if size is not None:
                try:
                    if int(size) != os.path.getsize(local):
                        problems.append("fixture_bytes:" + basename)
                except (TypeError, ValueError):
                    problems.append("fixture_bad_bytes_meta:" + basename)
        entries = extract_array_entries(manifest)
        if entries is None:
            problems.append("fixture_manifest_no_arrays")
        else:
            array_problems = 0
            for raw in entries:
                entry = normalise_array_entry(raw)
                if entry is None or not isinstance(entry["name"], str):
                    array_problems += 1
                    continue
                name = entry["name"]
                if "::" not in name:
                    array_problems += 1
                    continue
                filename, array_name = name.split("::", 1)
                local = os.path.join(fixture_dir,
                                     os.path.basename(filename))
                if not os.path.isfile(local):
                    array_problems += 1
                    continue
                try:
                    with np.load(local) as archive:
                        if array_name not in archive.files:
                            array_problems += 1
                            continue
                        array = archive[array_name]
                        if entry["dtype"] is not None and not _is_dtype_name(
                                entry["dtype"], str(array.dtype)):
                            array_problems += 1
                        if entry["shape"] is not None and tuple(
                                entry["shape"]) != tuple(array.shape):
                            array_problems += 1
                        if entry["sha256"] and sha256_c_order(
                                array) != entry["sha256"]:
                            array_problems += 1
                except Exception:
                    array_problems += 1
            if array_problems:
                problems.append("fixture_array_mismatches=%d" %
                                array_problems)
    log.category("integrity.fixture", not problems, 0.0, len(problems),
                 "files_checked=%d problems=%s" %
                 (fixture_checked, shortlist(problems)))
    return manifest


def load_fixture_data(log, fixture_dir):
    data_path = os.path.join(fixture_dir, "data_fixture.npz")
    schedule_path = os.path.join(fixture_dir, "schedule_fixture.npz")
    data = None
    schedule = None
    try:
        data = load_npz(data_path)
    except Exception as exc:
        log.category("fixture.load", False, math.inf, 1,
                     "%s:%s" % (type(exc).__name__, exc))
    try:
        schedule = load_npz(schedule_path)
    except Exception as exc:
        log.category("fixture.schedule_load", False, math.inf, 1,
                     "%s:%s" % (type(exc).__name__, exc))
    return data, schedule


def check_bundle_integrity(log, bundle, bundle_path, reference_manifests):
    names = canonical_bundle_names()
    problems = []
    if not isinstance(bundle, dict) or not bundle:
        log.category("integrity.bundle", False, 0.0, 1, "bundle_not_loaded")
        return {}
    actual = set(bundle)
    missing = sorted(set(names) - actual)
    extra = sorted(actual - set(names))
    if missing or extra:
        problems.append("name_set")
    shapes = {}
    for name in names:
        if name not in bundle:
            continue
        array = bundle[name]
        shapes[name] = tuple(array.shape)
        if array.dtype != np.float32:
            problems.append("dtype:" + name)
        if not np.all(np.isfinite(array)):
            problems.append("nonfinite:" + name)
    total = sum(int(np.prod(shape)) for shape in shapes.values())
    recorded_checked = 0
    for manifest in reference_manifests:
        if not isinstance(manifest, dict):
            continue
        extra_keys = ("bundle_sha256", "initial_bundle_sha256",
                      "bundle_file_sha256")
        recorded, _, _ = find_recorded_sha(
            manifest, os.path.basename(bundle_path), extra_keys=extra_keys)
        if recorded is not None:
            recorded_checked += 1
            if sha256_file(bundle_path) != recorded:
                problems.append("file_sha256")
        recorded_hash = _first_value(
            manifest, ("bundle_hash", "bundle_array_hash",
                       "bundle_array_sha256"), None)
        if isinstance(recorded_hash, str) and _is_hash(recorded_hash):
            recorded_checked += 1
            canonical = bundle_hash_of_arrays(
                [bundle[name] for name in names if name in bundle])
            ordered = bundle_hash_of_arrays(
                [bundle[name] for name in bundle])
            if recorded_hash.lower() not in (canonical, ordered):
                problems.append("bundle_array_hash")
    problems = sorted(set(problems))
    log.category("integrity.bundle", not problems, 0.0, len(problems),
                 "arrays=%d parameters=%d hashes_checked=%d problems=%s" %
                 (len(bundle), total, recorded_checked, shortlist(problems)))
    return shapes


def check_reference_integrity(log, framework, reference, manifest, path,
                              expected, bundle_path):
    label = "integrity.reference.%s" % framework
    problems = []
    if reference is None:
        problems.append("npz_not_loaded")
    if not isinstance(manifest, dict):
        problems.append("manifest_not_loaded")
    if not os.path.isfile(path):
        problems.append("npz_missing")

    if isinstance(manifest, dict):
        sha_problems, source = check_recorded_file_hash(
            manifest, path,
            extra_keys=("reference_sha256", "phase4_reference_sha256",
                        "phase3_reference_sha256"))
        problems.extend(sha_problems)
        if source:
            log.diagnostic("reference.%s.hash_source" % framework, source)
        for recorded_path, digest, size in file_entries(manifest):
            basename = os.path.basename(recorded_path)
            candidates = [os.path.join(os.path.dirname(path), basename),
                          os.path.join(os.path.dirname(bundle_path), basename)]
            local = next((item for item in candidates
                          if os.path.isfile(item)), None)
            if local is None:
                continue
            if isinstance(digest, str) and sha256_file(local) != digest.lower():
                problems.append("recorded_file_sha:" + basename)
            if size is not None:
                try:
                    if int(size) != os.path.getsize(local):
                        problems.append("recorded_file_bytes:" + basename)
                except (TypeError, ValueError):
                    problems.append("recorded_file_bad_bytes:" + basename)

        entries = extract_array_entries(manifest)
        if entries is None:
            problems.append("manifest_no_array_entries")
        else:
            parsed = {}
            malformed = 0
            for raw in entries:
                entry = normalise_array_entry(raw)
                if entry is None or not isinstance(entry["name"], str):
                    malformed += 1
                    continue
                parsed[artifact_array_name(entry["name"])] = entry
            if malformed:
                problems.append("malformed_array_entries=%d" % malformed)
            missing = sorted(set(expected) - set(parsed))
            extra = sorted(set(parsed) - set(expected))
            if missing or extra:
                problems.append("manifest_array_set")
            if isinstance(reference, dict):
                for name, (dtype_name, shape) in expected.items():
                    if name not in reference or name not in parsed:
                        continue
                    entry = parsed[name]
                    array = reference[name]
                    allowed = (shape, (1,), ()) if shape == (1,) else (shape,)
                    if entry["dtype"] is not None and not _is_dtype_name(
                            entry["dtype"], dtype_name):
                        problems.append("manifest_dtype:" + name)
                    if entry["shape"] is not None and tuple(
                            entry["shape"]) not in allowed:
                        problems.append("manifest_shape:" + name)
                    if entry["sha256"] and sha256_c_order(
                            array) != entry["sha256"]:
                        problems.append("manifest_sha256:" + name)

    if isinstance(reference, dict):
        actual = set(reference)
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        if missing or extra:
            problems.append("npz_array_set")
        for name, (dtype_name, shape) in expected.items():
            if name not in reference:
                continue
            array = reference[name]
            allowed = (shape, (1,), ()) if shape == (1,) else (shape,)
            if not _is_dtype_name(str(array.dtype), dtype_name):
                problems.append("npz_dtype:" + name)
            if tuple(array.shape) not in allowed:
                problems.append("npz_shape:" + name)
            if dtype_name.startswith("float") and not np.all(
                    np.isfinite(array)):
                problems.append("npz_nonfinite:" + name)
    problems = sorted(set(problems))
    log.category(label, not problems, 0.0, len(problems),
                 "arrays=%d problems=%s" %
                 (len(reference) if isinstance(reference, dict) else 0,
                  shortlist(problems)))
    return not problems


def check_reference_inputs(log, framework, reference, data):
    if not isinstance(reference, dict) or not isinstance(data, dict):
        log.category("reference.inputs.%s" % framework, False, math.inf, 1,
                     "reference_or_fixture_missing")
        return
    train_ref = data.get("train_transformed_ref")
    labels = data.get("train_labels")
    results = []
    for update in UPDATES:
        index = INPUT_UPDATE_INDEX[update]
        actual = reference.get("input." + update)
        expected = train_ref[index] if isinstance(
            train_ref, np.ndarray) and train_ref.ndim >= 1 else None
        results.append(compare_exact(actual, expected))
        results.append(compare_exact(reference.get("label." + update),
                                     labels))
        eta_actual = reference.get("eta." + update)
        eta_expected = np.asarray([schedule_eta(int(update[1]))])
        results.append(compare_arrays(eta_actual, eta_expected, 0.0, 1e-12))
    log.from_result("reference.inputs.%s" % framework, merge_results(results))


def check_ddp_integrity(log, run, expected, batch, bundle_shapes,
                        framework):
    label = "integrity.ddp.%s.w%d" % (framework, run["world"])
    problems = list(run["errors"])
    checked_arrays = 0
    checked_hashes = 0
    for record in run["ranks"]:
        rank = record["rank"]
        npz = record["npz"]
        meta = record["json"]
        if npz is None:
            problems.append("rank%d:npz" % rank)
            continue
        actual = set(npz)
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        if missing or extra:
            problems.append("rank%d:array_set" % rank)
        for name, (dtype_name, shape) in expected.items():
            if name not in npz:
                continue
            array = npz[name]
            checked_arrays += 1
            if dtype_name == "int":
                if array.dtype.kind not in "iu":
                    problems.append("rank%d:%s:dtype" % (rank, name))
            elif not _is_dtype_name(str(array.dtype), dtype_name):
                problems.append("rank%d:%s:dtype" % (rank, name))
            if shape == ():
                allowed = ((), (1,))
            elif shape == (1,):
                allowed = ((1,), ())
            else:
                allowed = (shape,)
            if tuple(array.shape) not in allowed:
                problems.append("rank%d:%s:shape" % (rank, name))
            if dtype_name.startswith("float") and not np.all(
                    np.isfinite(array)):
                problems.append("rank%d:%s:nonfinite" % (rank, name))
        rank_meta = {
            "rank": _as_int(npz.get("meta.rank")),
            "world_size": _as_int(npz.get("meta.world_size")),
            "device_index": _as_int(npz.get("meta.device_index")),
            "updates": _as_int(npz.get("meta.updates")),
        }
        if rank_meta["rank"] != rank:
            problems.append("rank%d:meta.rank" % rank)
        if rank_meta["world_size"] != run["world"]:
            problems.append("rank%d:meta.world_size" % rank)
        if rank_meta["device_index"] is None:
            problems.append("rank%d:meta.device_index" % rank)
        if rank_meta["updates"] != EXPECTED_UPDATES:
            problems.append("rank%d:meta.updates" % rank)
        if not isinstance(meta, dict):
            problems.append("rank%d:json" % rank)
            continue
        json_rank = _as_int(extract_meta_value(meta, "rank"))
        json_world = _as_int(extract_meta_value(meta, "world_size"))
        json_updates = _as_int(extract_meta_value(meta, "updates"))
        if json_rank is not None and json_rank != rank:
            problems.append("rank%d:json.rank" % rank)
        if json_world is not None and json_world != run["world"]:
            problems.append("rank%d:json.world_size" % rank)
        if json_updates is not None and json_updates != EXPECTED_UPDATES:
            problems.append("rank%d:json.updates" % rank)

        recorded_sha, entries = extract_npz_record(meta)
        if recorded_sha is None:
            problems.append("rank%d:npz_sha256_missing" % rank)
        else:
            checked_hashes += 1
            if sha256_file(record["npz_path"]) != recorded_sha:
                problems.append("rank%d:npz_sha256" % rank)
        if entries is None:
            problems.append("rank%d:npz_array_metadata_missing" % rank)
        else:
            parsed = {}
            for raw in entries:
                entry = normalise_array_entry(raw)
                if entry is None or not isinstance(entry["name"], str):
                    problems.append("rank%d:malformed_array_entry" % rank)
                    continue
                parsed[artifact_array_name(entry["name"])] = entry
            missing_entries = sorted(set(npz) - set(parsed))
            extra_entries = sorted(set(parsed) - set(npz))
            if missing_entries or extra_entries:
                problems.append("rank%d:array_metadata_set" % rank)
            for name, entry in parsed.items():
                if name not in npz:
                    continue
                array = npz[name]
                checked_hashes += 1
                if entry["sha256"] and sha256_c_order(
                        array) != entry["sha256"]:
                    problems.append("rank%d:array_sha256:%s" % (rank, name))

        shard_ids = extract_shard_ids(meta)
        local = batch // run["world"]
        global_expected = np.arange(rank * local, (rank + 1) * local)
        local_expected = np.arange(local)
        for update in ("u1", "u3"):
            ids = shard_ids.get(update)
            if ids is None:
                problems.append("rank%d:shard_ids_%s_missing" %
                                (rank, update))
            elif not (np.array_equal(ids, global_expected) or
                      np.array_equal(ids, local_expected) or
                      np.array_equal(ids, global_expected + 1) or
                      np.array_equal(ids, local_expected + 1)):
                problems.append("rank%d:shard_ids_%s" % (rank, update))
    problems = sorted(set(problems))
    log.category(label, not problems, 0.0, len(problems),
                 "arrays_checked=%d hashes_checked=%d problems=%s" %
                 (checked_arrays, checked_hashes, shortlist(problems)))
    return not problems


# ---------------------------------------------------------------------------
# Shard checks
# ---------------------------------------------------------------------------

def check_schedule_fixture(log, schedule):
    if not isinstance(schedule, dict):
        log.category("shards.schedule_fixture", False, math.inf, 1,
                     "schedule_fixture_missing")
        return
    permutation = schedule.get("epoch0_permutation")
    if not isinstance(permutation, np.ndarray):
        log.category("shards.schedule_fixture", False, math.inf, 1,
                     "epoch0_permutation_missing")
        return
    problems = []
    ids = epoch_sample_ids(permutation, SCHEDULE_BATCH)
    if len(ids) != SCHEDULE_SAMPLES:
        problems.append("samples=%d" % len(ids))
    updates = len(ids) // SCHEDULE_BATCH
    if updates != SCHEDULE_STEPS:
        problems.append("updates=%d" % updates)
    violations = len(problems)
    max_abs = 0.0
    details = list(problems)
    for world in (1, 2, 4):
        sequences = [rank_sequence(permutation, SCHEDULE_BATCH, rank, world)
                     for rank in range(world)]
        expected = sequences[0] if world == 1 else np.stack(sequences)
        key = "rank_ids_R%d" % world
        actual = schedule.get(key)
        result = compare_exact(actual, expected)
        violations += result.violations
        if result.max_abs != 0.0:
            max_abs = max(max_abs, result.max_abs)
        if not result.ok:
            details.append(key)
        if world > 1 and isinstance(actual, np.ndarray):
            chunk = SCHEDULE_BATCH // world
            for batch_index in range(updates):
                global_batch = ids[batch_index * SCHEDULE_BATCH:
                                   (batch_index + 1) * SCHEDULE_BATCH]
                for rank in range(world):
                    start = batch_index * chunk
                    expected_chunk = global_batch[rank * chunk:
                                                  (rank + 1) * chunk]
                    if not np.array_equal(actual[rank, start:start + chunk],
                                          expected_chunk):
                        violations += 1
                        details.append("%s:batch%d:rank%d" %
                                       (key, batch_index, rank))
    log.category("shards.schedule_fixture", violations == 0, max_abs,
                 violations,
                 "samples=%d updates=%d problems=%s" %
                 (len(ids), updates, shortlist(details)))


def check_global_batch_shards(log, run, data):
    label = "shards.global_batch.%s.w%d" % (run["framework"], run["world"])
    if not isinstance(data, dict):
        log.category(label, False, math.inf, 1, "fixture_missing")
        return
    train_ref = data.get("train_transformed_ref")
    labels = data.get("train_labels")
    if not isinstance(train_ref, np.ndarray) or not isinstance(
            labels, np.ndarray):
        log.category(label, False, math.inf, 1, "fixture_arrays_missing")
        return
    global_input = train_ref[0]
    world = run["world"]
    if global_input.shape[0] % world != 0:
        log.category(label, False, math.inf, 1, "batch_not_divisible")
        return
    chunk = global_input.shape[0] // world
    problems = []
    violations = 0
    max_abs = 0.0
    pieces = {"u1": [], "u3": [], "label.u1": [], "label.u3": []}
    for record in run["ranks"]:
        rank = record["rank"]
        npz = record["npz"]
        if npz is None:
            problems.append("rank%d:npz_missing" % rank)
            violations += 1
            continue
        expected_input = global_input[rank * chunk:(rank + 1) * chunk]
        expected_label = labels[rank * chunk:(rank + 1) * chunk]
        for update in ("u1", "u3"):
            actual = npz.get("input." + update)
            result = compare_exact(actual, expected_input)
            violations += result.violations
            max_abs = max(max_abs, result.max_abs)
            if not result.ok:
                problems.append("rank%d:input.%s" % (rank, update))
            label_actual = npz.get("label." + update)
            label_result = compare_exact(label_actual, expected_label)
            violations += label_result.violations
            max_abs = max(max_abs, label_result.max_abs)
            if not label_result.ok:
                problems.append("rank%d:label.%s" % (rank, update))
        u1 = npz.get("input.u1")
        u3 = npz.get("input.u3")
        if u1 is not None and u3 is not None:
            same = compare_exact(u1, u3)
            violations += same.violations
            if not same.ok:
                problems.append("rank%d:input.u1_ne_u3" % rank)
        label1 = npz.get("label.u1")
        label3 = npz.get("label.u3")
        if label1 is not None and label3 is not None:
            same = compare_exact(label1, label3)
            violations += same.violations
            if not same.ok:
                problems.append("rank%d:label.u1_ne_u3" % rank)
        if u1 is not None:
            pieces["u1"].append(u1)
        if u3 is not None:
            pieces["u3"].append(u3)
        if label1 is not None:
            pieces["label.u1"].append(label1)
        if label3 is not None:
            pieces["label.u3"].append(label3)
    for key, expected in (("u1", global_input), ("u3", global_input),
                          ("label.u1", labels), ("label.u3", labels)):
        if len(pieces[key]) == world:
            result = compare_exact(np.concatenate(pieces[key], axis=0),
                                   expected)
            violations += result.violations
            max_abs = max(max_abs, result.max_abs)
            if not result.ok:
                problems.append("concat." + key)
        else:
            problems.append("concat.%s_incomplete" % key)
            violations += 1
    log.category(label, violations == 0, max_abs, violations,
                 "local=%d problems=%s" % (chunk, shortlist(problems)))


# ---------------------------------------------------------------------------
# Rank replica checks
# ---------------------------------------------------------------------------

def check_rank_replicas(log, run):
    label = "replicas.%s.w%d" % (run["framework"], run["world"])
    hashes = {}
    for record in run["ranks"]:
        hashes[record["rank"]] = extract_state_hashes(record["json"])
    states = required_state_names(run["framework"])
    violations = 0
    details = []
    for update in UPDATES:
        for state in states:
            values = []
            missing = False
            for rank in range(run["world"]):
                value = hashes.get(rank, {}).get(update, {}).get(state)
                if value is None:
                    missing = True
                    values.append(None)
                else:
                    values.append(value)
            if missing:
                violations += 1
                details.append("%s.%s_missing" % (update, state))
                continue
            reference = values[0]
            for index, value in enumerate(values[1:], start=1):
                if value != reference:
                    violations += 1
                    details.append("%s.%s:rank%d" %
                                   (update, state, index))

    max_abs = 0.0
    compare_names = ["param.u3.", "m.u3.", "v.u3."]
    if run["framework"] == "flux":
        compare_names.append("beta_power.u3.")
    else:
        compare_names.append("step.u3")
    base = run["ranks"][0]["npz"] if run["ranks"] else None
    for record in run["ranks"][1:]:
        rank = record["rank"]
        other = record["npz"]
        if base is None or other is None:
            violations += 1
            details.append("rank%d:npz_missing" % rank)
            continue
        for prefix in compare_names:
            if prefix.endswith("."):
                names = sorted(name for name in other
                               if name.startswith(prefix))
                base_names = sorted(name for name in base
                                    if name.startswith(prefix))
                if names != base_names:
                    violations += 1
                    details.append("rank%d:%s:name_set" % (rank, prefix))
                for name in names:
                    if name not in base:
                        continue
                    result = compare_exact(base[name], other[name])
                    violations += result.violations
                    max_abs = max(max_abs, result.max_abs)
                    if not result.ok:
                        details.append("rank%d:%s" % (rank, name))
            else:
                result = compare_exact(base.get(prefix), other.get(prefix))
                violations += result.violations
                max_abs = max(max_abs, result.max_abs)
                if not result.ok:
                    details.append("rank%d:%s" % (rank, prefix))
    log.category(label, violations == 0, max_abs, violations,
                 "states=%s problems=%s" %
                 ("/".join(states), shortlist(details)))


def check_updates(log, run):
    label = "updates.%s.w%d" % (run["framework"], run["world"])
    violations = 0
    details = []
    if len(run["ranks"]) != run["world"]:
        violations += 1
        details.append("rank_files=%d_want_%d" %
                       (len(run["ranks"]), run["world"]))
    for record in run["ranks"]:
        npz = record["npz"]
        if npz is None:
            violations += 1
            details.append("rank%d:npz_missing" % record["rank"])
            continue
        value = _as_int(npz.get("meta.updates"))
        if value != EXPECTED_UPDATES:
            violations += 1
            details.append("rank%d:npz=%r" % (record["rank"], value))
        json_value = _as_int(extract_meta_value(record["json"], "updates"))
        if json_value is not None and json_value != EXPECTED_UPDATES:
            violations += 1
            details.append("rank%d:json=%r" % (record["rank"], json_value))
    log.category(label, violations == 0, 0.0, violations,
                 "expected=%d problems=%s" %
                 (EXPECTED_UPDATES, shortlist(details)))


# ---------------------------------------------------------------------------
# Synchronized gradients and oracle
# ---------------------------------------------------------------------------

def record_same_pairs(log, name, pairs, atol, rtol, args=None,
                      device="cpu", kind="", detail=""):
    results = [compare_arrays(a, b, atol, rtol) for a, b in pairs]
    log.from_result(name, merge_results(results), detail=detail)


def check_flux_syncgrads(log, run, reference, bundle_shapes, atol, rtol,
                         args=None, device="cpu"):
    world = run["world"]
    sync = {update: {} for update in UPDATES}
    violations = 0
    details = []
    for name in bundle_shapes:
        for update in UPDATES:
            arrays = []
            for record in run["ranks"]:
                npz = record["npz"]
                if npz is None:
                    arrays = []
                    break
                value = npz.get("localgrad.%s.%s" % (update, name))
                if value is None:
                    arrays = []
                    break
                arrays.append(np.asarray(value, dtype=np.float64))
            if len(arrays) != world:
                violations += 1
                details.append("%s:missing:%s" % (update, name))
                continue
            shape = bundle_shapes[name]
            if any(array.shape != shape for array in arrays):
                violations += 1
                details.append("%s:shape:%s" % (update, name))
                continue
            total = np.zeros(shape, dtype=np.float64)
            for array in arrays:
                total += array
            sync[update][name] = total / float(world)

    for update in ("u1", "u3"):
        cat = "same.flux.w%d.grads.%s" % (world, update)
        detail = "synchronized=sum_r(localgrad)/%d" % world
        if isinstance(reference, dict) and all(
                ("grad.%s.%s" % (update, name)) in reference
                for name in bundle_shapes):
            pairs = [(sync[update].get(name),
                      reference.get("grad.%s.%s" % (update, name)))
                     for name in bundle_shapes]
            record_same_pairs(log, cat, pairs, atol, rtol, args=args,
                              device=device, kind="grads.%s" % update,
                              detail=detail)
        else:
            results = []
            if isinstance(reference, dict):
                for name in bundle_shapes:
                    expected = reference.get("grad.%s.%s" % (update, name))
                    if expected is None:
                        results.append(CompareResult(
                            False, math.inf, 1, 0,
                            "missing:grad.%s.%s" % (update, name)))
                    else:
                        results.append(compare_arrays(
                            sync[update].get(name), expected, atol, rtol))
            else:
                results.append(CompareResult(False, math.inf, 1, 0,
                                             "reference_missing"))
            log.from_result(cat, merge_results(results), detail=detail)
    if isinstance(reference, dict):
        u2_expected = any(("grad.u2." + name) in reference
                          for name in bundle_shapes)
        if u2_expected:
            pairs = [(sync["u2"].get(name), reference.get("grad.u2." + name))
                     for name in bundle_shapes]
            record_same_pairs(
                log, "same.flux.w%d.grads.u2" % world, pairs, atol, rtol,
                args=args, device=device, kind="grads.u2",
                detail="synchronized=sum_r(localgrad)/%d" % world)
        else:
            log.diagnostic(
                "flux.w%d.grads.u2" % world,
                "no grad.u2 in the 337-array reference; u2 is checked by "
                "the exactly-once oracle and cross-framework categories")
    if violations:
        log.diagnostic("flux.w%d.syncgrad_missing" % world,
                       shortlist(details))
    return sync


def check_pytorch_syncgrads(log, run, reference, bundle_shapes, atol, rtol,
                            args=None, device="cpu"):
    world = run["world"]
    sync = {update: {} for update in UPDATES}
    replica_violations = 0
    replica_details = []
    max_abs = 0.0
    base = run["ranks"][0]["npz"] if run["ranks"] else None
    if base is None:
        log.category("py.syncgrad_replicas.w%d" % world, False, math.inf, 1,
                     "rank0_npz_missing")
    else:
        for record in run["ranks"][1:]:
            other = record["npz"]
            if other is None:
                replica_violations += 1
                replica_details.append("rank%d:npz_missing" %
                                       record["rank"])
                continue
            for update in UPDATES:
                for name in bundle_shapes:
                    key = "syncgrad.%s.%s" % (update, name)
                    result = compare_exact(base.get(key), other.get(key))
                    replica_violations += result.violations
                    max_abs = max(max_abs, result.max_abs)
                    if not result.ok:
                        replica_details.append("rank%d:%s" %
                                               (record["rank"], key))
        log.category("py.syncgrad_replicas.w%d" % world,
                     replica_violations == 0, max_abs, replica_violations,
                     "world=%d problems=%s" %
                     (world, shortlist(replica_details)))

    if base is None:
        return sync
    missing = 0
    for update in UPDATES:
        for name in bundle_shapes:
            key = "syncgrad.%s.%s" % (update, name)
            value = base.get(key)
            if value is None:
                missing += 1
            else:
                sync[update][name] = np.asarray(value, dtype=np.float64)
    for update in ("u1", "u3"):
        cat = "same.pytorch.w%d.grads.%s" % (world, update)
        detail = "rank0 native DDP syncgrad"
        if isinstance(reference, dict) and all(
                ("grad.%s.%s" % (update, name)) in reference
                for name in bundle_shapes):
            pairs = [(sync[update].get(name),
                      reference.get("grad.%s.%s" % (update, name)))
                     for name in bundle_shapes]
            record_same_pairs(log, cat, pairs, atol, rtol, args=args,
                              device=device, kind="grads.%s" % update,
                              detail=detail)
        else:
            results = []
            if isinstance(reference, dict):
                for name in bundle_shapes:
                    expected = reference.get("grad.%s.%s" % (update, name))
                    if expected is None:
                        results.append(CompareResult(
                            False, math.inf, 1, 0,
                            "missing:grad.%s.%s" % (update, name)))
                    else:
                        results.append(compare_arrays(
                            sync[update].get(name), expected, atol, rtol))
            else:
                results.append(CompareResult(False, math.inf, 1, 0,
                                             "reference_missing"))
            log.from_result(cat, merge_results(results), detail=detail)
    if isinstance(reference, dict):
        u2_expected = any(("grad.u2." + name) in reference
                          for name in bundle_shapes)
        if u2_expected:
            pairs = [(sync["u2"].get(name), reference.get("grad.u2." + name))
                     for name in bundle_shapes]
            record_same_pairs(
                log, "same.pytorch.w%d.grads.u2" % world, pairs, atol, rtol,
                args=args, device=device, kind="grads.u2",
                detail="rank0 native DDP syncgrad")
        else:
            log.diagnostic(
                "pytorch.w%d.grads.u2" % world,
                "no grad.u2 in the 337-array reference; u2 is checked by "
                "the exactly-once oracle and cross-framework categories")
    if missing:
        log.diagnostic("pytorch.w%d.syncgrad_missing" % world, missing)
    return sync


def check_oracle(log, framework, world, sync, observed, bundle,
                 bundle_shapes, etas, atol, rtol):
    x0 = {name: np.asarray(bundle[name], dtype=np.float64)
          for name in bundle_shapes}
    try:
        final = adamw_oracle(x0, sync, etas, list(bundle_shapes))
    except Exception as exc:
        log.category("oracle.%s.w%d" % (framework, world), False, math.inf, 1,
                     "%s:%s" % (type(exc).__name__, exc))
        return
    results = []
    for state in ("param", "m", "v"):
        for name in bundle_shapes:
            actual = observed.get("%s.u3.%s" % (state, name))
            expected = final["u3"][state][name]
            results.append(compare_arrays(actual, expected, atol, rtol))
    log.from_result("oracle.%s.w%d" % (framework, world),
                    merge_results(results),
                    detail="float64_decoupled_adamw_updates=3")


# ---------------------------------------------------------------------------
# Same-framework comparisons
# ---------------------------------------------------------------------------

def check_same_framework(log, run, reference, data, sync, bundle_shapes,
                         atol, rtol, args=None, device="cpu"):
    framework = run["framework"]
    world = run["world"]
    base = run["ranks"][0]["npz"] if run["ranks"] else None
    if base is None or not isinstance(reference, dict) or not isinstance(
            data, dict):
        for state in ("param", "m", "v"):
            log.category("same.%s.w%d.%s.u3" %
                         (framework, world, state), False, math.inf, 1,
                         "artifact_missing")
        for suffix in ("eta", "loss", "logits.u1", "logits.u3"):
            log.category("same.%s.w%d.%s" % (framework, world, suffix),
                         False, math.inf, 1, "artifact_missing")
        if framework == "flux":
            log.category("same.flux.w%d.beta_power.u3" % world, False,
                         math.inf, 1, "artifact_missing")
        else:
            log.category("same.pytorch.w%d.step.u3" % world, False,
                         math.inf, 1, "artifact_missing")
        return

    for state in ("param", "m", "v"):
        pairs = [(base.get("%s.u3.%s" % (state, name)),
                  reference.get("%s.u3.%s" % (state, name)))
                 for name in bundle_shapes]
        record_same_pairs(
            log, "same.%s.w%d.%s.u3" % (framework, world, state),
            pairs, atol, rtol, args=args, device=device,
            kind="%s.u3" % state, detail="arrays=%d" % len(pairs))

    if framework == "flux":
        step = _as_int(reference.get("step.u3"))
        if step is None:
            step = 3
        expected_beta = beta_power_expected(BETAS, int(step))
        results = [compare_arrays(base.get("beta_power.u3." + name),
                                  expected_beta, atol, rtol)
                   for name in bundle_shapes]
        log.from_result("same.flux.w%d.beta_power.u3" % world,
                        merge_results(results),
                        detail="expected=beta^(%d+1)" % int(step))
    else:
        reference_step = _as_int(reference.get("step.u3"))
        actual_step = _as_int(base.get("step.u3"))
        if actual_step is None or reference_step is None:
            log.category("same.pytorch.w%d.step.u3" % world, False,
                         math.inf, 1, "step_missing")
        else:
            log.category("same.pytorch.w%d.step.u3" % world,
                         actual_step == reference_step, 0.0,
                         0 if actual_step == reference_step else 1,
                         "actual=%d reference=%d" %
                         (actual_step, reference_step))

    eta_results = []
    for update in UPDATES:
        expected = reference.get("eta." + update)
        actual = base.get("eta." + update)
        eta_results.append(compare_arrays(actual, expected, atol, rtol))
        if expected is not None:
            schedule_expected = np.asarray([schedule_eta(int(update[1]))])
            eta_results.append(compare_arrays(actual, schedule_expected,
                                              0.0, 1e-12))
    log.from_result("same.%s.w%d.eta" % (framework, world),
                    merge_results(eta_results))

    loss_results = []
    for update in UPDATES:
        local_losses = []
        for record in run["ranks"]:
            value = (record["npz"].get("loss." + update)
                     if record["npz"] is not None else None)
            if value is None:
                local_losses = []
                break
            local_losses.append(np.asarray(value, dtype=np.float64))
        if len(local_losses) != world:
            loss_results.append(CompareResult(False, math.inf, 1, 0,
                                              "missing:" + update))
        else:
            global_loss = sum(float(np.asarray(item).reshape(-1)[0])
                              for item in local_losses) / float(world)
            loss_results.append(compare_arrays(
                np.asarray([global_loss]),
                reference.get("loss." + update), atol, rtol))
    log.from_result("same.%s.w%d.loss" % (framework, world),
                    merge_results(loss_results),
                    detail="mean_of_rank_local_losses")

    for update in ("u1", "u3"):
        pieces = []
        for record in run["ranks"]:
            value = (record["npz"].get("logits." + update)
                     if record["npz"] is not None else None)
            if value is None:
                pieces = []
                break
            pieces.append(np.asarray(value))
        if len(pieces) != world:
            log.category("same.%s.w%d.logits.%s" %
                         (framework, world, update), False, math.inf, 1,
                         "missing_logits")
        else:
            concatenated = np.concatenate(pieces, axis=0)
            record_same_pairs(
                log, "same.%s.w%d.logits.%s" % (framework, world, update),
                [(concatenated, reference.get("logits." + update))],
                atol, rtol, args=args, device=device,
                kind="logits.%s" % update,
                detail="rank_order_concatenation")


# ---------------------------------------------------------------------------
# Cross-framework comparisons
# ---------------------------------------------------------------------------

def _cross_rule(args, kind: str):
    base = (float(args.cross_atol), float(args.cross_rtol))
    if FP32_ABSOLUTE_LIMIT is not None:
        return base, base, False
    override = {
        "grads.u1": getattr(args, "grads_u1_atol", None),
        "grads.u2": getattr(args, "grads_u2_atol", None),
        "grads.u3": getattr(args, "grads_u3_atol", None),
        "m.u3": getattr(args, "m_u3_atol", None),
    }.get(kind)
    if override is None:
        return base, base, False
    return base, (float(override), float(args.cross_rtol)), True


def record_cross(log, name, actual, expected, args, kind, detail=""):
    base, effective, has_override = _cross_rule(args, kind)
    # These categories originate from FP32 tensors, including values promoted
    # to Float64 for offline averaging/concatenation. Eta is truly Float64.
    fp32 = kind != "eta"
    base_result = compare_arrays(actual, expected, base[0], base[1], fp32=fp32)
    effective_result = compare_arrays(actual, expected,
                                      effective[0], effective[1], fp32=fp32)
    if has_override:
        status = "PASS" if base_result.ok else "FAIL"
        print("PHASE4_BASE_RULE %s %s max_abs=%s violations=%d" %
              (name, status, fmt_float(base_result.max_abs),
               base_result.violations), flush=True)
        detail = (detail + " rule=override atol=%s rtol=%s base_rule=%s" %
                  (fmt_float(effective[0]), fmt_float(effective[1]),
                   status)).strip()
    log.from_result(name, effective_result, detail=detail)


def combined_logits(run, update):
    pieces = []
    for record in run["ranks"]:
        value = (record["npz"].get("logits." + update)
                 if record["npz"] is not None else None)
        if value is None:
            return None
        pieces.append(np.asarray(value))
    if len(pieces) != run["world"]:
        return None
    return np.concatenate(pieces, axis=0)


def averaged_loss(run, update):
    values = []
    for record in run["ranks"]:
        value = (record["npz"].get("loss." + update)
                 if record["npz"] is not None else None)
        if value is None:
            return None
        values.append(float(np.asarray(value).reshape(-1)[0]))
    if len(values) != run["world"]:
        return None
    return np.asarray([sum(values) / float(run["world"])])


def first_npz(run):
    return run["ranks"][0]["npz"] if run["ranks"] else None


def check_cross_framework(log, flux_run, torch_run, flux_sync, torch_sync,
                          pytorch_reference, bundle_shapes, args):
    world = flux_run["world"]
    flux_npz = first_npz(flux_run)
    torch_npz = first_npz(torch_run)
    if flux_npz is None or torch_npz is None:
        for suffix in ("grads.u1", "grads.u2", "grads.u3", "params.u3",
                       "m.u3", "v.u3", "logits.u1", "logits.u3", "loss",
                       "eta"):
            log.category("cross.ddp.w%d.%s" % (world, suffix), False,
                         math.inf, 1, "artifact_missing")
        return

    for update in UPDATES:
        missing = [name for name in bundle_shapes
                   if flux_sync[update].get(name) is None or
                   torch_sync[update].get(name) is None]
        if missing:
            log.category("cross.ddp.w%d.grads.%s" % (world, update), False,
                         math.inf, len(missing),
                         "missing_leaves=%s" % shortlist(missing))
            continue
        actual = np.concatenate([
            np.asarray(flux_sync[update][name], dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        expected = np.concatenate([
            np.asarray(torch_sync[update][name], dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        record_cross(log, "cross.ddp.w%d.grads.%s" % (world, update),
                     actual, expected, args, "grads." + update,
                     detail="flux_offline_vs_pytorch_ddp")

    for state in ("param", "m", "v"):
        missing = [
            name for name in bundle_shapes
            if flux_npz.get("%s.u3.%s" % (state, name)) is None or
            torch_npz.get("%s.u3.%s" % (state, name)) is None]
        if missing:
            log.category("cross.ddp.w%d.%s.u3" % (world, state), False,
                         math.inf, len(missing),
                         "missing_leaves=%s" % shortlist(missing))
            continue
        actual = np.concatenate([
            np.asarray(flux_npz["%s.u3.%s" % (state, name)],
                       dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        expected = np.concatenate([
            np.asarray(torch_npz["%s.u3.%s" % (state, name)],
                       dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        record_cross(log, "cross.ddp.w%d.%s.u3" % (world, state),
                     actual, expected, args, "%s.u3" % state,
                     detail="flux_vs_pytorch_ddp")

    for update in ("u1", "u3"):
        actual = combined_logits(flux_run, update)
        expected = combined_logits(torch_run, update)
        record_cross(log, "cross.ddp.w%d.logits.%s" % (world, update),
                     actual, expected, args, "logits", detail="combined")
    for update in UPDATES:
        actual = averaged_loss(flux_run, update)
        expected = averaged_loss(torch_run, update)
        record_cross(log, "cross.ddp.w%d.loss.%s" % (world, update),
                     actual, expected, args, "loss", detail="rank_average")
    for update in UPDATES:
        record_cross(log, "cross.ddp.w%d.eta.%s" % (world, update),
                     flux_npz.get("eta." + update),
                     torch_npz.get("eta." + update), args, "eta")

    if not isinstance(pytorch_reference, dict):
        for suffix in ("grads.u1", "grads.u3", "params.u3", "m.u3", "v.u3",
                       "logits.u1", "logits.u3", "loss", "eta"):
            log.category("cross.reference.flux.w%d.%s" % (world, suffix),
                         False, math.inf, 1, "pytorch_reference_missing")
        return
    for update in ("u1", "u3"):
        missing = [name for name in bundle_shapes
                   if flux_sync[update].get(name) is None or
                   pytorch_reference.get("grad.%s.%s" % (update, name))
                   is None]
        if missing:
            log.category("cross.reference.flux.w%d.grads.%s" %
                         (world, update), False, math.inf, len(missing),
                         "missing_leaves=%s" % shortlist(missing))
            continue
        actual = np.concatenate([
            np.asarray(flux_sync[update][name], dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        expected = np.concatenate([
            np.asarray(pytorch_reference["grad.%s.%s" % (update, name)],
                       dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        record_cross(log, "cross.reference.flux.w%d.grads.%s" %
                     (world, update), actual, expected, args,
                     "grads." + update,
                     detail="flux_ddp_vs_pytorch_reference")
    for state in ("param", "m", "v"):
        missing = [
            name for name in bundle_shapes
            if flux_npz.get("%s.u3.%s" % (state, name)) is None or
            pytorch_reference.get("%s.u3.%s" % (state, name)) is None]
        if missing:
            log.category("cross.reference.flux.w%d.%s.u3" %
                         (world, state), False, math.inf, len(missing),
                         "missing_leaves=%s" % shortlist(missing))
            continue
        actual = np.concatenate([
            np.asarray(flux_npz["%s.u3.%s" % (state, name)],
                       dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        expected = np.concatenate([
            np.asarray(pytorch_reference["%s.u3.%s" % (state, name)],
                       dtype=np.float64).reshape(-1)
            for name in bundle_shapes])
        record_cross(log, "cross.reference.flux.w%d.%s.u3" %
                     (world, state), actual, expected, args,
                     "%s.u3" % state,
                     detail="flux_ddp_vs_pytorch_reference")
    for update in ("u1", "u3"):
        record_cross(log, "cross.reference.flux.w%d.logits.%s" %
                     (world, update), combined_logits(flux_run, update),
                     pytorch_reference.get("logits." + update), args,
                     "logits", detail="combined")
    for update in UPDATES:
        record_cross(log, "cross.reference.flux.w%d.loss.%s" %
                     (world, update), averaged_loss(flux_run, update),
                     pytorch_reference.get("loss." + update), args, "loss",
                     detail="rank_average")
    for update in UPDATES:
        record_cross(log, "cross.reference.flux.w%d.eta.%s" %
                     (world, update), flux_npz.get("eta." + update),
                     pytorch_reference.get("eta." + update), args, "eta")


# ---------------------------------------------------------------------------
# Collective protocol and DDP configuration
# ---------------------------------------------------------------------------

def _find_ddp_config(value, depth=0):
    if depth > 6:
        return None
    if isinstance(value, dict):
        keys = {_norm_key(key) for key in value}
        if "findunusedparameters" in keys or "staticgraph" in keys:
            return value
        for item in value.values():
            if isinstance(item, (dict, list)):
                found = _find_ddp_config(item, depth + 1)
                if found:
                    return found
    elif isinstance(value, list):
        for item in value:
            found = _find_ddp_config(item, depth + 1)
            if found:
                return found
    return None


def _config_flag(config, name):
    if not isinstance(config, dict):
        return None
    value = _first_value(config, (name,), _MISSING)
    if value is _MISSING:
        normalised = {_norm_key(key): item for key, item in config.items()}
        value = normalised.get(_norm_key(name), _MISSING)
    if value is _MISSING:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def check_flux_protocol(log, run, bundle_names, bundle_shapes, device,
                        protocol=DEFAULT_FLUX_COLLECTIVE_PROTOCOL):
    if protocol not in FLUX_COLLECTIVE_PROTOCOLS:
        raise ValueError("unknown Flux collective protocol %r" % (protocol,))
    world = run["world"]
    label = "protocol.flux.w%d" % world
    rank0_json = run["ranks"][0]["json"] if run["ranks"] else None
    traces = extract_traces(run["summary"], rank0_json)
    if not traces:
        log.category(label, False, 0.0, 1,
                     "collective_trace_missing traces=0")
        return
    violations = 0
    details = []
    signatures = {}
    expected_count = len(bundle_names) + (
        2 if protocol == "guarded" else 0)
    for update in UPDATES:
        entries = traces.get(update)
        if entries is None:
            violations += 1
            details.append(update + ":missing")
            continue
        trace_violations, trace_details, signature = validate_flux_trace(
            entries, bundle_names, bundle_shapes, device, protocol=protocol)
        violations += trace_violations
        if trace_violations:
            details.append("%s:%s" % (update, trace_details))
        if len(entries) != expected_count:
            details.append("%s:count=%d" % (update, len(entries)))
        signatures[update] = signature
    if len(signatures) == 3:
        base = signatures.get("u1")
        for update in ("u2", "u3"):
            if signatures.get(update) != base:
                violations += 1
                details.append(update + ":signature_differs_from_u1")
    summary_digests = _trace_digest_mapping(run["summary"]) or {}
    rank0_digests = _trace_digest_mapping(rank0_json) or {}
    reference_digests = summary_digests or rank0_digests
    for update, digest in reference_digests.items():
        if digest.get("count") is not None and digest["count"] != expected_count:
            violations += 1
            details.append("%s:digest_count=%d" % (update, digest["count"]))
    rank_traces = 1
    for record in run["ranks"][1:]:
        rank_digests = _trace_digest_mapping(record["json"]) or {}
        if rank_digests:
            rank_traces += 1
            for update in UPDATES:
                expected_digest = reference_digests.get(update)
                actual_digest = rank_digests.get(update)
                if expected_digest is None or actual_digest is None:
                    violations += 1
                    details.append("rank%d:%s:digest_missing" %
                                   (record["rank"], update))
                    continue
                if actual_digest.get("count") != expected_digest.get("count"):
                    violations += 1
                    details.append("rank%d:%s:digest_count" %
                                   (record["rank"], update))
                if expected_digest.get("sha256") is not None and (
                        actual_digest.get("sha256") !=
                        expected_digest.get("sha256")):
                    violations += 1
                    details.append("rank%d:%s:digest_sha256" %
                                   (record["rank"], update))
        other = extract_traces(record["json"])
        if not other:
            continue
        if not rank_digests:
            rank_traces += 1
        for update in UPDATES:
            other_signature = other.get(update)
            if other_signature is None:
                violations += 1
                details.append("rank%d:%s:missing" % (record["rank"], update))
            elif other_signature != signatures.get(update):
                violations += 1
                details.append("rank%d:%s:signature" %
                               (record["rank"], update))
    log.category(label, violations == 0, 0.0, violations,
                 "updates=%d entries_per_update=%d rank_traces=%d "
                 "problems=%s" %
                 (len(signatures), expected_count, rank_traces,
                  shortlist(details)))


def check_pytorch_protocol(log, run, device):
    world = run["world"]
    label = "protocol.pytorch.w%d" % world
    rank0_json = run["ranks"][0]["json"] if run["ranks"] else None
    config = _find_ddp_config(run["summary"]) or _find_ddp_config(rank0_json)
    violations = 0
    details = []
    if config is None:
        violations += 1
        details.append("ddp_config_missing")
    else:
        find_unused = _config_flag(config, "find_unused_parameters")
        static_graph = _config_flag(config, "static_graph")
        if find_unused is not False:
            violations += 1
            details.append("find_unused_parameters=%r" % (find_unused,))
        if static_graph is not False:
            violations += 1
            details.append("static_graph=%r" % (static_graph,))
    backend = extract_meta_value(run["summary"], "backend")
    if backend is _MISSING:
        backend = extract_meta_value(rank0_json, "backend")
    backend_text = str(backend).lower() if backend is not _MISSING else None
    expected_backend = "nccl" if device == "cuda" else "gloo"
    if backend_text is None or expected_backend not in backend_text:
        violations += 1
        details.append("backend=%r_want=%s" % (backend_text,
                                               expected_backend))
    recorded_world = extract_meta_value(run["summary"], "world_size")
    if recorded_world is _MISSING:
        recorded_world = extract_meta_value(rank0_json, "world_size")
    if recorded_world is _MISSING and isinstance(config, dict):
        recorded_world = _first_value(
            config, ("world_size", "world", "nranks"), _MISSING)
    if _as_int(recorded_world) != world:
        violations += 1
        details.append("world_size=%r" % (recorded_world,))
    log.category(label, violations == 0, 0.0, violations,
                 "find_unused_parameters=false static_graph=false "
                 "backend=%s problems=%s" %
                 (backend_text, shortlist(details)))


# ---------------------------------------------------------------------------
# Split-control diagnostics
# ---------------------------------------------------------------------------

_SPLIT_FULL = ("full", "global", "full_batch", "batch_full", "whole")
_SPLIT_HALF1 = ("half1", "half_1", "half-1", "first_half", "half_a",
                "halfa", "h1")
_SPLIT_HALF2 = ("half2", "half_2", "half-2", "second_half", "half_b",
                "halfb", "h2")


def _split_role(key):
    text = key.lower()
    for role, aliases in (("full", _SPLIT_FULL),
                          ("half1", _SPLIT_HALF1),
                          ("half2", _SPLIT_HALF2)):
        for alias in aliases:
            pattern = r"(^|[._\-])" + re.escape(alias) + r"([._\-]|$)"
            if re.search(pattern, text):
                return role
    return None


def find_split_array(npz, update, name, role):
    aliases = {"full": _SPLIT_FULL, "half1": _SPLIT_HALF1,
               "half2": _SPLIT_HALF2}[role]
    explicit = []
    for alias in aliases:
        explicit.extend([
            "grad.%s.%s.%s" % (update, alias, name),
            "grad.%s.%s.%s" % (alias, update, name),
            "%s.%s.%s" % (update, alias, name),
            "%s.grad.%s.%s" % (update, alias, name),
            "grad.%s.%s" % (alias, name),
            "%s.grad.%s" % (alias, name),
            "%s.%s" % (alias, name),
            "grad.%s.%s.%s" % (update, name, alias),
        ])
    for key in explicit:
        if key in npz:
            return npz[key]
    normalised_update = _norm_key(update)
    normalised_name = _norm_key(name)
    for key, value in npz.items():
        if _split_role(key) != role:
            continue
        normalised_key = _norm_key(key)
        if normalised_update in normalised_key and normalised_name in normalised_key:
            return value
    return None


def _split_diff(full, half1, half2):
    full64 = np.asarray(full, dtype=np.float64)
    h1 = np.asarray(half1, dtype=np.float64)
    h2 = np.asarray(half2, dtype=np.float64)
    if full64.shape != h1.shape or full64.shape != h2.shape:
        return None
    half_mean = 0.5 * (h1 + h2)
    return {
        "max_abs_full_vs_halfmean": float(np.max(np.abs(full64 - half_mean))),
        "max_abs_full_vs_half1": float(np.max(np.abs(full64 - h1))),
        "max_abs_full_vs_half2": float(np.max(np.abs(full64 - h2))),
    }


def check_split_control(log, split_paths, bundle_shapes):
    report = {}
    for framework, path in split_paths.items():
        entry = {"path": path, "status": "not_available", "updates": {}}
        if not path or not os.path.isfile(path):
            print("PHASE4_SPLIT framework=%s status=not_available path=%s" %
                  (framework, path), flush=True)
            report[framework] = entry
            continue
        try:
            npz = load_npz(path)
        except Exception as exc:
            entry["status"] = "load_failed"
            entry["error"] = "%s:%s" % (type(exc).__name__, exc)
            print("PHASE4_SPLIT framework=%s status=load_failed error=%s" %
                  (framework, entry["error"]), flush=True)
            report[framework] = entry
            continue
        json_path = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(json_path):
            try:
                entry["recorded"] = read_json(json_path)
            except Exception as exc:
                entry["recorded_error"] = "%s:%s" % (type(exc).__name__, exc)
        entry["status"] = "ok"
        found_any = False
        for update in UPDATES:
            max_values = {"max_abs_full_vs_halfmean": [],
                          "max_abs_full_vs_half1": [],
                          "max_abs_full_vs_half2": []}
            leaves = 0
            for name in bundle_shapes:
                full = find_split_array(npz, update, name, "full")
                half1 = find_split_array(npz, update, name, "half1")
                half2 = find_split_array(npz, update, name, "half2")
                if full is None or half1 is None or half2 is None:
                    continue
                differences = _split_diff(full, half1, half2)
                if differences is None:
                    continue
                leaves += 1
                for key, value in differences.items():
                    max_values[key].append(value)
            if leaves == 0:
                continue
            found_any = True
            summary = {"leaves": leaves}
            for key, values in max_values.items():
                summary[key] = max(values) if values else 0.0
            entry["updates"][update] = summary
            print("PHASE4_SPLIT framework=%s update=%s leaves=%d "
                  "max_abs_full_vs_halfmean=%s max_abs_full_vs_half1=%s "
                  "max_abs_full_vs_half2=%s" %
                  (framework, update, leaves,
                   fmt_float(summary["max_abs_full_vs_halfmean"]),
                   fmt_float(summary["max_abs_full_vs_half1"]),
                   fmt_float(summary["max_abs_full_vs_half2"])), flush=True)
        if not found_any:
            entry["status"] = "no_gradient_arrays"
            print("PHASE4_SPLIT framework=%s status=no_gradient_arrays path=%s"
                  % (framework, path), flush=True)
        report[framework] = entry
    log.diagnostic("split_control", report)
    return report


def discover_split_paths(ddp_root, frameworks, device, explicit):
    result = {}
    if explicit:
        flattened = []
        for item in explicit:
            if isinstance(item, (list, tuple)):
                flattened.extend(item)
            else:
                flattened.append(item)
        for item in flattened:
            for token in str(item).split(","):
                token = token.strip()
                if not token:
                    continue
                if "=" in token:
                    key, path = token.split("=", 1)
                    key = key.strip().lower()
                    if key in ("torch", "pytorch"):
                        key = "pytorch"
                    path = os.path.abspath(path.strip())
                    if os.path.isdir(path):
                        path = os.path.join(path, "split_control.npz")
                    result[key] = path
                else:
                    path = os.path.abspath(token)
                    if os.path.isdir(path):
                        path = os.path.join(path, "split_control.npz")
                    lowered = path.lower()
                    if "flux" in lowered:
                        result["flux"] = path
                    elif "pytorch" in lowered or "torch" in lowered:
                        result["pytorch"] = path
                    else:
                        for framework in frameworks:
                            result.setdefault(framework, path)
        return result
    candidates = [
        os.path.join(ddp_root, "split"),
        os.path.join(ddp_root, "..", "split"),
    ]
    for framework in frameworks:
        for base in candidates:
            path = os.path.abspath(os.path.join(
                base, framework, device, "split_control.npz"))
            if os.path.isfile(path):
                result[framework] = path
                break
    return result


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_reports(out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "phase4_report.json")
    check_json_path = os.path.join(out_dir, "phase4_check_report.json")
    text_path = os.path.join(out_dir, "phase4_report.txt")
    summary_md_path = os.path.join(out_dir, "phase4_check_summary.md")
    for jpath in (json_path, check_json_path):
        with open(jpath, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=False)
            handle.write("\n")
    lines = ["PHASE4_REPORT schema=%s created=%s" %
             (report.get("schema"), report.get("created"))]
    for category in report.get("categories", []):
        lines.append("PHASE4_CATEGORY %s %s max_abs=%s violations=%d%s" %
                     (category["name"], category["status"],
                      fmt_float(category["max_abs"]),
                      category["violations"],
                      (" " + category["detail"])
                      if category.get("detail") else ""))
    for name, value in sorted(report.get("diagnostics", {}).items()):
        lines.append("PHASE4_DIAG %s %s" % (name, json.dumps(value)))
    split = report.get("split_control", {})
    for framework, entry in split.items():
        lines.append("PHASE4_SPLIT %s %s" % (framework, json.dumps(entry)))
    final = "PHASE4_CHECK PASS" if report.get("pass") else (
        "PHASE4_CHECK FAIL " + " ".join(report.get("failed", [])))
    lines.append(final)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    md_lines = [
        "# Phase 4 Distributed-Correctness Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Schema | `%s` |" % report.get("schema"),
        "| Created | `%s` |" % report.get("created"),
        "| Finished | `%s` |" % report.get("finished"),
        "| Device | `%s` |" % report.get("device"),
        "| World sizes | `%s` |" % ",".join(
            str(w) for w in report.get("world_sizes", [])),
        "| Frameworks | `%s` |" % ",".join(report.get("frameworks", [])),
        "| Categories | `%d` |" % len(report.get("categories", [])),
        "| Verdict | **%s** |" % ("PASS" if report.get("pass") else "FAIL"),
        "",
        "## Categories",
        "",
        "| Category | Status | max_abs | Violations | Detail |",
        "|---|---|---:|---:|---|",
    ]
    for category in report.get("categories", []):
        md_lines.append("| `%s` | %s | `%s` | %d | %s |" % (
            category["name"],
            category["status"],
            fmt_float(category["max_abs"]),
            category["violations"],
            category.get("detail", ""),
        ))
    md_lines.extend([
        "",
        "## Verdict",
        "",
        "```text",
        final,
        "```",
    ])
    with open(summary_md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")
    return json_path, text_path


# ---------------------------------------------------------------------------
# Self-tests for the pure checker functions
# ---------------------------------------------------------------------------

def _selftest_check(results, name, condition, detail=""):
    results.append((name, bool(condition), detail))


def run_self_tests():
    global FP32_ABSOLUTE_LIMIT
    results = []
    saved_limit = FP32_ABSOLUTE_LIMIT
    FP32_ABSOLUTE_LIMIT = 1e-3
    for name, a, b, want in (
            ("below", [0.0], [0.000999], True),
            ("boundary", [0.0], [0.001], False),
            ("above", [0.0], [0.001001], False),
            ("no_relative_escape", [100.0], [100.002], False),
            ("nonfinite", [0.0], [float("nan")], False)):
        _selftest_check(results, "absolute." + name,
                        compare_arrays(np.asarray(a, dtype=np.float32), b, 2e-3, 1e-4).ok == want)
    _selftest_check(results, "absolute.exact_preserved",
                    not compare_arrays([0.0], [1e-8], 0, 0).ok and
                    not compare_exact([0.0], [1e-8]).ok)
    _selftest_check(results, "absolute.float64_preserved",
                    not compare_arrays([0.0], [1e-5], 1e-6, 0).ok)
    _selftest_check(results, "absolute.promoted_fp32",
                    compare_arrays([0.0], [0.0009], 1e-6, 0, fp32=True).ok and
                    not compare_arrays([0.0], [0.001], 1e-6, 0, fp32=True).ok)
    FP32_ABSOLUTE_LIMIT = saved_limit

    # Tolerance rule and shape/nonfinite handling.
    a = np.asarray([0.0, 1.0, -2.0])
    result = compare_arrays(a, a.copy(), 2e-6, 1e-5)
    _selftest_check(results, "compare.equal", result.ok and
                    result.violations == 0 and result.max_abs == 0.0)
    result = compare_arrays(a, a + 1e-6, 2e-6, 1e-5)
    _selftest_check(results, "compare.within", result.ok)
    result = compare_arrays(a, a + 1e-2, 2e-6, 1e-5)
    _selftest_check(results, "compare.violation",
                    (not result.ok) and result.violations == 3 and
                    result.max_abs > 0.0)
    result = compare_arrays(a, a[:2], 2e-6, 1e-5)
    _selftest_check(results, "compare.shape", not result.ok)
    bad = a.copy()
    bad[0] = np.nan
    result = compare_arrays(bad, a, 2e-6, 1e-5)
    _selftest_check(results, "compare.nonfinite",
                    (not result.ok) and math.isinf(result.max_abs))
    result = compare_exact(np.asarray([1, 2, 3]), np.asarray([1, 2, 4]))
    _selftest_check(results, "compare.integer_exact",
                    (not result.ok) and result.violations == 1)

    # Shard reconstruction.
    permutation = np.arange(12, dtype=np.int64)
    ids = epoch_sample_ids(permutation, 4)
    _selftest_check(results, "shards.epoch_ids",
                    np.array_equal(ids, permutation) and len(ids) == 12)
    rank0 = rank_sequence(permutation, 4, 0, 2)
    rank1 = rank_sequence(permutation, 4, 1, 2)
    _selftest_check(results, "shards.rank_sequence",
                    np.array_equal(rank0, np.asarray([0, 1, 4, 5, 8, 9])) and
                    np.array_equal(rank1, np.asarray([2, 3, 6, 7, 10, 11])))
    batch = permutation[:4]
    _selftest_check(results, "shards.contiguous",
                    np.array_equal(rank_shard(batch, 0, 2), [0, 1]) and
                    np.array_equal(rank_shard(batch, 1, 2), [2, 3]))
    ids512 = np.arange(1024, dtype=np.int64)
    seq = rank_sequence(ids512, 512, 3, 4)
    expected = np.concatenate([
        ids512[start + 384:start + 512]
        for start in (0, 512)])
    _selftest_check(results, "shards.batch512", np.array_equal(seq, expected))

    # Schedule values.
    _selftest_check(
        results, "schedule.boundaries",
        abs(schedule_eta(1) - ETA_MAX / WARMUP_UPDATES) == 0.0 and
        abs(schedule_eta(WARMUP_UPDATES) - ETA_MAX) == 0.0 and
        abs(schedule_eta(TOTAL_UPDATES)) == 0.0)

    # Two-step hand-computed AdamW case (decay weight, lambda=1e-4?  The
    # hand case uses lambda=0.1 through the explicit argument below).
    weight = "stem.conv.weight"
    bias = "stem.gn.bias"
    x0 = {weight: np.asarray([1.0]), bias: np.asarray([0.0])}
    gradients = {
        "u1": {weight: np.asarray([0.5]), bias: np.asarray([0.0])},
        "u2": {weight: np.asarray([-0.25]), bias: np.asarray([0.0])},
        "u3": {weight: np.asarray([0.0]), bias: np.asarray([0.0])},
    }
    # The hand case uses lambda=0.1 for the decay weight.
    oracle = adamw_oracle(x0, gradients,
                          {"u1": 0.01, "u2": 0.01, "u3": 0.01},
                          [weight, bias], weight_decay=0.1)
    hand_x = 0.9853476298705847
    hand_m = 0.019999999999999997
    hand_v = 0.0003122500000000003
    _selftest_check(
        results, "oracle.hand_two_step",
        abs(float(oracle["u2"]["param"][weight][0]) - hand_x) < 1e-15 and
        abs(float(oracle["u2"]["m"][weight][0]) - hand_m) < 1e-15 and
        abs(float(oracle["u2"]["v"][weight][0]) - hand_v) < 1e-15)
    _selftest_check(results, "oracle.decay_group",
                    is_decay_name("stem.conv.weight") and
                    not is_decay_name("stem.gn.weight") and
                    not is_decay_name("classifier.bias"))

    # C-order hash stability, including a non-contiguous view.
    array = np.arange(6, dtype="<f4").reshape(2, 3)
    expected_hash = ("e2c0a71510b5394df7773b63fb5f54372"
                     "b84c3564e67811bde7d665be227976d")
    _selftest_check(results, "hash.c_order",
                    sha256_c_order(array) == expected_hash)
    transposed_hash = "0c9d0bb54e4f5a0121543129f106617549c7ff2b34c6842c5a2e19186c5a7914"
    _selftest_check(results, "hash.noncontiguous",
                    sha256_c_order(array.T) == transposed_hash)
    _selftest_check(results, "hash.file_roundtrip",
                    _is_hash("0" * 64) and not _is_hash("0" * 63))

    # Collective protocol validation.
    names = ["n%02d" % index for index in range(62)]
    shapes = {name: (index + 1,) for index, name in enumerate(names)}
    expected = expected_flux_trace(names, shapes, "cpu")
    entries = []
    for item in expected:
        entries.append({
            "size": item["size"],
            "type": item["type"],
            "dtype": "float32",
            "op": "avg",
        })
    violations, details, signature = validate_flux_trace(
        entries, names, shapes, "cpu")
    _selftest_check(results, "protocol.valid",
                    violations == 0 and len(signature) == 64)
    mutated = [dict(item) for item in entries]
    mutated[0]["size"] = 61
    violations, _, _ = validate_flux_trace(mutated, names, shapes, "cpu")
    _selftest_check(results, "protocol.size_mutation", violations > 0)
    mutated = [dict(item) for item in entries]
    mutated[1]["type"] = "device"
    violations, _, _ = validate_flux_trace(mutated, names, shapes, "cpu")
    _selftest_check(results, "protocol.type_mutation", violations > 0)
    mutated = [dict(item) for item in entries]
    mutated[63]["op"] = "sum"
    violations, _, _ = validate_flux_trace(mutated, names, shapes, "cpu")
    _selftest_check(results, "protocol.op_mutation", violations > 0)
    violations, _, _ = validate_flux_trace(entries[:-1], names, shapes, "cpu")
    _selftest_check(results, "protocol.count_mutation", violations > 0)

    # The Phase 4 Flux runner records trace_entries as
    # {index, kind, name, buftype: "Vector{Float32}", size: [..], avg: true}.
    actual_format = []
    for index, item in enumerate(expected, start=1):
        if index == 1:
            kind = "presence"
        elif index == len(expected):
            kind = "vote"
        else:
            kind = "gradient"
        actual_format.append({
            "index": index,
            "kind": kind,
            "name": item["name"],
            "buftype": ("Matrix{Float32}" if index == 2
                        else "Vector{Float32}"),
            "size": (list(shapes[item["name"]]) if kind == "gradient"
                     else [item["size"]]),
            "avg": True,
        })
    violations, details, signature = validate_flux_trace(
        actual_format, names, shapes, "cpu")
    _selftest_check(results, "protocol.actual_runner_format",
                    violations == 0 and len(signature) == 64, details)
    mutated = [dict(item) for item in actual_format]
    mutated[1]["size"] = [2]
    violations, _, _ = validate_flux_trace(mutated, names, shapes, "cpu")
    _selftest_check(results, "protocol.actual_size_mutation", violations > 0)
    mutated = [dict(item) for item in actual_format]
    mutated[1]["avg"] = False
    violations, _, _ = validate_flux_trace(mutated, names, shapes, "cpu")
    _selftest_check(results, "protocol.actual_avg_mutation", violations > 0)
    mutated = [dict(item) for item in actual_format]
    mutated[1]["buftype"] = "CuArray{Float32, 1, CUDA.DeviceMemory}"
    violations, _, _ = validate_flux_trace(mutated, names, shapes, "cpu")
    _selftest_check(results, "protocol.actual_type_mutation", violations > 0)

    # Historical static protocol: 62 gradient entries in canonical bundle
    # order, with no presence reduction and no validity vote.
    static_expected = expected_flux_trace(
        names, shapes, "cpu", protocol="static")
    static_entries = []
    for item in static_expected:
        static_entries.append({
            "kind": "gradient",
            "name": item["name"],
            "buftype": "Vector{Float32}",
            "size": list(item["flux_shape"]),
            "canonical_size": list(item["canonical_size"]),
            "layout": item["layout"],
            "avg": True,
        })
    violations, details, signature = validate_flux_trace(
        static_entries, names, shapes, "cpu", protocol="static")
    _selftest_check(results, "protocol.static_valid",
                    violations == 0 and len(signature) == 62, details)
    violations, _, _ = validate_flux_trace(
        static_entries, names, shapes, "cpu", protocol="guarded")
    _selftest_check(results, "protocol.static_rejected_as_guarded",
                    violations > 0)
    violations, _, _ = validate_flux_trace(
        actual_format, names, shapes, "cpu", protocol="static")
    _selftest_check(results, "protocol.guarded_rejected_as_static",
                    violations > 0)
    mutated = [dict(item) for item in static_entries]
    mutated.append({
        "kind": "presence",
        "name": None,
        "buftype": "Vector{Float32}",
        "size": [62],
        "avg": True,
    })
    violations, _, _ = validate_flux_trace(
        mutated, names, shapes, "cpu", protocol="static")
    _selftest_check(results, "protocol.static_presence_extra_rejected",
                    violations > 0)
    mutated = [dict(item) for item in static_entries]
    mutated.append({
        "kind": "vote",
        "name": None,
        "buftype": "Vector{Float32}",
        "size": [1],
        "avg": True,
    })
    violations, _, _ = validate_flux_trace(
        mutated, names, shapes, "cpu", protocol="static")
    _selftest_check(results, "protocol.static_vote_extra_rejected",
                    violations > 0)
    mutated = [dict(item) for item in static_entries]
    mutated[0] = {
        "kind": "presence",
        "name": None,
        "buftype": "Vector{Float32}",
        "size": [62],
        "avg": True,
    }
    violations, _, _ = validate_flux_trace(
        mutated, names, shapes, "cpu", protocol="static")
    _selftest_check(results, "protocol.static_presence_entry_rejected",
                    violations > 0)
    mutated = [dict(item) for item in static_entries]
    del mutated[10]
    violations, _, _ = validate_flux_trace(
        mutated, names, shapes, "cpu", protocol="static")
    _selftest_check(results, "protocol.static_missing_gradient_rejected",
                    violations > 0)

    # The Flux runner records Julia 1-based shard ids and a *_0based alias.
    shard_record = {
        "shard_ids": {
            "u1": [1, 2, 3],
            "u1_0based": [0, 1, 2],
            "u3": [1, 2, 3],
            "u3_0based": [0, 1, 2],
        }
    }
    shard_ids = extract_shard_ids(shard_record)
    _selftest_check(results, "shard_ids.prefer_0based",
                    np.array_equal(shard_ids["u1"], [0, 1, 2]) and
                    np.array_equal(shard_ids["u3"], [0, 1, 2]))

    # Manifest/hash helpers.
    _selftest_check(results, "update_tags",
                    _update_from_tag("u1") == "u1" and
                    _update_from_tag("t3") == "u3" and
                    _update_from_tag("step2") == "u2" and
                    _update_from_tag("other") is None)
    parsed = parse_state_hash_mapping({
        "u1": {"param": "a" * 64, "m": "b" * 64, "v": "c" * 64},
        "u2": {"param": "d" * 64, "m": "e" * 64, "v": "f" * 64},
        "u3": {"param": "0" * 64, "m": "1" * 64, "v": "2" * 64},
        "beta_power": {"u1": "3" * 64, "u2": "4" * 64, "u3": "5" * 64},
    })
    _selftest_check(results, "state_hash_mapping",
                    parsed["u1"]["param"] == "a" * 64 and
                    parsed["u3"]["v"] == "2" * 64 and
                    parsed["u2"]["beta_power"] == "4" * 64)

    failed = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        print("PHASE4_SELFTEST CHECK %s %s%s" %
              (name, "PASS" if ok else "FAIL",
               (" " + detail) if detail else ""), flush=True)
    if failed:
        print("PHASE4_SELFTEST FAIL " + " ".join(failed), flush=True)
        return 1
    print("PHASE4_SELFTEST PASS checks=%d" % len(results), flush=True)
    return 0


def discover_reference_manifest(npz_path, explicit):
    if explicit:
        return os.path.abspath(explicit)
    if not npz_path:
        return None
    directory = os.path.dirname(os.path.abspath(npz_path))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    candidates = [
        os.path.join(directory, stem + "_manifest.json"),
        os.path.join(directory, stem + ".json"),
        os.path.join(directory, "phase4_reference_manifest.json"),
        os.path.join(directory, "phase3_reference_manifest.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def infer_device(explicit, path_map, manifest_map):
    if explicit in ("cpu", "cuda"):
        return explicit
    for framework, path in path_map.items():
        if not path:
            continue
        parts = [part.lower() for part in os.path.abspath(path).split(os.sep)]
        if "cuda" in parts:
            return "cuda"
        if "cpu" in parts:
            return "cpu"
    for manifest in manifest_map.values():
        if isinstance(manifest, dict):
            device = manifest.get("device")
            if isinstance(device, dict):
                device = device.get("type") or device.get("name")
            if isinstance(device, str):
                text = device.lower()
                if "cuda" in text:
                    return "cuda"
                if "cpu" in text:
                    return "cpu"
    return "cpu"


def discover_world_sizes(ddp_root, frameworks):
    found = set()
    roots = [ddp_root, os.path.join(ddp_root, "ddp")]
    for root in roots:
        for framework in frameworks:
            directory = os.path.join(root, framework)
            if not os.path.isdir(directory):
                continue
            for entry in os.listdir(directory):
                match = re.fullmatch(r"w(\d+)", entry)
                if match and os.path.isdir(os.path.join(directory, entry)):
                    world = int(match.group(1))
                    if world > 0:
                        found.add(world)
    return sorted(found)


def default_out_dir(ddp_root):
    if os.path.basename(os.path.normpath(ddp_root)) == "ddp":
        return os.path.join(os.path.dirname(os.path.abspath(ddp_root)),
                            "check")
    nested = os.path.join(ddp_root, "ddp")
    if os.path.isdir(nested):
        return os.path.join(os.path.abspath(ddp_root), "check")
    return os.path.join(os.path.abspath(ddp_root), "check")


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

def _parse_world_sizes(text):
    values = []
    for token in re.split(r"[,\s]+", str(text).strip()):
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            raise ValueError("invalid world size %r" % token)
        if value < 1:
            raise ValueError("world size must be positive")
        if value not in (1, 2, 4):
            raise ValueError("static Phase 4 supports world sizes 1, 2, 4")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("no world sizes selected")
    return values


def _parse_frameworks(text):
    values = []
    for token in re.split(r"[,\s]+", str(text).strip()):
        if not token:
            continue
        token = token.strip().lower()
        if token in ("torch", "pytorch"):
            token = "pytorch"
        if token not in ("flux", "pytorch"):
            raise ValueError("unknown framework %r" % token)
        if token not in values:
            values.append(token)
    if not values:
        raise ValueError("no frameworks selected")
    return values


def _load_optional_npz(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        return load_npz(path)
    except Exception:
        return None


def _load_optional_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _fixture_contract(data, bundle):
    batch = BATCH
    input_hw = (64, 64)
    n_classes = 200
    if isinstance(data, dict):
        train_ref = data.get("train_transformed_ref")
        if isinstance(train_ref, np.ndarray) and train_ref.ndim == 5:
            batch = int(train_ref.shape[1])
            input_hw = (int(train_ref.shape[-2]), int(train_ref.shape[-1]))
    if isinstance(bundle, dict) and "classifier.weight" in bundle:
        n_classes = int(bundle["classifier.weight"].shape[0])
    return batch, input_hw, n_classes


def discover_shard_reference(ddp_root, reference_path, framework, device,
                             world):
    base_dirs = []
    if ddp_root:
        abs_ddp = os.path.abspath(ddp_root)
        base_dirs.append(os.path.dirname(abs_ddp))
        base_dirs.append(abs_ddp)
    if reference_path:
        cur = os.path.abspath(reference_path)
        for _ in range(5):
            cur = os.path.dirname(cur)
            if cur and cur not in base_dirs:
                base_dirs.append(cur)
    candidates = []
    for base_dir in base_dirs:
        fw_dir = os.path.join(base_dir, "shard_reference", framework)
        if device:
            candidates.append(os.path.join(
                fw_dir, str(device), "w%d" % world, "phase4_reference.npz"))
        for dev in ("cpu", "cuda"):
            candidates.append(os.path.join(
                fw_dir, dev, "w%d" % world, "phase4_reference.npz"))
        if os.path.isdir(fw_dir):
            try:
                for entry in sorted(os.listdir(fw_dir)):
                    candidates.append(os.path.join(
                        fw_dir, entry, "w%d" % world, "phase4_reference.npz"))
            except OSError:
                pass
        candidates.append(os.path.join(
            fw_dir, "w%d" % world, "phase4_reference.npz"))
    seen = set()
    for path in candidates:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm):
            return norm
    return None


def check_shard_reference_integrity(log, framework, world, shard_ref,
                                    shard_manifest, shard_path,
                                    expected_reference):
    cat = "integrity.shard_reference.%s.w%d" % (framework, world)
    if not isinstance(shard_ref, dict):
        log.category(cat, False, math.inf, 1,
                     "npz_missing_or_unreadable:%s" % shard_path)
        return False
    if not isinstance(expected_reference, dict) or not expected_reference:
        log.category(cat, False, math.inf, 1, "expected_reference_unavailable")
        return False
    violations = 0
    problems = []
    expected_keys = set(expected_reference.keys())
    actual_keys = set(shard_ref.keys())
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        violations += len(missing)
        problems.append("missing=%s" % shortlist(missing))
    if extra:
        violations += len(extra)
        problems.append("extra=%s" % shortlist(extra))
    for key, spec in expected_reference.items():
        if key not in shard_ref:
            continue
        arr = np.asarray(shard_ref[key])
        if isinstance(spec, dict):
            exp_shape = spec.get("shape")
        elif (isinstance(spec, tuple) and len(spec) == 2 and
              isinstance(spec[1], tuple)):
            exp_shape = spec[1]
        elif isinstance(spec, tuple) and spec and isinstance(spec[0], tuple):
            exp_shape = spec[0]
        else:
            exp_shape = spec
        if exp_shape is not None and tuple(arr.shape) != tuple(exp_shape):
            violations += 1
            problems.append("shape:%s:%s!=%s" % (key, arr.shape, exp_shape))
        if not np.all(np.isfinite(arr)):
            violations += 1
            problems.append("nonfinite:%s" % key)
    if isinstance(shard_manifest, dict):
        manifest_count = shard_manifest.get("array_count")
        if manifest_count is not None and int(manifest_count) != len(
                expected_reference):
            violations += 1
            problems.append("manifest_count=%s" % manifest_count)
    detail = ("arrays=%d expected=%d path=%s" %
              (len(shard_ref), len(expected_reference), shard_path))
    if problems:
        detail += " problems=" + ";".join(problems[:8])
    log.category(cat, violations == 0, 0.0, violations, detail)
    return violations == 0


def check_shard_vs_full_control(log, framework, world, shard_ref, full_ref,
                                bundle_shapes, sync_grads=None):
    cat = "control.shard_vs_full.%s.w%d" % (framework, world)
    if not isinstance(shard_ref, dict) or not isinstance(full_ref, dict):
        log.diagnostic(cat, {"status": "reference_missing"})
        return {}
    envelope = {}
    overall_max_abs = 0.0
    parts = []
    key_map = (
        ("grads.u1", "grad.u1."),
        ("grads.u2", "grad.u2."),
        ("grads.u3", "grad.u3."),
        ("param.u3", "param.u3."),
        ("m.u3", "m.u3."),
        ("v.u3", "v.u3."),
    )
    for label, prefix in key_map:
        results = []
        if label == "grads.u2" and not any(
                (prefix + name) in shard_ref and (prefix + name) in full_ref
                for name in bundle_shapes):
            w_u2 = (sync_grads or {}).get((framework, world), {}).get("u2", {})
            w1_u2 = (sync_grads or {}).get((framework, 1), {}).get("u2", {})
            if w_u2 and w1_u2:
                for name in bundle_shapes:
                    if name in w_u2 and name in w1_u2:
                        results.append(compare_arrays(
                            w_u2[name], w1_u2[name], 0.0, 0.0))
        else:
            for name in bundle_shapes:
                k = prefix + name
                if k in shard_ref and k in full_ref:
                    results.append(compare_arrays(
                        shard_ref.get(k), full_ref.get(k), 0.0, 0.0))
        if results:
            merged = merge_results(results)
            max_abs = float(merged.max_abs)
            overall_max_abs = max(overall_max_abs, max_abs)
            envelope[label] = {
                "max_abs": max_abs,
                "leaves": len(results),
            }
            parts.append("%s=%s" % (label, fmt_float(max_abs)))
        else:
            envelope[label] = {
                "max_abs": None,
                "leaves": 0,
            }
            parts.append("%s=n/a" % label)
    detail = "non_gating_batch_split_envelope " + " ".join(parts)
    log.diagnostic(cat, envelope)
    log.category(cat, True, overall_max_abs, 0, detail)
    return envelope


def main(argv=None):
    global FP32_ABSOLUTE_LIMIT
    parser = argparse.ArgumentParser(
        description="Independent Phase 4 distributed-correctness checker")
    parser.add_argument("--fixture-dir")
    parser.add_argument("--bundle")
    parser.add_argument("--pytorch-reference", "--torch-reference",
                        dest="pytorch_reference")
    parser.add_argument("--pytorch-reference-manifest",
                        "--torch-reference-manifest",
                        dest="pytorch_reference_manifest")
    parser.add_argument("--flux-reference")
    parser.add_argument("--flux-reference-manifest")
    parser.add_argument("--ddp-root", "--ddp-dir", dest="ddp_root")
    parser.add_argument("--out-dir")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--world-sizes", default=None)
    parser.add_argument("--frameworks", default="flux,pytorch")
    parser.add_argument("--flux-collective-protocol",
                        choices=FLUX_COLLECTIVE_PROTOCOLS,
                        default=DEFAULT_FLUX_COLLECTIVE_PROTOCOL,
                        help="Flux trace protocol: guarded (64 entries) or "
                             "static (62 gradient entries)")
    parser.add_argument("--atol", type=float, default=2.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--cross-atol", type=float, default=1.0e-5)
    parser.add_argument("--cross-rtol", type=float, default=1.0e-4)
    parser.add_argument("--grads-u1-atol", type=float, default=1.0e-3)
    parser.add_argument("--grads-u2-atol", type=float, default=2.0e-3)
    parser.add_argument("--grads-u3-atol", type=float, default=2.0e-3)
    parser.add_argument("--m-u3-atol", type=float, default=2.0e-4)
    parser.add_argument("--split-control", action="append", nargs="+",
                        default=[],
                        help="optional split-control .npz path or "
                             "framework=path; repeatable or comma-separated")
    parser.add_argument("--self-test", action="store_true",
                         help="run unit tests for the pure helper functions")
    parser.add_argument("--fp32-absolute-1e3", action="store_true",
                        help="require finite numerical errors strictly <1e-3; preserve exact checks")
    args = parser.parse_args(argv)
    FP32_ABSOLUTE_LIMIT = 1.0e-3 if args.fp32_absolute_1e3 else None
    if FP32_ABSOLUTE_LIMIT is not None:
        print("PHASE4_ACCEPTANCE_RULE abs(a-b)<1e-3 rtol=0 structural_and_rank_checks=exact", flush=True)

    if args.self_test:
        return run_self_tests()

    for name in ("fixture_dir", "bundle", "ddp_root"):
        if not getattr(args, name):
            parser.error("--%s is required" % name.replace("_", "-"))
    try:
        frameworks = _parse_frameworks(args.frameworks)
    except ValueError as exc:
        parser.error(str(exc))
    for name in ("atol", "rtol", "cross_atol", "cross_rtol",
                 "grads_u1_atol", "grads_u2_atol", "grads_u3_atol",
                 "m_u3_atol"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error("--%s must be non-negative" % name.replace("_", "-"))
    if "flux" in frameworks and not args.flux_reference:
        parser.error("--flux-reference is required when flux is selected")
    if "pytorch" in frameworks and not args.pytorch_reference:
        parser.error("--pytorch-reference is required when pytorch is selected")

    fixture_dir = os.path.abspath(args.fixture_dir)
    bundle_path = os.path.abspath(args.bundle)
    ddp_root = os.path.abspath(args.ddp_root)
    env_out_dir = os.environ.get("CHECK_DIR")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else (
        os.path.abspath(env_out_dir) if env_out_dir else default_out_dir(
            ddp_root))
    if args.world_sizes is None:
        world_sizes = discover_world_sizes(ddp_root, frameworks)
        if not world_sizes:
            env_worlds = os.environ.get("WORLD_SIZES", "")
            try:
                world_sizes = (_parse_world_sizes(env_worlds)
                               if env_worlds.strip()
                               else _parse_world_sizes(DEFAULT_WORLD_SIZES))
            except ValueError as exc:
                parser.error(str(exc))
    else:
        try:
            world_sizes = _parse_world_sizes(args.world_sizes)
        except ValueError as exc:
            parser.error(str(exc))
    log = CheckLog()
    started = now_utc()

    fixture_manifest = check_fixture_integrity(log, fixture_dir)
    data, schedule = load_fixture_data(log, fixture_dir)
    bundle = _load_optional_npz(bundle_path)
    if bundle is None:
        log.category("bundle.load", False, math.inf, 1,
                     "bundle_missing_or_unreadable")
    else:
        log.category("bundle.load", True, 0.0, 0, "arrays=%d" % len(bundle))

    reference_paths = {
        "flux": args.flux_reference,
        "pytorch": args.pytorch_reference,
    }
    reference_manifest_paths = {
        "flux": args.flux_reference_manifest or os.environ.get("FLUX_REF_JSON"),
        "pytorch": (args.pytorch_reference_manifest or
                    os.environ.get("TORCH_MANIFEST")),
    }
    references = {}
    reference_manifests = {}
    for framework in frameworks:
        path = os.path.abspath(reference_paths[framework])
        manifest_path = discover_reference_manifest(
            path, reference_manifest_paths[framework])
        references[framework] = _load_optional_npz(path)
        reference_manifests[framework] = _load_optional_json(manifest_path)
        log.category("reference.load.%s" % framework,
                     references[framework] is not None,
                     0.0, 0 if references[framework] is not None else 1,
                     "npz=%s manifest=%s" %
                     (path, manifest_path))

    explicit_device = args.device
    if explicit_device is None:
        env_device = os.environ.get("DEVICE", "").strip().lower()
        if env_device in ("cpu", "cuda"):
            explicit_device = env_device
    device = infer_device(explicit_device, reference_paths, reference_manifests)
    log.diagnostic("device", device)
    log.diagnostic("world_sizes", world_sizes)

    bundle_shapes = check_bundle_integrity(
        log, bundle, bundle_path,
        [reference_manifests.get(framework) for framework in frameworks])
    batch, input_hw, n_classes = _fixture_contract(data, bundle)

    expected_reference = {}
    if bundle_shapes:
        expected_reference = expected_reference_arrays(
            bundle_shapes, batch, input_hw, n_classes)
        for framework in frameworks:
            reference = references.get(framework)
            manifest = reference_manifests.get(framework)
            path = os.path.abspath(reference_paths[framework])
            check_reference_integrity(
                log, framework, reference, manifest, path,
                expected_reference, bundle_path)
            check_reference_inputs(log, framework, reference, data)
    else:
        log.category("reference.expected_schema", False, math.inf, 1,
                     "bundle_shapes_unavailable")

    if isinstance(schedule, dict):
        check_schedule_fixture(log, schedule)
    else:
        log.category("shards.schedule_fixture", False, math.inf, 1,
                     "schedule_fixture_missing")

    runs = {}
    sync_grads = {}
    for framework in frameworks:
        for world in world_sizes:
            if batch % world != 0:
                log.category("ddp.world_size.%s.w%d" % (framework, world),
                             False, 0.0, 1, "batch_not_divisible")
                continue
            run = load_ddp_run(ddp_root, framework, world)
            runs[(framework, world)] = run
            expected_ddp = expected_ddp_arrays(
                framework, world, batch, input_hw, n_classes, bundle_shapes)
            check_ddp_integrity(log, run, expected_ddp, batch, bundle_shapes,
                                framework)
            check_updates(log, run)
            check_global_batch_shards(log, run, data)
            check_rank_replicas(log, run)
            shard_ref_path = discover_shard_reference(
                ddp_root, reference_paths.get(framework), framework, device,
                world)
            shard_ref = None
            if shard_ref_path is not None:
                shard_ref = _load_optional_npz(shard_ref_path)
                shard_manifest_path = discover_reference_manifest(
                    shard_ref_path, None)
                shard_manifest = _load_optional_json(shard_manifest_path)
                check_shard_reference_integrity(
                    log, framework, world, shard_ref, shard_manifest,
                    shard_ref_path, expected_reference)
            same_ref = (shard_ref if isinstance(shard_ref, dict)
                        else references.get(framework))
            if framework == "flux":
                sync = check_flux_syncgrads(
                    log, run, same_ref, bundle_shapes,
                    args.atol, args.rtol, args=args, device=device)
            else:
                sync = check_pytorch_syncgrads(
                    log, run, same_ref, bundle_shapes,
                    args.atol, args.rtol, args=args, device=device)
            sync_grads[(framework, world)] = sync
            if isinstance(shard_ref, dict):
                check_shard_vs_full_control(
                    log, framework, world, shard_ref,
                    references.get(framework), bundle_shapes, sync_grads)
            etas = {update: schedule_eta(int(update[1]))
                    for update in UPDATES}
            observed = first_npz(run) or {}
            if bundle is not None and bundle_shapes:
                check_oracle(log, framework, world, sync, observed, bundle,
                             bundle_shapes, etas, args.atol, args.rtol)
            else:
                log.category("oracle.%s.w%d" % (framework, world), False,
                             math.inf, 1, "bundle_missing")
            check_same_framework(
                log, run, same_ref, data, sync,
                bundle_shapes, args.atol, args.rtol, args=args, device=device)
            if framework == "flux":
                check_flux_protocol(
                    log, run, list(bundle_shapes), bundle_shapes, device,
                    protocol=args.flux_collective_protocol)
            else:
                check_pytorch_protocol(log, run, device)

    if "flux" in frameworks and "pytorch" in frameworks:
        for world in world_sizes:
            flux_run = runs.get(("flux", world))
            torch_run = runs.get(("pytorch", world))
            if flux_run is None or torch_run is None:
                continue
            flux_sync = sync_grads.get(("flux", world), {})
            torch_sync = sync_grads.get(("pytorch", world), {})
            check_cross_framework(
                log, flux_run, torch_run, flux_sync, torch_sync,
                references.get("pytorch"), bundle_shapes, args)

    split_paths = discover_split_paths(
        ddp_root, frameworks, device, args.split_control)
    split_report = check_split_control(log, split_paths, bundle_shapes)

    passed = not log.failed
    report = {
        "schema": "phase4-check-v1",
        "created": started,
        "finished": now_utc(),
        "device": device,
        "world_sizes": world_sizes,
        "frameworks": frameworks,
        "flux_collective_protocol": args.flux_collective_protocol,
        "fixture_dir": fixture_dir,
        "bundle": bundle_path,
        "ddp_root": ddp_root,
        "tolerances": {
            "effective_policy": "FP32: finite abs(a-b)<1e-3, rtol=0; exact structural/rank checks; non-FP32 retains legacy rules" if FP32_ABSOLUTE_LIMIT is not None else "legacy per-category rules",
            "absolute_limit": FP32_ABSOLUTE_LIMIT,
            "argument_scope": "The following legacy arguments do not override the effective FP32 absolute policy.",
            "same_framework": {"atol": args.atol, "rtol": args.rtol},
            "cross_framework": {"atol": args.cross_atol,
                                "rtol": args.cross_rtol},
            "category_overrides": {
                "grads.u1": args.grads_u1_atol,
                "grads.u2": args.grads_u2_atol,
                "grads.u3": args.grads_u3_atol,
                "m.u3": args.m_u3_atol,
            },
        },
        "categories": log.categories,
        "diagnostics": log.diagnostics,
        "split_control": split_report,
        "pass": passed,
        "failed": log.failed,
    }
    json_path, text_path = write_reports(out_dir, report)
    if passed:
        print("PHASE4_CHECK PASS categories=%d report=%s" %
              (len(log.categories), json_path), flush=True)
        return 0
    print("PHASE4_CHECK FAIL " + " ".join(log.failed), flush=True)
    print("PHASE4_REPORT %s" % text_path, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
