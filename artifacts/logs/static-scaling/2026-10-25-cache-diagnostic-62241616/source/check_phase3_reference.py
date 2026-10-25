#!/usr/bin/env python3
"""
Phase 3 reference artifact checker (numpy only; no torch, no Flux).

Validates the PyTorch reference written by make_phase3_reference.py against the
frozen Phase 2 data fixture and initial bundle:

  * manifest structural fields, provenance, device/torch/cuda and TF32 records,
  * AdamW configuration and schedule W/T/eta values,
  * sha256 of the data fixture, the initial bundle, and the reference archive,
  * the exact 337-array set, dtype/shape, and per-array sha256 of C-order bytes,
  * input.u*/label.u* recomputed from data_fixture.npz (exact equality),
  * eta.u1..u3 recomputed from the warm-up schedule (rtol 1e-12),
  * finiteness of every float array, nonzero m/v after u3, step.u3 == 3,
  * the 21/41 decay/no-decay groupings and the model counts.

The --atol/--rtol arguments are accepted for CLI parity with
check_phase3_flux.jl; this checker does not compare two frameworks.

Manifest key alternatives accepted for robustness (the contract fixes the
array-entry schema, not every top-level key spelling):
  created | created_utc | created_at
  files (list/dict) | fixture_files | reference_files | artifact_files |
  reference_file {path, bytes, sha256}
  bundle_sha256 | initial_bundle_sha256 | files entry for initial_bundle.npz
  torch | torch_version | runtime.torch_version
  cuda | cuda_version | runtime.cuda_available
  tf32 (bool/string, or PyTorch backend flags such as
  cuda_matmul_allow_tf32/cudnn_allow_tf32/float32_matmul_precision=highest)
  adamw | optimizer (dict) | optimizer_config
  schedule.W/warmup_updates, schedule.T/total_updates,
  schedule.eta/etas/eta_per_update or schedule.eta_u1..u3
  optimizer_steps | steps | step_numbers | optimizer_step_numbers |
  schedule.steps | training.optimizer_step_numbers
  model.groups / model.decay + model.no_decay / manifest groups /
  optimizer.decay_group.parameters + optimizer.no_decay_group.parameters /
  PyTorch-style [{"params": [...], "weight_decay": ...}] lists.

Prints one `CHECK <name> PASS|FAIL <detail>` line per check and a final
`PHASE3_REFERENCE_CHECK PASS|FAIL ...`; exits 0 only on PASS.

Reproduction (Leonardo compute node, after make_phase3_reference.py):
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/check_phase3_reference.py \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \\
    --reference artifacts/baselines/static-scaling/2026-10-17-phase3/phase3_reference.npz \\
    --manifest artifacts/baselines/static-scaling/2026-10-17-phase3/phase3_reference_manifest.json

Local inspection (numpy only; after the reference is copied home):
  python3 scripts/static_scaling/check_phase3_reference.py \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \\
    --reference <copy>/phase3_reference.npz \\
    --manifest <copy>/phase3_reference_manifest.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np

PARAM_COUNT = 11_271_432
ARRAY_COUNT = 62
ETA_MAX = 1.0e-3
WARMUP_UPDATES = 975
TOTAL_UPDATES = 9_750
ETA_RTOL = 1.0e-12
BATCH = 12
UPDATES = ("u1", "u2", "u3")

_MISSING = object()
_HEX64 = set("0123456789abcdef")


class CheckLog:
    def __init__(self):
        self.failed = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print("CHECK %s %s%s" % (name, "PASS" if ok else "FAIL",
                                 (" " + detail) if detail else ""), flush=True)
        if not ok:
            self.failed.append(name)
        return bool(ok)


def _fmt(x) -> str:
    try:
        return "%.9g" % float(x)
    except (TypeError, ValueError):
        return repr(x)


def _shortlist(items, limit: int = 5) -> str:
    items = [str(x) for x in items]
    if not items:
        return "none"
    shown = ";".join(items[:limit])
    return shown + (";+%dmore" % (len(items) - limit) if len(items) > limit else "")


def _first(container, keys, default=None):
    if isinstance(container, dict):
        for key in keys:
            if key in container:
                return container[key]
    return default


def _as_count(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        for key in ("count", "total", "num", "n"):
            if key in value:
                return _as_count(value[key])
    return None


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _close(value, target: float, rtol: float = 1.0e-12) -> bool:
    number = _as_number(value)
    return number is not None and math.isclose(number, target, rel_tol=rtol, abs_tol=0.0)


def sha256_c_order(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_hash_of_arrays(arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def expected_bundle_shapes():
    shapes = [
        ("stem.conv.weight", (64, 3, 3, 3)),
        ("stem.gn.weight", (64,)),
        ("stem.gn.bias", (64,)),
    ]
    widths = (64, 128, 256, 512)
    previous = 64
    for stage in range(1, 5):
        channels = widths[stage - 1]
        for block in (0, 1):
            cin = previous if block == 0 else channels
            prefix = "stage%d.block%d" % (stage, block)
            shapes.extend([
                (prefix + ".conv1.weight", (channels, cin, 3, 3)),
                (prefix + ".conv2.weight", (channels, channels, 3, 3)),
                (prefix + ".gn1.weight", (channels,)),
                (prefix + ".gn1.bias", (channels,)),
                (prefix + ".gn2.weight", (channels,)),
                (prefix + ".gn2.bias", (channels,)),
            ])
        if stage >= 2:
            prefix = "stage%d.block0" % stage
            shapes.extend([
                (prefix + ".proj.weight", (channels, channels // 2, 1, 1)),
                (prefix + ".proj_gn.weight", (channels,)),
                (prefix + ".proj_gn.bias", (channels,)),
            ])
        previous = channels
    shapes.extend([
        ("classifier.weight", (200, 512)),
        ("classifier.bias", (200,)),
    ])
    return shapes


def expected_reference_arrays(bundle_shapes):
    expected = {}
    for update in UPDATES:
        expected["input." + update] = ("float32", (BATCH, 3, 64, 64))
        expected["label." + update] = ("int64", (BATCH,))
        expected["eta." + update] = ("float64", ())
        expected["loss." + update] = ("float32", ())
        expected["logits." + update] = ("float32", (BATCH, 200))
    activations = [
        ("act.stem.gn", (BATCH, 64, 64, 64)),
        ("act.stem", (BATCH, 64, 64, 64)),
        ("act.stage1.block0", (BATCH, 64, 64, 64)),
        ("act.stage1.block1", (BATCH, 64, 64, 64)),
        ("act.stage2.block0", (BATCH, 128, 32, 32)),
        ("act.stage2.block1", (BATCH, 128, 32, 32)),
        ("act.stage3.block0", (BATCH, 256, 16, 16)),
        ("act.stage3.block1", (BATCH, 256, 16, 16)),
        ("act.stage4.block0", (BATCH, 512, 8, 8)),
        ("act.stage4.block1", (BATCH, 512, 8, 8)),
        ("act.pooled", (BATCH, 512)),
    ]
    for name, shape in activations:
        expected[name] = ("float32", shape)
    for prefix in ("grad.u1.", "grad.u3.", "param.u3.", "m.u3.", "v.u3."):
        for name, shape in bundle_shapes:
            expected[prefix + name] = ("float32", shape)
    expected["step.u3"] = ("int64", ())
    return expected


def is_group_norm_name(name: str) -> bool:
    for suffix in (".gn.weight", ".gn.bias", ".gn1.weight", ".gn1.bias",
                   ".gn2.weight", ".gn2.bias", ".proj_gn.weight", ".proj_gn.bias"):
        if name.endswith(suffix):
            return True
    return False


def schedule_eta(update_index: int) -> float:
    if update_index <= WARMUP_UPDATES:
        return ETA_MAX * update_index / WARMUP_UPDATES
    return ETA_MAX * 0.5 * (1.0 + math.cos(
        math.pi * (update_index - WARMUP_UPDATES) / (TOTAL_UPDATES - WARMUP_UPDATES)))


def parse_utc(text) -> bool:
    if not isinstance(text, str):
        return False
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == datetime.timedelta(0)


def file_entries(manifest):
    entries = []
    for key in ("files", "fixture_files", "reference_files", "artifact_files"):
        value = manifest.get(key)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    entries.append((entry["path"], entry.get("sha256"), entry.get("bytes")))
        elif isinstance(value, dict):
            for path, meta in value.items():
                if isinstance(meta, str):
                    entries.append((path, meta, None))
                elif isinstance(meta, dict):
                    entries.append((path, meta.get("sha256"), meta.get("bytes")))
    return entries


def find_recorded_sha(manifest, filename: str, extra_keys=()):
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]
    for path, digest, size in file_entries(manifest):
        if os.path.basename(path) == basename:
            return digest, size, path
    keys = list(extra_keys) + [
        stem + "_sha256",
        stem.replace(".", "_") + "_sha256",
        basename + "_sha256",
    ]
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str):
            return value, None, key
    reference_entry = manifest.get("reference_file")
    if isinstance(reference_entry, dict) and isinstance(reference_entry.get("sha256"), str):
        entry_path = reference_entry.get("path")
        if not isinstance(entry_path, str) or os.path.basename(entry_path) == basename:
            return reference_entry["sha256"], reference_entry.get("bytes"), "reference_file"
    for container_key in ("fixture_sha256", "fixture_hashes", "file_sha256", "hashes"):
        container = manifest.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in (basename, stem):
            value = container.get(key)
            if isinstance(value, str):
                return value, None, container_key + "." + key
            if isinstance(value, dict) and isinstance(value.get("sha256"), str):
                return value["sha256"], value.get("bytes"), container_key + "." + key
    return None, None, None


def find_tf32(manifest):
    keys = ("tf32", "tf32_enabled", "allow_tf32", "torch_tf32", "use_tf32")
    for key in keys:
        if key in manifest:
            return manifest[key]
    for container_key in ("torch", "cuda", "settings", "precision"):
        container = manifest.get(container_key)
        if isinstance(container, dict):
            for key in keys:
                if key in container:
                    return container[key]
    return _MISSING


def tf32_disabled(value) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "false", "disabled", "off", "0", "no", "not_applicable", "n/a",
            "highest", "ieee", "float32",
        }
    if isinstance(value, (list, tuple)):
        return bool(value) and all(tf32_disabled(item) for item in value)
    if isinstance(value, dict):
        tf32_values = [item for key, item in value.items() if "tf32" in str(key).lower()]
        precision = value.get("float32_matmul_precision")
        precision_ok = True
        if precision is not None:
            precision_ok = (isinstance(precision, str) and
                            precision.strip().lower() in {"highest", "ieee", "float32"})
        if tf32_values:
            return precision_ok and all(tf32_disabled(item) for item in tf32_values)
        if precision is not None:
            return precision_ok
        return bool(value) and all(tf32_disabled(item) for item in value.values())
    return False


def check_manifest_fields(log, manifest):
    required = [
        ("producer", ("producer",)),
        ("created", ("created", "created_utc", "created_at")),
        ("synthetic", ("synthetic",)),
        ("schema", ("schema",)),
        ("source_commit", ("source_commit",)),
        ("source_state_sha256", ("source_state_sha256",)),
        ("flux_commit", ("flux_commit",)),
        ("flux_state_sha256", ("flux_state_sha256",)),
    ]
    missing = [label for label, keys in required if _first(manifest, keys, _MISSING) is _MISSING]
    log.check("manifest.required_fields", not missing, "missing=" + _shortlist(missing))

    producer = manifest.get("producer")
    log.check("manifest.producer", isinstance(producer, str) and bool(producer.strip()),
              "producer=%r" % (producer,))

    created = _first(manifest, ("created", "created_utc", "created_at"))
    log.check("manifest.created_utc", parse_utc(created), "created=%r" % (created,))

    log.check("manifest.synthetic_false", manifest.get("synthetic") is False,
              "synthetic=%r" % (manifest.get("synthetic"),))

    schema = manifest.get("schema")
    log.check("manifest.schema", isinstance(schema, str) and bool(schema.strip()),
              "schema=%r" % (schema,))

    provenance_bad = []
    for key in ("source_commit", "source_state_sha256", "flux_commit", "flux_state_sha256"):
        value = manifest.get(key)
        if value is not None and not isinstance(value, str):
            provenance_bad.append(key)
    log.check("manifest.provenance", not provenance_bad,
              "bad_type=" + _shortlist(provenance_bad) +
              " source_commit=%r flux_commit=%r" % (manifest.get("source_commit"),
                                                     manifest.get("flux_commit")))

    device = manifest.get("device")
    if isinstance(device, dict):
        device_name = str(device.get("name") or device.get("type") or "")
    else:
        device_name = str(device or "")
    log.check("manifest.device", bool(device_name.strip()), "device=%r" % (device,))

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    torch_info = manifest.get("torch")
    if torch_info is None:
        torch_info = manifest.get("torch_version")
    if torch_info is None:
        torch_info = runtime.get("torch_version")
    torch_ok = isinstance(torch_info, str) and bool(torch_info.strip())
    if isinstance(torch_info, dict):
        torch_ok = bool(torch_info) and isinstance(
            _first(torch_info, ("version", "torch_version", "name")), str)
    log.check("manifest.torch_info", torch_ok, "torch=%r" % (torch_info,))

    is_cuda = "cuda" in device_name.lower()
    cuda_info = manifest.get("cuda")
    if cuda_info is None:
        cuda_info = manifest.get("cuda_version")
    if cuda_info is None:
        cuda_info = runtime.get("cuda_version")
    if is_cuda:
        cuda_ok = (cuda_info is not None and bool(cuda_info)) or (
            runtime.get("cuda_available") is True)
        log.check("manifest.cuda_info", cuda_ok, "cuda=%r runtime.cuda_available=%r" %
                  (cuda_info, runtime.get("cuda_available")))
    else:
        log.check("manifest.cuda_info", True, "device is not CUDA; cuda record optional")

    tf32 = find_tf32(manifest)
    if is_cuda:
        tf32_ok = tf32 is not _MISSING and tf32_disabled(tf32)
    else:
        tf32_ok = tf32 is not _MISSING
    log.check("manifest.tf32_disabled", tf32_ok, "tf32=%r" % (tf32,))


def check_file_hashes(log, manifest, fixture_dir, bundle_path, reference_path, bundle):
    problems = []
    details = []

    fixture_path = os.path.join(fixture_dir, "data_fixture.npz")
    recorded, size, source = find_recorded_sha(manifest, "data_fixture.npz")
    if recorded is None:
        problems.append("data_fixture.npz:no-recorded-sha256")
    else:
        actual = sha256_file(fixture_path)
        if recorded.lower() != actual:
            problems.append("data_fixture.npz:sha256")
        if size is not None:
            try:
                if int(size) != os.path.getsize(fixture_path):
                    problems.append("data_fixture.npz:bytes")
            except (TypeError, ValueError):
                problems.append("data_fixture.npz:bytes-meta")
        details.append("data_fixture.npz<-%s" % source)

    recorded, size, source = find_recorded_sha(
        manifest, "initial_bundle.npz", extra_keys=("bundle_sha256",))
    if recorded is None:
        problems.append("initial_bundle.npz:no-recorded-sha256")
    else:
        actual_file = sha256_file(bundle_path)
        actual_arrays = bundle_hash_of_arrays(
            [bundle[name] for name in sorted(bundle)]) if bundle else None
        recorded_lower = recorded.lower()
        if recorded_lower == actual_file:
            details.append("initial_bundle.npz<-file-sha256")
        elif actual_arrays is not None and recorded_lower == actual_arrays:
            details.append("initial_bundle.npz<-array-concat-sha256")
        else:
            problems.append("initial_bundle.npz:sha256")
        if size is not None:
            try:
                if int(size) != os.path.getsize(bundle_path):
                    problems.append("initial_bundle.npz:bytes")
            except (TypeError, ValueError):
                problems.append("initial_bundle.npz:bytes-meta")

    recorded, size, source = find_recorded_sha(
        manifest, "phase3_reference.npz",
        extra_keys=("reference_sha256", "phase3_reference_sha256"))
    if recorded is None:
        problems.append("phase3_reference.npz:no-recorded-sha256")
    else:
        actual = sha256_file(reference_path)
        if recorded.lower() != actual:
            problems.append("phase3_reference.npz:sha256")
        if size is not None:
            try:
                if int(size) != os.path.getsize(reference_path):
                    problems.append("phase3_reference.npz:bytes")
            except (TypeError, ValueError):
                problems.append("phase3_reference.npz:bytes-meta")
        details.append("phase3_reference.npz<-%s" % source)

    fixture_manifest_path = os.path.join(fixture_dir, "fixture_manifest.json")
    recorded, _, source = find_recorded_sha(manifest, "fixture_manifest.json")
    if recorded is not None:
        if recorded.lower() != sha256_file(fixture_manifest_path):
            problems.append("fixture_manifest.json:sha256")
        details.append("fixture_manifest.json<-%s" % source)

    log.check("manifest.file_sha256", not problems,
              "checked=%s mismatches=%s" % (";".join(details), _shortlist(problems)))


def check_bundle(log, bundle, expected):
    want = sorted(name for name, _ in expected)
    actual = sorted(bundle)
    missing = sorted(set(want) - set(actual))
    extra = sorted(set(actual) - set(want))
    log.check("bundle.name_set",
              len(bundle) == ARRAY_COUNT and not missing and not extra,
              "n=%d want=%d missing=%s extra=%s" %
              (len(bundle), ARRAY_COUNT, _shortlist(missing), _shortlist(extra)))

    metadata_bad = []
    total = 0
    for name, shape in expected:
        if name not in bundle:
            continue
        array = bundle[name]
        if array.dtype != np.float32:
            metadata_bad.append(name + ":dtype=" + str(array.dtype))
        if tuple(array.shape) != shape:
            metadata_bad.append(name + ":shape=" + str(tuple(array.shape)))
        total += int(array.size)
    log.check("bundle.dtypes_shapes", not metadata_bad,
              "n_checked=%d mismatches=%s" % (len(want), _shortlist(metadata_bad)))
    log.check("bundle.parameter_count", total == PARAM_COUNT and not missing,
              "total=%d want=%d" % (total, PARAM_COUNT))


def parse_array_entries(log, manifest, expected):
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list):
        log.check("manifest.array_entries", False, "'arrays' is not a list")
        return {}
    entries = {}
    malformed = []
    duplicates = []
    for entry in arrays:
        if not isinstance(entry, dict):
            malformed.append("<non-object>")
            continue
        name = entry.get("name")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        layout = entry.get("layout")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(dtype, str):
            malformed.append("<name/dtype>")
            continue
        if not (isinstance(shape, list) and all(
                isinstance(v, int) and not isinstance(v, bool) for v in shape)):
            malformed.append(name + ":shape")
            continue
        if not isinstance(layout, str):
            malformed.append(name + ":layout")
        if not (isinstance(digest, str) and len(digest) == 64 and
                set(digest.lower()) <= _HEX64):
            malformed.append(name + ":sha256")
        if name in entries:
            duplicates.append(name)
        entries[name] = {
            "dtype": dtype.lower(),
            "shape": tuple(shape),
            "layout": layout,
            "sha256": digest.lower() if isinstance(digest, str) else None,
        }
    log.check("manifest.array_entries",
              not malformed and not duplicates,
              "n=%d malformed=%s duplicates=%s" %
              (len(entries), _shortlist(malformed), _shortlist(duplicates)))

    want = set(expected)
    got = set(entries)
    missing = sorted(want - got)
    extra = sorted(got - want)
    log.check("manifest.array_set",
              not missing and not extra and len(entries) == len(expected),
              "n=%d want=%d missing=%s extra=%s" %
              (len(entries), len(expected), _shortlist(missing), _shortlist(extra)))
    return entries


def check_reference_arrays(log, reference, entries, expected):
    want = set(expected)
    got = set(reference)
    missing = sorted(want - got)
    extra = sorted(got - want)
    log.check("arrays.actual_set",
              not missing and not extra and len(reference) == len(expected),
              "n=%d want=%d missing=%s extra=%s" %
              (len(reference), len(expected), _shortlist(missing), _shortlist(extra)))

    metadata_bad = []
    manifest_bad = []
    hash_bad = []
    n_metadata = 0
    n_hash = 0
    for name, (dtype_name, shape) in expected.items():
        array = reference.get(name)
        entry = entries.get(name)
        if array is None or entry is None:
            continue
        n_metadata += 1
        # Scalar artifacts are accepted as 0-d or one-element arrays. The
        # PyTorch generator stores one-element arrays; older tooling may
        # store true 0-d scalars.
        allowed_shapes = (shape, (1,)) if shape == () else (shape,)
        if str(array.dtype) != dtype_name or tuple(array.shape) not in allowed_shapes:
            metadata_bad.append("%s:%s%s" % (name, array.dtype, tuple(array.shape)))
        if entry["dtype"] != dtype_name or tuple(entry["shape"]) not in allowed_shapes:
            manifest_bad.append("%s:%s%s" % (name, entry["dtype"], tuple(entry["shape"])))
        if entry["sha256"] is not None:
            n_hash += 1
            if sha256_c_order(array) != entry["sha256"]:
                hash_bad.append(name)
    log.check("arrays.metadata", not metadata_bad,
              "n_checked=%d mismatches=%s" % (n_metadata, _shortlist(metadata_bad)))
    log.check("arrays.manifest_metadata", not manifest_bad,
              "n_checked=%d mismatches=%s" % (n_metadata, _shortlist(manifest_bad)))
    log.check("arrays.sha256", not hash_bad and n_hash == len(expected),
              "n_checked=%d want=%d mismatches=%s" %
              (n_hash, len(expected), _shortlist(hash_bad)))


def check_fixture_inputs(log, data, reference):
    if not isinstance(data, dict):
        for update in UPDATES:
            log.check("fixture.input_" + update, False, "data fixture not loaded")
            log.check("fixture.label_" + update, False, "data fixture not loaded")
        return

    train_ref = data.get("train_transformed_ref")
    labels = data.get("train_labels")
    contract_ok = (
        isinstance(train_ref, np.ndarray) and train_ref.dtype == np.float32 and
        tuple(train_ref.shape) == (2, BATCH, 3, 64, 64) and
        isinstance(labels, np.ndarray) and labels.dtype == np.int64 and
        tuple(labels.shape) == (BATCH,)
    )
    log.check("fixture.data_contract", contract_ok,
              "train_transformed_ref=%s/%s train_labels=%s/%s" %
              (getattr(train_ref, "dtype", None), getattr(train_ref, "shape", None),
               getattr(labels, "dtype", None), getattr(labels, "shape", None)))

    indices = {"u1": 0, "u2": 1, "u3": 0}
    for update in UPDATES:
        actual = reference.get("input." + update)
        expected = (train_ref[indices[update]]
                    if contract_ok and train_ref is not None else None)
        ok = (actual is not None and expected is not None and
              actual.dtype == np.float32 and tuple(actual.shape) == (BATCH, 3, 64, 64) and
              np.array_equal(actual, expected))
        if ok:
            detail = "exact"
        elif actual is None or expected is None:
            detail = "missing"
        elif tuple(actual.shape) != tuple(expected.shape):
            detail = "shape=%s want=%s" % (tuple(actual.shape), tuple(expected.shape))
        else:
            detail = "max_abs_diff=" + _fmt(np.max(np.abs(
                actual.astype(np.float64) - expected.astype(np.float64))))
        log.check("fixture.input_" + update, ok, detail)

        actual_label = reference.get("label." + update)
        ok = (actual_label is not None and labels is not None and
              actual_label.dtype == np.int64 and tuple(actual_label.shape) == (BATCH,) and
              np.array_equal(actual_label, labels))
        if ok:
            detail = "exact"
        elif actual_label is None or labels is None:
            detail = "missing"
        elif (tuple(actual_label.shape) != tuple(labels.shape) or
              actual_label.dtype != np.int64):
            detail = "dtype=%s shape=%s want=int64/%s" % (
                actual_label.dtype, tuple(actual_label.shape), tuple(labels.shape))
        else:
            detail = "differing=%d" % int(np.count_nonzero(actual_label != labels))
        log.check("fixture.label_" + update, ok, detail)


def check_eta(log, reference):
    for offset, update in enumerate(UPDATES, start=1):
        actual = reference.get("eta." + update)
        expected = schedule_eta(offset)
        ok = (actual is not None and actual.dtype == np.float64 and
              tuple(actual.shape) in ((), (1,)) and bool(np.all(np.isfinite(actual))) and
              _close(float(np.asarray(actual).reshape(-1)[0]), expected, rtol=ETA_RTOL))
        log.check("schedule.eta_" + update, ok,
                  "value=%s expected=%s rtol=%.1e" %
                  (_fmt(actual) if actual is not None else "missing",
                   _fmt(expected), ETA_RTOL))


def check_finiteness(log, reference, expected):
    bad = []
    checked = 0
    for name, (dtype_name, _) in expected.items():
        if not dtype_name.startswith("float"):
            continue
        array = reference.get(name)
        if array is None:
            continue
        checked += 1
        if not np.all(np.isfinite(array)):
            bad.append(name)
    log.check("arrays.finite", not bad,
              "n_checked=%d nonfinite=%s" % (checked, _shortlist(bad)))


def check_moments(log, reference, bundle_shapes):
    zero_m = []
    zero_v = []
    negative_v = []
    for name, _ in bundle_shapes:
        moment_m = reference.get("m.u3." + name)
        moment_v = reference.get("v.u3." + name)
        if moment_m is None or not np.any(moment_m != 0):
            zero_m.append(name)
        if moment_v is None or not np.any(moment_v != 0):
            zero_v.append(name)
        elif np.any(moment_v < 0):
            negative_v.append(name)
    log.check("optimizer.moments_nonzero", not zero_m and not zero_v,
              "n_zero_m=%d zero_m=%s n_zero_v=%d zero_v=%s" %
              (len(zero_m), _shortlist(zero_m), len(zero_v), _shortlist(zero_v)))
    log.check("optimizer.v_nonnegative", not negative_v,
              "n_negative=%d arrays=%s" % (len(negative_v), _shortlist(negative_v)))


def check_step_counter(log, reference):
    step = reference.get("step.u3")
    scalar_ok = (step is not None and tuple(step.shape) in ((), (1,)))
    ok = (scalar_ok and step.dtype == np.int64 and
          int(np.asarray(step).reshape(-1)[0]) == 3)
    log.check("optimizer.step_u3", ok,
              "value=%s dtype=%s shape=%s" %
              (int(np.asarray(step).reshape(-1)[0]) if scalar_ok else "missing",
               getattr(step, "dtype", None), getattr(step, "shape", None)))


def _names(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        inner = _first(value, ("names", "params", "parameters", "arrays"))
        if isinstance(inner, (list, tuple)):
            return [str(item) for item in inner]
        return [str(key) for key in value]
    return None


def extract_groups(manifest):
    containers = [("manifest", manifest)]
    model = manifest.get("model")
    if isinstance(model, dict):
        containers.insert(0, ("manifest.model", model))
    optimizer = manifest.get("optimizer")
    if isinstance(optimizer, dict):
        containers.append(("manifest.optimizer", optimizer))

    decay_keys = ("decay", "decay_group", "decay_names", "decay_parameters",
                  "with_decay")
    nodecay_keys = ("no_decay", "nodecay", "no_decay_group", "no_decay_names",
                    "no_decay_parameters", "without_decay")

    for label, container in containers:
        for key in ("groups", "param_groups", "parameter_groups"):
            value = container.get(key)
            if isinstance(value, dict):
                decay = _first(value, decay_keys)
                nodecay = _first(value, nodecay_keys)
                if decay is not None and nodecay is not None:
                    return _names(decay), _names(nodecay), label + "." + key
            if isinstance(value, list):
                decay = []
                nodecay = []
                structured = False
                for entry in value:
                    if not isinstance(entry, dict):
                        continue
                    params = _first(entry, ("params", "names", "parameters", "arrays"))
                    weight_decay = _first(entry, ("weight_decay", "lambda", "wd"))
                    if not isinstance(params, (list, tuple)) or weight_decay is None:
                        continue
                    structured = True
                    target = nodecay if _close(weight_decay, 0.0) else decay
                    target.extend(str(item) for item in params)
                if structured:
                    return decay or None, nodecay or None, label + "." + key
        decay = _first(container, decay_keys)
        nodecay = _first(container, nodecay_keys)
        if decay is not None and nodecay is not None:
            return _names(decay), _names(nodecay), label
    return None, None, None


def check_groups(log, manifest, bundle_shapes):
    names = [name for name, _ in bundle_shapes]
    decay_expected = sorted(name for name in names
                            if name.endswith(".weight") and not is_group_norm_name(name))
    nodecay_expected = sorted(name for name in names if name not in set(decay_expected))
    log.check("groups.expected_membership",
              len(decay_expected) == 21 and len(nodecay_expected) == 41 and
              not (set(decay_expected) & set(nodecay_expected)) and
              set(decay_expected) | set(nodecay_expected) == set(names),
              "decay=%d no_decay=%d total=%d" %
              (len(decay_expected), len(nodecay_expected), len(names)))

    decay, nodecay, source = extract_groups(manifest)
    if decay is None or nodecay is None:
        log.check("groups.decay", False, "no 21/41 grouping found in the manifest/model section")
        log.check("groups.no_decay", False, "no 21/41 grouping found in the manifest/model section")
        return
    decay = sorted(decay)
    nodecay = sorted(nodecay)
    log.check("groups.decay", decay == decay_expected,
              "source=%s n=%d missing=%s extra=%s" %
              (source, len(decay),
               _shortlist(sorted(set(decay_expected) - set(decay))),
               _shortlist(sorted(set(decay) - set(decay_expected)))))
    log.check("groups.no_decay", nodecay == nodecay_expected,
              "source=%s n=%d missing=%s extra=%s" %
              (source, len(nodecay),
               _shortlist(sorted(set(nodecay_expected) - set(nodecay))),
               _shortlist(sorted(set(nodecay) - set(nodecay_expected)))))


def check_model_counts(log, manifest, bundle_shapes):
    model = manifest.get("model")
    if not isinstance(model, dict):
        model = {}
    param_keys = ("parameter_count", "param_count", "params", "parameters",
                  "trainable_parameters", "num_parameters", "total_parameters",
                  "model_parameter_count")
    array_keys = ("array_count", "num_arrays", "trainable_arrays", "arrays",
                  "parameter_array_count", "model_array_count")

    parameter_count = None
    array_count = None
    for container in (model, manifest):
        if parameter_count is None:
            parameter_count = _as_count(_first(container, param_keys))
        if array_count is None:
            array_count = _as_count(_first(container, array_keys))

    computed_total = sum(int(np.prod(shape)) for _, shape in bundle_shapes)
    log.check("model.parameter_count",
              parameter_count == PARAM_COUNT and computed_total == PARAM_COUNT,
              "manifest=%s computed=%d want=%d" %
              (parameter_count, computed_total, PARAM_COUNT))
    log.check("model.array_count",
              array_count == ARRAY_COUNT and len(bundle_shapes) == ARRAY_COUNT,
              "manifest=%s computed=%d want=%d" %
              (array_count, len(bundle_shapes), ARRAY_COUNT))


def check_adamw(log, manifest):
    config = manifest.get("adamw")
    if not isinstance(config, dict):
        config = manifest.get("optimizer")
    if not isinstance(config, dict):
        config = manifest.get("optimizer_config")
    if not isinstance(config, dict):
        config = _first(manifest, ("training", "training_contract"))
        config = config.get("optimizer") if isinstance(config, dict) else None
    if isinstance(config, dict) and isinstance(config.get("config"), dict):
        config = config["config"]
    if not isinstance(config, dict) or isinstance(config.get("adamw"), dict):
        if isinstance(config, dict):
            config = config["adamw"]
    if not isinstance(config, dict):
        log.check("manifest.adamw", False, "no AdamW configuration object found")
        return

    problems = []
    lr = _first(config, ("lr", "eta", "learning_rate", "eta_max"))
    if not _close(lr, 1.0e-3):
        problems.append("lr=%r" % (lr,))
    betas = _first(config, ("betas", "beta"))
    if isinstance(betas, dict):
        betas = [betas.get("beta1"), betas.get("beta2")]
    betas_ok = (isinstance(betas, (list, tuple)) and len(betas) == 2 and
                _close(betas[0], 0.9) and _close(betas[1], 0.999))
    if not betas_ok:
        problems.append("betas=%r" % (betas,))
    eps = _first(config, ("eps", "epsilon"))
    if not _close(eps, 1.0e-8):
        problems.append("eps=%r" % (eps,))
    weight_decay = _first(config, ("weight_decay", "lambda", "wd"))
    if isinstance(weight_decay, dict):
        decay_value = _first(weight_decay, ("decay", "with_decay", "lambda",
                                           "weight_decay"))
        nodecay_value = _first(weight_decay, ("no_decay", "nodecay", "without_decay"))
        weight_decay_ok = (_close(decay_value, 1.0e-4) and
                           (nodecay_value is None or _close(nodecay_value, 0.0)))
    else:
        weight_decay_ok = _close(weight_decay, 1.0e-4)
    if not weight_decay_ok:
        problems.append("weight_decay=%r" % (weight_decay,))
    if "amsgrad" in config and config["amsgrad"] is not False:
        problems.append("amsgrad=%r" % (config["amsgrad"],))
    if "amsgrad" not in config:
        problems.append("amsgrad-missing")
    bias_correction = _first(config, ("bias_correction", "correct_bias",
                                      "bias_correction_on"))
    if bias_correction is not None and bias_correction is not True:
        problems.append("bias_correction=%r" % (bias_correction,))
    couple = config.get("couple")
    if couple is not None and couple is not True:
        problems.append("couple=%r" % (couple,))
    log.check("manifest.adamw", not problems, "problems=" + _shortlist(problems))


def _eta_map(value):
    if isinstance(value, dict):
        result = {}
        for update, index in zip(UPDATES, (1, 2, 3)):
            for key in (update, update.upper(), str(index), "t%d" % index):
                if key in value:
                    result[update] = value[key]
                    break
        return result
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return dict(zip(UPDATES, value))
    return None


def check_schedule(log, manifest):
    schedule = manifest.get("schedule")
    if not isinstance(schedule, dict):
        log.check("manifest.schedule", False, "no schedule object found")
        return
    warmup = _first(schedule, ("W", "warmup_updates", "warmup_W",
                               "warmup_updates_b512"))
    total = _first(schedule, ("T", "total_updates", "total_T",
                              "learning_curve_updates",
                              "learning_curve_updates_b512"))
    problems = []
    if _as_count(warmup) != WARMUP_UPDATES:
        problems.append("W=%r" % (warmup,))
    if _as_count(total) != TOTAL_UPDATES:
        problems.append("T=%r" % (total,))

    values = _first(schedule, ("eta", "etas", "eta_per_update", "learning_rates", "lr"))
    if values is None:
        values = {update: schedule.get("eta_" + update) for update in UPDATES
                  if "eta_" + update in schedule}
    eta_map = _eta_map(values)
    if eta_map is None:
        problems.append("eta-per-update-missing")
    else:
        for offset, update in enumerate(UPDATES, start=1):
            expected = schedule_eta(offset)
            value = eta_map.get(update)
            if value is None or not _close(value, expected, rtol=ETA_RTOL):
                problems.append("%s=%r(want %s)" % (update, value, _fmt(expected)))
    log.check("manifest.schedule", not problems,
              "W=%r T=%r problems=%s" %
              (warmup, total, _shortlist(problems)))


def check_optimizer_steps(log, manifest):
    steps = _first(manifest, ("optimizer_steps", "steps", "step_numbers",
                              "optimizer_step_numbers"))
    if steps is None:
        schedule = manifest.get("schedule")
        if isinstance(schedule, dict):
            steps = _first(schedule, ("steps", "optimizer_steps"))
    if steps is None:
        optimizer = manifest.get("optimizer")
        if isinstance(optimizer, dict):
            steps = _first(optimizer, ("steps", "optimizer_steps"))
    if steps is None:
        training = manifest.get("training")
        if isinstance(training, dict):
            steps = _first(training, ("steps", "optimizer_steps",
                                      "optimizer_step_numbers"))
    if steps is None:
        log.check("optimizer.steps", False, "no optimizer step numbers recorded")
        return

    expected = {"u1": 1, "u2": 2, "u3": 3}
    values = {}
    if isinstance(steps, dict):
        for update, index in zip(UPDATES, (1, 2, 3)):
            for key in (update, update.upper(), str(index), "t%d" % index):
                if key in steps:
                    values[update] = steps[key]
                    break
    elif isinstance(steps, (list, tuple)) and len(steps) == 3:
        values = dict(zip(UPDATES, steps))
    problems = []
    for update in UPDATES:
        value = values.get(update)
        if not (isinstance(value, int) and not isinstance(value, bool) and
                value == expected[update]):
            problems.append("%s=%r" % (update, value))
    log.check("optimizer.steps", not problems,
              "steps=%r problems=%s" % (steps, _shortlist(problems)))


def load_npz_dict(path):
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the Phase 3 PyTorch reference artifacts (numpy only)")
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    args = parser.parse_args(argv)

    log = CheckLog()
    fixture_dir = os.path.abspath(args.fixture_dir)
    bundle_path = os.path.abspath(args.bundle)
    reference_path = os.path.abspath(args.reference)
    manifest_path = os.path.abspath(args.manifest)

    paths = {
        "data_fixture.npz": os.path.join(fixture_dir, "data_fixture.npz"),
        "fixture_manifest.json": os.path.join(fixture_dir, "fixture_manifest.json"),
        "initial_bundle.npz": bundle_path,
        "phase3_reference.npz": reference_path,
        "phase3_reference_manifest.json": manifest_path,
    }
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    log.check("files.present", not missing,
              "missing=" + _shortlist(missing) if missing else
              "fixture_dir=%s reference=%s" % (fixture_dir, reference_path))
    if missing:
        print("PHASE3_REFERENCE_CHECK FAIL " + " ".join(log.failed))
        return 1

    data = bundle = reference = manifest = None
    try:
        data = load_npz_dict(paths["data_fixture.npz"])
        log.check("fixture.load", True, "arrays=%d" % len(data))
    except Exception as exc:
        log.check("fixture.load", False, "%s: %s" % (type(exc).__name__, exc))

    try:
        bundle = load_npz_dict(bundle_path)
        log.check("bundle.load", True, "arrays=%d" % len(bundle))
    except Exception as exc:
        log.check("bundle.load", False, "%s: %s" % (type(exc).__name__, exc))

    try:
        reference = load_npz_dict(reference_path)
        log.check("reference.load", True, "arrays=%d" % len(reference))
    except Exception as exc:
        log.check("reference.load", False, "%s: %s" % (type(exc).__name__, exc))

    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        log.check("manifest.load", isinstance(manifest, dict),
                  "type=%s" % (type(manifest).__name__,))
    except Exception as exc:
        log.check("manifest.load", False, "%s: %s" % (type(exc).__name__, exc))

    bundle_shapes = expected_bundle_shapes()
    expected = expected_reference_arrays(bundle_shapes)
    log.check("reference.expected_array_count", len(expected) == 337,
              "n=%d want=337" % (len(expected),))

    if isinstance(bundle, dict):
        check_bundle(log, bundle, bundle_shapes)

    if isinstance(manifest, dict):
        check_manifest_fields(log, manifest)
        check_file_hashes(log, manifest, fixture_dir, bundle_path, reference_path,
                          bundle if isinstance(bundle, dict) else None)
        entries = parse_array_entries(log, manifest, expected)
        if isinstance(reference, dict):
            check_reference_arrays(log, reference, entries, expected)
        check_groups(log, manifest, bundle_shapes)
        check_model_counts(log, manifest, bundle_shapes)
        check_adamw(log, manifest)
        check_schedule(log, manifest)
        check_optimizer_steps(log, manifest)
    else:
        log.check("manifest.object", False, "manifest is not a JSON object")

    if isinstance(reference, dict):
        if isinstance(data, dict):
            check_fixture_inputs(log, data, reference)
        check_eta(log, reference)
        check_finiteness(log, reference, expected)
        check_moments(log, reference, bundle_shapes)
        check_step_counter(log, reference)

    if log.failed:
        print("PHASE3_REFERENCE_CHECK FAIL " + " ".join(log.failed))
        return 1
    print("PHASE3_REFERENCE_CHECK PASS arrays=%d atol=%s rtol=%s" %
          (len(expected), _fmt(args.atol), _fmt(args.rtol)))
    return 0


if __name__ == "__main__":
    sys.exit(main())




