#!/usr/bin/env python3
"""Plot Phase K conditional-graph DDP demonstration artifacts.

The tool reads the train/resume TSV files defined by
``temp/phase_k_k2_interface.md`` and writes deterministic PNG figures plus a
``figures-metadata.txt`` provenance record. It never invents metric values: a
malformed or incomplete schema stops the run with exit status 2 before any
figure is written.

Reproducible invocation in this checkout (system python3 has no numpy or
matplotlib; ``uv`` provides them in an ephemeral environment):

    uv run --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_demo.py \
        --run <RUN_DIR> --out <FIG_DIR> [--title TITLE]

See scripts/plots/README.md for details.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shlex
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    print(
        f"ERROR: missing python dependency {exc.name!r}.\n"
        f"Run with: uv run --with matplotlib --with numpy python3 {sys.argv[0]} ...",
        file=sys.stderr,
    )
    sys.exit(2)

VALIDATION_COLUMNS = (
    "run_id mode schedule_digest dataset_digest step model task n loss accuracy "
    "exposure present checkpoint_step".split()
)
RATE_COLUMNS = "run_id mode schedule_digest step model rank leaf eta".split()
PREDICTION_COLUMNS = (
    "run_id mode schedule_digest step model task sample_index target predicted "
    "pixels".split()
)
METRICS_COLUMNS = (
    "step cycle present_a present_b counts_a counts_b loss_ref loss_local_rank0 "
    "peer_param_maxerr peer_mom1_maxerr peer_mom2_maxerr peer_beta_exact val_a_loss "
    "val_a_acc val_b_loss val_b_acc".split()
)
LEAVES = (
    "trunk_weight",
    "trunk_bias",
    "head_a_weight",
    "head_a_bias",
    "head_b_weight",
    "head_b_bias",
)
SUBSET_COLUMNS = "task split pool_index official_index label".split()
SOURCES_COLUMNS = "task dataset split url filename sha256".split()
FILE_STEMS = ("validation", "learning-rate", "predictions", "metrics")
NEW_STYLE_STEMS = ("validation", "learning-rate", "predictions")
MODEL_ORDER = ("ddp", "reference")
MODEL_DISPLAY = {"ddp": "DDP", "reference": "reference"}
TASK_ORDER = ("a", "b")
COLORS = {"ddp": "#1f77b4", "reference": "#d62728"}
TASK_COLORS = {"a": "#2ca02c", "b": "#9467bd"}
TASK_LABELS = {
    "mnist-fashion": {"a": "Task A (MNIST digits)", "b": "Task B (Fashion-MNIST clothing)"},
    "toy": {"a": "Task A (toy)", "b": "Task B (toy)"},
}
MARKERS = {"ddp": "o", "reference": "s"}
TASK_MARKERS = {"a": "o", "b": "^"}
FIGURE_DPI = 160
RESUME_STYLE = (0, (4.0, 2.0))
RANK_LINESTYLES = ("-", "--", "-.", ":")
RANK_MARKERS = ("o", "s", "^", "D")


class SchemaError(Exception):
    """A Phase K TSV input violates the fixed interface."""


class WarningList(list):
    """A list that prints each recorded warning immediately."""

    def append(self, message):
        super().append(message)
        print(f"WARNING: {message}", file=sys.stderr)


def _die(message: str) -> None:
    raise SchemaError(message)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path, columns, label: str):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _die(f"{path}: cannot read {label}: {exc}")
    lines = text.splitlines()
    if not lines:
        _die(f"{path}: {label} is empty")
    header = lines[0].rstrip("\r").split("\t")
    if header != list(columns):
        _die(
            f"{path}: {label} header mismatch\n"
            f"  expected: {' '.join(columns)}\n"
            f"  found:    {' '.join(header)}"
        )
    rows = []
    for lineno, raw in enumerate(lines[1:], start=2):
        if raw.strip("\r") == "":
            _die(f"{path}:{lineno}: blank row in {label}")
        fields = raw.rstrip("\r").split("\t")
        if len(fields) != len(columns):
            _die(
                f"{path}:{lineno}: expected {len(columns)} tab-separated fields, "
                f"found {len(fields)}"
            )
        rows.append((lineno, dict(zip(columns, fields))))
    if not rows:
        _die(f"{path}: {label} has a header but no data rows")
    return rows


def field_str(path: Path, lineno: int, row, key: str) -> str:
    value = row[key]
    if value == "":
        _die(f"{path}:{lineno}: empty {key}")
    return value


def field_int(path: Path, lineno: int, row, key: str) -> int:
    value = row[key]
    try:
        return int(value)
    except ValueError:
        _die(f"{path}:{lineno}: {key}={value!r} is not an integer")


def field_float(path: Path, lineno: int, row, key: str) -> float:
    value = row[key]
    try:
        parsed = float(value)
    except ValueError:
        _die(f"{path}:{lineno}: {key}={value!r} is not a number")
    if not math.isfinite(parsed):
        _die(f"{path}:{lineno}: {key}={value!r} is not finite")
    return parsed


def field_bool(path: Path, lineno: int, row, key: str) -> bool:
    value = row[key]
    if value == "true":
        return True
    if value == "false":
        return False
    _die(f"{path}:{lineno}: {key}={value!r} is not 'true' or 'false'")


def contiguous(steps) -> bool:
    ordered = sorted(set(steps))
    return ordered == list(range(ordered[0], ordered[-1] + 1))


def parse_validation(path: Path, mode: str):
    raw_rows = read_tsv(path, VALIDATION_COLUMNS, "validation table")
    rows = []
    seen = set()
    run_ids = set()
    schedule_digests = set()
    dataset_digests = set()
    checkpoint_steps = set()
    task_counts = defaultdict(set)
    for lineno, raw in raw_rows:
        if field_str(path, lineno, raw, "mode") != mode:
            _die(f"{path}:{lineno}: mode does not match directory name {mode!r}")
        model = field_str(path, lineno, raw, "model")
        if model not in MODEL_ORDER:
            _die(f"{path}:{lineno}: model={model!r} is not one of {MODEL_ORDER}")
        task = field_str(path, lineno, raw, "task")
        if task not in TASK_ORDER:
            _die(f"{path}:{lineno}: task={task!r} is not one of {TASK_ORDER}")
        step = field_int(path, lineno, raw, "step")
        n = field_int(path, lineno, raw, "n")
        exposure = field_int(path, lineno, raw, "exposure")
        checkpoint_step = field_int(path, lineno, raw, "checkpoint_step")
        loss = field_float(path, lineno, raw, "loss")
        accuracy = field_float(path, lineno, raw, "accuracy")
        present = field_bool(path, lineno, raw, "present")
        if step < 0:
            _die(f"{path}:{lineno}: step={step} is negative")
        if n <= 0:
            _die(f"{path}:{lineno}: n={n} must be positive")
        if exposure < 0:
            _die(f"{path}:{lineno}: exposure={exposure} is negative")
        if checkpoint_step < 0:
            _die(f"{path}:{lineno}: checkpoint_step={checkpoint_step} is negative")
        if not 0.0 <= accuracy <= 1.0:
            _die(f"{path}:{lineno}: accuracy={accuracy} is outside [0, 1]")
        key = (step, model, task)
        if key in seen:
            _die(f"{path}:{lineno}: duplicate validation key {key}")
        seen.add(key)
        run_ids.add(field_str(path, lineno, raw, "run_id"))
        schedule_digests.add(field_str(path, lineno, raw, "schedule_digest"))
        dataset_digests.add(field_str(path, lineno, raw, "dataset_digest"))
        checkpoint_steps.add(checkpoint_step)
        task_counts[task].add(n)
        rows.append(
            {
                "run_id": raw["run_id"],
                "schedule_digest": raw["schedule_digest"],
                "dataset_digest": raw["dataset_digest"],
                "step": step,
                "model": model,
                "task": task,
                "n": n,
                "loss": loss,
                "accuracy": accuracy,
                "exposure": exposure,
                "present": present,
                "checkpoint_step": checkpoint_step,
            }
        )
    if len(checkpoint_steps) != 1:
        _die(f"{path}: checkpoint_step is not a single value: {sorted(checkpoint_steps)}")
    for task, counts in task_counts.items():
        if len(counts) != 1:
            _die(f"{path}: task {task} has inconsistent validation sample counts {sorted(counts)}")
    steps = sorted({row["step"] for row in rows})
    if not contiguous(steps):
        _die(f"{path}: validation steps are not contiguous: {steps[:3]}...{steps[-3:]}")
    for step in steps:
        expected = {(step, model, task) for model in MODEL_ORDER for task in TASK_ORDER}
        actual = {key for key in seen if key[0] == step}
        if actual != expected:
            _die(
                f"{path}: step {step} is incomplete; expected keys {sorted(expected)}, "
                f"found {sorted(actual)}"
            )
        for task in TASK_ORDER:
            exposures = {
                row["exposure"]
                for row in rows
                if row["step"] == step and row["task"] == task
            }
            if len(exposures) != 1:
                _die(f"{path}: exposure differs between models at step {step} task {task}")
            present_values = {
                row["present"]
                for row in rows
                if row["step"] == step and row["task"] == task
            }
            if len(present_values) != 1:
                _die(f"{path}: present differs between models at step {step} task {task}")
    return {
        "path": path,
        "rows": rows,
        "steps": steps,
        "checkpoint_step": checkpoint_steps.pop(),
        "run_ids": run_ids,
        "schedule_digests": schedule_digests,
        "dataset_digests": dataset_digests,
        "row_count": len(rows),
    }


def parse_rates(path: Path, mode: str):
    raw_rows = read_tsv(path, RATE_COLUMNS, "learning-rate table")
    rows = []
    seen = set()
    per_step = defaultdict(list)
    run_ids = set()
    schedule_digests = set()
    for lineno, raw in raw_rows:
        if field_str(path, lineno, raw, "mode") != mode:
            _die(f"{path}:{lineno}: mode does not match directory name {mode!r}")
        model = field_str(path, lineno, raw, "model")
        if model not in MODEL_ORDER:
            _die(f"{path}:{lineno}: model={model!r} is not one of {MODEL_ORDER}")
        rank = field_int(path, lineno, raw, "rank")
        leaf = field_str(path, lineno, raw, "leaf")
        if leaf not in LEAVES:
            _die(f"{path}:{lineno}: leaf={leaf!r} is not a canonical Phase K leaf")
        eta = field_float(path, lineno, raw, "eta")
        step = field_int(path, lineno, raw, "step")
        if rank < 0:
            _die(f"{path}:{lineno}: rank={rank} is negative")
        if model == "reference" and rank != 0:
            _die(f"{path}:{lineno}: reference must report rank 0, found {rank}")
        if eta <= 0:
            _die(f"{path}:{lineno}: eta={eta} must be positive")
        key = (step, model, rank, leaf)
        if key in seen:
            _die(f"{path}:{lineno}: duplicate learning-rate key {key}")
        seen.add(key)
        run_ids.add(field_str(path, lineno, raw, "run_id"))
        schedule_digests.add(field_str(path, lineno, raw, "schedule_digest"))
        clean = {
            "step": step,
            "model": model,
            "rank": rank,
            "leaf": leaf,
            "eta": eta,
        }
        rows.append(clean)
        per_step[step].append(clean)
    steps = sorted(per_step)
    if not contiguous(steps):
        _die(f"{path}: learning-rate steps are not contiguous")
    ddp_ranks = sorted({row["rank"] for row in rows if row["model"] == "ddp"})
    if not ddp_ranks or ddp_ranks != list(range(ddp_ranks[-1] + 1)):
        _die(f"{path}: DDP ranks must be contiguous from 0, found {ddp_ranks}")
    workers = ddp_ranks[-1] + 1
    for step, step_rows in per_step.items():
        etas = {row["eta"] for row in step_rows}
        if len(etas) != 1:
            _die(f"{path}: eta differs across ranks/leaves at step {step}: {sorted(etas)}")
        expected = {
            (model, rank, leaf)
            for model in MODEL_ORDER
            for rank in (range(workers) if model == "ddp" else [0])
            for leaf in LEAVES
        }
        actual = {(row["model"], row["rank"], row["leaf"]) for row in step_rows}
        if actual != expected:
            _die(
                f"{path}: step {step} learning-rate rows incomplete; "
                f"missing {sorted(expected - actual)}, extra {sorted(actual - expected)}"
            )
    return {
        "path": path,
        "rows": rows,
        "steps": steps,
        "workers": workers,
        "etas": sorted({row["eta"] for row in rows}),
        "run_ids": run_ids,
        "schedule_digests": schedule_digests,
        "row_count": len(rows),
    }


def parse_predictions(path: Path, mode: str, nclasses=None):
    raw_rows = read_tsv(path, PREDICTION_COLUMNS, "predictions table")
    rows = []
    seen = set()
    run_ids = set()
    schedule_digests = set()
    for lineno, raw in raw_rows:
        if field_str(path, lineno, raw, "mode") != mode:
            _die(f"{path}:{lineno}: mode does not match directory name {mode!r}")
        model = field_str(path, lineno, raw, "model")
        if model not in MODEL_ORDER:
            _die(f"{path}:{lineno}: model={model!r} is not one of {MODEL_ORDER}")
        task = field_str(path, lineno, raw, "task")
        if task not in TASK_ORDER:
            _die(f"{path}:{lineno}: task={task!r} is not one of {TASK_ORDER}")
        step = field_int(path, lineno, raw, "step")
        sample_index = field_int(path, lineno, raw, "sample_index")
        target = field_int(path, lineno, raw, "target")
        predicted = field_int(path, lineno, raw, "predicted")
        if sample_index < 0:
            _die(f"{path}:{lineno}: sample_index={sample_index} is negative")
        if target < 0 or predicted < 0:
            _die(f"{path}:{lineno}: labels must be non-negative")
        if nclasses is not None:
            limit = nclasses.get(task)
            if limit is not None and (target >= limit or predicted >= limit):
                _die(f"{path}:{lineno}: label outside [0, {limit}) for task {task}")
        pixel_text = raw["pixels"]
        if pixel_text == "":
            _die(f"{path}:{lineno}: empty pixels field")
        try:
            pixels = [float(value) for value in pixel_text.split(",")]
        except ValueError:
            _die(f"{path}:{lineno}: pixels are not comma-separated numbers")
        if not all(math.isfinite(value) for value in pixels):
            _die(f"{path}:{lineno}: pixels contain a non-finite value")
        key = (step, model, task, sample_index)
        if key in seen:
            _die(f"{path}:{lineno}: duplicate predictions key {key}")
        seen.add(key)
        run_ids.add(field_str(path, lineno, raw, "run_id"))
        schedule_digests.add(field_str(path, lineno, raw, "schedule_digest"))
        rows.append(
            {
                "step": step,
                "model": model,
                "task": task,
                "sample_index": sample_index,
                "target": target,
                "predicted": predicted,
                "pixels": pixels,
            }
        )
    steps = sorted({row["step"] for row in rows})
    if len(steps) != 2:
        _die(f"{path}: expected exactly two prediction snapshots, found {steps}")
    for step in steps:
        for model in MODEL_ORDER:
            for task in TASK_ORDER:
                group = [
                    row
                    for row in rows
                    if row["step"] == step and row["model"] == model and row["task"] == task
                ]
                if not group:
                    _die(f"{path}: missing predictions for step {step} {model} task {task}")
                if len(group) > 20:
                    _die(f"{path}: step {step} {model} task {task} has more than 20 samples")
    for task in TASK_ORDER:
        pixel_counts = {
            len(row["pixels"]) for row in rows if row["task"] == task
        }
        if len(pixel_counts) != 1:
            _die(f"{path}: task {task} has inconsistent pixel counts {sorted(pixel_counts)}")
        index_sequences = set()
        for step in steps:
            for model in MODEL_ORDER:
                indices = tuple(
                    row["sample_index"]
                    for row in rows
                    if row["step"] == step and row["model"] == model and row["task"] == task
                )
                index_sequences.add(indices)
        if len(index_sequences) != 1:
            _die(
                f"{path}: task {task} sample index sequence differs across "
                "snapshots/models (order matters for the first-20 panel)"
            )
    sample_signatures = defaultdict(set)
    for row in rows:
        sample_signatures[(row["task"], row["sample_index"])].add(
            (row["target"], tuple(row["pixels"]))
        )
    for key, signatures in sample_signatures.items():
        if len(signatures) != 1:
            _die(f"{path}: target/pixels are inconsistent for task/sample {key}")
    return {
        "path": path,
        "rows": rows,
        "snapshot_steps": steps,
        "nclasses": nclasses,
        "run_ids": run_ids,
        "schedule_digests": schedule_digests,
        "row_count": len(rows),
    }


def parse_metrics(path: Path, mode: str):
    raw_rows = read_tsv(path, METRICS_COLUMNS, "metrics table")
    rows = []
    seen = set()
    beta_false_steps = []
    for lineno, raw in raw_rows:
        step = field_int(path, lineno, raw, "step")
        cycle = field_int(path, lineno, raw, "cycle")
        present_a = field_bool(path, lineno, raw, "present_a")
        present_b = field_bool(path, lineno, raw, "present_b")
        counts_a = field_int(path, lineno, raw, "counts_a")
        counts_b = field_int(path, lineno, raw, "counts_b")
        loss_ref = field_float(path, lineno, raw, "loss_ref")
        loss_local = field_float(path, lineno, raw, "loss_local_rank0")
        param_err = field_float(path, lineno, raw, "peer_param_maxerr")
        mom1_err = field_float(path, lineno, raw, "peer_mom1_maxerr")
        mom2_err = field_float(path, lineno, raw, "peer_mom2_maxerr")
        beta_exact = field_bool(path, lineno, raw, "peer_beta_exact")
        val_a_loss = field_float(path, lineno, raw, "val_a_loss")
        val_a_acc = field_float(path, lineno, raw, "val_a_acc")
        val_b_loss = field_float(path, lineno, raw, "val_b_loss")
        val_b_acc = field_float(path, lineno, raw, "val_b_acc")
        if step < 1:
            _die(f"{path}:{lineno}: step={step} must be >= 1")
        if cycle != ((step - 1) % 6) + 1:
            _die(f"{path}:{lineno}: cycle={cycle} does not match step={step}")
        if counts_a < 0 or counts_b < 0:
            _die(f"{path}:{lineno}: counts must be non-negative")
        if present_a != (counts_a > 0):
            _die(f"{path}:{lineno}: present_a={present_a} is inconsistent with counts_a={counts_a}")
        if present_b != (counts_b > 0):
            _die(f"{path}:{lineno}: present_b={present_b} is inconsistent with counts_b={counts_b}")
        if param_err < 0 or mom1_err < 0 or mom2_err < 0:
            _die(f"{path}:{lineno}: peer max errors must be non-negative")
        if not beta_exact:
            beta_false_steps.append(step)
        for key, value in (
            ("val_a_acc", val_a_acc),
            ("val_b_acc", val_b_acc),
        ):
            if not 0.0 <= value <= 1.0:
                _die(f"{path}:{lineno}: {key}={value} is outside [0, 1]")
        if step in seen:
            _die(f"{path}:{lineno}: duplicate metrics step {step}")
        seen.add(step)
        rows.append(
            {
                "step": step,
                "cycle": cycle,
                "present_a": present_a,
                "present_b": present_b,
                "counts_a": counts_a,
                "counts_b": counts_b,
                "loss_ref": loss_ref,
                "loss_local_rank0": loss_local,
                "peer_param_maxerr": param_err,
                "peer_mom1_maxerr": mom1_err,
                "peer_mom2_maxerr": mom2_err,
                "peer_beta_exact": beta_exact,
                "val_a_loss": val_a_loss,
                "val_a_acc": val_a_acc,
                "val_b_loss": val_b_loss,
                "val_b_acc": val_b_acc,
            }
        )
    steps = sorted(seen)
    if not contiguous(steps):
        _die(f"{path}: metrics steps are not contiguous")
    absent_steps = {
        "a": [row["step"] for row in rows if not row["present_a"]],
        "b": [row["step"] for row in rows if not row["present_b"]],
    }
    return {
        "path": path,
        "rows": rows,
        "steps": steps,
        "absent_steps": absent_steps,
        "beta_false_steps": beta_false_steps,
        "row_count": len(rows),
    }


def parse_metadata(path: Path, mode: str, warnings):
    info = {"present": True, "path": path}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"{path}: cannot read optional metadata.txt: {exc}")
        return {"present": False, "path": path}
    for raw in lines:
        fields = raw.rstrip("\r").split("\t")
        if not fields or fields[0] == "":
            continue
        key = fields[0]
        if key == "dataset" and len(fields) >= 2:
            info["dataset"] = fields[1]
            for extra in fields[2:]:
                if extra.startswith("digest="):
                    info["dataset_digest"] = extra[len("digest=") :]
        elif key == "schedule" and len(fields) >= 2:
            for extra in fields[1:]:
                if "=" in extra:
                    name, value = extra.split("=", 1)
                    info[name] = value
        elif key == "extra" and len(fields) >= 3:
            info.setdefault("extra", {})[fields[1]] = fields[2]
        elif key in ("workers", "dtype", "backend", "mode", "label", "julia", "host") and len(fields) >= 2:
            info[key] = fields[1]
    extra = info.get("extra", {})
    for name in ("run_id", "schedule_digest", "prepared_dataset_digest", "optimizer", "on_gpu"):
        if name in extra:
            info[name] = extra[name]
    if "optimizer" in info:
        for token in str(info["optimizer"]).split():
            if token.startswith("eta="):
                try:
                    info["eta"] = float(token[len("eta=") :])
                except ValueError:
                    warnings.append(f"{path}: cannot parse optimizer eta from {token!r}")
    for name in ("steps", "workers", "global_batch", "local_batch"):
        if name in info:
            try:
                info[name] = int(info[name])
            except (TypeError, ValueError):
                warnings.append(f"{path}: cannot parse {name}={info[name]!r} as an integer")
                del info[name]
    if info.get("mode") is not None and info["mode"] != mode:
        _die(f"{path}: metadata mode={info['mode']!r} does not match directory {mode!r}")
    return info


class ModeData:
    def __init__(self, mode, directory, paths):
        self.mode = mode
        self.directory = directory
        self.paths = paths
        self.validation = None
        self.rates = None
        self.predictions = None
        self.metrics = None
        self.metadata = {"present": False}
        self.row_counts = {}


class Pair:
    def __init__(self, label, run_root, pair_id, status):
        self.label = label
        self.run_root = run_root
        self.pair_id = pair_id
        self.status = status
        self.modes = {}
        self.prepared_dir = None
        self.optional_files = []
        self.workers = None
        self.steps = None
        self.checkpoint_step = None
        self.run_id = None
        self.schedule_digest = None
        self.dataset_digest = None
        self.dataset_name = None
        self.backend = None
        self.dtype = None
        self.eta = None
        self.nclasses = None
        self.pixel_counts = {}


def mode_file_paths(directory, mode, warnings):
    paths = {stem: Path(directory) / f"{stem}-{mode}.tsv" for stem in FILE_STEMS}
    present_new_style = [paths[stem].is_file() for stem in NEW_STYLE_STEMS]
    if not any(present_new_style):
        warnings.append(
            f"skipping {directory}: no K2-style validation/learning-rate/predictions TSVs "
            "(legacy or unrelated directory)"
        )
        return None
    missing = [str(paths[stem]) for stem in FILE_STEMS if not paths[stem].is_file()]
    if missing:
        _die(
            f"{directory}: partial K2 output; missing required file(s): "
            + ", ".join(missing)
        )
    return paths


def has_new_style_files(directory, mode):
    return any(
        (Path(directory) / f"{stem}-{mode}.tsv").is_file() for stem in NEW_STYLE_STEMS
    )


def discover_pairs(run_root, warnings):
    pairs = {}

    def register(directory, mode, pair_id):
        entry = pairs.setdefault(os.path.abspath(pair_id), {})
        directory = os.path.abspath(directory)
        if mode in entry:
            if entry[mode] == directory:
                return
            existing_ok = has_new_style_files(entry[mode], mode)
            new_ok = has_new_style_files(directory, mode)
            if new_ok and not existing_ok:
                warnings.append(
                    f"{run_root}: replacing legacy {mode} directory for run {pair_id}: "
                    f"{entry[mode]} -> {directory}"
                )
                entry[mode] = directory
            else:
                warnings.append(
                    f"{run_root}: duplicate {mode} directory for run {pair_id}: "
                    f"{entry[mode]} and {directory}; keeping {entry[mode]}"
                )
            return
        entry[mode] = directory

    for dirpath, dirnames, _ in os.walk(run_root):
        rel = os.path.relpath(dirpath, run_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 3:
            dirnames[:] = []
            continue
        base = os.path.basename(dirpath.rstrip(os.sep)) or dirpath
        mode = None
        pair_id = None
        if base in ("train", "resume"):
            mode = base
            pair_id = os.path.dirname(dirpath)
        elif base.endswith("-train") or base.endswith("-resume"):
            mode = "train" if base.endswith("-train") else "resume"
            pair_id = os.path.join(os.path.dirname(dirpath), base[: -len("-" + mode)])
        if mode is not None:
            register(dirpath, mode, pair_id)
            dirnames[:] = []
    for pair_id, entry in pairs.items():
        for mode in ("train", "resume"):
            if mode in entry:
                continue
            for candidate in (os.path.join(pair_id, mode), pair_id + "-" + mode):
                if os.path.isdir(candidate):
                    entry[mode] = os.path.abspath(candidate)
                    break
    return pairs


def parse_subset_classes(path: Path, warnings):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"{path}: cannot read optional subset_indices.tsv: {exc}")
        return None
    if not lines:
        warnings.append(f"{path}: optional subset_indices.tsv is empty")
        return None
    header = lines[0].rstrip("\r").split("\t")
    if header != SUBSET_COLUMNS:
        warnings.append(
            f"{path}: optional subset_indices.tsv header {header} does not match "
            f"{SUBSET_COLUMNS}; class range check disabled"
        )
        return None
    class_max = {}
    for lineno, raw in enumerate(lines[1:], start=2):
        fields = raw.rstrip("\r").split("\t")
        if len(fields) != len(SUBSET_COLUMNS):
            warnings.append(f"{path}:{lineno}: malformed optional subset row; class range check disabled")
            return None
        task, label = fields[0], fields[4]
        if task not in TASK_ORDER:
            warnings.append(f"{path}:{lineno}: unknown task {task!r}; class range check disabled")
            return None
        try:
            label_value = int(label)
        except ValueError:
            warnings.append(f"{path}:{lineno}: label={label!r} is not an integer; class range check disabled")
            return None
        if label_value < 0:
            warnings.append(f"{path}:{lineno}: negative label; class range check disabled")
            return None
        class_max[task] = max(class_max.get(task, -1), label_value)
    if not class_max:
        warnings.append(f"{path}: optional subset_indices.tsv has no data rows")
        return None
    return {task: limit + 1 for task, limit in class_max.items()}


def find_prepared_dir(pair_id, run_root):
    candidates = [
        os.path.join(pair_id, "..", "prepared"),
        os.path.join(pair_id, "..", "..", "prepared"),
        os.path.join(run_root, "prepared"),
        os.path.join(os.path.dirname(os.path.abspath(run_root)), "prepared"),
    ]
    seen = set()
    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if os.path.isdir(resolved):
            return Path(resolved)
    return None


def load_mode(mode, directory, paths, nclasses, warnings):
    data = ModeData(mode, directory, paths)
    data.validation = parse_validation(paths["validation"], mode)
    data.rates = parse_rates(paths["learning-rate"], mode)
    data.predictions = parse_predictions(paths["predictions"], mode, nclasses=nclasses)
    data.metrics = parse_metrics(paths["metrics"], mode)
    if data.metrics["beta_false_steps"]:
        warnings.append(
            f"{paths['metrics']}: peer_beta_exact is false at steps "
            f"{data.metrics['beta_false_steps']}; plotted as measured (not an error)"
        )
    data.row_counts = {
        "validation": data.validation["row_count"],
        "learning-rate": data.rates["row_count"],
        "predictions": data.predictions["row_count"],
        "metrics": data.metrics["row_count"],
    }
    metadata_path = Path(directory) / "metadata.txt"
    if metadata_path.is_file():
        data.metadata = parse_metadata(metadata_path, mode, warnings)
    else:
        warnings.append(f"{metadata_path}: optional run metadata.txt is absent")
        data.metadata = {"present": False, "path": metadata_path}
    return data


def identity_union(mode_datas, key):
    values = set()
    for data in mode_datas:
        for table in (data.validation, data.rates, data.predictions):
            if key in table:
                values |= table[key]
    return values


def single_identity(mode_datas, key, label, warnings):
    values = identity_union(mode_datas, key)
    if len(values) != 1:
        _die(f"{label}: {key} is not unique across train/resume inputs: {sorted(values)}")
    return values.pop()


def load_pair(pair, warnings):
    mode_datas = []
    for mode in ("train", "resume"):
        if mode not in pair.status:
            continue
        data = load_mode(
            mode,
            pair.status[mode]["directory"],
            pair.status[mode],
            pair.nclasses,
            warnings,
        )
        pair.modes[mode] = data
        mode_datas.append(data)
    if not mode_datas:
        _die(f"{pair.label}: no train or resume data loaded")
    pair.run_id = single_identity(mode_datas, "run_ids", pair.label, warnings)
    pair.schedule_digest = single_identity(mode_datas, "schedule_digests", pair.label, warnings)
    pair.dataset_digest = single_identity(
        [data for data in mode_datas if data.validation["dataset_digests"]],
        "dataset_digests",
        pair.label,
        warnings,
    )
    checkpoint_steps = {data.validation["checkpoint_step"] for data in mode_datas}
    if len(checkpoint_steps) != 1:
        _die(
            f"{pair.label}: checkpoint_step differs between train and resume: "
            f"{sorted(checkpoint_steps)}"
        )
    pair.checkpoint_step = checkpoint_steps.pop()
    if pair.checkpoint_step < 1:
        _die(f"{pair.label}: checkpoint_step={pair.checkpoint_step} is not positive")
    horizons = {data.metrics["steps"][-1] for data in mode_datas}
    if len(horizons) != 1:
        _die(f"{pair.label}: train and resume end at different steps: {sorted(horizons)}")
    pair.steps = horizons.pop()
    workers = {data.rates["workers"] for data in mode_datas}
    if len(workers) != 1:
        _die(f"{pair.label}: train and resume report different rank counts: {sorted(workers)}")
    pair.workers = workers.pop()
    for mode, data in pair.modes.items():
        first_metrics = 1 if mode == "train" else pair.checkpoint_step + 1
        expected_metrics = list(range(first_metrics, pair.steps + 1))
        if data.metrics["steps"] != expected_metrics:
            _die(
                f"{pair.label}/{mode}: metrics steps {data.metrics['steps'][:2]}..."
                f"{data.metrics['steps'][-2:]} do not match expected "
                f"{expected_metrics[:2]}...{expected_metrics[-2:]}"
            )
        first_validation = 0 if mode == "train" else pair.checkpoint_step
        expected_validation = list(range(first_validation, pair.steps + 1))
        if data.validation["steps"] != expected_validation:
            _die(
                f"{pair.label}/{mode}: validation steps do not match expected "
                f"0/checkpoint..{pair.steps}"
            )
        if data.rates["steps"] != expected_metrics:
            _die(f"{pair.label}/{mode}: learning-rate steps do not match metrics steps")
        expected_snapshots = [0, pair.steps] if mode == "train" else [pair.checkpoint_step, pair.steps]
        if data.predictions["snapshot_steps"] != expected_snapshots:
            _die(
                f"{pair.label}/{mode}: prediction snapshots {data.predictions['snapshot_steps']} "
                f"do not match expected {expected_snapshots}"
            )
        for task in TASK_ORDER:
            counts = {
                len(row["pixels"])
                for row in data.predictions["rows"]
                if row["task"] == task
            }
            if len(counts) != 1:
                _die(f"{pair.label}/{mode}: task {task} pixel counts differ: {sorted(counts)}")
    all_pixel_counts = {}
    for task in TASK_ORDER:
        counts = set()
        for data in mode_datas:
            counts |= {
                len(row["pixels"]) for row in data.predictions["rows"] if row["task"] == task
            }
        if len(counts) != 1:
            _die(
                f"{pair.label}: train/resume disagree on pixel count for task {task}: "
                f"{sorted(counts)}"
            )
        all_pixel_counts[task] = counts.pop()
    pair.pixel_counts = all_pixel_counts
    if len(set(all_pixel_counts.values())) != 1:
        _die(
            f"{pair.label}: tasks use different pixel counts "
            f"{all_pixel_counts}; cannot build one prediction panel type"
        )
    dataset_names = {data.metadata.get("dataset") for data in mode_datas if data.metadata.get("present")}
    if len(dataset_names) > 1:
        _die(f"{pair.label}: metadata reports different datasets: {sorted(dataset_names)}")
    pair.dataset_name = next(iter(dataset_names)) if dataset_names else None
    backends = {data.metadata.get("backend") for data in mode_datas if data.metadata.get("present")}
    if len(backends) > 1:
        _die(f"{pair.label}: metadata reports different backends: {sorted(backends)}")
    pair.backend = next(iter(backends)) if backends else None
    dtypes = {data.metadata.get("dtype") for data in mode_datas if data.metadata.get("present")}
    pair.dtype = next(iter(dtypes)) if dtypes else None
    etas = set()
    for data in mode_datas:
        etas |= set(data.rates["etas"])
    pair.eta = next(iter(etas)) if len(etas) == 1 else None
    for mode, data in pair.modes.items():
        if not data.metadata.get("present"):
            continue
        meta = data.metadata
        if meta.get("run_id") is not None and meta["run_id"] != pair.run_id:
            _die(f"{pair.label}/{mode}: metadata run_id differs from TSV run_id")
        if meta.get("schedule_digest") is not None and meta["schedule_digest"] != pair.schedule_digest:
            _die(f"{pair.label}/{mode}: metadata schedule_digest differs from TSV digest")
        if meta.get("dataset_digest") is not None and meta["dataset_digest"] != pair.dataset_digest:
            _die(f"{pair.label}/{mode}: metadata dataset digest differs from TSV digest")
        if meta.get("workers") is not None and meta["workers"] != pair.workers:
            _die(
                f"{pair.label}/{mode}: metadata workers={meta['workers']} but TSV ranks "
                f"imply {pair.workers}"
            )
        local_batch = meta.get("local_batch")
        global_batch = meta.get("global_batch")
        if local_batch is not None and global_batch is not None:
            if global_batch != local_batch * pair.workers:
                _die(
                    f"{pair.label}/{mode}: global_batch={global_batch} != "
                    f"local_batch={local_batch} x workers={pair.workers}"
                )
        if local_batch is None:
            warnings.append(
                f"{pair.label}/{mode}: metadata lacks local_batch; exposure cross-check skipped"
            )
            continue
        counts = {
            row["step"]: (row["counts_a"], row["counts_b"]) for row in data.metrics["rows"]
        }
        exposures = defaultdict(set)
        for row in data.validation["rows"]:
            exposures[(row["step"], row["task"])].add(row["exposure"])
        for task in TASK_ORDER:
            index = TASK_ORDER.index(task)
            for step in data.validation["steps"]:
                value = next(iter(exposures[(step, task)]))
                if step == 0:
                    if value != 0:
                        _die(
                            f"{pair.label}/{mode}: step 0 exposure for task {task} is "
                            f"{value}, expected 0"
                        )
                    continue
                previous = exposures.get((step - 1, task))
                if previous is None:
                    continue
                expected_delta = counts.get(step, (None, None))[index]
                if expected_delta is None:
                    continue
                delta = value - next(iter(previous))
                if delta != expected_delta * local_batch:
                    _die(
                        f"{pair.label}/{mode}: exposure delta {delta} at step {step} task "
                        f"{task} != counts {expected_delta} x local_batch {local_batch}"
                    )
    pair.prepared_dir = find_prepared_dir(pair.pair_id, pair.run_root)
    if pair.prepared_dir is None:
        warnings.append(
            f"{pair.label}: no prepared directory found; sources.tsv and "
            "subset_indices.tsv not recorded"
        )
    return pair


def model_series(data, model, task, field):
    rows = [
        row for row in data.validation["rows"] if row["model"] == model and row["task"] == task
    ]
    rows.sort(key=lambda row: row["step"])
    xs = np.array([row["step"] for row in rows], dtype=float)
    ys = np.array([row[field] for row in rows], dtype=float)
    return xs, ys


def difference_series(data, task, field):
    x_ddp, y_ddp = model_series(data, "ddp", task, field)
    x_ref, y_ref = model_series(data, "reference", task, field)
    if not np.array_equal(x_ddp, x_ref):
        _die(f"{data.validation['path']}: DDP and reference steps differ for task {task}")
    return x_ddp, y_ddp - y_ref


def task_title(pair, task):
    mapping = TASK_LABELS.get(pair.dataset_name)
    if mapping:
        return mapping[task]
    return f"Task {task.upper()}"


def figure_title(user_title, pair, kind):
    bits = []
    if user_title:
        bits.append(user_title)
    bits.append(pair.label)
    descriptors = []
    if pair.dataset_name:
        descriptors.append(pair.dataset_name)
    if pair.backend:
        descriptors.append(pair.backend)
    if pair.workers:
        descriptors.append(f"{pair.workers} ranks")
    if descriptors:
        bits.append(", ".join(descriptors))
    bits.append(kind)
    return " - ".join(bits)


def absent_union(pair, task):
    steps = set()
    for data in pair.modes.values():
        steps.update(data.metrics["absent_steps"][task])
    return sorted(steps)


def annotate_checkpoint(ax, pair, y=0.97):
    ax.axvline(pair.checkpoint_step, color="black", linestyle=":", linewidth=1.0)
    ax.annotate(
        f"checkpoint/resume boundary (step {pair.checkpoint_step})",
        xy=(pair.checkpoint_step, y),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=7,
        color="black",
        va="top",
    )


def shade_absences(ax, pair, tasks, alpha=0.10):
    handles = []
    for task in tasks:
        steps = absent_union(pair, task)
        if not steps:
            continue
        for step in steps:
            ax.axvspan(step - 0.5, step + 0.5, color=TASK_COLORS[task], alpha=alpha, lw=0)
        handles.append(
            Patch(
                facecolor=TASK_COLORS[task],
                alpha=0.35,
                label=f"Task {task.upper()} globally absent",
            )
        )
    return handles


def save_figure(fig, path: Path, outputs):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)


def fig_validation(pair, out_dir, field, ylabel, kind, user_title, outputs):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, task in zip(axes, TASK_ORDER):
        for mode in ("train", "resume"):
            data = pair.modes.get(mode)
            if data is None:
                continue
            for model in MODEL_ORDER:
                xs, ys = model_series(data, model, task, field)
                reference = model == "reference"
                linewidth = 2.6 if reference else 1.2
                alpha = 0.55 if reference else 1.0
                zorder = 2 if reference else 3
                if mode == "train":
                    ax.plot(
                        xs,
                        ys,
                        color=COLORS[model],
                        linestyle="-",
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=zorder,
                        label=f"{MODEL_DISPLAY[model]} (train, solid)",
                    )
                else:
                    ax.plot(
                        xs,
                        ys,
                        color=COLORS[model],
                        linestyle=RESUME_STYLE,
                        marker=MARKERS[model],
                        markersize=3.0 if reference else 3.4,
                        markevery=max(1, len(xs) // 12),
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=zorder,
                        label=f"{MODEL_DISPLAY[model]} (resume, dashed)",
                    )
        shade_absences(ax, pair, (task,))
        annotate_checkpoint(ax, pair)
        differences = []
        for mode, data in pair.modes.items():
            _, diff = difference_series(data, task, field)
            differences.append(np.max(np.abs(diff)))
        max_diff = max(differences) if differences else float("nan")
        ax.text(
            0.02,
            0.03,
            f"max |DDP - reference| = {max_diff:.3e}",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.set_title(task_title(pair, task))
        ax.set_xlabel("global update index")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(figure_title(user_title, pair, kind), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_dir / f"{pair.label}-{kind}.png", outputs)


def fig_difference(pair, out_dir, user_title, outputs):
    fields = (
        ("loss", "DDP - reference loss"),
        ("accuracy", "DDP - reference accuracy"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4))
    for row_index, (field, ylabel) in enumerate(fields):
        for column_index, task in enumerate(TASK_ORDER):
            ax = axes[row_index][column_index]
            for mode in ("train", "resume"):
                data = pair.modes.get(mode)
                if data is None:
                    continue
                xs, diff = difference_series(data, task, field)
                if mode == "train":
                    ax.plot(
                        xs,
                        diff,
                        color=COLORS["ddp"],
                        linestyle="-",
                        linewidth=1.3,
                        label="DDP - reference (train)",
                    )
                else:
                    ax.plot(
                        xs,
                        diff,
                        color=COLORS["ddp"],
                        linestyle=RESUME_STYLE,
                        marker=MARKERS["ddp"],
                        markersize=3.0,
                        markevery=max(1, len(xs) // 12),
                        linewidth=1.1,
                        label="DDP - reference (resume)",
                    )
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            shade_absences(ax, pair, (task,))
            if row_index == 0:
                annotate_checkpoint(ax, pair, y=0.99)
            ax.set_title(task_title(pair, task))
            ax.set_xlabel("global update index")
            ax.set_ylabel(ylabel)
            limits = ax.get_ylim()
            if limits[0] == limits[1]:
                ax.set_ylim(limits[0] - 1e-12, limits[1] + 1e-12)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(figure_title(user_title, pair, "validation differences"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out_dir / f"{pair.label}-validation-difference.png", outputs)


def fig_learning_rate(pair, out_dir, user_title, outputs):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    palette = plt.get_cmap("tab10").colors
    for model in MODEL_ORDER:
        ranks = sorted(
            {
                row["rank"]
                for data in pair.modes.values()
                for row in data.rates["rows"]
                if row["model"] == model
            }
        )
        for rank in ranks:
            for mode in ("train", "resume"):
                data = pair.modes.get(mode)
                if data is None:
                    continue
                rows = sorted(
                    [
                        row
                        for row in data.rates["rows"]
                        if row["model"] == model and row["rank"] == rank
                    ],
                    key=lambda row: row["step"],
                )
                xs = np.array([row["step"] for row in rows], dtype=float)
                ys = np.array([row["eta"] for row in rows], dtype=float)
                if model == "reference":
                    color = "black"
                    linestyle = (0, (5.0, 1.2, 1.0, 1.2))
                    marker = "*"
                    linewidth = 1.6
                    markersize = 4.6
                else:
                    color = palette[rank % len(palette)]
                    linestyle = RANK_LINESTYLES[rank % len(RANK_LINESTYLES)]
                    marker = RANK_MARKERS[rank % len(RANK_MARKERS)]
                    linewidth = 1.3
                    markersize = 3.4
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=markersize,
                    alpha=1.0 if mode == "train" else 0.75,
                    markevery=max(1, len(xs) // 12),
                    linewidth=linewidth,
                    label=f"{MODEL_DISPLAY[model]} rank {rank} ({mode})",
                )
    etas = sorted({row["eta"] for data in pair.modes.values() for row in data.rates["rows"]})
    ax.set_xlabel("global update index")
    ax.set_ylabel("configured Adam learning rate eta")
    ax.set_title("configured Adam learning rate eta per rank and reference")
    ax.text(
        0.02,
        0.05,
        "unique eta values: " + ", ".join(f"{value:g}" for value in etas),
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(figure_title(user_title, pair, "learning rate"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_dir / f"{pair.label}-learning-rate.png", outputs)


def fig_discrepancy(pair, out_dir, user_title, outputs):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))
    error_fields = (
        ("peer_param_maxerr", "peer_param_maxerr", "#1f77b4"),
        ("peer_mom1_maxerr", "peer_mom1_maxerr", "#ff7f0e"),
        ("peer_mom2_maxerr", "peer_mom2_maxerr", "#2ca02c"),
    )
    positives = []
    for data in pair.modes.values():
        for row in data.metrics["rows"]:
            for field, _, _ in error_fields:
                if row[field] > 0:
                    positives.append(row[field])
    default_floor = 1e-12 if (pair.dtype or "").lower().startswith("float32") else 1e-18
    floor = min(positives) / 10.0 if positives else default_floor
    floor = max(floor, 1e-300)
    has_zero = False
    for mode in ("train", "resume"):
        data = pair.modes.get(mode)
        if data is None:
            continue
        rows = sorted(data.metrics["rows"], key=lambda row: row["step"])
        xs = np.array([row["step"] for row in rows], dtype=float)
        for field, display, color in error_fields:
            ys = np.array([row[field] for row in rows], dtype=float)
            has_zero = has_zero or bool(np.any(ys == 0))
            axes[0].plot(
                xs,
                np.maximum(ys, floor),
                color=color,
                linestyle="-" if mode == "train" else RESUME_STYLE,
                linewidth=1.2,
                markersize=2.6,
                markevery=max(1, len(xs) // 12),
                marker="o" if mode == "resume" else None,
                label=f"{display} ({mode})",
            )
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=floor / 10.0)
    if has_zero:
        axes[0].text(
            0.02,
            0.03,
            f"exact zeros shown at display floor {floor:.0e}",
            transform=axes[0].transAxes,
            fontsize=7,
        )
    axes[0].set_title("peer vs reference: parameter and Adam moment errors (log)")
    axes[0].set_xlabel("global update index")
    axes[0].set_ylabel("max absolute error (log scale)")
    axes[0].legend(fontsize=6.5, ncol=2, loc="best")
    for mode in ("train", "resume"):
        data = pair.modes.get(mode)
        if data is None:
            continue
        rows = sorted(data.metrics["rows"], key=lambda row: row["step"])
        xs = np.array([row["step"] for row in rows], dtype=float)
        axes[1].plot(
            xs,
            [row["loss_ref"] for row in rows],
            color="#d62728",
            linestyle="-" if mode == "train" else RESUME_STYLE,
            linewidth=1.2,
            label=f"loss_ref (reference, {mode})",
        )
        axes[1].plot(
            xs,
            [row["loss_local_rank0"] for row in rows],
            color="#1f77b4",
            linestyle="-" if mode == "train" else RESUME_STYLE,
            linewidth=1.2,
            label=f"loss_local_rank0 (DDP rank 0, {mode})",
        )
        axes[2].plot(
            xs,
            [1.0 if row["peer_beta_exact"] else 0.0 for row in rows],
            color="#2ca02c",
            linestyle="-" if mode == "train" else RESUME_STYLE,
            marker="o",
            markersize=2.6,
            markevery=max(1, len(xs) // 12),
            linewidth=1.2,
            label=f"peer_beta_exact ({mode})",
        )
    axes[1].set_title("training loss: reference vs rank-0 local")
    axes[1].set_xlabel("global update index")
    axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=7)
    axes[2].set_yticks([0.0, 1.0], ["false", "true"])
    axes[2].set_ylim(-0.25, 1.25)
    axes[2].set_title("peer_beta_exact (required true at every step)")
    axes[2].set_xlabel("global update index")
    axes[2].legend(fontsize=7, loc="lower right")
    for index, ax in enumerate(axes):
        shade_absences(ax, pair, TASK_ORDER)
        if index == 0:
            annotate_checkpoint(ax, pair, y=0.75)
    fig.suptitle(figure_title(user_title, pair, "per-step discrepancy"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, out_dir / f"{pair.label}-discrepancy.png", outputs)


def prediction_groups(data, task):
    groups = defaultdict(list)
    for row in data.predictions["rows"]:
        if row["task"] == task:
            groups[(row["step"], row["model"])].append(row)
    return groups


def collect_snapshots(pair, task):
    steps = sorted(
        {
            step
            for data in pair.modes.values()
            for step in data.predictions["snapshot_steps"]
        }
    )
    initial, final = steps[0], steps[-1]
    snapshots = {}
    for tag, step in (("initial", initial), ("final", final)):
        for model in MODEL_ORDER:
            chosen = None
            for mode in ("train", "resume"):
                data = pair.modes.get(mode)
                if data is None:
                    continue
                group = prediction_groups(data, task).get((step, model))
                if group:
                    chosen = group
                    break
            if chosen is None:
                _die(
                    f"{pair.label}: no prediction rows for task {task} step {step} model {model}"
                )
            snapshots[(tag, model)] = chosen
    return initial, final, snapshots


def fig_predictions_image(pair, task, snapshots, initial, final, out_dir, user_title, outputs):
    rows = snapshots[("initial", "ddp")]
    n_samples = len(rows)
    pixel_count = len(rows[0]["pixels"])
    side = int(round(math.sqrt(pixel_count)))
    if side * side != pixel_count:
        _die(
            f"{pair.label}: task {task} has {pixel_count} pixels, which is not a square "
            "image grid and not a two-feature toy sample"
        )
    specs = (
        ("initial", None),
        ("initial", "ddp"),
        ("initial", "reference"),
        ("final", None),
        ("final", "ddp"),
        ("final", "reference"),
    )
    all_pixels = [
        value
        for (_, rows) in snapshots.items()
        for row in rows
        for value in row["pixels"]
    ]
    vmin = min(all_pixels)
    vmax = max(all_pixels)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    fig, axes = plt.subplots(
        len(specs),
        n_samples,
        figsize=(max(7.0, 0.78 * n_samples), 0.95 * len(specs)),
        squeeze=False,
    )
    row_labels = {
        ("initial", None): "initial\ntarget",
        ("initial", "ddp"): "initial\nDDP pred",
        ("initial", "reference"): "initial\nref pred",
        ("final", None): "final\ntarget",
        ("final", "ddp"): "final\nDDP pred",
        ("final", "reference"): "final\nref pred",
    }
    for row_index, (tag, model) in enumerate(specs):
        target_rows = snapshots[(tag, "ddp")]
        for column_index in range(n_samples):
            ax = axes[row_index][column_index]
            if model is None:
                row = target_rows[column_index]
                image = np.asarray(row["pixels"], dtype=float).reshape(side, side)
                ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"t{row['target']}", fontsize=6)
            else:
                row = snapshots[(tag, model)][column_index]
                image = np.asarray(row["pixels"], dtype=float).reshape(side, side)
                ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                correct = row["predicted"] == row["target"]
                color = "#2ca02c" if correct else "#d62728"
                ax.set_title(f"p{row['predicted']}", fontsize=6, color=color)
                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(0.9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_index][0].set_ylabel(row_labels[(tag, model)], fontsize=7)
    summary = []
    for tag, _ in (("initial", None), ("final", None)):
        parts = []
        for model in MODEL_ORDER:
            group = snapshots[(tag, model)]
            correct = sum(row["predicted"] == row["target"] for row in group)
            parts.append(f"{MODEL_DISPLAY[model]} {correct}/{len(group)}")
        summary.append(f"{tag}: " + ", ".join(parts))
    fig.text(
        0.5,
        0.005,
        f"prediction accuracy - {' | '.join(summary)}; titles t=target label, p=predicted label",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.suptitle(
        figure_title(user_title, pair, f"predictions task {task.upper()} (image grid)"),
        fontsize=11,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.07, hspace=0.45, wspace=0.08)
    save_figure(fig, out_dir / f"{pair.label}-predictions-task-{task}.png", outputs)


def fig_predictions_scatter(pair, task, snapshots, initial, final, out_dir, user_title, outputs):
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), squeeze=False)
    columns = (None, "ddp", "reference")
    labels_seen = sorted(
        {row["target"] for rows in snapshots.values() for row in rows}
        | {row["predicted"] for rows in snapshots.values() for row in rows}
    )
    palette = plt.get_cmap("tab10").colors
    class_colors = {
        label: palette[index % len(palette)] for index, label in enumerate(labels_seen)
    }
    for row_index, tag in enumerate(("initial", "final")):
        target_rows = snapshots[(tag, "ddp")]
        for column_index, model in enumerate(columns):
            ax = axes[row_index][column_index]
            rows = target_rows if model is None else snapshots[(tag, model)]
            xs = [row["pixels"][0] for row in rows]
            ys = [row["pixels"][1] for row in rows]
            if model is None:
                for label, color in class_colors.items():
                    selected = [
                        (x, y)
                        for row, x, y in zip(rows, xs, ys)
                        if row["target"] == label
                    ]
                    if selected:
                        ax.scatter(
                            [point[0] for point in selected],
                            [point[1] for point in selected],
                            color=color,
                            marker="o",
                            s=42,
                            label=f"target class {label}",
                        )
                ax.set_title(f"{tag}: target labels")
            else:
                for label, color in class_colors.items():
                    correct = [
                        (x, y)
                        for row, x, y in zip(rows, xs, ys)
                        if row["predicted"] == label and row["predicted"] == row["target"]
                    ]
                    wrong = [
                        (x, y)
                        for row, x, y in zip(rows, xs, ys)
                        if row["predicted"] == label and row["predicted"] != row["target"]
                    ]
                    if correct:
                        ax.scatter(
                            [point[0] for point in correct],
                            [point[1] for point in correct],
                            color=color,
                            marker="o",
                            s=42,
                        )
                    if wrong:
                        ax.scatter(
                            [point[0] for point in wrong],
                            [point[1] for point in wrong],
                            color=color,
                            marker="x",
                            s=52,
                        )
                correct_count = sum(row["predicted"] == row["target"] for row in rows)
                ax.set_title(
                    f"{tag}: {MODEL_DISPLAY[model]} predictions "
                    f"({correct_count}/{len(rows)} correct)"
                )
            ax.set_xlabel("feature 1")
            ax.set_ylabel("feature 2")
            ax.grid(True, alpha=0.3)
    handles = [
        Line2D([], [], color="none", marker="o", markerfacecolor="gray", markersize=7, label="correct"),
        Line2D([], [], color="gray", marker="x", linestyle="none", markersize=7, label="incorrect"),
    ] + [
        Patch(facecolor=class_colors[label], label=f"class {label}")
        for label in labels_seen
    ]
    axes[0][0].legend(handles=handles, fontsize=7, loc="best")
    mismatches = [
        f"{row['sample_index']}({row['target']}->{row['predicted']})"
        for row in snapshots[("final", "ddp")]
        if row["predicted"] != row["target"]
    ]
    footer = "final DDP mismatches (sample(target->predicted)): " + (
        ", ".join(mismatches) if mismatches else "none"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        figure_title(user_title, pair, f"predictions task {task.upper()} (two-feature scatter)"),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save_figure(fig, out_dir / f"{pair.label}-predictions-task-{task}.png", outputs)


def fig_predictions(pair, task, out_dir, user_title, outputs):
    initial, final, snapshots = collect_snapshots(pair, task)
    pixel_count = pair.pixel_counts[task]
    if pixel_count == 2:
        fig_predictions_scatter(pair, task, snapshots, initial, final, out_dir, user_title, outputs)
    else:
        fig_predictions_image(pair, task, snapshots, initial, final, out_dir, user_title, outputs)


def fig_schedule(pair, out_dir, user_title, outputs):
    primary = pair.modes.get("train") or pair.modes["resume"]
    rows = sorted(primary.metrics["rows"], key=lambda row: row["step"])
    steps = np.array([row["step"] for row in rows], dtype=float)
    present_a = np.array([row["present_a"] for row in rows])
    present_b = np.array([row["present_b"] for row in rows])
    counts_a = np.array([row["counts_a"] for row in rows], dtype=float)
    counts_b = np.array([row["counts_b"] for row in rows], dtype=float)
    exposure_source = pair.modes.get("train") or pair.modes["resume"]
    exposures = {}
    for task in TASK_ORDER:
        exp_rows = sorted(
            [
                row
                for row in exposure_source.validation["rows"]
                if row["model"] == "ddp" and row["task"] == task
            ],
            key=lambda row: row["step"],
        )
        exposures[task] = (
            np.array([row["step"] for row in exp_rows], dtype=float),
            np.array([row["exposure"] for row in exp_rows], dtype=float),
        )
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 8.0), sharex=True)
    axes[0].scatter(
        steps[present_a],
        np.full(int(present_a.sum()), 1.0),
        marker="s",
        s=13,
        color=TASK_COLORS["a"],
        label="Task A contributing rank present",
    )
    axes[0].scatter(
        steps[present_b],
        np.full(int(present_b.sum()), 0.55),
        marker="s",
        s=13,
        color=TASK_COLORS["b"],
        label="Task B contributing rank present",
    )
    axes[0].set_yticks([0.55, 1.0], ["Task B", "Task A"])
    axes[0].set_ylim(0.30, 1.25)
    axes[0].set_title("per-step task presence (a gap means the task is globally absent)")
    axes[1].step(steps, counts_a, where="mid", color=TASK_COLORS["a"], linewidth=1.3, label="Task A")
    axes[1].step(steps, counts_b, where="mid", color=TASK_COLORS["b"], linewidth=1.3, label="Task B")
    axes[1].plot(steps, counts_a, "o", color=TASK_COLORS["a"], markersize=2.6)
    axes[1].plot(steps, counts_b, "o", color=TASK_COLORS["b"], markersize=2.6)
    axes[1].set_ylabel("contributing ranks")
    axes[1].set_title("number of ranks assigned to each task per global step")
    axes[1].legend(fontsize=7)
    for task in TASK_ORDER:
        xs, ys = exposures[task]
        axes[2].step(xs, ys, where="mid", color=TASK_COLORS[task], linewidth=1.3, label=f"Task {task.upper()}")
        axes[2].plot(xs, ys, "o", color=TASK_COLORS[task], markersize=2.6)
        axes[2].annotate(
            f"{int(ys[-1])}",
            xy=(xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=TASK_COLORS[task],
        )
    axes[2].set_ylabel("cumulative training samples")
    axes[2].set_title("cumulative training exposure per task")
    axes[2].set_xlabel("global update index")
    axes[2].legend(fontsize=7)
    absence_handles = []
    for index, ax in enumerate(axes):
        handles = shade_absences(ax, pair, TASK_ORDER)
        if handles and not absence_handles:
            absence_handles = handles
        if index == 0:
            annotate_checkpoint(ax, pair, y=0.99)
    legend_handles = (
        axes[0].get_legend_handles_labels()[0]
        + absence_handles
        + [
            Line2D(
                [],
                [],
                color="black",
                linestyle=":",
                linewidth=1.0,
                label=f"checkpoint/resume boundary (step {pair.checkpoint_step})",
            )
        ]
    )
    legend_labels = (
        axes[0].get_legend_handles_labels()[1]
        + [handle.get_label() for handle in absence_handles]
        + [f"checkpoint/resume boundary (step {pair.checkpoint_step})"]
    )
    axes[0].legend(legend_handles, legend_labels, fontsize=7, loc="lower left", ncol=2)
    fig.suptitle(figure_title(user_title, pair, "schedule and participation"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out_dir / f"{pair.label}-schedule.png", outputs)


def expected_collective_count(row):
    """One presence reduction plus two gradient reductions per present head."""
    present = int(row["present_a"]) + int(row["present_b"])
    if present not in (1, 2):
        _die(
            f"step {row['step']}: {present} globally present tasks; expected one or two"
        )
    return 3 + 2 * present


def fig_exposure(pair, out_dir, user_title, outputs):
    if not pair.workers:
        _die(f"{pair.label}: unknown worker count; cannot compute task weights")
    primary = pair.modes.get("train") or pair.modes["resume"]
    rows = sorted(primary.metrics["rows"], key=lambda row: row["step"])
    steps = np.array([row["step"] for row in rows], dtype=float)
    weights = {
        task: np.array(
            [row[f"counts_{task}"] / pair.workers for row in rows], dtype=float
        )
        for task in TASK_ORDER
    }
    exposures = {}
    exposure_source = pair.modes.get("train") or pair.modes["resume"]
    for task in TASK_ORDER:
        exp_rows = sorted(
            [
                row
                for row in exposure_source.validation["rows"]
                if row["model"] == "ddp" and row["task"] == task
            ],
            key=lambda row: row["step"],
        )
        exposures[task] = (
            np.array([row["step"] for row in exp_rows], dtype=float),
            np.array([row["exposure"] for row in exp_rows], dtype=float),
        )
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 5.8), sharex=True)
    top = axes[0]
    for task in TASK_ORDER:
        top.vlines(
            steps, 0.0, weights[task], color=TASK_COLORS[task], linewidth=0.8, alpha=0.65
        )
        top.plot(
            steps,
            weights[task],
            "o",
            color=TASK_COLORS[task],
            markersize=3.0,
            label=f"Task {task.upper()}",
        )
    top.set_ylim(-0.04, 1.08)
    top.set_ylabel("task weight (counts / ranks)")
    top.set_title(
        f"per-step gradient weight g = (1/R) * sum_r g_r with R = {pair.workers}; "
        "locally absent parameters contribute zero"
    )
    top.legend(fontsize=7, ncol=2)
    mixed = [row for row in rows if row["present_a"] and row["present_b"]]
    if mixed:
        first = mixed[0]
        first_index = rows.index(first)
        top.annotate(
            f"step {first['step']}: A {first['counts_a']}/{pair.workers}, "
            f"B {first['counts_b']}/{pair.workers}",
            xy=(first["step"], max(weights["a"][first_index], weights["b"][first_index])),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.6},
        )
    bottom = axes[1]
    for task in TASK_ORDER:
        xs, ys = exposures[task]
        bottom.step(xs, ys, where="mid", color=TASK_COLORS[task], linewidth=1.3, label=f"Task {task.upper()}")
        bottom.plot(xs, ys, "o", color=TASK_COLORS[task], markersize=2.6)
        bottom.annotate(
            f"{int(ys[-1])}",
            xy=(xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=TASK_COLORS[task],
        )
    bottom.set_ylabel("cumulative samples")
    bottom.set_title("cumulative training exposure per task (global samples)")
    bottom.set_xlabel("global update index")
    bottom.legend(fontsize=7)
    for ax in axes:
        shade_absences(ax, pair, TASK_ORDER)
    annotate_checkpoint(axes[0], pair, y=0.99)
    fig.suptitle(figure_title(user_title, pair, "task weights and exposure"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, out_dir / f"{pair.label}-exposure.png", outputs)


def fig_collectives(pair, out_dir, user_title, outputs):
    primary = pair.modes.get("train") or pair.modes["resume"]
    rows = sorted(primary.metrics["rows"], key=lambda row: row["step"])
    steps = np.array([row["step"] for row in rows], dtype=float)
    counts = np.array([expected_collective_count(row) for row in rows], dtype=float)
    n_five = int((counts == 5).sum())
    n_seven = int((counts == 7).sum())
    n_five + n_seven == len(counts) or _die(
        f"{pair.label}: unexpected collective counts {sorted(set(counts))}"
    )
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.step(steps, counts, where="mid", color="#333333", linewidth=1.2)
    ax.plot(steps, counts, "o", color="#333333", markersize=3.0)
    ax.set_yticks([5, 7])
    ax.set_ylim(4.5, 7.5)
    ax.set_xlabel("global update index")
    ax.set_ylabel("collectives per rank")
    ax.set_title(
        "expected collectives per step: 1 CPU presence reduction, then one gradient "
        "reduction per present leaf (GPU gradients use NCCL)"
    )
    ax.annotate(
        f"{n_seven} steps with 7 collectives (both tasks present)",
        xy=(0.01, 0.95),
        xycoords="axes fraction",
        fontsize=7.5,
        va="top",
    )
    ax.annotate(
        f"{n_five} steps with 5 collectives (one task globally absent)",
        xy=(0.01, 0.84),
        xycoords="axes fraction",
        fontsize=7.5,
        va="top",
    )
    shade_absences(ax, pair, TASK_ORDER)
    annotate_checkpoint(ax, pair, y=0.70)
    fig.suptitle(figure_title(user_title, pair, "collective protocol"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_dir / f"{pair.label}-collectives.png", outputs)


def ddp_validation_lookup(data, task, field):
    return {
        row["step"]: row[field]
        for row in data.validation["rows"]
        if row["model"] == "ddp" and row["task"] == task
    }


def fig_continuation(pair, out_dir, user_title, outputs, warnings):
    train = pair.modes.get("train")
    resume = pair.modes.get("resume")
    if train is None or resume is None:
        warnings.append(
            f"{pair.label}: continuation figure needs both train and resume; skipped"
        )
        return
    steps = np.array(
        list(range(pair.checkpoint_step + 1, pair.steps + 1)), dtype=float
    )
    if steps.size == 0:
        _die(f"{pair.label}: no steps after checkpoint {pair.checkpoint_step}")
    fields = (("loss", "validation loss difference (train - resume)"),
              ("accuracy", "validation accuracy difference (train - resume)"))
    differences = {}
    summary = {}
    for task in TASK_ORDER:
        for field, _ in fields:
            train_values = ddp_validation_lookup(train, task, field)
            resume_values = ddp_validation_lookup(resume, task, field)
            missing = [int(step) for step in steps if step not in train_values or step not in resume_values]
            if missing:
                _die(
                    f"{pair.label}: continuation steps missing for task {task} {field}: "
                    f"{missing[:5]}"
                )
            diff = np.array(
                [train_values[int(step)] - resume_values[int(step)] for step in steps],
                dtype=float,
            )
            differences[(task, field)] = diff
            summary[(task, field)] = (
                int(np.count_nonzero(diff == 0.0)),
                int(diff.size),
                float(np.max(np.abs(diff))),
            )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for ax, (field, ylabel) in zip(axes, fields):
        for task in TASK_ORDER:
            values = differences[(task, field)]
            exact, total, max_diff = summary[(task, field)]
            ax.plot(
                steps,
                values,
                TASK_MARKERS[task],
                color=TASK_COLORS[task],
                markersize=3.0,
                linewidth=0.8,
                label=(
                    f"Task {task.upper()}: {exact}/{total} exact, "
                    f"max |diff| {max_diff:.3e}"
                ),
            )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("global update index")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
        if all(np.all(differences[(task, field)] == 0.0) for task in TASK_ORDER):
            ax.annotate(
                "all differences are exactly 0.0",
                xy=(0.5, 0.06),
                xycoords="axes fraction",
                ha="center",
                fontsize=7.5,
            )
        annotate_checkpoint(ax, pair, y=0.99)
    fig.suptitle(
        figure_title(user_title, pair, "fresh-process continuation difference"), fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_dir / f"{pair.label}-continuation.png", outputs)


def final_validation_row(pair, task, model):
    for mode in ("train", "resume"):
        data = pair.modes.get(mode)
        if data is None:
            continue
        for row in data.validation["rows"]:
            if row["step"] == pair.steps and row["model"] == model and row["task"] == task:
                return row
    _die(
        f"{pair.label}: no final (step {pair.steps}) validation row for task {task} "
        f"model {model}"
    )


def comparison_label(pair):
    parent = os.path.basename(pair.pair_id.rstrip(os.sep)) or pair.label
    parts = [parent]
    if pair.dataset_name:
        parts.append(pair.dataset_name)
    if pair.backend:
        parts.append(pair.backend)
    if pair.workers:
        parts.append(f"{pair.workers} ranks")
    return " / ".join(parts)


def fig_comparison(pairs, out_dir, user_title, outputs):
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 2.6 * len(pairs) + 3.0), 5.2))
    positions = np.arange(len(pairs), dtype=float)
    width = 0.36
    max_loss_diff = 0.0
    max_acc_diff = 0.0
    for task_index, task in enumerate(TASK_ORDER):
        offset = (task_index - 0.5) * width
        losses = []
        accuracies = []
        for pair in pairs:
            ddp_row = final_validation_row(pair, task, "ddp")
            reference_row = final_validation_row(pair, task, "reference")
            losses.append(ddp_row["loss"])
            accuracies.append(ddp_row["accuracy"])
            max_loss_diff = max(max_loss_diff, abs(ddp_row["loss"] - reference_row["loss"]))
            max_acc_diff = max(
                max_acc_diff, abs(ddp_row["accuracy"] - reference_row["accuracy"])
            )
        loss_bars = axes[0].bar(
            positions + offset, losses, width, color=TASK_COLORS[task], label=f"Task {task.upper()}"
        )
        acc_bars = axes[1].bar(
            positions + offset,
            accuracies,
            width,
            color=TASK_COLORS[task],
            label=f"Task {task.upper()}",
        )
        for bars, values, ax in ((loss_bars, losses, axes[0]), (acc_bars, accuracies, axes[1])):
            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:.4g}",
                    xy=(bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )
    labels = [comparison_label(pair) for pair in pairs]
    ylabels = ("validation loss (final step)", "validation accuracy (final step)")
    for ax, ylabel in zip(axes, ylabels):
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        if ylabel.startswith("validation accuracy"):
            ax.set_ylim(0.0, 1.0)
    footer = "; ".join(f"{pair.label} = {pair.run_id}" for pair in pairs)
    fig.text(
        0.5,
        0.005,
        f"max final |DDP - reference|: loss {max_loss_diff:.3e}, accuracy {max_acc_diff:.3e}"
        f"; {footer}",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    title = "final validation metrics per configuration"
    if user_title:
        title = f"{user_title} - {title}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, out_dir / "phase-k-comparison.png", outputs)


def generate_figures(pair, out_dir, user_title, outputs, warnings):
    fig_validation(pair, out_dir, "loss", "validation loss", "validation-loss", user_title, outputs)
    fig_validation(
        pair, out_dir, "accuracy", "validation accuracy", "validation-accuracy", user_title, outputs
    )
    fig_difference(pair, out_dir, user_title, outputs)
    fig_learning_rate(pair, out_dir, user_title, outputs)
    fig_discrepancy(pair, out_dir, user_title, outputs)
    fig_exposure(pair, out_dir, user_title, outputs)
    fig_collectives(pair, out_dir, user_title, outputs)
    fig_continuation(pair, out_dir, user_title, outputs, warnings)
    for task in TASK_ORDER:
        fig_predictions(pair, task, out_dir, user_title, outputs)
    fig_schedule(pair, out_dir, user_title, outputs)
    if not pair.optional_files:
        warnings.append(f"{pair.label}: no prepared provenance files recorded")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Plot Phase K conditional-graph DDP demonstration artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="RUN_DIR",
        help="campaign root containing <config>/train and <config>/resume; repeatable",
    )
    parser.add_argument("--out", required=True, metavar="FIG_DIR", help="output directory")
    parser.add_argument("--title", default=None, help="optional figure title prefix")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_roots = [os.path.abspath(os.fspath(Path(name).expanduser())) for name in args.run]
    for run_root in run_roots:
        if not os.path.isdir(run_root):
            _die(f"--run {run_root} is not a directory")
    out_dir = Path(os.path.abspath(os.fspath(Path(args.out).expanduser())))
    if out_dir.exists() and not out_dir.is_dir():
        _die(f"--out {out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = WarningList()
    entries = []
    for root_index, run_root in enumerate(run_roots):
        discovered = discover_pairs(run_root, warnings)
        for pair_id in sorted(discovered):
            status = {}
            skip = False
            for mode in ("train", "resume"):
                if mode not in discovered[pair_id]:
                    continue
                paths = mode_file_paths(discovered[pair_id][mode], mode, warnings)
                if paths is None:
                    skip = True
                    break
                status[mode] = {"directory": discovered[pair_id][mode], **paths}
            if skip or not status:
                continue
            entries.append((root_index, run_root, pair_id, status))
    if not entries:
        _die(
            "no complete Phase K train/resume output found under: " + ", ".join(run_roots)
        )
    roots_with_entries = {entry[0] for entry in entries}
    for root_index, run_root in enumerate(run_roots):
        if root_index not in roots_with_entries:
            warnings.append(f"{run_root}: no complete K2 train/resume output found")
    multi = len(run_roots) > 1
    used_labels = set()
    pairs = []
    for root_index, run_root, pair_id, status in entries:
        relative = os.path.relpath(pair_id, run_root)
        if relative == ".":
            label = os.path.basename(run_root)
        elif relative.startswith(".."):
            label = os.path.basename(pair_id)
        else:
            label = relative.replace(os.sep, "_")
        label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label) or "run"
        if multi:
            root_tag = "".join(
                ch if ch.isalnum() or ch in "._-" else "_" for ch in os.path.basename(run_root)
            ) or f"run{root_index + 1}"
            label = f"{root_tag}-{label}"
        base_label = label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}-{suffix}"
            suffix += 1
        used_labels.add(label)
        pair = Pair(label=label, run_root=run_root, pair_id=pair_id, status=status)
        pair.nclasses = None
        pair.prepared_dir = find_prepared_dir(pair_id, run_root)
        if pair.prepared_dir is not None:
            subset_path = pair.prepared_dir / "subset_indices.tsv"
            sources_path = pair.prepared_dir / "sources.tsv"
            if subset_path.is_file():
                pair.nclasses = parse_subset_classes(subset_path, warnings)
                pair.optional_files.append(subset_path)
            else:
                warnings.append(f"{subset_path}: optional prepared subset_indices.tsv is absent")
            if sources_path.is_file():
                pair.optional_files.append(sources_path)
            else:
                warnings.append(f"{sources_path}: optional prepared sources.tsv is absent")
        else:
            warnings.append(
                f"{label}: prepared directory not found; sources.tsv/subset_indices.tsv "
                "not recorded"
            )
        pairs.append(pair)
    for pair in pairs:
        load_pair(pair, warnings)
    outputs = []
    for pair in pairs:
        generate_figures(pair, out_dir, args.title, outputs, warnings)
    if len(pairs) > 1:
        fig_comparison(pairs, out_dir, args.title, outputs)
    metadata_path = out_dir / "figures-metadata.txt"
    lines = ["phase_k_demo_plot_metadata"]
    add = lines.append
    add(f"generated_at_utc\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add(f"python\t{sys.version.split()[0]}")
    add(f"matplotlib\t{matplotlib.__version__}")
    add(f"numpy\t{np.__version__}")
    add("command\t" + " ".join(shlex.quote(name) for name in sys.argv))
    add(
        "reproducible_command\tuv run --offline --with-requirements "
        "scripts/plots/requirements-phase-k.txt python3 "
        + " ".join(shlex.quote(name) for name in sys.argv)
    )
    for run_root in run_roots:
        add(f"run_root\t{run_root}")
    for pair in pairs:
        add(
            f"pair\t{pair.label}\t{pair.pair_id}\tsteps={pair.steps}"
            f"\tcheckpoint_step={pair.checkpoint_step}\tworkers={pair.workers}"
            f"\tdataset={pair.dataset_name or 'unknown'}\tbackend={pair.backend or 'unknown'}"
            f"\tmodes={','.join(sorted(pair.modes))}"
        )
        add(f"run_id\t{pair.label}\t{pair.run_id}")
        add(f"schedule_digest\t{pair.label}\t{pair.schedule_digest}")
        add(f"dataset_digest\t{pair.label}\t{pair.dataset_digest}")
        if pair.eta is not None:
            add(f"eta\t{pair.label}\t{pair.eta:g}")
        for mode in sorted(pair.modes):
            data = pair.modes[mode]
            for stem in FILE_STEMS:
                path = data.paths[stem]
                add(
                    f"input\t{pair.label}\t{mode}\t{stem}\t{path}\t{sha256_file(path)}"
                    f"\trows={data.row_counts[stem]}"
                )
            if data.metadata.get("present"):
                path = data.metadata["path"]
                add(f"input\t{pair.label}\t{mode}\tmetadata\t{path}\t{sha256_file(path)}")
        for path in pair.optional_files:
            add(
                f"optional_input\t{pair.label}\t{path}\t{sha256_file(path)}"
                f"\trows={max(0, len(path.read_text(encoding='utf-8').splitlines()) - 1)}"
            )
    for path in outputs:
        add(f"output\t{path}\t{sha256_file(path)}\tbytes={path.stat().st_size}")
    for warning in warnings:
        add(f"warning\t{warning}")
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"phase_k_demo_plot: wrote {len(outputs)} figures and {metadata_path}")
    for pair in pairs:
        print(
            f"  {pair.label}: run_id={pair.run_id} steps={pair.steps} "
            f"checkpoint_step={pair.checkpoint_step} workers={pair.workers}"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SchemaError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
