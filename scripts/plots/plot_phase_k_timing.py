#!/usr/bin/env python3
"""Plot Phase K timing-stretch records and run the live viewer.

The tool reads the tab-separated files defined by
``temp/phase_k_timing_stretch_plan.md`` (``records.tsv``, ``steps.tsv``,
``repeats.tsv``) from one baseline directory and one DDP directory, and writes
deterministic PNG figures plus a ``figures-metadata.txt`` provenance record.
Speedup is the ratio of the two per-mode median durations (baseline / DDP
slowest rank) for training-only and total time; repeats are never paired by
index, and a slowdown is a valid result.

Reproducible invocation in this checkout (system python3 has no numpy or
matplotlib; ``uv`` provides them in an ephemeral environment):

    uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py \
        --baseline <BASELINE_DIR> --ddp <DDP_DIR> --out <FIG_DIR> \
        [--title TITLE] [--label offline|replay|live]

Live follow mode polls a rank-0 records file and refreshes ``timing-live.png``:

    uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py \
        --follow --records <DIR>/records.tsv [--baseline <BASELINE_DIR>] \
        --out <FIG_DIR> --interval 2 --status-log <FILE> [--replay]

See scripts/plots/README.md for details.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    print(
        f"ERROR: missing python dependency {exc.name!r}.\n"
        f"Run with: uv run --offline --with matplotlib --with numpy python3 {sys.argv[0]} ...",
        file=sys.stderr,
    )
    sys.exit(2)

RECORD_COLUMNS = (
    "run_id mode repeat step elapsed_s train_s samples_total samples_a samples_b "
    "updates_per_s samples_per_s val_a_loss val_a_acc val_b_loss val_b_acc".split()
)
STEP_COLUMNS = "mode repeat step elapsed_s train_s samples_total samples_a samples_b".split()
REPEAT_COLUMNS = (
    "run_id mode repeat t_total_s t_train_s t_val_s t_write_s slowest_rank_s "
    "slowest_train_s steps samples_total updates_per_s samples_per_s "
    "train_updates_per_s train_samples_per_s".split()
)

STATIC_FIGURES = (
    "timing-updates-vs-elapsed.png",
    "timing-samples-vs-elapsed.png",
    "timing-val-loss-a.png",
    "timing-val-loss-b.png",
    "timing-val-acc-a.png",
    "timing-val-acc-b.png",
    "timing-repeat-times.png",
    "timing-speedup.png",
)
LIVE_FIGURE = "timing-live.png"
FIGURE_DPI = 160
XLABEL_TRAIN = "elapsed training time (s), validation excluded"

MODE_STYLE = {
    "baseline": {"color": "#1f77b4", "display": "baseline (single process)"},
    "ddp": {"color": "#d62728", "display": "DDP"},
}
TASK_LABELS = {"a": "Task A", "b": "Task B"}


def _tint(color, white_fraction):
    """Blend ``color`` toward white; used for deterministic bar shading."""
    rgb = to_rgb(color)
    return tuple(component + (1.0 - component) * white_fraction for component in rgb)


def _repeat_offsets(count):
    """Deterministic horizontal jitter ordered by repeat index."""
    if count <= 1:
        return np.zeros(count, dtype=float)
    return np.linspace(-0.08, 0.08, count)


def ratio_of_medians(numerator, denominator):
    """median(numerator) / median(denominator); nan when the denominator is invalid."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.size == 0 or denominator.size == 0:
        return float("nan")
    denominator_median = float(np.median(denominator))
    if denominator_median <= 0.0 or not math.isfinite(denominator_median):
        return float("nan")
    return float(np.median(numerator)) / denominator_median


class SchemaError(Exception):
    """A timing TSV input violates the fixed schema."""


def _die(message: str) -> None:
    raise SchemaError(message)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_lines(text: str, path: Path, label: str, strict: bool):
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if strict and text and not text.endswith("\n"):
        _die(f"{path}: {label} does not end with a newline (partial write?)")
    return lines


def parse_rows(text: str, path: Path, columns, label: str, strict: bool = True):
    lines = _clean_lines(text, path, label, strict)
    if not lines:
        if strict:
            _die(f"{path}: {label} is empty")
        return []
    header = lines[0].split("\t")
    if header != list(columns):
        _die(
            f"{path}: {label} header mismatch\n"
            f"  expected: {' '.join(columns)}\n"
            f"  found:    {' '.join(header)}"
        )
    rows = []
    for lineno, raw in enumerate(lines[1:], start=2):
        if raw == "":
            _die(f"{path}:{lineno}: blank row in {label}")
        fields = raw.split("\t")
        if len(fields) != len(columns):
            _die(
                f"{path}:{lineno}: expected {len(columns)} tab-separated fields, "
                f"found {len(fields)}"
            )
        rows.append((lineno, dict(zip(columns, fields))))
    if strict and not rows:
        _die(f"{path}: {label} has a header but no data rows")
    return rows


def number(path: Path, lineno: int, row, key: str) -> float:
    value = row[key]
    try:
        parsed = float(value)
    except ValueError:
        _die(f"{path}:{lineno}: {key}={value!r} is not a number")
    if not math.isfinite(parsed):
        _die(f"{path}:{lineno}: {key}={value!r} is not finite")
    return parsed


def integer(path: Path, lineno: int, row, key: str) -> int:
    value = row[key]
    try:
        return int(value)
    except ValueError:
        _die(f"{path}:{lineno}: {key}={value!r} is not an integer")


NUMERIC_RECORDS = tuple(RECORD_COLUMNS[3:])
NUMERIC_STEPS = tuple(STEP_COLUMNS[2:])
NUMERIC_REPEATS = tuple(REPEAT_COLUMNS[3:])


def load_table(directory: Path, stem: str, columns, numeric, mode: str, strict: bool):
    path = directory / f"{stem}.tsv"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _die(f"{path}: cannot read {stem}.tsv: {exc}")
    raw_rows = parse_rows(text, path, columns, f"{stem}.tsv", strict=strict)
    rows = []
    run_ids = set()
    for lineno, raw in raw_rows:
        row = {"mode": raw["mode"]}
        if "run_id" in raw:
            row["run_id"] = raw["run_id"]
            run_ids.add(raw["run_id"])
        for key in columns:
            if key in ("run_id", "mode"):
                continue
            row[key] = integer(path, lineno, raw, key) if key in ("step", "repeat") \
                else number(path, lineno, raw, key)
        if row["mode"] != mode:
            _die(f"{path}:{lineno}: mode={row['mode']!r}, expected {mode!r}")
        rows.append(row)
    if run_ids and len(run_ids) != 1:
        _die(f"{path}: multiple run_id values: {sorted(run_ids)}")
    return {"path": path, "rows": rows, "run_ids": run_ids}


class ModeData:
    def __init__(self, name, directory):
        self.name = name
        self.directory = directory
        self.records = None
        self.steps = None
        self.repeats = None
        self.metadata_path = None

    def load(self, strict=True):
        expected = "single" if self.name == "baseline" else "ddp"
        self.records = load_table(self.directory, "records", RECORD_COLUMNS,
                                  NUMERIC_RECORDS, expected, strict=strict)
        self.steps = load_table(self.directory, "steps", STEP_COLUMNS,
                                NUMERIC_STEPS, expected, strict=strict)
        self.repeats = load_table(self.directory, "repeats", REPEAT_COLUMNS,
                                  NUMERIC_REPEATS, expected, strict=strict)
        candidate = self.directory / "metadata.txt"
        self.metadata_path = candidate if candidate.is_file() else None
        return self

    def records_by_repeat(self):
        grouped = {}
        for row in self.records["rows"]:
            grouped.setdefault(int(row["repeat"]), []).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: row["step"])
        return grouped

    def steps_by_repeat(self):
        grouped = {}
        for row in self.steps["rows"]:
            grouped.setdefault(int(row["repeat"]), []).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: row["step"])
        return grouped

    def repeat_rows(self):
        return sorted(self.repeats["rows"], key=lambda row: row["repeat"])


def band_curve(series, grid, value_key):
    """Interpolate ``value_key`` from each repeat's step series onto ``grid``.

    Returns ``(median, low, high, n_repeats)`` where the arrays are aligned with
    ``grid``; per-repeat values are interpolated in step space so repeats with
    different record boundaries still share one x grid.
    """
    curves = []
    for rows in series.values():
        if not rows:
            continue
        ys = np.array([row["step"] for row in rows], dtype=float)
        xs = np.array([row[value_key] for row in rows], dtype=float)
        if len(ys) == 0 or np.any(np.diff(ys) <= 0):
            continue
        curves.append(np.interp(grid, ys, xs))
    if not curves:
        return None
    stacked = np.vstack(curves)
    return np.median(stacked, axis=0), stacked.min(axis=0), stacked.max(axis=0), len(curves)


def common_grid(mode_data, value_key):
    steps = set()
    for rows in mode_data.steps_by_repeat().values():
        for row in rows:
            steps.add(row["step"])
    for rows in mode_data.records_by_repeat().values():
        for row in rows:
            steps.add(row["step"])
    return np.array(sorted(steps), dtype=float)


def plot_value_or_step(baseline, ddp, value_key, ylabel, title, label, out_path):
    """Plot ``step`` or a cumulative value key against elapsed training time.

    Every repeat contributes one curve: its ``train_s`` values are interpolated
    onto the shared step grid, and the figure draws the per-repeat lines, the
    min/max spread band, and the median line.
    """
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for name, data in (("baseline", baseline), ("ddp", ddp)):
        if data is None:
            continue
        series = data.steps_by_repeat()
        grid = common_grid(data, value_key)
        if len(grid) == 0:
            continue
        curves_x = []
        curves_y = []
        for rows in series.values():
            step_axis = np.array([row["step"] for row in rows], dtype=float)
            xvals = np.array([row["train_s"] for row in rows], dtype=float)
            if np.any(np.diff(step_axis) <= 0):
                continue
            curves_x.append(np.interp(grid, step_axis, xvals))
            if value_key == "step":
                curves_y.append(grid)
            else:
                values = np.array([row[value_key] for row in rows], dtype=float)
                curves_y.append(np.interp(grid, step_axis, values))
        if not curves_x:
            continue
        xstack = np.vstack(curves_x)
        ystack = np.vstack(curves_y)
        median_x = np.median(xstack, axis=0)
        median_y = np.median(ystack, axis=0)
        n = xstack.shape[0]
        style = MODE_STYLE[name]
        if n > 1:
            for i in range(n):
                ax.plot(xstack[i], ystack[i], color=style["color"], linewidth=0.7,
                        alpha=0.28, zorder=2)
            ax.fill_between(median_x, ystack.min(axis=0), ystack.max(axis=0),
                            color=style["color"], alpha=0.16, lw=0, zorder=1)
        ax.plot(median_x, median_y, color=style["color"], linewidth=2.2, zorder=3,
                label=f"{style['display']} (median of {n} repeat{'s' if n > 1 else ''})")
    ax.set_xlabel(XLABEL_TRAIN)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    note = "curves are per-repeat rank-0 training time"
    if label == "replay":
        note += "; replayed data, not a live measurement"
    elif label == "live":
        note += "; captured live"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_validation(baseline, ddp, task, field, ylabel, title, label, out_path):
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    value_key = f"val_{task}_{field}"
    for name, data in (("baseline", baseline), ("ddp", ddp)):
        if data is None:
            continue
        series = data.records_by_repeat()
        grid = common_grid(data, value_key)
        if len(grid) == 0:
            continue
        band = band_curve(series, grid, "train_s")
        if band is None:
            continue
        median_x, low_x, high_x, n = band
        curves = []
        for rows in series.values():
            step_axis = np.array([row["step"] for row in rows], dtype=float)
            values = np.array([row[value_key] for row in rows], dtype=float)
            curves.append(np.interp(grid, step_axis, values))
        stacked = np.vstack(curves)
        ys = np.median(stacked, axis=0)
        style = MODE_STYLE[name]
        if n > 1:
            for rows in series.values():
                ax.plot([row["train_s"] for row in rows],
                        [row[value_key] for row in rows],
                        color=style["color"], linewidth=0.7, alpha=0.28, zorder=2)
            ax.fill_between(median_x, stacked.min(axis=0), stacked.max(axis=0),
                            color=style["color"], alpha=0.16, lw=0, zorder=1)
        ax.plot(median_x, ys, color=style["color"], linewidth=2.2, zorder=3,
                label=f"{style['display']} (median of {n} repeat{'s' if n > 1 else ''})")
    ax.set_xlabel(XLABEL_TRAIN)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    note = "validation at record boundaries; x is rank-0 elapsed training time"
    if label == "replay":
        note += "; replayed data, not a live measurement"
    elif label == "live":
        note += "; captured live"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_repeat_times(baseline, ddp, title, label, out_path):
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    width = 0.36
    xmax = 0
    medians = {}
    for offset, (name, data) in enumerate((("baseline", baseline), ("ddp", ddp))):
        if data is None:
            continue
        rows = data.repeat_rows()
        xmax = max(xmax, len(rows))
        positions = np.arange(len(rows), dtype=float) + (offset - 0.5) * width
        style = MODE_STYLE[name]
        trains = np.array([row["t_train_s"] for row in rows], dtype=float)
        validations = np.array([row["t_val_s"] for row in rows], dtype=float)
        writes = np.array([row["t_write_s"] for row in rows], dtype=float)
        totals = np.array([row["t_total_s"] for row in rows], dtype=float)
        remainder = np.maximum(totals - (trains + validations + writes), 0.0)
        segments = (
            (trains, style["color"], "training (t_train_s)"),
            (validations, _tint(style["color"], 0.60), "validation (t_val_s)"),
            (writes, _tint(style["color"], 0.35), "record writes (t_write_s)"),
            (remainder, _tint(style["color"], 0.85), "remainder"),
        )
        bottom = np.zeros(len(rows), dtype=float)
        for values, color, segment_label in segments:
            ax.bar(positions, values, width, bottom=bottom, color=color,
                   edgecolor=style["color"], linewidth=0.4,
                   label=segment_label if name == "baseline" else None)
            bottom = bottom + values
        if name == "ddp":
            ax.plot(positions, [row["slowest_rank_s"] for row in rows],
                    linestyle="none", marker="x", markersize=6, color=style["color"],
                    label="DDP slowest-rank total")
            ax.plot(positions, [row["slowest_train_s"] for row in rows],
                    linestyle="none", marker="+", markersize=7, color=style["color"],
                    label="DDP slowest-rank training")
        medians[name] = (float(np.median(totals)), float(np.median(trains)))
    for name, (total, train) in medians.items():
        style = MODE_STYLE[name]
        ax.axhline(total, color=style["color"], linestyle="--", linewidth=1.0)
        ax.axhline(train, color=style["color"], linestyle=":", linewidth=1.0)
    note = ("stacked rank-0 local: training + validation + writes + remainder "
            "(blue = baseline, red = DDP)\n"
            "DDP markers: x = slowest-rank total, + = slowest-rank training; "
            "dashed/dotted = median rank-0 total/training")
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7)
    ax.set_xticks(np.arange(xmax), [f"repeat {i + 1}" for i in range(xmax)])
    ax.set_xlabel("repeat")
    ax.set_ylabel("wall time (s)")
    ax.set_title(title)
    if label == "replay":
        ax.text(0.02, 0.12, "replayed data, not a live measurement",
                transform=ax.transAxes, fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_speedup(baseline, ddp, title, label, out_path):
    if baseline is None or ddp is None:
        return
    base_rows = baseline.repeat_rows()
    ddp_rows = ddp.repeat_rows()
    if not base_rows or not ddp_rows:
        return
    panels = (
        ("training only",
         np.array([row["t_train_s"] for row in base_rows], dtype=float),
         np.array([row["slowest_train_s"] for row in ddp_rows], dtype=float)),
        ("total (validation + writes included)",
         np.array([row["t_total_s"] for row in base_rows], dtype=float),
         np.array([row["slowest_rank_s"] for row in ddp_rows], dtype=float)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax, (panel_title, base_values, ddp_values) in zip(axes, panels):
        for center, name, values in ((0.0, "baseline", base_values),
                                     (1.0, "ddp", ddp_values)):
            style = MODE_STYLE[name]
            ax.plot(center + _repeat_offsets(values.size), values, linestyle="none",
                    marker="o", markersize=5.5, color=style["color"],
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=style["display"], zorder=3)
            median = float(np.median(values))
            ax.plot([center - 0.18, center + 0.18], [median, median],
                    color=style["color"], linewidth=2.4, zorder=4,
                    label=f"{style['display']} median")
        speedup = ratio_of_medians(base_values, ddp_values)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, zorder=1,
                   label="1.0 s reference")
        ax.set_title(panel_title)
        ax.set_ylabel("duration (s)")
        ax.set_xticks([0.0, 1.0], ["baseline", "DDP"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(0.03, 0.96, f"speedup (ratio of medians) = {speedup:.3f}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox={"boxstyle": "round", "facecolor": "white",
                      "edgecolor": "#999999", "alpha": 0.9})
        ax.text(0.02, 0.02, "individual repeats shown, not paired",
                transform=ax.transAxes, fontsize=7)
        if label == "replay":
            ax.text(0.02, 0.07, "replayed data, not a live measurement",
                    transform=ax.transAxes, fontsize=7)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(fig, out_path)


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def figure_title(user_title, label, kind):
    bits = []
    if user_title:
        bits.append(user_title)
    bits.append(f"phase K timing ({label})")
    bits.append(kind)
    return " - ".join(bits)


def write_metadata(out_dir, args, baseline, ddp, outputs):
    lines = ["phase_k_timing_plot_metadata"]
    add = lines.append
    add(f"generated_at_utc\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add(f"python\t{sys.version.split()[0]}")
    add(f"matplotlib\t{matplotlib.__version__}")
    add(f"numpy\t{np.__version__}")
    add(f"label\t{args.label}")
    add("command\t" + " ".join(shlex.quote(name) for name in sys.argv))
    add("reproducible_command\tuv run --offline --with-requirements "
        "scripts/plots/requirements-phase-k.txt python3 "
        + " ".join(shlex.quote(name) for name in sys.argv))
    for name, data in (("baseline", baseline), ("ddp", ddp)):
        if data is None:
            continue
        add(f"input_dir\t{name}\t{data.directory}")
        for stem, table in (("records", data.records), ("steps", data.steps),
                            ("repeats", data.repeats)):
            add(f"input\t{name}\t{stem}\t{table['path']}\t{sha256_file(table['path'])}"
                f"\trows={len(table['rows'])}")
        if data.metadata_path is not None:
            add(f"input\t{name}\tmetadata\t{data.metadata_path}"
                f"\t{sha256_file(data.metadata_path)}")
    for path in outputs:
        add(f"output\t{path}\t{sha256_file(path)}\tbytes={path.stat().st_size}")
    (out_dir / "figures-metadata.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_static(args):
    baseline_dir = Path(os.path.abspath(os.fspath(Path(args.baseline).expanduser())))
    ddp_dir = Path(os.path.abspath(os.fspath(Path(args.ddp).expanduser())))
    for name, directory in (("baseline", baseline_dir), ("ddp", ddp_dir)):
        if not directory.is_dir():
            _die(f"--{name} {directory} is not a directory")
    out_dir = Path(os.path.abspath(os.fspath(Path(args.out).expanduser())))
    if out_dir.exists() and not out_dir.is_dir():
        _die(f"--out {out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label
    baseline = ModeData("baseline", baseline_dir).load(strict=True)
    ddp = ModeData("ddp", ddp_dir).load(strict=True)
    outputs = []
    for name, data in (("baseline", baseline), ("ddp", ddp)):
        if not data.repeat_rows():
            _die(f"{data.repeats['path']}: no repeats for {name}")
    plot_value_or_step(
        baseline, ddp, "step", "completed global updates",
        figure_title(args.title, label, "completed updates versus elapsed training time"),
        label, out_dir / "timing-updates-vs-elapsed.png")
    outputs.append(out_dir / "timing-updates-vs-elapsed.png")
    plot_value_or_step(
        baseline, ddp, "samples_total", "processed samples (global batch count)",
        figure_title(args.title, label, "processed samples versus elapsed training time"),
        label, out_dir / "timing-samples-vs-elapsed.png")
    outputs.append(out_dir / "timing-samples-vs-elapsed.png")
    for task in ("a", "b"):
        plot_validation(
            baseline, ddp, task, "loss", f"{TASK_LABELS[task]} validation loss",
            figure_title(args.title, label, f"{TASK_LABELS[task]} validation loss"),
            label, out_dir / f"timing-val-loss-{task}.png")
        outputs.append(out_dir / f"timing-val-loss-{task}.png")
        plot_validation(
            baseline, ddp, task, "acc", f"{TASK_LABELS[task]} validation accuracy",
            figure_title(args.title, label, f"{TASK_LABELS[task]} validation accuracy"),
            label, out_dir / f"timing-val-acc-{task}.png")
        outputs.append(out_dir / f"timing-val-acc-{task}.png")
    plot_repeat_times(
        baseline, ddp,
        figure_title(args.title, label, "per-repeat wall time decomposition"),
        label, out_dir / "timing-repeat-times.png")
    outputs.append(out_dir / "timing-repeat-times.png")
    plot_speedup(
        baseline, ddp,
        figure_title(args.title, label,
                     "per-mode repeat durations and ratio-of-medians speedup"),
        label, out_dir / "timing-speedup.png")
    outputs.append(out_dir / "timing-speedup.png")
    write_metadata(out_dir, args, baseline, ddp, outputs)
    print(f"phase_k_timing_plot: wrote {len(outputs)} figures and "
          f"{out_dir / 'figures-metadata.txt'}")
    summary = {}
    for name, data in (("baseline", baseline), ("ddp", ddp)):
        rows = data.repeat_rows()
        total_key = "slowest_rank_s" if name == "ddp" else "t_total_s"
        train_key = "slowest_train_s" if name == "ddp" else "t_train_s"
        totals = np.array([row[total_key] for row in rows], dtype=float)
        trains = np.array([row[train_key] for row in rows], dtype=float)
        summary[name] = (totals, trains)
        print(f"  {name}: repeats={len(rows)} median_total_s={np.median(totals):.6f} "
              f"median_train_s={np.median(trains):.6f}")
    train_speedup = ratio_of_medians(summary["baseline"][1], summary["ddp"][1])
    total_speedup = ratio_of_medians(summary["baseline"][0], summary["ddp"][0])
    print("  speedup (ratio of medians, baseline / DDP): "
          f"training_only={train_speedup:.6f} total={total_speedup:.6f}")
    return 0


def read_records_partial(path: Path):
    """Read complete rows from a records.tsv that may still be growing."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    complete = data.endswith(b"\n")
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    elif not complete and lines:
        lines = lines[:-1]
    if not lines:
        return []
    header = lines[0].split("\t")
    if header != list(RECORD_COLUMNS):
        return []
    rows = []
    for raw in lines[1:]:
        fields = raw.split("\t")
        if len(fields) != len(RECORD_COLUMNS) or raw == "":
            continue
        row = dict(zip(RECORD_COLUMNS, fields))
        try:
            row["step"] = int(row["step"])
            for key in NUMERIC_RECORDS:
                if key != "step":
                    row[key] = float(row[key])
        except ValueError:
            continue
        rows.append(row)
    return rows


def load_baseline_for_follow(path) -> "ModeData | None":
    if path is None:
        return None
    directory = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if not directory.is_dir():
        _die(f"--baseline {directory} is not a directory")
    try:
        return ModeData("baseline", directory).load(strict=True)
    except SchemaError as exc:
        print(f"WARNING: ignoring baseline for live view: {exc}", file=sys.stderr)
        return None


def draw_live_figure(out_dir, rows, baseline, label):
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    if baseline is not None:
        for repeat, series in baseline.steps_by_repeat().items():
            ax.plot([row["train_s"] for row in series],
                    [row["step"] for row in series],
                    color=MODE_STYLE["baseline"]["color"], linewidth=1.6,
                    label="baseline (saved run)")
    if rows:
        xs = [row["train_s"] for row in rows]
        ys = [row["step"] for row in rows]
        ax.plot(xs, ys, color=MODE_STYLE["ddp"]["color"], linewidth=2.0, marker="o",
                markersize=3.0, label="live rank-0 records")
        last = rows[-1]
        ax.annotate(f"last step {last['step']}", xy=(xs[-1], ys[-1]),
                    xytext=(6, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel(XLABEL_TRAIN)
    ax.set_ylabel("completed global updates")
    ax.set_title(f"phase K timing live view ({label})")
    if label == "replay":
        ax.text(0.02, 0.02, "replayed records, not a live measurement",
                transform=ax.transAxes, fontsize=7)
    ax.grid(True, alpha=0.25)
    if rows or baseline is not None:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_figure(fig, Path(out_dir) / LIVE_FIGURE)


def follow(args):
    records = Path(os.path.abspath(os.fspath(Path(args.records).expanduser())))
    out_dir = Path(os.path.abspath(os.fspath(Path(args.out).expanduser())))
    out_dir.mkdir(parents=True, exist_ok=True)
    status_log = Path(os.path.abspath(os.fspath(
        Path(args.status_log).expanduser()))) if args.status_log else out_dir / "status.log"
    status_log.parent.mkdir(parents=True, exist_ok=True)
    label = "replay" if args.replay else args.label
    if args.replay:
        print("NOTICE: --replay is set; all viewer output is labelled replay "
              "and must not be presented as live.", file=sys.stderr)
    baseline = load_baseline_for_follow(args.baseline)
    done_marker = records.parent / "done"
    interval = max(0.1, float(args.interval))
    last_size = -1
    stable_polls = 0
    last_change = None
    print(f"[viewer] watching {records} (interval {interval:g}s) out={out_dir}")
    while True:
        now = time.monotonic()
        exists = records.is_file()
        size = records.stat().st_size if exists else -1
        if size != last_size:
            last_change = now
            stable_polls = 0
        else:
            stable_polls += 1
        if last_change is None:
            last_change = now
        rows = read_records_partial(records) if exists else []
        done_exists = done_marker.is_file()
        if not exists:
            status = "WAITING"
        elif done_exists and stable_polls >= 2:
            status = "DONE"
        else:
            status = "RUNNING"
        age = (now - last_change) if exists else -1.0
        last_step = rows[-1]["step"] if rows else 0
        updates_per_s = rows[-1]["updates_per_s"] if rows else 0.0
        wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = (f"[viewer] wall={wall} status={status} last_step={last_step} "
                f"record_age_s={age:.2f} updates_per_s={updates_per_s:.6f}")
        print(line, flush=True)
        with open(status_log, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        draw_live_figure(out_dir, rows, baseline, label)
        last_size = size
        if status == "DONE":
            print(f"[viewer] done marker found; wrote {out_dir / LIVE_FIGURE}")
            return 0
        time.sleep(interval)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Plot Phase K timing-stretch records and run the live viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", default=None,
                        help="directory with baseline records.tsv/steps.tsv/repeats.tsv")
    parser.add_argument("--ddp", default=None,
                        help="directory with DDP records.tsv/steps.tsv/repeats.tsv")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--title", default=None, help="optional figure title prefix")
    parser.add_argument("--label", default=None,
                        choices=("offline", "replay", "live"),
                        help="provenance label shown in every figure "
                             "(default: offline for static, live for --follow, replay with --replay)")
    parser.add_argument("--follow", action="store_true",
                        help="poll a growing records.tsv and refresh timing-live.png")
    parser.add_argument("--records", default=None,
                        help="live records.tsv path (follow mode)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="live polling interval in seconds")
    parser.add_argument("--status-log", default=None,
                        help="append live status lines to this file")
    parser.add_argument("--replay", action="store_true",
                        help="label follow output as replay, never as live")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.follow:
        if not args.records:
            _die("--follow requires --records <records.tsv>")
        if args.replay:
            args.label = "replay"
        elif args.label is None:
            args.label = "live"
        return follow(args)
    if not args.baseline or not args.ddp:
        _die("static mode requires --baseline <dir> and --ddp <dir>")
    if args.label is None:
        args.label = "offline"
    return generate_static(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SchemaError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
