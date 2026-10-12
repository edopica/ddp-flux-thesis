#!/usr/bin/env python3
"""Draw the Phase K K2 conditional network and its six-step routing cycle.

The figure content is parsed from the repository sources, not invented:

- the model dimensions come from the K2 line of
  ``scripts/examples/conditional_graph_demo.jl``
  (``CGModel.build_toy_model(seed_init; indim=784, hidden=32, nclasses=10)``);
- the layer definitions come from ``scripts/examples/conditional_graph/model.jl``
  (``TwoHeadModel``: one ``Dense(indim => hidden, tanh)`` trunk and two
  ``Dense(hidden => nclasses)`` heads);
- the routing cycle and its purpose strings come from ``ROUTING_CYCLE`` in
  ``scripts/examples/conditional_graph_demo.jl``;
- the global batch default comes from the ``prepare`` option parser.

A missing or ambiguous source pattern stops the run with exit status 2 before
any figure is written. The figures are deterministic; ``figures-metadata.txt``
records the parsed values and the SHA256 of every consumed source and produced
figure.

Reproducible invocation in this checkout (system python3 has no numpy or
matplotlib; ``uv`` provides them in an ephemeral environment):

    uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_network.py \
        --out reports/figures/phase-k-network [--worlds 2,4]

See scripts/plots/README.md for details.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    print(
        f"ERROR: missing python dependency {exc.name!r}.\n"
        f"Run with: uv run --offline --with matplotlib --with numpy python3 {sys.argv[0]} ...",
        file=sys.stderr,
    )
    sys.exit(2)

FIGURE_DPI = 160
NETWORK_FIGURE = "phase-k-k2-network.png"
ROUTING_FIGURE = "phase-k-k2-routing.png"

COLOR_TRUNK_FACE, COLOR_TRUNK_EDGE = "#dfe8f5", "#3b5b8c"
COLOR_HEAD_A_FACE, COLOR_HEAD_A_EDGE = "#d8ecf8", "#1f77b4"
COLOR_HEAD_B_FACE, COLOR_HEAD_B_EDGE = "#fde6ce", "#d2691e"
COLOR_ABSENT_FACE, COLOR_ABSENT_EDGE = "#f0f0f0", "#9e9e9e"
COLOR_INPUT_FACE, COLOR_INPUT_EDGE = "#eeeeee", "#666666"
COLOR_OUTPUT_FACE, COLOR_OUTPUT_EDGE = "#f3f0fa", "#6a5fa5"
COLOR_TEXT = "#20242b"
TASK_COLORS = {"a": COLOR_HEAD_A_EDGE, "b": COLOR_HEAD_B_EDGE}

MODEL_BUILD_RE = re.compile(
    r"CGModel\.build_toy_model\(seed_init;\s*indim=(\d+),\s*hidden=(\d+),\s*nclasses=(\d+)\)"
)
MODEL_TRUNK_RE = re.compile(r"Dense\(indim => hidden, tanh")
MODEL_HEAD_RE = re.compile(r"Dense\(hidden => nclasses;")
ROUTING_ENTRY_RE = re.compile(
    r"\(;\s*cycle\s*=\s*(\d+),\s*(.*?),\s*purpose\s*=\s*\"([^\"]+)\"\)", re.DOTALL
)
ROUTING_ALL_RE = re.compile(r"all\s*=\s*:(\w+)")
ROUTING_RANK0_RE = re.compile(r"rank0\s*=\s*:(\w+),\s*others\s*=\s*:(\w+)")
GLOBAL_BATCH_RE = re.compile(r'int_opt\(opts, "global-batch", (\d+)\)')


class SourceError(Exception):
    """A repository source does not expose the expected Phase K pattern."""


def _die(message: str) -> None:
    raise SourceError(message)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source(path: Path, label: str) -> str:
    if not path.is_file():
        _die(f"missing source {label}: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        _die(f"source {label} is not UTF-8: {exc}")


def _one(pattern: re.Pattern, text: str, label: str):
    matches = pattern.findall(text)
    if len(matches) != 1:
        _die(f"expected exactly one {label} match, found {len(matches)}")
    return matches[0]


def parse_sources(repo_root: Path):
    demo_path = repo_root / "scripts" / "examples" / "conditional_graph_demo.jl"
    model_path = repo_root / "scripts" / "examples" / "conditional_graph" / "model.jl"
    schedule_path = repo_root / "scripts" / "examples" / "conditional_graph" / "schedule.jl"
    common_path = repo_root / "scripts" / "examples" / "conditional_graph" / "common.jl"

    demo = _read_source(demo_path, "conditional_graph_demo.jl")
    model = _read_source(model_path, "model.jl")
    _read_source(schedule_path, "schedule.jl")
    _read_source(common_path, "common.jl")

    indim_s, hidden_s, nclasses_s = _one(
        MODEL_BUILD_RE, demo, "K2 build_toy_model(seed_init; indim=..., hidden=..., nclasses=...)"
    )
    indim, hidden, nclasses = int(indim_s), int(hidden_s), int(nclasses_s)
    if not MODEL_TRUNK_RE.search(model):
        _die(f"{model_path}: No trunk Dense(indim => hidden, tanh) definition")
    if not MODEL_HEAD_RE.search(model):
        _die(f"{model_path}: No head Dense(hidden => nclasses; ...) definition")

    routing = []
    for cycle_s, body, purpose in ROUTING_ENTRY_RE.findall(demo):
        cycle = int(cycle_s)
        all_match = ROUTING_ALL_RE.search(body)
        rank0_match = ROUTING_RANK0_RE.search(body)
        if all_match:
            routing.append((cycle, "all", all_match.group(1), None, purpose))
        elif rank0_match:
            routing.append((cycle, "split", rank0_match.group(1), rank0_match.group(2), purpose))
        else:
            _die(f"cannot parse ROUTING_CYCLE entry {cycle}: {body!r}")
    routing.sort(key=lambda entry: entry[0])
    if [entry[0] for entry in routing] != [1, 2, 3, 4, 5, 6]:
        _die(f"expected ROUTING_CYCLE cycles 1..6, found {[e[0] for e in routing]}")

    global_batch = int(_one(GLOBAL_BATCH_RE, demo, "prepare --global-batch default"))

    trunk_params = indim * hidden + hidden
    head_params = hidden * nclasses + nclasses
    return {
        "paths": {
            "demo": demo_path,
            "model": model_path,
            "schedule": schedule_path,
            "common": common_path,
        },
        "indim": indim,
        "hidden": hidden,
        "nclasses": nclasses,
        "trunk_params": trunk_params,
        "head_params": head_params,
        "total_params": trunk_params + 2 * head_params,
        "routing": routing,
        "global_batch": global_batch,
    }


def add_box(ax, x, y, w, h, title, lines=(), face="#ffffff", edge="#333333",
            title_size=10.5, body_size=8.5, linestyle="solid", text_color=COLOR_TEXT,
            title_color=None, zorder=3):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            linewidth=1.3, edgecolor=edge, facecolor=face, linestyle=linestyle,
            zorder=zorder,
        )
    )
    items = [title, *lines]
    n = len(items)
    cx = x + w / 2.0
    for index, item in enumerate(items):
        item_y = y + h * (n - index - 0.5) / n
        ax.text(
            cx, item_y, item,
            ha="center", va="center", zorder=zorder + 1,
            fontsize=title_size if index == 0 else body_size,
            fontweight="bold" if index == 0 else "normal",
            color=(title_color or text_color) if index == 0 else text_color,
        )


def add_arrow(ax, start, end, color="#3b5b8c", rad=0.0, lw=1.5, linestyle="solid",
              zorder=2):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
            color=color, linestyle=linestyle,
            connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5,
            zorder=zorder,
        )
    )


def draw_architecture(path: Path, parsed):
    fig = plt.figure(figsize=(16.0, 9.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 96)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        80, 93.6,
        "Phase K K2 network: task-conditioned MNIST / Fashion-MNIST "
        "(TwoHeadModel, Float32)",
        ha="center", va="center", fontsize=15, fontweight="bold", color=COLOR_TEXT,
    )
    ax.text(
        80, 90.2,
        "one shared trunk; the local batch selects exactly one head",
        ha="center", va="center", fontsize=10.5, color="#555555",
    )

    pattern = np.fromfunction(
        lambda i, j: np.exp(-(((i - 14.0) ** 2 + (j - 14.0) ** 2) / 180.0)), (28, 28)
    )
    ax.imshow(pattern, extent=(4, 22, 56, 74), cmap="Greys", vmin=0.0, vmax=1.0,
              interpolation="nearest", zorder=1)
    ax.add_patch(
        FancyBboxPatch((3.4, 55.4), 19.2, 19.2,
                       boxstyle="round,pad=0.0,rounding_size=1.0", linewidth=1.3,
                       edgecolor=COLOR_INPUT_EDGE, facecolor="none", zorder=2)
    )
    ax.text(13, 52.6, "input x", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=COLOR_TEXT)
    ax.text(13, 49.6, "28 x 28 grayscale, row-major flatten", ha="center", va="center",
            fontsize=8.5, color=COLOR_TEXT)
    ax.text(13, 47.0, f"({parsed['indim']},) Float32", ha="center", va="center",
            fontsize=8.5, color=COLOR_TEXT)
    add_arrow(ax, (23.5, 65), (30.5, 65))

    add_box(
        ax, 31, 51, 30, 28,
        f"trunk : Dense({parsed['indim']} => {parsed['hidden']}, tanh)",
        [
            f"W {parsed['hidden']} x {parsed['indim']} = "
            f"{parsed['hidden'] * parsed['indim']:,}".replace(",", " "),
            f"b {parsed['hidden']}",
            f"{parsed['trunk_params']:,}".replace(",", " ") + " parameters",
            "task-independent, always active",
        ],
        face=COLOR_TRUNK_FACE, edge=COLOR_TRUNK_EDGE, title_color=COLOR_TRUNK_EDGE,
    )

    ax.add_patch(
        FancyBboxPatch((66, 53), 4.6, 24,
                       boxstyle="round,pad=0.0,rounding_size=0.8", linewidth=1.3,
                       edgecolor=COLOR_TRUNK_EDGE, facecolor="#eef4fb", zorder=3)
    )
    ax.text(68.3, 65, f"h ({parsed['hidden']})", ha="center", va="center", fontsize=9,
            rotation=90, color=COLOR_TRUNK_EDGE, zorder=4)
    add_arrow(ax, (61.5, 65), (65.5, 65))

    add_arrow(ax, (71, 67), (78.5, 76), rad=-0.12)
    add_arrow(ax, (71, 63), (78.5, 54), rad=0.12)

    add_box(
        ax, 79, 69, 34, 14,
        f"head_a : Dense({parsed['hidden']} => {parsed['nclasses']})",
        ["MNIST digits (task :a)", f"{parsed['head_params']} parameters"],
        face=COLOR_HEAD_A_FACE, edge=COLOR_HEAD_A_EDGE, title_color=COLOR_HEAD_A_EDGE,
    )
    add_box(
        ax, 79, 47, 34, 14,
        f"head_b : Dense({parsed['hidden']} => {parsed['nclasses']})",
        ["Fashion-MNIST clothing (task :b)", f"{parsed['head_params']} parameters"],
        face=COLOR_HEAD_B_FACE, edge=COLOR_HEAD_B_EDGE, title_color=COLOR_HEAD_B_EDGE,
    )

    add_arrow(ax, (113.5, 76), (118.5, 76), color=COLOR_HEAD_A_EDGE)
    add_arrow(ax, (113.5, 54), (118.5, 54), color=COLOR_HEAD_B_EDGE)

    add_box(
        ax, 119, 69, 17, 14,
        f"logits_a ({parsed['nclasses']})", ["softmax", "cross-entropy L_a"],
        face=COLOR_OUTPUT_FACE, edge=COLOR_OUTPUT_EDGE, title_color=COLOR_OUTPUT_EDGE,
        title_size=9.5, body_size=8.5,
    )
    add_box(
        ax, 119, 47, 17, 14,
        f"logits_b ({parsed['nclasses']})", ["softmax", "cross-entropy L_b"],
        face=COLOR_OUTPUT_FACE, edge=COLOR_OUTPUT_EDGE, title_color=COLOR_OUTPUT_EDGE,
        title_size=9.5, body_size=8.5,
    )
    ax.text(146, 76, "per-task loss\non the selected head only", ha="left", va="center",
            fontsize=8.5, color="#555555")
    ax.text(146, 54, "the unused head is\nnever called", ha="left", va="center",
            fontsize=8.5, color="#555555")

    ax.text(
        80, 41.5,
        f"total {parsed['total_params']:,}".replace(",", " ")
        + " Float32 parameters = trunk"
        f" {parsed['trunk_params']:,}".replace(",", " ")
        + " + 2 x head"
        f" {parsed['head_params']}"
        + "; model_definition_version 1",
        ha="center", va="center", fontsize=10, color=COLOR_TEXT,
    )
    ax.plot([2, 158], [36.5, 36.5], color="#cccccc", linewidth=1.0)

    ax.text(
        80, 33.6,
        "conditional execution: the host-side branch `(m::TwoHeadModel)(x, head)` "
        "runs exactly one head per local batch",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=COLOR_TEXT,
    )

    for label, head, lane_y, lane_name in (
        ("task :a - MNIST", "a", 16.0, "L_a"),
        ("task :b - Fashion-MNIST", "b", 4.0, "L_b"),
    ):
        ax.text(3, lane_y + 4.0, label, ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=TASK_COLORS[head])
        add_box(
            ax, 30, lane_y, 19, 8, "trunk", ["active"],
            face=COLOR_TRUNK_FACE, edge=COLOR_TRUNK_EDGE, title_size=9, body_size=8,
            title_color=COLOR_TRUNK_EDGE,
        )
        head_name = f"head_{head}"
        add_box(
            ax, 53, lane_y, 23, 8, head_name, ["active"],
            face=TASK_COLORS[head], edge=TASK_COLORS[head],
            title_size=9, body_size=8, title_color="#ffffff",
        )
        add_box(
            ax, 80, lane_y, 17, 8, lane_name, ["scalar loss"],
            face=COLOR_OUTPUT_FACE, edge=COLOR_OUTPUT_EDGE, title_size=9.5,
            body_size=8, title_color=COLOR_OUTPUT_EDGE,
        )
        add_arrow(ax, (49.5, lane_y + 4.0), (52.5, lane_y + 4.0))
        add_arrow(ax, (76.5, lane_y + 4.0), (79.5, lane_y + 4.0))
        skip = "b" if head == "a" else "a"
        add_box(
            ax, 103, lane_y, 23, 8, f"head_{skip}", ["absent"],
            face=COLOR_ABSENT_FACE, edge=COLOR_ABSENT_EDGE, linestyle="dashed",
            title_size=9, body_size=8, title_color=COLOR_ABSENT_EDGE,
        )
        ax.text(
            129, lane_y + 4.0,
            f"not executed locally, no AD node;\nzero buffer if globally present",
            ha="left", va="center", fontsize=8.3, color="#555555",
        )

    ax.text(
        80, 0.8,
        "DDP protocol per update: one host-Float32 presence allreduce over all 6 "
        "trainable leaves, then every rank allreduces the globally present leaves in "
        "canonical order = 5 collectives when one task is globally present "
        "(cycles 3-4), 7 when both are (cycles 1-2, 5-6)",
        ha="center", va="center", fontsize=9, color="#333333",
    )

    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def heads_for_cycle(entry, rank):
    cycle, kind, first, second, _purpose = entry
    if kind == "all":
        return first
    return first if rank == 0 else second


def draw_routing(path: Path, parsed, worlds):
    routing = parsed["routing"]
    width = max(14.0, 5.0 * len(worlds))
    fig, axes = plt.subplots(
        1, len(worlds), figsize=(width, 6.0),
        gridspec_kw={"wspace": 0.30},
    )
    if len(worlds) == 1:
        axes = [axes]

    for ax, workers in zip(axes, worlds):
        ax.set_xlim(-2.4, workers - 1 + 7.6)
        ax.set_ylim(7.1, -1.9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(
            (workers - 1) / 2.0, -1.55,
            f"R = {workers} ranks", ha="center", va="center", fontsize=13,
            fontweight="bold", color=COLOR_TEXT,
        )
        ax.text(-0.85, -0.25, "rank", ha="right", va="center",
                fontsize=9, color="#444444")
        for rank in range(workers):
            ax.text(rank, -0.25, str(rank), ha="center", va="center",
                    fontsize=9, color="#444444")
        for entry in routing:
            cycle, _kind, _first, _second, purpose = entry
            present = set()
            for rank in range(workers):
                head = heads_for_cycle(entry, rank)
                present.add(head)
                ax.add_patch(
                    FancyBboxPatch(
                        (rank - 0.42, cycle - 0.38), 0.84, 0.76,
                        boxstyle="round,pad=0.0,rounding_size=0.18", linewidth=1.2,
                        edgecolor=TASK_COLORS[head], facecolor=TASK_COLORS[head],
                        zorder=3,
                    )
                )
                ax.text(rank, cycle, head.upper(), ha="center", va="center",
                        fontsize=11, fontweight="bold", color="#ffffff", zorder=4)
            ax.text(-0.85, cycle, f"step {cycle}", ha="right", va="center",
                    fontsize=9, color="#444444")
            presence = "A + B" if present == {"a", "b"} else f"{next(iter(present)).upper()} only"
            ax.text(workers - 0.15, cycle,
                    f"{purpose}\n[global presence: {presence}]",
                    ha="left", va="center", fontsize=8.3, color="#333333")

    fig.suptitle(
        "Phase K K2 routing cycle: head trained by each rank per step "
        "(repeats every 6 steps)",
        fontsize=14, fontweight="bold", color=COLOR_TEXT, y=0.99,
    )
    local_batches = ", ".join(
        f"R={workers}: {parsed['global_batch'] // workers}" for workers in worlds
    )
    fig.text(
        0.5, 0.035,
        f"A = MNIST (task :a), B = Fashion-MNIST (task :b); global batch "
        f"{parsed['global_batch']}, local batch = global / R ({local_batches})",
        ha="center", va="center", fontsize=9.5, color="#333333",
    )
    fig.text(
        0.5, 0.005,
        "parsed from ROUTING_CYCLE in scripts/examples/conditional_graph_demo.jl "
        "and six_step_head in scripts/examples/conditional_graph/schedule.jl",
        ha="center", va="center", fontsize=8, color="#666666",
    )
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Draw the Phase K K2 conditional network and routing cycle.",
    )
    parser.add_argument(
        "--repo", default=str(Path(__file__).resolve().parents[2]),
        help="repository root holding scripts/examples (default: script location)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output directory (default: <repo>/reports/figures/phase-k-network)",
    )
    parser.add_argument(
        "--worlds", default="2,4",
        help="comma-separated rank counts for the routing figure (default: 2,4)",
    )
    return parser.parse_args(argv)


def write_metadata(out_dir: Path, parsed, worlds, outputs):
    lines = ["phase_k_network_plot_metadata"]
    add = lines.append
    add(f"generated_at_utc\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add(f"python\t{sys.version.split()[0]}")
    add(f"matplotlib\t{matplotlib.__version__}")
    add(f"numpy\t{np.__version__}")
    add("command\t" + " ".join(shlex.quote(name) for name in sys.argv))
    add("reproducible_command\tuv run --offline --with-requirements "
        "scripts/plots/requirements-phase-k.txt python3 "
        + " ".join(shlex.quote(name) for name in sys.argv))
    for name, path in parsed["paths"].items():
        add(f"source\t{name}\t{path}\t{sha256_file(path)}")
    add(f"model\tindim\t{parsed['indim']}\thidden\t{parsed['hidden']}"
        f"\tnclasses\t{parsed['nclasses']}")
    add(f"parameters\ttrunk\t{parsed['trunk_params']}\thead\t{parsed['head_params']}"
        f"\ttotal\t{parsed['total_params']}")
    add(f"global_batch\t{parsed['global_batch']}")
    add(f"worlds\t{','.join(str(workers) for workers in worlds)}")
    for cycle, kind, first, second, purpose in parsed["routing"]:
        add(f"routing\tcycle={cycle}\t{kind}\t{first}\t{second}\t{purpose}")
    for path in outputs:
        add(f"output\t{path}\t{sha256_file(path)}\tbytes={path.stat().st_size}")
    (out_dir / "figures-metadata.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(os.path.abspath(os.fspath(Path(args.repo).expanduser())))
    if not repo_root.is_dir():
        _die(f"--repo {repo_root} is not a directory")
    out_dir = Path(
        os.path.abspath(os.fspath(Path(args.out).expanduser()))
        if args.out else repo_root / "reports" / "figures" / "phase-k-network"
    )
    if out_dir.exists() and not out_dir.is_dir():
        _die(f"--out {out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        worlds = sorted({int(part) for part in args.worlds.split(",") if part})
    except ValueError:
        _die(f"--worlds must be comma-separated integers, got {args.worlds!r}")
    if not worlds or worlds[0] < 2:
        _die(f"--worlds must contain integers >= 2, got {args.worlds!r}")

    parsed = parse_sources(repo_root)
    for cycle, kind, first, second, _purpose in parsed["routing"]:
        for head in (first, second):
            if head is not None and head not in ("a", "b"):
                _die(f"unknown head {head!r} in routing cycle {cycle}")

    outputs = [out_dir / NETWORK_FIGURE, out_dir / ROUTING_FIGURE]
    draw_architecture(outputs[0], parsed)
    draw_routing(outputs[1], parsed, worlds)
    write_metadata(out_dir, parsed, worlds, outputs)

    print(f"phase_k_network_plot: wrote {len(outputs)} figures and "
          f"{out_dir / 'figures-metadata.txt'}")
    print(f"  model: indim={parsed['indim']} hidden={parsed['hidden']} "
          f"nclasses={parsed['nclasses']} total_params={parsed['total_params']}")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
