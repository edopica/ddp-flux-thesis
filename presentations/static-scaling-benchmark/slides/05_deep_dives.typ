// 05_deep_dives.typ - Technical deep dives
#import "@preview/touying:0.6.1": *
#import "../components/cards.typ": *
#import "../components/charts.typ": *

= Technical Deep Dives

== The First-Repeat Outlier

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 12pt,
  align: top,
  card(
    title: "What happened on the first repeat",
    badge: badge("one-off", fill: cand-current-bg, text-color: cand-current-dark),
  )[
    - Flux (guarded), single GPU:
      - First repeat: #text(weight: "bold", fill: cand-current-dark)[680.8 s]
      - Repeats two to five: 91.0–91.1 s
    - The first repeat is about 7.5 times slower. Later repeats agree
      closely, within a tenth of a second.
    - The five-repeat *median* is 91.1 s. Dropping the first repeat changes
      it by less than 0.02%, so the reported median is not sensitive to that
      run.
  ],
  card(
    title: "Why the median, not the mean",
    badge: badge("method", fill: rgb("#e2e8f0"), text-color: slate-dark),
  )[
    - The mean is dragged upward by the first repeat.
    - A mean-based speedup would imply *more than linear scaling* on two and
      four GPUs, which is not plausible for this workload.
    - The median is the more representative summary when one run is
      disturbed by an external event.
  ],
)

#v(8pt)

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  callout(
    title: "Cause: still unconfirmed",
    fill: rgb("#f8fafc"),
    stroke: 1pt + slate-border,
    title-color: slate-dark,
  )[
    The most plausible explanation is a cold file-cache load of the training
    data on the first read. No page-fault, disk, or cache telemetry was
    recorded, so this remains a hypothesis rather than a claim.
  ],
  callout(
    title: "It is not compilation time",
    fill: rgb("#f8fafc"),
    stroke: 1pt + slate-border,
    title-color: slate-dark,
  )[
    Twenty warm-up updates run before the timer starts. Julia compilation and
    GPU kernel selection happen there. The timer starts only after the
    weights and optimizer state are reset.
  ],
)

== Repeat Consistency

#align(center)[
  #chart-dispersion(show-header: false)
]

#v(2pt)

#text(size: 0.9em)[
  *Takeaway:* Repeats two to five agree closely for all three candidates.
  The median removes the single cold-start event without discarding any
  run.
]

== Guard Cost and Synchronization Strategy

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  align: top,
  card(
    title: "Flux (guarded)",
    title-color: cand-current-dark,
    badge: badge("64 calls", fill: cand-current-bg, text-color: cand-current-dark),
  )[
    - One presence check, so no GPU starts an update alone.
    - One gradient all-reduce per parameter array: 62 calls.
    - One finite-value check that rejects non-finite gradients.
    - 64 synchronization calls per update, none of them overlapped with
      computation.
  ],
  card(
    title: "Flux (static)",
    title-color: cand-historical-dark,
    badge: badge("62 calls", fill: cand-historical-bg, text-color: cand-historical-dark),
  )[
    - One gradient all-reduce per parameter array: 62 calls.
    - No presence check and no finite-value check.
    - The leanest path, and the baseline for the cost of the two extra
      checks.
  ],
  card(
    title: "PyTorch DDP",
    title-color: cand-pytorch-dark,
    badge: badge("bucketed", fill: cand-pytorch-bg, text-color: cand-pytorch-dark),
  )[
    - Gradients are packed into large contiguous buffers.
    - Fewer, larger synchronization calls.
    - Communication overlaps the backward pass instead of following it.
  ],
)

#v(8pt)

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 12pt,
  card(title: "What the numbers show")[
    Flux (static) is about 9% faster than Flux (guarded) at two and four
    GPUs. The gap is consistent with per-update guard work, but the two
    Flux builds also differ in other implementation details.
  ],
  callout(
    title: "What this does not establish",
    fill: rgb("#f8fafc"),
    stroke: 1pt + slate-border,
    title-color: slate-dark,
  )[
    The throughput difference cannot be attributed to the two extra checks
    alone. The benchmark compares two builds, not two isolated lines of code.
  ],
)

== Numerical Reproducibility: What It Does and Does Not Support

#grid(
  columns: (1.05fr, 0.95fr),
  gutter: 12pt,
  align: top,
  card(title: "What was compared")[
    - Parameter snapshots after the same 390 updates, with identical data,
      initial weights, and learning-rate schedule.
    - 62 parameter arrays per model, about 11.27 million values.
    - Six comparisons between GPU counts within a framework; nine across
      frameworks.
    - Within each distributed run, all ranks agree on the final state;
      the differences are between separate runs.
  ],
  card(title: "What the snapshots show")[
    - *Weights:* about 2% relative L2, the same order as the difference
      between GPU counts of one framework.
    - *AdamW moments:* 6–9% for the second moment and 18–30% for the first,
      as expected for state accumulated over all updates.
    - *Schedule counters:* exact match.
    - *Low-magnitude leaves:* one GroupNorm bias reaches 37% on an absolute
      difference of 0.003, because its reference norm is close to zero.
  ],
)

#v(8pt)

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  callout(title: "Why Flux repeats differ")[
    - Five repeats per configuration: Flux gives five distinct final-state
      hashes at every GPU count; PyTorch gives one.
    - Fresh processes can select numerically different cuDNN convolution
      algorithms: selection is timing-based and has no determinism filter.
      This happens at one GPU too, where there are no collectives at all.
    - PyTorch requires deterministic algorithms and disables autotuning.
  ],
  callout(title: "What that means for the comparison")[
    Run-to-run variation is the same order as the differences between GPU
    counts and between frameworks. With one snapshot per configuration it
    cannot be separated from them. The comparison is descriptive; it is not
    evidence that one candidate computes the wrong answer.
  ],
)
