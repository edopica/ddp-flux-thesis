// 01_overview.typ - Executive summary, the three candidates, and the study design
#import "@preview/touying:0.6.1": *
#import "../components/cards.typ": *

= Overview & Study Design

== Executive Summary: What the Benchmark Shows

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  card(title: "1 · Both Flux builds scale")[
    Adding GPUs on a single node gives close-to-proportional throughput gains.
    Both Flux builds gain throughput from one to four GPUs.
  ],
  card(title: "2 · Flux is competitive with PyTorch")[
    Throughput is close to PyTorch at every GPU count. Framework differences
    are smaller than the gain from doubling the GPU count.
  ],
)

#v(8pt)

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  card(title: "3 · Guards cost throughput at two GPUs")[
    The Flux build with runtime guards trails at two GPUs. At four GPUs it is
    still behind Flux (static), but it moves ahead of PyTorch and has the most
    efficient two-to-four-GPU transition of the three.
  ],
  card(title: "4 · The result is stable across repeats")[
    Re-measuring every configuration in alternating order produced stable
    medians. One slow first repeat does not move the reported values.
  ],
)

#v(8pt)

#callout(
  title: "Verdict",
  fill: rgb("#f8fafc"),
  stroke: 1pt + slate-border,
  title-color: slate-dark,
)[
  Strong scaling holds on this node: all three implementations convert
  additional GPUs into useful throughput, and the Flux builds are credible
  alternatives to PyTorch. The result is a *descriptive throughput
  comparison*. It says nothing about convergence, accuracy, or
  time-to-quality.
]

== The Three Candidates: Two Flux Builds and PyTorch

#text(size: 0.9em, fill: slate-text)[
  All three candidates run the same model, data order, optimizer, and number
  of updates. Their main difference is how they synchronize gradients.
  #emph["Static" here means no runtime guards, not a static computation graph.]
]

#v(6pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card(
    title: "Flux (static)",
    title-color: cand-historical-dark,
    badge: badge("earlier build", fill: cand-historical-bg, text-color: cand-historical-dark),
  )[
    The original Flux data-parallel implementation.
    #v(3pt)
    - One synchronization call per gradient array.
    - No runtime checks around synchronization.
    - Leanest communication path; this is the baseline for measuring the
      cost of the newer safety layer.
  ],
  card(
    title: "Flux (guarded)",
    title-color: cand-current-dark,
    badge: badge("current build", fill: cand-current-bg, text-color: cand-current-dark),
  )[
    The Flux build with runtime guards.
    #v(3pt)
    - Same per-array synchronization as Flux (static).
    - Adds a presence check: every GPU confirms it has started the update.
    - Adds a finite-value check: non-finite gradients are rejected before
      the optimizer applies them.
  ],
  card(
    title: "PyTorch DDP",
    title-color: cand-pytorch-dark,
    badge: badge("external reference", fill: cand-pytorch-bg, text-color: cand-pytorch-dark),
  )[
    The comparison point outside Julia.
    #v(3pt)
    - Native distributed data parallel.
    - Packs gradients into large buffers instead of one synchronization call
      per array.
    - Overlaps communication with the backward pass; deterministic kernels
      are enabled.
  ],
)

#v(8pt)

#callout(
  title: "What is held constant",
  fill: rgb("#f8fafc"),
  stroke: 1pt + slate-border,
  title-color: slate-dark,
)[
  Model weights, sample order, batch composition, optimizer schedule, and
  update count are identical in all runs. The intended contrast is the
  *synchronization layer*: guards in Flux, bucketing in PyTorch. The builds
  also differ in other implementation details.
]

== Study Design: Fixed Work, More GPUs

#grid(
  columns: (1.05fr, 0.95fr),
  gutter: 12pt,
  align: top,
  card(title: "Research question")[
    When the global batch is fixed, how does training throughput change as
    the same run moves from one to two and four GPUs? And how do the two
    Flux builds compare with PyTorch along the way?
    #v(3pt)
    This is *strong scaling*: the problem stays the same size while the
    hardware grows.
    #v(4pt)
    *Platform:* one Leonardo node with four A100-SXM-64GB GPUs connected by
    NVLink. Tiny ImageNet-200 ResNet-18, AdamW, global batch 256.
  ],
  card(title: "Scaling design")[
    The global batch never changes. Each additional GPU receives a smaller
    slice of the same work, so every run performs the same 390 optimizer
    updates.
    #v(4pt)
    #let shard(count, per) = {
      block(width: 100%)[
        #grid(
          columns: (1fr,) * count,
          gutter: 3pt,
          ..range(count).map(_ => rect(
            width: 100%,
            height: 16pt,
            radius: 2pt,
            fill: rgb("#cbd5e1"),
            align(center + horizon)[#text(size: 0.64em, weight: "bold", fill: slate-dark)[#per]],
          )),
        )
      ]
    }
    #grid(
      columns: (0.55fr, 1.45fr),
      gutter: 8pt,
      row-gutter: 5pt,
      align: (right + horizon, left + horizon),
      text(size: 0.78em, weight: "bold")[1 GPU],
      shard(1, [256]),
      text(size: 0.78em, weight: "bold")[2 GPUs],
      shard(2, [128]),
      text(size: 0.78em, weight: "bold")[4 GPUs],
      shard(4, [64]),
    )
  ],
)

#v(8pt)

#card(title: "Measures")[
  Throughput $Q_N = (K B) / T_N$ #h(14pt) Speedup $S_N = T_1 / T_N$ #h(14pt) Efficiency $E_N = S_N / N$
  #v(3pt)
  $K = 390$ updates, $B = 256$ images, $T_N$ = epoch time on $N$ GPUs.
  #v(4pt)
  *Scope:* fixed-work throughput only. No validation loss, accuracy, or
  convergence claim.
]
