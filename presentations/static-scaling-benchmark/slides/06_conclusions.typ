// 06_conclusions.typ - Takeaways, limits, and next steps
#import "@preview/touying:0.6.1": *
#import "../components/cards.typ": *

= Conclusions & Next Steps

== What the Benchmark Supports

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 12pt,
  align: top,
  card(title: "Findings")[
    - *Strong scaling holds on this node.* Both Flux builds turn additional
      GPUs into useful throughput and stay competitive with PyTorch.
    - *Flux (static) is the throughput leader at four GPUs.* Flux (guarded)
      remains behind it, but has the best two-to-four-GPU step efficiency.
    - *The result is not an artifact of run order.* Re-measuring every
      configuration in alternating order produced stable medians. Dropping
      the one anomalous first repeat changed the affected median by less
      than 0.02%.
  ],
  card(title: "What it does not claim")[
    - *No convergence claim.* The measured epoch sits inside the
      learning-rate ramp; there is no validation accuracy or time-to-quality
      result.
    - *No claim of numerical equivalence.* Aggregate weight difference is
      about 2% relative L2; individual arrays can differ more.
    - *No causal claim about the guards.* The two Flux builds differ in more
      than the two guard calls.
    - *No generalization beyond this node.* One node with four GPUs.
  ],
)

#v(8pt)

#callout(
  title: "Bottom line",
  fill: rgb("#f8fafc"),
  stroke: 1pt + slate-border,
  title-color: slate-dark,
)[
  On this node, for this model and batch size, Flux is a *better* alternative
  than PyTorch. The benchmark does not separate framework effects from
  GPU-count effects. I still wouldn't merge the current version upstream, because
  the conditional graph overhead still has an impact (which should be zero in a static graph).
]

== Next Steps

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  card(title: "Reduce synchronization overhead")[
    - *Gradient bucketing in Flux.* Pack small arrays into large contiguous
      buffers to cut the number of synchronization calls.
    - *Overlapped backward communication.* Start synchronization while the
      backward pass is still running.
    - Both changes reduce synchronization work per update.
  ],
  card(title: "Extend the evidence")[
    - *Cold-cache experiment.* Repeat the first measurement with page-fault
      and disk telemetry to confirm or reject the cache hypothesis.
    - *Deterministic-mode sensitivity.* Repeat the comparison with cuDNN
      algorithm selection restricted to a deterministic set, or with a
      fixed algorithm, to separate framework differences from run-to-run
      variation.
    - *Multi-node scaling.* Move from NVLink to the InfiniBand fabric.
  ],
)
