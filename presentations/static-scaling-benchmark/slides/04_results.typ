// 04_results.typ - Throughput, scaling, and comparative results
#import "@preview/touying:0.6.1": *
#import "../components/cards.typ": *
#import "../components/charts.typ": *

= Empirical Results & Scaling Analysis

== All Measurements at a Glance

#text(size: 0.82em, fill: slate-text)[All times are for one epoch of 390
  updates at global batch 256.]

#v(2pt)

#align(center)[
  #set text(size: 0.72em)
  #table(
    columns: (1.35fr, 0.85fr, 0.85fr, 0.85fr, 1.05fr, 1.05fr, 1.05fr, 1.05fr, 1.05fr),
    align: (left, center, center, center, center, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt + primary-color) } else { 0.5pt + slate-border },
    fill: (x, y) => if y == 0 { rgb("#f1f5f9") } else if calc.even(y) { rgb("#f8fafc") } else { none },
    table.header(
      [*Candidate*], [*1 GPU (s)*], [*2 GPUs (s)*], [*4 GPUs (s)*],
      [*1 GPU img/s*], [*2 GPUs img/s*], [*4 GPUs img/s*],
      [*Speedup (4 GPUs)*], [*Efficiency (4 GPUs)*]
    ),
    [#text(weight: "bold", fill: cand-historical-dark)[Flux (static)]],
    [91.09], [47.46], [25.53],
    [1,096.1], [2,103.8], [#text(weight: "bold", fill: cand-historical-dark)[3,910.8]],
    [#text(weight: "bold", fill: cand-historical-dark)[3.568x]], [#text(weight: "bold", fill: cand-historical-dark)[89.20%]],

    [#text(weight: "bold", fill: cand-current-dark)[Flux (guarded)]],
    [91.10], [52.21], [27.92],
    [1,095.9], [1,912.2], [3,575.5],
    [3.263x], [81.57%],

    [#text(weight: "bold", fill: cand-pytorch-dark)[PyTorch DDP]],
    [95.74], [49.79], [28.59],
    [1,042.9], [2,005.3], [3,492.0],
    [3.348x], [83.71%]
  )
]

#v(4pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card(title: "One GPU")[
    Both Flux builds run at the same speed. Both are slightly faster than
    PyTorch before any communication begins.
  ],
  card(title: "Two GPUs")[
    Flux (static) has the highest throughput. PyTorch and Flux (static) are
    effectively tied on speedup; Flux (guarded) is behind, consistent with
    extra work per update.
  ],
  card(title: "Four GPUs")[
    Flux (static) leads overall. Both Flux builds overtake PyTorch, and the
    guarded build nearly closes its two-GPU gap.
  ],
)

== Absolute Throughput

#align(center)[
  #chart-throughput(show-header: false)
]

#v(2pt)

#text(size: 0.9em)[
  *Takeaway:* Flux is competitive at every GPU count, and the guarded build
  finishes ahead of PyTorch at four GPUs. Flux (static) leads overall.
]

== Strong Scaling Speedup

#align(center)[
  #chart-speedup(show-header: false)
]

#v(2pt)

#text(size: 0.9em)[
  *Takeaway:* Flux (static) has the highest overall speedup. PyTorch is
  marginally ahead at two GPUs, but the gap is 0.2% and disappears at four.
  Speedup is relative to each candidate's own one-GPU time.
]

== Parallel Efficiency

#align(center)[
  #chart-efficiency(show-header: false)
]

#v(2pt)

#text(size: 0.9em)[
  *Takeaway:* Overall efficiency falls as GPUs are added, for every
  candidate. Flux (guarded) has the strongest step from two to four GPUs;
  PyTorch drops the most.
]

== Comparisons by GPU Count

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card(title: "One GPU", badge: badge("reference", fill: rgb("#e2e8f0"), text-color: slate-dark))[
    #align(center)[
      #text(size: 1.15em, weight: "bold", fill: cand-historical-dark)[1,096] #text(size: 0.75em)[(Flux)]
      #h(6pt)
      #text(size: 1.15em, weight: "bold", fill: cand-pytorch-dark)[1,043] #text(size: 0.75em)[(PyTorch)]
    ]
    #v(3pt)
    - The two Flux builds are indistinguishable.
    - Both Flux builds are slightly faster than PyTorch before communication.
  ],
  card(title: "Two GPUs", badge: badge("mixed", fill: cand-current-bg, text-color: cand-current-dark))[
    #align(center)[
      #text(size: 1.12em, weight: "bold", fill: cand-historical-dark)[2,104] #text(size: 0.7em)[(static)] #h(4pt)
      #text(size: 1.12em, weight: "bold", fill: cand-pytorch-dark)[2,005] #text(size: 0.7em)[(PyTorch)] #h(4pt)
      #text(size: 1.12em, weight: "bold", fill: cand-current-dark)[1,912] #text(size: 0.7em)[(guarded)]
    ]
    #v(3pt)
    - Flux (static) leads.
    - PyTorch scales better than Flux (guarded) at this size.
  ],
  card(title: "Four GPUs", badge: badge("peak", fill: cand-historical-bg, text-color: cand-historical-dark))[
    #align(center)[
      #text(size: 1.12em, weight: "bold", fill: cand-historical-dark)[3,911] #text(size: 0.7em)[(static)] #h(4pt)
      #text(size: 1.12em, weight: "bold", fill: cand-current-dark)[3,576] #text(size: 0.7em)[(guarded)] #h(4pt)
      #text(size: 1.12em, weight: "bold", fill: cand-pytorch-dark)[3,492] #text(size: 0.7em)[(PyTorch)]
    ]
    #v(3pt)
    - Both Flux builds beat PyTorch.
    - Flux (static) has the largest margin; Flux (guarded) is just ahead.
  ]
)

#v(4pt)

#callout(title: "The shape of the result")[
  On one GPU the frameworks are close. PyTorch is strongest at two GPUs.
  At four GPUs, both Flux builds overtake it: the guarded build moves from
  behind PyTorch at two GPUs to ahead at four.
]

== Incremental Scaling: Two to Four GPUs

#grid(
  columns: (1.15fr, 0.85fr),
  gutter: 12pt,
  card(title: "The two-to-four-GPU transition")[
    Efficiency at four GPUs mixes two effects: the quality of the step from
    one to two, and the quality of the step from two to four. The second
    transition is the better test of how a candidate behaves once
    distribution is already in place.
    #v(4pt)
    $ E_(2 -> 4) = (T_2 / T_4) / 2 $
    #v(4pt)
    Flux (guarded) has the weakest first transition but the *strongest
    second*: its two-to-four-GPU step is the most efficient, even though it
    remains behind Flux (static) overall.
  ],
  [
    #metric-card("93.5%", "Flux (guarded)", sub: "highest step efficiency", accent: cand-current)
    #v(4pt)
    #metric-card("93.0%", "Flux (static)", sub: "close second", accent: cand-historical)
    #v(4pt)
    #metric-card("87.1%", "PyTorch DDP", sub: "lowest step efficiency", accent: cand-pytorch)
  ]
)

#v(4pt)

#callout(
  title: "Interpretation",
  fill: rgb("#f8fafc"),
  stroke: 1pt + slate-border,
  title-color: slate-dark,
)[
  Flux (static) remains the strongest overall. Flux (guarded) is the most
  efficient when scaling from two to four GPUs, consistent with a fixed
  per-update cost being spread over more hardware. The benchmark is
  descriptive and does not establish that mechanism.
]
