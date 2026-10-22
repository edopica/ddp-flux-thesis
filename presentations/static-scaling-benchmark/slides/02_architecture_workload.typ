// 02_architecture_workload.typ - Model and workload (kept intentionally slim)
#import "@preview/touying:0.6.1": *
#import "../components/cards.typ": *
#import "../components/network.typ": *

= Model & Workload

== The Model at a Glance

A ResNet-18 adapted to 64#sym.times 64 inputs. GroupNorm computes its
statistics within each image, so the local batch size does not change the
normalization.

#v(4pt)

#align(center)[
  #resnet18-full-diagram(scale: 82%)
]

#v(4pt)

#architecture-stats-cards()

== Block Design & Parameter Layout

#grid(
  columns: (1fr, 1.25fr),
  gutter: 12pt,
  card(title: "Residual block")[
    #align(center)[
      #resnet18-block-diagram(scale: 52%, downsample: false)
    ]
    #v(4pt)
    - Two 3#sym.times 3 convolutions with GroupNorm and ReLU, plus a skip
      connection.
    - All convolutions are bias-free; normalization carries the shift.
    - GroupNorm-32 computes statistics within each image, so the output for
      an image does not depend on the other images on that GPU.
  ],
  card(title: "Where the 62 parameter arrays come from")[
    #set text(size: 0.9em)
    #architecture-summary-table()
  ],
)

== Workload, Batching & Optimizer

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  card(title: "Data and batches")[
    - *Dataset:* Tiny ImageNet-200, 100,000 training images.
    - *Effective epoch:* 99,840 images, after dropping a 160-image tail so
      the batch divides exactly at one, two, and four GPUs.
    - *Global batch:* 256 images per optimizer update, always.
    - *Local batch:* 256 on one GPU, 128 on two, 64 on four.
    - *Measured work:* 390 updates per epoch.
  ],
  card(title: "Optimizer")[
    - *AdamW* with $beta_1 = 0.9$, $beta_2 = 0.999$, $epsilon = 10^(-8)$,
      learning rate $10^(-3)$.
    - Weight decay applies to convolution and linear weights; biases and
      normalization parameters are excluded.
    - A linear learning-rate ramp followed by cosine decay. The measured
      epoch sits inside the ramp, so the study compares throughput, not
      converged quality.
  ],
)

#v(6pt)

#callout(title: "Fixed-work rule")[
  Every candidate and every GPU count sees the same sample permutation,
  the same deterministic transforms, the same initial weights, and exactly
  390 gradient updates.
]
