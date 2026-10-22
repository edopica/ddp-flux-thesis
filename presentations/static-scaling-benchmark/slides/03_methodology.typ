// 03_methodology.typ - Measurement protocol (slim, jargon-free)
#import "@preview/touying:0.6.1": *
#import "../components/cards.typ": *

= Measurement Protocol

== How a Run Is Timed

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  card(title: "One run")[
    - *Reset to* the same initial weights and optimizer state.
    - Run 20 untimed warm-up updates. Compilation and kernel selection
      happen here, before the timer.
    - *Time* 390 updates over one full epoch.
    - *Wait for all GPUs to finish* before stopping the timer, so the
      recorded duration covers every process.
  ],
  card(title: "The repeat design")[
    - Every candidate is measured at one, two, and four GPUs.
    - *Five repeats* per configuration.
    - Repeat order *alternates*: one-to-four GPUs, then four-to-one, so slow
      drift is spread across candidates rather than concentrated in one.
    - The reported value is the *median* of the five repeats. A single slow
      start does not dominate the result.
  ],
)

#v(8pt)

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  card(title: "Platform")[
    One Leonardo node with four A100-SXM-64GB GPUs connected by NVLink.
    Flux runs through MPI; PyTorch runs through torchrun. Single-GPU runs use
    the same code path without any distributed communication.
  ],
  card(title: "Memory limit")[
    Before timing, each configuration must stay below 80% of device memory
    during a short trial. Every configuration passed at global batch 256.
    A larger batch exceeded the margin and was not used, so all candidates
    run under the same memory constraint.
  ],
)
