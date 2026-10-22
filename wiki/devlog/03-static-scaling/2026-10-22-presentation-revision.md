# Supervisor presentation revision — 2026-10-22

Status: deck rebuilt for supervisor review. No benchmark data or claims changed.

The supervisor reviewed the first static-scaling deck and asked for a clearer
story, fewer internal process labels, one candidate palette, and larger charts.
The source in `presentations/static-scaling-benchmark/` now reflects that
review.

## Content changes

- Added a candidate slide that explains the difference between the two Flux
  builds: static (no runtime checks) and guarded (presence and finite-value
  checks), plus PyTorch's bucketed synchronization.
- Removed all performance statistics from the executive summary. It now states
  four findings in prose.
- Replaced the experimental-arms paragraph on the study-design slide with a
  batch-partitioning diagram and the throughput/speedup/efficiency definitions.
- Removed the network-specification slide and the upstream correctness-harness
  slide. The model, block layout, and workload slides remain.
- Removed the campaign-structure/evidence-integrity slide and the standalone
  memory-preflight slide. One timing-protocol slide now covers reset, 20 untimed
  updates, 390 timed updates, five repeats, alternating order, median, and the
  80% memory limit.
- Removed phase numbers, commit hashes, job IDs, node names, OD/ADR codes,
  Slurm/SHA evidence language, and W1/W2/W4 shorthand from the visible deck.
- Softened causal and absolute wording where the benchmark does not isolate the
  mechanism.

## Visual changes

- One palette is now used everywhere:
  historical/static Flux red `#dc2626`, current/guarded Flux yellow `#ca8a04`,
  PyTorch blue `#2563eb`. The theme header is neutral slate.
- Charts use a 25.4 cm card, roughly 23–24 cm plot widths, 10.5 pt axis text,
  and 8–10 pt annotations. Legend order matches bar/line order.
- Chart slides now use a short takeaway line instead of a second callout, so the
  chart itself fills the slide.

## Build

The deck is 27 pages, down from 30. `make pdf` and `make png` regenerate
`presentation.pdf` and `dist/slides/`. The `png` target now clears stale slide
images before rendering.

```bash
cd presentations/static-scaling-benchmark
make pdf
make png
pdftotext presentation.pdf - | rg 'Phase|commit|Job|W1|W2|W4|Slurm|SHA|lrdn'
```

The final `rg` check returns no matches.

## Next action

Send `presentation.pdf` to the supervisor for review. If the narrative is
approved, present from that PDF and keep the source as the editable master.
