// Publication-Grade CeTZ Charts for Static Scaling Benchmark
// Built with @preview/cetz:0.3.4 and @preview/cetz-plot:0.1.1
//
// Candidate palette is imported from cards.typ (single source of truth):
//   Historical Flux = RED, Current Flux = YELLOW/GOLD, PyTorch = BLUE.

#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot, chart
#import "cards.typ": (
  cand-historical,
  cand-historical-dark,
  cand-historical-bg,
  cand-current,
  cand-current-dark,
  cand-current-bg,
  cand-pytorch,
  cand-pytorch-dark,
  cand-pytorch-bg,
)

// Non-candidate entries are semantic (status/grid/text) and stay neutral.
#let chart-colors = (
  flux-current: cand-current,
  flux-current-dark: cand-current-dark,
  flux-current-light: cand-current-bg,
  flux-historical: cand-historical,
  flux-historical-dark: cand-historical-dark,
  flux-historical-light: cand-historical-bg,
  pytorch: cand-pytorch,
  pytorch-dark: cand-pytorch-dark,
  pytorch-light: cand-pytorch-bg,
  ideal: rgb("#64748b"),       // Cool slate for reference
  ideal-light: rgb("#f1f5f9"), // Subtle slate fill
  grid: rgb("#e2e8f0"),        // Subtle grid lines
  axis: rgb("#94a3b8"),        // Crisp axis stroke
  text: rgb("#1e293b"),        // Dark slate text
  muted: rgb("#64748b"),       // Muted secondary text
  card-bg: rgb("#ffffff"),     // White surface
  border: rgb("#e2e8f0"),      // Border stroke
  success: rgb("#15803d"),     // Forest green
  success-bg: rgb("#dcfce7"),  // Green tint
  danger: rgb("#b91c1c"),      // Alert red
  danger-bg: rgb("#fee2e2"),   // Red tint
)

// Card geometry: 25.4 cm leaves visible margins on a ~28 cm slide body.
#let card-width = 25.4cm
// Base canvas text size for axis ticks; labels use explicit smaller sizes.
#let canvas-text = 10.5pt
#let chart-title-size = 13.5pt
#let chart-subtitle-size = 9.5pt
#let legend-text-size = 10pt
#let value-label-size = 8.5pt
#let callout-size = 8pt

// Reusable legend pill
#let chart-legend-item(color, label, size: legend-text-size) = {
  box(baseline: 20%, inset: (right: 13pt), [
    #box(width: 9pt, height: 9pt, radius: 2pt, fill: color)
    #std.h(5pt)
    #text(size: size, weight: "medium", fill: chart-colors.text, label)
  ])
}

// Shared card header: title + subtitle on the left, legend on the right.
#let chart-header(
  c,
  title,
  subtitle,
  show-header,
  legend-items,
) = {
  if show-header [
    #grid(
      columns: (1fr, auto),
      align: (left + horizon, right + horizon),
      column-gutter: 10pt,
      [
        #text(size: chart-title-size, weight: "bold", fill: c.text, title) \
        #std.v(-2pt)
        #text(size: chart-subtitle-size, fill: c.muted, subtitle)
      ],
      legend-items,
    )
    #std.v(4pt)
  ] else [
    #align(right)[#legend-items]
    #std.v(2pt)
  ]
}

// ---------------------------------------------------------------------------
// 1. Throughput Chart
// ---------------------------------------------------------------------------
/// Clustered column chart showing absolute throughput across W1, W2, W4
/// Data from Table 4.1 in report:
/// - W1: Flux Current 1095.9 | Flux Historical 1096.1 | PyTorch 1042.9
/// - W2: Flux Current 1912.2 | Flux Historical 2103.8 | PyTorch 2005.3
/// - W4: Flux Current 3575.5 | Flux Historical 3910.8 | PyTorch 3492.0
#let chart-throughput(
  size: (23.4, 8.4),
  title: [Absolute Throughput by GPU Count],
  subtitle: [Effective images processed per second (higher is better) • Leonardo A100-SXM-64GB],
  show-header: true,
  show-annotations: true,
) = {
  let c = chart-colors
  let bar-palette(i) = {
    let cols = (c.flux-historical, c.flux-current, c.pytorch)
    let col = cols.at(calc.rem(i, cols.len()))
    (stroke: 0.5pt + col.darken(15%), fill: col)
  }

  box(
    width: card-width,
    stroke: 0.5pt + c.border,
    radius: 6pt,
    fill: c.card-bg,
    inset: (x: 12pt, top: 10pt, bottom: 8pt),
  )[
    #chart-header(
      c,
      title,
      subtitle,
      show-header,
      [
        #chart-legend-item(c.flux-historical, [Flux (static)])
        #chart-legend-item(c.flux-current, [Flux (guarded)])
        #chart-legend-item(c.pytorch, [PyTorch])
      ],
    )

    #set text(size: canvas-text)
    #align(center)[
      #cetz.canvas({
        plot.plot(
          size: size,
          x-min: -0.65,
          x-max: 2.65,
          y-min: 0,
          y-max: 4700,
          x-ticks: ((0, [1 GPU]), (1, [2 GPUs]), (2, [4 GPUs])),
          x-tick-step: none,
          x-label: [],
          y-label: [Images per second],
          y-ticks: ((0, [0]), (1000, [1k]), (2000, [2k]), (3000, [3k]), (4000, [4k])),
          y-tick-step: none,
          y-grid: true,
          plot-style: bar-palette,
          axis-style: "scientific-auto",
          {
            plot.add-bar(
              (
                (0, (1096.1, 1095.9, 1042.9)),
                (1, (2103.8, 1912.2, 2005.3)),
                (2, (3910.8, 3575.5, 3492.0)),
              ),
              bar-width: 0.62,
              mode: "clustered",
            )
          }
        )

        // Annotations are drawn in canvas coordinates with an explicit
        // data -> cm mapping. Using `plot.annotate` here inflates the canvas
        // bounds by the label width (CeTZ fixed-size annotation quirk) and
        // shifts the whole plot, so we keep annotations outside the plot.
        if show-annotations {
          let x-min = -0.65
          let x-max = 2.65
          let y-max = 4700
          let (w, h) = size
          let sx(x) = (x - x-min) / (x-max - x-min) * w
          let sy(y) = y / y-max * h
          let offset = 0.207
          let y-pad = 175

          // W1 Values
          cetz.draw.content((sx(0 - offset), sy(1096.1 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-historical-dark, [1,096]))
          cetz.draw.content((sx(0),          sy(1095.9 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-current-dark, [1,096]))
          cetz.draw.content((sx(0 + offset), sy(1042.9 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.pytorch-dark, [1,043]))

          // W2 Values
          cetz.draw.content((sx(1 - offset), sy(2103.8 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-historical-dark, [2,104]))
          cetz.draw.content((sx(1),          sy(1912.2 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-current-dark, [1,912]))
          cetz.draw.content((sx(1 + offset), sy(2005.3 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.pytorch-dark, [2,005]))

          // W4 Values
          cetz.draw.content((sx(2 - offset), sy(3910.8 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-historical-dark, [3,911]))
          cetz.draw.content((sx(2),          sy(3575.5 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-current-dark, [3,576]))
          cetz.draw.content((sx(2 + offset), sy(3492.0 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.pytorch-dark, [3,492]))

          // Highlight Callout Pill above the W4 winner
          cetz.draw.content((sx(2), sy(4455)), [
            #box(fill: c.flux-historical-light, stroke: 0.5pt + c.flux-historical, radius: 3pt, inset: (x: 5pt, y: 2pt))[
              #text(size: callout-size, weight: "bold", fill: c.flux-historical-dark)[Flux (static): +12.0% vs PyTorch, +9.4% vs guarded]
            ]
          ])

          // W1 Descriptive tie note
          cetz.draw.content((sx(0), sy(1650)), [
            #box(fill: rgb("#f8fafc"), stroke: 0.5pt + c.border, radius: 2.5pt, inset: (x: 4pt, y: 2pt))[
              #text(size: callout-size, fill: c.muted)[static and guarded tied (+5.1% vs PyTorch)]
            ]
          ])
        }
      })
    ]
  ]
}

// ---------------------------------------------------------------------------
// 2. Speedup Chart
// ---------------------------------------------------------------------------
/// Line and point plot comparing strong scaling speedup ($S_N = T_1 / T_N$)
/// Data from Section 4.1 in report:
/// - W1: 1.0x (all)
/// - W2: Current 1.745x | Historical 1.919x | PyTorch 1.923x | Ideal 2.0x
/// - W4: Current 3.263x | Historical 3.568x | PyTorch 3.348x | Ideal 4.0x
#let chart-speedup(
  size: (23.2, 8.2),
  title: [Strong Scaling Speedup ($S_N = T_1 / T_N$)],
  subtitle: [Speedup relative to each candidate's own one-GPU time (ideal = linear)],
  show-header: true,
  show-annotations: true,
) = {
  let c = chart-colors

  box(
    width: card-width,
    stroke: 0.5pt + c.border,
    radius: 6pt,
    fill: c.card-bg,
    inset: (x: 12pt, top: 10pt, bottom: 8pt),
  )[
    #chart-header(
      c,
      title,
      subtitle,
      show-header,
      [
        #chart-legend-item(c.flux-historical, [Flux (static)])
        #chart-legend-item(c.flux-current, [Flux (guarded)])
        #chart-legend-item(c.pytorch, [PyTorch])
        #chart-legend-item(c.ideal, [Ideal])
      ],
    )

    #set text(size: canvas-text)
    #align(center)[
      #cetz.canvas({
        plot.plot(
          size: size,
          x-min: 0.7,
          x-max: 4.85,
          y-min: 0.8,
          y-max: 4.3,
          x-ticks: ((1, [1 GPU]), (2, [2 GPUs]), (4, [4 GPUs])),
          x-tick-step: none,
          y-ticks: ((1.0, [1.0×]), (2.0, [2.0×]), (3.0, [3.0×]), (4.0, [4.0×])),
          y-tick-step: none,
          y-grid: true,
          x-grid: true,
          x-label: [Number of GPUs],
          y-label: [Speedup],
          axis-style: "scientific-auto",
          {
            // 1. Ideal Scaling Reference Line
            plot.add(
              ((1.0, 1.0), (2.0, 2.0), (4.0, 4.0)),
              style: (stroke: (paint: c.ideal, dash: "dashed", thickness: 1.2pt)),
              mark: none,
            )

            // 2. Flux Historical (Static DDP)
            plot.add(
              ((1.0, 1.0), (2.0, 1.919), (4.0, 3.568)),
              style: (stroke: 2pt + c.flux-historical),
              mark: "square",
              mark-style: (fill: c.flux-historical, stroke: 1.5pt + white),
              mark-size: 0.15,
            )

            // 3. PyTorch (Native DDP)
            plot.add(
              ((1.0, 1.0), (2.0, 1.923), (4.0, 3.348)),
              style: (stroke: 2pt + c.pytorch),
              mark: "triangle",
              mark-style: (fill: c.pytorch, stroke: 1.5pt + white),
              mark-size: 0.15,
            )

            // 4. Flux Current (Guarded DDP)
            plot.add(
              ((1.0, 1.0), (2.0, 1.745), (4.0, 3.263)),
              style: (stroke: 2pt + c.flux-current),
              mark: "o",
              mark-style: (fill: c.flux-current, stroke: 1.5pt + white),
              mark-size: 0.15,
            )
          }
        )

        if show-annotations {
          let (w, h) = size
          let sx(x) = (x - 0.7) / (4.85 - 0.7) * w
          let sy(y) = (y - 0.8) / (4.3 - 0.8) * h

          // End-Point Value Badges at W4
          cetz.draw.content((sx(4.08), sy(4.02)), anchor: "west", text(size: value-label-size, weight: "medium", fill: c.ideal, [4.00× ideal]))
          cetz.draw.content((sx(4.08), sy(3.60)), anchor: "west", text(size: value-label-size, weight: "bold", fill: c.flux-historical-dark, [3.57× static (89.2%)]))
          cetz.draw.content((sx(4.08), sy(3.34)), anchor: "west", text(size: value-label-size, weight: "bold", fill: c.pytorch-dark, [3.35× PyTorch (83.7%)]))
          cetz.draw.content((sx(4.08), sy(3.10)), anchor: "west", text(size: value-label-size, weight: "bold", fill: c.flux-current-dark, [3.26× guarded (81.6%)]))

          // Two-GPU callout (below the data lines, above the x axis)
          cetz.draw.content((sx(2.0), sy(1.20)), [
            #box(fill: rgb("#f8fafc"), stroke: 0.5pt + c.border, radius: 2.5pt, inset: (x: 4pt, y: 2pt))[
              #text(size: callout-size, fill: c.text)[
                *2 GPUs*: static / PyTorch $approx 1.92times$ (96%)\
                guarded $1.74times$ (87%)
              ]
            ]
          ])
        }
      })
    ]
  ]
}

// ---------------------------------------------------------------------------
// 3. Parallel Efficiency Chart
// ---------------------------------------------------------------------------
/// Clustered column chart showing Parallel Efficiency at W2 and W4,
/// plus the critical W2 -> W4 Incremental Efficiency finding.
/// Data from Section 4.1 & 4.3 in report:
/// - W2 Overall: Current 87.25% | Historical 95.97% | PyTorch 96.14%
/// - W4 Overall: Current 81.57% | Historical 89.20% | PyTorch 83.71%
/// - W2->W4 Step: Current 93.49% | Historical 92.95% | PyTorch 87.07%
#let chart-efficiency(
  size: (23.2, 8.2),
  title: [Parallel Scaling Efficiency ($E_N = S_N / N$)],
  subtitle: [Overall efficiency at 2 and 4 GPUs, plus the two-to-four-GPU transition],
  show-header: true,
  show-annotations: true,
) = {
  let c = chart-colors
  let eff-palette(i) = {
    let cols = (c.flux-historical, c.flux-current, c.pytorch)
    let col = cols.at(calc.rem(i, cols.len()))
    (stroke: 0.5pt + col.darken(15%), fill: col)
  }

  box(
    width: card-width,
    stroke: 0.5pt + c.border,
    radius: 6pt,
    fill: c.card-bg,
    inset: (x: 12pt, top: 10pt, bottom: 8pt),
  )[
    #chart-header(
      c,
      title,
      subtitle,
      show-header,
      [
        #chart-legend-item(c.flux-historical, [Flux (static)])
        #chart-legend-item(c.flux-current, [Flux (guarded)])
        #chart-legend-item(c.pytorch, [PyTorch])
        #chart-legend-item(c.ideal, [Ideal])
      ],
    )

    #set text(size: canvas-text)
    #align(center)[
      #cetz.canvas({
        plot.plot(
          size: size,
          x-min: -0.65,
          x-max: 2.65,
          y-min: 70,
          y-max: 106,
          x-ticks: ((0, [2 GPUs]), (1, [4 GPUs]), (2, [2$arrow.r$4 GPUs])),
          x-tick-step: none,
          x-label: [],
          y-label: [Parallel efficiency (%)],
          y-ticks: ((70, [70%]), (80, [80%]), (90, [90%]), (100, [100%])),
          y-tick-step: none,
          y-grid: true,
          plot-style: eff-palette,
          axis-style: "scientific-auto",
          {
            plot.add-bar(
              (
                (0, (95.97, 87.25, 96.14)),
                (1, (89.20, 81.57, 83.71)),
                (2, (92.95, 93.49, 87.07)),
              ),
              bar-width: 0.62,
              mode: "clustered",
            )

            // 100% Ideal Reference Line
            plot.add-hline(100, style: (stroke: (paint: c.ideal, dash: "dashed", thickness: 1pt)))
          }
        )

        if show-annotations {
          let (w, h) = size
          let sx(x) = (x + 0.65) / (2.65 + 0.65) * w
          let sy(y) = (y - 70) / (106 - 70) * h
          let offset = 0.207
          let y-pad = 2.2

          // Group 0: 2 GPUs
          cetz.draw.content((sx(0 - offset), sy(95.97 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-historical-dark, [96.0%]))
          cetz.draw.content((sx(0),          sy(87.25 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-current-dark, [87.3%]))
          cetz.draw.content((sx(0 + offset), sy(96.14 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.pytorch-dark, [96.1%]))

          // Group 1: 4 GPUs
          cetz.draw.content((sx(1 - offset), sy(89.20 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-historical-dark, [89.2%]))
          cetz.draw.content((sx(1),          sy(81.57 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-current-dark, [81.6%]))
          cetz.draw.content((sx(1 + offset), sy(83.71 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.pytorch-dark, [83.7%]))

          // Group 2: two-to-four-GPU transition
          cetz.draw.content((sx(2 - offset), sy(92.95 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-historical-dark, [93.0%]))
          cetz.draw.content((sx(2),          sy(93.49 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.flux-current-dark, [93.5%]))
          cetz.draw.content((sx(2 + offset), sy(87.07 + y-pad)), text(size: value-label-size, weight: "bold", fill: c.pytorch-dark, [87.1%]))

          // 100% Ideal Label
          cetz.draw.content((sx(2.6), sy(99.4)), anchor: "north-east", text(size: value-label-size, weight: "medium", fill: c.ideal, [100% linear]))

          // Callout Badge on Incremental Step
          cetz.draw.content((sx(2), sy(103.9)), [
            #box(fill: c.flux-current-light, stroke: 0.5pt + c.flux-current, radius: 2.5pt, inset: (x: 4pt, y: 1.5pt))[
              #text(size: callout-size, weight: "bold", fill: c.flux-current-dark)[Guarded: best\ two-to-four step]
            ]
          ])
        }
      })
    ]
  ]
}

// ---------------------------------------------------------------------------
// 4. Repeats & Dispersion Chart
// ---------------------------------------------------------------------------
/// Coordinated dual-panel visualization contrasting the Repeat-1 outlier
/// with the ultra-stable steady-state repeats r2..r5.
/// Data from Section 4.2 in report:
/// - Current W1: r1=680.79s (7.47x outlier), r2=91.04s, r3=91.07s, r4=91.10s, r5=91.11s. Median=91.10s.
/// - Historical W1: 91.05s - 91.10s (stdev 0.02s). Median=91.09s.
/// - PyTorch W1: 95.72s - 95.74s (stdev 0.006s). Median=95.74s.
#let chart-dispersion(
  size: (23.6, 8.0),
  title: [Repeat Variance and the First-Repeat Outlier (Single GPU)],
  subtitle: [Full scale vs. zoomed steady state • effect of using the median],
  show-header: true,
) = {
  let c = chart-colors
  let gap = 2.0
  let panel-w = (size.at(0) - gap) / 2
  let panel-h = size.at(1) - 1.25

  box(
    width: card-width,
    stroke: 0.5pt + c.border,
    radius: 6pt,
    fill: c.card-bg,
    inset: (x: 12pt, top: 10pt, bottom: 8pt),
  )[
    #chart-header(
      c,
      title,
      subtitle,
      show-header,
      [
        #chart-legend-item(c.flux-historical, [Flux (static)])
        #chart-legend-item(c.flux-current, [Flux (guarded)])
        #chart-legend-item(c.pytorch, [PyTorch])
      ],
    )

    #set text(size: canvas-text)
    #align(center)[
      #cetz.canvas({
        // Panel 1: Full Scale (0 to 760s)
        cetz.draw.group({
          plot.plot(
            size: (panel-w, panel-h),
            x-min: 0.5,
            x-max: 5.5,
            y-min: 0,
            y-max: 760,
            x-ticks: ((1, [r1]), (2, [r2]), (3, [r3]), (4, [r4]), (5, [r5])),
            x-tick-step: none,
            y-ticks: ((0, [0s]), (200, [200s]), (400, [400s]), (600, [600s])),
            y-tick-step: none,
            y-grid: true,
            x-label: [Repeats (1–5)],
            y-label: [Epoch duration (s)],
            axis-style: "scientific-auto",
            {
              // PyTorch Reference
              plot.add(
                ((1, 95.73), (2, 95.74), (3, 95.73), (4, 95.72), (5, 95.74)),
                style: (stroke: 1.4pt + c.pytorch),
                mark: "triangle",
                mark-size: 0.1,
                mark-style: (fill: c.pytorch),
              )
              // Historical Reference
              plot.add(
                ((1, 91.09), (2, 91.10), (3, 91.05), (4, 91.08), (5, 91.09)),
                style: (stroke: 2.4pt + c.flux-historical),
                mark: "square",
                mark-size: 0.12,
                mark-style: (fill: c.flux-historical),
              )
              // Current W1 (Contains Outlier)
              plot.add(
                ((1, 680.79), (2, 91.04), (3, 91.07), (4, 91.10), (5, 91.11)),
                style: (stroke: 1.6pt + c.flux-current),
                mark: "o",
                mark-size: 0.13,
                mark-style: (fill: c.flux-current),
              )
            }
          )
          // Annotations in canvas coordinates (see throughput note).
          let sx(x) = (x - 0.5) / (5.5 - 0.5) * panel-w
          let sy(y) = y / 760 * panel-h
          cetz.draw.content((sx(1.32), sy(680.79)), anchor: "west", [
            #box(fill: c.flux-current-light, stroke: 0.5pt + c.flux-current, radius: 2.5pt, inset: 3pt)[
              #text(size: callout-size, weight: "bold", fill: c.flux-current-dark)[r1: 680.8 s\ (7.47× slower)]
            ]
          ])
          cetz.draw.content((sx(3.5), sy(26)), anchor: "south", text(size: value-label-size, fill: c.muted, [r2–r5 cluster tightly at ~91 s]))
          cetz.draw.content((panel-w / 2, panel-h + 0.42), anchor: "center", text(size: 11pt, weight: "bold", fill: c.text, [Full scale: repeat 1]))
        })

        // Panel 2: Zoomed Steady-State (r2 to r5)
        cetz.draw.set-origin((panel-w + gap, 0))
        cetz.draw.group({
          plot.plot(
            size: (panel-w, panel-h),
            x-min: 1.5,
            x-max: 5.5,
            y-min: 88,
            y-max: 98,
            x-ticks: ((2, [r2]), (3, [r3]), (4, [r4]), (5, [r5])),
            x-tick-step: none,
            y-ticks: ((90, [90s]), (92, [92s]), (94, [94s]), (96, [96s]), (98, [98s])),
            y-tick-step: none,
            y-grid: true,
            x-label: [Repeats (2–5)],
            y-label: [Epoch duration (s)],
            axis-style: "scientific-auto",
            {
              // PyTorch
              plot.add(
                ((2, 95.74), (3, 95.73), (4, 95.72), (5, 95.74)),
                style: (stroke: 1.4pt + c.pytorch),
                mark: "triangle",
                mark-size: 0.1,
                mark-style: (fill: c.pytorch),
              )
              // Historical
              plot.add(
                ((2, 91.10), (3, 91.05), (4, 91.08), (5, 91.09)),
                style: (stroke: 2.4pt + c.flux-historical),
                mark: "square",
                mark-size: 0.12,
                mark-style: (fill: c.flux-historical),
              )
              // Current
              plot.add(
                ((2, 91.04), (3, 91.07), (4, 91.10), (5, 91.11)),
                style: (stroke: 1.6pt + c.flux-current),
                mark: "o",
                mark-size: 0.13,
                mark-style: (fill: c.flux-current),
              )
            }
          )
          let sx(x) = (x - 1.5) / (5.5 - 1.5) * panel-w
          let sy(y) = (y - 88) / (98 - 88) * panel-h
          // Steady-State Success Box
          cetz.draw.content((sx(3.55), sy(93.35)), anchor: "center", [
            #box(fill: c.success-bg, stroke: 0.5pt + c.success, radius: 2.5pt, inset: 3pt)[
              #text(size: callout-size, weight: "bold", fill: c.success)[Ultra-stable: $sigma < 0.04$s\ Median: 91.10s (Δ = 0.017%)]
            ]
          ])
          // PyTorch label
          cetz.draw.content((sx(2.05), sy(97.05)), anchor: "west", text(size: value-label-size, fill: c.pytorch-dark, [PyTorch 95.7 s]))
          // Flux labels (static and guarded overlap at this scale)
          cetz.draw.content((sx(2.05), sy(90.05)), anchor: "west", text(size: value-label-size, fill: c.flux-current-dark, [guarded 91.1 s]))
          cetz.draw.content((sx(2.05), sy(89.25)), anchor: "west", text(size: value-label-size, fill: c.flux-historical-dark, [static 91.1 s]))
          cetz.draw.content((panel-w / 2, panel-h + 0.42), anchor: "center", text(size: 11pt, weight: "bold", fill: c.text, [Steady state: repeats 2–5]))
        })
      })
    ]

    #std.v(3pt)
    #align(center)[
      #text(size: 8.5pt, fill: c.muted)[
        *Key conclusion*: the cause of the first-repeat slowdown is unconfirmed.
        Dropping repeat 1 changes the reported median by only 0.017%.
      ]
    ]
  ]
}

// ---------------------------------------------------------------------------
// 5. Diagnostics Chart
// ---------------------------------------------------------------------------
/// Category-level relative L2 difference ranges across all 15 comparisons
/// Data from Section 3.5 in report:
/// - param: 1.8877% - 2.0819% (62 arrays, 11,271,432 parameters)
/// - v (second moment): 5.8652% - 8.7686%
/// - m (first moment): 18.1823% - 30.3619%
/// - beta_power: 0.0% (exact match across all 62 leaves in all Flux comparisons)
#let chart-diagnostics(
  size: (23.2, 7.4),
  title: [Numerical Differences After 390 Updates],
  subtitle: [Relative $L_2$ difference ranges across 15 comparisons],
  show-header: true,
) = {
  let c = chart-colors
  // Neutral categorical hues: deliberately outside the red/yellow/blue candidate set.
  let c-param = rgb("#047857")
  let c-v = rgb("#6d28d9")
  let c-m = rgb("#475569")

  let W = size.at(0)
  let x0 = 5.1
  let x1 = W - 0.2
  let plot-w = x1 - x0
  let x-scale(val) = x0 + (val / 35.0) * plot-w
  let axis-y = 0.55

  box(
    width: card-width,
    stroke: 0.5pt + c.border,
    radius: 6pt,
    fill: c.card-bg,
    inset: (x: 12pt, top: 10pt, bottom: 8pt),
  )[
    #if show-header [
      #align(left)[
        #text(size: chart-title-size, weight: "bold", fill: c.text, title) \
        #std.v(-2pt)
        #text(size: chart-subtitle-size, fill: c.muted, subtitle)
      ]
      #std.v(5pt)
    ]

    #set text(size: canvas-text)
    #align(center)[
      #cetz.canvas({
        let row-param = 1.25
        let row-v = 3.55
        let row-m = 5.85

        // Subtle vertical gridlines + x tick labels
        for x-val in (0, 5, 10, 15, 20, 25, 30, 35) {
          let x-pos = x-scale(x-val)
          cetz.draw.line((x-pos, axis-y), (x-pos, 6.55), stroke: (paint: c.grid, dash: "dotted", thickness: 0.6pt))
          cetz.draw.content((x-pos, axis-y - 0.16), anchor: "north", text(size: 9.5pt, fill: c.muted, str(x-val) + "%"))
        }

        // Baseline X axis
        cetz.draw.line((x0, axis-y), (x1, axis-y), stroke: 1pt + c.axis)

        // Category 1: weights (bottom)
        cetz.draw.content((x0 - 0.3, row-param), anchor: "east", block(width: 4.5cm, align(right)[
          #text(size: 10pt, weight: "bold", fill: c.text)[Weights] \
          #text(size: 8.5pt, fill: c.muted)[62 arrays • 11.27M values] \
          #text(size: 8.5pt, fill: c.muted)[stays within a narrow band]
        ]))
        let p-min = x-scale(1.8877)
        let p-max = x-scale(2.0819)
        let p-draw-max = calc.max(p-max, p-min + 0.22)
        cetz.draw.rect((p-min, row-param - 0.27), (p-draw-max, row-param + 0.27), fill: c-param, stroke: none, radius: 2pt)
        cetz.draw.content((p-draw-max + 0.22, row-param), anchor: "west", text(size: 10pt, weight: "bold", fill: c-param)[1.89% – 2.08%])

        // Category 2: variance (middle)
        cetz.draw.content((x0 - 0.3, row-v), anchor: "east", block(width: 4.5cm, align(right)[
          #text(size: 10pt, weight: "bold", fill: c.text)[Variance] \
          #text(size: 8.5pt, fill: c.muted)[AdamW second moment ($v_t$)] \
          #text(size: 8.5pt, fill: c.muted)[squared-gradient state]
        ]))
        let v-min = x-scale(5.8652)
        let v-max = x-scale(8.7686)
        cetz.draw.rect((v-min, row-v - 0.27), (v-max, row-v + 0.27), fill: c-v, stroke: none, radius: 2pt)
        cetz.draw.content((v-max + 0.22, row-v), anchor: "west", text(size: 10pt, weight: "bold", fill: c-v)[5.87% – 8.77%])

        // Category 3: momentum (top)
        cetz.draw.content((x0 - 0.3, row-m), anchor: "east", block(width: 4.5cm, align(right)[
          #text(size: 10pt, weight: "bold", fill: c.text)[Momentum] \
          #text(size: 8.5pt, fill: c.muted)[AdamW first moment ($m_t$)] \
          #text(size: 8.5pt, fill: c.muted)[accumulated gradient state]
        ]))
        let m-min = x-scale(18.1823)
        let m-max = x-scale(30.3619)
        cetz.draw.rect((m-min, row-m - 0.27), (m-max, row-m + 0.27), fill: c-m, stroke: none, radius: 2pt)
        cetz.draw.content((m-max + 0.22, row-m), anchor: "west", text(size: 10pt, weight: "bold", fill: c-m)[18.18% – 30.36%])

        // X axis title
        cetz.draw.content(((x0 + x1) / 2, axis-y - 0.85), anchor: "north", text(size: 9.5pt, weight: "medium", fill: c.muted)[
          Relative $L_2$ difference after 390 updates
        ])
      })
    ]

    #std.v(3pt)
    #align(left)[
      #text(size: 8.5pt, fill: c.muted)[
        *Key observation*: cross-framework weight differences ($approx 2.0\%$) are comparable to differences between GPU counts within one framework ($approx 2.0\%$). Schedule counters match exactly ($0.0\%$), consistent with identical update counts.
      ]
    ]
  ]
}
