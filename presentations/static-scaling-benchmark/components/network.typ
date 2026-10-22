// network.typ - Neural Network Visualizations for Static Scaling Benchmark
// Tiny ImageNet ResNet-18 (Small-image stem, GroupNorm-32)
// Reference: experiments/static-scaling/specification.md:110-152

#import "@preview/neural-netz:0.1.0": draw-network
#import "@preview/cetz:0.4.2": canvas, draw

// Theme Colors (harmonized with theme-setup.typ and cards.typ)
#let brand-primary = rgb("#006494")   // Blue Accent
#let brand-dark    = rgb("#13293d")   // Dark Navy
#let brand-teal    = rgb("#247ba0")   // Teal / Secondary
#let brand-light   = rgb("#e8f1f5")   // Soft Light Blue
#let brand-accent  = rgb("#e76f51")   // Coral / ReLU / Accent
#let brand-amber   = rgb("#f4a261")   // Amber / GN / Attention
#let brand-green   = rgb("#2a9d8f")   // Emerald Green / Sum / Residual
#let text-main     = rgb("#1e293b")   // Body text
#let text-muted    = rgb("#64748b")   // Subtext / captions
#let border-color  = rgb("#e2e8f0")   // Subtle borders

/// 3D Isometric Overview Diagram of the Tiny ImageNet ResNet-18 Architecture
/// Uses `@preview/neural-netz:0.1.0` to render input, stem, 4 stages, GAP, and classifier.
///
/// Parameters:
/// - `scale`: Diagram scale percentage (default 60%, ideal for 16:9 slides)
/// - `palette`: Color palette ("warm" or "cold", default "warm")
/// - `detailed-blocks`: If true (default), renders all 8 individual BasicBlocks across the 4 stages.
///   If false, renders 4 composite stage blocks.
#let resnet18-full-diagram(
  scale: 60%,
  palette: "warm",
  detailed-blocks: true,
) = {
  let layers = if detailed-blocks {
    (
      // Input RGB image (64x64, 3 channels)
      (
        type: "input",
        image-file: "default",
        height: 5.5,
        depth: 5.5,
        label: "Input",
        channels: (3, 64),
      ),
      
      // Stem: 3x3 Conv, 64ch, s=1, p=1, no bias -> GN(32) -> ReLU (No max-pooling!)
      (
        type: "conv",
        widths: (0.4,),
        height: 5.5,
        depth: 5.5,
        label: "Stem",
        xlabel: "3×3, s=1",
        channels: (64, 64),
        offset: 1.6,
        name: "stem",
      ),
      
      // Stage 1: 2 BasicBlocks, 64ch, 64x64, stride 1, no projection
      (
        type: "convres",
        widths: (0.35, 0.35),
        height: 5.5,
        depth: 5.5,
        label: "Stage 1.1",
        xlabel: "s=1",
        channels: (64, 64, 64),
        offset: 2.0,
        name: "b1_1",
      ),
      (
        type: "convres",
        widths: (0.35, 0.35),
        height: 5.5,
        depth: 5.5,
        label: "Stage 1.2",
        xlabel: "s=1",
        channels: (64, 64, 64),
        offset: 1.0,
        name: "b1_2",
      ),
      
      // Stage 2: 2 BasicBlocks, 128ch, 32x32 (Block 1 stride 2 with 1x1 projection)
      (
        type: "convres",
        widths: (0.45, 0.45),
        height: 4.2,
        depth: 4.2,
        label: "Stage 2.1",
        xlabel: "s=2 (proj)",
        channels: (128, 128, 32),
        offset: 2.2,
        name: "b2_1",
      ),
      (
        type: "convres",
        widths: (0.45, 0.45),
        height: 4.2,
        depth: 4.2,
        label: "Stage 2.2",
        xlabel: "s=1",
        channels: (128, 128, 32),
        offset: 1.0,
        name: "b2_2",
      ),
      
      // Stage 3: 2 BasicBlocks, 256ch, 16x16 (Block 1 stride 2 with 1x1 projection)
      (
        type: "convres",
        widths: (0.55, 0.55),
        height: 3.0,
        depth: 3.0,
        label: "Stage 3.1",
        xlabel: "s=2 (proj)",
        channels: (256, 256, 16),
        offset: 2.2,
        name: "b3_1",
      ),
      (
        type: "convres",
        widths: (0.55, 0.55),
        height: 3.0,
        depth: 3.0,
        label: "Stage 3.2",
        xlabel: "s=1",
        channels: (256, 256, 16),
        offset: 1.0,
        name: "b3_2",
      ),
      
      // Stage 4: 2 BasicBlocks, 512ch, 8x8 (Block 1 stride 2 with 1x1 projection)
      (
        type: "convres",
        widths: (0.65, 0.65),
        height: 2.0,
        depth: 2.0,
        label: "Stage 4.1",
        xlabel: "s=2 (proj)",
        channels: (512, 512, 8),
        offset: 2.2,
        name: "b4_1",
      ),
      (
        type: "convres",
        widths: (0.65, 0.65),
        height: 2.0,
        depth: 2.0,
        label: "Stage 4.2",
        xlabel: "s=1",
        channels: (512, 512, 8),
        offset: 1.0,
        name: "b4_2",
      ),
      
      // Global Average Pooling: 8x8x512 -> 512
      (
        type: "gap",
        height: 1.4,
        depth: 1.4,
        label: "GAP",
        xlabel: "1×1",
        channels: (512,),
        offset: 1.8,
        name: "gap",
      ),
      
      // Classifier: Linear 512 -> 200 logits
      (
        type: "fc",
        height: 3.2,
        depth: 0.25,
        label: "FC-200",
        xlabel: "Logits",
        channels: (200,),
        offset: 1.5,
        name: "fc",
      ),
    )
  } else {
    // Macro Stage View (4 stages)
    (
      (type: "input", image-file: "default", height: 5.5, depth: 5.5, label: "Input", channels: (3, 64)),
      (type: "conv", widths: (0.4,), height: 5.5, depth: 5.5, label: "Stem", xlabel: "Conv 3×3", channels: (64, 64), offset: 1.6),
      (type: "convres", widths: (0.45, 0.45), height: 5.5, depth: 5.5, label: "Stage 1 (×2)", xlabel: "64 ch, s=1", channels: (64, 64, 64), offset: 2.0),
      (type: "convres", widths: (0.55, 0.55), height: 4.2, depth: 4.2, label: "Stage 2 (×2)", xlabel: "128 ch, s=2", channels: (128, 128, 32), offset: 2.2),
      (type: "convres", widths: (0.65, 0.65), height: 3.0, depth: 3.0, label: "Stage 3 (×2)", xlabel: "256 ch, s=2", channels: (256, 256, 16), offset: 2.2),
      (type: "convres", widths: (0.75, 0.75), height: 2.0, depth: 2.0, label: "Stage 4 (×2)", xlabel: "512 ch, s=2", channels: (512, 512, 8), offset: 2.2),
      (type: "gap", height: 1.4, depth: 1.4, label: "GAP", channels: (512,), offset: 1.8),
      (type: "fc", height: 3.2, depth: 0.25, label: "Classifier", xlabel: "Dense 200", channels: (200,), offset: 1.5),
    )
  }

  align(center)[
    #draw-network(
      layers,
      scale: scale,
      show-relu: true,
      palette: palette,
    )
  ]
}

/// Detailed Zoom-In of a Single ResNet BasicBlock
/// Drawn with CeTZ showing: Conv3x3 -> GN32 -> ReLU -> Conv3x3 -> GN32 -> Residual Addition -> ReLU
///
/// Parameters:
/// - `scale`: Scaling factor for slide embedding (default 80%)
/// - `downsample`: If true, illustrates a downsampling block with 1x1 projection; if false, shows identity bypass.
#let resnet18-block-diagram(
  scale: 80%,
  downsample: false,
) = {
  let content-block = canvas({
    import draw: *

    let box-w = 2.25
    let box-h = 1.15
    let gap = 0.65
    let y0 = 0.0

    // Palette & Styles
    let conv-fill = rgb("#e0f2fe")
    let conv-stroke = 1.2pt + brand-primary
    let gn-fill = rgb("#fef3c7")
    let gn-stroke = 1.2pt + brand-amber.darken(20%)
    let relu-fill = rgb("#fee2e2")
    let relu-stroke = 1.2pt + brand-accent
    let arrow-style = (stroke: 1.2pt + brand-dark, mark: (end: ">", fill: brand-dark, size: 0.22))

    // Input annotation
    let in-x = -2.1
    content((in-x, y0), align(center)[
      #text(weight: "bold", size: 10pt, fill: text-main)[Input $x$]\
      #text(size: 8pt, fill: text-muted)[$H times W times C$]
    ], name: "input")

    // Block 1: First 3x3 Conv
    let x1 = 0.0
    rect((x1, y0 - box-h/2), (x1 + box-w, y0 + box-h/2), fill: conv-fill, stroke: conv-stroke, radius: 4pt, name: "conv1")
    let s-desc = if downsample [stride 2, pad 1] else [stride $s$, pad 1]
    content((x1 + box-w/2, y0), align(center)[
      #text(weight: "bold", size: 9pt, fill: brand-primary)[3×3 Conv]\
      #text(size: 7.2pt, fill: text-main)[#s-desc\ no bias]
    ])

    // Block 2: GroupNorm (32 groups)
    let x2 = x1 + box-w + gap
    let gn-w = box-w * 0.92
    rect((x2, y0 - box-h/2), (x2 + gn-w, y0 + box-h/2), fill: gn-fill, stroke: gn-stroke, radius: 4pt, name: "gn1")
    content((x2 + gn-w/2, y0), align(center)[
      #text(weight: "bold", size: 9pt, fill: brand-amber.darken(30%))[GroupNorm]\
      #text(size: 7.2pt, fill: text-main)[32 groups\ affine $(gamma, beta)$]
    ])

    // Block 3: First ReLU
    let x3 = x2 + gn-w + gap
    let relu-w = 1.2
    rect((x3, y0 - box-h/2), (x3 + relu-w, y0 + box-h/2), fill: relu-fill, stroke: relu-stroke, radius: 4pt, name: "relu1")
    content((x3 + relu-w/2, y0), align(center)[
      #text(weight: "bold", size: 9pt, fill: brand-accent)[ReLU]\
      #text(size: 7.2pt, fill: text-main)[activation]
    ])

    // Block 4: Second 3x3 Conv
    let x4 = x3 + relu-w + gap
    rect((x4, y0 - box-h/2), (x4 + box-w, y0 + box-h/2), fill: conv-fill, stroke: conv-stroke, radius: 4pt, name: "conv2")
    content((x4 + box-w/2, y0), align(center)[
      #text(weight: "bold", size: 9pt, fill: brand-primary)[3×3 Conv]\
      #text(size: 7.2pt, fill: text-main)[stride 1, pad 1\ no bias]
    ])

    // Block 5: Second GroupNorm (32 groups)
    let x5 = x4 + box-w + gap
    rect((x5, y0 - box-h/2), (x5 + gn-w, y0 + box-h/2), fill: gn-fill, stroke: gn-stroke, radius: 4pt, name: "gn2")
    content((x5 + gn-w/2, y0), align(center)[
      #text(weight: "bold", size: 9pt, fill: brand-amber.darken(30%))[GroupNorm]\
      #text(size: 7.2pt, fill: text-main)[32 groups\ affine $(gamma, beta)$]
    ])

    // Node 6: Residual Addition Circle (+)
    let x6 = x5 + gn-w + gap + 0.35
    let sum-r = 0.38
    circle((x6, y0), radius: sum-r, fill: rgb("#dcfce7"), stroke: 1.5pt + brand-green, name: "sum")
    content((x6, y0), text(weight: "bold", size: 14pt, fill: brand-green.darken(25%))[+])

    // Block 7: Second / Final ReLU
    let x7 = x6 + sum-r + gap
    rect((x7, y0 - box-h/2), (x7 + relu-w, y0 + box-h/2), fill: relu-fill, stroke: relu-stroke, radius: 4pt, name: "relu2")
    content((x7 + relu-w/2, y0), align(center)[
      #text(weight: "bold", size: 9pt, fill: brand-accent)[ReLU]\
      #text(size: 7.2pt, fill: text-main)[activation]
    ])

    // Output annotation (with ample space to prevent collision)
    let out-x = x7 + relu-w + 2.0
    let out-shape = if downsample [ $H/2 times W/2 times C'$ ] else [ $H/s times W/s times C'$ ]
    content((out-x, y0), align(center)[
      #text(weight: "bold", size: 10pt, fill: text-main)[Output $y$]\
      #text(size: 8pt, fill: text-muted)[#out-shape]
    ], name: "output")

    // Main forward line connections
    line((-1.3, y0), (x1, y0), ..arrow-style)
    line((x1 + box-w, y0), (x2, y0), ..arrow-style)
    line((x2 + gn-w, y0), (x3, y0), ..arrow-style)
    line((x3 + relu-w, y0), (x4, y0), ..arrow-style)
    line((x4 + box-w, y0), (x5, y0), ..arrow-style)
    line((x5 + gn-w, y0), (x6 - sum-r, y0), ..arrow-style)
    line((x6 + sum-r, y0), (x7, y0), ..arrow-style)
    line((x7 + relu-w, y0), (out-x - 0.8, y0), ..arrow-style)

    // Residual Shortcut Branch
    let branch-x = -0.75
    let skip-y = 2.1

    // Branch takeoff point
    circle((branch-x, y0), radius: 0.08, fill: brand-teal, stroke: none)
    line((branch-x, y0), (branch-x, skip-y), stroke: 1.3pt + brand-teal)

    if downsample {
      // Downsampling path: 1x1 Conv + GN(32)
      let proj-w = 5.8
      let proj-h = 1.45
      let proj-x1 = (branch-x + x6) / 2 - proj-w / 2
      let proj-x2 = proj-x1 + proj-w
      
      line((branch-x, skip-y), (proj-x1, skip-y), stroke: 1.3pt + brand-teal)
      
      rect((proj-x1, skip-y - proj-h/2), (proj-x2, skip-y + proj-h/2), fill: rgb("#f0fdf4"), stroke: 1.3pt + brand-teal, radius: 4pt)
      content(((proj-x1 + proj-x2) / 2, skip-y + 0.20), text(weight: "bold", size: 8.5pt, fill: brand-teal.darken(20%))[1×1 Conv (stride 2) + GroupNorm(32)])
      content(((proj-x1 + proj-x2) / 2, skip-y - 0.22), text(size: 7.2pt, fill: text-muted)[Projection Shortcut ($s=2, C arrow.r C'$, $H/2 times W/2$)])
      
      line((proj-x2, skip-y), (x6, skip-y), stroke: 1.3pt + brand-teal)
    } else {
      // Identity shortcut path with informational badge
      let badge-w = 5.8
      let badge-h = 1.45
      let badge-x1 = (branch-x + x6) / 2 - badge-w / 2
      let badge-x2 = badge-x1 + badge-w
      
      line((branch-x, skip-y), (badge-x1, skip-y), stroke: 1.3pt + brand-teal)
      
      rect((badge-x1, skip-y - badge-h/2), (badge-x2, skip-y + badge-h/2), fill: rgb("#f0fdf4"), stroke: 1.3pt + brand-green, radius: 4pt)
      content(((badge-x1 + badge-x2) / 2, skip-y + 0.20), text(weight: "bold", size: 8.5pt, fill: brand-green.darken(25%))[Residual Shortcut ($x + cal(F)(x)$)])
      content(((badge-x1 + badge-x2) / 2, skip-y - 0.22), text(size: 7.2pt, fill: text-muted)[Identity Shortcut ($s=1$, No Projection)])
      
      line((badge-x2, skip-y), (x6, skip-y), stroke: 1.3pt + brand-teal)
    }

    // Shortcut arrival into (+) node
    line((x6, skip-y), (x6, y0 + sum-r), stroke: 1.3pt + brand-teal, mark: (end: ">", fill: brand-teal, size: 0.22))
  })

  align(center)[
    #std.scale(scale, reflow: true)[#content-block]
  ]
}

/// Statistics highlight cards summarizing ResNet-18 parameters, arrays, and normalization
#let architecture-stats-cards() = {
  grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 10pt,
    
    // Total Parameters Card
    rect(
      width: 100%,
      stroke: 1pt + brand-primary.lighten(60%),
      fill: rgb("#f8fafc"),
      radius: 6pt,
      inset: (x: 10pt, y: 8pt),
      align(center)[
        #text(size: 1.45em, weight: "bold", fill: brand-primary)[11.27M] \
        #v(2pt)
        #text(size: 0.85em, weight: "medium", fill: text-main)[Trainable Parameters] \
        #v(1pt)
        #text(size: 0.7em, fill: text-muted)[11,271,432 exact count]
      ]
    ),
    
    // Arrays Card
    rect(
      width: 100%,
      stroke: 1pt + brand-dark.lighten(60%),
      fill: rgb("#f8fafc"),
      radius: 6pt,
      inset: (x: 10pt, y: 8pt),
      align(center)[
        #text(size: 1.45em, weight: "bold", fill: brand-dark)[62] \
        #v(2pt)
        #text(size: 0.85em, weight: "medium", fill: text-main)[Parameter Arrays] \
        #v(1pt)
        #text(size: 0.7em, fill: text-muted)[20 Conv + 40 GN + 2 Dense]
      ]
    ),
    
    // Normalization Card
    rect(
      width: 100%,
      stroke: 1pt + brand-teal.lighten(60%),
      fill: rgb("#f8fafc"),
      radius: 6pt,
      inset: (x: 10pt, y: 8pt),
      align(center)[
        #text(size: 1.45em, weight: "bold", fill: brand-teal)[20 Layers] \
        #v(2pt)
        #text(size: 0.85em, weight: "medium", fill: text-main)[GroupNorm(32)] \
        #v(1pt)
        #text(size: 0.7em, fill: text-muted)[Batch-size invariant (no sync)]
      ]
    ),
    
    // Resolution Card
    rect(
      width: 100%,
      stroke: 1pt + rgb("#475569").lighten(60%),
      fill: rgb("#f8fafc"),
      radius: 6pt,
      inset: (x: 10pt, y: 8pt),
      align(center)[
        #text(size: 1.45em, weight: "bold", fill: rgb("#475569"))[64×64×3] \
        #v(2pt)
        #text(size: 0.85em, weight: "medium", fill: text-main)[Input Resolution] \
        #v(1pt)
        #text(size: 0.7em, fill: text-muted)[Tiny ImageNet (200 classes)]
      ]
    ),
  )
}

/// Detailed Summary Table of the ResNet-18 Architecture & Trainable Parameters
/// Verified against experiments/static-scaling/specification.md:133-152
#let architecture-summary-table() = {
  align(center)[
    #table(
      columns: (1.4fr, 2.6fr, 1.2fr, 1.2fr, 1.0fr),
      align: (left, left, center, right, right),
      stroke: (x, y) => if y == 0 {
        (bottom: 1.5pt + brand-primary)
      } else if y == 8 {
        (top: 1.5pt + brand-dark)
      } else {
        (bottom: 0.5pt + border-color)
      },
      fill: (x, y) => if y == 0 {
        brand-light
      } else if y == 8 {
        rgb("#f1f5f9")
      } else if calc.even(y) {
        rgb("#f8fafc")
      } else {
        none
      },
      
      // Header
      table.header(
        [*Component*],
        [*Specification & Layers*],
        [*Output Shape*],
        [*Parameters*],
        [*Arrays*]
      ),
      
      // Rows
      [Stem], [3×3 Conv (s1, p1, no bias) + GroupNorm(32) + ReLU], [$64 times 64 times 64$], [1,728], [3],
      [Stage 1], [2 BasicBlocks (64 ch, stride 1, no projection)], [$64 times 64 times 64$], [147,456], [12],
      [Stage 2], [2 BasicBlocks (128 ch, stride 2, 1×1 proj + GN)], [$32 times 32 times 128$], [524,288], [15],
      [Stage 3], [2 BasicBlocks (256 ch, stride 2, 1×1 proj + GN)], [$16 times 16 times 256$], [2,097,152], [15],
      [Stage 4], [2 BasicBlocks (512 ch, stride 2, 1×1 proj + GN)], [$8 times 8 times 512$], [8,388,608], [15],
      [Pooling], [Global spatial average pooling (GAP)], [$512$], [0], [0],
      [Classifier], [Dense Linear (512 $arrow$ 200 logits) + bias], [200 logits], [102,600], [2],
      
      // Total Row
      [*Total*], [*ResNet-18 GN-32 (20 Conv + 20 GN + 1 Dense)*], [*200 classes*], [*11,271,432*], [*62*],
    )
  ]
}
