// Reusable UI components for Touying presentation slides
// Optimized for 16:9 presentation layout with clear visual hierarchy

// --- Candidate palette: single source of truth for every slide and chart ---
// Historical Flux = red, Current Flux = yellow, PyTorch = blue.
// `-dark` variants are for text on light backgrounds, `-bg` for tints.
#let cand-historical = rgb("#dc2626")      // Red 600
#let cand-historical-dark = rgb("#991b1b") // Red 800 (readable text)
#let cand-historical-bg = rgb("#fef2f2")   // Red 50

#let cand-current = rgb("#ff9100")         // Yellow 600
#let cand-current-dark = rgb("#ff6d00")    // Yellow 800 (readable text)
#let cand-current-bg = rgb("#fefce8")      // Yellow 50

#let cand-pytorch = rgb("#2563eb")         // Blue 600
#let cand-pytorch-dark = rgb("#1e40af")    // Blue 800 (readable text)
#let cand-pytorch-bg = rgb("#eff6ff")      // Blue 50

#let primary-color = rgb("#1e3a8a")     // Navy Blue
#let secondary-color = rgb("#0284c7")   // Blue Accent
#let success-color = rgb("#059669")     // Green (status only)
#let warning-color = rgb("#d97706")     // Amber (status only)
#let danger-color = rgb("#dc2626")      // Red / Non-Claims
#let purple-color = rgb("#7c3aed")      // Indigo / Deep Dive
#let slate-dark = rgb("#0f172a")        // Dark Slate
#let slate-text = rgb("#334155")        // Body Slate
#let slate-light = rgb("#f8fafc")       // Card Background
#let slate-border = rgb("#e2e8f0")      // Subtle Border

/// Base card container with optional header, custom fill and stroke
#let card(
  title: none,
  badge: none,
  fill: slate-light,
  stroke: 1pt + slate-border,
  radius: 6pt,
  inset: 9pt,
  title-color: slate-dark,
  body,
) = {
  block(
    fill: fill,
    stroke: stroke,
    radius: radius,
    inset: inset,
    width: 100%,
    [
      #if title != none or badge != none [
        #grid(
          columns: (1fr, auto),
          align: (left + horizon, right + horizon),
          [
            #if title != none [
              #text(weight: "bold", size: 0.9em, fill: title-color)[#title]
            ]
          ],
          [
            #if badge != none [ #badge ]
          ],
        )
        #v(3pt)
        #line(length: 100%, stroke: 0.5pt + stroke.paint.lighten(40%))
        #v(4pt)
      ]
      #set text(size: 0.82em, fill: slate-text)
      #body
    ],
  )
}

/// Highlight metric box for prominent numerical values
#let metric-card(
  value,
  label,
  sub: none,
  accent: primary-color,
  fill: none,
) = {
  let bg = if fill != none { fill } else { accent.lighten(92%) }
  let border = accent.lighten(50%)
  block(
    fill: bg,
    stroke: 1pt + border,
    radius: 6pt,
    inset: (x: 8pt, y: 7pt),
    width: 100%,
    align(center)[
      #text(size: 1.35em, weight: "bold", fill: accent)[#value] \
      #v(1pt)
      #text(size: 0.78em, weight: "bold", fill: slate-dark)[#label]
      #if sub != none [
        \ #v(1pt)
        #text(size: 0.70em, fill: slate-text.lighten(20%))[#sub]
      ]
    ],
  )
}

/// Small pill badge for tags, states, and framework labels
#let badge(
  label,
  fill: slate-border,
  text-color: slate-dark,
) = {
  box(
    fill: fill,
    radius: 4pt,
    inset: (x: 5pt, y: 2pt),
    baseline: 0%,
    text(size: 0.72em, weight: "bold", fill: text-color)[#label],
  )
}

/// Status badges
#let pass-badge() = badge("PASS", fill: rgb("#d1fae5"), text-color: rgb("#065f46"))
#let notice-badge(txt) = badge(txt, fill: rgb("#fef3c7"), text-color: rgb("#92400e"))
#let info-badge(txt) = badge(txt, fill: rgb("#e0f2fe"), text-color: rgb("#0369a1"))
#let alert-badge(txt) = badge(txt, fill: rgb("#fee2e2"), text-color: rgb("#991b1b"))

/// Callout block for warnings, caveats, non-claims
#let callout(
  title: none,
  fill: rgb("#fffbeb"),
  stroke: 1pt + rgb("#fde68a"),
  title-color: rgb("#b45309"),
  body,
) = {
  block(
    fill: fill,
    stroke: stroke,
    radius: 6pt,
    inset: 8pt,
    width: 100%,
    [
      #if title != none [
        #text(weight: "bold", size: 0.85em, fill: title-color)[#title]
        #v(2pt)
      ]
      #set text(size: 0.8em, fill: slate-text)
      #body
    ],
  )
}
