// Theme setup and reusable visual components for Static Scaling Benchmark presentation

#let brand-dark = rgb("#13293d")
#let brand-primary = rgb("#006494")
#let brand-teal = rgb("#247ba0")
#let brand-light = rgb("#e8f1f5")
#let brand-accent = rgb("#e76f51")
#let brand-amber = rgb("#f4a261")
#let brand-green = rgb("#2a9d8f")
#let bg-page = rgb("#fdfdfd")
#let bg-surface = rgb("#ffffff")
#let border-color = rgb("#e2e8f0")
#let text-muted = rgb("#64748b")
#let text-main = rgb("#1e293b")

// Metric highlight card
#let metric-card(value, label, subtext: none, color: brand-primary, fill: rgb("#f8fafc")) = {
  rect(
    width: 100%,
    stroke: 1pt + color.lighten(60%),
    fill: fill,
    radius: 6pt,
    inset: (x: 10pt, y: 8pt),
    [
      #align(center)[
        #text(size: 1.5em, weight: "bold", fill: color, value) \
        #v(2pt)
        #text(size: 0.85em, weight: "medium", fill: text-main, label)
        #if subtext != none [
          \ #v(1pt)
          #text(size: 0.7em, fill: text-muted, subtext)
        ]
      ]
    ]
  )
}

// Styled Card container
#let card(title: none, body, fill: bg-surface, stroke: 1pt + border-color, radius: 6pt, inset: 10pt, title-color: brand-primary) = {
  rect(
    width: 100%,
    stroke: stroke,
    fill: fill,
    radius: radius,
    inset: inset,
    [
      #if title != none [
        #text(weight: "bold", size: 0.95em, fill: title-color, title)
        #v(4pt)
        #line(length: 100%, stroke: 0.5pt + stroke.paint)
        #v(4pt)
      ]
      #set text(size: 0.85em, fill: text-main)
      #body
    ]
  )
}

// Status Badge
#let status-badge(status, level: "pass") = {
  let (bg, fg) = if level == "pass" {
    (rgb("#dcfce7"), rgb("#15803d"))
  } else if level == "warn" or level == "qualified" {
    (rgb("#fef3c7"), rgb("#b45309"))
  } else if level == "info" {
    (rgb("#e0f2fe"), rgb("#0369a1"))
  } else {
    (rgb("#fee2e2"), rgb("#b91c1c"))
  }
  box(
    fill: bg,
    radius: 4pt,
    inset: (x: 6pt, y: 3pt),
    outset: 0pt,
    text(size: 0.72em, weight: "bold", fill: fg, status)
  )
}

// Key Takeaway callout
#let takeaway-box(body, title: [Key Takeaway], icon-symbol: "💡") = {
  rect(
    width: 100%,
    stroke: (left: 3.5pt + brand-primary),
    fill: rgb("#f0f9ff"),
    radius: (right: 4pt),
    inset: (x: 10pt, y: 8pt),
    [
      #text(weight: "bold", size: 0.88em, fill: brand-primary)[#icon-symbol #title] \
      #v(2pt)
      #set text(size: 0.8em, fill: text-main)
      #body
    ]
  )
}

// Warning / Caution callout
#let caution-box(body, title: [Important Qualification]) = {
  rect(
    width: 100%,
    stroke: (left: 3.5pt + brand-accent),
    fill: rgb("#fff5f5"),
    radius: (right: 4pt),
    inset: (x: 10pt, y: 8pt),
    [
      #text(weight: "bold", size: 0.88em, fill: brand-accent)[⚠️ #title] \
      #v(2pt)
      #set text(size: 0.8em, fill: text-main)
      #body
    ]
  )
}
