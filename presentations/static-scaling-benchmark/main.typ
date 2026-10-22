// main.typ - Top-level presentation entry point
// Compiles all modular slide files using Touying metropolis theme

#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "components/cards.typ": *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Strong Scaling of Flux.jl DDP and PyTorch],
    subtitle: [One Leonardo node · 4× A100-SXM-64GB · fixed global batch],
    author: [Flux.jl DDP Thesis Research Group],
    date: datetime.today(),
    institution: [Leonardo EuroHPC / CINECA],
  ),
  config-colors(
    primary: rgb("#475569"),
    secondary: rgb("#1e293b"),
    neutral-dark: rgb("#0f172a"),
    neutral-light: rgb("#f8fafc"),
  )
)

#set text(font: ("Liberation Sans", "Noto Sans", "Arial"), size: 15pt)
#set list(spacing: 0.5em)

#title-slide()

#include "slides/01_overview.typ"
#include "slides/02_architecture_workload.typ"
#include "slides/03_methodology.typ"
#include "slides/04_results.typ"
#include "slides/05_deep_dives.typ"
#include "slides/06_conclusions.typ"
#include "slides/07_backup.typ"
