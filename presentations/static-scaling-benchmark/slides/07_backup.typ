// 07_backup.typ - Appendix / backup material
#import "@preview/touying:0.6.1": *
#import "../components/cards.typ": *
#import "../components/charts.typ": *

#show: appendix

== Backup: Diagnostic Snapshot Differences

#align(center)[
  #chart-diagnostics(show-header: false)
]

#v(2pt)

#text(size: 0.9em)[
  *Takeaway:* Schedule counters match exactly, weights stay within a narrow
  band, and optimizer moments diverge more, as expected for accumulated
  state. A few low-magnitude arrays can look large in relative terms.
]
