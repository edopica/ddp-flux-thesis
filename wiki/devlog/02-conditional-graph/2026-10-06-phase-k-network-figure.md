# Phase K K2 network diagram

Date: 2026-10-06

The user requested a graphical representation of the Phase K MNIST/Fashion-MNIST
(K2) conditional network. No training, remote execution, or Flux source change
occurred.

Added `scripts/plots/plot_phase_k_network.py`, which parses the network and
routing from the repository sources instead of hardcoding them:

- model dimensions from the K2 line in `conditional_graph_demo.jl`
  (`build_toy_model(seed_init; indim=784, hidden=32, nclasses=10)`), and the
  trunk/head layer definitions in `conditional_graph/model.jl`;
- the six `ROUTING_CYCLE` entries and purposes from `conditional_graph_demo.jl`;
- the `prepare` global-batch default (128).

The tool writes two deterministic PNGs and a provenance record:

- `reports/figures/phase-k-network/phase-k-k2-network.png` - architecture
  (input 784, shared trunk `Dense(784 => 32, tanh)` 25 120 parameters, two
  `Dense(32 => 10)` heads 330 parameters each, total 25 780 Float32), plus one
  lane per task showing the active path and the locally absent head, and the
  DDP presence protocol (5 collectives with one task globally present, 7 with
  both).
- `reports/figures/phase-k-network/phase-k-k2-routing.png` - the six routing
  cycles per rank for R=2 and R=4 with the source purpose strings.

Verification:

- the recorded source hashes `model.jl 07de4f0f...`, `common.jl 9dfa74ee...`,
  and `conditional_graph_demo.jl b6e6725d...` match the K2 campaign
  `prepare_metadata.txt` (deathstar `k2-2026-10-05-deathstar-mpi`), so the
  drawing describes the network and sources used by the stored evidence;
- two runs produced byte-identical PNGs;
- command:
  `uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_network.py --out reports/figures/phase-k-network`.

The DDP collective count in the figure was checked against
`CGCommon.expected_collective_count` and the presence allreduce in
`flux-integration/src/distributed/public_api.jl` (`_sync_gradients`).

**Exact next action:** unchanged - write the deferred-behavior report; it is the
only remaining Phase K item. The GPU recap notebook is the follow-up after the
review (`reports/conditional-graph-phase-k-review.md`).
