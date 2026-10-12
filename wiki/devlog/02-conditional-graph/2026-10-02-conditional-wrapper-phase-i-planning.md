# Phase I agenda and post-1.0 performance goals

## Request and result

The user requested guidance for Phase I and a review of metadata storage costs for models with many parameters.
The user also requested parallelization, SIMD, allocation analysis, and performance goals after stable conditional-graph support 1.0.

The implementation plan now defines:

- A Phase I review procedure, decision table, and release follow-ups in section 12.
- An updated current entry point and exact next action in section 22.
- A post-1.0 performance roadmap under C9/C10 in section 23.

The initial planning pass left Phase I acceptance pending.
The subsequent user decisions appear in the next section. The README progress table and current context reflect those decisions.

## User decisions and validation extensions

The user accepted the wrapper API, explicit `.tree` access, synchronization contract, and checkpoint recipe for the supervisor prototype.
The user requested native GPU validation. GPU-specific failures can move to the next upgrade batch after characterization.
Prototype 1.0 records reproducible results and limitations. It does not imply production readiness or support for a failed GPU configuration.

The user requested a specific report before deciding whether uncommon behavioral limitations need handling.
Created `temp/conditional_graph_deferred_behaviors_prompt.md` for that later agent session.
The prompt prioritizes rank-local non-finite loss behavior and distinguishes it from non-finite gradients, unequal participation, and configuration mismatch.
It requires bounded probes, passing controls, source evidence, realistic reachability, impact, and explicit deferral conditions.

The user also requested a concrete MNIST or toy-dataset demonstration.
The plan recommends a deterministic toy classifier first, then MNIST through the same two-head validation harness.
The reference differentiates the mean of the scheduled local losses without the distributed wrapper.
Per-step comparisons cover model parameters, Adam state, replica agreement, and active-to-globally-absent preservation.
Checkpoint continuation and presentation plots complete the demonstration.

New source observations:

- `test/ext_distributed/conditional_distributedtest.jl:26-42` uses CPU-specific vectors. NCCL selection alone does not establish GPU coverage.
- `test/ext_distributed/distributed_setup.jl` initializes the selected backend but does not adapt those fixtures.
- `scripts/examples/train_ddp.jl:156` still calls the removed `resolve_unused_parameters!!` helper.
- `src/train.jl:375-378` still contains the rank-local loss-finiteness guard documented by the audit.

Added Phase J GPU validation and Phase K dataset validation to section 12 of the plan.
Both phases remain planned. The deferred-behavior report remains a later task.

## Source observations

Inspected sources in `../flux-integration`:

- `src/distributed/public_api.jl:575-736`: buffer representation, occurrence plan, traversal, presence reduction, and reconstruction.
- `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:62-84`: NCCL reduction and MPI fallback.

The presence payload uses one Float32 per synchronization buffer.
Mutable aliases share a buffer. Each isbits trainable occurrence has a separate buffer.
The payload therefore scales with tensor leaves rather than scalar weights.

Each update allocates an identity map, buffer records, occurrence entries, gradient buffers, presence flags, and a presence mask.
Repeated tied contributions also allocate temporary arrays.
The traversal prepares gradient buffers before the presence reduction reveals globally absent leaves.
The protocol issues one presence collective plus one gradient collective per globally present buffer.

These observations identify measurement candidates. They do not establish a performance bottleneck.
The roadmap requires separate leaf-count and gradient-byte sweeps, warmed timings, allocation attribution, and complete-step results.

## Proposed priority

1. Establish the stable baseline and component costs after 1.0 acceptance.
2. Reduce metadata allocations and evaluate reusable structural plans and owned buffers.
3. Evaluate SIMD for contiguous, independent CPU loops.
4. Evaluate threading for independent local preparation with controlled resources per rank.
5. Evaluate transport or collective changes only when the cost breakdown supports them.

The plan records cache invalidation, non-mutating ownership, packed-Boolean write races, tied accumulation, and ordered collective requirements.
Numeric performance budgets will come from C9 measurements. No speedup target is presented as an observed result.

## Checks and commands

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

```bash
git status --short
git status --short --untracked-files=no
git diff --stat -- README.md temp/conditional_graph_wrapper_implementation_plan.md wiki/context/current.md
git diff -- README.md temp/conditional_graph_wrapper_implementation_plan.md wiki/context/current.md
git diff --check
```

Flux status command, from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
git status --short --branch
```

The Flux status showed the existing three modified files: `NEWS.md`, `src/distributed/public_api.jl`, and the synchronization test file.
`git diff --check` passed with exit 0. All shell commands in this session returned exit 0.
This session changed only thesis documentation. It ran no Julia tests, profiling jobs, or remote commands.

## Exact next action

Implement the focused native CUDA/NCCL regression in section 12, Phase J1, of the implementation plan.
Then run the bounded two-rank/two-GPU case through the selected host's documented workflow.
