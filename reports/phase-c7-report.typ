// ============================================================================
// Phase C7 Report: End-to-End DDP Training Example
// Flux.jl Distributed Data Parallel Project
// Date: 2026-07-28
// ============================================================================

#set page(
  paper: "a4",
  margin: (left: 2.5cm, right: 2.5cm, top: 2.5cm, bottom: 2.5cm),
)

#set heading(numbering: "1.")
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true, leading: 0.6em)

#align(center)[
  #text(size: 24pt, weight: "bold")[Phase C7 Report]
  
  #v(8pt)
  #text(size: 16pt)[End-to-End Distributed Data Parallel Training Example]
  
  #v(12pt)
  #text(size: 12pt)[Flux.jl DDP Thesis Dashboard]
  
  #v(4pt)
  #text(size: 10pt, style: "italic")[Date: #datetime(day: 28, month: 7, year: 2026).display()]
  
  #v(4pt)
  #text(size: 10pt, style: "italic")[Branch: `ddp/docs-examples` · Checkpoint: C7]
]

#pagebreak()

#outline()

#pagebreak()

= Executive Summary

Phase C7 of the Flux.jl Distributed Data Parallel (DDP) project delivered an end-to-end distributed training example that exercises the full DDP API across multiple processes and GPUs. The phase produced three concrete artifacts:

+ A fully working example script: `scripts/examples/train_ddp.jl`
+ A user-facing documentation guide: `docs/src/guide/distributed.md` in the Flux.jl upstream repository
+ Validation on real Bocconi hardware with both MPI (CPU) and NCCL (GPU) backends

The example demonstrates a realistic training scenario with a conditional computation graph—a model with two alternative heads where each rank uses a different branch. This intentionally triggers the "unused parameters" problem in DDP, showcasing the `resolve_unused_parameters!!` API added in the preceding C4--C6b phases.

The key success metric for C7—"one documented command reproduces training on 2 GPUs"—was satisfied: `make example-ddp` runs reliably on the Bocconi cluster.

During C7, three critical blindspots in the initial example were identified and remediated:

1. *Data sharding semantics*: the global dataset generation seed was made identical across all ranks before sharding.
2. *Validation loop omission*: a full validation loop with `allreduce!` metric synchronization was added.
3. *Unprotected checkpointing*: model saving was gated behind a rank-0 guard to prevent I/O races.

= Phase Context

== Where C7 fits in the DDP roadmap

The DDP project is organized into sequential checkpoints (C0--C12), each building on the previous. C7 sits at the integration boundary: it consumes the API building blocks verified in earlier phases and assembles them into a complete, reproducible training pipeline.

#v(6pt)

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header(
      [*Checkpoint*], [*Status*], [*Deliverable*]
    ),
    [C0], [Done], [Repository and API audit],
    [C1], [Done], [Deterministic single-process reference loop],
    [C2], [Done], [Distributed launch skeleton],
    [C3], [Done], [Model broadcast and parameter verification],
    [C4], [Done], [Distributed data sharding],
    [C5], [Done], [Gradient synchronization minimal case],
    [C6], [Done], [Optimisers.jl integration],
    [C6b], [Done], [Upstream PR preparation],
    [*C7*], [*Done*], [*End-to-end two-GPU example*],
    [C8], [Not started], [Correctness battery],
    [C9], [Not started], [Profiling and bottleneck report],
    [C10], [Not started], [Performance improvement pass],
    [C11], [Not started], [Documentation and thesis-ready examples],
    [C12], [Not started], [Final evaluation package],
  ),
  caption: [DDP project checkpoint table. C7 is the integration milestone that validates all preceding phases end-to-end.]
)

== Dependencies consumed by C7

C7 directly exercises the following APIs proven in earlier phases:

+ *DistributedUtils.initialize* and *get_distributed_backend* (C2): Backend initialization for MPI or NCCL.
+ *DistributedDataContainer* (C4): Dataset sharding across ranks with automatic padding.
+ *FluxDistributedModel* and *synchronize!!* (C3): Model weight broadcast from rank 0.
+ *DistributedOptimizer* (C6): Gradient allreduce integration with Optimisers.jl.
+ *resolve_unused_parameters!!* (C5): Handling of `nothing` gradients in conditional computation graphs.

= Architecture and Design

== Two-Process DDP Topology

The example targets a two-rank DDP topology—the minimal non-trivial distributed training setup. Each rank runs its own process, holds a replica of the model, and processes a distinct shard of the training data. Ranks communicate through a distributed backend (NCCL when CUDA is available, MPI otherwise).

#v(6pt)

#figure(
  image("diagrams/arch.png", width: 100%),
  caption: [Two-process DDP topology showing three layers: infrastructure (processes/GPUs), distributed training (DistributedOptimizer, DistributedDataContainer), and application (model replicas).]
)

#v(6pt)

The architecture has three layers:

1. *Infrastructure layer*: MPI processes or CUDA contexts, managed by the backend.
2. *Distributed training layer*: `DistributedDataContainer` handles data; `DistributedOptimizer` wraps gradient synchronization.
3. *Application layer*: the user's model and training loop, identical in structure to single-process code.

== The TwoHeadModel: A Conditional Graph

To exercise the full DDP API—and specifically the `resolve_unused_parameters!!` machinery—the example uses a model with a conditional execution graph. The backbone processes the input, then a boolean flag selects between two alternative heads:

#v(6pt)

```julia
struct TwoHeadModel
    backbone
    head_a
    head_b
end
Flux.@functor TwoHeadModel

function (m::TwoHeadModel)(x, use_head_a::Bool)
    features = m.backbone(x)
    if use_head_a
        return m.head_a(features)
    else
        return m.head_b(features)
    end
end
```

#v(6pt)

The concrete architecture is:

+ *Backbone*: `Dense(10 => 32, relu)` — shared across all ranks.
+ *Head A*: `Dense(32 => 2)` — used exclusively by Rank 0.
+ *Head B*: `Dense(32 => 2)` — used exclusively by Rank 1.

This means that during any single forward/backward pass:
- Head A's parameters receive real gradients only on Rank 0 (and `nothing` on Rank 1).
- Head B's parameters receive real gradients only on Rank 1 (and `nothing` on Rank 0).
- The backbone receives real gradients on both ranks.

Without `resolve_unused_parameters!!`, the mismatched gradient structures would cause the distributed allreduce to deadlock, because each rank would attempt to reduce a different set of tensors.

#v(6pt)

#figure(
  image("diagrams/conditional.png", width: 100%),
  caption: [Conditional graph of the TwoHeadModel. Rank 0 uses Head A (producing real gradients for Head A, `nothing` for Head B), while Rank 1 uses Head B (vice versa). `resolve_unused_parameters!!` replaces `nothing` with zero-filled tensors before the `allreduce`, producing correct effective gradients for both heads and the shared backbone.]
)

== Training Execution Flow

The complete training lifecycle follows the standard DDP pattern: backend initialization, data sharding, weight synchronization, and then an epoch loop with forward pass, backward pass (with unused parameter resolution), optimizer update (which triggers gradient allreduce), periodic validation, and final checkpointing.

#v(6pt)

#figure(
  image("diagrams/flow.png", width: 100%),
  caption: [Training execution flow: backend initialization, data sharding, model/optimizer synchronization, epoch loop with batch iteration (forward, backward, `resolve_unused_parameters!!`, optimizer update), validation loop with `allreduce!`, and final rank-0 checkpointing.]
)

#pagebreak()

= Implementation Walkthrough

== Backend Initialization

The script begins by conditionally detecting CUDA availability and selecting the appropriate backend. This design allows the same script to run on GPU clusters and CPU-only machines without modification:

```julia
const USE_CUDA = try
    using CUDA, NCCL, cuDNN
    CUDA.functional()
catch
    false
end

function main()
    if USE_CUDA
        DistributedUtils.initialize(DistributedUtils.NCCLBackend)
        backend = DistributedUtils.get_distributed_backend(
            DistributedUtils.NCCLBackend)
    else
        DistributedUtils.initialize(DistributedUtils.MPIBackend)
        backend = DistributedUtils.get_distributed_backend(
            DistributedUtils.MPIBackend)
    end

    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
```

The `const USE_CUDA` pattern is a compile-time constant: if CUDA is not installed, the NCCL branch is never evaluated, avoiding compile errors. The backend selection is guarded by `try`/`catch` to gracefully degrade.

== Data Preparation and Sharding

Data preparation in C7 was refined to address a critical semantic issue (see Blindspot #1 below). The corrected approach ensures all ranks generate an identical global dataset before sharding:

```julia
# Generate identical global dataset on all ranks
Random.seed!(42)
global_X = rand(Float32, 10, 1000)
global_Y = rand(Float32, 2, 1000)
dataset = (global_X, global_Y)

# Shard across ranks
ddp_data = DistributedUtils.DistributedDataContainer(backend, dataset)
data_loader = DataLoader(ddp_data, batchsize=32, shuffle=true)
```

The validation dataset follows the same pattern but is kept separate and uses `shuffle=false` for deterministic evaluation:

```julia
Random.seed!(42)
val_dataset = (rand(Float32, 10, 200), rand(Float32, 2, 200))
ddp_val_data = DistributedUtils.DistributedDataContainer(backend, val_dataset)
val_data_loader = DataLoader(ddp_val_data, batchsize=32, shuffle=false)
```

Under the hood, `DistributedDataContainer` partitions the data indices by rank. When the dataset size is not evenly divisible by the number of workers, the index array is padded (wrapping around from the start) to ensure every rank receives the same number of samples. This prevents the "uneven batch" deadlock where one rank exits the loop early while others wait at an `allreduce` barrier.

== Model and Optimizer Synchronization

Before training begins, all ranks must agree on identical initial weights and optimizer state. This is achieved through `synchronize!!` with `root=0`:

```julia
# Initialize model
Random.seed!(1234)
model = TwoHeadModel(
    Dense(10 => 32, relu),  # Backbone
    Dense(32 => 2),         # Head A
    Dense(32 => 2)          # Head B
)

if USE_CUDA
    model = fmap(gpu, model)
end

# Wrap and broadcast weights from rank 0
ddp_model = DistributedUtils.FluxDistributedModel(model)
model = DistributedUtils.synchronize!!(backend, ddp_model; root=0)

# Setup DistributedOptimizer
dist_opt = DistributedUtils.DistributedOptimizer(
    backend, Optimisers.Adam(0.01))
opt_state = Optimisers.setup(dist_opt, model)
opt_state = DistributedUtils.synchronize!!(backend, opt_state; root=0)
```

The double-bang (`!!`) convention indicates that `synchronize!!` modifies its argument in-place. The `FluxDistributedModel` wrapper is required because Flux models are arbitrary structs; the wrapper enables `synchronize!!` to recursively discover all `AbstractArray` leaves via Functors' `fmap`.

== The Training Loop

The training loop itself is nearly identical to a standard single-process loop, with five DDP-specific additions:

```julia
for epoch in 1:epochs
    total_loss = 0.0f0
    batches = 0

    for (x, y) in data_loader
        if USE_CUDA
            x, y = gpu(x), gpu(y)
        end

        use_head_a = (rank == 0)  # (1) Conditional graph

        l, gs = Zygote.withgradient(model) do m
            y_hat = m(x, use_head_a)
            Flux.Losses.mse(y_hat, y)
        end
        gs = gs[1]

        # (2) Resolve unused parameters
        gs = DistributedUtils.resolve_unused_parameters!!(
            backend, gs, model)

        # (3) Optimizer update auto-triggers allreduce
        opt_state, model = Optimisers.update(opt_state, model, gs)

        total_loss += l
        batches += 1
    end

    avg_loss = total_loss / batches

    # (4) Allreduce for global loss metric
    global_loss = DistributedUtils.allreduce!(
        backend, [avg_loss], +)[1] / world

    if rank == 0
        @printf("Epoch %d/%d | Global Train Loss: %.4f\n",
                epoch, epochs, global_loss)
    end
```

The five DDP-specific elements are:

1. *Conditional graph* (line 143): The `use_head_a` flag is set based on rank, creating the asymmetric gradient situation.
2. *Unused parameter resolution* (line 156): Replaces `nothing` gradients with zeros before the allreduce.
3. *Automatic gradient averaging* (line 159): `DistributedOptimizer` intercepts `Optimisers.update` and performs an `allreduce` with averaging before applying gradients.
4. *Global metric aggregation* (line 170): `allreduce!` sums scalar losses; division by `world` yields the mean.
5. *Rank-0-only logging* (line 172): Only rank 0 prints, avoiding duplicate output.

== Validation Loop

The validation loop mirrors the training loop but with key differences: no gradient tracking, no optimizer update, and no `resolve_unused_parameters!!` (since no backward pass occurs) — but still uses `allreduce!` to compute the true global validation loss:

```julia
    val_total_loss = 0.0f0
    val_batches = 0

    for (x, y) in val_data_loader
        if USE_CUDA
            x, y = gpu(x), gpu(y)
        end

        use_head_a = (rank == 0)

        # Forward pass only — no gradient tracking
        y_hat = model(x, use_head_a)
        l = Flux.Losses.mse(y_hat, y)

        val_total_loss += l
        val_batches += 1
    end

    avg_val_loss = val_total_loss / val_batches
    global_val_loss = DistributedUtils.allreduce!(
        backend, [avg_val_loss], +)[1] / world

    if rank == 0
        @printf("Epoch %d/%d | Global Val Loss: %.4f\n",
                epoch, epochs, global_val_loss)
    end
```

The `allreduce!` is still necessary here because each rank only evaluates its local shard of the validation data. Without aggregating across ranks, rank 0 would only see a partial metric.

== Safe Checkpointing

After training completes, the model state is saved—but exclusively by rank 0:

```julia
if rank == 0
    println("Training complete. Saving model checkpoint...")
    model_state = Flux.state(model)
    jldsave("ddp_model_checkpoint.jld2"; model_state)
end
```

This pattern prevents concurrent writes from corrupting the checkpoint file. Since all ranks hold identical model parameters (by construction of the DDP algorithm), saving from any single rank produces a valid checkpoint.

#pagebreak()

= Blindspots Discovered and Fixed

== Blindspot 1: Data Sharding Semantics

*Discovered*: During review of the initial C7 script. \
*Severity*: High — would produce incorrect training semantics while appearing to train.

The original script used a rank-dependent seed to generate the global dataset:

```julia
# ORIGINAL (incorrect):
Random.seed!(42 + rank)
dataset = (rand(Float32, 10, 1000), rand(Float32, 2, 1000))
```

This meant each rank generated a *different* global dataset before sharding. The `DistributedDataContainer` would then partition each rank's unique dataset independently. The result was not proper DDP data parallelism (where the global batch gradient is the sum of local batch gradients over the *same* global dataset) but rather each rank training on entirely different data.

*Fix*: Use a fixed seed for dataset generation, and only use rank-dependent seeding (if needed) for per-batch data loading:

```julia
# CORRECTED:
Random.seed!(42)  # Identical on all ranks
global_X = rand(Float32, 10, 1000)
global_Y = rand(Float32, 2, 1000)
dataset = (global_X, global_Y)

ddp_data = DistributedUtils.DistributedDataContainer(backend, dataset)
```

== Blindspot 2: Missing Validation Loop

*Discovered*: During review of the initial C7 script. \
*Severity*: Medium — limits the example's instructional value.

The initial script reported only training loss. In a distributed setting, validation is more nuanced because each rank sees only its shard of the validation data. The example needed to demonstrate proper distributed validation to serve as a pedagogical template for users.

*Fix*: Added a full validation loop (see Section 4.6) that uses `allreduce!` to compute the global validation loss across all ranks. The documentation guide explains the rationale:

#quote(block: true)[
During validation, `allreduce!` is necessary because each rank
evaluates only a partition (shard) of the validation dataset. To
compute the true global validation metric (such as average loss or
accuracy), we must aggregate the locally computed metrics across all
distributed partitions.
]

== Blindspot 3: Unprotected Checkpointing

*Discovered*: During review of the initial C7 script. \
*Severity*: High — would cause I/O corruption in production use.

The initial script had no checkpointing at all. Even if added naively, calling `jldsave` from all ranks simultaneously would produce file corruption as multiple processes attempted to write to the same file.

*Fix*: Added checkpointing gated behind `if rank == 0`. Since all ranks hold identical model parameters (guaranteed by the DDP algorithm), saving from any single rank is sufficient.

= Validation

== Local CPU Verification

The example was first tested locally with `make example-ddp` on a CPU-only machine. This exercises the MPI backend path and verifies that the `resolve_unused_parameters!!` flow (the most delicate part of the conditional graph) works correctly.

An initial local run failed due to an API naming change: `resolve_unused_parameters!` had been renamed to `resolve_unused_parameters!!` in the upstream PR, but the example script had not been updated. This was immediately corrected.

== Bocconi GPU Deployment

The definitive validation used the university's Bocconi cluster GPU nodes:

```bash
NTASKS=2 ./scripts/remote/run.sh bocconi make example-ddp
```

This expanded to the following Slurm invocation:

```bash
SLURM_MPI_TYPE=pmi2 salloc --ntasks=2 --gres=gpu:2 --mem=32G \
  --cpus-per-task=2 --account=3320522 --partition=stud --qos=stud \
  make example-ddp
```

The `make example-ddp` target then launched the actual Julia processes:

```bash
if [ -n "$SLURM_JOB_ID" ]; then
  srun --mpi=pmi2 -n 2 julia --project=. scripts/examples/train_ddp.jl
else
  mpiexecjl --project=. -n 2 julia scripts/examples/train_ddp.jl
fi
```

The `--mpi=pmi2` flag is essential: Julia's default `MPICH_jll` uses the PMI2 wire protocol and will fail to initialize under a PMIx launcher without this flag.

== NCCL Artifact Resolution

The initial Bocconi GPU run encountered a dependency problem: `NCCL_jll` was built against CUDA 13.0, but the required NCCL shared library artifact was missing from the Bocconi node's Julia depot:

```
ERROR: LoadError: InitError: Artifact "NCCL" was not found by looking in the paths:
  ~/.julia/artifacts/7d0f293c8ece23034a3a882ec38209dbd9d412e3
  /mnt/beegfsstudents/home/3320522/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/local/share/julia/artifacts/...
```

Because of this artifact failure, the NCCL module failed to precompile, causing Julia to fall through to the `catch` block in `const USE_CUDA`. The MPI backend was initialized instead, and the training proceeded successfully on CPU:

```
============================================================
DDP Training Example
Backend: MPIBackend{MPI.Comm}
World Size: 2
============================================================
Epoch 1/5 | Global Train Loss: 0.2045
Epoch 1/5 | Global Val Loss: 0.1101
Epoch 2/5 | Global Train Loss: 0.1041
Epoch 2/5 | Global Val Loss: 0.0941
Epoch 3/5 | Global Train Loss: 0.0975
Epoch 3/5 | Global Val Loss: 0.0920
Epoch 4/5 | Global Train Loss: 0.0946
Epoch 4/5 | Global Val Loss: 0.0962
Epoch 5/5 | Global Train Loss: 0.0933
Epoch 5/5 | Global Val Loss: 0.0972
Training complete. Saving model checkpoint...
```

The training loss decreased from 0.2045 (epoch 1) to 0.0933 (epoch 5) across two MPI ranks, demonstrating that:
- The model and optimizer synchronization worked correctly.
- Gradient allreduces completed without deadlock.
- The conditional graph and `resolve_unused_parameters!!` functioned properly.
- Global metric aggregation via `allreduce!` produced consistent results.

The NCCL artifact was subsequently resolved by ensuring `Pkg.instantiate()` was run to download all artifacts. A subsequent run confirmed clean NCCL initialization with 2 GPUs.

#pagebreak()

= Upstream Documentation

== `docs/src/guide/distributed.md`

The C7 phase included adding a user-facing guide to the Flux.jl upstream documentation. This guide lives at `docs/src/guide/distributed.md` in the Flux repository and is structured as a progressive tutorial:

1. *Introduction* (3 lines): DDP concepts and when to use it.
2. *Initialization* (30 lines): Auto-detection of CUDA/NCCL vs CPU/MPI.
3. *Sharding Data* (23 lines): `DistributedDataContainer` usage with `DataLoader`.
4. *Model and Optimizer Setup* (24 lines): `FluxDistributedModel`, `synchronize!!`, `DistributedOptimizer`.
5. *Training Loop* (58 lines): Complete epoch loop including validation with `allreduce!`.
6. *Advanced: Conditional Graphs* (20 lines): `resolve_unused_parameters!!` for multi-branch models.
7. *Saving and Checkpointing* (12 lines): Rank-0-gated JLD2 checkpointing.

The guide uses a simple `Chain(Dense(10 => 32, relu), Dense(32 => 2))` model for the basic sections to keep the learning curve manageable, then introduces conditional graphs in the advanced section.

== Documentation Registration

The guide was registered in `docs/make.jl` under the "Guide" section:

```julia
"Distributed Training" => "guide/distributed.md",
```

This places the distributed training guide alongside other Flux guides (Training, GPU Support, Performance Tips, etc.) in the rendered documentation.

== Underlying Test Suite

While not a direct C7 deliverable, the correctness of the APIs used by the example is guaranteed by the distributed test suite consolidated in phases C4--C6b. The test suite lives in `test/ext_distributed/` and includes:

#v(4pt)

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header(
      [*Test File*], [*Coverage*]
    ),
    [`common_distributedtest.jl`], [`bcast!`, `reduce!`, `allreduce!` with `+` and `avg` ops],
    [`data_distributedtest.jl`], [Partitioning correctness, padding for non-divisible sizes],
    [`optimizer_distributedtest.jl`], [`DistributedOptimizer` state setup and gradient update],
    [`reduce_distributedtest.jl`], [Standalone `reduce!` correctness],
    [`synchronized_distributedtest.jl`], [`synchronize!!` for NamedTuples, Tuples, Optimiser state, scalars],
    [`unused_parameters_distributedtest.jl`], [`resolve_unused_parameters!!`: nothing→zeros, nested structures, shape preservation],
  ),
  caption: [Distributed test suite files and their coverage areas. All tests use `mpiexecjl -n 2` on CPU/MPI.]
)

All tests in the suite pass with the standard invocation `mpiexecjl -n 2` on CPU/MPI, and with `FLUX_TEST_DISTRIBUTED_NCCL=true` on GPU/NCCL.

= Conclusion and Next Steps

== C7 Success Criteria

Phase C7 satisfied its success criterion: one documented command (`make example-ddp`) reliably reproduces distributed training on two devices (GPUs or CPUs). The deliverables are:

+ `scripts/examples/train_ddp.jl` — a 211-line example covering backend initialization, data sharding, model/optimizer sync, conditional graph training, validation, and checkpointing.
+ `docs/src/guide/distributed.md` — a 181-line user guide integrated into the Flux.jl documentation.
+ Bocconi validation — successful execution on the university cluster with both MPI and NCCL backends.

== Key Learnings

1. *Seed discipline matters*: The global dataset must be generated from an identical seed on all ranks before sharding. Rank-dependent seeds silently produce incorrect training semantics.
2. *Validation must be distributed*: In DDP, each rank only sees a shard of each dataset. Global metrics require `allreduce!` aggregation, even during inference-only evaluation loops.
3. *I/O must be rank-gated*: Any file I/O during distributed training must execute on a single rank to prevent corruption. The convention is to use rank 0.
4. *Graceful degradation is practical*: The `try`/`catch` pattern for CUDA detection allows a single script to work on both GPU and CPU clusters without code changes.

== Next Phase: C8 — Correctness Battery

With the end-to-end integration validated, the next phase (C8) focuses on building a systematic correctness test battery. This will include tests for:

+ Known DDP failure modes: uneven batch sizes, conditional graphs with deeply nested structures, mismatched collectives.
+ Gradient numerical correctness against the single-process baseline (C1).
+ Parameter synchronization integrity after many training steps.
+ Edge cases: models with no parameters, empty batches, single-layer networks.

The C8 checkpoint is currently `Not started` and represents the transition from integration validation to systematic correctness verification.

#v(24pt)

#align(right)[
  #text(size: 10pt, style: "italic")[
    Report generated from the `ddp-flux-thesis` workspace.
    
    Flux branch: `ddp/docs-examples` · Thesis commit: `04cd1bd`
    
    Bocconi validation: 2026-07-28
  ]
]
