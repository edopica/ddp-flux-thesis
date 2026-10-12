# ==============================================================================
# Conditional-graph DDP probe (child process, runs under mpiexec)
# ==============================================================================
#
# Exercises conditional branches through real AD and a distributed optimizer step.
# Every mode must finish with correct values on every rank.
#
# Modes (first positional argument, default "deadlock"):
#
#   deadlock    rank 0 uses (trunk, a, b); rank 1 uses (trunk, c)
#               -> one presence reduction, then trunk, a, b, c on both ranks.
#
#   corruption  rank 0 uses (trunk, a); rank 1 uses (trunk, b)
#               -> one presence reduction, then trunk, a, b on both ranks.
#                  The globally absent c remains unchanged.
#
#   control     every rank uses (trunk, a, b, c)
#               -> one presence reduction, then trunk, a, b, c on both ranks.
#
# The script prints machine-readable markers so the driver
# (`conditional_deadlock_check.jl`) can detect a hang without waiting for a
# fixed startup timeout:
#
#   PROBE_READY rank=<r>
#   TRACE rank=<r> allreduce #<n> kind=<k> dtype=<t> op=<op> input=<v> output=<v>
#   PROBE_UPDATE_DONE rank=<r>
#   PROBE_VERDICT mode=<m> rank=<r> correct=<bool> ...
#   PROBE_FINISHED rank=<r>
#
# Run directly with the selected Flux environment:
#   mpiexec -n 2 julia --project=../flux-integration/test scripts/checks/conditional_deadlock_probe.jl deadlock
# ==============================================================================

using MPI
using Flux
using Optimisers
using Functors

const MODE = isempty(ARGS) ? "deadlock" : ARGS[1]
const VALID_MODES = ("deadlock", "corruption", "control")

MODE in VALID_MODES || error("mode must be one of $(VALID_MODES); got $(repr(MODE))")

Flux.DistributedUtils.initialize(Flux.MPIBackend)
const backend = Flux.DistributedUtils.get_distributed_backend(Flux.MPIBackend)
const rank = Flux.DistributedUtils.local_rank(backend)
const world = Flux.DistributedUtils.total_workers(backend)

world == 2 || error("this probe needs exactly 2 ranks; got $(world)")

# Send each record and its newline in one write to the MPI output pipe.
function marker(message)
    write(stdout, message * "\n")
    flush(stdout)
end

# ------------------------------------------------------------------
# Model: four independent parameter vectors, so each rank can choose a
# different subset in the forward pass.
# ------------------------------------------------------------------

struct Branchy
    trunk::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
end

Functors.@functor Branchy

const TARGET_TRUNK = 1.0
const TARGET_A = 2.0
const TARGET_B = 5.0
const TARGET_C = 3.0
const ETA = 0.1

function branch_loss(mode, r, m)
    l = sum(abs2, m.trunk .- TARGET_TRUNK)
    if mode == "deadlock"
        l += r == 0 ? sum(abs2, m.a .- TARGET_A) + sum(abs2, m.b .- TARGET_B) :
                      sum(abs2, m.c .- TARGET_C)
    elseif mode == "corruption"
        l += r == 0 ? sum(abs2, m.a .- TARGET_A) : sum(abs2, m.b .- TARGET_B)
    else # control
        l += sum(abs2, m.a .- TARGET_A) +
             sum(abs2, m.b .- TARGET_B) +
             sum(abs2, m.c .- TARGET_C)
    end
    return l
end

# ------------------------------------------------------------------
# Backend wrapper that logs every collective before it happens. This is what
# makes the hang diagnosable: the last trace line names the collective that
# never matched.
# ------------------------------------------------------------------

struct LoggingBackend{B} <: Flux.AbstractFluxDistributedBackend
    inner::B
    rank::Int
    counter::Base.RefValue{Int}
end

function Flux.DistributedUtils.allreduce!(b::LoggingBackend, buf, op)
    b.counter[] += 1
    kind = eltype(buf) === Float32 ? "presence" : "gradient"
    opname = op === Flux.DistributedUtils.avg ? "avg" : "other"
    input = join(buf, ',')
    marker("PROBE_COLLECTIVE_BEGIN rank=$(b.rank) number=$(b.counter[]) kind=$(kind) input=$(input)")
    result = Flux.DistributedUtils.allreduce!(b.inner, buf, op)
    marker("TRACE rank=$(b.rank) allreduce #$(b.counter[]) kind=$(kind) dtype=$(eltype(buf)) op=$(opname) input=$(input) output=$(join(buf, ',')) END")
    return result
end

# ------------------------------------------------------------------
# One distributed optimizer step
# ------------------------------------------------------------------

model = Branchy(ones(4), ones(4), ones(4), ones(4))
log_backend = LoggingBackend(backend, rank, Ref(0))
state = Optimisers.setup(
    Flux.DistributedUtils.DistributedOptimizer(log_backend, Optimisers.Descent(ETA)), model)

_, gs = Flux.withgradient(m -> branch_loss(MODE, rank, m), model)

marker("PROBE_READY rank=$(rank)")

state, model = Optimisers.update!(state, model, gs[1])

marker("PROBE_UPDATE_DONE rank=$(rank)")

# ------------------------------------------------------------------
# Analytical values on every rank, including locally and globally absent branches.
# ------------------------------------------------------------------

weights = MODE == "control" ? (1.0, 1.0, 1.0, 1.0) :
          MODE == "deadlock" ? (1.0, 0.5, 0.5, 0.5) : (1.0, 0.5, 0.5, 0.0)
targets = (TARGET_TRUNK, TARGET_A, TARGET_B, TARGET_C)
expected = map((target, weight) -> 1.0 - ETA * 2 * (1.0 - target) * weight, targets, weights)
params = (model.trunk, model.a, model.b, model.c)
ok = all(all(isapprox(x, want; atol=1e-8, rtol=0) for x in p) for (p, want) in zip(params, expected))
marker("PROBE_VERDICT mode=$(MODE) rank=$(rank) correct=$(ok) " *
        "expected=$(join(expected, ',')) actual=$(join(first.(params), ','))")

# The final barrier also detects ranks that fail to finish the update.
MPI.Barrier(backend.comm)
marker("PROBE_FINISHED rank=$(rank)")

MPI.Finalize()
exit(ok ? 0 : 1)
