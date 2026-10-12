# Device-generic two-head conditional model for the Phase K demo.
#
# The shared trunk feeds exactly one head per local batch; the host-side branch
# on `head` means the unused head is absent from the local AD graph. Both heads
# keep equal shapes so silent cross-parameter gradient mixing stays detectable.
module CGModel

using Flux, Functors, Random

export TwoHeadModel, build_toy_model, MODEL_DEFINITION_VERSION

const MODEL_DEFINITION_VERSION = 1

struct TwoHeadModel{T,H1,H2}
    trunk::T
    head_a::H1
    head_b::H2
end

Functors.@functor TwoHeadModel

function (m::TwoHeadModel)(x, head::Symbol)
    head === :a && return m.head_a(m.trunk(x))
    head === :b && return m.head_b(m.trunk(x))
    error("unknown head $(repr(head)); expected :a or :b")
end

head_field(head::Symbol) = head === :a ? :head_a : head === :b ? :head_b :
                           error("unknown head $(repr(head))")

# `winit` is captured with the explicit RNG so the prepared artifact is
# reproducible in the preparing process; the artifact, not the seed, is what
# every rank loads.
function build_toy_model(seed::Integer; indim::Int=2, hidden::Int=16, nclasses::Int=2)
    rng = MersenneTwister(seed)
    winit = (out, in) -> Flux.glorot_uniform(rng, out, in)
    return TwoHeadModel(
        Dense(indim => hidden, tanh; init=winit),
        Dense(hidden => nclasses; init=winit),
        Dense(hidden => nclasses; init=winit),
    )
end

end # module
