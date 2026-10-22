"""
Matched Flux implementation of the frozen static-scaling model
(Tiny ImageNet ResNet-18, small-image stem, GroupNorm-32) plus the
single-device training path used by Phase 3 and later phases.

The file defines `module StaticScalingFlux`. Include it and use
`StaticScalingFlux.make_model(bundle)`; there is no second training
implementation in the check scripts.

Public API (all exported):

* `ResNet18Small`, `Stem`, `BasicBlock`, `Stage` - model types.
* `bundle_parameter_shapes() -> Vector{Pair{String,Tuple}}` - the 62
  frozen bundle names and canonical PyTorch shapes, in bundle order.
* `make_model(bundle::AbstractDict) -> ResNet18Small` - constructs the
  model with every parameter taken from `bundle`; framework default
  initializers are never used.
* `load_bundle!(model, bundle) -> model` - overwrites every parameter
  array in place from `bundle` (works on CPU and GPU models).
* `each_parameter(model) -> Vector{Pair{String,AbstractArray}}` - the 62
  trainable arrays in bundle order, in Flux layout.
* `param_by_name(model, name)` - Flux-layout array for one bundle name.
* `parameter_count(model) -> Int` - 11,271,432 for this architecture.
* `canonical_param(name, a)` - Flux layout -> canonical PyTorch layout
  for conv weights; identity for GroupNorm and classifier arrays.
* `build_optimizer_state(model; eta) -> Optimisers state` - AdamW with
  lambda=1e-4 on the 21 decay arrays and lambda=0 on the 41 no-decay
  arrays (40 GroupNorm gamma/beta + classifier bias).
* `state_leaves(state) -> Vector{Pair{String,Optimisers.Leaf}}` - leaves
  keyed by bundle names.
* `is_decay_param(name) -> Bool`.
* `schedule_eta(t; eta_max, W, T) -> Float64` - warm-up/cosine schedule.
* `forward_with_activations(model, x; capture=true)` - returns
  `(logits, acts)`; `acts` is `nothing` when `capture=false`, otherwise a
  named tuple with keys `stem_gn`, `stem`, `stageN_blockB`, `pooled`.
* `train_step!(model, opt_state, x, y; eta, capture_activations=false,
  apply_update=true)` - one Zygote forward/backward, capture, then
  `Optimisers.adjust!(state, eta)` and `Optimisers.update!`. Returns a
  named tuple `(model, opt_state, loss, logits, activations, grads)`.

Layouts: inputs are Flux WHCN (`permutedims(nchw, (4,3,2,1))`), conv
weights are Flux `(kw,kh,cin,cout)`, logits are `(200,N)`, and
`Flux.CrossCor` is used for every convolution. Bundle arrays stay in
canonical PyTorch layout; conversions happen only at the boundary.

GPU precision policy: before any GPU work (including moving a model or
array to the device), call
`StaticScalingPrecision.configure_precision!()` from `precision.jl`.
That single reusable initialization path enforces
`CUDA.allowscalar(false)`, cuBLAS `PEDANTIC_MATH`, and cuDNN
`CUDNN_FMA_MATH`, which is the accepted FP32/TF32-off policy. It is
process-wide and must be called once before GPU work. The model math is
unchanged. Acceptance checks, benchmarks, and later training runners
must reuse this path instead of setting CUDA math state locally.
"""
module StaticScalingFlux

using Flux
using Functors
using Optimisers
using Statistics
import Zygote

export ResNet18Small, Stem, BasicBlock, Stage,
       bundle_parameter_shapes, bundle_parameter_names, make_model, load_bundle!,
       each_parameter, param_by_name, parameter_count, canonical_param,
       build_optimizer_state, state_leaves, is_decay_param,
       schedule_eta, forward_with_activations, train_step!,
       gradient_for,
       GN_GROUPS, GN_EPS, PARAMETER_COUNT, ETA_MAX, WARMUP_UPDATES, TOTAL_UPDATES,
       WEIGHT_DECAY, BETA, EPSILON

const GN_GROUPS = 32
const GN_EPS = 1.0f-5
const PARAMETER_COUNT = 11_271_432
const ETA_MAX = 1.0e-3
const WARMUP_UPDATES = 975
const TOTAL_UPDATES = 9750
const WEIGHT_DECAY = 1.0e-4
const BETA = (0.9, 0.999)
const EPSILON = 1.0e-8

struct Stem{C,G}
    conv::C
    gn::G
end
Flux.@layer Stem trainable = (conv, gn)

struct BasicBlock{C1,G1,C2,G2,P,PG}
    conv1::C1
    gn1::G1
    conv2::C2
    gn2::G2
    proj::P
    proj_gn::PG
end
Flux.@layer BasicBlock trainable = (conv1, gn1, conv2, gn2, proj, proj_gn)

struct Stage{B0,B1}
    block0::B0
    block1::B1
end
Flux.@layer Stage trainable = (block0, block1)

struct ResNet18Small{S,T1,T2,T3,T4,C}
    stem::S
    stage1::T1
    stage2::T2
    stage3::T3
    stage4::T4
    classifier::C
end
Flux.@layer ResNet18Small trainable = (stem, stage1, stage2, stage3, stage4, classifier)

const _GN_SUFFIXES = (".gn.weight", ".gn.bias",
                      ".gn1.weight", ".gn1.bias",
                      ".gn2.weight", ".gn2.bias",
                      ".proj_gn.weight", ".proj_gn.bias")

_is_groupnorm_param(name::AbstractString) = any(s -> endswith(name, s), _GN_SUFFIXES)

_is_conv_weight(name::AbstractString) =
    endswith(name, ".conv.weight") ||
    endswith(name, ".conv1.weight") ||
    endswith(name, ".conv2.weight") ||
    endswith(name, ".proj.weight")

"""
    bundle_parameter_shapes() -> Vector{Pair{String,Tuple}}

The 62 frozen bundle names and canonical PyTorch shapes, in bundle order.
"""
function bundle_parameter_shapes()
    shapes = Pair{String,Tuple}[]
    push!(shapes, "stem.conv.weight" => (64, 3, 3, 3))
    push!(shapes, "stem.gn.weight" => (64,))
    push!(shapes, "stem.gn.bias" => (64,))
    widths = (64, 128, 256, 512)
    prev = 64
    for s in 1:4
        ch = widths[s]
        for b in 0:1
            cin = b == 0 ? prev : ch
            pre = "stage$(s).block$(b)"
            push!(shapes, "$(pre).conv1.weight" => (ch, cin, 3, 3))
            push!(shapes, "$(pre).conv2.weight" => (ch, ch, 3, 3))
            push!(shapes, "$(pre).gn1.weight" => (ch,))
            push!(shapes, "$(pre).gn1.bias" => (ch,))
            push!(shapes, "$(pre).gn2.weight" => (ch,))
            push!(shapes, "$(pre).gn2.bias" => (ch,))
        end
        if s >= 2
            push!(shapes, "stage$(s).block0.proj.weight" => (ch, ch ÷ 2, 1, 1))
            push!(shapes, "stage$(s).block0.proj_gn.weight" => (ch,))
            push!(shapes, "stage$(s).block0.proj_gn.bias" => (ch,))
        end
        prev = ch
    end
    push!(shapes, "classifier.weight" => (200, 512))
    push!(shapes, "classifier.bias" => (200,))
    return shapes
end

bundle_parameter_names() = first.(bundle_parameter_shapes())

function _check_bundle_names(bundle::AbstractDict)
    expected = bundle_parameter_names()
    actual = String.(collect(keys(bundle)))
    missing = setdiff(expected, actual)
    extra = setdiff(actual, expected)
    (isempty(missing) && isempty(extra)) ||
        error("bundle name-set mismatch: n=$(length(actual)) want=$(length(expected)) " *
              "missing=$(first(missing, 5)) extra=$(first(extra, 5))")
    return nothing
end

function _bundle_array(bundle::AbstractDict, name::AbstractString)
    haskey(bundle, name) || error("bundle is missing array $(name)")
    a = bundle[name]
    a isa AbstractArray || error("bundle entry $(name) is not an array: $(typeof(a))")
    eltype(a) === Float32 || error("bundle entry $(name) has eltype $(eltype(a)); expected Float32")
    return a
end

function _source_array(bundle::AbstractDict, name::AbstractString)
    a = _bundle_array(bundle, name)
    return _is_conv_weight(name) ? permutedims(a, (4, 3, 2, 1)) : copy(a)
end

function _flux_path(name::AbstractString)
    parts = String.(split(name, "."))
    if _is_groupnorm_param(name)
        parts[end] = parts[end] == "weight" ? "γ" : "β"
    end
    return parts
end

"""
    param_by_name(model, name) -> AbstractArray

Return the Flux-layout parameter array for a canonical bundle name.
"""
function param_by_name(model, name::AbstractString)
    node = model
    for part in _flux_path(name)
        node = getproperty(node, Symbol(part))
    end
    return node
end

"""
    each_parameter(model) -> Vector{Pair{String,AbstractArray}}

The 62 trainable arrays in bundle order, in Flux layout.
"""
each_parameter(model) = [name => param_by_name(model, name) for name in bundle_parameter_names()]

parameter_count(model) = sum(length(a) for (_, a) in each_parameter(model); init = 0)

"""
    canonical_param(name, a)

Convert a Flux-layout parameter array to canonical PyTorch layout.
Convolution weights are `(kw,kh,cin,cout)` in Flux and `(out,in,kh,kw)`
in the bundle; GroupNorm and classifier arrays are already canonical.
"""
function canonical_param(name::AbstractString, a)
    _is_conv_weight(name) || return a
    return permutedims(a, (4, 3, 2, 1))
end

function _make_gn(bundle::AbstractDict, prefix::AbstractString, ch::Int)
    w = _source_array(bundle, prefix * ".weight")
    b = _source_array(bundle, prefix * ".bias")
    return GroupNorm(ch, GN_GROUPS;
                     initγ = _ -> copy(w), initβ = _ -> copy(b),
                     affine = true, eps = GN_EPS)
end

function _make_block(bundle::AbstractDict, pre::AbstractString, cin::Int, cout::Int;
                     stride::Int, projection::Bool)
    w1 = _source_array(bundle, "$(pre).conv1.weight")
    size(w1) == (3, 3, cin, cout) ||
        error("$(pre).conv1.weight has Flux size $(size(w1)); expected $((3, 3, cin, cout))")
    w2 = _source_array(bundle, "$(pre).conv2.weight")
    size(w2) == (3, 3, cout, cout) ||
        error("$(pre).conv2.weight has Flux size $(size(w2)); expected $((3, 3, cout, cout))")

    conv1 = CrossCor(w1, false; stride = stride, pad = 1)
    gn1 = _make_gn(bundle, "$(pre).gn1", cout)
    conv2 = CrossCor(w2, false; stride = 1, pad = 1)
    gn2 = _make_gn(bundle, "$(pre).gn2", cout)

    if projection
        wp = _source_array(bundle, "$(pre).proj.weight")
        size(wp) == (1, 1, cin, cout) ||
            error("$(pre).proj.weight has Flux size $(size(wp)); expected $((1, 1, cin, cout))")
        proj = CrossCor(wp, false; stride = stride, pad = 0)
        proj_gn = _make_gn(bundle, "$(pre).proj_gn", cout)
    else
        proj = nothing
        proj_gn = nothing
    end

    return BasicBlock(conv1, gn1, conv2, gn2, proj, proj_gn)
end

function _make_stage(bundle::AbstractDict, s::Int, cin::Int, cout::Int)
    stride = s == 1 ? 1 : 2
    projection = s >= 2
    return Stage(
        _make_block(bundle, "stage$(s).block0", cin, cout;
                    stride = stride, projection = projection),
        _make_block(bundle, "stage$(s).block1", cout, cout;
                    stride = 1, projection = false),
    )
end

"""
    make_model(bundle::AbstractDict) -> ResNet18Small

Build the frozen architecture with every parameter loaded from `bundle`.
No framework default initializer remains in the model.
"""
function make_model(bundle::AbstractDict)
    _check_bundle_names(bundle)
    for (name, shape) in bundle_parameter_shapes()
        a = _bundle_array(bundle, name)
        size(a) == shape ||
            error("bundle array $(name) has size $(size(a)); expected $(shape)")
    end

    stem = Stem(
        CrossCor(_source_array(bundle, "stem.conv.weight"), false; stride = 1, pad = 1),
        _make_gn(bundle, "stem.gn", 64),
    )

    stage1 = _make_stage(bundle, 1, 64, 64)
    stage2 = _make_stage(bundle, 2, 64, 128)
    stage3 = _make_stage(bundle, 3, 128, 256)
    stage4 = _make_stage(bundle, 4, 256, 512)

    classifier = Dense(
        _source_array(bundle, "classifier.weight"),
        _source_array(bundle, "classifier.bias"),
        identity,
    )

    model = ResNet18Small(stem, stage1, stage2, stage3, stage4, classifier)
    length(each_parameter(model)) == length(bundle_parameter_shapes()) ||
        error("model exposes $(length(each_parameter(model))) arrays; expected 62")
    parameter_count(model) == PARAMETER_COUNT ||
        error("model has $(parameter_count(model)) parameters; expected $(PARAMETER_COUNT)")
    return model
end

"""
    load_bundle!(model, bundle) -> model

Overwrite every parameter array from `bundle`, using the same layout
conversions as `make_model`. Returns `model`.
"""
function load_bundle!(model::ResNet18Small, bundle::AbstractDict)
    _check_bundle_names(bundle)
    for (name, dest) in each_parameter(model)
        src = _source_array(bundle, name)
        size(src) == size(dest) ||
            error("bundle array $(name) has size $(size(src)); model parameter has size $(size(dest))")
        copyto!(dest, src)
    end
    return model
end

function _global_mean(h)
    return dropdims(mean(h; dims = (1, 2)), dims = (1, 2))
end

function (block::BasicBlock)(x)
    y = relu.(block.gn1(block.conv1(x)))
    y = block.gn2(block.conv2(y))
    if block.proj === nothing
        y = y .+ x
    else
        y = y .+ block.proj_gn(block.proj(x))
    end
    return relu.(y)
end

"""
    forward_with_activations(model, x; capture=true) -> (logits, acts)

Run one forward pass. With `capture=true`, `acts` is a named tuple with
keys `stem_gn`, `stem`, `stage1_block0`, ..., `stage4_block1`, `pooled`;
with `capture=false` it is `nothing`. Activations are in Flux layout
(WHCN for conv outputs, `(C,N)` for `pooled`). Logits are `(200,N)`.
"""
function forward_with_activations(model::ResNet18Small, x; capture::Bool = true)
    stem_gn = model.stem.gn(model.stem.conv(x))
    stem = relu.(stem_gn)

    h1a = model.stage1.block0(stem)
    h1b = model.stage1.block1(h1a)
    h2a = model.stage2.block0(h1b)
    h2b = model.stage2.block1(h2a)
    h3a = model.stage3.block0(h2b)
    h3b = model.stage3.block1(h3a)
    h4a = model.stage4.block0(h3b)
    h4b = model.stage4.block1(h4a)

    pooled = _global_mean(h4b)
    logits = model.classifier(pooled)

    acts = capture ? (
        stem_gn = stem_gn,
        stem = stem,
        stage1_block0 = h1a,
        stage1_block1 = h1b,
        stage2_block0 = h2a,
        stage2_block1 = h2b,
        stage3_block0 = h3a,
        stage3_block1 = h3b,
        stage4_block0 = h4a,
        stage4_block1 = h4b,
        pooled = pooled,
    ) : nothing

    return logits, acts
end

(model::ResNet18Small)(x) = first(forward_with_activations(model, x; capture = false))

"""
    is_decay_param(name) -> Bool

The 21 decay arrays are the 20 convolution weights and
`classifier.weight`. The 41 no-decay arrays are the 20 GroupNorm gamma,
20 GroupNorm beta, and `classifier.bias`.
"""
function is_decay_param(name::AbstractString)
    name == "classifier.weight" && return true
    endswith(name, ".weight") || return false
    return !_is_groupnorm_param(name)
end

"""
    build_optimizer_state(model; eta=ETA_MAX)

Create `Optimisers.setup(AdamW(...), model)` and set the per-leaf
`lambda` explicitly: 1e-4 for the 21 decay arrays, 0 for the 41
no-decay arrays. Moments and beta powers stay at their initial values.
"""
function build_optimizer_state(model; eta::Real = ETA_MAX)
    rule = Optimisers.AdamW(; eta = eta, beta = BETA,
                            lambda = WEIGHT_DECAY, epsilon = EPSILON,
                            couple = true)
    state = Optimisers.setup(rule, model)
    leaves = state_leaves(state)
    length(leaves) == length(bundle_parameter_shapes()) ||
        error("optimizer state has $(length(leaves)) leaves; expected 62")

    want_beta_power = (Float32(BETA[1]), Float32(BETA[2]))
    ndecay = 0
    for (name, leaf) in leaves
        leaf.rule isa Optimisers.AdamW ||
            error("optimizer leaf $(name) uses $(typeof(leaf.rule)); expected AdamW")
        want_lambda = is_decay_param(name) ? WEIGHT_DECAY : 0.0
        Optimisers.adjust!(leaf; lambda = want_lambda)
        leaf.rule.lambda == want_lambda ||
            error("optimizer leaf $(name) has lambda=$(leaf.rule.lambda); expected $(want_lambda)")
        leaf.rule.eta == float(eta) ||
            error("optimizer leaf $(name) has eta=$(leaf.rule.eta); expected $(float(eta))")
        leaf.rule.beta == BETA ||
            error("optimizer leaf $(name) has beta=$(leaf.rule.beta); expected $(BETA)")
        leaf.rule.epsilon == EPSILON ||
            error("optimizer leaf $(name) has epsilon=$(leaf.rule.epsilon); expected $(EPSILON)")
        leaf.rule.couple === true ||
            error("optimizer leaf $(name) has couple=$(leaf.rule.couple); expected true")
        m, v, beta_power = leaf.state
        all(iszero, m) ||
            error("optimizer leaf $(name) has nonzero initial m")
        all(iszero, v) ||
            error("optimizer leaf $(name) has nonzero initial v")
        beta_power == want_beta_power ||
            error("optimizer leaf $(name) has beta_power=$(beta_power); expected $(want_beta_power)")
        ndecay += want_lambda != 0.0
    end
    ndecay == 21 ||
        error("optimizer has $(ndecay) decay leaves; expected 21 (41 no-decay)")
    return state
end

function _collect_state_leaves!(out, path::Vector{String}, node)
    if node isa Optimisers.Leaf
        push!(out, join(path, ".") => node)
        return out
    end
    children = Functors.functor(node)[1]
    if children isa NamedTuple
        for k in keys(children)
            _collect_state_leaves!(out, vcat(path, String(k)), getproperty(children, k))
        end
    elseif children isa Tuple
        for (i, child) in enumerate(children)
            _collect_state_leaves!(out, vcat(path, string(i)), child)
        end
    elseif children isa AbstractDict
        for k in sort!(collect(keys(children)))
            _collect_state_leaves!(out, vcat(path, string(k)), children[k])
        end
    end
    return out
end

function _leaf_bundle_name(raw::AbstractString)
    parts = String.(split(raw, "."))
    if length(parts) >= 2 && parts[end] in ("γ", "β") &&
       parts[end - 1] in ("gn", "gn1", "gn2", "proj_gn")
        parts[end] = parts[end] == "γ" ? "weight" : "bias"
    end
    return join(parts, ".")
end

"""
    state_leaves(state) -> Vector{Pair{String,Optimisers.Leaf}}

Optimizer leaves keyed by canonical bundle names. The order follows the
state tree; use the names, not the order, for comparisons.
"""
function state_leaves(state)
    raw = Pair{String,Any}[]
    _collect_state_leaves!(raw, String[], state)
    return [(_leaf_bundle_name(name) => leaf) for (name, leaf) in raw]
end

"""
    schedule_eta(t; eta_max=ETA_MAX, W=WARMUP_UPDATES, T=TOTAL_UPDATES)

Global update schedule. `t=1..W` is linear warm-up; `t=W+1..T` is
cosine decay to zero. Returns a Float64.
"""
function schedule_eta(t::Integer; eta_max::Real = ETA_MAX,
                      W::Integer = WARMUP_UPDATES, T::Integer = TOTAL_UPDATES)
    1 <= t <= T || throw(ArgumentError("schedule index t=$(t) is outside 1:$(T)"))
    t <= W && return eta_max * t / W
    return eta_max * 0.5 * (1 + cos(pi * (t - W) / (T - W)))
end

"""
    gradient_for(grads, name)

Look up the gradient array for a canonical bundle name in a Zygote
gradient tree (`grads` is the first element returned by `withgradient`).
GroupNorm names map to the `γ` and `β` fields.
"""
function gradient_for(grads, name::AbstractString)
    node = grads
    for part in _flux_path(name)
        node = getproperty(node, Symbol(part))
    end
    return node
end

"""
    train_step!(model, opt_state, x, y; eta, capture_activations=false,
                apply_update=true) -> NamedTuple

One update: Zygote forward/backward of the mean cross-entropy loss,
then (when `apply_update=true`) `Optimisers.adjust!(opt_state, eta)` and
`Optimisers.update!(opt_state, model, grads)`. The gradient and logits
correspond to the parameters before the update. Returns
`(model, opt_state, loss, logits, activations, grads)`.

On the GPU path, call `StaticScalingPrecision.configure_precision!()`
before any GPU work; do not set CUDA math state locally in a runner.
"""
function train_step!(model::ResNet18Small, opt_state, x, y;
                     eta::Real, capture_activations::Bool = false,
                     apply_update::Bool = true)
    val, grads = Zygote.withgradient(model) do m
        logits, acts = forward_with_activations(m, x; capture = capture_activations)
        loss = Flux.Losses.logitcrossentropy(logits, y)
        (loss = loss, logits = logits, acts = acts)
    end

    g = grads[1]
    if apply_update
        Optimisers.adjust!(opt_state, eta)
        opt_state, model = Optimisers.update!(opt_state, model, g)
    end

    return (model = model, opt_state = opt_state, loss = val.loss,
            logits = val.logits, activations = val.acts, grads = g)
end

end # module StaticScalingFlux
