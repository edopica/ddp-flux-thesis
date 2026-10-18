"""
Phase 3 Flux acceptance check for the static-scaling experiment.

Loads the frozen Phase 2 bundle into the matched Flux model and compares
three single-device updates against the PyTorch reference artifact:

* every intermediate activation, logits, and loss;
* all 62 gradients after u1 and u3;
* all 62 parameters, AdamW m, and AdamW v after u3;
* the reference step counter and the Flux beta-power state;
* schedule eta values at t = 1, 2, 3 (rtol 1e-12);
* bundle/model/optimizer contract: 62 arrays, 11,271,432 parameters,
  exact bundle load, 21 decay / 41 no-decay leaves with lambda 1e-4 / 0,
  and zero initial moments.

Float comparison rule (frozen specification section 11, OD-10 recorded
2026-10-17):
`|a - b| <= atol + rtol * max(|a|, |b|)`, base defaults atol = 1e-5,
rtol = 1e-4. Three backward categories use frozen category-specific
absolute floors with the same rtol = 1e-4: `phase3.grads.u1`
atol = 1e-3, `phase3.grads.u3` atol = 2e-3, `phase3.m.u3`
atol = 2e-4. Every other category keeps the base rule. Prints one
`CHECK <category> PASS|FAIL ...` line per category, one
`PHASE3_MAXABS <category> <value>` line for tolerance calibration, a
`PHASE3_ACCEPTANCE_RULE` line with every applied rule, and a final
`PHASE3_FLUX_CHECK PASS|FAIL` line; exits 0 only on PASS. Per-leaf
`PHASE3_LEAF_L2*` and `PHASE3_PARAM_INCREMENT_SUMMARY` lines are
supplementary diagnostics and do not change the acceptance rule.

The model and training path come from `model_flux.jl`; this script only
drives training and compares arrays. On `--device cuda`, the precision
policy comes from `precision.jl`: `configure_precision!(fma=true)` sets
`CUDA.allowscalar(false)`, cuBLAS `PEDANTIC_MATH`, and cuDNN
`CUDNN_FMA_MATH` before any GPU work, clears the cuDNN algorithm perf
caches, and the module-load algorithm chooser enforces
`ps.mathType == CUDNN_FMA_MATH`. After training, every recorded cuDNN
algorithm choice is printed as `PHASE3_CUDNN_ALGO`, summarized in
`PHASE3_CUDNN_ALGO_SUMMARY`, and accepted by
`CHECK cudnn.fma_algo_selection`. That check passes only when every
choice reports `CUDNN_FMA_MATH` and is not a fallback; a fallback is
tolerated only for a successful non-tensor-core algo 0/1 and only when
all acceptance categories pass. A TF32/tensor-op candidate never passes.

CLI:
  julia --startup-file=no scripts/static_scaling/check_phase3_flux.jl \\
    [--fixture-dir DIR] [--bundle FILE] [--reference FILE] \\
    [--manifest FILE] [--device cpu|cuda] [--atol FLOAT] [--rtol FLOAT] \\
    [--grads-u1-atol FLOAT] [--grads-u3-atol FLOAT] [--m-u3-atol FLOAT]

Defaults: the frozen Phase 2 fixture directory, its initial_bundle.npz,
`artifacts/baselines/static-scaling/2026-10-17-phase3/phase3_reference.npz`
and the matching `phase3_reference_manifest.json`; device cpu. The base
comparison defaults are atol = 1e-5 and rtol = 1e-4. The frozen OD-10
category floors default to `--grads-u1-atol 1e-3`,
`--grads-u3-atol 2e-3`, and `--m-u3-atol 2e-4`; all three use rtol = 1e-4.

Reproduction (Leonardo compute node; never on the login node):
  JULIA_LOAD_PATH="<THESIS>/scripts/static_scaling/julia-env:<FLUX>/test:@stdlib" \\
    julia --startup-file=no <THESIS>/scripts/static_scaling/check_phase3_flux.jl \\
    --fixture-dir <ARTIFACTS>/2026-10-16-phase2 \\
    --reference <ARTIFACTS>/2026-10-17-phase3/phase3_reference.npz \\
    --manifest <ARTIFACTS>/2026-10-17-phase3/phase3_reference_manifest.json \\
    --device cuda

The GPU-to-host copies below are explicit acceptance diagnostics. They
are outside the training path; the training module itself never copies
parameters or gradients to the host.
"""
;
include(joinpath(@__DIR__, "replay.jl"))
include(joinpath(@__DIR__, "model_flux.jl"))
include(joinpath(@__DIR__, "precision.jl"))
using .StaticScalingFlux
using .StaticScalingPrecision
using Flux
using Optimisers
using SHA
using Printf
using CUDA
using cuDNN

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_FIXTURE_DIR =
    joinpath(REPO_ROOT, "artifacts", "baselines", "static-scaling", "2026-10-16-phase2")
const DEFAULT_REFERENCE =
    joinpath(REPO_ROOT, "artifacts", "baselines", "static-scaling", "2026-10-17-phase3",
             "phase3_reference.npz")

const ALLOWED_OPTS = Set(["fixture-dir", "bundle", "reference", "manifest",
                          "device", "atol", "rtol",
                          "grads-u1-atol", "grads-u3-atol", "m-u3-atol"])

const REF_ACTIVATIONS = [
    ("act.stem.gn", (12, 64, 64, 64)),
    ("act.stem", (12, 64, 64, 64)),
    ("act.stage1.block0", (12, 64, 64, 64)),
    ("act.stage1.block1", (12, 64, 64, 64)),
    ("act.stage2.block0", (12, 128, 32, 32)),
    ("act.stage2.block1", (12, 128, 32, 32)),
    ("act.stage3.block0", (12, 256, 16, 16)),
    ("act.stage3.block1", (12, 256, 16, 16)),
    ("act.stage4.block0", (12, 512, 8, 8)),
    ("act.stage4.block1", (12, 512, 8, 8)),
    ("act.pooled", (12, 512)),
]

function expected_reference_names()
    names = String[]
    for t in 1:3
        for prefix in ("input", "label", "eta", "loss", "logits")
            push!(names, "$(prefix).u$(t)")
        end
    end
    append!(names, first.(REF_ACTIVATIONS))
    for prefix in ("grad.u1", "grad.u3", "param.u3", "m.u3", "v.u3")
        for name in bundle_parameter_names()
            push!(names, "$(prefix).$(name)")
        end
    end
    push!(names, "step.u3")
    return names
end

as_array(x::AbstractArray) = x
as_array(x::Number) = reshape([x], ())

function scalar_value(x)
    x isa Number && return x
    a = x isa Array ? x : Array(x)
    length(a) == 1 || error("expected a scalar or a one-element array, got size $(size(a))")
    return a[]
end

host_array(a::Array) = copy(a)
host_array(a::AbstractArray) = Array(a)

shortlist(names, maxn = 5) = isempty(names) ? "none" :
    (length(names) <= maxn ? join(names, ";") :
     join(names[1:maxn], ";") * ";+$(length(names) - maxn)more")

function general_c_order_bytes(a::AbstractArray)
    isbitstype(eltype(a)) || error("general_c_order_bytes: unsupported eltype $(eltype(a))")
    b = ndims(a) <= 1 ? Array(a) : permutedims(a, reverse(1:ndims(a)))
    return Vector{UInt8}(reinterpret(UInt8, vec(b)))
end

general_c_order_sha256(a::AbstractArray) = bytes2hex(sha256(general_c_order_bytes(a)))

function parse_float_opt(opts, key, default)
    s = get(opts, key, "")
    isempty(s) && return default
    v = tryparse(Float64, s)
    v === nothing && error("--$(key) must be a floating-point number, got $(s)")
    return v
end

rule_text(atol, rtol) =
    "|a - b| <= $(string(Float64(atol))) + $(string(Float64(rtol))) * max(|a|, |b|)"

activation_key(refname::AbstractString) =
    Symbol(replace(replace(refname, "act." => ""), "." => "_"))

function activation_to_canonical(key::Symbol, a)
    key === :pooled && return permutedims(a, (2, 1))
    return permutedims(a, (4, 3, 2, 1))
end

function check_specs(log::CheckLog, fields, label::AbstractString, specs)
    missing = String[]
    invalid = String[]
    for (name, T, shape) in specs
        if !haskey(fields, name)
            push!(missing, name)
            continue
        end
        a = as_array(fields[name])
        shape_ok = shape === :scalar ? length(a) == 1 : size(a) == shape
        (eltype(a) === T && shape_ok) ||
            push!(invalid, "$(name):$(eltype(a))$(size(a))")
    end
    ok = isempty(missing) && isempty(invalid)
    log_check(log, label, ok,
              "n=$(length(specs)) missing=$(shortlist(missing)) invalid=$(shortlist(invalid))")
    return ok
end

function check_reference_contract(log::CheckLog, ref)
    input_specs = Tuple{String,Type,Any}[]
    eta_specs = Tuple{String,Type,Any}[]
    loss_logit_specs = Tuple{String,Type,Any}[]
    for t in 1:3
        push!(input_specs, ("input.u$(t)", Float32, (12, 3, 64, 64)))
        push!(input_specs, ("label.u$(t)", Int64, (12,)))
        push!(eta_specs, ("eta.u$(t)", Float64, :scalar))
        push!(loss_logit_specs, ("loss.u$(t)", Float32, :scalar))
        push!(loss_logit_specs, ("logits.u$(t)", Float32, (12, 200)))
    end
    act_specs = [(name, Float32, shape) for (name, shape) in REF_ACTIVATIONS]

    shape_map = Dict(bundle_parameter_shapes())
    names = first.(bundle_parameter_shapes())
    specs_for(prefix) =
        [("$(prefix).$(name)", Float32, shape_map[name]) for name in names]

    ok = true
    ok &= check_specs(log, ref, "reference.inputs_labels", input_specs)
    ok &= check_specs(log, ref, "reference.eta", eta_specs)
    ok &= check_specs(log, ref, "reference.loss_logits", loss_logit_specs)
    ok &= check_specs(log, ref, "reference.activations", act_specs)
    ok &= check_specs(log, ref, "reference.grad_u1", specs_for("grad.u1"))
    ok &= check_specs(log, ref, "reference.grad_u3", specs_for("grad.u3"))
    ok &= check_specs(log, ref, "reference.param_u3", specs_for("param.u3"))
    ok &= check_specs(log, ref, "reference.m_u3", specs_for("m.u3"))
    ok &= check_specs(log, ref, "reference.v_u3", specs_for("v.u3"))
    ok &= check_specs(log, ref, "reference.step", [("step.u3", Int64, :scalar)])

    expected_names = expected_reference_names()
    actual_names = collect(keys(ref))
    missing = setdiff(expected_names, actual_names)
    extra = setdiff(actual_names, expected_names)
    ok &= log_check(log, "reference.array_set",
                    length(expected_names) == 337 && length(actual_names) == 337 &&
                    isempty(missing) && isempty(extra),
                    "n=$(length(actual_names)) want=337 " *
                    "missing=$(shortlist(missing)) extra=$(shortlist(extra))")
    return ok
end

function check_protocol(log::CheckLog, fixture_dir::String, ref)
    data_path = joinpath(fixture_dir, "data_fixture.npz")
    if !isfile(data_path)
        log_check(log, "protocol.data_fixture", false, "not found: $(data_path)")
        return false
    end
    df = read_npz_fields(data_path)
    if !all(haskey(df, k) for k in ("train_transformed_ref", "train_labels"))
        log_check(log, "protocol.data_fixture", false, "train_transformed_ref or train_labels missing")
        return false
    end
    tr = df["train_transformed_ref"]
    labels = df["train_labels"]
    ok = true
    for t in 1:3
        row = t == 2 ? 2 : 1
        exp_x = tr[row, :, :, :, :]
        key_x = "input.u$(t)"
        key_y = "label.u$(t)"
        if !haskey(ref, key_x) || !haskey(ref, key_y)
            log_check(log, "protocol.u$(t)", false, "reference input or label missing")
            ok = false
            continue
        end
        x = as_array(ref[key_x])
        y = as_array(ref[key_y])
        ok_x = size(x) == size(exp_x) && all(x .== exp_x)
        ok_y = size(y) == size(labels) && all(y .== labels)
        log_check(log, "protocol.u$(t)", ok_x && ok_y,
                  "input_matches_fixture=$(ok_x) labels_match_fixture=$(ok_y) row=$(row)")
        ok &= ok_x && ok_y
    end
    if all(haskey(ref, k) for k in ("input.u1", "input.u3", "label.u1", "label.u3"))
        same_x = all(as_array(ref["input.u3"]) .== as_array(ref["input.u1"]))
        same_y = all(as_array(ref["label.u3"]) .== as_array(ref["label.u1"]))
        log_check(log, "protocol.u3_reuses_u1", same_x && same_y,
                  "input_equal=$(same_x) labels_equal=$(same_y)")
        ok &= same_x && same_y
    else
        log_check(log, "protocol.u3_reuses_u1", false, "input.u1/u3 or label.u1/u3 missing")
        ok = false
    end
    return ok
end

function check_model_and_bundle(log::CheckLog, bundle)
    expected = bundle_parameter_shapes()
    names = first.(expected)
    shape_map = Dict(expected)
    bundle_names = collect(keys(bundle))
    missing = setdiff(names, bundle_names)
    extra = setdiff(bundle_names, names)
    log_check(log, "model.bundle_names",
              length(bundle) == 62 && isempty(missing) && isempty(extra),
              "n=$(length(bundle)) missing=$(shortlist(missing)) extra=$(shortlist(extra))")

    bad_shape = String[]
    bad_dtype = String[]
    for name in names
        haskey(bundle, name) || continue
        a = bundle[name]
        size(a) == shape_map[name] || push!(bad_shape, name)
        eltype(a) === Float32 || push!(bad_dtype, name)
    end
    log_check(log, "model.bundle_shapes", isempty(bad_shape),
              "n_checked=$(length(names)) mismatches=$(shortlist(bad_shape))")
    log_check(log, "model.bundle_dtypes", isempty(bad_dtype),
              "n_checked=$(length(names)) non_float32=$(shortlist(bad_dtype))")

    model = make_model(bundle)
    params = each_parameter(model)
    log_check(log, "model.array_count", length(params) == 62,
              "n=$(length(params)) want=62")
    total = parameter_count(model)
    log_check(log, "model.parameter_count", total == 11271432,
              "n=$(total) want=11271432")

    bad_load = String[]
    for (name, a) in params
        isequal(canonical_param(name, a), bundle[name]) || push!(bad_load, name)
    end
    log_check(log, "model.load_exact", isempty(bad_load),
              "n_checked=$(length(params)) mismatches=$(shortlist(bad_load))")

    load_bundle!(model, bundle)
    bad_reload = String[]
    for (name, a) in each_parameter(model)
        isequal(canonical_param(name, a), bundle[name]) || push!(bad_reload, name)
    end
    log_check(log, "model.load_bundle_reload", isempty(bad_reload),
              "n_checked=$(length(params)) mismatches=$(shortlist(bad_reload))")
    return model
end

function check_optimizer_contract(log::CheckLog, state)
    leaves = state_leaves(state)
    names = first.(leaves)
    expected = bundle_parameter_names()
    missing = setdiff(expected, names)
    extra = setdiff(names, expected)
    log_check(log, "optimizer.leaf_names",
              length(leaves) == 62 && isempty(missing) && isempty(extra),
              "n=$(length(leaves)) missing=$(shortlist(missing)) extra=$(shortlist(extra))")

    decay = [n for n in names if is_decay_param(n)]
    nodecay = [n for n in names if !is_decay_param(n)]
    log_check(log, "optimizer.group_counts",
              length(decay) == 21 && length(nodecay) == 41,
              "decay=$(length(decay)) no_decay=$(length(nodecay))")

    bad_rule = String[]
    bad_lambda = String[]
    for (name, leaf) in leaves
        r = leaf.rule
        if !(r isa Optimisers.AdamW)
            push!(bad_rule, "$(name):$(typeof(r))")
            continue
        end
        want = is_decay_param(name) ? WEIGHT_DECAY : 0.0
        r.lambda == want || push!(bad_lambda, "$(name):$(r.lambda)")
    end
    log_check(log, "optimizer.rule_type", isempty(bad_rule),
              "mismatches=$(shortlist(bad_rule))")
    log_check(log, "optimizer.lambda", isempty(bad_lambda),
              "n_checked=$(length(leaves)) mismatches=$(shortlist(bad_lambda))")

    bad_m = String[]
    bad_v = String[]
    bad_beta = String[]
    want_beta = (0.9f0, 0.999f0)
    for (name, leaf) in leaves
        m, v, bp = leaf.state
        all(iszero, m) || push!(bad_m, name)
        all(iszero, v) || push!(bad_v, name)
        bp == want_beta || push!(bad_beta, name)
    end
    log_check(log, "optimizer.initial_m", isempty(bad_m),
              "n_checked=$(length(leaves)) mismatches=$(shortlist(bad_m))")
    log_check(log, "optimizer.initial_v", isempty(bad_v),
              "n_checked=$(length(leaves)) mismatches=$(shortlist(bad_v))")
    log_check(log, "optimizer.initial_beta_power", isempty(bad_beta),
              "n_checked=$(length(leaves)) mismatches=$(shortlist(bad_beta))")

    if isempty(leaves) || !(first(leaves).second.rule isa Optimisers.AdamW)
        log_check(log, "optimizer.rule_config", false, "no AdamW leaves")
    else
        r = first(leaves).second.rule
        ok = r.beta == BETA && r.epsilon == EPSILON && r.couple === true
        log_check(log, "optimizer.rule_config", ok,
                  "beta=$(r.beta) epsilon=$(r.epsilon) couple=$(r.couple)")
    end
    return leaves
end

function configure_cuda!(log::CheckLog)
    cfg = configure_precision!(fma = true)
    blocked = try
        _ = CUDA.zeros(Float32, 1)[1]
        false
    catch err
        occursin("scalar", lowercase(sprint(showerror, err)))
    end
    log_check(log, "cuda.allowscalar_false", blocked,
              "CUDA.allowscalar(false); probe scalar index rejected=$(blocked)")
    log_check(log, "cuda.pedantic_math", CUDA.math_mode() === CUDA.PEDANTIC_MATH,
              "CUDA.math_mode()=$(CUDA.math_mode())")
    log_check(log, "cudnn.fma_math",
              cfg.cudnn_requested == cuDNN.CUDNN_FMA_MATH &&
              cfg.cudnn_effective == cuDNN.CUDNN_FMA_MATH,
              "requested=$(math_type_label(cfg.cudnn_requested)) " *
              "effective=$(math_type_label(cfg.cudnn_effective)) " *
              "cudnn_version=$(cfg.cudnn_version)")
    log_check(log, "cudnn.algo_caches_cleared", cfg.caches_cleared,
              "before=$(cfg.caches_before) after=(fwd=0,bwd_data=0,bwd_filter=0)")
    dev = Flux.gpu_device(; force = true)
    log_check(log, "cuda.device_selected", dev isa Flux.CUDADevice,
              "device=$(dev) (cuDNN loaded=$(isdefined(@__MODULE__, :cuDNN)))")
    println("PHASE3_FLUX_CHECK_CONFIG cuda_math_mode=$(CUDA.math_mode()) " *
            "cuda_math_precision=$(CUDA.math_precision()) " *
            "cudnn_requested=$(math_type_label(cfg.cudnn_requested)) " *
            "cudnn_effective=$(math_type_label(cfg.cudnn_effective)) " *
            "cudnn_version=$(cfg.cudnn_version)")
    return dev
end

function cudnn_algo_choice_line(choice)
    shape = choice.shape === nothing ? "NA" :
            join(string.(choice.shape), "x")
    return @sprintf("kind=%s algo=%d math_type=%s fallback=%s time=%.9g memory=%d shape=%s",
                    choice.kind, choice.algo, choice.math_type_label,
                    choice.fallback, choice.time, choice.memory, shape)
end

"""
    check_cudnn_algo_selection(log, device)

Print every recorded cuDNN convolution algorithm choice and enforce the
accepted FMA policy. A choice passes when it reports `CUDNN_FMA_MATH` and
is not a fallback. A fallback passes only when its algorithm is a
successful non-tensor-core algo 0/1 (`math_unknown=false`) **and** every
acceptance category has already passed. Tensor-op/TF32 candidates never
pass. On a CPU run there are no cuDNN choices, so the line is SKIP.
"""
function check_cudnn_algo_selection(log::CheckLog, device::AbstractString)
    choices = precision_algo_choices()
    enabled = device == "cuda"
    for choice in choices
        println("PHASE3_CUDNN_ALGO ", cudnn_algo_choice_line(choice))
    end
    n = length(choices)
    nfma = count(c -> c.math_type == Int(cuDNN.CUDNN_FMA_MATH), choices)
    nfallback = count(c -> c.fallback, choices)
    println("PHASE3_CUDNN_ALGO_SUMMARY choices=$(n) fma=$(nfma) " *
            "fallback=$(nfallback) enabled=$(enabled)")
    if !enabled
        println("CHECK cudnn.fma_algo_selection SKIP device=$(device) choices=$(n)")
        return true
    end
    categories_ok = isempty(log.failed)
    all_fma = n > 0 && nfma == n
    fallback_ok = all(choices) do choice
        !choice.fallback ||
            (Int(choice.algo) in (0, 1) && !choice.math_unknown)
    end
    ok = all_fma && fallback_ok && (nfallback == 0 || categories_ok)
    log_check(log, "cudnn.fma_algo_selection", ok,
              "choices=$(n) fma=$(nfma) fallback=$(nfallback) " *
              "fallback_algo_0_1=$(fallback_ok) categories_pass=$(categories_ok) " *
              "rule=\"all choices math_type=CUDNN_FMA_MATH and fallback=false; " *
              "fallback allowed only for successful non-tensor-core algo 0/1 " *
              "when all acceptance categories pass; tensor-op candidates never pass\"")
    return ok
end

function check_schedule(log::CheckLog, ref)
    pairs = Tuple{String,Any,Any}[]
    for t in 1:3
        key = "eta.u$(t)"
        if !haskey(ref, key)
            log_check(log, "phase3.eta", false, "$(key) missing")
            return false
        end
        push!(pairs, (key, [schedule_eta(t)], [Float64(scalar_value(ref[key]))]))
    end
    return compare_category(log, "phase3.eta", pairs; atol = 0.0, rtol = 1.0e-12)
end

function compare_category(log::CheckLog, category::AbstractString, pairs;
                          atol::Float64, rtol::Float64)
    max_abs = 0.0
    max_rel = 0.0
    worst = "-"
    worst_ratio = -1.0
    nviolations = 0
    bad = String[]
    for (name, actual, expected) in pairs
        if actual === nothing || expected === nothing
            push!(bad, "$(name):missing")
            max_abs = Inf
            max_rel = Inf
            worst = String(name)
            continue
        end
        if size(actual) != size(expected)
            push!(bad, "$(name):shape$(size(actual))vs$(size(expected))")
            max_abs = Inf
            max_rel = Inf
            worst = String(name)
            continue
        end
        array_violations = 0
        for (x, y) in zip(actual, expected)
            xf = Float64(x)
            yf = Float64(y)
            if !isfinite(xf) || !isfinite(yf)
                max_abs = Inf
                max_rel = Inf
                worst = String(name)
                worst_ratio = Inf
                nviolations += 1
                array_violations += 1
                continue
            end
            d = abs(xf - yf)
            m = max(abs(xf), abs(yf))
            r = m == 0.0 ? (d == 0.0 ? 0.0 : Inf) : d / m
            tol = atol + rtol * m
            ratio = d == 0.0 ? 0.0 : (tol > 0.0 ? d / tol : Inf)
            max_abs = max(max_abs, d)
            max_rel = max(max_rel, r)
            if ratio > worst_ratio
                worst_ratio = ratio
                worst = String(name)
            end
            if d > tol
                nviolations += 1
                array_violations += 1
            end
        end
        array_violations > 0 && push!(bad, "$(name):$(array_violations)")
    end
    detail = "max_abs_diff=$(fmt_float(max_abs)) max_rel_diff=$(fmt_float(max_rel)) " *
             "worst=$(worst) violations=$(nviolations)"
    isempty(bad) || (detail *= " arrays=" * shortlist(unique(bad)))
    log_check(log, category, isempty(bad) && nviolations == 0, detail)
    println("PHASE3_MAXABS ", category, " ", fmt_float(max_abs))
    return isempty(bad) && nviolations == 0
end

function array_l2_64(a)
    s = 0.0
    for x in a
        xf = Float64(x)
        s += xf * xf
    end
    return sqrt(s)
end

function array_diff_l2_64(a, b)
    s = 0.0
    for (x, y) in zip(a, b)
        d = Float64(x) - Float64(y)
        s += d * d
    end
    return sqrt(s)
end

fmt_l2(x) = @sprintf("%.6e", Float64(x))

"""
    leaf_l2_report(log, category, pairs; floor=1.0e-12, worst_n=10)

Print supplementary per-leaf L2 diagnostics for `(name, actual, expected)`
pairs. A relative L2 ratio is reported only when `ref_l2 > floor`; otherwise
the absolute norms and difference are printed with `rel_l2=NA`. Exact
zero/nonzero disagreements and the worst valid ratios are also printed.
This function never changes the pass/fail result.
"""
function leaf_l2_report(log::CheckLog, category::AbstractString, pairs;
                        floor::Float64 = 1.0e-12, worst_n::Int = 10,
                        extra_summary_prefix::Union{Nothing,AbstractString} = nothing)
    leaves = 0
    max_rel = nothing
    worst_leaf = "-"
    zero_ref_nonzero_actual = 0
    nonzero_ref_zero_actual = 0
    below_floor = 0
    worst = Tuple{Float64,String}[]
    for (name, actual, expected) in pairs
        leaves += 1
        leaf = String(name)
        if actual === nothing || expected === nothing || size(actual) != size(expected)
            println("PHASE3_LEAF_L2 category=", category, " leaf=", leaf,
                    " ref_l2=NA actual_l2=NA diff_l2=NA rel_l2=NA")
            continue
        end
        ref_l2 = array_l2_64(expected)
        actual_l2 = array_l2_64(actual)
        diff_l2 = array_diff_l2_64(actual, expected)
        if ref_l2 == 0.0 && actual_l2 > 0.0
            zero_ref_nonzero_actual += 1
            println(@sprintf("PHASE3_LEAF_ZERO category=%s leaf=%s ref_l2=%.6e actual_l2=%.6e",
                             category, leaf, ref_l2, actual_l2))
        elseif ref_l2 > 0.0 && actual_l2 == 0.0
            nonzero_ref_zero_actual += 1
            println(@sprintf("PHASE3_LEAF_ZERO category=%s leaf=%s ref_l2=%.6e actual_l2=%.6e",
                             category, leaf, ref_l2, actual_l2))
        end
        if ref_l2 > floor
            rel_l2 = diff_l2 / ref_l2
            println(@sprintf("PHASE3_LEAF_L2 category=%s leaf=%s ref_l2=%.6e actual_l2=%.6e diff_l2=%.6e rel_l2=%.6e",
                             category, leaf, ref_l2, actual_l2, diff_l2, rel_l2))
            push!(worst, (rel_l2, leaf))
            if isnothing(max_rel) || rel_l2 > max_rel
                max_rel = rel_l2
                worst_leaf = leaf
            end
        else
            below_floor += 1
            println(@sprintf("PHASE3_LEAF_L2 category=%s leaf=%s ref_l2=%.6e actual_l2=%.6e diff_l2=%.6e rel_l2=NA",
                             category, leaf, ref_l2, actual_l2, diff_l2))
        end
    end
    sort!(worst; by = x -> (-x[1], x[2]))
    summary(prefix) = println(@sprintf(
        "%s category=%s leaves=%d max_rel_l2=%s worst_leaf=%s zero_ref_nonzero_actual=%d nonzero_ref_zero_actual=%d below_floor=%d",
        prefix, category, leaves, isnothing(max_rel) ? "NA" : fmt_l2(max_rel),
        worst_leaf, zero_ref_nonzero_actual, nonzero_ref_zero_actual, below_floor))
    summary("PHASE3_LEAF_L2_SUMMARY")
    extra_summary_prefix === nothing || summary(extra_summary_prefix)
    for (rank, (rel_l2, leaf)) in enumerate(worst[1:min(worst_n, length(worst))])
        println(@sprintf("PHASE3_LEAF_L2_WORST category=%s rank=%d leaf=%s rel_l2=%.6e",
                         category, rank, leaf, rel_l2))
    end
    return nothing
end

function beta_power_expected(beta::NTuple{2,Float32}, s::Int)
    b1, b2 = beta
    p1, p2 = b1, b2
    for _ in 1:s
        p1 *= b1
        p2 *= b2
    end
    return [p1, p2]
end

function make_batch(ref, t::Int, to_device)
    x_nchw = as_array(ref["input.u$(t)"])
    labels = vec(Int.(as_array(ref["label.u$(t)"])))
    x = to_device(permutedims(x_nchw, (4, 3, 2, 1)))
    y = to_device(Flux.onehotbatch(labels .+ 1, 1:200))
    return x, y
end

function run_training(model, opt_state, ref, to_device)
    losses = Dict{Int,Any}()
    logits = Dict{Int,Any}()
    grads_u1 = nothing
    grads_u3 = nothing
    acts_u1 = nothing
    nupdates = 0
    for t in 1:3
        x, y = make_batch(ref, t, to_device)
        eta = schedule_eta(t)
        capture = t == 1
        res = train_step!(model, opt_state, x, y;
                          eta = eta, capture_activations = capture)
        model = res.model
        opt_state = res.opt_state
        losses[t] = host_array(res.loss isa Number ? [res.loss] : res.loss)
        logits[t] = host_array(res.logits)
        nupdates += 1
        if t == 1
            grads_u1 = res.grads
            acts_u1 = res.activations
        elseif t == 3
            grads_u3 = res.grads
        end
    end
    return (; model, opt_state, losses, logits, grads_u1, grads_u3, acts_u1, nupdates)
end

function build_param_map(model, names)
    return Dict(name => canonical_param(name, host_array(param_by_name(model, name)))
                for name in names)
end

function build_grad_map(grads, names)
    out = Dict{String,Any}()
    for name in names
        try
            g = gradient_for(grads, name)
            out[name] = g isa AbstractArray ?
                canonical_param(name, host_array(g)) : nothing
        catch
            out[name] = nothing
        end
    end
    return out
end

function build_moment_map(state, names, which::Int)
    leaves = Dict(name => leaf for (name, leaf) in state_leaves(state))
    out = Dict{String,Any}()
    for name in names
        if haskey(leaves, name)
            out[name] = canonical_param(name, host_array(leaves[name].state[which]))
        else
            out[name] = nothing
        end
    end
    return out, leaves
end

function named_pairs(prefix::String, actual::Dict, ref, names)
    return [(("$(prefix).$(name)"),
             actual[name],
             haskey(ref, "$(prefix).$(name)") ? as_array(ref["$(prefix).$(name)"]) : nothing)
            for name in names]
end

function run_flux_checks(log::CheckLog, opts::Dict{String,String})
    for key in keys(opts)
        key in ALLOWED_OPTS || error("unknown option --$(key)")
    end
    fixture_dir = get(opts, "fixture-dir", "")
    isempty(fixture_dir) && (fixture_dir = DEFAULT_FIXTURE_DIR)
    isdir(fixture_dir) || error("--fixture-dir not found: $(fixture_dir)")
    bundle_path = get(opts, "bundle", "")
    isempty(bundle_path) && (bundle_path = joinpath(fixture_dir, "initial_bundle.npz"))
    reference_path = get(opts, "reference", "")
    isempty(reference_path) && (reference_path = DEFAULT_REFERENCE)
    manifest_path = get(opts, "manifest", "")
    isempty(manifest_path) &&
        (manifest_path = joinpath(dirname(reference_path), "phase3_reference_manifest.json"))
    device = lowercase(get(opts, "device", "cpu"))
    device in ("cpu", "cuda") || error("--device must be cpu or cuda, got $(device)")
    atol = parse_float_opt(opts, "atol", 1.0e-5)
    rtol = parse_float_opt(opts, "rtol", 1.0e-4)
    grads_u1_atol = parse_float_opt(opts, "grads-u1-atol", 1.0e-3)
    grads_u3_atol = parse_float_opt(opts, "grads-u3-atol", 2.0e-3)
    m_u3_atol = parse_float_opt(opts, "m-u3-atol", 2.0e-4)
    all(t >= 0.0 for t in (atol, rtol, grads_u1_atol, grads_u3_atol, m_u3_atol)) ||
        error("--atol, --rtol, --grads-u1-atol, --grads-u3-atol, and " *
              "--m-u3-atol must be non-negative")

    println("PHASE3_FLUX_CHECK_CONFIG device=$(device) atol=$(fmt_float(atol)) " *
            "rtol=$(fmt_float(rtol)) " *
            "grads_u1_atol=$(fmt_float(grads_u1_atol)) grads_u1_rtol=$(fmt_float(rtol)) " *
            "grads_u3_atol=$(fmt_float(grads_u3_atol)) grads_u3_rtol=$(fmt_float(rtol)) " *
            "m_u3_atol=$(fmt_float(m_u3_atol)) m_u3_rtol=$(fmt_float(rtol))")
    println("PHASE3_ACCEPTANCE_RULE base=\"" * rule_text(atol, rtol) * "\" " *
            "phase3.grads.u1=\"" * rule_text(grads_u1_atol, rtol) * "\" " *
            "phase3.grads.u3=\"" * rule_text(grads_u3_atol, rtol) * "\" " *
            "phase3.m.u3=\"" * rule_text(m_u3_atol, rtol) * "\" " *
            "other_categories=base")
    println("PHASE3_FLUX_CHECK_CONFIG fixture_dir=$(fixture_dir)")
    println("PHASE3_FLUX_CHECK_CONFIG bundle=$(bundle_path)")
    println("PHASE3_FLUX_CHECK_CONFIG reference=$(reference_path)")
    println("PHASE3_FLUX_CHECK_CONFIG manifest=$(manifest_path)")

    files_ok = true
    for (label, path) in (("bundle", bundle_path), ("reference", reference_path),
                          ("manifest", manifest_path))
        exists = isfile(path)
        log_check(log, "files.$(label)", exists, path)
        files_ok &= exists
    end
    files_ok || return

    bundle = read_npz_fields(bundle_path)
    ref = read_npz_fields(reference_path)

    local manifest
    try
        manifest = json_parse_file(manifest_path)
        log_check(log, "manifest.parse", true, manifest_path)
    catch err
        manifest = nothing
        log_check(log, "manifest.parse", false, one_line(sprint(showerror, err)))
    end

    model = check_model_and_bundle(log, bundle)
    check_protocol(log, fixture_dir, ref)
    ref_ok = check_reference_contract(log, ref)
    verify_manifest(log, manifest, ref, reference_path)

    to_device = identity
    if device == "cuda"
        dev = configure_cuda!(log)
        model = dev(model)
        to_device = dev
    end

    opt_state = build_optimizer_state(model; eta = schedule_eta(1))
    check_optimizer_contract(log, opt_state)
    check_schedule(log, ref)
    ref_ok || return

    result = run_training(model, opt_state, ref, to_device)
    names = bundle_parameter_names()

    act_pairs = Tuple{String,Any,Any}[]
    if result.acts_u1 === nothing
        log_check(log, "phase3.activations", false, "no u1 activations captured")
    else
        for (refname, _) in REF_ACTIVATIONS
            key = activation_key(refname)
            actual = activation_to_canonical(key, host_array(getproperty(result.acts_u1, key)))
            expected = haskey(ref, refname) ? as_array(ref[refname]) : nothing
            push!(act_pairs, (refname, actual, expected))
        end
        compare_category(log, "phase3.activations", act_pairs; atol = atol, rtol = rtol)
    end

    logit_pairs = Tuple{String,Any,Any}[]
    for t in 1:3
        key = "logits.u$(t)"
        actual = permutedims(result.logits[t], (2, 1))
        expected = haskey(ref, key) ? as_array(ref[key]) : nothing
        push!(logit_pairs, (key, actual, expected))
    end
    compare_category(log, "phase3.logits", logit_pairs; atol = atol, rtol = rtol)

    loss_pairs = Tuple{String,Any,Any}[]
    for t in 1:3
        key = "loss.u$(t)"
        actual = [Float32(scalar_value(result.losses[t]))]
        expected = haskey(ref, key) ? [Float32(scalar_value(ref[key]))] : nothing
        push!(loss_pairs, (key, actual, expected))
    end
    compare_category(log, "phase3.loss", loss_pairs; atol = atol, rtol = rtol)

    grad_u1 = build_grad_map(result.grads_u1, names)
    grad_u3 = build_grad_map(result.grads_u3, names)
    grad_u1_pairs = named_pairs("grad.u1", grad_u1, ref, names)
    grad_u3_pairs = named_pairs("grad.u3", grad_u3, ref, names)
    compare_category(log, "phase3.grads.u1", grad_u1_pairs;
                     atol = grads_u1_atol, rtol = rtol)
    compare_category(log, "phase3.grads.u3", grad_u3_pairs;
                     atol = grads_u3_atol, rtol = rtol)
    leaf_l2_report(log, "grads.u1", grad_u1_pairs)
    leaf_l2_report(log, "grads.u3", grad_u3_pairs)

    params_u3 = build_param_map(result.model, names)
    params_u3_pairs = named_pairs("param.u3", params_u3, ref, names)
    compare_category(log, "phase3.params.u3", params_u3_pairs; atol = atol, rtol = rtol)
    leaf_l2_report(log, "params.u3", params_u3_pairs)

    delta_pairs = Tuple{String,Any,Any}[]
    for name in names
        key = "param.u3.$(name)"
        actual = params_u3[name]
        expected = haskey(ref, key) ? as_array(ref[key]) : nothing
        if actual === nothing || expected === nothing
            push!(delta_pairs, ("delta.u3.$(name)", nothing, nothing))
        else
            push!(delta_pairs, ("delta.u3.$(name)",
                                actual - bundle[name],
                                expected - bundle[name]))
        end
    end
    leaf_l2_report(log, "param_increment.u3", delta_pairs;
                   extra_summary_prefix = "PHASE3_PARAM_INCREMENT_SUMMARY")

    m_u3, leaves_u3 = build_moment_map(result.opt_state, names, 1)
    v_u3, _ = build_moment_map(result.opt_state, names, 2)
    m_u3_pairs = named_pairs("m.u3", m_u3, ref, names)
    v_u3_pairs = named_pairs("v.u3", v_u3, ref, names)
    compare_category(log, "phase3.m.u3", m_u3_pairs;
                     atol = m_u3_atol, rtol = rtol)
    compare_category(log, "phase3.v.u3", v_u3_pairs; atol = atol, rtol = rtol)
    leaf_l2_report(log, "m.u3", m_u3_pairs)
    leaf_l2_report(log, "v.u3", v_u3_pairs)

    step_ref = haskey(ref, "step.u3") ? scalar_value(ref["step.u3"]) : nothing
    s = step_ref isa Real ? Int(round(step_ref)) : 3
    expected_beta = beta_power_expected((0.9f0, 0.999f0), s)
    beta_pairs = Tuple{String,Any,Any}[]
    for name in names
        if haskey(leaves_u3, name)
            push!(beta_pairs, ("beta_power.u3.$(name)",
                               collect(Float32, leaves_u3[name].state[3]),
                               expected_beta))
        else
            push!(beta_pairs, ("beta_power.u3.$(name)", nothing, expected_beta))
        end
    end
    compare_category(log, "phase3.beta_power.u3", beta_pairs; atol = atol, rtol = rtol)

    if step_ref === nothing
        log_check(log, "phase3.step", false, "step.u3 missing")
    else
        ok = step_ref isa Integer && Int(step_ref) == 3 && result.nupdates == 3
        log_check(log, "phase3.step", ok,
                  "reference_step=$(step_ref) flux_updates=$(result.nupdates) expected=3")
    end

    # Algorithm-selection evidence is supplementary to the numerical
    # categories above, but a non-FMA/tensor-op choice is never accepted.
    # `log.failed` already contains every acceptance-category result.
    check_cudnn_algo_selection(log, device)
    return nothing
end

function manifest_top_or_section(manifest, section::AbstractString, key::AbstractString)
    if manifest isa JSONObject && haskey(manifest, key)
        return manifest[key]
    end
    node = json_find(manifest, section)
    if node isa JSONObject && haskey(node, key)
        return node[key]
    end
    return JSON_MISSING
end

manifest_dtype_name(::Type{Float32}) = "float32"
manifest_dtype_name(::Type{Float64}) = "float64"
manifest_dtype_name(::Type{Int64}) = "int64"
manifest_dtype_name(T::Type) = lowercase(string(T))

function check_manifest_count(log::CheckLog, manifest, label::AbstractString,
                              candidates, want::Int)
    for (section, key) in candidates
        v = manifest_top_or_section(manifest, section, key)
        if v === JSON_MISSING
            continue
        end
        if v isa Integer
            log_check(log, "manifest.$(label)", Int(v) == want,
                      "got=$(v) want=$(want) key=$(key)")
        else
            log_check(log, "manifest.$(label)", false,
                      "not an integer: $(typeof(v)) key=$(key)")
        end
        return nothing
    end
    println("CHECK manifest.$(label) SKIP not recorded")
    return nothing
end

function manifest_step_sequence(manifest)
    for key in ("optimizer_step_numbers", "step_numbers")
        v = json_find(manifest, key)
        v === JSON_MISSING && continue
        if v isa JSONObject
            values = Int[]
            for t in 1:3
                tag = "u$(t)"
                (haskey(v, tag) && v[tag] isa Integer) || return nothing
                push!(values, Int(v[tag]))
            end
            return values
        elseif v isa Vector && all(x -> x isa Integer, v)
            return Int.(v)
        else
            return nothing
        end
    end
    return nothing
end

function verify_manifest(log::CheckLog, manifest, ref, reference_path::AbstractString)
    manifest === nothing && return nothing
    syn = json_find(manifest, "synthetic")
    if syn === JSON_MISSING
        println("CHECK manifest.synthetic SKIP not recorded")
    else
        log_check(log, "manifest.synthetic", syn === false, "value=$(syn)")
    end

    check_manifest_count(log, manifest, "parameter_count",
                         [("model", "parameter_count"),
                          ("", "model_parameter_count")], 11271432)
    check_manifest_count(log, manifest, "array_count",
                         [("model", "parameter_array_count"),
                          ("", "model_array_count"),
                          ("model", "array_count")], 62)

    steps = manifest_step_sequence(manifest)
    if steps === nothing
        recorded = json_find(manifest, "optimizer_step_numbers") !== JSON_MISSING ||
                   json_find(manifest, "step_numbers") !== JSON_MISSING
        if recorded
            log_check(log, "manifest.step_numbers", false,
                      "step-number entry is not a u1/u2/u3 integer mapping")
        else
            println("CHECK manifest.step_numbers SKIP not recorded")
        end
    else
        log_check(log, "manifest.step_numbers", steps == [1, 2, 3],
                  "got=$(steps) want=[1, 2, 3]")
    end

    entries = json_bundle_entries(manifest)
    entry_names = String[]
    for (raw_name, _) in entries
        name = occursin("::", raw_name) ? split(raw_name, "::"; limit = 2)[2] : raw_name
        push!(entry_names, String(name))
    end
    expected_names = expected_reference_names()
    missing_entries = setdiff(expected_names, entry_names)
    extra_entries = setdiff(entry_names, expected_names)
    duplicates = [n for n in unique(entry_names) if count(==(n), entry_names) > 1]
    log_check(log, "manifest.array_set",
              !isempty(entries) && length(entries) == 337 &&
              length(entry_names) == 337 && isempty(missing_entries) &&
              isempty(extra_entries) && isempty(duplicates),
              "n=$(length(entries)) want=337 missing=$(shortlist(missing_entries)) " *
              "extra=$(shortlist(extra_entries)) duplicates=$(shortlist(duplicates))")

    missing_ref = String[]
    bad_hash = String[]
    bad_shape = String[]
    bad_dtype = String[]
    for (raw_name, entry) in entries
        entry isa JSONObject || continue
        name = occursin("::", raw_name) ? split(raw_name, "::"; limit = 2)[2] : raw_name
        if !haskey(ref, name)
            push!(missing_ref, name)
            continue
        end
        a = as_array(ref[name])
        want = get(entry, "sha256", nothing)
        if !(want isa String)
            push!(bad_hash, "$(name):no-sha256")
        elseif general_c_order_sha256(a) != lowercase(String(want))
            push!(bad_hash, name)
        end
        shape = get(entry, "shape", nothing)
        if !(shape isa Vector && all(x -> x isa Integer, shape))
            push!(bad_shape, "$(name):no-shape")
        elseif collect(Int64, shape) != collect(Int64, size(a))
            push!(bad_shape, name)
        end
        dtype = get(entry, "dtype", nothing)
        if !(dtype isa String)
            push!(bad_dtype, "$(name):no-dtype")
        elseif lowercase(dtype) != manifest_dtype_name(eltype(a))
            push!(bad_dtype, "$(name):$(dtype)")
        end
    end
    log_check(log, "manifest.arrays_present", isempty(missing_ref),
              "n_checked=$(length(entries)) missing=$(shortlist(missing_ref))")
    log_check(log, "manifest.arrays_sha256", isempty(bad_hash) && length(entries) == 337,
              "n_checked=$(length(entries)) want=337 mismatch=$(shortlist(bad_hash))")
    log_check(log, "manifest.arrays_shape", isempty(bad_shape) && length(entries) == 337,
              "n_checked=$(length(entries)) want=337 mismatch=$(shortlist(bad_shape))")
    log_check(log, "manifest.arrays_dtype", isempty(bad_dtype) && length(entries) == 337,
              "n_checked=$(length(entries)) want=337 mismatch=$(shortlist(bad_dtype))")

    if manifest isa JSONObject
        rf = json_find(manifest, "reference_file")
        if rf isa JSONObject
            recorded = get(rf, "sha256", nothing)
            actual = open(reference_path, "r") do io
                bytes2hex(sha256(io))
            end
            log_check(log, "manifest.reference_sha256",
                      recorded isa String && lowercase(String(recorded)) == actual,
                      "recorded=$(recorded) actual=$(actual)")
            n = get(rf, "array_count", nothing)
            if n isa Integer
                log_check(log, "manifest.reference_array_count", Int(n) == length(ref),
                          "got=$(n) actual=$(length(ref))")
            else
                println("CHECK manifest.reference_array_count SKIP not recorded")
            end
        else
            println("CHECK manifest.reference_file SKIP not recorded")
        end
    end
    return nothing
end

"""
    main_flux(argv) -> exit code

Run the acceptance checks and return 0 only when every CHECK passed.
"""
function main_flux(argv)
    log = CheckLog()
    try
        opts = parse_cli_args(argv)
        run_flux_checks(log, opts)
    catch err
        log_check(log, "unhandled_exception", false, one_line(sprint(showerror, err)))
    end
    return final_line(log, "PHASE3_FLUX_CHECK")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main_flux(ARGS))
end
