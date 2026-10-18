# Cross-framework optimizer-contract probe: Flux side.
#
# Run on a Leonardo compute node through
#   scripts/remote/leonardo/run-julia-phase3.sh
# or locally with the thesis julia-env plus the Flux test project on
# JULIA_LOAD_PATH.  The probe is independent of the training run; it only
# exercises library optimizers and the frozen schedule.
#
# Probe-specific comparison rule (not the Phase 3 acceptance rule):
#   |a - b| <= 1e-7 + 5e-7 * max(|a|, |b|)
# Float32 has ~1.19e-7 relative spacing near |x| = 1.  A three-update
# library-vs-Float64-oracle calibration leaves at most ~0.17 of this bound;
# the bound therefore has ~6x margin over observed library rounding.  A wrong
# decay on a no-decay leaf moves x by eta*lambda*|x| = 1e-6*|x|, which exceeds
# the bound by at least ~1.8x for the probe's |x| >= 2 no-decay leaf.  Wrong
# epsilon placement changes the tiny-gradient update by ~1e-3 and is rejected
# by orders of magnitude.
#
# The real training-path rule and per-leaf lambda assignment are asserted on
# the full 11,271,432-parameter bundle model without running 11M-parameter
# updates.  Controlled updates use small leaves and eta = 1e-2 so the
# decoupled decay term is visible at Float32 scale; this is a sensitivity
# probe with the same equation, not a change to the training schedule.

using Optimisers
using NPZ

const HERE = @__DIR__
const THESIS = normpath(joinpath(HERE, "..", "..", ".."))
const SCRIPTS = joinpath(THESIS, "scripts", "static_scaling")

include(joinpath(SCRIPTS, "model_flux.jl"))
using .StaticScalingFlux
const SF = StaticScalingFlux

# Reuse the existing constructor and beta-power checks unchanged.  The include
# runs the original @assert checks and prints its R3/R4 lines.
include(joinpath(HERE, "probe_optimizer_contract.jl"))
const LEGACY_PROBE_OK = true

const PROBE_ETA = 1.0e-2
const PROBE_BETA = (0.9, 0.999)
const PROBE_EPS = 1.0e-8
const PROBE_WD = 1.0e-4

# ---------------------------------------------------------------------------
# JSON writer (the probe emits only dictionaries, vectors, strings, numbers)
# ---------------------------------------------------------------------------

_json_escape(s::AbstractString) = replace(String(s), "\\" => "\\\\", "\"" => "\\\"")

function _json_write(io::IO, x)
    if x isa AbstractDict
        print(io, "{")
        first = true
        for k in sort!(collect(keys(x)); by = string)
            first || print(io, ",")
            first = false
            print(io, "\"", _json_escape(string(k)), "\":")
            _json_write(io, x[k])
        end
        print(io, "}")
    elseif x isa AbstractVector || x isa Tuple
        print(io, "[")
        first = true
        for v in x
            first || print(io, ",")
            first = false
            _json_write(io, v)
        end
        print(io, "]")
    elseif x isa AbstractString
        print(io, "\"", _json_escape(x), "\"")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, string(x))
    elseif x isa Real
        print(io, repr(Float64(x)))
    elseif x === nothing
        print(io, "null")
    else
        error("unsupported JSON value: $(typeof(x))")
    end
end

function write_json(path::AbstractString, value)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        _json_write(io, value)
        print(io, "\n")
    end
end

# ---------------------------------------------------------------------------
# Controlled library updates
# ---------------------------------------------------------------------------

function run_case(name::AbstractString, x0, grads, lambda::Real)
    rule = Optimisers.AdamW(; eta = PROBE_ETA, beta = PROBE_BETA,
                            lambda = lambda, epsilon = PROBE_EPS, couple = true)
    x = Float32.(x0)
    gsteps = [Float32.(g) for g in grads]
    state = Optimisers.init(rule, x)

    x_steps = Vector{Vector{Float64}}()
    m_before = Float64[]
    v_before = Float64[]
    bp_before = Float64[]
    m_final = Float64[]
    v_final = Float64[]
    bp_final = Float64[]

    for (i, g) in enumerate(gsteps)
        if i == length(gsteps)
            m_before = Float64.(state[1])
            v_before = Float64.(state[2])
            bp_before = Float64.(state[3])
        end
        state, dx = Optimisers.apply!(rule, state, x, g)
        x = x .- dx
        push!(x_steps, Float64.(x))
    end
    m_final = Float64.(state[1])
    v_final = Float64.(state[2])
    bp_final = Float64.(state[3])

    return Dict{String,Any}(
        "name" => String(name),
        "lambda" => Float64(lambda),
        "x0" => Float64.(Float32.(x0)),
        "grads" => [Float64.(g) for g in gsteps],
        "x_steps" => x_steps,
        "x_final" => x_steps[end],
        "m_before_measured" => m_before,
        "v_before_measured" => v_before,
        "beta_power_before_measured" => bp_before,
        "m_final" => m_final,
        "v_final" => v_final,
        "beta_power_final" => bp_final,
        "all_nonzero_m_before_measured" => all(!iszero, m_before),
        "all_nonzero_v_before_measured" => all(!iszero, v_before),
        "rule" => Dict{String,Any}(
            "eta" => rule.eta,
            "beta" => collect(rule.beta),
            "lambda" => rule.lambda,
            "epsilon" => rule.epsilon,
            "couple" => rule.couple,
        ),
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function parse_args(args)
    opts = Dict{String,String}(
        "bundle" => joinpath(THESIS, "artifacts", "baselines", "static-scaling",
                             "2026-10-16-phase2", "initial_bundle.npz"),
        "out" => joinpath(HERE, "probe_flux.json"),
    )
    i = 1
    while i <= length(args)
        if args[i] == "--bundle" && i < length(args)
            opts["bundle"] = args[i + 1]; i += 2
        elseif args[i] == "--out" && i < length(args)
            opts["out"] = args[i + 1]; i += 2
        else
            error("usage: probe_optimizer_contract_cross.jl [--bundle PATH] [--out PATH]")
        end
    end
    return opts
end

function main(args)
    opts = parse_args(args)
    bundle_path = abspath(opts["bundle"])
    out_path = abspath(opts["out"])
    isfile(bundle_path) || error("bundle not found: $(bundle_path)")

    # Actual training-path model and rule/group configuration.
    bundle = npzread(bundle_path)
    model = SF.make_model(bundle)
    train_state = SF.build_optimizer_state(model; eta = SF.ETA_MAX)
    leaves = SF.state_leaves(train_state)
    decay_leaves = [(name, leaf) for (name, leaf) in leaves if leaf.rule.lambda != 0]
    nodecay_leaves = [(name, leaf) for (name, leaf) in leaves if leaf.rule.lambda == 0]
    rule_meta_ok = all(
        leaf.rule isa Optimisers.AdamW &&
        leaf.rule.beta == SF.BETA &&
        leaf.rule.epsilon == SF.EPSILON &&
        leaf.rule.couple === true
        for (_, leaf) in leaves)

    training_path = Dict{String,Any}(
        "parameter_count" => SF.parameter_count(model),
        "leaf_count" => length(leaves),
        "decay_leaf_count" => length(decay_leaves),
        "nodecay_leaf_count" => length(nodecay_leaves),
        "decay_lambda_values" => sort(unique(Float64[leaf.rule.lambda for (_, leaf) in decay_leaves])),
        "nodecay_lambda_values" => sort(unique(Float64[leaf.rule.lambda for (_, leaf) in nodecay_leaves])),
        "decay_leaf_names_head" => first([name for (name, _) in decay_leaves], 5),
        "nodecay_leaf_names_head" => first([name for (name, _) in nodecay_leaves], 5),
        "rule_meta_ok" => rule_meta_ok,
        "eta_max" => SF.ETA_MAX,
        "beta" => collect(SF.BETA),
        "epsilon" => SF.EPSILON,
        "weight_decay" => SF.WEIGHT_DECAY,
    )

    # Schedule boundary values and out-of-range rejection.
    schedule = Dict{String,Any}(
        "eta_max" => SF.ETA_MAX,
        "W" => SF.WARMUP_UPDATES,
        "T" => SF.TOTAL_UPDATES,
        "t1" => SF.schedule_eta(1),
        "tW" => SF.schedule_eta(SF.WARMUP_UPDATES),
        "tW1" => SF.schedule_eta(SF.WARMUP_UPDATES + 1),
        "tT" => SF.schedule_eta(SF.TOTAL_UPDATES),
    )
    for (key, t) in (("reject_t0", 0), ("reject_tTplus1", SF.TOTAL_UPDATES + 1))
        ok = try
            SF.schedule_eta(t)
            false
        catch err
            err isa ArgumentError
        end
        schedule[key] = ok
        schedule[key * "_type"] = "ArgumentError"
    end
    schedule["rejection_ok"] = schedule["reject_t0"] && schedule["reject_tTplus1"]

    # Controlled cases: one decay leaf and one no-decay leaf per case.
    case_specs = Dict(
        "main" => Dict(
            "decay" => ([1.0, -0.5, 0.25],
                        [[0.1, -0.2, 0.05],
                         [-0.3, 0.4, -0.15],
                         [0.2, 0.15, -0.25]]),
            "nodecay" => ([2.0, -1.5, 0.75],
                          [[0.05, -0.1, 0.2],
                           [-0.4, 0.3, 0.1],
                           [0.15, -0.2, 0.05]]),
        ),
        "tiny" => Dict(
            "decay" => ([1.0, -1.0, 0.5],
                        [[1.0e-9, -1.0e-12, 1.0e-9],
                         [-2.0e-9, 1.0e-12, -1.0e-9],
                         [1.5e-9, -2.0e-12, 0.5e-9]]),
            "nodecay" => ([2.0, -1.5, 0.75],
                          [[-1.0e-9, 2.0e-12, -1.0e-9],
                           [1.0e-9, -1.0e-12, 2.0e-9],
                           [-1.5e-9, 2.0e-12, -0.5e-9]]),
        ),
    )

    cases = Dict{String,Any}()
    npz = Dict{String,Any}()
    for case_name in ("main", "tiny")
        cases[case_name] = Dict{String,Any}()
        for leaf_name in ("decay", "nodecay")
            x0, grads = case_specs[case_name][leaf_name]
            lambda = leaf_name == "decay" ? PROBE_WD : 0.0
            c = run_case("$(case_name).$(leaf_name)", x0, grads, lambda)
            cases[case_name][leaf_name] = c
            prefix = "$(case_name)_$(leaf_name)"
            npz["$(prefix)_x0"] = Float64.(c["x0"])
            npz["$(prefix)_x_final"] = Float64.(c["x_final"])
            npz["$(prefix)_m_final"] = Float64.(c["m_final"])
            npz["$(prefix)_v_final"] = Float64.(c["v_final"])
        end
    end

    result = Dict{String,Any}(
        "framework" => "flux",
        "probe_eta" => PROBE_ETA,
        "probe_beta" => collect(PROBE_BETA),
        "probe_epsilon" => PROBE_EPS,
        "probe_weight_decay" => PROBE_WD,
        "legacy_probe_included" => LEGACY_PROBE_OK,
        "julia_version" => string(VERSION),
        "optimisers_version" => string(pkgversion(Optimisers)),
        "training_path" => training_path,
        "schedule" => schedule,
        "cases" => cases,
    )
    write_json(out_path, result)
    npz_path = replace(out_path, r"\.json$" => "_arrays.npz")
    npzwrite(npz_path, npz)

    println("PROBE_FLUX training_path leaves=$(length(leaves)) decay=$(length(decay_leaves)) nodecay=$(length(nodecay_leaves)) rule_meta_ok=$(rule_meta_ok)")
    println("PROBE_FLUX schedule t1=$(schedule["t1"]) tW=$(schedule["tW"]) tW1=$(schedule["tW1"]) tT=$(schedule["tT"]) rejection_ok=$(schedule["rejection_ok"])")
    for case_name in ("main", "tiny")
        for leaf_name in ("decay", "nodecay")
            c = cases[case_name][leaf_name]
            println("PROBE_FLUX case=$(case_name) leaf=$(leaf_name) lambda=$(c["lambda"]) " *
                    "m_nonzero=$(c["all_nonzero_m_before_measured"]) v_nonzero=$(c["all_nonzero_v_before_measured"]) " *
                    "x_final=$(c["x_final"])")
        end
    end
    println("PROBE_FLUX WROTE $(out_path) $(npz_path)")
    return 0
end

exit(main(ARGS))
