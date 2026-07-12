#!/usr/bin/env julia
"""
verify.jl — Determinism verifier for C1.

Loads the saved baseline, re-runs the reference loop with the same config,
and asserts bit-identical equality for:
  - Loss at every step
  - All parameter arrays at every step
  - All gradient arrays at every step
  - Optimizer state at every step

Prints PASS or FAIL with details of the first mismatch.

Usage:
    julia --project=. scripts/reference/verify.jl
"""

include(joinpath(@__DIR__, "ReferenceLoop.jl"))
using .ReferenceLoop
using Functors

const BASELINE_PATH = joinpath(@__DIR__, "..", "..", "artifacts", "baselines", "reference_loop_baseline.jld2")

"""
    collect_arrays(x)

Collects all AbstractArray leaves from a Functors-compatible structure.
Returns a flat Vector{AbstractArray}.
"""
function collect_arrays(x)
    arrays = AbstractArray[]
    Functors.fmap(x; exclude=a -> a isa AbstractArray) do a
        push!(arrays, a)
        a
    end
    return arrays
end

"""
    compare_structures(label, expected, actual, step)

Compare two Functors-compatible structures element-by-element.
Returns (pass::Bool, message::String).
"""
function compare_structures(label::String, expected, actual, step::Int)
    exp_arrays = collect_arrays(expected)
    act_arrays = collect_arrays(actual)

    if length(exp_arrays) != length(act_arrays)
        return (false, "Step $step $label: array count mismatch (expected=$(length(exp_arrays)), actual=$(length(act_arrays)))")
    end

    for (i, (e, a)) in enumerate(zip(exp_arrays, act_arrays))
        if size(e) != size(a)
            return (false, "Step $step $label array $i: size mismatch (expected=$(size(e)), actual=$(size(a)))")
        end
        if e != a
            # Find first differing element
            idx = findfirst(e .!= a)
            return (false, "Step $step $label array $i: value mismatch at index $idx (expected=$(e[idx]), actual=$(a[idx]))")
        end
    end

    return (true, "")
end

function main()
    println("=" ^ 60)
    println("C1 — Deterministic Reference Loop: Verification")
    println("=" ^ 60)

    # Check baseline exists
    if !isfile(BASELINE_PATH)
        println("\nFAIL: Baseline file not found at $(BASELINE_PATH)")
        println("      Run `make reference` first to generate the baseline.")
        exit(1)
    end

    println("\nLoading baseline from: $(BASELINE_PATH)")
    baseline = load_baseline(BASELINE_PATH)

    println("Re-running reference loop with same config...")
    println("  seed       = $(baseline.config.seed)")
    println("  num_steps  = $(baseline.config.num_steps)")
    result = run_reference_loop(baseline.config)

    num_steps = baseline.config.num_steps
    pass = true
    first_failure = ""

    println("\nVerifying bit-identical equality for $num_steps steps...")

    for step in 1:num_steps
        b = baseline.steps[step]
        r = result.steps[step]

        # Check loss
        if b.loss != r.loss
            msg = "Step $step loss: expected=$(b.loss), actual=$(r.loss)"
            if pass
                first_failure = msg
            end
            pass = false
            println("  FAIL: $msg")
            break
        end

        # Check gradients
        if b.gradients !== nothing && r.gradients !== nothing
            ok, msg = compare_structures("gradients", b.gradients, r.gradients, step)
            if !ok
                if pass
                    first_failure = msg
                end
                pass = false
                println("  FAIL: $msg")
                break
            end
        end

        # Check parameters
        if b.parameters !== nothing && r.parameters !== nothing
            ok, msg = compare_structures("parameters", b.parameters, r.parameters, step)
            if !ok
                if pass
                    first_failure = msg
                end
                pass = false
                println("  FAIL: $msg")
                break
            end
        end

        # Check optimizer state
        if b.optimizer_state !== nothing && r.optimizer_state !== nothing
            ok, msg = compare_structures("optimizer_state", b.optimizer_state, r.optimizer_state, step)
            if !ok
                if pass
                    first_failure = msg
                end
                pass = false
                println("  FAIL: $msg")
                break
            end
        end
    end

    # Also check initial model and gradients
    if pass
        ok, msg = compare_structures("initial_model", baseline.initial_model, result.initial_model, 0)
        if !ok
            pass = false
            first_failure = msg
            println("  FAIL: $msg")
        end
    end

    if pass
        ok, msg = compare_structures("initial_gradients", baseline.initial_gradients, result.initial_gradients, 0)
        if !ok
            pass = false
            first_failure = msg
            println("  FAIL: $msg")
        end
    end

    println()
    println("=" ^ 60)
    if pass
        println("PASS — All $num_steps steps are bit-identical.")
        println("       Loss, gradients, parameters, and optimizer state match.")
    else
        println("FAIL — Determinism check failed.")
        println("       First mismatch: $first_failure")
        exit(1)
    end
    println("=" ^ 60)
end

main()
