#!/usr/bin/env julia
"""
save.jl — Baseline generator for C1.

Constructs the default ReferenceConfig, runs the deterministic reference loop,
saves the result to artifacts/baselines/reference_loop_baseline.jld2, and
prints a summary (loss curve, gradient norms, total time) to stdout.

Usage:
    julia --project=. scripts/reference/save.jl
"""

include(joinpath(@__DIR__, "ReferenceLoop.jl"))
using .ReferenceLoop
using LinearAlgebra: norm
using Printf
using Functors

const BASELINE_PATH = joinpath(@__DIR__, "..", "..", "artifacts", "baselines", "reference_loop_baseline.jld2")

function main()
    println("=" ^ 60)
    println("C1 — Deterministic Reference Loop: Baseline Generator")
    println("=" ^ 60)

    config = ReferenceConfig()
    println("\nConfiguration:")
    println("  seed         = $(config.seed)")
    println("  num_steps    = $(config.num_steps)")
    println("  model_fn     = $(config.model_fn)")
    println("  loss_fn      = $(config.loss_fn)")
    println("  setup_fn     = $(config.setup_fn)")
    println()

    println("Running reference loop...")
    result = run_reference_loop(config)

    # Save baseline
    save_baseline(BASELINE_PATH, result)
    println("\n✓ Baseline saved to: $(BASELINE_PATH)")

    # Print summary
    println("\n" * "-" ^ 60)
    println("Loss curve:")
    println("-" ^ 60)
    for (i, step) in enumerate(result.steps)
        loss_str = @sprintf("%.8f", step.loss)
        println("  Step $(lpad(i, 3)): loss = $(loss_str)")
    end

    println("\n" * "-" ^ 60)
    println("Gradient norms (per step, Frobenius over all params):")
    println("-" ^ 60)
    for (i, step) in enumerate(result.steps)
        if step.gradients !== nothing
            # Collect all gradient arrays and compute global norm
            grad_arrays = []
            Functors.fmap(step.gradients; exclude=x -> x isa AbstractArray) do x
                push!(grad_arrays, x)
                x
            end
            gnorm = sqrt(sum(norm(g)^2 for g in grad_arrays))
            println("  Step $(lpad(i, 3)): ||∇|| = $(@sprintf("%.8f", gnorm))")
        end
    end

    println("\n" * "-" ^ 60)
    println("Summary:")
    println("-" ^ 60)
    println("  Initial loss:  $(@sprintf("%.8f", result.steps[1].loss))")
    println("  Final loss:    $(@sprintf("%.8f", result.steps[end].loss))")
    println("  Total time:    $(@sprintf("%.4f", result.elapsed_total)) s")
    println("  Baseline file: $(BASELINE_PATH)")
    println("\n✓ Done.")
end

main()
