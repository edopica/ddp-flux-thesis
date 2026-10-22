# Forward-only Phase 3 diagnostic: compare Flux u1 activations and logits
# against the PyTorch reference, with per-array maxima. No optimizer update.
#
# This is a debugging aid. The acceptance check is check_phase3_flux.jl.
# On the cuda path, precision.jl configures the accepted FP32/TF32-off
# policy before any GPU work.
#
# Julia 1.12 note: A[1] on a 5-D array is scalar indexing; use full slices.

include(joinpath(@__DIR__, "replay.jl"))
include(joinpath(@__DIR__, "model_flux.jl"))
include(joinpath(@__DIR__, "precision.jl"))

using .StaticScalingFlux
using .StaticScalingPrecision
using Flux
using Printf
using CUDA
using cuDNN

const ATOL = 1.0e-5
const RTOL = 1.0e-4

function getopt(args, key, default)
    idx = findfirst(==("--" * key), args)
    idx === nothing && return default
    return args[idx + 1]
end

function maxdiff(actual, expected)
    size(actual) == size(expected) ||
        return (Inf, Inf, length(expected), "shape $(size(actual)) vs $(size(expected))")
    max_abs = 0.0
    max_rel = 0.0
    bad = 0
    for (a, b) in zip(actual, expected)
        d = abs(Float64(a) - Float64(b))
        s = max(abs(Float64(a)), abs(Float64(b)))
        r = s == 0 ? 0.0 : d / s
        max_abs = max(max_abs, d)
        max_rel = max(max_rel, r)
        if !(d <= ATOL + RTOL * s)
            bad += 1
        end
    end
    return (max_abs, max_rel, bad, "")
end

function main(args)
    fixture_dir = getopt(args, "fixture-dir",
        joinpath(@__DIR__, "..", "..", "artifacts", "baselines", "static-scaling",
                 "2026-10-16-phase2"))
    bundle_path = getopt(args, "bundle", joinpath(fixture_dir, "initial_bundle.npz"))
    reference_path = getopt(args, "reference",
        joinpath(@__DIR__, "..", "..", "artifacts", "baselines", "static-scaling",
                 "2026-10-17-phase3", "phase3_reference.npz"))
    device = lowercase(getopt(args, "device", "cuda"))

    bundle = read_npz_fields(bundle_path)
    ref = read_npz_fields(reference_path)
    data = read_npz_fields(joinpath(fixture_dir, "data_fixture.npz"))

    model = make_model(bundle)
    x_nchw = data["train_transformed_ref"][1, :, :, :, :]
    x = permutedims(x_nchw, (4, 3, 2, 1))
    if device == "cuda"
        cfg = configure_precision!(fma = true)
        dev = Flux.gpu_device(; force = true)
        model = dev(model)
        x = dev(x)
        println("DEVICE cuda math_mode=$(CUDA.math_mode()) precision=$(CUDA.math_precision())")
        println("DIAGNOSE_CONFIG cudnn_requested=$(math_type_label(cfg.cudnn_requested)) " *
                "cudnn_effective=$(math_type_label(cfg.cudnn_effective)) " *
                "cudnn_version=$(cfg.cudnn_version)")
    end

    logits, acts = forward_with_activations(model, x)
    logits_host = Array(logits)
    println(@sprintf("LOGITS max_abs=%.9g max_rel=%.9g violations=%d",
                     maxdiff(permutedims(logits_host, (2, 1)), ref["logits.u1"])[1:3]...))

    for (refname, key) in (
        ("act.stem.gn", :stem_gn), ("act.stem", :stem),
        ("act.stage1.block0", :stage1_block0), ("act.stage1.block1", :stage1_block1),
        ("act.stage2.block0", :stage2_block0), ("act.stage2.block1", :stage2_block1),
        ("act.stage3.block0", :stage3_block0), ("act.stage3.block1", :stage3_block1),
        ("act.stage4.block0", :stage4_block0), ("act.stage4.block1", :stage4_block1),
        ("act.pooled", :pooled),
    )
        actual = Array(getproperty(acts, key))
        actual = key === :pooled ? permutedims(actual, (2, 1)) : permutedims(actual, (4, 3, 2, 1))
        d = maxdiff(actual, ref[refname])
        println(@sprintf("%-22s max_abs=%.9g max_rel=%.9g violations=%d %s",
                         refname, d[1], d[2], d[3], d[4]))
    end
    return 0
end

exit(main(ARGS))
