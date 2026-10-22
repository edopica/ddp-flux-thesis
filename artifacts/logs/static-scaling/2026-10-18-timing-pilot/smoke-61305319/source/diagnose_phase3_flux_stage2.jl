include(joinpath(@__DIR__, "replay.jl"))
include(joinpath(@__DIR__, "model_flux.jl"))
include(joinpath(@__DIR__, "precision.jl"))
using .StaticScalingFlux
using .StaticScalingPrecision
using Flux
using Printf
using CUDA
using cuDNN

function opt(args, key)
    i = findfirst(==("--" * key), args)
    i === nothing && error("missing --" * key)
    return args[i + 1]
end

function maxdiff(a, b)
    size(a) == size(b) || return (Inf, Inf, 0)
    max_abs = 0.0; max_rel = 0.0; bad = 0
    for (x, y) in zip(a, b)
        d = abs(Float64(x) - Float64(y)); s = max(abs(Float64(x)), abs(Float64(y)))
        max_abs = max(max_abs, d)
        max_rel = max(max_rel, s == 0 ? 0.0 : d / s)
        bad += !(d <= 1e-5 + 1e-4 * s)
    end
    return (max_abs, max_rel, bad)
end

fixture_dir = opt(ARGS, "fixture-dir")
bundle_path = opt(ARGS, "bundle")
reference_path = opt(ARGS, "reference")
diag_path = opt(ARGS, "diag")
device = get(ARGS, findfirst(==("--device"), ARGS) === nothing ? 0 : findfirst(==("--device"), ARGS) + 1, "cuda")
precision = get(ARGS, findfirst(==("--precision"), ARGS) === nothing ? 0 : findfirst(==("--precision"), ARGS) + 1, "fma")
precision in ("current", "fma") ||
    error("--precision must be 'current' or 'fma', got '$(precision)'")

bundle = read_npz_fields(bundle_path)
ref = read_npz_fields(reference_path)
diag = read_npz_fields(diag_path)
model = make_model(bundle)
if device == "cuda"
    cfg = configure_precision!(fma = precision == "fma")
    eff_conv1 = cudnn_effective_math_type(; pad = 1, stride = 2)
    eff_conv2 = cudnn_effective_math_type(; pad = 1, stride = 1)
    eff_proj = cudnn_effective_math_type(; pad = 0, stride = 2)
    println("FLUX_STAGE2_CONFIG precision=$(precision) " *
            "cublas_math_mode=$(cfg.cublas_math_mode) " *
            "cudnn_requested=$(math_type_label(cfg.cudnn_requested)) " *
            "cudnn_effective_conv1=$(math_type_label(eff_conv1)) " *
            "cudnn_effective_conv2=$(math_type_label(eff_conv2)) " *
            "cudnn_effective_proj=$(math_type_label(eff_proj)) " *
            "cudnn_version=$(cudnn_version()) " *
            "CUDNN_FMA_MATH=$(math_type_label(cuDNN.CUDNN_FMA_MATH))")
    dev = Flux.gpu_device(; force = true)
    model = dev(model)
else
    println("FLUX_STAGE2_CONFIG precision=$(precision) " *
            "cublas_math_mode=n/a cudnn_requested=n/a " *
            "cudnn_effective_conv1=n/a cudnn_effective_conv2=n/a " *
            "cudnn_effective_proj=n/a cudnn_version=n/a " *
            "CUDNN_FMA_MATH=n/a device=cpu")
end
blk = model.stage2.block0
x = permutedims(ref["act.stage1.block1"], (4, 3, 2, 1))
if device == "cuda"
    x = Flux.gpu_device(; force = true)(x)
end
conv1 = blk.conv1(x)
gn1 = blk.gn1(conv1)
r1 = relu.(gn1)
conv2 = blk.conv2(r1)
gn2 = blk.gn2(conv2)
proj = blk.proj(x)
pgn = blk.proj_gn(proj)
total = gn2 .+ pgn
out = relu.(total)
for (name, val) in [("conv1", conv1), ("gn1", gn1), ("relu1", r1), ("conv2", conv2),
                    ("gn2", gn2), ("proj", proj), ("pgn", pgn), ("total", total), ("out", out)]
    a = permutedims(Array(val), (4, 3, 2, 1))
    d = maxdiff(a, diag[name])
    @printf("FLUX_STAGE2 %-6s max_abs=%.9g max_rel=%.9g violations=%d\n", name, d[1], d[2], d[3])
end
