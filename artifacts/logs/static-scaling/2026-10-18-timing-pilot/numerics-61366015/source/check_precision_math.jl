"""
Focused GPU precision regression for the static-scaling Phase 3 policy.

The accepted policy is FP32 with TF32 disabled, realized by
`StaticScalingPrecision.configure_precision!()` (cuBLAS PEDANTIC_MATH and
cuDNN `CUDNN_FMA_MATH`). This script checks that policy directly:

* the loaded CUDA.jl and cuDNN.jl package versions are the supported
  6.2.2 pair (the loaded cuDNN library version is recorded separately);
* requested cuDNN math type is `CUDNN_FMA_MATH` (`--mode fma`);
* the effective math type of the memoized stage-2 conv1, conv2, and proj
  descriptors is `CUDNN_FMA_MATH`;
* a stage-2 conv2-shaped convolution forward, data-gradient, and
  weight-gradient match a Float64 CPU arithmetic reference at FP32 level;
* a classifier-sized (200x512 times 512x12) matrix multiplication matches
  a Float64 CPU reference at FP32 level, confirming the cuBLAS pedantic
  path;
* an end-to-end NNlib path convolution (`Flux.CrossCor`, stage-2 conv1
  shape 64->128, stride 2, pad 1, batch 12) matches a Float64 CPU
  reference. In `--mode fma` this also asserts that the recorded
  algorithm choices are non-empty, every choice reports
  `CUDNN_FMA_MATH`, and no choice is a fallback. This is the check that
  exercises the process-wide `cudnnConvolutionAlgoPerfChoose` override,
  not just pinned kernels.

Why the convolution regression pins the cuDNN algorithm
-------------------------------------------------------

The stock cuDNN autotuner chooses algorithms by timing, not by the
requested math type. For 3x3 stride-1 convolutions it can choose FP32
Winograd in both modes; Winograd's transform arithmetic has a relative
error around 3e-4, which masks the TF32-versus-FMA difference. The
controlled stage-2 experiment and dedicated probes on 2026-10-17 showed
that pinning `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` and
`CUDNN_CONVOLUTION_BWD_DATA/FILTER_ALGO_1` makes the effect deterministic:
FMA errors are ~1e-7 to 1e-6, stock DEFAULT_MATH errors are ~3e-4.
The regression therefore calls those cuDNN kernels explicitly, which are
the same backend kernels Flux/NNlib use. Those items isolate the math-mode
contract; the separate autotuned `Flux.CrossCor` item exercises the
process-wide chooser override. The full three-update model path is covered
by the Phase 3 driver.

Negative control
----------------

`--mode current` uses the stock cuDNN.jl mapping
(`PEDANTIC_MATH -> CUDNN_DEFAULT_MATH`, i.e. TF32 permitted) and is
expected to FAIL on the pinned convolution items. It exists to prove the
regression detects the reproduced failure; it is not part of the accepted
pass path. The driver runs it in a separate process because descriptors and
algorithm caches are process-wide.

The declared numerical bound is `REL_L2_BOUND = 1e-4` (relative L2
against the Float64 CPU reference). This is a regression bound, not a
change to the Phase 3 acceptance tolerances (atol=1e-5, rtol=1e-4).

Run on a compute node through `run-julia-phase3.sh`; never on a login node.
"""

include(joinpath(@__DIR__, "precision.jl"))
using .StaticScalingPrecision
using CUDA
using cuDNN
using Flux
using LinearAlgebra
using Printf
using Random

const REL_L2_BOUND = 1.0e-4
const SEED = 20261017

function getopt(args, key, default)
    idx = findfirst(==("--" * key), args)
    idx === nothing && return default
    idx == length(args) && error("missing value for --" * key)
    return args[idx + 1]
end

function host(a)
    return a isa Array ? a : Array(a)
end

function rel_l2(actual, expected)
    a = Float64.(vec(host(actual)))
    b = Float64.(vec(host(expected)))
    denominator = norm(b)
    numerator = norm(a .- b)
    return denominator == 0 ? numerator : numerator / denominator
end

function max_abs(actual, expected)
    a = Float64.(vec(host(actual)))
    b = Float64.(vec(host(expected)))
    length(a) == length(b) || return Inf
    return maximum(abs.(a .- b); init = 0.0)
end

function check_item(name, actual, expected)
    r = rel_l2(actual, expected)
    m = max_abs(actual, expected)
    passed = r <= REL_L2_BOUND
    @printf("CHECK_PRECISION_MATH_ITEM %-20s rel_l2=%.6e max_abs=%.6e bound=%.1e %s\n",
            name, r, m, REL_L2_BOUND, passed ? "PASS" : "FAIL")
    return passed
end

function cpu_reference(x, w, dy)
    x64 = Float64.(x)
    w64 = Float64.(w)
    dy64 = Float64.(dy)
    y64 = Flux.CrossCor(w64, false; stride = 1, pad = 1)(x64)
    _, (gx64, gw64) = Flux.withgradient(
        (xx, ww) -> sum(Flux.CrossCor(ww, false; stride = 1, pad = 1)(xx) .* dy64),
        x64, w64)
    return y64, gx64, gw64
end

function fixed_algo_conv(x, w, dy)
    xc = CuArray(x)
    wc = CuArray(w)
    dyc = CuArray(dy)
    yc = CuArray(zeros(Float32, size(dy)))
    dxc = CuArray(zeros(Float32, size(x)))
    dwc = CuArray(zeros(Float32, size(w)))

    xDesc = cuDNN.cudnnTensorDescriptor(xc)
    wDesc = cuDNN.cudnnFilterDescriptor(wc)
    yDesc = cuDNN.cudnnTensorDescriptor(dyc)
    dxDesc = cuDNN.cudnnTensorDescriptor(dxc)
    dwDesc = cuDNN.cudnnFilterDescriptor(dwc)
    convDesc = cuDNN.cudnnConvolutionDescriptor(
        cuDNN.convdims(1, size(x), 0),
        cuDNN.convdims(1, size(x), 1),
        cuDNN.convdims(1, size(x), 1),
        cuDNN.CUDNN_CROSS_CORRELATION,
        cuDNN.cudnnDataType(Float32),
        cudnn_requested_math_type(),
        cuDNN.CUDNN_DEFAULT_REORDER,
        Cint(1),
    )

    algo_fwd = cuDNN.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    fwd_ws_size = Ref{Csize_t}(0)
    cuDNN.cudnnGetConvolutionForwardWorkspaceSize(
        cuDNN.handle(), xDesc, wDesc, convDesc, yDesc, algo_fwd, fwd_ws_size)
    fwd_ws = CuArray{UInt8}(undef, fwd_ws_size[])
    cuDNN.cudnnConvolutionForward(
        cuDNN.handle(), Ref{Float32}(1.0f0), xDesc, xc, wDesc, wc, convDesc,
        algo_fwd, fwd_ws, sizeof(fwd_ws), Ref{Float32}(0.0f0), yDesc, yc)

    algo_data = cuDNN.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
    data_ws_size = Ref{Csize_t}(0)
    cuDNN.cudnnGetConvolutionBackwardDataWorkspaceSize(
        cuDNN.handle(), wDesc, yDesc, convDesc, dxDesc, algo_data, data_ws_size)
    data_ws = CuArray{UInt8}(undef, data_ws_size[])
    cuDNN.cudnnConvolutionBackwardData(
        cuDNN.handle(), Ref{Float32}(1.0f0), wDesc, wc, yDesc, dyc, convDesc,
        algo_data, data_ws, sizeof(data_ws), Ref{Float32}(0.0f0), dxDesc, dxc)

    algo_filter = cuDNN.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
    filter_ws_size = Ref{Csize_t}(0)
    cuDNN.cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cuDNN.handle(), xDesc, yDesc, convDesc, dwDesc, algo_filter, filter_ws_size)
    filter_ws = CuArray{UInt8}(undef, filter_ws_size[])
    cuDNN.cudnnConvolutionBackwardFilter(
        cuDNN.handle(), Ref{Float32}(1.0f0), xDesc, xc, yDesc, dyc, convDesc,
        algo_filter, filter_ws, sizeof(filter_ws), Ref{Float32}(0.0f0), dwDesc, dwc)

    CUDA.synchronize()
    println("CHECK_PRECISION_MATH_BACKEND algo_fwd=$(Int(algo_fwd)) " *
            "algo_data=$(Int(algo_data)) algo_filter=$(Int(algo_filter)) " *
            "fwd_ws=$(fwd_ws_size[]) data_ws=$(data_ws_size[]) " *
            "filter_ws=$(filter_ws_size[])")
    return Array(yc), Array(dxc), Array(dwc)
end

function classifier_matmul_item(w, x)
    w64 = Float64.(w)
    x64 = Float64.(x)
    y64 = w64 * x64
    y32 = CuArray(w) * CuArray(x)
    return check_item("classifier_matmul", y32, y64)
end

# End-to-end NNlib/cuDNN path: no pinned algorithm, so the process-wide
# chooser override decides. Stage-2 conv1 shape: (32,32,64,12) -> stride 2,
# pad 1, 64 input channels, 128 output channels, batch 12 (WHCN).
function nnlib_path_conv(x, w)
    y64 = Flux.CrossCor(Float64.(w), false; stride = 2, pad = 1)(Float64.(x))
    y32 = Flux.CrossCor(CuArray(w), false; stride = 2, pad = 1)(CuArray(x))
    CUDA.synchronize()
    return check_item("conv_nnlib_fwd", Array(y32), y64)
end

function algo_choice_line(choice)
    shape = choice.shape === nothing ? "NA" :
            join(string.(choice.shape), "x")
    return @sprintf("kind=%s algo=%d math_type=%s fallback=%s time=%.9g memory=%d shape=%s",
                    choice.kind, choice.algo, choice.math_type_label,
                    choice.fallback, choice.time, choice.memory, shape)
end

function print_algo_choices(prefix)
    choices = precision_algo_choices()
    for choice in choices
        println(prefix, " ", algo_choice_line(choice))
    end
    return choices
end

function main(args)
    mode = lowercase(getopt(args, "mode", "fma"))
    mode in ("fma", "current") ||
        error("--mode must be 'fma' or 'current', got '$(mode)'")

    cfg = configure_precision!(fma = mode == "fma")
    effective_conv1 = cudnn_effective_math_type(; pad = 1, stride = 2)
    effective_conv2 = cudnn_effective_math_type(; pad = 1, stride = 1)
    effective_proj = cudnn_effective_math_type(; pad = 0, stride = 2)
    expected = mode == "fma" ? cuDNN.CUDNN_FMA_MATH : cuDNN.CUDNN_DEFAULT_MATH

    println("CHECK_PRECISION_MATH_CONFIG mode=$(mode) " *
            "cublas_math_mode=$(cfg.cublas_math_mode) " *
            "cudnn_requested=$(math_type_label(cfg.cudnn_requested)) " *
            "cudnn_effective_conv1=$(math_type_label(effective_conv1)) " *
            "cudnn_effective_conv2=$(math_type_label(effective_conv2)) " *
            "cudnn_effective_proj=$(math_type_label(effective_proj)) " *
            "cuda_package_version=$(cfg.cuda_package_version) " *
            "cudnn_package_version=$(cfg.cudnn_package_version) " *
            "cudnn_library_version=$(cfg.cudnn_library_version) " *
            "cudnn_version=$(cfg.cudnn_version) " *
            "CUDNN_FMA_MATH=$(math_type_label(cuDNN.CUDNN_FMA_MATH))")
    if mode == "current"
        println("CHECK_PRECISION_MATH_MODE mode=current " *
                "note=\"negative control; a conv discrepancy is expected\"")
    end

    ok = true
    config_ok = cfg.cudnn_requested == expected &&
                effective_conv1 == expected &&
                effective_conv2 == expected &&
                effective_proj == expected
    println("CHECK_PRECISION_MATH_CONFIG_CHECK requested=$(math_type_label(cfg.cudnn_requested)) " *
            "expected=$(math_type_label(expected)) " *
            (config_ok ? "PASS" : "FAIL"))
    ok &= config_ok

    version_ok = cfg.versions_supported === true
    println("CHECK_PRECISION_MATH_VERSION " *
            "cuda_package_version=$(cfg.cuda_package_version) " *
            "cudnn_package_version=$(cfg.cudnn_package_version) " *
            "cudnn_library_version=$(cfg.cudnn_library_version) " *
            "versions_supported=$(cfg.versions_supported) " *
            (version_ok ? "PASS" : "FAIL"))
    ok &= version_ok

    Random.seed!(SEED)

    # Stage-2 conv2 shape: (32,32,128,12) -> (32,32,128,12), stride 1, pad 1.
    x = randn(Float32, 32, 32, 128, 12)
    w = randn(Float32, 3, 3, 128, 128) .* Float32(1 / sqrt(3 * 3 * 128))
    dy = randn(Float32, 32, 32, 128, 12)
    y64, gx64, gw64 = cpu_reference(x, w, dy)
    y32, gx32, gw32 = fixed_algo_conv(x, w, dy)
    ok &= check_item("conv_fixed_fwd", y32, y64)
    ok &= check_item("conv_fixed_dx", gx32, gx64)
    ok &= check_item("conv_fixed_dw", gw32, gw64)

    # Classifier-sized matrix multiplication (200x512 * 512x12), cuBLAS path.
    wc = randn(Float32, 200, 512) .* Float32(1 / sqrt(512))
    xc = randn(Float32, 512, 12)
    ok &= classifier_matmul_item(wc, xc)

    # End-to-end NNlib path (autotuned; exercises the chooser override).
    x_nn = randn(Float32, 32, 32, 64, 12)
    w_nn = randn(Float32, 3, 3, 64, 128) .* Float32(1 / sqrt(3 * 3 * 64))
    nnlib_ok = nnlib_path_conv(x_nn, w_nn)
    choices = print_algo_choices("CHECK_PRECISION_MATH_ALGO")
    if mode == "fma"
        choices_ok = !isempty(choices) &&
                     all(c -> c.math_type == Int(cuDNN.CUDNN_FMA_MATH) &&
                              !c.fallback, choices)
        @printf("CHECK_PRECISION_MATH_ALGO_CHOICES choices=%d fma=%d fallback=%d %s\n",
                length(choices),
                count(c -> c.math_type == Int(cuDNN.CUDNN_FMA_MATH), choices),
                count(c -> c.fallback, choices),
                choices_ok ? "PASS" : "FAIL")
        ok &= nnlib_ok
        ok &= choices_ok
    else
        # Negative control: print the NNlib-path result and choices for
        # diagnosis, but the deterministic fixed-algo path decides FAIL.
        println("CHECK_PRECISION_MATH_NNLIB_DIAG mode=current " *
                "nnlib_ok=$(nnlib_ok) choices=$(length(choices)) " *
                "note=\"diagnostic only; fixed-algo negative control decides\"")
    end

    if ok
        println("CHECK_PRECISION_MATH PASS")
        return 0
    end
    println("CHECK_PRECISION_MATH FAIL")
    return 1
end

exit(main(ARGS))
