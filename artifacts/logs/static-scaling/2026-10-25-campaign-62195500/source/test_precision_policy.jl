"""
Focused, GPU-free regression for the static-scaling precision policy.

This script includes `precision.jl`, imports CUDA.jl/cuDNN.jl only to use
their enums, and drives the pure candidate-selection, descriptor-policy,
and version-gate helpers with synthetic named-tuple performance results.
It covers ten cases and never calls `configure_precision!`, so no GPU or
cuDNN handle is required.

Prints one `CHECK_PRECISION_POLICY` line per case and ends with
`CHECK_PRECISION_POLICY PASS` or `CHECK_PRECISION_POLICY FAIL`; exits
0 only on PASS.
"""

include(joinpath(@__DIR__, "precision.jl"))
using .StaticScalingPrecision
import CUDA
import cuDNN

const SP = StaticScalingPrecision
const SUCCESS = cuDNN.CUDNN_STATUS_SUCCESS
const FAILED = cuDNN.CUDNN_STATUS_BAD_PARAM
const FMA = cuDNN.CUDNN_FMA_MATH
const DEFAULT = cuDNN.CUDNN_DEFAULT_MATH

candidate(; algo, status = SUCCESS, mathType, time, memory) =
    (algo = algo, status = status, mathType = mathType, time = time,
     memory = memory)

function caught_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

function check_case(name, f)
    passed = try
        f()
    catch err
        println("CHECK_PRECISION_POLICY case=$(name) FAIL " *
                "exception=$(sprint(showerror, err))")
        return false
    end
    println("CHECK_PRECISION_POLICY case=$(name) $(passed ? "PASS" : "FAIL")")
    return passed
end

function case_strict_prefers_fma()
    perf = [
        candidate(algo = 1, mathType = DEFAULT, time = 1.0, memory = 100),
        candidate(algo = 7, mathType = FMA, time = 2.0, memory = 100),
    ]
    n = length(perf)
    selected = SP._select_candidate(perf, n;
                                    strict_fma = true,
                                    descriptor_math_type = FMA,
                                    skip_math_type_check = true)
    selected == 2 || return false
    ibest, fallback, math_unknown = SP._resolve_candidate(
        perf, n, selected; strict_fma = true, kind = "strict_faster_non_fma")
    return ibest == 2 && !fallback && !math_unknown &&
           perf[ibest].mathType == FMA
end

function case_strict_rejects_non_fma()
    perf = [
        candidate(algo = 0, mathType = DEFAULT, time = 1.0, memory = 100),
        candidate(algo = 1, mathType = DEFAULT, time = 2.0, memory = 100),
    ]
    n = length(perf)
    selected = SP._select_candidate(perf, n;
                                    strict_fma = true,
                                    descriptor_math_type = FMA,
                                    skip_math_type_check = false)
    selected == 0 || return false
    err = caught_error(() -> SP._resolve_candidate(
        perf, n, selected; strict_fma = true, kind = "only_non_fma"))
    err isa SP.PrecisionPolicyError || return false
    msg = sprint(showerror, err)
    return occursin("only_non_fma", msg) && occursin("algo=0", msg) &&
           occursin("algo=1", msg) && occursin("mathType=", msg) &&
           occursin("status=", msg)
end

function case_strict_rejects_all_failed()
    perf = [
        candidate(algo = 0, status = FAILED, mathType = FMA,
                  time = 1.0, memory = 100),
        candidate(algo = 1, status = FAILED, mathType = DEFAULT,
                  time = 2.0, memory = 100),
    ]
    n = length(perf)
    selected = SP._select_candidate(perf, n;
                                    strict_fma = true,
                                    descriptor_math_type = FMA,
                                    skip_math_type_check = false)
    selected == 0 || return false
    err = caught_error(() -> SP._resolve_candidate(
        perf, n, selected; strict_fma = true, kind = "all_failed"))
    err isa SP.PrecisionPolicyError || return false
    msg = sprint(showerror, err)
    return occursin("all_failed", msg) && occursin("status=$(Int(FAILED))", msg)
end

function case_stock_float32_selection()
    perf = [
        candidate(algo = 0, mathType = DEFAULT, time = 1.0, memory = 100),
        candidate(algo = 4, mathType = FMA, time = 2.0, memory = 200),
    ]
    n = length(perf)
    selected = SP._select_candidate(perf, n;
                                    strict_fma = false,
                                    descriptor_math_type = FMA,
                                    skip_math_type_check = true)
    selected == 1 || return false
    ibest, fallback, math_unknown = SP._resolve_candidate(
        perf, n, selected; strict_fma = false, kind = "stock_float32")
    return ibest == 1 && !fallback && !math_unknown &&
           perf[ibest].mathType == DEFAULT
end

function case_versions_supported()
    supported, cuda_v, cudnn_v = SP._check_versions(v"6.2.2", v"6.2.2")
    return supported === true && cuda_v == v"6.2.2" && cudnn_v == v"6.2.2"
end

function case_versions_unsupported_gate()
    err = caught_error(() -> SP._check_versions(v"6.2.2", v"6.2.1"))
    err isa SP.PrecisionPolicyError || return false
    supported, cuda_v, cudnn_v =
        SP._check_versions(v"6.2.1", v"6.2.2";
                           allow_unsupported_versions = true)
    return supported === false && cuda_v == v"6.2.1" && cudnn_v == v"6.2.2"
end

function case_strict_resolve_never_returns_non_fma()
    perf = [
        candidate(algo = 0, mathType = DEFAULT, time = 1.0, memory = 100),
        candidate(algo = 2, mathType = FMA, time = 2.0, memory = 100),
    ]
    n = length(perf)
    for i in eachindex(perf)
        perf[i].mathType == FMA && continue
        err = caught_error(() -> SP._resolve_candidate(
            perf, n, i; strict_fma = true, kind = "resolve_probe"))
        err isa SP.PrecisionPolicyError || return false
    end

    selected = SP._select_candidate(perf, n;
                                    strict_fma = true,
                                    descriptor_math_type = FMA,
                                    skip_math_type_check = true)
    selected == 2 || return false
    ibest, fallback, math_unknown = SP._resolve_candidate(
        perf, n, selected; strict_fma = true, kind = "resolve_probe")
    return ibest == 2 && !fallback && !math_unknown &&
           perf[ibest].mathType == FMA
end

function case_stock_algo0_fallback()
    perf = [
        candidate(algo = 0, status = FAILED, mathType = DEFAULT,
                  time = 1.0, memory = 100),
        candidate(algo = 1, status = FAILED, mathType = DEFAULT,
                  time = 2.0, memory = 100),
    ]
    n = length(perf)
    selected = SP._select_candidate(perf, n;
                                    strict_fma = false,
                                    descriptor_math_type = DEFAULT,
                                    skip_math_type_check = true)
    selected == 0 || return false
    ibest, fallback, math_unknown = SP._resolve_candidate(
        perf, n, selected; strict_fma = false, kind = "stock_algo0_fallback")
    return ibest == 1 && fallback === true && math_unknown === true &&
           perf[ibest].algo == 0
end

function case_n_truncation()
    perf = [
        candidate(algo = 1, mathType = DEFAULT, time = 1.0, memory = 100),
        candidate(algo = 2, mathType = FMA, time = 2.0, memory = 100),
        candidate(algo = 3, mathType = FMA, time = 0.1, memory = 50),
    ]
    selected2 = SP._select_candidate(perf, 2;
                                     strict_fma = true,
                                     descriptor_math_type = FMA,
                                     skip_math_type_check = true)
    selected3 = SP._select_candidate(perf, 3;
                                     strict_fma = true,
                                     descriptor_math_type = FMA,
                                     skip_math_type_check = true)
    selected2 == 2 || return false
    selected3 == 3 || return false
    ibest, fallback, math_unknown = SP._resolve_candidate(
        perf, 2, selected2; strict_fma = true, kind = "n_truncation")
    return ibest == 2 && !fallback && !math_unknown && ibest != 3
end

function case_descriptor_policy()
    SP._check_descriptor_math_type(FMA; strict_fma = true) === nothing ||
        return false
    err = caught_error(() -> SP._check_descriptor_math_type(
        DEFAULT; strict_fma = true))
    err isa SP.PrecisionPolicyError || return false
    SP._check_descriptor_math_type(DEFAULT; strict_fma = false) === nothing ||
        return false
    return true
end

ok = true
ok &= check_case("strict_fma_prefers_fma", case_strict_prefers_fma)
ok &= check_case("strict_fma_rejects_non_fma", case_strict_rejects_non_fma)
ok &= check_case("strict_fma_rejects_all_failed", case_strict_rejects_all_failed)
ok &= check_case("stock_float32_selection", case_stock_float32_selection)
ok &= check_case("stock_algo0_fallback", case_stock_algo0_fallback)
ok &= check_case("n_truncation", case_n_truncation)
ok &= check_case("versions_supported", case_versions_supported)
ok &= check_case("versions_unsupported_gate", case_versions_unsupported_gate)
ok &= check_case("strict_resolve_never_returns_non_fma",
                 case_strict_resolve_never_returns_non_fma)
ok &= check_case("descriptor_policy", case_descriptor_policy)

if ok
    println("CHECK_PRECISION_POLICY PASS")
    exit(0)
end
println("CHECK_PRECISION_POLICY FAIL")
exit(1)
