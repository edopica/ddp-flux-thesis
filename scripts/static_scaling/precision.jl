"""
Reusable precision configuration for the static-scaling GPU path.

The accepted policy is FP32 with TF32 disabled: cuBLAS runs in
`CUDA.PEDANTIC_MATH`, and cuDNN convolutions request `CUDNN_FMA_MATH`
(FP32 multiply-add, no TF32 tensor-core inputs). CUDNN_FMA_MATH is the
concrete cuDNN realization of that policy. It is not a numerical
tolerance change and it does not alter the model math.

Process-wide overrides installed at module load
-----------------------------------------------

This module adds two process-wide method overrides to cuDNN.jl. Both are
defined when the module is loaded, before `configure_precision!` runs, so
there is no world-age window in which stock dispatch can be observed:

1. `cuDNN.math_mode(::CUDACore.MathMode)` (more specific than the stock
   `math_mode(::Any)` method). It reads the module-global
   `FMA_ENABLED::Ref{Bool}`:
   * `FMA_ENABLED[] == true`: PEDANTIC -> `CUDNN_FMA_MATH`,
     DEFAULT -> `CUDNN_TENSOR_OP_MATH`,
     FAST -> `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION`;
   * `FMA_ENABLED[] == false`: exactly the stock mapping
     (PEDANTIC -> `CUDNN_DEFAULT_MATH`, DEFAULT -> tensor-op,
     FAST -> tensor-op-allow-conversion).
   The stock `Any` method is never overwritten.

2. `cuDNN.cudnnConvolutionAlgoPerfChoose(convDesc, tensorDesc,
   perfResults, n)` for the forward, backward-data, and
   backward-filter performance-result types. The stock method skips the
   math-type check whenever the tensor dtype is Float32, so cuDNN can
   select a TF32-capable algorithm even when the descriptor requests
   `CUDNN_FMA_MATH`. The override reads the descriptor math type with
   `cudnnGetConvolutionMathType`; when `FMA_ENABLED[]` is true and that
   type is `CUDNN_FMA_MATH`, every candidate must report
   `ps.mathType == CUDNN_FMA_MATH`. Otherwise the stock Float32 rule is
   reproduced exactly. Selection order, status, memory, and the 10%
   time window are unchanged. If no candidate survives in FMA mode, the
   chooser falls back only to a successful non-tensor-core algorithm
   (`algo == 0` or `algo == 1`) and records `fallback=true`; when no such
   candidate exists it keeps cuDNN.jl's algo-0 backup and records
   `fallback=true, math_unknown=true`. It never silently accepts a TF32
   candidate. Every choice is recorded; setting
   `STATIC_SCALING_PRECISION_LOG_CANDIDATES=1` also records the candidate
   list. `precision_algo_choices()` returns a copy for acceptance logs.

Perf-cache clearing requirement
-------------------------------

The three cuDNN.jl algorithm performance caches
(`cudnnConvolutionFwdAlgoPerfCache`,
`cudnnConvolutionBwdDataAlgoPerfCache`,
`cudnnConvolutionBwdFilterAlgoPerfCache`) are keyed by descriptor tuples
that do **not** include the convolution math type. A cached algorithm
chosen under the stock mapping can therefore be reused after the
descriptor math type changes. `configure_precision!` clears all three
caches under their locks and verifies that they are empty. This must run
in a fresh process before any GPU work (model/array transfer, descriptor
construction, or convolution). Later calls with the same `fma` value
return the recorded configuration; calls with a different value throw.

Package-version assumptions
---------------------------

The overrides target the cuDNN.jl 6.2.2 interface shared by the pinned
Leonardo stack (CUDA.jl 6.2.2, cuDNN library 9.26.0):

* `cuDNN.math_mode(mode)` maps `CUDACore.MathMode` values to
  `cudnnMathType_t` values. The stock method is `math_mode(mode::Any)`.
* `cuDNN.CUDNN_FMA_MATH` exists (`cudnnMathType_t` value 3).
* `cuDNN.cudnnConvolutionDescriptor(pad, stride, dilation, mode,
  dataType, mathType, reorderType, groupCount)` is the memoized
  descriptor constructor used by NNlib's cuDNN extension.
* `cuDNN.cudnnConvolutionFwdAlgoPerf_t`,
  `cuDNN.cudnnConvolutionBwdDataAlgoPerf_t`, and
  `cuDNN.cudnnConvolutionBwdFilterAlgoPerf_t` expose `algo`, `status`,
  `time`, `memory`, and `mathType` fields.
* The three perf caches and their `ReentrantLock`s are module globals.

If any of these change, `configure_precision!` fails loudly instead of
silently falling back to the stock `PEDANTIC_MATH -> CUDNN_DEFAULT_MATH`
mapping, which leaves TF32 enabled on Ampere for FP32 convolutions.

Descriptor cache
----------------

`cudnn_effective_math_type` builds the descriptor with the same argument
order and values that NNlib uses, then queries it with
`cudnnGetConvolutionMathType`. cuDNN.jl memoizes descriptors by argument
tuple, which includes the math type. Query the effective type *before*
the real convolution in a fresh process so the cached descriptor is the
one NNlib reuses. A separate process is required for each precision
case: descriptors, algorithm caches, and the chooser override state are
process-wide.

No error is swallowed: CUDA availability, enum presence, and the
requested/effective math types are all checked and reported.
"""
module StaticScalingPrecision

import CUDA
import cuDNN

export configure_precision!, cudnn_requested_math_type,
       cudnn_effective_math_type, cudnn_version, math_type_label,
       precision_algo_choices, FMA_ENABLED

const CUDACore = cuDNN.CUDACore

# Set exactly once by `configure_precision!`. The module-load method
# overrides below read this flag on every call.
const FMA_ENABLED = Ref(false)

const _configured = Ref(false)
const _configuration = Ref{Any}(nothing)

const _ALGO_CHOICES = Any[]
const _ALGO_CHOICES_LOCK = ReentrantLock()

# Only the number of dimensions of `x` influences cuDNN.convdims; the
# stage-2 diagnostic is a 4-D WHCN convolution. Values are descriptive.
const _X_SIZE = (32, 32, 64, 12)

"""
    math_type_label(t) -> String

Human-readable cuDNN math type with its numeric value, for log lines.
"""
function math_type_label(t)
    t == cuDNN.CUDNN_DEFAULT_MATH &&
        return "CUDNN_DEFAULT_MATH($(Int(t)))"
    t == cuDNN.CUDNN_TENSOR_OP_MATH &&
        return "CUDNN_TENSOR_OP_MATH($(Int(t)))"
    t == cuDNN.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION &&
        return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION($(Int(t)))"
    if isdefined(cuDNN, :CUDNN_FMA_MATH) && t == cuDNN.CUDNN_FMA_MATH
        return "CUDNN_FMA_MATH($(Int(t)))"
    end
    return "UNKNOWN($(Int(t)))"
end

"""
    cudnn_version() -> VersionNumber

Directory-free cuDNN library version from `cuDNN.version()`.
"""
cudnn_version() = cuDNN.version()

# ---------------------------------------------------------------------------
# Process-wide override 1: CUDA math mode -> cuDNN math type
#
# Defined at module load, more specific than cuDNN.jl's stock
# `math_mode(mode::Any)` method. The stock method is left untouched.
# ---------------------------------------------------------------------------

function cuDNN.math_mode(mode::CUDACore.MathMode)
    if FMA_ENABLED[]
        if mode == CUDACore.PEDANTIC_MATH
            return cuDNN.CUDNN_FMA_MATH
        elseif mode == CUDACore.DEFAULT_MATH
            return cuDNN.CUDNN_TENSOR_OP_MATH
        elseif mode == CUDACore.FAST_MATH
            return cuDNN.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
        end
    else
        # Stock cuDNN.jl 6.2.2 mapping, reproduced exactly.
        if mode == CUDACore.PEDANTIC_MATH
            return cuDNN.CUDNN_DEFAULT_MATH
        elseif mode == CUDACore.DEFAULT_MATH
            return cuDNN.CUDNN_TENSOR_OP_MATH
        elseif mode == CUDACore.FAST_MATH
            return cuDNN.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
        end
    end
    throw(ArgumentError("unsupported CUDA math mode"))
end

"""
    cudnn_requested_math_type()

The `cudnnMathType_t` that the current process requests for new cuDNN
convolution descriptors. With the FMA mapping installed this is
`CUDNN_FMA_MATH` under `CUDA.PEDANTIC_MATH`.

Uses `invokelatest` for safety even though the mapping method is
installed at module load.
"""
cudnn_requested_math_type() = Base.invokelatest(cuDNN.math_mode)

# ---------------------------------------------------------------------------
# Process-wide override 2: algorithm selection honors the FMA math type
#
# Defined at module load for the three convolution performance-result
# types. The stock untyped method remains available for any other type.
# ---------------------------------------------------------------------------

function _log_candidates()
    return get(ENV, "STATIC_SCALING_PRECISION_LOG_CANDIDATES", "") == "1"
end

function _candidate_log(perfResults, n)
    _log_candidates() || return nothing
    candidates = Any[]
    for (i, ps) in enumerate(perfResults)
        i > n && break
        push!(candidates, (
            algo = Int(ps.algo),
            status = Int(ps.status),
            math_type = Int(ps.mathType),
            time = Float64(ps.time),
            memory = Int(ps.memory),
        ))
    end
    return candidates
end

function _record_algo_choice(entry)
    lock(_ALGO_CHOICES_LOCK) do
        push!(_ALGO_CHOICES, entry)
    end
    return nothing
end

"""
    precision_algo_choices() -> Vector

Snapshot of every convolution algorithm choice made through the
process-wide chooser override. Each entry is a named tuple with fields
`kind` (`"fwd"`, `"bwd_data"`, or `"bwd_filter"`), `algo`, `math_type`,
`math_type_label`, `math_type_expected`, `fallback`, `math_unknown`,
`status`, `time`, `memory`, `shape`, `n`, and `candidates` (the last is
`nothing` unless `STATIC_SCALING_PRECISION_LOG_CANDIDATES=1`).
"""
function precision_algo_choices()
    return lock(_ALGO_CHOICES_LOCK) do
        copy(_ALGO_CHOICES)
    end
end

# First successful candidate among the known non-tensor-core FP32
# algorithms. cuDNN returns candidates sorted by time, so the first match
# is the fastest of these.
function _first_fallback_candidate(perfResults, n)
    for (i, ps) in enumerate(perfResults)
        i > n && break
        ps.status == cuDNN.CUDNN_STATUS_SUCCESS || continue
        Int(ps.algo) in (0, 1) || continue
        return i
    end
    return nothing
end

function _choose_and_record(kind::AbstractString, convDesc, tensorDesc,
                            perfResults, n)
    descriptor_math_type = Ref{cuDNN.cudnnMathType_t}(cuDNN.CUDNN_DEFAULT_MATH)
    cuDNN.cudnnGetConvolutionMathType(convDesc, descriptor_math_type)
    dtype, dims, _ = cuDNN.cudnnGetTensorDescriptor(tensorDesc)

    strict_fma = FMA_ENABLED[] &&
                 descriptor_math_type[] == cuDNN.CUDNN_FMA_MATH
    # Stock cuDNN.jl rule: Float32 tensors skip the math-type check.
    skip_math_type_check = dtype == Float32

    candidates = _candidate_log(perfResults, n)

    ibest, mbest, tbest = 0, Inf, Inf
    for (i, ps) in enumerate(perfResults)
        i > n && break
        math_ok = strict_fma ? (ps.mathType == cuDNN.CUDNN_FMA_MATH) :
                  (skip_math_type_check ||
                   ps.mathType == descriptor_math_type[])
        # Same selection rule as cuDNN.jl 6.2.2.
        if math_ok && ps.status == cuDNN.CUDNN_STATUS_SUCCESS &&
           ps.memory < mbest && ps.time < tbest * 1.1
            ibest, mbest, tbest = i, ps.memory, ps.time
        end
    end

    fallback = false
    math_unknown = false
    if ibest == 0
        if FMA_ENABLED[]
            ibest = _first_fallback_candidate(perfResults, n)
            if ibest !== nothing
                fallback = true
                chosen_algo = Int(perfResults[ibest].algo)
                @warn "No FMA-matching cuDNN algorithm; using successful " *
                      "non-tensor-core fallback" kind = kind algo = chosen_algo
            else
                ibest = findfirst(p -> p.algo == 0, perfResults)
                ibest === nothing &&
                    error("Cannot find backup algorithm for convolution, giving up.")
                fallback = true
                math_unknown = true
                @warn "No FMA-matching cuDNN algorithm and no successful " *
                      "algo 0/1 fallback; using stock algo-0 backup " *
                      "without status or math-type verification" kind = kind
            end
        else
            @warn "No valid algorithm found, probably bad params for convolution." maxlog=1
            ibest = findfirst(p -> p.algo == 0, perfResults)
            ibest === nothing &&
                error("Cannot find backup algorithm for convolution, giving up.")
            fallback = true
            math_unknown = true
        end
    end

    if candidates !== nothing
        detail = join(("algo=$(c.algo):math_type=$(c.math_type):status=$(c.status):" *
                       "time=$(c.time):memory=$(c.memory)" for c in candidates), ",")
        println("PRECISION_ALGO_CANDIDATES kind=$(kind) n=$(length(candidates)) ",
                detail)
    end

    chosen = perfResults[ibest]
    _record_algo_choice((
        kind = String(kind),
        algo = Int(chosen.algo),
        math_type = Int(chosen.mathType),
        math_type_label = math_type_label(chosen.mathType),
        math_type_expected = Int(descriptor_math_type[]),
        fallback = fallback,
        math_unknown = math_unknown,
        status = Int(chosen.status),
        time = Float64(chosen.time),
        memory = Int(chosen.memory),
        shape = dims,
        n = Int(n),
        candidates = candidates,
    ))
    return chosen
end

function cuDNN.cudnnConvolutionAlgoPerfChoose(
        convDesc::cuDNN.cudnnConvolutionDescriptor,
        tensorDesc::cuDNN.cudnnTensorDescriptor,
        perfResults::AbstractVector{cuDNN.cudnnConvolutionFwdAlgoPerf_t},
        n)
    return _choose_and_record("fwd", convDesc, tensorDesc, perfResults, n)
end

function cuDNN.cudnnConvolutionAlgoPerfChoose(
        convDesc::cuDNN.cudnnConvolutionDescriptor,
        tensorDesc::cuDNN.cudnnTensorDescriptor,
        perfResults::AbstractVector{cuDNN.cudnnConvolutionBwdDataAlgoPerf_t},
        n)
    return _choose_and_record("bwd_data", convDesc, tensorDesc, perfResults, n)
end

function cuDNN.cudnnConvolutionAlgoPerfChoose(
        convDesc::cuDNN.cudnnConvolutionDescriptor,
        tensorDesc::cuDNN.cudnnTensorDescriptor,
        perfResults::AbstractVector{cuDNN.cudnnConvolutionBwdFilterAlgoPerf_t},
        n)
    return _choose_and_record("bwd_filter", convDesc, tensorDesc, perfResults, n)
end

# ---------------------------------------------------------------------------
# Descriptor and configuration helpers
# ---------------------------------------------------------------------------

function _cache_length(cache_lock, cache)
    return lock(cache_lock) do
        length(cache)
    end
end

function _clear_algo_perf_caches()
    before = (
        fwd = _cache_length(cuDNN.cudnnConvolutionFwdAlgoPerfCacheLock,
                            cuDNN.cudnnConvolutionFwdAlgoPerfCache),
        bwd_data = _cache_length(cuDNN.cudnnConvolutionBwdDataAlgoPerfCacheLock,
                                 cuDNN.cudnnConvolutionBwdDataAlgoPerfCache),
        bwd_filter = _cache_length(cuDNN.cudnnConvolutionBwdFilterAlgoPerfCacheLock,
                                   cuDNN.cudnnConvolutionBwdFilterAlgoPerfCache),
    )
    lock(cuDNN.cudnnConvolutionFwdAlgoPerfCacheLock) do
        empty!(cuDNN.cudnnConvolutionFwdAlgoPerfCache)
    end
    lock(cuDNN.cudnnConvolutionBwdDataAlgoPerfCacheLock) do
        empty!(cuDNN.cudnnConvolutionBwdDataAlgoPerfCache)
    end
    lock(cuDNN.cudnnConvolutionBwdFilterAlgoPerfCacheLock) do
        empty!(cuDNN.cudnnConvolutionBwdFilterAlgoPerfCache)
    end
    after = (
        fwd = _cache_length(cuDNN.cudnnConvolutionFwdAlgoPerfCacheLock,
                            cuDNN.cudnnConvolutionFwdAlgoPerfCache),
        bwd_data = _cache_length(cuDNN.cudnnConvolutionBwdDataAlgoPerfCacheLock,
                                 cuDNN.cudnnConvolutionBwdDataAlgoPerfCache),
        bwd_filter = _cache_length(cuDNN.cudnnConvolutionBwdFilterAlgoPerfCacheLock,
                                   cuDNN.cudnnConvolutionBwdFilterAlgoPerfCache),
    )
    (after.fwd == 0 && after.bwd_data == 0 && after.bwd_filter == 0) ||
        error("failed to clear cuDNN algorithm perf caches: " *
              "before=$(before) after=$(after)")
    return (before = before, cleared = true)
end

"""
    cudnn_effective_math_type(; pad=1, stride=2, dilation=1, group=1)

Build (or fetch from the cuDNN.jl descriptor cache) the 4-D
cross-correlation descriptor that NNlib uses for a Flux `CrossCor` with
these parameters, then return the math type recorded in it via
`cudnnGetConvolutionMathType`.

Argument order matches NNlib's cuDNN extension:
`pad, stride, dilation, mode, dataType, mathType, reorderType, groupCount`.
Call this before the real convolution in a fresh process.
"""
function cudnn_effective_math_type(; pad::Integer = 1, stride::Integer = 2,
                                   dilation::Integer = 1, group::Integer = 1)
    requested = cudnn_requested_math_type()
    desc = cuDNN.cudnnConvolutionDescriptor(
        cuDNN.convdims(pad, _X_SIZE, 0),
        cuDNN.convdims(stride, _X_SIZE, 1),
        cuDNN.convdims(dilation, _X_SIZE, 1),
        cuDNN.CUDNN_CROSS_CORRELATION,
        cuDNN.cudnnDataType(Float32),
        requested,
        cuDNN.CUDNN_DEFAULT_REORDER,
        Cint(group),
    )
    effective = Ref{cuDNN.cudnnMathType_t}(cuDNN.CUDNN_DEFAULT_MATH)
    cuDNN.cudnnGetConvolutionMathType(desc, effective)
    return effective[]
end

"""
    configure_precision!(; fma=true) -> NamedTuple

Configure the process for FP32/TF32-off GPU work:

1. set the module-global `FMA_ENABLED[]` exactly once (conflicting
   reconfiguration throws);
2. `CUDA.allowscalar(false)`;
3. `CUDA.math_mode!(CUDA.PEDANTIC_MATH)` for cuBLAS;
4. clear the three cuDNN convolution algorithm performance caches under
   their locks (their keys omit the descriptor math type) and verify
   they are empty;
5. verify `CUDNN_FMA_MATH` exists, CUDA is functional, and the requested
   and effective descriptor math types match the case.

The process-wide `math_mode` and algorithm-chooser overrides are
installed at module load, so they are active before this function runs.

Must be called once per process before any GPU work. Returns
`(fma, cublas_math_mode, cudnn_requested, cudnn_effective, cudnn_version,
caches_before, caches_cleared)`. With `fma=false` the stock mapping is
used, so `PEDANTIC_MATH` requests `CUDNN_DEFAULT_MATH`; that is the
negative control, not the accepted policy.
"""
function configure_precision!(; fma::Bool = true)
    if _configured[]
        existing = _configuration[]
        existing.fma == fma ||
            error("configure_precision! already configured fma=$(existing.fma); " *
                  "refusing fma=$(fma) in the same process")
        return existing
    end

    isdefined(cuDNN, :CUDNN_FMA_MATH) ||
        error("cuDNN.CUDNN_FMA_MATH is not defined; cannot request FP32 FMA math")
    CUDA.functional() ||
        error("CUDA.functional() is false; precision configuration needs a functional GPU")

    FMA_ENABLED[] = fma
    CUDA.allowscalar(false)
    CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    cache_info = _clear_algo_perf_caches()

    requested = cudnn_requested_math_type()
    effective = cudnn_effective_math_type()
    expected = fma ? cuDNN.CUDNN_FMA_MATH : cuDNN.CUDNN_DEFAULT_MATH
    requested == expected ||
        error("cuDNN requested math type $(math_type_label(requested)); " *
              "expected $(math_type_label(expected))")
    effective == expected ||
        error("cuDNN effective descriptor math type $(math_type_label(effective)); " *
              "expected $(math_type_label(expected))")

    configuration = (
        fma = fma,
        cublas_math_mode = CUDA.math_mode(),
        cudnn_requested = requested,
        cudnn_effective = effective,
        cudnn_version = cudnn_version(),
        caches_before = cache_info.before,
        caches_cleared = cache_info.cleared,
    )
    _configuration[] = configuration
    _configured[] = true
    return configuration
end

end # module StaticScalingPrecision
