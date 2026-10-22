"""
Reusable precision configuration for the static-scaling GPU path.

The accepted policy is FP32 with TF32 disabled: cuBLAS runs in
`CUDA.PEDANTIC_MATH`, and cuDNN convolutions request `CUDNN_FMA_MATH`
(FP32 multiply-add, no TF32 tensor-core inputs). CUDNN_FMA_MATH is the
concrete cuDNN realization of that policy. It is not a numerical
tolerance change and it does not alter the model math.

Required initialization
-----------------------

Every process must call `configure_precision!` once before any
convolution work, and before moving a model or arrays to the GPU or
building cuDNN descriptors. The algorithm-chooser override throws
`PrecisionPolicyError` if a convolution runs before that call. The
public `cudnn_effective_math_type` helper also requires configuration.
`configure_precision!` rejects unsupported CUDA.jl/cuDNN.jl package
versions, checks that the installed chooser methods are this module's
methods on both the abstract and concrete call signatures, sets the
process math state, and clears the algorithm caches. Cached calls
re-validate the override methods and the process math state. A failure
in the state-mutating configuration section is permanent for the
process: later calls throw and the caller must start a fresh process.

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
   `cudnnGetConvolutionMathType`. With the committed configuration
   `fma=true` (and `FMA_ENABLED[]` matching) it is fail-closed:

   * the descriptor math type must be `CUDNN_FMA_MATH`, otherwise it
     throws `PrecisionPolicyError` before selecting;
   * every candidate must report `ps.mathType == CUDNN_FMA_MATH`;
   * if no successful FMA candidate exists, it throws
     `PrecisionPolicyError` with the kind and each candidate's
     algo/mathType/status;
   * it never returns a non-FMA candidate and never records a fallback.

   With the committed configuration `fma=false` the stock cuDNN.jl
   6.2.2 rule is reproduced exactly: Float32 tensors skip the math-type check,
   success/status, memory, and the 10% time window are unchanged, and
   the stock algo-0 backup and warnings are retained. Every choice is
   recorded; setting `STATIC_SCALING_PRECISION_LOG_CANDIDATES=1` also
   records the candidate list. `precision_algo_choices()` returns a copy
   for acceptance logs.

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
in a fresh process before any GPU work. Later calls with the same `fma`
value re-validate the override methods and process math state and return
the recorded configuration; calls with a different value throw. A cached
configuration recorded with unsupported package versions is rejected
unless `allow_unsupported_versions=true`.

Package versions and override revalidation
------------------------------------------

The supported and enforced Julia package versions are CUDA.jl 6.2.2 and
cuDNN.jl 6.2.2, the pinned Leonardo/flux-integration runtime. Package
versions are read with `Base.pkgversion`; the loaded cuDNN library
version is recorded separately from `cuDNN.version()`. An unsupported
pair throws `PrecisionPolicyError` before any state changes. The
explicit diagnostic escape hatch is
`allow_unsupported_versions=true`, which emits a warning and records
`versions_supported=false`; it is not an accepted production path.

`_check_override_methods` uses `which` to verify that
`cuDNN.math_mode(::CUDACore.MathMode)` and all three
`cuDNN.cudnnConvolutionAlgoPerfChoose` methods for the forward,
backward-data, and backward-filter performance vectors resolve to
methods defined by this module. It checks both the abstract
`AbstractVector{T}` signatures and the concrete
`Vector{T}`/`Cint` call signatures, so a future more-specific method
cannot silently bypass the override. A changed dispatch target throws
`PrecisionPolicyError`.

Revalidation path for a future stack: update
`SUPPORTED_CUDA_VERSION`/`SUPPORTED_CUDNN_VERSION`, review the interface
assumptions below, rerun `test_precision_policy.jl` and the GPU
precision regression, then record the new stack. Use
`allow_unsupported_versions=true` only for a clearly marked diagnostic
run.

Interface assumptions (cuDNN.jl 6.2.2)
--------------------------------------

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

The public `cudnn_effective_math_type` requires `configure_precision!`
and delegates to `_cudnn_effective_math_type`. The internal helper
builds the descriptor with the same argument order and values that NNlib
uses, then queries it with `cudnnGetConvolutionMathType`. cuDNN.jl
memoizes descriptors by argument tuple, which includes the math type.
Query the effective type *before* the real convolution in a fresh process
so the cached descriptor is the one NNlib reuses. `configure_precision!`
uses the internal helper because it validates the descriptor before
`_configured` is set. A separate process is required for each precision
case: descriptors, algorithm caches, and the chooser override state are
process-wide.

No error is swallowed: CUDA availability, package versions, override
methods, enum presence, cache clearing, and the requested/effective math
types are all checked and reported.
"""

module StaticScalingPrecision

import CUDA
import cuDNN

export configure_precision!, cudnn_requested_math_type,
       cudnn_effective_math_type, cudnn_version, math_type_label,
       precision_algo_choices, PrecisionPolicyError

const CUDACore = cuDNN.CUDACore

const SUPPORTED_CUDA_VERSION = v"6.2.2"
const SUPPORTED_CUDNN_VERSION = v"6.2.2"

struct PrecisionPolicyError <: Exception
    msg::String
end

Base.showerror(io::IO, err::PrecisionPolicyError) =
    print(io, "PrecisionPolicyError: ", err.msg)

# Set exactly once by `configure_precision!`. The module-load method
# overrides below read this flag on every call.
const FMA_ENABLED = Ref(false)

const _configured = Ref(false)
const _CONFIGURE_FAILED = Ref{Any}(nothing)
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

"""
    _package_versions() -> NamedTuple

VersionNumbers of the loaded CUDA.jl and cuDNN.jl packages, read with
`Base.pkgversion`. Throws `PrecisionPolicyError` if either cannot be
determined.
"""
function _package_versions()
    cuda_v = Base.pkgversion(CUDA)
    cudnn_v = Base.pkgversion(cuDNN)
    if cuda_v === nothing || cudnn_v === nothing
        throw(PrecisionPolicyError(
            "cannot determine CUDA.jl/cuDNN.jl package versions: " *
            "Base.pkgversion(CUDA)=$(cuda_v), " *
            "Base.pkgversion(cuDNN)=$(cudnn_v); load both packages from " *
            "installed package environments"))
    end
    return (cuda = cuda_v, cudnn = cudnn_v)
end

"""
    _check_versions(cuda_v, cudnn_v; allow_unsupported_versions=false)
        -> (supported, cuda, cudnn)

Enforce the pinned CUDA.jl 6.2.2 / cuDNN.jl 6.2.2 package pair. An
unsupported pair throws `PrecisionPolicyError` unless
`allow_unsupported_versions=true`; that diagnostic escape hatch emits an
explicit warning and returns `supported=false`.
"""
function _check_versions(cuda_v::VersionNumber, cudnn_v::VersionNumber;
                         allow_unsupported_versions::Bool = false)
    supported = cuda_v == SUPPORTED_CUDA_VERSION &&
                cudnn_v == SUPPORTED_CUDNN_VERSION
    if !supported
        msg = "unsupported CUDA.jl/cuDNN.jl package versions: " *
              "CUDA.jl=$(cuda_v) (supported $(SUPPORTED_CUDA_VERSION)), " *
              "cuDNN.jl=$(cudnn_v) (supported $(SUPPORTED_CUDNN_VERSION))"
        allow_unsupported_versions ||
            throw(PrecisionPolicyError(
                msg * "; pass allow_unsupported_versions=true only for a " *
                "marked diagnostic run"))
        @warn "StaticScalingPrecision: continuing with unsupported package versions" *
              " because allow_unsupported_versions=true" package_versions = msg supported = false
    end
    return (supported, cuda_v, cudnn_v)
end

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

function _candidate_description(ps)
    return "algo=$(Int(ps.algo)) mathType=$(math_type_label(ps.mathType)) " *
           "status=$(Int(ps.status))"
end

function _candidate_listing(perfResults, n)
    parts = String[]
    for (i, ps) in enumerate(perfResults)
        i > n && break
        push!(parts, "candidate=$(i) " * _candidate_description(ps))
    end
    return isempty(parts) ? "no candidates" : join(parts, "; ")
end

"""
    _check_descriptor_math_type(descriptor_math_type; strict_fma)

Throw `PrecisionPolicyError` when strict FMA is enabled and the
descriptor math type is not `CUDNN_FMA_MATH`; otherwise return `nothing`.
"""
function _check_descriptor_math_type(descriptor_math_type; strict_fma::Bool)
    if strict_fma && descriptor_math_type != cuDNN.CUDNN_FMA_MATH
        throw(PrecisionPolicyError(
            "FMA precision policy is enabled but the cuDNN convolution " *
            "descriptor math type is $(math_type_label(descriptor_math_type)); " *
            "expected $(math_type_label(cuDNN.CUDNN_FMA_MATH))"))
    end
    return nothing
end

"""
    _select_candidate(perfResults, n; strict_fma, descriptor_math_type,
                      skip_math_type_check) -> Int

Pure candidate-selection rule. Returns the 1-based index of the chosen
performance result, or 0 when none qualifies. With `strict_fma=true`,
only successful candidates whose `mathType` is `CUDNN_FMA_MATH` are
eligible; `skip_math_type_check` is ignored. With `strict_fma=false` the
stock cuDNN.jl 6.2.2 rule is used: Float32 tensors skip the math-type
check, and the memory/10% time-window rule is unchanged.
"""
function _select_candidate(perfResults, n; strict_fma::Bool,
                           descriptor_math_type, skip_math_type_check::Bool)
    ibest, mbest, tbest = 0, Inf, Inf
    for (i, ps) in enumerate(perfResults)
        i > n && break
        math_ok = strict_fma ? (ps.mathType == cuDNN.CUDNN_FMA_MATH) :
                  (skip_math_type_check || ps.mathType == descriptor_math_type)
        # Same selection rule as cuDNN.jl 6.2.2.
        if math_ok && ps.status == cuDNN.CUDNN_STATUS_SUCCESS &&
           ps.memory < mbest && ps.time < tbest * 1.1
            ibest, mbest, tbest = i, ps.memory, ps.time
        end
    end
    return ibest
end

"""
    _resolve_candidate(perfResults, n, ibest; strict_fma, kind)
        -> (ibest, fallback, math_unknown)

Apply the policy to a selected index. A positive index is checked for
bounds and successful status; under strict FMA it must also report
`CUDNN_FMA_MATH`. Under strict FMA, `ibest=0` and any non-FMA selected
candidate throw `PrecisionPolicyError`; a returned candidate is always
`CUDNN_FMA_MATH` with `fallback=false`. Otherwise the stock algo-0
backup and warnings are reproduced exactly.
"""
function _resolve_candidate(perfResults, n, ibest; strict_fma::Bool, kind)
    if ibest > 0
        if ibest > n
            throw(PrecisionPolicyError(
                "candidate index $(ibest) is outside 1:$(n) for kind=$(kind); " *
                _candidate_listing(perfResults, n)))
        end
        chosen = perfResults[ibest]
        if chosen.status != cuDNN.CUDNN_STATUS_SUCCESS
            throw(PrecisionPolicyError(
                "selected cuDNN candidate is not successful for kind=$(kind): " *
                _candidate_description(chosen) * "; " *
                _candidate_listing(perfResults, n)))
        end
        if strict_fma && chosen.mathType != cuDNN.CUDNN_FMA_MATH
            throw(PrecisionPolicyError(
                "strict FMA policy rejected a non-FMA cuDNN algorithm for " *
                "kind=$(kind): $(_candidate_description(chosen)); " *
                _candidate_listing(perfResults, n)))
        end
        return (ibest, false, false)
    end

    if strict_fma
        throw(PrecisionPolicyError(
            "strict FMA policy found no successful CUDNN_FMA_MATH candidate " *
            "for kind=$(kind); " * _candidate_listing(perfResults, n)))
    end

    fallback = false
    math_unknown = false
    @warn "No valid algorithm found, probably bad params for convolution." maxlog=1
    ibest = findfirst(p -> p.algo == 0, perfResults)
    ibest === nothing &&
        error("Cannot find backup algorithm for convolution, giving up.")
    fallback = true
    math_unknown = true
    return (ibest, fallback, math_unknown)
end

function _choose_and_record(kind::AbstractString, convDesc, tensorDesc,
                            perfResults, n)
    _configured[] ||
        throw(PrecisionPolicyError(
            "precision policy is not initialized; call " *
            "StaticScalingPrecision.configure_precision! before convolution work"))

    strict_fma = _configuration[].fma
    if FMA_ENABLED[] != strict_fma
        throw(PrecisionPolicyError(
            "mutable FMA state was changed after precision configuration: " *
            "FMA_ENABLED[]=$(FMA_ENABLED[]) but configured fma=$(strict_fma); " *
            "start a fresh process and run configure_precision! again"))
    end

    descriptor_math_type = Ref{cuDNN.cudnnMathType_t}(cuDNN.CUDNN_DEFAULT_MATH)
    cuDNN.cudnnGetConvolutionMathType(convDesc, descriptor_math_type)
    dtype, dims, _ = cuDNN.cudnnGetTensorDescriptor(tensorDesc)

    _check_descriptor_math_type(descriptor_math_type[]; strict_fma = strict_fma)
    # Stock cuDNN.jl rule: Float32 tensors skip the math-type check.
    skip_math_type_check = dtype == Float32

    candidates = _candidate_log(perfResults, n)

    ibest = _select_candidate(perfResults, n;
                              strict_fma = strict_fma,
                              descriptor_math_type = descriptor_math_type[],
                              skip_math_type_check = skip_math_type_check)
    ibest, fallback, math_unknown = _resolve_candidate(
        perfResults, n, ibest; strict_fma = strict_fma, kind = kind)

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

"""
    _check_override_methods()

Use `which` to verify that the installed `cuDNN.math_mode(::CUDACore.MathMode)`
and the three `cuDNN.cudnnConvolutionAlgoPerfChoose` methods for the
forward, backward-data, and backward-filter performance vectors resolve
to methods defined by `StaticScalingPrecision`. Both the abstract
`AbstractVector{T}` signatures and the concrete `Vector{T}`/`Cint` call
signatures are checked. Throws `PrecisionPolicyError` if a lookup fails
or resolves elsewhere.
"""
function _check_override_methods()
    checks = [
        ("cuDNN.math_mode(::CUDACore.MathMode)",
         () -> which(cuDNN.math_mode, Tuple{CUDACore.MathMode})),
        ("cuDNN.cudnnConvolutionAlgoPerfChoose fwd abstract",
         () -> which(cuDNN.cudnnConvolutionAlgoPerfChoose,
                     Tuple{cuDNN.cudnnConvolutionDescriptor,
                           cuDNN.cudnnTensorDescriptor,
                           AbstractVector{cuDNN.cudnnConvolutionFwdAlgoPerf_t},
                           Any})),
        ("cuDNN.cudnnConvolutionAlgoPerfChoose fwd concrete",
         () -> which(cuDNN.cudnnConvolutionAlgoPerfChoose,
                     Tuple{cuDNN.cudnnConvolutionDescriptor,
                           cuDNN.cudnnTensorDescriptor,
                           Vector{cuDNN.cudnnConvolutionFwdAlgoPerf_t},
                           Cint})),
        ("cuDNN.cudnnConvolutionAlgoPerfChoose bwd_data abstract",
         () -> which(cuDNN.cudnnConvolutionAlgoPerfChoose,
                     Tuple{cuDNN.cudnnConvolutionDescriptor,
                           cuDNN.cudnnTensorDescriptor,
                           AbstractVector{cuDNN.cudnnConvolutionBwdDataAlgoPerf_t},
                           Any})),
        ("cuDNN.cudnnConvolutionAlgoPerfChoose bwd_data concrete",
         () -> which(cuDNN.cudnnConvolutionAlgoPerfChoose,
                     Tuple{cuDNN.cudnnConvolutionDescriptor,
                           cuDNN.cudnnTensorDescriptor,
                           Vector{cuDNN.cudnnConvolutionBwdDataAlgoPerf_t},
                           Cint})),
        ("cuDNN.cudnnConvolutionAlgoPerfChoose bwd_filter abstract",
         () -> which(cuDNN.cudnnConvolutionAlgoPerfChoose,
                     Tuple{cuDNN.cudnnConvolutionDescriptor,
                           cuDNN.cudnnTensorDescriptor,
                           AbstractVector{cuDNN.cudnnConvolutionBwdFilterAlgoPerf_t},
                           Any})),
        ("cuDNN.cudnnConvolutionAlgoPerfChoose bwd_filter concrete",
         () -> which(cuDNN.cudnnConvolutionAlgoPerfChoose,
                     Tuple{cuDNN.cudnnConvolutionDescriptor,
                           cuDNN.cudnnTensorDescriptor,
                           Vector{cuDNN.cudnnConvolutionBwdFilterAlgoPerf_t},
                           Cint})),
    ]
    failures = String[]
    for (label, lookup) in checks
        method = try
            lookup()
        catch err
            push!(failures, "$(label): method lookup failed " *
                            "($(sprint(showerror, err)))")
            continue
        end
        if method.module !== @__MODULE__
            push!(failures,
                  "$(label): resolved to $(method.module).$(method.name), " *
                  "not $(@__MODULE__)")
        end
    end
    isempty(failures) ||
        throw(PrecisionPolicyError(
            "cuDNN precision override methods are not installed as expected: " *
            join(failures, "; ")))
    return nothing
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

Public descriptor helper. Requires `configure_precision!` to have run in
this process, then delegates to `_cudnn_effective_math_type`.

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
    _configured[] ||
        throw(PrecisionPolicyError(
            "cudnn_effective_math_type requires precision configuration; call " *
            "StaticScalingPrecision.configure_precision! before descriptor work"))
    return _cudnn_effective_math_type(; pad = pad, stride = stride,
                                      dilation = dilation, group = group)
end

function _cudnn_effective_math_type(; pad::Integer = 1, stride::Integer = 2,
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
    configure_precision!(; fma=true, allow_unsupported_versions=false)
        -> NamedTuple

Configure the process for FP32/TF32-off GPU work:

1. read the loaded CUDA.jl and cuDNN.jl package versions with
   `Base.pkgversion` and enforce the supported 6.2.2 pair. An unsupported
   pair throws `PrecisionPolicyError` unless
   `allow_unsupported_versions=true`; that diagnostic path warns and
   records `versions_supported=false`;
2. use `which` to verify that the four cuDNN math-mode/algorithm-chooser
   methods installed by this module are still the dispatch targets for
   both the abstract `AbstractVector` and concrete `Vector`/`Cint` call
   signatures;
3. verify `CUDNN_FMA_MATH` exists and CUDA is functional;
4. set the module-global `FMA_ENABLED[]` exactly once (conflicting
   reconfiguration throws);
5. `CUDA.allowscalar(false)`;
6. `CUDA.math_mode!(CUDA.PEDANTIC_MATH)` for cuBLAS and assert the
   process math mode changed;
7. clear the three cuDNN convolution algorithm performance caches under
   their locks (their keys omit the descriptor math type) and verify
   they are empty;
8. verify the requested and effective descriptor math types match the
   case with the internal `_cudnn_effective_math_type`; the public helper
   is unavailable until `_configured` is set.

The version and override checks run before any process state is mutated.
A failure in the state-mutating section (steps 4-8) is permanent for the
process: later calls throw `PrecisionPolicyError` and the caller must
start a fresh process. Cached calls with the same `fma` value re-run the
override-method check, assert `FMA_ENABLED[]` and `CUDA.math_mode()`
still match the recorded state, reject a cached unsupported-version
configuration unless `allow_unsupported_versions=true`, and then return
the recorded configuration. Calls with a different `fma` value throw.

The process-wide `math_mode` and algorithm-chooser overrides are
installed at module load, so they are active before this function runs.

Must be called once per process before any convolution work and before
moving a model or arrays to the GPU. Returns
`(fma, cublas_math_mode, cudnn_requested, cudnn_effective, cudnn_version,
cuda_package_version, cudnn_package_version, cudnn_library_version,
versions_supported, caches_before, caches_cleared)`. `cudnn_version` and
`cudnn_library_version` are both the loaded cuDNN library version from
`cuDNN.version()`; the package versions are separate. With `fma=false`
the stock mapping is used, so `PEDANTIC_MATH` requests
`CUDNN_DEFAULT_MATH`; that is the negative control, not the accepted
policy.
"""
function configure_precision!(; fma::Bool = true,
                              allow_unsupported_versions::Bool = false)
    if _CONFIGURE_FAILED[] !== nothing
        failed = _CONFIGURE_FAILED[]
        throw(PrecisionPolicyError(
            "this process already failed precision configuration and cannot " *
            "be reused: $(sprint(showerror, failed)); start a fresh process"))
    end

    if _configured[]
        existing = _configuration[]
        existing.fma == fma ||
            throw(PrecisionPolicyError(
                "configure_precision! already configured fma=$(existing.fma); " *
                "refusing fma=$(fma) in the same process"))
        if !allow_unsupported_versions && existing.versions_supported !== true
            throw(PrecisionPolicyError(
                "cached precision configuration was recorded with unsupported " *
                "CUDA.jl/cuDNN.jl package versions; start a fresh process or " *
                "pass allow_unsupported_versions=true for a marked diagnostic run"))
        end
        _check_override_methods()
        FMA_ENABLED[] == existing.fma ||
            throw(PrecisionPolicyError(
                "mutable FMA state was changed after precision configuration: " *
                "FMA_ENABLED[]=$(FMA_ENABLED[]) but cached fma=$(existing.fma); " *
                "start a fresh process and run configure_precision! again"))
        CUDA.math_mode() == CUDA.PEDANTIC_MATH ||
            throw(PrecisionPolicyError(
                "CUDA math mode was changed after precision configuration: " *
                "CUDA.math_mode()=$(CUDA.math_mode()), expected " *
                "CUDA.PEDANTIC_MATH; start a fresh process and run " *
                "configure_precision! again"))
        return existing
    end

    package_versions = _package_versions()
    supported, cuda_v, cudnn_v = _check_versions(
        package_versions.cuda, package_versions.cudnn;
        allow_unsupported_versions = allow_unsupported_versions)
    _check_override_methods()

    isdefined(cuDNN, :CUDNN_FMA_MATH) ||
        throw(PrecisionPolicyError(
            "cuDNN.CUDNN_FMA_MATH is not defined; cannot request FP32 FMA math"))
    CUDA.functional() ||
        throw(PrecisionPolicyError(
            "CUDA.functional() is false; precision configuration needs a " *
            "functional GPU"))

    cache_info = nothing
    requested = nothing
    effective = nothing
    expected = nothing
    try
        FMA_ENABLED[] = fma
        CUDA.allowscalar(false)
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
        CUDA.math_mode() == CUDA.PEDANTIC_MATH ||
            throw(PrecisionPolicyError(
                "CUDA.math_mode! did not set CUDA.PEDANTIC_MATH; got " *
                "$(CUDA.math_mode())"))
        cache_info = _clear_algo_perf_caches()

        requested = cudnn_requested_math_type()
        effective = _cudnn_effective_math_type()
        expected = fma ? cuDNN.CUDNN_FMA_MATH : cuDNN.CUDNN_DEFAULT_MATH
        requested == expected ||
            throw(PrecisionPolicyError(
                "cuDNN requested math type $(math_type_label(requested)); " *
                "expected $(math_type_label(expected))"))
        effective == expected ||
            throw(PrecisionPolicyError(
                "cuDNN effective descriptor math type $(math_type_label(effective)); " *
                "expected $(math_type_label(expected))"))
    catch err
        _CONFIGURE_FAILED[] = err
        rethrow()
    end

    library_version = cudnn_version()
    configuration = (
        fma = fma,
        cublas_math_mode = CUDA.math_mode(),
        cudnn_requested = requested,
        cudnn_effective = effective,
        cudnn_version = library_version,
        cuda_package_version = cuda_v,
        cudnn_package_version = cudnn_v,
        cudnn_library_version = library_version,
        versions_supported = supported,
        caches_before = cache_info.before,
        caches_cleared = cache_info.cleared,
    )
    _configuration[] = configuration
    _configured[] = true
    return configuration
end

end # module StaticScalingPrecision
