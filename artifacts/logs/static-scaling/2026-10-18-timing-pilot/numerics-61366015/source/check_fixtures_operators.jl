"""
Phase 2 operator/fixture acceptance check (Flux + Zygote).

Verifies against the frozen cross-framework contract:
  * linear_fixture.npz: rectangular Dense(7 => 3), forward and Zygote gradients
    of sum(y); the weight must be used untransposed, size(W) == (3, 7).
  * kernel_fixture.npz case A (2x3 kernel, (out,in,kh,kw)=(3,2,2,3), stride 1,
    pad 0) and case B (2x3 kernel, out=2, in=3, stride 2, pad 1) with a manual
    orientation canary: plain Flux.Conv on the same permuted kernel must differ
    by more than 1e-3, reversed-kernel Conv must match Flux.CrossCor.
Float tolerance: cross-framework provisional atol=1e-5, rtol=1e-4
(|a-b| <= atol + rtol*max(|a|,|b|)); integer arrays are exact.
Prints `CHECK <name> PASS|FAIL <detail>` lines with max abs diffs and a final
`PHASE2_JULIA_OPERATOR_CHECK PASS|FAIL`; exits 0/1.

Reproduction (local, NPZ + Flux stacked load path):
  cd /home/kurapica/Projects/ddp_flux/ddp-flux-thesis
  JULIA_LOAD_PATH="\$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" \\
    julia --startup-file=no scripts/static_scaling/check_fixtures_operators.jl \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2

Reproduction (Leonardo: Flux project is the test project of the remote worktree):
  JULIA_LOAD_PATH="<THESIS>/scripts/static_scaling/julia-env:<REMOTE_FLUX>/test:@stdlib" \\
    julia --startup-file=no <THESIS>/scripts/static_scaling/check_fixtures_operators.jl \\
    --fixture-dir <ARTIFACTS>/2026-10-16-phase2
  (<REMOTE_FLUX> example: /leonardo_work/IscrC_UNMASKED/epicazio/flux-integration)
"""
;
include(joinpath(@__DIR__, "replay.jl"))
using Flux
using Zygote

const ATOL = 1.0e-5
const RTOL = 1.0e-4

const LINEAR_ARRAYS = [
    ("weight", Float32, (3, 7)),
    ("bias", Float32, (3,)),
    ("x", Float32, (7, 5)),
    ("y", Float32, (3, 5)),
    ("x_grad", Float32, (7, 5)),
    ("w_grad", Float32, (3, 7)),
    ("b_grad", Float32, (3,)),
]

const KERNEL_ARRAYS = [
    ("a_weight", Float32, (3, 2, 2, 3)),
    ("a_x", Float32, (2, 2, 5, 6)),
    ("a_y", Float32, (2, 3, 4, 4)),
    ("a_x_grad", Float32, (2, 2, 5, 6)),
    ("a_w_grad", Float32, (3, 2, 2, 3)),
    ("a_stride", Int64, (2,)),
    ("a_padding", Int64, (2,)),
    ("b_weight", Float32, (2, 3, 2, 3)),
    ("b_x", Float32, (1, 3, 9, 7)),
    ("b_y", Float32, (1, 2, 5, 4)),
    ("b_x_grad", Float32, (1, 3, 9, 7)),
    ("b_w_grad", Float32, (2, 3, 2, 3)),
    ("b_stride", Int64, (2,)),
    ("b_padding", Int64, (2,)),
]

function compare_float(log::CheckLog, name, actual, expected)
    if size(actual) != size(expected)
        return log_check(log, name, false, "shape $(size(actual)) vs $(size(expected))")
    end
    ok = all(within_tol(x, y; atol = ATOL, rtol = RTOL) for (x, y) in zip(actual, expected))
    return log_check(log, name, ok, "max_abs_diff=" * fmt_float(max_abs_diff(actual, expected)))
end

function check_linear(log::CheckLog, fields)
    W = fields["weight"]
    b = fields["bias"]
    x = fields["x"]
    # R1 canary: the classifier-style weight is canonical (out, in); Flux.Dense
    # stores exactly this shape and must not transpose it.
    log_check(log, "linear.weight_shape_out_in", size(W) == (3, 7),
              "size=$(size(W)) expected=(3, 7) no_transpose")
    d = Flux.Dense(W, b)
    compare_float(log, "linear.forward", d(x), fields["y"])
    grads = only(Zygote.gradient(m -> sum(m(x)), d))
    gx = only(Zygote.gradient(xx -> sum(d(xx)), x))
    compare_float(log, "linear.x_grad", gx, fields["x_grad"])
    compare_float(log, "linear.w_grad", grads.weight, fields["w_grad"])
    compare_float(log, "linear.b_grad", grads.bias, fields["b_grad"])
end

function kernel_setup(fields, tag)
    w = fields["$(tag)_weight"]
    x = fields["$(tag)_x"]
    stride = Tuple(Int.(fields["$(tag)_stride"]))
    padding = Tuple(Int.(fields["$(tag)_padding"]))
    wf = permutedims(w, (4, 3, 2, 1))       # (out,in,kh,kw) -> (kw,kh,in,out)
    xf = nchw_to_whcn(x)                    # NCHW -> WHCN
    return wf, xf, stride, padding
end

function check_kernel(log::CheckLog, fields, tag)
    wf, xf, stride, padding = kernel_setup(fields, tag)
    layer = Flux.CrossCor(wf, false; stride = stride, pad = padding)
    y = layer(xf)
    # forward back to canonical NCHW (the (4,3,2,1) permutation is self-inverse)
    compare_float(log, "kernel_$(tag).forward", y, permutedims(fields["$(tag)_y"], (4, 3, 2, 1)))
    grads = only(Zygote.gradient(m -> sum(m(xf)), layer))
    gx = only(Zygote.gradient(xx -> sum(layer(xx)), xf))
    compare_float(log, "kernel_$(tag).w_grad", permutedims(grads.weight, (4, 3, 2, 1)), fields["$(tag)_w_grad"])
    compare_float(log, "kernel_$(tag).x_grad", permutedims(gx, (4, 3, 2, 1)), fields["$(tag)_x_grad"])
end

function check_orientation_canary(log::CheckLog, fields, tag)
    wf, xf, stride, padding = kernel_setup(fields, tag)
    y_cc = Flux.CrossCor(wf, false; stride = stride, pad = padding)(xf)
    # Forward-only canary: Flux.Conv computes true convolution; it must not match
    # CrossCor on an asymmetric kernel, which is what R2 was about.
    y_conv = Flux.Conv(wf, false; stride = stride, pad = padding)(xf)
    d = max_abs_diff(y_conv, y_cc)
    log_check(log, "kernel_$(tag).orientation_canary", d > 1.0e-3,
              "conv_vs_crosscor_max_abs_diff=" * fmt_float(d) * " (must be > 1e-3)")
    y_rev = Flux.Conv(reverse(wf; dims = (1, 2)), false; stride = stride, pad = padding)(xf)
    compare_float(log, "kernel_$(tag).reversed_conv_matches", y_rev, y_cc)
end

function run_operator_checks(log::CheckLog, opts::Dict{String,String})
    fixture_dir = get(opts, "fixture-dir", "")
    isempty(fixture_dir) && error("--fixture-dir is required")
    isdir(fixture_dir) || error("--fixture-dir not found: $(fixture_dir)")

    lf = read_npz_fields(joinpath(fixture_dir, "linear_fixture.npz"))
    check_contract(log, lf, LINEAR_ARRAYS, "linear.contract")
    if all(haskey(lf, n) for (n, _, _) in LINEAR_ARRAYS)
        check_linear(log, lf)
    end

    kf = read_npz_fields(joinpath(fixture_dir, "kernel_fixture.npz"))
    check_contract(log, kf, KERNEL_ARRAYS, "kernel.contract")
    for tag in ("a", "b")
        if all(haskey(kf, "$(tag)_$(f)") for f in ("weight", "x", "y", "x_grad", "w_grad", "stride", "padding"))
            check_kernel(log, kf, tag)
        end
    end
    if all(haskey(kf, "$(f)") for f in ("a_weight", "a_x", "a_stride", "a_padding"))
        check_orientation_canary(log, kf, "a")
    end
    return nothing
end

function main_operators(argv)
    opts = parse_cli_args(argv)
    log = CheckLog()
    try
        run_operator_checks(log, opts)
    catch err
        log_check(log, "unhandled_exception", false, one_line(sprint(showerror, err)))
    end
    return final_line(log, "PHASE2_JULIA_OPERATOR_CHECK")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main_operators(ARGS))
end
