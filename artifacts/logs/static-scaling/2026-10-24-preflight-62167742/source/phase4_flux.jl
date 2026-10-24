"""
Phase 4 static-scaling Flux runner (reference, DDP, split-control).

The heavy includes run only when this file is executed as a script, so
`julia --project=scripts/static_scaling/julia-env -e 'include("phase4_flux.jl")'`
is a pure parse check. Running it as a script loads `replay.jl`,
`model_flux.jl`, `precision.jl` only for `--device cuda`, and
`phase4_common.jl`, then dispatches on `--mode`.

Modes
-----
* `reference`: one process, the three Phase 3 updates on the full 12-sample
  batch, plain `Optimisers` state, 337-array Phase 3 NPZ schema.
* `ddp`: one process per rank, `DistributedOptimizer` wrapper with a
  `RecordingBackend`, equal Phase 2 shards. The default `--distributed-api
  guarded` records 64 wrapper collectives per update; `--distributed-api
  static` selects the historical per-leaf wrapper and records 62.
* `split-control`: one process, full-batch gradient versus the average of two
  half-batch gradients for u1 and u3.

Examples
--------
Reference (CPU, local):
  JULIA_LOAD_PATH="\$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration/test:@stdlib" \\
    julia --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test \\
    scripts/static_scaling/phase4_flux.jl \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \\
    --mode reference --device cpu --out /tmp/phase4-out --force

DDP (CPU/MPI, world 2):
  JULIA_LOAD_PATH="\$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration/test:@stdlib" \\
    ~/.julia/bin/mpiexecjl -n 2 julia --startup-file=no \\
    --project=/home/kurapica/Projects/ddp_flux/flux-integration/test \\
    scripts/static_scaling/phase4_flux.jl \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \\
    --mode ddp --device cpu --backend mpi --world-size 2 --out /tmp/phase4-out \\
    --rank-out /tmp/phase4-out/ddp/flux/w2/rank_0.npz --force \\
    --source-commit <sha> --source-state-sha256 <sha256> \\
    --flux-commit <sha> --flux-state-sha256 <sha256>

Add `--distributed-api static` to any DDP command to select the historical
per-leaf API (62 gradient allreduces; no presence or validity vote).
"""

function phase4_run(argv)
    try
        phase4 = Main.StaticScalingPhase4
        opts = phase4.parse_phase4_args(argv)
        mode = opts["mode"]
        code = if mode == "reference"
            phase4.run_reference(opts)
        elseif mode == "shard-reference"
            phase4.run_shard_reference(opts)
        elseif mode == "ddp"
            phase4.run_ddp(opts)
        elseif mode == "split-control"
            phase4.run_split_control(opts)
        else
            error("unsupported mode $(mode)")
        end
        code == 0 || return code
        println("PHASE4_FLUX PASS mode=$(mode) device=$(opts["device"]) " *
                "backend=$(opts["backend"])")
        return 0
    catch err
        println(stderr, "PHASE4_FLUX ERROR: ", sprint(showerror, err))
        for (i, frame) in enumerate(stacktrace(catch_backtrace()))
            i > 8 && break
            println(stderr, "  ", frame)
        end
        println("PHASE4_FLUX FAIL")
        return 1
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    include(joinpath(@__DIR__, "replay.jl"))
    include(joinpath(@__DIR__, "model_flux.jl"))
    let pre_opts = parse_cli_args(ARGS)
        mode = lowercase(get(pre_opts, "mode", ""))
        device = lowercase(get(pre_opts, "device", "cpu"))
        backend = lowercase(get(pre_opts, "backend", "mpi"))
        device == "cuda" && include(joinpath(@__DIR__, "precision.jl"))
        if mode == "ddp"
            Base.require(Base.PkgId(
                Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
            if backend == "nccl"
                Base.require(Base.PkgId(
                    Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
                Base.require(Base.PkgId(
                    Base.UUID("3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"), "NCCL"))
            end
        end
    end
    include(joinpath(@__DIR__, "phase4_common.jl"))
    exit(phase4_run(ARGS))
end
