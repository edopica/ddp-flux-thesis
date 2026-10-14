# Conditional-graph success regression (single-process driver).
# Exit 0 requires correct values and ordered collectives on both ranks in all modes.
# Startup and update timeouts are separate. The update timer starts after both
# PROBE_READY markers, so package loading and AD compilation use the startup budget.
# Usage: make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration

using MPI
using Test

const PROBE = joinpath(@__DIR__, "conditional_deadlock_probe.jl")
const PROJECT_DIR = dirname(Base.active_project())
const JULIA = Base.julia_cmd()
const STARTUP_TIMEOUT = parse(Float64, get(ENV, "DEADLOCK_STARTUP_TIMEOUT", "300.0"))
const HANG_TIMEOUT = parse(Float64, get(ENV, "DEADLOCK_HANG_TIMEOUT", "20.0"))
const NPROCS = 2

read_log(path) = read(path, String)

function child_command(mode)
    return `$(MPI.mpiexec()) -n $(NPROCS) $(JULIA) --project=$(PROJECT_DIR) --startup-file=no --color=no $(PROBE) $(mode)`
end

function stop_probe(proc)
    kill(proc)
    if timedwait(() -> !process_running(proc), 5.0) == :timed_out
        kill(proc, Base.SIGKILL)
    end
    wait(proc)
end

function run_probe(mode)
    log, io = mktemp()
    proc = nothing
    started = time()
    try
        proc = run(pipeline(child_command(mode), stdout=io, stderr=io), wait=false)
        ready = false
        deadline = started + STARTUP_TIMEOUT
        timeout = :none
        while process_running(proc)
            out = read_log(log)
            if !ready && all(r -> occursin("PROBE_READY rank=$(r)\n", out), 0:NPROCS-1)
                ready = true
                deadline = time() + HANG_TIMEOUT
            end
            if time() > deadline
                timeout = ready ? :update : :startup
                stop_probe(proc)
                break
            end
            sleep(0.2)
        end
        wait(proc)
        return (; output=read_log(log), timeout, exitcode=proc.exitcode,
                elapsed=time() - started)
    finally
        proc !== nothing && process_running(proc) && stop_probe(proc)
        close(io)
        rm(log; force=true)
    end
end

# Independent analytical protocol: fields are trunk, a, b, c, all length four.
# Each rank contributes its local gradient or zero. The mean divides by ALL ranks.
# The trailing gradient-validity vote is appended after the gradient entries.
function expected_trace(mode, rank)
    active = mode == "control" ? [true, true, true, true] :
             mode == "deadlock" ? [true, rank == 0, rank == 0, rank == 1] :
                                  [true, rank == 0, rank == 1, false]
    global_presence = mode == "control" ? [1.0, 1.0, 1.0, 1.0] :
                      mode == "deadlock" ? [1.0, 0.5, 0.5, 0.5] :
                                           [1.0, 0.5, 0.5, 0.0]
    gradients = [0.0, -2.0, -8.0, -4.0]
    expected = [(kind="presence", dtype="Float32", input=Float64.(active), output=global_presence)]
    for i in eachindex(gradients)
        global_presence[i] == 0 && continue
        push!(expected, (kind="gradient", dtype="Float64",
                         input=fill(active[i] ? gradients[i] : 0.0, 4),
                         output=fill(gradients[i] * global_presence[i], 4)))
    end
    # Gradient-validity vote: host Float32 length-1 flag, valid (0) on every rank.
    # The probe classifies collectives by dtype, so this one logs as "presence".
    push!(expected, (kind="presence", dtype="Float32", input=[0.0], output=[0.0]))
    return expected
end

function check_trace(out, mode, rank)
    pattern = Regex("TRACE rank=$(rank) allreduce #([0-9]+) kind=(\\w+) dtype=(\\w+) op=(\\w+) input=([^\\s]+) output=([^\\s]+) END")
    records = collect(eachmatch(pattern, out))
    expected = expected_trace(mode, rank)
    @test length(records) == length(expected)
    # Count all entries too: malformed, duplicate, or incomplete traces must fail.
    @test count("TRACE rank=$(rank) allreduce #", out) == length(expected)
    for (i, (record, want)) in enumerate(zip(records, expected))
        number, kind, dtype, op, input, output = record.captures
        @test parse(Int, number) == i
        @test kind == want.kind
        @test dtype == want.dtype
        @test op == "avg"
        @test parse.(Float64, split(input, ',')) ≈ want.input
        @test parse.(Float64, split(output, ',')) ≈ want.output
    end
end

function main()
    @testset "conditional-graph DDP success regression" begin
        for mode in ("deadlock", "corruption", "control")
            @testset "$mode" begin
                result = run_probe(mode)
                out = result.output
                println("\n--- $mode mode trace ---\n", out,
                        "\n[driver] mode=$mode exit=$(result.exitcode) timeout=$(result.timeout) elapsed=$(round(result.elapsed; digits=3))s")
                @test result.timeout == :none
                @test result.exitcode == 0
                for rank in 0:NPROCS-1
                    @test count("PROBE_READY rank=$(rank)\n", out) == 1
                    @test count("PROBE_UPDATE_DONE rank=$(rank)\n", out) == 1
                    @test count("PROBE_FINISHED rank=$(rank)\n", out) == 1
                    @test count("PROBE_VERDICT mode=$(mode) rank=$(rank) correct=true ", out) == 1
                    check_trace(out, mode, rank)
                end
            end
        end
    end
    return true
end

ok = try
    main()
catch err
    err isa Test.TestSetException || rethrow()
    false
end
println("\nDETECTOR SUMMARY: ", ok ? "conditional-graph support works (CPU/MPI, two ranks)" :
                                      "conditional-graph regression FAILED")
exit(ok ? 0 : 1)
