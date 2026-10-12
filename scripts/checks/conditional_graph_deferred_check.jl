# ==============================================================================
# Driver for the conditional-graph deferred-behavior probe.
# ==============================================================================
#
# Runs `conditional_graph_deferred_probe.jl` at 2 MPI ranks per mode with a
# startup timer and an external hang watchdog, saves each raw console log under
# `artifacts/logs/conditional-wrapper/deferred/`, and asserts the expected
# marker signature per mode. A timeout alone never passes: the hang modes must
# show the unmatched `COLLECTIVE_BEGIN` of the finite rank plus the stop/hold
# marker of the diverging rank.
#
# Usage (from the thesis workspace):
#   julia --project=../flux-integration/test \
#       scripts/checks/conditional_graph_deferred_check.jl
#
# Optional environment overrides:
#   DEFERRED_STARTUP_TIMEOUT (default 420 s), DEFERRED_HANG_TIMEOUT (default 60 s),
#   DEFERRED_OUTDIR (default artifacts/logs/conditional-wrapper/deferred).
# ==============================================================================

using MPI

const PROBE = joinpath(@__DIR__, "conditional_graph_deferred_probe.jl")
const PROJECT_DIR = dirname(Base.active_project())
const JULIA = Base.julia_cmd()
const STARTUP_TIMEOUT = parse(Float64, get(ENV, "DEFERRED_STARTUP_TIMEOUT", "420.0"))
const HANG_TIMEOUT = parse(Float64, get(ENV, "DEFERRED_HANG_TIMEOUT", "60.0"))
const DATE = "2026-10-06"
const OUTDIR = abspath(get(ENV, "DEFERRED_OUTDIR",
    joinpath(@__DIR__, "..", "..", "artifacts", "logs", "conditional-wrapper", "deferred")))

const FAILURES = Ref(0)

function check(condition, label)
    condition || (FAILURES[] += 1)
    println("    ", condition ? "PASS" : "FAIL", "  ", label)
    return condition
end

read_log(path) = read(path, String)

function stop_probe(proc)
    kill(proc)
    if timedwait(() -> !process_running(proc), 5.0) == :timed_out
        kill(proc, Base.SIGKILL)
    end
    wait(proc)
end

function reap_orphans()
    pids = try
        strip(read(`pgrep -f conditional_graph_deferred_probe.jl`, String))
    catch
        ""
    end
    isempty(pids) && return ""
    try
        run(pipeline(`pkill -f conditional_graph_deferred_probe.jl`, devnull))
    catch
    end
    sleep(1.0)
    try
        run(pipeline(`pkill -9 -f conditional_graph_deferred_probe.jl`, devnull))
    catch
    end
    return pids
end

function run_probe(mode)
    logpath = joinpath(OUTDIR, "raw-$(mode)-mpi2-$DATE.log")
    # One shared IO handle for stdout and stderr: two separate opens of the same
    # path would overwrite each other's offsets when a hung run is killed.
    log, io = mktemp()
    started = time()
    proc = nothing
    timeout = :none
    try
        cmd = `$(MPI.mpiexec()) -n 2 $(JULIA) --project=$PROJECT_DIR --startup-file=no --color=no $(PROBE) $mode`
        proc = run(pipeline(cmd, stdout=io, stderr=io), wait=false)
        deadline = started + STARTUP_TIMEOUT
        ready = false
        while process_running(proc)
            out = read_log(log)
            if !ready && count("PROBE_READY", out) >= 2
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
        output = read_log(log)
        write(logpath, output)
        return (; mode, output, timeout,
                exitcode=proc.exitcode, elapsed=time() - started)
    finally
        proc !== nothing && process_running(proc) && stop_probe(proc)
        close(io)
        rm(log; force=true)
        if timeout != :none
            pids = reap_orphans()
            isempty(pids) || println("    reaped orphan pids: $(replace(pids, '\n' => ' '))")
        end
    end
end

# ---- log helpers -------------------------------------------------------------

# Warmup runs with a non-logging backend, so every COLLECTIVE marker in the
# console log belongs to the requested mode; rank output interleaving must not
# be sliced apart.
body(run) = run.output

countpat(s, pat) = length(collect(eachmatch(Regex(pat), s)))

function last_begin(s, r)
    ms = collect(eachmatch(Regex("COLLECTIVE_BEGIN rank=$r number=(\\d+) kind=(\\w+)"), s))
    isempty(ms) && return nothing
    return (number=parse(Int, ms[end].captures[1]), kind=String(ms[end].captures[2]))
end

function last_end(s, r)
    ms = collect(eachmatch(Regex("COLLECTIVE_END rank=$r number=(\\d+)"), s))
    isempty(ms) && return nothing
    return parse(Int, ms[end].captures[1])
end

function num_after(s, pat)
    m = match(Regex(pat), s)
    m === nothing && return nothing
    return parse(Float64, m.captures[1])
end

function mode_result_ok(run, mode, rank)
    return occursin("MODE_RESULT mode=$mode rank=$rank ok=true", body(run))
end

function matched_collectives(run, rank; begin_count, end_count, last_begin_number)
    b = body(run)
    nb = countpat(b, "COLLECTIVE_BEGIN rank=$rank ")
    ne = countpat(b, "COLLECTIVE_END rank=$rank ")
    lb = last_begin(b, rank)
    check(nb == begin_count, "rank $rank: $begin_count collectives begun (got $nb)")
    check(ne == end_count, "rank $rank: $end_count collectives completed (got $ne)")
    if last_begin_number !== nothing
        check(lb !== nothing && lb.number == last_begin_number && lb.kind == "presence",
              "rank $rank: last begun collective is number $last_begin_number (presence)")
    end
end

# ---- per-mode expectations ---------------------------------------------------

function assert_control(run)
    b = body(run)
    check(run.timeout == :none, "no timeout")
    check(run.exitcode == 0, "exit 0 (got $(run.exitcode))")
    for r in 0:1
        check(occursin("MODE_RESULT mode=control rank=$r ok=true", b), "rank $r result ok")
        check(occursin("CONTROL_MATCH rank=$r ok=true", b), "rank $r matches native Adam shadow")
    end
    matched_collectives(run, 0; begin_count=12, end_count=12, last_begin_number=nothing)
    matched_collectives(run, 1; begin_count=12, end_count=12, last_begin_number=nothing)
end

function assert_grad_nonfinite(run)
    b = body(run)
    check(run.timeout == :none, "no timeout")
    check(run.exitcode == 0, "exit 0 (got $(run.exitcode))")
    check(occursin(Regex("GRAD rank=0 .*loss_finite=true wgrad_finite=true"), b),
          "rank 0 finite loss and finite w gradient")
    check(occursin(Regex("GRAD rank=1 .*loss_finite=true wgrad_finite=false"), b),
          "rank 1 finite loss and NaN w gradient")
    for r in 0:1
        check(occursin(Regex("CORRUPTION rank=$r exc=none w_finite=false v_finite=true " *
                             "wmom_finite=false"), b),
              "rank $r updated without exception; w and its Adam moments became non-finite, " *
              "v stayed finite")
        check(occursin("MODE_RESULT mode=grad-nonfinite rank=$r ok=true", b), "rank $r result ok")
    end
    matched_collectives(run, 0; begin_count=3, end_count=3, last_begin_number=nothing)
    matched_collectives(run, 1; begin_count=3, end_count=3, last_begin_number=nothing)
end

function assert_nan_loss_all(run)
    b = body(run)
    check(run.timeout == :none, "no timeout")
    check(run.exitcode == 0, "exit 0 (got $(run.exitcode))")
    for r in 0:1
        check(occursin("NONFINITE_STOP rank=$r item=3", b),
              "rank $r stopped with DomainError at step 3")
        check(occursin(Regex("STOP_STATE_MATCH rank=$r caught=true .*state_ok=true"), b),
              "rank $r state matches the two-step shadow after the stop")
        check(occursin("MODE_RESULT mode=nan-loss-all rank=$r ok=true", b), "rank $r result ok")
    end
    matched_collectives(run, 0; begin_count=6, end_count=6, last_begin_number=nothing)
    matched_collectives(run, 1; begin_count=6, end_count=6, last_begin_number=nothing)
end

function assert_nan_loss_hang(run)
    b = body(run)
    check(run.timeout == :update, "watchdog fired after startup (got $(run.timeout))")
    check(occursin("NONFINITE_STOP rank=1 item=3 exc=DomainError", b),
          "rank 1 stopped with DomainError at step 3")
    check(occursin("HOLDING rank=1 after=DomainError", b), "rank 1 held after the DomainError")
    check(!occursin("NONFINITE_STOP rank=0", b), "rank 0 never stopped")
    matched_collectives(run, 0; begin_count=7, end_count=6, last_begin_number=7)
    matched_collectives(run, 1; begin_count=6, end_count=6, last_begin_number=nothing)
end

function assert_early_stop(run)
    b = body(run)
    check(run.timeout == :update, "watchdog fired after startup (got $(run.timeout))")
    check(occursin("LOOP_STOP rank=1 i=4 reason=rank-local", b), "rank 1 stopped its loop at step 4")
    check(occursin("HOLDING rank=1 after=early-stop", b), "rank 1 held after the early stop")
    matched_collectives(run, 0; begin_count=10, end_count=9, last_begin_number=10)
    matched_collectives(run, 1; begin_count=9, end_count=9, last_begin_number=nothing)
end

function assert_step_desync(run)
    b = body(run)
    check(run.timeout == :update, "watchdog fired after startup (got $(run.timeout))")
    check(occursin("LOSS_NAN rank=1 i=3", b), "rank 1 loss was NaN at step 3")
    check(occursin("HOLDING rank=1 after=step-desync", b), "rank 1 held after its loop")
    dc = num_after(b, "DESYNC_RESULT rank=1 .*delta_vs_correct=([0-9.eE+-]+)")
    ds = num_after(b, "DESYNC_RESULT rank=1 .*delta_vs_shifted=([0-9.eE+-]+)")
    check(dc !== nothing && dc > 1e-6,
          "rank 1 trajectory differs from the correct 5-step reference (delta=$(dc))")
    check(ds !== nothing && ds <= 1e-10,
          "rank 1 trajectory matches the shifted-pairing reference (delta=$(ds))")
    matched_collectives(run, 0; begin_count=13, end_count=12, last_begin_number=13)
    matched_collectives(run, 1; begin_count=12, end_count=12, last_begin_number=nothing)
end

function assert_config_eta(run)
    b = body(run)
    check(run.timeout == :none, "no timeout")
    check(run.exitcode == 0, "exit 0 (got $(run.exitcode))")
    check(occursin("LOCAL_ADJUST rank=1 i=2 eta=0.1", b), "rank 1 adjusted its learning rate locally")
    d = num_after(b, "CONFIG_ETA rank=0 self_maxdiff=([0-9.eE+-]+)")
    cross = num_after(b, "CONFIG_ETA rank=0 .*cross_delta=([0-9.eE+-]+)")
    check(d !== nothing && d <= 1e-13,
          "rank 0 matches its own eta=0.05 reference (maxdiff=$(d))")
    check(cross !== nothing && cross > 1e-6,
          "replicas diverged silently (cross delta=$(cross))")
    for r in 0:1
        check(occursin("MODE_RESULT mode=config-eta rank=$r ok=true", b), "rank $r result ok")
    end
    matched_collectives(run, 0; begin_count=12, end_count=12, last_begin_number=nothing)
    matched_collectives(run, 1; begin_count=12, end_count=12, last_begin_number=nothing)
end

function assert_config_frozen(run)
    b = body(run)
    check(run.timeout == :none, "no timeout")
    check(run.exitcode == 0, "exit 0 (got $(run.exitcode))")
    check(occursin("LOCAL_FREEZE rank=1", b), "rank 1 froze its wrapper state locally")
    cross = num_after(b, "CONFIG_FROZEN rank=1 cross_delta=([0-9.eE+-]+)")
    check(cross !== nothing && cross > 1e-6,
          "replicas diverged silently (cross delta=$(cross))")
    check(occursin(Regex("CONFIG_FROZEN rank=1 cross_delta=[0-9.eE+-]+ unchanged=true"), b),
          "rank 1 parameters stayed frozen")
    for r in 0:1
        check(occursin("MODE_RESULT mode=config-frozen rank=$r ok=true", b), "rank $r result ok")
    end
    matched_collectives(run, 0; begin_count=12, end_count=12, last_begin_number=nothing)
    matched_collectives(run, 1; begin_count=12, end_count=12, last_begin_number=nothing)
end

# ---- main --------------------------------------------------------------------

const CASES = [
    ("control", assert_control),
    ("nan-loss-all", assert_nan_loss_all),
    ("grad-nonfinite", assert_grad_nonfinite),
    ("config-eta", assert_config_eta),
    ("config-frozen", assert_config_frozen),
    ("nan-loss-hang", assert_nan_loss_hang),
    ("early-stop", assert_early_stop),
    ("step-desync", assert_step_desync),
]

mkpath(OUTDIR)
println("conditional-graph deferred-behavior check")
println("probe       : $PROBE")
println("project     : $PROJECT_DIR")
println("output dir  : $OUTDIR")
println("timeouts    : startup=$STARTUP_TIMEOUT s, hang=$HANG_TIMEOUT s")

for (mode, assert_fn) in CASES
    println("\n=== mode: $mode ===")
    run = run_probe(mode)
    println("    timeout=$(run.timeout) exitcode=$(run.exitcode) elapsed=$(round(run.elapsed; digits=1))s")
    versions = match(r"VERSIONS (.*)", run.output)
    versions !== nothing && println("    ", versions.captures[1])
    assert_fn(run)
end

println("\nfailures: $(FAILURES[])")
exit(FAILURES[] == 0 ? 0 : 1)
