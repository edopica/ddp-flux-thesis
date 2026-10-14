# ==============================================================================
# Driver for the DB-1/DB-4 loss and DB-3 gradient numerical-guard regressions.
# ==============================================================================
#
# Runs `conditional_graph_numerical_guard_probe.jl` under mpiexec per case,
# with a separate startup watchdog and execution watchdog, saves each
# interleaved stdout+stderr log under the requested outdir, validates every
# rank's collective trace by ordinal and metadata against the scenario's
# explicit expected stage schedule, and writes `summary.tsv` and `metadata.md`.
# Only this parent writes shared files; the children only print markers. The
# probe is group-agnostic: `--group` selects the driver's case table, its
# expected schedules, and its metadata only.
#
# Usage (from the thesis workspace):
#   julia --startup-file=no --project=../flux-integration/test \
#       scripts/checks/conditional_graph_numerical_guard_check.jl \
#       --group loss --ranks 2 --backend mpi \
#       --outdir artifacts/logs/conditional-wrapper/numerical-guards/loss-mpi2
#
# `--group` (default loss) selects the case table:
#   loss      DB-1/DB-4 coordinated non-finite-loss rejection
#   gradient  DB-3 synchronized-gradient rejection
#   all       loss cases followed by gradient cases in one campaign
#
# Frozen post-fix collective protocol: every nonempty `_sync_gradients` runs one
# extra host Float32 len-1 `avg` collective (the "gradient vote") after the
# presence and gradient reductions; `np == 0` keeps zero collectives. On a
# gradient rejection every rank completes the full schedule (eager: loss vote ->
# presence -> P reductions -> gradient vote; explicit: presence -> P reductions
# -> gradient vote) before throwing. Loss-rejected eager steps issue only the
# loss vote. The gradient vote is metadata-identical to the loss vote (and to
# the presence vector when np == 1), so `:vote`, `:presence` and `:gradvote` are
# classified by ordinal within the explicit expected schedule, never by dtype
# or length alone.
#
# Optional flags:
#   --case NAME          run only matching case names
#   --selftest-cleanup   run the campaign-scoped cleanup isolation selftest
#                        (no MPI child launch) and exit
#
# Environment overrides:
#   NUMERICAL_GUARD_STARTUP_TIMEOUT (default 420 s)
#   NUMERICAL_GUARD_EXECUTION_TIMEOUT (default 120 s)
#   NUMERICAL_GUARD_PHASE (default green)
#   NUMERICAL_GUARD_CAMPAIGN (default "<pid>-<time_ns()>")
#
# Every probe child carries `--campaign TOKEN`; cleanup matches TOKEN as an
# exact separate argv element (no regular expressions), so prefix-extended and
# metacharacter-colliding tokens cannot select another campaign. Cleanup only
# kills processes whose argv contains the probe file as a separate element and
# the exact adjacent pair `--campaign TOKEN`. Operator-supplied
# NUMERICAL_GUARD_CAMPAIGN values must be unique across concurrent runs: the
# default "<pid>-<time_ns()>" is unique, but uniqueness of a user-supplied
# token is the operator's responsibility. `--selftest-cleanup` covers exact
# match, prefix extension, and metacharacter collision. The `healthy` case
# trains a conditional model through the eager `train!` path: rank-distinct
# U/V branches (full, partial, globally absent, reactivated) are compared per
# step against a native reference; the driver additionally requires the
# probe's STEP_STATE_OK, ABSENCE_FROZEN, and REACTIVATED markers.
# ==============================================================================

using MPI
using Dates

# ------------------------------------------------------------------
# Campaign-scoped cleanup. Every probe child carries `--campaign TOKEN`; the
# cleanup scans /proc and matches only processes whose argv contains the probe
# filename as a separate element and the exact adjacent pair
# `--campaign TOKEN`, so concurrent campaigns cannot kill each other.
# `--selftest-cleanup` checks this isolation without launching MPI and exits
# before option parsing.
# ------------------------------------------------------------------

const PROBE_BASENAME = "conditional_graph_numerical_guard_probe.jl"

function campaign_pids(token)
    pids = Int[]
    for entry in readdir("/proc")
        all(isdigit, entry) || continue
        pid = parse(Int, entry)
        pid == getpid() && continue
        cmdline = try
            read("/proc/$pid/cmdline", String)
        catch
            continue
        end
        argv = split(cmdline, '\0'; keepempty=false)
        any(a -> basename(a) == PROBE_BASENAME, argv) || continue
        matched = false
        for i in 1:(length(argv) - 1)
            if argv[i] == "--campaign" && argv[i + 1] == token
                matched = true
                break
            end
        end
        matched && push!(pids, pid)
    end
    sort!(pids)
    return pids
end

function reap_campaign(token)
    pids = campaign_pids(token)
    isempty(pids) && return pids
    for pid in pids
        try
            run(pipeline(`kill -TERM $pid`, devnull))
        catch
        end
    end
    sleep(1.0)
    for pid in campaign_pids(token)
        try
            run(pipeline(`kill -KILL $pid`, devnull))
        catch
        end
    end
    return pids
end

function selftest_cleanup()
    base = getpid()
    t1 = "selftestA-$base"
    t2 = "$t1-rerun"
    t3 = "selftest.$base"
    t4 = "selftestX$base"
    tokens = [t1, t2, t3, t4]

    procs = Dict{String, Base.Process}()
    for tok in tokens
        procs[tok] = run(pipeline(
            `$(Base.julia_cmd()) --startup-file=no -e "sleep(300)" $PROBE_BASENAME --campaign $tok`,
            stdout=devnull, stderr=devnull), wait=false)
    end

    deadline = time() + 15.0
    ready = false
    while time() < deadline
        ready = all(tok -> length(campaign_pids(tok)) == 1, tokens)
        ready && break
        sleep(0.25)
    end

    pids1 = campaign_pids(t1)
    pids2 = campaign_pids(t2)
    pids3 = campaign_pids(t3)
    pids4 = campaign_pids(t4)
    started = ready && length(pids1) == 1 && length(pids2) == 1 &&
              length(pids3) == 1 && length(pids4) == 1

    killed1 = Int[]
    killed2 = Int[]
    killed3 = Int[]
    killed4 = Int[]
    t1_gone = t2_gone = t3_gone = t4_gone = false
    t2_alive_after_t1 = t3_alive_after_t1 = t4_alive_after_t1 = false
    t2_alive_after_t3 = t4_alive_after_t3 = false
    t2_alive_after_t4 = false
    prefix_ok = metachar_ok = false

    if started
        killed1 = reap_campaign(t1)
        t1_gone = isempty(campaign_pids(t1))
        t2_alive_after_t1 = length(campaign_pids(t2)) == 1
        t3_alive_after_t1 = length(campaign_pids(t3)) == 1
        t4_alive_after_t1 = length(campaign_pids(t4)) == 1

        killed3 = reap_campaign(t3)
        t3_gone = isempty(campaign_pids(t3))
        t2_alive_after_t3 = length(campaign_pids(t2)) == 1
        t4_alive_after_t3 = length(campaign_pids(t4)) == 1

        killed4 = reap_campaign(t4)
        t4_gone = isempty(campaign_pids(t4))
        t2_alive_after_t4 = length(campaign_pids(t2)) == 1

        killed2 = reap_campaign(t2)
        t2_gone = isempty(campaign_pids(t2))

        prefix_ok = t2_alive_after_t1 && t2_alive_after_t3 && t2_alive_after_t4
        metachar_ok = t3_alive_after_t1 && t4_alive_after_t1 &&
                      t2_alive_after_t3 && t4_alive_after_t3
    end

    for tok in tokens
        proc = procs[tok]
        if process_running(proc)
            kill(proc)
            if timedwait(() -> !process_running(proc), 5.0) == :timed_out
                kill(proc, Base.SIGKILL)
            end
        end
        wait(proc)
    end

    final_empty = all(tok -> isempty(campaign_pids(tok)), tokens)
    passed = started && t1_gone && t2_gone && t3_gone && t4_gone &&
             prefix_ok && metachar_ok && final_empty

    evidence = "t1=$t1 pids1=$(join(pids1, ',')) killed1=$(join(killed1, ',')) " *
               "t2=$t2 pids2=$(join(pids2, ',')) killed2=$(join(killed2, ',')) " *
               "t3=$t3 pids3=$(join(pids3, ',')) killed3=$(join(killed3, ',')) " *
               "t4=$t4 pids4=$(join(pids4, ',')) killed4=$(join(killed4, ','))"

    if passed
        println("SELFTEST_CLEANUP PASS prefix_ok=$prefix_ok metachar_ok=$metachar_ok " *
                "final_empty=$final_empty $evidence")
        return 0
    end
    println("SELFTEST_CLEANUP FAIL reason=\"started=$started t1_gone=$t1_gone t2_gone=$t2_gone " *
            "t3_gone=$t3_gone t4_gone=$t4_gone prefix_ok=$prefix_ok metachar_ok=$metachar_ok " *
            "final_empty=$final_empty t2_alive_after_t1=$t2_alive_after_t1 " *
            "t3_alive_after_t1=$t3_alive_after_t1 t4_alive_after_t1=$t4_alive_after_t1 " *
            "t2_alive_after_t3=$t2_alive_after_t3 t4_alive_after_t3=$t4_alive_after_t3 " *
            "t2_alive_after_t4=$t2_alive_after_t4\" $evidence")
    return 1
end

if "--selftest-cleanup" in ARGS
    exit(selftest_cleanup())
end

function parse_cli(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $(repr(a))")
        key = a[3:end]
        key in ("group", "ranks", "backend", "outdir", "case") ||
            error("unknown option $(repr(a))")
        i += 1
        i <= length(args) || error("missing value for $(repr(a))")
        opts[key] = args[i]
        i += 1
    end
    return opts
end

const OPTS = parse_cli(ARGS)
const GROUP = get(OPTS, "group", "loss")
const RANKS = parse(Int, get(OPTS, "ranks", "2"))
const BACKEND = get(OPTS, "backend", "mpi")
const OUTDIR = abspath(get(OPTS, "outdir", "artifacts/logs/conditional-wrapper/numerical-guards"))
const CASE_FILTER = get(OPTS, "case", nothing)

GROUP in ("loss", "gradient", "all") ||
    error("--group must be loss, gradient or all; got $(repr(GROUP))")
BACKEND in ("mpi", "nccl") || error("--backend must be mpi or nccl; got $(repr(BACKEND))")

const PROBE = joinpath(@__DIR__, PROBE_BASENAME)
const PROJECT_DIR = dirname(Base.active_project())
const JULIA = Base.julia_cmd()
const STARTUP_TIMEOUT = parse(Float64, get(ENV, "NUMERICAL_GUARD_STARTUP_TIMEOUT", "420.0"))
const EXECUTION_TIMEOUT = parse(Float64, get(ENV, "NUMERICAL_GUARD_EXECUTION_TIMEOUT", "120.0"))
const PHASE = get(ENV, "NUMERICAL_GUARD_PHASE", "green")
const CAMPAIGN = get(ENV, "NUMERICAL_GUARD_CAMPAIGN", "$(getpid())-$(time_ns())")

# ------------------------------------------------------------------
# Case tables. Each entry is (case, fault_rank, fault_value); fault_rank = -1
# means no rank-local fault (all-rank or no-fault scenario). `--group` selects
# the loss table, the gradient table, or their concatenation (`all`). NCCL
# reuses the MPI tables; the overflow cases stay listed because the probe
# reports them UNTESTED under NCCL (the average arithmetic differs), never
# silently skipped.
# ------------------------------------------------------------------

function loss_case_table(ranks)
    ranks == 2 && return [
        ("train-one", 0, "nan"),
        ("train-one", 1, "nan"),
        ("train-one-inf", 1, "inf"),
        ("train-one-ninf", 1, "ninf"),
        ("loop-one", 1, "nan"),
        ("loop-withgradient-one", 1, "nan"),
        ("train-all", -1, "nan"),
        ("aux-nonfinite-ok", -1, "nan"),
        ("aux-nan-first", 1, "nan"),
        ("no-trainable", 1, "nan"),
        ("recover-one", 1, "nan"),
        ("healthy", -1, "nan"),
        ("dup-train-one", 1, "nan"),
    ]
    ranks == 4 && return [
        ("train-one", 0, "nan"),
        ("train-one", 2, "nan"),
        ("train-one-inf", 3, "inf"),
        ("loop-one", 3, "nan"),
        ("aux-nan-first", 2, "nan"),
        ("no-trainable", 2, "nan"),
        ("recover-one", 3, "nan"),
        ("healthy", -1, "nan"),
    ]
    error("no loss case table for --ranks $ranks (supported: 2, 4)")
end

function gradient_case_table(ranks)
    ranks == 2 && return [
        ("grad-train-one", 0, "nan"),
        ("grad-train-one", 1, "nan"),
        ("grad-audit-zero", 1, "nan"),
        ("grad-late", 1, "nan"),
        ("grad-inf", 1, "inf"),
        ("grad-ninf", 1, "ninf"),
        ("grad-update-both", 0, "nan"),
        ("grad-overflow", -1, "inf"),
        ("grad-tied-overflow", -1, "inf"),
        ("grad-asym-vote", -1, "nan"),
        ("grad-caller", 0, "nan"),
        ("grad-conditional-absence", 0, "nan"),
        ("grad-absent-history", -1, "nan"),
        ("grad-all-absent", -1, "nan"),
        ("grad-no-trainable", -1, "nan"),
        ("grad-empty", -1, "nan"),
        ("grad-frozen", 1, "nan"),
        ("grad-recover", 1, "nan"),
        ("grad-healthy", -1, "nan"),
    ]
    ranks == 4 && return [
        ("grad-train-one", 2, "nan"),
        ("grad-late", 3, "nan"),
        ("grad-inf", 3, "inf"),
        ("grad-asym-vote", -1, "nan"),
        ("grad-conditional-absence", 2, "nan"),
        ("grad-recover", 3, "nan"),
        ("grad-healthy", -1, "nan"),
        ("grad-tied-overflow", -1, "inf"),
    ]
    error("no gradient case table for --ranks $ranks (supported: 2, 4)")
end

function case_table(ranks)
    GROUP == "loss" && return loss_case_table(ranks)
    GROUP == "gradient" && return gradient_case_table(ranks)
    return vcat(loss_case_table(ranks), gradient_case_table(ranks))
end

# ------------------------------------------------------------------
# Expected collective schedules. Classification is driven by the scenario's
# explicit stage schedule, never by dtype or length alone: the loss vote, the
# gradient vote (both host Float32 len 1) and the presence vector (len np) can
# share a dtype and even a length. `:vote`, `:presence` and `:gradvote` are
# therefore told apart only by ordinal within the expected schedule. GuardModel
# has np = 2 canonical gradient buffers: tied a/b (len 2) and independent c
# (len 3). CondModel has np = 3: w (len 2), u (len 3) and v (len 4).
# ------------------------------------------------------------------

const NP_GUARD = 2
const NP_COND = 3

# Eager and explicit GuardModel steps. A valid step is presence -> P gradient
# reductions -> gradient vote; the eager path adds the leading loss vote.
const EAGER_STEP = [:vote, :presence, :grad_a, :grad_c, :gradvote]
const EXPLICIT_STEP = [:presence, :grad_a, :grad_c, :gradvote]
# Eager and explicit CondModel steps. `COND_ABSENT` omits the globally absent
# v reduction but still runs the gradient vote.
const COND_FULL = [:vote, :presence, :grad_w, :grad_u, :grad_v, :gradvote]
const COND_ABSENT = [:vote, :presence, :grad_w, :grad_u, :gradvote]
const COND_EXPLICIT = [:presence, :grad_w, :grad_u, :grad_v, :gradvote]

function step_stages(np)
    np == 0 && return [:vote]
    return EAGER_STEP
end

function case_np(case)
    case in ("healthy", "grad-conditional-absence", "grad-absent-history", "grad-healthy") &&
        return NP_COND
    case in ("no-trainable", "grad-no-trainable") && return 0
    case == "grad-empty" && return 1
    return NP_GUARD
end

function expected_stages(case)
    if case == "healthy"
        # Steps 1 and 2 reduce U and V (globally present); step 3 is globally
        # absent for V; step 4 reactivates V.
        return vcat(COND_FULL, COND_FULL, COND_ABSENT, COND_FULL)
    elseif case == "grad-train-one" || case == "grad-late"
        # Two valid eager steps, then a rejected third that still completes the
        # full schedule before throwing.
        return vcat(EAGER_STEP, EAGER_STEP, EAGER_STEP)
    elseif case == "grad-audit-zero"
        # The audit 002 reproducer: steps 1 and 2 are ordinary valid steps
        # (the tied buffer and `c` reduce). At step 3 the loss uses only the
        # tied buffer, so `c` is absent and the zero-residual sqrt poisons only
        # grad_a before the vote.
        audit_step = [:vote, :presence, :grad_a, :gradvote]
        return vcat(EAGER_STEP, EAGER_STEP, audit_step)
    elseif case in ("grad-inf", "grad-ninf", "grad-update-both", "grad-overflow",
                    "grad-tied-overflow", "grad-asym-vote", "grad-caller")
        # Explicit updates: every step reduces then votes; the third rejects.
        return vcat(EXPLICIT_STEP, EXPLICIT_STEP, EXPLICIT_STEP)
    elseif case == "grad-conditional-absence"
        return vcat(COND_EXPLICIT, COND_EXPLICIT)
    elseif case == "grad-absent-history" || case == "grad-healthy"
        # Globally present steps 1, 2 and 4; v is globally absent at step 3.
        return vcat(COND_FULL, COND_FULL, COND_ABSENT, COND_FULL)
    elseif case == "grad-all-absent"
        # All gradients globally absent: presence and gradient vote only.
        return vcat(EXPLICIT_STEP, EXPLICIT_STEP, [:presence, :gradvote])
    elseif case == "grad-no-trainable"
        # No canonical buffers: the wrapper runs zero collectives, so only the
        # eager loss vote per step.
        return [:vote, :vote, :vote]
    elseif case == "grad-empty"
        # One empty canonical buffer (len 0): vacuously valid, but still one
        # presence exchange and one gradient vote.
        return [:presence, :grad_e, :gradvote]
    elseif case == "grad-frozen"
        return vcat(EXPLICIT_STEP, EXPLICIT_STEP)
    elseif case == "grad-recover"
        # One invalid eager step rejects mid-run; the next full step recovers.
        return vcat(EAGER_STEP, EAGER_STEP, EAGER_STEP, EAGER_STEP)
    end
    step = step_stages(case_np(case))
    if case == "aux-nonfinite-ok"
        return repeat(step, 3)
    elseif case == "recover-one"
        return vcat(step, step, [:vote], step)
    else
        # All remaining loss-group cases reject at item 3 after two valid steps;
        # a loss-rejected eager step issues only the loss vote.
        return vcat(step, step, [:vote])
    end
end

const GRAD_STAGES = (:grad_a, :grad_c, :grad_w, :grad_u, :grad_v, :grad_e)

function stage_meta(stage, np)
    op = "avg"
    eltype_s = stage in (:vote, :presence, :gradvote) ? "Float32" :
               (BACKEND == "nccl" ? "Float32" : "Float64")
    len = stage in (:vote, :gradvote) ? 1 : stage === :presence ? np :
          stage === :grad_a ? 2 : stage === :grad_c ? 3 :
          stage === :grad_w ? 2 : stage === :grad_u ? 3 :
          stage === :grad_v ? 4 : stage === :grad_e ? 0 :
          error("unknown stage $(repr(stage))")
    bt = stage in GRAD_STAGES && BACKEND == "nccl" ? "device" : "host"
    return (; op, eltype=eltype_s, len, buftype=bt)
end

# ------------------------------------------------------------------
# Child launch
# ------------------------------------------------------------------

function stop_probe(proc)
    kill(proc)
    if timedwait(() -> !process_running(proc), 5.0) == :timed_out
        kill(proc, Base.SIGKILL)
    end
    wait(proc)
end

function probe_command(case, fault_rank, fault_value)
    base = `$(MPI.mpiexec()) -n $RANKS $(JULIA) --project=$PROJECT_DIR --startup-file=no --color=no $PROBE --case $case --backend $BACKEND --ranks $RANKS --campaign $CAMPAIGN`
    if fault_rank >= 0
        return `$base --fault-rank $fault_rank --fault-value $fault_value`
    end
    return base
end

function run_case(spec)
    case, fault_rank, fault_value = spec
    tag = fault_rank >= 0 ? "$(case)-$(BACKEND)$(RANKS)-fault$(fault_rank)" :
                            "$(case)-$(BACKEND)$(RANKS)"
    logpath = joinpath(OUTDIR, "raw-$tag.log")
    cmd = probe_command(case, fault_rank, fault_value)

    # One shared IO handle for stdout and stderr.
    log, io = mktemp()
    started = time()
    proc = nothing
    timeout = :none
    ready = false
    try
        proc = run(pipeline(cmd, stdout=io, stderr=io), wait=false)
        deadline = started + STARTUP_TIMEOUT
        while process_running(proc)
            out = read(log, String)
            if !ready && count("GUARD_PROBE_READY rank=", out) >= RANKS
                ready = true
                deadline = time() + EXECUTION_TIMEOUT
            end
            if time() > deadline
                timeout = ready ? :execution : :startup
                stop_probe(proc)
                break
            end
            sleep(0.2)
        end
        wait(proc)
        output = read(log, String)
        write(logpath, output)
        ready = count("GUARD_PROBE_READY rank=", output) >= RANKS
        # Record argv only. Printing the `Cmd` itself would embed the parent's
        # full environment (including secrets) into metadata.md.
        cmdline = join(cmd.exec, " ")
        return (; spec, case, fault_rank, fault_value, tag, logpath, output, timeout,
                ready, exitcode=proc.exitcode, elapsed=time() - started, cmdline)
    finally
        proc !== nothing && process_running(proc) && stop_probe(proc)
        close(io)
        rm(log; force=true)
        if timeout != :none
            pids = reap_campaign(CAMPAIGN)
            isempty(pids) || println("    reaped orphan pids: $(join(pids, ' '))")
        end
    end
end

# ------------------------------------------------------------------
# Log parsing and trace validation
# ------------------------------------------------------------------

const BEGIN_RE = r"COLLECTIVE_BEGIN rank=(\d+) number=(\d+) op=(\S+) eltype=(\S+) len=(\d+) buftype=(\S+)"
const END_RE = r"COLLECTIVE_END rank=(\d+) number=(\d+)"
const RESULT_RE = r"CASE_RESULT rank=(\d+) case=(\S+) status=(\w+)(?: reason=\"([^\"]*)\")?(?: error=\"([^\"]*)\")?"
const STEP_STATE_RE = r"STEP_STATE_OK rank=(\d+) step=(\d+) ok=(true|false)"
const ABSENCE_FROZEN_RE = r"ABSENCE_FROZEN rank=(\d+) ok=(true|false)"
const REACTIVATED_RE = r"REACTIVATED rank=(\d+) ok=(true|false)"
const VERSIONS_RE = r"^VERSIONS (.*)$"m

function rank_trace(output, rank)
    begins = Tuple{Int, String, String, Int, String}[]
    for m in eachmatch(BEGIN_RE, output)
        parse(Int, m.captures[1]) == rank || continue
        push!(begins, (parse(Int, m.captures[2]), m.captures[3], m.captures[4],
            parse(Int, m.captures[5]), m.captures[6]))
    end
    ends = Int[]
    for m in eachmatch(END_RE, output)
        parse(Int, m.captures[1]) == rank || continue
        push!(ends, parse(Int, m.captures[2]))
    end
    sort!(begins, by=first)
    sort!(ends)
    return begins, ends
end

function check_trace(output, rank, expected, np)
    begins, ends = rank_trace(output, rank)
    problems = String[]
    if length(begins) != length(expected)
        push!(problems, "expected $(length(expected)) collectives, got $(length(begins))")
    end
    if length(ends) != length(begins)
        push!(problems, "$(length(begins)) begins but $(length(ends)) ends (unmatched collective)")
    end
    for (i, b) in enumerate(begins)
        b[1] == i || (push!(problems, "begin ordinal gap at position $i (number=$(b[1]))"); break)
    end
    for (i, e) in enumerate(ends)
        e == i || (push!(problems, "end ordinal gap at position $i (number=$(e))"); break)
    end
    mismatches = 0
    for i in 1:min(length(begins), length(expected))
        want = stage_meta(expected[i], np)
        got = begins[i]
        if got[2] != want.op || got[3] != want.eltype || got[4] != want.len || got[5] != want.buftype
            if mismatches < 3
                push!(problems, "ordinal $i expected $(expected[i]) (op=$(want.op) eltype=$(want.eltype) " *
                                "len=$(want.len) buftype=$(want.buftype)) got (op=$(got[2]) " *
                                "eltype=$(got[3]) len=$(got[4]) buftype=$(got[5]))")
            end
            mismatches += 1
        end
    end
    mismatches > 3 && push!(problems, "... $(mismatches - 3) more metadata mismatches")
    return problems
end

function marker_flags(re, output)
    flags = Dict{Int, Bool}()
    for m in eachmatch(re, output)
        flags[parse(Int, m.captures[1])] = m.captures[2] == "true"
    end
    return flags
end

# The healthy case must match the native reference after every step and show
# the required conditional behavior: V frozen at the globally absent step,
# then reactivated. A missing marker fails, never silently skips.
function healthy_problems(output)
    problems = String[]
    step_ok = Dict{Tuple{Int, Int}, Bool}()
    for m in eachmatch(STEP_STATE_RE, output)
        step_ok[(parse(Int, m.captures[1]), parse(Int, m.captures[2]))] = m.captures[3] == "true"
    end
    for r in 0:(RANKS - 1), i in 1:4
        if !haskey(step_ok, (r, i))
            push!(problems, "rank $r: missing STEP_STATE_OK step=$i")
        elseif !step_ok[(r, i)]
            push!(problems, "rank $r: STEP_STATE_OK step=$i ok=false")
        end
    end
    frozen = marker_flags(ABSENCE_FROZEN_RE, output)
    reactivated = marker_flags(REACTIVATED_RE, output)
    for r in 0:(RANKS - 1)
        if !haskey(frozen, r)
            push!(problems, "rank $r: missing ABSENCE_FROZEN")
        elseif !frozen[r]
            push!(problems, "rank $r: ABSENCE_FROZEN ok=false")
        end
        if !haskey(reactivated, r)
            push!(problems, "rank $r: missing REACTIVATED")
        elseif !reactivated[r]
            push!(problems, "rank $r: REACTIVATED ok=false")
        end
    end
    return problems
end

function case_results(output)
    results = Dict{Int, NamedTuple}()
    for m in eachmatch(RESULT_RE, output)
        rank = parse(Int, m.captures[1])
        results[rank] = (; case=m.captures[2], status=m.captures[3],
            reason=m.captures[4], error=m.captures[5])
    end
    return results
end

function evaluate(run)
    output = run.output
    expected = expected_stages(run.case)
    np = case_np(run.case)
    problems = String[]

    if run.timeout != :none
        push!(problems, "watchdog timeout ($(run.timeout)) after $(round(run.elapsed; digits=1))s")
    end
    if !run.ready
        push!(problems, "no GUARD_PROBE_READY on all $RANKS ranks")
    end

    results = case_results(output)
    statuses = Dict{Int, String}()
    for r in 0:(RANKS - 1)
        if haskey(results, r)
            statuses[r] = results[r].status
            results[r].error !== nothing && push!(problems, "rank $r error: $(results[r].error)")
        else
            statuses[r] = "MISSING"
            push!(problems, "missing CASE_RESULT rank=$r")
        end
    end

    untested = all(==("UNTESTED"), values(statuses)) && length(statuses) == RANKS
    if untested && run.timeout == :none && run.ready && run.exitcode == 0
        return (; verdict="UNTESTED", problems=String[], statuses)
    end

    for r in 0:(RANKS - 1)
        statuses[r] == "PASS" || statuses[r] == "MISSING" ||
            push!(problems, "rank $r CASE_RESULT status=$(statuses[r])")
    end

    for r in 0:(RANKS - 1)
        append!(problems, ["rank $r: $p" for p in check_trace(output, r, expected, np)])
    end

    run.case in ("healthy", "grad-absent-history", "grad-healthy") &&
        append!(problems, healthy_problems(output))

    if run.timeout == :none && run.exitcode != 0
        push!(problems, "child exit code $(run.exitcode)")
    end

    verdict = isempty(problems) ? "PASS" : "FAIL"
    return (; verdict, problems, statuses)
end

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

function source_identity()
    flux_root = dirname(PROJECT_DIR)
    head = try
        strip(read(`git -C $flux_root rev-parse HEAD`, String))
    catch
        "unknown"
    end
    hashes = String[]
    for rel in ("src/train.jl", "src/distributed/public_api.jl")
        h = try
            first(split(strip(read(`md5sum $flux_root/$rel`, String))))
        catch
            "unknown"
        end
        push!(hashes, "$rel=$h")
    end
    return head, join(hashes, " ")
end

function main_body()
    mkpath(OUTDIR)
    table = case_table(RANKS)
    selected = CASE_FILTER === nothing ? table : filter(spec -> spec[1] == CASE_FILTER, table)
    isempty(selected) && error("--case $(repr(CASE_FILTER)) matched no cases for ranks=$RANKS")

    println("DB-1/DB-4/DB-3 numerical-guard check (group=$GROUP)")
    println("probe       : $PROBE")
    println("project     : $PROJECT_DIR")
    println("backend     : $BACKEND, ranks: $RANKS")
    println("output dir  : $OUTDIR")
    println("phase       : $PHASE")
    println("watchdogs   : startup=$(STARTUP_TIMEOUT) s, execution=$(EXECUTION_TIMEOUT) s")
    println("cases       : $(length(selected))")

    head, hashes = source_identity()
    versions = "unknown"
    runs = NamedTuple[]

    for spec in selected
        println("\n=== case: $(spec[1]) backend=$BACKEND ranks=$RANKS fault=$(spec[2]) value=$(spec[3]) ===")
        run = run_case(spec)
        evaluation = evaluate(run)
        push!(runs, merge(run, evaluation))
        v = match(VERSIONS_RE, run.output)
        if v !== nothing && versions == "unknown"
            versions = v.captures[1]
        end
        println("    log=$(basename(run.logpath)) timeout=$(run.timeout) exitcode=$(run.exitcode) " *
                "elapsed=$(round(run.elapsed; digits=1))s")
        println("    verdict=$(evaluation.verdict)")
        for p in evaluation.problems
            println("      - $p")
        end
        if evaluation.verdict == "UNTESTED"
            println("      UNTESTED: $(spec[1]) did not run in this environment (not silently skipped)")
        end
    end

    # summary.tsv
    summary_path = joinpath(OUTDIR, "summary.tsv")
    open(summary_path, "w") do io
        println(io, "case\tbackend\tranks\tfault_rank\tfault_value\tverdict\treason")
        for r in runs
            reason = isempty(r.problems) ? "" : replace(join(r.problems, "; "), '\t' => ' ', '\n' => ' ')
            println(io, join((r.case, BACKEND, RANKS, r.fault_rank, r.fault_value, r.verdict, reason), '\t'))
        end
    end

    # metadata.md
    metadata_path = joinpath(OUTDIR, "metadata.md")
    open(metadata_path, "w") do io
        println(io, "# DB-1/DB-4/DB-3 numerical-guard check metadata")
        println(io)
        println(io, "- date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "- phase: $PHASE")
        println(io, "- group: $GROUP (loss = DB-1/DB-4 loss vote; gradient = DB-3 gradient vote; " *
                    "all = loss cases followed by gradient cases in one campaign)")
        println(io, "- backend: $BACKEND")
        println(io, "- ranks: $RANKS")
        println(io, "- outdir: $OUTDIR")
        println(io, "- parent julia: $(VERSION)")
        println(io, "- child project: $PROJECT_DIR")
        println(io, "- Flux source: HEAD $head ($hashes)")
        println(io, "- child versions (VERSIONS markers): $versions")
        println(io, "- startup watchdog: $STARTUP_TIMEOUT s (NUMERICAL_GUARD_STARTUP_TIMEOUT)")
        println(io, "- execution watchdog: $EXECUTION_TIMEOUT s (NUMERICAL_GUARD_EXECUTION_TIMEOUT)")
        println(io, "- timeout semantics: a timeout always fails the case, even with plausible markers")
        println(io, "- gradient vote: every nonempty `_sync_gradients` runs one extra host Float32 " *
                    "len-1 `avg` collective after the presence and gradient reductions; `np == 0` " *
                    "keeps zero collectives")
        println(io, "- rejected steps: every rank completes the full expected schedule before " *
                    "throwing (eager: loss vote -> presence -> P reductions -> gradient vote; " *
                    "explicit: presence -> P reductions -> gradient vote); a loss-rejected eager " *
                    "step issues only the loss vote")
        println(io, "- classification: `:vote`, `:presence` and `:gradvote` are metadata-identical " *
                    "(host Float32 len 1; `:presence` when np == 1), so stages are classified by " *
                    "ordinal within the explicit expected schedule")
        println(io, "- schedules: eager GuardModel = vote, presence, grad_a, grad_c, gradvote; " *
                    "explicit GuardModel = presence, grad_a, grad_c, gradvote; explicit CondModel = " *
                    "presence, grad_w, grad_u, grad_v, gradvote; eager CondModel = vote, presence, " *
                    "grad_w, grad_u, grad_v, gradvote, or without grad_v when v is globally absent; " *
                    "no-trainable = one loss vote per eager step and zero wrapper collectives; " *
                    "empty buffer = presence, grad_e (len 0), gradvote")
        println(io)
        println(io, "## Cases")
        println(io)
        println(io, "| case | fault_rank | fault_value | verdict | timeout | exit | elapsed_s | log |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for r in runs
            println(io, "| $(r.case) | $(r.fault_rank) | $(r.fault_value) | $(r.verdict) | " *
                        "$(r.timeout) | $(r.exitcode) | $(round(r.elapsed; digits=1)) | " *
                        "$(basename(r.logpath)) |")
        end
        println(io)
        println(io, "## Per-case commands and reasons")
        for r in runs
            println(io)
            println(io, "### $(r.case) fault_rank=$(r.fault_rank) fault_value=$(r.fault_value)")
            println(io)
            println(io, "- command: `$(r.cmdline)`")
            println(io, "- timeout: $(r.timeout)")
            println(io, "- exit status: $(r.exitcode)")
            println(io, "- verdict: $(r.verdict)")
            if isempty(r.problems)
                println(io, "- reason: none")
            else
                for p in r.problems
                    println(io, "- reason: $p")
                end
            end
        end
    end

    npass = count(r -> r.verdict == "PASS", runs)
    nfail = count(r -> r.verdict == "FAIL", runs)
    nun = count(r -> r.verdict == "UNTESTED", runs)
    pids = reap_campaign(CAMPAIGN)
    isempty(pids) || println("reaped final orphan pids: $(join(pids, ' '))")
    println("\nsummary: PASS=$npass FAIL=$nfail UNTESTED=$nun")
    println("artifacts: $(joinpath(OUTDIR, "summary.tsv"))")
    for r in runs
        r.verdict == "UNTESTED" &&
            println("UNTESTED $(r.tag) (reported explicitly, not silently skipped)")
    end
    return nfail == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_body())
end
