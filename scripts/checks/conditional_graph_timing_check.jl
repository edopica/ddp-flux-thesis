# Bounded Phase K timing-stretch driver.
#
# Orchestrates, with captured exit codes and elapsed times:
#   correctness  single and DDP fixed-work runs compared against the saved
#                trajectory snapshots, or against each other through
#                `--dump-final`;
#   baseline     one single-process run (`--device cpu|cuda`);
#   ddp          one MPI/NCCL run (`--backend mpi|nccl`, `--ranks` processes);
#   timing       baseline + ddp only, to measure the speedup of the timed
#                training phase (equivalent to `all` without correctness).
#
# Each child writes its console log and artifacts under `--root/<phase>`. The
# driver checks exit codes, timing markers, correctness PASS lines, and writes
# `summary-<phase>.txt` plus `TIMING_STRETCH_SUMMARY`. A measured speedup below
# 1 is a valid result and never fails the campaign.
#
# Run under the Flux implementation environment, for example:
#   julia --project=../flux-integration/test scripts/checks/conditional_graph_timing_check.jl \
#       --root artifacts/logs/conditional-wrapper/timing/stretch-local \
#       --prepared <p.jls> --backend mpi --device cpu --ranks 4 --phase all

using MPI, Test, Dates, Printf, Serialization

const TIMING = normpath(joinpath(@__DIR__, "..", "examples", "conditional_graph_timing.jl"))
const PROJECT = dirname(Base.active_project())
const JULIA = Base.julia_cmd()

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $(repr(a))")
        key = a[3:end]
        i + 1 <= length(args) || error("option --$key requires a value")
        opts[key] = args[i + 1]
        i += 2
    end
    return opts
end

const OPTS = parse_args(copy(ARGS))
const PHASE = get(OPTS, "phase", "all")
PHASE in ("all", "correctness", "baseline", "ddp", "timing") ||
    error("--phase must be all, correctness, baseline, ddp, or timing")
const PHASES = PHASE == "all" ? ["correctness", "baseline", "ddp"] :
    PHASE == "timing" ? ["baseline", "ddp"] : [PHASE]
const ROOT = abspath(get(OPTS, "root",
    joinpath("artifacts", "logs", "conditional-wrapper", "timing",
        "stretch-" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))))
const PREPARED = get(OPTS, "prepared", nothing)
PREPARED === nothing && error("--prepared is required")
const TRAJECTORY = get(OPTS, "trajectory", nothing)
const BACKEND = get(OPTS, "backend", "mpi")
const DEVICE = get(OPTS, "device", BACKEND == "nccl" ? "cuda" : "cpu")
const RANKS = parse(Int, get(OPTS, "ranks", "4"))
const STEPS = parse(Int, get(OPTS, "steps", "120"))
const WARMUP = parse(Int, get(OPTS, "warmup", "3"))
const REPEATS = parse(Int, get(OPTS, "repeats", "5"))
const RECORD_EVERY = parse(Int, get(OPTS, "record-every", "5"))
const CHECK_STEPS = parse(Int, get(OPTS, "check-steps", "12"))
const TIMEOUT = parse(Float64, get(OPTS, "timeout", "1800"))
const LABEL = get(OPTS, "label", "phase-k-timing")
const ATOL = parse(Float64, get(OPTS, "atol", "2.0e-6"))
const RTOL = parse(Float64, get(OPTS, "rtol", "1.0e-5"))

for phase in PHASES
    dir = joinpath(ROOT, phase)
    ispath(dir) && error("refusing existing phase directory $dir")
end
mkpath(ROOT)

const RESULTS = NamedTuple[]
const PROBLEMS = String[]
const SPEEDUP = Ref{Union{Nothing,NamedTuple}}(nothing)

function run_child(name, nranks, args; out, timeout = TIMEOUT)
    dir = joinpath(ROOT, name)
    mkpath(dir)
    log = joinpath(dir, "console.log")
    cmd = nranks > 1 ?
        `$(MPI.mpiexec()) -n $(nranks) $(JULIA) --startup-file=no --project=$(PROJECT) $(TIMING) $(args)` :
        `$(JULIA) --startup-file=no --project=$(PROJECT) $(TIMING) $(args)`
    cmd = addenv(cmd, "JULIA_PKG_PRECOMPILE_AUTO" => "0")
    write(joinpath(dir, "command.txt"), string(cmd) * "\n")
    started = time()
    io = open(log, "w")
    proc = run(pipeline(cmd, stdout = io, stderr = io), wait = false)
    timed_out = false
    deadline = started + timeout
    while process_running(proc)
        if time() > deadline
            timed_out = true
            kill(proc)
            if timedwait(() -> !process_running(proc), 5.0) == :timed_out
                kill(proc, Base.SIGKILL)
            end
            break
        end
        sleep(0.5)
    end
    wait(proc)
    close(io)
    elapsed = time() - started
    output = read(log, String)
    result = (; name, nranks, out, exitcode = proc.exitcode, timed_out, elapsed, log, output)
    push!(RESULTS, result)
    status = result.exitcode == 0 ? "PASS" : "FAIL"
    @printf("[%s] %-18s ranks=%d exit=%s elapsed=%.1fs log=%s\n",
        status, name, nranks, string(result.exitcode), elapsed, log)
    return result
end

function count_occurrences(text, needle)
    return length(collect(eachmatch(Regex(escape_string(needle)), text)))
end

function timing_args(mode; out, check = false, dump = nothing)
    args = [mode, "--prepared", PREPARED, "--out", out, "--label", LABEL,
        "--steps", string(STEPS), "--warmup", string(WARMUP),
        "--repeats", string(REPEATS), "--record-every", string(RECORD_EVERY),
        "--world", string(RANKS)]
    if mode == "single"
        append!(args, ["--device", DEVICE])
    else
        append!(args, ["--backend", BACKEND])
    end
    if check
        append!(args, ["--check-trajectory", TRAJECTORY, "--check-steps", string(CHECK_STEPS)])
    end
    dump === nothing || append!(args, ["--dump-final", dump])
    return args
end

function check_job(result, mode; trajectory_check = false)
    problems = String[]
    result.exitcode == 0 || push!(problems,
        "$(result.name): exit code $(result.exitcode) (timed_out=$(result.timed_out))")
    result.timed_out && push!(problems, "$(result.name): exceeded $(TIMEOUT)s timeout")
    occursin("TIMING_START", result.output) ||
        push!(problems, "$(result.name): missing TIMING_START marker")
    occursin("TIMING_DONE mode=$mode", result.output) ||
        push!(problems, "$(result.name): missing TIMING_DONE mode=$mode marker")
    occursin("TIMING_SUMMARY mode=$mode", result.output) ||
        push!(problems, "$(result.name): missing TIMING_SUMMARY mode=$mode marker")
    if result.nranks > 1
        n_done = count_occurrences(result.output, "TIMING_RANK_DONE mode=$mode")
        n_expected = result.nranks
        n_done == n_expected || push!(problems,
            "$(result.name): $(n_done) per-rank TIMING_RANK_DONE markers, expected $(n_expected)")
    end
    if trajectory_check
        occursin("TIMING_CHECK_RESULT source=trajectory PASS", result.output) ||
            push!(problems, "$(result.name): missing trajectory PASS line")
    end
    occursin("ERROR: LoadError", result.output) &&
        push!(problems, "$(result.name): log contains an ERROR: LoadError")
    done = joinpath(result.out, "done")
    isfile(done) || push!(problems, "$(result.name): missing done marker $done")
    isfile(joinpath(result.out, "records.tsv")) ||
        push!(problems, "$(result.name): missing records.tsv")
    isfile(joinpath(result.out, "repeats.tsv")) ||
        push!(problems, "$(result.name): missing repeats.tsv")
    return problems
end

function compare_dumps(path_a, path_b)
    a = open(deserialize, path_a)
    b = open(deserialize, path_b)
    length(a.values) == length(b.values) || error("dump length mismatch")
    maxabs = 0.0
    nbad = 0
    for i in eachindex(a.values)
        x = a.values[i]
        y = b.values[i]
        e = abs(x - y)
        maxabs = max(maxabs, e)
        nbad += e <= ATOL + RTOL * abs(y) ? 0 : 1
    end
    return (; maxabs, nbad, n = length(a.values), ok = nbad == 0)
end

function rank_dump_problems(ddp_dump)
    problems = String[]
    base = open(deserialize, ddp_dump)
    for r in 0:(RANKS - 1)
        path = "$ddp_dump.rank$r.jls"
        isfile(path) || continue
        other = open(deserialize, path)
        equal = length(other.values) == length(base.values) &&
                all(other.values .== base.values) &&
                all(other.layout[i] == base.layout[i] for i in eachindex(base.layout))
        @printf("TIMING_RANK_DUMP rank=%d equal=%s\n", r, string(equal))
        equal || push!(problems, "rank $r dump differs from rank 0")
    end
    return problems
end

function run_correctness()
    problems = String[]
    dir = joinpath(ROOT, "correctness")
    if TRAJECTORY !== nothing
        single = run_child("correctness-single", 1,
            timing_args("single"; out = joinpath(dir, "single"), check = true);
            out = joinpath(dir, "single"))
        append!(problems, check_job(single, "single"; trajectory_check = true))
        ddp = run_child("correctness-ddp", RANKS,
            timing_args("ddp"; out = joinpath(dir, "ddp"), check = true);
            out = joinpath(dir, "ddp"))
        append!(problems, check_job(ddp, "ddp"; trajectory_check = true))
    else
        single_dump = joinpath(dir, "single-final.jls")
        ddp_dump = joinpath(dir, "ddp-final.jls")
        single = run_child("correctness-single", 1,
            timing_args("single"; out = joinpath(dir, "single"), dump = single_dump);
            out = joinpath(dir, "single"))
        append!(problems, check_job(single, "single"))
        ddp = run_child("correctness-ddp", RANKS,
            timing_args("ddp"; out = joinpath(dir, "ddp"), dump = ddp_dump);
            out = joinpath(dir, "ddp"))
        append!(problems, check_job(ddp, "ddp"))
        if single.exitcode == 0 && ddp.exitcode == 0
            cmp = compare_dumps(single_dump, ddp_dump)
            @printf("TIMING_CHECK_RESULT source=single-vs-ddp %s maxabs=%.6e nbad=%d\n",
                cmp.ok ? "PASS" : "FAIL", cmp.maxabs, cmp.nbad)
            cmp.ok || push!(problems,
                "single-vs-ddp snapshot mismatch: maxabs=$(cmp.maxabs) nbad=$(cmp.nbad)")
            append!(problems, rank_dump_problems(ddp_dump))
        end
    end
    return problems
end

function parse_summary(output, mode)
    pattern = Regex("TIMING_SUMMARY mode=" * mode *
        " repeats=(\\d+) median_total_s=([0-9.eE+-]+) min_total_s=([0-9.eE+-]+)" *
        " max_total_s=([0-9.eE+-]+) median_train_s=([0-9.eE+-]+) min_train_s=([0-9.eE+-]+)" *
        " max_train_s=([0-9.eE+-]+) median_updates_per_s=([0-9.eE+-]+)" *
        " median_train_updates_per_s=([0-9.eE+-]+)")
    m = match(pattern, output)
    m === nothing && return nothing
    return (; repeats = parse(Int, m[1]), median_total_s = parse(Float64, m[2]),
        min_total_s = parse(Float64, m[3]), max_total_s = parse(Float64, m[4]),
        median_train_s = parse(Float64, m[5]), min_train_s = parse(Float64, m[6]),
        max_train_s = parse(Float64, m[7]), median_updates_per_s = parse(Float64, m[8]),
        median_train_updates_per_s = parse(Float64, m[9]))
end

function run_phase(name)
    mode = name == "baseline" ? "single" : "ddp"
    nranks = name == "baseline" ? 1 : RANKS
    out = joinpath(ROOT, name)
    result = run_child(name, nranks, timing_args(mode; out = out); out = out)
    return result, check_job(result, mode)
end

function summary_text(speedups, phase_results)
    io = IOBuffer()
    println(io, "Phase K timing stretch driver summary")
    println(io, "date\t", Dates.now())
    println(io, "root\t", ROOT)
    println(io, "prepared\t", abspath(PREPARED))
    println(io, "trajectory\t", TRAJECTORY === nothing ? "none" : abspath(TRAJECTORY))
    println(io, "backend\t", BACKEND)
    println(io, "device\t", DEVICE)
    println(io, "ranks\t", RANKS)
    println(io, "label\t", LABEL)
    println(io, "julia\t", VERSION)
    println(io, "project\t", PROJECT)
    println(io, "steps\t", STEPS)
    println(io, "warmup\t", WARMUP)
    println(io, "repeats\t", REPEATS)
    println(io, "record_every\t", RECORD_EVERY)
    println(io, "check_steps\t", CHECK_STEPS)
    println(io, "phase\t", PHASE)
    if speedups === nothing
        println(io, "speedup_train\tna")
        println(io, "speedup_total\tna")
        println(io, "median_single_train_s\tna")
        println(io, "median_ddp_slowest_train_s\tna")
        println(io, "median_single_total_s\tna")
        println(io, "median_ddp_slowest_total_s\tna")
    else
        println(io, "speedup_train\t", @sprintf("%.6f", speedups.train))
        println(io, "speedup_total\t", @sprintf("%.6f", speedups.total))
        println(io, "median_single_train_s\t", @sprintf("%.6f", speedups.median_single_train_s))
        println(io, "median_ddp_slowest_train_s\t",
            @sprintf("%.6f", speedups.median_ddp_slowest_train_s))
        println(io, "median_single_total_s\t", @sprintf("%.6f", speedups.median_single_total_s))
        println(io, "median_ddp_slowest_total_s\t",
            @sprintf("%.6f", speedups.median_ddp_slowest_total_s))
    end
    println(io, "command\t", Base.julia_cmd(), " ", join(ARGS, ' '))
    for r in RESULTS
        println(io, "job\t", r.name, "\tranks=", r.nranks, "\texit=", r.exitcode,
            "\telapsed=", round(r.elapsed; digits = 3), "\tlog=", r.log)
    end
    for p in PROBLEMS
        println(io, "problem\t", p)
    end
    return String(take!(io))
end

function main()
    @testset "Phase K timing stretch ($BACKEND, $PHASE)" begin
        if "correctness" in PHASES
            append!(PROBLEMS, run_correctness())
        end
        baseline_result = nothing
        ddp_result = nothing
        if "baseline" in PHASES
            baseline_result, problems = run_phase("baseline")
            append!(PROBLEMS, problems)
        end
        if "ddp" in PHASES
            ddp_result, problems = run_phase("ddp")
            append!(PROBLEMS, problems)
        end
        if baseline_result !== nothing && ddp_result !== nothing &&
                baseline_result.exitcode == 0 && ddp_result.exitcode == 0
            base = parse_summary(baseline_result.output, "single")
            ddp = parse_summary(ddp_result.output, "ddp")
            if base !== nothing && ddp !== nothing &&
                    ddp.median_train_s > 0 && ddp.median_total_s > 0
                speedup_train = base.median_train_s / ddp.median_train_s
                speedup_total = base.median_total_s / ddp.median_total_s
                SPEEDUP[] = (; train = speedup_train, total = speedup_total,
                    median_single_train_s = base.median_train_s,
                    median_ddp_slowest_train_s = ddp.median_train_s,
                    median_single_total_s = base.median_total_s,
                    median_ddp_slowest_total_s = ddp.median_total_s)
                @printf("TIMING_SPEEDUP backend=%s ranks=%d median_single_train_s=%.6f median_ddp_slowest_train_s=%.6f speedup_train=%.6f median_single_total_s=%.6f median_ddp_slowest_total_s=%.6f speedup_total=%.6f\n",
                    BACKEND, RANKS, base.median_train_s, ddp.median_train_s, speedup_train,
                    base.median_total_s, ddp.median_total_s, speedup_total)
            else
                push!(PROBLEMS, "could not parse TIMING_SUMMARY lines for the speedup")
            end
        end
        @test isempty(PROBLEMS)
    end
    return SPEEDUP[]
end

function guarded()
    speedups = try
        main()
    catch err
        showerror(stderr, err)
        println(stderr)
        push!(PROBLEMS, "driver error: $(sprint(showerror, err))")
        nothing
    end
    ok = isempty(PROBLEMS) && all(r -> r.exitcode == 0, RESULTS)
    write(joinpath(ROOT, "summary-$(PHASE).txt"),
        summary_text(speedups, RESULTS) * "status\t$(ok ? "PASS" : "FAIL")\n")
    @printf("\nTIMING_STRETCH_SUMMARY backend=%s ranks=%d speedup_train=%s speedup_total=%s status=%s\n",
        BACKEND, RANKS,
        speedups === nothing ? "na" : @sprintf("%.6f", speedups.train),
        speedups === nothing ? "na" : @sprintf("%.6f", speedups.total),
        ok ? "PASS" : "FAIL")
    for p in PROBLEMS
        @info "problem" p
    end
    return ok
end

ok = guarded()
exit(ok ? 0 : 1)
