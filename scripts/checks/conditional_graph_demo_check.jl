# Bounded Phase K toy campaign driver.
#
# Orchestrates, with captured exit codes and elapsed times:
#   prepare (single process, no MPI backend)
#   train + fresh-process resume per configuration (two/four MPI ranks, or
#   two NCCL ranks on a GPU host)
# Each child writes its console log and artifacts under `--root`. The driver
# checks exit codes, per-rank step counts, done markers, and resume match
# markers, then writes `summary.txt` and exits 0 only if every cell passed.
#
# Run under the Flux implementation environment, for example:
#   julia --project=../flux-integration/test scripts/checks/conditional_graph_demo_check.jl \
#       --root artifacts/logs/conditional-wrapper/demo/toy-<id>-mpi --backend mpi --ranks 2,4
#   julia --project=../flux-integration/test scripts/checks/conditional_graph_demo_check.jl \
#       --root artifacts/logs/conditional-wrapper/demo/toy-<id>-nccl --backend nccl --ranks 2

using MPI, Test, Dates, Printf, Serialization, Flux, Optimisers
include(joinpath(@__DIR__, "..", "examples", "conditional_graph", "model.jl"))
include(joinpath(@__DIR__, "..", "examples", "conditional_graph", "schedule.jl"))
include(joinpath(@__DIR__, "..", "examples", "conditional_graph", "dataset.jl"))
include(joinpath(@__DIR__, "..", "examples", "conditional_graph", "metrics.jl"))

const DEMO = normpath(joinpath(@__DIR__, "..", "examples", "conditional_graph_demo.jl"))
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
const ROOT = abspath(get(OPTS, "root",
    joinpath("artifacts", "logs", "conditional-wrapper", "demo",
        "toy-" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))))
const BACKEND = get(OPTS, "backend", "mpi")
const RANKS = parse.(Int, split(get(OPTS, "ranks", "2,4"), ','))
const TIMEOUT = parse(Float64, get(OPTS, "timeout", "1800"))
const DATASET = get(OPTS, "dataset", "toy")
const LABEL = get(OPTS, "label", "phase-k-$DATASET")
const STEPS = parse(Int, get(OPTS, "steps", DATASET == "toy" ? "16" : "120"))
ispath(ROOT) && error("refusing existing campaign root $ROOT")

const RESULTS = NamedTuple[]

function run_child(name, nranks, args; timeout = TIMEOUT)
    dir = joinpath(ROOT, name)
    mkpath(dir)
    log = joinpath(dir, "console.log")
    cmd = nranks > 1 ?
        `$(MPI.mpiexec()) -n $(nranks) $(JULIA) --startup-file=no --project=$(PROJECT) $(DEMO) $(args)` :
        `$(JULIA) --startup-file=no --project=$(PROJECT) $(DEMO) $(args)`
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
    result = (; name, nranks, exitcode = proc.exitcode,
        timed_out, elapsed, log, output)
    push!(RESULTS, result)
    status = result.exitcode == 0 ? "PASS" : "FAIL"
    @printf("[%s] %-14s ranks=%d exit=%s elapsed=%.1fs log=%s\n",
        status, name, nranks, string(result.exitcode), elapsed, log)
    return result
end

function count_occurrences(text, needle)
    return length(collect(eachmatch(Regex(escape_string(needle)), text)))
end

function check_job(result, nranks, mode, expected_steps)
    problems = String[]
    result.exitcode == 0 || push!(problems,
        "$(result.name): exit code $(result.exitcode) (timed_out=$(result.timed_out))")
    result.timed_out && push!(problems, "$(result.name): exceeded $(TIMEOUT)s timeout")
    n_step = count_occurrences(result.output, "DEMO_STEP mode=$mode")
    n_expect = nranks * expected_steps
    n_step == n_expect || push!(problems,
        "$(result.name): $(n_step) DEMO_STEP lines, expected $(n_expect) ($nranks ranks × $expected_steps steps)")
    n_rank_done = count_occurrences(result.output, "DEMO_RANK_DONE mode=$mode")
    n_rank_done == nranks || push!(problems,
        "$(result.name): $(n_rank_done) DEMO_RANK_DONE lines, expected $(nranks)")
    upper = uppercase(mode)
    occursin("DEMO_$(upper)_DONE", result.output) ||
        push!(problems, "$(result.name): missing DEMO_$(upper)_DONE marker")
    if mode == "resume"
        n_match = count_occurrences(result.output, "resume_match=true")
        n_match == n_expect || push!(problems,
            "$(result.name): $(n_match) resume_match=true markers, expected $(n_expect)")
    end
    occursin("ERROR: LoadError", result.output) &&
        push!(problems, "$(result.name): log contains an ERROR: LoadError")
    occursin("DEMO_SCHEMA_OK mode=$mode", result.output) || push!(problems, "missing schema validation marker")
    # A total count alone could hide duplicate ranks or missing steps.
    keys_seen = [(parse(Int,m[1]), parse(Int,m[2])) for m in eachmatch(
        Regex("DEMO_STEP mode=$mode step=(\\d+) cycle=\\d+ rank=(\\d+)"), result.output)]
    first_step = mode == "train" ? 1 : 3
    expected_keys = Set((s,r) for s in first_step:(first_step+expected_steps-1) for r in 0:nranks-1)
    length(keys_seen) == length(expected_keys) && Set(keys_seen) == expected_keys ||
        push!(problems, "missing or duplicate rank/step markers")
    return problems
end

function summary_text()
    io = IOBuffer()
    println(io, "Phase K $DATASET campaign summary")
    println(io, "date\t", Dates.now())
    println(io, "root\t", ROOT)
    println(io, "backend\t", BACKEND)
    println(io, "ranks\t", RANKS)
    println(io, "label\t", LABEL)
    println(io, "julia\t", VERSION)
    println(io, "project\t", PROJECT)
    println(io, "steps\t", STEPS)
    println(io, "command\t", Base.julia_cmd(), " ", join(ARGS, ' '))
    for r in RESULTS
        println(io, "job\t", r.name, "\tranks=", r.nranks, "\texit=", r.exitcode,
            "\telapsed=", round(r.elapsed; digits = 3), "\tlog=", r.log)
    end
    return String(take!(io))
end

function main()
    DATASET in ("toy", "mnist-fashion") || error("invalid dataset")
    @testset "Phase K $DATASET campaign ($BACKEND)" begin
        # 1. Prepare the dataset, schedule, initial model, and native tree.
        problems = String[]
        prepared = get(OPTS, "prepared", joinpath(ROOT, "prepared", "$(DATASET)_prepared.jls"))
        if !haskey(OPTS, "prepared")
            args = ["prepare", "--out", joinpath(ROOT, "prepared"), "--dataset", DATASET,
                "--worlds", join(RANKS, ','), "--steps", string(STEPS)]
            for key in ("cache", "eta", "seed-train", "seed-val", "seed-subset", "seed-init", "seed-order", "train-per-task", "val-per-task")
                haskey(OPTS, key) && append!(args, ["--$key", OPTS[key]])
            end
            prep = run_child("prepared", 1, args)
            @test prep.exitcode == 0
            @test occursin("DEMO_PREPARE_DONE", prep.output)
            prep.exitcode == 0 || error("preparation failed; see $(prep.log)")
        end
        payload = open(deserialize, prepared)
        string(payload.dataset.name) == DATASET || error("prepared dataset choice differs")
        payload.default_steps == STEPS || error("prepared step count differs")
        steps = STEPS
        for R in RANKS
            # 2. Uninterrupted trajectory with per-step reference checks and a
            #    checkpoint after step 2.
            train_dir = joinpath(ROOT, "$(BACKEND)$(R)", "train")
            run_id = basename(ROOT) * "-$(BACKEND)$(R)"
            train = run_child("$(BACKEND)$(R)-train", R,
                ["train", "--prepared", prepared, "--out", train_dir,
                    "--backend", BACKEND, "--save-at", "2", "--save-trajectory",
                    "--label", LABEL, "--run-id", run_id])
            append!(problems, check_job(train, R, "train", steps))
            @test occursin("DEMO_TRAIN_DONE", train.output)
            @test occursin("DEMO_CHECKPOINT step_saved=2 step_next=3", train.output)
            train.exitcode == 0 || error("training failed; see $(train.log)")
            sched = payload.schedules[R]
            @test CGMetrics.validate_outputs(train_dir, "train", 1, steps, R, sched, payload.dataset;
                run_id, schedule_digest=CGSchedule.schedule_digest(sched), eta=payload.optimizer.eta)

            # 3. Fresh processes restore the data-only checkpoint and continue
            #    from step 3.
            resume_dir = joinpath(ROOT, "$(BACKEND)$(R)", "resume")
            resume = run_child("$(BACKEND)$(R)-resume", R,
                ["resume",
                    "--checkpoint", joinpath(train_dir, "checkpoint-step2.jls"),
                    "--expected", joinpath(train_dir, "trajectory.jls"),
                    "--out", resume_dir, "--backend", BACKEND,
                    "--label", LABEL])
            append!(problems, check_job(resume, R, "resume", steps - 2))
            @test occursin("DEMO_RESUME_DONE", resume.output)
            resume.exitcode == 0 || error("resume failed; see $(resume.log)")
            @test CGMetrics.validate_outputs(resume_dir, "resume", 3, steps, R, sched, payload.dataset;
                run_id, schedule_digest=CGSchedule.schedule_digest(sched), eta=payload.optimizer.eta)
        end
        @test isempty(problems)
        for p in problems
            @info "problem" p
        end
    end
    return true
end

ok = try
    main()
catch err
    showerror(stderr, err)
    println(stderr)
    false
end

CG_ok = ok && all(r -> r.exitcode == 0, RESULTS)
mkpath(ROOT)
write(joinpath(ROOT, "summary.txt"), summary_text() * "status\t$(CG_ok ? "PASS" : "FAIL")\n")
println("\nPHASE_K_SUMMARY dataset=$DATASET backend=$BACKEND ranks=$(join(RANKS, ',')) ",
    CG_ok ? "PASS" : "FAIL", " root=$ROOT")
exit(CG_ok ? 0 : 1)
