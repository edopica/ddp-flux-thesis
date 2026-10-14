# Acceptance check for the pinned Phase K collective protocol (review R1).
#
# `collective_summary` must select the expected collective count from the
# bundle's explicit `collective_protocol` pin, never from the observed record
# counts: historical pre-DB-3 evidence keeps the 5/7 protocol, while guarded
# DB-3 campaigns require the 6/8 protocol and its trailing host `Float32`
# length-1 `avg` gradient-validity vote. The negative controls prove that a
# wrong or missing protocol pin throws instead of silently accepting either
# count.
#
# Pure Julia stdlib: no project, no Flux, no MPI, no devices. Run from the
# thesis root:
#
#   julia --startup-file=no scripts/checks/phase_k_collective_protocol_check.jl [ROOT]
#
# ROOT optionally overrides the guarded DB-3 toy campaign root (repo-relative).
# The guarded artifacts are produced on deathstar; the default root is
# `artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi`.
#
# Prints `PHASE_K_PROTOCOL <name> PASS` or `PHASE_K_PROTOCOL <name> FAIL: <error>`
# for every check, then `PHASE_K_PROTOCOL_SUMMARY PASS` and exit 0 only when
# every check passed; otherwise it prints the failure list and exits 1.

include(joinpath(@__DIR__, "..", "..", "notebook", "phase_k_workflow_helpers.jl"))

using .PhaseKWorkflow

const DEFAULT_DB3_TOY_ROOT =
    joinpath(PHASE_K_ROOT, "demo", "db3-toy-2026-10-14-mpi")

const FAILURES = String[]

function report(name, ok, detail)
    if ok
        println("PHASE_K_PROTOCOL ", name, " PASS")
    else
        println("PHASE_K_PROTOCOL ", name, " FAIL",
            isempty(detail) ? "" : ": " * detail)
        push!(FAILURES, name)
    end
    return ok
end

function captured_error(body)
    try
        body()
        return nothing
    catch err
        return sprint(showerror, err)
    end
end

function check_passes(name, body)
    message = captured_error(body)
    return report(name, message === nothing, something(message, ""))
end

function check_throws(name, body; contains)
    message = captured_error(body)
    message === nothing &&
        return report(name, false, "call unexpectedly succeeded")
    matches = occursin(contains, message)
    return report(name, matches, matches ? "" :
        "threw, but the error does not contain $(repr(contains)): " * message)
end

# The guarded bundle needs `campaign_record` because `bundle_artifacts` builds
# its path; `load_bundle` and `collective_summary` never read the file.
function db3_toy_bundle(root; collective_protocol = "guarded-6/8")
    fields = (
        key = "db3-toy-mpi2",
        config = "mpi2",
        ranks = 2,
        run_root = root,
        prepared = joinpath(root, "prepared"),
        campaign_record = joinpath(root, "CAMPAIGN.md"),
        checkpoint_step = 2,
        steps = 16,
    )
    collective_protocol === nothing && return fields
    return merge(fields, (collective_protocol = collective_protocol,))
end

for key in ("tutorial-2rank", "tutorial-4rank")
    check_passes(key,
        () -> collective_summary(load_bundle(resolve_bundle(key))))
end

guarded_root = isempty(ARGS) ? DEFAULT_DB3_TOY_ROOT : ARGS[1]
if !isdir(joinpath(REPO_ROOT, guarded_root))
    report("db3-toy-mpi2", false,
        "DB-3 toy campaign root $(guarded_root) is missing; the guarded artifacts " *
        "are required and exist on deathstar")
else
    check_passes("db3-toy-mpi2",
        () -> collective_summary(load_bundle(db3_toy_bundle(guarded_root))))

    check_throws("guarded-pinned-legacy-5/7",
        () -> collective_summary(load_bundle(
            db3_toy_bundle(guarded_root; collective_protocol = "legacy-5/7")));
        contains = "collectives, metrics record")

    check_throws("guarded-without-protocol",
        () -> collective_summary(load_bundle(
            db3_toy_bundle(guarded_root; collective_protocol = nothing)));
        contains = "does not declare collective_protocol")
end

check_throws("tutorial-2rank-pinned-guarded-6/8",
    () -> collective_summary(load_bundle(merge(resolve_bundle("tutorial-2rank"),
        (collective_protocol = "guarded-6/8",))));
    contains = "collectives, metrics record")

if isempty(FAILURES)
    println("PHASE_K_PROTOCOL_SUMMARY PASS")
    exit(0)
else
    println("PHASE_K_PROTOCOL_SUMMARY FAIL")
    for name in FAILURES
        println("  failed: ", name)
    end
    exit(1)
end
