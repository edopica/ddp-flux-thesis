"""
Phase 2 convenience runner: runs check_fixtures_data.jl, then
check_fixtures_operators.jl when Flux is on the load path (otherwise only the
data check). Both checks receive the same CLI arguments.

Reproduction (local; NPZ-only, data check only):
  cd /home/kurapica/Projects/ddp_flux/ddp-flux-thesis
  julia --startup-file=no --project=scripts/static_scaling/julia-env \\
    scripts/static_scaling/check_all.jl --fixture-dir <dir> --bundle <npz> --bundle-manifest <json>

Reproduction (local; NPZ + Flux stack, runs both checks):
  JULIA_LOAD_PATH="\$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" \\
    julia --startup-file=no scripts/static_scaling/check_all.jl --fixture-dir <dir> --bundle <npz> --bundle-manifest <json>

Reproduction (Leonardo; the Flux project is the remote worktree test project):
  JULIA_LOAD_PATH="<THESIS>/scripts/static_scaling/julia-env:<REMOTE_FLUX>/test:@stdlib" \\
    julia --startup-file=no <THESIS>/scripts/static_scaling/check_all.jl --fixture-dir <dir> --bundle <npz> --bundle-manifest <json>
"""
;
function run_child(script::AbstractString, argv)
    base = Base.julia_cmd()
    ap = Base.active_project()
    script_path = joinpath(@__DIR__, script)
    args = String[String(a) for a in argv]
    cmd = (ap !== nothing && isfile(ap)) ?
        `$base --startup-file=no --project=$ap $script_path $args` :
        `$base --startup-file=no $script_path $args`
    p = run(pipeline(ignorestatus(cmd); stdout = stdout, stderr = stderr))
    return p.exitcode
end

# Flux counts as available only when it resolves from an explicit
# JULIA_LOAD_PATH entry (the documented stacked command), not from a default
# environment such as @v1.12 which would silently use a different Flux build.
function flux_on_explicit_load_path()
    flux = Base.find_package("Flux")
    flux === nothing && return false
    lp = get(ENV, "JULIA_LOAD_PATH", "")
    isempty(lp) && return false
    for e in split(lp, ':')
        isempty(e) && continue
        if e == "@"
            proj = Base.active_project()
            proj !== nothing && startswith(flux, dirname(proj)) && return true
        elseif !startswith(e, "@")
            startswith(flux, abspath(e)) && return true
        end
    end
    return false
end

status = run_child("check_fixtures_data.jl", ARGS)
if flux_on_explicit_load_path()
    status |= run_child("check_fixtures_operators.jl", ARGS)
else
    println("CHECK operators SKIP Flux not on an explicit JULIA_LOAD_PATH entry")
end
exit(status)
