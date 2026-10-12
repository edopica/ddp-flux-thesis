# Bounded two-job CPU/MPI characterization. Run from the thesis repository.
using MPI
const project = dirname(Base.active_project())
const probe = joinpath(@__DIR__, "conditional_wrapper_compatibility_probe.jl")
const output = abspath("artifacts/logs/conditional-wrapper")
isdir(output) || error("Evidence directory does not exist: $output")
directory = mktempdir("/tmp/opencode"; prefix="phaseH-")
try
    for mode in ("write", "resume")
        cmd = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --color=no --project=$project $probe $mode $directory`
        log = joinpath(output, "phaseH-$mode-mpi2-2026-10-01.log")
        println("COMMAND ", join(cmd.exec, " ")) # Do not print the inherited environment.
        started = time()
        open(log, "w") do io
            println(io, "COMMAND ", join(cmd.exec, " "))
            flush(io)
            proc = run(pipeline(ignorestatus(cmd); stdout=io, stderr=io); wait=false)
            expired = timedwait(() -> !process_running(proc), 180.0; pollint=0.2) == :timed_out
            if expired
                kill(proc)
                timedwait(() -> !process_running(proc), 5.0) == :timed_out && kill(proc, Base.SIGKILL)
            end
            wait(proc)
            elapsed = round(time()-started; digits=3)
            println(io, "RESULT exit=$(proc.exitcode) timeout=$expired elapsed=$elapsed")
            println("RESULT mode=$mode exit=$(proc.exitcode) timeout=$expired elapsed=$elapsed log=$log")
            (expired || proc.exitcode != 0) && error("Phase H $mode failed. See $log")
        end
    end
finally
    rm(directory; recursive=true)
end
