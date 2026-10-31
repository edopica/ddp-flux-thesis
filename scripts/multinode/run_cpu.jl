# One parent; MPI launches only direct workers. Run after explicit precompile.
using MPI
const project = joinpath(@__DIR__,"env")
const source = realpath(joinpath(@__DIR__,"../../../flux-multinode"))
const launcher = MPI.mpiexec()
for w in (2,4)
    for file in sort(filter(f -> endswith(f,"_distributedtest.jl"),readdir(joinpath(source,"test/ext_distributed");join=true)))
        cmd = `timeout -k 15s 900s $(launcher.exec) -n $w $(Base.julia_cmd()) --startup-file=no --compiled-modules=existing --project=$project $file`
        run(addenv(setenv(cmd,launcher.env),"FLUX_TEST_DISTRIBUTED_BACKEND"=>"mpi"))
    end
    run(setenv(`timeout -k 15s 300s $(launcher.exec) -n $w $(Base.julia_cmd()) --startup-file=no --compiled-modules=existing --project=$project $(joinpath(@__DIR__,"acceptance.jl")) --backend=mpi --world=$w --nodes=1 --ranks-per-node=$w`,launcher.env))
end
