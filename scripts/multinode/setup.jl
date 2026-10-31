using Pkg
Pkg.develop(path=realpath(joinpath(@__DIR__, "../../../flux-multinode")))
Pkg.instantiate()
Pkg.precompile()
using Flux, MPI, Optimisers, Functors, JSON, MLUtils
println("BASE_CPU_PRECOMPILE PASS flux=",pathof(Flux))
