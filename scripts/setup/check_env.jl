println("=== DDP thesis environment check ===")
println("Julia version: ", VERSION)
println("Active project: ", Base.active_project())

packages = [
    "Flux",
    "MPI",
    "Optimisers",
    "Zygote",
    "Functors",
    "MLUtils",
    "Adapt",
    "BenchmarkTools",
    "Revise",
]

for name in packages
    sym = Symbol(name)
    try
        @eval import $(sym)
        mod = getfield(Main, sym)
        println("OK: ", name, " loaded from ", pathof(mod))
    catch err
        println("FAIL: ", name, " -> ", sprint(showerror, err))
    end
end

try
    import CUDA
    println("CUDA package available.")
    println("CUDA.functional() = ", CUDA.functional())
catch err
    println("CUDA not available or not installed yet: ", sprint(showerror, err))
end

println("=== Done ===")
