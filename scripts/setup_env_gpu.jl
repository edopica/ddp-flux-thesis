using Pkg

println("Adding optional GPU/profiling packages.")
println("This is intended for the GPU/NCCL phase, not the initial CPU-only audit.")

Pkg.add(["CUDA", "NCCL", "NVTX", "Preferences"])
Pkg.instantiate()
Pkg.precompile()

println("GPU/profiling packages installed.")
println("Run: julia --project=. -e 'using CUDA; CUDA.versioninfo(); @show CUDA.functional()'")
