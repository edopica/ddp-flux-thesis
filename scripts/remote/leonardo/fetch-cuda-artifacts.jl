# Pre-fetch the platform-specific CUDA artifacts on a login node.
#
# Leonardo compute nodes have no outbound HTTPS. CUDA_Runtime_jll and its
# siblings select their artifact at load time from the visible driver (cuda
# tag), so a driverless login-node resolve/instantiate does not download them.
# Run this script on the login node with JULIA_DEPOT_PATH set to the shared
# depot before the first GPU precompile:
#
#   JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot \
#     /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia \
#     --startup-file=no scripts/remote/leonardo/fetch-cuda-artifacts.jl
#
# The cuda tags below match what CUDA_Runtime_jll selects on Leonardo:
# driver 535 (CUDA 12.x minor-version compatibility) -> newest 12.x toolkit,
# CUDA_Runtime and NCCL 12.9, CUDNN 12.0, CUDA_Compiler 12. Update the tags if
# the resolved CUDA.jl versions change.
using Pkg.Artifacts, Base.BinaryPlatforms

depot = ENV["JULIA_DEPOT_PATH"]
items = [
    ("CUDA_Compiler", "CUDA_Compiler_jll", "12"),
    ("CUDNN",         "CUDNN_jll",         "12.0"),
    ("NCCL",          "NCCL_jll",          "12.9"),
    ("CUDA_Runtime",  "CUDA_Runtime_jll",  "12.9"),
]

for (name, pkg, cuda_tag) in items
    pkgdir = joinpath(depot, "packages", pkg)
    sub = first(filter(isdir, readdir(pkgdir; join=true)))
    toml = joinpath(sub, "Artifacts.toml")
    platform = Platform("x86_64", "linux"; libc="glibc", cuda=cuda_tag, cuda_local="false")
    arts = select_downloadable_artifacts(toml; platform=platform, include_lazy=true)
    haskey(arts, name) || error("no artifact '$name' for cuda=$cuda_tag in $toml")
    path = ensure_artifact_installed(name, arts[name], toml; platform=platform, verbose=true)
    println("OK ", name, " cuda=", cuda_tag, " -> ", path)
end
print("ALL ARTIFACTS INSTALLED")
