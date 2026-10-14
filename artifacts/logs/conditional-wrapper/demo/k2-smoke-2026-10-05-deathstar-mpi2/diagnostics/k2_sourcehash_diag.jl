using Serialization, SHA, Flux, Optimisers, Functors, MPI, Zygote
const THESIS = "/home/picazio/projects/ddp-flux-thesis"
const SRC = joinpath(THESIS, "scripts/examples/conditional_graph")
include(joinpath(SRC, "common.jl")); include(joinpath(SRC, "model.jl")); include(joinpath(SRC, "dataset.jl"))
include(joinpath(SRC, "schedule.jl")); include(joinpath(SRC, "reference.jl")); include(joinpath(SRC, "checkpoint.jl"))
using .CGCommon, .CGModel, .CGDataset, .CGSchedule, .CGReference, .CGCheckpoint
const ROOT = joinpath(THESIS, "artifacts/logs/conditional-wrapper/demo/k2-smoke-2026-10-05-deathstar-mpi2")
payload = open(deserialize, joinpath(ROOT, "mpi2", "train", "checkpoint-step2.jls"))
println("LOADED_ORDER=", collect(keys(payload.source_hashes)))

# Rebuild exactly as conditional_graph_demo.jl:demo_source_hashes() does.
files = Dict{String,String}()
for f in ("common.jl", "model.jl", "dataset.jl", "schedule.jl", "reference.jl", "checkpoint.jl", "metrics.jl")
    files["scripts/examples/conditional_graph/$f"] = CGCommon.source_hash(joinpath(SRC, f))
end
files["scripts/examples/conditional_graph_demo.jl"] = CGCommon.source_hash(joinpath(THESIS, "scripts/examples/conditional_graph_demo.jl"))
checker = joinpath(THESIS, "scripts/checks/conditional_graph_demo_check.jl")
isfile(checker) && (files["scripts/checks/conditional_graph_demo_check.jl"] = CGCommon.source_hash(checker))
flux_root = normpath(joinpath(dirname(pathof(Flux)), ".."))
for rel in ("src/Flux.jl", "src/distributed/public_api.jl")
    p = joinpath(flux_root, rel)
    isfile(p) && (files["flux-integration/$rel"] = CGCommon.source_hash(p))
end
println("REBUILT_ORDER=", collect(keys(files)))
println("DICT_EQUAL=", files == payload.source_hashes)
p2 = merge(payload, (; source_hashes=files))
println("DIGEST_LOADED=", CGCheckpoint.checkpoint_digest(payload))
println("DIGEST_REBUILT_SOURCE_HASHES=", CGCheckpoint.checkpoint_digest(p2))
p3 = merge(payload, (; source_hashes=files, package_versions=CGCommon.package_versions()))
println("DIGEST_REBUILT_BOTH=", CGCheckpoint.checkpoint_digest(p3))
# Also compare the serialized bytes of just the Dict.
b(d) = (io=IOBuffer(); serialize(io, d); copy(io.data[1:position(io)]))
println("DICT_BYTES_EQUAL=", b(files) == b(payload.source_hashes))
println("SRC_DIAG_OK")
