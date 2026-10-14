# Read-only diagnostic for the K2 smoke resume digest mismatch.
# Recreates the demo modules in Main (same include order as the demo) so the
# serialized payload resolves. No MPI backend is initialized.
using Serialization, SHA, Dates, Flux, Optimisers, Functors, MPI, Zygote

const SRC = "/home/picazio/projects/ddp-flux-thesis/scripts/examples/conditional_graph"
include(joinpath(SRC, "common.jl"))
include(joinpath(SRC, "model.jl"))
include(joinpath(SRC, "dataset.jl"))
include(joinpath(SRC, "schedule.jl"))
include(joinpath(SRC, "reference.jl"))
include(joinpath(SRC, "checkpoint.jl"))
using .CGCommon, .CGModel, .CGDataset, .CGSchedule, .CGReference, .CGCheckpoint

const ROOT = "/home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k2-smoke-2026-10-05-deathstar-mpi2"
const CKPT = joinpath(ROOT, "mpi2", "train", "checkpoint-step2.jls")
const TRAJ = joinpath(ROOT, "mpi2", "train", "trajectory.jls")

payload = open(deserialize, CKPT)
expected = open(deserialize, TRAJ)
println("STORED_DIGEST=", expected.checkpoint_digest)
println("FRESH_DIGEST=", CGCheckpoint.checkpoint_digest(payload))
buf = IOBuffer()
serialize(buf, payload)
payload2 = deserialize(seekstart(buf))
println("ROUNDTRIP_DIGEST=", CGCheckpoint.checkpoint_digest(payload2))
println("CKPT_SIZE=", filesize(CKPT))

const FIELDS = (:schema_version, :step_next, :world_size, :backend_name, :dtype,
    :model_definition_version, :optimizer, :schedule_digest, :dataset_digest,
    :model, :tree, :ref_model, :ref_tree, :cursors, :package_versions, :source_hashes,
    :julia_version, :tolerances, :run_id)

function digest_columns(p)
    io = IOBuffer()
    cols = Pair{String,String}[]
    for field in FIELDS
        start = position(io)
        serialize(io, getproperty(p, field))
        stop = position(io)
        push!(cols, string(field) => bytes2hex(SHA.sha256(view(io.data, start+1:stop))))
    end
    return cols, bytes2hex(SHA.sha256(seekstart(io)))
end

cols, full = digest_columns(payload)
cols2, full2 = digest_columns(payload2)
for (a, b) in zip(cols, cols2)
    flag = a.second == b.second ? "same" : "DIFF"
    println("COL ", a.first, " ", a.second[1:16], " ", flag)
end
println("COL_TOTAL=", full[1:16], " ROUNDTRIP_TOTAL=", full2[1:16])
println("SAME_AS_STORED=", full == expected.checkpoint_digest)

for field in (:model, :tree, :ref_model, :ref_tree)
    x = getproperty(payload, field)
    println("TYPE ", field, " => ", typeof(x))
end
println("MODEL_W_TYPE=", typeof(payload.model.trunk.weight), " ELTYPE=", eltype(payload.model.trunk.weight))
println("SNAPSHOT_N=", length(expected.snapshots), " SNAP2_LEN=", length(expected.snapshots[payload.step_saved]))
println("SNAP2_TYPE=", typeof(expected.snapshots[payload.step_saved]))
println("CURSORS=", payload.cursors)
println("OPTIMIZER=", payload.optimizer)

# Compare the checkpointed model/tree values with the saved uninterrupted
# step-2 snapshots (the check that would run after the digest gate).
loaded = CGCommon.snapshot(payload.model, payload.tree)
ref_loaded = CGCommon.snapshot(payload.ref_model, payload.ref_tree)
println("DDP_SNAP_MATCH=", loaded.values == expected.snapshots[payload.step_saved])
println("REF_SNAP_MATCH=", ref_loaded.values == expected.ref_snapshots[payload.step_saved])
if loaded.values != expected.snapshots[payload.step_saved]
    d = findall(!, loaded.values .== expected.snapshots[payload.step_saved])
    println("DDP_SNAP_FIRST_DIFF_IDX=", first(d), " loaded=", loaded.values[first(d)], " saved=", expected.snapshots[payload.step_saved][first(d)])
end
println("DIAG_OK")
