# Pure-stdlib regression for CGCheckpoint.checkpoint_digest.
#
# The digest serializes the checkpoint payload. A `Dict` deserialized from a
# payload can iterate in a different order than the `Dict` that was saved, even
# though the two compare equal, so the digest must canonicalize dictionaries.
# This script needs no Flux, MPI, or device; it fails with a nonzero exit code
# if the digest is not stable across Serialization roundtrips, is sensitive to
# a changed value, or depends on dict insertion order.
include(joinpath(@__DIR__, "..", "examples", "conditional_graph", "checkpoint.jl"))

using .CGCheckpoint
using Serialization

function roundtrip(payload)
    io = IOBuffer()
    serialize(io, payload)
    return deserialize(IOBuffer(take!(io)))
end

# Eleven source-hash entries: the size of a real checkpoint's `source_hashes`
# crosses the Dict rehash boundary where a roundtrip changes iteration order.
function build_payload()
    source_hashes = Dict{String,String}()
    files = ("common.jl", "model.jl", "dataset.jl", "schedule.jl", "reference.jl",
        "checkpoint.jl", "metrics.jl", "conditional_graph_demo.jl",
        "conditional_graph_demo_check.jl")
    for f in files
        source_hashes["scripts/examples/conditional_graph/$f"] = "hash-$f"
    end
    source_hashes["flux-integration/src/Flux.jl"] = "hash-flux"
    source_hashes["flux-integration/src/distributed/public_api.jl"] = "hash-public-api"
    return (; schema_version = CGCheckpoint.SCHEMA_VERSION,
        kind = :phase_k_checkpoint,
        step_saved = 2, step_next = 3, world_size = 4, backend_name = "mpi",
        dtype = "Float32", model_definition_version = 1,
        optimizer = (; name = "Adam", eta = 1.0e-3, beta = (0.9, 0.999),
            epsilon = 1.0e-8),
        schedule_digest = "schedule", dataset_digest = "dataset",
        model = [1.5 2.5; 3.5 4.5],
        tree = (; trunk = (; weight = (; rule = (; eta = 1.0e-3),
                state = ([0.1, 0.2], [0.3, 0.4]), frozen = false),
            bias = (; rule = (; eta = 1.0e-3),
                state = ([0.0], [0.0]), frozen = false))),
        ref_model = [0.0, 1.0],
        ref_tree = (; head = (; bias = (; rule = (; eta = 1.0e-3),
                state = ([0.0], [0.0]), frozen = true))),
        cursors = (; a = 128, b = 256),
        package_versions = [(name = "Flux", version = "0.16.11", path = "/tmp/Flux.jl")],
        source_hashes,
        julia_version = "1.12.6",
        tolerances = (; atol = 2.0e-6, rtol = 1.0e-5),
        run_id = "roundtrip")
end

function main()
    payload = build_payload()
    expected = CGCheckpoint.checkpoint_digest(payload)
    println("CHECKPOINT_DIGEST ", expected)

    current = payload
    for i in 1:3
        current = roundtrip(current)
        current == payload ||
            error("payload values changed across roundtrip $i")
        got = CGCheckpoint.checkpoint_digest(current)
        got == expected ||
            error("digest changed after roundtrip $i: $got != $expected")
    end
    println("CHECKPOINT_ROUNDTRIP_STABLE roundtrips=3")

    # Dict insertion order must not matter.
    reversed_hashes = Dict{String,String}(reverse(collect(payload.source_hashes)))
    permuted = merge(payload, (; source_hashes = reversed_hashes))
    CGCheckpoint.checkpoint_digest(permuted) == expected ||
        error("dict insertion order changed the digest")
    println("CHECKPOINT_DICT_ORDER_CANONICAL entries=$(length(payload.source_hashes))")

    # A nested dict must be canonical too.
    nested_a = Dict("y" => 2, "x" => 1)
    nested_b = Dict("x" => 1, "y" => 2)
    CGCheckpoint.checkpoint_digest(merge(payload, (; cursors = nested_a))) ==
        CGCheckpoint.checkpoint_digest(merge(payload, (; cursors = nested_b))) ||
        error("nested dict insertion order changed the digest")
    println("CHECKPOINT_NESTED_DICT_CANONICAL")

    # A changed value must change the digest.
    changed_hashes = copy(payload.source_hashes)
    changed_hashes["scripts/examples/conditional_graph/common.jl"] = "mutated"
    CGCheckpoint.checkpoint_digest(merge(payload, (; source_hashes = changed_hashes))) !=
        expected || error("mutated source_hashes did not change the digest")
    CGCheckpoint.checkpoint_digest(merge(payload, (; step_next = 4))) != expected ||
        error("mutated step_next did not change the digest")
    CGCheckpoint.checkpoint_digest(merge(payload, (; tolerances = (; atol = 1.0e-3, rtol = 1.0e-5)))) !=
        expected || error("mutated tolerances did not change the digest")
    println("CHECKPOINT_MUTATION_SENSITIVE")

    println("CHECKPOINT_ROUNDTRIP_OK")
    return nothing
end

ok = try
    main()
    true
catch err
    println(stderr, "CHECKPOINT_ROUNDTRIP_FAIL")
    showerror(stderr, err)
    println(stderr)
    false
end
exit(ok ? 0 : 1)
