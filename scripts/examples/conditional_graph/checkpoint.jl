# Data-only checkpoint payloads for the Phase K demo.
#
# A payload contains plain Julia data plus the recursive `Schedule` and dataset
# NamedTuples. It never contains an MPI/NCCL communicator, a live backend, or a
# recording log. Device arrays are copied to the host before serialization by
# the caller. `Serialization` is the documented format for the toy campaign:
# continuation runs use the same Julia version and environment, and the schema
# version plus recorded package versions guard compatibility.
module CGCheckpoint

using Serialization, SHA

export SCHEMA_VERSION, save_checkpoint, load_checkpoint, checkpoint_digest

const SCHEMA_VERSION = 2

function save_checkpoint(path::AbstractString, payload)
    payload.schema_version == SCHEMA_VERSION || error(
        "refusing to save payload with schema $(payload.schema_version); " *
        "expected $(SCHEMA_VERSION)")
    tmp = path * ".tmp-" * string(getpid())
    open(tmp, "w") do io
        serialize(io, payload)
    end
    mv(tmp, path; force=true)
    return path
end

function load_checkpoint(path::AbstractString)
    isfile(path) || error("checkpoint $path does not exist")
    payload = open(deserialize, path)
    payload isa NamedTuple || error("checkpoint $path does not hold a NamedTuple payload")
    payload.schema_version == SCHEMA_VERSION || error(
        "checkpoint schema $(payload.schema_version) does not match " *
        "supported schema $(SCHEMA_VERSION)")
    return payload
end

# Canonicalize a digest input before serialization. A `Dict` deserialized from
# a payload can iterate in a different order than the insertion-order `Dict`
# that was saved even when the two compare equal, so entries are sorted by
# `string(key)`. NamedTuple/Tuple/AbstractVector recurse so dictionaries nested
# in containers are canonical too; every other value is serialized unchanged.
canonical_digest_value(x::AbstractDict) =
    [first(kv) => canonical_digest_value(last(kv))
     for kv in sort!(collect(x); by = kv -> string(first(kv)))]
canonical_digest_value(x::NamedTuple) =
    NamedTuple{keys(x)}(map(canonical_digest_value, values(x)))
canonical_digest_value(x::Tuple) = map(canonical_digest_value, x)
canonical_digest_value(x::AbstractVector) = map(canonical_digest_value, x)
canonical_digest_value(x) = x

# Identity digest over the fields that must be fixed across a continuation.
function checkpoint_digest(payload)
    io = IOBuffer()
    for field in (:schema_version, :step_next, :world_size, :backend_name, :dtype,
        :model_definition_version, :optimizer, :schedule_digest, :dataset_digest,
        :model, :tree, :ref_model, :ref_tree, :cursors, :package_versions, :source_hashes,
        :julia_version, :tolerances, :run_id)
        serialize(io, canonical_digest_value(getproperty(payload, field)))
    end
    return bytes2hex(SHA.sha256(seekstart(io)))
end

end # module
