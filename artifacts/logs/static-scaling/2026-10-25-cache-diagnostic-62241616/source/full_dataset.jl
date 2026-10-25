"""
Minimal full-dataset loader for the static-scaling experiment.

The prepared `.npy` files are C-order and the loader maps them read-only with
`Mmap`, reversing the dimensions so that Julia's column-major memory layout
interprets the raw bytes exactly like numpy does.

Included after `replay.jl`; this file has no Flux dependency.
"""

using Mmap
using SHA

const FULL_DATASET_SCHEMA = "static-full-dataset-loader-v1"
const FULL_DATASET_TRANSFORM_SPEC =
    "pad=4|crop=64|top_left_zero_based|flip_when=1|uint8_to_float32|div255|sub_mean|div_std|layout=WHCN"

struct FullDataset
    train_images::Array{UInt8,4}
    train_labels::Vector{Int64}
    permutations::Matrix{Int64}
    augmentation::Array{UInt8,3}
    mean32::Vector{Float32}
    std32::Vector{Float32}
    dataset_dir::String
    metadata::Dict{String,Any}
    buffers::Vector{Vector{UInt8}}
end

function _npy_eltype(descr::AbstractString)
    isempty(descr) && error("npy: empty dtype descriptor")
    endian = descr[1]
    code = String(descr[2:end])
    endian in ('<', '|', '=') ||
        error("npy: unsupported byte order $(repr(endian)); little-endian only")
    code == "u1" && return UInt8
    code == "i8" && return Int64
    code == "i4" && return Int32
    code == "u4" && return UInt32
    code == "f4" && return Float32
    code == "f8" && return Float64
    error("npy: unsupported dtype $(repr(String(descr)))")
end

function _npy_shape(header::AbstractString)
    m = match(r"'shape'\s*:\s*\(([^)]*)\)", header)
    m === nothing && error("npy: header has no shape")
    parts = filter(!isempty, strip.(split(m.captures[1], ',')))
    return Tuple(parse.(Int, parts))
end

function _map_npy(path::AbstractString)
    isfile(path) || error("npy: file not found: $(path)")
    filesize(path) > 0 || error("npy: empty file: $(path)")
    io = open(path, "r")
    bytes = Mmap.mmap(io, Vector{UInt8})
    close(io)
    length(bytes) >= 10 || error("npy: truncated header: $(path)")
    bytes[1:6] == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] ||
        error("npy: bad magic in $(path)")
    major = Int(bytes[7])
    major in (1, 2) || error("npy: unsupported format version $(major) in $(path)")
    if major == 1
        hlen = Int(bytes[9]) | (Int(bytes[10]) << 8)
        header_start = 11
        offset = 10 + hlen
    else
        hlen = Int(bytes[9]) | (Int(bytes[10]) << 8) |
               (Int(bytes[11]) << 16) | (Int(bytes[12]) << 24)
        header_start = 13
        offset = 12 + hlen
    end
    offset <= length(bytes) || error("npy: header length exceeds file: $(path)")
    header = String(bytes[header_start:offset])
    dm = match(r"'descr'\s*:\s*'([^']+)'", header)
    dm === nothing && error("npy: header has no dtype descriptor: $(path)")
    descr = dm.captures[1]
    occursin(r"'fortran_order'\s*:\s*False", header) ||
        error("npy: only C-order arrays are supported: $(path)")
    shape = _npy_shape(header)
    T = _npy_eltype(descr)
    expected = offset + prod(shape; init = 1) * sizeof(T)
    expected == length(bytes) ||
        error("npy: size mismatch in $(path): header says $(expected) bytes, file has $(length(bytes))")
    offset % sizeof(T) == 0 ||
        error("npy: data offset $(offset) is not aligned for $(T) in $(path)")
    dims = Tuple(reverse(shape))
    arr = unsafe_wrap(Array, Ptr{T}(pointer(bytes) + offset), dims; own = false)
    info = Dict{String,Any}(
        "path" => String(path),
        "descr" => String(descr),
        "dtype" => string(T),
        "shape" => collect(shape),
        "fortran_order" => false,
        "header_bytes" => offset,
        "file_bytes" => length(bytes),
    )
    return arr, bytes, info
end

"""
    open_npy_mmap(path) -> Array

Map a C-order `.npy` file read-only and return the array with reversed
dimensions. The mapping stays alive until the returned array is collected.
"""
function open_npy_mmap(path::AbstractString)
    arr, bytes, _ = _map_npy(path)
    finalizer(arr) do _
        sizeof(bytes)
        nothing
    end
    return arr
end

function _json_float32_triplet(v, name::AbstractString)
    v isa Vector || error("stats.json: $(name) must be a 3-element array")
    length(v) == 3 || error("stats.json: $(name) must have 3 entries")
    out = Float32[Float32(x) for x in v]
    all(isfinite, out) || error("stats.json: $(name) contains a non-finite value")
    return out
end

function _manifest_file_records(manifest, source::AbstractString)
    records = Dict{String,Any}()
    files = json_find(manifest, "files")
    files isa Vector || return records
    for entry in files
        entry isa JSONObject || continue
        p = get(entry, "path", nothing)
        p isa AbstractString || continue
        name = String(p)
        rec = get!(records, name, Dict{String,Any}())
        rec["source"] = String(source)
        if haskey(entry, "bytes")
            rec["declared_bytes"] = Int(entry["bytes"])
        end
        if haskey(entry, "sha256")
            rec["declared_sha256"] = String(entry["sha256"])
        end
        if haskey(entry, "dtype")
            rec["declared_dtype"] = String(entry["dtype"])
        end
        if haskey(entry, "shape")
            rec["declared_shape"] = Int[Int(x) for x in entry["shape"]]
        end
    end
    return records
end

function _check_declared_array(name, arr, decl)
    if haskey(decl, "declared_shape")
        want = Tuple(reverse(decl["declared_shape"]))
        size(arr) == want ||
            error("$(name): mapped size $(size(arr)) does not match declared reversed shape $(want)")
    end
    if haskey(decl, "declared_dtype")
        want = decl["declared_dtype"]
        got = string(eltype(arr))
        aliases = Dict("uint8" => "UInt8", "int64" => "Int64",
                       "float32" => "Float32", "float64" => "Float64")
        want_julia = get(aliases, want, want)
        got == want_julia ||
            error("$(name): mapped dtype $(got) does not match declared $(want)")
    end
    return nothing
end

function _sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function _file_record(dataset_dir, name, decl, verify_large_hashes)
    path = joinpath(dataset_dir, name)
    isfile(path) || error("dataset file not found: $(path)")
    bytes = filesize(path)
    large = name in ("train_images.npy", "train_labels.npy", "val_images.npy")
    always_hash = !large
    computed = (always_hash || verify_large_hashes) ? _sha256_file(path) : ""
    declared = get(decl, "declared_sha256", "")
    if !isempty(computed) && !isempty(declared) && computed != declared
        error("dataset file hash mismatch for $(name): actual=$(computed) declared=$(declared)")
    end
    if haskey(decl, "declared_bytes") && bytes != decl["declared_bytes"]
        error("dataset file size mismatch for $(name): actual=$(bytes) declared=$(decl["declared_bytes"])")
    end
    return Dict{String,Any}(
        "bytes" => bytes,
        "sha256" => !isempty(computed) ? computed : String(declared),
        "actual_sha256" => computed,
        "declared_sha256" => String(declared),
        "declared_bytes" => get(decl, "declared_bytes", nothing),
        "source" => get(decl, "source", "actual-only"),
        "hash_verified" => !isempty(computed),
        "large" => large,
    )
end

"""
    open_full_dataset(dataset_dir; verify_large_hashes=false) -> FullDataset

Open the prepared dataset with read-only memory maps and verify the manifest
metadata. Small files are always SHA-256 checked. Large image arrays are
stream-hashed only when `verify_large_hashes` is true; their sizes are always
checked.
"""
function open_full_dataset(dataset_dir::AbstractString; verify_large_hashes::Bool = false)
    dir = abspath(String(dataset_dir))
    isdir(dir) || error("dataset directory not found: $(dir)")
    required = ["train_images.npy", "train_labels.npy", "permutations.npy",
                "augmentation.npy", "stats.json",
                "dataset_manifest.json", "schedule_manifest.json"]
    for name in required
        isfile(joinpath(dir, name)) || error("required dataset file not found: $(joinpath(dir, name))")
    end

    stats_doc = json_parse_file(joinpath(dir, "stats.json"))
    mean_any = json_find(stats_doc, "mean")
    std_any = json_find(stats_doc, "std")
    (mean_any === JSON_MISSING || std_any === JSON_MISSING) &&
        error("stats.json: missing mean or std")
    mean32 = _json_float32_triplet(mean_any, "mean")
    std32 = _json_float32_triplet(std_any, "std")

    dataset_manifest = json_parse_file(joinpath(dir, "dataset_manifest.json"))
    schedule_manifest = json_parse_file(joinpath(dir, "schedule_manifest.json"))
    declared = Dict{String,Any}()
    merge!(declared, _manifest_file_records(dataset_manifest, "dataset_manifest.json"))
    merge!(declared, _manifest_file_records(schedule_manifest, "schedule_manifest.json"))

    images, images_bytes, images_info = _map_npy(joinpath(dir, "train_images.npy"))
    labels, labels_bytes, labels_info = _map_npy(joinpath(dir, "train_labels.npy"))
    permutations, permutations_bytes, permutations_info = _map_npy(joinpath(dir, "permutations.npy"))
    augmentation, augmentation_bytes, augmentation_info = _map_npy(joinpath(dir, "augmentation.npy"))
    images isa Array{UInt8,4} || error("train_images.npy: expected a 4-D UInt8 array")
    labels isa Vector{Int64} || error("train_labels.npy: expected a Vector{Int64}")
    permutations isa Matrix{Int64} || error("permutations.npy: expected a Matrix{Int64}")
    augmentation isa Array{UInt8,3} || error("augmentation.npy: expected a 3-D UInt8 array")

    for (name, arr) in (("train_images.npy", images),
                        ("train_labels.npy", labels),
                        ("permutations.npy", permutations),
                        ("augmentation.npy", augmentation))
        haskey(declared, name) && _check_declared_array(name, arr, declared[name])
    end

    file_names = copy(required)
    for optional in ("initial_bundle.npz", "initial_bundle.manifest.json")
        isfile(joinpath(dir, optional)) && push!(file_names, optional)
    end
    for name in sort!(collect(keys(declared)))
        name in file_names && continue
        isfile(joinpath(dir, name)) && push!(file_names, name)
    end
    files = Dict{String,Any}()
    for name in file_names
        file_names_known = haskey(declared, name)
        decl = file_names_known ? declared[name] : Dict{String,Any}()
        files[name] = _file_record(dir, name, decl, verify_large_hashes)
    end

    bundle_hash = ""
    bundle_manifest_path = joinpath(dir, "initial_bundle.manifest.json")
    if isfile(bundle_manifest_path)
        bundle_manifest = json_parse_file(bundle_manifest_path)
        recorded = json_find(bundle_manifest, "bundle_hash")
        recorded isa AbstractString && (bundle_hash = String(recorded))
    end

    metadata = Dict{String,Any}(
        "schema" => FULL_DATASET_SCHEMA,
        "dataset_dir" => dir,
        "files" => files,
        "declared_bundle_hash" => bundle_hash,
        "arrays" => Dict(
            "train_images" => images_info,
            "train_labels" => labels_info,
            "permutations" => permutations_info,
            "augmentation" => augmentation_info,
        ),
    )

    return FullDataset(images, labels, permutations, augmentation, mean32, std32,
                       dir, metadata,
                       Vector{UInt8}[images_bytes, labels_bytes,
                                     permutations_bytes, augmentation_bytes])
end

"""
    epoch_ids(ds, epoch, batch) -> Vector{Int64}

Effective sample ids for one epoch, after dropping the incomplete tail.
"""
function epoch_ids(ds::FullDataset, epoch::Integer, batch::Integer)
    batch > 0 || error("epoch_ids: batch must be positive")
    n = size(ds.permutations, 1)
    n_epochs = size(ds.permutations, 2)
    if n == 100000
        n_epochs == 50 || error("epoch_ids: expected 50 epoch rows, got $(n_epochs)")
        0 <= epoch <= 49 || error("epoch_ids: epoch $(epoch) outside 0:49")
        batch in (512, 256) || error("epoch_ids: batch must be 512 or 256, got $(batch)")
    else
        0 <= epoch < n_epochs ||
            error("epoch_ids: epoch $(epoch) outside 0:$(n_epochs - 1)")
    end
    row = ds.permutations[:, epoch + 1]
    ids = epoch_sample_ids(row, batch)
    if n == 100000
        length(ids) == 99840 ||
            error("epoch_ids: expected 99840 ids after tail drop, got $(length(ids))")
    else
        length(ids) % batch == 0 ||
            error("epoch_ids: effective length $(length(ids)) is not a multiple of $(batch)")
    end
    return Vector{Int64}(ids)
end

function updates_per_epoch(ds::FullDataset, batch::Integer)
    return length(epoch_ids(ds, 0, batch)) ÷ batch
end

"""
    global_batch_ids(ds, epoch, batch, update) -> Vector{Int64}

Ids of one global batch, formed from the epoch permutation before sharding.
"""
function global_batch_ids(ds::FullDataset, epoch::Integer, batch::Integer, update::Integer)
    ids = epoch_ids(ds, epoch, batch)
    n_updates = length(ids) ÷ batch
    1 <= update <= n_updates ||
        error("global_batch_ids: update $(update) outside 1:$(n_updates)")
    first = (update - 1) * batch + 1
    last = update * batch
    return ids[first:last]
end

"""
    local_batch(ds, epoch, batch, update, rank, world)

Assemble one rank's local training batch. `rank` is 1-based. Returns
`(x_whcn, labels, shard_ids, global_ids)` where H2D has not been applied.
"""
function local_batch(ds::FullDataset, epoch::Integer, batch::Integer, update::Integer,
                     rank::Integer, world::Integer)
    gids = global_batch_ids(ds, epoch, batch, update)
    sids = Vector{Int64}(rank_shard(gids, rank, world))
    idx = sids .+ 1
    raw = ds.train_images[:, :, :, idx]
    nchw = permutedims(raw, (4, 3, 2, 1))
    chunk = length(sids)
    size(nchw) == (chunk, 3, 64, 64) ||
        error("local_batch: gathered NCHW shape $(size(nchw)) is not ($(chunk), 3, 64, 64)")
    eltype(nchw) === UInt8 || error("local_batch: gathered images are not UInt8")
    triples = permutedims(ds.augmentation[:, idx, epoch + 1], (2, 1))
    eltype(triples) === UInt8 || error("local_batch: augmentation triples are not UInt8")
    size(triples) == (chunk, 3) ||
        error("local_batch: augmentation shape $(size(triples)) is not ($(chunk), 3)")
    x = replay_transform(nchw_to_whcn(nchw), triples, ds.mean32, ds.std32)
    labels = Int64.(ds.train_labels[idx])
    size(x) == (64, 64, 3, chunk) ||
        error("local_batch: transformed shape $(size(x)) is not (64, 64, 3, $(chunk))")
    length(labels) == chunk || error("local_batch: label count mismatch")
    return x, labels, sids, gids
end

"""
    transform_spec_sha256(ds) -> String

SHA-256 of the frozen transform description followed by the raw mean and std
float32 bytes.
"""
function transform_spec_sha256(ds::FullDataset)
    ctx = SHA.SHA2_256_CTX()
    SHA.update!(ctx, codeunits(FULL_DATASET_TRANSFORM_SPEC))
    SHA.update!(ctx, reinterpret(UInt8, ds.mean32))
    SHA.update!(ctx, reinterpret(UInt8, ds.std32))
    return bytes2hex(SHA.digest!(ctx))
end

sample_order_sha256(ds::FullDataset, epoch::Integer, batch::Integer) =
    c_order_sha256(epoch_ids(ds, epoch, batch))

update_order_sha256(ds::FullDataset, epoch::Integer, batch::Integer, update::Integer) =
    c_order_sha256(global_batch_ids(ds, epoch, batch, update))

"""
    dataset_metadata(ds) -> Dict

File hashes, normalization constants, order hashes, and frozen schedule sizes.
"""
function dataset_metadata(ds::FullDataset)
    n = size(ds.permutations, 1)
    n_eff_512 = length(epoch_ids(ds, 0, 512))
    n_eff_256 = length(epoch_ids(ds, 0, 256))
    return Dict{String,Any}(
        "schema" => FULL_DATASET_SCHEMA,
        "dataset_dir" => ds.dataset_dir,
        "dataset_files" => ds.metadata["files"],
        "stats_mean" => ds.mean32,
        "stats_std" => ds.std32,
        "n_samples" => n,
        "n_epochs" => size(ds.permutations, 2),
        "n_eff" => n_eff_512,
        "tail_size" => n - n_eff_512,
        "updates_per_epoch" => Dict("512" => n_eff_512 ÷ 512,
                                    "256" => n_eff_256 ÷ 256),
        "sample_order_sha256" => sample_order_sha256(ds, 0, 512),
        "sample_order_sha256_b256" => sample_order_sha256(ds, 0, 256),
        "transform_spec_sha256" => transform_spec_sha256(ds),
        "declared_bundle_hash" => ds.metadata["declared_bundle_hash"],
    )
end
