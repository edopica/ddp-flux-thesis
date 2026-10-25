"""
Full-dataset Julia checker for the static-scaling experiment.

Reads the frozen reference NPZ/JSON, rebuilds epoch ids, rank sequences,
update-1 transformed batches, onehot labels, and bundle hashes, then prints one
`CHECK <name> PASS|FAIL` line per check and a final summary line.
"""

include(joinpath(@__DIR__, "replay.jl"))
include(joinpath(@__DIR__, "full_dataset.jl"))
include(joinpath(@__DIR__, "model_flux.jl"))
using Flux
using SHA
using .StaticScalingFlux

const FROZEN_BUNDLE_HASH =
    "a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23"

function _parse_bool(value)
    return lowercase(strip(String(value))) in ("true", "1", "yes")
end

function _json_escape_check(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '"'
            write(io, "\\\"")
        elseif c == '\\'
            write(io, "\\\\")
        elseif c == '\n'
            write(io, "\\n")
        elseif c == '\r'
            write(io, "\\r")
        elseif c == '\t'
            write(io, "\\t")
        elseif UInt32(c) < 0x20
            print(io, "\\u", lpad(string(UInt32(c); base = 16), 4, '0'))
        else
            write(io, c)
        end
    end
    return String(take!(io))
end

function _json_write_check(io::IO, obj, indent::Int)
    pad = "  " ^ indent
    if obj === nothing || obj === missing
        write(io, "null")
    elseif obj isa Bool
        write(io, obj ? "true" : "false")
    elseif obj isa Integer
        print(io, obj)
    elseif obj isa AbstractFloat
        print(io, repr(Float64(obj)))
    elseif obj isa AbstractString || obj isa Symbol
        write(io, '"', _json_escape_check(String(obj)), '"')
    elseif obj isa AbstractDict
        keys_sorted = sort!(collect(keys(obj)); by = string)
        write(io, "{")
        if !isempty(keys_sorted)
            write(io, "\n")
            for (i, k) in enumerate(keys_sorted)
                write(io, pad, "  ", '"', _json_escape_check(string(k)), '"', ": ")
                _json_write_check(io, obj[k], indent + 1)
                i < length(keys_sorted) && write(io, ",")
                write(io, "\n")
            end
            write(io, pad)
        end
        write(io, "}")
    elseif obj isa AbstractVector || obj isa Tuple
        write(io, "[")
        if !isempty(obj)
            write(io, "\n")
            for (i, v) in enumerate(obj)
                write(io, pad, "  ")
                _json_write_check(io, v, indent + 1)
                i < length(obj) && write(io, ",")
                write(io, "\n")
            end
            write(io, pad)
        end
        write(io, "]")
    else
        _json_write_check(io, string(obj), indent)
    end
    return nothing
end

function _write_json_file(path::AbstractString, obj)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        _json_write_check(io, obj, 0)
        write(io, "\n")
    end
    return path
end

function _ref_updates_per_epoch(ref, batch::Integer)
    value = get(ref, "updates_per_epoch", nothing)
    if value isa JSONObject
        return Int(value[string(batch)])
    elseif value isa Number
        return Int(value)
    end
    error("reference JSON: updates_per_epoch is missing or invalid")
end

function _file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function main_full_dataset_check()
    opts = parse_cli_args(ARGS)
    for key in ("dataset-dir", "reference-dir", "out")
        haskey(opts, key) || error("--$(key) is required")
    end
    dataset_dir = abspath(opts["dataset-dir"])
    reference_dir = abspath(opts["reference-dir"])
    batch = parse(Int, get(opts, "batch-size", "512"))
    epoch = parse(Int, get(opts, "epoch", "0"))
    worlds = parse.(Int, split(get(opts, "worlds", "1,2,4"), ','))
    label = get(opts, "label", "flux-current")
    verify_large_hashes = _parse_bool(get(opts, "verify-large-hashes", "false"))
    strict_contract = _parse_bool(get(opts, "strict-contract", "true"))
    batch > 0 || error("--batch-size must be positive")
    for world in worlds
        world in (1, 2, 4) || error("--worlds entries must be 1, 2, or 4; got $(world)")
    end

    ref_npz = joinpath(reference_dir, "full_dataset_reference.npz")
    ref_json = joinpath(reference_dir, "full_dataset_reference.json")
    isfile(ref_npz) || error("reference NPZ not found: $(ref_npz)")
    isfile(ref_json) || error("reference JSON not found: $(ref_json)")

    ds = open_full_dataset(dataset_dir; verify_large_hashes = verify_large_hashes)
    ref_arrays = read_npz_fields(ref_npz)
    ref = json_parse_file(ref_json)

    log = CheckLog()
    results = Dict{String,Any}[]
    skipped = Dict{String,Any}[]
    function record(name, ok::Bool, detail = "")
        log_check(log, name, ok, detail)
        push!(results, Dict{String,Any}("name" => name, "pass" => ok, "detail" => String(detail)))
        return ok
    end
    function record_skip(name, detail)
        push!(skipped, Dict{String,Any}("name" => name, "detail" => String(detail)))
        return nothing
    end

    for key in ("epoch_ids", "rank_ids_R1", "rank_ids_R2", "rank_ids_R4",
                "update1_global_ids", "update1_labels", "update1_transformed_nchw",
                "stats_mean", "stats_std")
        haskey(ref_arrays, key) || error("reference NPZ is missing $(key)")
    end

    n = size(ds.permutations, 1)
    perm = ds.permutations[:, epoch + 1]
    ids = epoch_ids(ds, epoch, batch)
    n_eff = length(ids)
    tail = n - n_eff
    ref_epoch = Int(ref["epoch"])
    ref_batch = Int(ref["batch_size"])
    ref_n_eff = Int(ref["n_eff"])
    ref_tail = Int(ref["tail_size"])
    ref_updates = _ref_updates_per_epoch(ref, batch)
    record("reference.epoch", ref_epoch == epoch, "ref=$(ref_epoch) actual=$(epoch)")
    record("reference.batch_size", ref_batch == batch, "ref=$(ref_batch) actual=$(batch)")
    record("reference.n_eff", ref_n_eff == n_eff, "ref=$(ref_n_eff) actual=$(n_eff)")
    record("reference.tail_size", ref_tail == tail, "ref=$(ref_tail) actual=$(tail)")
    record("reference.updates_per_epoch", ref_updates == n_eff ÷ batch,
           "ref=$(ref_updates) actual=$(n_eff ÷ batch)")

    ref_ids = ref_arrays["epoch_ids"]
    record("epoch_ids.shape", size(ref_ids) == (n_eff,), "ref=$(size(ref_ids)) actual=$((n_eff,))")
    record("epoch_ids.eltype", eltype(ref_ids) === Int64, "ref=$(eltype(ref_ids))")
    record("epoch_ids.values", ref_ids == ids)
    ref_sample_sha = String(ref["sample_order_sha256"])
    julia_sample_sha = sample_order_sha256(ds, epoch, batch)
    record("sample_order_sha256", julia_sample_sha == ref_sample_sha &&
           c_order_sha256(ref_ids) == ref_sample_sha,
           "julia=$(julia_sample_sha) ref=$(ref_sample_sha)")
    ref_update1_sha = String(ref["update1_order_sha256"])
    julia_update1_sha = update_order_sha256(ds, epoch, batch, 1)
    record("update1_order_sha256", julia_update1_sha == ref_update1_sha,
           "julia=$(julia_update1_sha) ref=$(ref_update1_sha)")
    ref_transform_sha = String(ref["transform_spec_sha256"])
    julia_transform_sha = transform_spec_sha256(ds)
    record("transform_spec_sha256", julia_transform_sha == ref_transform_sha,
           "julia=$(julia_transform_sha) ref=$(ref_transform_sha)")

    ref_mean = Float32.(vec(ref_arrays["stats_mean"]))
    ref_std = Float32.(vec(ref_arrays["stats_std"]))
    record("stats.mean", ref_mean == ds.mean32, "ref=$(ref_mean) actual=$(ds.mean32)")
    record("stats.std", ref_std == ds.std32, "ref=$(ref_std) actual=$(ds.std32)")

    for world in worlds
        key = "rank_ids_R$(world)"
        haskey(ref_arrays, key) || error("reference NPZ is missing $(key)")
        ref_seq = ref_arrays[key]
        seqs = [rank_sequence(perm, batch, r, world) for r in 1:world]
        if world == 1
            record("rank_sequence.R1.shape", size(ref_seq) == (n_eff,),
                   "ref=$(size(ref_seq)) actual=$((n_eff,))")
            record("rank_sequence.R1.values", ref_seq == seqs[1])
        else
            want_shape = (world, n_eff ÷ world)
            record("rank_sequence.R$(world).shape", size(ref_seq) == want_shape,
                   "ref=$(size(ref_seq)) actual=$(want_shape)")
            for r in 1:world
                record("rank_sequence.R$(world).rank$(r)", ref_seq[r, :] == seqs[r])
            end
        end
        chunk = batch ÷ world
        concatenated = Int64[]
        for k in 1:(n_eff ÷ batch)
            for r in 1:world
                first = (k - 1) * chunk + 1
                append!(concatenated, seqs[r][first:(first + chunk - 1)])
            end
        end
        union_ids = collect(Iterators.flatten(seqs))
        record("rank_sequence.R$(world).concatenation", concatenated == ids,
               "per-update rank order")
        record("rank_sequence.R$(world).union", Set(union_ids) == Set(ids))
        record("rank_sequence.R$(world).length", length(concatenated) == n_eff,
               "length=$(length(concatenated)) expected=$(n_eff)")
        record("rank_sequence.R$(world).unique", length(unique(union_ids)) == length(union_ids))
        for r in 1:world
            record("rank_sequence.R$(world).rank$(r)_unique",
                   length(unique(seqs[r])) == length(seqs[r]))
        end
    end

    if strict_contract
        record("contract.n_eff", n_eff == 99840, "n_eff=$(n_eff)")
        record("contract.tail_size", tail == 160, "tail=$(tail)")
        record("contract.unique", length(unique(ids)) == length(ids))
        record("contract.range", all(x -> 0 <= x < 100000, ids))
        for b in (512, 256)
            ids_b = epoch_ids(ds, epoch, b)
            want = b == 512 ? 195 : 390
            record("contract.updates.b$(b)", length(ids_b) ÷ b == want,
                   "updates=$(length(ids_b) ÷ b) expected=$(want)")
            record("contract.length.b$(b)", length(ids_b) == 99840,
                   "length=$(length(ids_b))")
        end
        missing = setdiff(collect(0:99999), ids)
        expected_tail = perm[n_eff + 1:end]
        record("contract.missing_tail", Set(missing) == Set(expected_tail),
               "missing=$(length(missing)) tail=$(length(expected_tail))")
    else
        record("contract.n_eff_multiple", n_eff % batch == 0)
        record("contract.unique", length(unique(ids)) == length(ids))
        record("contract.range", all(x -> 0 <= x < n, ids))
        for b in (512, 256)
            ids_b = epoch_ids(ds, epoch, b)
            record("contract.n_eff.b$(b)", length(ids_b) == (n ÷ b) * b,
                   "length=$(length(ids_b)) expected=$((n ÷ b) * b)")
        end
        missing = setdiff(collect(0:(n - 1)), ids)
        expected_tail = perm[n_eff + 1:end]
        record("contract.missing_tail", Set(missing) == Set(expected_tail),
               "missing=$(length(missing)) tail=$(length(expected_tail))")
    end

    ref_gids = ref_arrays["update1_global_ids"]
    ref_labels = ref_arrays["update1_labels"]
    ref_x = ref_arrays["update1_transformed_nchw"]
    julia_gids = global_batch_ids(ds, epoch, batch, 1)
    julia_labels = Int64.(ds.train_labels[julia_gids .+ 1])
    record("update1.global_ids", ref_gids == julia_gids)
    record("update1.labels", ref_labels == julia_labels)
    record("update1.transformed.shape", size(ref_x) == (batch, 3, 64, 64),
           "ref=$(size(ref_x))")
    record("update1.transformed.eltype", eltype(ref_x) === Float32,
           "ref=$(eltype(ref_x))")

    for world in worlds
        xs = Vector{Array{Float32,4}}()
        labels = Vector{Vector{Int64}}()
        shard_ids = Vector{Vector{Int64}}()
        for r in 1:world
            x, y, sids, _ = local_batch(ds, epoch, batch, 1, r, world)
            push!(xs, x)
            push!(labels, y)
            push!(shard_ids, sids)
        end
        x_cat = cat(xs...; dims = 4)
        nchw = permutedims(x_cat, (4, 3, 2, 1))
        labels_cat = vcat(labels...)
        sids_cat = vcat(shard_ids...)
        record("update1.R$(world).transformed.shape", size(nchw) == size(ref_x),
               "actual=$(size(nchw)) ref=$(size(ref_x))")
        record("update1.R$(world).transformed.eltype", eltype(nchw) === Float32)
        if size(nchw) == size(ref_x)
            diff = max_abs_diff(nchw, ref_x)
            record("update1.R$(world).transformed.max_abs_diff", diff <= 1e-6,
                   "max_abs_diff=$(diff)")
        else
            record("update1.R$(world).transformed.max_abs_diff", false, "shape mismatch")
        end
        record("update1.R$(world).labels", labels_cat == ref_labels)
        record("update1.R$(world).shard_ids", sids_cat == ref_gids)
        record("update1.R$(world).local_size", all(length(y) == batch ÷ world for y in labels),
               "batch=$(batch) world=$(world)")
    end

    onehot = Flux.onehotbatch(ref_labels .+ 1, 1:200)
    onehot_matrix = Matrix{Float32}(onehot)
    record("onehot.shape", size(onehot_matrix) == (200, batch),
           "actual=$(size(onehot_matrix))")
    onehot_eltype_ok = eltype(onehot) in (Bool, Float32) && eltype(onehot_matrix) === Float32
    record("onehot.eltype", onehot_eltype_ok,
           "onehot=$(eltype(onehot)) matrix=$(eltype(onehot_matrix))")
    column_sums = vec(sum(onehot_matrix; dims = 1))
    record("onehot.column_sums", all(x -> x == 1.0f0, column_sums))
    argmax_ok = all(argmax(view(onehot_matrix, :, j)) == ref_labels[j] + 1
                    for j in eachindex(ref_labels))
    record("onehot.argmax", argmax_ok)
    k = min(4, batch)
    explicit = zeros(Float32, 200, k)
    for j in 1:k
        explicit[ref_labels[j] + 1, j] = 1.0f0
    end
    record("onehot.explicit", onehot_matrix[:, 1:k] == explicit)

    bundle_path = joinpath(dataset_dir, "initial_bundle.npz")
    isfile(bundle_path) || error("bundle not found: $(bundle_path)")
    bundle = read_npz_fields(bundle_path)
    names = bundle_parameter_names()
    shapes = Dict(bundle_parameter_shapes())
    record("bundle.array_count", length(bundle) == 62, "actual=$(length(bundle))")
    record("bundle.parameter_count", sum(length, values(bundle)) == 11_271_432,
           "actual=$(sum(length, values(bundle)))")
    names_ok = Set(String.(keys(bundle))) == Set(names)
    record("bundle.name_set", names_ok)
    record("bundle.shapes", all(haskey(bundle, n) && size(bundle[n]) == s for (n, s) in shapes))
    bundle_hash = names_ok ? _sha256_concat([c_order_bytes(bundle[name]) for name in names]) : ""
    ref_bundle_hash = String(ref["bundle_hash"])
    record("bundle.bundle_hash", bundle_hash == ref_bundle_hash,
           "julia=$(bundle_hash) ref=$(ref_bundle_hash)")
    bundle_file_hash = _file_sha256(bundle_path)
    ref_bundle_file_hash = String(ref["bundle_file_sha256"])
    record("bundle.file_sha256", bundle_file_hash == ref_bundle_file_hash,
           "julia=$(bundle_file_hash) ref=$(ref_bundle_file_hash)")
    if strict_contract
        record("bundle.frozen_hash", bundle_hash == FROZEN_BUNDLE_HASH,
               "actual=$(bundle_hash)")
    end

    ref_files = get(ref, "dataset_files", nothing)
    if ref_files isa JSONObject
        for (name, entry) in ref_files
            path = joinpath(dataset_dir, name)
            if !isfile(path)
                record("dataset_file.$(name).present", false, "missing $(path)")
                continue
            end
            actual_bytes = filesize(path)
            declared_bytes = Int(entry["bytes"])
            record("dataset_file.$(name).bytes", actual_bytes == declared_bytes,
                   "actual=$(actual_bytes) ref=$(declared_bytes)")
            large = name in ("train_images.npy", "train_labels.npy", "val_images.npy")
            declared_sha = String(entry["sha256"])
            if !large || verify_large_hashes
                actual_sha = _file_sha256(path)
                record("dataset_file.$(name).sha256", actual_sha == declared_sha,
                       "actual=$(actual_sha) ref=$(declared_sha)")
            else
                record_skip("dataset_file.$(name).sha256",
                            "large-file hash skipped (--verify-large-hashes false)")
            end
        end
    else
        record("dataset_files.present", false)
    end

    for key in ("train_images.npy", "train_labels.npy")
        path = joinpath(dataset_dir, key)
        record("dataset.$(key).size", isfile(path) && filesize(path) > 0,
               isfile(path) ? "bytes=$(filesize(path))" : "missing")
    end

    ref_x_sha = c_order_sha256(ref_x)

    payload = Dict{String,Any}(
        "schema" => "static-full-dataset-julia-check-v1",
        "label" => label,
        "dataset_dir" => ds.dataset_dir,
        "reference_dir" => reference_dir,
        "epoch" => epoch,
        "batch_size" => batch,
        "worlds" => worlds,
        "verify_large_hashes" => verify_large_hashes,
        "strict_contract" => strict_contract,
        "checks_total" => length(results),
        "checks_failed" => length(log.failed),
        "pass" => isempty(log.failed),
        "checks" => results,
        "skipped_checks" => skipped,
        "reference_metadata" => Dict(
            "epoch" => ref_epoch,
            "batch_size" => ref_batch,
            "n_eff" => ref_n_eff,
            "tail_size" => ref_tail,
            "updates_per_epoch" => ref_updates,
            "sample_order_sha256" => ref_sample_sha,
            "update1_order_sha256" => ref_update1_sha,
            "transform_spec_sha256" => ref_transform_sha,
            "transform_sample_sha256" => get(ref, "transform_sample_sha256", nothing),
        ),
        "dataset_metadata" => dataset_metadata(ds),
        "bundle" => Dict(
            "bundle_hash" => bundle_hash,
            "bundle_file_sha256" => bundle_file_hash,
            "array_count" => length(bundle),
            "parameter_count" => sum(length, values(bundle)),
        ),
        "sample_order_sha256" => julia_sample_sha,
        "update1_order_sha256" => julia_update1_sha,
        "transform_spec_sha256" => julia_transform_sha,
        "update1_transformed_sha256" => ref_x_sha,
    )
    _write_json_file(opts["out"], payload)
    if isempty(log.failed)
        println("FULL_DATASET_JULIA_CHECK PASS label=$(label) checks=$(length(results))")
        return 0
    end
    println("FULL_DATASET_JULIA_CHECK FAIL label=$(label) checks=$(length(results))")
    return 1
end

exit(main_full_dataset_check())
