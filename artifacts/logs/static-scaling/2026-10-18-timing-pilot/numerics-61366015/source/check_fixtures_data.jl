"""
Phase 2 data/fixture acceptance check (NPZ-only; no Flux, no PyTorch).

Verifies the frozen file/array contract of the static-scaling experiment.
fixture_manifest.json is mandatory and enforced first: canonical file set,
per-file sha256/bytes, exact 35-array set, and per-array dtype/shape/sha256.
When the manifest is missing or inconsistent, the fixture-dependent semantic
checks (labels, transforms, schedule) are skipped; bundle and stats checks
still run. Then verifies data_fixture.npz, schedule_fixture.npz,
initial_bundle.npz + initial_bundle.manifest.json, and optionally
<prep-dir>/stats.json.

Prints one `CHECK <name> PASS|FAIL <detail>` line per check and a final
`PHASE2_JULIA_DATA_CHECK PASS|FAIL ...` line; exits 0/1.

Reproduction (local; NPZ-only environment from scripts/static_scaling/julia-env):
  cd /home/kurapica/Projects/ddp_flux/ddp-flux-thesis
  julia --startup-file=no --project=scripts/static_scaling/julia-env \\
    scripts/static_scaling/check_fixtures_data.jl \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \\
    --bundle-manifest artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.manifest.json

Reproduction (Leonardo; same NPZ-only project on the thesis copy):
  JULIA_DEPOT_PATH=<remote depot> julia --startup-file=no --project=<THESIS>/scripts/static_scaling/julia-env \\
    <THESIS>/scripts/static_scaling/check_fixtures_data.jl \\
    --fixture-dir <ARTIFACTS>/2026-10-16-phase2 --prep-dir <PREP>
"""
;
include(joinpath(@__DIR__, "replay.jl"))
using SHA
using Printf

const DATA_FIXTURE_ARRAYS = [
    ("decoded_train_images", UInt8, (12, 3, 64, 64)),
    ("train_labels", Int64, (12,)),
    ("augmentation", UInt8, (2, 12, 3)),
    ("train_transformed_ref", Float32, (2, 12, 3, 64, 64)),
    ("decoded_val_images", UInt8, (8, 3, 64, 64)),
    ("val_labels", Int64, (8,)),
    ("val_transformed_ref", Float32, (8, 3, 64, 64)),
    ("normalization_mean", Float32, (3,)),
    ("normalization_std", Float32, (3,)),
]

const SCHEDULE_FIXTURE_ARRAYS = [
    ("epoch0_permutation", Int64, (100000,)),
    ("augmentation_epochs01", UInt8, (2, 100000, 3)),
    ("rank_ids_R1", Int64, (99840,)),
    ("rank_ids_R2", Int64, (2, 49920)),
    ("rank_ids_R4", Int64, (4, 24960)),
]

function expected_bundle_shapes()
    shapes = Pair{String,Tuple}[]
    push!(shapes, "stem.conv.weight" => (64, 3, 3, 3))
    push!(shapes, "stem.gn.weight" => (64,))
    push!(shapes, "stem.gn.bias" => (64,))
    widths = (64, 128, 256, 512)
    prev = 64
    for s in 1:4
        ch = widths[s]
        for b in 0:1
            cin = b == 0 ? prev : ch
            pre = "stage$(s).block$(b)"
            push!(shapes, "$(pre).conv1.weight" => (ch, cin, 3, 3))
            push!(shapes, "$(pre).conv2.weight" => (ch, ch, 3, 3))
            push!(shapes, "$(pre).gn1.weight" => (ch,))
            push!(shapes, "$(pre).gn1.bias" => (ch,))
            push!(shapes, "$(pre).gn2.weight" => (ch,))
            push!(shapes, "$(pre).gn2.bias" => (ch,))
        end
        if s >= 2
            push!(shapes, "stage$(s).block0.proj.weight" => (ch, ch ÷ 2, 1, 1))
            push!(shapes, "stage$(s).block0.proj_gn.weight" => (ch,))
            push!(shapes, "stage$(s).block0.proj_gn.bias" => (ch,))
        end
        prev = ch
    end
    push!(shapes, "classifier.weight" => (200, 512))
    push!(shapes, "classifier.bias" => (200,))
    return shapes
end

function artifact_equals(actual, expected)
    isequal(actual, expected) && return true, ""
    return false, "max_abs_diff=" * fmt_float(max_abs_diff(actual, expected))
end

function shortlist(names, maxn = 5)
    isempty(names) && return "none"
    shown = join(names[1:min(length(names), maxn)], ";")
    length(names) > maxn ? "$(shown);+$(length(names) - maxn)more" : shown
end

const FIXTURE_MANIFEST_NAME = "fixture_manifest.json"
const FIXTURE_CANONICAL_FILES = ["data_fixture.npz", "schedule_fixture.npz",
                                 "linear_fixture.npz", "kernel_fixture.npz"]
const FIXTURE_ARRAY_COUNT = 35
const FIXTURE_GN_WEIGHT_COUNT = 20
const FIXTURE_GN_BIAS_COUNT = 20

function fixture_dtype_name(T::Type)
    T === UInt8 && return "uint8"
    T === Int8 && return "int8"
    T === Int16 && return "int16"
    T === Int32 && return "int32"
    T === Int64 && return "int64"
    T === UInt16 && return "uint16"
    T === UInt32 && return "uint32"
    T === UInt64 && return "uint64"
    T === Float16 && return "float16"
    T === Float32 && return "float32"
    T === Float64 && return "float64"
    T === Bool && return "bool"
    return lowercase(string(T))
end

_fixture_as_array(x::AbstractArray) = x
_fixture_as_array(x::Number) = reshape([x], ())

function fixture_c_order_sha256(a::AbstractArray)
    T = eltype(a)
    (T === Float32 || T === UInt8 || T === Int64) && return c_order_sha256(a)
    b = ndims(a) <= 1 ? Array(a) : permutedims(a, reverse(1:ndims(a)))
    return bytes2hex(sha256(Vector{UInt8}(reinterpret(UInt8, vec(b)))))
end

function fixture_skip(name::AbstractString, detail::AbstractString)
    println("CHECK ", name, " SKIP ", detail)
    return nothing
end

"""
    check_fixture_manifest_integrity(log, fixture_dir) -> Bool

Enforce `fixture_manifest.json` BEFORE any semantic check. Logs six checks:
`fixtures.manifest_present`, `fixtures.file_set`, `fixtures.file_sha256_bytes`,
`fixtures.array_set`, `fixtures.array_metadata`, `fixtures.array_sha256`.
A missing manifest logs `manifest_present` FAIL and SKIP lines for the rest
(SKIP lines never enter `log.failed`). Returns `true` only when every check
passed.
"""
function check_fixture_manifest_integrity(log::CheckLog, fixture_dir::String)
    remaining = ["fixtures.file_set", "fixtures.file_sha256_bytes", "fixtures.array_set",
                 "fixtures.array_metadata", "fixtures.array_sha256"]
    manifest_path = joinpath(fixture_dir, FIXTURE_MANIFEST_NAME)
    if !isfile(manifest_path)
        log_check(log, "fixtures.manifest_present", false, "not found: $(manifest_path)")
        for nm in remaining
            fixture_skip(nm, "fixture manifest missing")
        end
        return false
    end
    log_check(log, "fixtures.manifest_present", true, manifest_path)

    local manifest
    try
        manifest = json_parse_file(manifest_path)
    catch err
        detail = "manifest unreadable: " * one_line(sprint(showerror, err))
        for nm in remaining
            log_check(log, nm, false, detail)
        end
        return false
    end
    ok = true

    # -- canonical file set + per-file sha256/bytes -------------------------
    files = get(manifest, "files", nothing)
    file_paths = String[]
    files_wellformed = files isa Vector
    if files isa Vector
        for e in files
            if e isa JSONObject && get(e, "path", nothing) isa String
                push!(file_paths, String(e["path"]))
            else
                files_wellformed = false
            end
        end
    end
    want_files = sort(FIXTURE_CANONICAL_FILES)
    missing_files = setdiff(want_files, sort(file_paths))
    extra_files = setdiff(sort(file_paths), want_files)
    file_set_ok = files_wellformed && length(file_paths) == length(FIXTURE_CANONICAL_FILES) &&
                  isempty(missing_files) && isempty(extra_files)
    log_check(log, "fixtures.file_set", file_set_ok,
              "n=$(length(file_paths)) missing=$(shortlist(missing_files)) extra=$(shortlist(extra_files))")
    ok &= file_set_ok

    bad_hashes = String[]
    bad_bytes = String[]
    for e in (files isa Vector ? files : [])
        if !(e isa JSONObject) || !(get(e, "path", nothing) isa String)
            push!(bad_hashes, "<malformed-file-entry>")
            continue
        end
        rel = String(e["path"])
        path = joinpath(fixture_dir, rel)
        if !isfile(path)
            push!(bad_hashes, "$(rel):missing-on-disk")
            continue
        end
        want_sha = get(e, "sha256", nothing)
        if !(want_sha isa String)
            push!(bad_hashes, "$(rel):no-sha256")
        else
            got_sha = bytes2hex(sha256(read(path)))
            lowercase(got_sha) == lowercase(String(want_sha)) || push!(bad_hashes, "$(rel):sha256")
        end
        if haskey(e, "bytes")
            want_bytes = get(e, "bytes", nothing)
            if !(want_bytes isa Real) || filesize(path) != want_bytes
                push!(bad_bytes, "$(rel):bytes=$(filesize(path)) want=$(want_bytes)")
            end
        end
    end
    file_hash_ok = files_wellformed && isempty(bad_hashes) && isempty(bad_bytes)
    log_check(log, "fixtures.file_sha256_bytes", file_hash_ok,
              "n_checked=$(length(file_paths)) sha_mismatch=$(shortlist(bad_hashes)) " *
              "bytes_mismatch=$(shortlist(bad_bytes))")
    ok &= file_hash_ok

    # -- load the canonical npz archives (needed for the array checks) ------
    npz_fields = Dict{String,Dict{String,Any}}()
    unreadable = String[]
    for f in FIXTURE_CANONICAL_FILES
        path = joinpath(fixture_dir, f)
        if !isfile(path)
            push!(unreadable, "$(f):missing")
            continue
        end
        try
            npz_fields[f] = read_npz_fields(path)
        catch err
            push!(unreadable, "$(f):" * one_line(sprint(showerror, err)))
        end
    end
    expected_names = String[]
    for f in FIXTURE_CANONICAL_FILES
        haskey(npz_fields, f) || continue
        for k in keys(npz_fields[f])
            push!(expected_names, "$(f)::$(k)")
        end
    end

    # -- manifest `arrays` must be exactly the loaded 35-name set -----------
    arrays = get(manifest, "arrays", nothing)
    manifest_names = String[]
    arrays_wellformed = arrays isa Vector
    if arrays isa Vector
        for e in arrays
            if e isa JSONObject && get(e, "name", nothing) isa String
                push!(manifest_names, String(e["name"]))
            else
                arrays_wellformed = false
            end
        end
    end
    seen = Set{String}()
    dups = String[]
    for nm in manifest_names
        nm in seen ? push!(dups, nm) : push!(seen, nm)
    end
    expected_set = Set(expected_names)
    manifest_set = Set(manifest_names)
    missing_arrays = sort(collect(setdiff(expected_set, manifest_set)))
    extra_arrays = sort(collect(setdiff(manifest_set, expected_set)))
    array_set_ok = arrays_wellformed && isempty(unreadable) && isempty(missing_arrays) &&
                   isempty(extra_arrays) && isempty(dups) &&
                   length(manifest_names) == FIXTURE_ARRAY_COUNT
    log_check(log, "fixtures.array_set", array_set_ok,
              "n=$(length(manifest_names)) want=$(FIXTURE_ARRAY_COUNT) " *
              "missing=$(shortlist(missing_arrays)) extra=$(shortlist(extra_arrays)) " *
              "duplicates=$(shortlist(dups)) unreadable=$(shortlist(unreadable))")
    ok &= array_set_ok

    # -- per-entry dtype/shape and per-array sha256 -------------------------
    meta_bad = String[]
    hash_bad = String[]
    n_meta = 0
    n_hash = 0
    for e in (arrays isa Vector ? arrays : [])
        if !(e isa JSONObject) || !(get(e, "name", nothing) isa String)
            push!(meta_bad, "<malformed-array-entry>")
            push!(hash_bad, "<malformed-array-entry>")
            continue
        end
        name = String(e["name"])
        parts = split(name, "::"; limit = 2)
        if length(parts) != 2 || !haskey(npz_fields, parts[1]) || !haskey(npz_fields[parts[1]], parts[2])
            push!(meta_bad, "$(name):array-unavailable")
            push!(hash_bad, "$(name):array-unavailable")
            continue
        end
        arr = _fixture_as_array(npz_fields[parts[1]][parts[2]])
        n_meta += 1
        n_hash += 1
        want_dtype = fixture_dtype_name(eltype(arr))
        dtype = get(e, "dtype", nothing)
        dtype isa String && lowercase(dtype) == want_dtype ||
            push!(meta_bad, "$(name):dtype=$(dtype) want=$(want_dtype)")
        shape = get(e, "shape", nothing)
        if !(shape isa Vector && all(x -> x isa Integer, shape) &&
             collect(Int64, shape) == collect(Int64, size(arr)))
            push!(meta_bad, "$(name):shape=$(shape) want=$(collect(size(arr)))")
        end
        want_sha = get(e, "sha256", nothing)
        want_sha isa String && fixture_c_order_sha256(arr) == lowercase(String(want_sha)) ||
            push!(hash_bad, name)
    end
    log_check(log, "fixtures.array_metadata", isempty(meta_bad),
              "n_checked=$(n_meta) mismatches=$(shortlist(meta_bad))")
    log_check(log, "fixtures.array_sha256", isempty(hash_bad),
              "n_checked=$(n_hash) mismatch=$(shortlist(hash_bad))")
    ok &= isempty(meta_bad) && isempty(hash_bad)

    return ok
end

function _is_gn_name(n::AbstractString, suffix::AbstractString)
    for pre in (".gn", ".gn1", ".gn2", ".proj_gn")
        endswith(n, pre * suffix) && return true
    end
    return false
end

function run_data_checks(log::CheckLog, opts::Dict{String,String})
    fixture_dir = get(opts, "fixture-dir", "")
    isempty(fixture_dir) && error("--fixture-dir is required")
    isdir(fixture_dir) || error("--fixture-dir not found: $(fixture_dir)")
    bundle_path = get(opts, "bundle", joinpath(fixture_dir, "initial_bundle.npz"))
    manifest_path = get(opts, "bundle-manifest", joinpath(fixture_dir, "initial_bundle.manifest.json"))
    prep_dir = get(opts, "prep-dir", "")

    integrity_ok = check_fixture_manifest_integrity(log, fixture_dir)
    df = nothing
    if !integrity_ok
        println("CHECK fixtures.semantic_checks SKIP fixture manifest integrity not satisfied")
    else
        # ----- data fixture ---------------------------------------------------
        data_path = joinpath(fixture_dir, "data_fixture.npz")
        df = read_npz_fields(data_path)
        check_contract(log, df, DATA_FIXTURE_ARRAYS, "data.contract")

        if all(haskey(df, n) for (n, _, _) in DATA_FIXTURE_ARRAYS)
            train_labels = df["train_labels"]
            val_labels = df["val_labels"]
            ok = all(0 .<= train_labels .<= 199)
            log_check(log, "labels.train_range_0_199", ok,
                      ok ? "n=$(length(train_labels))" : "min=$(minimum(train_labels)) max=$(maximum(train_labels))")
            ok = all(1 .<= (train_labels .+ 1) .<= 200)
            log_check(log, "labels.train_julia_conversion_1_200", ok, ok ? "" : "labels+1 out of range")
            ok = all(0 .<= val_labels .<= 199)
            log_check(log, "labels.val_range_0_199", ok,
                      ok ? "n=$(length(val_labels))" : "min=$(minimum(val_labels)) max=$(maximum(val_labels))")
            ok = all(1 .<= (val_labels .+ 1) .<= 200)
            log_check(log, "labels.val_julia_conversion_1_200", ok, ok ? "" : "labels+1 out of range")

            mean32 = df["normalization_mean"]
            std32 = df["normalization_std"]
            train_imgs = nchw_to_whcn(df["decoded_train_images"])
            train_aug = df["augmentation"]
            train_ref = df["train_transformed_ref"]
            for e in 1:2
                triples = train_aug[e, :, :]
                out = replay_transform(train_imgs, triples, mean32, std32)
                ref_whcn = permutedims(selectdim(train_ref, 1, e), (4, 3, 2, 1))
                ok, detail = artifact_equals(out, ref_whcn)
                log_check(log, "transform.train_epoch$(e - 1)", ok, detail)
            end
            val_imgs = nchw_to_whcn(df["decoded_val_images"])
            # validation has no augmentation: the identity crop under padding-4 is
            # the center window (top=4, left=4), which returns the original image
            val_triples = zeros(UInt8, size(val_imgs, 4), 3)
            val_triples[:, 1] .= 4
            val_triples[:, 2] .= 4
            out = replay_transform(val_imgs, val_triples, mean32, std32)
            ref_whcn = permutedims(df["val_transformed_ref"], (4, 3, 2, 1))
            ok, detail = artifact_equals(out, ref_whcn)
            log_check(log, "transform.val", ok, detail)
        else
            log_check(log, "transform.train_epoch0", false, "data fixture incomplete")
            log_check(log, "transform.train_epoch1", false, "data fixture incomplete")
            log_check(log, "transform.val", false, "data fixture incomplete")
        end

        # ----- schedule fixture ----------------------------------------------
        sched_path = joinpath(fixture_dir, "schedule_fixture.npz")
        sf = read_npz_fields(sched_path)
        check_contract(log, sf, SCHEDULE_FIXTURE_ARRAYS, "schedule.contract")

        if haskey(sf, "epoch0_permutation")
            perm = sf["epoch0_permutation"]
            ok = length(perm) == 100000 && sort(perm) == collect(Int64, 0:99999)
            log_check(log, "schedule.permutation_is_0_to_99999", ok,
                      ok ? "n=$(length(perm))" : "length=$(length(perm)) unique=$(length(unique(perm))) min=$(minimum(perm)) max=$(maximum(perm))")
        else
            perm = Int64[]
            log_check(log, "schedule.permutation_is_0_to_99999", false, "epoch0_permutation missing")
        end

        if haskey(sf, "augmentation_epochs01")
            aug = sf["augmentation_epochs01"]
            ok = all(0 .<= aug[:, :, 1] .<= 8) && all(0 .<= aug[:, :, 2] .<= 8) && all(aug[:, :, 3] .<= 1)
            log_check(log, "schedule.augmentation_ranges", ok,
                      ok ? "top/left in 0..8, flip in 0..1, n=$(length(aug))" : "range violation")
        else
            log_check(log, "schedule.augmentation_ranges", false, "augmentation_epochs01 missing")
        end

        if !isempty(perm)
            for (R, key) in ((1, "rank_ids_R1"), (2, "rank_ids_R2"), (4, "rank_ids_R4"))
                if !haskey(sf, key)
                    log_check(log, "schedule.rank_sequence_R$(R)", false, "$(key) missing")
                    continue
                end
                ref = sf[key]
                if R == 1
                    seq = rank_sequence(perm, 512, 1, 1)
                    ok = length(ref) == 99840 && seq == vec(ref)
                    detail = ok ? "n=$(length(seq))" : "length(ref)=$(length(ref)) mismatch_reason=$(seq == vec(ref) ? "length" : "values")"
                else
                    seqs = [rank_sequence(perm, 512, r, R) for r in 1:R]
                    ok = size(ref) == (R, 99840 ÷ R) && all(seqs[r] == ref[r, :] for r in 1:R)
                    bad = ok ? 0 : findfirst(r -> seqs[r] != ref[r, :], 1:R)
                    detail = ok ? "n=$(length(seqs[1])) per rank" : "shape=$(size(ref)) first_bad_rank=$(something(bad, 0))"
                end
                log_check(log, "schedule.rank_sequence_R$(R)", ok, detail)
            end
            ids = epoch_sample_ids(perm, 512)
            for (R, key) in ((2, "rank_ids_R2"), (4, "rank_ids_R4"))
                haskey(sf, key) || continue
                ref = sf[key]
                size(ref) == (R, length(ids) ÷ R) || continue
                concat = sort(vcat([Vector(ref[r, :]) for r in 1:R]...))
                ok = concat == sort(ids)
                log_check(log, "schedule.rank_coverage_R$(R)", ok,
                          ok ? "ranks cover $(length(ids)) ids" : "coverage mismatch")
            end
            log_check(log, "tail.arithmetic",
                      (100000 - 99840 == 160) && (99840 ÷ 512 == 195) && (99840 ÷ 256 == 390),
                      "100000-99840=$(100000 - 99840) 99840/512=$(99840 ÷ 512) 99840/256=$(99840 ÷ 256)")
            ids512 = epoch_sample_ids(perm, 512)
            ids256 = epoch_sample_ids(perm, 256)
            tail512 = Set(perm[(length(ids512) + 1):end])
            tail256 = Set(perm[(length(ids256) + 1):end])
            ok = length(ids512) == 99840 && length(ids256) == 99840 && tail512 == tail256 && length(tail512) == 160
            log_check(log, "tail.same_set_all_configs", ok,
                      "dropped512=$(length(perm) - length(ids512)) dropped256=$(length(perm) - length(ids256)) same_set=$(tail512 == tail256)")
        end
    end

    # ----- initial bundle -------------------------------------------------
    bundle = read_npz_fields(bundle_path)
    expected = expected_bundle_shapes()
    expected_names = [p.first for p in expected]
    expected_shapes = Dict(expected)
    actual_names = sort(collect(keys(bundle)))
    want_names = sort(expected_names)
    missing = setdiff(want_names, actual_names)
    extra = setdiff(actual_names, want_names)
    log_check(log, "bundle.name_set",
              length(bundle) == 62 && isempty(missing) && isempty(extra),
              "n=$(length(bundle)) missing=$(shortlist(missing)) extra=$(shortlist(extra))")

    bad_shapes = String[]
    bad_dtypes = String[]
    for name in expected_names
        haskey(bundle, name) || continue
        a = bundle[name]
        eltype(a) === Float32 || push!(bad_dtypes, name)
        size(a) == expected_shapes[name] || push!(bad_shapes, name)
    end
    log_check(log, "bundle.shapes", isempty(bad_shapes),
              "n_checked=$(length(expected_names)) mismatches=$(shortlist(bad_shapes))")
    log_check(log, "bundle.dtypes_float32", isempty(bad_dtypes),
              "n_checked=$(length(expected_names)) mismatches=$(shortlist(bad_dtypes))")

    any_missing = any(!haskey(bundle, n) for n in expected_names)
    total_elems = any_missing ? -1 : sum(prod(size(bundle[n])) for n in expected_names)
    log_check(log, "bundle.element_count", total_elems == 11271432,
              "total=$(total_elems) want=11271432")

    gn_names = [n for n in expected_names if _is_gn_name(n, ".weight")]
    gn_bias_names = [n for n in expected_names if _is_gn_name(n, ".bias")]
    if any_missing
        log_check(log, "bundle.gn_initial_values", false, "arrays missing")
    else
        counts_ok = length(gn_names) == FIXTURE_GN_WEIGHT_COUNT &&
                    length(gn_bias_names) == FIXTURE_GN_BIAS_COUNT
        okw = counts_ok && all(all(x -> x == 1.0f0, bundle[n]) for n in gn_names)
        okb = counts_ok && all(all(x -> x == 0.0f0, bundle[n]) for n in gn_bias_names)
        log_check(log, "bundle.gn_initial_values", okw && okb,
                  "weights$(okw ? "==1" : "!=1") ($(length(gn_names)) arrays) biases$(okb ? "==0" : "!=0") ($(length(gn_bias_names)) arrays)")
    end
    if any_missing
        log_check(log, "bundle.classifier_shapes", false, "classifier arrays missing")
    else
        ok = size(bundle["classifier.weight"]) == (200, 512) && size(bundle["classifier.bias"]) == (200,)
        log_check(log, "bundle.classifier_shapes", ok,
                  "weight=$(size(bundle["classifier.weight"])) bias=$(size(bundle["classifier.bias"]))")
    end

    # ----- bundle manifest ------------------------------------------------
    if !isfile(manifest_path)
        log_check(log, "bundle.manifest_sha256", false, "manifest not found: $(manifest_path)")
        log_check(log, "bundle.bundle_hash", false, "manifest not found")
    else
        manifest = json_parse_file(manifest_path)
        entries = json_bundle_entries(manifest)
        digest_hex = Dict{String,String}()
        dup_names = String[]
        metadata_bad = String[]
        for (nm, e) in entries
            haskey(digest_hex, nm) && push!(dup_names, nm)
            e isa JSONObject || continue
            d = get(e, "sha256", nothing)
            d isa String && (digest_hex[nm] = lowercase(d))
            if haskey(bundle, nm)
                dt = get(e, "dtype", nothing)
                sh = get(e, "shape", nothing)
                dt isa String && lowercase(dt) != "float32" && push!(metadata_bad, "$(nm):dtype=$(dt)")
                sh isa Vector && collect(Int64, sh) != collect(Int64, size(bundle[nm])) && push!(metadata_bad, "$(nm):shape")
            end
        end
        extra_entries = setdiff(sort(collect(keys(digest_hex))), sort(expected_names))
        log_check(log, "bundle.manifest_entries",
                  !isempty(entries) && isempty(dup_names) && isempty(extra_entries),
                  "n_entries=$(length(entries)) duplicates=$(shortlist(dup_names)) extra=$(shortlist(extra_entries))")
        missing_entries = [n for n in expected_names if !haskey(digest_hex, n)]
        hash_mismatch = String[]
        if !any_missing
            for n in expected_names
                haskey(digest_hex, n) || continue
                c_order_sha256(bundle[n]) == digest_hex[n] || push!(hash_mismatch, n)
            end
        end
        log_check(log, "bundle.manifest_sha256",
                  isempty(missing_entries) && isempty(hash_mismatch),
                  "n_checked=$(length(expected_names)) missing=$(shortlist(missing_entries)) mismatch=$(shortlist(hash_mismatch))")
        log_check(log, "bundle.manifest_metadata", isempty(metadata_bad),
                  "mismatches=$(shortlist(metadata_bad))")

        bh = json_find(manifest, "bundle_hash")
        if bh === JSON_MISSING
            log_check(log, "bundle.bundle_hash", false, "bundle_hash not recorded in manifest")
        elseif !(bh isa String) || !occursin(r"^[0-9a-f]{64}$", lowercase(String(bh)))
            log_check(log, "bundle.bundle_hash", false, "bundle_hash is not a 64-hex string")
        else
            definition = JSON_MISSING
            for k in ("bundle_hash_definition", "bundle_hash_rule", "bundle_hash_method", "bundle_hash_recipe")
                v = json_find(manifest, k)
                v === JSON_MISSING || (definition = v; break)
            end
            if definition === JSON_MISSING
                log_check(log, "bundle.bundle_hash", true,
                          "format ok; no definition recorded, recomputation skipped")
            elseif !any_missing
                raw = Dict(n => c_order_bytes(bundle[n]) for n in expected_names)
                order = [nm for (nm, _) in entries if haskey(bundle, nm)]
                cands = bundle_hash_candidates(raw, digest_hex, order)
                match = findfirst(p -> p.second == lowercase(String(bh)), cands)
                if match === nothing
                    defstr = definition isa String ? one_line(definition) : "structured"
                    log_check(log, "bundle.bundle_hash", false,
                              "recompute failed def=\"$(first(defstr, 100))\" tried=$(length(cands)) rules")
                else
                    log_check(log, "bundle.bundle_hash", true, "recomputed rule=$(cands[match].first)")
                end
            else
                log_check(log, "bundle.bundle_hash", false, "bundle arrays missing; cannot recompute")
            end
        end
    end

    # ----- stats.json consistency ----------------------------------------
    if isempty(prep_dir)
        log_check(log, "normalization.stats_json", true, "skipped: --prep-dir not provided")
    elseif !integrity_ok
        println("CHECK normalization.stats_json SKIP fixture manifest integrity not satisfied")
    else
        stats_path = joinpath(prep_dir, "stats.json")
        if !isfile(stats_path)
            log_check(log, "normalization.stats_json", false, "not found: $(stats_path)")
        else
            stats = json_parse_file(stats_path)
            jmean = json_find(stats, "mean")
            jstd = json_find(stats, "std")
            if !(jmean isa Vector) || !(jstd isa Vector)
                log_check(log, "normalization.stats_json", false, "stats.json lacks mean/std arrays")
            else
                fmean = df["normalization_mean"]
                fstd = df["normalization_std"]
                okm = Float32.(jmean) == fmean
                oks = Float32.(jstd) == fstd
                detail = (okm ? "mean==fixture" : "mean differs fixture=$(fmean) stats=$(Float32.(jmean))") * " " *
                         (oks ? "std==fixture" : "std differs fixture=$(fstd) stats=$(Float32.(jstd))")
                log_check(log, "normalization.stats_json", okm && oks, detail)
            end
        end
    end

    return log
end

function main_data(argv)
    opts = parse_cli_args(argv)
    log = CheckLog()
    try
        run_data_checks(log, opts)
    catch err
        log_check(log, "unhandled_exception", false, one_line(sprint(showerror, err)))
    end
    return final_line(log, "PHASE2_JULIA_DATA_CHECK")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main_data(ARGS))
end
