# Shared utilities for the static-scaling Phase 2 fixtures (Julia track).
# No Flux dependency; needs only NPZ.jl and the SHA standard library.
#
# Purpose
#   * read .npz/.npy artifacts produced by the Python track,
#   * hash arrays with the exact C-order bytes Python's hashlib sees,
#   * rebuild the epoch schedule, rank shards, and the training transform
#     (zero-pad 4 -> 64x64 crop -> horizontal flip -> float32 /255, -mean, /std).
#
# Reproduction (local, NPZ-only environment):
#   cd /home/kurapica/Projects/ddp_flux/ddp-flux-thesis
#   julia --startup-file=no --project=scripts/static_scaling/julia-env -e \
#     'include("scripts/static_scaling/replay.jl"); println(bytes2hex(sha256(c_order_bytes(zeros(Float32,2))))); println("replay OK")'
#
# Reproduction (local, NPZ + Flux stacked load path):
#   JULIA_LOAD_PATH="$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" \
#     julia --startup-file=no -e 'using NPZ, Flux; println("stack OK")'
#
# Reproduction (Leonardo: Flux lives in the test project of the remote worktree):
#   JULIA_LOAD_PATH="<THESIS>/scripts/static_scaling/julia-env:<REMOTE_FLUX>/test:@stdlib" \
#     julia --startup-file=no -e 'using NPZ, Flux; println("stack OK")'
#   (<REMOTE_FLUX> example: /leonardo_work/IscrC_UNMASKED/epicazio/flux-integration)

using NPZ
using SHA
using Printf

# ---------------------------------------------------------------------------
# NPZ reading
# ---------------------------------------------------------------------------

"""
    read_np(path)

Read a `.npz` archive (returns `Dict{String,Any}`) or a `.npy` array (returns the
array). NPZ.jl 0.4.3 supports both; if a `.npy` file cannot be read, print an
explicit workaround telling the caller to pass an `.npz` copy, then rethrow.
"""
function read_np(path::AbstractString)
    isfile(path) || error("read_np: file not found: $(path)")
    if endswith(path, ".npy")
        try
            return NPZ.npzread(path)
        catch err
            println(stderr, "read_np: NPZ.jl v$(pkgversion(NPZ)) could not read this .npy file: $(path)")
            println(stderr, "read_np: ", sprint(showerror, err))
            println(stderr, "read_np: workaround - write an .npz copy and pass that path instead:")
            println(stderr, "read_np:   python3 -c 'import numpy as np; a = np.load(\"$(path)\"); np.savez(\"$(splitext(path)[1]).npz\", a=a)'")
            rethrow()
        end
    end
    return NPZ.npzread(path)
end

"""
    read_npz_fields(path)

Read an `.npz` archive as a `Dict{String,Any}` of arrays. Errors if `path` is a
bare `.npy` array (use `read_np` for those).
"""
function read_npz_fields(path::AbstractString)
    isfile(path) || error("read_npz_fields: file not found: $(path)")
    d = NPZ.npzread(path)
    d isa AbstractDict || error("read_npz_fields: $(path) is not an .npz archive (NPZ.jl returned $(typeof(d))); use read_np for .npy files")
    return Dict{String,Any}(d)
end

# ---------------------------------------------------------------------------
# C-order hashing
# ---------------------------------------------------------------------------

# True when the array's linear (column-major) element order equals its logical
# C-order traversal. All Julia `Array`s are contiguous; this also accepts views.
function is_c_contiguous(a::AbstractArray)
    try
        st = strides(a)
        expected = 1
        for d in 1:ndims(a)
            st[d] == expected || return false
            expected = Base.checked_mul(expected, size(a, d))
        end
        return true
    catch
        return false
    end
end

"""
    c_order_bytes(a)

Raw bytes of `a` in numpy C order (`arr.tobytes(order="C")`).

Trick: a Julia array loaded from a C-order numpy array is column-major, so its
memory layout is the reverse of the numpy layout. `permutedims(a, reverse(1:ndims(a)))`
reverses the dimensions; the Julia memory of the result then equals the original
C-order byte sequence. 0/1-dimensional arrays need no permutation.
"""
function c_order_bytes(a::AbstractArray)
    T = eltype(a)
    T === Float32 || T === UInt8 || T === Int64 ||
        error("c_order_bytes: unsupported eltype $(T); Float32/UInt8/Int64 only")
    is_c_contiguous(a) || error("c_order_bytes: array is not contiguous")
    b = ndims(a) <= 1 ? Array(a) : permutedims(a, reverse(1:ndims(a)))
    return Vector{UInt8}(reinterpret(UInt8, vec(b)))
end

c_order_sha256(a::AbstractArray) = bytes2hex(sha256(c_order_bytes(a)))

function _sha256_concat(parts)
    ctx = SHA.SHA2_256_CTX()
    for p in parts
        SHA.update!(ctx, p)
    end
    return bytes2hex(SHA.digest!(ctx))
end

"""
    bundle_hash_candidates(raw_bytes, digest_hex, manifest_order)

Candidate recomputations of a manifest `bundle_hash` from per-array C-order
bytes and per-array sha256 hex digests. The frozen contract fixes the per-array
hashes but not the bundle rule, so the checker tries the common rules and
reports which one matches. Returns `Vector{Pair{String,String}}` (rule => hex).
"""
function bundle_hash_candidates(raw_bytes::Dict{String,Vector{UInt8}},
                                digest_hex::Dict{String,String},
                                manifest_order::Vector{String})
    cands = Pair{String,String}[]
    sorted = sort(manifest_order)
    if all(haskey(raw_bytes, x) for x in manifest_order)
        push!(cands, "sha256(concat C-order bytes, manifest order)" =>
              _sha256_concat([raw_bytes[x] for x in manifest_order]))
        push!(cands, "sha256(concat C-order bytes, name-sorted)" =>
              _sha256_concat([raw_bytes[x] for x in sorted]))
    end
    if all(haskey(digest_hex, x) for x in manifest_order)
        push!(cands, "sha256(concat sha256 hex, manifest order)" =>
              _sha256_concat([Vector{UInt8}(codeunits(digest_hex[x])) for x in manifest_order]))
        push!(cands, "sha256(concat sha256 hex, name-sorted)" =>
              _sha256_concat([Vector{UInt8}(codeunits(digest_hex[x])) for x in sorted]))
        push!(cands, "sha256(concat raw sha256 digests, manifest order)" =>
              _sha256_concat([hex2bytes(digest_hex[x]) for x in manifest_order]))
        push!(cands, "sha256(concat raw sha256 digests, name-sorted)" =>
              _sha256_concat([hex2bytes(digest_hex[x]) for x in sorted]))
        push!(cands, "sha256(concat 'name:hex\\n', manifest order)" =>
              _sha256_concat([Vector{UInt8}(codeunits(x * ":" * digest_hex[x] * "\n")) for x in manifest_order]))
        push!(cands, "sha256(concat 'name:hex\\n', name-sorted)" =>
              _sha256_concat([Vector{UInt8}(codeunits(x * ":" * digest_hex[x] * "\n")) for x in sorted]))
    end
    return cands
end

# ---------------------------------------------------------------------------
# Epoch schedule and rank sharding
# ---------------------------------------------------------------------------

"""
    epoch_sample_ids(perm, batch) -> ids

Effective sample ids of one epoch: drop the incomplete tail so that
`length(ids)` is a multiple of `batch` (invariant: 100000 -> 99840 at 512/256).
"""
function epoch_sample_ids(perm::AbstractVector, batch::Integer)
    batch > 0 || error("epoch_sample_ids: batch must be positive")
    n_eff = (length(perm) ÷ batch) * batch
    return perm[1:n_eff]
end

"""
    rank_shard(batch_ids, rank, world) -> shard

Chunk `rank` (1-based) of one global batch split into `world` equal contiguous
shards. Errors when the batch does not divide evenly.
"""
function rank_shard(batch_ids::AbstractVector, rank::Integer, world::Integer)
    world >= 1 || error("rank_shard: world must be >= 1")
    1 <= rank <= world || error("rank_shard: rank $(rank) outside 1:$(world)")
    length(batch_ids) % world == 0 ||
        error("rank_shard: batch length $(length(batch_ids)) not divisible by world $(world)")
    chunk = length(batch_ids) ÷ world
    first = (rank - 1) * chunk + 1
    return batch_ids[first:(first + chunk - 1)]
end

"""
    rank_sequence(perm, batch, rank, world) -> ids

Concatenation of rank `rank`'s shard over every global batch of the epoch
(the Julia side of `rank_ids_R{world}`).
"""
function rank_sequence(perm::AbstractVector, batch::Integer, rank::Integer, world::Integer)
    ids = epoch_sample_ids(perm, batch)
    out = Vector{eltype(perm)}(undef, length(ids) ÷ world)
    pos = 1
    for start in 1:batch:length(ids)
        shard = rank_shard(ids[start:(start + batch - 1)], rank, world)
        out[pos:(pos + length(shard) - 1)] .= shard
        pos += length(shard)
    end
    return out
end

# ---------------------------------------------------------------------------
# Layout conversion and training transform
# ---------------------------------------------------------------------------

# Canonical numpy/PyTorch NCHW -> Flux WHCN (W, H, C, N). The permutation is its
# own inverse: applying it again returns NCHW.
nchw_to_whcn(a::AbstractArray) = permutedims(a, (4, 3, 2, 1))

"""
    replay_transform(images_whcn, triples, mean32, std32) -> Float32 WHCN

Replay the frozen training transform on uint8 WHCN images:
zero-pad 4 on W and H (dims 1 and 2 of WHCN), crop at zero-based `(top, left)`
(output index `w` maps to padded column `left + w`), reverse the W axis when
`flip == 1`, then `((Float32(u) / 255) - mean_c) / std_c` in float32 with the
same operation order as numpy. `triples[i, :] = (top, left, flip)` belongs to
image `i`; padding zeros are normalized like real pixels because the Python
track applies the transform to the padded uint8 image.
"""
function replay_transform(images_whcn::Array{UInt8}, triples::AbstractMatrix{UInt8},
                          mean32::AbstractVector{Float32}, std32::AbstractVector{Float32})
    W, H, C, N = size(images_whcn)
    size(triples, 1) == N ||
        error("replay_transform: $(size(triples,1)) triples for $(N) images")
    size(triples, 2) == 3 || error("replay_transform: triples must have 3 columns")
    length(mean32) == C == length(std32) ||
        error("replay_transform: mean/std length must match the $(C) channels")
    out = Array{Float32}(undef, W, H, C, N)
    @inbounds for i in 1:N
        top = Int(triples[i, 1])
        left = Int(triples[i, 2])
        flip = Int(triples[i, 3])
        (0 <= top <= 8 && 0 <= left <= 8 && (flip == 0 || flip == 1)) ||
            error("replay_transform: invalid triple (top=$top, left=$left, flip=$flip) at row $i")
        for c in 1:C
            m = mean32[c]
            s = std32[c]
            for h in 1:H
                src_h = h + top - 4
                in_h = 1 <= src_h <= H
                for w in 1:W
                    dst_w = flip == 1 ? (W + 1 - w) : w
                    src_w = dst_w + left - 4
                    u = (in_h && 1 <= src_w <= W) ? images_whcn[src_w, src_h, c, i] : 0x00
                    out[w, h, c, i] = (Float32(u) / 255.0f0 - m) / s
                end
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Tolerances (frozen specification section 11)
# ---------------------------------------------------------------------------

# Elementwise rule: |a - b| <= atol + rtol * max(|a|, |b|).
# Note this differs from Base.isapprox, which uses max(atol, rtol*...).
within_tol(a, b; atol, rtol) = abs(a - b) <= atol + rtol * max(abs(a), abs(b))

function max_abs_diff(a, b)
    length(a) == length(b) || error("max_abs_diff: length mismatch $(length(a)) vs $(length(b))")
    m = 0.0
    for (x, y) in zip(a, b)
        d = abs(Float64(x) - Float64(y))
        m = max(m, d)
    end
    return m
end

fmt_float(x) = @sprintf("%.9g", x)

# ---------------------------------------------------------------------------
# Deterministic CHECK lines
# ---------------------------------------------------------------------------

mutable struct CheckLog
    failed::Vector{String}
end
CheckLog() = CheckLog(String[])

function log_check(log::CheckLog, name::AbstractString, ok::Bool, detail::AbstractString = "")
    println("CHECK ", name, " ", ok ? "PASS" : "FAIL", isempty(detail) ? "" : " " * detail)
    ok || push!(log.failed, name)
    return ok
end

function final_line(log::CheckLog, tag::AbstractString)
    if isempty(log.failed)
        println(tag, " PASS")
        return 0
    end
    println(tag, " FAIL ", join(log.failed, ","))
    return 1
end

one_line(s::AbstractString) = replace(s, r"\s+" => " ")

# Contract check over a list of (name, eltype, shape) entries; one CHECK line
# per array so a failure names the exact array.
function check_contract(log::CheckLog, fields, contract, label)
    for (name, T, shape) in contract
        if !haskey(fields, name)
            log_check(log, "$(label).$(name)", false, "missing")
            continue
        end
        a = fields[name]
        if !(a isa AbstractArray)
            log_check(log, "$(label).$(name)", false, "not an array: $(typeof(a))")
            continue
        end
        ok = eltype(a) === T && size(a) == shape
        detail = ok ? "" : "got $(eltype(a)) $(size(a)), want $(T) $(shape)"
        log_check(log, "$(label).$(name)", ok, detail)
    end
    return nothing
end

function parse_cli_args(argv::AbstractVector{<:AbstractString})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument: $(a)")
        body = a[3:end]
        if occursin('=', body)
            k, v = split(body, '=', limit = 2)
            opts[k] = v
        elseif i < length(argv) && !startswith(argv[i + 1], "--")
            opts[body] = argv[i + 1]
            i += 1
        else
            opts[body] = "true"
        end
        i += 1
    end
    return opts
end

# ---------------------------------------------------------------------------
# Minimal JSON reader (no external dependency)
# ---------------------------------------------------------------------------

# Objects keep document order as a vector of pairs; `Dict` would not.
struct JSONObject
    pairs::Vector{Pair{String,Any}}
end

Base.length(o::JSONObject) = length(o.pairs)
Base.iterate(o::JSONObject, args...) = iterate(o.pairs, args...)
Base.haskey(o::JSONObject, k::AbstractString) = any(p -> p.first == k, o.pairs)

function Base.getindex(o::JSONObject, k::AbstractString)
    for p in o.pairs
        p.first == k && return p.second
    end
    throw(KeyError(k))
end

function Base.get(o::JSONObject, k::AbstractString, default)
    for p in o.pairs
        p.first == k && return p.second
    end
    return default
end

struct _JSONMissing end
const JSON_MISSING = _JSONMissing()

const _JSON_WS = Set{UInt8}((0x20, 0x09, 0x0a, 0x0d))

function json_parse(text::AbstractString)
    b = codeunits(text)
    val, pos = _json_value(b, 1)
    pos = _json_skipws(b, pos)
    pos <= length(b) && error("json_parse: trailing content at byte $(pos)")
    return val
end

json_parse_file(path::AbstractString) = json_parse(read(path, String))

function _json_skipws(b, pos)
    while pos <= length(b) && b[pos] in _JSON_WS
        pos += 1
    end
    return pos
end

function _json_expect(b, pos, lit::Vector{UInt8})
    view(b, pos:(pos + length(lit) - 1)) == lit ||
        error("json_parse: expected $(String(lit)) at byte $(pos)")
    return nothing
end

function _json_value(b, pos)
    pos = _json_skipws(b, pos)
    pos <= length(b) || error("json_parse: unexpected end of input")
    c = b[pos]
    if c == UInt8('{')
        return _json_object(b, pos + 1)
    elseif c == UInt8('[')
        return _json_array(b, pos + 1)
    elseif c == UInt8('"')
        return _json_string(b, pos + 1)
    elseif c == UInt8('t')
        _json_expect(b, pos, Vector{UInt8}(codeunits("true")))
        return true, pos + 4
    elseif c == UInt8('f')
        _json_expect(b, pos, Vector{UInt8}(codeunits("false")))
        return false, pos + 5
    elseif c == UInt8('n')
        _json_expect(b, pos, Vector{UInt8}(codeunits("null")))
        return nothing, pos + 4
    else
        return _json_number(b, pos)
    end
end

function _json_object(b, pos)
    pairs = Pair{String,Any}[]
    pos = _json_skipws(b, pos)
    pos <= length(b) && b[pos] == UInt8('}') && return JSONObject(pairs), pos + 1
    while true
        pos = _json_skipws(b, pos)
        pos <= length(b) && b[pos] == UInt8('"') ||
            error("json_parse: object key must be a string at byte $(pos)")
        key, pos = _json_string(b, pos + 1)
        pos = _json_skipws(b, pos)
        pos <= length(b) && b[pos] == UInt8(':') ||
            error("json_parse: expected ':' at byte $(pos)")
        val, pos = _json_value(b, pos + 1)
        push!(pairs, key => val)
        pos = _json_skipws(b, pos)
        pos <= length(b) || error("json_parse: unterminated object")
        if b[pos] == UInt8(',')
            pos += 1
        elseif b[pos] == UInt8('}')
            return JSONObject(pairs), pos + 1
        else
            error("json_parse: expected ',' or '}' at byte $(pos)")
        end
    end
end

function _json_array(b, pos)
    vals = Any[]
    pos = _json_skipws(b, pos)
    pos <= length(b) && b[pos] == UInt8(']') && return vals, pos + 1
    while true
        val, pos = _json_value(b, pos)
        push!(vals, val)
        pos = _json_skipws(b, pos)
        pos <= length(b) || error("json_parse: unterminated array")
        if b[pos] == UInt8(',')
            pos += 1
        elseif b[pos] == UInt8(']')
            return vals, pos + 1
        else
            error("json_parse: expected ',' or ']' at byte $(pos)")
        end
    end
end

function _json_string(b, pos)
    io = IOBuffer()
    while true
        pos <= length(b) || error("json_parse: unterminated string")
        c = b[pos]
        if c == UInt8('"')
            return String(take!(io)), pos + 1
        elseif c == UInt8('\\')
            pos += 1
            pos <= length(b) || error("json_parse: unterminated escape")
            e = b[pos]
            if e == UInt8('"')
                write(io, UInt8('"'))
            elseif e == UInt8('\\')
                write(io, UInt8('\\'))
            elseif e == UInt8('/')
                write(io, UInt8('/'))
            elseif e == UInt8('b')
                write(io, UInt8('\b'))
            elseif e == UInt8('f')
                write(io, UInt8('\f'))
            elseif e == UInt8('n')
                write(io, UInt8('\n'))
            elseif e == UInt8('r')
                write(io, UInt8('\r'))
            elseif e == UInt8('t')
                write(io, UInt8('\t'))
            else
                error("json_parse: unsupported escape \\$(Char(e)) at byte $(pos)")
            end
            pos += 1
        else
            write(io, c)
            pos += 1
        end
    end
end

function _json_number(b, pos)
    start = pos
    while pos <= length(b) && (UInt8('0') <= b[pos] <= UInt8('9') || b[pos] in (UInt8('-'), UInt8('+'), UInt8('.'), UInt8('e'), UInt8('E')))
        pos += 1
    end
    tok = String(b[start:(pos - 1)])
    isempty(tok) && error("json_parse: invalid value at byte $(start)")
    if occursin('.', tok) || occursin('e', tok) || occursin('E', tok)
        return parse(Float64, tok), pos
    end
    return parse(Int64, tok), pos
end

# Depth-first search for the first occurrence of `key`; JSON_MISSING if absent.
function json_find(node, key::AbstractString)
    if node isa JSONObject
        for (k, v) in node
            k == key && return v
            r = json_find(v, key)
            r !== JSON_MISSING && return r
        end
    elseif node isa Vector
        for v in node
            r = json_find(v, key)
            r !== JSON_MISSING && return r
        end
    end
    return JSON_MISSING
end

# Bundle entries in document order: either `arrays` as a list of objects, or as
# an object name -> entry, or (fallback) every object carrying name + sha256.
function json_bundle_entries(manifest)
    entries = Tuple{String,Any}[]
    arrays = json_find(manifest, "arrays")
    if arrays isa Vector
        for v in arrays
            if v isa JSONObject && haskey(v, "name") && haskey(v, "sha256") && v["name"] isa String
                push!(entries, (v["name"], v))
            end
        end
    elseif arrays isa JSONObject
        for (k, v) in arrays
            if v isa JSONObject && haskey(v, "sha256")
                nm = haskey(v, "name") ? String(v["name"]) : k
                push!(entries, (nm, v))
            end
        end
    end
    isempty(entries) && _collect_entry_objects!(entries, manifest)
    return entries
end

function _collect_entry_objects!(entries, node)
    if node isa JSONObject
        if haskey(node, "name") && haskey(node, "sha256") && node["name"] isa String
            push!(entries, (String(node["name"]), node))
        end
        for (_, v) in node
            _collect_entry_objects!(entries, v)
        end
    elseif node isa Vector
        for v in node
            _collect_entry_objects!(entries, v)
        end
    end
    return nothing
end
