"""Opt-in, synchronized W=1 allocator diagnostic; never used by normal timing."""

function _cache_api_evidence(name::Symbol)
    try
        f = getproperty(CUDA, name)
        method = which(f, Tuple{})
        path = String(method.file)
        doc = string(Base.Docs.doc(f))
        evidence = Dict{String,Any}("api" => "CUDA.$(name)()", "doc" => doc,
            "method_file" => path, "method_line" => method.line)
        if isfile(path)
            evidence["source_sha256"] = bytes2hex(SHA.sha256(read(path)))
            lines = readlines(path)
            excerpt_range = max(1, method.line - 12):min(length(lines), method.line + 18)
            evidence["source_excerpt"] = join(lines[excerpt_range], "\n")
        end
        return evidence
    catch err
        return Dict{String,Any}("api" => "CUDA.$(name)()",
            "error" => sprint(showerror, err), "doc" => "")
    end
end

function _cache_diagnostic_init()
    used = _cache_api_evidence(:used_memory)
    reserved = _cache_api_evidence(:cached_memory)
    Dict{String,Any}(
        "schema" => "static-cache-diagnostic-v1", "pid" => getpid(),
        "device_uuid" => "GPU-" * string(CUDA.uuid(CUDA.device())),
        "cuda_version" => string(Base.pkgversion(CUDA)),
        "runtime_api_evidence" => Dict("allocator_used_bytes" => used,
                                       "allocator_reserved_bytes" => reserved),
        "retention_policy" => "At post_training/post_gc/post_reclaim the same model, optimizer state, dataset maps, initial CPU bundle and final loss are retained via GC.@preserve; per-step input, output and gradient temporaries do not escape steps(). Flux optimizer does not retain a parameter-gradient buffer. Raw post_training is captured before state hashing; model/optimizer copies to CPU for invariant hashing occur between post_training and explicit GC, and again after post_reclaim. This host work may trigger ordinary Julia GC.",
        "counter_semantics" => Dict(
            "allocator_used_bytes" => "application-in-use CUDA pool bytes; not a pure model-tensor total",
            "allocator_reserved_bytes" => "CUDA pool backing/reserved bytes INCLUDING in-use bytes; CUDA.cached_memory is not unused-cache bytes",
            "process_bytes" => "nvidia-smi per-process resident GPU memory, MiB converted to bytes",
            "device_used_bytes" => "device total minus free; includes other processes and runtime allocations",
            "total_bytes" => "device total bytes"),
        "peak_policy" => "snapshots are synchronized point samples, not within-step high-water marks; external nvidia-smi trace remains sampled process footprint",
        "snapshots" => Any[])
end

function _cache_allocator_counter(name::Symbol, evidence, expected_doc::AbstractString)
    occursin(expected_doc, lowercase(get(evidence, "doc", ""))) ||
        return nothing, "unavailable: loaded API documentation does not verify expected pool semantics"
    try
        value = getproperty(CUDA, name)()
        (value === missing || value === nothing) &&
            return nothing, "unavailable: allocator backend does not expose this counter"
        return Int(value), "measured"
    catch err
        return nothing, "unavailable: " * sprint(showerror, err)
    end
end

function _cache_process_bytes()
    # Querying at an explicitly synchronized boundary avoids extrapolating the
    # asynchronous external trace to the exact cache-release instant.
    output = read(`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`, String)
    values = Int[]
    for line in split(output, '\n')
        fields = split(line, ',')
        length(fields) >= 2 || continue
        pid = tryparse(Int, strip(fields[1]))
        memory = tryparse(Int, strip(fields[2]))
        pid == getpid() && memory !== nothing && push!(values, memory * 1024^2)
    end
    isempty(values) && error("cache diagnostic: nvidia-smi did not report this process")
    return maximum(values)
end

function _cache_snapshot!(diagnostic, phase::AbstractString)
    CUDA.synchronize()
    epoch = time()
    evidence = diagnostic["runtime_api_evidence"]
    used, used_status = _cache_allocator_counter(:used_memory,
        evidence["allocator_used_bytes"], "in use")
    reserved, reserved_status = _cache_allocator_counter(:cached_memory,
        evidence["allocator_reserved_bytes"], "backing memory")
    free, total = CUDA.memory_info()
    process = _cache_process_bytes()
    push!(diagnostic["snapshots"], Dict{String,Any}(
        "phase" => String(phase), "epoch" => epoch, "end_epoch" => time(),
        "allocator_used_bytes" => used, "allocator_reserved_bytes" => reserved,
        "process_bytes" => process, "device_used_bytes" => Int(total - free),
        "total_bytes" => Int(total),
        "counter_status" => Dict("allocator_used_bytes" => used_status,
            "allocator_reserved_bytes" => reserved_status, "process_bytes" => "measured",
            "device_used_bytes" => "measured", "total_bytes" => "measured")))
    println("CACHE_SNAPSHOT ", phase, " process_MiB=", process / 1024^2,
        " allocator_used_bytes=", used, " allocator_reserved_bytes=", reserved)
    flush(stdout)
    return nothing
end
