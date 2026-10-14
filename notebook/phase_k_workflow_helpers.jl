# Shared helpers for the Phase K IJulia notebooks.
#
# Artifact-first: every function reads the saved campaign files under
# `artifacts/logs/conditional-wrapper/`. The module never loads Flux, MPI, or
# CUDA, and it never joins a collective. Fresh training runs, when requested,
# are started as child processes through the bounded campaign driver.
#
# Used by:
#   notebook/conditional_graph_distributed_workflow.ipynb
#
# The plotting plotters are Python programs called through `uv` with the pinned
# environment in `scripts/plots/requirements-phase-k.txt`; their PNG outputs are
# displayed through IJulia MIME display.

module PhaseKWorkflow

using SHA, Markdown, Printf, Dates

export find_repo_root, PHASE_K_BUNDLES, resolve_bundle, bundle_artifacts,
    BUNDLE_ARTIFACT_PATHS, demo_figures, display_png, render_demo_figures,
    render_network_figures, md_table, sha256_file, relative_path,
    parse_demo_console, parse_demo_steps, parse_collectives,
    bundle_overview, provenance_table, prepared_provenance_table,
    validation_summary, discrepancy_summary, learning_rate_summary,
    collective_summary, continuation_summary, exposure_summary,
    schedule_cross_check, representative_traces, artifact_inventory,
    campaign_commands, local_reproduction_command, reproduction_command,
    plotter_environment_command, run_command, run_child_command, PHASE_K_ROOT,
    load_bundle, BundleData, mode_rows, mode_collectives,
    provenance_entries, max_reference_difference, summary_status,
    metadata_lines, meta_first, meta_value_after, meta_extra, metadata_version,
    read_table, run_plotter, run_training_child, FLUX_WORKTREE,
    FLUX_TEST_PROJECT, REPO_ROOT, PLOT_REQUIREMENTS

# ---------------------------------------------------------------------------
# Paths and bundle map
# ---------------------------------------------------------------------------

const PHASE_K_ROOT = joinpath("artifacts", "logs", "conditional-wrapper")
const PLOT_REQUIREMENTS = joinpath("scripts", "plots", "requirements-phase-k.txt")

function find_repo_root(start)
    dir = abspath(start)
    for _ in 1:8
        if isfile(joinpath(dir, "README.md")) &&
           isdir(joinpath(dir, "artifacts")) &&
           isdir(joinpath(dir, "scripts", "examples"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && break
        dir = parent
    end
    error("cannot locate the thesis repository root above $(start)")
end

const REPO_ROOT = find_repo_root(@__DIR__)
const FLUX_WORKTREE = normpath(joinpath(REPO_ROOT, "..", "flux-integration"))
const FLUX_TEST_PROJECT = joinpath(FLUX_WORKTREE, "test")

# A tutorial bundle is a complete run bundle: the campaign root, the exact
# rank configuration, and the shared prepared-dataset provenance. The default
# is the four-rank Leonardo run; `tutorial-2rank` selects the two-rank
# deathstar run. Both read the same prepared MNIST/Fashion-MNIST artifact.
const PHASE_K_BUNDLES = Dict(
    "tutorial-4rank" => (
        key = "tutorial-4rank",
        config = "nccl4",
        description = "Leonardo, 4x A100-SXM-64GB, CUDA/NCCL",
        host = "leonardo",
        node = "lrdn0436",
        devices = "[1, 2, 3, 0]",
        gpu = "NVIDIA A100-SXM-64GB",
        backend = "nccl",
        ranks = 4,
        # Pre-DB-3 evidence: presence + 2 trunk gradients + 2 per present head.
        collective_protocol = "legacy-5/7",
        label = "k2-leonardo-nccl4",
        run_root = joinpath(PHASE_K_ROOT, "demo", "k2-2026-10-05-leonardo-nccl4"),
        prepared = joinpath(PHASE_K_ROOT, "demo", "k2-2026-10-05-deathstar-mpi", "prepared"),
        campaign_record = joinpath(PHASE_K_ROOT, "demo",
            "CAMPAIGN-2026-10-05-leonardo-k2.md"),
        steps = 120,
        checkpoint_step = 2,
        timeout = 3600,
    ),
    "tutorial-2rank" => (
        key = "tutorial-2rank",
        config = "nccl2",
        description = "deathstar, 2x TITAN RTX, CUDA/NCCL",
        host = "deathstar",
        node = "gnode01",
        devices = "[1, 2]",
        gpu = "NVIDIA TITAN RTX",
        backend = "nccl",
        ranks = 2,
        # Pre-DB-3 evidence: presence + 2 trunk gradients + 2 per present head.
        collective_protocol = "legacy-5/7",
        label = "k2-deathstar-nccl2",
        run_root = joinpath(PHASE_K_ROOT, "demo", "k2-2026-10-05-deathstar-nccl2"),
        prepared = joinpath(PHASE_K_ROOT, "demo", "k2-2026-10-05-deathstar-mpi", "prepared"),
        campaign_record = joinpath(PHASE_K_ROOT, "demo",
            "CAMPAIGN-2026-10-05-deathstar-k2.md"),
        steps = 120,
        checkpoint_step = 2,
        timeout = 5400,
    ),
)

# Collective protocol identity. Historical pre-DB-3 campaigns use the 5/7
# protocol (presence + 2 trunk gradients + 2 per globally present head).
# Guarded campaigns (DB-3) add the trailing host `Float32` length-1 `avg`
# gradient-validity vote and use the 6/8 protocol. The protocols are never
# inferred from record counts: every bundle must pin its protocol explicitly.
const COLLECTIVE_PROTOCOLS = Dict(
    "legacy-5/7" => (name = "5/7", base = 3, vote_required = false),
    "guarded-6/8" => (name = "6/8", base = 4, vote_required = true),
)

function collective_protocol(bundle)
    name = get(bundle, :collective_protocol, nothing)
    name === nothing && error(
        "bundle $(bundle.key) does not declare collective_protocol; historical " *
        "bundles pin \"legacy-5/7\" and post-DB-3 bundles pin \"guarded-6/8\"")
    protocol = get(COLLECTIVE_PROTOCOLS, name, nothing)
    protocol === nothing && error(
        "bundle $(bundle.key): unknown collective protocol $(repr(name)); known: " *
        join(sort(collect(keys(COLLECTIVE_PROTOCOLS))), ", "))
    return protocol
end

function resolve_bundle(key::AbstractString)
    haskey(PHASE_K_BUNDLES, key) || error(
        "unknown tutorial configuration $(repr(key)); available: " *
        join(sort(collect(keys(PHASE_K_BUNDLES))), ", "))
    bundle = PHASE_K_BUNDLES[key]
    missing = bundle_missing_files(bundle)
    isempty(missing) || error(
        "selected bundle $(key) is incomplete; missing required artifact(s):\n  " *
        join(missing, "\n  "))
    return bundle
end

function bundle_artifacts(bundle)
    root = joinpath(REPO_ROOT, bundle.run_root)
    config = bundle.config
    train = joinpath(root, config, "train")
    resume = joinpath(root, config, "resume")
    prepared = joinpath(REPO_ROOT, bundle.prepared)
    return (
        root = root,
        summary = joinpath(root, "summary.txt"),
        train_dir = train,
        resume_dir = resume,
        train_console = joinpath(root, "$(config)-train", "console.log"),
        resume_console = joinpath(root, "$(config)-resume", "console.log"),
        train_command = joinpath(root, "$(config)-train", "command.txt"),
        resume_command = joinpath(root, "$(config)-resume", "command.txt"),
        network_diagram = joinpath(REPO_ROOT, "reports", "figures", "phase-k-network"),
        prepared_dir = prepared,
        prepared_file = joinpath(prepared, "mnist-fashion_prepared.jls"),
        prepare_metadata = joinpath(prepared, "prepare_metadata.txt"),
        sources = joinpath(prepared, "sources.tsv"),
        subset_indices = joinpath(prepared, "subset_indices.tsv"),
        schedule = joinpath(prepared, "schedule-ranks$(bundle.ranks).tsv"),
        train_metadata = joinpath(train, "metadata.txt"),
        resume_metadata = joinpath(resume, "metadata.txt"),
        checkpoint = joinpath(train, "checkpoint-step$(bundle.checkpoint_step).jls"),
        trajectory = joinpath(train, "trajectory.jls"),
        campaign_record = joinpath(REPO_ROOT, bundle.campaign_record),
    )
end

const BUNDLE_ARTIFACT_PATHS = (
    :summary, :train_console, :resume_console, :train_command, :resume_command,
    :prepare_metadata, :sources, :subset_indices, :schedule, :train_metadata,
    :resume_metadata, :checkpoint, :trajectory,
)

const MODE_STEMS = ("validation", "learning-rate", "predictions", "metrics")

function bundle_missing_files(bundle)
    paths = bundle_artifacts(bundle)
    missing = String[]
    for name in BUNDLE_ARTIFACT_PATHS
        isfile(paths[name]) || push!(missing, relative_path(paths[name]))
    end
    for mode in ("train", "resume")
        dir = mode == "train" ? paths.train_dir : paths.resume_dir
        for stem in MODE_STEMS
            file = joinpath(dir, "$(stem)-$(mode).tsv")
            isfile(file) || push!(missing, relative_path(file))
        end
    end
    return missing
end

relative_path(path) = begin
    absolute = abspath(path)
    startswith(absolute, REPO_ROOT * "/") ? absolute[(length(REPO_ROOT) + 2):end] : absolute
end

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

sha256_file(path) = open(path) do io
    bytes2hex(SHA.sha256(io))
end

# Tables are shown as HTML. Displaying the Markdown.MD value directly publishes
# a text/latex representation that uses the tabular environment, which KaTeX
# renderers (for example VS Code notebooks) reject. text/plain keeps the pipe
# table for terminals.
struct HtmlTable
    html::String
    markdown::String
end

Base.show(io::IO, ::MIME"text/plain", table::HtmlTable) = print(io, table.markdown)
Base.show(io::IO, ::MIME"text/html", table::HtmlTable) = print(io, table.html)

function md_table(headers::AbstractVector, rows::AbstractVector)
    io = IOBuffer()
    print(io, "| ", join(string.(headers), " | "), " |\n")
    print(io, "|", join(fill(" --- ", length(headers)), "|"), "|\n")
    for row in rows
        cells = [replace(string(cell), "|" => "\\|") for cell in row]
        print(io, "| ", join(cells, " | "), " |\n")
    end
    source = String(take!(io))
    return HtmlTable(Markdown.html(Markdown.parse(source)), source)
end

function run_command(cmd::Cmd)
    out = IOBuffer()
    err = IOBuffer()
    process = run(pipeline(cmd, stdout = out, stderr = err), wait = false)
    wait(process)
    return (; cmd, exitcode = process.exitcode,
        stdout = String(take!(out)), stderr = String(take!(err)))
end

function display_png(path::AbstractString)
    isfile(path) || error("figure $(path) does not exist")
    display("image/png", read(path))
    return nothing
end

function read_table(path; required = String[])
    isfile(path) || error("missing required table $(relative_path(path))")
    lines = readlines(path)
    isempty(lines) && error("$(path) is empty")
    header = split(chomp(lines[1]), '\t')
    for name in required
        name in header || error("$(relative_path(path)): missing column $(name); header is $(header)")
    end
    rows = Vector{Dict{String,String}}()
    for (index, line) in enumerate(lines[2:end])
        isempty(strip(line)) && error("$(relative_path(path)): blank row at line $(index + 1)")
        fields = split(chomp(line), '\t')
        length(fields) == length(header) || error(
            "$(relative_path(path)):$(index + 1): expected $(length(header)) fields, got $(length(fields))")
        push!(rows, Dict{String,String}(zip(header, fields)))
    end
    return (; header, rows)
end

function row_value(rows, index, column)
    haskey(rows[index], column) || error("row $(index): no column $(column)")
    return rows[index][column]
end

function parse_float_field(rows, index, column)
    value = row_value(rows, index, column)
    parsed = tryparse(Float64, value)
    parsed === nothing && error("row $(index): $(column)=$(repr(value)) is not a number")
    return parsed
end

function parse_int_field(rows, index, column)
    value = tryparse(Int, row_value(rows, index, column))
    value === nothing && error("row $(index): column $(column) is not an integer")
    return value
end

# ---------------------------------------------------------------------------
# Metadata files (tab-separated key[/value] lines)
# ---------------------------------------------------------------------------

function metadata_lines(path)
    isfile(path) || error("missing metadata file $(relative_path(path))")
    return [split(chomp(line), '\t') for line in readlines(path) if !isempty(strip(line))]
end

function meta_values(lines, key)
    return [fields for fields in lines if !isempty(fields) && fields[1] == key]
end

function meta_first(lines, key, index = 2, default = nothing)
    entries = meta_values(lines, key)
    isempty(entries) && return default
    return length(entries[1]) >= index ? entries[1][index] : default
end

function meta_value_after(lines, key, prefix)
    for fields in lines
        fields[1] == key || continue
        for field in fields[2:end]
            startswith(field, prefix) && return field[(length(prefix) + 1):end]
        end
    end
    return nothing
end

function meta_extra(lines, key, default = nothing)
    for fields in lines
        fields[1] == "extra" && length(fields) >= 3 && fields[2] == key && return fields[3]
    end
    return default
end

# ---------------------------------------------------------------------------
# Console parsing (regex scan over concatenated records)
# ---------------------------------------------------------------------------

const STEP_RE = r"DEMO_STEP mode=(\w+) step=(\d+) cycle=(\d+) rank=(\d+) head=([a-z]) present_a=(true|false) present_b=(true|false) counts_a=(\d+) counts_b=(\d+) loss_ref=(\S+) loss_local=(\S+) peer_param_maxerr=(\S+) peer_mom1_maxerr=(\S+) peer_mom2_maxerr=(\S+) beta_exact=(true|false) collectives=(\d+) resume_match=(true|false|n/a)"

const COLLECTIVE_RE = r"DEMO_COLLECTIVE mode=(\w+) step=(\d+) rank=(\d+) order=(\d+) type=(.+?) shape=\(([^)]*)\) avg=(true|false) local=(nothing|Float32\[[^\]]*\]) global=(nothing|Float32\[[^\]]*\])"

function parse_demo_steps(path)
    isfile(path) || error("missing console log $(relative_path(path))")
    text = read(path, String)
    steps = Vector{NamedTuple}()
    for m in eachmatch(STEP_RE, text)
        push!(steps, (
            mode = String(m.captures[1]),
            step = parse(Int, m.captures[2]),
            cycle = parse(Int, m.captures[3]),
            rank = parse(Int, m.captures[4]),
            head = String(m.captures[5]),
            present_a = m.captures[6] == "true",
            present_b = m.captures[7] == "true",
            counts_a = parse(Int, m.captures[8]),
            counts_b = parse(Int, m.captures[9]),
            loss_ref = parse(Float64, m.captures[10]),
            loss_local = parse(Float64, m.captures[11]),
            peer_param_maxerr = parse(Float64, m.captures[12]),
            peer_mom1_maxerr = parse(Float64, m.captures[13]),
            peer_mom2_maxerr = parse(Float64, m.captures[14]),
            beta_exact = m.captures[15] == "true",
            collectives = parse(Int, m.captures[16]),
            resume_match = m.captures[17] == "n/a" ? nothing : m.captures[17] == "true",
        ))
    end
    return steps
end

function parse_flag_vector(text)
    text == "nothing" && return nothing
    m = match(r"Float32\[(.*)\]", text)
    m === nothing && error("cannot parse presence flags $(repr(text))")
    body = strip(m.captures[1])
    isempty(body) && return Float32[]
    return parse.(Float32, split(body, ","))
end

# The console prints shapes as `(1,)`, `(16, 2)`, so the captured text keeps
# the trailing comma of one-dimensional shapes; parse the actual dimensions.
function shape_dims(text)
    return [parse(Int, strip(dim)) for dim in split(text, ',') if !isempty(strip(dim))]
end

function parse_collectives(path)
    isfile(path) || error("missing console log $(relative_path(path))")
    text = read(path, String)
    entries = Vector{NamedTuple}()
    for m in eachmatch(COLLECTIVE_RE, text)
        push!(entries, (
            mode = String(m.captures[1]),
            step = parse(Int, m.captures[2]),
            rank = parse(Int, m.captures[3]),
            order = parse(Int, m.captures[4]),
            buftype = strip(m.captures[5]),
            shape = String(m.captures[6]),
            avg = m.captures[7] == "true",
            local_flags = parse_flag_vector(m.captures[8]),
            global_flags = parse_flag_vector(m.captures[9]),
        ))
    end
    return entries
end

function parse_demo_console(path)
    steps = parse_demo_steps(path)
    text = read(path, String)
    return (
        steps = steps,
        collectives = parse_collectives(path),
        rank_done = length(collect(eachmatch(r"DEMO_RANK_DONE mode=(\w+)", text))),
        schema_ok = collect(eachmatch(r"DEMO_SCHEMA_OK mode=(\w+) validation_steps=(\d+)", text)),
        train_done = occursin("DEMO_TRAIN_DONE", text),
        resume_done = occursin("DEMO_RESUME_DONE", text),
    )
end

# ---------------------------------------------------------------------------
# Parsed bundle view
# ---------------------------------------------------------------------------

struct BundleData
    bundle::NamedTuple
    paths::NamedTuple
    steps::Vector{NamedTuple}
    collectives::Vector{NamedTuple}
    train_meta::Vector{Vector{SubString{String}}}
    resume_meta::Vector{Vector{SubString{String}}}
    prepare_meta::Vector{Vector{SubString{String}}}
end

function load_bundle(bundle)
    paths = bundle_artifacts(bundle)
    steps = vcat(parse_demo_steps(paths.train_console), parse_demo_steps(paths.resume_console))
    collectives = vcat(parse_collectives(paths.train_console),
        parse_collectives(paths.resume_console))
    return BundleData(bundle, paths, steps, collectives,
        metadata_lines(paths.train_metadata), metadata_lines(paths.resume_metadata),
        metadata_lines(paths.prepare_metadata))
end

function mode_rows(data::BundleData, mode::AbstractString)
    return [record for record in data.steps if record.mode == mode]
end

function mode_collectives(data::BundleData, mode::AbstractString)
    return [entry for entry in data.collectives if entry.mode == mode]
end

# ---------------------------------------------------------------------------
# Overview and provenance tables
# ---------------------------------------------------------------------------

function metadata_version(lines, name)
    for fields in meta_values(lines, "package")
        length(fields) >= 3 && fields[2] == name && return fields[3]
    end
    return "unknown"
end

function summary_status(path)
    isfile(path) || return "summary missing"
    for line in readlines(path)
        startswith(line, "status\t") && return strip(line[(length("status\t") + 1):end])
    end
    return "unknown"
end

function bundle_overview(data::BundleData)
    bundle = data.bundle
    train = data.train_meta
    schedule_digest = meta_value_after(train, "schedule", "digest=")
    eta = meta_extra(train, "optimizer")
    rows = [
        ["selection", bundle.key],
        ["description", bundle.description],
        ["campaign root", relative_path(data.paths.root)],
        ["config / ranks", "$(bundle.config) / $(bundle.ranks)"],
        ["backend / dtype", "$(meta_first(train, "backend")) / $(meta_first(train, "dtype"))"],
        ["host / node", "$(meta_first(train, "host")) / $(bundle.node)"],
        ["GPU devices", "$(bundle.gpu), devices $(bundle.devices)"],
        ["Julia", meta_first(train, "julia")],
        ["Flux / Optimisers", "$(metadata_version(train, "Flux")) / $(metadata_version(train, "Optimisers"))"],
        ["MPI", metadata_version(train, "MPI")],
        ["dataset", meta_first(train, "dataset")],
        ["steps / checkpoint", "120 / step $(bundle.checkpoint_step)"],
        ["global / local batch", "$(meta_value_after(train, "schedule", "global_batch=")) / " *
            meta_value_after(train, "schedule", "local_batch=")],
        ["schedule digest", schedule_digest],
        ["optimizer", eta],
        ["tolerances", meta_first(train, "tolerances")],
        ["campaign status", summary_status(data.paths.summary)],
    ]
    return md_table(["field", "value"], rows)
end

function provenance_entries(data::BundleData)
    recorded = Dict{String,String}()
    for fields in meta_values(data.train_meta, "source_hash")
        length(fields) >= 3 && (recorded[fields[2]] = fields[3])
    end
    entries = Vector{NamedTuple}()
    for rel in sort(collect(keys(recorded)))
        local_path = startswith(rel, "flux-integration/") ?
            joinpath(FLUX_WORKTREE, rel[(length("flux-integration/") + 1):end]) :
            joinpath(REPO_ROOT, rel)
        current = isfile(local_path) ? sha256_file(local_path) : "missing"
        push!(entries, (; rel, recorded = recorded[rel], current,
            match = current == recorded[rel]))
    end
    return entries
end

function provenance_table(data::BundleData)
    entries = provenance_entries(data)
    rows = [[entry.rel, entry.recorded[1:12] * "...",
        entry.match ? entry.current[1:12] * "..." : entry.current[1:12] * "... (changed)",
        entry.match ? "match" : "changed"] for entry in entries]
    n_match = count(entry -> entry.match, entries)
    table = md_table(["source", "campaign SHA-256", "current SHA-256", "state"], rows)
    return (; entries, n_match, n_total = length(entries), table)
end

function prepared_provenance_table(data::BundleData)
    lines = data.prepare_meta
    rows = [
        ["loader", meta_first(lines, "loader")],
        ["preprocessing", meta_first(lines, "preprocessing")],
        ["label encoding", meta_first(lines, "label_encoding")],
        ["task mapping", meta_first(lines, "task_mapping")],
        ["seeds", meta_first(lines, "seeds")],
        ["dataset digest", meta_first(lines, "dataset_digest")],
        ["model definition", "version $(meta_first(lines, "model_definition_version"))"],
        ["optimizer", meta_first(lines, "optimizer")],
        ["prepared SHA-256 (recorded)", meta_extra(data.train_meta, "prepared_sha256")],
    ]
    for R in (2, 4)
        digest = nothing
        for fields in meta_values(lines, "schedule_workers")
            length(fields) >= 3 && fields[2] == string(R) && (digest = fields[3])
        end
        digest === nothing || push!(rows,
            ["schedule digest, $(R) ranks", digest])
    end
    return md_table(["prepared artifact field", "value"], rows)
end

function artifact_inventory(data::BundleData)
    paths = data.paths
    entries = Pair{String,String}[]
    for name in BUNDLE_ARTIFACT_PATHS
        push!(entries, relative_path(paths[name]) => sha256_file(paths[name]))
    end
    for mode in ("train", "resume")
        dir = mode == "train" ? paths.train_dir : paths.resume_dir
        for stem in MODE_STEMS
            file = joinpath(dir, "$(stem)-$(mode).tsv")
            push!(entries, relative_path(file) => sha256_file(file))
        end
    end
    rows = [[first(entry), last(entry)[1:16] * "..."] for entry in entries]
    return md_table(["artifact", "SHA-256"], rows)
end

# ---------------------------------------------------------------------------
# Data-derived summaries
# ---------------------------------------------------------------------------

function validation_rows(path, model, task)
    table = read_table(path; required = ["step", "model", "task", "loss", "accuracy", "exposure"])
    return sort(
        [(parse_int_field(table.rows, index, "step"),
          parse_float_field(table.rows, index, "loss"),
          parse_float_field(table.rows, index, "accuracy"),
          parse_int_field(table.rows, index, "exposure"))
         for index in eachindex(table.rows)
         if table.rows[index]["model"] == model && table.rows[index]["task"] == task];
        by = first)
end

function validation_summary(data::BundleData)
    paths = data.paths
    rows = Vector{Vector{Any}}()
    for mode in ("train", "resume")
        dir = mode == "train" ? paths.train_dir : paths.resume_dir
        file = joinpath(dir, "validation-$(mode).tsv")
        for model in ("ddp", "reference"), task in ("a", "b")
            series = validation_rows(file, model, task)
            initial = series[1]
            final = series[end]
            push!(rows, [mode, model, uppercase(task), initial[1],
                @sprintf("%.4f", initial[2]), @sprintf("%.4f", initial[3]),
                final[1], @sprintf("%.4f", final[2]), @sprintf("%.4f", final[3]),
                final[4]])
        end
    end
    table = md_table(["mode", "model", "task", "first step", "first loss", "first acc",
        "final step", "final loss", "final acc", "final exposure"], rows)
    return (; rows, table)
end

function max_reference_difference(data::BundleData)
    paths = data.paths
    result = Dict{Tuple{String,String},NamedTuple}()
    for mode in ("train", "resume")
        dir = mode == "train" ? paths.train_dir : paths.resume_dir
        file = joinpath(dir, "validation-$(mode).tsv")
        for task in ("a", "b")
            ddp = Dict(step => (loss, acc) for (step, loss, acc, _) in
                validation_rows(file, "ddp", task))
            ref = Dict(step => (loss, acc) for (step, loss, acc, _) in
                validation_rows(file, "reference", task))
            steps = sort(collect(intersect(keys(ddp), keys(ref))))
            isempty(steps) && continue
            max_loss = maximum(abs(ddp[s][1] - ref[s][1]) for s in steps)
            max_acc = maximum(abs(ddp[s][2] - ref[s][2]) for s in steps)
            result[(task, mode)] = (; n = length(steps), max_loss, max_acc)
        end
    end
    return result
end

function discrepancy_summary(data::BundleData)
    paths = data.paths
    maxima = (param = 0.0, mom1 = 0.0, mom2 = 0.0)
    beta_false = String[]
    for mode in ("train", "resume")
        dir = mode == "train" ? paths.train_dir : paths.resume_dir
        table = read_table(joinpath(dir, "metrics-$(mode).tsv");
            required = ["step", "peer_param_maxerr", "peer_mom1_maxerr",
                "peer_mom2_maxerr", "peer_beta_exact"])
        for index in eachindex(table.rows)
            step = parse_int_field(table.rows, index, "step")
            maxima = (param = max(maxima.param, parse_float_field(table.rows, index, "peer_param_maxerr")),
                mom1 = max(maxima.mom1, parse_float_field(table.rows, index, "peer_mom1_maxerr")),
                mom2 = max(maxima.mom2, parse_float_field(table.rows, index, "peer_mom2_maxerr")))
            table.rows[index]["peer_beta_exact"] == "true" ||
                push!(beta_false, "$(mode) step $(step)")
        end
    end
    return (; maxima, beta_false, n_false = length(beta_false))
end

function learning_rate_summary(data::BundleData)
    paths = data.paths
    etas = Set{Float64}()
    leaves = Set{String}()
    ddp_ranks = Set{Int}()
    reference_records = 0
    ddp_records = 0
    for mode in ("train", "resume")
        dir = mode == "train" ? paths.train_dir : paths.resume_dir
        table = read_table(joinpath(dir, "learning-rate-$(mode).tsv");
            required = ["step", "model", "rank", "leaf", "eta"])
        for index in eachindex(table.rows)
            push!(etas, parse_float_field(table.rows, index, "eta"))
            push!(leaves, table.rows[index]["leaf"])
            if table.rows[index]["model"] == "ddp"
                push!(ddp_ranks, parse_int_field(table.rows, index, "rank"))
                ddp_records += 1
            else
                reference_records += 1
            end
        end
    end
    return (; etas = sort(collect(etas)), leaves = sort(collect(leaves)),
        ranks = sort(collect(ddp_ranks)), ddp_records, reference_records)
end

function expected_collectives(record, protocol)
    present = (record.present_a ? 1 : 0) + (record.present_b ? 1 : 0)
    present in (1, 2) || error("step $(record.step): $(present) present tasks")
    # `base` is presence + 2 trunk gradients, plus the guarded protocol's
    # trailing host gradient-validity vote; 2 per globally present head.
    return protocol.base + 2 * present
end

function collective_summary(data::BundleData)
    protocol = collective_protocol(data.bundle)
    rows = Vector{Vector{Any}}()
    checks = String[]
    observed = Dict{Tuple{String,Int,Int},Int}()
    for entry in data.collectives
        key = (entry.mode, entry.step, entry.rank)
        observed[key] = get(observed, key, 0) + 1
    end
    for record in data.steps
        expected = expected_collectives(record, protocol)
        seen = get(observed, (record.mode, record.step, record.rank), 0)
        expected == record.collectives == seen || error(
            "$(record.mode) step $(record.step) rank $(record.rank): expected $(expected) " *
            "collectives, metrics record $(record.collectives), console $(seen)")
        # The guarded protocol ends every trace with the host gradient-validity
        # vote; the legacy protocol never emits it (the exact count rejects it).
        if protocol.vote_required
            entries = [entry for entry in data.collectives
                       if entry.mode == record.mode && entry.step == record.step &&
                          entry.rank == record.rank]
            vote = entries[findmax(entry -> entry.order, entries)[2]]
            vote.buftype == "Vector{Float32}" && shape_dims(vote.shape) == [1] && vote.avg &&
                vote.local_flags === nothing && vote.global_flags === nothing || error(
                "$(record.mode) step $(record.step) rank $(record.rank): trailing collective " *
                "is $(vote.buftype) shape ($(vote.shape)), expected the trailing host " *
                "Float32 length-1 avg gradient-validity vote")
        end
    end
    for mode in ("train", "resume")
        counts = Dict{Int,Int}()
        for record in mode_rows(data, mode)
            counts[record.collectives] = get(counts, record.collectives, 0) + 1
        end
        for count in sort(collect(keys(counts)))
            label = count == protocol.base + 4 ? "both tasks present" :
                count == protocol.base + 2 ? "one task globally absent" :
                error("$(mode): collective count $(count) is outside the " *
                      "$(protocol.name) protocol")
            push!(rows, [mode, count, counts[count], label])
        end
    end
    # The first collective of every step is the CPU presence reduction.
    for entry in data.collectives
        entry.order != 1 && continue
        entry.buftype == "Vector{Float32}" || error(
            "first collective is $(entry.buftype), expected the CPU presence Vector{Float32}")
        entry.local_flags !== nothing && entry.global_flags !== nothing || error(
            "presence reduction at step $(entry.step) lacks local/global flags")
        entry.avg || error("presence reduction at step $(entry.step) is not an averaging reduction")
    end
    # Every (mode, step, rank) trace is a contiguous order sequence starting at 1.
    for ((mode, step, rank), count) in observed
        orders = sort([entry.order for entry in data.collectives
                       if entry.mode == mode && entry.step == step && entry.rank == rank])
        orders == collect(1:count) || error(
            "$(mode) step $(step) rank $(rank): collective orders $(orders) are not 1:$(count)")
    end
    # The global flags must equal the rank mean of the local flags.
    flag_mismatches = 0
    for mode in ("train", "resume"), step in unique(record.step for record in data.steps if record.mode == mode)
        entries = [entry for entry in mode_collectives(data, mode)
                   if entry.step == step && entry.order == 1]
        isempty(entries) && continue
        locals = [entry.local_flags for entry in entries]
        all(length(flags) == 6 for flags in locals) || error(
            "$(mode) step $(step): presence flag length $(length(first(locals))), expected 6")
        expected_global = reduce(+, locals) ./ length(locals)
        for entry in entries
            isapprox(entry.global_flags, expected_global; atol = 1.0e-6, rtol = 0.0) ||
                (flag_mismatches += 1)
        end
    end
    flag_mismatches == 0 || error(
        "$(flag_mismatches) presence-reduction flag averages disagree with the console")
    push!(checks, "presence reduction is one CPU Vector{Float32} average, first in every step")
    push!(checks, "order 1 precedes every gradient reduction (orders 2..n are contiguous)")
    push!(checks, "observed collectives per rank equal the $(protocol.name) protocol on every step")
    protocol.vote_required && push!(checks,
        "the final collective of every step is the host Float32 length-1 avg gradient vote")
    push!(checks, "global presence flags equal the rank mean of the local flags")
    if all(endswith(entry.buftype, "CuArray{Float32, 2, CUDACore.DeviceMemory}") ||
           endswith(entry.buftype, "CuArray{Float32, 1, CUDACore.DeviceMemory}") ||
           entry.buftype == "Vector{Float32}"
           for entry in data.collectives)
        push!(checks, "every gradient reduction uses a CUDA buffer (NCCL transport)")
    end
    return (; rows, checks)
end

function representative_traces(data::BundleData; steps = (1, 3), rank = 0)
    tables = Dict{Int,Any}()
    for (mode, step) in (("train", steps[1]), ("train", steps[2]))
        entries = sort([entry for entry in data.collectives
                        if entry.mode == mode && entry.step == step && entry.rank == rank],
            by = entry -> entry.order)
        rows = [[entry.order, entry.buftype, "(" * entry.shape * ")",
            entry.avg ? "avg" : "sum",
            entry.local_flags === nothing ? "--" : string(entry.local_flags),
            entry.global_flags === nothing ? "--" : string(entry.global_flags)]
            for entry in entries]
        tables[step] = md_table(["order", "type", "shape", "op", "local flags", "global flags"], rows)
    end
    return tables
end

function continuation_summary(data::BundleData)
    bundle = data.bundle
    paths = data.paths
    rows = Vector{Vector{Any}}()
    resume_records = mode_rows(data, "resume")
    n_match = count(record -> record.resume_match === true, resume_records)
    n_total = length(resume_records)
    expected = bundle.ranks * (bundle.steps - bundle.checkpoint_step)
    n_total == expected || error(
        "resume step records $(n_total) != ranks $(bundle.ranks) x steps after checkpoint $(expected)")
    push!(rows, ["resume markers (console)", n_match, n_total,
        n_match == n_total ? "all matched" : "MISMATCH"])
    # Metric intersections: validation DDP rows at steps after the checkpoint.
    for task in ("a", "b")
        train = Dict(step => (loss, acc) for (step, loss, acc, _) in
            validation_rows(joinpath(paths.train_dir, "validation-train.tsv"), "ddp", task)
            if step > bundle.checkpoint_step)
        resume = Dict(step => (loss, acc) for (step, loss, acc, _) in
            validation_rows(joinpath(paths.resume_dir, "validation-resume.tsv"), "ddp", task)
            if step > bundle.checkpoint_step)
        steps = sort(collect(intersect(keys(train), keys(resume))))
        exact = count(step -> train[step] == resume[step], steps)
        max_loss = maximum(abs(train[step][1] - resume[step][1]) for step in steps)
        max_acc = maximum(abs(train[step][2] - resume[step][2]) for step in steps)
        push!(rows, ["validation task $(uppercase(task))", exact, length(steps),
            @sprintf("max |diff| loss %.3e, acc %.3e", max_loss, max_acc)])
    end
    return (; rows, n_match, n_total, expected)
end

function exposure_summary(data::BundleData)
    bundle = data.bundle
    paths = data.paths
    schedule = read_table(paths.schedule; required = ["step", "rank", "task", "pool_indices"])
    from_schedule = Dict("a" => 0, "b" => 0)
    for index in eachindex(schedule.rows)
        task = schedule.rows[index]["task"]
        n = length(split(schedule.rows[index]["pool_indices"], ','))
        from_schedule[task] = get(from_schedule, task, 0) + n
    end
    # The prepared schedule lists every sample identity once, so its row count
    # per task is the exposure.
    rows = Vector{Vector{Any}}()
    for task in ("a", "b")
        table = read_table(joinpath(paths.train_dir, "validation-train.tsv");
            required = ["step", "model", "task", "exposure"])
        exposures = [parse_int_field(table.rows, index, "exposure")
                     for index in eachindex(table.rows)
                     if table.rows[index]["model"] == "ddp" &&
                        table.rows[index]["task"] == task]
        final = maximum(exposures)
        final == from_schedule[task] || error(
            "task $(task): final exposure $(final) != prepared schedule rows $(from_schedule[task])")
        mixed = [record for record in mode_rows(data, "train")
                 if record.present_a && record.present_b]
        weights = sort(unique(["$(record.counts_a)/$(bundle.ranks), $(record.counts_b)/$(bundle.ranks)"
                               for record in mixed]))
        push!(rows, [uppercase(task), final, join(weights, "; "),
            "samples = $(final) (global batch 128 x $(bundle.steps) steps = $(128 * bundle.steps))"])
    end
    total = sum(values(from_schedule))
    total == 128 * bundle.steps || error(
        "schedule exposure $(total) != global batch x steps $(128 * bundle.steps)")
    return (; rows, from_schedule, total)
end

function schedule_cross_check(data::BundleData)
    bundle = data.bundle
    table = read_table(data.paths.schedule; required = ["step", "rank", "task", "head"])
    canonical(phase, rank) = phase == 3 ? "a" :
        phase == 4 ? "b" :
        phase == 6 ? (rank == 0 ? "b" : "a") :
        (rank == 0 ? "a" : "b")
    phase_counts = Dict{Int,Dict{String,Int}}()
    for index in eachindex(table.rows)
        step = parse_int_field(table.rows, index, "step")
        rank = parse_int_field(table.rows, index, "rank")
        task = table.rows[index]["task"]
        table.rows[index]["head"] == task || error(
            "schedule step $(step) rank $(rank): task $(task) != head $(table.rows[index]["head"])")
        phase = mod1(step, 6)
        task == canonical(phase, rank) || error(
            "schedule step $(step) rank $(rank): task $(task), expected the six-step rule $(canonical(phase, rank))")
        counts = get!(phase_counts, phase, Dict("a" => 0, "b" => 0))
        counts[task] += 1
    end
    rows = Vector{Vector{Any}}()
    for phase in 1:6
        counts = phase_counts[phase]
        description = phase == 3 ? "all task A" :
            phase == 4 ? "all task B" :
            phase == 6 ? "rank 0 task B, others task A" :
            "rank 0 task A, others task B"
        push!(rows, [phase, counts["a"], counts["b"], description])
    end
    return md_table(["cycle phase", "task A ranks", "task B ranks", "routing rule"], rows)
end

# ---------------------------------------------------------------------------
# Python plotting through the pinned uv environment
# ---------------------------------------------------------------------------

function plotter_environment_command()
    return "uv run --offline --with-requirements $(PLOT_REQUIREMENTS) python3 <plotter> ..."
end

function run_plotter(script, arguments; workdir = REPO_ROOT)
    cmd = Cmd(vcat(["uv", "run", "--offline", "--with-requirements",
        PLOT_REQUIREMENTS, "python3", script], String[String(arg) for arg in arguments]))
    result = run_command(Cmd(cmd; dir = workdir))
    result.exitcode == 0 || error(
        "plotter $(script) failed with exit code $(result.exitcode)\n" *
        "command: $(result.cmd)\n" *
        join(last(split(result.stderr, "\n"), 20), "\n"))
    return result
end

function demo_figures(out_dir)
    prefixes = Set{String}()
    for file in readdir(out_dir)
        endswith(file, "-schedule.png") || continue
        push!(prefixes, file[1:(end - length("-schedule.png"))])
    end
    length(prefixes) == 1 || error(
        "expected exactly one figure prefix under $(out_dir), found $(sort(collect(prefixes)))")
    prefix = only(prefixes)
    figure(name) = joinpath(out_dir, "$(prefix)-$(name).png")
    return (
        prefix = prefix,
        directory = out_dir,
        validation_loss = figure("validation-loss"),
        validation_accuracy = figure("validation-accuracy"),
        validation_difference = figure("validation-difference"),
        learning_rate = figure("learning-rate"),
        discrepancy = figure("discrepancy"),
        exposure = figure("exposure"),
        collectives = figure("collectives"),
        continuation = figure("continuation"),
        schedule = figure("schedule"),
        predictions_a = figure("predictions-task-a"),
        predictions_b = figure("predictions-task-b"),
    )
end

function render_demo_figures(bundle; out_dir = joinpath(REPO_ROOT, "notebook", "figures",
        "conditional_graph_distributed_workflow", bundle.key))
    mkpath(out_dir)
    run_plotter("scripts/plots/plot_phase_k_demo.py",
        ["--run", bundle.run_root, "--out", relative_path(out_dir),
            "--title", "Phase K tutorial: $(bundle.label)"])
    return demo_figures(out_dir)
end

function render_network_figures(bundle; out_dir = joinpath(REPO_ROOT, "notebook", "figures",
        "conditional_graph_distributed_workflow", bundle.key, "network"))
    mkpath(out_dir)
    run_plotter("scripts/plots/plot_phase_k_network.py",
        ["--out", relative_path(out_dir), "--worlds", string(bundle.ranks)])
    return (
        network = joinpath(out_dir, "phase-k-k2-network.png"),
        routing = joinpath(out_dir, "phase-k-k2-routing.png"),
    )
end

# ---------------------------------------------------------------------------
# Reproduction commands and child-process runs
# ---------------------------------------------------------------------------

function extract_backtick_command(path)
    isfile(path) || error("missing command record $(relative_path(path))")
    text = read(path, String)
    m = match(r"setenv\(`(.+?)`,", text)
    m === nothing && return strip(text)
    return strip(m.captures[1])
end

function campaign_commands(bundle)
    paths = bundle_artifacts(bundle)
    return (
        train = extract_backtick_command(paths.train_command),
        resume = extract_backtick_command(paths.resume_command),
    )
end

function local_reproduction_command(bundle; backend = "mpi", ranks = "2,4",
        root = joinpath(PHASE_K_ROOT, "notebook", "k2-notebook-<timestamp>"))
    return "julia --startup-file=no --project=$(joinpath("..", "flux-integration", "test")) " *
           "scripts/checks/conditional_graph_demo_check.jl" *
           " --root $(root) --dataset mnist-fashion --backend $(backend)" *
           " --ranks $(ranks) --steps 120" *
           " --prepared $(joinpath(bundle.prepared, "mnist-fashion_prepared.jls"))" *
           " --timeout 1800 --label k2-notebook"
end

reproduction_command(bundle; backend = "mpi", ranks = "2,4") =
    local_reproduction_command(bundle; backend = backend, ranks = ranks)

function run_child_command(cmd::Cmd; label = "child")
    started = time()
    result = run_command(cmd)
    elapsed = round(time() - started; digits = 1)
    result.exitcode == 0 || error(
        "$(label) failed with exit code $(result.exitcode) after $(elapsed) s\n" *
        "command: $(result.cmd)\n" *
        join(last(split(result.stderr, "\n"), 30), "\n"))
    return (; result, elapsed)
end

function run_training_child(bundle; backend = "mpi", ranks = 2,
        timeout = 1800, label = "k2-notebook")
    stamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    root = joinpath(REPO_ROOT, PHASE_K_ROOT, "notebook", "$(bundle.key)-$(stamp)")
    ispath(root) && error("refusing existing run root $(root)")
    prepared = joinpath(REPO_ROOT, bundle.prepared, "mnist-fashion_prepared.jls")
    isfile(prepared) || error("prepared artifact $(prepared) is not present")
    driver = joinpath(REPO_ROOT, "scripts", "checks", "conditional_graph_demo_check.jl")
    cmd = `julia --startup-file=no --project=$(FLUX_TEST_PROJECT) $(driver) --root $(root) --dataset mnist-fashion --backend $(backend) --ranks $(ranks) --steps 120 --prepared $(prepared) --timeout $(timeout) --label $(label)`
    return run_child_command(cmd; label = "Phase K driver $(bundle.key)")
end

end # module
