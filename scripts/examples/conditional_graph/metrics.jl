# Rank-zero presentation output, plus strict production schema validation.
module CGMetrics

const VALIDATION_COLUMNS = split("run_id mode schedule_digest dataset_digest step model task n loss accuracy exposure present checkpoint_step")
const RATE_COLUMNS = split("run_id mode schedule_digest step model rank leaf eta")
const PREDICTION_COLUMNS = split("run_id mode schedule_digest step model task sample_index target predicted pixels")
const DISCREPANCY_COLUMNS = split("step cycle present_a present_b counts_a counts_b loss_ref loss_local_rank0 peer_param_maxerr peer_mom1_maxerr peer_mom2_maxerr peer_beta_exact val_a_loss val_a_acc val_b_loss val_b_acc")
const LEAVES = ("trunk_weight", "trunk_bias", "head_a_weight", "head_a_bias", "head_b_weight", "head_b_bias")

function init_files(out, mode)
    for (stem, columns) in (("validation", VALIDATION_COLUMNS),
            ("learning-rate", RATE_COLUMNS), ("predictions", PREDICTION_COLUMNS))
        path = joinpath(out, "$stem-$mode.tsv")
        isfile(path) && error("refusing to overwrite $path")
        open(path, "w") do io
            println(io, join(columns, '\t'))
        end
    end
end

function append_row(out, stem, mode, row)
    open(joinpath(out, "$stem-$mode.tsv"), "a") do io
        println(io, join(row, '\t'))
    end
end

function read_rows(path, columns)
    lines = readlines(path)
    isempty(lines) && error("empty TSV: $path")
    split(lines[1], '\t') == columns || error("incorrect schema: $path")
    rows = [split(l, '\t'; keepempty=true) for l in lines[2:end]]
    all(r -> length(r) == length(columns), rows) || error("invalid row width: $path")
    return [Dict(zip(columns, row)) for row in rows]
end

number(r, key) = begin
    v = parse(Float64, r[key])
    isfinite(v) || error("non-finite $key")
    v
end

function validate_outputs(out, mode, first_step, last_step, R, sched, dataset;
        run_id, schedule_digest, eta, checkpoint_step=2)
    # This validator is called by the producer and the bounded parent driver.
    identity(r) = begin
        r["run_id"] == run_id || error("run identity mismatch")
        r["mode"] == mode || error("mode mismatch")
        r["schedule_digest"] == schedule_digest || error("schedule identity mismatch")
    end
    rows = read_rows(joinpath(out, "validation-$mode.tsv"), VALIDATION_COLUMNS)
    expected = Set((s, m, t) for s in (first_step-1):last_step for m in ("ddp", "reference") for t in ("a", "b"))
    seen = Set{Tuple{Int,String,String}}()
    for r in rows
        identity(r)
        r["dataset_digest"] == dataset.digest || error("dataset identity mismatch")
        s = parse(Int, r["step"]); t = r["task"]
        key = (s, r["model"], t)
        key in expected && !(key in seen) || error("unexpected or duplicate validation key $key")
        push!(seen, key)
        ti = t == "a" ? 1 : 2
        parse(Int, r["n"]) == size(dataset.pools[Symbol(t)].val.y, 2) || error("validation count mismatch")
        parse(Int, r["exposure"]) == sum(sched.task_requests[1:s, ti]) || error("exposure mismatch")
        r["present"] == string(s == 0 || sched.task_requests[s, ti] > 0) || error("presence mismatch")
        parse(Int, r["checkpoint_step"]) == checkpoint_step || error("checkpoint marker mismatch")
        number(r, "loss") >= 0 || error("negative loss")
        0 <= number(r, "accuracy") <= 1 || error("invalid accuracy")
    end
    seen == expected || error("missing validation rows")
    rows = read_rows(joinpath(out, "learning-rate-$mode.tsv"), RATE_COLUMNS)
    expected_rates = Set((s, m, r, l) for s in first_step:last_step for m in ("ddp", "reference")
        for r in (m == "ddp" ? (0:R-1) : (0:0)) for l in LEAVES)
    seen_rates = Set{Tuple{Int,String,Int,String}}()
    for r in rows
        identity(r)
        key = (parse(Int, r["step"]), r["model"], parse(Int, r["rank"]), r["leaf"])
        key in expected_rates && !(key in seen_rates) || error("unexpected or duplicate eta key $key")
        push!(seen_rates, key)
        number(r, "eta") == eta || error("configured eta mismatch")
    end
    seen_rates == expected_rates || error("missing learning-rate rows")
    rows = read_rows(joinpath(out, "predictions-$mode.tsv"), PREDICTION_COLUMNS)
    expected_predictions = Set((s, m, string(t),
        hasproperty(dataset.pools[t].val, :indices) ? dataset.pools[t].val.indices[i] : i)
        for s in (first_step-1, last_step) for m in ("ddp", "reference") for t in (:a, :b)
        for i in 1:min(20, size(dataset.pools[t].val.x, 2)))
    seen_predictions = Set{Tuple{Int,String,String,Int}}()
    for r in rows
        identity(r)
        key = (parse(Int, r["step"]), r["model"], r["task"], parse(Int, r["sample_index"]))
        key in expected_predictions && !(key in seen_predictions) || error("invalid prediction key $key")
        push!(seen_predictions, key)
        for col in ("target", "predicted")
            0 <= parse(Int, r[col]) < dataset.nclasses || error("invalid class")
        end
        pool = dataset.pools[Symbol(r["task"])].val
        i = hasproperty(pool, :indices) ? findfirst(==(key[4]), pool.indices) : key[4]
        parse(Int, r["target"]) == argmax(pool.y[:, i])-1 || error("prediction target mismatch")
        pixels = parse.(Float64, split(r["pixels"], ','))
        pixels == Float64.(pool.x[:, i]) || error("prediction pixels mismatch")
    end
    seen_predictions == expected_predictions || error("missing prediction rows")
    rows = read_rows(joinpath(out, "metrics-$mode.tsv"), DISCREPANCY_COLUMNS)
    seen_steps = Set{Int}()
    for r in rows
        s = parse(Int, r["step"])
        s in first_step:last_step && !(s in seen_steps) || error("invalid discrepancy step")
        push!(seen_steps, s)
        parse(Int, r["cycle"]) == mod1(s, 6) || error("cycle mismatch")
        r["peer_beta_exact"] == "true" || error("scalar Adam state differs")
        for (ti, task) in enumerate(("a", "b"))
            parse(Int, r["counts_$task"]) == sched.task_requests[s, ti] ÷ sched.local_batch || error("rank count mismatch")
            r["present_$task"] == string(sched.task_requests[s, ti] > 0) || error("presence mismatch")
        end
        for key in DISCREPANCY_COLUMNS[7:11]
            number(r, key)
        end
    end
    seen_steps == Set(first_step:last_step) || error("missing discrepancy steps")
    return true
end

end
