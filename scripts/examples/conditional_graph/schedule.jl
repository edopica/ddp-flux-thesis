# Explicit global sample schedule for the Phase K demo.
#
# The schedule is data saved before training: for every (step, rank) it records
# the selected head and the sample-pool indices. It is built deterministically
# from saved permutations and cursors, so every rank and the independent
# reference consume exactly the same identity `(task, pool_index)` sequence.
#
# Repeating six-step routing cycle (from the Phase K prompt):
#   step 1: rank 0 -> A, others -> B   (local absence / one-versus-rest)
#   step 2: rank 0 -> A, others -> B   (nonzero Adam history on both heads)
#   step 3: all -> A                   (B globally absent)
#   step 4: all -> B                   (A globally absent)
#   step 5: rank 0 -> A, others -> B   (reactivation after absence)
#   step 6: rank 0 -> B, others -> A   (reversed participation)
module CGSchedule

using Random, SHA, Serialization

export Schedule, six_step_head, build_schedule, validate_schedule,
    schedule_present, schedule_digest, cycle_step, schedule_summary

struct Schedule
    steps::Int
    workers::Int
    local_batch::Int
    global_batch::Int
    heads::Matrix{Symbol}
    samples::Matrix{Vector{Int}}
    task_requests::Matrix{Int}          # steps × (a, b), samples requested
    exposures::NamedTuple{(:a, :b),Tuple{Int,Int}}
    completed_passes::NamedTuple{(:a, :b),Tuple{Int,Int}}
    partial_samples::NamedTuple{(:a, :b),Tuple{Int,Int}}
    reused_samples::NamedTuple{(:a, :b),Tuple{Int,Int}}
    pool_sizes::NamedTuple{(:a, :b),Tuple{Int,Int}}
    cursor_end::NamedTuple{(:a, :b),Tuple{Int,Int}}
    permutation_a::Vector{Int}
    permutation_b::Vector{Int}
    order_seed::UInt64
end

cycle_step(step::Int) = mod1(step, 6)

function six_step_head(cycle::Int, rank::Int)
    cycle == 3 && return :a
    cycle == 4 && return :b
    cycle == 6 && return rank == 0 ? :b : :a
    return rank == 0 ? :a : :b
end

function build_schedule(; steps::Int, workers::Int, local_batch::Int,
        pool_sizes::NamedTuple, order_seed::Integer)
    workers >= 2 || throw(ArgumentError("workers must be >= 2, got $workers"))
    local_batch >= 1 || throw(ArgumentError("local_batch must be >= 1"))
    global_batch = workers * local_batch

    rng = MersenneTwister(order_seed)
    permutation_a = randperm(rng, pool_sizes.a)
    permutation_b = randperm(rng, pool_sizes.b)
    perms = (; a = permutation_a, b = permutation_b)

    heads = Matrix{Symbol}(undef, steps, workers)
    samples = Matrix{Vector{Int}}(undef, steps, workers)
    task_requests = zeros(Int, steps, 2)
    cursors = Dict(:a => 0, :b => 0)

    for step in 1:steps
        for rank in 0:(workers - 1)
            heads[step, rank + 1] = six_step_head(cycle_step(step), rank)
        end
        for (ti, task) in enumerate((:a, :b))
            assigned = [r for r in 1:workers if heads[step, r] == task]
            request = length(assigned) * local_batch
            task_requests[step, ti] = request
            isempty(assigned) && continue
            perm = perms[task]
            n = getproperty(pool_sizes, task)
            start = cursors[task]
            # Cyclic contiguous slice of the saved permutation. The request is
            # always smaller than the pool, so every sample identity inside one
            # global batch is unique, including at pool boundaries.
            idxs = [perm[mod1(start + i, n)] for i in 1:request]
            cursors[task] = start + request
            for (j, rank) in enumerate(assigned)
                samples[step, rank] = idxs[((j - 1) * local_batch + 1):(j * local_batch)]
            end
        end
    end

    exposure_a = sum(task_requests[:, 1])
    exposure_b = sum(task_requests[:, 2])
    p = pool_sizes
    return Schedule(steps, workers, local_batch, global_batch, heads, samples,
        task_requests,
        (; a = exposure_a, b = exposure_b),
        (; a = div(exposure_a, p.a), b = div(exposure_b, p.b)),
        (; a = mod(exposure_a, p.a), b = mod(exposure_b, p.b)),
        (; a = max(exposure_a - p.a, 0), b = max(exposure_b - p.b, 0)),
        p,
        (; a = cursors[:a], b = cursors[:b]),
        permutation_a, permutation_b, UInt64(order_seed))
end

schedule_present(sched::Schedule, step::Int) = (;
    a = any(==(:a), view(sched.heads, step, :)),
    b = any(==(:b), view(sched.heads, step, :)))

function schedule_digest(sched::Schedule)
    io = IOBuffer()
    for field in (:steps, :workers, :local_batch, :global_batch, :heads, :samples,
        :task_requests, :exposures, :pool_sizes, :cursor_end,
        :permutation_a, :permutation_b, :order_seed)
        serialize(io, getfield(sched, field))
    end
    return bytes2hex(SHA.sha256(seekstart(io)))
end

function validate_schedule(sched::Schedule)
    sched.steps >= 6 || error("schedule needs a complete routing cycle")
    cursors = Dict(:a=>0, :b=>0)
    for (task, perm) in ((:a, sched.permutation_a), (:b, sched.permutation_b))
        sort(perm) == collect(1:sched.pool_sizes[task]) || error("invalid saved permutation")
    end
    sched.global_batch == sched.workers * sched.local_batch ||
        error("global batch $(sched.global_batch) != workers × local batch")
    for step in 1:sched.steps
        identity = Set{Tuple{Symbol,Int}}()
        for rank in 1:sched.workers
            head = sched.heads[step, rank]
            head == six_step_head(cycle_step(step), rank-1) || error("invalid routing cycle")
            head in (:a, :b) || error("step $step rank $rank invalid head $head")
            chosen = sched.samples[step, rank]
            length(chosen) == sched.local_batch ||
                error("step $step rank $rank local batch $(length(chosen)) != " *
                      "$(sched.local_batch)")
            for i in chosen
                1 <= i <= sched.pool_sizes[head] || error("sample outside task pool")
                perm = head == :a ? sched.permutation_a : sched.permutation_b
                cursors[head] += 1
                i == perm[mod1(cursors[head], length(perm))] || error("saved schedule violates task cursor")
                key = (head, i)
                key in identity && error("step $step global batch repeats identity $key")
                push!(identity, key)
            end
        end
        length(identity) == sched.global_batch ||
            error("step $step global batch has $(length(identity)) identities, " *
                  "expected $(sched.global_batch)")
        for (ti, task) in enumerate((:a, :b))
            want = count(==(task), view(sched.heads, step, :)) * sched.local_batch
            got = sched.task_requests[step, ti]
            want == got || error("step $step task $task request $got != $want")
        end
    end
    for (ti, task) in enumerate((:a, :b))
        n = sched.pool_sizes[task]
        exposure = sum(sched.task_requests[:, ti])
        exposure == sched.exposures[task] == sched.cursor_end[task] || error("exposure/cursor mismatch")
        sched.completed_passes[task] == div(exposure, n) || error("completed pass mismatch")
        sched.partial_samples[task] == mod(exposure, n) || error("partial pass mismatch")
        sched.reused_samples[task] == max(exposure-n, 0) || error("reuse mismatch")
    end
    return sched
end

function schedule_summary(sched::Schedule)
    tasks = map((:a, :b)) do task
        (; task,
            exposure = getproperty(sched.exposures, task),
            pool = getproperty(sched.pool_sizes, task),
            completed_passes = getproperty(sched.completed_passes, task),
            partial = getproperty(sched.partial_samples, task),
            reused = getproperty(sched.reused_samples, task))
    end
    return (; steps = sched.steps, workers = sched.workers,
        local_batch = sched.local_batch, global_batch = sched.global_batch, tasks)
end

end # module
