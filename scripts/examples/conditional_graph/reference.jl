# Independent native reference for the Phase K demo.
#
# The reference differentiates the scheduled global objective
#     L_global(model) = sum_r L_r(model) / R
# once, in one process, through plain Zygote and Optimisers. It never calls the
# distributed wrapper, `_sync_gradients`, or any distributed reduced gradient,
# and it never resets its state from the distributed replicas.
module CGReference

using Flux, Optimisers, Zygote

export local_loss, global_loss, global_gradient, reference_update!, reference_step!

function local_loss(model, x, y, head::Symbol)
    logits = model(x, head)
    return Flux.Losses.logitcrossentropy(logits, y)
end

# `batches` is an ordered collection of `(head, x, y)` shard tuples. Unused
# subtrees are never evaluated: a globally absent head stays absent from the
# reference AD graph, exactly like the native skip semantics.
function global_loss(model, batches)
    total = nothing
    for (head, x, y) in batches
        l = local_loss(model, x, y, head)
        total = total === nothing ? l : total + l
    end
    return total / length(batches)
end

function global_gradient(model, batches)
    loss, gs = Zygote.withgradient(m -> global_loss(m, batches), model)
    return loss, gs[1]
end

# One native Optimisers update per scheduled global step. The native tree is
# the reference's own tree, created from the same saved initial artifact.
function reference_update!(model, tree, batches)
    loss, grads = global_gradient(model, batches)
    tree, model = Optimisers.update!(tree, model, grads)
    return (; loss, grads, tree, model)
end

# Full parameter and native Adam-state view of one head, as host copies. It is
# used only to assert that a globally absent head stays untouched by an update.
function head_state(model, tree, head::Symbol)
    sub = head === :a ? :head_a : :head_b
    layer = getfield(model, sub)
    leaf_w = getfield(getfield(tree, sub), :weight)
    leaf_b = getfield(getfield(tree, sub), :bias)
    return (; w = Array(layer.weight), b = Array(layer.bias),
        mw = Array(leaf_w.state[1]), vw = Array(leaf_w.state[2]), bw = leaf_w.state[3],
        mb = Array(leaf_b.state[1]), vb = Array(leaf_b.state[2]), bb = leaf_b.state[3])
end

# Native scheduled step with the demo's globally-absent-head contract: an
# absent head must not appear in the AD result and must not change through the
# update. `present` is `CGSchedule.schedule_present(sched, step)`.
function reference_step!(model, tree, batches, present)
    absent = Dict{Symbol,Any}()
    for head in (:a, :b)
        getproperty(present, head) || (absent[head] = head_state(model, tree, head))
    end
    loss, grads = global_gradient(model, batches)
    for head in (:a, :b)
        sub = head === :a ? :head_a : :head_b
        g = getproperty(grads, sub)
        want = getproperty(present, head)
        want ? (g === nothing && error("native reference: globally present head " *
                                       "$head is absent from the AD result")) :
               (g !== nothing && error("native reference: globally absent head " *
                                       "$head is present in the AD result"))
    end
    tree, model = Optimisers.update!(tree, model, grads)
    for (head, before) in absent
        head_state(model, tree, head) == before ||
            error("native reference: globally absent head $head changed during the update")
    end
    return (; loss, grads, tree, model)
end

end # module
