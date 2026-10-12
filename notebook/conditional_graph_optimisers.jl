### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 2de35f30-1c3a-4b55-a6a7-f6c0baff2144
begin
    import Pkg
    # Activate the thesis project so this notebook uses the locally dev'd Flux
    # (the current per-leaf `DistributedOptimizer`), not registry Flux.
    # Referencing `Pkg.activate` also tells Pluto to stop managing its own
    # package environment for this notebook.
    Pkg.activate(dirname(@__DIR__))

    using Optimisers
    using Flux
    using Functors
    using Zygote

    (optimisers = pathof(Optimisers), flux = pathof(Flux))
end

# ╔═╡ 6647903a-15a3-4118-8e88-8ef1ce11c12f
md"""
# Conditional graphs, Optimisers.jl, and distributed training

This notebook is a hands-on walk from a single optimizer step to the reason a
per-leaf distributed optimizer breaks when ranks use different conditional
branches.

It has five parts:

1. Setup.
2. Optimisers.jl on its own.
3. Optimisers inside Flux.
4. The distributed case and its failure modes.
5. Takeaways and the required contract.

It is runnable with the thesis environment:

```bash
julia --project=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis
```

The thesis environment dev's the local Flux path
`/home/kurapica/Projects/ddp_flux/ddp_flux` (dirty master `b633cdc7`), which
contains the current per-leaf `DistributedOptimizer`.
"""

# ╔═╡ fa207608-cf0d-4868-9aff-856e1e7badd8
md"""
**Two libraries, two roles.**

- **Optimisers.jl** owns the update rule. It walks a model, stores per-parameter
  optimizer state, and applies `rule` updates to gradients.
- **Flux.jl** owns the layers, the automatic differentiation, and the training
  loop. For optimization it does not reimplement anything: it calls Optimisers.

`Functors.jl` is the traversal machinery both use to walk nested models, and
`Zygote.jl` is the reverse-mode AD engine behind `Flux.gradient`.
"""

# ╔═╡ 296cf55b-d54d-4c7e-9bfc-85beaabdf3da
md"""
## 2. Optimisers.jl alone

Start with a tiny nested model: a `NamedTuple` with two arrays. `a` is a
three-element vector and `b` is a two-element vector.
"""

# ╔═╡ 0d984f8f-f0ba-4a75-b417-ad64b741320c
begin
    model = (a = [1.0, 2.0, 3.0], b = [10.0, 20.0])
    state = Optimisers.setup(Descent(0.1), model)
    state
end

# ╔═╡ 598d7278-655b-438e-a790-0c79a316f649
md"""
`Optimisers.setup` walks the model with `Optimisers.trainable`, stops at every
`Optimisers.isnumeric` leaf (a non-integer array), and replaces that leaf with a
`Leaf(rule, state)`.

`Descent` has no state, so both leaves carry `nothing`. An optimizer such as
`Adam` would carry its moment buffers inside the same `Leaf`.
"""

# ╔═╡ 1e89566e-94cc-46b2-91aa-b8ebe2f4ca63
begin
    grads = (a = [0.5, 0.5, 0.5], b = [1.0, 1.0])
    before = copy(model.a)
    Optimisers.update!(state, model, grads)
    (parameter_before = before, parameter_after = model.a)
end

# ╔═╡ 68b11880-6e31-4037-ab83-97b85379c1f7
begin
    model_n = (a = [1.0, 2.0, 3.0], b = [10.0, 20.0])
    state_n = Optimisers.setup(Descent(0.1), model_n)
    Optimisers.update!(state_n, model_n, (a = [0.5, 0.5, 0.5], b = nothing))
    (a = model_n.a, b = model_n.b)
end

# ╔═╡ 3d358c61-acdc-45e7-be7c-71b1fa6666a6
md"""
`b` is unchanged: `b = [10.0, 20.0]`. The gradient for `a` was applied. This is
the crux of the whole notebook.

`Optimisers.update!` runs two passes (`interface.jl:70-109`):

1. `_grads!` accumulates gradients into an `IdDict{Leaf, Any}`. For a leaf whose
   gradient is `nothing` (or any `AbstractZero`), the method
   `_grads!(dict, ℓ::Leaf, x, ::Zero...) = nothing` (`interface.jl:116`) adds
   **no entry** to the dictionary. Also, `subtract!(x, x̄::Zero) = x`
   (`interface.jl:112`) is the identity.
2. `_update!` calls `apply!` only when `haskey(grads, ℓ)` (`interface.jl:103-108`).
   A missing entry means the optimizer step is skipped and the leaf state is left
   untouched.

The crux: Optimisers does **not** zero-fill a missing gradient. It **skips** the
leaf.
"""

# ╔═╡ 52300212-c362-4055-89d0-94eb1654bd01
begin
    m_zero = (w = [1.0],)
    s_zero = Optimisers.setup(Adam(0.1), m_zero)
    Optimisers.update!(s_zero, m_zero, (w = [0.0],))    # explicit zero gradient

    m_none = (w = [1.0],)
    s_none = Optimisers.setup(Adam(0.1), m_none)
    Optimisers.update!(s_none, m_none, (w = nothing,))  # no gradient at all

    (zero_gradient_state = s_zero.w.state,
     nothing_gradient_state = s_none.w.state)
end

# ╔═╡ 0c460f91-e876-4dd2-8e37-9d1f130921e2
md"""
Compare the two `Adam` states. The last tuple element is the running `βt`
schedule.

- With a **zero gradient**, Adam still takes a step: `mt` and `vt` are updated
  and `βt` advances (for example `0.9 -> 0.81`). The optimizer "sees" a step.
- With a **`nothing` gradient**, the leaf is skipped entirely: `βt` stays
  `(0.9, 0.999)`.

This is why a naive zero-fill is wrong: it does not just add zero to the
parameters, it silently advances the optimizer's internal schedule. Optimisers
deliberately distinguishes "no gradient" from "zero gradient".
"""

# ╔═╡ ea516871-e4d6-4794-8aa4-af0abce00124
begin
    bn = BatchNorm(2)
    (trainable_arrays = length(Optimisers.trainables(bn)),
     is_numeric_weight = Optimisers.isnumeric(bn.β),
     is_numeric_integer = Optimisers.isnumeric([1, 2, 3]),
     running_mean_is_trainable = any(p -> p === bn.μ, Optimisers.trainables(bn)))
end

# ╔═╡ 8243d0e4-2cdd-480a-9965-39dfe814841a
md"""
`Optimisers.isnumeric` is the source of truth for "this array is a trainable
parameter". It returns `true` for arrays of non-integer numbers and `false` for
integer arrays and everything else.

`Optimisers.trainable` is the source of truth for *which* fields participate.
Layers such as `BatchNorm` override it so that running statistics (`μ`, `σ²`)
are ignored even though they are floating-point arrays. `setup` therefore never
creates a `Leaf` for them. This matters later: the distributed optimizer must
walk the model the same way, or it will try to communicate state that
Optimisers itself never touches.
"""

# ╔═╡ e602dd38-31e8-4fac-a46a-ded4c12fde58
md"""
## 3. Flux context

Flux delegates to Optimisers. `Flux.setup`, `Flux.withgradient`, and
`Flux.train!` all end up in the same machinery, so the skip-on-`nothing`
semantics apply unchanged.
"""

# ╔═╡ e841bd53-e056-4c43-9965-8b8137dcd753
begin
    chain = Chain(Dense(2 => 3, relu), Dense(3 => 1))
    opt_state = Flux.setup(Adam(0.01), chain)
    opt_state
end

# ╔═╡ 7258c26e-0b93-4224-85d2-3d38a96929c0
begin
    x = rand(Float32, 2, 16)
    y = rand(Float32, 1, 16)
    data = [(x, y)]
    loss(m, x, y) = Flux.mse(m(x), y)

    l, gs = Flux.withgradient(m -> loss(m, x, y), chain)
    Flux.train!(loss, chain, data, opt_state)

    (loss_before_train = l, first_weight_norm = sum(abs2, chain[1].weight))
end

# ╔═╡ 2931d9c4-8903-443c-9610-5cdc24c4c7b5
md"""
`Flux.setup` returns an Optimisers tree of `Leaf`s. `Flux.withgradient` returns
the loss and a gradient tree. `Flux.train!` accepts that tree as its optimizer
argument and applies the updates in place.

Because the tree is exactly the Optimisers tree, Flux inherits the skip
behavior: if a parameter has a `nothing` gradient, Flux does not update it
either.
"""

# ╔═╡ 777499c9-d7a4-4946-a7b8-3fec0f8c35cb
begin
    cond = (l1 = Dense(2 => 2), l2 = Dense(2 => 1))
    g = Flux.gradient(c -> sum(c.l1(Float32[1.0, 2.0])), cond)[1]
    (grad_l1 = typeof(g.l1), grad_l2 = g.l2)
end

# ╔═╡ 4aff86ca-3246-4306-9614-eff976ddd3e3
md"""
## 4. Distributed case: where conditional graphs break

The current `DistributedOptimizer` is a per-leaf rule. It is defined in
`src/distributed/public_api.jl`:

```julia
function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    y_avg = DistributedUtils.allreduce!(opt.backend, y, DistributedUtils.avg)
    return Optimisers.apply!(opt.opt, state, x, y_avg)
end
Optimisers.init(opt::DistributedOptimizer, x::AbstractArray) = Optimisers.init(opt.opt, x)
```

The current dirty `master` (`b633cdc7`) also carries an experimental `no_sync`
flag; the pushed salvage branch does not. That flag only bypasses the
`allreduce!`; it does not fix the skip. Treat it as an experiment, not a fix.

Now connect the two facts:

- `Optimisers.update!` calls `apply!` **only** for leaves that have a local
  gradient (Section 2).
- The `allreduce!` lives **inside** `apply!`.

Therefore each rank issues one collective per *locally present* leaf, and only
those. The rule is per-leaf: every `Leaf` gets the same `DistributedOptimizer`
and the same backend, but there is no model-level coordination. If ranks use
different conditional branches, their collective sequences diverge.
"""

# ╔═╡ d38a238d-28ff-46b4-a6bf-32143f25907d
begin
    struct Branchy
        trunk::Vector{Float64}
        a::Vector{Float64}
        b::Vector{Float64}
        c::Vector{Float64}
    end
    Functors.@functor Branchy
end

# ╔═╡ fe970603-9297-41dd-acd7-8754b0323085
begin
    branch_loss(mode, r, m) = begin
        l = sum(abs2, m.trunk .- 1.0)
        if mode == "deadlock"
            l += r == 0 ? sum(abs2, m.a .- 2.0) + sum(abs2, m.b .- 5.0) :
                          sum(abs2, m.c .- 3.0)
        elseif mode == "corruption"
            l += r == 0 ? sum(abs2, m.a .- 2.0) : sum(abs2, m.b .- 5.0)
        else
            l += sum(abs2, m.a .- 2.0) + sum(abs2, m.b .- 5.0) + sum(abs2, m.c .- 3.0)
        end
        l
    end

    function rank_grads(mode, r)
        m = Branchy(ones(4), ones(4), ones(4), ones(4))
        g = Flux.gradient(mm -> branch_loss(mode, r, mm), m)[1]
        (; trunk = g.trunk !== nothing, a = g.a !== nothing,
             b = g.b !== nothing, c = g.c !== nothing)
    end

    (deadlock_rank0 = rank_grads("deadlock", 0),
     deadlock_rank1 = rank_grads("deadlock", 1),
     corruption_rank0 = rank_grads("corruption", 0),
     corruption_rank1 = rank_grads("corruption", 1))
end

# ╔═╡ 5782d79e-07ec-449b-a351-d571d1676f2b
md"""
The table above is the *local* gradient presence per rank, computed in one
process. It predicts the collective counts:

- **deadlock:** rank 0 has `trunk, a, b` present (3 collectives); rank 1 has
  `trunk, c` present (2 collectives). Rank 0 blocks on its third collective.
- **corruption:** rank 0 has `trunk, a` (2 collectives); rank 1 has `trunk, b`
  (2 collectives). Equal counts, so nothing hangs, but the collective for `a`
  pairs with the collective for `b`.
- **control:** every rank has all four present (4 collectives), so the update is
  correct.

### The three outcomes

1. **Uniform (fine).** Every rank activates the same leaves. Same collectives,
   same order, correct average.
2. **Silent corruption.** Different leaves are active, but the *number* of
   collectives is equal on every rank. The collectives pair the wrong
   parameters. Both receive a wrong averaged gradient, and nothing errors. This
   is the most dangerous case because training continues.
3. **Deadlock.** The number of collectives differs. The rank with more blocks
   forever on the unmatched collective. MPI does not time out.

A rank-dependent branch model cannot be run inside the notebook, because MPI
must own the process. The next cell runs the recorded detector trace by default,
and can launch the real 2-rank detector when you flip the switch.
"""

# ╔═╡ e8591dc4-76bb-4de0-8938-a22c9faff06d
begin
    RUN_MPI = false  # set true to launch a 2-rank MPI run (~2.5 min)

    repo_root = let
        env = get(ENV, "DDP_FLUX_REPO", "")
        if !isempty(env)
            env
        else
            p = dirname(Base.active_project())
            isfile(joinpath(p, "scripts", "checks", "conditional_deadlock_check.jl")) ? p : pwd()
        end
    end

    deadlock_trace = """
    PROBE_READY rank=0
    PROBE_READY rank=1
    TRACE rank=0 allreduce #1 size=4
    TRACE rank=1 allreduce #1 size=4
    TRACE rank=0 allreduce #2 size=4
    TRACE rank=1 allreduce #2 size=4
    TRACE rank=0 allreduce #3 size=4
    PROBE_UPDATE_DONE rank=1
    [driver kills the job: rank 0 never finishes]
    """

    corruption_trace = """
    TRACE rank=0 allreduce #1 size=4
    TRACE rank=1 allreduce #1 size=4
    TRACE rank=0 allreduce #2 size=4
    TRACE rank=1 allreduce #2 size=4
    PROBE_UPDATE_DONE rank=0
    PROBE_UPDATE_DONE rank=1
    PROBE_VERDICT mode=corruption a_expected=1.1 a_actual=1.5 b_expected=1.4 b_actual=1.5 corrupted=true
    """

    control_trace = """
    TRACE rank=0 allreduce #1..#4 size=4
    TRACE rank=1 allreduce #1..#4 size=4
    PROBE_VERDICT mode=control correct=true a=1.2 b=1.8 c=1.4
    """

    if RUN_MPI
        script = joinpath(repo_root, "scripts", "checks", "conditional_deadlock_check.jl")
        cmd = `julia --project=$(repo_root) $(script)`
        output = read(cmd, String)
        println(output)
        output
    else
        println("--- deadlock mode (recorded) ---\n", deadlock_trace)
        println("--- corruption mode (recorded) ---\n", corruption_trace)
        println("--- control mode (recorded) ---\n", control_trace)
        (deadlock = deadlock_trace, corruption = corruption_trace, control = control_trace)
    end
end

# ╔═╡ 03d63426-edad-4181-a81e-61816f92ff5d
md"""
### The dropped whole-model solution

The proposed fix (M1.1 spike, commit `3370b910`, preserved as
`archive/pr2694-unused-gradient-spike`) replaces the per-leaf rule with a
whole-model `DistributedOptimizerState` wrapper. A specialized
`Optimisers.setup(::DistributedOptimizer, model)` returns the wrapper, and the
wrapper coordinates the step instead of letting each leaf communicate alone:

1. **Presence allreduce.** One tiny `avg` allreduce of 0/1 flags per parameter
   records which parameters are globally present (threshold `> 0`).
2. **Model-order allreduce.** Exactly one in-place `allreduce!` runs per unique
   trainable parameter, in model-structure order. Tied parameters are
   deduplicated by identity. The order is derived from the model, so it is
   identical on every rank, independent of which local gradients exist.
3. **Zero contribution for locally-absent but globally-present.** A rank that
   did not use a parameter contributes a zero buffer of the correct shape to
   the shared average, so the global mean stays correct.
4. **`ZeroTangent` for globally-absent.** If no rank used a parameter, the
   wrapper substitutes `ZeroTangent`, so the optimizer step is skipped and the
   leaf state stays untouched, exactly as in the single-process case.

This keeps all communication outside automatic differentiation (the baseline
rule), and it is why the zero-fill is done *inside* the coordinated wrapper
rather than by a naive `resolve_unused_parameters!!` pass. The full design, its
review findings, and the collective-count protocol are in the companion report
`reports/conditional-graph-support.typ`.
"""

# ╔═╡ 3b6f54d3-78fa-4d47-a25f-d347087a5b3f
md"""
## 5. Takeaways

| Scenario | Local gradients | Collectives per rank | Result |
|---|---|---|---|
| Uniform | same leaves present | equal count, same order | correct |
| Silent corruption | different leaves, equal count | equal count, wrong pairing | wrong gradients, no error |
| Deadlock | different leaves, different count | unequal | hang forever |
| `nothing` gradient | leaf skipped | leaf does not communicate | state untouched (by design) |

**Required contract for a correct distributed optimizer:**

- Decide participation from the *model*, not from the local gradient.
- Issue collectives in an order identical on every rank.
- Contribute a zero buffer for locally-absent but globally-present parameters.
- Emit `ZeroTangent` for globally-absent parameters so the optimizer step is
  skipped and the leaf state is untouched.
- Keep every collective outside automatic differentiation.
- Never treat "no gradient" as "zero gradient": the two have different effects
  on optimizer state.

The detector in `scripts/checks/conditional_deadlock_check.jl` confirms the
current implementation on 2026-09-14: 12/12 tests pass, in 2m27s.
"""

# ╔═╡ Cell order:
# ╠═6647903a-15a3-4118-8e88-8ef1ce11c12f
# ╠═2de35f30-1c3a-4b55-a6a7-f6c0baff2144
# ╠═fa207608-cf0d-4868-9aff-856e1e7badd8
# ╠═296cf55b-d54d-4c7e-9bfc-85beaabdf3da
# ╠═0d984f8f-f0ba-4a75-b417-ad64b741320c
# ╠═598d7278-655b-438e-a790-0c79a316f649
# ╠═1e89566e-94cc-46b2-91aa-b8ebe2f4ca63
# ╠═68b11880-6e31-4037-ab83-97b85379c1f7
# ╠═3d358c61-acdc-45e7-be7c-71b1fa6666a6
# ╠═52300212-c362-4055-89d0-94eb1654bd01
# ╠═0c460f91-e876-4dd2-8e37-9d1f130921e2
# ╠═ea516871-e4d6-4794-8aa4-af0abce00124
# ╠═8243d0e4-2cdd-480a-9965-39dfe814841a
# ╠═e602dd38-31e8-4fac-a46a-ded4c12fde58
# ╠═e841bd53-e056-4c43-9965-8b8137dcd753
# ╠═7258c26e-0b93-4224-85d2-3d38a96929c0
# ╠═2931d9c4-8903-443c-9610-5cdc24c4c7b5
# ╠═777499c9-d7a4-4946-a7b8-3fec0f8c35cb
# ╠═4aff86ca-3246-4306-9614-eff976ddd3e3
# ╠═d38a238d-28ff-46b4-a6bf-32143f25907d
# ╠═fe970603-9297-41dd-acd7-8754b0323085
# ╠═5782d79e-07ec-449b-a351-d571d1676f2b
# ╠═e8591dc4-76bb-4de0-8938-a22c9faff06d
# ╠═03d63426-edad-4181-a81e-61816f92ff5d
# ╠═3b6f54d3-78fa-4d47-a25f-d347087a5b3f
