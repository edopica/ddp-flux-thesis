# Phase H characterization. Run through conditional_wrapper_compatibility_check.jl.
using Flux, Optimisers, Functors, MPI, Serialization, Test
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "../.."))) # JLD2 from thesis environment only
using JLD2
const DU = Flux.DistributedUtils
DU.initialize(Flux.MPIBackend; cuda_devices=missing, amdgpu_devices=missing)
const backend = DU.get_distributed_backend(Flux.MPIBackend)
const rank = DU.local_rank(backend)
const mode, directory = ARGS
const W = DU.DistributedOptimizerState
const rule = Optimisers.Adam(0.01)
function fixture()
    a = [1.0, 2.0]
    model = (a=a, b=a, frozen=[3.0, 4.0])
    state = Flux.setup(DU.DistributedOptimizer(backend, rule), model)
    # Give the frozen leaf real history before saving it. All ranks use this same
    # local initialization; ordinary training still updates through the wrapper.
    tree, model = Optimisers.update!(state.tree, model,
        (a=nothing, b=nothing, frozen=ones(2)))
    state = W(backend, tree)
    Optimisers.freeze!(state.tree.frozen)
    return model, state
end
grads() = (a=fill(Float64(rank + 1), 2), b=ones(2), frozen=ones(2))
function equal_checkpoint(model, state, payload)
    @test model.a ≈ payload.model.a
    @test model.b ≈ payload.model.b
    @test model.frozen == payload.model.frozen
    @test state.tree.a.state[1] ≈ payload.tree.a.state[1]
    @test state.tree.a.state[2] ≈ payload.tree.a.state[2]
    @test state.tree.a.state[3] == payload.tree.a.state[3]
    @test state.tree.a.rule == payload.tree.a.rule
    @test state.tree.frozen.frozen
    @test state.tree.frozen.state == payload.tree.frozen.state
    @test state.tree.frozen.rule == payload.tree.frozen.rule
    @test state.tree.a.frozen == payload.tree.a.frozen
    @test model.a === model.b
    @test state.tree.a === state.tree.b
end
rank == 0 && begin
    println("Julia=", VERSION, " mode=", mode, " ranks=", DU.total_workers(backend))
    for m in (Flux, Optimisers, Functors, MPI, JLD2)
        println(nameof(m), "=", pkgversion(m), " source=", pathof(m))
    end
    println("MPI_library=", MPI.MPI_LIBRARY)
end
started = time()
@testset "Phase H $mode rank=$rank" begin
    @test DU.total_workers(backend) == 2
    if mode == "write"
        model, state = fixture()
        @test any(!iszero, state.tree.frozen.state[1])
        @test propertynames(state) == (:backend, :tree)
        @test keys(Functors.children(state)) == (:tree,)
        @test_throws ErrorException state.a
        @test_throws MethodError state[:a]
        @test_throws MethodError keys(state)
        @test_throws MethodError length(state)
        @test_throws MethodError iterate(state)
        @test_throws MethodError collect(zip(state, model))
        @test keys(state.tree) == keys(model)
        @test state.tree.a === state.tree.b
        @test_throws Exception Functors.fmap((x, y) -> x, state, model)
        mapped = Functors.fmap(x -> x isa AbstractArray ? copy(x) : x, state)
        @test mapped.backend === backend
        @test mapped.tree.a === mapped.tree.b
        @test mapped.tree.a !== state.tree.a
        cpustate = Flux.cpu(state)
        @test cpustate.backend === backend
        @test cpustate.tree.a === cpustate.tree.b
        adjusted = Optimisers.adjust(state, Optimisers.Adam(0.02))
        @test adjusted.backend === backend
        @test adjusted.tree.a === adjusted.tree.b
        @test adjusted.tree.a !== state.tree.a
        @test adjusted.tree.a.state[1] !== state.tree.a.state[1]
        @test adjusted.tree.a.rule.eta == 0.02
        @test state.tree.a.rule.eta == 0.01
        @test adjusted.tree.frozen.frozen
        @test Optimisers.adjust!(adjusted; eta=0.03) === adjusted
        @test Optimisers.adjust!(adjusted, Optimisers.Adam(0.04)) === adjusted
        @test adjusted.tree.a.rule.eta == 0.04
        @test Optimisers.adjust(DU.DistributedOptimizer(backend, rule), 0.02).backend === backend
        # Non-mutating update copies tree/model, preserves their internal ties and backend.
        state2, model2 = Optimisers.update(state, model, grads())
        @test state2.backend === backend
        @test state2.tree.a === state2.tree.b
        @test model2.a === model2.b
        @test model2.a !== model.a
        @test state2.tree.a !== state.tree.a
        @test model.a == [1.0, 2.0]
        state, model = Optimisers.update!(state, model, grads())
        equal_checkpoint(model, state, (; model=model2, tree=state2.tree))
        # Persist only portable data. The next MPI job constructs its own backend.
        payload = (; model, tree=state.tree)
        if rank == 0
            serialize(joinpath(directory, "checkpoint.jls"), payload)
            JLD2.jldsave(joinpath(directory, "checkpoint.jld2"); payload)
            JLD2.jldsave(joinpath(directory, "model-only.jld2"); model_state=Flux.state(model))
        end
        state, model = Optimisers.update!(state, model, grads())
        if rank == 0
            serialize(joinpath(directory, "expected.jls"), (; model, tree=state.tree))
        end
        MPI.Barrier(backend.comm)
        # Arrays and scalar state synchronize through the wrapper.
        # Contract: rule and frozen configuration stay rank-local in both paths.
        # This deliberately violates update preconditions; it is not a restore recipe.
        syncmodel = (a=ones(2), b=ones(2))
        syncstate = Optimisers.setup(DU.DistributedOptimizer(backend, rule), syncmodel)
        leaf = syncstate.tree.a
        leaf.state = (fill(Float64(rank + 1), 2), fill(Float64(rank + 3), 2), (0.5 + rank/10, 0.6 + rank/10))
        Optimisers.adjust!(leaf, 0.01 * (rank + 1))
        rank == 0 && Optimisers.freeze!(leaf)
        @test DU.synchronize!!(backend, syncstate) === syncstate
        @test leaf.state[1] == ones(2)
        @test leaf.state[2] == fill(3.0, 2)
        @test leaf.state[3] == (0.5, 0.6)
        @test leaf.rule.eta == 0.01 * (rank + 1)
        @test leaf.frozen == (rank == 0)
        returned = DU.synchronize!!(backend, syncstate.tree)
        @test returned.a.state[3] == (0.5, 0.6)
        @test returned.a === leaf
        @test returned.a.rule.eta == 0.01 * (rank + 1)
        @test returned.a.frozen == (rank == 0)
        tiedmodel, tiedstate = fixture()
        returned_ties = DU.synchronize!!(backend, tiedstate.tree)
        @test tiedstate.tree.a === tiedstate.tree.b
        @test returned_ties.a === returned_ties.b
        @test returned_ties.a === tiedstate.tree.a
        println("SYNC_CONTRACT rank=$rank wrapper_scalar=$(leaf.state[3]) returned_scalar=$(returned.a.state[3]) rule_eta=$(leaf.rule.eta) frozen=$(leaf.frozen) native_return_preserves_leaf_ties=true requires_rank_consistent_configuration=true full_checkpoint_restore=false")
    elseif mode == "resume"
        expected = deserialize(joinpath(directory, "expected.jls"))
        for format in (:Serialization, :JLD2)
            payload = format == :Serialization ? deserialize(joinpath(directory, "checkpoint.jls")) :
                JLD2.load(joinpath(directory, "checkpoint.jld2"), "payload")
            model = payload.model
            state = W(backend, payload.tree)
            @test state.backend === backend
            @test model.a === model.b
            @test state.tree.a === state.tree.b
            @test state.tree.frozen.frozen
            @test state.tree.a.rule == rule
            state, model = Optimisers.update!(state, model, grads())
            equal_checkpoint(model, state, expected)
            println("RESUME rank=$rank format=$format fresh_process_backend=true values_moments_decay_rules_freeze_ties=pass")
        end
        model, _ = fixture()
        Flux.loadmodel!(model, JLD2.load(joinpath(directory, "model-only.jld2"), "model_state"))
        payload = deserialize(joinpath(directory, "checkpoint.jls"))
        @test model.a == payload.model.a
        @test model.a === model.b
        fresh = Flux.setup(DU.DistributedOptimizer(backend, rule), model)
        @test fresh.tree.a.state[1] != payload.tree.a.state[1]
        @test !fresh.tree.frozen.frozen
    else
        error("unknown mode: $mode")
    end
end
println("PHASE_H_DONE mode=$mode rank=$rank elapsed=$(round(time()-started; digits=3))")
MPI.Barrier(backend.comm)
