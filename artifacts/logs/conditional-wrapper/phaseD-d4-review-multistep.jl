include("/home/kurapica/Projects/ddp_flux/flux-integration/test/ext_distributed/conditional_distributedtest.jl")

# Characterization: exit 0 confirms the review defect, not correctness.
@testset "D4 review: setup selection is lost on step two" begin
    for update_fn in (Optimisers.update!, Optimisers.update)
        model = LegacyTupleTrainable(fill(1.0, 4), fill(1.0, 4), fill(9.0, 4))
        ref_model = deepcopy(model)
        rec = RecordingBackend(backend, Any[])
        state = Optimisers.setup(DistributedUtils.DistributedOptimizer(rec, Optimisers.Descent(0.1)), model)
        ref_state = Optimisers.setup(Optimisers.Descent(0.1), ref_model)
        grads = (a=fill(Float64(rank + 1), 4), b=fill(Float64(rank + 5), 4), excluded=fill(9.0, 4))
        mean_grads = deepcopy(grads)
        DistributedUtils.allreduce!(backend, mean_grads.a, DistributedUtils.avg)
        DistributedUtils.allreduce!(backend, mean_grads.b, DistributedUtils.avg)
        for step in 1:2
            empty!(rec.log)
            state, model = update_fn(state, model, grads)
            ref_state, ref_model = update_fn(ref_state, ref_model, mean_grads)
            @test state.tree.b isa Optimisers.Leaf
            @test model.a ≈ ref_model.a
            if step == 1
                @test model.b ≈ ref_model.b
                @test length(rec.log) == 3
            else
                @test !(model.b ≈ ref_model.b)
                @test length(rec.log) == 2
            end
            if rank == 0
                println("D4_REVIEW path=", update_fn, " step=", step,
                    " b=", model.b[1], " native_b=", ref_model.b[1],
                    " collectives=", length(rec.log))
            end
        end
    end
end
