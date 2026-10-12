# Devlog: 2026-07-11 — C3: Model broadcast and parameter verification

## Summary

Implemented and verified C3: all ranks start from identical model parameters after broadcasting from rank 0 using `DistributedUtils.synchronize!!`.

## What was done

### Implementation

1. **`scripts/sync/broadcast.jl`** — Main C3 script:
   - Each rank builds a model with a *different* random seed (proving parameters are genuinely different before sync).
   - Uses `DistributedUtils.synchronize!!(backend, FluxDistributedModel(model); root=0)` to broadcast rank 0's parameters.
   - Verifies all ranks hold bit-identical parameters via `allreduce` + max-deviation check.
   - Confirms rank 0's model is unchanged after sync (no mutation).
   - Also broadcasts optimizer state via `synchronize!!(backend, st_opt; root=0)`.

2. **`scripts/sync/verify.jl`** — Deep verification script with 4 tests:
   - **Test 1**: Per-tensor element-wise broadcast check (all 4 tensors: max_range=0.0).
   - **Test 2**: Optimizer state synchronization (Adam, with perturbed non-root state).
   - **Test 3**: Reference consistency — synced model matches rank 0's seed-42 original.
   - **Test 4**: Larger model (3-layer MLP, 10→128→64→1, 9729 params).

3. **Makefile targets**: Added `make sync-model` and `make verify-sync`.

### Verification

```bash
$ make sync-model   # ✅ PASSED — 2 ranks, 769 params, max_deviation=0.0
$ make verify-sync  # ✅ PASSED — All 4 tests green, including 9729-param model
```

Key output from `sync-model`:
```
[rank 0] PRE-sync param hash = 36.957302
[rank 1] PRE-sync param hash = 39.96411
[rank 0] POST-sync param hash = 36.957302
[rank 1] POST-sync param hash = 36.957302
✅ C3 PASSED: All 2 ranks have identical parameters
   Global max deviation: 0.0
```

Key output from `verify-sync`:
```
✅ model.layers[1].weight: max_range=0.0
✅ model.layers[1].bias: max_range=0.0
✅ model.layers[2].weight: max_range=0.0
✅ model.layers[2].bias: max_range=0.0
✅ Test 1 PASSED: All parameter tensors identical across ranks
✅ Optimizer state synchronized without errors
✅ Test 3 PASSED: Synced model matches rank 0's original (seed=42)
✅ Test 4 PASSED: 9729 parameters identical across ranks
✅ C3 VERIFICATION PASSED: All tests passed for 2 ranks
```

## Key API patterns used

```julia
# Broadcast model (requires FluxDistributedModel wrapper)
model = DistributedUtils.synchronize!!(
    backend,
    DistributedUtils.FluxDistributedModel(model);
    root=0
)

# Broadcast optimizer state (no wrapper needed)
st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0)
```

## Notes

- `synchronize!!` uses `fmap` internally to traverse the model tree, calling `bcast!` on each `AbstractArray{<:isbitstype}` leaf.
- The `FluxDistributedModel` wrapper is required because Flux models are arbitrary types — without it, `synchronize!!` cannot dispatch correctly.
- Non-bitstype, non-container leaves are silently returned unchanged (audit risk #1 from C0, not a problem for standard Dense/Chain models).
- Optimizer state (`Optimisers.Leaf`) has a dedicated `synchronize!!` method that handles the `.state` field via `@set!`.

## Next action

Start checkpoint C4: Distributed data sharding. Implement `DistributedDataContainer` usage and verify dataset coverage and duplication policy.
