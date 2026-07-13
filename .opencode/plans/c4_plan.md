# Implementation Plan for C4 (Distributed Data Sharding)

## 1. Reproduce the Bug
- Create `scripts/data/reproduce_bug.jl`.
- Set dataset size $N=9$ and run with 4 MPI ranks.
- `DistributedDataContainer` will assign 3 items each to ranks 0, 1, and 2. Rank 3 will hit a `BoundsError` due to `Iterators.partition` exhausting the dataset early.
- Run the script and save the output.

## 2. Notebook Creation (Before)
- Generate a Jupyter notebook `notebook/C4_Data_Sharding.ipynb`.
- **Before Section**: Illustrate the mathematical flaw in `Iterators.partition` for distributed sharding. Show how uneven batches not only cause a `BoundsError` but also risk DDP deadlocks if different ranks produce a different number of batches.

## 3. Fix Flux.jl
- Update `src/distributed/public_api.jl` in the `Flux.jl` repository.
- Modify `DistributedDataContainer` to introduce PyTorch-like dataset padding:
  - Calculate `total_padded = ceil(N/W) * W`.
  - Append indices `1:(total_padded - N)` to the dataset partitioning indices.
- This ensures all ranks receive the exact same number of items, mirroring PyTorch's default `DistributedSampler`.

## 4. Notebook Update (After)
- Add an **After Section** to the notebook.
- Demonstrate the new, balanced behavior showing perfectly even item counts and batch counts across all ranks.

## 5. Verification
- Write `scripts/data/verify_sharding.jl`.
- Add a `make verify-data` target.
- Run it in a real MPI environment to confirm the fix works and all ranks complete training loops without deadlocking.

## 6. Documentation
- Write the devlog `openwiki/devlog/202607XX-c4-data-sharding.md`.
- Update the thesis dashboard and `current.md` to reflect C4 completion.
