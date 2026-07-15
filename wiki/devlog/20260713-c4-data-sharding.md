# 2026-07-13: C4 — Distributed data sharding

## Objective
Fix a critical bug in `DistributedDataContainer` regarding data partitioning, which caused bounds errors and led to deadlock states during distributed training due to ranks processing a mismatched number of batches.

## Issue
The original `__construct_distributed_data_container` distributed data evenly using `Iterators.partition(1:total_size, size_per_worker)`. However, if `total_size` was not evenly divisible by `total_workers`, this would produce fewer partitions than workers. For example, $N=9$ over $W=4$ workers:
- `size_per_worker = ceil(9/4) = 3`
- `Iterators.partition` produces 3 partitions: `1:3`, `4:6`, `7:9`.
- Rank 3 (the 4th worker) tries to access the 4th partition and raises a `BoundsError`.

Even worse, if $N$ is just large enough to produce 4 partitions, but the last partition is smaller, Rank 3 might receive fewer items. This causes Rank 3 to finish its training epoch earlier than other ranks. Rank 3 then drops out of collective communication operations like `Allreduce`, resulting in a DDP deadlock.

## Solution
Implemented dataset padding in `Flux.jl`'s `DistributedUtils.jl` to precisely replicate PyTorch's `DistributedSampler` default behavior:
1. `total_padded = size_per_worker * split_across`
2. If `total_padded > total_size`, append indices from the start of the dataset: `1:(total_padded - total_size)`.

Now, if $N=9$ and $W=4$:
- `total_padded = 12`
- We append indices `1:3`.
- Rank 3 receives `1:3` as its partition.
- All ranks are guaranteed exactly `size_per_worker` items.

## Verification
- Added `scripts/data/verify_sharding.jl`.
- Runs a dummy training loop tracking the number of batches processed and forces an MPI `Allreduce`.
- The test successfully verified that all ranks process exactly 2 batches without bounds errors or hanging.
- A new Jupyter notebook `notebook/C4_Data_Sharding.ipynb` was created to explain the mathematics of the problem and the PyTorch-style fix.

## Status
Checkpoint C4 completed. Next: C5 — Gradient synchronization minimal case.
## The Padding Trade-Off: Duplication vs. Deadlocks

Because we **must** have the exact same number of batches on every rank to prevent `Allreduce` deadlocks, we have to either add fake data (padding) or throw data away (dropping). 

In our toy example ($N=9, W=4$), Rank 0 and Rank 3 receive the exact same data indices (`1:3`). This 33% duplication seems extreme, but it scales perfectly:
1. **Negligible Bias at Scale**: The maximum number of items ever padded is $W - 1$. For 1,000,000 images on 8 GPUs, padding 7 images introduces statistically negligible bias.
2. **Shuffling**: In practice, datasets are shuffled at the start of every epoch (before padding). The 1 to $W-1$ items that are duplicated are completely random every epoch, ensuring no single piece of data is systematically over-represented.
3. **Data Utilization**: Padding is preferred over truncation (`drop_last`) as it guarantees 100% dataset utilization per epoch.
