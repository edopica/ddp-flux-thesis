# GPU scaling experiment discussion

Date: 2026-10-15

## Request

The user proposes static and conditional training workloads in Flux and PyTorch on Leonardo.
Each workload starts with one process on one GPU, followed by two and four ranks.
The professor suggested “tiny-images” as a dataset between MNIST and a large demonstration dataset.

## Recommendations for discussion

- Confirm whether the dataset is Tiny ImageNet-200 before implementation.
- Use fixed-global-batch strong scaling as the primary experiment.
- Report fixed-local-batch throughput scaling separately because it changes the global batch.
- Use a substantial convolutional trunk and controlled conditional branches with comparable active computation.
- Keep sample identities and branch assignments independent of rank count.
- Match initial arrays, preprocessing, loss reduction, precision, optimizer equations, and update counts across frameworks.
- Run separate correctness checks for local absence, global absence, and reactivation after nonzero Adam history.
- Use native PyTorch DDP as the practical baseline and record its unused-parameter configuration.
- Record differences in numerical guards instead of attributing every framework difference to communication.
- Measure training-only throughput and end-to-end throughput separately.
- Exclude compilation and warm-up from steady-state timing. Warm up every conditional path.
- Use synchronized window boundaries, slowest-rank duration, repeated runs, and recorded hardware/software identities.

## Existing evidence

The October 6 campaign reported training-only speedup 1.888594 for its small one-/four-GPU workload.
That campaign predates the latest numerical guards and cannot represent current implementation performance.
ADR-0009 adds device finiteness reductions, synchronization, and an all-rank validity vote.
The current workload needs fresh correctness checks on Leonardo before timed runs.

## Session outcome

The experiment configuration remains a proposal. No training or remote command ran.
The local command `git status --short` returned exit 0 with no output before documentation edits.
The README progress table remains unchanged because no implementation milestone changed.

**Exact next action:** confirm the dataset identity, then specify the model pair and rank-independent global batch/routing schedule.

## Follow-up: supervisor draft

The user selected static networks, fixed global batch size, and eager PyTorch without explicit compilation.
Tiny ImageNet-200 is the intended dataset. The user explicitly requested a discussion draft before supervisor review.

Created [the draft report](../../../reports/static-graph-gpu-scaling-proposal.md) with external references and analytical memory estimates.
The strongest recipe reference is `zeyuanyin/tiny-imagenet`: small-image ResNet-18, batch 256, SGD, and reported one-A100 training.
The proposed GroupNorm replacement is explicitly an adaptation, not a reproduction of the reported accuracy.
The report explains local BatchNorm statistics, DDP buffer broadcasts, SyncBatchNorm communication, and the fixed-global-batch equivalence requirement.

Calculated values: 11,271,432 parameters, 43.00 MiB per FP32 parameter-sized array, 171.99 MiB for parameters, gradients, and Adam moments.
The convolution-output proxy is 9.375 MiB per sample. Illustrative activation multipliers are assumptions, not measured bounds.
The arithmetic command passed. `git diff --check` returned exit 0 for tracked changes.
No experiment or remote command ran. The README progress table remains unchanged because this is a proposal.

The research-agent invocation failed before execution with an unavailable model.
Direct web research completed the task. See [tool failure](../../failures/2026-10-15-research-agent-model-unavailable.md).

Arithmetic reproduction command (exit 0):

```bash
python -c 'p=11689512-513*800-(7*7-3*3)*3*64; print("parameters",p); print("FP32 MiB",p*4/2**20,"SGD momentum MiB",p*12/2**20,"Adam MiB",p*16/2**20); print("conv output MiB/sample",(64*64*64*(1+4)+32*32*128*5+16*16*256*5+8*8*512*5)*4/2**20); print("dataset uint8/float32 GiB",100000*64*64*3/2**30,100000*64*64*3*4/2**30); [(print("batch",b,"input MiB",b*64*64*3*4/2**20,"conv output GiB",b*9.375/1024,"scenario GiB",b*9.375*2/1024+p*16/2**30,b*9.375*4/1024+p*16/2**30)) for b in [64,128,256,512]]'
```

**Exact next action:** discuss the model and normalization choice with the supervisor before implementation or Leonardo allocation.

## Follow-up: short supervisor proposal

The report `reports/static-graph-gpu-scaling-proposal.md` was rewritten as a 1–2 page Italian proposal for supervisor submission.
The extended English draft is preserved at `reports/static-graph-gpu-scaling-proposal-detailed.md`.
No experiment, arithmetic change, or implementation occurred.

**Exact next action:** discuss the model and normalization choice with the supervisor before implementation or Leonardo allocation.

## Follow-up: global batch 512 proposed

After discussion, the proposal now uses global batch 512 as the primary candidate (local batches 512/256/128 at one/two/four GPUs).
Rationale: the DDP all-reduce cost is approximately constant per step while compute scales with batch, so local batch 128 at four GPUs uses the A100s better than 64.
The reference recipe learning rate (0.2 at batch 256) needs documented rescaling (linear rule: 0.4) with its warm-up schedule.
Global batch 256 remains the fallback if the single-GPU memory pilot lacks headroom.
Leftover policy unchanged: 195 full batches and the same 160 dropped images per epoch (99,840 exposures).
Both proposal versions were updated: `reports/static-graph-gpu-scaling-proposal.md` and `reports/static-graph-gpu-scaling-proposal-detailed.md`.
No experiment or implementation ran.

## Follow-up: AdamW selected and GroupNorm approved

Supervisor feedback (2026-10-15): keep the global batch fixed at the largest value that fits a single GPU, use AdamW, and use the standard ResNet normalization.
After the BatchNorm explanation, GroupNorm was approved as the documented adaptation, and AdamW was selected as the optimizer.
The detailed proposal now uses AdamW with decoupled weight decay; the reference recipe's SGD settings (momentum 0.9, learning rate 0.2) no longer transfer.
The short Italian proposal still lists "SGD momentum o Adam" and awaits alignment.
No experiment or implementation ran.

**Exact next action:** align the short Italian proposal with the selected AdamW optimizer, then confirm the single-GPU batch with the memory pilot.
