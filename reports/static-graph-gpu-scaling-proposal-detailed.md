# Draft proposal: static-network GPU scaling in Flux and PyTorch

Date: 2026-10-15  
Status: **Discussion draft for supervisor review**  
Target system: one Leonardo node, with one, two, or four GPUs

## 1. Summary

The proposed experiment measures training performance for the same static network in Flux.jl and PyTorch.
It uses Tiny ImageNet-200 and a fixed global batch size at every GPU count.
The single-GPU case uses one ordinary training process. The distributed cases use one process per GPU.

The recommended starting model is a documented small-image ResNet-18, with GroupNorm as an explicit architectural adaptation.
GroupNorm removes dependence on the local batch size and simplifies comparison against a single-GPU reference.
The proposed global batch is 512, with 256 as a fallback, subject to a later memory pilot and supervisor agreement.

PyTorch runs in eager mode, without `torch.compile` or an explicit graph-compilation pipeline.
The experiment retains ordinary native kernels and native DDP communication.
Conditional networks remain a later extension. Fixed-local-batch experiments are outside this draft.

**No experiment ran for this report.** Memory values are analytical estimates, and external accuracy figures are attributed reference results.
GroupNorm and AdamW are now selected. The exact model size, hyperparameters, training duration, and the final batch size remain open decisions.

## 2. Research question and scope

The main question is:

> How much does training time decrease when the same workload moves from one GPU to two or four GPUs?

A second question compares absolute throughput and scaling efficiency between Flux and eager PyTorch.
These measurements will establish a baseline for future Flux DDP improvements.

Here, a **static network** executes the same layers and uses the same trainable parameters on every step.
This definition does not imply a compiled execution graph.
PyTorch DDP's `static_graph` argument is a separate configuration flag, rather than a compiler switch [5].

The initial matrix contains six configurations:

| Framework | GPUs | Processes | Global batch, proposed | Local batch |
|---|---:|---:|---:|---:|
| Flux | 1 | 1 | 512 | 512 |
| Flux | 2 | 2 | 512 | 256 |
| Flux | 4 | 4 | 512 | 128 |
| PyTorch | 1 | 1 | 512 | 512 |
| PyTorch | 2 | 2 | 512 | 256 |
| PyTorch | 4 | 4 | 512 | 128 |

Every configuration consumes the same global batches and performs the same number of optimizer updates.
The learning-rate schedule does not scale with GPU count because the global batch does not change.
The reference recipe's SGD settings do not transfer to AdamW; learning rate, decoupled weight decay, betas, epsilon, and warm-up are a new documented choice, identical in both frameworks.
The initial design uses no gradient accumulation or activation checkpointing.

## 3. Dataset and documented networks

### 3.1 Tiny ImageNet-200

Tiny ImageNet contains 200 classes of 64×64 RGB images [1, 2]:

| Split | Images per class | Total images |
|---|---:|---:|
| Training | 500 | 100,000 |
| Validation | 50 | 10,000 |
| Test | 50 | 10,000 |

The proposal uses the labeled training and validation splits.
It retains native 64×64 resolution and trains from scratch.
Resizing to 224×224 or using pretrained weights changes the workload and complicates comparison with a small-image baseline.

Dataset size controls the available training examples, but it does not determine GPU utilization by itself.
Model computation, activation sizes, batch size, and communication determine scaling behavior.

### 3.2 Available precedents

There are documented ResNet baselines, although no single Tiny ImageNet architecture is mandatory.

| Reference | Documented workload | Relevance |
|---|---|---|
| Zeyuan Yin, `tiny-imagenet` [2] | ResNet-18 and ResNet-50, modified small-image stem, 200 outputs | Strong starting point: explicit recipe, checkpoints, and one-A100 training |
| `Tiny-ImageNet-Classifier` [3] | ResNet-18 with original and modified stems, from-scratch and transfer-learning experiments | Illustrates why the stem and pretraining history must be specified |
| TorchVision ResNet source [4] | Standard residual blocks, stage structure, initialization, and configurable normalization | Precise architecture reference for the cross-framework implementation |

Reference [2] reports ResNet-18 top-1 accuracy of 59.57%, 60.23%, and 60.50% after 50, 100, and 200 epochs.
It reports batch size 256, SGD momentum 0.9, weight decay 1e-4, and initial learning rate 0.2.
Its schedule uses cosine decay and five warm-up epochs.
These are author-reported results, not independently reproduced results or acceptance thresholds for this proposal.

Reference [3] includes transfer-learning results with different resolutions and weight histories.
Those numbers cannot serve as direct targets for a from-scratch GroupNorm experiment.

### 3.3 Recommended architecture

The recommended candidate follows the small-image adaptation in reference [2]:

| Component | Proposed specification | Output per sample |
|---|---|---|
| Input | RGB, native resolution | 64×64×3 |
| Stem | 3×3 convolution, 64 channels, stride 1, padding 1, no bias | 64×64×64 |
| Initial normalization and activation | GroupNorm, then ReLU | 64×64×64 |
| Initial max-pooling | None | — |
| Residual stage 1 | Two BasicBlocks, 64 channels | 64×64×64 |
| Residual stage 2 | Two BasicBlocks, 128 channels, first block downsamples | 32×32×128 |
| Residual stage 3 | Two BasicBlocks, 256 channels, first block downsamples | 16×16×256 |
| Residual stage 4 | Two BasicBlocks, 512 channels, first block downsamples | 8×8×512 |
| Pooling | Global spatial average | 512 |
| Classifier | Linear layer with bias | 200 logits |

Each BasicBlock contains two 3×3 convolutions, normalization layers, ReLU activations, and a residual addition.
Downsampling blocks use a stride-2 first convolution and a matching 1×1 projection with normalization.
The proposal replaces every BatchNorm layer with affine GroupNorm, including projection paths.
The initial GroupNorm choice is 32 groups with epsilon 1e-5. Every stage width is divisible by 32.

The label must be **“Tiny ImageNet ResNet-18, small-image stem, GroupNorm-32.”**
This is an adaptation of the documented model, not an exact reproduction of its BatchNorm accuracy results.

The original TorchVision stem also accepts 64×64 inputs, but immediately reduces spatial size to 16×16 through convolution and max-pooling.
The modified stem preserves spatial detail and gives a more substantial compute workload.
Its residual-stage activation maps contain 16 times as many elements as the original-stem maps at the same input resolution.
That ratio describes those maps, not total training memory or total runtime.

**Recommendation:** use ResNet-18 before ResNet-50 to limit implementation and review effort.
AdamW with decoupled weight decay is the selected optimizer [13].
Both frameworks must implement the same AdamW equations with identical hyperparameters and the same parameter groups.
The documented recipe's SGD settings (momentum 0.9, learning rate 0.2) do not transfer to AdamW.

## 4. Why BatchNorm complicates fixed-global-batch scaling

### 4.1 The normalization population changes

Training-mode BatchNorm normalizes each channel using the current batch and spatial positions [6]:

\[
y = \gamma\frac{x-\mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}}+\epsilon}}+\beta.
\]

With global batch 512, the ordinary single-GPU layer calculates statistics over 512 images.
With four ranks and ordinary BatchNorm, each layer calculates separate statistics over 128 local images.
Thus, the same image can produce different normalized features depending on the GPU count and its shard companions.
The difference exists even when 64 images provide statistically reasonable estimates.

For batch-independent networks and equal shards, averaging local mean-loss gradients gives the gradient of the global mean loss:

\[
\frac{1}{N}\sum_{r=1}^{N}\nabla L_r = \nabla L_{\mathrm{global}}.
\]

Local BatchNorm breaks the premise behind this equivalence.
Each shard evaluates a different batch-dependent forward function from the single-GPU global-batch function.
Gradient averaging synchronizes the resulting gradients. It does not retroactively replace the local forward statistics with global statistics.

For intuition, consider two equal shards with different channel means.
Local normalization centers each shard around its own mean. Global normalization centers both around one shared mean.
The global variance also contains the between-shard mean difference, which a simple average of local variances misses.

### 4.2 Why this matters to the experiment

The goal is to change resource count while preserving the training problem.
Local BatchNorm changes both resource count and the effective normalization population.
Consequently, differences in learning curves can reflect normalization changes rather than distributed-update defects.
A short single-GPU/DDP equivalence check also becomes harder to interpret.

Smaller local batches can additionally produce noisier statistics, especially in later layers with small spatial maps [7].
This does not mean that BatchNorm necessarily fails at local batch 64.
BatchNorm remains a valid practical training choice. Its batch dependence is a confounding factor for this particular controlled experiment.

### 4.3 Running buffers do not solve the problem

BatchNorm usually stores running means and variances for evaluation [6].
PyTorch DDP normally broadcasts buffers from rank 0 at the beginning of forward execution [5].
This buffer broadcast does **not** calculate global current-batch statistics.
Each ordinary BatchNorm layer still normalizes its training input using the local batch.

Cross-framework reproduction with BatchNorm also requires matched variance estimators, momentum conventions, and buffer-update behavior.
Synchronizing trainable parameters alone does not establish equivalent BatchNorm state.

### 4.4 Alternatives and their tradeoffs

| Choice | Effect on this experiment |
|---|---|
| Local BatchNorm | Closest to many published recipes, but changing rank count changes normalization populations |
| SyncBatchNorm | Restores global-batch statistics across ranks, but adds normalization communication and requires equivalent implementations |
| GroupNorm | Computes statistics within each sample, independent of batch partition, with no normalization collectives |
| Frozen BatchNorm | Removes current-batch dependence, but changes training behavior and needs meaningful fixed statistics |

SyncBatchNorm is conceptually appropriate for global-batch BatchNorm semantics [8].
However, it adds communication during normalization and backward propagation.
The timing then includes both gradient synchronization and normalization synchronization.
That is a useful future workload, but it expands the first experiment's implementation scope.

GroupNorm calculates statistics over channel groups and spatial positions within each sample [7].
It therefore preserves the forward normalization when the same batch splits across different GPU counts.
Small numerical differences can still arise from kernels and reduction order.

**Recommendation:** GroupNorm for the first controlled scaling study, with the architectural change stated explicitly.
Its accuracy and kernel performance can differ from BatchNorm. This proposal makes no claim that GroupNorm is faster or more accurate here.

## 5. Preliminary memory estimates

### 5.1 Assumptions and units

The estimates use the proposed small-image ResNet-18 with 200 outputs, affine normalization, and FP32 tensors.
They assume ordinary replicated DDP, with no optimizer sharding, mixed precision, accumulation, or activation checkpointing.
One MiB is 2^20 bytes. One GiB is 2^30 bytes.

The project records four A100-SXM GPUs with 64 GB each on a Leonardo node [10].
Actual available memory and the selected runtime stack require a later allocation check.

### 5.2 Parameters and optimizer state

TorchVision reports 11,689,512 parameters for standard ResNet-18 with 1,000 outputs [4].
Replacing its classifier and stem gives:

\[
P=11{,}689{,}512-(512+1)(1000-200)-64\cdot3(7^2-3^2)
=11{,}271{,}432.
\]

Affine GroupNorm and affine BatchNorm have the same trainable parameter count for matching channels.
BatchNorm also has non-trainable running buffers, which GroupNorm does not require.

| Storage component | Formula | Approximate size per GPU |
|---|---:|---:|
| Parameters | 4P bytes | 43.00 MiB |
| One complete gradient set | 4P bytes | 43.00 MiB |
| Two AdamW moment buffers | 8P bytes | 85.99 MiB |
| Parameters + gradients + AdamW moments | 16P bytes | 171.99 MiB |

These totals exclude activations, temporary tensors, communication storage, and runtime overhead.
Ordinary DDP replicates parameters and optimizer state on every GPU.
Four GPUs therefore do not divide this persistent storage by four.

An additional full-sized gradient copy costs approximately 43 MiB per GPU.
PyTorch communication buckets and Flux synchronization buffers can add storage of this order.
Exact allocation depends on buffer aliasing, framework configuration, and tensor lifetimes.
PyTorch documents that `gradient_as_bucket_view=True` can save one gradient-sized allocation [5].
The initial proposal leaves this optional optimization disabled and records the actual configuration.

### 5.3 Inputs and dataset storage

One FP32 input image occupies `64 × 64 × 3 × 4 = 49,152` bytes, or 48 KiB.

| Local batch | Input tensor alone |
|---:|---:|
| 64 | 3 MiB |
| 128 | 6 MiB |
| 256 | 12 MiB |
| 512 | 24 MiB |

The 100,000 training images occupy approximately 1.14 GiB as decoded uint8 RGB arrays, or 4.58 GiB as FP32 arrays.
These figures exclude labels, metadata, array objects, augmentation copies, and loader queues.
They describe decoded storage, not the compressed download size.

The full dataset need not reside on each GPU.
A host-side cache and bounded device batches keep input storage separate from model memory.
Data-loader workers and prefetch queues can multiply host memory use, so their configuration belongs in the experiment record.

### 5.4 Activation memory is the main uncertainty

One early feature map has `64 × 64 × 64` FP32 values per image: **1 MiB per image**.
At local batch 512, that single map occupies 512 MiB.
Training retains multiple tensors for backward computation, so parameter size alone substantially understates memory requirements.

A transparent activation-size proxy is the sum of all convolution outputs, counted once:

\[
A_{\mathrm{proxy}}/b =
4[5(64^2\cdot64)+5(32^2\cdot128)+5(16^2\cdot256)+5(8^2\cdot512)]
=9.375\ \mathrm{MiB}.
\]

Here, `b` is the local batch size.
The first term counts the stem and four stage-1 convolutions.
Each later term counts four residual convolutions and one projection convolution.
This proxy is **not** peak live memory or a strict lower bound: tensor retention, aliasing, and reuse differ across implementations.

| Local batch | Convolution-output proxy | Illustrative `2A + AdamW storage` | Illustrative `4A + AdamW storage` |
|---:|---:|---:|---:|
| 64 | 0.586 GiB | 1.34 GiB | 2.51 GiB |
| 128 | 1.172 GiB | 2.51 GiB | 4.86 GiB |
| 256 | 2.344 GiB | 4.86 GiB | 9.54 GiB |
| 512 | 4.688 GiB | 9.54 GiB | 18.92 GiB |

The multipliers illustrate sensitivity to additional normalization, activation, and backward tensors.
They are modeling assumptions, **not calibrated bounds or measured ranges**.
Inputs, convolution workspaces, communication buffers, allocator reservations, and CUDA context memory require additional space.
Temporary optimizer allocations and Flux AD retention can also change the peak.

**Planning conclusion:** batch 512 is the primary pilot candidate on a 64-GB A100, with 256 as the fallback, but its exact peak remains unmeasured.
The illustrative calculation suggests several GiB, up to roughly 19 GiB in the single-GPU tensors-plus-optimizer scenario, rather than hundreds of MiB.
It does not guarantee that the complete process fits with sufficient workspace headroom.

With global batch 512, the activation-dependent portion decreases approximately with local batch: 512, 256, then 128.
Persistent state remains replicated, and DDP adds communication storage.
Measured per-GPU memory therefore need not decrease exactly as `1/N`.

### 5.5 Proposed later memory pilot

After approval, the pilot will measure both frameworks after warm-up and optimizer-state initialization.
It will record peak live tensor memory, allocator-reserved memory where available, and device-level process memory.
Those categories must remain separate because allocator reservations do not equal live tensors.
The pilot will cover forward, backward, communication, and optimizer update at all three GPU counts.

The final fixed global batch must fit the one-GPU case in both frameworks with sufficient workspace headroom.
The same batch will then apply to the complete experiment matrix.

## 6. Meaning of an eager PyTorch baseline

The intended comparison excludes explicit model compilation and optional accelerator-specific tuning.
It does not require disabling the optimized libraries that implement ordinary tensor operations.

| Feature | Proposed initial policy |
|---|---|
| `torch.compile`, TorchScript, explicit CUDA Graph capture | Disabled / unused |
| Native CUDA, cuDNN, and cuBLAS operations | Retained |
| Native NCCL DDP, including normal bucketing and overlap | Retained, configuration recorded |
| `find_unused_parameters` | `False`, because all parameters participate |
| DDP `static_graph` | `False` for the initial default-style baseline |
| Mixed precision and explicit fused optimizer paths | Disabled |
| TF32 | Explicitly disabled in both frameworks for the initial FP32 comparison |
| Channels-last conversion and communication hooks | No optional tuning |

`torch.compile` has multiple backends [9]. Describing PyTorch compilation generally as “MLIR” is too specific.
The reproducible requirement is the absence of an explicit graph-compilation path.
Julia still performs normal JIT compilation, and native libraries can perform initialization or algorithm selection.
Steady-state timing excludes those startup costs in both frameworks.

Native DDP already buckets gradients and can overlap communication with backward computation [5].
Those features belong to the practical PyTorch baseline even without `torch.compile`.
The comparison measures complete training implementations, rather than an isolated comparison of all-reduce kernels.

## 7. Reproduction and correctness requirements

The shared experiment specification will define:

- Exact model structure, convolution padding, projection paths, normalization axes, epsilon, affine parameters, and classifier bias.
- Shared initial parameter arrays, with explicit layout conversion between frameworks.
- A saved global sample order and matching labels.
- Matching preprocessing and augmentation decisions, indexed by sample and epoch rather than rank.
- Identical mean-reduced cross-entropy, optimizer equations, hyperparameters, and schedule.
- Equal local batches and update counts, with explicit gradient averaging by world size.
- The same treatment of the final incomplete global batch.

The same integer seed does not guarantee the same initial weights or augmentations across Julia and Python.
Short correctness runs will use shared tensors and compare outputs, gradients, parameters, and optimizer state within predefined tolerances.
The checks will compare each distributed implementation with its single-GPU global-batch reference, then compare the two frameworks.
Long training trajectories need not match bit-for-bit because floating-point reduction order differs.

For batch 512, 100,000 training images produce 195 full batches and the same 160 remaining images.
A simple initial policy drops the same final 160 images from each saved epoch permutation in every configuration.
The record must state the resulting 99,840 sample exposures per epoch.
Rank-specific padding or independent shuffling must not silently change the global workload.

Learning curves provide context but do not prove distributed correctness.
Detailed reference comparisons and diagnostic host copies remain outside timed training windows.

## 8. Timing and reporting plan

The proposed campaign uses the same allocated Leonardo node for sequential configurations where feasible.
Run order alternates across frameworks and GPU counts to reduce systematic ordering effects.
The record includes device identities, node name, CPU allocation, affinity, software versions, source hashes, and exact commands.

At least five independent timed repeats are proposed for each configuration.
A later pilot will choose a fixed step count that gives the fastest configuration tens of seconds of steady-state work.
Every configuration then uses that same step count and global sample schedule.
Warm-up occurs before timing, followed by restoration of the intended model and optimizer starting state.

Two timing views answer different questions:

1. **Training-only:** prepared device batches, forward, backward, gradient communication, built-in guards, and optimizer update.
2. **End-to-end training:** input preparation, host-to-device transfer, and training under the documented input pipeline.

Both views use real Tiny ImageNet samples.
Device staging must remain bounded and its extra memory must be reported separately.
Validation, checkpoints, and report writes remain outside these timing windows.
Startup time can appear as a separate metric.

CUDA synchronization occurs at timing boundaries, with ranks aligned before the window.
The distributed duration uses the slowest rank's synchronized elapsed time.
Per-step global barriers and diagnostic synchronization are unnecessary for the primary throughput measurement.

For fixed work of `K` global updates and global batch `B`:

\[
Q_N=\frac{KB}{T_N},\qquad S_N=\frac{T_1}{T_N},\qquad E_N=\frac{S_N}{N}.
\]

The report will show absolute throughput, update time, speedup, efficiency, memory, and validation metrics.
It will show individual repeats and a dispersion statistic alongside medians.
Headline speedup uses the ratio of median durations, with a separate single-GPU baseline for each framework.
Absolute throughput prevents a weak single-GPU baseline from appearing competitive solely through a favorable speedup ratio.

Fixed CPU-resource and data-loader policies are necessary for interpreting end-to-end scaling.
For this first study, a fixed total CPU budget per job is the proposed policy.
Worker counts and thread counts must avoid oversubscription at four ranks.

## 9. Existing Flux evidence and interpretation limits

The earlier Leonardo MNIST/Fashion-MNIST timing campaign reported training-only speedup of approximately 1.89× at four GPUs [11].
That result describes a much smaller model and an earlier implementation.
It does not predict this experiment's speedup.

Current Flux includes coordinated numerical guards, including a post-reduction gradient check [12].
That check adds device reductions, synchronization, and an all-rank validity vote.
Default eager PyTorch DDP does not automatically provide the same rejection contract.

The proposal retains the current Flux guards and records this semantic difference.
The full Flux/PyTorch performance gap cannot be attributed solely to gradient communication.
Normalization kernels, AD, allocation behavior, optimizer implementation, and numerical guards also contribute.
Further profiling follows correctness acceptance and the initial measurements.

## 10. Decisions for the supervisor

| Decision | Proposed starting point |
|---|---|
| Dataset | Tiny ImageNet-200 at native 64×64 resolution |
| Model | Small-image ResNet-18 from reference [2], trained from scratch |
| Normalization | GroupNorm-32, explicitly documented as an adaptation |
| Optimizer | AdamW with decoupled weight decay; identical equations, hyperparameters, and parameter groups in both frameworks |
| Global batch | 512, subject to a later memory pilot (256 fallback) |
| Scaling | One, two, and four GPUs with fixed global work |
| PyTorch execution | Eager native DDP, without explicit graph compilation |
| Precision | FP32, matched TF32 policy, initially disabled |
| Training budget | Fixed-step throughput campaign plus an agreed learning-curve budget |
| Primary deliverable | Reproducible throughput, speedup, efficiency, memory, and correctness report |

The key tradeoff is documented-model fidelity versus rank-independent normalization.
GroupNorm provides the cleaner controlled scaling experiment, but changes the reference network's training recipe.
If exact BatchNorm reproduction takes priority, the proposal needs an explicit local-BatchNorm or SyncBatchNorm policy.

**Next action:** discuss the model and normalization choice with the supervisor before implementation or Leonardo allocation.

## Sources

Sources consulted on 2026-10-15. Repository recipes and versioned documentation provide design evidence, not tested environment pins.
Implementation work will pin exact source revisions and compatible package versions.

1. Ya Le and Xuan Yang, *Tiny ImageNet Visual Recognition Challenge*, Stanford CS231n, 2015. Dataset description. <https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf>
2. Zeyuan Yin, *Tiny-ImageNet training in Torchvision*. Small-image ResNet modification, training recipe, A100 context, and author-reported results. <https://github.com/zeyuanyin/tiny-imagenet>
3. *Tiny-ImageNet-Classifier*. Documented ResNet-18 baseline and modified-stem experiments. <https://github.com/tjmoon0104/Tiny-ImageNet-Classifier>
4. TorchVision, *ResNet source*. BasicBlock definitions, standard stem, classifier, configurable normalization, and parameter metadata. <https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html>
5. PyTorch 2.9, *DistributedDataParallel*. Gradient averaging, buffer broadcasts, bucketing, gradient views, and `static_graph`. <https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html>
6. PyTorch 2.9, *BatchNorm2d*. Normalization axes, variance estimators, running statistics, and momentum. <https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm2d.html>
7. Yuxin Wu and Kaiming He, *Group Normalization*, 2018. Batch-independent normalization and measured small-batch behavior. <https://arxiv.org/abs/1803.08494>
8. PyTorch, *SyncBatchNorm*. Normalization across a process group. <https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html>
9. PyTorch 2.9, *torch.compile*. Optional compilation interface and backend selection. <https://docs.pytorch.org/docs/2.9/generated/torch.compile.html>
10. Project hardware and execution record: [Leonardo context](../wiki/context/leonardo.md).
11. Earlier project measurements: [corrected Phase K timing report](conditional-graph-phase-k-timing-stretch.md).
12. Current Flux numerical contract: [ADR-0009](../wiki/decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md).
13. Ilya Loshchilov and Frank Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019. <https://arxiv.org/abs/1711.05101>
