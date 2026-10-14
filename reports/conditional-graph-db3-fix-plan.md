# DB-3: reject non-finite synchronized gradients

Date: 2026-10-13
Status: proposed implementation plan. No implementation or runtime validation occurred during planning.

## 1. Goal and failure mechanism

Prevent non-finite synchronized gradients from reaching the inner optimizer and corrupting parameters or optimizer state.
The initial policy is mandatory, coordinated rejection for the distributed wrapper.
No public opt-out or recovery policy enters this implementation.

The current wrapper builds fresh gradient buffers, reduces them, reconstructs the gradient tree, and calls Optimisers.jl.
It performs no finiteness check at this boundary.
Audit 002 demonstrates a finite scalar loss with a NaN gradient that contaminates parameters and Adam moments on both ranks.

Evidence:

- [Audit 002, DB-3](../wiki/audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md).
- `../flux-integration/src/distributed/public_api.jl:461-472,575-645,698-737`.
- `../flux-integration/ext/FluxMPIExt/FluxMPIExt.jl:100-116`.
- `../flux-integration/ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:63-84`.
- `scripts/checks/conditional_graph_deferred_probe.jl`, mode `grad-nonfinite`.

Line references describe the inspected source and can change during implementation.

## 2. Required behavior and boundaries

For every distributed wrapper update with at least one canonical trainable buffer:

1. Build the existing canonical buffers and occurrence plan.
2. Exchange the existing presence metadata.
3. Reduce every globally present buffer in the existing canonical order.
4. Evaluate finiteness of every globally present reduced buffer.
5. Participate in one all-rank invalid-gradient vote.
6. If any rank reports an invalid buffer, raise `DomainError` on every rank.
7. Otherwise, reconstruct the gradient tree and invoke Optimisers.jl as before.

The error must state that a synchronized gradient is non-finite and that no optimizer update occurred.
Do not report an offending source rank: a post-reduction check does not establish that information.

### Covered entry points

- `Optimisers.update!` and `Optimisers.update` with `DistributedOptimizerState`.
- `Flux.update!` where it dispatches to the same wrapper methods.
- Explicit `withgradient` plus either wrapper update method.
- Eager `train!`, `trainstep!`, and `trainstep_withgradient!` through those methods.

The [DB-1/DB-4 plan](conditional-graph-db1-db4-fix-plan.md) independently handles scalar-loss participation.
DB-3 alone does not correct a rank that skips the wrapper because of a non-finite loss.
Neither fix depends on the other for its focused finite-loss tests.

### Preservation guarantee

Rejection occurs before the inner optimizer call.
Parameters, existing optimizer state, and caller-owned gradients retain their pre-update values.
Fresh communication buffers can contain non-finite values and are then discarded.
Both mutating and non-mutating wrapper APIs must satisfy this guarantee.

The guard rejects non-finite values at the synchronized-gradient boundary.
It does not guarantee that finite gradients cannot overflow inside Adam or another optimizer rule.
It does not repair pre-existing invalid parameters, optimizer state, or rank-local configuration.
It does not roll back forward-pass side effects or support arbitrary process failure.

## 3. Why the check follows reduction

A post-reduction check inspects the actual values that the inner optimizer will consume.
It covers the demonstrated NaN propagation and non-finite results from local tied-gradient accumulation or communication arithmetic.

The MPI implementation currently sums the buffers before dividing by the worker count.
Finite inputs can therefore overflow during the sum, even when their mathematical average is representable.
A pre-reduction check alone misses this case.

The contract concerns the final synchronized values, not a complete audit of every original input or intermediate arithmetic operation.
Tests must establish NaN and Inf propagation on each claimed backend.
If a supported reduction masks invalid input into a finite result, assess an additional pre-reduction vote before claiming input-level rejection.

Do not throw after detecting a bad early buffer while later reductions remain outstanding.
Every rank must finish the same gradient collective schedule and reach the validity vote.
No parameter update can occur between reductions.

## 4. Implementation design

### 4.1 Insert the guard at the common wrapper boundary

Add the guard inside `_sync_gradients`, after its gradient-reduction loop and before `_build!`.
Both `Optimisers.update` and `Optimisers.update!` already call this function before the inner optimizer.
Keep that shared boundary rather than duplicating checks across public entry points.

Inspect `_SyncBuf.buf` for entries where the globally agreed `present[i]` is true.
Do not traverse the original gradient tree again.
The existing buffers already encode tied-gradient accumulation, copies into parameter-compatible storage, and zero contributions from locally absent gradients.

Preserve these cases:

- Locally absent but globally present gradients participate normally.
- Globally absent leaves still rebuild as `ZeroTangent`, preserving Adam skip semantics.
- Frozen leaves retain their existing participation rule. An invalid present frozen gradient is also rejected.
- Mutable tied parameters use one canonical buffer.
- Isbits parameters retain the existing per-occurrence buffer behavior.
- No trainable buffers retain the existing early return and zero update collectives.

If buffers exist but every gradient is globally absent, use a valid local flag and still perform the gradient vote.
This keeps one deterministic validation stage for every nonempty synchronization plan.

### 4.2 CPU and device reductions

Use a backend-compatible array reduction, conceptually `all(isfinite, buffer)`.
Confirm dispatch and result types for each supported buffer type before settling the helper implementation.
Do not assume the CPU implementation proves CUDA compatibility.

For CUDA:

- Evaluate finiteness on the device.
- Transfer only the scalar validity result needed for host control flow.
- Do not use `Array(buffer)`, `collect(buffer)`, host iteration, or scalar indexing.
- Establish ordering between NCCL writes and the finiteness reduction.
- Ensure the validity result reflects completed gradient communication before the host vote.
- Keep `CUDA.allowscalar(false)` enabled throughout focused tests.

The current NCCL extension returns after `NCCL.Allreduce!` without an explicit synchronization in that method.
Inspect the library's stream semantics and prove correct ordering with the actual test environment.
Do not treat return from that Julia method alone as proof of completed device writes.
If device-specific code is required, keep it in the appropriate extension rather than importing CUDA into core Flux.

### 4.3 One global invalid-gradient vote

Combine local buffer results into one invalid flag.
Encode invalid as `1f0` and valid as `0f0` in a one-element CPU buffer.
Use `DistributedUtils.allreduce!` with `DistributedUtils.avg` and reject any positive result.

Reuse the private invalid-flag helper from the loss fix if it exists.
Otherwise, introduce that helper here and let the loss fix reuse it later.
Do not add a collective per parameter or rely on a local throw after the scan.

The final vote guarantees a shared branch even if reduced results differ in finiteness across ranks.
A test backend must exercise that asymmetric-result case.
Normal NaN propagation alone does not prove the vote is necessary and correct.

### 4.4 Collective schedules

Let `P` be the number of globally present canonical buffers.

| Entry path | Schedule for a nonempty buffer plan |
|---|---|
| Explicit update | presence → `P` gradient reductions → gradient vote |
| Eager training with loss fix | loss vote → presence → `P` gradient reductions → gradient vote |
| Gradient rejection | Same complete schedule, then error on every rank |
| Globally absent gradients | presence → gradient vote. All inner leaves retain skip semantics |

The new collective changes exact counts in regression tests and demonstration log parsers.
Update those expectations while preserving explicit ordering checks.

## 5. RED-first acceptance matrix

Use the bounded numerical-guard harness defined in the DB-1/DB-4 plan.
Implement `--group gradient` independently of the loss guard.
Preserve the historical deferred driver and saved corruption evidence.

Before the fault step, perform valid updates that create nonzero Adam moments.
Snapshot every trainable parameter, array state, scalar state, and relevant tied identity.
Use exact snapshot equality for rejection tests, and established tolerances for healthy numerical-reference comparisons.

| Case | Required assertion |
|---|---|
| Finite loss, NaN gradient on one rank | Every rank raises. No parameter or optimizer state changes |
| Positive and negative Inf gradients | Same rejection contract |
| Audit zero-residual `sqrt(sum(abs2, ...))` reproducer | Real AD yields finite loss and invalid gradient, then coordinated rejection |
| One invalid rank out of four | Any-positive vote rejects regardless of source rank |
| Finite MPI inputs whose sum overflows | Post-reduction guard rejects before the optimizer |
| Finite tied contributions whose local sum overflows | Canonical-buffer guard rejects |
| Invalid gradient in a late buffer | Earlier parameters and moments remain unchanged |
| Test backend corrupts one rank after a real reduction | Final vote makes every rank reject, including ranks with finite local buffers |
| `update` and `update!` | Both reject without mutating their inputs |
| Shared or read-only caller gradient | Rejection preserves caller data and alias relationships |
| Conditional local absence | Remote invalid contribution rejects on every rank |
| Globally absent leaf with nonzero Adam history | Values, moments, and counters retain skip semantics |
| All gradients absent / no trainable parameters | Correct distinct schedules and unchanged state |
| Frozen present leaf with invalid gradient | Reject according to the retained participation contract |
| Nested, tied, wrapped, and isbits CPU cases | Guard uses existing canonical buffers without changing traversal semantics |
| Empty numeric buffer | Its finiteness result is valid and does not cause a spurious rejection |
| Healthy conditional model | Parameters and complete Adam state match the independent reference |

For overflow tests, prove the operands are finite and the reduced output is non-finite.
NCCL can use different average arithmetic.
Do not require MPI's overflow example to overflow identically under NCCL.
Use a controlled test backend to prove the generic post-reduction branch when native arithmetic does not reproduce it.
Label simulated and native evidence separately.

The test backend must wrap the real collective schedule and inject a numerical result, not replace all communication with a mock.
Its injection must occur before the production guard inspects the reduced buffer.

Add a coordinated catch-and-continue subcase with one subsequent healthy update.
Compare against a reference that omits the rejected update.
This checks that neither the optimizer counter nor collective sequence advanced incorrectly.

## 6. Files and documentation

Expected Flux changes:

- `src/distributed/public_api.jl`: shared validity helper, reduced-buffer scan, coordinated rejection, and wrapper docstrings.
- A focused `test/ext_distributed/nonfinite_gradient_distributedtest.jl`: CPU and native GPU cases.
- Existing conditional and backend tests: updated exact collective expectations.
- Device extension code only if device reduction or stream integration requires it.
- `NEWS.md`: default rejection of non-finite synchronized gradients and the new metadata collective.

Expected thesis changes:

- Shared numerical-guard probe and bounded driver.
- Focused DB-3 ADR before production edits, with the failure mode and the post-reduction rationale.
- Audit 002 resolution entry after acceptance, with evidence links.
- Current context, devlog, failure records, and progress status after each completed gate.

The documentation must state that scalar-loss checks and synchronized-gradient checks protect different boundaries.
It must describe scalar device-to-host validity transfers explicitly.
Do not claim protection against optimizer-internal overflow or arbitrary user exceptions.

## 7. Execution sequence and commands

### Phase A: establish RED evidence

1. Inspect and preserve the current Flux diff and record the source identity.
2. Write the DB-3 ADR and focused success regressions.
3. Run the finite-loss NaN case and the late-buffer preservation case against the existing source.
4. Retain the expected failures as evidence of silent mutation.

### Phase B: CPU implementation and gates

1. Add the post-reduction scan and all-rank vote.
2. Run the focused two-rank CPU matrix.
3. Run the four-rank matrix, including a single invalid non-root rank.
4. Run wrapper compatibility, collective-order, and method-ambiguity checks.
5. Run the dedicated distributed suites at two and four ranks.

Planned commands, from the thesis root, after the new harness exists:

```bash
julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl --group gradient --ranks 2 --backend mpi --outdir artifacts/logs/conditional-wrapper/numerical-guards/gradient-mpi2
julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl --group gradient --ranks 4 --backend mpi --outdir artifacts/logs/conditional-wrapper/numerical-guards/gradient-mpi4
```

The full-suite commands are specified in the DB-1/DB-4 plan.
Run the combined suite once for the final combined source state when no intervening change requires repetition.
These planned commands are not evidence of successful execution.

### Phase C: native GPU gate

After CPU correctness passes, run two-rank native CUDA/NCCL with scalar indexing disabled.
Cover real-AD finite-loss NaN, Inf input, late-buffer rejection, conditional absence, unchanged Adam history, and the healthy reference.
Exercise the asymmetric-result test backend on device buffers without host scalar indexing.
Prove device reduction ordering under the actual NCCL/CUDA versions.

Load the remote skill and server instructions before execution.
Record exact allocation, synchronization, precompilation, and test commands.
A four-rank GPU run adds useful coverage but is not the minimum gate for this focused fix.
Claims must identify the rank counts actually exercised.

### Phase D: combined loss/gradient acceptance

Run the shared harness with `--group all` at two and four CPU ranks and the focused two-rank GPU configuration.
Assert that invalid loss stops before the gradient path and finite loss with invalid gradients stops at the gradient vote.
Retain healthy conditional-reference checks in the combined campaign.

## 8. Cost assessment after correctness

The initial implementation adds one scan of the reduced gradients and one scalar metadata collective per nonempty wrapper update.
CUDA scans can add kernel launches and device-to-host synchronization per buffer.
The actual overhead remains unmeasured.

After correctness gates pass, compare the guarded version with the pinned pre-guard baseline on identical healthy workloads.
Use warm-up, identical update counts, the slowest-rank duration, and separate runs without per-step diagnostic logging.
Record scan strategy, buffer count, gradient bytes, rank count, and device model.
Report absolute step time and relative overhead without changing correctness tolerances.

No performance threshold is invented in this plan.
If measured cost is unacceptable, make a separate decision about batched device reductions or an explicit opt-out.
Do not silently weaken the mandatory-rejection contract to improve timing.

## 9. Completion criteria

- Every rejected update raises on all participating ranks with no unmatched collective.
- Parameters, complete optimizer state, and caller gradients retain their snapshot values.
- Both update APIs and the real-AD audit reproducer pass.
- Reduction-generated invalid results are covered by native or clearly labeled controlled evidence.
- Conditional absence, tied identity, frozen participation, and healthy trajectories retain their contracts.
- Native CUDA/NCCL evidence proves device-safe checks before a GPU-support claim.
- New collective counts, diagnostics, docs, ADR, and artifact metadata agree.
- Unmeasured overhead remains explicitly unmeasured until the post-correctness comparison occurs.

**Exact first implementation action:** record the current Flux diff, then write the DB-3 ADR and a finite-loss NaN regression with populated Adam state.
