# Conditional wrapper compatibility audit

Independent review (2026-10-02): no blocking findings within the documented CPU/MPI scope.
Combined dedicated suites passed 6/6 at two and four ranks. Checkpoint continuation passed 204 assertions again.
The combined gates recorded as pending in the historical sections are now complete.
Evidence: [review devlog](../../devlog/02-conditional-graph/2026-10-02-conditional-wrapper-phase-h-review.md). Status update (2026-10-07): Phase I–K are complete; see [current.md](../../context/current.md).

Date: 2026-10-01  
Plan: section 12, Phase H, `temp/conditional_graph_wrapper_implementation_plan.md`  
Flux baseline: `../flux-integration`, HEAD `55171e659b2f425da980cbc83177bf5c38f0a7a1`.
Current review: the intentional uncommitted Agent 1–3 corrections are present on that HEAD.

## Verdict

The wrapper supports the tested first-order CPU/MPI training path through Flux and Optimisers.
It is not a drop-in replacement for a model-shaped optimizer tree.
Direct tree consumers require an explicit `.tree` boundary.

**Current implementation:** direct assignment to `Leaf.state` retains supported scalar replacements and tied leaf identity.
Agent 1–2 regression evidence establishes subsequent update correctness for these corrections.
Agent 3 defines the unchanged configuration boundary: rules and frozen flags stay local and must match across ranks.
Root-only checkpoint loading followed by synchronization is not a complete restore.

**Historical findings at clean `55171e65`:** the wrapper discarded replacement scalar state, and the native return lost tied leaf identity.
The original logs and the historical mechanism section remain evidence for those findings.
They do not describe the corrected uncommitted method.

**Checkpoint resume passed with a data-only payload and a fresh backend in a second MPI job.**
Julia Serialization and JLD2 preserved the tested tied arrays, tied leaves, Adam state, rule, and frozen flags.
This result does not establish portability for serialized MPI or NCCL backends.

**Shape decision:** explicit `.tree` paths sufficiently resolve migration for the CPU experimental path.
The extra root remains an architecture cost. Upstream API acceptance remains under review.
The [practical migration guide](../../../temp/phase_h/04-state-shape-migration.md) gives inspection, configuration, and checkpoint recipes.
Full combined two-rank and four-rank gates for the uncommitted corrections remain pending.

### Current targeted continuation evidence

Agent 4 ran the current compatibility probe in two sequential local MPI jobs, with two ranks each.
The write job passed **61/61 per rank**. The fresh-process resume job passed **41/41 per rank**.
Total: **204 assertions**, exit 0 for both jobs, elapsed 17.331 s and 17.823 s.
New logs: `temp/phase_h/04-continuation-{write,resume}-mpi2.log`.
Both Serialization and JLD2 preserve the tested native tree, tied identities, configuration, and Adam history.
The frozen leaf now has nonzero history before serialization, and continuation compares that history explicitly.
The probe also establishes corrected scalar/tied behavior and unchanged rank-local configuration.
These jobs are targeted continuation evidence, not combined full-suite gates.

## Sources and evidence classes

Unless explicitly marked current, source line references identify the original audited HEAD.
The original `H` line numbers identify the historical probe. Agents 1–4 extended that probe.
Current references and exact commands appear in the migration guide.
Source references use these prefixes:

| Prefix | Location |
|---|---|
| `D` | Flux `src/distributed/public_api.jl` at the audited HEAD |
| `T` | Flux `src/train.jl` at the audited HEAD |
| `C` | Flux `test/ext_distributed/conditional_distributedtest.jl` at the audited HEAD |
| `O` | Installed Optimisers 0.4.9, `/home/kurapica/.julia/packages/Optimisers/tMaaf/src/interface.jl` |
| `A` | Installed Optimisers 0.4.9, `/home/kurapica/.julia/packages/Optimisers/tMaaf/src/adjust.jl` |
| `H` | New `scripts/checks/conditional_wrapper_compatibility_probe.jl` |

**Existing evidence:** Phase F ran the conditional tests at two and four ranks on deathstar.
Each rank passed 367 assertions across 30 testsets.
Both dedicated suite runs passed all six child files.
The source hashes match the local audited worktree.
These results come from stored evidence, not new remote runs.

Evidence paths:

- `artifacts/logs/conditional-wrapper/phaseF-conditional-mpi2-deathstar-2026-09-30.log`
- `artifacts/logs/conditional-wrapper/phaseF-conditional-mpi4-deathstar-2026-09-30.log`
- `artifacts/logs/conditional-wrapper/phaseF-suite-nprocs2-deathstar-2026-09-30.log`
- `artifacts/logs/conditional-wrapper/phaseF-suite-nprocs4-deathstar-2026-09-30.log`
- `artifacts/logs/conditional-wrapper/phaseF-metadata.md`

**Existing detector evidence:** Phase G passed 210 assertions with two local MPI ranks.
It checked numerical results, gradient means, collective counts, and collective order.
Its log is `artifacts/logs/conditional-wrapper/phaseG-detector-GREEN-mpi2-2026-09-30.log`.

**Historical runtime evidence at clean `55171e65`:** H ran two bounded local CPU/MPI jobs.
The write job passed 54 assertions per rank.
The resume job passed 35 assertions per rank.
The combined total is 178 passing assertions, with no failed assertions or commands.
Some assertions characterize incompatibilities. A passing characterization does not make that behavior supported.

**Source-only findings** are identified explicitly in the following sections.
The audit did not run the local main suite.

## Compatibility matrix

| Interface | Contract and limitations | Source | Evidence |
|---|---|---|---|
| `Flux.setup` | Returns the wrapper after native setup. Flux retains its mutable-parameter requirement. | `T:142-157`, `D:413-421` | Existing `C:1280-1294,1317-1329`. H constructs tied Adam state through Flux. |
| `Optimisers.setup` | Wraps the inner native tree. Supports native immutable parameter semantics. Direct and rule-chain nesting of distributed rules is rejected. | `D:396-425`, `O:29-58` | Existing `C:831-859,1213-1236` and `optimizer_distributedtest.jl:44-62`. |
| `Flux.train!` | The eager step calls wrapper `update!`. All ranks must take the same number of update calls. | `T:198-252,375-378,425-429` | Existing `C:1317-1356` covers real AD and a nested model. Rank-local non-finite loss handling is a source-only risk. |
| `Optimisers.update!` | Synchronizes first, mutates the inner tree, returns the same wrapper and the inner model result. | `D:434-439`, `O:78-109` | Existing `C:775-859,903-964`. H covers tied Adam state. |
| `Optimisers.update` | Returns a fresh wrapper and copied native tree/model. Retains the backend object. | `D:441-446`, `O:72-76` | Existing `C:860-901`. H checks separate input/output arrays and preserved ties. |
| `Optimisers.adjust` | Delegates numeric, keyword, and same-type rule replacement to the inner tree. Retains the backend. | `D:476-489`, `A:126-149,158-168` | Existing `C:1375-1404`. H adds rule replacement, tied leaves, frozen metadata, and copied moments. |
| `Optimisers.adjust!` | Delegates the same state forms in place and returns the wrapper. Rule replacement retains native type restrictions. | `D:491-498`, `A:118-124` | Existing numeric case. H adds keyword and rule replacement cases. |
| `Optimisers.freeze!` | Whole-wrapper operation delegates. Partial operation requires `state.tree.<path>`. Frozen leaves still participate in gradient communication. | `D:500,579-619`, `O:100-108`, `A:37-38` | Existing `C:1414-1489` compares Adam state and collective counts. |
| `Optimisers.thaw!` | Delegates and returns the wrapper. The preserved native state resumes updates. | `D:502`, `A:46-47` | Existing `C:1414-1489`. |
| `DistributedUtils.synchronize!!` | Current uncommitted method retains supported scalar state and tied leaves. Rules and frozen flags remain local. | Current `src/distributed/public_api.jl:244-298,739-750` | Agent 1–3 scalar/tied/metadata regressions. Agent 4 probe rerun. Historical defects appear separately below. |
| Functors traversal | Only `tree` is a child. Array transforms retain the backend. The extra root breaks model/state structural pairing. | `D:401`, `O:25` | Existing `C:1530-1598`. H checks children, paired-map rejection, and tied identity. |
| Device movement | `Flux.cpu(state)` retains the backend and tied leaves in H. Generic device-like adaptation works in existing tests. | Flux `src/functor.jl:61-85`, `D:401` | Existing simulated adaptation, plus new CPU runtime checks. Native CUDA/NCCL remains untested. |
| Checkpoint serialization | A data-only model/tree payload can resume with a fresh runtime backend. | `D:396-401` and repository checkpoint sources below | H passes Serialization and JLD2 across separate MPI jobs. Whole-backend serialization remains outside the tested scope. |
| Direct state inspection | The public wrapper fields are `backend` and `tree`. It has no model-field, indexing, or iteration forwarding. | `D:396-401` | H checks property names and expected failures for fields, indexing, keys, length, iteration, and zip. |

### Training restrictions

The wrapper accepts only first-order gradient arguments.
Higher-order arguments produce `ArgumentError` on both update paths (`D:427-455`, existing `C:1240-1250`).
Direct updates of Enzyme `Duplicated` models also produce `ArgumentError`.
Setup accepts `Duplicated`, and Flux unwraps the model before its eager update.
Existing tests establish setup and rejection, not a complete distributed Enzyme training run (`C:1280-1294`).

The `Chain` tuple-gradient diagnostic retains Flux's warning and unwraps the first gradient (`D:458-472`, existing `C:1252-1275`).
Calling `update!` on `state.tree` alone bypasses distributed synchronization.
This is a local optimizer call, not a partial distributed update API.

**Source-only training risk:** `T:375-378` skips the update for a non-finite local loss.
`T:244-245` then throws on that rank.
Other ranks can enter collectives and wait indefinitely.
The supported path therefore requires finite losses, equal step counts, and consistent rank participation.
H did not run a deliberately hanging case.

## State-shape incompatibilities and migration

### 1. The state root no longer mirrors the model

Native setup returns a tuple, named tuple, array, dictionary, or root `Leaf`, according to the model.
Distributed setup now returns `DistributedOptimizerState(backend, tree)`.

| Previous consumer | Required migration |
|---|---|
| `state.layers[1].weight` | Use `state.tree.layers[1].weight`. |
| `state.a`, `state[:a]`, `state[1]` | Apply the original field or index operation to `state.tree`. |
| Root-array `state.state`, `state.rule`, `state.frozen` | Use `state.tree.state`, `state.tree.rule`, `state.tree.frozen`. |
| `keys(state)`, `length(state)`, `iterate(state)`, `values(state)`, `pairs(state)` | Inspect the native `state.tree` container. Root leaves remain non-container objects. |
| `zip(state, model)` or positional recursive tree code | Pair `state.tree` with the model and retain native traversal rules. |
| `fmap(f, state, model)` | Pair the model with `state.tree`. Reconstruct the wrapper only after a deliberate state transformation. |
| Functors paths into state | Add the `tree` root component. Paths are no longer identical to model paths. |
| Partial `adjust!`, `freeze!`, `thaw!` | Select the subtree through `state.tree` and call the native operation there. |
| Generic `map`, destructuring, or dispatch restricted to `NamedTuple`/`Tuple`/`Leaf` | Add explicit wrapper handling at the application boundary. |

H executes representative direct-access and generic-zip failures.
The complete operation list also follows from the wrapper definition and its method set.
No forwarding methods were added during this audit.

### 2. Rule and backend locations changed

The old per-leaf implementation stored `DistributedOptimizer` inside each `Leaf.rule`.
The current tree stores the inner rule directly.

| Old per-leaf path | Current path |
|---|---|
| `state.a.rule.opt.eta` | `state.tree.a.rule.eta` |
| `state.a.rule.backend` | `state.backend` |
| Model-shaped state root | `state.tree` |
| Leaf-local distributed `apply!` | Whole-wrapper `update` or `update!` |

The historical source was inspected with `git show fee2c310:src/distributed/public_api.jl`.
Its `Optimisers.init` delegates to the inner rule, while its `apply!` communicates per leaf.
Current `D:423-425` deliberately rejects those per-leaf calls.

### 3. Return conventions require care

Both update APIs return `(state, model)`.
`update!` can return a replacement immutable model despite its name.
Custom loops must retain that model result (`O:89-111`, existing `C:831-859`).
Flux's mutation-based training path retains its existing restrictions.

Wrapper `adjust!`, `freeze!`, and `thaw!` return the wrapper.
Native leaf/tree mutators generally return `nothing` through `foreach` (`A:37-47,118-124`).
Code that uses their return value as a sentinel requires migration.

**Source-only root-leaf nuance:** native `adjust(::Leaf, ...)` shares the existing `Leaf.state` (`A:126-128`).
Native container `adjust` copies writable arrays through Functors (`A:135-148`).
The wrapper inherits this difference because it delegates to `state.tree`.
A root-array model therefore has different non-mutating adjustment aliasing from the named-tuple fixture in H.

### 4. Tree identity is semantic

Native setup caches mutable parameters and gives tied parameters the same `Leaf` (`O:18-25,45-58`).
Native gradient accumulation groups contributions by `Leaf` identity (`O:78-85,118-134`).
The wrapper also deduplicates mutable parameter buffers by parameter identity (`D:579-609`).
Manually rebuilt state must preserve both model ties and leaf ties.
Equal leaf values do not replace shared identity.

H proves that `update`, container `adjust`, Functors copying, `Flux.cpu`, Serialization, and JLD2 preserve the tested ties.
These results do not cover arbitrary hand-written tree transformations or custom serializers.
Model changes after setup also require a compatible new tree.
The state records the original trainable selection (`D:504-519`).

## Historical synchronization defects at clean `55171e65`

This section preserves the original observations and mechanism.
The current direct assignment corrects scalar retention and tied identity. It does not change metadata transport.
Agent reports: `temp/phase_h/01-scalar-state.md`, `02-tied-leaf-identity.md`, and `03-rule-frozen-contract.md`.

`synchronize!!(::Leaf)` reconstructs a leaf with its synchronized state (`D:712-715`).
Tuple and named-tuple methods return reconstructed containers (`D:254-263`).
The wrapper calls this operation but discards the returned tree (`D:718-721`).

Mutable arrays change through shared storage.
Scalar values and immutable replacement objects do not update the original leaf.
This distinction matters for Adam's decay tuple and optimizer step counters.

H deliberately gives the two ranks different moment arrays, decay tuples, learning rates, and frozen flags.
The results are:

| Item | Rank 0 after wrapper synchronization | Rank 1 after wrapper synchronization |
|---|---|---|
| First moment array | `[1, 1]` | `[1, 1]` |
| Second moment array | `[3, 3]` | `[3, 3]` |
| Decay tuple | `(0.5, 0.6)` | `(0.6, 0.7)` |
| Learning rate | `0.01` | `0.02` |
| Frozen flag | `true` | `false` |

Capturing the native tree return restores the scalar tuple to `(0.5, 0.6)` on both ranks.
However, a tied fixture then has `returned.a !== returned.b`.
The original wrapper retains its ties because it retains the original tree.
Replacing the wrapper tree with this native return is therefore not a general migration recipe.

The tie loss exists in the generic tree synchronization path.
The discarded replacement tree is specific to the new wrapper method.
Rule and frozen metadata exclusion exists in the leaf synchronization method.
This audit distinguishes these causes rather than assigning every limitation to the wrapper.

**Source-only additional limits:** generic synchronization has no dictionary-specific method.
A non-isbits dictionary state reaches the fallback and remains unchanged (`D:274-277`).
Custom non-isbits state structs without a method also remain unchanged.
The wrapper's explicit synchronization backend does not replace `state.backend`.
A subsequent update still uses the stored backend (`D:434-445,718-721`).

### Current operational scope

All ranks must start with matching complete optimizer state, rule settings, frozen flags, and model topology.
Rank-local schedule changes can produce different parameters despite identical reduced gradients.
The wrapper does not enforce agreement on these items.
Frozen leaves still communicate, so a frozen-flag mismatch can cause silent divergence rather than a collective-count mismatch.

For the tested checkpoint path, every rank loads the same data payload.
That path restores scalars and metadata without reliance on `synchronize!!`.
Root-only checkpoint loading followed by wrapper synchronization is not a complete optimizer restore procedure.

## Checkpoints and device movement

### Historical checkpoint evidence and current extension

The write job performs one distributed Adam step with a tied parameter and a frozen parameter.
Rank 0 saves `(; model, tree=state.tree)` with both Serialization and JLD2.
It then performs another step and saves the expected state.

The resume job starts new Julia processes and initializes a fresh MPI backend.
Every rank loads each payload and constructs `DistributedOptimizerState(backend, payload.tree)`.
It takes the next distributed step and compares against the uninterrupted result.
The comparisons cover parameters, both Adam moments, decay state, the rule, frozen metadata, model ties, and leaf ties.

The test uses the same package versions, model definition, CPU placement, and two-rank world size across both jobs.
The mean divides each summed gradient by all workers.
The tied parameter contributes through both model occurrences before that mean.
This is continuation equivalence, not a complete training replay guarantee.
The payload does not include random-number state, data position, or schedule counters.
Agent 4 repeated this recipe on the corrected implementation and added nonzero frozen-history comparisons.
The current result is 204 assertions, as recorded in the current evidence section.

### Repository checkpoint formats

`scripts/examples/train_ddp.jl:204-207` saves only `Flux.state(model)` under the JLD2 key `model_state`.
H reproduces that key and restores a preconstructed tied model with `Flux.loadmodel!`.
The parameter values and model ties survive.
Fresh setup does not recover the previous Adam moments or frozen flags.
That format supports model restoration, not exact optimizer continuation.

`scripts/reference/ReferenceLoop.jl:234-273` saves initial/final models, optimizer states, and per-step results through JLD2.
It has no distributed backend replacement procedure or optimizer-state migration version.
H exercises its serialization library and relevant model/tree objects, not the entire reference-result schema.

### Migration procedure

1. Record package versions, the model definition, and the checkpoint schema.
2. Save the model and inner optimizer tree in one data payload.
3. Save random-number state, data position, and schedule counters for complete training replay.
4. Initialize the distributed backend in each new process.
5. Load the same data payload on every rank.
6. Restore model and state arrays to compatible devices.
7. Construct `DistributedOptimizerState(fresh_backend, loaded_tree)`.
8. Check model topology, trainable selection, parameter ties, leaf ties, rule settings, and frozen flags.
9. Resume through the wrapper's update API.

The tested procedure uses complete model objects and their corresponding native trees.
Loading only model weights into a different architecture does not establish tree compatibility.

### Old checkpoints

A native inner Optimisers tree can be wrapped after a compatible model restore.
H proves this case for its current-version Adam fixture.
An old distributed tree contains `Leaf{DistributedOptimizer,...}` objects instead.
Wrapping that tree does not unwrap its rules.
Its next inner update reaches the current `apply!` rejection (`D:425`).

Exact continuation from that historical format needs an explicit migration.
Such a migration must unwrap each rule, preserve state and frozen flags, and cache shared leaves by identity.
Typed `Leaf.rule` fields prevent arbitrary in-place rule replacement (`O:18-22`).
This audit does not implement or runtime-test historical checkpoint conversion.
For model-only recovery, fresh setup discards optimizer history and therefore starts a new optimization trajectory.

### Backend identity and portability

Functors exposes only the wrapper's tree (`D:401`).
Its reconstruction retains the backend object.
`update` and `adjust` also explicitly reuse that backend.
H checks these identities in the current process.
Existing simulated device adaptation checks that state arrays change while the backend remains identical (`C:1548-1559`).

Device adaptation does not select a new communicator or assign a new rank/device mapping.
Changing the execution backend requires explicit runtime construction.
Native GPU movement and NCCL communicator behavior remain untested in H.

Julia's generic serialization and `deepcopy` do not derive their contract from Functors children.
The wrapper physically contains a backend field.
MPIBackend contains a communicator, and NCCLBackend contains both NCCL and MPI runtime objects (`src/distributed/backend.jl:13-39`).
No wrapper-specific serializer or `deepcopy` policy establishes fresh communicator ownership.
H does not serialize, copy, or resume a complete live MPI/NCCL backend.
An in-process object roundtrip alone cannot establish cross-process communicator portability.

## Wrapper versus a proposed Optimisers pre-update hook

| Question | Current wrapper | Proposed whole-model pre-update hook |
|---|---|---|
| State shape | Adds the `tree` root and explicit API delegation. | Can retain native tree shape if runtime context lives elsewhere. |
| Synchronization point | Wrapper update runs before native gradient accumulation/application. | Must run once before per-leaf gradient pruning or updates. |
| Absent gradients | Wrapper inspects all selected parameters before collectives. | Hook must receive the complete model, state selection, and local gradient. |
| Ties and traversal | Wrapper reproduces native traversal semantics. | An official contract can expose native selection and tie information. |
| API cost | Flux owns wrapper methods and migrations. | Optimisers needs an extension contract and runtime-context lifecycle. |
| Checkpoint cost | Data and backend separation is explicit but manual. | The hook still needs a rule for backend persistence and reattachment. |

This hook is a proposal, not an installed Optimisers API.
The installed `_prepare_rule` hook runs only during setup (`O:29-42`).
It does not provide a per-step synchronization point.
A per-leaf `apply!` hook cannot repair conditional collective order after absent-gradient pruning (`O:114-134`).

### Maintainer discussion questions

1. Is a wrapper state acceptable as public Flux API, including explicit `.tree` migration and changed mutator returns?
2. Must `synchronize!!` restore all optimizer state, including scalar counters, rules, frozen flags, and shared leaf identity?
3. Which layer owns reconstruction and identity caching for synchronized optimizer trees?
4. Is the explicit synchronization backend required to match the wrapper backend, or does the API permit intentional differences?
5. Can Optimisers expose a whole-model hook before gradient pruning, with a complete state/model/gradient contract?
6. How does such a hook receive runtime context without inserting communicators into every leaf or a portable checkpoint?
7. Which side owns canonical traversal, trainable selection, isbits occurrences, and tied-gradient accumulation?
8. Does the hook cover both update APIs while preserving non-mutating inputs and native return conventions?
9. How must Flux coordinate non-finite losses and other rank-local step rejection before collectives?
10. What versioned checkpoint format and historical migration policy will maintainers support?

## Historical reproduction and evidence

The command in this section records the original audit invocation.
Its driver overwrites historical logs. For a new run, use the separate Agent 4 runner with a fresh prefix.

From `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`:

```bash
timeout --signal=TERM --kill-after=10s 420s julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_wrapper_compatibility_check.jl
```

The driver uses `MPI.mpiexec()` from the active Flux test environment, as the Phase G detector does.
Each child job has a 180-second timeout and a five-second termination grace period.
Each log records the expanded command, package paths, package versions, assertion totals, exit status, and elapsed time.
Only rank 0 writes shared checkpoint files.
The driver deletes temporary checkpoints after both jobs.

| Historical job | Assertions per rank | Total | Exit | Elapsed | Timeout |
|---|---:|---:|---:|---:|---|
| Write and compatibility characterization | 54 | 108 | 0 | 21.362 s | No |
| Resume in fresh processes | 35 | 70 | 0 | 17.223 s | No |

Historical evidence, preserved unchanged:

- `artifacts/logs/conditional-wrapper/phaseH-write-mpi2-2026-10-01.log`
- `artifacts/logs/conditional-wrapper/phaseH-resume-mpi2-2026-10-01.log`
- `artifacts/logs/conditional-wrapper/phaseH-metadata.md`

**Exact next action:** main runs the combined two-rank and four-rank gates on the corrected working tree with fresh evidence filenames.
Then conduct the Phase I API review with this audit and the migration guide.
The experimental migration decision does not establish upstream acceptance of the wrapper shape or a historical checkpoint conversion policy.
