# P0 CPU launcher and fixture — 2026-10-31

Repaired c8-mpi/c8-mpi-4 to invoke one parent with explicit rank count and CPU
backend flag (audit F13). Launcher interception regression passes at W=2/4.
Added isolated base CPU environment, setup script, direct acceptance worker and
bounded local driver. Historical GPU/CG paths and both Flux source trees unchanged.

Executed the five upstream primitive files plus the new six-step Descent/Adam
fixture at W=2 and W=4; final driver exit 0. Both BASE_DDP_ACCEPTANCE markers PASS.

W=2: 528 leaf comparisons; max absolute error 2.7755575615628914e-17; hosts ['omarchy'].
W=4: 1056 leaf comparisons; max absolute error 2.2204460492503131e-16; hosts ['omarchy'].

Exact commands from thesis root:

```sh
python3 scripts/multinode/test_launcher.py
julia --startup-file=no --project=scripts/multinode/env -e 'using Pkg; Pkg.develop(path="../flux-multinode"); Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --project=scripts/multinode/env scripts/multinode/setup.jl
julia --startup-file=no --project=scripts/multinode/env scripts/multinode/run_cpu.jl
julia --startup-file=no -e 'Meta.parseall(read("scripts/multinode/acceptance.jl",String)); println("PARSE PASS")'
git diff --check
```

Evidence: artifacts/logs/multinode/p0/{launcher,setup,warm,stage1}.log and
acceptance-{2,4}.json. Environment manifest records resolved versions and selected
Flux development path. JSON records source content hash, manifest hash, MPI library,
placement and every rank's comparisons. Only rank zero writes shared JSON.

Earlier failed/terminated attempts are retained and explained in
wiki/failures/2026-10-31-multinode-p0.md. The final run used existing compiled
modules only, following successful explicit precompile and warm load. No new
cache-build output in final log. Numerical tolerances stayed fixed.

Scope: this is a passing initial CPU harness, not full closure of every stage-1
audit evidence requirement. Diagnostic collective traces are compared, but native
optimizer collective calls are not intercepted. Effective preference provenance,
full loader/shard evidence and explicit integer-step attestation need review.
Two-node CPU and all GPU gates remain unverified; no remote commands ran.

Exact next action: review and close the remaining stage-1 evidence gaps in
scripts/multinode/acceptance.jl before authorizing the stage-2 two-node campaign.

Post-run process snapshot: 0 Julia/MPI processes remain.

Commit preparation: reran `python3 scripts/multinode/test_launcher.py` (PASS).
Staged only P0 edits and evidence; prior migration/presentation work stays local.
The full machine process snapshot stays local; the zero-leftover result is recorded
above. Raw Julia warning whitespace is preserved (see failure record).
