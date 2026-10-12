# Conditional wrapper Phase H: compatibility review

Date: 2026-10-01

## Corrective follow-up: four sequential issue agents

The user requested corrections after the initial audit, with one sequential subagent per issue.
Work stopped after Agent 2 at the user's request. Agents 3 and 4 completed after the user resumed the next two issues.

| Issue | Outcome | Report |
|---|---|---|
| Scalar state | Direct `Leaf.state` assignment retains synchronized scalars. RED/GREEN and Adam continuation passed. | `temp/phase_h/01-scalar-state.md` |
| Tied identity | The same correction preserves tied leaves. Both update APIs match independent continuation references. | `temp/phase_h/02-tied-leaf-identity.md` |
| Rule/frozen metadata | Explicit contract retains local configuration. Tests demonstrate divergence before alignment and correct continuation afterward. | `temp/phase_h/03-rule-frozen-contract.md` |
| State-shape migration | Explicit `.tree` paths suffice for the experimental path. Native-tree checkpoints attach fresh backends. Upstream shape acceptance remains pending. | `temp/phase_h/04-state-shape-migration.md` |

Agent 3's focused synchronized file passed **499/499 assertions per rank**, 998 total, at two ranks.
Its full-restore hypothesis intentionally failed before the documentation correction. This is a contract counterexample, not a runtime RED/GREEN fix.
The failure record is `wiki/failures/2026-10-01-phase-h-rule-frozen-contract.md`.

Agent 4's checkpoint jobs passed **204 assertions** across two sequential two-rank jobs.
Write passed 61/61 per rank in 17.331 seconds. Fresh-process resume passed 41/41 per rank in 17.823 seconds.
Both jobs returned 0 without timeout. The frozen leaf has nonzero history, which continuation compares explicitly.
Historical logs remain intact. New logs and exact commands are in the issue reports.

The main agent reviewed the two reports and production diff. Both repository whitespace checks passed.
Flux retains uncommitted source, test, and NEWS changes on `55171e65`.
The audit now separates historical findings from the corrected working tree.

**Exact next action:** review the combined diff and run dedicated CPU/MPI gates at two and four ranks with fresh evidence filenames.
Full combined gates and Phase I acceptance remain pending. Native NCCL and rank-local non-finite-loss handling remain outside this follow-up.

## Initial audit, before corrections

## Result

Phase H is complete as an audit, with compatibility limitations recorded.
The [report](../../audits/02-conditional-graph/001-conditional-wrapper-compatibility.md) covers every requested interface and compares the wrapper with a proposed Optimisers pre-update hook.
Flux remains clean at `55171e659b2f425da980cbc83177bf5c38f0a7a1`.

## Findings

- The tested first-order CPU/MPI training path works through Flux and Optimisers.
- State inspection, subtree operations, and generic tree consumers require explicit `.tree` migration.
- Wrapper `synchronize!!` broadcasts mutable arrays but discards replacement scalar state.
- Native tree synchronization restores scalar state but loses shared `Leaf` identity for tied parameters.
- Leaf synchronization excludes rules and frozen flags. These remain rank-local.
- Serialization and JLD2 continuation passed with model/inner-tree payloads and fresh process-local backends.
- Historical per-leaf distributed checkpoints require explicit migration. Complete live-backend serialization has no portability evidence.
- Source inspection identifies a rank-local non-finite loss risk in `Flux.train!`. No deliberately hanging run occurred.

The report separates wrapper-specific defects, existing native synchronization limitations, runtime evidence, and source-only findings.
Passing characterization assertions record the incompatible behavior. They do not establish compatibility.

## Changes

- Added `wiki/audits/02-conditional-graph/001-conditional-wrapper-compatibility.md`.
- Added `scripts/checks/conditional_wrapper_compatibility_check.jl` and its probe.
- Stored new evidence under `artifacts/logs/conditional-wrapper/phaseH-*`.
- Updated the implementation plan, current context, README progress table, and wiki quickstart.

One sequential subagent performed the detailed audit and runtime characterization.
The main agent reviewed the report, probe, logs, and relevant wrapper methods before updating the checkpoint.

## Commands and evidence

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

```bash
ls artifacts/logs/conditional-wrapper /tmp/opencode && timeout --signal=TERM --kill-after=10s 420s julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_wrapper_compatibility_check.jl
git diff --check
git -C ../flux-integration diff --check
git -C ../flux-integration status --short
git -C ../flux-integration rev-parse HEAD
```

| Job | Assertions per rank | Total | Exit | Elapsed |
|---|---:|---:|---:|---:|
| Write and compatibility characterization | 54 | 108 | 0 | 21.362 s |
| Resume in fresh MPI processes | 35 | 70 | 0 | 17.223 s |

Both jobs used two ranks. Neither job timed out.
Only rank 0 wrote shared checkpoint files. Every rank loaded the same payload for continuation.
The driver deleted temporary checkpoint files after both jobs.

Versions: Julia 1.12.6, Flux 0.16.11, Optimisers 0.4.9, Functors 0.5.3, MPI.jl 0.20.27, and JLD2 0.6.6.
The probe loaded JLD2 through the thesis environment after loading Flux from the integration test environment.
Exact expanded commands, package paths, logs, and scope are in `artifacts/logs/conditional-wrapper/phaseH-metadata.md`.

Existing Phase F and G evidence supplies the previously tested API and collective contracts.
The new jobs cover compatibility gaps and checkpoint continuation.
No command failed. Expected exceptions appear as successful characterization assertions.

## Exact next action

Conduct Phase I user review of the production diff, CPU/MPI evidence, compatibility report, remaining limitations, and proposed Optimisers discussion points.
Decide the synchronization contract and state-shape migration policy before further implementation or native NCCL work.
