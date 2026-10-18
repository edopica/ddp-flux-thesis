# Static-scaling Phase 2 implementation review

Date: 2026-10-17

## Follow-up verdict — R1–R3 closed locally

The original reviewer inspected the corrections and reran the acceptance harness on 2026-10-17.
**R1–R3 are closed for the reviewed failure scenarios. No new blocking finding arose in this follow-up.**

- Canonical fixtures: `check_all.jl` passed all 81 checks, exit 0.
- Harness: all six cases passed, including Python positive acceptance and rejection of all three original mutations.
- Python positive acceptance used the existing synthetic PREP. Julia positive acceptance used the canonical real fixtures.
- The GroupNorm mutation failed the semantic check in both languages while its integrity checks passed.
- The corrected remote acceptance on the real Leonardo PREP remains pending SSH access restoration.

The initial harness attempt exceeded the scratch quota during a full synthetic-directory copy.
The corrected invocation reused the existing PREP through hardlinks and used the canonical fixture-only directory for Julia cases.
Details and exact commands: [follow-up devlog](../wiki/devlog/03-static-scaling/2026-10-17-phase2-acceptance-fixes.md#original-reviewer-follow-up).

The [response report](static-scaling-phase2-review-response.md) correctly separates integrity checks from independent source-data reconstruction.
Its recorded limitations do not reopen these findings.

**Exact next action:** restore Leonardo SSH access, sync the corrected checkers, and run `scripts/remote/leonardo/run-phase2-check.sh` on the real PREP.

## Verdict

**Original verdict: changes required in the acceptance checks, three P2 findings.**
The follow-up verdict supersedes this original verdict. The remaining sections preserve the initial review evidence.
The original local acceptance run passes all 75 checks again.
The review found no numerical defect in the saved fixtures.
However, altered scratch artifacts also pass checks that must reject them.

Scope: plan section 4, the shared specification, Phase 2 Python and Julia code,
remote driver source, and the saved campaign evidence.
This review ran local checks only. Remote job outcomes come from the saved evidence.

## R1 — P2: enforce prepared dataset and schedule hashes

Location: `scripts/static_scaling/check_dataset.py:227-267` and `:565-582`.

The checker never reads `dataset_manifest.json` or `schedule_manifest.json`.
It checks structural invariants and regenerates only epochs 0 and 1.
The epoch-independence check regenerates epoch 49 but discards that result.
It does not compare the generated result against the stored epoch-49 array.

Reproduction: copy the regenerated schedule into scratch space.
Change `augmentation[49, 0, 0]` to another valid crop offset without updating the manifest.
The saved file hash then differs from the manifest hash.
Nevertheless, `check_permutation_augmentation` reports all five checks as PASS.
The remaining checks use epoch 0 or fixture epochs 0 and 1, so they cannot detect this change.

Dataset checks have the same integrity gap.
Counts and global pixel statistics cannot detect every change in image identity or spatial pixel order.
Fixture comparisons cover only sampled images.

Impact: acceptance can approve prepared inputs that differ from the frozen artifacts.
This breaks the provenance guarantee for later training runs.

Required correction:

1. Require both manifests for real-data acceptance.
2. Require the expected file entries.
3. Compare each prepared file against its recorded hash before semantic checks.
4. Add a negative check for a valid-range change after epoch 1.

The successful original campaign and independent schedule regeneration remain valid evidence.
This finding concerns the reusable acceptance implementation.

## R2 — P2: enforce fixture integrity in the Julia acceptance entry point

Location: `scripts/static_scaling/check_fixtures_data.jl:99-116`.

The Julia checker never reads `fixture_manifest.json`.
It checks label ranges but does not compare labels against their recorded hashes or expected sample identities.
The full `check_all.jl` entry point inherits this gap.

Reproduction: copy the real fixture directory and change `data_fixture.train_labels[0]` from 0 to 1.
Keep the original fixture manifest unchanged.
Run the full Julia acceptance command against the copied directory and the original initial bundle.

Observed result:

```text
wrong_label_stale_manifest exit= 0
PHASE2_JULIA_DATA_CHECK PASS
PHASE2_JULIA_OPERATOR_CHECK PASS
```

Impact: the local acceptance command approves an incorrect training target after fixture corruption or copy errors.
Its transform and operator comparisons do not establish label identity.
The remote Python fixture check does detect this stale hash, but local Julia acceptance does not run that check.

Required correction:

1. Require the fixture manifest in the Julia entry point.
2. Require the expected file and array entries.
3. Compare file hashes or all array hashes before numerical checks.
4. Add a negative check for an in-range label change with an unchanged manifest.

## R3 — P2: check all GroupNorm initial values

Locations:

- `scripts/static_scaling/check_dataset.py:366-371`
- `scripts/static_scaling/check_fixtures_data.jl:242-250`

Both selectors match `.gn.weight` and `.proj_gn.weight`, with equivalent bias suffixes.
They omit `.gn1.*` and `.gn2.*` in all eight residual blocks.
Thus, each checker examines 4 weights and 4 biases instead of 20 weights and 20 biases.

The positive run exposes the coverage count:

```text
CHECK bundle.gn_initial_values PASS weights==1 (4 arrays) biases==0 (4 arrays)
```

Reproduction: change `stage1.block0.gn1.weight[0]` from 1 to 2 in a scratch bundle.
Update the per-array hashes and bundle hash to represent a self-consistent generator regression.
Both the Python bundle checks and full Julia acceptance still pass.
Hashes establish byte identity but cannot establish the initialization recipe.

Required correction:

1. Select all normalization arrays, including residual `gn1` and `gn2` arrays.
2. Assert exactly 20 weights and 20 biases.
3. Require weights equal to 1 and biases equal to 0.
4. Add a negative check with valid hashes and an invalid residual GroupNorm value.

A separate audit of the original bundle checked all 40 arrays and passed.
The saved bundle does not need regeneration on this evidence.

## What the evidence supports

- Global batches precede contiguous rank sharding.
- The adapters preserve the 99,840-sample prefix and common 160-sample tail at batches 512 and 256.
- Augmentation records use sample identity and epoch, independently of rank.
- The Python and Julia transform comparisons pass on the real saved fixtures.
- The rectangular linear and asymmetric-kernel checks pass, including gradients and the convolution-orientation canary.
- Local `selftest_local.py` passes all five groups.
- Local `check_all.jl` passes all 75 checks with exit 0.
- The saved remote logs distinguish dataset/data acceptance from the local operator check.

Deferring the remote Flux operator check to Phase 3 is consistent with this phase's data-and-fixture scope.
Phase 3 still requires model consumption, GPU numerical comparisons, and optimizer-update comparisons.

## Commands and evidence

Commands ran from the repository root:

```bash
git status --short
command -v julia && python3 -c 'import numpy; print(numpy.__version__)'
TMPDIR=/tmp/opencode python3 scripts/static_scaling/selftest_local.py
JULIA_LOAD_PATH="$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" julia --startup-file=no scripts/static_scaling/check_all.jl --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --bundle-manifest artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/initial_bundle.manifest.json
python3 /tmp/opencode/phase2_review_probe.py
```

The review probe exited 0 because it asserted the observed false-positive behavior.
It used NumPy 2.5.3 and scratch copies only.
The schedule source was `/tmp/opencode/verify-sched`, retained from the prior independent artifact audit.
The probe asserted a hash mismatch against that schedule's manifest after the alteration.

Scratch evidence:

- Probe: `/tmp/opencode/phase2_review_probe.py`
- Artifacts: `/tmp/opencode/phase2-review-611ydppm/`
- Label-case log: `wrong_label_stale_manifest.log` in that directory
- GroupNorm-case log: `wrong_gn_initial_value.log` in that directory

The reproduction descriptions in R1–R3 define the mutations independently of these temporary paths.
No production code or canonical artifact changed during this review.

## Exact next action

Original action, now complete locally: correct R1–R3 and rerun positive acceptance and the three negative cases.
The current next action is the real-PREP remote re-check described in the follow-up verdict.
