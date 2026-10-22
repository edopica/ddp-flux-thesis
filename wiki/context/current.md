# Current context

## Benchmark artifacts committed — 2026-10-22

The static-scaling benchmark worktree was committed: 582 files (15.4 MB) covering
the new remote runners and checkers, the experiment contract, the wiki records,
the reports, and the ten evidence bundles under `artifacts/logs/` and
`artifacts/baselines/`.
The 12 GB already ignored under `artifacts/` stayed ignored (arrays, caches).
Presentation slide renders (`dist/`, `components/test_*`), the stray root
`out.pdf`, and notebook checkpoints are now ignored as build products.
Raw campaign logs were kept because their `SHA256SUMS` and the
`RESULTS.md`/`PROVENANCE.md` documents reference them; dropping them would break
16 document links.
The presentation `Makefile` now targets `presentation.pdf`, the name used by the
revision devlog and the supervisor next action; the byte-identical
`benchmark_presentation.pdf` duplicate was removed.
Details: [devlog](../devlog/03-static-scaling/2026-10-22-artifact-commit-scope.md).

**EXACT NEXT ACTION:** push the commit, then send `presentation.pdf` to the
supervisor.

## Supervisor presentation revised — 2026-10-22

The static-scaling deck was rebuilt for a supervisor audience.
`presentations/static-scaling-benchmark/presentation.pdf` is now 27 pages.
It adds a candidate-comparison slide, removes internal phase/commit/job/OD/Slurm
labels, uses one palette (static Flux red, guarded Flux yellow, PyTorch blue),
and enlarges the charts to fill the slide.
The benchmark numbers and claim boundaries are unchanged.
Details: [devlog](../devlog/03-static-scaling/2026-10-22-presentation-revision.md).

**EXACT NEXT ACTION:** send `presentation.pdf` to the supervisor; if the
narrative is approved, present from that PDF.

## Full campaign claims checked — 2026-10-20

Job `61493236` completed `0:0` in `02:46:11` on `lrdn1300`.
The saved evidence contains 45/45 timing records, 9/9 diagnostics, and campaign/artifact PASS markers.
All 307 remote checksum entries pass. Independent arithmetic reproduces the submitted table.
Batch 256 passed all nine preflight configurations.

The result supports qualified descriptive throughput and strong scaling.
Historical Flux scales best at W4. PyTorch has slightly greater observed speedup at W2.
The first current-W1 repeat took 680.785 seconds, but a cold-cache cause remains unconfirmed.
Diagnostics report aggregate parameter relative L2 of 1.8877–2.0819%, with no equivalence threshold.
The review did not recompute snapshot differences or run new training.

Slurm step `61493236.9` was an operator diagnostic `srun` that polled `/proc` for 60 seconds, not a campaign workload. It exited 2 when the Julia process ended. The campaign evidence is unaffected.
**EXACT NEXT ACTION:** run the OD-14 deterministic-mode sensitivity and a controlled cache experiment before offering stronger claims.
Retain the qualified label and unconfirmed outlier cause in the summary.

Details: [structured report](../../reports/static-scaling-full-campaign-61493236-report.md),
[review](../../reports/static-scaling-full-campaign-61493236-review.md),
[commands](../devlog/03-static-scaling/2026-10-20-full-campaign-claims-review.md).

## Batch-256 launcher corrections — 2026-10-19

The user selected global batch **256** for the urgent full-dataset campaign.
The launcher now runs only the nine batch-256 preflight cases.
The campaign uses 390 measured updates, 20 discarded warmup updates, and five repeats.
The batch-512 fallback path is removed.

The precompile declaration is corrected. Both launcher branches run the new
`ready` checker before timing. Reused decisions require matching source, dataset,
runtime, hardware capacity, CPU policy, and a complete PASS matrix.
Each preflight rank writes its PID and GPU UUID to a separate file.
The memory checker requires repeated timestamped samples for every rank's PID/GPU pair.
The memory result is a sampled peak, not a guaranteed continuous high-water mark.

Both full-dataset runners now use the batch-dependent OD-4 learning-rate schedule.
Batch 256 uses 1,950 warmup updates and 19,500 total updates.
The 390-entry Float32 schedule hashes match across the Julia and Python helpers.

Local gates: six regression tests PASS, checker self-test PASS (21 checks),
shell syntax PASS, Julia parse PASS, and `git diff --check` PASS.
No remote command ran. GPU memory acceptance remains unmeasured.
Details: [devlog](../devlog/03-static-scaling/2026-10-19-batch256-launcher-corrections.md).

**EXACT NEXT ACTION:** resolve the historical worktree identity difference before sync.
Then sync the corrected thesis files and run the four-GPU preflight-only job.
The launcher precompiles both Flux worktrees on the compute node before workloads.
Require the new batch-256 preflight and `FULL_CAMPAIGN_READY PASS` before the campaign.

## Campaign readiness review — 2026-10-19

**NO-GO for the current launcher, including preflight.** Local review reproduced
an unset-variable failure in `precompile_variant` at launcher line 262.
The batch-512 memory rejection also exits before the batch-256 fallback
(lines 631–634). The Python self-test does not exercise these shell paths.

Campaign mode accepts a reused decision's batch before it checks the decision's
PASS status and source identity. Those checks occur after timing and diagnostics.
Memory acceptance also lacks per-rank sample coverage: one process sample can
pass a four-rank case. No remote command ran during this review.

Local checks passed: shell syntax, checker self-test (21 checks), and Julia parse.
These results do not establish launcher readiness.
Details and reproduction commands: [review](../devlog/03-static-scaling/2026-10-19-full-campaign-readiness-review.md).

**EXACT NEXT ACTION:** correct the shell failures and the preflight acceptance gaps.
Then run shell control-flow regressions before the nine-configuration GPU preflight.
Resolve the recorded historical worktree identity difference before any source sync.

## Full-dataset campaign launcher hardened — 2026-10-19

**Status: local hardening only.** The full-dataset launcher
`scripts/remote/leonardo/run-static-full-campaign.sh` implements `all`,
`preflight`, and `campaign` modes behind `CONFIRM_CAMPAIGN=1`. It verifies the
thesis commit and each Flux commit, state, and source-tree hash against pinned
constants, precompiles both Flux worktrees on the compute node, and
writes source/runtime/resource/dataset identity, campaign config, provenance,
preflight decision, timing records, diagnostics, `REPORT.json`, `SUMMARY.json`,
`SHA256SUMS`, and driver markers.

Preflight covers all 9 framework/world combinations (`flux-current`,
`flux-historical`, `pytorch` at W=1,2,4), the full dataset, 20 discarded warmup
plus 20 sustained updates, and external `nvidia-smi` per-process peak sampling.
Batch 512 is selected only if all 9 pass the OD-8 80%-of-64-GiB criterion;
otherwise batch 256 is tested for all 9 and selected only if all 9 pass. A
failed 512 candidate is excluded and no longer aborts the fallback.

The campaign uses the selected batch, K=195 at 512 or K=390 at 256, 5 repeats,
repeat-first (9 configurations per repeat, alternating order on even repeats),
then a separate untimed diagnostics matrix with snapshots.

`scripts/static_scaling/check_full_campaign.py` validates the pinned source
constants, current/historical distinctness, runtime-versus-record
Julia/Python/torch/numpy versions, exact timing and diagnostic record sets,
`eta_sha256` consistency, OD-14 descriptive diagnostics, and
`NUMERICAL_EQUIVALENCE_CLAIM none`. `full_dataset_timing.jl` and
`timing_torch.py` now emit `eta_sha256` for the Float32 learning-rate sequence
for updates `1..K`.

Local gates pass: `bash -n` exit 0, `py_compile` exit 0,
`FULL_CAMPAIGN_SELFTEST PASS checks=21`, and Julia parse `JULIA_PARSE_OK`.
`shellcheck` is not installed locally. The launcher and checker are untracked
in git. **No preflight or campaign ran. No throughput, scaling, memory-margin,
or numerical-equivalence claim exists.**

**Unresolved OD-8 memory risk:** the loader dry runs measured Flux device-used
at 86.34% (current) and 83.14% (historical) of the A100 total at batch 512/W=1,
above the 80% criterion. Batch 512 did not OOM and no 256 fallback ran then.
Peak process memory has not been measured by the new preflight.

**Preflight risk:** the local historical Flux worktree state is
`34005a430cc8be2aa662e0ca59328e07a824231cb9858f52fce3ac388afc330c`
versus the pinned `4af8409c...`. Remote state was not re-verified. The launcher
fails closed on a mismatch; this is a risk, not evidence that the remote tree
drifted.

This checkpoint supersedes the earlier Phase 5 memory-pilot next action below;
OD-8 is still unresolved and is the first preflight question.

**EXACT NEXT ACTION:** sync the thesis and both Flux worktrees, precompile on a
compute node, and submit the preflight-only job with 4 GPUs
(`CONFIRM_CAMPAIGN=1 CAMPAIGN_MODE=preflight`). Inspect the 9-config decision.
Do not launch the campaign until preflight is PASS.

```bash
ssh leonardo 'CONFIRM_CAMPAIGN=1 CAMPAIGN_MODE=preflight sbatch --parsable \
  -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 \
  --cpus-per-task=32 --gres=gpu:4 --time=04:00:00 \
  --job-name=static-full-campaign \
  --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-%j.log \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-campaign.sh"'
```

Evidence: [devlog](../devlog/03-static-scaling/2026-10-19-full-campaign-launcher-hardening.md),
[launcher](../../scripts/remote/leonardo/run-static-full-campaign.sh),
[checker](../../scripts/static_scaling/check_full_campaign.py),
[prior loader validation](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/RESULTS.md).

## Full-dataset loader validated — 2026-10-19

The minimal full-dataset path is implemented for PyTorch, Flux current
`f7b4520c`, and Flux historical `8add0e42`, and it passes the frozen loader
contract. Job `61450014` (`lrdn0667`, `boost_qos_dbg`, 1 node / 1 task /
32 CPUs / 4×A100, `00:14:09`, `COMPLETED 0:0`) printed
`FULL_VALIDATION PASS` after compute-node precompile and:

- `FULL_DATASET_REFERENCE PASS`;
- `FULL_DATASET_PYTHON_CHECK PASS checks=169`;
- `FULL_DATASET_JULIA_CHECK PASS label=flux-current checks=111`;
- `FULL_DATASET_JULIA_CHECK PASS label=flux-historical checks=108`;
- `FULL_DATASET_PYTHON_CHECK PASS checks=177` for both revisions;
- `FIXTURE_TIMING_PATH PASS` (the unchanged fixture path);
- `FULL_DATASET_DRYRUN PASS` for all three configurations.

The loader reproduces the epoch-0 sample order, the W=1/2/4 rank shards,
99,840 ids with the 160-id tail dropped, 195/390 update counts, the one-hot
mapping, and the frozen bundle. The Julia transformed update-1 tensor matches
the Python reference with `max_abs_diff=0.0` at W=1, 2, and 4. The three
1-update, batch-512, W=1 dry runs had finite loss
`5.492467880249023`; they are validation only, not benchmark results.

The dry runs expose a memory risk: Flux device-used was
`58,796,081,152 / 68,099,571,712` bytes (`86.34%`) for current and
`56,615,043,072 / 68,099,571,712` (`83.14%`) for historical. This
`total - free` counter includes context/allocator caches, but it is above the
proposed OD-8 80% margin. PyTorch max allocated was `15,394,088,960` bytes
(`22.61%`), max reserved `24,540,872,704` (`36.04%`). Batch 512 did not OOM;
no fallback to 256 was performed.

Evidence: [RESULTS](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/RESULTS.md),
[COMMANDS](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/COMMANDS.md),
[devlog](../devlog/03-static-scaling/2026-10-19-full-dataset-loader-validation.md),
[failure note](../failures/2026-10-19-full-loader-validation-driver.md),
[Slurm log](../../artifacts/logs/leonardo/2026-10-19-static-full-loader-validation-61450014.log).

**EXACT NEXT ACTION:** run the Phase 5 larger-batch memory pilot under OD-14,
measure peak process memory at batch 512/W=1 (and the W=2/4 matrix), and record
OD-7/OD-8 before any full-dataset campaign. The campaign command is documented
in the devlog but was **not** executed. Do not cite the dry-run timings as
throughput results.

## OD-14 recorded — 2026-10-19

The [OD-14 decision](../../experiments/static-scaling/OD-14-benchmark-acceptance.md)
records qualified descriptive benchmark claims under the existing FP32 policies.
Short-horizon correctness and long-run diagnostics remain separate.
No long-run numerical-equivalence threshold is adopted from the observed envelope.
Stronger equivalence or training-quality claims require prospective criteria.
The final Phase 6 report requires a deterministic-mode sensitivity in both frameworks.
Historical helper drift and missing runtime evidence remain explicit limitations.

The specification and TOML policy now reflect this decision.
Policy requirements do not imply automatic enforcement by current runners.
The printed diagnostic has parameter `rel_l2=0.00225564012` at Flux-versus-PyTorch
W=2, slightly above the earlier summary's `0.00224`.

**EXACT NEXT ACTION:** run the Phase 5 larger-batch memory pilot under OD-14.
Record OD-7 and OD-8 before the full-dataset campaign.

## Fixture timing review — 2026-10-19

Campaign `61314790` retains a **qualified timing PASS**. The reviewer accepted
the medians and scaling ratios and found no timing-boundary, sample-count, or
gradient-averaging defect. The four P2 findings and the response are recorded in
[REVIEW-RESPONSE.md](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md).

The checker was hardened to require the exact schema and precision metadata,
non-empty job and node, exact fixture/bundle hashes, complete state hashes, all
36 source-snapshot hashes, the required driver lines, and `provenance.json`
schema/source/policy/qualifications. Its self-test prints
`TIMING_ARTIFACT_SELFTEST PASS checks=13` (one positive and 12 mutations).

| Framework | W | median_s | images/s | speedup | efficiency |
|---|---|---:|---:|---:|---:|
| Flux | 1 | 5.483059 | 656.568 | 1.0000 | 1.0000 |
| Flux | 2 | 5.714868 | 629.936 | 0.9594 | 0.4797 |
| Flux | 4 | 5.664201 | 635.571 | 0.9680 | 0.2420 |
| PyTorch | 1 | 8.203725 | 438.825 | 1.0000 | 1.0000 |
| PyTorch | 2 | 5.258007 | 684.670 | 1.5602 | 0.7801 |
| PyTorch | 4 | 4.542426 | 792.528 | 1.8060 | 0.4515 |

Flux remains flat across W=1/2/4; PyTorch scales `1.56x` at W=2 and `1.81x` at
W=4 on this fixture.

**Review qualifications and follow-up:**

- Execution policies differ. PyTorch uses deterministic cuDNN
  (`benchmark=False`), TF32 off, and `float32_matmul_precision="highest"`; Flux
  uses strict FMA candidate filtering without a determinism filter. Flux DDP
  includes synchronized gradient-finiteness checks inside the timed window;
  PyTorch has no such guard in the timed loop. The cross-framework timing is
  descriptive, not a communication-only isolation.
- The 300-update final-state snapshot diagnostic is complete and descriptive
  (job `61366015`, `lrdn2256.leonardo.local`, `COMPLETED 0:0`, elapsed
  `00:13:26`; 6/6 JSON records and 6/6 snapshots; `STATIC_TIMING_numerics PASS`
  and `TIMING_DRIVER_EXIT=0`). All arrays are finite; `beta_power` is exact
  (62/62); no `param`, `m`, or `v` array is exactly equal. The envelope is
  `param` `max_abs` up to `0.0333` and `rel_l2` up to `0.00224`, `m` `rel_l2`
  up to `0.0485`, and `v` `rel_l2` up to `0.000333`. No equivalence rule is
  applied; `NUMERICAL_EQUIVALENCE_CLAIM none`. Do not claim correctness,
  equivalence, or an error bound. Flux produced five distinct final hash sets
  per configuration across campaign repeats; PyTorch one. This is consistent
  with Flux's non-deterministic FMA-compatible algorithm selection, not
  evidence of an incorrect result.
- The two loaded shared helpers differ from the Phase 4 accepted snapshot:
  `phase4_common.jl` (accepted `32d85954...`, timing `30892f68...`) and
  `phase4_torch.py` (accepted `75c19051...`, timing `7620d9d1...`). The drift
  is recorded, not asserted equivalent.
- `runtime-env.txt` capture is present for job `61366015` and records Julia
  1.12.6, Flux 0.16.11, CUDA.jl/cuDNN.jl 6.2.2, Python 3.11.17, torch
  2.9.1+cu126, and numpy 2.4.6 with executable/module paths; no
  `RUNTIME_ENV_CAPTURE_FAILED` line. Campaign `61314790` predates the capture
  and retains only the copied manifests and requirements freeze.
- The earlier Leonardo SSH publickey rejection (2026-10-19) is retained as
  history ([failure](../failures/2026-10-17-leonardo-ssh-publickey.md)); job
  `61366015` subsequently ran and completed.

The fixture evidence is bounded training-only: not full-dataset, not
end-to-end, not the Phase 5 memory pilot, and not learning curves. Timing
results, the descriptive snapshot, and provenance: [RESULTS.md](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/RESULTS.md),
[300-update snapshot results](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md),
[provenance.json](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/provenance.json),
[CAMPAIGN.md](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/CAMPAIGN.md),
[devlog](../devlog/03-static-scaling/2026-10-19-timing-review-response.md),
[failure record](../failures/2026-10-19-timing-numerics-attempts.md).

**Next-action update:** OD-14 is recorded in the section above. The Phase 5 larger-batch memory pilot is next.

## Fixture GPU timing campaign record — 2026-10-19 (qualified by the review above)

The bounded fixture GPU timing campaign (job `61314790`, `static-timing-campaign`,
`lrdn3064.leonardo.local`, `boost_qos_lprod`, 1 node / 1 task / 32 CPUs /
4× A100-SXM-64GB, `02:00:00`) is **complete and PASS**, qualified by the
2026-10-19 review above. It ran from
`2026-10-18T23:06:02Z` to `2026-10-18T23:48:09Z` (approx.; `00:42:07` elapsed),
and Slurm reports `COMPLETED` exit `0:0`. The compute-node precompile printed
`FLUX_PRECOMPILE_OK`; the driver printed `STATIC_TIMING_campaign PASS` and
`TIMING_DRIVER_EXIT=0`. All 30/30 JSON records are present (5 repeats × 6
configurations), with no failed runs. The artifact checker prints
`TIMING_ARTIFACT_CHECK PASS`.

Fixture: global batch 12, alternating two saved Phase 2 batches, 20 warmup +
300 measured training-only updates, strict FP32 FMA/TF32 off, native NCCL DDP
at W=2,4.

| Framework | W | median_s | images/s | speedup | efficiency |
|---|---:|---:|---:|---:|---:|
| Flux | 1 | 5.483059 | 656.568 | 1.0000 | 1.0000 |
| Flux | 2 | 5.714868 | 629.936 | 0.9594 | 0.4797 |
| Flux | 4 | 5.664201 | 635.571 | 0.9680 | 0.2420 |
| PyTorch | 1 | 8.203725 | 438.825 | 1.0000 | 1.0000 |
| PyTorch | 2 | 5.258007 | 684.670 | 1.5602 | 0.7801 |
| PyTorch | 4 | 4.542426 | 792.528 | 1.8060 | 0.4515 |

Flux is flat across W=1/2/4; PyTorch scales `1.56x` at W=2 and `1.81x` at W=4
on this fixture. These are bounded fixture measurements only.

This campaign supersedes canceled partial campaign `61305371` (23/30 records),
which remains non-evidence; smoke `61305319` remains the preflight. GPU timing
was previously deferred, and the user requested it on 2026-10-19.

It is **not** the full-dataset campaign, **not** end-to-end timing (no input
staging or H2D in the window), **not** the Phase 5 larger-batch memory pilot,
and **not** learning curves. Do not claim more than measured.

Evidence: [RESULTS.md](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/RESULTS.md),
[CAMPAIGN.md](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/CAMPAIGN.md),
[devlog](../devlog/03-static-scaling/2026-10-19-fixture-timing-campaign.md).
Local artifacts: `artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/`
(30 JSON, `driver.log`, manifests, source snapshot; 2.4 MB); Slurm log
`artifacts/logs/leonardo/2026-10-19-static-timing-campaign-61314790.log`.

**Historical next action (superseded by the review section above):** run the
Phase 5 larger-batch memory pilot before the full-dataset campaign, and decide
whether to add the end-to-end timing view.

## Phase 4 GPU accepted and finalized — 2026-10-18

The user approved finite absolute numerical error `<1e-3`, with exact structural and rank checks.
The fresh artifact recheck passes all 202 categories for Flux/PyTorch at W=1,2,4.
It finished at `2026-10-18T21:44:16Z`; the canonical report SHA-256 is
`a5ca161c3e7b7b57a9d5b3d24c7ddb9236ce613554db5ab95f391976a1fa7f8c`.
The checker self-test passes 36 checks.
CPU validation is deferred. The previous CUDA-only trajectory overrides are removed.
The original strict same-framework rule did not pass and is not the acceptance claim.
Evidence: [Phase 4 results](../../artifacts/logs/static-scaling/2026-10-18-phase4/RESULTS.md).

Phase 4 is finalized for the GPU scope.
GPU timing was the next phase; that statement is superseded by the completed
fixture campaign above.
The preflight smoke (`61305319`) and canceled partial campaign (`61305371`) are
not Phase 4 acceptance evidence. Campaign `61305371` was canceled at user
request after 23 of 30 records.
The full-dataset campaign and larger-batch memory pilot remain pending.
**Exact next action (historical; superseded by the fixture timing section above):**
review the Phase 4 changes; start timing only on explicit request.

## Historical Phase 4 handoff — superseded — 2026-10-18

Phase 4 is **not accepted**. The Flux runner, independent checker, and remote
drivers are implemented and pass local CPU checks; the PyTorch DDP runner is
missing. The remaining Flux issue is the `u3` same-framework comparison
(`grads.u3` ~4e-4, `m.u3` ~4e-5) caused by trajectory sensitivity after two
updates; an independent shard-arithmetic reference is the recommended control.
Leonardo SSH publickey and deathstar DNS were both unavailable at handoff.
Do not rerun the full local CPU matrix; use deathstar/Leonardo.

Read the complete state, evidence, and exact next actions in
[`temp/phase4-handoff.md`](../../temp/phase4-handoff.md). Details:
[devlog](../devlog/03-static-scaling/2026-10-18-phase4-implementation-handoff.md),
[failure note](../failures/2026-10-18-phase4-remote-access.md).

**Exact next action:** implement `scripts/static_scaling/phase4_torch.py`,
then add the shard-arithmetic reference mode and resolve the Flux `u3`
comparison before any remote acceptance run.

## Focused R1/R2 review (2026-10-18): GO for Phase 4

Focused R1/R2 review (2026-10-18): **GO for Phase 4**. Source inspection and
retained evidence revealed no blocking defect. Fresh local hash checks passed
for 23 evidence files and matched all seven scripts in the recorded provenance.
See the [review record](../devlog/03-static-scaling/2026-10-18-r1-r2-precision-policy-fixes.md#focused-phase-4-readiness-review-2026-10-18).

## Phase 3 acceptance-review fixes verified — 2026-10-18

R1/R2 are hardened and verified remotely; Phase 3 acceptance is retained. The
[acceptance review](../../reports/static-scaling-phase3-acceptance-review.md)
found no evidence of invalidating the accepted run, which used 32 FMA choices
and zero fallbacks. The independent-review hardening found no blocking defect
and closed the identified reuse risks.

- Canonical precompile: job `61187225` (`lrdn0241`, UTC
  `12:20:59`-`12:24:36`), `FLUX_PRECOMPILE_OK`, exit 0. Use
  `precompile-phase3.sh`; the default `precompile.sh` is the wrong target for
  Leonardo (root `Manifest.toml` excluded, compute nodes offline).
- Focused precision regression: job `61188854` (`lrdn1845`, UTC start
  `2026-10-18T12:24:49Z`, driver `12:24:54Z`, exit 0),
  `PHASE3_PRECISION_REGRESSION PASS`; fma PASS with `fallback=0`, current
  negative control as expected at `~2.932134e-4`, policy 10/10 PASS, 0 FAIL.
- Full Phase 3 recheck on the hardened helper: job `61191281` (`lrdn2911`,
  UTC start `2026-10-18T12:30:16Z`, driver `12:30:21Z`, elapsed `00:07:52`,
  exit 0), all four summaries PASS, 337/337 exact-equal to the preserved
  reference (`max_abs_diff=0`), 32/32 cuDNN choices `CUDNN_FMA_MATH` with
  `fallback=0`, zero violations in every category, and no
  `Precompiling packages...` line.

| Category | Max abs diff | Violations |
|---|---:|---:|
| eta | 0 | 0 |
| activations | 1.56164169e-05 | 0 |
| logits | 1.7285347e-06 | 0 |
| loss | 4.76837158e-07 | 0 |
| grads.u1 | 3.02582979e-04 | 0 |
| grads.u3 | 6.08410686e-04 | 0 |
| params.u3 | 4.77954745e-06 | 0 |
| m.u3 | 6.32628798e-05 | 0 |
| v.u3 | 2.32830644e-07 | 0 |
| beta_power.u3 | 0 | 0 |
| step | reference 3 = Flux 3 | 0 |

Reference identity: NPZ SHA-256
`97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783`,
manifest SHA-256
`bfd979e2d575e24004860c07c216779325923b73e9873bb063c899acab04109a`;
337/337 exact-equal to the preserved attempt-3 archive, independent check
PASS, remote/local SHA-256 3/3 precision and 5/5 Phase 3. Source identity is
generation-time: thesis state
`da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199`, Flux
`f7b4520c...` state `e3b0c442...` (clean); later doc edits move the live
worktree hash.

Evidence: [precision hardening](../../artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/),
[full recheck logs](../../artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/),
[recheck reference](../../artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/),
[response report](../../reports/static-scaling-phase3-acceptance-review-response.md),
[devlog](../devlog/03-static-scaling/2026-10-18-r1-r2-precision-policy-fixes.md).

R3 is recorded as open OD-14, a Phase 6 benchmark-policy decision with a
proposed default, not a Phase 4 blocker. R4 is corrected in the response
report: the "no sign flips" claim is withdrawn and the zero/nonzero check is
whole-leaf L2, not elementwise. The calibrated tolerances remain specific to
the cross-framework fixture, not automatic Phase 4 DDP tolerances. The
independent delta review's non-blocking reuse/test gaps are listed in the
[devlog](../devlog/03-static-scaling/2026-10-18-r1-r2-precision-policy-fixes.md#deferred-non-blocking-findings-2026-10-18);
none blocks Phase 4.

Leonardo access currently works. The earlier transient local DNS/network drop
did not affect the final hardened evidence; it is not evidence of a key
change. The superseded intermediate runs remain under
`artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-fixes/` and
`artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-recheck/`.

**Exact next action:** start Phase 4 distributed correctness at two and four
ranks per `temp/static-graph-gpu-scaling-implementation-phases.md` section 6.

## Phase 3 accepted — 2026-10-18

Static-scaling Phase 3 is **accepted**. The final matched single-GPU
Flux/PyTorch correctness run passed the frozen FP32/TF32-off contract and the
frozen OD-10 rule on the final source snapshot.

| Item | Value |
|---|---|
| Final driver | job `60956582`, node `lrdn0165`, UTC `2026-10-18T01:25:21Z`, exit 0 |
| Precompile | job `60954207`, node `lrdn3362`, `FLUX_PRECOMPILE_OK` |
| Precision regression | job `60954547`, node `lrdn2211`, `PHASE3_PRECISION_REGRESSION PASS` |
| Optimizer probe | job `60955690`, node `lrdn0279`, `PHASE3_OPTIMIZER_PROBE PASS` (69 checks, 34/34 controls rejected) |
| Source identity | thesis `3088f61020137912af32d9c969e87ea9ae2a71c3`, state `4a28adb1...`; Flux `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`, state `e3b0c442...` (clean) |

All four required summaries PASS: `PHASE3_REFERENCE PASS`,
`PHASE3_REFERENCE_CHECK PASS`, `PHASE3_FLUX_CHECK PASS`,
`PHASE3_REMOTE_CHECK PASS`. No later log contains
`Precompiling packages...`.

| Category | Status | max abs diff | violations |
|---|---|---:|---:|
| eta | PASS | 0 | 0 |
| activations | PASS | 1.56164169e-5 | 0 |
| logits | PASS | 2.2649765e-6 | 0 |
| loss | PASS | 4.76837158e-7 | 0 |
| grads.u1 | PASS | 3.02568078e-4 | 0 |
| grads.u3 | PASS | 6.21065497e-4 | 0 |
| params.u3 | PASS | 5.93136065e-6 | 0 |
| m.u3 | PASS | 7.2401017e-5 | 0 |
| v.u3 | PASS | 2.32830644e-7 | 0 |
| beta_power.u3 | PASS | 0 | 0 |
| step | PASS | ref 3 = Flux 3 | 0 |

All 32 cuDNN choices are `CUDNN_FMA_MATH(3)` with `fallback=0`
(`PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0`). The rule printed
by the checker is the frozen base `1e-5`/`1e-4` plus only
`grads.u1 atol=1e-3`, `grads.u3 atol=2e-3`, and `m.u3 atol=2e-4`, each with
`rtol=1e-4`. The final reference is byte-identical to the preserved attempt-3
reference (337/337 exact, max diff `0.0`); the independent NumPy checker
passes locally (`PHASE3_REFERENCE_CHECK PASS arrays=337`), remote/local
SHA-256 matches 15/15, and `sha256sum -c` passes.

Precision cause and correction: stock cuDNN.jl maps
`PEDANTIC_MATH -> CUDNN_DEFAULT_MATH` (TF32 permitted), and cuDNN.jl 6.2.2's
Float32 autotuner skips the math-type check. The isolated stage-2 convolution
reproduced the failure; explicit `CUDNN_FMA_MATH` removed it. The reusable
`scripts/static_scaling/precision.jl` now installs an FMA-aware
`math_mode`/chooser and clears the algorithm caches. The residual backward
differences are consistent with FP32 backend/algorithm differences, including
possible sensitivity near nonlinear boundaries; they are covered by the frozen
OD-10 floors with at least 2x margin. The controls support that explanation
but do not isolate every residual to backward-kernel rounding.

Evidence: [results](../../artifacts/logs/static-scaling/2026-10-18-phase3-final/RESULTS.md),
[commands](../../artifacts/logs/static-scaling/2026-10-18-phase3-final/COMMANDS.md),
[provenance](../../artifacts/logs/static-scaling/2026-10-18-phase3-final/PROVENANCE.md),
[SHA-256 comparison](../../artifacts/logs/static-scaling/2026-10-18-phase3-final/SHA256-COMPARISON.txt),
[review response](../../reports/static-scaling-phase3-review-response.md),
[devlog](../devlog/03-static-scaling/2026-10-18-phase3-final-acceptance.md).
The hardened Phase 2 prerequisite remains
[`60951247`](../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/PROVENANCE.md).

Benchmarks are **not** part of Phase 3. The later acceptance-review R1/R2
fixes and the hardened full recheck are recorded in the top section. **Exact
next action:** start Phase 4 distributed correctness at two and four ranks per
`temp/static-graph-gpu-scaling-implementation-phases.md` section 6.

## Phase 3 Step 5 complete: hardened Phase 2 re-check PASS, decisions frozen — 2026-10-18 (historical; superseded by acceptance above)

The hardened Phase 2 checkers ran against the real Leonardo `$PREP` and
fixtures for the first time after the R1-R3 hardening. Job `60951247`
(`lrdn0171`, `boost_qos_dbg`, 1 node, 1 task, 8 CPUs, `--gres=none`,
`COMPLETED 0:0`, elapsed `00:00:56`, UTC start `2026-10-18T00:44:27Z`)
printed `PHASE2_DATASET_CHECK PASS` (66 CHECK lines), `PHASE2_JULIA_DATA_CHECK
PASS` (47 CHECK lines), `python_check_exit=0 julia_check_exit=0`, and
`PHASE2_REMOTE_CHECK PASS`; driver exit 0. Total 113 PASS, 0 FAIL, 0 SKIP.
The hardened gates exercised the real manifests (`required=6 listed=6`,
`checked=2`), regenerated all 50 schedule epochs bitwise in 21.21 s, checked
20/20 GroupNorm arrays, and validated 35/35 fixture array hashes. Inputs were
not modified; checker hashes on the remote tree equal the local hardened
sources. Evidence:
[results/provenance](../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/PROVENANCE.md),
[commands](../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/COMMANDS.md),
[`SHA256SUMS`](../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/SHA256SUMS),
[devlog](../devlog/03-static-scaling/2026-10-18-phase3-step5-decisions.md).
This closes the Phase 2 remote prerequisite for Phase 3.

OD-2, OD-3, OD-4, and OD-11 are now recorded with user authority (OD-10 was
recorded 2026-10-17): AdamW `lr=1e-3`, `wd=1e-4`, `betas=(0.9,0.999)`,
`eps=1e-8`; decay only on the 21 convolution/classifier-weight leaves
(`lambda=1e-4`), no decay on the 41 GroupNorm/bias leaves (`lambda=0`); 5
warm-up epochs and a 50-epoch budget indexed by global update (`W=975`,
`T=9750` at batch 512; `W=1950`, `T=19500` alternate for Phase 5); explicit
`Zygote.withgradient` + `Optimisers.update!` with communication outside AD
(64 collectives per update). `open-decisions.md`, `specification.md`, and
`experiment.toml` are consistent; both Python `tomllib` and Julia `TOML`
parse the config. OD-7, OD-8, OD-9, OD-12, and OD-13 remain open with
unchanged proposed defaults.

Phase 3 status at this checkpoint: Steps 1-4 complete; Step 3c acceptance
PASS under the recorded OD-10 rule (job `60942933`); Step 5 complete.
Step 6 subsequently passed on 2026-10-18; see the acceptance section above.

**Exact next action (historical; completed in the acceptance section above):**
execute Step 6 of the Phase 3 handoff: sync the final source snapshot,
precompile on a Leonardo compute node, run
`scripts/remote/leonardo/run-phase3-check.sh` in a fresh `OUT`/`LOG_DIR`,
require all four PASS summaries and driver exit 0, then copy the reference,
manifest, logs, runtime records, and probe results home and run the
independent NumPy checker against the copied reference.

## Phase 3 Step 4 optimizer-contract probe PASS — 2026-10-17

The compact cross-framework probe passes on Leonardo (job `60948431`,
`lrdn0171`; CPU-only, 1 node, 8 CPUs, `--gres=none`). It asserts the real
training-path AdamW configuration (Flux 62 leaves = 21 decay with
`lambda=1e-4` / 41 no-decay with `lambda=0`; PyTorch two groups 21/41 with
`weight_decay=1e-4`/`0.0`), runs three-update controlled sequences with
nonzero `m`/`v` at probe `eta=1e-2`, includes `1e-9`/`1e-12` gradients for
epsilon placement, and checks the schedule at `1`, `W`, `W+1`, and `T` plus
out-of-range rejection.

Result: `PHASE3_OPTIMIZER_PROBE PASS`; 69 checks; 34/34 negative controls
rejected (omitted update, wrong decay scale `1e-3`, omitted decay, decay on
the no-decay leaf, wrong epsilon placement, wrong bias correction, warm-up
formula at `W+1`; weakest ratio 5.10). Flux/PyTorch/oracle final-update
maximum absolute differences are `5.38e-08..1.07e-07`; the cross-framework
maximum is `1.19e-07`. Schedule values are `eta(1)=1.0256410256410257e-06`,
`eta(W)=1e-3`, `eta(W+1)=9.9999996795609018e-04`, `eta(T)=0`; both frameworks
reject `t=0` and `t=T+1`. The probe rule `1e-7 + 5e-7*max(|a|,|b|)` is
probe-specific; the Phase 3 acceptance rule is untouched. The existing
optimizer probe is included unchanged, so its constructor and beta-power
checks still run. No training code, schedule, precision policy, tolerance,
or Flux source changed; no commit.

Evidence:
[results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/RESULTS.md),
[commands](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/COMMANDS.md),
[provenance](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/PROVENANCE.md),
[controlled cases](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/controlled-cases.txt),
[devlog](../devlog/03-static-scaling/2026-10-17-phase3-step4-optimizer-probe.md).

**Exact next action:** run Step 5 of the Phase 3 handoff: locate evidence
for the hardened Phase 2 remote re-check against the real `$PREP`; if it is
absent, run `scripts/remote/leonardo/run-phase2-check.sh` with the corrected
checkers. Then finish the decision record for OD-2, OD-3, OD-4, and OD-11
(OD-10 is already recorded 2026-10-17) in `open-decisions.md`,
`specification.md`, and `experiment.toml`, asking the user for approval where
a decision still requires authority. Do not start Step 6 yet.

## Phase 3 Step 3c acceptance PASS under recorded OD-10 rule — 2026-10-17

The user approved the OD-10 revision from the numerical calibration. The
elementwise rule stays `|a - b| <= atol + rtol * max(|a|, |b|)` with the base
`atol = 1e-5`, `rtol = 1e-4`. Only three backward categories override the
absolute floor with the same `rtol = 1e-4`:

| Category | atol | rtol |
|---|---:|---:|
| `phase3.grads.u1` | 1e-3 | 1e-4 |
| `phase3.grads.u3` | 2e-3 | 1e-4 |
| `phase3.m.u3` | 2e-4 | 1e-4 |

`check_phase3_flux.jl` now has matching CLI defaults
(`--grads-u1-atol`, `--grads-u3-atol`, `--m-u3-atol`), validates them,
prints `PHASE3_FLUX_CHECK_CONFIG` and `PHASE3_ACCEPTANCE_RULE`, and applies
them only to the three named categories. `experiment.toml`,
`specification.md` section 11, and `open-decisions.md` (OD-10 recorded
2026-10-17) are consistent. OD-2/3/4/11 remain open. Step 3b is superseded
by the approved calibration, not by a silent tolerance change.

Precompile job `60941770` (`lrdn2500`, `FLUX_PRECOMPILE_OK`). Full driver
job `60942933` (`lrdn0129`) ran in a new root
`$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-step3c/` with no
`ATOL`, `RTOL`, or category override. All four summaries PASS and the
driver exits 0:

| Category | Status | max_abs | violations |
|---|---|---:|---:|
| eta | PASS | 0 | 0 |
| activations | PASS | 1.56164169e-5 | 0 |
| logits | PASS | 1.66893005e-6 | 0 |
| loss | PASS | 4.76837158e-7 | 0 |
| grads.u1 | PASS | 3.02597880e-4 | 0 |
| grads.u3 | PASS | 5.43192029e-4 | 0 |
| params.u3 | PASS | 4.77582216e-6 | 0 |
| m.u3 | PASS | 6.515719e-5 | 0 |
| v.u3 | PASS | 2.32830644e-7 | 0 |
| beta_power.u3 | PASS | 0 | 0 |
| step | PASS | ref 3 = Flux 3 | 0 |

All 32 cuDNN choices are `CUDNN_FMA_MATH(3)` with `fallback=false`; all 225
candidate math types are FMA. Per-leaf relative L2 (supplementary):
`grads.u1` 1.998759e-3, `grads.u3` 4.288785e-3, `m.u3` 2.571253e-3; no
zero/nonzero disagreements. The new reference is again byte-identical to
the preserved attempt-3 reference (`97e5007c...`), 337/337 arrays
exact-equal; the independent NumPy checker passes locally.

Evidence: [results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3c/RESULTS.md),
[provenance](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3c/PROVENANCE.md),
[commands](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3c/COMMANDS.md),
[algorithm choices](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3c/algorithm-choices.txt),
[devlog](../devlog/03-static-scaling/2026-10-17-phase3-step3c-od10-acceptance.md).

**Exact next action (completed; see the Step 4 section above):** run Step 4
of the Phase 3 handoff: compact cross-framework optimizer-contract probe
(decay/no-decay leaves, nonzero moments, tiny-gradient epsilon placement,
negative controls) and schedule boundary checks at `1`, `W`, `W+1`, and `T`.

## Phase 3 numerical calibration: independent-backend FP32 envelope — 2026-10-17

The backward residual is ordinary FP32 algorithm/reduction variation, not an
outlier that suggests a Flux backward defect. A new PyTorch CPU reference
(job `60912525`, node `lrdn3450`) was generated with the same generator,
fixture, bundle, three updates, and provenance as the preserved GPU reference
(`phase3_reference.npz` SHA-256 `a7ff467b...`, manifest `device="cpu"`). The
full Flux checker ran on CPU against the preserved PyTorch GPU reference and
against the new CPU reference (job `60912664`, node `lrdn3450`).

| Comparison | grads.u1 | grads.u3 | m.u3 |
|---|---:|---:|---:|
| PyTorch CPU vs PyTorch GPU (same generator) | 3.021732e-4 / 14,365 violations / 1.84e-3 leaf L2 | 4.099458e-4 / 128,170 / 3.64e-3 | 5.057361e-5 / 1,137 / 2.30e-3 |
| Flux CPU vs PyTorch GPU | 4.126653e-4 / 32,212 / 2.55e-3 | 7.480253e-4 / 102,364 / 4.22e-3 | 7.226411e-5 / 946 / 2.39e-3 |
| Flux GPU vs PyTorch GPU (Step 3b) | 3.025942e-4 / 18,248 / 2.00e-3 | 6.672032e-4 / 105,443 / 4.54e-3 | 6.584823e-5 / 1,304 / 2.62e-3 |
| Flux CPU vs PyTorch CPU | 2.731457e-4 / 42,530 / 2.82e-3 | 6.272923e-4 / 156,257 / 4.58e-3 | 9.655068e-5 / 1,438 / 2.84e-3 |

Forward, logits, loss, final parameters, `v`, beta powers, and the step counter
pass in the Flux runs. The same-framework CPU/GPU control reproduces the
gradient/moment failures with the same worst leaves, and the Flux GPU residual
is within ~1.6x of that envelope. The CPU path does not use cuDNN, so the FMA
policy is not the cause.

Proposed but **not applied**: keep `rtol=1e-4` and raise only the absolute
floor — `grads.u1 atol=1e-3`, `grads.u3 atol=2e-3`, `m.u3 atol=2e-4` — each
with ≥2x margin over the four-comparison envelope; all other categories stay
at `1e-5`/`1e-4`. The user must approve before any specification, config, or
checker change.

Evidence: [results](../../artifacts/logs/static-scaling/2026-10-17-phase3-numerics/RESULTS.md),
[commands](../../artifacts/logs/static-scaling/2026-10-17-phase3-numerics/COMMANDS.md),
[provenance](../../artifacts/logs/static-scaling/2026-10-17-phase3-numerics/PROVENANCE.md),
[devlog](../devlog/03-static-scaling/2026-10-17-phase3-numerics-calibration.md).

**Exact next action:** user decision on the three category floors. If approved,
update OD-10 plus the specification, `experiment.toml`, and checker
consistently, then rerun the full driver in a fresh root and require all four
summaries to PASS. If rejected, force a deterministic/direct backward
algorithm family and repeat the full comparison without changing tolerances.
Do not start Steps 4–6 before that decision.

## Phase 3 Step 3b: FMA chooser fixed, backward residual FAIL — 2026-10-17

Completion instructions: [Phase 3 handoff](../../temp/static-graph-gpu-scaling-phase3-handoff.md).

`scripts/static_scaling/precision.jl` now installs its cuDNN overrides
at module load: the FMA-aware `math_mode(::CUDACore.MathMode)` mapping
and more-specific `cudnnConvolutionAlgoPerfChoose` methods for
forward/backward-data/backward-filter. With FMA enabled and an FMA
descriptor, the chooser requires `ps.mathType == CUDNN_FMA_MATH` and
does not apply cuDNN.jl's Float32 skip. `configure_precision!` clears
the three cuDNN algorithm perf caches (keys omit the math type) and
verifies they are empty. `check_phase3_flux.jl` records every choice as
`PHASE3_CUDNN_ALGO` / `PHASE3_CUDNN_ALGO_SUMMARY` and enforces
`CHECK cudnn.fma_algo_selection`; `check_precision_math.jl` adds an
autotuned `Flux.CrossCor` NNlib-path item.

Focused regression PASS: job `60910394` on `lrdn0365`,
`PHASE3_PRECISION_REGRESSION PASS`; the NNlib item chose `algo=0`,
`CUDNN_FMA_MATH(3)`, `fallback=false`, `rel_l2=4.248559e-7`; the
`--mode current` fixed-algorithm control still fails at `≈2.93e-4`.

Full comparison (precompile `60909956` on `lrdn0034`; driver `60910786`
on `lrdn0001`; new root
`$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/`;
unchanged `atol=1e-5`, `rtol=1e-4`): all 32 choices are
`CUDNN_FMA_MATH(3)` with `fallback=false`, and every candidate math type
is `3`. The forward path now passes; three gradient categories still
fail:

| Category | Result | max_abs | violations |
|---|---|---:|---:|
| activations | PASS | 1.56164169e-5 | 0 |
| logits | PASS | 1.66893005e-6 | 0 |
| loss | PASS | 4.76837158e-7 | 0 |
| grads.u1 | FAIL | 3.02594155e-4 | 18,248 |
| grads.u3 | FAIL | 6.67203218e-4 | 105,443 |
| params.u3 | PASS | 4.77582216e-6 | 0 |
| m.u3 | FAIL | 6.58482313e-5 | 1,304 |
| v.u3 | PASS | 2.32830644e-7 | 0 |
| eta, beta_power.u3, step | PASS | 0 | 0 |

Chosen algorithms: forward 0/6, backward data 0/4, backward filter
0/3/5 (FFT-tiling/Winograd/legacy direct). The residual is consistent
with FP32 backward-algorithm rounding against the PyTorch deterministic
reference (cuDNN 9.10.2.21), not TF32. The new reference is again
337/337 exact-equal to the preserved attempt-3 reference and the
independent NumPy checker passes. No tolerance, model-math, optimizer,
or Flux change; no commit.

Evidence: [provenance](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3b/PROVENANCE.md),
[commands](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3b/COMMANDS.md),
[results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3b/RESULTS.md),
[algorithm choices](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3b/algorithm-choices.txt),
[devlog](../devlog/03-static-scaling/2026-10-17-phase3-step3b-fma-algo-chooser.md),
[Step 3 failure](../failures/2026-10-17-phase3-step3-fma-full-model.md).

**Exact next action:** user decision required before Step 4. Either
(1) restrict backward algorithm selection to deterministic/direct
algorithms that match the PyTorch reference, or (2) justify a
cross-framework tolerance revision numerically. Then rerun the full
driver in a fresh root and require all four summaries to PASS. Do not
widen tolerances or pin algorithms without that decision.

## Phase 3 Step 3 full comparison FAIL — precision enforcement gap — 2026-10-17

Completion instructions: [Phase 3 handoff](../../temp/static-graph-gpu-scaling-phase3-handoff.md).

Step 3 ran the complete three-update comparison in a new evidence root on
Leonardo (driver job `60904928`, node `lrdn0125`; precompile `60904055`,
`lrdn3183`). The new PyTorch reference is array-for-array identical to the
preserved attempt-3 reference (337/337 exact, maximum difference 0.0), and
the independent NumPy checker passes. The Flux comparison does **not** pass
with unchanged `atol=1e-5`, `rtol=1e-4`:

| Category | max_abs | violations | Result |
|---|---:|---:|---|
| activations | 7.64757395e-3 | 2,705,938 | FAIL |
| logits | 5.64932823e-4 | 4,981 | FAIL |
| grads.u1 | 5.24188578e-3 | 3,842,324 | FAIL |
| grads.u3 | 7.48962909e-3 | 3,906,482 | FAIL |
| params.u3 | 1.1857599e-5 | 5 | FAIL |
| m.u3 | 1.47970638e-3 | 721,437 | FAIL |
| eta, loss, v.u3, beta_power.u3, step | — | 0 | PASS |

The checker printed requested and effective `CUDNN_FMA_MATH(3)`, but the
full-model forward output is exactly the pre-FMA TF32 run (same per-array
activation violation counts and maximum as job `60887065`). The FMA
descriptor did not change the executed kernels. The evidence is consistent
with cuDNN.jl 6.2.2's Float32 autotuner ignoring the reported math type
(`skipMathTypeCheck = dtype == Float32`): the isolated stage-2 experiment
happened to select FMA-compatible algorithms; the full run selected
TF32-capable ones. This is not ordinary FP32 reduction-order noise and not
an optimizer/model-math defect; loss, eta, beta powers, step, and `v.u3`
pass, and the forward already fails.

Per-leaf relative-L2 summaries (supplementary only): worst leaves are
`grad.u1.stage1.block1.conv1.weight` (4.897306e-2),
`grad.u3.stage1.block1.conv1.weight` (5.608752e-2),
`param.u3.stage1.block1.gn1.bias` (1.148905e-1),
`m.u3.stage1.block1.conv1.weight` (3.943915e-2),
`v.u3.stem.conv.weight` (5.166543e-2), and
`delta.u3.stage1.block1.gn1.bias` (1.148905e-1); no zero/nonzero
disagreements or below-floor norms.

OD-2, OD-3, OD-4, OD-10, and OD-11 remain open. The hardened Phase 2 remote
re-check still lacks evidence. No Flux source change, no commit, and no
tolerance, model-math, optimizer, or precision-config change.

Evidence:
[provenance](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3/PROVENANCE.md),
[commands](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3/COMMANDS.md),
[results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3/RESULTS.md),
[devlog](../devlog/03-static-scaling/2026-10-17-phase3-step3-full-comparison-fail.md),
[failure](../failures/2026-10-17-phase3-step3-fma-full-model.md).

**Exact next action (executed as Step 3b; see the section above):** do
not start Step 4. Enforce FMA-compatible algorithm selection for the
full-model convolution forward and backward in the reusable precision
path (pin algorithms or make the Float32 chooser honor
`ps.mathType == CUDNN_FMA_MATH`), add a full-forward regression, and
rerun Step 3 in a fresh `OUT`/`LOG_DIR` with unchanged tolerances.

## Phase 3 Step 2 precision experiment and focused regression PASS — 2026-10-17

Completion instructions: [Phase 3 handoff](../../temp/static-graph-gpu-scaling-phase3-handoff.md).
The user prefers a bounded precision diagnosis and justified numerical acceptance, followed by progress toward benchmarks.
The handoff does not require bitwise equality or a universal `1e-6` threshold.

The controlled stage-2 experiment resolved the GPU discrepancy. Three fresh
Julia processes compared the stock cuDNN mapping
(`PEDANTIC_MATH -> CUDNN_DEFAULT_MATH`, TF32 permitted) with an explicit
`PEDANTIC_MATH -> CUDNN_FMA_MATH` mapping, against the unchanged PyTorch
`stage2_diag.npz` and the unchanged `atol=1e-5` / `rtol=1e-4` rule:

| Case | conv1 | conv2 | proj |
|---|---|---|---|
| cuda current | 3.5725832e-3, 1,225,601 violations | 3.64590809e-3, 1,316,175 | 0 |
| cuda FMA | 0, 0 | 1.90734863e-6, 0 | 0 |
| CPU control | 7.62939453e-6, 0 | 5.48362732e-6, 0 | 0 |

Requested/effective math types were `CUDNN_DEFAULT_MATH(0)` for current and
`CUDNN_FMA_MATH(3)` for FMA on conv1/conv2/proj; cuBLAS stayed
`PEDANTIC_MATH`. cuDNN library 9.26.0 (parser prints `92.60.0`). FMA removes
the failure; the current case retains it. Experiment job 60896936 on
`lrdn0102` printed `PHASE3_PRECISION_EXPERIMENT PASS`.

cuDNN's autotuner selects algorithms by timing, so the failing convolution can
move between conv1 and conv2 (observed on job 60896513), and a synthetic
high-level test can select FP32 Winograd in both modes. The regression therefore
pins `IMPLICIT_PRECOMP_GEMM` / `BWD_DATA|FILTER_ALGO_1`, which separates FMA
(~1e-7..1e-6) from current (~2.9e-4) deterministically. Job 60900440 on
`lrdn2176` printed `PHASE3_PRECISION_REGRESSION PASS`.

The reusable path is `scripts/static_scaling/precision.jl`
(`configure_precision!`), wired into `check_phase3_flux.jl` and
`diagnose_phase3_flux.jl`; `model_flux.jl` docstrings require GPU runners to
reuse it before GPU work. No Flux source changed, no commit, no tolerance
change. Full Phase 3 acceptance still requires the Step 3 three-update
comparison; the earlier full-run failure (job 60885091) remains historical
evidence.

Leonardo access worked for this campaign. The hardened Phase 2 remote re-check
against the real PREP still lacks evidence in the inspected logs. OD-2, OD-3,
OD-4, OD-10, and OD-11 remain open.

Details: [experiment provenance](../../artifacts/logs/static-scaling/2026-10-17-phase3-step2-precision/PROVENANCE.md),
[devlog](../devlog/03-static-scaling/2026-10-17-phase3-step2-precision-experiment.md),
[failure history](../failures/2026-10-17-phase3-step2-precision-regression-attempts.md),
[prior review](../../reports/static-scaling-phase3-review.md).

**Exact next action:** run Step 3, the complete three-update Flux comparison
with `configure_precision!(fma=true)` and unchanged tolerances, in a new output
directory; retain per-category maxima, violation counts, and relative L2.

## Phase 2 acceptance checks corrected and independently verified — 2026-10-17

Original-reviewer follow-up: R1–R3 are closed locally after source inspection and a fresh six-case harness run.
Canonical Julia acceptance passed 81 checks. The real-PREP remote re-check ran later as job `60951247` and passed (113 checks, exit 0; see the Phase 3 Step 5 section).
The [review](../../reports/static-scaling-phase2-review.md) now records this closure.

The [implementation review](../../reports/static-scaling-phase2-review.md) found three P2 acceptance gaps (R1-R3): prepared-file hashes were not enforced, Julia ignored fixture integrity hashes, and both GroupNorm checks covered only 4 of 20 layers. All three are corrected and independently verified in the [response report](../../reports/static-scaling-phase2-review-response.md); the findings concerned acceptance coverage, not demonstrated corruption of the canonical artifacts.

- R1: `check_dataset.py` enforces `dataset_manifest.json` and `schedule_manifest.json` before semantic checks (`prep.dataset_manifest_present`, `prep.dataset_file_set`, `prep.dataset_files_sha256_bytes`, `prep.schedule_manifest_present`, `prep.schedule_file_set`, `prep.schedule_files_sha256_bytes`), regenerates all 50 schedule epochs bitwise (`schedule.regenerated_all_epochs_bitwise`, about 29 s locally), and compares the regenerated epoch 49.
- R2: `check_fixtures_data.jl` enforces `fixture_manifest.json` before numerical checks (4 files, 35 arrays, dtype/shape/C-order SHA-256: `fixtures.manifest_present`, `fixtures.file_set`, `fixtures.file_sha256_bytes`, `fixtures.array_set`, `fixtures.array_metadata`, `fixtures.array_sha256`); `check_all.jl` inherits the gate.
- R3: both checkers select all 40 GroupNorm arrays (20 weights, 20 biases) and fail `bundle.gn_initial_values` on count or value drift.

Positive reruns: `selftest_local.py` five groups PASS; local `check_all.jl` 81/81 PASS, exit 0 (47 data + 34 operator; was 75); Python positive 63 CHECK lines (62 PASS plus the usual local `--dataset-root` SKIP). The new [negative harness](../../scripts/static_scaling/negative_tests_local.py) ends `NEGTEST PASS`: the epoch-49 valid-range mutation is rejected by `prep.schedule_files_sha256_bytes FAIL`, the in-range label mutation by `fixtures.file_sha256_bytes FAIL`, and the self-consistent residual GroupNorm mutation by `bundle.gn_initial_values FAIL` in both checkers while `bundle.manifest_sha256` and `bundle.bundle_hash` PASS. With self-consistent schedule hashes, `schedule.regenerated_all_epochs_bitwise FAIL bad_epochs=['49']` (and `schedule.epoch_independence FAIL`) still rejects. Canonical artifacts are unchanged (`initial_bundle.npz` sha256 `a790b03c...`; all four fixture files match their manifest).

Residual limitations: a fully self-consistent rewrite of `data_fixture.npz` plus `fixture_manifest.json` still passes the local Julia-only check because Julia has no PREP reference; the Python PREP cross-check catches it (`data_fixture.train_labels FAIL`), and the remote combined acceptance runs Python first. A self-consistent dataset-image rewrite is bounded by the manifest hashes, global statistics, and sampled fixtures.

Remote re-check on the real `$PREP` was completed as job `60951247` (113 checks, 0 FAIL, exit 0). Earlier on 2026-10-17 `ssh leonardo` had failed with `epicazio@login.leonardo.cineca.it: Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`; access later worked. The recurrence and the separate transient 2026-10-18 DNS/network drop are recorded in the [failure note](../failures/2026-10-17-leonardo-ssh-publickey.md).

**Historical next action (completed):** the real-`$PREP` re-check ran, Phase 3 proceeded, and Phase 3 was accepted. The current exact next action is stated at the top of this file.

Details: [acceptance-fix devlog](../devlog/03-static-scaling/2026-10-17-phase2-acceptance-fixes.md), [prior review devlog](../devlog/03-static-scaling/2026-10-17-phase2-implementation-review.md).

## Phase 2 complete and independently verified — 2026-10-16 (historical; acceptance hardened 2026-10-17, see above)

Static-network GPU scaling Phase 2 is complete: the canonical decoded dataset,
50-epoch schedule, normalization statistics, 62-array initial bundle, and the
four shared correctness fixtures were generated on Leonardo and passed remote,
local, and independent acceptance. Generation-time provenance: thesis HEAD
`3088f61020137912af32d9c969e87ea9ae2a71c3`, source state
`4d4fd2ac75f192c049d0c7eedfcdf4c1daf83ece770e66ff7bf181d76b651ad1` (recipe:
`git diff HEAD` + sha256 of sorted untracked non-artifacts files; later
comment-only edits move the tree hash, so the state is generation-time
provenance). The canonical bundle is 11,271,432 float32 parameters in 62
arrays with `bundle_hash=a7632ca9...`, reproduced identically by local torch
2.14.1+cpu and remote torch 2.9.1+cu126; `initial_bundle.npz` sha256
`a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`.

Remote acceptance: prep job 60758615 (`lrdn0018`, lprod, COMPLETED 0:0,
00:01:55) and check job 60758839 (`lrdn0011`, dbg, COMPLETED 0:0, 00:02:15),
with `PHASE2_DATASET_CHECK PASS`, `PHASE2_JULIA_DATA_CHECK PASS`,
`PHASE2_REMOTE_CHECK PASS`; the prepared dataset lives at
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/`, fixtures at
`artifacts/baselines/static-scaling/2026-10-16-phase2/`. Local acceptance on
the copied real fixtures: `PHASE2_JULIA_DATA_CHECK PASS` +
`PHASE2_JULIA_OPERATOR_CHECK PASS` (75 checks, exit 0; the remote check job is
NPZ-only and does not run the Flux operator check). Independent verification:
[report](../../reports/static-scaling-phase2-verification.md), overall PASS,
all four adversarial perturbations caught; D1 was an error in the verification
prompt's rank-rule wording (artifacts match the documented per-512-batch
sharding bitwise), D2 was a stale `COMMANDS.md` note, now corrected.

Full command/hash/job record:
[COMMANDS.md](../../artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md).
Campaign record and failure history:
[2026-10-16-phase2-artifacts-complete](../devlog/03-static-scaling/2026-10-16-phase2-artifacts-complete.md),
[env-token/QoS failure](../failures/2026-10-16-phase2-prep-leonardo-env-and-qos.md),
[`std64`/masked-exit failure](../failures/2026-10-16-phase2-prep-std64-nameerror-masked-exit.md),
[linear-fixture failure](../failures/2026-10-16-phase2-prep-make-fixtures-linear-broadcast.md).

**Exact next action (historical; superseded by the 2026-10-17 section above):**
Phase 3 (training-side consumption / single-GPU baseline per
`temp/static-graph-gpu-scaling-implementation-phases.md`): sync
`flux-integration` `f7b4520c` to Leonardo (manual rsync, protect manifests),
resolve on the login node, precompile on a compute node under
`boost_qos_lprod`, then run the remote operator check.

## Phase 2 prep retry-4: steps 1-3 PASS, fixtures blocked — 2026-10-16 (historical; resolved in retry-5, see above)

Leonardo prep job 60755769 (`$WORK` target, node `lrdn0018`, `boost_qos_lprod`,
elapsed 00:25:34, FAILED 1:0) ran the synced commit
`3088f61020137912af32d9c969e87ea9ae2a71c3` (state
`3e9ba4d8013ec67383454d86d39ec0caaae2616e43cc5ea4cc963843d6e72561`) with
`FORCE=1`. The two local fixes took effect: `DATASET_PREP DONE` with
`train_mean=[0.48023695 0.44806704 0.39750364]`,
`train_std=[0.27643642 0.2688633  0.28158993]`,
`containers_train=[('JPEG', 100000)] containers_val=[('JPEG', 10000)]`;
`SCHEDULE epochs=50 DONE`; `BUNDLE arrays=62` verified with
`bundle_hash=a7632ca9...`. Step 4/4 `make_fixtures` then FAILED with
`RuntimeError: The size of tensor a (5) must match the size of tensor b (3)`
at `make_fixtures.py:212` (`y = w @ x + b`): the fixture bias is `(3,)` but
must broadcast to `(3, 5)`. The fixed prep driver aborted with
`PHASE2_PREP FAIL` and exit 1 (no more masked success).
The fixture directory is empty and the acceptance check, copy-back, and
COMMANDS.md were not run; no code was changed. Prepared dataset hashes:
`artifacts/logs/leonardo/2026-10-16-phase2-prepared-sha256.txt`
(`initial_bundle.npz` = `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`).
Details: [failure record](../failures/2026-10-16-phase2-prep-make-fixtures-linear-broadcast.md),
[devlog](../devlog/03-static-scaling/2026-10-16-phase2-prep-retry4-make-fixtures-bug.md).

**Exact next action (historical; completed in retry-5):** correct the
linear-fixture bias broadcasting in `scripts/static_scaling/make_fixtures.py`
(`build_linear_fixture_torch` line 210-212; also audit
`build_kernel_fixture_torch`, never exercised), sync, and re-run prep step 4/4
on Leonardo.

## OD-1, OD-5, OD-6 recorded; Phase 2 unblocked — 2026-10-16

The supervisor recorded the three Phase 2-blocking decisions in [open-decisions.md](../../experiments/static-scaling/open-decisions.md) (decision record dated 2026-10-16):
OD-1 freezes the small-image ResNet-18 (3x3 stride-1 stem, no max-pool, GroupNorm-32 eps 1e-5, biased 200-class classifier) at 11,271,432 trainable parameters across 62 arrays, with Kaiming-normal `fan_out`/ReLU convolution init, PyTorch `Linear` defaults for the classifier, GN `gamma = 1`/`beta = 0`, and one canonical initial bundle with recorded generator, version, and seed loaded by both frameworks.
OD-5 fixes per-channel RGB population mean and standard deviation from all 100,000 original training images (decoded uint8 scaled to [0,1], float64 accumulation, before augmentation/padding/tail drop; validation and test images excluded); float32 constants frozen in the fixture manifest.
OD-6 fixes the training transforms: zero-pad 4 per side, uniform 64x64 crop (zero-based `crop_top`/`crop_left` in `0..8`, independent), horizontal flip p=0.5; `(crop_top, crop_left, flip)` saved per (epoch, sample_id); padding/crop/flip on uint8, then float32, /255, normalize; validation uses conversion and normalization only.
`specification.md`, `experiment.toml`, and `open-decisions.md` are updated and consistent; the TOML parses. Numeric normalization constants and the parameter bundle are generated in Phase 2.
Details: [devlog](../devlog/03-static-scaling/2026-10-16-od1-od5-od6-recorded.md).

**Exact next action:** implement Phase 2 (plan section 4): dataset adapters, saved epoch permutations and augmentation records, normalization statistics and the canonical initial bundle, and the shared `.npz`+manifest correctness fixtures (including the review's rectangular-linear and asymmetric-kernel cases); PyTorch-side generation needs the pinned Leonardo venv, so sync `f7b4520c` and precompile on a compute node first.

## Phase 1 review independently closed — 2026-10-16

The reviewer inspected the corrected specification, configuration, and both probe scripts.
Both local probe reruns exited 0 and reproduced the saved printed output.
R1–R4 are closed in the [review report](../../reports/static-scaling-phase1-review.md).
The experiment decisions remain pending. Actual PyTorch numerical comparisons remain Phase 2/3 work.

**Exact next action:** record OD-1, OD-5, and OD-6 before generating Phase 2 fixtures.

## Phase 1 review response complete — 2026-10-16

All four review findings are corrected in `experiments/static-scaling/specification.md` and `experiment.toml`.
R1: no classifier transpose (`Dense(512 => 200)`, weight `(200, 512)`).
R2: convolutions use `Flux.CrossCor` with `permutedims(w, (4, 3, 2, 1))`; no kernel reversal; self-inverse export.
R3: the all-keyword `Optimisers.AdamW(; eta, beta, lambda, epsilon, couple = true)` constructor is documented.
R4: the shared counter is completed updates `s`; Flux stores `beta^(s+1)` and continuation reconstructs it by repeated float32 multiplication.
Focused acceptance probes pass with exit 0 (`experiments/static-scaling/probes/`, logs in `artifacts/logs/static-scaling/2026-10-16-review-response/`), including a manual PyTorch-semantics cross-correlation reference with an asymmetric kernel.
No GPU, remote, or training run; no commit.
Details: [response devlog](../devlog/03-static-scaling/2026-10-16-phase1-review-response.md).

**Exact next action:** record OD-1, OD-5, and OD-6 (they block Phase 2), then generate the Phase 2 shared fixtures under the corrected conversion rules.

## Phase 1 review: changes required — 2026-10-16

The [review](../../reports/static-scaling-phase1-review.md) identifies four contract defects in the shared specification and configuration.
R1 corrects the classifier transpose. R2 requires an explicit convolution kernel-orientation rule.
R3 corrects the AdamW constructor. R4 corrects the stored beta-power exponent relative to completed updates.
Local Optimisers 0.4.9 and NNlib 0.9.45 probes support the findings.
The specification remains unchanged pending corrections. No remote execution or training occurred.
Commands and evidence: [devlog](../devlog/03-static-scaling/2026-10-16-phase1-review.md).

**Exact next action:** correct R1–R4 in the specification and configuration before generating Phase 2 fixtures.

## Phase 1 shared specification drafted — 2026-10-15

Phase 1 of the [static scaling plan](../../temp/static-graph-gpu-scaling-implementation-phases.md) is drafted as documentation in `experiments/static-scaling/`: a readable [specification](../../experiments/static-scaling/specification.md), a machine-readable [config](../../experiments/static-scaling/experiment.toml) (Julia TOML and Python 3.11 tomllib), and an [open-decision table](../../experiments/static-scaling/open-decisions.md) with 13 decisions and proposed defaults.
The contract records the architecture (11,271,432 parameters, 62 trainable arrays), the data schedule (99,840 samples per epoch, 160-image tail dropped, 195/390 updates per epoch), AdamW equations and grouping, the schedule by global update, the shared `.npz` plus manifest fixture format, the fixed 32-core CPU policy, the 20%-free memory margin, tolerances, and the explicit Flux entry point (64 guard collectives per update).
Source pin: `flux-integration` is clean at `f7b4520c` with both guards committed; the Leonardo copy is a pre-guard snapshot and must be re-synced and precompiled before any run. The local manifest is stale.
An independent verification subagent passed all nine checks; one wording defect was found and fixed. No training, no remote execution, no Flux change, no commit.

Details: [devlog](../devlog/03-static-scaling/2026-10-15-phase1-shared-specification.md).

**Exact next action:** record the open decisions (at minimum OD-3, OD-5, OD-6, OD-10, OD-11), then implement Phase 2: data adapters, saved epoch permutations and augmentation records, and shared correctness fixtures.

## Static-network implementation phases — 2026-10-15

The [implementation plan](../../temp/static-graph-gpu-scaling-implementation-phases.md) defines nine phases after the completed Leonardo environment preparation.
The phases cover the shared specification, data fixtures, single-GPU training, DDP correctness, memory selection, timing, repeated measurements, learning curves, and reporting.
The final batch follows the memory pilot. GroupNorm and AdamW remain selected, with exact hyperparameters still open.
This session produced documentation only. No training or remote execution occurred.

**Exact next action:** write the Phase 1 shared specification and configuration draft, with remaining decisions listed explicitly.

## Leonardo benchmark environment ready — 2026-10-15

The static-network scaling benchmark environment is prepared on Leonardo; no experiment has run.
The PyTorch venv (torch 2.9.1+cu126, torchvision 0.24.1+cu126, CPython 3.11.17) and Tiny ImageNet-200 are staged on `$WORK`.
Compute-node verification PASS (`ENV_VERIFY=PASS`, one A100, single-kernel environment smoke); the dataset passed every checksum, layout, and count check.
Reusable setup, verification, and dataset scripts are in `scripts/remote/leonardo/`.
No test, benchmark, or training ran; no model or training code was written.

Details: [devlog](../devlog/03-static-scaling/2026-10-15-leonardo-benchmark-env-setup.md),
[Leonardo context](../context/leonardo.md),
[proposal draft](../../reports/static-graph-gpu-scaling-proposal-detailed.md) (discussion draft, no experiment).

**Exact next action:** write the shared experiment specification and the training code for both frameworks once the supervisor confirms the open decisions (AdamW hyperparameters and the batch size via the later memory pilot); the environment is ready but no model/training code exists yet and no experiment has run.

## Conditional MoE and PyTorch explanation — 2026-10-15

Source inspection and primary documentation support the supervisor answers about unused MoE parameters and PyTorch DDP.
The wrapper handles local absence, global absence, and reactivation with a fixed parameter structure.
This contract does not establish learned-router MoE or expert-parallel acceptance.
PyTorch DDP also distinguishes local and global absence. Its standard reducer uses autograd hooks and gradient buckets.
The comparison pins PyTorch documentation and reducer source to version 2.8.0 for reproducible claims.
Details: [discussion record](../devlog/02-conditional-graph/2026-10-15-moe-pytorch-conditional-graphs.md).

**Exact next action:** clarify whether the supervisor means temporary expert inactivity, token dropping, or structural expert removal.

## Static-network scaling proposal drafted — 2026-10-15

The user narrowed the proposal to static networks, eager PyTorch, and fixed-global-batch scaling on Leonardo.
Tiny ImageNet-200 is the intended dataset. This remains a drafting phase for supervisor discussion.
The [draft report](../../reports/static-graph-gpu-scaling-proposal.md) identifies documented small-image ResNet-18 recipes and explains the BatchNorm tradeoff.
The supervisor-facing version is now a 1–2 page Italian proposal; the extended drafting version is preserved at `reports/static-graph-gpu-scaling-proposal-detailed.md`.
The proposal now uses global batch 512 as the primary candidate (local 512/256/128 at one/two/four GPUs), with 256 as the memory-pilot fallback; supervisor feedback approved GroupNorm and selected AdamW, so the reference SGD settings no longer transfer.
It proposes GroupNorm as an explicit adaptation and calculates parameter, optimizer, input, and activation-storage estimates.
The activation scenarios are uncalibrated assumptions, not measured peak-memory bounds.
No training, remote execution, or implementation occurred. Batch size and training duration remain proposals, and the exact AdamW hyperparameters are still open.

**Exact next action:** align the short Italian proposal with AdamW, then confirm the single-GPU batch with the memory pilot.

## GPU scaling experiment discussion — 2026-10-15

The user proposes Flux/PyTorch comparisons on Leonardo at one, two, and four GPUs, with static and conditional networks.
The dataset name remains ambiguous: confirm whether “tiny-images” means Tiny ImageNet-200.
The recommendation is fixed-global-batch strong scaling first, followed by fixed-local-batch throughput scaling.
Both frameworks need the same data, model equations, initial parameters, optimizer semantics, and global routing schedule.
Conditional correctness checks must cover locally absent gradients, globally absent gradients, and reactivation after Adam history.
The latest numerical guards require a fresh Flux baseline. Historical October 6 timings describe an earlier implementation.
This discussion establishes no new performance result or approved experiment configuration.
Details: [discussion record](../devlog/02-conditional-graph/2026-10-15-gpu-scaling-discussion.md).

**Exact next action:** confirm the dataset identity, then specify the model pair and rank-independent global batch/routing schedule.

## DB-3 independent review closed — 2026-10-14

R1, R2, and R3 are closed. No blocking findings remain within the reviewed scope.
Both historical tutorial summaries passed an independent local rerun.
All six corrected file hashes match the response campaign, and Flux `git diff --check` passed.
The inspected remote logs support the guarded protocol checks, focused MPI/NCCL results, and old-weighting RED control.

Details: [final review](../../reports/conditional-graph-db3-review.md#independent-closure-review--2026-10-14).
The earlier local MPI environment failure remains separate from acceptance. No remote commands or commits occurred during closure.

**Exact next action:** decide whether to commit the reviewed DB-3 change set and prepare the upstream review.

## DB-3 review response complete — 2026-10-14

All three findings of [the DB-3 review](../../reports/conditional-graph-db3-review.md)
are corrected and gate-verified on `deathstar`. Production guard code did not
change; no commit was made (the worktrees remain dirty by project convention).

- R1 (thesis): `notebook/phase_k_workflow_helpers.jl` pins the collective
  protocol per bundle (`legacy-5/7` base 3 no vote, `guarded-6/8` base 4 with the
  trailing host `Float32` length-1 `avg` gradient-validity vote); a missing or
  unknown pin throws. The new `scripts/checks/phase_k_collective_protocol_check.jl`
  passed on deathstar: both tutorials, the guarded DB-3 toy bundle, and three
  negative controls, `PHASE_K_PROTOCOL_SUMMARY PASS`, exit 0. Log:
  `artifacts/logs/conditional-wrapper/numerical-guards/db3-review-response/r1-protocol-check.log`.
- R2 (Flux, test only): `test/ext_distributed/nonfinite_gradient_distributedtest.jl`
  now uses a nonzero `a` target and the correct step-3 weighting, and asserts
  nonzero `a` moments before the absence step. Healthy testset 59/59 per rank at
  mpi2, mpi4, and nccl2 (exit 0); the old-weighting RED control fails 6 of 59 per
  rank. Logs under
  `artifacts/logs/conditional-wrapper/numerical-guards/db3-review-response/`.
- R3 (Flux, docs only): wrapper docstrings, `NEWS.md`, the NCCL extension
  comment, and the ADR-0009 implementation summary now state the all-rank vote,
  coordinated `DomainError`, preservation guarantee, nonempty-plan rule, and
  scalar device-to-host transfer consistently; no executable line changed.

Evidence:
[devlog](../devlog/02-conditional-graph/2026-10-14-db3-review-response.md),
[COMMANDS.md](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-review-response-campaign-2026-10-14-deathstar),
[failure note](../failures/2026-10-14-db3-review-response-r1-shape-and-artifacts.md),
[ADR-0009](../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md).

**Exact next action:** Review the combined DB-3 diff against the corrected hashes, then decide on committing the DB-3 change set and updating the upstream PR/review.

## DB-3 independent review — 2026-10-14

The review requests corrections for two P2 findings and one P3 documentation finding.
[Report](../../reports/conditional-graph-db3-review.md): both saved tutorials fail under the new collective-count expectation.
The focused healthy reference also has incorrect step-3 weighting, hidden by zero gradients.
Public contract documentation needs corrections.

The inspected production guard follows the planned boundary. Production and focused-test hashes match the campaign.
The independent local MPI attempt failed during precompilation, before tests, with a CUDA dependency `KeyError`.
Evidence and commands: [devlog](../devlog/02-conditional-graph/2026-10-14-db3-review.md) and [failure record](../failures/2026-10-14-db3-review-local-environment.md).

**Exact next action:** none remains from this review; the review-response section above records the corrections and deathstar gates.

## DB-3 gradient rejection implemented and gate-accepted — 2026-10-14

The post-reduction non-finite-gradient guard is implemented and gate-accepted
([ADR-0009](../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md)).
`_sync_gradients` scans every globally present reduced buffer with
`_reduced_buf_finite` and casts one all-rank validity vote through the existing
`_any_rank_invalid` helper, after the complete reduction loop and before
`_build!`; `np > 0` always votes (including the all-gradients-absent case) and
`np == 0` keeps zero collectives. The NCCL extension overrides the scan for
`CuArray`: it synchronizes the CUDA stream and device-reduces. Rejection raises
`DomainError(nothing, _NONFINITE_GRADIENT_ERROR)` on every rank with "non-finite
synchronized gradient" and "No optimizer update occurred". No commit; the
`flux-integration` worktree is dirty on HEAD
`314aef2022333fa8f37035ed10cdc8e2360fa475`.

- GREEN on deathstar: gradient harness mpi2 19/19, mpi4 8/8, nccl2 17 PASS +
  2 UNTESTED (the native overflow fixtures are MPI-only; the controlled
  injection covers the device path); focused `nonfinite_gradient_distributedtest.jl`
  exit 0 at mpi2, mpi4, and nccl2; loss-group regression mpi2 13/13 and mpi4
  8/8; combined `--group all` (Phase D) mpi2 32/32, mpi4 16/16, nccl2 29 PASS +
  3 UNTESTED (`dup-train-one`, `grad-overflow`, `grad-tied-overflow`); full
  suite 9/9 files at mpi2 (2m53.8s) and mpi4 (3m09.9s), 18/18 at nccl2
  (7m37.9s, MPI + NCCL); single-process `train` 137/137; 0 ambiguities;
  deadlock regression 246/246; Phase K toy demo 13/13 at mpi 2 and 4; 0
  `Precompiling packages...` lines.
- RED retained: gradient harness mpi2 18 FAIL + 1 PASS (`grad-no-trainable` is
  a no-regression case); focused test mpi2 first testset 7 pass / 7 fail /
  1 error. The superseded v1 RED run and the failed first suite attempt are
  retained.
- New regression: `test/ext_distributed/nonfinite_gradient_distributedtest.jl`
  (848 lines, 15 testsets). The updated conditional and CUDA conditional test
  files, the driver, the deadlock check, and the Phase K helpers encode the
  extra trailing host `Float32` length-1 `avg` gradient vote per nonempty
  update; `NEWS.md` records the rejection.
- Harness/test defects found and fixed (all fixed; none production):
  [failure note](../failures/2026-10-14-db3-green-attempt1-harness-fixes.md).
- Evidence:
  [COMMANDS.md DB-3 campaign](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-campaign-2026-10-14-deathstar),
  [devlog](../devlog/02-conditional-graph/2026-10-14-db3-gradient-rejection.md),
  [audit 002 resolution](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md#43-db-3--finite-loss-with-non-finite-gradient-silent-corruption).

**Exact next action:** Review the DB-3 working-tree diff and evidence, then decide on committing to flux-integration and preparing the upstream contribution/review.

## DB-1/DB-4 independent review closed — 2026-10-14

R1, R2, and the NamedTuple coverage gap are closed.
The final exact-argv cleanup passed the independent local selftest:
`SELFTEST_CLEANUP PASS prefix_ok=true metachar_ok=true final_empty=true`.
All five hashes match the recorded final campaign. The green3 summaries support the reported MPI and NCCL results.
No blocking findings remain within the documented eager-path scope.

Details: [final review](../../reports/conditional-graph-db1-db4-review.md#final-review-verdict--2026-10-14),
[independent closure command](../devlog/02-conditional-graph/2026-10-14-db1-db4-v2-review.md#final-closure-after-the-exact-argv-correction).
NCCL `Duplicated` remains untested. DB-3 and combined acceptance remain pending.

**Exact next action:** decide whether to implement the separate DB-3 guard before combined `--group all` acceptance.

## DB-1/DB-4 R1 round-2 cleanup fix complete — 2026-10-14

R1 is closed: the v2 cleanup interpolated the campaign token into an unbounded
regular expression, so `campA` also selected `campA-rerun` and `camp.1` also
selected `campX1` (reproduced with the real `campaign_pids()`). Round 2 removes
regex/pgrep/pkill matching: `campaign_pids(token)` scans
`/proc/<pid>/cmdline`, requires the probe filename as a separate argv element,
and requires the exact adjacent pair `("--campaign", token)`, so cleanup is
prefix- and metacharacter-safe. Operator-supplied `NUMERICAL_GUARD_CAMPAIGN`
values must be unique across concurrent runs (default `<pid>-<time_ns>` is
unique).

- Re-gates on `deathstar` with the final driver (v3): harness `green3-*` mpi2
  13/13, mpi4 8/8, nccl2 12 PASS + 1 UNTESTED (`dup-train-one`); healthy
  markers all true (mpi2 8 `STEP_STATE_OK` + 2 `ABSENCE_FROZEN` + 2
  `REACTIVATED`; mpi4 16/4/4; nccl2 8/2/2); 0 `Precompiling packages...` lines
  in every v3 log.
- Suite v2 (mpi2 8/8, mpi4 8/8, nccl2 16/16; single-process `train` 137/137;
  0 ambiguities) remains authoritative because the test file is unchanged.
- Extended selftest (`selftest-cleanup-deathstar-v2.log`) PASS with
  `prefix_ok=true metachar_ok=true final_empty=true` on four real idle Julia
  children (prefix, regex-metacharacter, and wildcard-collision tokens).
- Fresh concurrency check (`cleanup-concurrency-deathstar/`): campB 13/13 PASS;
  campA (`healthy`, 1 s execution watchdog, token `campA`) timed out after
  37.7 s with both ranks at `GUARD_PROBE_READY`; no leftover processes; GPUs
  idle.
- Remaining: DB-3 not implemented (`--group gradient|all` prints
  `GROUP_UNAVAILABLE`, exit 2), so the combined Phase D acceptance is pending;
  the Duplicated eager route under NCCL is untested; the contract is eager-path
  only and forward-pass side effects are not rolled back.
- Evidence:
  [COMMANDS.md v3](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#v3-review-response-round-2-2026-10-13),
  [selftest log](../../artifacts/logs/conditional-wrapper/numerical-guards/selftest-cleanup-deathstar-v2.log),
  [concurrency](../../artifacts/logs/conditional-wrapper/numerical-guards/cleanup-concurrency-deathstar/),
  [devlog R1 round 2](../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md#review-response-r1-round-2),
  [audit 002 resolution](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md#resolution-2026-10-13).

**Exact next action:** decide whether to implement the DB-3 gradient guard
(separate plan) before the combined `--group all` acceptance.

## DB-1/DB-4 review response (v2) complete — 2026-10-13

The external review's two P2 harness issues and one coverage gap are fixed and
re-gated with no Flux source change; the v1 counts remain valid for the cases
that ran. Report:
[DB-1/DB-4 review](../../reports/conditional-graph-db1-db4-review.md).

- R1 (cleanup isolation): driver cleanup is campaign-token-scoped
  (`reap_campaign`, `--campaign TOKEN`, `--selftest-cleanup`); the selftest and
  a concurrent campA/campB check prove that one campaign cannot kill another's
  children.
- R2 (conditional healthy gate): `healthy` now runs four rank-distinct
  conditional eager `train!` steps with per-step parameter and full Adam-leaf
  state comparison against a native reference (`STEP_STATE_OK`,
  `ABSENCE_FROZEN`, `REACTIVATED`), at 2 and 4 MPI ranks and 2 NCCL ranks.
- Coverage: `nonfinite_loss_distributedtest.jl` gained a NamedTuple-loss
  testset (now 9 testsets).
- GREEN on `deathstar` (v2): harness mpi2 13/13, mpi4 8/8, nccl2 12 PASS + 1
  UNTESTED (`dup-train-one`); suite v2 mpi2 8/8, mpi4 8/8, nccl2 16/16;
  0 `Precompiling packages...` lines; deathstar left clean.
- Remaining: DB-3 not implemented (`--group gradient|all` prints
  `GROUP_UNAVAILABLE`, exit 2), so the combined Phase D acceptance is pending;
  the Duplicated eager route under NCCL is untested; the contract is eager-path
  only and forward-pass side effects are not rolled back.
- Evidence:
  [COMMANDS.md v2](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#v2-review-response-campaign-2026-10-13),
  [devlog review response](../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md#review-response-second-pass),
  [failure note](../failures/2026-10-13-db1-db4-healthy-probe-fielderror.md),
  [audit 002 resolution](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md#resolution-2026-10-13).

**Exact next action:** decide whether to implement the DB-3 gradient guard
(separate plan) before the combined `--group all` acceptance.

## DB-1/DB-4 coordinated non-finite-loss fix accepted — 2026-10-13

The coordinated non-finite-loss rejection is implemented and gate-accepted
([ADR-0008](../decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md)).
The private hook `Flux.Train._loss_update_allowed` runs in `_eager_step!` after
`withgradient` and before `_update!`; the `DistributedOptimizerState`
specialization votes with one host `Float32` 0/1 `avg` allreduce (threshold
`> 0`) and raises `DomainError` on every rank before the presence and gradient
collectives. No commit; Flux HEAD stays `55171e65`. The review-response v2
re-gate is recorded in the section above.

- New device-generic regression
  `test/ext_distributed/nonfinite_loss_distributedtest.jl` (9 testsets after
  the v2 NamedTuple addition) plus
  14 lines in `test/train.jl` and a `NEWS.md` entry.
- GREEN on `deathstar` (v1 campaign; re-gated in the v2 section above): harness
  loss mpi2 13/13, mpi4 8/8 (one invalid rank of
  four), nccl2 12 PASS + 1 UNTESTED (`dup-train-one`); dedicated suite mpi2 8/8
  and mpi4 8/8 files, nccl2 16/16 (MPI + NCCL passes); single-process `train`
  137/137; 0 ambiguities; 0 `Precompiling packages...` lines.
- Remaining coverage: DB-3 not implemented (`--group gradient|all` prints
  `GROUP_UNAVAILABLE`, exit 2), so the combined Phase D acceptance is pending;
  the Duplicated eager route under NCCL is untested; the contract is eager-path
  only and forward-pass side effects are not rolled back.
- Evidence: [COMMANDS.md](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md),
  [devlog](../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md),
  [failure note](../failures/2026-10-13-db1-db4-nccl-dup-and-testfile.md),
  [audit 002 resolution](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md#resolution-2026-10-13).

**Exact next action:** decide whether to implement the DB-3 gradient guard
(separate plan) before the combined `--group all` acceptance.

## Numerical-safeguard implementation plans — 2026-10-13

The user requested detailed plans for both numerical safeguards after the initial scope discussion.

- [DB-1/DB-4 plan](../../reports/conditional-graph-db1-db4-fix-plan.md): a shared eager-training hook coordinates loss rejection before update collectives.
- [DB-3 plan](../../reports/conditional-graph-db3-fix-plan.md): a post-reduction gradient check coordinates rejection before Optimisers.jl changes parameters or state.

Both plans specify RED-first tests, CPU gates, focused NCCL gates, error semantics, and documentation requirements.
The proposed gradient check is mandatory. Its overhead remains unmeasured.
The plans distinguish optimizer-update preservation from forward-pass side effects.
Implementation and runtime acceptance remain pending.

**Exact next action:** record the Flux worktree diff, then write the DB-1/DB-4 ADR and bounded asymmetric-loss success regression.

## Deferred-behavior scope discussion — 2026-10-13

The user and tutor want important, bounded fixes, with other work tracked through GitHub issues.
The initial recommendation groups DB-1 and DB-4 into one coordinated non-finite-loss fix.
DB-3 remains a separate decision because gradient checks add device reductions and synchronization.
Accepted configuration and checkpoint contracts remain distinct from defects.
This recommendation is provisional. The user has not selected the final scope.
Details: [scope discussion](../devlog/02-conditional-graph/2026-10-13-deferred-behavior-triage.md).

**Exact next action:** agree on the DB-1/DB-4 error policy and whether DB-3 enters the immediate fix batch.

## Wiki reorganized by project scope — 2026-10-07

The flat `wiki/devlog/` (76 entries) and `wiki/audits/` (3 reports) folders are
grouped by project scope: [`01-pr2694/`](../devlog/01-pr2694/) (C0–C8, M0/M1,
PS1–PS8, Windows fix, era tooling) and
[`02-conditional-graph/`](../devlog/02-conditional-graph/) (wrapper Phases A–K,
GPU validation, timing, notebooks). Complete indexes:
[`wiki/devlog/README.md`](../devlog/README.md) and
[`wiki/audits/README.md`](../audits/README.md). All relative links were
rewritten; a repo-wide markdown-link inventory reports zero broken links among
the moved files.

The same pass corrected stale wiki facts: quickstart next action and indexes,
Leonardo status, README status rows, the `../Flux.jl` fork path, ADR
superseded-notes, resolved open questions in this file, and missing reference
files in testing docs. No Julia, MPI, CUDA, notebook, or Makefile change.
Details: [2026-10-07-wiki-reorganization](../devlog/02-conditional-graph/2026-10-07-wiki-reorganization.md).

**Exact next action:** unchanged — implement `notebook/conditional_graph_benchmark.ipynb` from the corrected `timing/stretch-2026-10-06-leonardo/` artifacts (two-notebook plan implementation item 6).

## Conditional graph implementation report written — 2026-10-06

A detailed implementation report is ready for the supervisor meeting:
[`reports/conditional-graph-implementation.md`](../../reports/conditional-graph-implementation.md).
It documents how conditional graph support was implemented, with file paths,
line numbers, code snippets, the defect history, the test evidence, and the
audit 002 limitations. The report uses the simple-english rules and was
cross-checked against the implementation worktree (commit chain `6065ce8f` to
`55171e65`, production diff +384/-13 in `src/distributed/public_api.jl`).
Documentation only: no Flux source, test, Julia, MPI, CUDA, or remote command
changed this session. Devlog:
[2026-10-06-conditional-graph-implementation-report](../devlog/02-conditional-graph/2026-10-06-conditional-graph-implementation-report.md).

**Exact next action:** unchanged — implement `notebook/conditional_graph_benchmark.ipynb` from the corrected `timing/stretch-2026-10-06-leonardo/` artifacts (two-notebook plan implementation item 6).

## Distributed workflow notebook implemented — 2026-10-06

Deliverable 1 of the two-notebook plan is implemented and executed from fresh
IJulia kernels in artifact mode.
`notebook/conditional_graph_distributed_workflow.ipynb` selects a complete run
bundle: four-rank Leonardo NCCL by default, two-rank deathstar NCCL through
`PHASE_K_TUTORIAL_CONFIG=tutorial-2rank`. The kernel reads saved artifacts,
calls the Python plotters through `uv` with the pinned environment in
`scripts/plots/requirements-phase-k.txt`, and never joins a collective; fresh
training is an opt-in child process through the bounded driver.

Acceptance checks passed: four-rank resume 472/472 and exposures 6400/8960;
two-rank resume 236/236 and exposures 7680/7680; both HTML exports show 12
figures and nine sections with no error outputs; missing artifacts produce an
explicit error listing every path. The child-process path was executed
separately (MPI 2 ranks, driver 6/6 PASS, 151.8 s). New plotter figures cover
exposure, collectives, and continuation. Records:
[devlog](../devlog/02-conditional-graph/2026-10-06-phase-k-workflow-notebook-implementation.md),
`artifacts/logs/conditional-wrapper/notebook/2026-10-06/COMMANDS.md`.

**Exact next action:** implement `notebook/conditional_graph_benchmark.ipynb`
from the corrected `timing/stretch-2026-10-06-leonardo/` artifacts (plan
implementation item 6).

## Two Phase K notebooks planned — 2026-10-06

The user selected two IJulia notebooks: a distributed-wrapper tutorial and a separate one-A100/four-A100 benchmark.
The tutorial defaults to four ranks and permits a two-rank selection.
Both notebooks analyze saved artifacts and include reproduction commands.

The [implementation plan](../../reports/conditional-graph-phase-k-notebook-plan.md) defines each graph, its data, interpretation, and acceptance checks.
The recommendation is to retain the Python plotters and call them from Julia through subprocesses.
New views cover task exposure, checkpoint continuation, throughput, and optional timing components.
This replaces the earlier single `conditional_graph_gpu_recap.ipynb` proposal.
Notebook implementation remains pending. Broader deferred-behavior acceptance remains a separate decision.

**Exact next action:** implement the artifact bundle map and shared IJulia display helper, then extend the existing Python figures.

## GPU notebook preconditions reviewed — 2026-10-06

The corrected timing summaries both report PASS. The recorded ratios are
1.888594 for training-only time and 1.397335 for total time.

Audit 002 section 6 needs three clarifications for the notebook:

- Finite loss does not imply finite gradients. Any future pre-update guard must
  reach an all-rank decision before any rank stops or skips an update.
- `Flux.train!` is not a workaround for asymmetric non-finite losses (DB-1).
  The notebook can retain the existing explicit fixed-step driver.
- Equal participation is the requirement. The existing saved Phase K schedule
  already supplies equal batch sizes and update counts without padding.
  `DistributedDataContainer` is not required for this demonstration.

Proposed scope: existing two-/four-rank correctness artifacts, fixed model and
native Adam configuration, data-only fresh-process continuation, and the corrected
one-/four-A100 timing results. Numerical-failure recovery and untested GPU paths
remain documented limitations. User acceptance remains pending.

**Exact next action:** obtain the user's decision on these demonstration
preconditions, then create `notebook/conditional_graph_gpu_recap.ipynb`.

## Timing corrections applied and re-measured on Leonardo — 2026-10-06

The 2026-10-06 [review](../../reports/conditional-graph-phase-k-review.md)
found three timing findings in the optional post-Phase K stretch. All three are
corrected and the campaign was re-measured.

- P1: the timed single-GPU baseline took asymmetric diagnostic host copies
  (480 `Array` copies per 120-step repeat through `CGReference.reference_step!`
  absent-head snapshots); DDP had none. Timed and warm-up baseline steps now
  use the plain native `CGReference.reference_update!`; the assertions remain
  in the untimed `--check-trajectory` correctness path.
- P2a: timing did not isolate training-only duration (record writes stayed in
  `t_train_s`; no slowest-rank training-only field). A barrier-guarded
  diagnostic window now excludes validation and record writes; `repeats.tsv`
  gained `t_write_s` and `slowest_train_s`; `TIMING_SUMMARY` reports
  `median_total_s` and `median_train_s`.
- P2b: the speedup figure used the median of paired per-repeat ratios while the
  report used the ratio of medians. Driver and figure now both use the ratio of
  medians; the figure shows per-mode individual repeat durations.
- New driver phase `--phase timing` (baseline + DDP, one `summary-timing.txt`).
  `repeats.tsv` now has 15 columns; the strict plotter exits 2 on the old
  11-column header, so the 2026-10-05 campaign root is no longer replottable.

Corrected campaign (Leonardo, node `lrdn0009`, 4× A100-SXM-64GB, driver
535.274.02; correctness job `59514262` on `lrdn1934`, timing job `59514959` on
`lrdn0009`):

- Correctness preflight PASS: `single` trajectory `maxabs=2.980232e-08`, `ddp`
  `maxabs=0.000000e+00`, tolerances unchanged (`atol=2.0e-6`, `rtol=1.0e-5`).
- Timing: baseline 5/5 (1× A100), DDP 5/5 (4 NCCL ranks, devices `[1,2,3,0]`),
  fixed work 120 updates × batch 128 = 15360 samples per repeat, 0
  `Precompiling packages` lines, no Flux source change, no commits.
- Corrected headline (ratio of medians): training-only speedup
  `0.263440 / 0.139490 = 1.888594`; total speedup
  `0.438678 / 0.313939 = 1.397335`.
- Historical instrumented run (2026-10-05) reported total speedup 1.480203
  (baseline median `t_total_s` 0.465324, DDP median `slowest_rank_s` 0.314365,
  baseline median `t_train_s` 0.295540). The corrected baseline median
  `t_train_s` is 0.263440 (−0.032100 s, −10.9%), consistent with removing the
  diagnostic host copies; the corrected DDP slowest total is 0.313939
  (−0.14%). The historical 1.480203 describes the instrumented runs only and
  is superseded.

Files: run root
`artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/`
(`summary-correctness.txt`, `summary-timing.txt`, both status PASS); local
driver logs `artifacts/logs/leonardo/2026-10-06-timing-correctness-r2.log` and
`...-timing-r2.log`; figures regenerated (8 PNGs plus `figures-metadata.txt`)
under `reports/figures/phase-k-timing/`; rewritten report
`reports/conditional-graph-phase-k-timing-stretch.md`; campaign record
`artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/CAMPAIGN.md`;
devlog [2026-10-06-timing-corrections](../devlog/02-conditional-graph/2026-10-06-timing-corrections.md).

Timeline: local implementation and independent local CPU/MPI verification,
then remote correctness preflight and timing campaign, then documentation. The
core correctness matrix remains untouched and valid; the GPU recap notebook
next action is unchanged.

**Exact next action:** review the deferred findings and decide the
demonstration preconditions (report section 6); then create
`notebook/conditional_graph_gpu_recap.ipynb` from the existing two-rank and
four-rank correctness artifacts.

## Deferred-behavior report complete — 2026-10-06

The deferred-behavior investigation is complete; it was the last remaining
Phase K item. Report:
[002-conditional-wrapper-deferred-behaviors](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md).

New bounded two-rank CPU/MPI probes characterized the wrapper boundaries.
Final run: **8/8 modes PASS, 83 assertions, 0 failures, exit 0**
(summary `artifacts/logs/conditional-wrapper/deferred/check-summary-2026-10-06.log`;
raw logs and `deferred-metadata.md` in the same directory). Runtime-confirmed
findings:

- DB-1: one rank's non-finite loss makes `Flux.train!` hang (the offender
  throws `DomainError` after skipping its update; other ranks block in the
  presence `allreduce!`).
- DB-2: all-rank non-finite loss is a clean coordinated stop with the model and
  state unchanged.
- DB-3: a finite loss with a non-finite gradient silently corrupts parameters
  and Adam moments on every rank (no gradient finiteness check exists).
- DB-4: a manual `trainstep!` loop with one skipped update silently shifts
  gradient pairing (`delta_vs_shifted=0.0` against an independent reference)
  and then hangs.
- DB-5: a rank-local loop stop hangs the remaining ranks.
- DB-6/DB-7: rank-local learning-rate or frozen-flag changes diverge replicas
  silently with an unchanged collective schedule (accepted configuration
  precondition).
- DB-8: `synchronize!!` excludes rules/frozen/backend and is not a checkpoint
  restore (accepted contract).
- DB-10: non-finite behavior is untested under CUDA/NCCL; recorded GPU failures
  are harness-only.

No Flux source or test changed; no commits; no remote execution. Probe/driver:
`scripts/checks/conditional_graph_deferred_probe.jl` and
`..._check.jl`; follow-up backlog A1-A8 is in the report (section 7) and the
implementation plan (section 21). Probe-tooling iterations are recorded in
[the failure note](../failures/2026-10-06-deferred-probe-driver-defects.md).

**Exact next action:** review the deferred-behavior report and decide which
deferred findings the demonstration carries (report section 6); then proceed to
the GPU recap notebook or the next campaign. No other Phase K item remains.

## Phase K K2 network diagram — 2026-10-06

Added `scripts/plots/plot_phase_k_network.py`, which parses the K2 network
(`TwoHeadModel`: trunk `Dense(784 => 32, tanh)`, two `Dense(32 => 10)` heads,
25 780 Float32 parameters) and the six-step `ROUTING_CYCLE` from the example
sources and writes two deterministic figures plus provenance metadata under
`reports/figures/phase-k-network/`. The recorded source hashes match the K2
deathstar campaign `prepare_metadata.txt`; repeated runs produce byte-identical
PNGs. No training ran; no Flux source changed.

Devlog: [2026-10-06-phase-k-network-figure](../devlog/02-conditional-graph/2026-10-06-phase-k-network-figure.md).

**Exact next action:** unchanged — create `notebook/conditional_graph_gpu_recap.ipynb` from the existing two-rank and four-rank correctness artifacts; the deferred-behavior report remains pending.

## Phase K review and GPU notebook plan — 2026-10-06

The implementation review found no blocking defect in the inspected correctness path.
Stored K2 GPU evidence passed local checks: two-rank resume 236/236 and four-rank resume 472/472.
Local figure regeneration produced 17 PNGs plus metadata. No new training ran.

The optional timing path needs corrections before a new training-speed comparison.
The timed single-GPU baseline copies absent-head parameters and Adam state to CPU, unlike DDP.
The timing fields also retain record-write overhead, and the speedup figure uses a different summary statistic.
The historical 1.480203 ratio describes those instrumented runs only.

Review and notebook outline: [Phase K review](../../reports/conditional-graph-phase-k-review.md).
The required correctness matrix remains complete. The deferred-behavior report remains pending.

**Exact next action:** create `notebook/conditional_graph_gpu_recap.ipynb` from the existing two-rank and four-rank correctness artifacts.

## Phase K timing stretch complete on Leonardo — 2026-10-05

The optional post-Phase K timing stretch passed its correctness preflight and
completed 5/5 timed repeats in both modes on Leonardo (4× A100-SXM-64GB,
driver 535.274.02). Jobs: correctness `lrdn0790` (salloc `59434349`),
baseline `lrdn1990` (`59434611`), DDP `lrdn1847` (`59435161`), GPU check
`lrdn3078` (`59434335`). No Flux source changed; no commits; 0
`Precompiling packages` lines in every console log.

- Correctness preflight PASS: local CPU 4-rank `single`-vs-`ddp` snapshot check
  `TIMING_CHECK_RESULT source=single-vs-ddp PASS maxabs=2.980232e-08`; Leonardo
  trajectory checks `single` `maxabs=2.980232e-08` and `ddp`
  `maxabs=0.0`, tolerances unchanged (`atol=2.0e-6`, `rtol=1.0e-5`).
- Fixed work: 120 global updates × global batch 128 = 15360 samples per repeat
  (a=6400, b=8960). Baseline = 1 process on 1 A100; DDP = 4 NCCL ranks on
  devices `[1,2,3,0]`.
- Baseline `t_total_s` per repeat 0.626333, 0.439836, 0.832694, 0.465324,
  0.444683; median 0.465324 (median 257.884817 updates/s).
- DDP `slowest_rank_s` per repeat 0.714849, 0.344812, 0.282049, 0.314365,
  0.266138; median 0.314365 (median 381.721898 updates/s).
- Measured diagnostic `speedup = 0.465324 / 0.314365 = 1.480203` for this
  tiny fixed schedule; not a production-speedup claim.
- 8 PNGs plus `figures-metadata.txt` in `reports/figures/phase-k-timing/`.
  Live viewer captured WAITING/RUNNING/DONE
  (`.../stretch-2026-10-05-leonardo/viewer/timing-live.png`), with the
  documented caveat that the ssh stream arrived in a late buffered burst.

Evidence: report
[conditional-graph-phase-k-timing-stretch.md](../../reports/conditional-graph-phase-k-timing-stretch.md);
campaign record
[CAMPAIGN.md](../../artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/CAMPAIGN.md);
devlog [2026-10-05-phase-k-timing-stretch](../devlog/02-conditional-graph/2026-10-05-phase-k-timing-stretch.md).

**Exact next action:** write the deferred-behavior report
(`temp/conditional_graph_deferred_behaviors_prompt.md`); it is the only
remaining Phase K item. Nothing else from Phase K is pending.

## Phase K complete on Leonardo (4-rank NCCL) — 2026-10-05

Both remaining Phase K acceptance cells PASS on Leonardo (4× A100-SXM-64GB,
driver 535.274.02) with CUDA/NCCL at 4 ranks; driver exit 0 for both roots.
No Flux source changed this session.

- K1 toy, node `lrdn3394`, root
  `artifacts/logs/conditional-wrapper/demo/k1-2026-10-05-leonardo-nccl4`:
  prepared ranks=1 exit 0 (12.7 s), `nccl4-train` ranks=4 exit 0 (183.6 s),
  `nccl4-resume` ranks=4 exit 0 (67.7 s); testset
  `Phase K toy campaign (nccl) | 8 8 4m27.5s`;
  `PHASE_K_SUMMARY dataset=toy backend=nccl ranks=4 PASS`; `DEMO_SCHEMA_OK`
  train `validation_steps=17`, resume `validation_steps=15`; 64/56
  `DEMO_STEP` records; `resume_match` 56/56; collectives 24×5 + 40×7 (train)
  and 24×5 + 32×7 (resume); maxima (train = resume) param 2.980e-08,
  mom1 2.980e-08, mom2 1.164e-10; `peer_beta_exact=false` count 0.
- K2 MNIST/Fashion-MNIST, node `lrdn0436`, root
  `artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4`:
  `nccl4-train` ranks=4 exit 0 (184.657 s), `nccl4-resume` ranks=4 exit 0
  (77.235 s); testset `Phase K mnist-fashion campaign (nccl) | 6 6 4m25.6s`;
  `PHASE_K_SUMMARY dataset=mnist-fashion backend=nccl ranks=4 PASS`;
  `DEMO_SCHEMA_OK` train `validation_steps=121`, resume
  `validation_steps=119`; 480/472 `DEMO_STEP` records; `resume_match`
  472/472; collectives 160×5 + 320×7 (train) and 160×5 + 312×7 (resume);
  maxima (train = resume) param 5.960e-08, mom1 4.657e-09, mom2 1.455e-11;
  `peer_beta_exact=false` count 0.
- Both runs map rank 0→1, 1→2, 2→3, 3→0 (`devices=[1, 2, 3, 0]`).
  Tolerances unchanged (`atol=2.0e-6`, `rtol=1.0e-5`); the nonzero 4-rank
  maxima are Float32 accumulation differences inside them.
- K2 reused the deathstar prepared dataset
  `/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls`
  (SHA-256 `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6`,
  39804054 bytes); schedule digest
  `ce6b41aa81c9eb9d3c4cc526a25f50e4e302738e47e660db0b4e5ad2aa78bc57`
  (identical to deathstar mpi4); dataset digest
  `658e3dd341ca4e4fa9e5e4a05eacf53c0bebca626a05b17a87e917062dec2971`;
  exposures a=6400 / b=8960, local batch 32, 120 steps, Adam eta=0.001. No
  download on compute nodes and 0 `Precompiling packages` lines.
- Figures: `reports/figures/phase-k-k2/` regenerated to 41 PNGs plus
  `figures-metadata.txt`; 8 new `k2-2026-10-05-leonardo-nccl4-nccl4-*` files.
- Local driver logs: `artifacts/logs/leonardo/2026-10-05-k1-leonardo-nccl4.log`
  and `...-k2-leonardo-nccl4.log`.

Evidence: campaign record
[CAMPAIGN-2026-10-05-leonardo-k2.md](../../artifacts/logs/conditional-wrapper/demo/CAMPAIGN-2026-10-05-leonardo-k2.md);
devlog [2026-10-05-leonardo-k1-k2-4rank-nccl](../devlog/02-conditional-graph/2026-10-05-leonardo-k1-k2-4rank-nccl.md);
failure record
[2026-10-05-leonardo-k1-k2-nccl4](../failures/2026-10-05-leonardo-k1-k2-nccl4.md).

**Exact next action:** the required Phase K acceptance matrix is complete; only
the optional post-Phase K live one-A100 versus four-rank A100 timing stretch
and the deferred-behavior report remain unrun.

## Phase K K2 complete on deathstar — 2026-10-05

The K2 task-conditioned MNIST/Fashion-MNIST campaign is complete on `deathstar`.
All driver summaries are PASS: smoke2 mpi2 8/8, K1 toy regression mpi2 8/8,
full CPU/MPI (mpi2 and mpi4, 120 steps) 13/13, CUDA/NCCL 2 ranks 6/6.

- mpi2 and NCCL parameter/Adam-moment maxima are exactly 0.0 with
  `resume_match` 236/236; mpi4 maxima are within the declared Float32
  `atol=2e-6, rtol=1e-5` (train param 7.451e-09, resume param 5.960e-08) with
  `resume_match` 472/472. Tolerances were not changed.
- Final validation (initial digits 0.1170, clothing 0.0460): mpi2 and NCCL
  digits acc 0.8560 / loss 0.6935, clothing acc 0.7270 / loss 0.7968
  (exposure 7680/7680); mpi4 digits acc 0.8470 / loss 0.7658 (exposure 6400),
  clothing acc 0.7390 / loss 0.7740 (exposure 8960). mpi2 and NCCL
  trajectories are identical; mpi4 is a different routed objective by design
  (steps 1-2 route A to 1 of 4 ranks and B to 3 of 4).
- Per-step evidence covers parameters, both Adam moments, decay products,
  rules/frozen/shape/dtype/device, local and global AD absence, equal update
  counts, rank-vs-reference and rank-vs-rank comparisons, and the ordered
  collective trace per rank on every step (5 collectives when one head is
  globally absent, 7 otherwise).
- One defect was found by the smoke and fixed:
  `CGCheckpoint.checkpoint_digest` was unstable across a `Serialization`
  roundtrip. Canonicalization (Dict/NamedTuple/Tuple/AbstractVector) plus the
  new `scripts/checks/conditional_graph_checkpoint_roundtrip.jl` regression
  (RED then GREEN) close it. Pre-fix checkpoints are intentionally
  incompatible.
- Presentation assets: 33 PNGs plus `figures-metadata.txt` under
  `reports/figures/phase-k-k2/`; supervisor report
  `reports/conditional-graph-phase-k.md`.

Evidence: `artifacts/logs/conditional-wrapper/demo/CAMPAIGN-2026-10-05-deathstar-k2.md`;
devlog `wiki/devlog/02-conditional-graph/2026-10-05-phase-k-k2-mnist-implementation.md`; failure
record `wiki/failures/2026-10-05-phase-k-k2-campaign.md`.

Unrun/pending (not counted as passing): the optional post-Phase K live
one-versus-four A100 timing comparison and the deferred-behavior report. The
Leonardo 4-rank K1/K2 correctness cells now PASS; see the top section.

**Exact next action:** run the optional live one-A100 versus four-rank A100
timing stretch and write the deferred-behavior report.

## Phase K visualization plan — 2026-10-05

The Phase K prompt and implementation plan now require reference-overlay validation loss and accuracy curves for each task.
They also require plots of the configured Adam learning rate for DDP and the independent reference.
The reference consumes each distributed run's saved schedule in one process, including its task weights and virtual shards.
Separate metric records cover initialization, every update, task exposure, and checkpoint continuation.

An optional post–Phase K stretch compares one A100 with four NCCL ranks on Leonardo through an external live chart.
The timing runs use equal work, separate jobs, warm-up, CUDA synchronization, and repeated measurements.
The chart can compare live DDP progress with a saved baseline and replay explicitly labeled records.
Speedup remains unmeasured. The small model can be communication-bound.

This session changed documentation only. Existing K1 evidence remains the latest runtime evidence.
K2 and the required Leonardo K1 cell remain pending.
Details: `temp/conditional_graph_phase_k_prompt.md`, sections 8 and 11.

**Exact next action:** inspect the K1 metric schema and native reference, then extend K2 with separate reference metrics and learning-rate records.

## Phase K toy implementation complete — 2026-10-04

The deterministic toy case from `temp/conditional_graph_phase_k_prompt.md` (K1)
is implemented and validated on CPU/MPI and on native CUDA/NCCL.

- New files: `scripts/examples/conditional_graph_demo.jl` and
  `scripts/examples/conditional_graph/{common,model,dataset,schedule,reference,checkpoint}.jl`;
  bounded campaign driver `scripts/checks/conditional_graph_demo_check.jl`.
- Contract per step: shared-trunk two-head model, saved six-step routing cycle,
  comparison against an independent single-process native Adam reference,
  ordered collective trace (7 or 5 collectives), data-only Serialization
  checkpoints with fresh-process continuation, rank-0 metrics.
- CPU/MPI matrix (2 and 4 ranks) PASS, 9/9 driver assertions: per-step maximum
  parameter error 0.0 (R=2) and 1.11e-16 (R=4); moments ≤5.6e-17; `beta_exact`
  on all rows. Evidence:
  `artifacts/logs/conditional-wrapper/demo/toy-2026-10-04-local-mpi/`.
- deathstar CUDA/NCCL 2 ranks PASS, 6/6 driver assertions: per-step errors
  exactly 0.0 against the same-device Float32 native reference; 20×7 + 12×5
  collectives per rank at the scheduled presence pattern; task-B validation
  accuracy 0.000 → 1.000 by step 11; fresh-process resume matched all 28
  markers. Evidence: `.../toy-2026-10-04-deathstar-nccl2/`.
- Toy learning rate set to 1e-2: the prompt's 1e-3 (a recommendation) leaves
  head B inverted at 0.0 accuracy after the 16-update budget for the prepared
  seed; documented in the devlog.
- First NCCL attempt failed in GPU-only demo code (`CUDA` referenced in a
  device-generic module); recorded in
  `wiki/failures/2026-10-04-phasek-nccl-cuda-ref.md`, failed console kept under
  `toy-2026-10-04-deathstar-nccl2-attempt1/`.
- No Flux production or test source changed; runs use the existing wrapper at
  `55171e65` through `--project=../flux-integration/test`.

Devlog: `wiki/devlog/02-conditional-graph/2026-10-04-conditional-wrapper-phase-k-toy-implementation.md`.

**Exact next action:** implement K2, the task-conditioned MNIST/Fashion-MNIST
presentation campaign (120 updates, rank-0 metrics, fresh-process
continuation), and run it through the same driver at the validated
configurations.

## Phase K task-conditioned dataset decision — 2026-10-04

The user requested a standalone Phase K prompt and implementation discussion.
The user then selected toy diagnostics followed by a task-conditioned MNIST/Fashion-MNIST demonstration.
The prompt is `temp/conditional_graph_phase_k_prompt.md`.
The presentation model shares a trunk and uses separate ten-class digit and clothing heads.
Known dataset identity selects the head without the target class or a learned router.
The saved schedule records task identity and samples together for the independent native Adam reference.
Mixed-task, single-task, and reactivation steps exercise local absence and global absence after nonzero Adam history.
It also specifies per-step comparisons, ordered collective traces, and data-only checkpoints with fresh-process continuation.
The presentation reports task-specific metrics and sample exposure counts.
Suggested model sizes, sample counts, and run lengths are recommendations for discussion.
No Phase K implementation or runtime result exists from this session.

**Exact next action:** implement the saved toy schedule and independent native reference from the isolated prompt.

## Leonardo 4-rank NCCL validation complete — 2026-10-04

The previously deferred 4-rank/NCCL weighting case is closed on CINECA
Leonardo (4× A100-SXM-64GB, driver 535.274.02, MPICH via `srun --mpi=pmi2`):

- Focused Phase J regression: **217/217 assertions per rank**, exit 0; ranks on
  devices `[1, 2, 3, 0]`. Log:
  `artifacts/logs/conditional-wrapper/phaseJ-gpu-focused-nccl4-leonardo-2026-10-04.log`.
- Full dedicated suite at 4 ranks under both backends:
  `Distributed | 14 14 5m55.7s`, exit 0. Log:
  `phaseJ-remote-suite-nccl4-leonardo-2026-10-04.log`.
- 4-rank MPI and NCCL smokes pass; stack check: CUDA/cuDNN functional, 4 A100s.
- Bring-up: Julia 1.12.6 on `$WORK`, `scripts/remote/hosts/leonardo.conf`,
  GPU-node precompile (278 deps, 221 s), login-side CUDA artifact prefetch
  (`scripts/remote/leonardo/fetch-cuda-artifacts.jl`).
- Traps recorded in `wiki/failures/2026-10-04-leonardo-cuda-artifact-prefetch.md`
  (no-internet compute nodes, driverless precompile, `cd && srun` wrapper).
- No product change; `flux-integration` still dirty at `55171e65`.

The pre-bring-up readiness record
`artifacts/logs/leonardo/2026-10-04-node-readiness.md` (written before Julia was
installed) is superseded by this entry and by the updated
`wiki/context/leonardo.md`.

**Exact next action:** Phase K (plan section 12): deterministic toy dataset
reference first, then the MNIST presentation run, on the validated 2-rank
(deathstar) and 4-rank (leonardo) GPU configurations.

## Phase J complete — 2026-10-04

Native CUDA/NCCL validation passed on `deathstar` at 2 ranks (4 GPUs are
unavailable on both hosts; the user authorized the 2-rank gate).

- Focused device-generic regression `test/ext_distributed/cuda_conditional_distributedtest.jl`
  on real `CuArray` parameters with `CUDA.allowscalar(false)`: **217/217
  assertions per rank**, 6 testsets, exit 0, 52.2 s; ranks mapped to devices 1
  and 2. Log: `phaseJ-gpu-focused-nccl2-deathstar-2026-10-04.log`.
- Complete dedicated suite, MPI only: `Distributed | 7 7 2m03.6s`, exit 0.
  Stretch: with `FLUX_TEST_DISTRIBUTED_NCCL=true`, all 7 child files under both
  backends: `Distributed | 14 14 5m01.9s`, exit 0.
- No product failure; no production source changed. Test-only changes: the new
  file, a `cuDNN` import in the NCCL branch of `distributed_setup.jl` (required
  because `MLDataDevices.functional(CUDADevice)` is defined only in its cuDNN
  extension), and a device-aware `NUMERIC_ATOL` for Float32 reference
  comparisons.
- Both harness failures and the deferred cases are recorded in
  `artifacts/logs/conditional-wrapper/phaseJ-metadata.md` and
  `wiki/failures/2026-10-04-phaseJ-nccl-harness-failures.md`.
- Flux remains dirty at `55171e65`; nothing committed or pushed.

**Exact next action:** Phase K (plan section 12): deterministic toy dataset
reference first, then the MNIST presentation run, on the 2-rank GPU
configuration that passed Phase J.

## Phase J attempt — 2026-10-04 (superseded by the completion entry above)

Phase J (native CUDA/NCCL validation) implementation is complete in the
`flux-integration` worktree and CPU-validated; the native GPU gate has **not**
passed yet. Handoff: `temp/phaseJ_handoff.md`; devlog:
`wiki/devlog/02-conditional-graph/2026-10-04-conditional-wrapper-phase-j-native-gpu-attempt.md`.

- New device-generic regression `test/ext_distributed/cuda_conditional_distributedtest.jl`
  (CuArray under NCCL with `allowscalar(false)`; identical body on host Arrays
  under MPI). Local 2-rank CPU evidence: 212 assertions/rank; full dedicated
  suite 7/7, exit 0 (log `phaseJ-local-suite-mpi2-2026-10-04.log`).
- Test env now adds CUDA, NCCL, cuDNN in `test/Project.toml`; runner imports
  CUDA only for the NCCL pass.
- deathstar is healthy (3× TITAN RTX, driver 555.42.02, CUDA functional). The
  stale Phase F remote manifest (StridedViews conflict) was backed up as
  `Manifest.toml.phaseF-bak` and regenerated; precompile passed (188 deps,
  137 s; Flux 0.16.11, Optimisers 0.4.9, Functors 0.5.3, MPI 0.20.27, CUDA
  6.2.2, NCCL 0.1.3).
- GPU run blocked by a diagnosed harness issue: `FluxMPIExt` calls
  `MLDataDevices.functional(CUDADevice)`, which in MLDataDevices 1.17.10 is
  defined only in its cuDNN extension; without cuDNN the fallback is `false`
  and `force_cuda` errors. cuDNN was added to `test/Project.toml` locally and
  must now be synced, resolved, precompiled, and rerun on deathstar.
- Sync hazard: the root workspace `Manifest.toml` must be excluded from the
  flux-integration rsync (or re-resolved after every sync), otherwise the stale
  local manifest overwrites the regenerated remote one.
- 4-GPU runs are impossible on both hosts; the user authorized the 2-rank gate.

**Exact next action:** follow section 8 of `temp/phaseJ_handoff.md`: sync with
the root-manifest exclusion, re-run `.phaseJ_resolve_env.jl` on deathstar, then
run the focused 2-rank NCCL regression.

## Phase I user decisions — 2026-10-02

The user accepted the wrapper API, explicit `.tree` migration, synchronization contract, and fresh-backend checkpoint recipe for a supervisor prototype.
The user requested native GPU validation and a concrete dataset-level demonstration.
GPU-specific failures can move to the next upgrade batch after characterization. The CPU evidence remains the current correctness baseline.

The implementation plan now defines Phase J native CUDA/NCCL validation and Phase K toy/MNIST validation.
The demonstration will compare complete training trajectories against an independent native Optimisers global-batch reference.
The existing example still calls the removed `resolve_unused_parameters!!` helper and cannot establish the current wrapper behavior unchanged.
The current conditional fixture uses CPU `Vector` fields. Selecting NCCL alone does not prove CUDA gradient communication.

The later agent prompt is `temp/conditional_graph_deferred_behaviors_prompt.md`.
It requests an evidence-based report on non-finite values, rank participation, configuration mismatch, state boundaries, and available GPU findings.
The report must distinguish realistic reachability from unmeasured prevalence before recommending deferral.
No new runtime result exists from this planning update.

**Exact next action:** implement the focused native CUDA/NCCL regression in section 12, Phase J1, of the implementation plan.

## Phase I planning — 2026-10-02

The user requested guidance for Phase I and performance goals after stable conditional-graph support 1.0.
The implementation plan now includes an acceptance agenda in section 12 and a C9/C10 roadmap in section 23.
Phase I acceptance and the stable 1.0 backend scope remain pending.

Source inspection shows one CPU presence flag per synchronization buffer, not per scalar weight.
Each update constructs the identity map, occurrence plan, gradient buffers, and presence arrays again.
The roadmap separates traversal/allocation costs from gradient bytes and collective latency.
It covers reusable storage, cache lifecycle, SIMD candidates, safe threading, and metadata transport.
These are source-based hypotheses and future goals. No profiling or performance result exists from this session.

The combined Phase H correctness evidence remains the latest result. Flux retains its existing uncommitted changes on `55171e65`.
Session record: [Phase I planning](../devlog/02-conditional-graph/2026-10-02-conditional-wrapper-phase-i-planning.md).
**Exact next action:** review the Phase I decision table, starting with the supported backends and release limitations.

## Current checkpoint — 2026-10-02

Independent Phase H review found no blocking defects within the documented CPU/MPI scope.
The combined dedicated suites passed 6/6 child files at two ranks (1 m 58.7 s) and four ranks (2 m 22.8 s).
Checkpoint continuation passed 204 assertions across two fresh MPI jobs. The ambiguity count is zero.
Flux retains its existing uncommitted changes on `55171e65`.
Evidence and exact commands: [Phase H review](../devlog/02-conditional-graph/2026-10-02-conditional-wrapper-phase-h-review.md).
**Exact next action:** conduct Phase I user review of the API, synchronization contract, migration policy, remaining limitations, and combined evidence.

## Current checkpoint — 2026-10-01

The four Phase H issue agents completed their focused work sequentially.
The scalar-state correction also preserves tied-leaf identity. Both have RED/GREEN evidence and continuation regressions.
The explicit synchronization contract covers supported `Leaf.state` values. Rules and frozen flags remain local and must match across ranks.
The state-shape review retains explicit `.tree` paths and documents checkpoint migration without forwarding methods.

Latest evidence: **499/499 assertions per rank** in the synchronized file at two ranks, plus **204 assertions** across checkpoint write/resume jobs.
The checkpoint jobs passed with Serialization and JLD2, fresh backends, and nonzero frozen-leaf history.
Reports and exact commands: `temp/phase_h/01-scalar-state.md`, `02-tied-leaf-identity.md`, `03-rule-frozen-contract.md`, and `04-state-shape-migration.md`.
Flux HEAD remains `55171e65`, with uncommitted source, tests, and NEWS changes.
**Exact next action:** review the combined diff and run the dedicated CPU/MPI gates at two and four ranks with fresh evidence filenames.
Full combined gates and Phase I acceptance remain pending. The non-finite-loss risk remains outside these four corrections.

### Historical Phase H audit result, before corrections

Phase H compatibility audit is complete. The report covers all requested interfaces, state-shape migration, checkpoint continuation, and a proposed Optimisers pre-update hook.
Report: [wrapper compatibility audit](../audits/02-conditional-graph/001-conditional-wrapper-compatibility.md).

Two sequential local MPI jobs passed **178 assertions**: 54 per rank for characterization and 35 per rank for fresh-process resume.
Both jobs used two ranks and returned 0. Serialization and JLD2 continuation preserved the tested model ties, leaf ties, Adam state, and frozen flags.
Each rank loaded the data-only checkpoint and attached a fresh backend.

The audit found synchronization limitations. Wrapper `synchronize!!` discards replacement scalar state, although mutable moment arrays synchronize.
The native synchronized tree loses tied `Leaf` identity. Rules and frozen flags remain rank-local.
These are passing characterizations of unsupported behavior, not resolved defects.
The report also records a source-only risk from rank-local non-finite losses in `Flux.train!`.

Flux remains clean at `55171e659b2f425da980cbc83177bf5c38f0a7a1`.
Exact commands and evidence: `artifacts/logs/conditional-wrapper/phaseH-metadata.md` and `wiki/devlog/02-conditional-graph/2026-10-01-conditional-wrapper-phase-h-compatibility.md`.
**Next action:** conduct Phase I user review of the production diff, CPU/MPI evidence, compatibility report, remaining limitations, and Optimisers discussion points.
The review must decide the synchronization contract and state-shape migration policy before further implementation or native NCCL work.

## Earlier session records — Phase G

Independent Phase G review (2026-09-30): no blocking findings. The local two-rank rerun passed 210/210 assertions, exit 0, with a testset time of 1 m 5.2 s. Both ranks completed all three modes with the expected values and collective order. Flux remains clean at `55171e65`, and `git diff --check` passed. The review covered the detector, probe, Makefile target, and stored evidence. Timeout termination was inspected but not exercised in this review. Exact commands and scope are in the Phase G devlog. Next action: Phase H, wrapper compatibility review.

Phase G complete (2026-09-30): the external detector passed **210/210 assertions** locally at two MPI ranks, exit 0, 1 m 7.1 s. Both ranks finished `deadlock`, `corruption`, and `control` with correct values. Traces prove one CPU presence reduction followed by gradients in model order (5/4/5 collectives per rank). The Makefile now uses the selected Flux test environment. A first-run trace-framing defect is corrected and recorded in `wiki/failures/2026-09-30-conditional-detector-trace-framing.md`. Flux remains clean at `55171e65`. Next action: **Phase H, wrapper compatibility review**. Command: `make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration`. Evidence: `wiki/devlog/02-conditional-graph/2026-09-30-conditional-wrapper-phase-g-detector.md` and `artifacts/logs/conditional-wrapper/phaseG-detector-GREEN-mpi2-2026-09-30.log`.

## Earlier session records

Phase F (CPU/MPI gates) complete on `deathstar` (2026-09-30): **all gates GREEN at two and four ranks**. Focused conditional file 367/367 per rank at 2 ranks (exit 0, 1 m 3.6 s) and at 4 ranks (exit 0, 1 m 7.9 s); focused optimizer 6/6 per rank (8.5 s); complete dedicated suite `Distributed | 6 6 1m33.5s` at 2 ranks and `Distributed | 6 6 1m43.6s` at 4 ranks, all exit 0; 0 ambiguities; `git diff --check` clean. The four-rank weighting is proven analytically by the first testset (rank 0 uses branch A = 1/4 weight; ranks 1-3 use branch B = 3/4 weight; both updated values asserted on every rank). The correct worktree was synced manually with rsync because `sync_code.sh` syncs the frozen C8 `ddp_flux`, not `flux-integration`; the remote conditional test md5 matched (`c26ef49b...`). Flux was unchanged at `55171e65` (Phase F changed no source or tests). Precompile on deathstar: 128 deps in 115 s. The remote environment was verified after a brief VPN drop was restored: Optimisers 0.4.9, MPI 0.20.27, Flux 0.16.11, MPICH launcher artifact identical to local, and the three remote source/test md5s match the local worktree exactly. Next action: Phase G (convert the deadlock detector to a success regression). Details: `wiki/devlog/02-conditional-graph/2026-09-30-conditional-wrapper-phase-f-cpu-mpi-gates.md`, `artifacts/logs/conditional-wrapper/phaseF-metadata.md`.

Remote tooling update (2026-09-30): the `hpc` SSH alias and every reference were renamed to `bocconi` (real hostname `slogin.hpc.unibocconi.it` unchanged; SSH config alias only, no key change). `scripts/remote/` was cleaned up: stale `run_tests.sh` and `setup_mpi.jl` deleted, `update_wiki.sh` moved to `scripts/setup/`, `hosts/hpc.conf` -> `hosts/bocconi.conf` (wrapper byte-identical), `run.sh` now rejects unknown host names, and both syncs use `rsync --delete` with the remote-resolved `test/Manifest.toml` protected. Per-server docs now live in `wiki/context/bocconi.md` / `deathstar.md` / `leonardo.md` (new stub) behind a slimmed `remote` skill index. No Flux code, tests, or checkpoint state changed. Next action unchanged: Phase F (two- and four-rank CPU/MPI gates; the four-rank case must prove weighting). Details: `wiki/devlog/02-conditional-graph/2026-09-30-remote-tooling-bocconi-rename.md`.

Independent Phase E follow-up (2026-09-30): **both P2 findings closed; Phase E approved within its CPU/MPI scope**. The corrected adaptation test transforms nonzero Adam arrays and updates through the adapted state. The missing M1.1 diagnostic and delegation checks are present. Independent focused run: 367/367 per rank at two MPI ranks, exit 0, 78.972 seconds wall time. Flux diff remains tests-only (+560/-0), with clean whitespace. Stored correction logs support the optimizer 6/6, complete suite 6/6, and zero-ambiguity results. Exact review command is in the Phase E devlog. Next action: Phase F, focused child followed by the dedicated suite at two and four ranks, with analytical four-rank weighting.

Phase E review correction (2026-09-30): **both P2 gaps corrected; correction gate GREEN**. The device-adaptation test now uses Adam, transforms exactly the two nonzero state arrays (`mt`, `vt`), asserts adapted types, preserved values, backend identity, and an untouched source state, then updates through the adapted state and matches a native Adam reference. The M1.1 API checks are restored (`Chain tuple-gradient diagnostic`, `Enzyme Duplicated model setup and update`, the shared-gradient tied subcase, whole-wrapper `freeze!`/`thaw!`), and the freeze test now uses Adam with retained returned states, frozen-moment assertions, and a native reference. Correction gate at two ranks: focused conditional **367/367 per rank** across 30 testsets (exit 0, 1 m 22.0 s; aggregate `conditional file | 367 367 | 1m22.0s`), optimizer 6/6 (10.3 s), full dedicated suite `Distributed | 6 6 2m02.6s` (exit 0, 2 m 4.8 s), 0 ambiguities, `git diff --check` clean. The correction runs passed on the first attempt; no production defect and no production change. Test-only diff is +560/−0, committed at Flux `55171e65` ("Add Phase E conditional wrapper tests and restore M1.1 API checks", not pushed). Next action: Phase F (two- and four-rank CPU/MPI gates; the four-rank case must prove weighting). Evidence: `artifacts/logs/conditional-wrapper/phaseE-correction-*` and the updated `phaseE-metadata.md`; devlog "Corrections applied" section.

Date: 2026-09-30
Phase E (restore and improve the M1.1 test battery) complete, review-corrected, and GREEN, committed at Flux `55171e65`. `test/ext_distributed/conditional_distributedtest.jl` gained 15 testsets (+560 lines, tests only; no production change) ported from `3370b910` plus the two Phase E items and the restored M1.1 API checks: `real AD conditional branches average over multiple steps` (11), `Adam matches an independent global-mean reference` (9), `tied parameters through real AD` (13, incl. the M1.1 shared-gradient subcase), `tied parameter through a transpose with real AD` (5), `isbits parameters (FillArrays) sync independently` (5), `higher-order gradients rejected` (2), `Chain tuple-gradient diagnostic` (5, M1.1 API restore), `Enzyme Duplicated model setup and update` (5, M1.1 API restore), `collective count protocol with real AD` (4), `Flux.train! with DistributedOptimizerState` (3), `nested model (Chain of custom layer and Dense)` (2), `synchronize!! on DistributedOptimizerState` (3), `adjust / adjust! on wrapped state and rule` (11), `freeze! and thaw! preserve wrapper state and collectives` (31; Adam, retained states, whole-wrapper + partial delegation, collective counts, native reference), `Functor device adaptation keeps the backend` (26; Adam state arrays transformed and reused, native reference). M1.1 testsets already superseded by stronger Phase D5 testsets (BatchNorm, transpose/adjoint, caller gradients, immutable returns, globally absent Adam) were not duplicated. Two-rank GREEN: focused conditional **367/367 per rank** across 30 testsets, focused optimizer 6/6, full dedicated suite `Distributed | 6 6 2m02.6s`, 0 ambiguities, `git diff --check` clean. Optimisers 0.4.9. All RED iterations were new-test defects; no production defect was found. Next action: Phase F (two- and four-rank CPU/MPI gates; the four-rank case must prove weighting). Evidence: `artifacts/logs/conditional-wrapper/phaseE-*`; devlog `wiki/devlog/02-conditional-graph/2026-09-30-conditional-wrapper-phase-e-m11-battery.md`.

Phase D is complete and committed at `d6046103` (D1-D3 at `cbdd3edc`, D4 + D5 at `d6046103`, not pushed). Active implementation branch `ddp/integration-m1-spike`; Phase E is committed at `55171e65` in the same worktree (not pushed).

Independent follow-up review: **Both D5 P2 findings are closed.** The corrected source assertions cover active-to-absent Adam state and caller-gradient non-mutation. The independent two-rank conditional run passed 232/232 per rank (exit 0, 59.032 s). The production checksum matches the correction report, and Flux `git diff --check` passed. The stored mutation-test and full-suite logs support the report. D5 is approved within its CPU/MPI scope.

Review result (2026-09-30): **D5 reopened for two P2 test gaps, now corrected**. (1) The absent Adam parameter now takes two active steps first (nonzero moments), snapshots value and full state, then becomes globally absent; exact preservation is asserted, and a reset to the fresh state would fail because `snap_state != fresh.absent.state` is asserted before the absent steps. (2) The caller-gradient tests now use rank-distinct gradients on both update paths and a genuinely read-only gradient, compare each caller gradient with its local snapshot, and check each updated model against an independent global mean; identical gradients and the writable `view` could not detect aliasing. Targeted RED: with a temporary aliased buffer the corrected caller testset fails `gs.w == snap` with the global mean on both ranks (exit 1, 1 m 10.8 s); with a temporary absent-leaf zero-gradient emission the corrected absent testset fails both snapshots with changed moments and step products (exit 1, 1 m 14.3 s). The production source was restored byte-for-byte after each injection. Repeated D5 gate GREEN: conditional 232/232 per rank, optimizer 6/6, dedicated suite `Distributed | 6 6 1m45.2s`, 0 ambiguities, clean `git diff --check`. Exact commands and logs are in the D5 devlog correction section.

Phase D5 (regression gate) complete and committed at Flux `d6046103` (together with the corrected D4; not pushed). Eight focused testsets pin the retained contracts; the gate found and fixed one real defect: `_buildmap`/`_foreach_child` mapped multidimensional array containers over `eachindex`, flattening a 2×2 `Matrix` child to a length-4 vector, so array containers lost their shape during gradient reconstruction. The production now maps array containers over `CartesianIndices`. RED: focused 2-rank conditional exit 1, array-container 18 passed / 2 failed (`size` `(4,) == (2, 2)`, `axes` mismatch). GREEN after the review corrections: focused conditional 232/232 per rank, focused optimizer 6/6, full dedicated 2-rank suite `Distributed | 6 6 1m45.2s`, 0 ambiguities, `git diff --check` clean. Next action: Phase E (restore the M1.1 battery). Evidence: `artifacts/logs/conditional-wrapper/phaseD-d5-*`; devlog `wiki/devlog/02-conditional-graph/2026-09-30-conditional-wrapper-phase-d5-regression-gate.md`.

Review update: **D4 P1 resolved — D4 + D5 committed at `d6046103`**. Selection is now read from the setup state tree (`_sync_gradients(backend, tree, model, grads)`), so later updates keep exactly the leaves setup chose; `_trainable_child` is removed. RED2 confirmed the second-step defect, and the new two-step regression (both update paths, momentum state, collective counts) is GREEN. D5 passed on top of that, including the review corrections. Details and commands are in the D4 and D5 devlogs.

Active checkpoint: Phase K required matrix complete; the benchmark notebook is the remaining deliverable (see the top sections).
Active implementation branch: `ddp/integration-m1-spike` at `55171e65` in `/home/kurapica/Projects/ddp_flux/flux-integration` (Phase E committed; later-phase changes uncommitted in the worktree; not pushed).
Archived research branch: `archive/pr2694-unused-gradient-spike` at `3370b910`.
Flux repo paths: `/home/kurapica/Projects/ddp_flux/ddp_flux` (C8 dirty work, frozen) · `/home/kurapica/Projects/ddp_flux/flux-integration` (conditional-graph implementation worktree) · `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` (PR #2694 worktree; currently on `ddp/windows-path-fix` at `fee2c310` — work ONLY here for that PR)
PR #2694 head: `fee2c310`. Candidate base: `fa3bf228`. Windows-path fix commits: `69703025`, `fee2c310`.
Upstream merge base: `404ff37d`. The latest push fast-forwarded `ddp/upstream-pr` from `c64f695f` to `fee2c310`.

## What is known

- The local Flux.jl fork exists.
- The first implementation priority was understanding the current distributed code (C0-C6b done).
- CPU/MPI verification and upstream PR prep is mostly done.
- C7 is complete: an end-to-end example `scripts/examples/train_ddp.jl` was successfully created, executed on Bocconi using `make example-ddp`, and added as a documentation guide (`distributed.md`) to the Flux upstream docs.
- Bocconi GPUs are reachable via `salloc --gres=gpu:2` + `mpiexecjl` (NOT via `run.sh` which wraps in `bash -l` that resets env).
- Bocconi login node (slnode01) ≠ GPU compute node (gnode01); salloc grants allocation but `bash -c` runs on login node; use `mpiexecjl`/`srun` to launch on GPU nodes.
- **Precompilation (2026-09-03; the temporary `experiment/precompile` branch was deleted after adoption):** julia caches bake the host CPU feature set. Login node (graniterapids) and compute nodes (gnode01/02, icelake-server, mutually compatible) do NOT share caches. Old workflow precompiled on the login node (~10 min per Flux change, wasted) and the first compute-node test after a source change silently recompiled for ~4 min inside the test watchdog -> "timeouts on precomp". New workflow: precompile ONLY on compute nodes via `srun` (`scripts/remote/precompile.sh bocconi`, now runs `make precompile-all FLUX_REPO_PATH=../ddp_flux`), `make env` never precompiles, `JULIA_PKG_PRECOMPILE_AUTO=0` in bocconi.conf. Validated end-to-end on bocconi: warm 2-rank suite 4m25 with zero in-test compilation; ADR-0005 + devlog `2026-09-03-precompile-experiment.md`. CPU-target pinning (`JULIA_CPU_TARGET`/`-C`) was tested and rejected (env var ignored at load; gnode images carry `pconfig` absent on login).
- **M0 (2026-09-08, ADR-0006):** integration workspace isolated. Dirty master (`b633cdc7`) preserved untouched (working tree verified identical) and snapshotted on `archive/dirty-master-c8.2-wip` @ `b0034f55`. Worktree `/home/kurapica/Projects/ddp_flux/flux-integration` on branch `ddp/integration-m1-spike` @ `fa3bf228` (tip of `ddp/pr2694-ci-fix`), tree clean, no C8 content. Metadata, port inventory, and checklist in ADR-0006 (scope user-approved). Policy: dirty master not merged wholesale; C8 test expansions port later as separate commits only after M1 proof.
- **M1 (2026-09-08, commit `04ddeaa0`, NOT pushed):** automatic unused-gradient architecture. `DistributedOptimizer` stays `<: Optimisers.AbstractRule` but per-leaf `apply!` no longer communicates. New `DistributedUtils.DistributedOptimizerState` wrapper returned by specialized `Optimisers.setup(::DistributedOptimizer, model)`; specialized `Optimisers.update`/`update!` fill missing (`nothing`/AbstractZero) gradients with device-matched zeros (`similar(p)`), run one in-place allreduce per unique trainable parameter (model-structure order, tied params deduped by identity), then delegate to the native inner update. Communication stays outside AD. Added state-level adjust/adjust!/freeze!/thaw!, `DistributedUtils.synchronize!!`, updated docstrings; `resolve_unused_parameters!!` kept as legacy fallback (manual use forbidden). Public APIs only — no private Optimisers internals. **Gate REOPENED by review (2026-09-08):** 7 blocking findings + additional risks + test-proof gaps — see M1.1 below.
- **M1.1 (2026-09-08, commit `3370b910` on top of `04ddeaa0`, NOT pushed):** corrective spike after the M1 gate reopen. Fixed all 7 blocking findings: (1) trainable-semantics traversal via `Optimisers.trainable`, stopping at `Optimisers.isnumeric` leaves, exactly mirroring `Optimisers.setup` (BatchNorm μ/σ² excluded, proven by CountingBackend test); (2) `update!` propagates the model returned by the inner update; (3) structural gradient pairing via `Functors.functor(typeof(model_node), base(grad_node))[1]` — no KeyPath replay, Transpose/Adjoint get parent-space grads; (4) presence protocol (one Float32 `avg` allreduce of 0/1 flags per step, threshold `> 0`; globally-absent → `ZeroTangent`, optimizer step skipped, Adam state untouched); (5) nested composition rejected with ArgumentError (init/apply!); (6) caller grads never mutated (copied into fresh `similar(p)`); (7) device-preserving buffers (`similar(p)` + `copyto!`, `collect(g)` removed). Extras: higher-order grads rejected; Enzyme Duplicated setup unwraps `model.val` / Duplicated update rejected; Chain-tuple diagnostic; rule-level adjust moved to public `Optimisers.adjust`; `Functors.@functor DistributedOptimizerState (tree,)`; `resolve_unused_parameters!!` + `unused_parameters_distributedtest.jl` deleted (automatic path supersedes); `Test.detect_ambiguities(DistributedUtils; recursive=false)` → 0.
- M1.1 evidence: full ext_distributed MPI suite 6/6 PASS via direct runner `JULIA_MPI_TEST_NPROCS=2 julia --project=test test/ext_distributed/runtests.jl mpi` (1m35.6s) and `=4` (1m59.7s) — run directly, NOT via `make c8-mpi` (the earlier "4-rank" evidence was duplicated 4-rank children). Orchestrator independently re-ran both suites (6/6, EXIT=0). Logs: `artifacts/logs/m1.1/suite-nprocs2.log`, `suite-nprocs4.log`. New `conditional_distributedtest.jl` (570 lines): real-AD branches, analytic global-mean per param (proves averaging not broadcast, 4-rank weighting), Adam vs in-process global-mean reference incl. moment states, globally-absent skip semantics, BatchNorm collective count + μ/σ² unchanged, transpose/adjoint parent-space, tied/isbits params through AD, composition rejected, grad non-mutation, immutable model return, collective-count protocol. 6 test files now (unused_parameters deleted).
- M1.1 known limitations (accepted, documented in docstrings): DistributedOptimizer must be the outermost rule; only first-order grads; Enzyme Duplicated update unsupported; grads copied into fresh buffers (in-place reuse deferred to the perf pass, after correctness); `docs/src/guide/gpu.md:366-377` still shows pre-M1 Leaf output (docs update deferred). NCCL path API-compatible but runtime untested (CPU/MPI only per agreed scope).

## What was done last (2026-09-30) — conditional-graph wrapper Phase E (M1.1 battery restored, review-corrected)

- **Phase E implemented and review-corrected** (committed at Flux `55171e65`, tests only): `test/ext_distributed/conditional_distributedtest.jl` gained 15 testsets (+560 lines, no deletions). The plan requires restoring the M1.1 battery from `3370b910` without its defects.
- **Ported from M1.1** (real-AD and public-API cases with no Phase D equivalent): `real AD conditional branches average over multiple steps` (4-step rank-gated forward, analytic global mean integrated alongside updates, cross-rank equality), `Adam matches an independent global-mean reference` (5 steps; parameters, moments, and decayed `beta`), `tied parameters through real AD` (distinct per-occurrence Zygote gradients plus the M1.1 shared-gradient subcase; both update paths vs native reference), `tied parameter through a transpose with real AD`, `isbits parameters (FillArrays) sync independently`, `higher-order gradients rejected`, `Chain tuple-gradient diagnostic` (M1.1 API restore: warning and equal step for both update paths), `Enzyme Duplicated model setup and update` (M1.1 API restore: supported setup, `Flux.setup`, rejection of both updates), `collective count protocol with real AD`, `Flux.train! with DistributedOptimizerState`, `nested model (Chain of custom layer and Dense)`, `synchronize!! on DistributedOptimizerState`, `adjust / adjust! on wrapped state and rule`.
- **Not duplicated**: BatchNorm exclusion, transpose/adjoint parent space, caller-gradient non-mutation, `update!` immutable returns, globally absent Adam state — the D5 versions are stronger (real AD, rank-distinct, read-only, active-then-absent snapshots).
- **Phase E additions**: `freeze! and thaw! preserve wrapper state and collectives` (Adam with retained returned states; whole-wrapper `freeze!`/`thaw!` delegation; partial freeze through the inner tree; frozen value and moments held; 1 presence + 2 gradients per step unchanged; native reference freezes/thaws at the same steps) and `Functor device adaptation keeps the backend` (public `fmap` transforms exactly the two nonzero Adam state arrays, asserts types/values/backend identity/source immutability, and updates through the adapted state against a native Adam reference; `AdaptedBackend` delegates adapted buffers to the real backend on raw storage).
- **Review correction (2026-09-30)**: the first version used stateless `Descent` for both Phase E additions (zero array transformations, fresh state), omitted the M1.1 Chain/Enzyme checks and the shared-gradient tied subcase, and bypassed whole-wrapper freeze delegation. All corrected; `import Enzyme` added for `Enzyme.make_zero`; `AdaptedArray` gained `Base.convert` so an adapted Adam state stays adapted across updates. The correction runs passed on the first attempt.
- **GREEN evidence (correction gate)**: focused conditional at 2 MPI ranks exit 0, 1 m 22.0 s, **367/367 per rank** across 30 testsets (aggregate: `conditional file | 367 367 | 1m22.0s`); focused optimizer 6/6, 10.3 s; full dedicated 2-rank suite `Distributed | 6 6 2m02.6s`, exit 0, 2 m 4.8 s; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean.
- **RED iterations were test defects only** (5 attempts): freeze snapshot vs in-place mutation, freeze step reference model, `_avg_sum` length-4 constraint, `AdaptedArray` missing MPI `strides`, and inner-update container type. No production defect was found; no production source changed.
- Tied real-AD diagnostics: both ranks return `gs.a = [0.0, 2.0]`, `gs.b = [-2.0, 0.0]` as distinct objects; the test proves the wrapper's local tied sum matches native Optimisers. Rank-distinct tied averaging remains covered by D2.
- Optimisers 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf`). No mock or simulated value. Not committed; commit/push only with explicit authorization.
- Phase F (two- and four-rank gates; the four-rank case must prove weighting), Phase G (detector conversion), and Phase H (compatibility review) remain. No four-rank run, no GPU/NCCL, no detector, no Bocconi run, no main suite.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseE-*` and `phaseE-correction-*`. Details: `wiki/devlog/02-conditional-graph/2026-09-30-conditional-wrapper-phase-e-m11-battery.md` (see the "Corrections applied" section).

## What was done before (2026-09-30) — conditional-graph wrapper Phase D5 (regression gate)

- **D5 implemented** (uncommitted; Flux HEAD still `cbdd3edc`, D4 changes preserved in the same working tree): the retained contracts from plan section 12 D5 are now pinned by eight testsets in `test/ext_distributed/conditional_distributedtest.jl`: `BatchNorm running statistics stay excluded`, `transpose and adjoint gradients pair in parent space`, `caller gradients stay unchanged on both update paths`, `update! propagates immutable model returns and keeps wrapper identity`, `update returns a fresh wrapper without mutating inputs`, `globally absent parameters keep Adam state after active steps`, `empty trainable model performs no collectives`, and `array containers keep their shape during reconstruction`. `using FillArrays` added at the top for the isbits immutable-parameter case; `ImmutableReturnLayer` and `EmptyTrainableModel` fixtures added.
- **Defect found by the gate and corrected**: `_buildmap(f, ch::AbstractArray)` mapped over `eachindex(ch)`. For an `IndexLinear` container (`Matrix{Vector{Float64}}`), that flattened a 2×2 model child to a length-4 `Vector`; values paired by column-major order (so the downstream update could appear to succeed), but the rebuilt gradient did not mirror the model shape that Optimisers consumes (plan section 8.2). `_buildmap` and `_foreach_child` now map array containers over `CartesianIndices(ch)`: shape, axes, and column-major order are preserved, and both passes still consume the occurrence plan in lockstep (`_child` already accepts `CartesianIndex`). D5 production diff relative to D4: `src` +5/−2, test +332/−0.
- **RED evidence**: focused file at 2 MPI ranks before the array fix → exit 1, 1 m 4.3 s. Per rank: control 5/5, D1 10/10, guard 11/11, D2 38/38, D3 16/16, D4 20/20, D4-review 30/30, BatchNorm 6/6, transpose/adjoint 16/16, caller gradients 8/8, `update!` immutable 9/9, `update` fresh wrapper 15/15, globally absent 11/11, empty model 9/9; the new array-container testset **18 passed / 2 failed / 20**, exactly the intended defect: `size(synced.w) == (4,)` vs `(2, 2)` and `axes` mismatch. Values, protocol count, and end-to-end wrapper/reference comparison passed on the flattened gradient. No fixture error.
- **GREEN evidence**: exit 0, 1 m 18.0 s (one local Flux + FluxMPIExt recompile): all 15 testsets **224/224 per rank**; focused optimizer 6/6, 11.7 s; full dedicated 2-rank suite `Distributed | 6 6 1m38.2s`, 1 m 41.4 s; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean. Combined uncommitted D4 + D5 diff relative to `cbdd3edc`: 2 files, +581/−66 (`src` +67/−65, test +514/−1).
- Optimisers 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf`). No mock or simulated value in the D5 testsets. Not committed; commit/push only with explicit authorization.
- Phase D (D1-D5) is complete. Phase E (restore the M1.1 battery), Phase F (four ranks), Phase G (detector conversion), and Phase H (compatibility review) remain. No four-rank run, no GPU/NCCL, no detector, no Bocconi run, no main suite.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseD-d5-*`. Details: `wiki/devlog/02-conditional-graph/2026-09-30-conditional-wrapper-phase-d5-regression-gate.md`.

## What was done before (2026-09-29) — conditional-graph wrapper Phase D4 (P1 corrected)

- **D4 corrected after review** (uncommitted; Flux HEAD `cbdd3edc`): the first fix's `_trainable_child(ch::NamedTuple, tr::Tuple, k) = ch[k] in tr` predicate reevaluated value membership on every update, so once the first update made the equal-valued fixture fields `a` and `b` differ, `b` silently stopped being synchronized although its `Leaf` remained in the tree. The walk now reads selection from the state tree returned by `Optimisers.setup`, the retained setup-time record: `_sync_gradients(backend, tree, model, grads)` passes the tree to `_walk!`/`_build!`, which pair each model child with its tree node through the shared structural `_child` helper; a numeric child is trainable iff its tree node is an `Optimisers.Leaf`, and an empty tree node (`()`) skips the subtree. `_trainable_child` is deleted (no value-membership rule remains), no warning is emitted in the walk, and canonical model order is unchanged, so the D1-D3 order and occurrence-plan contracts still hold. The D3 test now builds `tree = Optimisers.setup(Descent(1.0), model)` for its direct `_sync_gradients` call.
- **Review characterization** (exit 0 asserts the defect): both update paths showed step 2 wrapper `b=0.45` vs native `-0.10` and 2 collectives instead of 3. `artifacts/logs/conditional-wrapper/phaseD-d4-review-multistep.jl`, `phaseD-d4-review-mpi2-2026-09-29.log`.
- **Regression added**: `legacy tuple trainable selection persists across updates` runs two steps through both `Optimisers.update!` and `Optimisers.update` with `Momentum`, distinct per-rank and per-step gradients, and native global-mean references; every step asserts parameters (`a`, `b`), untouched `excluded`, `!(model.a ≈ model.b)` (the trigger), `state.tree == ref_state` (rule, momentum state, frozen flags), cumulative collective counts (`3 * step`), and cross-rank count equality.
- **RED2 evidence**: focused file at 2 MPI ranks with corrected tests on the unchanged buggy source → exit 1, 37.8 s; prior testsets unchanged 5/5, 10/10, 11/11, 38/38, 16/16, 20/20; new testset **24 passed / 6 failed / 30 per rank** (3 intended failures per update path at step 2: `b [0.55,…]` vs `[-0.605,…]`, momentum `[0.45,…]` vs `[1.155,…]`, collectives 5 vs 6). No fixture error.
- **GREEN2 evidence**: exit 0, 48 s focused conditional (control 5/5 + D1 10/10 + guard 11/11 + D2 38/38 + D3 16/16 + D4 20/20 + D4-review 30/30 per rank); focused optimizer 6/6, 10 s; full dedicated 2-rank suite `Distributed | 6 6 1m12.0s`, 74 s; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean with only the two expected files.
- Optimisers 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf`). Final uncommitted D4 diff: 2 files, +244/−64 relative to `cbdd3edc` (`src` +62/−63 including the D3 direct-call update, test +182/−1). Not committed; commit/push only with explicit authorization.
- D5 (regression gate) remains. No four-rank run (Phase F), no GPU/NCCL, no detector, no Bocconi run, no main suite. No mock or simulated value is used in the D4 testsets.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseD-d4-*` (metadata Correction section). Details: `wiki/devlog/02-conditional-graph/2026-09-29-conditional-wrapper-phase-d4-legacy-trainable.md`.

## What was done before (2026-09-29) — conditional-graph wrapper Phase D3

- **Commit follow-up**: user authorized documentation and Flux commits. Commit `cbdd3edc` contains the reviewed D1-D3 source and tests (2 files, +480/−34). The Flux worktree is clean. The plan header and wiki quickstart now identify D4 as the next phase. Earlier uncommitted-state descriptions record the state at test time.
- **D3 review**: no blocking code findings. An independent two-rank conditional run passed control 5/5, D1 10/10, isbits guard 11/11, D2 38/38, and D3 16/16 per rank. Exit 0, 27.649 s. The ambiguity count remained zero, and Flux `git diff --check` passed. Stored RED and full-suite GREEN logs support the implementation report. The plan header and wiki quickstart still show the pre-D3 checkpoint. Exact review commands and limits are in the D3 devlog.
- **Phase D3 implemented** (uncommitted; Flux HEAD still `f5b87975`, D1+D2 changes preserved in the same working tree): CPU presence metadata. `_sync_gradients` now allocates the presence flags with `pres = zeros(Float32, np)` instead of `similar(bufs[anchor].buf, Float32, np)`; the `anchor = findfirst(b -> !isbits(b.p), bufs)` lookup and the `_SyncBuf.p` field (used only for that anchoring) are removed. The comment documents the split: presence metadata is host CPU metadata reduced once with `avg` (NCCL falls back to MPI for CPU buffers), while parameter gradient buffers keep their own device and backend. The one `avg` presence reduction and the `> 0` threshold are unchanged.
- **Defect corrected**: with a GPU parameter the old code allocated the presence vector on the parameter device; the scalar flag writes and the globally-present decision then touched device memory from host code (rejected by CUDA scalar-indexing checks, and a device sync for control flow).
- **Regression**: `test/ext_distributed/conditional_distributedtest.jl` gained `DeviceLikeArray` (custom `AbstractArray` whose `similar` preserves the wrapper type; `copyto!`/`fill!` operate on raw storage; scalar access increments `DEVICE_SCALAR_ACCESSES`), `DeviceMetadataBackend` (records every collective `(buftype, eltype, ndims, size, avg)`; supplies a simulated presence fraction `0.25f0` — one active rank out of four — and delegates gradient reductions to the real backend), and testset `CPU presence metadata for device-like parameters`, which calls `DistributedUtils._sync_gradients` directly. Assertions: zero host scalar accesses; 3 collectives (1 presence + 2 gradients); exactly one 1-D `Float32 avg` collective with `buftype === Vector{Float32}`; parameter buffers are `DeviceLikeArray{Float64,1}`; rebuilt gradients equal explicit `avg` 2-rank references; equal counts across ranks. The `0.25f0` fraction makes `> 0` observable (`> 0.5` would run no gradient collective).
- **RED evidence**: focused file at 2 MPI ranks before production changes → exit 1, 29 s, control 5/5, D1 10/10, guard 11/11, D2 38/38, D3 **14 passed / 2 failed / 16 per rank**. The two failures are exactly the intended defect: line 445 `DEVICE_SCALAR_ACCESSES[] == 0` evaluated `4 == 0`; line 455 `presence.buftype === Vector{Float32}` evaluated `DeviceLikeArray{Float32, 1} === Vector{Float32}`. No fixture errors.
- **GREEN evidence**: after the fix → exit 0, 44 s (one local Flux + FluxMPIExt recompile), control 5/5 + D1 10/10 + guard 11/11 + D2 38/38 + D3 16/16 per rank; focused optimizer 6/6, 12 s; full dedicated 2-rank suite `Distributed | 6 6 1m03.6s`, exit 0, 66 s; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean.
- Optimisers 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf`). Combined uncommitted D1+D2+D3 working-tree diff: 2 files, +480/−34 (`src` +74/−34, test +406/−0). Not committed; commit/push only with explicit authorization.
- The `0.25f0` presence result is simulated by the mock and labeled as such. No four-rank run (Phase F), no GPU/NCCL, no detector, no Bocconi run, no main suite. D4 (legacy tuple `trainable`) and D5 (regression gate) remain.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseD-d3-*`. Details: `wiki/devlog/02-conditional-graph/2026-09-29-conditional-wrapper-phase-d3-cpu-presence-metadata.md`.

## What was done before (2026-09-28) — conditional-graph wrapper Phase D2

- **D2 review**: no blocking code findings. Independent two-rank runs passed conditional control 5/5, D1 10/10, isbits guard 11/11, D2 35/35, and optimizer composition 6/6 per rank. Ambiguities remained zero and Flux `git diff --check` passed. One non-blocking coverage caveat: `IsoVec` becomes `Vector{Float64}` after the first update. Steps 2–3 prove state continuity and tied-gradient handling, but only step 1 exercises mixed tied/isbits mapping. Exact review commands and results are in the D2 devlog.
- **Phase D2 implemented** (uncommitted; Flux HEAD still `f5b87975`, D1 changes preserved in the same working tree): explicit occurrence mapping. `_walk!` now pushes a `_SyncOccurrence(buffer_index, emit)` entry for every numeric trainable occurrence, and `_build!` consumes the plan through a cursor instead of inferring buffer indices from an occurrence counter. First mutable occurrence and isbits occurrences `emit`; tied repeats emit `ZeroTangent`; `_sync_gradients` asserts complete plan consumption (`cursor[] == length(plan)`) and throws `ArgumentError` otherwise. `seen`/`counter`/`first_seen` are gone from the rebuild pass. Collectives, presence protocol, and buffer allocation unchanged.
- **Defect corrected**: a model ordered `p, p2 (= p), isbits c, mutable q` has three buffers but four counted occurrences; the old counter mapped `c` to `q`'s buffer, so `c` received `q`'s global-mean gradient and Adam update while its own buffer was never emitted. With a trailing isbits parameter the counter could also exceed the buffer list (`BoundsError`).
- **Regression**: `test/ext_distributed/conditional_distributedtest.jl` gained `MixedTiedIsbits` with three Adam steps (both tied occurrences present; first absent while the second contributes; second absent while the first contributes), a native Adam global-mean reference, model-value and Adam-state comparisons, recording-backend protocol checks (1 presence + 3 gradient reductions, tied sum pre-reduction), cross-rank count equality, plus an equal-valued isbits guard testset (`(c1 = v, c2 = v)`, `c1 === c2`, distinct gradients, independent buffers). The guard runs before the regression so RED cannot abort it. The mixed fixture is type-preserving (`c::IsoVec{4}` plus a reconstruction `convert`), with `isbits(model.c)` asserted before steps 2-3 and after step 3, so every step exercises the isbits path.
- **RED evidence**: focused file at 2 MPI ranks before production changes → exit 1, 30 s, control 5/5, D1 10/10, guard 11/11, D2 30 passed / 5 failed per rank; all 5 failures on `c` (values and Adam state diverge from the reference after steps 1-3; tied `p`/`p2` and `q` were correct). Strengthened type-preserving rerun (D2 production temporarily reverted to D1-only, verified by diff): exit 1, 40 s, D2 33 passed / 5 failed per rank, all on `c`; the `isbits` assertions pass, so steps 2-3 are isbits-path failures. One earlier attempt was discarded: `_avg_sum(local...)` used the reserved word `local` and raised a `ParseError` before tests ran (fixture error, not RED).
- **GREEN evidence**: after the fix → exit 0, 40 s (type-preserving fixture), control 5/5 + D1 10/10 + guard 11/11 + D2 38/38 per rank; focused optimizer 6/6, 10 s; full dedicated 2-rank suite `Distributed | 6 6 57.4s`, exit 0, 59 s; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean.
- Optimisers 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf`). Combined uncommitted D1+D2 working-tree diff: 2 files, +355/−27 (`src` +63/−27, test +292/−0). Not committed; commit/push only with explicit authorization.
- D3 (CPU presence metadata), D4 (legacy selection), D5 (regression gate) remain. Four-rank gates remain Phase F; no GPU/NCCL, detector, Bocconi, or main-suite runs.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseD-d2-*`. Details: `wiki/devlog/02-conditional-graph/2026-09-28-conditional-wrapper-phase-d2-occurrence-plan.md`.

## What was done before (2026-09-28) — conditional-graph wrapper Phase D1

- **D1 review**: no blocking code findings. Independent two-rank rerun passed control 5/5 and D1 10/10 per rank. Ambiguity count remained zero, and Flux `git diff --check` passed. The stored RED and full-suite GREEN logs support the implementation report. Review details and exact commands are in the D1 devlog.
- **Phase D1 implemented** (uncommitted; Flux HEAD still `f5b87975`): canonical traversal order for gradient synchronization. `_walk!` now iterates the model's functor children through a new `_foreach_child` (mirroring `_buildmap`) and filters them with `_trainable_child`; both passes use the same normalized selection in model order. Previously `_walk!` followed `pairs(Optimisers.trainable(x))` while `_build!` rebuilt in functor order, so a reversed named `trainable` `(b=..., a=...)` allocated isbits buffers in reverse order and the rebuild consumed them by position, swapping equal-shaped isbits gradients (silent, no tied parameters needed).
- **Regression**: `test/ext_distributed/conditional_distributedtest.jl` gained a D1 testset: `IsoVec` isbits `AbstractArray` leaf (SVector-like, no StaticArrays dependency), `ReversedTrainable` with reversed `trainable`, `RecordingBackend` logging pre-allreduce buffer values, native-Optimisers global-mean reference from explicit `allreduce!(..., avg)`, value assertions, collective order (presence `Float32` first, then `a`, then `b`) and cross-rank count equality.
- **RED evidence**: focused file at 2 MPI ranks before production changes → exit 1, 25 s, control 5/5 + D1 6 passed / 4 failed per rank. Rank 0 `model2.a = 0.45` vs reference `0.85`, `model2.b = 0.85` vs `0.45`; `rec.log[2]` was local `gb` (reverse model order). Intended defect, no fixture error.
- **GREEN evidence**: after the fix → exit 0, 37 s cold (one Flux + FluxMPIExt recompile), control 5/5 + D1 10/10 per rank; focused optimizer file 6/6, 11 s; full dedicated 2-rank suite `Distributed | 6 6 55.5s`, exit 0, 59 s; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean.
- Optimisers 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf`). Production diff `src/distributed/public_api.jl` +20/−1; test +92. Not committed; commit/push only with explicit authorization.
- Verified at baseline that a legacy tuple `trainable` already fails with `MethodError: haskey(::Tuple, ::Symbol)`; D1 preserves that failure mode. D3 (CPU presence metadata), D4 (legacy selection), and D5 (regression gate) remained open after D1. Four-rank gates remain Phase F.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseD-d1-*`. Details: `wiki/devlog/02-conditional-graph/2026-09-28-conditional-wrapper-phase-d1-canonical-traversal.md`.

## What was done before (2026-09-22) - Supervisor meeting note

- Reviewed the Italian report against its meeting-preparation goal, then rewrote it with user approval.
- `reports/conditional-graph-support-it.pdf` had 3 pages instead of 10: problem, proposed design, evidence and meeting decisions. (Status 2026-10-07: the compiled PDF is no longer present in the workspace; the `reports/conditional-graph-support-it.typ` source remains.)
- Removed the detailed diagrams, REPL quotation, and compatibility inventory. Kept the conditional-branch example and the absent-versus-zero distinction.
- Added explicit total-rank averaging, replica preconditions, composition limits, and architectural tradeoffs.
- Separated earlier prototype results at 2 and 4 ranks from current implementation results at 2 ranks. Compatibility corrections remain open.
- Typst compilation passed. PDF metadata and extracted text were checked. All three pages were inspected visually.
- No Flux code or tests changed. Implementation checkpoint and README progress remain unchanged.
- Next report action: discuss the state-wrapper boundary, initial scope, and acceptance criteria with the supervisor.
- Details: `wiki/devlog/02-conditional-graph/2026-09-22-supervisor-meeting-report.md`.

## What was done before (2026-09-21) — report italiano compatto

- Created `reports/conditional-graph-support-it.typ` (614 lines at creation; now 177 lines after the 2026-09-22 rewrite) and compiled `reports/conditional-graph-support-it.pdf` (10 pages, 170,568 bytes, A4, Typst 0.15.1, exit 0) as a compact Italian design report for the thesis supervisor. The English `reports/conditional-graph-support.typ` is unchanged (980 lines; 18-page PDF per `pdfinfo`).
- First draft: three sequential subagents — Italian translation; compaction of the Optimisers.jl `update!` contract section; section 6 built around a verbatim quotation of the Optimisers docs "Incomplete or nothing gradients" (REPL Adam example; the `generaly` typo stays).
- Revision (user request, simple-english skill applied): removed every reference to PR #2694, spike M1.1, branches, commits, local paths, and internal probe/log identifiers; section 7 rewritten from an implementation walkthrough to a design section (traversata, presenza, riduzione e ricostruzione); compatibility table reduced to two columns; "Evidenze" replaced by a short "Stato della verifica"; `hang` -> `blocco`. After a user question, section 3 now states explicitly that it describes the integration Flux ships today, not the proposal (heading: "L'integrazione distribuita che Flux fornisce oggi"); the section 4 caption is aligned.
- Kept in full: all 3 native-Typst diagrams and the error taxonomy.
- Typst fixes: `\#140`/`\#1`/`\#2`/`\#3` escapes (an unescaped `#` plus digits is evaluated and drops the hash — pre-existing defect), `#outline(title: [Indice], depth: 1)`, and `#set par(justify: false)` in the title block.
- Verification: `typst compile ...` exit 0; `pdfinfo` 10 pages; `pdftotext` finds `Incomplete or nothing gradients`, `generaly`, `0.997003`, `Indice`, `update completato`, `21 settembre 2026`; `rg` for PR/spike/worktree terms returns no match; all 10 pages inspected as PNG.
- No Flux source, tests, or commits touched. Checkpoint unchanged: Phase D1 planned on `ddp/integration-m1-spike`, Flux `f5b87975`.
- Details: `wiki/devlog/02-conditional-graph/2026-09-21-conditional-graph-report-italian.md`.

## What was done before (2026-09-17) - Phase D plan revision

- Phase C review found no blocking issues in the direct-nesting correction at `f5b87975`.
- Revised `temp/conditional_graph_wrapper_implementation_plan.md` without changing Flux source or tests. Phase D is planned, not implemented.
- D1 covers canonical traversal order with reversed named `trainable` fields. D2 covers explicit occurrence mapping for mixed tied/isbits parameters.
- D3 covers CPU presence metadata through a device-like fixture. D4 covers legacy tuple selection, including equal-valued distinct fields.
- Both traversal passes must use the same normalized model order. A sequential occurrence list alone does not correct mismatched traversal order.
- Optimisers 0.4.9 uses value membership for legacy tuple selection (`src/interface.jl:193-195`). Identity-only selection is not equivalent.
- D5 adds focused contract checks and a complete two-rank regression gate. Four-rank gates remain Phase F, with native GPU/NCCL work deferred.
- Corrected the plan's obsolete branch ancestry and entry instructions. Phase A is historical and must not repeat.
- Details: `wiki/devlog/02-conditional-graph/2026-09-17-conditional-wrapper-phase-d-plan-revision.md`.

## What was done before (2026-09-17) — Phase C review follow-up

- **Review finding**: the Phase C composition guard missed direct nesting of `DistributedOptimizer`. `Optimisers.setup(opt::DistributedOptimizer, model)` delegated straight to `Optimisers.setup(opt.opt, model)`, so `DistributedOptimizer(backend, DistributedOptimizer(backend, Adam()))` dispatched back to the specialized method, bypassed the per-leaf `init`/`apply!` guards, and built nested `DistributedOptimizerState` wrappers; updates would have synchronized twice. Plan section 7.5 requires `DistributedOptimizer` to be the outermost rule.
- **Fix**: both setup methods (plain model and `EnzymeCore.Duplicated`) now reject `opt.opt isa DistributedOptimizer` with `ArgumentError(_COMPOSE_ERROR)` before the inner setup runs. `_COMPOSE_ERROR` was moved above the setup methods and the guarding comment rewritten. No new methods were added (no ambiguity risk). `DistributedOptimizer(backend, OptimiserChain(...))` remains valid with an ordinary inner state tree; nesting inside an `OptimiserChain` is still rejected by the per-leaf `init` guard.
- **Three commits on `ddp/integration-m1-spike`**: `109aad9c` "Synchronize whole-model gradients before optimizer updates" (Phase C checkpoint; 2 files, +311/-13; development checkpoint, not final correctness approval); `a18392dd` "Add nested DistributedOptimizer composition regression (RED)" (tests only, 1 file, +33); `f5b87975` "Reject directly nested DistributedOptimizer at setup" (production fix only, 1 file, +15/-10). Flux HEAD `f5b87975`, worktree clean, not pushed.
- **Tests**: `@testset "DistributedOptimizer composition guard"` in `test/ext_distributed/optimizer_distributedtest.jl` asserts directly nested setup throws `ArgumentError` for the plain-model and `EnzymeCore.Duplicated` `Optimisers.setup` paths, nested-inside-`OptimiserChain` still throws, and a valid outermost `OptimiserChain` setups successfully with `Leaf`/tuple inner state. Model `(a=zeros(4), b=zeros(4))`.
- **RED evidence** (before fix, at `a18392dd`): focused file at 2 MPI ranks (`FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl`) → exit 1, 13 s, composition testset `4 passed / 2 failed` per rank; the 2 failures are exactly the direct-nesting assertions ("Expected: ArgumentError, No exception thrown"), not fixture/dependency errors. Log: `phaseC-review-guard-RED-mpi2-2026-09-17.log`.
- **GREEN evidence** (after fix, at `f5b87975`): focused optimizer file at 2 ranks → exit 0, 15 s cold with a one-time local `FluxMPIExt` precompile, 6/6 per rank; warm rerun exit 0, 10 s, no precompile; focused `conditional_distributedtest.jl` at 2 ranks → exit 0, 19 s, 5/5 per rank; full dedicated two-rank suite `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` → exit 0, 56 s, `Distributed | 6 6 54.4s`, all 6 children; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean; `using Flux` exit 0, 14 s.
- **Scope**: only the composition guard changed. Traversal ordering, mixed tied/isbits occurrence handling, CPU presence metadata, and legacy tuple-`trainable` handling remain deferred to Phase D and were not touched. No remote/GPU/NCCL/main-suite runs.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseC-review-metadata.md`, `phaseC-review-guard-RED-mpi2-2026-09-17.log`, `phaseC-review-guard-GREEN-mpi2-2026-09-17.log`, `phaseC-review-guard-GREEN-mpi2-warm-2026-09-17.log`, `phaseC-review-conditional-mpi2-2026-09-17.log`, `phaseC-review-suite-nprocs2-2026-09-17.log`, `phaseC-review-ambiguities-2026-09-17.log`, `phaseC-review-diff-check-2026-09-17.log`. Details: `wiki/devlog/02-conditional-graph/2026-09-17-conditional-wrapper-phase-c-review-followup.md`.

## What was done before (2026-09-17) — Phase C wrapper boundary

- **Phase C implemented** (committed as `109aad9c`): communication moved out of per-leaf `Optimisers.apply!(::DistributedOptimizer, ...)` into specialized whole-model `Optimisers.update`/`update!` methods on a new `DistributedOptimizerState{B,T}` wrapper (fields `backend`, `tree`; declared with `Functors.@functor DistributedOptimizerState (tree,)`). Added `Optimisers.setup(::DistributedOptimizer, model)` and an `EnzymeCore.Duplicated` variant returning the wrapper; composition errors (per-leaf `Optimisers.init`/`apply!` throw `ArgumentError`; `DistributedOptimizer` cannot be nested in an `OptimiserChain`); higher-order-gradient rejection on `update`/`update!`; delegation for `Optimisers.adjust`/`adjust!`/`freeze!`/`thaw!` on the wrapper state and rule; `DistributedUtils.synchronize!!(backend, ::DistributedOptimizerState)` delegating to the inner tree.
- **Minimal whole-model synchronization** ported from the preserved M1.1 reference `3370b910`: model-order walk via public `Optimisers.trainable`/`Optimisers.isnumeric`, structural gradient pairing, one presence-flag allreduce plus one allreduce per globally present buffer, rebuilt gradient tree with `ZeroTangent` for globally absent/tied-repeat occurrences. Public Optimisers APIs only; `resolve_unused_parameters!!` not restored.
- **GREEN evidence** (run from `/home/kurapica/Projects/ddp_flux/flux-integration`; `MPIEXEC=/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`): focused `conditional_distributedtest.jl` at 2 ranks → exit 0, 5/5 per rank, 28 s first run (one-time local `FluxMPIExt` precompile), 21 s warm; `optimizer_distributedtest.jl` at 2 ranks → exit 0, 9 s; full suite `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` → exit 0, `Distributed | 6 6 58.0s`, 61 s wall; `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` → 0; `git diff --check` clean.
- **Committed**: the wrapper boundary is commit `109aad9c` (2 files: `src/distributed/public_api.jl`, `test/ext_distributed/optimizer_distributedtest.jl`). Flux HEAD then advanced to `f5b87975` with the direct-nesting fix (see the follow-up section above). Thesis HEAD `7e098319`; wiki and artifact changes uncommitted.
- **Known Phase C limitations deferred to Phase D** (plan sections 9.2-9.4): rebuilt tree maps isbits occurrences by a global counter (mixed tied+isbits order defect); presence vector allocated from a parameter buffer (GPU host-control defect); old tuple-returning `Optimisers.trainable` behavior not yet proven.
- Not done (Phases F, G, H): four-rank runs, detector conversion, compatibility review.
- **Evidence**: `artifacts/logs/conditional-wrapper/phaseC-metadata.md`, `phaseC-GREEN-mpi2-2026-09-17.log`, `phaseC-GREEN-mpi2-warm-2026-09-17.log`, `phaseC-GREEN-optimizer-mpi2-2026-09-17.log`, `phaseC-suite-nprocs2-2026-09-17.log`, `phaseC-ambiguities-2026-09-17.log`, `phaseC-diff-check-2026-09-17.log`. Details: `wiki/devlog/02-conditional-graph/2026-09-17-conditional-wrapper-phase-c-wrapper-boundary.md`.

## What was done before (2026-09-16) — conditional-graph wrapper Phase B RED test

- Added `test/ext_distributed/conditional_distributedtest.jl` (55 lines, one testset), commit `6065ce8f` "Add conditional-graph distributed corruption regression (RED)". Follow-up commit `8add0e42` "Strengthen conditional-graph RED test preconditions" adds `@test tworkers >= 2` and `gs[1]` inactive-branch `nothing` assertions, so a one-rank run or changed AD behavior cannot produce a false green. No production code changed; not pushed.
- The test uses a real-AD `TwoBranch` model with two equal-shaped branches: rank 0 trains `a`, every other rank trains `b`, inactive branch gets `nothing`. It computes the analytic global mean per branch and asserts both updated values on every rank.
- RED evidence: `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl` → exit 1, 24 s, `3 passed, 2 failed` per rank (the 3 preconditions pass; the original run before the follow-up was `0 passed, 2 failed`). Fast numerical failure, no timeout. For `eta=0.1, ta=zeros(4), tb=fill(3.0,4)`: `a` actual 1.1 (rank 0) / 1.0 (rank 1) vs expected 0.9; `b` actual 1.1 / 1.0 vs expected 1.2. The current per-leaf `apply!` pairs the two single-rank allreduces by call order, so `avg(g_a, g_b)` mixes the branches and each rank updates only its own branch.
- Evidence: `artifacts/logs/conditional-wrapper/phaseB-RED-mpi2-2026-09-16.log`, `phaseB-RED-mpi2-strengthened-2026-09-16.log`, `phaseB-RED-metadata.md`. Details: `wiki/devlog/02-conditional-graph/2026-09-16-conditional-wrapper-phase-b-red-test.md`.
- Worktree clean; `ddp/windows-path-fix` and `ddp/pr2694-salvage` untouched; `archive/pr2694-unused-gradient-spike` still at `3370b910`.

## What was done before (2026-09-16) — conditional-graph wrapper Phase A baseline

- **Branch move (plan deviation)**: the plan claimed `fee2c310` descends from `3370b910`. It does not; both are siblings off `fa3bf228`, so `git merge --ff-only ddp/windows-path-fix` is impossible. The user approved `git reset --hard fee2c310` on `ddp/integration-m1-spike`. Only that branch moved; `ddp/windows-path-fix` and `ddp/pr2694-salvage` were not touched; `archive/pr2694-unused-gradient-spike` still holds `3370b910`.
- **Baseline**: HEAD `fee2c310`, worktree clean, `DistributedOptimizerState` absent from `src/` and `test/`. Routing test `julia --startup-file=no --project=test test/distributed_routing.jl` → 17/17, exit 0, 2 s. Two-rank suite `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` → 5/5, exit 0, 55 s (5 child files: common, data, optimizer, reduce, synchronized). `conditional_distributedtest.jl` is absent after the reset, as expected.
- **Evidence**: `artifacts/logs/conditional-wrapper/` (`baseline-routing-fee2c310-2026-09-16.log`, `baseline-mpi2-fee2c310-2026-09-16.log`, `baseline-metadata.md`). Details: `wiki/devlog/02-conditional-graph/2026-09-16-conditional-wrapper-phase-a-baseline.md`.
- No Flux source changed in this session.

## What was done before (2026-09-16) — conditional-graph wrapper implementation plan

- The user approved completion of the `DistributedOptimizerState` wrapper before a possible Optimisers.jl API proposal.
- The implementation will remain on `ddp/integration-m1-spike` in `/home/kurapica/Projects/ddp_flux/flux-integration`.
- The branch will first fast-forward from preserved M1.1 commit `3370b910` to PR head `fee2c310`. The feature will then return as narrow new commits.
- The full handoff is `temp/conditional_graph_wrapper_implementation_plan.md`. It contains context, invariants, branch procedure, phases, test matrix, commands, evidence requirements, stop conditions, and acceptance criteria.
- The plan identifies two unresolved M1.1 defects: mixed tied/isbits buffer indexing and GPU-host control over the presence vector. It also requires old tuple-returning `trainable` coverage.
- No Flux source changed in this session.

## What was done before (2026-09-16) — Windows routing-test fix

- Windows CI exposed a test-only bug in `test/distributed_routing.jl`: `occursin(runner_path, string(expr))` fails because rendering a Julia string literal escapes Windows backslashes.
- Commit `69703025` replaced rendered-text matching with inspection of the stored expression argument. Follow-up commit `fee2c310` removed the redundant injected Windows-path testset and uses `@test expr.args[2] == runner_path`, retaining the existing expression-shape assertions.
- `julia --startup-file=no test/distributed_routing.jl` passed 17/17 locally. `git diff --check` passed.
- `git push origin HEAD:ddp/upstream-pr` fast-forwarded the PR branch from `c64f695f` to `fee2c310`. GitHub confirms PR #2694 now points to `fee2c310a32958ca2f44c5ce5c1ddbef3db87c75`; live checks started, including Windows.
- Details: `temp/2026-09-15-pr2694-windows-distributed-routing-failure.md`, `wiki/devlog/01-pr2694/2026-09-16-pr2694-windows-routing-fix.md`.

## What was done before (2026-09-14) — conditional-graph onboarding (report + detector + notebook)

- PR #2694 is pushed and live CI is running; this session restarts the dropped conditional-graph (unused-parameter) work with onboarding material. No Flux source modified; all changes in the thesis repo.
- **Report**: `reports/conditional-graph-support.typ` → `reports/conditional-graph-support.pdf` (Typst 0.15.1, exit 0, A4, 18 pages, 300,488 bytes). Explains the Optimisers.jl update contract, the current per-leaf `DistributedOptimizer.apply!`, the failure scenarios, three native-Typst diagrams, why `nothing -> zeros` breaks Adam skip semantics, the M1.1 `DistributedOptimizerState` design, and the compatibility surface. Cites Optimisers.jl 0.4.7 (`ZOK0f`, pinned by this project): `subtract!` `interface.jl:104`, leaf `_grads!` `108`, non-leaf `_grads!` `119`; footnotes the +8 offset in 0.4.8/0.4.9. No external Typst packages.
- **Detector**: `scripts/checks/conditional_deadlock_probe.jl` (child, mpiexec) + `scripts/checks/conditional_deadlock_check.jl` (driver) + `make verify-conditional-deadlock`. Three modes: `deadlock` (rank 0 uses trunk+a+b, rank 1 uses trunk+c; rank 0 blocks on collective #3), `corruption` (rank 0 uses a, rank 1 uses b; equal counts, wrong pairing, silent), `control` (symmetric, correct). A `LoggingBackend` prints `TRACE rank=<r> allreduce #<n> size=<s>` before each collective.
- **Detector result**: `make verify-conditional-deadlock` → **12/12 PASS, exit 0, 1m20.0s** (2026-09-14). Log `artifacts/logs/conditional-deadlock/detector-2026-09-14.log`. Deadlock trace shows `TRACE rank=0 allreduce #3` with `PROBE_UPDATE_DONE rank=1` and no rank-0 finish; corruption shows `a_expected=1.1 a_actual=1.5 b_expected=1.4 b_actual=1.5 corrupted=true`; control `correct=true a=1.2 b=1.8 c=1.4`. This is a characterization test (passes while the bug exists); it must be updated when the fix lands.
- **Driver bug fixed**: `pipeline(cmd, stdout=log, stderr=log)` opened the same path twice (each handle at offset 0) so stderr overwrote the stdout markers; fixed by sharing one `IO` handle. Evidence: first run captured an empty trace, rerun captured it.
- **Notebook**: `notebook/conditional_graph_optimisers.jl`, valid Pluto `.jl` (25 cells = 13 Markdown + 12 code, matching `Cell order`). Optimisers alone → Flux context → distributed failure modes, with a `RUN_MPI=false` toggle. Validation: `STRUCTURE_OK cells=25 code_cells=12`, `PARSE_OK cells=25 markdown=13`, extracted code cells run `julia --project=.` `EXIT=0` no warnings, value assertions pass.
- Details: `wiki/devlog/02-conditional-graph/2026-09-14-conditional-graph-onboarding.md`, `wiki/devlog/02-conditional-graph/2026-09-14-conditional-graph-optimisers-notebook.md`. Consistent with ADR-0007 (unused-parameter support deferred).

## What was done last (2026-09-11) — final Flux commit and gate rerun

- **Review**: a six-agent review of PR #2694 salvage phases 2-7 produced findings. The orchestrator fixed all in-scope findings with RED/GREEN subagents. Commit `15ca2d9c` contains these review fixes. Details: `wiki/devlog/01-pr2694/2026-09-10-pr2694-review-fixes.md`.
- **Push blocker 1 (NCCL guard bypass)**: `initialize(NCCLBackend; force=true)` raised `MethodError`. Fixed `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` — added `force::Bool=false` to `__initialize(::Type{NCCLBackend}; ...)` and forwarded `force` to the MPI initializer; `src/distributed/public_api.jl` docstring now says the guardrail protects `MPI.Init()` including the `MPI.Init()` performed while bootstrapping `NCCLBackend`, the error prefix is backend-neutral `"Distributed initialization failed:"`, and it recommends `force=true` to the same `initialize` call; `NEWS.md` updated likewise.
- **Push blocker 2 (main-suite distributed opt-in)**: `ParallelTestRunner` discovered `ext_distributed/*` child files individually, so workers lacked `FLUX_TEST_DISTRIBUTED_BACKEND`. Fixed by making `test/ext_distributed/runtests.jl` the sole entry point: new `test/test_utils_distributed.jl` (`strip_distributed_tests!`, `add_distributed_runner!`); `test/runtests.jl` always strips `ext_distributed` from discovery, deletes the helper from discovery, and conditionally adds one `"ext_distributed"` entry when `FLUX_TEST_DIST_MPI || FLUX_TEST_DIST_NCCL`.
- **Push blocker 3 (padding docs)**: `DistributedDataContainer` docstring, `docs/src/guide/gpu.md`, and `NEWS.md` now state duplicated observations enter training batches, get extra weight in the averaged gradients, and change the effective training objective (not only epoch/metric accounting).
- **Additional in-scope corrections**: `"MPItrampoline"` → `"MPIwrapper"` (checker docstring + guard fixtures); `check_launcher_compat` renamed to internal `__check_launcher_compat` (users call `initialize`) across `public_api.jl`, `FluxMPIExt.jl`, `distributed_launcher_guard.jl`; `force` moved out of the "Undocumented internal kwargs" comment (it is public); `MPI.Initialized()` captured once and reused in `FluxMPIExt.jl`; dead `BoundsError → NaN32` sentinel removed from `data_distributedtest.jl` (records failure but still reaches the collective to preserve collective order) and the empty-data test now asserts `"empty"` / `"numobs(data) == 0"`; false "children MUST call exit(1)" comment removed from `ext_distributed/runtests.jl`; `.github/workflows/distributed_ci.yml` child watchdog `FLUX_TEST_DISTRIBUTED_TIMEOUT` `"1200"` → `"900"` in both jobs (step timeout stays 20 min); trailing whitespace removed in `public_api.jl`; final newline added to `FluxMPINCCLExt.jl`. NOT fixed: latent `CUDA.devices()` without `using CUDA` (phase-7 finding 4, NCCL-only) and stale out-of-scope `gpu.md` examples (finding 8).
- **RED evidence**: `artifacts/logs/pr2694-salvage/review-fixes/red-nccl-force.log` (new NCCL static-contract test 4 passed / 2 failed: `:force in declared`, `:force in forwarded`); `red-routing.log` (`test/distributed_routing.jl` failed at include — `test/test_utils_distributed.jl` did not exist).
- **GREEN + final verification** (logs under `artifacts/logs/pr2694-salvage/review-fixes/`, final under `.../final/`): guard tests 169/169 + NCCL static contract 6/6, exit 0 (`final/guard-final.log`); routing regression 18/18, exit 0 (`final/routing-final.log`); routing against REAL `ParallelTestRunner.find_tests("test")` — BEFORE = 6 `ext_distributed/*` child keys, AFTER_STRIP = none, AFTER_ADD = `["ext_distributed"]`, `ROUTING_OK` (`final/routing-real-discovery.log`); 2-rank dedicated runner 5/5 exit 0 (`final/mpi-nprocs2.log`); 4-rank dedicated runner 5/5 exit 0 (`final/mpi-nprocs4.log`); docs build `julia --project=docs docs/make.jl` exit 0, Doctests 1/1, rendered `docs/build/guide/gpu.html` contains the new wording (`final/docs-build.log`); static scope checks clean (`final/scope-checks.log`); repository-root `Manifest.toml` unchanged, sha256 `3b86244bc58065456acd34494db9d936953678e3824b107f8fdb4f68ecbb0046`.
- **Local main-suite failure (important)**: running the main suite `test/runtests.jl` locally is NOT possible on this host and twice crashed the desktop (opencode + herdr). `test/runtests.jl` calls `Pkg.add("MPI")` unconditionally when distributed flags are set; at the time `test/Project.toml` listed `Reactant` in `[deps]` but the resolved `Manifest.toml` lacked it, so Pkg instantiation pulled in Reactant/XLA, and `test/test_module.jl` loads any package where `find_package(...)` resolves — every worker loaded Reactant; on a 14 GiB host this exhausted memory. `FLUX_TEST_REACTANT=false` does NOT help (`Pkg.add("MPI")` still instantiates). Side effect: the aborted first run appended a `Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"` line to `test/Project.toml`, reverted with `git restore test/Project.toml`; no tracked env changes remain. The main-suite route is verified locally via the real-discovery routing check + routing unit test + direct runner; the full route remains a CI gate. Full write-up: `wiki/failures/2026-09-10-main-suite-reactant-instantiation.md`.
- **Final commit**: `c64f695f857a4248846163b0455b1a002978f869` only corrects stale comments in `test/distributed_routing.jl`. Its parent is review-fix commit `15ca2d9c`.
- **Gate rerun**: the rootless Podman/OpenMPI hwloc topology-emulation gate used exact `c64f695f`. All original 9 steps exited 0. The 2-rank and 4-rank suites each passed 5/5. Then exact workflow `timeout=900` reruns passed 5/5 at both rank counts. The final step restored plain MPI preferences.
- **Evidence**: `temp/docker_ci_fix/podman/podman-c64f695f-*.log` and `temp/docker_ci_fix/podman/podman-c64f695f-timeout900-*.log`.
- **Scope**: this gate emulates the OpenMPI topology with hwloc. It does not provide real cgroup-cpuset parity.
- **Git state**: `ddp/pr2694-salvage` is clean at exact `c64f695f857a4248846163b0455b1a002978f869`. PR #2694 points to this commit.
- **Push**: `git push origin ddp/pr2694-salvage:ddp/upstream-pr` fast-forwarded the PR branch from `08f468e3` to `c64f695f`. GitHub reports the PR as open and mergeable. Live checks started.

## What was done before (2026-09-10) — rootless podman topology-emulation validation

- Ran the rootful-docker → rootless-podman migration validation (evidence `temp/docker_ci_fix/podman/`, details `wiki/devlog/01-pr2694/2026-09-10-rootless-podman-ci-parity.md`). Key result: the planned `taskset -c 0-3 podman run ...` does NOT restrict the container (rootless `conmon` is forked by the systemd user manager; container PID 1 affinity `0-11`, `nproc` = 12), and inside-container `taskset` restricts affinity (`nproc` = 4) but OpenMPI 4.1.6 counts hwloc slots, not affinity, so plain `mpiexec -n 4` still succeeds. The literal negative control passed 5/5 (44s) — taskset-only parity is invalid.
- Real cgroup cpuset is impossible rootless: `podman run --cpuset-cpus=0-3` errors (`cpuset.cpus: no such file or directory`, exit 127; cpuset controller not delegated — `user@1000.service/cgroup.controllers = cpu memory pids`), direct `cgroup.subtree_control` write is ENOENT, and `systemd-run --user --scope -p AllowedCPUs=0-3` is silently ignored.
- Working emulation: OpenMPI MCA `hwloc_base_topo_file` + hwloc-generated topology XML (2 cores / 4 PUs from CPUs 0-3; `/tmp/opencode/podman_migrate/topo-gen.c` → `topo-2c4t.xml`). Probe: plain `mpiexec -n 4` → "not enough slots" exit 1; plain `-n 2` → exit 0; `--use-hwthread-cpus -n 4` → exit 0.
- Full validation with the emulation (all real exit codes): R1a negative 4-rank 0 passed / 5 failed, banner ×5, EXIT 1 (8s); R2b positive 4-rank 5/5, EXIT 0 (59s); R3 2-rank control 5/5, EXIT 0 (51s); R4 CI-order 4-rank job 5/5, EXIT 0 (63s); R5 prefs restored byte-identical to backup (md5 `183acda9c034313d0b27f63abe08f2a5`). Staging repo left as found (only `test/LocalPreferences.toml` written during runs, restored).
- Failure recorded: `wiki/failures/2026-09-10-podman-taskset-cpuset.md`.
- **Gate 4 follow-up (2026-09-10)**: the real runner `scripts/podman/run-ci-parity.sh` was then used for the full phase-7 gate 4 sequence against staged salvage `caf47145`; all 9 steps exit 0, both suites 5/5 (2 ranks 35.8s, 4 ranks 1m01.4s), OpenMPI 4.1.6 probe `4-rank launch OK`. Gate 4 is now **PASS**. Details in the phase-7 section below.

## What was done before (2026-09-10) — PR #2694 salvage phase 7: verification gates

- **Phase 7 done**: Phase 7 of `temp/pr_salvage_plan.md` (lines 268-290) — the pre-push verification gates — is complete. Method: the orchestrator launched a swarm of subagents (gate checks + independent audits). All subagents completed; no subagent failed. Details: `wiki/devlog/01-pr2694/2026-09-10-pr2694-salvage-phase7-verification.md`.
- **Gate results (all six PASS)**: Gate 1 static scope checks PASS after fix; Gate 2 PMI guard tests PASS (169/169); Gate 3 MPI suite PASS (`JULIA_MPI_TEST_NPROCS=2` 5/5, 49s; `=4` 5/5, 54s); Gate 4 OpenMPI topology emulation **PASS via rootless podman** (see below); Gate 5 docs build PASS (`julia --project=docs docs/make.jl`, exit 0); Gate 6 final review PASS.
- **Gate 4 detail (2026-09-10, rootless podman)**: ran the real runner `scripts/podman/run-ci-parity.sh` only (no docker/sudo/`newgrp`/`--cpuset-cpus`), per `scripts/podman/README.md`. Staged salvage content `caf47145ba1ea2ca06370a2b3422df0b7ddf4a0f` to `/tmp/opencode/docker_ci/repo-salvage/` (Flux worktree untouched); all 9 steps exit 0. `mpi_fast` (2 ranks) exit 0, 5/5, 35.8s. `mpi_edge` (4 ranks) exit 0, 5/5, 1m01.4s. Probe: `mpiexec (OpenRTE) 4.1.6`, `nproc`=4, affinity `0-3`, `MPI.mpiexec()` = `mpiexec --use-hwthread-cpus`, `MPI.MPI_LIBRARY` = `OpenMPI`, `4-rank launch OK`; no `not enough slots` in any log. Non-blocking anomaly: the 4-rank log warns `155 dependencies precompiled but different versions are currently loaded (...)`; suite still exit 0 and 5/5. No podman runner problems; nothing needed fixing. Logs/summary: `temp/docker_ci_fix/podman/podman-phase7-00..08-*.log`, `temp/docker_ci_fix/podman/podman-phase7-SUMMARY.md`.
- **Gate 1 fix commit `caf47145`** "Document distributed changes and fix whitespace hygiene (phase 6)": 8 files, `+23/-19` — the 4 phase-6 docs/consistency files plus `runtests.jl` trailing whitespace and 3 test files missing final newlines. `git diff --check 404ff37d...HEAD` clean (exit 0); worktree clean; **NOT pushed**.
- **mpiexecjl conflict resolved**: an auditor claimed moving `--project` off the `mpiexecjl` wrapper breaks it; verified false — `MPI.jl`'s `bin/mpiexecjl` loops over ALL args and extracts `--project`/`--project=*` regardless of position, so the phase-6 `mpiexecjl -n <np> julia --project=. <file>.jl` order is correct.
- **In-scope audit findings (NOT fixed — future decision)**: (1) guard documented as MPI-only but also runs for `NCCLBackend`; `initialize(NCCLBackend; force=true)` is a `MethodError` (no `force` kwarg in `FluxMPINCCLExt`); (2) `"MPItrampoline"` is not a valid `MPI.MPI_LIBRARY` value in the `check_launcher_compat` docstring/example; (3) duplicated-observation wording should note training-gradient bias, not only metrics; (4) latent `CUDA.devices()` without `using CUDA` in `runtests.jl:28`; (5) dead `BoundsError → NaN32` in `data_distributedtest.jl:47-51`; (6) `force` labelled "Undocumented internal kwargs" though it is public; (7) redundant double `MPI.Initialized()` call; (8) stale out-of-scope `gpu.md` examples.
- **Explicitly NOT verified**: native NCCL runtime (out of scope). Gate 4 uses hwloc topology-file emulation under rootless podman, not a real cgroup cpuset; do not claim native NCCL or real-cpuset parity.
- Logs: `artifacts/logs/pr2694-salvage/phase7/` (`guard-focused.log`, `guard-repro.log`, `mpi-nprocs2.log`, `mpi-nprocs4.log`, `mpi-config.log`, `docs-build.log`, `docs-instantiate.log`, `instantiate.log`); gate 4 podman logs `temp/docker_ci_fix/podman/podman-phase7-00..08-*.log` + `podman-phase7-SUMMARY.md`.

## What was done before (2026-09-10) — PR #2694 salvage phase 6: documentation & consistency sweep

- **Phase 6 done**: the 4 docs/consistency edits (`+15/-11`) were committed in phase 7 as part of `caf47145` "Document distributed changes and fix whitespace hygiene (phase 6)" (8 files total, `+23/-19`), which also fixed whitespace hygiene. Worktree clean; NOT pushed.
  1. `docs/src/guide/gpu.md` (3 line edits) — both launcher occurrences corrected from `mpiexecjl --project=. -n <np> julia <filename>.jl` to `mpiexecjl -n <np> julia --project=. <filename>.jl` (lines ~317, ~397); line ~349 "split the data uniformly" replaced with equal-length/cyclic-duplication wording (every process gets an equal-length shard; when `N` is not divisible by workers the container repeats observations cyclically so each process receives `cld(N, workers)` observations while every index stays in `1:N`; dataset not modified; duplicates affect epoch accounting and metrics; empty dataset rejected). No unused-gradient instructions, no native-NCCL-testing claims; "Known shortcomings" CUDA-aware warning untouched.
  2. `NEWS.md` — `### Distributed training fixes` now has EXACTLY two bullets, each ending with `([#2694](https://github.com/FluxML/Flux.jl/pull/2694))`: PMI guardrail (known MPICH/PMIx launcher mismatch diagnostic + `force=true`) and Data sharding (equal-length cyclic sharding + empty-data `ArgumentError`). Removed the old `**Distributed test fixes**` test-renaming/CI bullet and the vague `**Data padding**` bullet. No other NEWS section changed.
  3. `src/distributed/public_api.jl:342` — docstring typo `AbstractFluxDistributedBacked` → `AbstractFluxDistributedBackend`.
  4. `ext/FluxMPIExt/FluxMPIExt.jl` — restored the missing final newline (no semantic change).
- Method: delegated — three implementation subagents (gpu.md; NEWS.md; typo + consistency sweep) applied the edits (no commit), then an INDEPENDENT verification subagent returned APPROVE. No subagent failed. No RED/GREEN cycle (documentation/consistency sweep, no semantic change — orchestrator decision).
- Checks passed (independent verifier, reproduced): `git diff --check` clean (exit 0); old launcher order absent; "split uniformly" absent; both `#2694` links present; `docs/src/guide/distributed.md` absent and `docs/make.jl` has no registration; no stale terms (`resolve_unused_parameters`, `unused_parameters`, `DistributedOptimizerState`, `conditional-gradient`, `AbstractFluxDistributedBacked`, `split uniformly`); no child distributed test reads `ARGS`; only the 4 expected files modified, no untracked files. **Correction**: the phase-6 verifier checked the 4 edited files but the claim that the whole changeset was newline-clean was inaccurate — 3 committed test files lacked final newlines and `runtests.jl` had 5 trailing-whitespace lines; fixed in `caf47145` during gate 1. Full MPI suite and docs build NOT run in phase 6 — both are Phase 7 gates.
- Scope boundaries honored: narrow edits only (4 files); no unused-gradient content, no `DistributedOptimizerState` content, no conditional-gradient API, no re-added distributed guide, no NCCL-validation claim.
- Note: `docs/src/guide/gpu.md:366-377` (pre-M1 Leaf output) is unrelated to this PR scope and remains deferred.
- Details: `wiki/devlog/01-pr2694/2026-09-10-pr2694-salvage-phase6-docs-consistency.md`; plan: `temp/pr_salvage_plan.md` (lines 251-267).

## What was done last (2026-09-09) — PR #2694 salvage phase 5: retained & polished Carlo's test changes

- **Phase 5 done**: commit `23cae338` "Polish distributed CI labels and harness wording (phase 5)" on `ddp/pr2694-salvage`, one commit on top of `4eb5708e` (phase 4 GREEN). Worktree clean. Exactly 2 files changed; no behavioral change (comment/CI-label polish only):
  1. `.github/workflows/distributed_ci.yml` — job `mpi_edge` renamed "MPI DDP Sharding & Asymmetry (4 Ranks)" → "MPI DDP Correctness (4 Ranks)" (both jobs now read "MPI DDP Correctness (2 Ranks)" / "(4 Ranks)"); removed the `FLUX_TEST_DISTRIBUTED_MPI: "true"` env line from BOTH `mpi_fast` and `mpi_edge` (that var is only read by the main suite `test/runtests.jl`, NOT by the standalone runner `test/ext_distributed/runtests.jl`, which reads `JULIA_MPI_TEST_NPROCS`, `FLUX_TEST_DISTRIBUTED_TIMEOUT`, `FLUX_TEST_DISTRIBUTED_NCCL`); run step renamed "Run Distributed Tests (Edge cases)" → "Run Distributed Tests"; verified no commented-out NCCL CI job remains (already removed in phase 2 `5c6ea561`); `--use-hwthread-cpus` still only in the four-rank job (`mpi_edge`), 8-line ubuntu-latest slots comment kept verbatim.
  2. `test/ext_distributed/runtests.jl` — comment above `proc = run(...)` rephrased to "Explicitly inherit the child streams so the child's output and errors are visible in the CI logs."; the `proc = run(...)` code line is byte-identical; NCCL route (`FLUX_TEST_DISTRIBUTED_NCCL` backends + `nccl` child branch) untouched; no claim added that CPU-buffer fallback tests validate native NCCL.
- Method: delegated — implementation subagent applied the edits (no commit), then an INDEPENDENT verification subagent audited against a 10-point checklist distilled from Phase 5 of `temp/pr_salvage_plan.md` (lines 228-251) and returned APPROVE. No RED/GREEN cycle (wording/CI-label polish, no semantic change — orchestrator decision).
- Checks passed: `git diff --check` clean; workflow YAML parses (jobs `mpi_fast`, `mpi_edge`); `Meta.parseall` on runtests.jl ok; grep confirms no `FLUX_TEST_DISTRIBUTED_MPI`, no "Sharding & Asymmetry", no "Edge cases", no commented NCCL block, no `supervisor`/`native`/`CPU-buffer`/`fallback` claims in the changed files or `test/ext_distributed/`. Full MPI suite NOT re-run (comment-only change); full two-rank/four-rank MPI runs remain a Phase 7 verification gate.
- Retained-from-`fa3bf228` items re-confirmed present and untouched: `test/ext_distributed/distributed_setup.jl`; child test files do not read `ARGS`; backend selected from `FLUX_TEST_DISTRIBUTED_BACKEND` env in `distributed_setup.jl`; deterministic `sort!(distributedtestfiles)`; `MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"` in `test/Project.toml`.
- Pre-existing out-of-scope observation (note, NOT a phase-5 defect): `test/ext_distributed/runtests.jl` ~line 28 references `CUDA.devices()` with no `using CUDA` visible at file top; would only error if `FLUX_TEST_DISTRIBUTED_NCCL=true` were set, which the current CI never does. Predates phase 5 and untouched.
- Details: `wiki/devlog/01-pr2694/2026-09-09-pr2694-salvage-phase5-polish-carlo-tests.md`; plan: `temp/pr_salvage_plan.md` (lines 228-251).

## What was done last (2026-09-09) — PR #2694 salvage phase 3: PMI guardrail narrowed (RED tests + GREEN implementation)

- **3a RED**: committed `af728837` — `test/distributed_launcher_guard.jl`, decision-matrix tests for the pure checker `DistributedUtils.check_launcher_compat(env; library, force, mpi_initialized)`. 52 errored / 0 passed (checker did not exist). Log: `artifacts/logs/pr2694-salvage/phase3-launcher-guard-RED.log`. Rows pin: MPICH+PMIX_RANK/OMPI vars → actionable error (names library + trigger var + force=true); force / already-initialized → ok; OpenMPI+PMIx vars → ok; PMI2/SLURM-only envs → ok; non-MPICH non-OpenMPI libraries never judged.
- **3b GREEN**: committed `cd8de233` (+53/-41). (1) New pure MPI-free `DistributedUtils.check_launcher_compat` in `src/distributed/public_api.jl` — returns `nothing` unless `library=="MPICH"` && PMIX_RANK/OMPI_COMM_WORLD_RANK present && !force && !mpi_initialized; message names library, trigger vars, `force=true`, launch alternatives. (2) `ext/FluxMPIExt/FluxMPIExt.jl` inline guard (40 lines: broad non-OpenMPI-PMI2 errors + SLURM-no-PMI2 `@warn`) replaced by 4-line delegation, still gated pre-`MPI.Init()`. (3) `initialize` docstring + NEWS narrowed to the limited behavior; `force=true` stays the documented expert bypass.
- Phase 3 evidence: guard tests **169/169 pass** (GREEN log `artifacts/logs/pr2694-salvage/phase3-launcher-guard-GREEN.log`); integration smokes on local MPICH_jll — `PMIX_RANK=0` set → `initialize(MPIBackend)` throws before `MPI.Init()` (init stays false); clean env → initializes fine. No stale "expects PMI2"/"is not OpenMPI"/SLURM-warning references left in src/ext/docs/NEWS.
- Details: `wiki/devlog/01-pr2694/2026-09-09-pr2694-salvage-phase3a-launcher-guard-tests.md`, `wiki/devlog/01-pr2694/2026-09-09-pr2694-salvage-phase3b-pmi-guardrail.md`.

## What was done last (2026-09-09) — PR #2694 salvage phase 4: cyclic data padding (RED + GREEN)

- **4a RED**: committed `e64165d3` — rewrote `test/ext_distributed/data_distributedtest.jl` as the five-size cyclic-padding matrix (N=0/1/2/7/10) driven by `_cyclic_spec(N, tworkers)`: `per = cld(N, W)`, `total = per*W`, `full = [mod1(i, N) for i in 1:total]`, rank `r` gets the contiguous slice `full[(r*per+1):((r+1)*per)]`. Each non-empty N asserts identity-container obs == cyclic chunk, all indices valid in `1:N`, originals-before-padding, and an allreduce aggregate equal to `sum(data[full])`. N=0 asserts `ArgumentError`. Against the old padding (`append!(1:pad_count)`), N=1 @ 4 ranks fails with BoundsError (invalid indices 2,3) + NaN allreduce aggregate → RED evidence. Logs: `artifacts/logs/pr2694-salvage/phase4-data-RED-n2.log` (EXIT 0), `phase4-data-RED-n4.log` (EXIT 1).
- **4b GREEN**: committed `4eb5708e` — `__construct_distributed_data_container` now throws a clear `ArgumentError` when `numobs(data) == 0` (before partitioning), uses `cld(N, split_across)` (no float `ceil`), and builds `indices = [mod1(i, N) for i in 1:total_padded]` so padding repeats cyclically from the start and every index stays in `1:N`. Docstring for `DistributedDataContainer` extended: dataset not modified (only the index sequence is padded), shard length `cld(N, workers)`, padding duplicates observations (metrics over sharded data count duplicates and must account for them), empty dataset rejected. Matrix now **45/45 pass per rank** at both `-n 2` and `-n 4`, EXIT 0 both. Logs: `artifacts/logs/pr2694-salvage/phase4-data-GREEN-n2.log`, `phase4-data-GREEN-n4.log`.
- Details: `wiki/devlog/01-pr2694/2026-09-09-pr2694-salvage-phase4-cyclic-data-padding.md`; plan: `temp/pr_salvage_plan.md`.

## What was done last (2026-09-09) — PR #2694 salvage phases 1-2

- **Phase 1**: archived `3370b910` as `archive/pr2694-unused-gradient-spike` (local); retired the clean `/tmp/opencode/pr2694-ci-fix` worktree (branch `ddp/pr2694-ci-fix` kept @ `fa3bf228`); created permanent worktree `../flux-pr2694-salvage` on `ddp/pr2694-salvage` @ `fa3bf228`. Dirty `master`, `flux-integration` @ `3370b910`, and `archive/dirty-master-c8.2-wip` untouched and verified identical.
- **Phase 2**: single scope-reset commit `5c6ea561` (pure deletions, 6 files / 267 lines): removed `resolve_unused_parameters!!` from `public_api.jl`, deleted `unused_parameters_distributedtest.jl`, removed unused-parameter NEWS text, deleted `docs/src/guide/distributed.md` + `docs/make.jl` entry, removed commented NCCL CI placeholder and NCCL-validation claims from `distributed_ci.yml`. Nothing ported from `04ddeaa0`/`3370b910`; `gpu.md` untouched (fa3bf228 cleanup retained).
- Details: `wiki/devlog/01-pr2694/2026-09-09-pr2694-salvage-phases-1-2.md`; plan: `temp/pr_salvage_plan.md`.

## What was done last (2026-09-08) — M0 + M1 + M1.1 PR #2694 integration spike

- M0: preserved dirty master (verified identical), snapshotted `archive/dirty-master-c8.2-wip` @ `b0034f55`, created the `../flux-integration` worktree @ `fa3bf228`, recorded metadata/port inventory/checklist in ADR-0006 (scope approved via checkpoint question).
- M1: implemented the `DistributedOptimizerState` wrapper spike (`04ddeaa0`): fixed `optimizer_distributedtest.jl` wrapper access; added `conditional_distributedtest.jl` (rank-gated branches, analytic global-mean checks, tied-param and frozen-param checks, CountingBackend collective counts); full ext_distributed MPI suite green.
- M1.1: review (2026-09-08) REOPENED the M1 gate with 7 blocking findings + risks + test-proof gaps; fixed all of them in `3370b910` (trainable-semantics traversal, update! model return, structural grad pairing, presence protocol, composition rejected, grad non-mutation, device-preserving buffers, plus extras); rebuilt `conditional_distributedtest.jl` (570 lines) with real AD; suite 6/6 at nprocs=2 and 4 via the direct runner.
- Details: `wiki/devlog/01-pr2694/2026-09-08-m0-m1-integration-spike.md`, `wiki/devlog/01-pr2694/2026-09-08-m1.1-corrective-spike.md`, `wiki/decisions/ADR-0006-integration-baseline.md`.

## What was done before (2026-09-03) — precompile experiment

- Phase A audit + Phase B baseline + Phase C mitigation experiments + Phase D
  implementation + Phase E end-to-end validation on bocconi (details: `wiki/devlog/01-pr2694/2026-09-03-precompile-experiment.md`, ADR-0005, branch `experiment/precompile`).
- Local uncommitted changes on `experiment/precompile`: Makefile (env/precompile/precompile-flux-test/precompile-all/install-mpiexec), scripts/remote/{precompile.sh,setup_node.sh,hosts/bocconi.conf}, README.md, wiki/{context/remote-nodes.md, decisions/ADR-0005-*, devlog/01-pr2694/2026-09-03-*}.

## What was done before (2026-08-17) — PR #2694 frozen-branch restore + rebase

- Pushed the frozen branch `upsteam-pr` to the PR head `ddp/upstream-pr`,
  restoring the CI-enabling `test/runtests.jl` default, the `distributed.md`
  guide, and its `docs/make.jl` registration.
- Rebased the PR onto current upstream master (`404ff37d`); resolved the only
  conflict (`NEWS.md`) by merging our distributed section with upstream's
  Unreleased bullets.
- PR now mergeable (`mergeable: true`); CI running.
- Local backups: `backup/pr2694-old-head` (`df95f602`),
  `backup/upsteam-pr-pre-rebase` (`504e816e`).
- Details: `wiki/devlog/01-pr2694/2026-08-17-pr2694-restore.md`.

## What was done before (2026-08-01) — C8.3 NCCL verification

- **Fixed `check_cross_rank_sync` in `helper.jl`:**
  - Was calling `MPI.Bcast!` with `backend.comm` (NCCL.Communicator), not MPI.Comm.
  - Switched to `DistributedUtils.bcast!(backend, ref; root=0)` which dispatches correctly for both MPI and NCCL backends.
- **NCCL+GPU verification on Bocconi (gnode01, 2 ranks, 2x A100):**
  - `end_to_end_distributedtest.jl nccl`: **78/78 PASS**
  - `data_distributedtest.jl nccl`: **116/116 PASS**
  - `reduce_distributedtest.jl nccl`: **1/1 PASS**
- **Pre-existing NCCL failures (not C8.3 related):**
  - `common_distributedtest.jl`: 11/17 (6 errors — pre-existing NCCL comm pattern mismatches)
  - `synchronized_distributedtest.jl`: 4/4? (needs investigation)
  - `optimizer_distributedtest.jl`: fails at line 101 (pre-existing)
  - `deadlock_distributedtest.jl`: ProcessExited(1) (known issue)

## Commands that pass

- `julia --startup-file=no test/distributed_routing.jl` (run in `../flux-pr2694-salvage` at `fee2c310`; 17/17 PASS, exit 0) — Windows routing-test fix
- `julia --project=test test/distributed_launcher_guard.jl` (run in `../flux-pr2694-salvage`; 169/169 + NCCL static contract 6/6, exit 0) — review-fix session (`final/guard-final.log`)
- `scripts/podman/run-ci-parity.sh <step>` (rootless Podman, exact `c64f695f`; original 9 steps exit 0; 2 ranks 5/5; 4 ranks 5/5) — final topology-emulation gate (`podman-c64f695f-*.log`)
- Exact workflow `timeout=900` suite reruns (rootless Podman, exact `c64f695f`; 2 ranks 5/5; 4 ranks 5/5; plain MPI preferences restored) — `podman-c64f695f-timeout900-*.log`
- `julia --project=test test/distributed_routing.jl` (run in `../flux-pr2694-salvage`, 18/18, exit 0) — review-fix session (`final/routing-final.log`)
- `ParallelTestRunner.find_tests("test")` real-discovery routing check (run in `../flux-pr2694-salvage`; BEFORE 6 child keys, AFTER_STRIP none, AFTER_ADD `["ext_distributed"]`, `ROUTING_OK`) — review-fix session (`final/routing-real-discovery.log`)
- `JULIA_MPI_TEST_NPROCS=2 julia --project=test test/ext_distributed/runtests.jl` (run in `../flux-pr2694-salvage`, 5/5 PASS, 33.8s) — review-fix session (`final/mpi-nprocs2.log`)
- `JULIA_MPI_TEST_NPROCS=4 julia --project=test test/ext_distributed/runtests.jl` (run in `../flux-pr2694-salvage`, 5/5 PASS, 40.9s) — review-fix session (`final/mpi-nprocs4.log`)
- `julia --project=test test/distributed_launcher_guard.jl` (run in `../flux-pr2694-salvage`, 169/169 PASS) — phase 7 gate 2
- `JULIA_MPI_TEST_NPROCS=2 julia --project=test test/ext_distributed/runtests.jl` (run in `../flux-pr2694-salvage`, 5/5 PASS, 49s) — phase 7 gate 3
- `JULIA_MPI_TEST_NPROCS=4 julia --project=test test/ext_distributed/runtests.jl` (run in `../flux-pr2694-salvage`, 5/5 PASS, 54s) — phase 7 gate 3
- `julia --project=docs docs/make.jl` (run in `../flux-pr2694-salvage`, exit 0) — phase 7 gate 5
- `scripts/podman/run-ci-parity.sh <step>` (rootless podman; staging `/tmp/opencode/docker_ci/repo-salvage` @ `caf47145`; 9 steps exit 0; 2-rank 5/5 35.8s, 4-rank 5/5 1m01.4s; OpenMPI 4.1.6 probe `4-rank launch OK`) — phase 7 gate 4
- `JULIA_MPI_TEST_NPROCS=2 julia --project=test test/ext_distributed/runtests.jl mpi` (run in `../flux-integration`, direct runner, 6/6 PASS, 1m35.6s) — M1.1 evidence
- `JULIA_MPI_TEST_NPROCS=4 julia --project=test test/ext_distributed/runtests.jl mpi` (run in `../flux-integration`, direct runner, 6/6 PASS, 1m59.7s) — M1.1 evidence
- `make c8-mpi FLUX_REPO_PATH=../flux-integration` (local, 2 ranks, children 4-rank) — M1 evidence (superseded by the direct-runner commands above; the earlier 4-rank evidence was duplicated 4-rank children)
- `mpiexecjl -n 2 julia --project=... end_to_end_distributedtest.jl mpi` (local, 78/78)
- `mpiexecjl -n 4 julia --project=... end_to_end_distributedtest.jl mpi` (local, 78/78)
- `FLUX_TEST_DISTRIBUTED_NCCL=true salloc --gres=gpu:2 ... mpiexecjl -n 2 julia --project=... end_to_end_distributedtest.jl nccl` (Bocconi, 78/78)
- `make example-ddp` (Bocconi and local)
- Bocconi precompile workflow (validated 2026-09-03): `scripts/remote/sync_code.sh bocconi` -> `scripts/remote/setup_node.sh bocconi` (resolve-only, no compile) -> `scripts/remote/precompile.sh bocconi` (compute node, ~2-10 min) -> `NTASKS=2 CPUS_PER_TASK=4 scripts/remote/run.sh bocconi "make c8-mpi FLUX_REPO_PATH=../ddp_flux"` (warm, 4m25, 7/8 PASS)
- `make precompile` (local)

## Open questions

- Port dirty-master C8 tests as a separate commit only after the conditional-graph scope is stable; the C8 worktree (`ddp_flux` master at `b633cdc7`) is frozen.
- `docs/src/guide/gpu.md:366-377` still shows pre-M1 `Leaf` output — update deferred.
- The full local main-suite route remains unverified: `test/runtests.jl` on this 14 GiB host instantiates Reactant/XLA and exhausts memory; routing checks cover test selection and CI covers the direct runner. See `wiki/failures/2026-09-10-main-suite-reactant-instantiation.md`.
- The pre-existing NCCL failures recorded during C8.3 (`common`/`synchronized`/`optimizer`/`deadlock` tests on Bocconi) belong to the frozen `ddp_flux` master; re-check only if that work resumes. The Phase J/K NCCL validation used the active wrapper branch and passed.

Resolved and closed since earlier updates: `test/Project.toml` resolves with CUDA/NCCL (Phase J precompiled it); deathstar GPUs are healthy (3× TITAN RTX, Phase J/K ran there); NCCL runtime validation of `DistributedOptimizerState` is complete (Phase J 217/217 and 14/14; Phase K K1/K2); the precompile hardening question closed with the compute-node workflow (ADR-0005).

## Next exact action

1. Implement `notebook/conditional_graph_benchmark.ipynb` from the corrected `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/` artifacts (two-notebook plan implementation item 6).
2. Keep PR #2694 separate from the conditional-graph feature; do not add wrapper commits to that PR.

Phases A–K are complete at the level described at the top of this file. Evidence is in `artifacts/logs/conditional-wrapper/`.

PR #2694 remains separate from the conditional-graph feature. Do not add wrapper commits to that PR. The full local main-suite route remains unsafe on this host because Reactant/XLA exhausts memory.

Note: `docs/src/guide/gpu.md:366-377` (pre-M1 Leaf output) is unrelated to this PR scope and remains deferred. When replying to Carlo, explain that `distributed_setup.jl` centralizes the setup instead of `test/test_module.jl`: each MPI rank is a fresh standalone Julia process, so importing the full general test module into every rank is unnecessary.
