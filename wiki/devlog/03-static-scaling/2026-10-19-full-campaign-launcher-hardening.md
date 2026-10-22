# Full-dataset campaign launcher hardened — 2026-10-19

**Status: local hardening only. No remote job ran for this launcher. No
preflight and no campaign has been executed. No throughput, scaling,
memory-margin, or numerical-equivalence claim exists.** The launcher and checker
are currently untracked in git.

## Summary

`scripts/remote/leonardo/run-static-full-campaign.sh` is no longer a template.
It implements bounded preflight and campaign modes with fail-closed source
identity checks, compute-node precompile, runtime/resource capture, external
GPU memory sampling, and artifact generation. The companion checker
`scripts/static_scaling/check_full_campaign.py` validates the resulting
artifact tree. Local syntax, compile, self-test, and Julia-parse gates pass.

## Launcher changes

`run-static-full-campaign.sh` provides three modes:

| Mode | Behavior |
|---|---|
| `all` | Run preflight, then the timing campaign and diagnostics in one job |
| `preflight` | Run only the bounded preflight and write the batch/memory decision |
| `campaign` | Reuse `PREFLIGHT_DECISION` and run timing and diagnostics only |

The driver requires `CONFIRM_CAMPAIGN=1`, rejects a login-node host, requires
one task, requires a fresh output root, and fails if a required path is
missing.

Source identity is verified against pinned constants before any workload:

| Item | Pinned value |
|---|---|
| Thesis commit | `a3786d97f3263cad20b254206f09c095880f624d` |
| Flux current commit | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux current state | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Flux current source tree | `83da2929ab7ebd66dbceb83452c8e55eae5e0444c8e96866cc2a25b513c3eb0d` |
| Flux historical commit | `8add0e42b7d9388ff365335152126e5b1e9de657` |
| Flux historical state | `4af8409c2a2ed1c661e8ae7742877f1ee9789d00dc7e404365ecefd98ab0d34c` |
| Flux historical source tree | `ad114ed385f7cc78e8409fa8e1f09d38133555f7fd58d5d9aca6a846bcf5e757` |

For Flux, state is the SHA-256 of `git diff HEAD` plus the sorted hashes of
untracked files that git does not ignore. Flux source tree is the hash of the
sorted `src/` file hashes. A no-git source falls back to the expected constants but still requires
the expected source-tree hash. The artifact checker additionally requires the
current and historical commits and source trees to differ.

The launcher precompiles both Flux worktrees on the compute node and requires
`FLUX_PRECOMPILE_OK` in each log before runtime capture. It then writes:

- `source-identity.txt`, `source/`, `source-sha256.txt`, `launcher-sha256.txt`;
- `runtime-env.txt`, `runtime.json` (Julia, Flux, CUDA, NCCL, MPI, Optimisers,
  Functors, Python, torch, numpy, cuDNN, GPU UUIDs and totals, driver);
- `resources.txt`, `resources.json` (Slurm job, node, partition, QoS, account,
  CPU/task, GRES, time limit, GPU mapping);
- `dataset-identity.json` (dataset file hashes, bundle identity, transform
  spec, per-batch sample/update-1 order hashes, workload counts);
- `campaign-config.json`, `provenance.json`, `preflight/decision.json`;
- timing records, diagnostic records and snapshots;
- `REPORT.json`, `SUMMARY.json`, `SHA256SUMS`, `driver.log`, and driver markers.

### Preflight

The preflight covers all nine framework/world combinations:
`flux-current`, `flux-historical`, and `pytorch` at W=1, 2, and 4. Each
configuration runs the full dataset with 20 discarded warmup updates and 20
sustained updates. A separate `nvidia-smi` sampler records per-process device
memory (default interval 0.25 s).

Batch 512 is selected only if all nine configurations pass the recorded OD-8
criterion: peak per-process device memory must not exceed 80% of the 64-GiB
device total. A missing sample or memory counter is `INCOMPLETE` and cannot
pass.

If batch 512 does not pass, batch 256 is tested for all nine configurations.
A 512 candidate whose runs fail is excluded from the fallback check; a
completed 512 candidate that fails the memory criterion remains recorded but no
longer aborts the 256 fallback. Batch 256 is selected only if all nine
configurations pass.
If neither batch passes, the driver exits non-zero; the campaign does not run.

### Campaign

The campaign uses the selected global batch:

| Batch | K | Update count source |
|---:|---:|---|
| 512 | 195 | 99,840 ids / 512 |
| 256 | 390 | 99,840 ids / 256 |

The default is 5 repeats, repeat-first: each repeat runs all nine
configurations before the next repeat starts. Even-numbered repeats reverse
the world-size and framework order. After the timed campaign, a separate
untimed diagnostics matrix runs the same batch, epoch, K, and warmup for all
nine configurations and writes one state snapshot per configuration.

## Checker changes

`scripts/static_scaling/check_full_campaign.py` has `preflight`, `campaign`,
and `--self-test` subcommands. It validates:

- the accepted source constants above and current/historical distinctness;
- `source-identity.txt`, the copied source snapshot, launcher/checker hashes,
  campaign config, provenance, preflight decision, dataset identity, runtime,
  and resources;
- runtime-versus-record Julia, Python, torch, and numpy versions;
- the exact timing record set (`<framework>-w<world>-r<repeat>.json`) and
  diagnostic record set (`<framework>-w<world>.json`), with no missing or
  unexpected files;
- record timing arithmetic, rank durations, replica hashes, optimizer step,
  dataset/bundle/order/transform hashes, policy fields, and finite loss;
- `eta_sha256` consistency across the campaign and between each timing and
  diagnostic pair;
- the nine diagnostic snapshots, including shape, finiteness, `beta_power`,
  and the OD-14 descriptive comparison set;
- `NUMERICAL_EQUIVALENCE_CLAIM none` in provenance; the diagnostics report
  retains the same claim.

`full_dataset_timing.jl` and `timing_torch.py` now emit `eta_sha256`, the
SHA-256 of the Float32 learning-rate sequence for updates `1..K`. The checker
treats a missing or non-hex value as a failure. It checks consistency, not an
independent recomputation of the schedule.

The self-test builds a synthetic campaign tree, exercises a passing preflight,
a batch-512 failure with batch-256 fallback, and mutation rejection. Its
printed timing and diagnostic tables are synthetic and are not evidence.

## Local verification

Commands run locally on 2026-10-19:

```bash
bash -n scripts/remote/leonardo/run-static-full-campaign.sh
python3 -m py_compile scripts/static_scaling/check_full_campaign.py \
  scripts/static_scaling/timing_torch.py
python3 scripts/static_scaling/check_full_campaign.py --self-test
julia --startup-file=no -e \
  'Meta.parseall(read("scripts/static_scaling/full_dataset_timing.jl", String)); println("JULIA_PARSE_OK")'
command -v shellcheck
```

| Check | Result |
|---|---|
| `bash -n` | exit 0; wrapper printed `BASH_SYNTAX_OK` |
| `py_compile` | exit 0 |
| Checker self-test | `FULL_CAMPAIGN_SELFTEST PASS checks=21` |
| Julia parse | `JULIA_PARSE_OK` |
| `shellcheck` | not installed locally; no shellcheck run |

## Preflight and campaign procedure

Sync the thesis and both Flux worktrees first. Precompile both Flux worktrees on
a compute node. The launcher also precompiles them before runtime capture and
fails without `FLUX_PRECOMPILE_OK`.

Submit the preflight-only job with 4 GPUs:

```bash
ssh leonardo 'CONFIRM_CAMPAIGN=1 CAMPAIGN_MODE=preflight sbatch --parsable \
  -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 \
  --cpus-per-task=32 --gres=gpu:4 --time=04:00:00 \
  --job-name=static-full-campaign \
  --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-%j.log \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-campaign.sh"'
```

Inspect the job output and `OUT/preflight/decision.json` (and
`decision-b512.json` when present). Require a selected batch of 512 or 256 and
a selected candidate status of `PASS`. Review the per-configuration peak,
limit, margin, and sample count. Do not submit the campaign if no candidate is
selected, the selected candidate is not `PASS`, or a required sample is
missing.

After a reviewed PASS, either submit `CAMPAIGN_MODE=all` (which reruns
preflight in the same job) or submit `CAMPAIGN_MODE=campaign` with
`PREFLIGHT_DECISION=<preflight-out>/preflight/decision.json` in a fresh output
root. The campaign command was not executed.

## Risks

- **OD-8 memory margin is unresolved.** The earlier loader validation measured
  Flux device-used at 86.34% (current) and 83.14% (historical) of the A100 total
  at batch 512/W=1, above the 80% criterion. Batch 512 did not OOM and no 256
  fallback ran then. Peak process memory has not been measured by this
  launcher's preflight.
- **Local historical Flux state drift.** The local historical worktree state is
  `34005a430cc8be2aa662e0ca59328e07a824231cb9858f52fce3ac388afc330c`
  versus the pinned `4af8409c...`. Remote state was not re-verified. This is a
  preflight risk, not evidence that the remote tree drifted. The launcher fails
  closed on a mismatch.
- **Untracked files.** The launcher and checker are untracked. A commit-only
  sync would omit them; sync the worktree and confirm both files exist on the
  remote before submission.
- **No remote evidence.** No job has run, so remote paths, pinned worktrees,
  and runtime versions are unverified in this checkpoint.
- **Shellcheck unavailable.** Only `bash -n` was run locally; shellcheck has not
  been run.

## Exact next action

**Sync the thesis and both Flux worktrees, precompile on a compute node, submit
the preflight-only job with 4 GPUs, inspect the nine-configuration decision,
and do not launch the campaign until preflight is PASS.**

## Evidence

- Launcher: `scripts/remote/leonardo/run-static-full-campaign.sh`
- Checker: `scripts/static_scaling/check_full_campaign.py`
- Timing paths: `scripts/static_scaling/full_dataset_timing.jl`,
  `scripts/static_scaling/timing_torch.py`
- Prior loader validation:
  [RESULTS](../../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/RESULTS.md),
  [devlog](2026-10-19-full-dataset-loader-validation.md)
- Prior validation memory risk: [current context](../../context/current.md)
