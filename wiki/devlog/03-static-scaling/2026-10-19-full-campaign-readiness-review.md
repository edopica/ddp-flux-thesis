# Campaign readiness review — 2026-10-19

Status: **NO-GO for the current launcher, including preflight.**
This review changed documentation only. No remote command ran.

## Findings

1. **P1: precompile fails before execution.**
   `run-static-full-campaign.sh:262` declares `name` and expands `$name` in
   the same `local` command under `set -u`. A fresh shell reports
   `name: unbound variable`. Split the dependent declaration.

2. **P1: memory rejection prevents the fallback.**
   At launcher lines 631–634, successful batch-512 processes give `status512=0`.
   The checker returns 1 when no candidate passes the memory criterion.
   This combination exits before the batch-256 runs at line 642.
   The Python fallback self-test bypasses this shell control flow.
   Distinguish a valid memory rejection from a checker execution failure.

3. **P1: campaign mode checks the reused decision too late.**
   Launcher lines 817–826 copy the decision and accept its selected batch.
   They do not require a PASS candidate or matching source identity before timing.
   `_check_preflight_decision` runs through the final checker at launcher line 871.
   Thus an invalid or stale decision can consume the complete campaign allocation.
   Run the acceptance checks before the first timed case.
   Include runtime, resource, dataset, and source compatibility in those checks.

4. **P1: memory acceptance lacks per-rank sample coverage.**
   `check_full_campaign.py:222–250` accepts any numeric process sample.
   It discards the collected GPU UUID set and does not associate samples with training ranks.
   Lines 828–841 accept the maximum available sample without requiring all ranks.
   The existing self-test passes W=4 with one process sample.
   A partial sampler result can therefore establish a false memory-margin PASS.
   Require samples for each training process and its assigned GPU.
   Record sample times and describe the result as a sampled peak.

## Local checks

These commands passed:

```bash
bash -n scripts/remote/leonardo/run-static-full-campaign.sh && python3 scripts/static_scaling/check_full_campaign.py --self-test
julia --startup-file=no -e 'Meta.parseall(read("scripts/static_scaling/full_dataset_timing.jl", String)); println("JULIA_PARSE_OK")'
```

The checker printed `FULL_CAMPAIGN_SELFTEST PASS checks=21`.
The Julia parser printed `JULIA_PARSE_OK`.
Synthetic timing output is not benchmark evidence.

The following controlled reproductions returned wrapper exit 0 and exposed the shell defects:

```bash
bash -c 'set -u; OUT=/tmp/opencode; precompile_variant() { local name=$1 dir=$2 log="$OUT/logs/precompile-$name.log"; }; if (precompile_variant current /unused); then printf "REPRO unexpectedly passed\n"; else printf "REPRO CONFIRMED: precompile local declaration fails under nounset\n"; fi'
bash -c 'status512=0; checker512=1; selected=""; if [ "$status512" -eq 0 ] && [ "$checker512" -ne 0 ]; then printf "REPRO CONFIRMED: completed batch 512 over memory limit takes exit branch before batch 256\n"; else printf "fallback reachable\n"; fi'
```

The first reproduction printed `name: unbound variable` and the confirmation marker.
The second reproduction printed the confirmation marker for the premature exit branch.

## Exact next action

Correct these findings and add shell control-flow regressions.
Resolve the historical worktree identity difference before any source sync.
Then run the compute-node precompile and nine-configuration preflight.
Require a reviewed PASS before the campaign.
