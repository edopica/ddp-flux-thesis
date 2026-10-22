# Full campaign claims review — 2026-10-20

Status: saved campaign PASS and timing arithmetic supported, with qualifications.

The review checked Leonardo job `61493236`, its saved artifacts, its source snapshot, and the OD-14 policy.
Slurm reports `COMPLETED 0:0` in `02:46:11`.
All 45 timing records and nine diagnostic records exist.
All 307 remote checksum entries pass. The local copy passes all 298 non-snapshot entries.
Independent arithmetic reproduces all submitted timing values.

The report records four important qualifications:

- Historical Flux scales best at W4. PyTorch has slightly greater observed speedup at W2.
- The first-repeat outlier exists, but its cold-cache attribution lacks direct evidence.
- The timers include data preparation and transfers. The implementations have different guards and determinism policies.
- Aggregate parameter differences are about 2%. Optimizer-moment differences are substantially greater.

The saved diagnostic report has 15 descriptive comparisons and no numerical threshold.
This review did not recompute snapshot differences or run new GPU workloads.
Slurm step `61493236.9` failed, but its command and purpose remain unresolved.

Report: [claim review](../../../reports/static-scaling-full-campaign-61493236-review.md).
Review-tool failures: [failure note](../../failures/2026-10-20-campaign-review-tools.md).

## Reproduction commands

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 leonardo 'sacct -X -j 61493236 --format=JobID,State,ExitCode,Elapsed,Start,End,NodeList -P'
ssh -o BatchMode=yes -o ConnectTimeout=20 leonardo 'sacct -j 61493236.9 --format=JobID,JobName%40,State,ExitCode,Start,End -P'
rsync -az --exclude='*.npz' leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-61493236/ /tmp/opencode/review-61493236/
ssh -o BatchMode=yes -o ConnectTimeout=20 leonardo 'python3 -c '\''import subprocess; p="/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-61493236"; r=subprocess.run(["sha256sum","--check","--quiet","SHA256SUMS"],cwd=p,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE); print("CHECKSUM_EXIT",r.returncode); print(r.stdout,r.stderr)'\'''
python3 artifacts/logs/static-scaling/2026-10-20-campaign-review/verify_claims.py /tmp/opencode/review-61493236
git diff --check
```

The arithmetic script first ran as `/tmp/opencode/verify_campaign_claims.py` with the same local evidence root.
The retained script accepts that root as an argument.

## Structured report and step `.9` resolution — 2026-10-20

Added `reports/static-scaling-full-campaign-61493236-report.md`. It covers the
research question, campaign design, execution pipeline, correctness/accuracy
harness, results, threats to validity, and next steps.

Resolved the review's open item: Slurm step `61493236.9` was the operator's
60-second `srun --overlap` `/proc` poll, not a campaign workload. `sacct` shows
the step's SubmitLine as `srun --jobid=614932+`; it exited 2 when the Julia PID
disappeared and `cat /proc/<pid>/io` failed. The campaign driver and all 45
records are unaffected.

The first-repeat outlier cause remains unconfirmed. Next action: run the OD-14
deterministic-mode sensitivity and a controlled cache experiment before offering
stronger claims.
