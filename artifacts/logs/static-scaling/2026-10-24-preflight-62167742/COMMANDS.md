# Memory preflight — commands

Job `62167742`, `2026-10-24`. See [RESULTS.md](RESULTS.md).

## Sync

The synced thesis tree had no `.git`; the launcher therefore used
`THESIS_COMMIT_SOURCE=expected-no-git`.

```bash
rsync -avz --delete --exclude-from='.rsyncignore' \
  -e 'ssh -F ~/.ssh/config' \
  . leonardo:'~/projects/ddp-flux-thesis/'
```

The two Flux worktrees already had the pinned source trees; their `src`
SHA-256 values were verified before submission and were not re-synced.

## Submit preflight-only

```bash
ssh -F ~/.ssh/config leonardo \
  'cd $HOME/projects/ddp-flux-thesis && \
   CONFIRM_CAMPAIGN=1 sbatch --parsable \
     --export=ALL,CONFIRM_CAMPAIGN=1 \
     -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_lprod \
     -N1 -n1 --cpus-per-task=4 --gres=gpu:4 --time=02:00:00 \
     --job-name=static-full-preflight \
     --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-preflight-%j.log \
     ./scripts/remote/leonardo/run-static-full-campaign.sh preflight'
```

The launcher creates
`$WORK/epicazio/static-scaling-evidence/static-full-campaign-62167742`.
It runs the nine batch-256 cases only; it does not start the 390-update
campaign.

## Local copy and independent check

```bash
rsync -avz -e 'ssh -F ~/.ssh/config' \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-62167742/' \
  artifacts/logs/static-scaling/2026-10-24-preflight-62167742/

cd scripts/static_scaling
python3 check_full_campaign.py preflight \
  --root ../../artifacts/logs/static-scaling/2026-10-24-preflight-62167742 \
  --candidate 256=../../artifacts/logs/static-scaling/2026-10-24-preflight-62167742/preflight/b256 \
  --updates 20 --warmup 20 --select \
  --out ../../artifacts/logs/static-scaling/2026-10-24-preflight-62167742/recheck-decision.json
```

Observed local recheck: nine `PASS` rows and `PREFLIGHT_SELECTED 256`.

```bash
python3 check_full_campaign.py ready \
  --root ../../artifacts/logs/static-scaling/2026-10-24-preflight-62167742
# FULL_CAMPAIGN_READY PASS batch=256 updates=390
```
