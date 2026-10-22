# Phase 4 remote access unavailable — 2026-10-18

## What failed

Two remote prerequisites for Phase 4 could not be reached at the end of the
implementation session.

### Leonardo

```text
$ ssh -o ConnectTimeout=10 leonardo 'hostname; date -u'
epicazio@login.leonardo.cineca.it: Permission denied (publickey,gssapi-keyex,gssapi-with-mic).
```

A prior attempt through `scripts/remote/run.sh leonardo` failed with the same
publickey error. This is a recurrence of
`wiki/failures/2026-10-17-leonardo-ssh-publickey.md`, not a new key change:
Leonardo access worked earlier in the static-scaling campaign (jobs
`60956582`, `61191281`, `61188854`, `61187225`).

### deathstar

```text
$ ssh -o ConnectTimeout=10 deathstar 'hostname'
ssh: Could not resolve hostname deathstar-hpc.sm.unibocconi.it: Name or service not known
```

## Impact

- No Phase 4 GPU/NCCL run on Leonardo.
- No Phase 4 CPU/MPI smoke on deathstar.
- The remote drivers are implemented and pass local `bash -n` and
  fake-runner end-to-end tests, but nested `srun --overlap`, GPU visibility,
  and torchrun port behavior remain unverified.

## Workaround / next action

Retry both hosts before any Phase 4 remote work. If Leonardo still denies the
key, re-check the agent/key state per the 2026-10-17 note; if deathstar DNS is
still unresolved, use the host IP or wait for the workstation to come back.
Then follow `temp/phase4-handoff.md` section 5.
