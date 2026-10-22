# Leonardo SSH publickey rejection blocks the Phase 2 remote re-check

Date: 2026-10-17 (recurrence 2026-10-18)
Scope: static-scaling Phase 2 acceptance re-check; Leonardo sync/re-check
Status: **RESOLVED for the campaign — recurring operational drops.** Access
worked for all required Phase 2/3 jobs; a later publickey rejection did not
block the campaign because access worked again when needed. No key change was
recorded or claimed. A separate transient local DNS/network drop occurred
during the 2026-10-18 recheck; the Slurm job continued and all evidence was
recovered.

## Update 2026-10-19 (recurrence during timing review follow-up)

The SSH publickey rejection recurred during the timing review follow-up. No
remote job was submitted. The existing campaign `61314790` evidence was
already local; it is unaffected.

## Update 2026-10-18 (update 2: successful jobs and transient network drops)

The final hardened Phase 3 review-fix campaign completed remotely:

- Precompile job `61187225` (`lrdn0241`): `FLUX_PRECOMPILE_OK`, exit 0.
- Focused precision regression job `61188854` (`lrdn1845`):
  `PHASE3_PRECISION_REGRESSION PASS`, exit 0, policy 10/10.
- Full Phase 3 recheck job `61191281` (`lrdn2911`): all four summaries PASS,
  exit 0.

The earlier intermediate runs (precompile `61074887`, focused `61077382`,
full recheck `61091402`) are superseded history.

`ssh -o BatchMode=yes leonardo true` returned exit 0 on 2026-10-18. A transient
local DNS/network drop occurred at `08:36:52Z` during the intermediate job
`61091402`. The Slurm job continued, completed `0:0`, and the evidence was
copied. A separate transient DNS interruption occurred while recovering the
deathstar ten-case policy log; the log was recovered. These drops are not
proven to have the same cause as the publickey rejection, and they are not
evidence that the key was changed or fixed.

## Update 2026-10-18 (earlier resolution and recurrence)

Leonardo access worked for the hardened Phase 2 re-check (job `60951247`) and
the Phase 3 final campaign: precompile `60954207`, precision regression
`60954547`, optimizer probe `60955690`, final driver `60956582`, all PASS.
Immediately after the final evidence copy, `ssh leonardo` again returned

```text
epicazio@login.leonardo.cineca.it: Permission denied (publickey,gssapi-keyex,gssapi-with-mic)
```

At that point the reference NPZ, manifest, logs, probe outputs, runtime record,
and remote SHA-256 transcript were already local; the remote/local SHA-256
comparison is 15/15 MATCH. The recurrence is operational only and does not
change the Phase 3 verdict. If more remote work is needed, check access with
`ssh leonardo true` first. See update 2 above: access later worked again
without a recorded key change.

## Symptom

An SSH connection to Leonardo fails during authentication, before any remote
command runs:

```text
$ ssh leonardo
epicazio@login.leonardo.cineca.it: Permission denied (publickey,gssapi-keyex,gssapi-with-mic)
```

The workstation has no identity loaded in the agent:

```text
$ ssh-add -l
The agent has no identities.
```

The on-disk key that SSH offers has fingerprint `SHA256:TDVaBv4y...`
(`~/.ssh/id_ed25519`, ED25519). The server rejects it.

## Evidence

Commands observed on 2026-10-17 from the thesis workspace:

```bash
ssh leonardo
# epicazio@login.leonardo.cineca.it: Permission denied (publickey,gssapi-keyex,gssapi-with-mic)

ssh-add -l
# The agent has no identities.
```

Fingerprint of the on-disk key offered for authentication:
`SHA256:TDVaBv4y...` (`~/.ssh/id_ed25519`, ED25519).

No remote shell was opened and no sync, precompile, or check job ran. The
failure is at authentication, not in the Phase 2 checker code.

## Context

The Leonardo path worked through 2026-10-16: Phase 2 prep job 60758615 and
check job 60758839 completed (0:0), and the artifacts were copied back. The
2026-10-17 attempt was the first sync/re-check after the acceptance fixes, and
it failed before the sync step.

The acceptance-fix campaign itself is local: R1-R3 are corrected, the positive
checks pass, and `negative_tests_local.py` ends `NEGTEST PASS`. The only
outstanding Phase 2 acceptance item is the remote re-check on the real `$PREP`.

## Root cause

Unknown. The server rejects the offered on-disk key and the local agent has no
identity. It is not determined whether the key was rotated or removed from the
CINECA account, whether the workstation is offering the wrong key, or whether
the agent is required by the current server policy. The failure is not caused
by the Phase 2 code or artifacts.

## Workaround and next steps

1. Restore the authorized key/agent on the workstation. Confirm that
   `~/.ssh/id_ed25519.pub` is present in the Leonardo account's
   `~/.ssh/authorized_keys` (use the CINECA user portal or support if the key
   must be re-registered), then load it with `ssh-agent` and `ssh-add`.
2. Verify with `ssh leonardo true`.
3. Sync the repository and `flux-integration` with
   `bash scripts/remote/sync_code.sh leonardo`.
4. If the sync changed Julia code, precompile on a compute node; otherwise run
   the Phase 2 check job:
   `scripts/remote/leonardo/run-phase2-check.sh` on the real `$PREP` and
   fixtures. The response report also records the real-`$PREP` invocation of
   `scripts/static_scaling/negative_tests_local.py`.
5. Then proceed to Phase 3 per
   `temp/static-graph-gpu-scaling-implementation-phases.md`.

## Impact

The remote re-check on the real `$PREP` is deferred. Local positive and
negative acceptance evidence is unaffected, and no canonical artifact changed.
The blocker is operational only: it must be cleared before the Phase 3 entry
gate can include the real-data remote run.

## Links

- [Phase 2 acceptance-fix devlog](../devlog/03-static-scaling/2026-10-17-phase2-acceptance-fixes.md)
- [Phase 2 review response](../../reports/static-scaling-phase2-review-response.md)
- [Leonardo context](../context/leonardo.md)
