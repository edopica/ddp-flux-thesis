# Phase 2 review harness exceeded the local scratch quota

Date: 2026-10-17
Status: recovered with an existing synthetic PREP and canonical fixture-only directory

## Command and failure

```bash
TMPDIR=/tmp/opencode python3 scripts/static_scaling/negative_tests_local.py --prep-dir /tmp/opencode/phase2-negtest/synthetic --fixture-dir /tmp/opencode/phase2-negtest/synthetic --bundle-manifest /tmp/opencode/phase2-negtest/synthetic/initial_bundle.manifest.json --workdir /tmp/opencode/phase2-reviewer-followup
```

The command exited 1.
Positive Python and Julia checks passed, and the epoch-49 mutation received the expected rejection.
The fixture mutation then copied the entire synthetic directory, including `train_images.npy`.
That copy failed with `[Errno 122] Disk quota exceeded`.
The subsequent GroupNorm scratch copies failed with the same error.

`df -h /tmp/opencode` reported 2.4 GiB available before the run.
That filesystem-wide value did not prevent the quota failure.

## Recovery

Only this review's incomplete mutation directories were removed. Its log directory remained available.

```bash
rm -rf -- /tmp/opencode/phase2-reviewer-followup/mutation_fixture_label /tmp/opencode/phase2-reviewer-followup/mutation_gn_julia /tmp/opencode/phase2-reviewer-followup/mutation_gn_python
ls -d /tmp/opencode /tmp/opencode/phase2-negtest/synthetic
mkdir /tmp/opencode/phase2-reviewer-followup-green && cp -al /tmp/opencode/phase2-negtest/synthetic /tmp/opencode/phase2-reviewer-followup-green/synthetic && TMPDIR=/tmp/opencode python3 scripts/static_scaling/negative_tests_local.py --workdir /tmp/opencode/phase2-reviewer-followup-green
```

The corrected invocation used canonical real fixtures for Julia cases and reused synthetic PREP for Python cases.
All six cases passed, final `NEGTEST PASS`, exit 0.
The harness detached the augmentation hardlink before mutation and checked that the source hash remained unchanged.

No production code or canonical artifact changed.
The failure concerns local scratch capacity, not acceptance correctness.
