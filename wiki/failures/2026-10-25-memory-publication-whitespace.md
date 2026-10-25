# Publication whitespace check on archived evidence — 2026-10-25

After staging both completed campaign bundles, `git diff --cached --check`
reported whitespace errors in original captured logs and archived upstream
Flux source (including trailing whitespace and space-before-tab indentation).
Those files are preserved evidence; reformatting them would change the source
snapshot and invalidate its hashes. This is unrelated to training correctness.

The pre-staging `git diff --check` passed for tracked working changes.
For publication, validate authored code/documents separately:

```bash
git diff --cached --check -- README.md reports scripts wiki
```

That scoped check passes. Keep original log/source bytes unchanged, backed by
the passing campaign checkers and original diagnostic checksum manifest.
No test, campaign or source-identity gate failed.
