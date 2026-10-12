# Conditional detector: MPI trace framing

Date: 2026-09-30

## Failure

The first Phase G run returned 1 after 206 passed assertions and four parser errors.
All three child jobs returned 0 with correct parameter values on both ranks.

Command:

```bash
make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration
```

Log: `artifacts/logs/conditional-wrapper/phaseG-detector-mpi2-2026-09-30.log`.
The exact capture command is in the Phase G devlog.

## Cause and correction

`println` wrote a message and its newline separately. MPI output from another rank sometimes arrived between these writes.
The parser read `0.0TRACE` as one numerical field and raised `ArgumentError`.

The probe now writes each record and newline together. Each collective trace also ends with `END`, which the parser requires.
The corrected run passed 210/210 assertions with exit code 0.

Log: `artifacts/logs/conditional-wrapper/phaseG-detector-GREEN-mpi2-2026-09-30.log`.
