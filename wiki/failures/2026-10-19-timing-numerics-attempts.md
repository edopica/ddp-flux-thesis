# Timing numerics failed attempts — 2026-10-19

Two jobs attempted the 300-update final-state snapshot diagnostic before
`61366015` passed. Neither attempt produced accepted evidence. Any partial
records are not accepted evidence and must not be analyzed as results.

## Attempt 1 — job 61362264

The job failed after precompile and runtime capture. The runtime capture ran
`python -m pip`, but the benchmark environment is a uv-managed venv with no
`pip` module (`No module named pip`). The driver runs with `set -e`, so it
aborted before training.

- Timing records: none.
- Snapshots: none.
- Accepted evidence: none.

Fix: read the installed distribution metadata with `importlib.metadata`
instead of invoking pip, and make runtime capture non-fatal.

## Attempt 2 — job 61363645

The job failed after the Flux W1 configuration. `timing_torch.py` called
`mt.param_cpu_np` and `mt.optimizer_leaf_state_np`, but those functions live in
`phase4_torch.py`. The PyTorch configuration therefore stopped after Flux W1.

- Flux W1: one JSON record and one snapshot produced.
- Remaining configurations: no records.
- Accepted evidence: none; the partial Flux W1 record is not accepted.

Fix: import and call `param_cpu_np` and `optimizer_leaf_state_np` from
`phase4_torch`.

## Final run

Job `61366015` (`static-timing-numerics`) completed the diagnostic with 6/6
JSON records and 6/6 `state.npz` snapshots; all six configurations printed
`TIMING_PASS`, and the driver printed `STATIC_TIMING_numerics PASS` and
`TIMING_DRIVER_EXIT=0`. The comparison is descriptive only; no equivalence rule
is applied.

- [Final results](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md)
- [Review response](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md)
- [Devlog](../devlog/03-static-scaling/2026-10-19-timing-review-response.md)
