# Results — Phase 3 hardening recheck (2026-10-18)

Status: **PASS — hardened helper reproduces the accepted Phase 3 result.**
Driver job `61191281` (`lrdn2911`, UTC start `2026-10-18T12:30:16Z`, driver
`12:30:21Z`, elapsed `00:07:52`, exit 0).

## 1. Required summaries

| Summary | Result |
|---|---|
| `PHASE3_REFERENCE` | PASS, `checks=32` |
| `PHASE3_REFERENCE_CHECK` | PASS, `arrays=337` |
| `PHASE3_FLUX_CHECK` | PASS |
| `PHASE3_REMOTE_CHECK` | PASS |

The log contains no `Precompiling packages...` line. The canonical precompile
was job `61187225` (`lrdn0241`, UTC `12:20:59`-`12:24:36`,
`FLUX_PRECOMPILE_OK`, exit 0).

## 2. Per-category acceptance table

| Category | Max abs diff | Violations |
|---|---:|---:|
| eta | 0 | 0 |
| activations | 1.56164169e-05 | 0 |
| logits | 1.7285347e-06 | 0 |
| loss | 4.76837158e-07 | 0 |
| grads.u1 | 3.02582979e-04 | 0 |
| grads.u3 | 6.08410686e-04 | 0 |
| params.u3 | 4.77954745e-06 | 0 |
| m.u3 | 6.32628798e-05 | 0 |
| v.u3 | 2.32830644e-07 | 0 |
| beta_power.u3 | 0 | 0 |
| step | reference 3 = Flux 3 | 0 |

## 3. Algorithm choices

```text
PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0 enabled=true
CHECK cudnn.fma_algo_selection PASS
```

All 32 choices are `CUDNN_FMA_MATH(3)` with `fallback=false`. The run used
CUDA.jl `6.2.2`, cuDNN.jl `6.2.2`, and cuDNN library `92.60.0`.

## 4. Reference identity

| Item | Value |
|---|---|
| Reference NPZ SHA-256 | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| Reference manifest SHA-256 | `bfd979e2d575e24004860c07c216779325923b73e9873bb063c899acab04109a` |
| Preserved archive | `artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/` |
| Comparison | 337/337 exact-equal, `max_abs_diff=0` |
| Independent reference check | PASS |
| Remote/local SHA-256 | precision 3/3 MATCH; Phase 3 5/5 MATCH |

## 5. Source identity

Generation-time identity for job `61191281`:

```text
THESIS_HEAD=a3786d97f3263cad20b254206f09c095880f624d
THESIS_STATE_SHA256=da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Later documentation edits move the live worktree hash. The values above are
the run-time provenance.

## 6. Superseded intermediate runs

The pre-hardening helper runs are retained as history, not primary evidence:
precompile `61074887`, focused `61077382`, full recheck `61091402`. Their roots
are `artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-fixes/`,
`artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-recheck/`, and
`artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-recheck/`.

## 7. Evidence

- [`logs/remote-check.log`](logs/remote-check.log)
- [`console/phase3-driver-console.log`](console/phase3-driver-console.log)
- [`category-table.txt`](category-table.txt)
- [`local-reference-check.log`](local-reference-check.log)
- [`local-array-comparison.txt`](local-array-comparison.txt)
- [`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt)
- [`PROVENANCE.md`](PROVENANCE.md)
- [`COMMANDS.md`](COMMANDS.md)
