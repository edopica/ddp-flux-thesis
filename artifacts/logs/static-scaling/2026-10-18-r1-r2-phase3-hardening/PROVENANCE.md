# Provenance — Phase 3 hardening recheck (2026-10-18)

Primary driver job `61191281` (`lrdn2911`, UTC start `2026-10-18T12:30:16Z`,
driver `12:30:21Z`, elapsed `00:07:52`, exit 0). Canonical precompile job
`61187225` (`lrdn0241`, UTC `12:20:59`-`12:24:36`, `FLUX_PRECOMPILE_OK`,
exit 0).

## 1. Generation-time source identity

```text
THESIS_HEAD=a3786d97f3263cad20b254206f09c095880f624d
THESIS_STATE_SHA256=da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The thesis state hash is generation-time. Later documentation edits (including
this file and the evidence reports) move the live worktree hash; they do not
change the source identity that produced the run. The Flux worktree was clean
at the generation-time commit. The remote trees are rsync deployments, so the
run log records `THESIS_GIT=<no-git>` and `FLUX_GIT=<no-git>`.

The generator and checker both record the same source and Flux identity in the
reference manifest and the run log.

## 2. Script identity

The hardened helper scripts were synced from the thesis tree. Their SHA-256
values at the generation-time source are:

```text
3efd8f4a1b37da847a7bd456138dff86a8dfb927e9cdfb0ebfb64011564c2b81  scripts/static_scaling/precision.jl
e0aef52ec41a4bf5822793141141522d8b09af166d0fb595df0673d1363141fe  scripts/static_scaling/test_precision_policy.jl
dfd28b9f4911b5d7f10558904a146970ef5c3624d51bdd0d55ed5920ca39497e  scripts/static_scaling/check_precision_math.jl
43ad641bfaad58f52e5aa2ab1405113dde29b107a27c5e21f8f549d978b985a1  scripts/static_scaling/check_phase3_flux.jl
92ceb205008029a72b9d9ff7ba5d354005a84cfae86bdd50b36032a81217217c  scripts/remote/leonardo/run-phase3-precision-regression.sh
f52c454de1ecbadaa45a7c7be9556ab59ab0628d99269e3d0a0ad2abc5432f6b  scripts/remote/leonardo/run-phase3-check.sh
ddb4e3b8a107341ccb8f1004248450836dc1bad9a88b0b6837169adfd65212b1  scripts/remote/leonardo/precompile-phase3.sh
```

The remote tree is pinned by `THESIS_STATE_SHA256`; separate remote per-file
hash transcripts were not retained in this root. The precision hardening root
contains the focused run's remote/local log comparison.

## 3. Reference identity and SHA comparison

```text
reference/phase3_reference.npz
  97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783
reference/phase3_reference_manifest.json
  bfd979e2d575e24004860c07c216779325923b73e9873bb063c899acab04109a
```

The reference is 337/337 exact-equal to
`artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/`
with `max_abs_diff=0`. The independent NumPy checker passes. The remote/local
comparison is `SHA256_COMPARISON PASS files=5 mismatches=0`; the precision
hardening root reports `files=3 mismatches=0`. See
[`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt),
[`remote-SHA256.txt`](remote-SHA256.txt), and
[`local-array-comparison.txt`](local-array-comparison.txt).

## 4. Superseded intermediate runs

The pre-hardening helper runs are retained as history, not primary evidence:

| Run | Job | Root |
|---|---|---|
| Precompile | `61074887` (`lrdn0421`) | `artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-fixes/` |
| Focused regression | `61077382` (`lrdn0200`) | `artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-fixes/` |
| Full recheck | `61091402` (`lrdn0870`) | `artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-recheck/` |

They used the pre-hardening helper and 7 policy cases. The final hardened
evidence is jobs `61187225`, `61188854`, and `61191281`.

## 5. Evidence files

- [`logs/remote-check.log`](logs/remote-check.log)
- [`console/phase3-driver-console.log`](console/phase3-driver-console.log)
- [`console/precompile-phase3.log`](console/precompile-phase3.log)
- [`category-table.txt`](category-table.txt)
- [`local-reference-check.log`](local-reference-check.log)
- [`local-array-comparison.txt`](local-array-comparison.txt)
- [`remote-SHA256.txt`](remote-SHA256.txt)
- [`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt)
- [`SHA256SUMS`](SHA256SUMS)
