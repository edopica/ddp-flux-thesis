# Commands — static-scaling Phase 3 final acceptance (2026-10-18)

All commands were run from the thesis repository root unless noted.
No commit was made. No tolerance, model-math, optimizer, schedule, or
precision-policy value was changed during this campaign.

## 1. Final source snapshot

```bash
git status --short
git diff --check
git -C ../flux-integration status --short
git -C ../flux-integration diff --check

SOURCE_COMMIT=$(git rev-parse HEAD)
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } \
  | sha256sum | cut -d' ' -f1 )
FLUX_COMMIT=$(git -C ../flux-integration rev-parse HEAD)
FLUX_STATE=$( { git -C ../flux-integration diff HEAD; \
  git -C ../flux-integration ls-files --others --exclude-standard \
  | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

Result (2026-10-18, before the run):

```text
SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3
SOURCE_STATE_SHA256=4a28adb1b031ddacb14aff3bce314d204e693c2d55910cdfe766d5ff33f131fe
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
GIT_DIFF_CHECK_OK
FLUX_DIFF_CHECK_OK
```

93 worktree entries, all Phase 2/3 work; no unintended changes and no
secrets found. `git -C ../flux-integration status --short` was empty.
Per-file SHA-256 values are in [`source-sha256.txt`](source-sha256.txt).

## 2. Sync to Leonardo

The first full explicit-exclusion rsync (only the two preserved attempt-3
directories excluded) was killed by the 900 s local timeout while uploading
the historical 286 MB reference archives. No source file was affected.

The successful sync excluded all historical `2026-10-17-phase3-*` reference
archives (a superset of the two preserved dirs; the remote Phase 2 fixture
tree was already present and unchanged):

```bash
rsync -az --delete --stats --exclude-from='.rsyncignore' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-*' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-*' \
  . leonardo:projects/ddp-flux-thesis/
```

Completed in 8.6 s: 1,696 files, 31 transferred, 2 deleted. The protected
remote `test/Manifest.toml` stayed in place (`.rsyncignore`).

Remote key-source SHA-256 values matched local before any run:

```text
MATCH scripts/static_scaling/precision.jl                         60e003ce...
MATCH scripts/static_scaling/check_phase3_flux.jl                 16a9a4c5...
MATCH scripts/remote/leonardo/run-phase3-check.sh                 f52c454d...
MATCH scripts/remote/leonardo/precompile-phase3.sh                ddb4e3b8...
MATCH scripts/remote/leonardo/run-phase3-precision-regression.sh  8bfccb66...
MATCH scripts/remote/leonardo/run-phase3-optimizer-probe.sh       4a70f3ac...
MATCH experiments/static-scaling/probes/probe_optimizer_contract_cross.jl 243e69e8...
MATCH experiments/static-scaling/probes/probe_optimizer_contract_cross.py 7e3a58f9...
MATCH experiments/static-scaling/probes/compare_optimizer_contract.py     56c7e043...
MATCH experiments/static-scaling/probes/probe_optimizer_contract.jl       e51b8ea6...
MATCH experiments/static-scaling/experiment.toml                  3330437c...
MATCH experiments/static-scaling/specification.md                 a542988a...
MATCH experiments/static-scaling/open-decisions.md                631ce1e0...
```

## 3. Compute-node precompile

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/precompile-phase3.sh"
```

Job `60954207`, node `lrdn3362`, completed `2026-10-18T01:13:20Z`.
Output ended with `FLUX_PRECOMPILE_OK`; exit 0.
Log: [`precompile.log`](precompile.log).

## 4. Focused precision regression

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/precision \
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/precision/logs \
    SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
    SOURCE_STATE_SHA256=4a28adb1b031ddacb14aff3bce314d204e693c2d55910cdfe766d5ff33f131fe \
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
    bash scripts/remote/leonardo/run-phase3-precision-regression.sh'"
```

Job `60954547`, node `lrdn2211`, UTC start `2026-10-18T01:14:29Z`.
Result: `PHASE3_PRECISION_REGRESSION PASS`. The FMA case passed
(`CHECK_PRECISION_MATH PASS`, autotuned item `algo=0`,
`CUDNN_FMA_MATH(3)`, `fallback=false`, `rel_l2=4.248559e-7`); the stock
negative control failed on `conv_fixed_fwd/dx/dw` at `~2.93e-4`.
Logs: [`precision-regression.log`](precision-regression.log),
[`precision/logs/fma.log`](precision/logs/fma.log),
[`precision/logs/current.log`](precision/logs/current.log).

## 5. Optimizer-contract and schedule probe

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'ROOT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/optimizer \
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/optimizer/logs \
    SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
    SOURCE_STATE_SHA256=4a28adb1b031ddacb14aff3bce314d204e693c2d55910cdfe766d5ff33f131fe \
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
    bash scripts/remote/leonardo/run-phase3-optimizer-probe.sh'"
```

Job `60955690`, node `lrdn0279`, UTC start `2026-10-18T01:20:02Z`.
Result: `PHASE3_OPTIMIZER_PROBE PASS`; 69 checks, 34/34 negative controls
rejected; driver `PHASE3_OPTIMIZER_PROBE_DRIVER PASS`, exit 0.
Logs: [`optimizer-probe.log`](optimizer-probe.log),
[`optimizer/logs/`](optimizer/logs/), [`optimizer/comparison.json`](optimizer/comparison.json).

## 6. Final acceptance driver

Fresh evidence root. No `ATOL`, `RTOL`, or category override was passed; the
frozen checker defaults supplied the rule.

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/reference \
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/logs \
    SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
    SOURCE_STATE_SHA256=4a28adb1b031ddacb14aff3bce314d204e693c2d55910cdfe766d5ff33f131fe \
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
    STATIC_SCALING_PRECISION_LOG_CANDIDATES=1 \
    bash scripts/remote/leonardo/run-phase3-check.sh'"
```

Job `60956582`, node `lrdn0165`, UTC start `2026-10-18T01:25:21Z`, duration
`5m26.4s`. Driver exit 0. All four summaries PASS:
`PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK PASS`, `PHASE3_REMOTE_CHECK PASS`.
No `Precompiling packages...` line appeared.
Logs: [`remote-check.log`](remote-check.log),
[`local-console.log`](local-console.log).

## 7. Runtime versions

A short follow-up compute-node job wrote the version record:

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 TIME=00:05:00 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'date -u; hostname; printenv SLURM_JOB_ID; nvidia-smi ...; \
    /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --version; \
    grep -A6 deps.Flux .../Manifest.toml; ...; \
    sha256sum .../Manifest.toml .../Project.toml .../test/Project.toml \
      scripts/static_scaling/julia-env/Manifest.toml \
      scripts/static_scaling/julia-env/Project.toml \
      .../requirements-freeze.txt; \
    cat .../requirements-freeze.txt' \
   > /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/logs/runtime-versions.txt"
```

Job `60959396`, node `lrdn0125`, UTC `2026-10-18T01:36:35Z`.
File: [`runtime-versions.txt`](runtime-versions.txt). The Julia manifest
hashes and the 28-line Python freeze are also extracted unchanged into
[`julia-manifest-hashes.txt`](julia-manifest-hashes.txt) and
[`python-freeze.txt`](python-freeze.txt).

Two earlier metadata-collection attempts are retained here for honesty:
job `60959152` (quoting error: unset `$R`) and job `60959247` (header
evaluated on the login node; `pip` is not installed in the uv venv).
Neither touched acceptance evidence.

## 8. Copy evidence home

```bash
rsync -avz --info=progress2 \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/reference/phase3_reference.npz' \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/reference/phase3_reference_manifest.json' \
  artifacts/baselines/static-scaling/2026-10-18-phase3-final/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/logs/remote-check.log' \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/precision/' \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/precision/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/optimizer/' \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/optimizer/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase3-final/logs/runtime-versions.txt' \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/
cp artifacts/logs/leonardo/2026-10-18-phase3-final.log \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/local-console.log
cp artifacts/logs/leonardo/2026-10-18-phase3-final-precompile.log \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/precompile.log
cp artifacts/logs/leonardo/2026-10-18-phase3-final-precision-regression.log \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/precision-regression.log
cp artifacts/logs/leonardo/2026-10-18-phase3-final-optimizer-probe.log \
  artifacts/logs/static-scaling/2026-10-18-phase3-final/optimizer-probe.log
```

Remote SHA-256 values were captured before the SSH publickey rejection
returned; the transcript is [`remote-sha256.txt`](remote-sha256.txt).

## 9. Local independent checks

```bash
python3 scripts/static_scaling/check_phase3_reference.py \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --reference artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference.npz \
  --manifest artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference_manifest.json

python3 - <<'PY'
# exact key-set, dtype/shape, and np.array_equal comparison over all arrays
# against artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/
PY

python3 - <<'PY'
# reject missing/empty/null/unknown source identities in the final manifest
PY

python3 - <<'PY'
# SHA-256 comparison of remote-sha256.txt against the local copies
PY

sha256sum -c SHA256SUMS
```

Results: `PHASE3_REFERENCE_CHECK PASS arrays=337`; 337/337 exact-equal to the
preserved attempt-3 reference, maximum absolute difference `0.0`;
`PROVENANCE_CHECK PASS fields=4 bad=[]`;
`SHA256_COMPARISON PASS files=15 mismatches=0`; `SHA256SUMS` check OK
(all evidence files except the root `SHA256SUMS` and its check log; the
nested optimizer `SHA256SUMS` is included).
Outputs: [`local-reference-check.log`](local-reference-check.log),
[`reference-comparison.txt`](reference-comparison.txt),
[`provenance-check.log`](provenance-check.log),
[`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt),
[`sha256sum-check.log`](sha256sum-check.log).

## 10. Step 1-5 evidence links

| Step | Evidence |
|---|---|
| Step 1 preserved attempt-3 | `artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/`, `artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/` |
| Step 2 precision experiment | `artifacts/logs/static-scaling/2026-10-17-phase3-step2-precision/` |
| Step 3/3b/3c numerics and acceptance | `artifacts/logs/static-scaling/2026-10-17-phase3-numerics/`, `.../2026-10-17-phase3-step3/`, `.../2026-10-17-phase3-step3b/`, `.../2026-10-17-phase3-step3c/` |
| Step 4 optimizer probe | `artifacts/logs/static-scaling/2026-10-17-phase3-step4/` |
| Step 5 hardened Phase 2 re-check | `artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/` |
| Step 6 (this run) | `artifacts/logs/static-scaling/2026-10-18-phase3-final/` |

## 11. Failures during this campaign

- Full rsync killed by the local 900 s timeout (no source effect); retried
  with historical reference archives excluded.
- Metadata job `60959152`: quoting error, no file written.
- Metadata job `60959247`: header evaluated on the login node; no `pip`
  module in the uv venv; superseded by job `60959396`.
- Leonardo SSH publickey rejection returned after the final evidence copy;
  all remote artifacts and remote SHA-256 values were already retained
  locally. This is an access/operations issue, not a numerical result.
