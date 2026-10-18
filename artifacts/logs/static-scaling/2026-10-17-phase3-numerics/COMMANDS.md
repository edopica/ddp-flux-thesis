# Commands — Phase 3 numerical calibration (2026-10-17)

All commands were run from the thesis repository root unless noted.
`sync_code.sh` was not used. No Flux.jl source changed. No precompile was
needed: only Python and shell files were transferred, and the remote Julia
checker, model, and precision files already had the run-time hashes from
Step 3b (verified by `sha256sum` before the runs). The checker logs contain no
`Precompiling packages...` line.

## 1. New read-only comparison and driver files

New files (no modifications to existing checkers or tolerances):

```text
scripts/static_scaling/compare_reference_arrays.py
scripts/remote/leonardo/run-phase3-numerics-reference.sh
scripts/remote/leonardo/run-phase3-numerics-flux-cpu.sh
```

Local syntax checks:

```bash
bash -n scripts/remote/leonardo/run-phase3-numerics-reference.sh
bash -n scripts/remote/leonardo/run-phase3-numerics-flux-cpu.sh
python3 -m py_compile scripts/static_scaling/compare_reference_arrays.py
```

Local self-tests of the comparison script:

```bash
python3 scripts/static_scaling/compare_reference_arrays.py \
  --ref artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz \
  --cand artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/phase3_reference.npz \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --no-leaves
# COMPARE_RESULT PASS, 0 violations, max_abs=0
```

## 2. Transfer the new files (no full sync)

```bash
rsync -avz scripts/static_scaling/compare_reference_arrays.py \
  leonardo:projects/ddp-flux-thesis/scripts/static_scaling/
rsync -avz \
  scripts/remote/leonardo/run-phase3-numerics-reference.sh \
  scripts/remote/leonardo/run-phase3-numerics-flux-cpu.sh \
  leonardo:projects/ddp-flux-thesis/scripts/remote/leonardo/
```

Remote SHA-256 matched local:

```text
27edd256f917cc8101c3d881d16949695c71f4fac7265eaccb1a8498c44885e5  compare_reference_arrays.py
1696a698022c9b63fb5b9b0dd6cf91e139d38160f39361a13974d55071b15c3a  run-phase3-numerics-reference.sh
3528c286cdc82c6515d0bf89889bf913ca7973b1f24369946eb45c486e4eedce  run-phase3-numerics-flux-cpu.sh
```

## 3. Generate the PyTorch CPU reference (compute node)

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=4 TIME=00:20:00 \
  scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c4 --chdir=\$HOME/projects/ddp-flux-thesis \
   bash scripts/remote/leonardo/run-phase3-numerics-reference.sh"
```

Job `60912525`, node `lrdn3450`. The driver ran:

```text
python scripts/static_scaling/make_phase3_reference.py \
  --fixture-dir <THESIS>/artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle <FIX>/initial_bundle.npz \
  --out-dir $WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics/torch_cpu_reference \
  --device cpu --force \
  --source-commit 3088f61020137912af32d9c969e87ea9ae2a71c3 \
  --source-state-sha256 82623bbb516a981d6c795792d1433521d25a8ee3bbcc44a2b66f6f4631a452b9 \
  --flux-commit f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
  --flux-state-sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

It then ran `check_phase3_reference.py` on the new archive and
`compare_reference_arrays.py` against the preserved GPU reference. Results:
`PHASE3_REFERENCE PASS checks=32 failed=0`,
`PHASE3_REFERENCE_CHECK PASS arrays=337`, `COMPARE_RESULT FAIL` with
143,672 violations (the diagnostic result, not a driver failure).
Console log: `2026-10-17-phase3-numerics-reference.log`.

## 4. Flux CPU checks (compute node)

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 TIME=00:30:00 \
  scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
   bash scripts/remote/leonardo/run-phase3-numerics-flux-cpu.sh"
```

Job `60912664`, node `lrdn3450`. The driver ran the unchanged checker twice:

```text
bash scripts/remote/leonardo/run-julia-phase3.sh \
  scripts/static_scaling/check_phase3_flux.jl \
  --fixture-dir <FIX> --bundle <FIX>/initial_bundle.npz \
  --reference <preserved GPU npz> \
  --manifest <preserved GPU manifest> \
  --device cpu --atol 1e-5 --rtol 1e-4
```

and the same command with the new CPU reference and manifest. Both runs exited
1 on the numerical categories; the job itself completed and printed
`FLUX_CPU_DONE`. Console log: `2026-10-17-phase3-numerics-flux-cpu.log`.

## 5. Bring evidence home

```bash
mkdir -p artifacts/baselines/static-scaling/2026-10-17-phase3-numerics \
  artifacts/logs/static-scaling/2026-10-17-phase3-numerics
rsync -avz \
  leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics/torch_cpu_reference/phase3_reference.npz \
  leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics/torch_cpu_reference/phase3_reference_manifest.json \
  artifacts/baselines/static-scaling/2026-10-17-phase3-numerics/
rsync -avz \
  leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics/logs/ \
  artifacts/logs/static-scaling/2026-10-17-phase3-numerics/
cp artifacts/logs/leonardo/2026-10-17-phase3-numerics-reference.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-numerics/
cp artifacts/logs/leonardo/2026-10-17-phase3-numerics-flux-cpu.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-numerics/
```

## 6. Local reproduction and rule sanity check

```bash
python3 scripts/static_scaling/compare_reference_arrays.py \
  --ref artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz \
  --cand artifacts/baselines/static-scaling/2026-10-17-phase3-numerics/phase3_reference.npz \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --atol 1e-5 --rtol 1e-4 \
  --json-out artifacts/logs/static-scaling/2026-10-17-phase3-numerics/torch-cpu-vs-gpu-reference.local.json \
  > artifacts/logs/static-scaling/2026-10-17-phase3-numerics/torch-cpu-vs-gpu-reference.local.log
```

The six-decimal `COMPARE_*` lines are identical to the remote run. The JSON
differs only in paths and last-ULP floating arithmetic (remote numpy 2.4.6,
local numpy 2.5.3).

```bash
python3 scripts/static_scaling/compare_reference_arrays.py \
  --ref artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz \
  --cand artifacts/baselines/static-scaling/2026-10-17-phase3-numerics/phase3_reference.npz \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --atol 2e-3 --rtol 1e-4 --no-leaves \
  > artifacts/logs/static-scaling/2026-10-17-phase3-numerics/torch-cpu-vs-gpu-reference.atol2e-3.log
# COMPARE_RESULT PASS, 0 violations
```
