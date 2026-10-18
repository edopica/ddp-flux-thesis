# Commands — static-scaling Phase 3 Step 4 optimizer probe

All commands were run from the thesis repository root unless noted.
Date: 2026-10-17 (UTC run start 2026-10-18T00:33:04Z).

## 1. Local parse and syntax checks

```bash
bash -n scripts/remote/leonardo/run-phase3-optimizer-probe.sh

python3 -m py_compile \
  experiments/static-scaling/probes/probe_optimizer_contract_cross.py \
  experiments/static-scaling/probes/compare_optimizer_contract.py

julia --startup-file=no -e \
  'for f in ["experiments/static-scaling/probes/probe_optimizer_contract_cross.jl"]
     Meta.parseall(read(f, String)); println("PARSE_OK ", f)
   end'
```

Result: `BASH_OK`, `PY_OK`, `PARSE_OK`.

## 2. Local Flux probe (CPU logic check)

```bash
JULIA_LOAD_PATH="$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration/test:@stdlib" \
JULIA_PKG_PRECOMPILE_AUTO=0 \
julia --startup-file=no \
  experiments/static-scaling/probes/probe_optimizer_contract_cross.jl \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --out /tmp/opencode/probe_flux.json
```

Result: the legacy R3/R4 checks, the 21/41 group assertions, and all four
controlled leaves printed; the probe wrote its JSON/NPZ. The local output was
used only to check the code and comparator; the evidence below comes from the
Leonardo run.

The comparator was also exercised locally with a duplicated JSON (framework
label changed to `pytorch`) to check its logic. That duplicate is not
evidence and is not retained.

## 3. Source identity

```bash
COMMIT=$(git rev-parse HEAD)
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
FLUX_COMMIT=$(git -C /home/kurapica/Projects/ddp_flux/flux-integration rev-parse HEAD)
FLUX_STATE=$( { git -C /home/kurapica/Projects/ddp_flux/flux-integration diff HEAD; \
  git -C /home/kurapica/Projects/ddp_flux/flux-integration ls-files --others --exclude-standard \
  | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

Result:

```text
COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3
STATE=39b1051f27077581b30ff49ae09aada8d58086f75d446a8a278718c992106164
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

New source hashes:

```text
243e69e827854141f6fbec05b6c21d99a94d8c6e6f29f8cc49bf86480ec273c9  experiments/static-scaling/probes/probe_optimizer_contract_cross.jl
7e3a58f93224e8adbe3c4cd0bbce1e0387c76dd61fdff5eb754b999e9a9844f3  experiments/static-scaling/probes/probe_optimizer_contract_cross.py
56c7e043c6b8d6c28edc6ffc5261ec0dd32bf929c29d96e2bd62dca5b5d2e889  experiments/static-scaling/probes/compare_optimizer_contract.py
4a70f3acc0205c975e242370ef36895d76fbaf0f5d20bb81ce2ac0b0078c4e6d  scripts/remote/leonardo/run-phase3-optimizer-probe.sh
```

## 4. Sync to Leonardo (explicit exclusions, not `sync_code.sh`)

The Step 3c exclusion list was repeated, with the newer local-only Step 3c
evidence directories also excluded so that local reference copies were not
uploaded:

```bash
rsync -avz --delete --exclude-from='.rsyncignore' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step3/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step3/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step3b/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-numerics/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-numerics/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step3c/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step3c/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step4/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step4/' \
  . leonardo:projects/ddp-flux-thesis/
```

Remote hashes of the four new files matched the local values above. Python
`__pycache__` directories created by the local parse check were removed from
the remote tree before execution; they are not part of the run.

## 5. Compute-node run (CPU-only)

```bash
NTASKS=1 CPUS_PER_TASK=8 GRES=none TIME=00:30:00 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'ROOT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step4 \
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step4/logs \
    SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
    SOURCE_STATE_SHA256=39b1051f27077581b30ff49ae09aada8d58086f75d446a8a278718c992106164 \
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
    bash scripts/remote/leonardo/run-phase3-optimizer-probe.sh'"
```

Job `60948431`, node `lrdn0171`, `boost_qos_dbg`, 1 node, 1 task, 8 CPUs,
`--gres=none`, elapsed about 2 minutes. The console log is
`2026-10-17-phase3-step4.log`.

Results:

```text
PROBE_FLUX training_path leaves=62 decay=21 nodecay=41 rule_meta_ok=true
PROBE_TORCH training_path leaves=62 decay=21 nodecay=41 rule_meta_ok=True
SUMMARY checks=69 negative_controls=34 rejected=34 failures=0
PHASE3_OPTIMIZER_PROBE PASS
PHASE3_OPTIMIZER_PROBE_DRIVER PASS
```

No `Precompiling packages...` line appeared.

## 6. Copy evidence home

```bash
mkdir -p artifacts/logs/static-scaling/2026-10-17-phase3-step4
rsync -avz \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step4/' \
  artifacts/logs/static-scaling/2026-10-17-phase3-step4/
cp artifacts/logs/leonardo/2026-10-17-phase3-step4.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-step4/
```

## 7. Hash verification

```bash
(cd artifacts/logs/static-scaling/2026-10-17-phase3-step4 && \
  sha256sum comparison.json probe_flux.json probe_flux_arrays.npz \
    probe_torch.json probe_torch_arrays.npz logs/*.log)
```

The values match the remote `logs/SHA256SUMS`; see `PROVENANCE.md` and
`SHA256SUMS`.

## 8. Outcome

`PHASE3_OPTIMIZER_PROBE PASS`; 69 checks, 34/34 negative controls rejected;
driver exit 0. No training code, tolerance, acceptance checker, schedule,
model math, or Flux source changed. Steps 5–6 were not run.
