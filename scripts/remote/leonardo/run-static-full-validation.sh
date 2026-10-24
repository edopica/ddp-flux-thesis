#!/usr/bin/env bash
# =============================================================================
# Leonardo COMPUTE-NODE validation driver for the full-dataset loader.
#
# ALLOCATION (from the thesis root on a login node; do not run on login):
#   bash scripts/remote/run.sh leonardo \
#     "srun -n1 -c32 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-static-full-validation.sh"
#
# The driver precompiles both Flux worktrees on the compute node, then:
#   1. builds the Python full-dataset reference (epoch 0, batch 512),
#   2. runs the Python and Julia loader checks at W=1,2,4,
#   3. runs the unchanged fixture timing path once per framework,
#   4. runs exactly one 1-update dry run at batch 512, W=1 for
#      Flux current, Flux historical, and PyTorch (20 discarded warmup updates).
#
# It does not launch the full campaign and does not fall back to batch 256.
# Set RUN_W4=1 to add one optional W=4 dry run for Flux current.
#
# Outputs are written only under a fresh $WORK evidence root by this single
# driver task (rank 0 of the allocation).
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
FLUX_CURRENT=${FLUX_CURRENT:-$HOME/projects/flux-integration}
FLUX_HISTORICAL=${FLUX_HISTORICAL:-$HOME/projects/flux-static-baseline}
PREP=${PREP:-$WORK/epicazio/static-scaling-prepared}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
PYTHON_BIN=${PYTHON_BIN:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
SCRIPTS=$THESIS/scripts/static_scaling
PRECOMPILE=$THESIS/scripts/remote/leonardo/precompile-phase3.sh
TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-32}
JOB_ID=${SLURM_JOB_ID:-manual}
OUT=${OUT:-$WORK/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-$JOB_ID}
LOG_DIR=$OUT/logs
RUN_W4=${RUN_W4:-0}

CURRENT_COMMIT=${CURRENT_COMMIT:-f7b4520c4bd307c67bbc5ac92a1c68493eccff6e}
CURRENT_STATE=${CURRENT_STATE:-e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855}
HISTORICAL_COMMIT=${HISTORICAL_COMMIT:-8add0e42b7d9388ff365335152126e5b1e9de657}
HISTORICAL_STATE=${HISTORICAL_STATE:-4af8409c2a2ed1c661e8ae7742877f1ee9789d00dc7e404365ecefd98ab0d34c}
THESIS_COMMIT=${THESIS_COMMIT:-660e363e6de5b6e4b398e542d79b6ce1adb381da}

host=$(hostname)
case "$host" in
  login*)
    echo "ERROR: run this driver under srun on a compute node (host=$host)" >&2
    exit 1
    ;;
esac
if [ "${SLURM_NTASKS:-1}" != "1" ] || [ "${SLURM_PROCID:-0}" != "0" ]; then
  echo "ERROR: driver must run as one task; got SLURM_NTASKS=${SLURM_NTASKS:-unset} SLURM_PROCID=${SLURM_PROCID:-unset}" >&2
  exit 1
fi
for path in "$JULIA_BIN" "$PYTHON_BIN" "$PREP/train_images.npy" "$FIX/data_fixture.npz"; do
  [ -e "$path" ] || { echo "ERROR: missing required path: $path" >&2; exit 1; }
done
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null || true)" ]; then
  echo "ERROR: output root exists and is not empty: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG_DIR" "$OUT/reference" "$OUT/fixture-smoke" "$OUT/dryrun"

export WORK THESIS FLUX_CURRENT FLUX_HISTORICAL PREP FIX OUT LOG_DIR
export JULIA_BIN JULIA_DEPOT_PATH PYTHON_BIN SCRIPTS PRECOMPILE
export JULIA_PKG_PRECOMPILE_AUTO=0

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
sha256_file() { sha256sum "$1" | cut -d' ' -f1; }
source_tree_sha256() {
  ( cd "$1" && find . -type f -not -path '*/__pycache__/*' -print0 \
      | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1 )
}
flux_source_sha256() {
  ( cd "$1" && find src -type f -print0 | sort -z \
      | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1 )
}

SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-$(source_tree_sha256 "$THESIS")}
CURRENT_SOURCE_SHA256=$(flux_source_sha256 "$FLUX_CURRENT")
HISTORICAL_SOURCE_SHA256=$(flux_source_sha256 "$FLUX_HISTORICAL")

julia_current() {
  ( unset JULIA_PROJECT
    export JULIA_DEPOT_PATH JULIA_PKG_PRECOMPILE_AUTO=0
    export JULIA_LOAD_PATH="$SCRIPTS/julia-env:$FLUX_CURRENT/test:@stdlib"
    export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$JULIA_BIN" --startup-file=no "$@" )
}
julia_historical() {
  ( unset JULIA_PROJECT
    export JULIA_DEPOT_PATH JULIA_PKG_PRECOMPILE_AUTO=0
    export JULIA_LOAD_PATH="$SCRIPTS/julia-env:$FLUX_HISTORICAL/test:@stdlib"
    export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$JULIA_BIN" --startup-file=no "$@" )
}

run_logged() {
  local name=$1
  shift
  echo "STAGE $name"
  echo "RUN $*"
  if ! "$@" 2>&1 | tee "$LOG_DIR/$name.log"; then
    echo "FULL_VALIDATION FAIL step=$name" >&2
    exit 1
  fi
  if grep -q "Precompiling packages" "$LOG_DIR/$name.log"; then
    echo "FULL_VALIDATION FAIL step=$name: unexpected precompilation" >&2
    exit 1
  fi
}

precompile_variant() {
  local name=$1
  local dir=$2
  local log="$LOG_DIR/precompile-$name.log"
  echo "STAGE precompile-$name"
  if ! ( unset JULIA_PROJECT JULIA_LOAD_PATH
         export JULIA_DEPOT_PATH JULIA_PKG_PRECOMPILE_AUTO=1
         FLUX_DIR="$dir" bash "$PRECOMPILE" ) 2>&1 | tee "$log"; then
    echo "FULL_VALIDATION FAIL step=precompile-$name" >&2
    exit 1
  fi
  grep -q "FLUX_PRECOMPILE_OK" "$log" || {
    echo "FULL_VALIDATION FAIL step=precompile-$name: missing FLUX_PRECOMPILE_OK" >&2
    exit 1
  }
  echo "PRECOMPILE_$name OK"
}

echo "FULL_VALIDATION START utc=$(utc_now) host=$host job=$JOB_ID"
echo "OUT=$OUT"
echo "PREP=$PREP"
echo "CURRENT=$FLUX_CURRENT ($CURRENT_COMMIT/$CURRENT_STATE)"
echo "HISTORICAL=$FLUX_HISTORICAL ($HISTORICAL_COMMIT/$HISTORICAL_STATE)"
echo "THESIS_STATE=$SOURCE_STATE_SHA256"

precompile_variant current "$FLUX_CURRENT"
precompile_variant historical "$FLUX_HISTORICAL"

run_logged reference "$PYTHON_BIN" "$SCRIPTS/full_dataset_reference.py" \
  --dataset-dir "$PREP" --out-dir "$OUT/reference" \
  --epoch 0 --batch-size 512 --worlds 1,2,4 --verify-large-hashes

run_logged python-check "$PYTHON_BIN" "$SCRIPTS/check_full_dataset.py" \
  --dataset-dir "$PREP" --reference-dir "$OUT/reference" \
  --out "$OUT/reference/python-check.json" \
  --epoch 0 --batch-size 512 --worlds 1,2,4 --strict-contract true

run_logged julia-check-current julia_current "$SCRIPTS/check_full_dataset.jl" \
  --dataset-dir "$PREP" --reference-dir "$OUT/reference" \
  --out "$OUT/check-julia-current.json" \
  --batch-size 512 --epoch 0 --worlds 1,2,4 \
  --label flux-current --verify-large-hashes true --strict-contract true

run_logged python-check-current "$PYTHON_BIN" "$SCRIPTS/check_full_dataset.py" \
  --dataset-dir "$PREP" --reference-dir "$OUT/reference" \
  --out "$OUT/reference/python-check-current-combined.json" \
  --epoch 0 --batch-size 512 --worlds 1,2,4 --strict-contract true \
  --julia-json "$OUT/check-julia-current.json"

run_logged julia-check-historical julia_historical "$SCRIPTS/check_full_dataset.jl" \
  --dataset-dir "$PREP" --reference-dir "$OUT/reference" \
  --out "$OUT/check-julia-historical.json" \
  --batch-size 512 --epoch 0 --worlds 1,2,4 \
  --label flux-historical --verify-large-hashes false --strict-contract true

run_logged python-check-historical "$PYTHON_BIN" "$SCRIPTS/check_full_dataset.py" \
  --dataset-dir "$PREP" --reference-dir "$OUT/reference" \
  --out "$OUT/reference/python-check-historical-combined.json" \
  --epoch 0 --batch-size 512 --worlds 1,2,4 --strict-contract true \
  --julia-json "$OUT/check-julia-historical.json"

run_logged fixture-flux-current julia_current "$SCRIPTS/timing_flux.jl" \
  --fixture-dir "$FIX" --world-size 1 --updates 3 --warmup 2 \
  --out "$OUT/fixture-smoke/flux-current-w1.json" \
  --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256" \
  --flux-commit "$CURRENT_COMMIT" --flux-state-sha256 "$CURRENT_STATE"

run_logged fixture-pytorch "$PYTHON_BIN" "$SCRIPTS/timing_torch.py" \
  --fixture-dir "$FIX" --world-size 1 --updates 3 --warmup 2 \
  --out "$OUT/fixture-smoke/pytorch-w1.json" \
  --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256"

"$PYTHON_BIN" - "$OUT/fixture-smoke" <<'PY'
import json
import os
import sys

root = sys.argv[1]
expected = {
    "flux-current-w1.json": ("flux", "rank_seconds"),
    "pytorch-w1.json": ("pytorch", "ranks"),
}
for name, (framework, rank_key) in expected.items():
    with open(os.path.join(root, name), "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    assert doc["schema"] == "static-timing-pilot-v1", (name, doc.get("schema"))
    assert doc["framework"] == framework, (name, doc.get("framework"))
    assert doc["global_batch"] == 12, (name, doc.get("global_batch"))
    assert doc["updates"] == 3, (name, doc.get("updates"))
    assert doc["warmup"] == 2, (name, doc.get("warmup"))
    assert doc["workload"] == "alternating-two-fixture-batches", (name, doc.get("workload"))
    assert doc["timing"] == "training-only", (name, doc.get("timing"))
    assert doc["pass_checks"] is True, (name, doc.get("pass_checks"))
    assert rank_key in doc, (name, rank_key)
    print("FIXTURE_TIMING_JSON PASS", name, "schema", doc["schema"])
print("FIXTURE_TIMING_PATH PASS")
PY

run_logged dryrun-flux-current julia_current "$SCRIPTS/timing_flux.jl" \
  --dataset-dir "$PREP" --world-size 1 --updates 1 --warmup 20 \
  --out "$OUT/dryrun/flux-current-w1.json" \
  --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256" \
  --flux-commit "$CURRENT_COMMIT" --flux-state-sha256 "$CURRENT_STATE"

run_logged dryrun-flux-historical julia_historical "$SCRIPTS/timing_flux.jl" \
  --dataset-dir "$PREP" --world-size 1 --updates 1 --warmup 20 \
  --out "$OUT/dryrun/flux-historical-w1.json" \
  --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256" \
  --flux-commit "$HISTORICAL_COMMIT" --flux-state-sha256 "$HISTORICAL_STATE"

run_logged dryrun-pytorch "$PYTHON_BIN" "$SCRIPTS/timing_torch.py" \
  --dataset-dir "$PREP" --world-size 1 --updates 1 --warmup 20 \
  --out "$OUT/dryrun/pytorch-w1.json" \
  --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256" \
  --flux-commit "" --flux-state-sha256 ""

if [ "$RUN_W4" = "1" ]; then
  echo "STAGE dryrun-flux-current-w4"
  if ! srun --overlap --mpi=pmi2 --kill-on-bad-exit=1 -n4 -c$((TOTAL_CPUS / 4)) \
      bash -c "unset JULIA_PROJECT; export JULIA_DEPOT_PATH='$JULIA_DEPOT_PATH' JULIA_LOAD_PATH='$SCRIPTS/julia-env:$FLUX_CURRENT/test:@stdlib' JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0; exec '$JULIA_BIN' --startup-file=no '$SCRIPTS/timing_flux.jl' --dataset-dir '$PREP' --world-size 4 --updates 1 --warmup 20 --distributed-api guarded --out '$OUT/dryrun/flux-current-w4.json' --source-commit '$THESIS_COMMIT' --source-state-sha256 '$SOURCE_STATE_SHA256' --flux-commit '$CURRENT_COMMIT' --flux-state-sha256 '$CURRENT_STATE'" \
      2>&1 | tee "$LOG_DIR/dryrun-flux-current-w4.log"; then
    echo "FULL_VALIDATION FAIL step=dryrun-flux-current-w4" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - "$OUT/dryrun" "$RUN_W4" <<'PY'
import json
import math
import os
import sys

root = sys.argv[1]
run_w4 = sys.argv[2] == "1"
expected = {
    "flux-current-w1.json": "flux",
    "flux-historical-w1.json": "flux",
    "pytorch-w1.json": "pytorch",
}
if run_w4:
    expected["flux-current-w4.json"] = "flux"
for name, framework in expected.items():
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        raise AssertionError("missing dry-run record: %s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    world = doc.get("world_size")
    assert doc["schema"] == "static-timing-full-dataset-v1", (name, doc.get("schema"))
    assert doc["framework"] == framework, (name, doc.get("framework"))
    assert doc["epoch"] == 0, (name, doc.get("epoch"))
    assert doc["batch_size"] == 512, (name, doc.get("batch_size"))
    assert doc["global_batch"] == 512, (name, doc.get("global_batch"))
    assert doc["updates"] == 1, (name, doc.get("updates"))
    assert doc["warmup"] == 20, (name, doc.get("warmup"))
    assert doc["updates_per_epoch"] == 195, (name, doc.get("updates_per_epoch"))
    assert doc["n_eff"] == 99840, (name, doc.get("n_eff"))
    assert doc["tail_size"] == 160, (name, doc.get("tail_size"))
    assert doc["loss_finite"] is True, name
    assert math.isfinite(float(doc["final_loss"])), (name, doc["final_loss"])
    assert doc["bundle_hash"] == "a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23", (name, doc.get("bundle_hash"))
    assert len(doc["sample_order_sha256"]) == 64, name
    assert len(doc["transform_spec_sha256"]) == 64, name
    assert doc["dataset_hashes"], name
    assert doc["timing"] == "full-dataset-end-to-end", name
    assert doc["pass_checks"] is True, name
    if framework == "flux":
        assert len(doc["rank_seconds"]) == world, (name, doc["rank_seconds"])
        memory = doc["memory"]
        assert memory["device_total_bytes"] > 0, name
        assert memory["device_used_bytes"] >= 0, name
        used = memory["device_used_bytes"] / memory["device_total_bytes"]
        print("FULL_DATASET_DRYRUN PASS", name, "world", world,
              "loss", doc["final_loss"], "device_used_fraction", "%.6f" % used)
    else:
        assert len(doc["ranks"]) == world, (name, doc["ranks"])
        memory = doc["ranks"][0]["memory"]
        assert memory["mem_get_info_total"] > 0, name
        used = memory["max_memory_allocated"] / memory["mem_get_info_total"]
        print("FULL_DATASET_DRYRUN PASS", name, "world", world,
              "loss", doc["final_loss"], "max_memory_allocated_fraction", "%.6f" % used)
PY

"$PYTHON_BIN" - "$OUT" "$JOB_ID" "$host" "$THESIS_COMMIT" "$SOURCE_STATE_SHA256" \
  "$CURRENT_COMMIT" "$CURRENT_STATE" "$CURRENT_SOURCE_SHA256" \
  "$HISTORICAL_COMMIT" "$HISTORICAL_STATE" "$HISTORICAL_SOURCE_SHA256" <<'PY'
import json
import os
import sys

out, job, host, thesis_commit, thesis_state, current_commit, current_state, current_source, \
    historical_commit, historical_state, historical_source = sys.argv[1:]
with open(os.path.join(out, "reference", "full_dataset_reference.json"), "r", encoding="utf-8") as handle:
    reference = json.load(handle)
summary = {
    "schema": "static-full-dataset-validation-v1",
    "job": job,
    "node": host,
    "dataset_dir": reference["dataset_dir"],
    "dataset_files": reference["dataset_files"],
    "sample_order_sha256": reference["sample_order_sha256"],
    "update1_order_sha256": reference["update1_order_sha256"],
    "transform_spec_sha256": reference["transform_spec_sha256"],
    "transform_sample_sha256": reference["transform_sample_sha256"],
    "bundle_hash": reference["bundle_hash"],
    "bundle_file_sha256": reference["bundle_file_sha256"],
    "n_eff": reference["n_eff"],
    "tail_size": reference["tail_size"],
    "updates_per_epoch": reference["updates_per_epoch"],
    "source": {
        "thesis_commit": thesis_commit,
        "thesis_state_sha256": thesis_state,
        "thesis_state_recipe": "sha256 of sorted (sha256sum) over the synced thesis tree, excluding __pycache__",
        "flux_current_commit": current_commit,
        "flux_current_state_sha256": current_state,
        "flux_current_source_tree_sha256": current_source,
        "flux_historical_commit": historical_commit,
        "flux_historical_state_sha256": historical_state,
        "flux_historical_source_tree_sha256": historical_source,
    },
}
with open(os.path.join(out, "SUMMARY.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("FULL_VALIDATION_SUMMARY_WRITTEN", os.path.join(out, "SUMMARY.json"))
PY

( cd "$OUT" && find . -type f -not -name SHA256SUMS -print0 \
    | sort -z | xargs -0 -r sha256sum > SHA256SUMS )

echo "FULL_VALIDATION PASS utc=$(utc_now) host=$host job=$JOB_ID OUT=$OUT"
