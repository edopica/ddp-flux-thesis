#!/usr/bin/env bash
set -euo pipefail

FLUX_REPO_PATH="${1:-${FLUX_REPO_PATH:-../Flux.jl}}"
OUT_DIR="artifacts/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/audit_flux_distributed_${STAMP}.txt"

mkdir -p "${OUT_DIR}"

if ! git -C "${FLUX_REPO_PATH}" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "ERROR: Not a git repository: ${FLUX_REPO_PATH}" >&2
  exit 1
fi

{
  echo "# Flux distributed audit grep"
  echo "Date: $(date)"
  echo "Flux repo: ${FLUX_REPO_PATH}"
  echo "Flux commit: $(git -C "${FLUX_REPO_PATH}" rev-parse HEAD)"
  echo "Flux branch: $(git -C "${FLUX_REPO_PATH}" branch --show-current)"
  echo

  echo "## Candidate distributed files"
  for path in \
    "src/distributed" \
    "ext/FluxMPIExt" \
    "ext/FluxMPINCCLExt" \
    "test/ext_distributed" \
    "docs"
  do
    if [ -e "${FLUX_REPO_PATH}/${path}" ]; then
      echo
      echo "### ${path}"
      find "${FLUX_REPO_PATH}/${path}" -maxdepth 3 -type f | sort
    else
      echo
      echo "### ${path}"
      echo "Missing"
    fi
  done

  echo
  echo "## Grep results"
  git -C "${FLUX_REPO_PATH}" grep -n \
    -e "DistributedUtils" \
    -e "MPIBackend" \
    -e "NCCLBackend" \
    -e "DistributedOptimizer" \
    -e "DistributedDataContainer" \
    -e "synchronize!!" \
    -e "allreduce" \
    -e "bcast" \
    -e "reduce" \
    -- src ext test docs 2>/dev/null || true
} | tee "${OUT_FILE}"

echo
echo "Wrote audit log: ${OUT_FILE}"
