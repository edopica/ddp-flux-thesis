#!/usr/bin/env bash
# Rootless-podman runner for the PR #2694 distributed CI pipeline with the
# GitHub runner topology (4 vCPU over 2 physical cores) emulated.
#
# Why this exists: rootless podman cannot use `--cpuset-cpus` (the cpuset
# cgroup controller is not delegated to the user session), `taskset` outside
# podman does not reach the container (rootless conmon is forked by the
# systemd user manager), and OpenMPI counts slots from the hwloc machine
# topology, not from the process affinity mask. This wrapper therefore injects
# an hwloc topology XML restricted to CPUs 0-3 (2 physical cores x 2 SMT
# threads on this host) via OMPI_MCA_hwloc_base_topo_file and pins the
# workload with taskset inside the container.
#
# Usage:
#   scripts/podman/run-ci-parity.sh <command> [args...]
#
# Examples:
#   scripts/podman/run-ci-parity.sh bash -lc 'nproc; taskset -cp $$'
#   scripts/podman/run-ci-parity.sh env JULIA_MPI_TEST_NPROCS=4 \
#     FLUX_TEST_DISTRIBUTED_TIMEOUT=600 julia --project=test test/ext_distributed/runtests.jl
#
# Environment overrides:
#   IMAGE=ddp-ci-repro:1.12.7   container image
#   DEPOT=ddp-ci-depot          podman volume mounted at /home/runner/.julia
#   REPO=/tmp/opencode/docker_ci/repo-salvage
#                               Flux checkout mounted at /home/runner/Flux.jl
#   CPUS=0-3                    CPU range to restrict/emulate
#   REGEN=1                     regenerate the topology XML even if present
#   NATIVE_CPUSET=1             use `podman --cpuset-cpus` instead of the hwloc
#                               emulation (only after cpuset delegation; see
#                               temp/docker_ci_fix/RESULTS-podman.md Plan B)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

IMAGE=${IMAGE:-ddp-ci-repro:1.12.7}
DEPOT=${DEPOT:-ddp-ci-depot}
REPO=${REPO:-/tmp/opencode/docker_ci/repo-salvage}
CPUS=${CPUS:-0-3}
TOPO="$SCRIPT_DIR/topo-2c4t.xml"

if (( $# == 0 )); then
  echo "usage: $(basename "$0") <command> [args...]" >&2
  exit 2
fi

if ! podman image exists "$IMAGE"; then
  echo "error: podman image '$IMAGE' not found (load it or set IMAGE=)" >&2
  exit 1
fi

if [[ ! -d "$REPO" ]]; then
  echo "error: repo '$REPO' not found; stage it with:" >&2
  echo "  rsync -a --delete --exclude .git <worktree>/ $REPO/" >&2
  exit 1
fi

COMMON_ARGS=(
  --rm --userns=keep-id
  -v "$DEPOT:/home/runner/.julia"
  -v "$REPO:/home/runner/Flux.jl"
  -w /home/runner/Flux.jl
)

if [[ "${NATIVE_CPUSET:-0}" == "1" ]]; then
  echo "podman-ci-parity: native cpuset CPUs=$CPUS image=$IMAGE repo=$REPO" >&2
  exec podman run "${COMMON_ARGS[@]}" --cpuset-cpus="$CPUS" "$IMAGE" "$@"
fi

if [[ ! -f "$TOPO" || "${REGEN:-0}" == "1" ]]; then
  FIRST=${CPUS%-*}
  LAST=${CPUS#*-}
  echo "podman-ci-parity: generating $TOPO from CPUs $CPUS" >&2
  gen_out=$(podman run --rm --userns=keep-id -v "$SCRIPT_DIR:/mnt/topo" "$IMAGE" bash -lc \
    "gcc -O2 -o /tmp/topo-gen /mnt/topo/topo-gen.c -lhwloc && \
     /tmp/topo-gen /mnt/topo/topo-2c4t.xml $FIRST $LAST")
  echo "podman-ci-parity: $gen_out" >&2
  if [[ "$gen_out" != *"cores=2 pus=4"* ]]; then
    echo "warning: CPUs $CPUS are not 2 physical cores x 2 threads on this host;" >&2
    echo "         the emulated topology may not match GitHub runners." >&2
  fi
fi

echo "podman-ci-parity: emulated 2-core/4-thread topology (CPUs $CPUS), image=$IMAGE" >&2

exec podman run "${COMMON_ARGS[@]}" \
  -v "$TOPO:/mnt/topo-2c4t.xml:ro" \
  -e OMPI_MCA_hwloc_base_topo_file=/mnt/topo-2c4t.xml \
  "$IMAGE" taskset -c "$CPUS" "$@"
