#!/usr/bin/env bash
# Run from the repository root. Artifact checks only, no model workload.
set -euo pipefail
BASE=artifacts/baselines/static-scaling
OUT=artifacts/logs/static-scaling/2026-10-18-phase4
python3 scripts/static_scaling/check_phase4.py --fp32-absolute-1e3 \
  --fixture-dir "$BASE/2026-10-16-phase2" \
  --bundle "$BASE/2026-10-16-phase2/initial_bundle.npz" \
  --flux-reference "$BASE/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz" \
  --pytorch-reference "$BASE/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz" \
  --ddp-root "$OUT/ddp" --device cuda --world-sizes 1,2,4 \
  --split-control "flux=$OUT/split/flux/cuda/split_control.npz" "pytorch=$OUT/split/pytorch/cuda/split_control.npz" \
  --out-dir "$OUT/approved-check" | tee "$OUT/approved-check.log"
