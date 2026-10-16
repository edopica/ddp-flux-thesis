#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) login-node setup for the Python + PyTorch environment of
# the planned Flux-vs-PyTorch GPU scaling benchmark.
#
#   RUN ON A LOGIN NODE (compute nodes have no internet / no pip):
#     ssh leonardo 'bash ~/leonardo-setup/setup-pytorch-env.sh'
#
# Idempotent: a re-run verifies the existing install instead of reinstalling.
#   --reinstall   remove the venv and rebuild it from scratch
#
# Scope: environment setup + import verification ONLY. No model, training,
# dataset, or benchmark code lives here. See wiki/context/leonardo.md.
# =============================================================================
set -euo pipefail

REINSTALL=0
for arg in "$@"; do
  case "$arg" in
    --reinstall) REINSTALL=1 ;;
    *) echo "ERROR: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Paths (override via environment); all on $WORK, never in the 50 GB $HOME.
WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
UV_DIR=${UV_DIR:-$WORK/epicazio/tools/uv}
UV_BIN=${UV_BIN:-$UV_DIR/uv}
UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/epicazio/uv-cache}
UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$WORK/epicazio/uv-python}
VENV=${VENV:-$WORK/epicazio/venvs/pytorch-scaling}
VENV_PY=$VENV/bin/python

PYTHON_REQUEST=${PYTHON_REQUEST:-3.11}
TORCH_VERSION=${TORCH_VERSION:-2.9.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.24.1}
INDEX_URL=${INDEX_URL:-https://download.pytorch.org/whl/cu126}

export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

echo "=== leonardo setup-pytorch-env: START ==="
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) user=$(id -un)"
echo "WORK=$WORK"
echo "UV_DIR=$UV_DIR"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
echo "VENV=$VENV"
echo "PYTHON_REQUEST=$PYTHON_REQUEST"
echo "INDEX_URL=$INDEX_URL"
echo "TORCH_VERSION=$TORCH_VERSION"
echo "TORCHVISION_VERSION=$TORCHVISION_VERSION"
echo "REINSTALL=$REINSTALL"

# ---- 1/5 uv ---------------------------------------------------------------
if [ ! -x "$UV_BIN" ]; then
  echo "[1/5] uv missing: installing into $UV_DIR"
  mkdir -p "$UV_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
  export UV_INSTALL_DIR="$UV_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  unset UV_INSTALL_DIR
  if [ ! -x "$UV_BIN" ]; then
    echo "ERROR: uv binary not found at $UV_BIN after installer run" >&2
    exit 1
  fi
else
  echo "[1/5] uv already present: $UV_BIN"
fi
echo "[1/5] uv --version: $("$UV_BIN" --version)"

# ---- 2/5 venv -------------------------------------------------------------
if [ "$REINSTALL" -eq 1 ] && [ -e "$VENV" ]; then
  echo "[2/5] --reinstall: removing $VENV"
  rm -rf "$VENV"
fi
if [ ! -x "$VENV_PY" ]; then
  echo "[2/5] creating venv $VENV (uv-managed CPython $PYTHON_REQUEST)"
  "$UV_BIN" venv --python "$PYTHON_REQUEST" --python-preference only-managed "$VENV"
else
  echo "[2/5] venv already present: $VENV (not recreating)"
fi
echo "[2/5] uv python list --only-installed:"
"$UV_BIN" python list --only-installed
echo "[2/5] venv python version: $("$VENV_PY" --version)"

# ---- 3/5 torch + torchvision ---------------------------------------------
NEED_INSTALL=1
if [ -x "$VENV_PY" ] && [ "$REINSTALL" -eq 0 ]; then
  if "$VENV_PY" - <<PY 2>/dev/null
import torch, torchvision
assert torch.__version__.split("+")[0] == "$TORCH_VERSION", torch.__version__
assert torchvision.__version__.split("+")[0] == "$TORCHVISION_VERSION", torchvision.__version__
PY
  then
    NEED_INSTALL=0
    echo "[3/5] pinned torch/torchvision already installed: skipping install"
  else
    echo "[3/5] existing venv does not match pins: (re)installing"
  fi
fi
if [ "$NEED_INSTALL" -eq 1 ]; then
  echo "[3/5] uv pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url $INDEX_URL"
  "$UV_BIN" pip install --python "$VENV_PY" --index-url "$INDEX_URL" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
fi

# ---- 4/5 freeze -----------------------------------------------------------
echo "[4/5] freeze -> $VENV/requirements-freeze.txt"
"$UV_BIN" pip freeze --python "$VENV_PY" | tee "$VENV/requirements-freeze.txt"

# ---- 5/5 login-node import check -----------------------------------------
echo "[5/5] login-node import check (no GPU on login nodes; torch.cuda.is_available() is expected False here)"
"$VENV_PY" -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.version.cuda)"

echo "=== leonardo setup-pytorch-env: DONE ==="
echo "DONE uv:     $UV_BIN ($("$UV_BIN" --version))"
echo "DONE python: $VENV_PY ($("$VENV_PY" --version 2>&1))"
echo "DONE venv:   $VENV"
echo "DONE freeze: $VENV/requirements-freeze.txt"
echo "NEXT (copy freeze into the local repo):"
echo "  scp leonardo:$VENV/requirements-freeze.txt artifacts/logs/leonardo/2026-10-15-pytorch-env-freeze.txt"
echo "NEXT (GPU environment verification only, NOT a benchmark):"
echo "  stage scripts/remote/leonardo/verify-pytorch-env.py to $VENV/verify-pytorch-env.py"
echo "  ssh leonardo \"srun -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg -N1 -n1 --cpus-per-task=4 --gpus-per-task=1 --time=00:05:00 bash -lc '$VENV_PY $VENV/verify-pytorch-env.py'\""
