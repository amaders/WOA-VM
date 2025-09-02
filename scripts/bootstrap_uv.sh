#!/usr/bin/env bash
# Ephemeral GPU VM bootstrap (Voltage Park friendly)
# - Installs uv (user-local)
# - Creates .venv and installs deps
# - Sets Hugging Face cache (HOME by default; /mnt if present)
# Usage:
#   bash scripts/bootstrap_uv.sh
set -euo pipefail

# --------------------------- SETTINGS ---------------------------------------
PY_VER="${PY_VER:-3.11}"          # override: PY_VER=3.10 bash scripts/bootstrap_uv.sh
VENV_DIR="${VENV_DIR:-.venv}"     # local virtual env directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Optional: user override for cache path
# If provided, wins over autodetection.
USER_HF_CACHE="${HF_CACHE_DIR:-}"

# Minimum free GB to avoid first-download surprises for 8B–13B models
THRESHOLD_GB="${HF_MIN_FREE_GB:-20}"

# ------------------------ UV INSTALL / PATH ---------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
  fi
else
  echo "[bootstrap] uv already present."
fi

# -------------------------- VENV & BASE DEPS --------------------------------
echo "[bootstrap] creating venv with Python ${PY_VER} at ${VENV_DIR} ..."
uv venv "${VENV_DIR}" --python "${PY_VER}"

echo "[bootstrap] syncing project dependencies (excluding torch)..."
cd "${PROJECT_ROOT}"
uv sync

# ------------------------- TORCH (CUDA WHEELS) ------------------------------
try_install_torch() {
  local index_url="$1"
  echo "[bootstrap] attempting torch install from: ${index_url}"
  if uv pip install --extra-index-url "${index_url}" "torch==2.4.0"; then
    echo "[bootstrap] torch installed from ${index_url}"
    return 0
  fi
  return 1
}

TORCH_OK=0
if try_install_torch "https://download.pytorch.org/whl/cu121"; then
  TORCH_OK=1
else
  echo "[bootstrap] cu121 failed, trying cu118..."
  if try_install_torch "https://download.pytorch.org/whl/cu118"; then
    TORCH_OK=1
  fi
fi

if [ "${TORCH_OK}" -ne 1 ]; then
  echo "[bootstrap] ERROR: torch install failed (cu121 & cu118)."
  echo "Check drivers with 'nvidia-smi' and ensure compatibility, then retry."
  exit 1
fi

# ------------------------- GPU INVENTORY ------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] GPU inventory:"
  nvidia-smi || true
else
  echo "[bootstrap] NOTE: 'nvidia-smi' not found; ensure this VM image has NVIDIA drivers."
fi

# ---------------------- HUGGING FACE CACHE ----------------------------------
# Priority:
#   1) explicit HF_CACHE_DIR env provided by user
#   2) /mnt/hf-cache if /mnt exists
#   3) $HOME/.cache/huggingface
if [ -n "${USER_HF_CACHE}" ]; then
  HF_CACHE="${USER_HF_CACHE}"
elif [ -d /mnt ]; then
  HF_CACHE="/mnt/hf-cache"
else
  HF_CACHE="$HOME/.cache/huggingface"
fi

mkdir -p "${HF_CACHE}"

# Persist env for future shells (won't overwrite if already present)
if ! grep -q 'HUGGINGFACE_HUB_CACHE' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo "export HF_HOME='${HF_CACHE}'"
    echo "export HUGGINGFACE_HUB_CACHE='${HF_CACHE}'"
  } >> "$HOME/.bashrc"
fi

# ------------------------ FREE-SPACE CHECK ----------------------------------
check_space_gb() {
  # Prints free GiB for filesystem containing the given path
  df -BG "$1" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}'
}

FREE_GB="$(check_space_gb "${HF_CACHE}" || echo 0)"
if [ -z "${FREE_GB}" ]; then FREE_GB=0; fi

if [ "${FREE_GB}" -lt "${THRESHOLD_GB}" ]; then
  echo "[bootstrap] WARNING: Only ~${FREE_GB} GiB free at cache path: ${HF_CACHE}"
  echo "           Large models (8B–13B) may require 15–30 GiB on first download."
  echo "           To override, re-run with a bigger path, e.g.:"
  echo "             HF_CACHE_DIR=/path/with/space bash scripts/bootstrap_uv.sh"
  echo "           Or set before loading models:"
  echo "             export HF_HOME=/path/with/space"
  echo "             export HUGGINGFACE_HUB_CACHE=/path/with/space"
fi

# ---------------------------- DONE ------------------------------------------
echo
echo "[bootstrap] done! Open a new shell or run:  source ~/.bashrc"
echo "[bootstrap] quick test:"
echo "  MODEL=meta-llama/Llama-3.1-8B-Instruct PROMPT='Say hi.' uv run python app/generate.py"