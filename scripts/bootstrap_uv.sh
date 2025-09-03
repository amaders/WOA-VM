#!/usr/bin/env bash
# Bootstrap with uv (no Docker, default HF cache, Torch-first)
# - Installs uv and fixes PATH (handles ~/.local/bin and ~/.cargo/bin)
# - Deactivates conda (base) to avoid /opt/anaconda3 pollution
# - Creates a fresh .venv via `uv venv`
# - Installs torch FIRST (cu121 -> cu118), then the rest (only-if-needed)
# - DOES NOT set or change HF cache vars (uses default ~/.cache/huggingface)

set -euo pipefail

# --------------------------- SETTINGS ---------------------------------------
PY_VER="${PY_VER:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"    # pinned to avoid large CUDA churn
INSTALL_ACCELERATE="${INSTALL_ACCELERATE:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/${VENV_DIR}"
VENV_BIN="${VENV_PATH}/bin"
VENV_PY="${VENV_BIN}/python"
VENV_PIP="${VENV_BIN}/pip"

# ------------------------ HELPERS -------------------------------------------
add_path_persist() {
  local dir="$1"
  case ":$PATH:" in *":$dir:"*) ;; *) export PATH="$dir:$PATH" ;; esac
  if [ -f "$HOME/.bashrc" ] && ! grep -q "export PATH=\"$dir:\$PATH\"" "$HOME/.bashrc" 2>/dev/null; then
    echo "export PATH=\"$dir:\$PATH\"" >> "$HOME/.bashrc"
  fi
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  # common install locations
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -x "$d/uv" ] && add_path_persist "$d"
  done
  # some uv installers drop an env file—source & persist it
  if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.local/bin/env"
    if ! grep -q 'source "$HOME/.local/bin/env"' "$HOME/.bashrc" 2>/dev/null; then
      echo 'source "$HOME/.local/bin/env"' >> "$HOME/.bashrc"
    fi
  fi
  hash -r || true
  command -v uv >/dev/null 2>&1 || {
    echo "[bootstrap] ERROR: uv not on PATH. Run: export PATH=\"\$HOME/.local/bin:\$PATH\""; exit 1;
  }
}

ensure_pip_in_venv() {
  if "${VENV_PY}" -m pip --version >/dev/null 2>&1; then return 0; fi
  echo "[bootstrap] pip missing in venv; trying ensurepip..."
  if "${VENV_PY}" -Im ensurepip --upgrade >/dev/null 2>&1; then
    "${VENV_PY}" -m pip install --upgrade pip wheel setuptools
    return 0
  fi
  echo "[bootstrap] ensurepip unavailable; fetching get-pip.py..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  else
    wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
  fi
  "${VENV_PY}" /tmp/get-pip.py
  "${VENV_PY}" -m pip install --upgrade pip wheel setuptools
}

# --------------------- AVOID CONDA HIJACKING --------------------------------
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[bootstrap] deactivating conda base..."
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda deactivate || true
  fi
fi

# ------------------------ ENSURE uv & CREATE VENV ---------------------------
ensure_uv

echo "[bootstrap] creating fresh venv ${VENV_DIR} with Python ${PY_VER} ..."
# remove existing venv to avoid interactive uv prompt
[ -d "${VENV_PATH}" ] && rm -rf "${VENV_PATH}"
uv venv "${VENV_PATH}" --python "${PY_VER}"

"${VENV_PY}" -V
ensure_pip_in_venv

# --------------------- INSTALL TORCH FIRST (CUDA WHEELS) --------------------
try_torch() { "${VENV_PY}" -m pip install --upgrade --extra-index-url "$1" "torch==${TORCH_VERSION}"; }

echo "[bootstrap] installing torch==${TORCH_VERSION} (CUDA wheels)..."
if try_torch https://download.pytorch.org/whl/cu121; then
  echo "[bootstrap] torch (cu121) installed into ${VENV_DIR}"
elif try_torch https://download.pytorch.org/whl/cu118; then
  echo "[bootstrap] torch (cu118) installed into ${VENV_DIR}"
else
  echo "[bootstrap] ERROR: torch install failed for cu121 & cu118. Check 'nvidia-smi'/drivers."; exit 1
fi

# --------------------- INSTALL THE REST (reuse torch) -----------------------
DEPS=(
  "transformers==4.44.2"
  "sentencepiece==0.2.0"
  "huggingface-hub==0.24.6"
  "hf-transfer==0.1.6"
  "requests>=2.32,<3"
)
if [[ "${INSTALL_ACCELERATE}" == "1" ]]; then
  DEPS+=("accelerate==0.33.0")
fi

echo "[bootstrap] installing deps (won't upgrade torch)..."
"${VENV_PY}" -m pip install --upgrade --upgrade-strategy only-if-needed "${DEPS[@]}"

# ------------------------- OPTIONAL: GPU INFO --------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] GPU inventory:"
  nvidia-smi || true
fi

# --------------------------- HF CACHE NOTE ----------------------------------
if [[ -n "${HF_HOME:-}" || -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  echo "[bootstrap] NOTE: HF cache env vars detected (HF_HOME/HUGGINGFACE_HUB_CACHE)."
  echo "           Using those instead of the default (~/.cache/huggingface)."
  echo "           To use the default, unset them in your shell."
fi

# -------------------------------- DONE --------------------------------------
echo
echo "[bootstrap] done."
echo "Activate venv and run inference:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  MODEL=meta-llama/Llama-3.1-8B-Instruct PROMPT='Say hi' python app/generate.py"
echo
echo "Or, without activating:"
echo "  MODEL=meta-llama/Llama-3.1-8B-Instruct PROMPT='Say hi' ${VENV_DIR}/bin/python app/generate.py"