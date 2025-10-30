#!/usr/bin/env bash
# Vendor a fixed snapshot of timm into ./timm (project root).
# Usage:
#   bash scripts/setup_timm_vendor.sh               # default TIMM_REF=v0.6.13
#   TIMM_REF=v0.9.16 bash scripts/setup_timm_vendor.sh
#   TIMM_REF=<commit-hash> bash scripts/setup_timm_vendor.sh
#
# Requirements: git, rsync (optional but recommended)

set -euo pipefail

# --- config ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${PROJECT_ROOT}/timm"
WORKDIR="${PROJECT_ROOT}/.vendor_tmp_timm"
REPO_URL="https://github.com/huggingface/pytorch-image-models.git"
: "${TIMM_REF:=v0.6.13}"   # default tag; change via env TIMM_REF

echo "[timm-vendor] project root: ${PROJECT_ROOT}"
echo "[timm-vendor] target dir  : ${VENDOR_DIR}"
echo "[timm-vendor] source repo  : ${REPO_URL}"
echo "[timm-vendor] git ref      : ${TIMM_REF}"

# --- prepare temp workspace ---
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
pushd "${WORKDIR}" >/dev/null

# --- shallow clone specific ref ---
echo "[timm-vendor] cloning timm @ ${TIMM_REF} ..."
git init -q
git remote add origin "${REPO_URL}"
git fetch --depth 1 origin "${TIMM_REF}"
git checkout -q FETCH_HEAD

# --- copy package directory ---
echo "[timm-vendor] copying package to ${VENDOR_DIR} ..."
rm -rf "${VENDOR_DIR}"
mkdir -p "${VENDOR_DIR}"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    "timm/" "${VENDOR_DIR}/" \
    --exclude "__pycache__" --exclude "*.pyc" --exclude "models/_hub.py" --exclude "examples" --exclude "results" --exclude ".github" --exclude "tests" --exclude ".git" --exclude ".gitignore"
else
  # Fallback without rsync
  cp -r timm/* "${VENDOR_DIR}/"
  find "${VENDOR_DIR}" -name "__pycache__" -type d -exec rm -rf {} +
  find "${VENDOR_DIR}" -name "*.pyc" -delete
fi

# --- write vendor meta / license notice ---
echo "__version__ = 'vendored:${TIMM_REF}'" > "${VENDOR_DIR}/_vendor_version.py"
if [ -f "LICENSE" ]; then
  cp -f LICENSE "${PROJECT_ROOT}/LICENSE.timm"
fi

popd >/dev/null
rm -rf "${WORKDIR}"

echo "[timm-vendor] Done. Local 'timm/' now pinned to ${TIMM_REF}"
echo "[timm-vendor] Tip: ensure local package has priority (sys.path insert) in main.py"
