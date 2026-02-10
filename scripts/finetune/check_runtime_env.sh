#!/usr/bin/env bash
set -eo pipefail

STRICT=false
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=true
fi

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() {
  echo "[PASS] $*"
  PASS_COUNT=$((PASS_COUNT + 1))
}

warn() {
  echo "[WARN] $*"
  WARN_COUNT=$((WARN_COUNT + 1))
}

fail() {
  echo "[FAIL] $*"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_TEXT_SH="${GREC_ROOT}/scripts/finetune/train_text.sh"
SMOKE_SH="${GREC_ROOT}/scripts/finetune/smoke_train.sh"

if [[ -f "${TRAIN_TEXT_SH}" ]]; then
  pass "Found train launcher: ${TRAIN_TEXT_SH}"
else
  fail "Missing train launcher: ${TRAIN_TEXT_SH}"
fi

if [[ -f "${SMOKE_SH}" ]]; then
  pass "Found smoke launcher: ${SMOKE_SH}"
else
  fail "Missing smoke launcher: ${SMOKE_SH}"
fi

if grep -q 'PYTHONNOUSERSITE' "${TRAIN_TEXT_SH}"; then
  pass "train_text.sh sets PYTHONNOUSERSITE"
else
  fail "train_text.sh does not set PYTHONNOUSERSITE"
fi

if grep -q 'PYTHON_BIN_PATH' "${TRAIN_TEXT_SH}"; then
  pass "train_text.sh resolves PYTHON_BIN_PATH"
else
  fail "train_text.sh does not resolve PYTHON_BIN_PATH"
fi

if grep -q 'torch.distributed.run' "${TRAIN_TEXT_SH}"; then
  pass "train_text.sh uses 'python -m torch.distributed.run'"
else
  fail "train_text.sh is not using 'python -m torch.distributed.run'"
fi

if grep -q 'NPROC:=4' "${SMOKE_SH}" && grep -q 'GPUS:=0,1,2,3' "${SMOKE_SH}"; then
  pass "smoke_train.sh defaults to 4 GPUs / NPROC=4"
else
  warn "smoke_train.sh default GPU/NPROC is not 4"
fi

# Match train_text.sh launcher selection logic
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  EXPECTED_PY="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
else
  if command -v python >/dev/null 2>&1; then
    EXPECTED_PY="${PYTHON_BIN:-python}"
  else
    EXPECTED_PY="${PYTHON_BIN:-python3}"
  fi
fi

if [[ "${EXPECTED_PY}" == /* ]]; then
  EXPECTED_PY_PATH="${EXPECTED_PY}"
else
  EXPECTED_PY_PATH="$(command -v "${EXPECTED_PY}" || true)"
fi

if [[ -z "${EXPECTED_PY_PATH}" || ! -x "${EXPECTED_PY_PATH}" ]]; then
  fail "Cannot resolve executable python launcher: ${EXPECTED_PY}"
  echo ""
  echo "Summary: pass=${PASS_COUNT}, warn=${WARN_COUNT}, fail=${FAIL_COUNT}"
  exit 1
fi

pass "Launcher python: ${EXPECTED_PY_PATH}"

CUR_PY="$(command -v python || true)"
CUR_PY3="$(command -v python3 || true)"
CUR_TORCHRUN="$(command -v torchrun || true)"

echo "[INFO] which python  = ${CUR_PY:-<not found>}"
echo "[INFO] which python3 = ${CUR_PY3:-<not found>}"
echo "[INFO] which torchrun= ${CUR_TORCHRUN:-<not found>}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if [[ "${EXPECTED_PY_PATH}" == "${CONDA_PREFIX}"/* ]]; then
    pass "Launcher python is inside active CONDA_PREFIX"
  else
    fail "Launcher python is outside active CONDA_PREFIX (${CONDA_PREFIX})"
  fi
else
  warn "CONDA_PREFIX is empty (no active conda env?)"
fi

if [[ -n "${CUR_PY3}" && -n "${CONDA_PREFIX:-}" && "${CUR_PY3}" != "${CONDA_PREFIX}"/* ]]; then
  warn "python3 is outside conda env (this is ok now, but avoid using raw python3 in launch)"
fi

RUNTIME_CHECK_FAIL=0
if ! PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" "${EXPECTED_PY_PATH}" <<'PY'
import importlib
import os
import site
import sys

fail = 0

def ok(msg):
    print(f"[PASS] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def bad(msg):
    global fail
    fail += 1
    print(f"[FAIL] {msg}")

print(f"[INFO] sys.executable={sys.executable}")
print(f"[INFO] python_version={sys.version.split()[0]}")
print(f"[INFO] ENABLE_USER_SITE={site.ENABLE_USER_SITE}")

if site.ENABLE_USER_SITE:
    bad("site.ENABLE_USER_SITE=True (expected False for clean env)")
else:
    ok("site.ENABLE_USER_SITE=False")

modules = ["torch", "transformers", "accelerate", "deepspeed", "triton"]
for name in modules:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        bad(f"import {name} failed: {type(exc).__name__}: {exc}")
        continue

    path = getattr(mod, "__file__", "<no __file__>")
    ver = getattr(mod, "__version__", "<no __version__>")
    print(f"[INFO] {name}={ver} @ {path}")
    if "/.local/lib/python" in str(path):
        bad(f"{name} is imported from ~/.local (environment polluted)")

try:
    import torch
    ok(f"torch.cuda.is_available={torch.cuda.is_available()}")
    ok(f"torch.cuda.device_count={torch.cuda.device_count()}")
    ok(f"torch.version.cuda={torch.version.cuda}")
    ok(f"hasattr(torch, 'compile')={hasattr(torch, 'compile')}")
except Exception as exc:
    bad(f"torch runtime check failed: {type(exc).__name__}: {exc}")

sys.exit(1 if fail else 0)
PY
then
  RUNTIME_CHECK_FAIL=1
fi

if [[ "${RUNTIME_CHECK_FAIL}" -eq 1 ]]; then
  fail "Python runtime/module checks failed"
else
  pass "Python runtime/module checks passed"
fi

PY_INCLUDE="$("${EXPECTED_PY_PATH}" - <<'PY'
import sysconfig
print(sysconfig.get_paths().get('include', ''))
PY
)"

if [[ -z "${PY_INCLUDE}" || ! -d "${PY_INCLUDE}" ]]; then
  fail "Python include dir not found: ${PY_INCLUDE}"
else
  pass "Python include dir: ${PY_INCLUDE}"
fi

if [[ -f "/usr/include/crypt.h" || -f "${CONDA_PREFIX:-}/include/crypt.h" ]]; then
  pass "Found crypt.h candidate (/usr/include or conda include)"
else
  warn "crypt.h not found in /usr/include or ${CONDA_PREFIX:-<no conda>}/include"
fi

CC_BIN=""
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc" ]]; then
  CC_BIN="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
elif command -v gcc >/dev/null 2>&1; then
  CC_BIN="$(command -v gcc)"
fi

if [[ -z "${CC_BIN}" ]]; then
  warn "No C compiler found for compile probe"
else
  pass "Compiler probe with: ${CC_BIN}"
  TMP_DIR="$(mktemp -d)"
  cat > "${TMP_DIR}/check_crypt.c" <<'C'
#include <Python.h>
#include <crypt.h>
int main(void) { return 0; }
C

  if "${CC_BIN}" -c "${TMP_DIR}/check_crypt.c" -I"${PY_INCLUDE}" -o "${TMP_DIR}/check_crypt.o" >/tmp/check_runtime_env_cc.log 2>&1; then
    pass "C compile probe succeeded (Python.h + crypt.h)"
  else
    fail "C compile probe failed (likely missing crypt.h/dev headers). See /tmp/check_runtime_env_cc.log"
    head -n 20 /tmp/check_runtime_env_cc.log || true
  fi
  rm -rf "${TMP_DIR}"
fi

echo ""
echo "Summary: pass=${PASS_COUNT}, warn=${WARN_COUNT}, fail=${FAIL_COUNT}"

if [[ "${STRICT}" == "true" ]]; then
  if [[ "${FAIL_COUNT}" -gt 0 || "${WARN_COUNT}" -gt 0 ]]; then
    exit 2
  fi
else
  if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    exit 1
  fi
fi
