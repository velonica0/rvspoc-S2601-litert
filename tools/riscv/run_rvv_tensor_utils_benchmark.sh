#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-riscv-bench}"
OUT_FILE="${OUT_FILE:-${ROOT_DIR}/docs/rvv_scalar_benchmarks.md}"

mkdir -p "${BUILD_DIR}"

CXX="${CXX:-g++}"
COMMON_FLAGS=(
  -std=c++17
  -O3
  -DNDEBUG
  -I"${ROOT_DIR}"
  -I"${ROOT_DIR}/litert/cmake_build/gemmlowp"
)
SCALAR_FLAGS=(
  -march=rv64gc
  -mabi=lp64d
  -fno-tree-vectorize
  -fno-tree-slp-vectorize
)
RVV_FLAGS=(
  -march=rv64gcv_zvl128b
  -mabi=lp64d
)

echo "[build] scalar portable object"
"${CXX}" "${COMMON_FLAGS[@]}" "${SCALAR_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/reference/portable_tensor_utils.cc" \
  -o "${BUILD_DIR}/portable_scalar.o"

echo "[build] scalar common object"
"${CXX}" "${COMMON_FLAGS[@]}" "${SCALAR_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/common.cc" \
  -o "${BUILD_DIR}/common_scalar.o"

echo "[build] rvv object"
"${CXX}" "${COMMON_FLAGS[@]}" "${RVV_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/optimized/rvv_tensor_utils.cc" \
  -o "${BUILD_DIR}/rvv_tensor_utils.o"

echo "[build] benchmark main"
"${CXX}" "${COMMON_FLAGS[@]}" "${RVV_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/optimized/rvv_tensor_utils_benchmark.cc" \
  -o "${BUILD_DIR}/rvv_tensor_utils_benchmark.o"

echo "[link] benchmark binary"
"${CXX}" "${RVV_FLAGS[@]}" \
  -no-pie \
  "${BUILD_DIR}/common_scalar.o" \
  "${BUILD_DIR}/portable_scalar.o" \
  "${BUILD_DIR}/rvv_tensor_utils.o" \
  "${BUILD_DIR}/rvv_tensor_utils_benchmark.o" \
  -lm \
  -o "${BUILD_DIR}/rvv_tensor_utils_benchmark"

{
  echo "# RVV Scalar Benchmark Results"
  echo
  echo "- Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- Host: $(uname -a)"
  echo "- Compiler: $(${CXX} --version | head -n 1)"
  echo "- Build dir: \`${BUILD_DIR}\`"
  echo
  "${BUILD_DIR}/rvv_tensor_utils_benchmark"
} | tee "${OUT_FILE}"

echo
echo "[done] wrote ${OUT_FILE}"
