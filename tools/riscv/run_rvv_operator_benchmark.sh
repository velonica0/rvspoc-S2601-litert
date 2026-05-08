#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-riscv-op-bench}"
OUT_FILE="${OUT_FILE:-${ROOT_DIR}/docs/rvv_operator_benchmarks.md}"

mkdir -p "${BUILD_DIR}"

CXX="${CXX:-g++}"
COMMON_FLAGS=(
  -std=c++17
  -O3
  -DNDEBUG
  -I"${ROOT_DIR}"
  -I"${ROOT_DIR}/litert/cmake_build/gemmlowp"
  -I"${ROOT_DIR}/litert/cmake_build/eigen"
  -I"${ROOT_DIR}/litert/cmake_build/ml_dtypes/third_party/eigen"
  -I"${ROOT_DIR}/litert/cmake_build/ruy"
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

echo "[build] scalar operator benchmark wrapper"
"${CXX}" "${COMMON_FLAGS[@]}" "${SCALAR_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/optimized/rvv_operator_benchmark_lib.cc" \
  -o "${BUILD_DIR}/rvv_operator_benchmark_scalar.o"

echo "[build] runtime_shape support object"
"${CXX}" "${COMMON_FLAGS[@]}" "${SCALAR_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/runtime_shape.cc" \
  -o "${BUILD_DIR}/runtime_shape_scalar.o"

echo "[build] quantized common support object"
"${CXX}" "${COMMON_FLAGS[@]}" "${SCALAR_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/common.cc" \
  -o "${BUILD_DIR}/common_scalar.o"

echo "[build] rvv operator benchmark wrapper"
"${CXX}" "${COMMON_FLAGS[@]}" "${RVV_FLAGS[@]}" \
  -DTFLITE_RVV_OPERATOR_BENCH_VARIANT_RVV -c \
  "${ROOT_DIR}/tflite/kernels/internal/optimized/rvv_operator_benchmark_lib.cc" \
  -o "${BUILD_DIR}/rvv_operator_benchmark_rvv.o"

echo "[build] benchmark main"
"${CXX}" "${COMMON_FLAGS[@]}" "${RVV_FLAGS[@]}" -c \
  "${ROOT_DIR}/tflite/kernels/internal/optimized/rvv_operator_benchmark_main.cc" \
  -o "${BUILD_DIR}/rvv_operator_benchmark_main.o"

echo "[link] benchmark binary"
"${CXX}" "${RVV_FLAGS[@]}" \
  -no-pie \
  "${BUILD_DIR}/runtime_shape_scalar.o" \
  "${BUILD_DIR}/common_scalar.o" \
  "${BUILD_DIR}/rvv_operator_benchmark_scalar.o" \
  "${BUILD_DIR}/rvv_operator_benchmark_rvv.o" \
  "${BUILD_DIR}/rvv_operator_benchmark_main.o" \
  -lm \
  -o "${BUILD_DIR}/rvv_operator_benchmark"

{
  echo "# RVV Operator Benchmark Results"
  echo
  echo "- Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- Host: $(uname -a)"
  echo "- Compiler: $(${CXX} --version | head -n 1)"
  echo "- Build dir: \`${BUILD_DIR}\`"
  echo
  "${BUILD_DIR}/rvv_operator_benchmark"
} | tee "${OUT_FILE}"

echo
echo "[done] wrote ${OUT_FILE}"
