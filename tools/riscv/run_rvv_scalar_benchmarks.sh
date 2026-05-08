#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FINAL_OUT="${OUT_FILE:-${ROOT_DIR}/docs/rvv_scalar_benchmarks.md}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

TENSOR_UTILS_OUT="${TMP_DIR}/rvv_tensor_utils_benchmarks.md"
OPERATOR_OUT="${TMP_DIR}/rvv_operator_benchmarks.md"
RVV_COMMITS=(
  '8edc8e0d|2026-05-08|integer_ops'
  '36a4fa2f|2026-05-08|optimized_ops.h'
  'd447fe69|2026-05-07|Portable 2 RVV'
  '84e86354|2026-05-07|all rvv_tensor_utils ops'
  '82d3365e|2026-05-06|rvv_tensor_utils_impl'
)

echo "[run] tensor_utils benchmark"
OUT_FILE="${TENSOR_UTILS_OUT}" "${ROOT_DIR}/tools/riscv/run_rvv_tensor_utils_benchmark.sh"

echo "[run] operator benchmark"
OUT_FILE="${OPERATOR_OUT}" "${ROOT_DIR}/tools/riscv/run_rvv_operator_benchmark.sh"

mkdir -p "$(dirname "${FINAL_OUT}")"

{
  echo "# RVV Scalar Benchmark Results"
  echo
  echo "- Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- Host: $(uname -a)"
  echo "- Compiler: $(${CXX:-g++} --version | head -n 1)"
  echo "- Scope: recent five commits that introduced or wired RVV coverage, plus follow-up P1 gap closures for \`reduce.h\` and \`resize_bilinear.h\`"
  echo
  echo "## Recent Five Commits"
  echo
  echo "| Commit | Date | Summary |"
  echo "| --- | --- | --- |"
  for commit_meta in "${RVV_COMMITS[@]}"; do
    IFS='|' read -r commit date summary <<<"${commit_meta}"
    echo "| \`${commit}\` | ${date} | ${summary} |"
  done
  echo
  echo "## Coverage Map"
  echo
  echo "| Source file | Recent commit(s) | Benchmark coverage |"
  echo "| --- | --- | --- |"
  echo "| \`tflite/kernels/internal/optimized/rvv_tensor_utils.cc\` | \`82d3365e\`, \`84e86354\`, \`d447fe69\` | Shared helper-level scalar vs RVV coverage |"
  echo "| \`tflite/kernels/internal/optimized/optimized_ops.h\` | \`36a4fa2f\` | Float arithmetic, affine quantize, uint8 pooling, HardSwish, ArgMin/ArgMax |"
  echo "| \`tflite/kernels/fully_connected.cc\` | \`36a4fa2f\` | Dense float \`EvalPie\` route benchmark |"
  echo "| \`tflite/kernels/lstm_eval.cc\` | \`36a4fa2f\` | Float gate and output/projection operator benchmarks |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/add.h\` | \`8edc8e0d\` | int8/int16 add plus scalar-broadcast kernels |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/mul.h\` | \`8edc8e0d\` | int8 mul plus scalar-broadcast kernels |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/sub.h\` | \`8edc8e0d\` | int16 sub |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/pooling.h\` | \`8edc8e0d\` | int8 average/max pooling |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h\` | \`8edc8e0d\` | int8 depthwise conv general and specialized kernels |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h\` | \`8edc8e0d\` | Dispatch-only; shares numeric coverage with \`depthwise_conv.h\` |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/leaky_relu.h\` | \`8edc8e0d\` | int16 LeakyReLU |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/lut.h\` | \`8edc8e0d\` | uint8/int8 lookup table |"
  echo "| \`tflite/kernels/internal/optimized/integer_ops/mean.h\` | \`8edc8e0d\` | int8 mean reduction |"
  echo "| \`tflite/kernels/internal/optimized/reduce.h\` | Follow-up P1 closure (2026-05-08) | uint8 keep-dims height-width mean and float last-dim mean |"
  echo "| \`tflite/kernels/internal/optimized/resize_bilinear.h\` | Follow-up P1 closure (2026-05-08) | float and uint8 generic bilinear resize kernels |"
  echo
  echo "## \`tflite/kernels/internal/optimized/rvv_tensor_utils.cc\`"
  echo
  echo "- Summary: shared tensor_utils helper coverage accumulated across the first three of the recent five RVV commits."
  echo
  sed -n '/^- Runtime VLEN bits:/,$p' "${TENSOR_UTILS_OUT}" | sed 's/^## /### /'
  echo
  sed -n '/^## `/,$p' "${OPERATOR_OUT}"
} > "${FINAL_OUT}"

echo
echo "[done] wrote ${FINAL_OUT}"
