/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_QUANTIZATION_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_QUANTIZATION_UTILS_H_

#include "tflite/kernels/internal/optimized/rvv_check.h"

#ifdef USE_RVV

#include <limits>

#include "tflite/kernels/internal/compatibility.h"

namespace tflite {
namespace rvv_utils {

// Saturating rounding doubling high multiply: equivalent to NEON vqrdmulhq.
// Computes (a * b + (1 << 30)) >> 31 with int64 intermediates, saturating.
inline vint32m1_t SaturatingRoundingDoublingHighMul(vint32m1_t a, int32_t b,
                                                    size_t vl) {
  vint64m2_t product = __riscv_vwmul_vx_i64m2(a, b, vl);
  vint64m2_t nudge = __riscv_vmv_v_x_i64m2(1LL << 30, vl);
  product = __riscv_vadd_vv_i64m2(product, nudge, vl);
  product = __riscv_vsra_vx_i64m2(product, 31, vl);
  return __riscv_vnclip_wx_i32m1(product, 0, __RISCV_VXRM_RDN, vl);
}

inline vint32m2_t SaturatingRoundingDoublingHighMul_m2(vint32m2_t a, int32_t b,
                                                       size_t vl) {
  vint64m4_t product = __riscv_vwmul_vx_i64m4(a, b, vl);
  vint64m4_t nudge = __riscv_vmv_v_x_i64m4(1LL << 30, vl);
  product = __riscv_vadd_vv_i64m4(product, nudge, vl);
  product = __riscv_vsra_vx_i64m4(product, 31, vl);
  return __riscv_vnclip_wx_i32m2(product, 0, __RISCV_VXRM_RDN, vl);
}

inline vint32m4_t SaturatingRoundingDoublingHighMul_m4(vint32m4_t a, int32_t b,
                                                       size_t vl) {
  vint64m8_t product = __riscv_vwmul_vx_i64m8(a, b, vl);
  vint64m8_t nudge = __riscv_vmv_v_x_i64m8(1LL << 30, vl);
  product = __riscv_vadd_vv_i64m8(product, nudge, vl);
  product = __riscv_vsra_vx_i64m8(product, 31, vl);
  return __riscv_vnclip_wx_i32m4(product, 0, __RISCV_VXRM_RDN, vl);
}

// Rounding-divide-by-power-of-two: equivalent to gemmlowp::RoundingDivideByPOT
// for vector types. Computes (x + rounding) >> exponent.
inline vint32m1_t RoundingDivideByPOT(vint32m1_t x, int exponent, size_t vl) {
  TFLITE_DCHECK_GE(exponent, 0);
  if (exponent == 0) return x;
  vint32m1_t mask = __riscv_vsra_vx_i32m1(x, 31, vl);
  vint32m1_t remainder = __riscv_vand_vx_i32m1(x, (1 << exponent) - 1, vl);
  vint32m1_t threshold =
      __riscv_vadd_vx_i32m1(mask, (1 << exponent) >> 1, vl);
  vbool32_t cmp = __riscv_vmsgt_vv_i32m1_b32(remainder, threshold, vl);
  vint32m1_t shifted = __riscv_vsra_vx_i32m1(x, exponent, vl);
  return __riscv_vadd_vx_i32m1_mu(cmp, shifted, shifted, 1, vl);
}

inline vint32m2_t RoundingDivideByPOT_m2(vint32m2_t x, int exponent,
                                          size_t vl) {
  TFLITE_DCHECK_GE(exponent, 0);
  if (exponent == 0) return x;
  vint32m2_t mask = __riscv_vsra_vx_i32m2(x, 31, vl);
  vint32m2_t remainder = __riscv_vand_vx_i32m2(x, (1 << exponent) - 1, vl);
  vint32m2_t threshold =
      __riscv_vadd_vx_i32m2(mask, (1 << exponent) >> 1, vl);
  vbool16_t cmp = __riscv_vmsgt_vv_i32m2_b16(remainder, threshold, vl);
  vint32m2_t shifted = __riscv_vsra_vx_i32m2(x, exponent, vl);
  return __riscv_vadd_vx_i32m2_mu(cmp, shifted, shifted, 1, vl);
}

inline vint32m4_t RoundingDivideByPOT_m4(vint32m4_t x, int exponent,
                                          size_t vl) {
  TFLITE_DCHECK_GE(exponent, 0);
  if (exponent == 0) return x;
  vint32m4_t mask = __riscv_vsra_vx_i32m4(x, 31, vl);
  vint32m4_t remainder = __riscv_vand_vx_i32m4(x, (1 << exponent) - 1, vl);
  vint32m4_t threshold =
      __riscv_vadd_vx_i32m4(mask, (1 << exponent) >> 1, vl);
  vbool8_t cmp = __riscv_vmsgt_vv_i32m4_b8(remainder, threshold, vl);
  vint32m4_t shifted = __riscv_vsra_vx_i32m4(x, exponent, vl);
  return __riscv_vadd_vx_i32m4_mu(cmp, shifted, shifted, 1, vl);
}

// Multiply by quantized multiplier: applies left_shift, then SRDH multiply,
// then right_shift. Combined left_shift includes params.left_shift +
// params.input_shift. If left_shift >= 0, apply before multiply; if < 0,
// apply -left_shift as right shift after multiply.
inline vint32m1_t MultiplyByQuantizedMultiplier(vint32m1_t x,
                                                int32_t multiplier,
                                                int32_t total_shift,
                                                size_t vl) {
  const int left_shift = std::max(0, total_shift);
  const int right_shift = std::max(0, -total_shift);
  if (left_shift > 0) {
    x = __riscv_vsll_vx_i32m1(x, left_shift, vl);
  }
  x = SaturatingRoundingDoublingHighMul(x, multiplier, vl);
  return RoundingDivideByPOT(x, right_shift, vl);
}

inline vint32m2_t MultiplyByQuantizedMultiplier_m2(vint32m2_t x,
                                                   int32_t multiplier,
                                                   int32_t total_shift,
                                                   size_t vl) {
  const int left_shift = std::max(0, total_shift);
  const int right_shift = std::max(0, -total_shift);
  if (left_shift > 0) {
    x = __riscv_vsll_vx_i32m2(x, left_shift, vl);
  }
  x = SaturatingRoundingDoublingHighMul_m2(x, multiplier, vl);
  return RoundingDivideByPOT_m2(x, right_shift, vl);
}

inline vint32m4_t MultiplyByQuantizedMultiplier_m4(vint32m4_t x,
                                                   int32_t multiplier,
                                                   int32_t total_shift,
                                                   size_t vl) {
  const int left_shift = std::max(0, total_shift);
  const int right_shift = std::max(0, -total_shift);
  if (left_shift > 0) {
    x = __riscv_vsll_vx_i32m4(x, left_shift, vl);
  }
  x = SaturatingRoundingDoublingHighMul_m4(x, multiplier, vl);
  return RoundingDivideByPOT_m4(x, right_shift, vl);
}

}  // namespace rvv_utils
}  // namespace tflite

#endif  // USE_RVV
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_QUANTIZATION_UTILS_H_
