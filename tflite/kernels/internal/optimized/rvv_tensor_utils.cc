/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tflite/kernels/internal/optimized/rvv_tensor_utils_impl.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/reference/portable_tensor_utils_impl.h"

namespace tflite {
namespace tensor_utils {

#ifdef USE_RVV
namespace {

constexpr int kMaxRvvBits = 512;
constexpr int kMaxRvvInt16Lanes = kMaxRvvBits / 16;
constexpr int kMaxRvvInt32Lanes = kMaxRvvBits / 16;

template <typename T>
inline T Clamp(T value, T min_value, T max_value) {
  return std::min(std::max(value, min_value), max_value);
}

float HorizontalSum(vfloat32m1_t values, size_t vl) {
  const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  const vfloat32m1_t reduced =
      __riscv_vfredusum_vs_f32m1_f32m1(values, zero, vl);
  return __riscv_vfmv_f_s_f32m1_f32(reduced);
}

int32_t HorizontalSum(vint32m2_t values, size_t vl) {
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t reduced = __riscv_vredsum_vs_i32m2_i32m1(values, zero, vl);
  return __riscv_vmv_x_s_i32m1_i32(reduced);
}

int32_t HorizontalSum(vint32m4_t values, size_t vl) {
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t reduced = __riscv_vredsum_vs_i32m4_i32m1(values, zero, vl);
  return __riscv_vmv_x_s_i32m1_i32(reduced);
}

int64_t HorizontalSum(vint64m4_t values, size_t vl) {
  const vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, 1);
  const vint64m1_t reduced = __riscv_vredsum_vs_i64m4_i64m1(values, zero, vl);
  return __riscv_vmv_x_s_i64m1_i64(reduced);
}

vint32m4_t RvvRoundToNearestAwayFromZero(vfloat32m4_t values, size_t vl) {
  const vfloat32m4_t half = __riscv_vfmv_v_f_f32m4(0.5f, vl);
  const vfloat32m4_t signed_half =
      __riscv_vfsgnj_vv_f32m4(half, values, vl);
  const vfloat32m4_t adjusted =
      __riscv_vfadd_vv_f32m4(values, signed_half, vl);
  return __riscv_vfcvt_rtz_x_f_v_i32m4(adjusted, vl);
}

vint32m2_t RvvRoundToNearestAwayFromZero(vfloat32m2_t values, size_t vl) {
  const vfloat32m2_t half = __riscv_vfmv_v_f_f32m2(0.5f, vl);
  const vfloat32m2_t signed_half =
      __riscv_vfsgnj_vv_f32m2(half, values, vl);
  const vfloat32m2_t adjusted =
      __riscv_vfadd_vv_f32m2(values, signed_half, vl);
  return __riscv_vfcvt_rtz_x_f_v_i32m2(adjusted, vl);
}

void RvvQuantizeFloats(const float* values, int size, float scaling_factor_inv,
                       int32_t offset, int32_t clamp_min, int32_t clamp_max,
                       int8_t* quantized_values) {
  int col = 0;
  while (col < size) {
    const size_t vl = __riscv_vsetvl_e32m4(size - col);
    const vfloat32m4_t input =
        __riscv_vle32_v_f32m4(values + col, vl);
    vfloat32m4_t transformed =
        __riscv_vfmul_vf_f32m4(input, scaling_factor_inv, vl);
    if (offset != 0) {
      transformed = __riscv_vfadd_vf_f32m4(
          transformed, static_cast<float>(offset), vl);
    }
    vint32m4_t rounded = RvvRoundToNearestAwayFromZero(transformed, vl);
    rounded = __riscv_vmax_vx_i32m4(rounded, clamp_min, vl);
    rounded = __riscv_vmin_vx_i32m4(rounded, clamp_max, vl);
    const vint16m2_t narrowed16 =
        __riscv_vnclip_wx_i16m2(rounded, 0, __RISCV_VXRM_RDN, vl);
    const vint8m1_t narrowed8 =
        __riscv_vnclip_wx_i8m1(narrowed16, 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse8_v_i8m1(quantized_values + col, narrowed8, vl);
    col += static_cast<int>(vl);
  }
}

vint32m2_t RvvMultiplyByQuantizedMultiplier(vint32m2_t values,
                                            int32_t multiplier, int shift,
                                            size_t vl);

vint32m2_t RvvRoundingDivideByPOT(vint64m4_t values, int exponent, size_t vl) {
  assert(exponent > 0);
  assert(exponent <= 31);

  const vbool16_t nonpositive = __riscv_vmslt_vx_i64m4_b16(values, 1, vl);
  const vint64m4_t nudge = __riscv_vmerge_vxm_i64m4(
      __riscv_vmv_v_x_i64m4(static_cast<int64_t>(1) << (exponent - 1), vl),
      -(static_cast<int64_t>(1) << (exponent - 1)), nonpositive, vl);
  const vint64m4_t adjusted = __riscv_vadd_vv_i64m4(values, nudge, vl);
  vint32m2_t quotient = __riscv_vnsra_wx_i32m2(adjusted, exponent, vl);

  const vint64m4_t remainder = __riscv_vand_vx_i64m4(
      adjusted, (static_cast<int64_t>(1) << exponent) - 1, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i64m4_b16(adjusted, 0, vl);
  const vbool16_t has_remainder = __riscv_vmsne_vx_i64m4_b16(remainder, 0, vl);
  const vint32m2_t correction_mask = __riscv_vand_vv_i32m2(
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, negative, vl),
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, has_remainder,
                               vl),
      vl);
  return __riscv_vadd_vv_i32m2(quotient, correction_mask, vl);
}

inline void RvvAccumulateInt8Product(vint32m4_t* acc, const int8_t* lhs,
                                     const int8_t* rhs, size_t vl) {
  const vint8m1_t lhs8 = __riscv_vle8_v_i8m1(lhs, vl);
  const vint8m1_t rhs8 = __riscv_vle8_v_i8m1(rhs, vl);
  const vint16m2_t product16 = __riscv_vwmul_vv_i16m2(lhs8, rhs8, vl);
  *acc = __riscv_vadd_vv_i32m4(*acc, __riscv_vwadd_vx_i32m4(product16, 0, vl),
                               vl);
}

template <typename OutputT>
void RvvMatrixBatchVectorPostAccumulate(int32_t multiplier, int32_t shift,
                                        int32_t output_zp, int total_size,
                                        const int32_t* scratch,
                                        OutputT* output) {
  std::array<int32_t, kMaxRvvInt32Lanes> temp_i32;

  int col = 0;
  while (col < total_size) {
    const size_t vl = __riscv_vsetvl_e32m2(total_size - col);
    assert(static_cast<int>(vl) <= kMaxRvvInt32Lanes);
    vint32m2_t values = __riscv_vle32_v_i32m2(scratch + col, vl);
    values = RvvMultiplyByQuantizedMultiplier(values, multiplier, shift, vl);
    values = __riscv_vadd_vx_i32m2(values, output_zp, vl);
    __riscv_vse32_v_i32m2(temp_i32.data(), values, vl);

    for (size_t i = 0; i < vl; ++i) {
      int32_t acc = temp_i32[i] + output[col + static_cast<int>(i)];
      acc = Clamp<int32_t>(acc, std::numeric_limits<OutputT>::min(),
                           std::numeric_limits<OutputT>::max());
      output[col + static_cast<int>(i)] = static_cast<OutputT>(acc);
    }
    col += static_cast<int>(vl);
  }
}

float RvvReduceSumFloat(const float* data, int size) {
  if (size <= 0) {
    return 0.0f;
  }

  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  const int step = static_cast<int>(vlmax);
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(data + col, vlmax);
    acc = __riscv_vfadd_vv_f32m1(acc, values, vlmax);
  }

  float sum = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    sum += data[col];
  }

  return sum;
}

int32_t RvvReduceSumInt8(const int8_t* data, int size) {
  if (size <= 0) {
    return 0;
  }

  const size_t vlmax = __riscv_vsetvlmax_e8m1();
  const int step = static_cast<int>(vlmax);
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vint8m1_t values8 = __riscv_vle8_v_i8m1(data + col, vlmax);
    const vint16m2_t values16 = __riscv_vwadd_vx_i16m2(values8, 0, vlmax);
    const vint32m4_t values32 = __riscv_vwadd_vx_i32m4(values16, 0, vlmax);
    acc = __riscv_vadd_vv_i32m4(acc, values32, vlmax);
  }

  int32_t sum = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    sum += data[col];
  }

  return sum;
}

int32_t RvvInt8DotProduct(const int8_t* lhs, const int8_t* rhs, int size) {
  if (size <= 0) {
    return 0;
  }

  const size_t vlmax = __riscv_vsetvlmax_e8m1();
  const int step = static_cast<int>(vlmax);
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vint8m1_t lhs8 = __riscv_vle8_v_i8m1(lhs + col, vlmax);
    const vint8m1_t rhs8 = __riscv_vle8_v_i8m1(rhs + col, vlmax);
    const vint16m2_t prod16 = __riscv_vwmul_vv_i16m2(lhs8, rhs8, vlmax);
    const vint32m4_t prod32 = __riscv_vwadd_vx_i32m4(prod16, 0, vlmax);
    acc = __riscv_vadd_vv_i32m4(acc, prod32, vlmax);
  }

  int32_t dot = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    dot += lhs[col] * rhs[col];
  }

  return dot;
}

int32_t RvvReduceSumInt16(const int16_t* data, int size) {
  if (size <= 0) {
    return 0;
  }

  const size_t vlmax = __riscv_vsetvlmax_e16m1();
  const int step = static_cast<int>(vlmax);
  vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vint16m1_t values16 = __riscv_vle16_v_i16m1(data + col, vlmax);
    const vint32m2_t values32 = __riscv_vwadd_vx_i32m2(values16, 0, vlmax);
    acc = __riscv_vadd_vv_i32m2(acc, values32, vlmax);
  }

  int32_t sum = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    sum += data[col];
  }

  return sum;
}

int32_t RvvReduceSumInt32(const int32_t* data, int size) {
  if (size <= 0) {
    return 0;
  }

  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  const int step = static_cast<int>(vlmax);
  vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vint32m2_t values32 = __riscv_vle32_v_i32m2(data + col, vlmax);
    acc = __riscv_vadd_vv_i32m2(acc, values32, vlmax);
  }

  int32_t sum = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    sum += data[col];
  }

  return sum;
}

float RvvReduceSumSquareFloat(const float* data, int size) {
  if (size <= 0) {
    return 0.0f;
  }

  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  const int step = static_cast<int>(vlmax);
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(data + col, vlmax);
    const vfloat32m1_t squared = __riscv_vfmul_vv_f32m1(values, values, vlmax);
    acc = __riscv_vfadd_vv_f32m1(acc, squared, vlmax);
  }

  float sum = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    sum += data[col] * data[col];
  }

  return sum;
}

int32_t RvvInt8DotProductOffset(const int8_t* lhs, int32_t lhs_offset,
                                const int8_t* rhs, int size) {
  if (size <= 0) {
    return 0;
  }

  const size_t vlmax = __riscv_vsetvlmax_e8m1();
  const int step = static_cast<int>(vlmax);
  const int16_t offset16 = static_cast<int16_t>(lhs_offset);
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vint8m1_t lhs8 = __riscv_vle8_v_i8m1(lhs + col, vlmax);
    const vint8m1_t rhs8 = __riscv_vle8_v_i8m1(rhs + col, vlmax);
    vint16m2_t lhs16 = __riscv_vwadd_vx_i16m2(lhs8, 0, vlmax);
    lhs16 = __riscv_vsub_vx_i16m2(lhs16, offset16, vlmax);
    const vint16m2_t rhs16 = __riscv_vwadd_vx_i16m2(rhs8, 0, vlmax);
    const vint32m4_t prod32 = __riscv_vwmul_vv_i32m4(lhs16, rhs16, vlmax);
    acc = __riscv_vadd_vv_i32m4(acc, prod32, vlmax);
  }

  int32_t dot = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    dot += (static_cast<int32_t>(lhs[col]) - lhs_offset) * rhs[col];
  }

  return dot;
}

int32_t RvvInt16DotProduct(const int16_t* lhs, const int16_t* rhs, int size) {
  if (size <= 0) {
    return 0;
  }

  const size_t vlmax = __riscv_vsetvlmax_e16m1();
  const int step = static_cast<int>(vlmax);
  vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, vlmax);
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vint16m1_t lhs16 = __riscv_vle16_v_i16m1(lhs + col, vlmax);
    const vint16m1_t rhs16 = __riscv_vle16_v_i16m1(rhs + col, vlmax);
    const vint32m2_t prod32 = __riscv_vwmul_vv_i32m2(lhs16, rhs16, vlmax);
    acc = __riscv_vadd_vv_i32m2(acc, prod32, vlmax);
  }

  int32_t dot = HorizontalSum(acc, vlmax);
  for (; col < size; ++col) {
    dot += lhs[col] * rhs[col];
  }

  return dot;
}

int64_t RvvInt16Int8DotProduct(const int16_t* lhs, const int8_t* rhs,
                               int size) {
  if (size <= 0) {
    return 0;
  }

  const size_t vlmax = __riscv_vsetvlmax_e8m1();
  const int step = static_cast<int>(vlmax);
  int64_t dot = 0;
  int col = 0;
  const int full_cols = size / step * step;
  for (; col < full_cols; col += step) {
    const vint16m2_t lhs16 = __riscv_vle16_v_i16m2(lhs + col, vlmax);
    const vint8m1_t rhs8 = __riscv_vle8_v_i8m1(rhs + col, vlmax);
    const vint16m2_t rhs16 = __riscv_vwadd_vx_i16m2(rhs8, 0, vlmax);
    const vint32m4_t prod32 = __riscv_vwmul_vv_i32m4(lhs16, rhs16, vlmax);
    dot += HorizontalSum(prod32, vlmax);
  }

  for (; col < size; ++col) {
    dot += static_cast<int32_t>(lhs[col]) * static_cast<int32_t>(rhs[col]);
  }

  return dot;
}

vint32m2_t RvvRoundingDivideByPOT(vint32m2_t values, int exponent, size_t vl) {
  if (exponent <= 0) {
    return values;
  }

  assert(exponent <= 31);
  const uint32_t mask_unsigned = (static_cast<uint32_t>(1) << exponent) - 1u;
  const int32_t mask = static_cast<int32_t>(mask_unsigned);
  const int32_t threshold = mask >> 1;

  const vint32m2_t remainder = __riscv_vand_vx_i32m2(values, mask, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i32m2_b16(values, 0, vl);
  const vint32m2_t threshold_vec = __riscv_vmerge_vxm_i32m2(
      __riscv_vmv_v_x_i32m2(threshold, vl), threshold + 1, negative, vl);
  const vbool16_t increment_mask =
      __riscv_vmslt_vv_i32m2_b16(threshold_vec, remainder, vl);
  const vint32m2_t quotient = __riscv_vsra_vx_i32m2(values, exponent, vl);
  const vint32m2_t increment = __riscv_vmerge_vxm_i32m2(
      __riscv_vmv_v_x_i32m2(0, vl), 1, increment_mask, vl);
  return __riscv_vadd_vv_i32m2(quotient, increment, vl);
}

vint32m2_t RvvSaturatingRoundingDoublingHighMul(vint32m2_t values,
                                                int32_t multiplier,
                                                size_t vl) {
  static constexpr int64_t kPositiveNudge = static_cast<int64_t>(1) << 30;
  static constexpr int64_t kNegativeNudge = 1 - kPositiveNudge;
  const vint64m4_t product = __riscv_vwmul_vx_i64m4(values, multiplier, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i64m4_b16(product, 0, vl);
  const vint64m4_t nudge = __riscv_vmerge_vxm_i64m4(
      __riscv_vmv_v_x_i64m4(kPositiveNudge, vl), kNegativeNudge, negative, vl);
  const vint64m4_t nudged = __riscv_vadd_vv_i64m4(product, nudge, vl);
  vint32m2_t result = __riscv_vnsra_wx_i32m2(nudged, 31, vl);
  // gemmlowp uses truncating division here, while arithmetic right shift rounds
  // negative non-multiples toward -inf. Correct those lanes back by 1.
  const vint64m4_t remainder =
      __riscv_vand_vx_i64m4(nudged, (static_cast<int64_t>(1) << 31) - 1, vl);
  const vbool16_t has_remainder = __riscv_vmsne_vx_i64m4_b16(remainder, 0, vl);
  const vint32m2_t negative_i32 = __riscv_vmerge_vxm_i32m2(
      __riscv_vmv_v_x_i32m2(0, vl), 1, negative, vl);
  const vint32m2_t remainder_i32 = __riscv_vmerge_vxm_i32m2(
      __riscv_vmv_v_x_i32m2(0, vl), 1, has_remainder, vl);
  const vint32m2_t correction =
      __riscv_vand_vv_i32m2(negative_i32, remainder_i32, vl);
  return __riscv_vadd_vv_i32m2(result, correction, vl);
}

vint32m2_t RvvMultiplyByQuantizedMultiplier(vint32m2_t values,
                                            int32_t multiplier, int shift,
                                            size_t vl) {
#if TFLITE_SINGLE_ROUNDING
  assert(multiplier >= 0);
  assert(shift >= -31 && shift <= 30);
  const int total_shift = 31 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  const vint64m4_t product = __riscv_vwmul_vx_i64m4(values, multiplier, vl);
  return __riscv_vnsra_wx_i32m2(__riscv_vadd_vx_i64m4(product, round, vl),
                                total_shift, vl);
#else
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  if (left_shift > 0) {
    values = __riscv_vsll_vx_i32m2(values, left_shift, vl);
  }
  values = RvvSaturatingRoundingDoublingHighMul(values, multiplier, vl);
  if (right_shift > 0) {
    values = RvvRoundingDivideByPOT(values, right_shift, vl);
  }
  return values;
#endif
}

inline vint16m1_t RvvBroadcastI16(int16_t value, size_t vl) {
  return __riscv_vmv_v_x_i16m1(value, vl);
}

inline vint16m1_t RvvSelectI16(vbool16_t mask, vint16m1_t then_val,
                               vint16m1_t else_val, size_t vl) {
  return __riscv_vmerge_vvm_i16m1(else_val, then_val, mask, vl);
}

inline vint16m1_t RvvAddI16(vint16m1_t lhs, vint16m1_t rhs, size_t vl) {
  return __riscv_vadd_vv_i16m1(lhs, rhs, vl);
}

inline vint16m1_t RvvSubI16(vint16m1_t lhs, vint16m1_t rhs, size_t vl) {
  return __riscv_vsub_vv_i16m1(lhs, rhs, vl);
}

inline vint16m1_t RvvNegI16(vint16m1_t values, size_t vl) {
  return __riscv_vneg_v_i16m1(values, vl);
}

inline vint16m1_t RvvSaturatingAddI16(vint16m1_t lhs, vint16m1_t rhs,
                                      size_t vl) {
  return __riscv_vsadd_vv_i16m1(lhs, rhs, vl);
}

inline vbool16_t RvvMaskIfZeroI16(vint16m1_t values, size_t vl) {
  return __riscv_vmseq_vx_i16m1_b16(values, 0, vl);
}

inline vbool16_t RvvMaskIfGreaterThanZeroI16(vint16m1_t values, size_t vl) {
  const vbool16_t nonzero = __riscv_vmsne_vx_i16m1_b16(values, 0, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i16m1_b16(values, 0, vl);
  const vint16m1_t false_vec = RvvBroadcastI16(0, vl);
  const vint16m1_t true_vec = RvvBroadcastI16(-1, vl);
  const vint16m1_t nonzero_vec =
      __riscv_vmerge_vvm_i16m1(false_vec, true_vec, nonzero, vl);
  const vint16m1_t negative_vec =
      __riscv_vmerge_vvm_i16m1(false_vec, true_vec, negative, vl);
  const vint16m1_t positive_vec = __riscv_vand_vv_i16m1(
      nonzero_vec, __riscv_vxor_vx_i16m1(negative_vec, -1, vl), vl);
  return __riscv_vmsne_vx_i16m1_b16(positive_vec, 0, vl);
}

inline vbool16_t RvvMaskIfLessThanI16(vint16m1_t lhs, int16_t rhs, size_t vl) {
  return __riscv_vmslt_vx_i16m1_b16(lhs, rhs, vl);
}

inline vint16m1_t RvvRoundingHalfSumI16(vint16m1_t lhs, vint16m1_t rhs,
                                        size_t vl) {
  const vint32m2_t lhs32 = __riscv_vwadd_vx_i32m2(lhs, 0, vl);
  const vint32m2_t rhs32 = __riscv_vwadd_vx_i32m2(rhs, 0, vl);
  const vint32m2_t sum32 = __riscv_vadd_vv_i32m2(lhs32, rhs32, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i32m2_b16(sum32, 0, vl);
  const vint32m2_t sign32 = __riscv_vmerge_vxm_i32m2(
      __riscv_vmv_v_x_i32m2(1, vl), -1, negative, vl);
  const vint32m2_t adjusted = __riscv_vadd_vv_i32m2(sum32, sign32, vl);
  vint32m2_t quotient = __riscv_vsra_vx_i32m2(adjusted, 1, vl);
  const vint32m2_t remainder = __riscv_vand_vx_i32m2(adjusted, 1, vl);
  const vbool16_t has_remainder =
      __riscv_vmsne_vx_i32m2_b16(remainder, 0, vl);
  const vint32m2_t correction = __riscv_vand_vv_i32m2(
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, negative, vl),
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, has_remainder,
                               vl),
      vl);
  quotient = __riscv_vadd_vv_i32m2(quotient, correction, vl);
  return __riscv_vnclip_wx_i16m1(quotient, 0, __RISCV_VXRM_RDN, vl);
}

inline vint16m1_t RvvSaturatingRoundingDoublingHighMulI16(vint16m1_t lhs,
                                                          vint16m1_t rhs,
                                                          size_t vl) {
  const vint32m2_t product32 = __riscv_vwmul_vv_i32m2(lhs, rhs, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i32m2_b16(product32, 0, vl);
  const vint32m2_t nudge = __riscv_vmerge_vxm_i32m2(
      __riscv_vmv_v_x_i32m2(1 << 14, vl), 1 - (1 << 14), negative, vl);
  const vint32m2_t adjusted = __riscv_vadd_vv_i32m2(product32, nudge, vl);
  vint32m2_t result32 = __riscv_vsra_vx_i32m2(adjusted, 15, vl);
  const vint32m2_t remainder =
      __riscv_vand_vx_i32m2(adjusted, (1 << 15) - 1, vl);
  const vbool16_t has_remainder =
      __riscv_vmsne_vx_i32m2_b16(remainder, 0, vl);
  const vint32m2_t correction = __riscv_vand_vv_i32m2(
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, negative, vl),
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, has_remainder,
                               vl),
      vl);
  result32 = __riscv_vadd_vv_i32m2(result32, correction, vl);
  vint16m1_t result =
      __riscv_vnclip_wx_i16m1(result32, 0, __RISCV_VXRM_RDN, vl);
  const vint16m1_t false_vec = RvvBroadcastI16(0, vl);
  const vint16m1_t true_vec = RvvBroadcastI16(-1, vl);
  const vint16m1_t min_mask = __riscv_vmerge_vvm_i16m1(
      false_vec, true_vec,
      __riscv_vmseq_vx_i16m1_b16(lhs, std::numeric_limits<int16_t>::min(), vl),
      vl);
  const vint16m1_t equal_mask = __riscv_vmerge_vvm_i16m1(
      false_vec, true_vec, __riscv_vmseq_vv_i16m1_b16(lhs, rhs, vl), vl);
  const vbool16_t overflow = __riscv_vmsne_vx_i16m1_b16(
      __riscv_vand_vv_i16m1(min_mask, equal_mask, vl), 0, vl);
  return __riscv_vmerge_vxm_i16m1(
      result, std::numeric_limits<int16_t>::max(), overflow, vl);
}

template <int IntegerBits>
constexpr int RvvFixedPointFractionalBits() {
  static_assert(IntegerBits >= 0);
  static_assert(IntegerBits <= 15);
  return 15 - IntegerBits;
}

template <int IntegerBits>
constexpr int16_t RvvFixedPointOneRaw() {
  if constexpr (IntegerBits == 0) {
    return std::numeric_limits<int16_t>::max();
  } else {
    return static_cast<int16_t>(1 << RvvFixedPointFractionalBits<IntegerBits>());
  }
}

template <int IntegerBits, int Exponent>
constexpr int16_t RvvFixedPointConstantPOTRaw() {
  static_assert(RvvFixedPointFractionalBits<IntegerBits>() + Exponent >= 0);
  return static_cast<int16_t>(
      1 << (RvvFixedPointFractionalBits<IntegerBits>() + Exponent));
}

template <int Exponent>
inline vint16m1_t RvvSaturatingRoundingMultiplyByPOTI16(vint16m1_t values,
                                                        size_t vl) {
  if constexpr (Exponent == 0) {
    return values;
  } else if constexpr (Exponent > 0) {
    vint32m2_t wide = __riscv_vwadd_vx_i32m2(values, 0, vl);
    wide = __riscv_vsll_vx_i32m2(wide, Exponent, vl);
    wide = __riscv_vmax_vx_i32m2(
        wide, std::numeric_limits<int16_t>::min(), vl);
    wide = __riscv_vmin_vx_i32m2(
        wide, std::numeric_limits<int16_t>::max(), vl);
    return __riscv_vnclip_wx_i16m1(wide, 0, __RISCV_VXRM_RDN, vl);
  } else {
    vint32m2_t wide = __riscv_vwadd_vx_i32m2(values, 0, vl);
    wide = RvvRoundingDivideByPOT(wide, -Exponent, vl);
    return __riscv_vnclip_wx_i16m1(wide, 0, __RISCV_VXRM_RDN, vl);
  }
}

template <int DstIntegerBits, int SrcIntegerBits>
inline vint16m1_t RvvRescaleI16(vint16m1_t values, size_t vl) {
  constexpr int kExponent = SrcIntegerBits - DstIntegerBits;
  return RvvSaturatingRoundingMultiplyByPOTI16<kExponent>(values, vl);
}

inline vint16m1_t RvvMulFixedPointI16(vint16m1_t lhs, vint16m1_t rhs,
                                      size_t vl) {
  return RvvSaturatingRoundingDoublingHighMulI16(lhs, rhs, vl);
}

template <int IntegerBits>
inline vint16m1_t RvvExpOnIntervalBetweenNegativeOneQuarterAndZeroExcl(
    vint16m1_t values, size_t vl) {
  static_assert(IntegerBits == 0);
  const vint16m1_t constant_term = RvvBroadcastI16(28918, vl);
  const vint16m1_t constant_one_over_three = RvvBroadcastI16(10923, vl);
  const vint16m1_t x = __riscv_vadd_vx_i16m1(
      values, RvvFixedPointConstantPOTRaw<0, -3>(), vl);
  const vint16m1_t x2 = RvvMulFixedPointI16(x, x, vl);
  const vint16m1_t x3 = RvvMulFixedPointI16(x2, x, vl);
  const vint16m1_t x4 = RvvMulFixedPointI16(x2, x2, vl);
  const vint16m1_t x4_over_4 =
      RvvSaturatingRoundingMultiplyByPOTI16<-2>(x4, vl);
  const vint16m1_t poly_term = RvvSaturatingRoundingMultiplyByPOTI16<-1>(
      RvvAddI16(
          RvvMulFixedPointI16(RvvAddI16(x4_over_4, x3, vl),
                              constant_one_over_three, vl),
          x2, vl),
      vl);
  return RvvSaturatingAddI16(
      constant_term,
      RvvMulFixedPointI16(constant_term, RvvAddI16(x, poly_term, vl), vl), vl);
}

template <int IntegerBits, int Exponent, int MultiplierRaw>
inline vint16m1_t RvvApplyExpBarrelShifter(vint16m1_t remainder,
                                           vint16m1_t result, size_t vl) {
  if constexpr (IntegerBits > Exponent) {
    constexpr int kShiftAmount =
        RvvFixedPointFractionalBits<IntegerBits>() + Exponent;
    const vint16m1_t bit =
        __riscv_vand_vx_i16m1(remainder, 1 << kShiftAmount, vl);
    const vbool16_t bit_is_set = __riscv_vmsne_vx_i16m1_b16(bit, 0, vl);
    const vint16m1_t shifted = RvvMulFixedPointI16(
        result, RvvBroadcastI16(static_cast<int16_t>(MultiplierRaw), vl), vl);
    return RvvSelectI16(bit_is_set, shifted, result, vl);
  } else {
    return result;
  }
}

template <int IntegerBits>
inline vint16m1_t RvvExpOnNegativeValues(vint16m1_t values, size_t vl) {
  const int16_t one_quarter = RvvFixedPointConstantPOTRaw<IntegerBits, -2>();
  const vint16m1_t mask = RvvBroadcastI16(static_cast<int16_t>(one_quarter - 1),
                                          vl);
  const vint16m1_t values_mod_quarter_minus_one_quarter =
      __riscv_vsub_vx_i16m1(__riscv_vand_vv_i16m1(values, mask, vl),
                            one_quarter, vl);
  vint16m1_t result = RvvExpOnIntervalBetweenNegativeOneQuarterAndZeroExcl<0>(
      RvvRescaleI16<0, IntegerBits>(values_mod_quarter_minus_one_quarter, vl),
      vl);
  const vint16m1_t remainder =
      __riscv_vsub_vv_i16m1(values_mod_quarter_minus_one_quarter, values, vl);

  result = RvvApplyExpBarrelShifter<IntegerBits, -2, 25520>(remainder, result,
                                                             vl);
  result = RvvApplyExpBarrelShifter<IntegerBits, -1, 19875>(remainder, result,
                                                             vl);
  result = RvvApplyExpBarrelShifter<IntegerBits, 0, 12055>(remainder, result,
                                                            vl);
  result = RvvApplyExpBarrelShifter<IntegerBits, 1, 4435>(remainder, result,
                                                           vl);
  result = RvvApplyExpBarrelShifter<IntegerBits, 2, 600>(remainder, result,
                                                          vl);
  result = RvvApplyExpBarrelShifter<IntegerBits, 3, 11>(remainder, result, vl);
  result = RvvApplyExpBarrelShifter<IntegerBits, 4, 0>(remainder, result, vl);

  if constexpr (IntegerBits > 5) {
    constexpr int kClampRaw = -(1 << (20 - IntegerBits));
    const vbool16_t clamp_mask = RvvMaskIfLessThanI16(values, kClampRaw, vl);
    result = RvvSelectI16(clamp_mask, RvvBroadcastI16(0, vl), result, vl);
  }

  return RvvSelectI16(RvvMaskIfZeroI16(values, vl),
                      RvvBroadcastI16(RvvFixedPointOneRaw<0>(), vl), result,
                      vl);
}

inline vint16m1_t RvvOneMinusXOverOnePlusXForXInZeroOne(vint16m1_t values,
                                                        size_t vl) {
  const vint16m1_t one_q0 = RvvBroadcastI16(RvvFixedPointOneRaw<0>(), vl);
  const vint16m1_t one_q2 = RvvBroadcastI16(RvvFixedPointOneRaw<2>(), vl);
  const vint16m1_t constant_48_over_17 = RvvBroadcastI16(23130, vl);
  const vint16m1_t constant_neg_32_over_17 = RvvBroadcastI16(-15420, vl);
  const vint16m1_t half_denominator =
      RvvRoundingHalfSumI16(values, one_q0, vl);
  vint16m1_t x = RvvAddI16(
      constant_48_over_17,
      RvvMulFixedPointI16(half_denominator, constant_neg_32_over_17, vl), vl);

  for (int iteration = 0; iteration < 3; ++iteration) {
    const vint16m1_t half_denominator_times_x =
        RvvMulFixedPointI16(half_denominator, x, vl);
    const vint16m1_t one_minus_half_denominator_times_x =
        RvvSubI16(one_q2, half_denominator_times_x, vl);
    x = RvvAddI16(
        x,
        RvvRescaleI16<2, 4>(
            RvvMulFixedPointI16(x, one_minus_half_denominator_times_x, vl),
            vl),
        vl);
  }
  return RvvRescaleI16<0, 2>(RvvSubI16(x, one_q2, vl), vl);
}

inline vint16m1_t RvvOneOverOnePlusXForXInZeroOne(vint16m1_t values,
                                                  size_t vl) {
  const vint16m1_t one_q0 = RvvBroadcastI16(RvvFixedPointOneRaw<0>(), vl);
  const vint16m1_t one_q2 = RvvBroadcastI16(RvvFixedPointOneRaw<2>(), vl);
  const vint16m1_t constant_48_over_17 = RvvBroadcastI16(23130, vl);
  const vint16m1_t constant_neg_32_over_17 = RvvBroadcastI16(-15420, vl);
  const vint16m1_t half_denominator =
      RvvRoundingHalfSumI16(values, one_q0, vl);
  vint16m1_t x = RvvAddI16(
      constant_48_over_17,
      RvvMulFixedPointI16(half_denominator, constant_neg_32_over_17, vl), vl);

  for (int iteration = 0; iteration < 3; ++iteration) {
    const vint16m1_t half_denominator_times_x =
        RvvMulFixedPointI16(half_denominator, x, vl);
    const vint16m1_t one_minus_half_denominator_times_x =
        RvvSubI16(one_q2, half_denominator_times_x, vl);
    x = RvvAddI16(
        x,
        RvvRescaleI16<2, 4>(
            RvvMulFixedPointI16(x, one_minus_half_denominator_times_x, vl),
            vl),
        vl);
  }
  // Matches Rescale<0>(ExactMulByPot<-1>(x)) in gemmlowp.
  return RvvSaturatingRoundingMultiplyByPOTI16<1>(x, vl);
}

template <int IntegerBits>
inline vint16m1_t RvvNegTanhOnNegativeValues(vint16m1_t values, size_t vl) {
  return RvvOneMinusXOverOnePlusXForXInZeroOne(
      RvvExpOnNegativeValues<IntegerBits + 1>(values, vl), vl);
}

template <int IntegerBits>
inline vint16m1_t RvvApplyTanhVector(vint16m1_t values, size_t vl) {
  const vbool16_t mask_if_negative = RvvMaskIfLessThanI16(values, 0, vl);
  const vbool16_t mask_if_zero = RvvMaskIfZeroI16(values, vl);
  const vint16m1_t negated = RvvNegI16(values, vl);
  const vint16m1_t negative_input =
      RvvSelectI16(mask_if_negative, values, negated, vl);
  const vint16m1_t negative_tanh =
      RvvNegTanhOnNegativeValues<IntegerBits>(negative_input, vl);
  const vint16m1_t signed_result =
      RvvSelectI16(mask_if_negative, RvvNegI16(negative_tanh, vl),
                   negative_tanh, vl);
  return RvvSelectI16(mask_if_zero, RvvBroadcastI16(0, vl), signed_result, vl);
}

template <int IntegerBits>
inline vint16m1_t RvvLogisticOnPositiveValues(vint16m1_t values, size_t vl) {
  return RvvOneOverOnePlusXForXInZeroOne(
      RvvExpOnNegativeValues<IntegerBits>(RvvNegI16(values, vl), vl), vl);
}

template <int IntegerBits>
inline vint16m1_t RvvApplyLogisticVector(vint16m1_t values, size_t vl) {
  const vbool16_t mask_if_positive = RvvMaskIfGreaterThanZeroI16(values, vl);
  const vbool16_t mask_if_zero = RvvMaskIfZeroI16(values, vl);
  const vint16m1_t absolute_input =
      RvvSelectI16(mask_if_positive, values, RvvNegI16(values, vl), vl);
  const vint16m1_t result_if_positive =
      RvvLogisticOnPositiveValues<IntegerBits>(absolute_input, vl);
  const vint16m1_t result_if_negative = RvvSubI16(
      RvvBroadcastI16(RvvFixedPointOneRaw<0>(), vl), result_if_positive, vl);
  const vint16m1_t one_half = RvvBroadcastI16(16384, vl);
  return RvvSelectI16(
      mask_if_zero, one_half,
      RvvSelectI16(mask_if_positive, result_if_positive, result_if_negative,
                   vl),
      vl);
}

}  // namespace

bool RvvIsZeroVector(const float* vector, int v_size) {
  int col = 0;
  while (col < v_size) {
    const size_t vl = __riscv_vsetvl_e32m1(v_size - col);
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + col, vl);
    const vbool32_t nonzero = __riscv_vmfne_vf_f32m1_b32(values, 0.0f, vl);
    if (__riscv_vfirst_m_b32(nonzero, vl) >= 0) {
      return false;
    }
    col += static_cast<int>(vl);
  }
  return true;
}

bool RvvIsZeroVector(const int8_t* vector, int v_size) {
  int col = 0;
  while (col < v_size) {
    const size_t vl = __riscv_vsetvl_e8m1(v_size - col);
    const vint8m1_t values = __riscv_vle8_v_i8m1(vector + col, vl);
    const vbool8_t nonzero = __riscv_vmsne_vx_i8m1_b8(values, 0, vl);
    if (__riscv_vfirst_m_b8(nonzero, vl) >= 0) {
      return false;
    }
    col += static_cast<int>(vl);
  }
  return true;
}

void RvvSymmetricQuantizeFloats(const float* values, int size,
                                int8_t* quantized_values, float* min_value,
                                float* max_value, float* scaling_factor) {
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;

  RvvSymmetricQuantizeFloats(values, size, quantized_values, *min_value,
                             *max_value, scaling_factor);
}

void RvvSymmetricQuantizeFloats(const float* values, int size,
                                int8_t* quantized_values, float min_value,
                                float max_value, float* scaling_factor) {
  constexpr int32_t kScale = 127;
  const float range = std::max(std::abs(min_value), std::abs(max_value));
  if (range == 0.0f) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1.0f;
    return;
  }

  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  RvvQuantizeFloats(values, size, scaling_factor_inv, /*offset=*/0, -kScale,
                    kScale, quantized_values);
}

void RvvAsymmetricQuantizeFloats(const float* values, int size,
                                 int8_t* quantized_values,
                                 float* scaling_factor, int32_t* offset) {
  constexpr int32_t kMinScale = -128;
  constexpr int32_t kMaxScale = 127;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const auto minmax = std::minmax_element(values, values + size);
  const double rmin = static_cast<double>(std::min(0.0f, *minmax.first));
  const double rmax = static_cast<double>(std::max(0.0f, *minmax.second));
  if (rmin == rmax) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1.0f;
    *offset = 0;
    return;
  }

  const double scale = (rmax - rmin) / (qmax_double - qmin_double);
  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);
  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  int8_t nudged_zero_point = 0;
  if (zero_point_double <= qmin_double) {
    nudged_zero_point = kMinScale;
  } else if (zero_point_double >= qmax_double) {
    nudged_zero_point = kMaxScale;
  } else {
    nudged_zero_point = static_cast<int8_t>(round(zero_point_double));
  }

  *scaling_factor = scale;
  *offset = nudged_zero_point;

  const float scaling_factor_inv = 1.0f / *scaling_factor;
  RvvQuantizeFloats(values, size, scaling_factor_inv, *offset, kMinScale,
                    kMaxScale, quantized_values);
}

float RvvVectorVectorDotProduct(const float* vector1, const float* vector2,
                                int v_size) {
  if (v_size <= 0) {
    return 0.0f;
  }

  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  const int step = static_cast<int>(vlmax);
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  int col = 0;
  const int full_cols = v_size / step * step;
  for (; col < full_cols; col += step) {
    const vfloat32m1_t lhs_vec = __riscv_vle32_v_f32m1(vector1 + col, vlmax);
    const vfloat32m1_t rhs_vec = __riscv_vle32_v_f32m1(vector2 + col, vlmax);
    acc = __riscv_vfmacc_vv_f32m1(acc, lhs_vec, rhs_vec, vlmax);
  }

  float dot_product = HorizontalSum(acc, vlmax);
  for (; col < v_size; ++col) {
    dot_product += vector1[col] * vector2[col];
  }

  return dot_product;
}

void RvvMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                            int m_cols, const float* vector,
                                            int n_batch, float* result) {
  float* result_in_batch = result;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float* vector_in_batch = vector + batch * m_cols;
    const float* matrix_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
      *result_in_batch +=
          RvvVectorVectorDotProduct(matrix_ptr, vector_in_batch, m_cols);
      matrix_ptr += m_cols;
      ++result_in_batch;
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t* matrix, int m_rows,
                                            int m_cols, const int8_t* vectors,
                                            const float* scaling_factors,
                                            int n_batch, float* result) {
  float* result_in_batch = result;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    const int8_t* vector_in_batch = vectors + batch * m_cols;
    const int8_t* matrix_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
      const int32_t dotprod =
          RvvInt8DotProduct(matrix_ptr, vector_in_batch, m_cols);
      *result_in_batch += dotprod * batch_scaling_factor;
      matrix_ptr += m_cols;
      ++result_in_batch;
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* matrix, int m_rows, int m_cols, const int8_t* vectors,
    const float* scaling_factors, int n_batch, int32_t* scratch,
    float* result, CpuBackendContext* context) {
  (void)scratch;
  (void)context;
  RvvMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                         scaling_factors, n_batch, result);
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context) {
  (void)context;
  std::vector<int32_t> local_scratch;
  if (scratch == nullptr) {
    local_scratch.resize(n_batch * n_output);
    scratch = local_scratch.data();
  }

  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t* input_ptr = input + batch * n_input;
    for (int row = 0; row < n_output; ++row) {
      int32_t acc = bias != nullptr ? bias[row] : 0;
      acc += RvvInt8DotProduct(input_ptr, input_to_gate_weights + row * n_input,
                               n_input);
      scratch[batch * n_output + row] = acc;
    }
  }

  RvvMatrixBatchVectorPostAccumulate(multiplier, shift, output_zp,
                                     n_batch * n_output, scratch, output);
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context) {
  (void)context;
  std::vector<int32_t> local_scratch;
  if (scratch == nullptr) {
    local_scratch.resize(n_batch * n_output);
    scratch = local_scratch.data();
  }

  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t* input_ptr = input + batch * n_input;
    for (int row = 0; row < n_output; ++row) {
      int32_t acc = bias != nullptr ? bias[row] : 0;
      acc += RvvInt8DotProduct(input_ptr, input_to_gate_weights + row * n_input,
                               n_input);
      scratch[batch * n_output + row] = acc;
    }
  }

  RvvMatrixBatchVectorPostAccumulate(multiplier, shift, output_zp,
                                     n_batch * n_output, scratch, output);
}

void RvvMatrixBatchVectorMultiply(const int8_t* input, int32_t input_zeropoint,
                                  const int8_t* input_to_gate_weights,
                                  int32_t input_to_gate_effective_scale_a,
                                  int32_t input_to_gate_effective_scale_b,
                                  int32_t n_batch, int32_t n_input,
                                  int32_t n_cell, int8_t* gate_output,
                                  int8_t gate_output_zp) {
  const int32_t int8_max = std::numeric_limits<int8_t>::max();
  const int32_t int8_min = std::numeric_limits<int8_t>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t* input_ptr = input + batch * n_input;
    for (int row = 0; row < n_cell; ++row) {
      const int8_t* row_weights = input_to_gate_weights + row * n_input;
      int32_t acc =
          RvvInt8DotProductOffset(input_ptr, input_zeropoint, row_weights,
                                  n_input);
      acc = MultiplyByQuantizedMultiplier(acc,
                                          input_to_gate_effective_scale_a,
                                          input_to_gate_effective_scale_b);
      acc += gate_output_zp;
      acc = Clamp<int32_t>(acc, int8_min, int8_max);
      gate_output[batch * n_cell + row] = static_cast<int8_t>(acc);
    }
  }
}

void RvvMatrixBatchVectorMultiply(const int16_t* hidden,
                                  const int8_t* hidden_to_output_weights,
                                  int32_t proj_effective_scale_a,
                                  int32_t proj_effective_scale_b,
                                  const int32_t* gate_bias, int32_t n_batch,
                                  int32_t n_hidden, int32_t n_output,
                                  int32_t output_zp, int8_t* proj_output) {
  const int32_t int8_max = std::numeric_limits<int8_t>::max();
  const int32_t int8_min = std::numeric_limits<int8_t>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    const int16_t* hidden_ptr = hidden + batch * n_hidden;
    for (int row = 0; row < n_output; ++row) {
      int64_t acc = gate_bias != nullptr ? gate_bias[row] : 0;
      acc += RvvInt16Int8DotProduct(hidden_ptr,
                                    hidden_to_output_weights + row * n_hidden,
                                    n_hidden);
      int32_t scaled = MultiplyByQuantizedMultiplier(
          acc, proj_effective_scale_a, proj_effective_scale_b);
      scaled += output_zp;
      scaled = Clamp<int32_t>(scaled, int8_min, int8_max);
      proj_output[batch * n_output + row] = static_cast<int8_t>(scaled);
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* matrix, int m_rows, int m_cols, const int8_t* vectors,
    const float* scaling_factors, int n_batch, float* result,
    const float* per_channel_scale, const int32_t* input_offset,
    int32_t* scratch, int32_t* row_sums, bool* compute_row_sums,
    CpuBackendContext* context) {
  (void)scratch;
  (void)context;

  if (input_offset == nullptr) {
    RvvMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                           scaling_factors, n_batch, result);
    return;
  }

  int32_t* row_sums_ptr = row_sums;
  std::vector<int32_t> local_row_sums;
  if (row_sums_ptr == nullptr) {
    local_row_sums.resize(m_rows);
    row_sums_ptr = local_row_sums.data();
  }
  if (row_sums_ptr == nullptr) {
    return;
  }

  if (row_sums == nullptr || !compute_row_sums || *compute_row_sums) {
    RvvReductionSumVector(matrix, row_sums_ptr, m_rows, m_cols);
    if (compute_row_sums) {
      *compute_row_sums = false;
    }
  }

  float* result_in_batch = result;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    const int32_t batch_offset = input_offset[batch];
    const int8_t* vector_in_batch = vectors + batch * m_cols;
    const int8_t* matrix_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
      int32_t dotprod = RvvInt8DotProduct(matrix_ptr, vector_in_batch, m_cols);
      dotprod -= row_sums_ptr[row] * batch_offset;
      float scale = batch_scaling_factor;
      if (per_channel_scale != nullptr) {
        scale *= per_channel_scale[row];
      }
      *result_in_batch += dotprod * scale;
      matrix_ptr += m_cols;
      ++result_in_batch;
    }
  }
}

void RvvMatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                       int32_t n_row, int32_t n_col,
                                       int32_t* output) {
  for (int row = 0; row < n_row; ++row) {
    const int32_t row_sum = RvvReduceSumInt8(matrix + row * n_col, n_col);
    output[row] += row_sum * scalar;
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* matrix, const int32_t* segments, const int32_t* indices,
    int m_rows, int m_cols, const float* vector, int n_batch, float* result) {
  constexpr int kBlockSize = 4;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);
  const size_t vl = __riscv_vsetvl_e32m1(kBlockSize);

  for (int batch = 0; batch < n_batch; ++batch) {
    const float* matrix_ptr = matrix;
    const float* vector_in_batch = vector + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      for (int i = segments[row]; i < segments[row + 1]; ++i) {
        const int block_start_index = indices[i] * kBlockSize;
        const vfloat32m1_t vector_block =
            __riscv_vle32_v_f32m1(vector_in_batch + block_start_index, vl);
        const vfloat32m1_t matrix_block =
            __riscv_vle32_v_f32m1(matrix_ptr, vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, matrix_block, vector_block, vl);
        matrix_ptr += kBlockSize;
      }
      result[batch * m_rows + row] += HorizontalSum(acc, vl);
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const float* matrix, const uint8_t* ledger, int m_rows, int m_cols,
    const float* vector, int n_batch, float* result) {
  constexpr int kVectorsPerBlock = 4;
  constexpr int kBlockSize = 16;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);
  const size_t vl = __riscv_vsetvl_e32m1(4);

  for (int batch = 0; batch < n_batch; ++batch) {
    const float* matrix_ptr = matrix;
    const uint8_t* ledger_ptr = ledger;
    const float* vector_in_batch = vector + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      const int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks == 0) {
        continue;
      }

      vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      for (int i = 0; i < num_nonzero_blocks; ++i) {
        const int block_start_index = *ledger_ptr++ * kBlockSize;
        const float* vector_block_ptr = vector_in_batch + block_start_index;
        for (int c = 0; c < kVectorsPerBlock; ++c) {
          const vfloat32m1_t vector_block =
              __riscv_vle32_v_f32m1(vector_block_ptr + c * 4, vl);
          const vfloat32m1_t matrix_block =
              __riscv_vle32_v_f32m1(matrix_ptr + c * 4, vl);
          acc = __riscv_vfmacc_vv_f32m1(acc, matrix_block, vector_block, vl);
        }
        matrix_ptr += kBlockSize;
      }
      result[batch * m_rows + row] += HorizontalSum(acc, vl);
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t* matrix, const int32_t* segments, const int32_t* indices,
    int m_rows, int m_cols, const int8_t* vector, const int32_t* bias_vector,
    int n_batch, int32_t input_offset, int32_t output_multiplier,
    int32_t output_shift, const int32_t* per_channel_scale,
    const int32_t* per_channel_shift, int32_t output_offset,
    int32_t output_activation_min, int32_t output_activation_max,
    int8_t* result) {
  constexpr int kBlockSize = 16;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);
  const size_t vl = __riscv_vsetvl_e8m1(kBlockSize);
  std::vector<int32_t> row_sums;
  if (input_offset != 0) {
    row_sums.resize(m_rows);
    const int8_t* row_matrix_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
      const int block_count = segments[row + 1] - segments[row];
      row_sums[row] = RvvReduceSumInt8(row_matrix_ptr, block_count * kBlockSize);
      row_matrix_ptr += block_count * kBlockSize;
    }
  }

  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t* matrix_ptr = matrix;
    const int8_t* vector_in_batch = vector + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      vint32m4_t dot_acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t dot_acc1 = __riscv_vmv_v_x_i32m4(0, vl);

      int i = segments[row];
      const int row_end = segments[row + 1];
      for (; i + 1 < row_end; i += 2) {
        const int block_start_index = indices[i] * kBlockSize;
        RvvAccumulateInt8Product(&dot_acc0, vector_in_batch + block_start_index,
                                 matrix_ptr, vl);
        matrix_ptr += kBlockSize;

        const int next_block_start_index = indices[i + 1] * kBlockSize;
        RvvAccumulateInt8Product(&dot_acc1,
                                 vector_in_batch + next_block_start_index,
                                 matrix_ptr, vl);
        matrix_ptr += kBlockSize;
      }
      if (i < row_end) {
        const int block_start_index = indices[i] * kBlockSize;
        RvvAccumulateInt8Product(&dot_acc0, vector_in_batch + block_start_index,
                                 matrix_ptr, vl);
        matrix_ptr += kBlockSize;
      }

      const vint32m4_t dot_acc = __riscv_vadd_vv_i32m4(dot_acc0, dot_acc1, vl);
      int32_t acc = HorizontalSum(dot_acc, vl);
      const int32_t bias_value = bias_vector != nullptr ? bias_vector[row] : 0;
      acc += bias_value;
      if (input_offset != 0) {
        acc += input_offset * row_sums[row];
      }
      acc = MultiplyByQuantizedMultiplier(
          acc, per_channel_scale != nullptr ? per_channel_scale[row]
                                            : output_multiplier,
          per_channel_shift != nullptr ? per_channel_shift[row] : output_shift);
      acc += output_offset;
      result[batch * m_rows + row] =
          static_cast<int8_t>(ActivationFunctionWithMinMax(
              acc, output_activation_min, output_activation_max));
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* matrix, const uint8_t* ledger, int m_rows, int m_cols,
    const int8_t* vectors, const float* scaling_factors, int n_batch,
    float* result, const float* per_channel_scale) {
  constexpr int kBlockSize = 16;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);
  const size_t vl = __riscv_vsetvl_e8m1(kBlockSize);

  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    const uint8_t* ledger_ptr = ledger;
    const int8_t* row_ptr = matrix;
    const int8_t* vector_in_batch = vectors + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      vint32m4_t dot_acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t dot_acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      const int num_nonzero_blocks = *ledger_ptr++;
      int i = 0;
      for (; i + 1 < num_nonzero_blocks; i += 2) {
        const int block_start_index = *ledger_ptr++ * kBlockSize;
        RvvAccumulateInt8Product(&dot_acc0, vector_in_batch + block_start_index,
                                 row_ptr, vl);
        row_ptr += kBlockSize;

        const int next_block_start_index = *ledger_ptr++ * kBlockSize;
        RvvAccumulateInt8Product(&dot_acc1,
                                 vector_in_batch + next_block_start_index,
                                 row_ptr, vl);
        row_ptr += kBlockSize;
      }
      if (i < num_nonzero_blocks) {
        const int block_start_index = *ledger_ptr++ * kBlockSize;
        RvvAccumulateInt8Product(&dot_acc0, vector_in_batch + block_start_index,
                                 row_ptr, vl);
        row_ptr += kBlockSize;
      }
      float scaling_factor = batch_scaling_factor;
      if (per_channel_scale != nullptr) {
        scaling_factor *= per_channel_scale[row];
      }
      const vint32m4_t dot_acc =
          __riscv_vadd_vv_i32m4(dot_acc0, dot_acc1, vl);
      result[batch * m_rows + row] += HorizontalSum(dot_acc, vl) * scaling_factor;
    }
  }
}

void RvvApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                       const int32_t* bias, int32_t layer_norm_scale_a,
                       int32_t layer_norm_scale_b, int32_t variance_limit,
                       int n_batch, int n_input, int16_t* output) {
  static constexpr int kTwoToPower20 = 1 << 20;
  const size_t vlmax = __riscv_vsetvlmax_e16m1();

  for (int batch = 0; batch < n_batch; ++batch) {
    const int base = batch * n_input;
    int64_t sum = 0;
    int64_t sum_sq = 0;

    int col = 0;
    while (col < n_input) {
      const size_t vl = __riscv_vsetvl_e16m1(n_input - col);
      const vint16m1_t values16 =
          __riscv_vle16_v_i16m1(input + base + col, vl);
      const vint32m2_t values32 = __riscv_vwadd_vx_i32m2(values16, 0, vl);
      const vint64m4_t squared64 =
          __riscv_vwmul_vv_i64m4(values32, values32, vl);
      sum += HorizontalSum(values32, vl);
      sum_sq += HorizontalSum(squared64, vl);
      col += static_cast<int>(vl);
    }

    int32_t mean = static_cast<int32_t>(sum * 1024 / n_input);
    const int32_t temp = kTwoToPower20 / n_input;
    int64_t variance =
        sum_sq * temp - static_cast<int64_t>(mean) * static_cast<int64_t>(mean);
    int32_t variance2 = static_cast<int32_t>(variance / kTwoToPower20);
    if (variance2 < 1) {
      variance2 = variance_limit;
    }
    int32_t stddev_inverse_a = 0;
    int stddev_inverse_b = 0;
    GetInvSqrtQuantizedMultiplierExp(variance2, /*reverse_shift=*/-1,
                                     &stddev_inverse_a, &stddev_inverse_b);

    col = 0;
    while (col < n_input) {
      const size_t vl = __riscv_vsetvl_e16m1(n_input - col);
      const vint16m1_t input16 = __riscv_vle16_v_i16m1(input + base + col, vl);
      const vint16m1_t weights16 =
          __riscv_vle16_v_i16m1(layer_norm_weights + col, vl);

      vint32m2_t shifted =
          __riscv_vwadd_vx_i32m2(input16, 0, vl);
      shifted = __riscv_vsll_vx_i32m2(shifted, 10, vl);
      shifted = __riscv_vsub_vx_i32m2(shifted, mean, vl);

      const vint32m2_t rescaled = RvvMultiplyByQuantizedMultiplier(
          shifted, stddev_inverse_a, stddev_inverse_b, vl);
      const vint32m2_t weights32 = __riscv_vwadd_vx_i32m2(weights16, 0, vl);
      vint64m4_t val3 = __riscv_vwmul_vv_i64m4(rescaled, weights32, vl);
      if (bias != nullptr) {
        const vint32m2_t bias32 = __riscv_vle32_v_i32m2(bias + col, vl);
        val3 = __riscv_vadd_vv_i64m4(val3, __riscv_vwadd_vx_i64m4(bias32, 0, vl),
                                     vl);
      }

      vint32m2_t val4 = RvvRoundingDivideByPOT(val3, 10, vl);
      vint32m2_t val5 = RvvMultiplyByQuantizedMultiplier(
          val4, layer_norm_scale_a, layer_norm_scale_b + 12, vl);
      val5 = __riscv_vmax_vx_i32m2(
          val5, std::numeric_limits<int16_t>::min(), vl);
      val5 = __riscv_vmin_vx_i32m2(
          val5, std::numeric_limits<int16_t>::max(), vl);
      const vint16m1_t output16 =
          __riscv_vnclip_wx_i16m1(val5, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse16_v_i16m1(output + base + col, output16, vl);
      col += static_cast<int>(vl);
    }
  }
}

void RvvApplyLayerNormFloat(const int16_t* input,
                            const int16_t* layer_norm_weights,
                            int32_t layer_norm_scale_a,
                            int32_t layer_norm_scale_b, const int32_t* bias,
                            int n_batch, int n_input, int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16_t>::max();
  const int32_t int16_min = std::numeric_limits<int16_t>::min();
  static constexpr float kOutputScale = static_cast<float>(1 << 12);
  const float layer_norm_scale =
      layer_norm_scale_a *
      std::pow(2.0, static_cast<double>(layer_norm_scale_b - 31));
  const float bias_scale =
      static_cast<float>(std::pow(2.0, -10)) * layer_norm_scale;

  for (int batch = 0; batch < n_batch; ++batch) {
    const int base = batch * n_input;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n_input; ++i) {
      const float value = static_cast<float>(input[base + i]);
      sum += value;
      sum_sq += value * value;
    }

    const float mean = sum / n_input;
    const float variance = sum_sq / n_input - mean * mean;
    const float stddev_inv =
        variance == 0.0f ? 1.0f / std::sqrt(1e-8f) : 1.0f / std::sqrt(variance);

    int col = 0;
    while (col < n_input) {
      const size_t vl = __riscv_vsetvl_e16m1(n_input - col);
      const vint16m1_t input16 = __riscv_vle16_v_i16m1(input + base + col, vl);
      const vint16m1_t weights16 =
          __riscv_vle16_v_i16m1(layer_norm_weights + col, vl);
      const vint32m2_t input32 = __riscv_vwadd_vx_i32m2(input16, 0, vl);
      const vint32m2_t weights32 = __riscv_vwadd_vx_i32m2(weights16, 0, vl);
      vfloat32m2_t normalized =
          __riscv_vfcvt_f_x_v_f32m2(input32, vl);
      normalized = __riscv_vfsub_vf_f32m2(normalized, mean, vl);
      normalized = __riscv_vfmul_vf_f32m2(normalized, stddev_inv, vl);
      vfloat32m2_t weighted = __riscv_vfcvt_f_x_v_f32m2(weights32, vl);
      weighted = __riscv_vfmul_vf_f32m2(weighted, layer_norm_scale, vl);
      weighted = __riscv_vfmul_vv_f32m2(normalized, weighted, vl);
      if (bias != nullptr) {
        const vint32m2_t bias32 = __riscv_vle32_v_i32m2(bias + col, vl);
        const vfloat32m2_t biasf = __riscv_vfcvt_f_x_v_f32m2(bias32, vl);
        const vfloat32m2_t bias_scaled =
            __riscv_vfmul_vf_f32m2(biasf, bias_scale, vl);
        weighted = __riscv_vfadd_vv_f32m2(weighted, bias_scaled, vl);
      }
      weighted = __riscv_vfmul_vf_f32m2(weighted, kOutputScale, vl);
      vint32m2_t quantized = RvvRoundToNearestAwayFromZero(weighted, vl);
      quantized = __riscv_vmax_vx_i32m2(quantized, int16_min, vl);
      quantized = __riscv_vmin_vx_i32m2(quantized, int16_max, vl);
      const vint16m1_t output16 =
          __riscv_vnclip_wx_i16m1(quantized, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse16_v_i16m1(output + base + col, output16, vl);
      col += static_cast<int>(vl);
    }
  }
}

void RvvApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                     int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const int base = batch * n_input;
    int col = 0;
    while (col < n_input) {
      const size_t vl = __riscv_vsetvl_e16m1(n_input - col);
      const vint16m1_t input16 = __riscv_vle16_v_i16m1(input + base + col, vl);
      const vint16m1_t output16 = RvvApplyLogisticVector<3>(input16, vl);
      __riscv_vse16_v_i16m1(output + base + col, output16, vl);
      col += static_cast<int>(vl);
    }
  }
}

void RvvApplySigmoidFloat(const int16_t* input, int32_t n_batch,
                          int32_t n_input, int16_t* output) {
  // Reuse the Q3.12 fixed-point logistic kernel instead of the scalar exp path.
  RvvApplySigmoid(input, n_batch, n_input, output);
}

void RvvApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
                  int32_t n_input, int16_t* output) {
  assert(integer_bits <= 6);
#define DISPATCH_RVV_TANH(i)                                                \
  case i:                                                                   \
    for (int batch = 0; batch < n_batch; ++batch) {                         \
      const int base = batch * n_input;                                     \
      int col = 0;                                                          \
      while (col < n_input) {                                               \
        const size_t vl = __riscv_vsetvl_e16m1(n_input - col);              \
        const vint16m1_t input16 =                                          \
            __riscv_vle16_v_i16m1(input + base + col, vl);                  \
        const vint16m1_t output16 = RvvApplyTanhVector<i>(input16, vl);     \
        __riscv_vse16_v_i16m1(output + base + col, output16, vl);           \
        col += static_cast<int>(vl);                                        \
      }                                                                     \
    }                                                                       \
    break;
  switch (integer_bits) {
    DISPATCH_RVV_TANH(0);
    DISPATCH_RVV_TANH(1);
    DISPATCH_RVV_TANH(2);
    DISPATCH_RVV_TANH(3);
    DISPATCH_RVV_TANH(4);
    DISPATCH_RVV_TANH(5);
    DISPATCH_RVV_TANH(6);
    default:
      return;
  }
#undef DISPATCH_RVV_TANH
}

void RvvApplyTanhFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                       int32_t integer_bits, int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16_t>::max();
  const int32_t int16_min = std::numeric_limits<int16_t>::min();
  const float input_scale =
      static_cast<float>(std::pow(2.0, static_cast<double>(integer_bits)));
  static constexpr float kOutputScale = static_cast<float>(1 << 15);
  std::array<float, kMaxRvvInt16Lanes> temp_input;
  std::array<float, kMaxRvvInt16Lanes> temp_output;

  for (int batch = 0; batch < n_batch; ++batch) {
    const int base = batch * n_input;
    int col = 0;
    while (col < n_input) {
      const size_t vl = __riscv_vsetvl_e16m1(n_input - col);
      const vint16m1_t input16 = __riscv_vle16_v_i16m1(input + base + col, vl);
      const vint32m2_t input32 = __riscv_vwadd_vx_i32m2(input16, 0, vl);
      vfloat32m2_t inputf = __riscv_vfcvt_f_x_v_f32m2(input32, vl);
      inputf = __riscv_vfmul_vf_f32m2(inputf, input_scale, vl);
      __riscv_vse32_v_f32m2(temp_input.data(), inputf, vl);

      for (size_t i = 0; i < vl; ++i) {
        temp_output[i] = std::tanh(temp_input[i]) * kOutputScale;
      }

      vfloat32m2_t outputf = __riscv_vle32_v_f32m2(temp_output.data(), vl);
      vint32m2_t quantized = __riscv_vfcvt_rtz_x_f_v_i32m2(outputf, vl);
      quantized = __riscv_vmax_vx_i32m2(quantized, int16_min, vl);
      quantized = __riscv_vmin_vx_i32m2(quantized, int16_max, vl);
      const vint16m1_t output16 =
          __riscv_vnclip_wx_i16m1(quantized, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse16_v_i16m1(output + base + col, output16, vl);
      col += static_cast<int>(vl);
    }
  }
}

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int shift, int16_t* output) {
  const size_t vlmax = __riscv_vsetvlmax_e16m1();
  const int step = static_cast<int>(vlmax);
  assert(step <= kMaxRvvInt32Lanes);
  std::array<int32_t, kMaxRvvInt32Lanes> temp_i32;

  for (int batch = 0; batch < n_batch; ++batch) {
    const int base = batch * n_input;
    int col = 0;
    const int full_cols = n_input / step * step;
    for (; col < full_cols; col += step) {
      const vint16m1_t lhs = __riscv_vle16_v_i16m1(input_1 + base + col, vlmax);
      const vint16m1_t rhs = __riscv_vle16_v_i16m1(input_2 + base + col, vlmax);
      const vint32m2_t prod = __riscv_vwmul_vv_i32m2(lhs, rhs, vlmax);
      __riscv_vse32_v_i32m2(temp_i32.data(), prod, vlmax);
      for (int i = 0; i < step; ++i) {
        output[base + col + i] = static_cast<int16_t>(
            gemmlowp::RoundingDivideByPOT(temp_i32[i], shift));
      }
    }
    for (; col < n_input; ++col) {
      const int32_t value = static_cast<int32_t>(input_1[base + col]) *
                            static_cast<int32_t>(input_2[base + col]);
      output[base + col] = static_cast<int16_t>(
          gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2,
                 int32_t multiplier, int32_t shift, int32_t n_batch,
                 int32_t n_input, int32_t output_zp, int8_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const int base = batch * n_input;
    int col = 0;
    while (col < n_input) {
      const size_t vl = __riscv_vsetvl_e16m1(n_input - col);
      const vint16m1_t lhs = __riscv_vle16_v_i16m1(input_1 + base + col, vl);
      const vint16m1_t rhs = __riscv_vle16_v_i16m1(input_2 + base + col, vl);
      vint32m2_t values = __riscv_vwmul_vv_i32m2(lhs, rhs, vl);
      values = RvvMultiplyByQuantizedMultiplier(values, multiplier, shift, vl);
      values = __riscv_vadd_vx_i32m2(values, output_zp, vl);
      values = __riscv_vmax_vx_i32m2(values, -128, vl);
      values = __riscv_vmin_vx_i32m2(values, 127, vl);
      const vint16m1_t narrowed16 =
          __riscv_vnclip_wx_i16m1(values, 0, __RISCV_VXRM_RDN, vl);
      const vint8mf2_t narrowed8 =
          __riscv_vnclip_wx_i8mf2(narrowed16, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse8_v_i8mf2(output + base + col, narrowed8, vl);
      col += static_cast<int>(vl);
    }
  }
}

void RvvCwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int16_t* output) {
  const size_t vlmax = __riscv_vsetvlmax_e16m1();
  const int step = static_cast<int>(vlmax);
  assert(step <= kMaxRvvInt32Lanes);
  std::array<int32_t, kMaxRvvInt32Lanes> temp_i32;

  for (int batch = 0; batch < n_batch; ++batch) {
    const int base = batch * n_input;
    int col = 0;
    const int full_cols = n_input / step * step;
    for (; col < full_cols; col += step) {
      const vint16m1_t lhs = __riscv_vle16_v_i16m1(input_1 + base + col, vlmax);
      const vint16m1_t rhs = __riscv_vle16_v_i16m1(input_2 + base + col, vlmax);
      const vint32m2_t sum = __riscv_vwadd_vv_i32m2(lhs, rhs, vlmax);
      __riscv_vse32_v_i32m2(temp_i32.data(), sum, vlmax);
      for (int i = 0; i < step; ++i) {
        output[base + col + i] = static_cast<int16_t>(Clamp<int32_t>(
            temp_i32[i], std::numeric_limits<int16_t>::min(),
            std::numeric_limits<int16_t>::max()));
      }
    }
    for (; col < n_input; ++col) {
      const int32_t sum =
          static_cast<int32_t>(input_1[base + col]) + input_2[base + col];
      output[base + col] = static_cast<int16_t>(Clamp<int32_t>(
          sum, std::numeric_limits<int16_t>::min(),
          std::numeric_limits<int16_t>::max()));
    }
  }
}

void RvvCwiseClipping(float* vector, int v_size, float clipping_value) {
  if (v_size <= 0) {
    return;
  }

  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  const int step = static_cast<int>(vlmax);
  const vfloat32m1_t max_dup = __riscv_vfmv_v_f_f32m1(clipping_value, vlmax);
  const vfloat32m1_t min_dup = __riscv_vfmv_v_f_f32m1(-clipping_value, vlmax);
  int col = 0;
  const int full_cols = v_size / step * step;
  for (; col < full_cols; col += step) {
    vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + col, vlmax);
    values = __riscv_vfmin_vv_f32m1(values, max_dup, vlmax);
    values = __riscv_vfmax_vv_f32m1(values, min_dup, vlmax);
    __riscv_vse32_v_f32m1(vector + col, values, vlmax);
  }
  for (; col < v_size; ++col) {
    vector[col] = Clamp<float>(vector[col], -clipping_value, clipping_value);
  }
}

void RvvCwiseClipping(int16_t* vector, int v_size, int16_t clipping_value) {
  if (v_size <= 0) {
    return;
  }

  const size_t vlmax = __riscv_vsetvlmax_e16m1();
  const int step = static_cast<int>(vlmax);
  const vint16m1_t max_dup = __riscv_vmv_v_x_i16m1(clipping_value, vlmax);
  const vint16m1_t min_dup = __riscv_vmv_v_x_i16m1(-clipping_value, vlmax);
  int col = 0;
  const int full_cols = v_size / step * step;
  for (; col < full_cols; col += step) {
    vint16m1_t values = __riscv_vle16_v_i16m1(vector + col, vlmax);
    values = __riscv_vmin_vv_i16m1(values, max_dup, vlmax);
    values = __riscv_vmax_vv_i16m1(values, min_dup, vlmax);
    __riscv_vse16_v_i16m1(vector + col, values, vlmax);
  }
  for (; col < v_size; ++col) {
    vector[col] = Clamp<int16_t>(vector[col], -clipping_value, clipping_value);
  }
}

void RvvCwiseClipping(int8_t* vector, int v_size, int8_t clipping_value) {
  if (v_size <= 0) {
    return;
  }

  const size_t vlmax = __riscv_vsetvlmax_e8m1();
  const int step = static_cast<int>(vlmax);
  const vint8m1_t max_dup = __riscv_vmv_v_x_i8m1(clipping_value, vlmax);
  const vint8m1_t min_dup = __riscv_vmv_v_x_i8m1(-clipping_value, vlmax);
  int col = 0;
  const int full_cols = v_size / step * step;
  for (; col < full_cols; col += step) {
    vint8m1_t values = __riscv_vle8_v_i8m1(vector + col, vlmax);
    values = __riscv_vmin_vv_i8m1(values, max_dup, vlmax);
    values = __riscv_vmax_vv_i8m1(values, min_dup, vlmax);
    __riscv_vse8_v_i8m1(vector + col, values, vlmax);
  }
  for (; col < v_size; ++col) {
    vector[col] = Clamp<int8_t>(vector[col], -clipping_value, clipping_value);
  }
}

void RvvVectorBatchVectorCwiseProductAccumulate(const int16_t* vector,
                                                int v_size,
                                                const int16_t* batch_vector,
                                                int n_batch,
                                                int32_t multiplier, int shift,
                                                int16_t* result) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const int16_t* batch_vector_ptr = batch_vector + batch * v_size;
    int16_t* result_ptr = result + batch * v_size;
    int col = 0;
    while (col < v_size) {
      const size_t vl = __riscv_vsetvl_e16m1(v_size - col);
      const vint16m1_t lhs = __riscv_vle16_v_i16m1(vector + col, vl);
      const vint16m1_t rhs =
          __riscv_vle16_v_i16m1(batch_vector_ptr + col, vl);
      vint32m2_t values = __riscv_vwmul_vv_i32m2(lhs, rhs, vl);
      values = RvvMultiplyByQuantizedMultiplier(values, multiplier, shift, vl);
      const vint16m1_t current = __riscv_vle16_v_i16m1(result_ptr + col, vl);
      const vint32m2_t current32 = __riscv_vwadd_vx_i32m2(current, 0, vl);
      values = __riscv_vadd_vv_i32m2(values, current32, vl);
      values = __riscv_vmax_vx_i32m2(
          values, std::numeric_limits<int16_t>::min(), vl);
      values = __riscv_vmin_vx_i32m2(
          values, std::numeric_limits<int16_t>::max(), vl);
      const vint16m1_t narrowed = __riscv_vnsra_wx_i16m1(values, 0, vl);
      __riscv_vse16_v_i16m1(result_ptr + col, narrowed, vl);
      col += static_cast<int>(vl);
    }
  }
}

void RvvBatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                         const int16_t* vector2, int v_size,
                                         int n_batch, int32_t* result) {
  for (int batch = 0; batch < n_batch; ++batch) {
    result[batch] = RvvInt16DotProduct(vector1, vector2, v_size);
    vector1 += v_size;
    vector2 += v_size;
  }
}

void RvvSub1Vector(const float* vector, int v_size, float* result) {
  if (v_size <= 0) {
    return;
  }

  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  const int step = static_cast<int>(vlmax);
  const vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vlmax);
  int col = 0;
  const int full_cols = v_size / step * step;
  for (; col < full_cols; col += step) {
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + col, vlmax);
    const vfloat32m1_t sub1 = __riscv_vfsub_vv_f32m1(one, values, vlmax);
    __riscv_vse32_v_f32m1(result + col, sub1, vlmax);
  }
  for (; col < v_size; ++col) {
    result[col] = 1.0f - vector[col];
  }
}

void RvvSub1Vector(const int16_t* vector, int v_size, int16_t* result) {
  if (v_size <= 0) {
    return;
  }

  const size_t vlmax = __riscv_vsetvlmax_e16m1();
  const int step = static_cast<int>(vlmax);
  const vint16m1_t one = __riscv_vmv_v_x_i16m1(32767, vlmax);
  int col = 0;
  const int full_cols = v_size / step * step;
  for (; col < full_cols; col += step) {
    const vint16m1_t values = __riscv_vle16_v_i16m1(vector + col, vlmax);
    const vint16m1_t sub1 = __riscv_vsub_vv_i16m1(one, values, vlmax);
    __riscv_vse16_v_i16m1(result + col, sub1, vlmax);
  }
  for (; col < v_size; ++col) {
    result[col] = static_cast<int16_t>(32767 - vector[col]);
  }
}

void RvvVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                             float* result) {
  int col = 0;
  while (col < v_size) {
    const size_t vl = __riscv_vsetvl_e8m1(v_size - col);
    const vint8m1_t values8 = __riscv_vle8_v_i8m1(vector + col, vl);
    const vint16m2_t values16 = __riscv_vwadd_vx_i16m2(values8, 0, vl);
    const vint32m4_t values32 = __riscv_vwadd_vx_i32m4(values16, 0, vl);
    const vfloat32m4_t valuesf = __riscv_vfcvt_f_x_v_f32m4(values32, vl);
    const vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(valuesf, scale, vl);
    __riscv_vse32_v_f32m4(result + col, scaled, vl);
    col += static_cast<int>(vl);
  }
}

void RvvReductionSumVector(const float* input_vector, float* output_vector,
                           int output_size, int reduction_size) {
  for (int output_index = 0; output_index < output_size; ++output_index) {
    output_vector[output_index] =
        RvvReduceSumFloat(input_vector, reduction_size);
    input_vector += reduction_size;
  }
}

void RvvReductionSumVector(const int32_t* input_vector, int32_t* output_vector,
                           int output_size, int reduction_size) {
  for (int output_index = 0; output_index < output_size; ++output_index) {
    output_vector[output_index] =
        RvvReduceSumInt32(input_vector, reduction_size);
    input_vector += reduction_size;
  }
}

void RvvReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                           int output_size, int reduction_size) {
  for (int output_index = 0; output_index < output_size; ++output_index) {
    output_vector[output_index] = RvvReduceSumInt8(input_vector, reduction_size);
    input_vector += reduction_size;
  }
}

void RvvMeanStddevNormalization(const float* input_vector, float* output_vector,
                                int v_size, int n_batch) {
  static constexpr float kNormalizationConstant = 1e-8f;
  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  const int step = static_cast<int>(vlmax);

  for (int batch = 0; batch < n_batch; ++batch) {
    const float sum = RvvReduceSumFloat(input_vector, v_size);
    const float mean = sum / v_size;

    float sum_diff_sq = 0.0f;
    int col = 0;
    const int full_cols = v_size / step * step;
    for (; col < full_cols; col += step) {
      const vfloat32m1_t values = __riscv_vle32_v_f32m1(input_vector + col, vlmax);
      const vfloat32m1_t diff = __riscv_vfsub_vf_f32m1(values, mean, vlmax);
      const vfloat32m1_t diff_sq = __riscv_vfmul_vv_f32m1(diff, diff, vlmax);
      sum_diff_sq += HorizontalSum(diff_sq, vlmax);
      __riscv_vse32_v_f32m1(output_vector + col, diff, vlmax);
    }
    for (; col < v_size; ++col) {
      const float diff = input_vector[col] - mean;
      sum_diff_sq += diff * diff;
      output_vector[col] = diff;
    }

    const float variance = sum_diff_sq / v_size;
    const float stddev_inv = 1.0f / std::sqrt(variance + kNormalizationConstant);
    col = 0;
    for (; col < full_cols; col += step) {
      const vfloat32m1_t diff = __riscv_vle32_v_f32m1(output_vector + col, vlmax);
      const vfloat32m1_t normalized =
          __riscv_vfmul_vf_f32m1(diff, stddev_inv, vlmax);
      __riscv_vse32_v_f32m1(output_vector + col, normalized, vlmax);
    }
    for (; col < v_size; ++col) {
      output_vector[col] *= stddev_inv;
    }

    input_vector += v_size;
    output_vector += v_size;
  }
}

void RvvTwoGateSaturatingAdd(const int8_t* input, int8_t input_zp,
                             const int8_t* recurrent, int8_t recurrent_zp,
                             int32_t input_effective_scale_a,
                             int32_t input_effective_scale_b,
                             int32_t recurrent_effective_scale_a,
                             int32_t recurrent_effective_scale_b,
                             int32_t n_batch, int32_t n_cell,
                             int16_t* output) {
  const int total = n_batch * n_cell;
  int col = 0;
  while (col < total) {
    const size_t vl = __riscv_vsetvl_e8mf2(total - col);
    const vint8mf2_t input8 = __riscv_vle8_v_i8mf2(input + col, vl);
    const vint8mf2_t recurrent8 = __riscv_vle8_v_i8mf2(recurrent + col, vl);
    vint16m1_t input16 = __riscv_vwadd_vx_i16m1(input8, 0, vl);
    vint16m1_t recurrent16 = __riscv_vwadd_vx_i16m1(recurrent8, 0, vl);
    input16 = __riscv_vsub_vx_i16m1(input16, input_zp, vl);
    recurrent16 = __riscv_vsub_vx_i16m1(recurrent16, recurrent_zp, vl);

    vint32m2_t input32 = __riscv_vwadd_vx_i32m2(input16, 0, vl);
    vint32m2_t recurrent32 = __riscv_vwadd_vx_i32m2(recurrent16, 0, vl);
    input32 = RvvMultiplyByQuantizedMultiplier(
        input32, input_effective_scale_a, input_effective_scale_b, vl);
    recurrent32 = RvvMultiplyByQuantizedMultiplier(
        recurrent32, recurrent_effective_scale_a,
        recurrent_effective_scale_b, vl);
    vint32m2_t summed = __riscv_vadd_vv_i32m2(input32, recurrent32, vl);
    summed = __riscv_vmax_vx_i32m2(
        summed, std::numeric_limits<int16_t>::min(), vl);
    summed = __riscv_vmin_vx_i32m2(
        summed, std::numeric_limits<int16_t>::max(), vl);
    const vint16m1_t output16 =
        __riscv_vnclip_wx_i16m1(summed, 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse16_v_i16m1(output + col, output16, vl);
    col += static_cast<int>(vl);
  }
}

#endif  // USE_RVV

}  // namespace tensor_utils
}  // namespace tflite
