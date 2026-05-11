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
#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_

#include "tflite/kernels/internal/optimized/rvv_check.h"

#ifdef USE_RVV

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/cpu_backend_threadpool.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

// ============================================================
// Float GEMM
// ============================================================

inline void RvvGemmFloatImpl(const float* lhs, const float* rhs,
                             float* dst, const float* bias,
                             int rows, int depth, int cols,
                             float clamp_min, float clamp_max,
                             int col_start, int col_end) {
  for (int col = col_start; col < col_end; col++) {
    const float* rhs_col = rhs + col * depth;
    float* dst_col = dst + col * rows;
    for (int row = 0; row < rows; row++) {
      const float* lhs_row = lhs + row * depth;
      size_t vl;
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f,
                                                   __riscv_vsetvl_e32m4(1));
      for (int k = 0; k < depth; k += vl) {
        vl = __riscv_vsetvl_e32m4(depth - k);
        vfloat32m4_t a = __riscv_vle32_v_f32m4(lhs_row + k, vl);
        vfloat32m4_t b = __riscv_vle32_v_f32m4(rhs_col + k, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, a, b, vl);
      }
      vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
      vfloat32m1_t s = __riscv_vfredusum_vs_f32m4_f32m1(
          acc, zero, __riscv_vsetvl_e32m4(depth));
      float result = __riscv_vfmv_f_s_f32m1_f32(s);
      if (bias) result += bias[row];
      result = std::min(clamp_max, std::max(clamp_min, result));
      dst_col[row] = result;
    }
  }
}

struct RvvGemmFloatTask : cpu_backend_threadpool::Task {
  const float* lhs;
  const float* rhs;
  float* dst;
  const float* bias;
  int rows, depth, cols;
  float clamp_min, clamp_max;
  int col_start, col_end;
  void Run() override {
    RvvGemmFloatImpl(lhs, rhs, dst, bias, rows, depth, cols,
                     clamp_min, clamp_max, col_start, col_end);
  }
};

inline bool RvvGemmFloat(
    const MatrixParams<float>& lhs_params, const float* lhs_data,
    const MatrixParams<float>& rhs_params, const float* rhs_data,
    const MatrixParams<float>& dst_params, float* dst_data,
    const GemmParams<float, float, QuantizationFlavor::kFloatingPoint>& params,
    CpuBackendContext* context) {
  if (lhs_params.order != Order::kRowMajor ||
      rhs_params.order != Order::kColMajor ||
      dst_params.order != Order::kColMajor) {
    return false;
  }
  const int rows = lhs_params.rows;
  const int depth = lhs_params.cols;
  const int cols = rhs_params.cols;
  const int num_threads =
      std::min(context->max_num_threads(), std::max(1, cols));

  if (num_threads <= 1) {
    RvvGemmFloatImpl(lhs_data, rhs_data, dst_data, params.bias, rows, depth,
                     cols, params.clamp_min, params.clamp_max, 0, cols);
    return true;
  }
  std::vector<RvvGemmFloatTask> tasks(num_threads);
  int cs = 0;
  for (int i = 0; i < num_threads; i++) {
    int ce = cs + (cols - cs) / (num_threads - i);
    auto& t = tasks[i];
    t.lhs = lhs_data; t.rhs = rhs_data; t.dst = dst_data;
    t.bias = params.bias; t.rows = rows; t.depth = depth; t.cols = cols;
    t.clamp_min = params.clamp_min; t.clamp_max = params.clamp_max;
    t.col_start = cs; t.col_end = ce;
    cs = ce;
  }
  cpu_backend_threadpool::Execute(num_threads, tasks.data(), context);
  return true;
}

// ============================================================
// Quantized GEMM — optimized with precomputed row/col sums
// ============================================================
// Key optimization: zero-point correction moved out of inner loop.
//
// Standard: acc = sum((a[k]-lhs_zp) * (b[k]-rhs_zp))
//         = sum(a[k]*b[k]) - rhs_zp*sum(a[k]) - lhs_zp*sum(b[k])
//           + depth*lhs_zp*rhs_zp
//
// Inner loop only computes sum(a[k]*b[k]) on raw int8 values.
// Zero-point correction applied once per output element.

inline int32_t RvvDotI8(const int8_t* a, const int8_t* b, int len) {
  size_t vl;
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(1));
  for (int k = 0; k < len; k += vl) {
    vl = __riscv_vsetvl_e8m1(len - k);
    vint8m1_t va = __riscv_vle8_v_i8m1(a + k, vl);
    vint8m1_t vb = __riscv_vle8_v_i8m1(b + k, vl);
    vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
    acc = __riscv_vwadd_wv_i32m4(acc, prod, vl);
  }
  vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(acc, z,
                                                  __riscv_vsetvl_e32m4(len));
  return __riscv_vmv_x_s_i32m1_i32(s);
}

inline int32_t RvvSumI8(const int8_t* data, int len) {
  size_t vl;
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(1));
  for (int k = 0; k < len; k += vl) {
    vl = __riscv_vsetvl_e8m1(len - k);
    vint8m1_t v = __riscv_vle8_v_i8m1(data + k, vl);
    vint16m2_t v16 = __riscv_vsext_vf2_i16m2(v, vl);
    acc = __riscv_vwadd_wv_i32m4(acc, v16, vl);
  }
  vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(acc, z,
                                                  __riscv_vsetvl_e32m4(len));
  return __riscv_vmv_x_s_i32m1_i32(s);
}

inline int32_t RvvDotU8(const uint8_t* a, const uint8_t* b, int len) {
  size_t vl;
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(1));
  for (int k = 0; k < len; k += vl) {
    vl = __riscv_vsetvl_e8m1(len - k);
    vuint8m1_t va = __riscv_vle8_v_u8m1(a + k, vl);
    vuint8m1_t vb = __riscv_vle8_v_u8m1(b + k, vl);
    vuint16m2_t prod = __riscv_vwmulu_vv_u16m2(va, vb, vl);
    vint16m2_t prod_s = __riscv_vreinterpret_v_u16m2_i16m2(prod);
    // Accumulate as unsigned in int32 to avoid overflow
    vuint32m4_t prod32 = __riscv_vzext_vf2_u32m4(prod, vl);
    vint32m4_t prod32_s = __riscv_vreinterpret_v_u32m4_i32m4(prod32);
    acc = __riscv_vadd_vv_i32m4(acc, prod32_s, vl);
  }
  vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(acc, z,
                                                  __riscv_vsetvl_e32m4(len));
  return __riscv_vmv_x_s_i32m1_i32(s);
}

inline int32_t RvvSumU8(const uint8_t* data, int len) {
  size_t vl;
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(1));
  for (int k = 0; k < len; k += vl) {
    vl = __riscv_vsetvl_e8m1(len - k);
    vuint8m1_t v = __riscv_vle8_v_u8m1(data + k, vl);
    vuint16m2_t v16 = __riscv_vzext_vf2_u16m2(v, vl);
    vuint32m4_t v32 = __riscv_vzext_vf2_u32m4(v16, vl);
    vint32m4_t v32s = __riscv_vreinterpret_v_u32m4_i32m4(v32);
    acc = __riscv_vadd_vv_i32m4(acc, v32s, vl);
  }
  vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(acc, z,
                                                  __riscv_vsetvl_e32m4(len));
  return __riscv_vmv_x_s_i32m1_i32(s);
}

template <typename SrcScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
inline void RvvGemmQuantizedImpl(
    const SrcScalar* lhs_data, const SrcScalar* rhs_data,
    DstScalar* dst_data, int rows, int depth, int cols,
    int32_t lhs_zp, int32_t rhs_zp, int32_t dst_zp,
    int32_t multiplier, int shift,
    const int32_t* multiplier_perchannel, const int* shift_perchannel,
    const int32_t* bias, int32_t clamp_min, int32_t clamp_max,
    int col_start, int col_end) {
  // Precompute LHS row sums: sum(lhs_data[row, :]) for each row.
  std::vector<int32_t> lhs_row_sums(rows);
  for (int row = 0; row < rows; row++) {
    if constexpr (std::is_same_v<SrcScalar, int8_t>) {
      lhs_row_sums[row] = RvvSumI8(lhs_data + row * depth, depth);
    } else {
      lhs_row_sums[row] = RvvSumU8(
          reinterpret_cast<const uint8_t*>(lhs_data + row * depth), depth);
    }
  }

  const int32_t depth_times_lhs_zp_times_rhs_zp =
      depth * lhs_zp * rhs_zp;

  for (int col = col_start; col < col_end; col++) {
    const SrcScalar* rhs_col = rhs_data + col * depth;
    DstScalar* dst_col = dst_data + col * rows;

    // Precompute RHS column sum for this column.
    int32_t rhs_col_sum;
    if constexpr (std::is_same_v<SrcScalar, int8_t>) {
      rhs_col_sum = RvvSumI8(rhs_col, depth);
    } else {
      rhs_col_sum = RvvSumU8(
          reinterpret_cast<const uint8_t*>(rhs_col), depth);
    }

    for (int row = 0; row < rows; row++) {
      const SrcScalar* lhs_row = lhs_data + row * depth;

      // Raw dot product (no zero-point subtraction in inner loop).
      int32_t raw_acc;
      if constexpr (std::is_same_v<SrcScalar, int8_t>) {
        raw_acc = RvvDotI8(lhs_row, rhs_col, depth);
      } else {
        raw_acc = RvvDotU8(
            reinterpret_cast<const uint8_t*>(lhs_row),
            reinterpret_cast<const uint8_t*>(rhs_col), depth);
      }

      // Apply zero-point correction:
      // acc = raw_acc - rhs_zp * row_sum - lhs_zp * col_sum
      //       + depth * lhs_zp * rhs_zp
      int32_t acc = raw_acc
                    - rhs_zp * lhs_row_sums[row]
                    - lhs_zp * rhs_col_sum
                    + depth_times_lhs_zp_times_rhs_zp;

      if (bias) acc += bias[row];

      int32_t row_mult = multiplier_perchannel ? multiplier_perchannel[row]
                                               : multiplier;
      int row_shift = shift_perchannel ? shift_perchannel[row] : shift;
      acc = MultiplyByQuantizedMultiplier(acc, row_mult, row_shift);
      acc += dst_zp;
      acc = std::min(std::max(acc, clamp_min), clamp_max);
      dst_col[row] = static_cast<DstScalar>(acc);
    }
  }
}

template <typename SrcScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct RvvGemmQuantizedTask : cpu_backend_threadpool::Task {
  const SrcScalar* lhs_data;
  const SrcScalar* rhs_data;
  DstScalar* dst_data;
  int rows, depth, cols;
  int32_t lhs_zp, rhs_zp, dst_zp;
  int32_t multiplier;
  int shift;
  const int32_t* multiplier_perchannel;
  const int* shift_perchannel;
  const int32_t* bias;
  int32_t clamp_min, clamp_max;
  int col_start, col_end;
  void Run() override {
    RvvGemmQuantizedImpl<SrcScalar, DstScalar, quantization_flavor>(
        lhs_data, rhs_data, dst_data, rows, depth, cols, lhs_zp, rhs_zp,
        dst_zp, multiplier, shift, multiplier_perchannel, shift_perchannel,
        bias, clamp_min, clamp_max, col_start, col_end);
  }
};

template <typename SrcScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
inline bool RvvGemmQuantized(
    const MatrixParams<SrcScalar>& lhs_params, const SrcScalar* lhs_data,
    const MatrixParams<SrcScalar>& rhs_params, const SrcScalar* rhs_data,
    const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
    const GemmParams<int32_t, DstScalar, quantization_flavor>& params,
    CpuBackendContext* context) {
  if (lhs_params.order != Order::kRowMajor ||
      rhs_params.order != Order::kColMajor ||
      dst_params.order != Order::kColMajor) {
    return false;
  }
  const int rows = lhs_params.rows;
  const int depth = lhs_params.cols;
  const int cols = rhs_params.cols;
  const int num_threads =
      std::min(context->max_num_threads(), std::max(1, cols));

  if (num_threads <= 1) {
    RvvGemmQuantizedImpl<SrcScalar, DstScalar, quantization_flavor>(
        lhs_data, rhs_data, dst_data, rows, depth, cols,
        lhs_params.zero_point, rhs_params.zero_point, dst_params.zero_point,
        params.multiplier_fixedpoint, params.multiplier_exponent,
        params.multiplier_fixedpoint_perchannel,
        params.multiplier_exponent_perchannel,
        params.bias, params.clamp_min, params.clamp_max, 0, cols);
    return true;
  }

  std::vector<RvvGemmQuantizedTask<SrcScalar, DstScalar, quantization_flavor>>
      tasks(num_threads);
  int cs = 0;
  for (int i = 0; i < num_threads; i++) {
    int ce = cs + (cols - cs) / (num_threads - i);
    auto& t = tasks[i];
    t.lhs_data = lhs_data; t.rhs_data = rhs_data; t.dst_data = dst_data;
    t.rows = rows; t.depth = depth; t.cols = cols;
    t.lhs_zp = lhs_params.zero_point;
    t.rhs_zp = rhs_params.zero_point;
    t.dst_zp = dst_params.zero_point;
    t.multiplier = params.multiplier_fixedpoint;
    t.shift = params.multiplier_exponent;
    t.multiplier_perchannel = params.multiplier_fixedpoint_perchannel;
    t.shift_perchannel = params.multiplier_exponent_perchannel;
    t.bias = params.bias;
    t.clamp_min = params.clamp_min; t.clamp_max = params.clamp_max;
    t.col_start = cs; t.col_end = ce;
    cs = ce;
  }
  cpu_backend_threadpool::Execute(num_threads, tasks.data(), context);
  return true;
}

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // USE_RVV
#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
