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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_gemm.h"
#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/cppmath.h"
#include "tflite/kernels/internal/optimized/rvv_quantization_utils.h"
#include "tflite/kernels/internal/optimized/rvv_tensor_utils_impl.h"
#include "tflite/kernels/internal/reference/portable_tensor_utils_impl.h"

#ifdef USE_RVV

namespace tflite {
namespace tensor_utils {

// --- Float matrix-vector multiply ---

void RvvMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                            int m_cols, const float* vector,
                                            int n_batch, float* result) {
  for (int b = 0; b < n_batch; b++) {
    const float* vector_in_batch = vector + b * m_cols;
    float* result_in_batch = result + b * m_rows;
    const float* matrix_row = matrix;
    for (int r = 0; r < m_rows; r++) {
      int c = 0;
      size_t vl;
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(m_cols));
      for (; c < m_cols; c += vl) {
        vl = __riscv_vsetvl_e32m4(m_cols - c);
        vfloat32m4_t vm = __riscv_vle32_v_f32m4(matrix_row + c, vl);
        vfloat32m4_t vv = __riscv_vle32_v_f32m4(vector_in_batch + c, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, vm, vv, vl);
      }
      vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
      vfloat32m1_t sum = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero,
                                                           __riscv_vsetvl_e32m4(m_cols));
      *result_in_batch += __riscv_vfmv_f_s_f32m1_f32(sum);
      matrix_row += m_cols;
      ++result_in_batch;
    }
  }
}

// --- Int8 matrix-vector multiply (delegate to Portable for complex variants) ---

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                            const int m_rows, const int m_cols,
                                            const int8_t* __restrict__ vectors,
                                            const float* scaling_factors,
                                            int n_batch,
                                            float* __restrict__ result) {
  for (int b = 0; b < n_batch; b++) {
    const float scale = scaling_factors[b];
    const int8_t* vec = vectors + b * m_cols;
    float* res = result + b * m_rows;
    const int8_t* mat_row = matrix;
    for (int r = 0; r < m_rows; r++) {
      int c = 0;
      size_t vl;
      vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(m_cols));
      for (; c < m_cols; c += vl) {
        vl = __riscv_vsetvl_e8m1(m_cols - c);
        vint8m1_t vm = __riscv_vle8_v_i8m1(mat_row + c, vl);
        vint8m1_t vv = __riscv_vle8_v_i8m1(vec + c, vl);
        vint16m2_t prod = __riscv_vwmul_vv_i16m2(vm, vv, vl);
        vint32m4_t prod32 = __riscv_vsext_vf2_i32m4(prod, vl);
        acc = __riscv_vadd_vv_i32m4(acc, prod32, vl);
      }
      vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t sum = __riscv_vredsum_vs_i32m4_i32m1(acc, zero,
                                                       __riscv_vsetvl_e32m4(m_cols));
      res[r] += __riscv_vmv_x_s_i32m1_i32(sum) * scale;
      mat_row += m_cols;
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                            const int m_rows, const int m_cols,
                                            const int8_t* __restrict__ vectors,
                                            const float* scaling_factors,
                                            int n_batch, int32_t* scratch,
                                            float* __restrict__ result,
                                            CpuBackendContext* context) {
  RvvMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                         scaling_factors, n_batch, result);
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      matrix, m_rows, m_cols, vectors, scaling_factors, n_batch, result,
      per_channel_scale, input_offset, scratch, row_sums, compute_row_sums,
      context);
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, scratch, output, context);
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, scratch, output, context);
}

void RvvMatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                       int32_t n_row, int32_t n_col,
                                       int32_t* output) {
  for (int r = 0; r < n_row; r++) {
    int32_t sum = 0;
    const int8_t* row = matrix + r * n_col;
    int c = 0;
    size_t vl;
    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvl_e32m4(n_col));
    for (; c < n_col; c += vl) {
      vl = __riscv_vsetvl_e8m1(n_col - c);
      vint8m1_t v = __riscv_vle8_v_i8m1(row + c, vl);
      vint16m2_t v16 = __riscv_vsext_vf2_i16m2(v, vl);
      vint32m4_t v32 = __riscv_vsext_vf2_i32m4(v16, vl);
      acc = __riscv_vadd_vv_i32m4(acc, v32, vl);
    }
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(acc, zero,
                                                    __riscv_vsetvl_e32m4(n_col));
    output[r] += __riscv_vmv_x_s_i32m1_i32(s) * scalar;
  }
}

// --- Dot product ---

float RvvVectorVectorDotProduct(const float* vector1, const float* vector2,
                                int v_size) {
  int i = 0;
  size_t vl;
  vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(v_size));
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e32m4(v_size - i);
    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(vector1 + i, vl);
    vfloat32m4_t v2 = __riscv_vle32_v_f32m4(vector2 + i, vl);
    acc = __riscv_vfmacc_vv_f32m4(acc, v1, v2, vl);
  }
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t sum = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero,
                                                       __riscv_vsetvl_e32m4(v_size));
  return __riscv_vfmv_f_s_f32m1_f32(sum);
}

// --- Sub1Vector ---

void RvvSub1Vector(const float* vector, int v_size, float* result) {
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e32m4(v_size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(vector + i, vl);
    vfloat32m4_t r = __riscv_vfrsub_vf_f32m4(v, 1.0f, vl);
    __riscv_vse32_v_f32m4(result + i, r, vl);
  }
}

void RvvSub1Vector(const int16_t* vector, int v_size, int16_t* result) {
  static const int16_t kOne = 32767;
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e16m4(v_size - i);
    vint16m4_t v = __riscv_vle16_v_i16m4(vector + i, vl);
    vint16m4_t r = __riscv_vxor_vx_i16m4(v, kOne, vl);
    __riscv_vse16_v_i16m4(result + i, r, vl);
  }
}

// --- CwiseMul ---

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int shift, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const int offset = batch * n_input;
    int i = 0;
    size_t vl;
    for (; i < n_input; i += vl) {
      vl = __riscv_vsetvl_e16m2(n_input - i);
      vint16m2_t a = __riscv_vle16_v_i16m2(input_1 + offset + i, vl);
      vint16m2_t b = __riscv_vle16_v_i16m2(input_2 + offset + i, vl);
      vint32m4_t product = __riscv_vwmul_vv_i32m4(a, b, vl);
      product = __riscv_vsra_vx_i32m4(product, shift, vl);
      vint16m2_t result = __riscv_vnclip_wx_i16m2(product, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse16_v_i16m2(output + offset + i, result, vl);
    }
  }
}

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2,
                 int32_t multiplier, int shift, int n_batch, int n_input,
                 int32_t output_zp, int8_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const int offset = batch * n_input;
    int i = 0;
    size_t vl;
    for (; i < n_input; i += vl) {
      vl = __riscv_vsetvl_e16m1(n_input - i);
      vint16m1_t a = __riscv_vle16_v_i16m1(input_1 + offset + i, vl);
      vint16m1_t b = __riscv_vle16_v_i16m1(input_2 + offset + i, vl);
      vint32m2_t product = __riscv_vwmul_vv_i32m2(a, b, vl);
      product = rvv_utils::MultiplyByQuantizedMultiplier_m2(
          product, multiplier, shift, vl);
      product = __riscv_vadd_vx_i32m2(product, output_zp, vl);
      product = __riscv_vmax_vx_i32m2(product, -128, vl);
      product = __riscv_vmin_vx_i32m2(product, 127, vl);
      vint16m1_t n16 = __riscv_vnclip_wx_i16m1(product, 0, __RISCV_VXRM_RDN, vl);
      vint8mf2_t n8 = __riscv_vnclip_wx_i8mf2(n16, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse8_v_i8mf2(output + offset + i, n8, vl);
    }
  }
}

// --- CwiseAdd ---

void RvvCwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const int offset = batch * n_input;
    int i = 0;
    size_t vl;
    for (; i < n_input; i += vl) {
      vl = __riscv_vsetvl_e16m2(n_input - i);
      vint16m2_t a = __riscv_vle16_v_i16m2(input_1 + offset + i, vl);
      vint16m2_t b = __riscv_vle16_v_i16m2(input_2 + offset + i, vl);
      vint32m4_t sum = __riscv_vwadd_vv_i32m4(a, b, vl);
      sum = __riscv_vmax_vx_i32m4(sum, std::numeric_limits<int16_t>::min(), vl);
      sum = __riscv_vmin_vx_i32m4(sum, std::numeric_limits<int16_t>::max(), vl);
      vint16m2_t result = __riscv_vnclip_wx_i16m2(sum, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse16_v_i16m2(output + offset + i, result, vl);
    }
  }
}

// --- CwiseClipping ---

void RvvCwiseClipping(float* vector, const int v_size,
                      const float clipping_value) {
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e32m4(v_size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(vector + i, vl);
    v = __riscv_vfmin_vf_f32m4(v, clipping_value, vl);
    v = __riscv_vfmax_vf_f32m4(v, -clipping_value, vl);
    __riscv_vse32_v_f32m4(vector + i, v, vl);
  }
}

void RvvCwiseClipping(int16_t* vector, const int v_size,
                      const int16_t clipping_value) {
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e16m4(v_size - i);
    vint16m4_t v = __riscv_vle16_v_i16m4(vector + i, vl);
    v = __riscv_vmin_vx_i16m4(v, clipping_value, vl);
    v = __riscv_vmax_vx_i16m4(v, static_cast<int16_t>(-clipping_value), vl);
    __riscv_vse16_v_i16m4(vector + i, v, vl);
  }
}

void RvvCwiseClipping(int8_t* vector, const int v_size,
                      const int8_t clipping_value) {
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e8m4(v_size - i);
    vint8m4_t v = __riscv_vle8_v_i8m4(vector + i, vl);
    v = __riscv_vmin_vx_i8m4(v, clipping_value, vl);
    v = __riscv_vmax_vx_i8m4(v, static_cast<int8_t>(-clipping_value), vl);
    __riscv_vse8_v_i8m4(vector + i, v, vl);
  }
}

// --- VectorScalarMultiply ---

void RvvVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                             float* result) {
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e8m1(v_size - i);
    vint8m1_t vi8 = __riscv_vle8_v_i8m1(vector + i, vl);
    vint16m2_t vi16 = __riscv_vsext_vf2_i16m2(vi8, vl);
    vint32m4_t vi32 = __riscv_vsext_vf2_i32m4(vi16, vl);
    vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vi32, vl);
    vf = __riscv_vfmul_vf_f32m4(vf, scale, vl);
    __riscv_vse32_v_f32m4(result + i, vf, vl);
  }
}

// --- IsZeroVector ---

bool RvvIsZeroVector(const float* vector, int v_size) {
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e32m4(v_size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(vector + i, vl);
    vbool8_t ne = __riscv_vmfne_vf_f32m4_b8(v, 0.0f, vl);
    if (__riscv_vcpop_m_b8(ne, vl) != 0) return false;
  }
  return true;
}

bool RvvIsZeroVector(const int8_t* vector, int v_size) {
  int i = 0;
  size_t vl;
  for (; i < v_size; i += vl) {
    vl = __riscv_vsetvl_e8m4(v_size - i);
    vint8m4_t v = __riscv_vle8_v_i8m4(vector + i, vl);
    vbool2_t ne = __riscv_vmsne_vx_i8m4_b2(v, 0, vl);
    if (__riscv_vcpop_m_b2(ne, vl) != 0) return false;
  }
  return true;
}

// --- ReductionSumVector ---

void RvvReductionSumVector(const float* input_vector, float* output_vector,
                           int output_size, int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    int i = 0;
    size_t vl;
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f,
                                                __riscv_vsetvl_e32m4(reduction_size));
    for (; i < reduction_size; i += vl) {
      vl = __riscv_vsetvl_e32m4(reduction_size - i);
      vfloat32m4_t v = __riscv_vle32_v_f32m4(input_vector + i, vl);
      acc = __riscv_vfadd_vv_f32m4(acc, v, vl);
    }
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero,
                                                         __riscv_vsetvl_e32m4(reduction_size));
    output_vector[o] = __riscv_vfmv_f_s_f32m1_f32(sum);
    input_vector += reduction_size;
  }
}

void RvvReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                           int output_size, int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    int i = 0;
    size_t vl;
    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0,
                                             __riscv_vsetvl_e32m4(reduction_size));
    for (; i < reduction_size; i += vl) {
      vl = __riscv_vsetvl_e8m1(reduction_size - i);
      vint8m1_t v8 = __riscv_vle8_v_i8m1(input_vector + i, vl);
      vint16m2_t v16 = __riscv_vsext_vf2_i16m2(v8, vl);
      vint32m4_t v32 = __riscv_vsext_vf2_i32m4(v16, vl);
      acc = __riscv_vadd_vv_i32m4(acc, v32, vl);
    }
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t sum = __riscv_vredsum_vs_i32m4_i32m1(acc, zero,
                                                     __riscv_vsetvl_e32m4(reduction_size));
    output_vector[o] = __riscv_vmv_x_s_i32m1_i32(sum);
    input_vector += reduction_size;
  }
}

// --- MeanStddevNormalization ---

void RvvMeanStddevNormalization(const float* __restrict__ input_vector,
                                float* __restrict__ output_vector, int v_size,
                                int n_batch) {
  for (int batch = 0; batch < n_batch; ++batch) {
    // Sum
    int i = 0;
    size_t vl;
    vfloat32m4_t sum_v = __riscv_vfmv_v_f_f32m4(0.0f,
                                                   __riscv_vsetvl_e32m4(v_size));
    for (; i < v_size; i += vl) {
      vl = __riscv_vsetvl_e32m4(v_size - i);
      vfloat32m4_t in = __riscv_vle32_v_f32m4(input_vector + i, vl);
      sum_v = __riscv_vfadd_vv_f32m4(sum_v, in, vl);
    }
    vfloat32m1_t zero_f = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_s = __riscv_vfredusum_vs_f32m4_f32m1(sum_v, zero_f,
                                                           __riscv_vsetvl_e32m4(v_size));
    const float mean = __riscv_vfmv_f_s_f32m1_f32(sum_s) / v_size;

    // Sum of squared differences
    i = 0;
    vfloat32m4_t ssd_v = __riscv_vfmv_v_f_f32m4(0.0f,
                                                   __riscv_vsetvl_e32m4(v_size));
    for (; i < v_size; i += vl) {
      vl = __riscv_vsetvl_e32m4(v_size - i);
      vfloat32m4_t in = __riscv_vle32_v_f32m4(input_vector + i, vl);
      vfloat32m4_t diff = __riscv_vfsub_vf_f32m4(in, mean, vl);
      ssd_v = __riscv_vfmacc_vv_f32m4(ssd_v, diff, diff, vl);
    }
    vfloat32m1_t ssd_s = __riscv_vfredusum_vs_f32m4_f32m1(ssd_v, zero_f,
                                                           __riscv_vsetvl_e32m4(v_size));
    const float variance = __riscv_vfmv_f_s_f32m1_f32(ssd_s) / v_size;
    constexpr float kEps = 1e-8f;
    const float stddev_inv = 1.0f / std::sqrt(variance + kEps);

    // Normalize
    i = 0;
    for (; i < v_size; i += vl) {
      vl = __riscv_vsetvl_e32m4(v_size - i);
      vfloat32m4_t in = __riscv_vle32_v_f32m4(input_vector + i, vl);
      vfloat32m4_t diff = __riscv_vfsub_vf_f32m4(in, mean, vl);
      vfloat32m4_t out = __riscv_vfmul_vf_f32m4(diff, stddev_inv, vl);
      __riscv_vse32_v_f32m4(output_vector + i, out, vl);
    }
    input_vector += v_size;
    output_vector += v_size;
  }
}

// --- SymmetricQuantizeFloats ---

void RvvSymmetricQuantizeFloats(const float* values, const int size,
                                int8_t* quantized_values, float* min,
                                float* max, float* scaling_factor) {
  // Find min and max
  float rmin = 0.0f, rmax = 0.0f;
  int i = 0;
  size_t vl;
  vfloat32m4_t vmin_v = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(size));
  vfloat32m4_t vmax_v = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(size));
  for (; i < size; i += vl) {
    vl = __riscv_vsetvl_e32m4(size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(values + i, vl);
    vmin_v = __riscv_vfmin_vv_f32m4(vmin_v, v, vl);
    vmax_v = __riscv_vfmax_vv_f32m4(vmax_v, v, vl);
  }
  {
    size_t full_vl = __riscv_vsetvl_e32m4(size);
    vfloat32m1_t id_min = __riscv_vfmv_v_f_f32m1(std::numeric_limits<float>::max(), 1);
    vfloat32m1_t id_max = __riscv_vfmv_v_f_f32m1(std::numeric_limits<float>::lowest(), 1);
    vfloat32m1_t rmin_v = __riscv_vfredmin_vs_f32m4_f32m1(vmin_v, id_min, full_vl);
    vfloat32m1_t rmax_v = __riscv_vfredmax_vs_f32m4_f32m1(vmax_v, id_max, full_vl);
    rmin = __riscv_vfmv_f_s_f32m1_f32(rmin_v);
    rmax = __riscv_vfmv_f_s_f32m1_f32(rmax_v);
  }
  *min = rmin;
  *max = rmax;
  const float range = std::max(std::abs(rmin), std::abs(rmax));
  if (range == 0) {
    memset(quantized_values, 0, size);
    *scaling_factor = 1.0f;
    return;
  }
  *scaling_factor = range / 127.0f;
  const float inv_scale = 127.0f / range;

  i = 0;
  for (; i < size; i += vl) {
    vl = __riscv_vsetvl_e32m4(size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(values + i, vl);
    vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(v, inv_scale, vl);
    vint32m4_t rounded = __riscv_vfcvt_x_f_v_i32m4(scaled, vl);
    rounded = __riscv_vmax_vx_i32m4(rounded, -128, vl);
    rounded = __riscv_vmin_vx_i32m4(rounded, 127, vl);
    vint16m2_t n16 = __riscv_vnclip_wx_i16m2(rounded, 0, __RISCV_VXRM_RDN, vl);
    vint8m1_t n8 = __riscv_vnclip_wx_i8m1(n16, 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse8_v_i8m1(quantized_values + i, n8, vl);
  }
}

void RvvSymmetricQuantizeFloats(const float* values, const int size,
                                int8_t* quantized_values, float min, float max,
                                float* scaling_factor) {
  const float range = std::max(std::abs(min), std::abs(max));
  if (range == 0) {
    memset(quantized_values, 0, size);
    *scaling_factor = 1.0f;
    return;
  }
  *scaling_factor = range / 127.0f;
  const float inv_scale = 127.0f / range;

  int i = 0;
  size_t vl;
  for (; i < size; i += vl) {
    vl = __riscv_vsetvl_e32m4(size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(values + i, vl);
    vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(v, inv_scale, vl);
    vint32m4_t rounded = __riscv_vfcvt_x_f_v_i32m4(scaled, vl);
    rounded = __riscv_vmax_vx_i32m4(rounded, -128, vl);
    rounded = __riscv_vmin_vx_i32m4(rounded, 127, vl);
    vint16m2_t n16 = __riscv_vnclip_wx_i16m2(rounded, 0, __RISCV_VXRM_RDN, vl);
    vint8m1_t n8 = __riscv_vnclip_wx_i8m1(n16, 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse8_v_i8m1(quantized_values + i, n8, vl);
  }
}

// --- AsymmetricQuantizeFloats ---

void RvvAsymmetricQuantizeFloats(const float* values, const int size,
                                 int8_t* quantized_values,
                                 float* scaling_factor, int32_t* offset) {
  float rmin = 0.0f, rmax = 0.0f;
  int i = 0;
  size_t vl;
  vfloat32m4_t vmin_v = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(size));
  vfloat32m4_t vmax_v = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(size));
  for (; i < size; i += vl) {
    vl = __riscv_vsetvl_e32m4(size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(values + i, vl);
    vmin_v = __riscv_vfmin_vv_f32m4(vmin_v, v, vl);
    vmax_v = __riscv_vfmax_vv_f32m4(vmax_v, v, vl);
  }
  {
    size_t full_vl = __riscv_vsetvl_e32m4(size);
    vfloat32m1_t id_min = __riscv_vfmv_v_f_f32m1(std::numeric_limits<float>::max(), 1);
    vfloat32m1_t id_max = __riscv_vfmv_v_f_f32m1(std::numeric_limits<float>::lowest(), 1);
    vfloat32m1_t rmin_v = __riscv_vfredmin_vs_f32m4_f32m1(vmin_v, id_min, full_vl);
    vfloat32m1_t rmax_v = __riscv_vfredmax_vs_f32m4_f32m1(vmax_v, id_max, full_vl);
    rmin = __riscv_vfmv_f_s_f32m1_f32(rmin_v);
    rmax = __riscv_vfmv_f_s_f32m1_f32(rmax_v);
  }
  if (rmin == rmax) {
    memset(quantized_values, 0, size);
    *scaling_factor = 1.0f;
    *offset = 0;
    return;
  }
  const float scale = (rmax - rmin) / 255.0f;
  *scaling_factor = scale;
  const float inv_scale = 1.0f / scale;
  const float zp = -128.0f - rmin * inv_scale;
  int32_t nudged_zp = static_cast<int32_t>(std::round(zp));
  nudged_zp = std::max(-128, std::min(127, nudged_zp));
  *offset = nudged_zp;

  i = 0;
  for (; i < size; i += vl) {
    vl = __riscv_vsetvl_e32m4(size - i);
    vfloat32m4_t v = __riscv_vle32_v_f32m4(values + i, vl);
    vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(v, inv_scale, vl);
    vfloat32m4_t shifted = __riscv_vfadd_vf_f32m4(scaled, static_cast<float>(nudged_zp), vl);
    vint32m4_t rounded = __riscv_vfcvt_x_f_v_i32m4(shifted, vl);
    rounded = __riscv_vmax_vx_i32m4(rounded, -128, vl);
    rounded = __riscv_vmin_vx_i32m4(rounded, 127, vl);
    vint16m2_t n16 = __riscv_vnclip_wx_i16m2(rounded, 0, __RISCV_VXRM_RDN, vl);
    vint8m1_t n8 = __riscv_vnclip_wx_i8m1(n16, 0, __RISCV_VXRM_RDN, vl);
    __riscv_vse8_v_i8m1(quantized_values + i, n8, vl);
  }
}

// --- Sparse float matrix-vector multiply ---

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result) {
  constexpr int kBlockSize = 4;
  for (int batch = 0; batch < n_batch; batch++) {
    const float* matrix_ptr = matrix;
    for (int row = 0; row < m_rows; row++) {
      const float* vector_in_batch = vector + batch * m_cols;
      float sum = 0.0f;
      for (int i = segments[row]; i < segments[row + 1]; i++) {
        const int block_start = indices[i] * kBlockSize;
        const float* vp = vector_in_batch + block_start;
        size_t vl = __riscv_vsetvl_e32m1(kBlockSize);
        vfloat32m1_t vv = __riscv_vle32_v_f32m1(vp, vl);
        vfloat32m1_t mv = __riscv_vle32_v_f32m1(matrix_ptr, vl);
        vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(vv, mv, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t s = __riscv_vfredusum_vs_f32m1_f32m1(prod, zero, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(s);
        matrix_ptr += kBlockSize;
      }
      result[batch * m_rows + row] += sum;
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result) {
  constexpr int kBlockSize = 16;
  for (int batch = 0; batch < n_batch; batch++) {
    const float* matrix_ptr = matrix;
    const uint8_t* ledger_ptr = ledger;
    for (int row = 0; row < m_rows; row++) {
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        const float* vector_in_batch = vector + batch * m_cols;
        size_t vl_block = __riscv_vsetvl_e32m4(kBlockSize);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl_block);
        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int block_start = *ledger_ptr++ * kBlockSize;
          vfloat32m4_t vv = __riscv_vle32_v_f32m4(
              vector_in_batch + block_start, vl_block);
          vfloat32m4_t mv = __riscv_vle32_v_f32m4(matrix_ptr, vl_block);
          acc = __riscv_vfmacc_vv_f32m4(acc, mv, vv, vl_block);
          matrix_ptr += kBlockSize;
        }
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t s = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero, vl_block);
        result[batch * m_rows + row] += __riscv_vfmv_f_s_f32m1_f32(s);
      }
    }
  }
}

// --- Sparse int8 matrix-vector multiply ---

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const int8_t* __restrict__ vector, const int32_t* __restrict__ bias_vector,
    int n_batch, const int32_t input_offset, const int32_t output_multiplier,
    int32_t output_shift, const int32_t* per_channel_scale,
    const int32_t* per_channel_shift, int32_t output_offset,
    const int32_t output_activation_min, const int32_t output_activation_max,
    int8_t* __restrict__ result) {
  constexpr int kBlockSize = 16;
  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t* matrix_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
      int32_t acc = 0;
      int32_t matrix_row_sum = 0;
      const int8_t* vector_in_batch = vector + batch * m_cols;
      for (int i = segments[row]; i < segments[row + 1]; ++i) {
        const int block_start = indices[i] * kBlockSize;
        const int8_t* vp = vector_in_batch + block_start;
        size_t vl = __riscv_vsetvl_e8m1(kBlockSize);
        vint8m1_t vv = __riscv_vle8_v_i8m1(vp, vl);
        vint8m1_t mv = __riscv_vle8_v_i8m1(matrix_ptr, vl);
        vint16m2_t prod16 = __riscv_vwmul_vv_i16m2(vv, mv, vl);
        vint32m4_t prod32 = __riscv_vsext_vf2_i32m4(prod16, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(prod32, zero, vl);
        acc += __riscv_vmv_x_s_i32m1_i32(s);
        vint16m2_t mv16 = __riscv_vsext_vf2_i16m2(mv, vl);
        vint32m4_t mv32 = __riscv_vsext_vf2_i32m4(mv16, vl);
        vint32m1_t ms = __riscv_vredsum_vs_i32m4_i32m1(mv32, zero, vl);
        matrix_row_sum += __riscv_vmv_x_s_i32m1_i32(ms);
        matrix_ptr += kBlockSize;
      }
      const int32_t bias_value = bias_vector != nullptr ? bias_vector[row] : 0;
      acc = acc + bias_value + input_offset * matrix_row_sum;
      acc = MultiplyByQuantizedMultiplier(
          acc, per_channel_scale ? per_channel_scale[row] : output_multiplier,
          per_channel_shift ? per_channel_shift[row] : output_shift);
      acc += output_offset;
      result[batch * m_rows + row] = static_cast<int8_t>(
          ActivationFunctionWithMinMax(acc, output_activation_min,
                                       output_activation_max));
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    const float* per_channel_scale) {
  constexpr int kBlockSize = 16;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    const int8_t* vec = vectors + batch * m_cols;
    const uint8_t* ledger_ptr = ledger;
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        int32_t dotprod = 0;
        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int col_index = *ledger_ptr++ * kBlockSize;
          size_t vl = __riscv_vsetvl_e8m1(kBlockSize);
          vint8m1_t v1 = __riscv_vle8_v_i8m1(vec + col_index, vl);
          vint8m1_t v2 = __riscv_vle8_v_i8m1(row_ptr, vl);
          vint16m2_t prod = __riscv_vwmul_vv_i16m2(v1, v2, vl);
          vint32m4_t prod32 = __riscv_vsext_vf2_i32m4(prod, vl);
          vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
          vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(prod32, zero, vl);
          dotprod += __riscv_vmv_x_s_i32m1_i32(s);
          row_ptr += kBlockSize;
        }
        float scaling = batch_scaling_factor;
        if (per_channel_scale) scaling *= per_channel_scale[row];
        result[batch * m_rows + row] += dotprod * scaling;
      }
    }
  }
}

// --- LayerNorm ---

void RvvApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                       const int32_t* bias, int32_t layer_norm_scale_a,
                       int32_t layer_norm_scale_b, int32_t variance_limit,
                       int n_batch, int n_input, int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16_t>::max();
  const int32_t int16_min = std::numeric_limits<int16_t>::min();
  const int32_t temp = 1048576 / n_input;

  for (int i = 0; i < n_batch; ++i) {
    int64_t sum = 0;
    int64_t sum_sq = 0;
    int j = 0;
    size_t vl;
    for (; j < n_input; j += vl) {
      vl = __riscv_vsetvl_e16m2(n_input - j);
      const int32_t index = i * n_input + j;
      vint16m2_t val = __riscv_vle16_v_i16m2(input + index, vl);
      vint32m4_t val32 = __riscv_vsext_vf2_i32m4(val, vl);
      vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(val32, zero32, vl);
      sum += __riscv_vmv_x_s_i32m1_i32(s);
      vint32m4_t sq = __riscv_vmul_vv_i32m4(val32, val32, vl);
      vint32m1_t ssq = __riscv_vredsum_vs_i32m4_i32m1(sq, zero32, vl);
      sum_sq += __riscv_vmv_x_s_i32m1_i32(ssq);
    }

    int32_t mean = static_cast<int32_t>(sum * 1024 / n_input);
    int64_t variance = sum_sq * temp - static_cast<int64_t>(mean) * mean;
    int32_t variance2 = static_cast<int32_t>(variance / 1048576);
    if (variance2 < 1) variance2 = variance_limit;
    int32_t stddev_inverse_a;
    int stddev_inverse_b;
    GetInvSqrtQuantizedMultiplierExp(variance2, -1, &stddev_inverse_a,
                                     &stddev_inverse_b);

    for (j = 0; j < n_input; j += vl) {
      vl = __riscv_vsetvl_e16m1(n_input - j);
      const int32_t index = i * n_input + j;
      vint16m1_t val = __riscv_vle16_v_i16m1(input + index, vl);
      vint32m2_t val32 = __riscv_vsext_vf2_i32m2(val, vl);
      val32 = __riscv_vsll_vx_i32m2(val32, 10, vl);
      val32 = __riscv_vsub_vx_i32m2(val32, mean, vl);
      val32 = rvv_utils::MultiplyByQuantizedMultiplier_m2(
          val32, stddev_inverse_a, stddev_inverse_b, vl);

      vint16m1_t lnw = __riscv_vle16_v_i16m1(layer_norm_weights + j, vl);
      vint32m2_t lnw32 = __riscv_vsext_vf2_i32m2(lnw, vl);
      vint32m2_t bias_v = __riscv_vle32_v_i32m2(bias + j, vl);

      vint64m4_t val3 = __riscv_vwmul_vv_i64m4(val32, lnw32, vl);
      vint64m4_t bias64 = __riscv_vsext_vf2_i64m4(bias_v, vl);
      val3 = __riscv_vadd_vv_i64m4(val3, bias64, vl);

      // (val3 + sign*512) / 1024
      vint64m4_t sign = __riscv_vsra_vx_i64m4(val3, 63, vl);
      vint64m4_t nudge = __riscv_vadd_vx_i64m4(
          __riscv_vand_vx_i64m4(sign, 1024, vl), 512, vl);
      vbool16_t neg = __riscv_vmslt_vx_i64m4_b16(val3, 0, vl);
      vint64m4_t val3_pos = __riscv_vadd_vv_i64m4(val3, nudge, vl);
      vint64m4_t val3_neg = __riscv_vsub_vv_i64m4(val3, nudge, vl);
      vint64m4_t val3_rnd = __riscv_vmerge_vvm_i64m4(val3_pos, val3_neg, neg, vl);
      val3_rnd = __riscv_vsra_vx_i64m4(val3_rnd, 10, vl);

      vint32m2_t val4 = __riscv_vnclip_wx_i32m2(val3_rnd, 0, __RISCV_VXRM_RDN, vl);
      val4 = rvv_utils::MultiplyByQuantizedMultiplier_m2(
          val4, layer_norm_scale_a, layer_norm_scale_b + 12, vl);
      val4 = __riscv_vmax_vx_i32m2(val4, int16_min, vl);
      val4 = __riscv_vmin_vx_i32m2(val4, int16_max, vl);
      vint16m1_t result = __riscv_vnclip_wx_i16m1(val4, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse16_v_i16m1(output + index, result, vl);
    }
  }
}

// Sigmoid and Tanh use gemmlowp fixed-point LUT which has no RVV equivalent.
void RvvApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                     int16_t* output) {
  PortableApplySigmoid(input, n_batch, n_input, output);
}

void RvvApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
                  int32_t n_input, int16_t* output) {
  PortableApplyTanh(integer_bits, input, n_batch, n_input, output);
}

// --- VectorBatchVectorCwiseProductAccumulate ---

void RvvVectorBatchVectorCwiseProductAccumulate(
    const int16_t* vector, int v_size, const int16_t* batch_vector, int n_batch,
    int32_t multiplier, int shift, int16_t* result) {
  for (int b = 0; b < n_batch; b++) {
    int v = 0;
    size_t vl;
    for (; v < v_size; v += vl) {
      vl = __riscv_vsetvl_e16m1(v_size - v);
      vint16m1_t a = __riscv_vle16_v_i16m1(vector + v, vl);
      vint16m1_t bv = __riscv_vle16_v_i16m1(batch_vector, vl);
      batch_vector += vl;

      vint32m2_t prod = __riscv_vwmul_vv_i32m2(a, bv, vl);
      prod = rvv_utils::MultiplyByQuantizedMultiplier_m2(prod, multiplier,
                                                          shift, vl);

      vint16m1_t prev = __riscv_vle16_v_i16m1(result, vl);
      vint32m2_t prev32 = __riscv_vsext_vf2_i32m2(prev, vl);
      prod = __riscv_vadd_vv_i32m2(prod, prev32, vl);
      prod = __riscv_vmax_vx_i32m2(prod, -32768, vl);
      prod = __riscv_vmin_vx_i32m2(prod, 32767, vl);
      vint16m1_t out = __riscv_vnclip_wx_i16m1(prod, 0, __RISCV_VXRM_RDN, vl);
      __riscv_vse16_v_i16m1(result, out, vl);
      result += vl;
    }
  }
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // USE_RVV
