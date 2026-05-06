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

#include <vector>

namespace tflite {
namespace tensor_utils {

#ifdef USE_RVV
namespace {

float HorizontalSum(vfloat32m1_t values, size_t vl) {
  const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  const vfloat32m1_t reduced =
      __riscv_vfredusum_vs_f32m1_f32m1(values, zero, vl);
  return __riscv_vfmv_f_s_f32m1_f32(reduced);
}

int32_t HorizontalSum(vint32m4_t values, size_t vl) {
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t reduced = __riscv_vredsum_vs_i32m4_i32m1(values, zero, vl);
  return __riscv_vmv_x_s_i32m1_i32(reduced);
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

}  // namespace

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

void RvvReductionSumVector(const float* input_vector, float* output_vector,
                           int output_size, int reduction_size) {
  for (int output_index = 0; output_index < output_size; ++output_index) {
    output_vector[output_index] =
        RvvReduceSumFloat(input_vector, reduction_size);
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

#endif  // USE_RVV

}  // namespace tensor_utils
}  // namespace tflite
