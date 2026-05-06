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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_IMPL_H_

#include <cstdint>

#include "tflite/kernels/internal/optimized/rvv_check.h"

namespace tflite {

class CpuBackendContext;

namespace tensor_utils {

#ifdef USE_RVV

float RvvVectorVectorDotProduct(const float* vector1, const float* vector2,
                                int v_size);

void RvvMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                            int m_cols, const float* vector,
                                            int n_batch, float* result);

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t* matrix, int m_rows,
                                            int m_cols, const int8_t* vectors,
                                            const float* scaling_factors,
                                            int n_batch, float* result);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* matrix, int m_rows, int m_cols, const int8_t* vectors,
    const float* scaling_factors, int n_batch, int32_t* scratch,
    float* result, CpuBackendContext* context);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* matrix, int m_rows, int m_cols, const int8_t* vectors,
    const float* scaling_factors, int n_batch, float* result,
    const float* per_channel_scale, const int32_t* input_offset,
    int32_t* scratch, int32_t* row_sums, bool* compute_row_sums,
    CpuBackendContext* context);

void RvvMatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                       int32_t n_row, int32_t n_col,
                                       int32_t* output);

void RvvReductionSumVector(const float* input_vector, float* output_vector,
                           int output_size, int reduction_size);

void RvvReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                           int output_size, int reduction_size);

#define RVV_OR_PORTABLE(funcname, ...) Rvv##funcname(__VA_ARGS__)

#else

#define RVV_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)

#endif  // USE_RVV

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_IMPL_H_
