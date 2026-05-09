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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_IMPL_H_

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

#ifdef USE_RVV

void RvvMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                            int m_cols, const float* vector,
                                            int n_batch, float* result);

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                            const int m_rows, const int m_cols,
                                            const int8_t* __restrict__ vectors,
                                            const float* scaling_factors,
                                            int n_batch,
                                            float* __restrict__ result);

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                            const int m_rows, const int m_cols,
                                            const int8_t* __restrict__ vectors,
                                            const float* scaling_factors,
                                            int n_batch, int32_t* scratch,
                                            float* __restrict__ result,
                                            CpuBackendContext* context);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

void RvvApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                       const int32_t* bias, int32_t layer_norm_scale_a,
                       int32_t layer_norm_scale_b, int32_t variance_limit,
                       int n_batch, int n_input, int16_t* output);

void RvvApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                     int16_t* output);

void RvvApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
                  int32_t n_input, int16_t* output);

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int shift, int16_t* output);

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2,
                 int32_t multiplier, int shift, int n_batch, int n_input,
                 int32_t output_zp, int8_t* output);

void RvvCwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int16_t* output);

void RvvCwiseClipping(float* vector, const int v_size,
                      const float clipping_value);
void RvvCwiseClipping(int16_t* vector, const int v_size,
                      const int16_t clipping_value);
void RvvCwiseClipping(int8_t* vector, const int v_size,
                      const int8_t clipping_value);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context);

void RvvMatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                       int32_t n_row, int32_t n_col,
                                       int32_t* output);

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const int8_t* __restrict__ vector, const int32_t* __restrict__ bias_vector,
    int n_batch, const int32_t input_offset, const int32_t output_multiplier,
    int32_t output_shift, const int32_t* per_channel_scale,
    const int32_t* per_channel_shift, int32_t output_offset,
    const int32_t output_activation_min, const int32_t output_activation_max,
    int8_t* __restrict__ result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    const float* per_channel_scale);

float RvvVectorVectorDotProduct(const float* vector1, const float* vector2,
                                int v_size);

void RvvSub1Vector(const float* vector, int v_size, float* result);
void RvvSub1Vector(const int16_t* vector, int v_size, int16_t* result);

void RvvVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                             float* result);

bool RvvIsZeroVector(const float* vector, int v_size);
bool RvvIsZeroVector(const int8_t* vector, int v_size);

void RvvSymmetricQuantizeFloats(const float* values, const int size,
                                int8_t* quantized_values, float* min,
                                float* max, float* scaling_factor);

void RvvSymmetricQuantizeFloats(const float* values, const int size,
                                int8_t* quantized_values, float min, float max,
                                float* scaling_factor);

void RvvAsymmetricQuantizeFloats(const float* values, const int size,
                                 int8_t* quantized_values,
                                 float* scaling_factor, int32_t* offset);

void RvvReductionSumVector(const float* input_vector, float* output_vector,
                           int output_size, int reduction_size);

void RvvReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                           int output_size, int reduction_size);

void RvvVectorBatchVectorCwiseProductAccumulate(
    const int16_t* vector, int v_size, const int16_t* batch_vector, int n_batch,
    int32_t multiplier, int shift, int16_t* result);

void RvvMeanStddevNormalization(const float* __restrict__ input_vector,
                                float* __restrict__ output_vector, int v_size,
                                int n_batch);

#endif  // USE_RVV

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_IMPL_H_
