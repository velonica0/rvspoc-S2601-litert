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

bool RvvIsZeroVector(const float* vector, int v_size);

bool RvvIsZeroVector(const int8_t* vector, int v_size);

void RvvSymmetricQuantizeFloats(const float* values, int size,
                                int8_t* quantized_values, float* min_value,
                                float* max_value, float* scaling_factor);

void RvvSymmetricQuantizeFloats(const float* values, int size,
                                int8_t* quantized_values, float min_value,
                                float max_value, float* scaling_factor);

void RvvAsymmetricQuantizeFloats(const float* values, int size,
                                 int8_t* quantized_values,
                                 float* scaling_factor, int32_t* offset);

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

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

void RvvMatrixBatchVectorMultiply(const int8_t* input, int32_t input_zeropoint,
                                  const int8_t* input_to_gate_weights,
                                  int32_t input_to_gate_effective_scale_a,
                                  int32_t input_to_gate_effective_scale_b,
                                  int32_t n_batch, int32_t n_input,
                                  int32_t n_cell, int8_t* gate_output,
                                  int8_t gate_output_zp);

void RvvMatrixBatchVectorMultiply(const int16_t* hidden,
                                  const int8_t* hidden_to_output_weights,
                                  int32_t proj_effective_scale_a,
                                  int32_t proj_effective_scale_b,
                                  const int32_t* gate_bias, int32_t n_batch,
                                  int32_t n_hidden, int32_t n_output,
                                  int32_t output_zp, int8_t* proj_output);

void RvvMatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                       int32_t n_row, int32_t n_col,
                                       int32_t* output);

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* matrix, const int32_t* segments, const int32_t* indices,
    int m_rows, int m_cols, const float* vector, int n_batch, float* result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const float* matrix, const uint8_t* ledger, int m_rows, int m_cols,
    const float* vector, int n_batch, float* result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t* matrix, const int32_t* segments, const int32_t* indices,
    int m_rows, int m_cols, const int8_t* vector, const int32_t* bias_vector,
    int n_batch, int32_t input_offset, int32_t output_multiplier,
    int32_t output_shift, const int32_t* per_channel_scale,
    const int32_t* per_channel_shift, int32_t output_offset,
    int32_t output_activation_min, int32_t output_activation_max,
    int8_t* result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* matrix, const uint8_t* ledger, int m_rows, int m_cols,
    const int8_t* vectors, const float* scaling_factors, int n_batch,
    float* result, const float* per_channel_scale);

void RvvApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                       const int32_t* bias, int32_t layer_norm_scale_a,
                       int32_t layer_norm_scale_b, int32_t variance_limit,
                       int n_batch, int n_input, int16_t* output);

void RvvApplyLayerNormFloat(const int16_t* input,
                            const int16_t* layer_norm_weights,
                            int32_t layer_norm_scale_a,
                            int32_t layer_norm_scale_b, const int32_t* bias,
                            int n_batch, int n_input, int16_t* output);

void RvvApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                     int16_t* output);

void RvvApplySigmoidFloat(const int16_t* input, int32_t n_batch,
                          int32_t n_input, int16_t* output);

void RvvApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
                  int32_t n_input, int16_t* output);

void RvvApplyTanhFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                       int32_t integer_bits, int16_t* output);

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int shift, int16_t* output);

void RvvCwiseMul(const int16_t* input_1, const int16_t* input_2,
                 int32_t multiplier, int32_t shift, int32_t n_batch,
                 int32_t n_input, int32_t output_zp, int8_t* output);

void RvvCwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
                 int n_input, int16_t* output);

void RvvCwiseClipping(float* vector, int v_size, float clipping_value);

void RvvCwiseClipping(int16_t* vector, int v_size, int16_t clipping_value);

void RvvCwiseClipping(int8_t* vector, int v_size, int8_t clipping_value);

void RvvVectorBatchVectorCwiseProductAccumulate(const int16_t* vector,
                                                int v_size,
                                                const int16_t* batch_vector,
                                                int n_batch,
                                                int32_t multiplier, int shift,
                                                int16_t* result);

void RvvBatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                         const int16_t* vector2, int v_size,
                                         int n_batch, int32_t* result);

void RvvSub1Vector(const float* vector, int v_size, float* result);

void RvvSub1Vector(const int16_t* vector, int v_size, int16_t* result);

void RvvVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                             float* result);

void RvvReductionSumVector(const float* input_vector, float* output_vector,
                           int output_size, int reduction_size);

void RvvReductionSumVector(const int32_t* input_vector, int32_t* output_vector,
                           int output_size, int reduction_size);

void RvvReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                           int output_size, int reduction_size);

void RvvMeanStddevNormalization(const float* input_vector, float* output_vector,
                                int v_size, int n_batch);

void RvvTwoGateSaturatingAdd(const int8_t* input, int8_t input_zp,
                             const int8_t* recurrent, int8_t recurrent_zp,
                             int32_t input_effective_scale_a,
                             int32_t input_effective_scale_b,
                             int32_t recurrent_effective_scale_a,
                             int32_t recurrent_effective_scale_b,
                             int32_t n_batch, int32_t n_cell,
                             int16_t* output);

#define RVV_OR_PORTABLE(funcname, ...) Rvv##funcname(__VA_ARGS__)

#else

#define RVV_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)

#endif  // USE_RVV

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_IMPL_H_
