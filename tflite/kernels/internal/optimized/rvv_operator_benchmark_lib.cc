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

#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h"
#include "tflite/kernels/internal/optimized/integer_ops/leaky_relu.h"
#include "tflite/kernels/internal/optimized/integer_ops/lut.h"
#include "tflite/kernels/internal/optimized/integer_ops/mean.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tflite/kernels/internal/optimized/integer_ops/sub.h"

#include <cmath>
#include <cstdlib>
#include <limits>

namespace {

inline float ClampFloat(float value, float activation_min,
                        float activation_max) {
  return std::min(activation_max, std::max(activation_min, value));
}

void BenchmarkAddElementwise(int size, const tflite::ArithmeticParams& params,
                             const float* lhs, const float* rhs,
                             float* output) {
  int i = 0;
#ifdef USE_RVV
  for (; i < size;) {
    const size_t vl = __riscv_vsetvl_e32m4(size - i);
    const vfloat32m4_t lhs_v = __riscv_vle32_v_f32m4(lhs + i, vl);
    const vfloat32m4_t rhs_v = __riscv_vle32_v_f32m4(rhs + i, vl);
    vfloat32m4_t sum = __riscv_vfadd_vv_f32m4(lhs_v, rhs_v, vl);
    sum = __riscv_vfmax_vf_f32m4(sum, params.float_activation_min, vl);
    sum = __riscv_vfmin_vf_f32m4(sum, params.float_activation_max, vl);
    __riscv_vse32_v_f32m4(output + i, sum, vl);
    i += static_cast<int>(vl);
  }
  return;
#endif

  for (; i < size; ++i) {
    output[i] = ClampFloat(lhs[i] + rhs[i], params.float_activation_min,
                           params.float_activation_max);
  }
}

void BenchmarkAddScalarBroadcast(int size,
                                 const tflite::ArithmeticParams& params,
                                 float lhs_scalar, const float* rhs,
                                 float* output) {
  int i = 0;
#ifdef USE_RVV
  for (; i < size;) {
    const size_t vl = __riscv_vsetvl_e32m4(size - i);
    const vfloat32m4_t rhs_v = __riscv_vle32_v_f32m4(rhs + i, vl);
    vfloat32m4_t sum = __riscv_vfadd_vf_f32m4(rhs_v, lhs_scalar, vl);
    sum = __riscv_vfmax_vf_f32m4(sum, params.float_activation_min, vl);
    sum = __riscv_vfmin_vf_f32m4(sum, params.float_activation_max, vl);
    __riscv_vse32_v_f32m4(output + i, sum, vl);
    i += static_cast<int>(vl);
  }
  return;
#endif

  for (; i < size; ++i) {
    output[i] = ClampFloat(lhs_scalar + rhs[i], params.float_activation_min,
                           params.float_activation_max);
  }
}

void BenchmarkMulElementwise(int size, const tflite::ArithmeticParams& params,
                             const float* lhs, const float* rhs,
                             float* output) {
  int i = 0;
#ifdef USE_RVV
  for (; i < size;) {
    const size_t vl = __riscv_vsetvl_e32m4(size - i);
    const vfloat32m4_t lhs_v = __riscv_vle32_v_f32m4(lhs + i, vl);
    const vfloat32m4_t rhs_v = __riscv_vle32_v_f32m4(rhs + i, vl);
    vfloat32m4_t product = __riscv_vfmul_vv_f32m4(lhs_v, rhs_v, vl);
    product = __riscv_vfmax_vf_f32m4(product, params.float_activation_min, vl);
    product = __riscv_vfmin_vf_f32m4(product, params.float_activation_max, vl);
    __riscv_vse32_v_f32m4(output + i, product, vl);
    i += static_cast<int>(vl);
  }
  return;
#endif

  for (; i < size; ++i) {
    output[i] = ClampFloat(lhs[i] * rhs[i], params.float_activation_min,
                           params.float_activation_max);
  }
}

void BenchmarkMulScalarBroadcast(int size,
                                 const tflite::ArithmeticParams& params,
                                 float lhs_scalar, const float* rhs,
                                 float* output) {
  int i = 0;
#ifdef USE_RVV
  for (; i < size;) {
    const size_t vl = __riscv_vsetvl_e32m4(size - i);
    const vfloat32m4_t rhs_v = __riscv_vle32_v_f32m4(rhs + i, vl);
    vfloat32m4_t product = __riscv_vfmul_vf_f32m4(rhs_v, lhs_scalar, vl);
    product = __riscv_vfmax_vf_f32m4(product, params.float_activation_min, vl);
    product = __riscv_vfmin_vf_f32m4(product, params.float_activation_max, vl);
    __riscv_vse32_v_f32m4(output + i, product, vl);
    i += static_cast<int>(vl);
  }
  return;
#endif

  for (; i < size; ++i) {
    output[i] = ClampFloat(lhs_scalar * rhs[i], params.float_activation_min,
                           params.float_activation_max);
  }
}

void BenchmarkBroadcastAddDispatch(const tflite::ArithmeticParams& params,
                                   const tflite::RuntimeShape& input1_shape,
                                   const float* input1_data,
                                   const tflite::RuntimeShape& input2_shape,
                                   const float* input2_data,
                                   const tflite::RuntimeShape& output_shape,
                                   float* output_data) {
  if (params.broadcast_category ==
      tflite::BroadcastableOpCategory::kGenericBroadcast) {
    tflite::optimized_ops::BroadcastAdd6DSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
    return;
  }

  tflite::optimized_ops::BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data, BenchmarkAddElementwise,
      BenchmarkAddScalarBroadcast);
}

void BenchmarkBroadcastMulDispatch(const tflite::ArithmeticParams& params,
                                   const tflite::RuntimeShape& input1_shape,
                                   const float* input1_data,
                                   const tflite::RuntimeShape& input2_shape,
                                   const float* input2_data,
                                   const tflite::RuntimeShape& output_shape,
                                   float* output_data) {
  if (params.broadcast_category ==
      tflite::BroadcastableOpCategory::kGenericBroadcast) {
    tflite::optimized_ops::BroadcastMul6DSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
    return;
  }

  tflite::optimized_ops::BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data, BenchmarkMulElementwise,
      BenchmarkMulScalarBroadcast);
}

inline float ApplyActivation(float value, TfLiteFusedActivation activation) {
  switch (activation) {
    case kTfLiteActNone:
      return value;
    case kTfLiteActSigmoid:
      return 1.0f / (1.0f + std::exp(-value));
    case kTfLiteActTanh:
      return std::tanh(value);
    default:
      TFLITE_DCHECK(false);
      return value;
  }
}

void ApplyActivationToBuffer(const float* input, int size,
                             TfLiteFusedActivation activation,
                             float* output) {
  for (int i = 0; i < size; ++i) {
    output[i] = ApplyActivation(input[i], activation);
  }
}

void VectorBatchVectorAssign(const float* vector, int v_size, int n_batch,
                             float* batch_vector) {
  for (int b = 0; b < n_batch; ++b) {
    std::copy_n(vector, v_size, batch_vector + b * v_size);
  }
}

void VectorVectorCwiseProduct(const float* lhs, const float* rhs, int v_size,
                              float* output) {
  for (int i = 0; i < v_size; ++i) {
    output[i] = lhs[i] * rhs[i];
  }
}

#ifdef USE_RVV
float HorizontalSum(vfloat32m1_t values, size_t vl) {
  const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  const vfloat32m1_t reduced =
      __riscv_vfredusum_vs_f32m1_f32m1(values, zero, vl);
  return __riscv_vfmv_f_s_f32m1_f32(reduced);
}
#endif

float VectorVectorDotProduct(const float* lhs, const float* rhs, int v_size) {
#ifdef USE_RVV
  if (v_size <= 0) {
    return 0.0f;
  }
  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  const int step = static_cast<int>(vlmax);
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  int col = 0;
  const int full_cols = v_size / step * step;
  for (; col < full_cols; col += step) {
    const vfloat32m1_t lhs_vec = __riscv_vle32_v_f32m1(lhs + col, vlmax);
    const vfloat32m1_t rhs_vec = __riscv_vle32_v_f32m1(rhs + col, vlmax);
    acc = __riscv_vfmacc_vv_f32m1(acc, lhs_vec, rhs_vec, vlmax);
  }
  float dot = HorizontalSum(acc, vlmax);
  for (; col < v_size; ++col) {
    dot += lhs[col] * rhs[col];
  }
  return dot;
#else
  float dot = 0.0f;
  for (int i = 0; i < v_size; ++i) {
    dot += lhs[i] * rhs[i];
  }
  return dot;
#endif
}

void LstmEvalMatrixBatchVectorMultiplyAccumulate(
    const float* matrix, const float* vector, const float* result,
    float* output, int m_rows, int m_cols, int n_batch) {
  std::copy_n(result, m_rows * n_batch, output);
  for (int batch = 0; batch < n_batch; ++batch) {
    const float* vector_in_batch = vector + batch * m_cols;
    const float* matrix_ptr = matrix;
    float* output_in_batch = output + batch * m_rows;
    for (int row = 0; row < m_rows; ++row) {
      output_in_batch[row] +=
          VectorVectorDotProduct(matrix_ptr, vector_in_batch, m_cols);
      matrix_ptr += m_cols;
    }
  }
}

void FullyConnectedMatrixBatchVectorMultiplyAccumulate(
    const float* matrix, int m_rows, int m_cols, const float* vector,
    int n_batch, float* result) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const float* vector_in_batch = vector + batch * m_cols;
    const float* matrix_ptr = matrix;
    float* result_in_batch = result + batch * m_rows;
    for (int row = 0; row < m_rows; ++row) {
      result_in_batch[row] +=
          VectorVectorDotProduct(matrix_ptr, vector_in_batch, m_cols);
      matrix_ptr += m_cols;
    }
  }
}

void LstmGateFloatImpl(const float* input, const float* input_to_gate_weights,
                       const float* output_state,
                       const float* recurrent_to_gate_weights,
                       const float* gate_bias, int n_batch, int n_input,
                       int n_output, int n_cell,
                       TfLiteFusedActivation activation, float* gate,
                       float* scratch) {
  VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  LstmEvalMatrixBatchVectorMultiplyAccumulate(
      input_to_gate_weights, input, gate, scratch, n_cell, n_input, n_batch);
  LstmEvalMatrixBatchVectorMultiplyAccumulate(recurrent_to_gate_weights,
                                              output_state, scratch, gate,
                                              n_cell, n_output, n_batch);
  ApplyActivationToBuffer(gate, n_batch * n_cell, activation, gate);
}

void LstmOutputFloatImpl(const float* cell_state, const float* output_gate,
                         const float* projection_weights,
                         const float* projection_bias, int n_batch, int n_cell,
                         int n_output, TfLiteFusedActivation activation,
                         float* output_state, float* scratch,
                         float* projection_bias_scratch) {
  ApplyActivationToBuffer(cell_state, n_batch * n_cell, activation, scratch);
  VectorVectorCwiseProduct(output_gate, scratch, n_batch * n_cell, scratch);

  if (projection_weights == nullptr) {
    std::copy_n(scratch, n_batch * n_output, output_state);
    return;
  }

  if (projection_bias != nullptr) {
    VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                            projection_bias_scratch);
  } else {
    std::fill_n(projection_bias_scratch, n_batch * n_output, 0.0f);
  }

  LstmEvalMatrixBatchVectorMultiplyAccumulate(
      projection_weights, scratch, projection_bias_scratch, output_state,
      n_output, n_cell, n_batch);
}

tflite::ArithmeticParams MakeFloatParams(float activation_min,
                                         float activation_max) {
  tflite::ArithmeticParams params;
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;
  return params;
}

tflite::ArithmeticParams MakeInt8AddParams() {
  tflite::ArithmeticParams params{};
  params.left_shift = 2;
  params.input1_offset = -3;
  params.input1_multiplier = 1 << 30;
  params.input1_shift = 0;
  params.input2_offset = 5;
  params.input2_multiplier = 1 << 30;
  params.input2_shift = 0;
  params.output_offset = -1;
  params.output_multiplier = 1 << 30;
  params.output_shift = 0;
  params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  params.quantized_activation_max = std::numeric_limits<int8_t>::max();
  return params;
}

tflite::ArithmeticParams MakeInt16AddSubParams() {
  tflite::ArithmeticParams params{};
  params.left_shift = 2;
  params.input1_offset = -13;
  params.input1_multiplier = 1 << 30;
  params.input1_shift = 0;
  params.input2_offset = 9;
  params.input2_multiplier = 1 << 30;
  params.input2_shift = 0;
  params.output_offset = 3;
  params.output_multiplier = 1 << 30;
  params.output_shift = 0;
  params.quantized_activation_min = std::numeric_limits<int16_t>::min();
  params.quantized_activation_max = std::numeric_limits<int16_t>::max();
  return params;
}

tflite::ArithmeticParams MakeInt8MulParams() {
  tflite::ArithmeticParams params{};
  params.input1_offset = 2;
  params.input2_offset = -1;
  params.output_offset = 0;
  params.output_multiplier = 1 << 30;
  params.output_shift = -4;
  params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  params.quantized_activation_max = std::numeric_limits<int8_t>::max();
  return params;
}

tflite::LeakyReluParams MakeLeakyReluInt16Params() {
  tflite::LeakyReluParams params{};
  params.alpha = 0.125f;
  params.input_offset = 0;
  params.output_offset = 0;
  params.output_multiplier_identity = 1 << 30;
  params.output_shift_identity = 1;
  params.output_multiplier_alpha = 1 << 30;
  params.output_shift_alpha = -2;
  return params;
}

void AddFloatElementwiseImpl(int size, float activation_min,
                             float activation_max, const float* lhs,
                             const float* rhs, float* output) {
  const tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  const tflite::RuntimeShape shape({size});
  tflite::optimized_ops::Add(params, shape, lhs, shape, rhs, shape, output);
}

void AddFloatScalarBroadcastImpl(int size, float activation_min,
                                 float activation_max, float lhs_scalar,
                                 const float* rhs, float* output) {
  tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  const tflite::RuntimeShape scalar_shape({1});
  const tflite::RuntimeShape vector_shape({size});
  tflite::optimized_ops::ProcessBroadcastShapes(scalar_shape, vector_shape,
                                                &params);
  BenchmarkBroadcastAddDispatch(params, scalar_shape, &lhs_scalar, vector_shape,
                                rhs, vector_shape, output);
}

void AddFloatScalarKernelImpl(int size, float activation_min,
                              float activation_max, float lhs_scalar,
                              const float* rhs, float* output) {
  const tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  BenchmarkAddScalarBroadcast(size, params, lhs_scalar, rhs, output);
}

void AddInt8ElementwiseImpl(int size, const int8_t* lhs, const int8_t* rhs,
                            int8_t* output) {
  const tflite::ArithmeticParams params = MakeInt8AddParams();
  const tflite::RuntimeShape shape({size});
  tflite::optimized_integer_ops::Add(params, shape, lhs, shape, rhs, shape,
                                     output);
}

void AddInt8ScalarBroadcastImpl(int size, int8_t lhs_scalar, const int8_t* rhs,
                                int8_t* output) {
  tflite::ArithmeticParams params = MakeInt8AddParams();
  const tflite::RuntimeShape scalar_shape({1});
  const tflite::RuntimeShape vector_shape({size});
  tflite::optimized_ops::ProcessBroadcastShapes(scalar_shape, vector_shape,
                                                &params);
  tflite::optimized_integer_ops::BroadcastAddDispatch(
      params, scalar_shape, &lhs_scalar, vector_shape, rhs, vector_shape,
      output);
}

void AddInt8ScalarKernelImpl(int size, int8_t lhs_scalar, const int8_t* rhs,
                             int8_t* output) {
  const tflite::ArithmeticParams params = MakeInt8AddParams();
  tflite::optimized_integer_ops::AddScalarBroadcast(size, params, lhs_scalar,
                                                    rhs, output);
}

void AddInt16ElementwiseImpl(int size, const int16_t* lhs, const int16_t* rhs,
                             int16_t* output) {
  const tflite::ArithmeticParams params = MakeInt16AddSubParams();
  const tflite::RuntimeShape shape({size});
  tflite::optimized_integer_ops::Add(params, shape, lhs, shape, rhs, shape,
                                     output);
}

void MulFloatElementwiseImpl(int size, float activation_min,
                             float activation_max, const float* lhs,
                             const float* rhs, float* output) {
  const tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  const tflite::RuntimeShape shape({size});
  tflite::optimized_ops::Mul(params, shape, lhs, shape, rhs, shape, output);
}

void MulFloatScalarBroadcastImpl(int size, float activation_min,
                                 float activation_max, float lhs_scalar,
                                 const float* rhs, float* output) {
  tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  const tflite::RuntimeShape scalar_shape({1});
  const tflite::RuntimeShape vector_shape({size});
  tflite::optimized_ops::ProcessBroadcastShapes(scalar_shape, vector_shape,
                                                &params);
  BenchmarkBroadcastMulDispatch(params, scalar_shape, &lhs_scalar, vector_shape,
                                rhs, vector_shape, output);
}

void MulFloatScalarKernelImpl(int size, float activation_min,
                              float activation_max, float lhs_scalar,
                              const float* rhs, float* output) {
  const tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  BenchmarkMulScalarBroadcast(size, params, lhs_scalar, rhs, output);
}

void MulInt8ElementwiseImpl(int size, const int8_t* lhs, const int8_t* rhs,
                            int8_t* output) {
  const tflite::ArithmeticParams params = MakeInt8MulParams();
  const tflite::RuntimeShape shape({size});
  tflite::optimized_integer_ops::Mul(params, shape, lhs, shape, rhs, shape,
                                     output);
}

void MulInt8ScalarBroadcastImpl(int size, int8_t lhs_scalar, const int8_t* rhs,
                                int8_t* output) {
  tflite::ArithmeticParams params = MakeInt8MulParams();
  const tflite::RuntimeShape scalar_shape({1});
  const tflite::RuntimeShape vector_shape({size});
  tflite::optimized_ops::ProcessBroadcastShapes(scalar_shape, vector_shape,
                                                &params);
  tflite::optimized_integer_ops::BroadcastMulDispatch(
      params, scalar_shape, &lhs_scalar, vector_shape, rhs, vector_shape,
      output);
}

void MulInt8ScalarKernelImpl(int size, int8_t lhs_scalar, const int8_t* rhs,
                             int8_t* output) {
  const tflite::ArithmeticParams params = MakeInt8MulParams();
  tflite::optimized_integer_ops::MulSimpleBroadcast(size, params, lhs_scalar,
                                                    rhs, output);
}

void SubFloatElementwiseImpl(int size, float activation_min,
                             float activation_max, const float* lhs,
                             const float* rhs, float* output) {
  const tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  const tflite::RuntimeShape shape({size});
  tflite::optimized_ops::SubWithActivation(params, shape, lhs, shape, rhs,
                                           shape, output);
}

void DivFloatElementwiseImpl(int size, float activation_min,
                             float activation_max, const float* lhs,
                             const float* rhs, float* output) {
  const tflite::ArithmeticParams params =
      MakeFloatParams(activation_min, activation_max);
  const tflite::RuntimeShape shape({size});
  tflite::optimized_ops::Div(params, shape, lhs, shape, rhs, shape, output);
}

void SubInt16ElementwiseImpl(int size, const int16_t* lhs, const int16_t* rhs,
                             int16_t* output) {
  const tflite::ArithmeticParams params = MakeInt16AddSubParams();
  const tflite::RuntimeShape shape({size});
  tflite::optimized_integer_ops::Sub(params, shape, lhs, shape, rhs, shape,
                                     output);
}

void LeakyReluInt16OperatorImpl(int size, const int16_t* input,
                                int16_t* output) {
  const tflite::LeakyReluParams params = MakeLeakyReluInt16Params();
  const tflite::RuntimeShape shape({size});
  tflite::optimized_integer_ops::QuantizeLeakyRelu(params, shape, input, shape,
                                                   output);
}

void LookupTableUint8OperatorImpl(int size, const uint8_t* input,
                                  const uint8_t* lut, uint8_t* output) {
  tflite::optimized_integer_ops::LookupTable(input, size, lut, output);
}

void LookupTableInt8OperatorImpl(int size, const int8_t* input,
                                 const int8_t* lut, int8_t* output) {
  tflite::optimized_integer_ops::LookupTable(input, size, lut, output);
}

void MeanInt8OperatorImpl(int batches, int input_height, int input_width,
                          int depth, const int8_t* input, int8_t* output) {
  const int area = input_height * input_width;
  TFLITE_CHECK_GT(area, 0);
  TFLITE_CHECK_EQ(area & (area - 1), 0);
  int log2_area = 0;
  for (int value = area; value > 1; value >>= 1) {
    ++log2_area;
  }

  tflite::MeanParams params{};
  params.axis_count = 2;
  params.axis[0] = 1;
  params.axis[1] = 2;

  const int32_t multiplier = 1 << 30;
  const int shift = 1 - log2_area;
  const int32_t bias = 0;
  const tflite::RuntimeShape input_shape(
      {batches, input_height, input_width, depth});
  const tflite::RuntimeShape output_shape({batches, 1, 1, depth});
  tflite::optimized_integer_ops::MeanImpl(params, input_shape, input,
                                          multiplier, shift, bias,
                                          output_shape, output,
                                          /*start_depth=*/0,
                                          /*end_depth=*/depth);
}

void FullyConnectedFloatOperatorImpl(int n_batch, int input_size, int num_units,
                                     const float* input, const float* weights,
                                     const float* bias, float* output) {
  if (bias != nullptr) {
    VectorBatchVectorAssign(bias, num_units, n_batch, output);
  } else {
    std::fill_n(output, n_batch * num_units, 0.0f);
  }

  FullyConnectedMatrixBatchVectorMultiplyAccumulate(
      weights, num_units, input_size, input, n_batch, output);
  tflite::tensor_utils::ApplyActivationToVector(
      output, n_batch * num_units, kTfLiteActNone, output);
}

void LstmGateFloatOperatorImpl(int n_batch, int n_input, int n_output,
                               int n_cell, const float* input,
                               const float* input_to_gate_weights,
                               const float* output_state,
                               const float* recurrent_to_gate_weights,
                               const float* gate_bias, float* gate,
                               float* scratch) {
  LstmGateFloatImpl(input, input_to_gate_weights, output_state,
                    recurrent_to_gate_weights, gate_bias, n_batch, n_input,
                    n_output, n_cell, kTfLiteActSigmoid, gate, scratch);
}

void LstmOutputFloatOperatorImpl(int n_batch, int n_cell, int n_output,
                                 const float* cell_state,
                                 const float* output_gate,
                                 const float* projection_weights,
                                 const float* projection_bias,
                                 float* output_state, float* scratch,
                                 float* projection_bias_scratch) {
  LstmOutputFloatImpl(cell_state, output_gate, projection_weights,
                      projection_bias, n_batch, n_cell, n_output,
                      kTfLiteActTanh, output_state, scratch,
                      projection_bias_scratch);
}

int ComputeValidOutputSize(int input_size, int filter_size, int stride) {
  return (input_size - filter_size) / stride + 1;
}

tflite::PoolParams MakeValidUint8PoolParams(int filter_height, int filter_width,
                                            int stride_height,
                                            int stride_width) {
  tflite::PoolParams params{};
  params.activation = tflite::FusedActivationFunctionType::kNone;
  params.padding_type = tflite::PaddingType::kValid;
  params.padding_values = {0, 0, 0, 0};
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.quantized_activation_min = 0;
  params.quantized_activation_max = 255;
  params.float_activation_min = std::numeric_limits<float>::lowest();
  params.float_activation_max = std::numeric_limits<float>::max();
  return params;
}

tflite::PoolParams MakeValidInt8PoolParams(int filter_height, int filter_width,
                                           int stride_height,
                                           int stride_width) {
  tflite::PoolParams params = MakeValidUint8PoolParams(
      filter_height, filter_width, stride_height, stride_width);
  params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  params.quantized_activation_max = std::numeric_limits<int8_t>::max();
  return params;
}

void AveragePoolUint8OperatorImpl(int batches, int input_height, int input_width,
                                  int depth, int filter_height,
                                  int filter_width, int stride_height,
                                  int stride_width, const uint8_t* input,
                                  uint8_t* output) {
  const tflite::PoolParams params = MakeValidUint8PoolParams(
      filter_height, filter_width, stride_height, stride_width);
  const int output_height =
      ComputeValidOutputSize(input_height, filter_height, stride_height);
  const int output_width =
      ComputeValidOutputSize(input_width, filter_width, stride_width);
  const tflite::RuntimeShape input_shape(
      {batches, input_height, input_width, depth});
  const tflite::RuntimeShape output_shape(
      {batches, output_height, output_width, depth});
  if (!tflite::optimized_ops::AveragePool(params, input_shape, input,
                                          output_shape, output)) {
    std::abort();
  }
}

void MaxPoolUint8OperatorImpl(int batches, int input_height, int input_width,
                              int depth, int filter_height, int filter_width,
                              int stride_height, int stride_width,
                              const uint8_t* input, uint8_t* output) {
  const tflite::PoolParams params = MakeValidUint8PoolParams(
      filter_height, filter_width, stride_height, stride_width);
  const int output_height =
      ComputeValidOutputSize(input_height, filter_height, stride_height);
  const int output_width =
      ComputeValidOutputSize(input_width, filter_width, stride_width);
  const tflite::RuntimeShape input_shape(
      {batches, input_height, input_width, depth});
  const tflite::RuntimeShape output_shape(
      {batches, output_height, output_width, depth});
  tflite::optimized_ops::MaxPool(params, input_shape, input, output_shape,
                                 output);
}

void AveragePoolInt8OperatorImpl(int batches, int input_height, int input_width,
                                 int depth, int filter_height,
                                 int filter_width, int stride_height,
                                 int stride_width, const int8_t* input,
                                 int8_t* output) {
  const tflite::PoolParams params = MakeValidInt8PoolParams(
      filter_height, filter_width, stride_height, stride_width);
  const int output_height =
      ComputeValidOutputSize(input_height, filter_height, stride_height);
  const int output_width =
      ComputeValidOutputSize(input_width, filter_width, stride_width);
  const tflite::RuntimeShape input_shape(
      {batches, input_height, input_width, depth});
  const tflite::RuntimeShape output_shape(
      {batches, output_height, output_width, depth});
  if (!tflite::optimized_integer_ops::AveragePool(params, input_shape, input,
                                                  output_shape, output)) {
    std::abort();
  }
}

void MaxPoolInt8OperatorImpl(int batches, int input_height, int input_width,
                             int depth, int filter_height, int filter_width,
                             int stride_height, int stride_width,
                             const int8_t* input, int8_t* output) {
  const tflite::PoolParams params = MakeValidInt8PoolParams(
      filter_height, filter_width, stride_height, stride_width);
  const int output_height =
      ComputeValidOutputSize(input_height, filter_height, stride_height);
  const int output_width =
      ComputeValidOutputSize(input_width, filter_width, stride_width);
  const tflite::RuntimeShape input_shape(
      {batches, input_height, input_width, depth});
  const tflite::RuntimeShape output_shape(
      {batches, output_height, output_width, depth});
  tflite::optimized_integer_ops::MaxPool(params, input_shape, input,
                                         output_shape, output);
}

void DepthwiseConvInt8OperatorImpl(
    int batches, int input_height, int input_width, int input_depth,
    int filter_height, int filter_width, int depth_multiplier,
    int stride_height, int stride_width, const int8_t* input,
    const int8_t* filter, const int32_t* bias,
    const int32_t* output_multiplier, const int32_t* output_shift,
    int8_t* output) {
  tflite::DepthwiseParams params{};
  params.padding_type = tflite::PaddingType::kValid;
  params.padding_values = {0, 0, 0, 0};
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.depth_multiplier = depth_multiplier;
  params.input_offset = 0;
  params.weights_offset = 0;
  params.output_offset = 0;
  params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  params.quantized_activation_max = std::numeric_limits<int8_t>::max();
  params.float_activation_min = std::numeric_limits<float>::lowest();
  params.float_activation_max = std::numeric_limits<float>::max();

  const int output_height =
      ComputeValidOutputSize(input_height, filter_height, stride_height);
  const int output_width =
      ComputeValidOutputSize(input_width, filter_width, stride_width);
  const int output_depth = input_depth * depth_multiplier;

  const tflite::RuntimeShape input_shape(
      {batches, input_height, input_width, input_depth});
  const tflite::RuntimeShape filter_shape(
      {1, filter_height, filter_width, output_depth});
  const tflite::RuntimeShape bias_shape({output_depth});
  const tflite::RuntimeShape output_shape(
      {batches, output_height, output_width, output_depth});
  tflite::optimized_integer_ops::depthwise_conv::DepthwiseConvGeneral(
      params, output_multiplier, output_shift, input_shape, input, filter_shape,
      filter, bias_shape, bias, output_shape, output,
      /*thread_start=*/0, /*thread_end=*/output_height, /*thread_dim=*/1);
}

template <typename T>
tflite::HardSwishParams MakeHardSwishParamsForBenchmark() {
  tflite::HardSwishParams params{};
  params.input_zero_point = std::is_same<T, uint8_t>::value ? 128 : 0;
  params.output_zero_point = std::is_same<T, uint8_t>::value ? 128 : 0;
  // Fixed benchmark quantization:
  //   input_scale = output_scale = 1 / 32
  // This gives stable reference-compatible params without linking
  // quantization_util.cc into the micro benchmark.
  params.output_multiplier_fixedpoint_int16 = 16384;
  params.output_multiplier_exponent = -6;
  params.reluish_multiplier_fixedpoint_int16 = 21845;
  params.reluish_multiplier_exponent = 2;
  return params;
}

template <typename T>
void HardSwishQuantizedOperatorImpl(int size, const T* input, T* output) {
  const tflite::HardSwishParams params = MakeHardSwishParamsForBenchmark<T>();
  const tflite::RuntimeShape shape({size});
  tflite::optimized_ops::HardSwish(params, shape, input, shape, output);
}

template <typename T>
void AffineQuantizeOperatorImpl(int size, int32_t zero_point, float scale,
                                const float* input, T* output) {
  tflite::QuantizationParams params;
  params.zero_point = zero_point;
  params.scale = scale;
  const tflite::RuntimeShape shape({size});
  tflite::optimized_ops::AffineQuantize(params, shape, input, shape, output);
}

int ArgMinFloatOperatorImpl(const float* input, int size) {
  return tflite::optimized_ops::ArgMinVector(input, size);
}

int ArgMaxFloatOperatorImpl(const float* input, int size) {
  return tflite::optimized_ops::ArgMaxVector(input, size);
}

int ArgMaxInt8OperatorImpl(const int8_t* input, int size) {
  return tflite::optimized_ops::ArgMaxVector(input, size);
}

int ArgMaxUint8OperatorImpl(const uint8_t* input, int size) {
  return tflite::optimized_ops::ArgMaxVector(input, size);
}

}  // namespace

#if defined(TFLITE_RVV_OPERATOR_BENCH_VARIANT_RVV)

extern "C" void RvvAddFloatElementwise(int size, float activation_min,
                                        float activation_max, const float* lhs,
                                        const float* rhs, float* output) {
  AddFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void RvvAddInt8Elementwise(int size, const int8_t* lhs,
                                      const int8_t* rhs, int8_t* output) {
  AddInt8ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void RvvAddInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output) {
  AddInt8ScalarBroadcastImpl(size, lhs_scalar, rhs, output);
}

extern "C" void RvvAddInt8ScalarKernel(int size, int8_t lhs_scalar,
                                       const int8_t* rhs, int8_t* output) {
  AddInt8ScalarKernelImpl(size, lhs_scalar, rhs, output);
}

extern "C" void RvvAddInt16Elementwise(int size, const int16_t* lhs,
                                       const int16_t* rhs, int16_t* output) {
  AddInt16ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void RvvAddFloatScalarBroadcast(int size, float activation_min,
                                            float activation_max,
                                            float lhs_scalar,
                                            const float* rhs, float* output) {
  AddFloatScalarBroadcastImpl(size, activation_min, activation_max, lhs_scalar,
                              rhs, output);
}

extern "C" void RvvAddFloatScalarKernel(int size, float activation_min,
                                         float activation_max,
                                         float lhs_scalar,
                                         const float* rhs, float* output) {
  AddFloatScalarKernelImpl(size, activation_min, activation_max, lhs_scalar,
                           rhs, output);
}

extern "C" void RvvMulFloatElementwise(int size, float activation_min,
                                        float activation_max, const float* lhs,
                                        const float* rhs, float* output) {
  MulFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void RvvMulInt8Elementwise(int size, const int8_t* lhs,
                                      const int8_t* rhs, int8_t* output) {
  MulInt8ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void RvvMulInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output) {
  MulInt8ScalarBroadcastImpl(size, lhs_scalar, rhs, output);
}

extern "C" void RvvMulInt8ScalarKernel(int size, int8_t lhs_scalar,
                                       const int8_t* rhs, int8_t* output) {
  MulInt8ScalarKernelImpl(size, lhs_scalar, rhs, output);
}

extern "C" void RvvMulFloatScalarBroadcast(int size, float activation_min,
                                            float activation_max,
                                            float lhs_scalar,
                                            const float* rhs, float* output) {
  MulFloatScalarBroadcastImpl(size, activation_min, activation_max, lhs_scalar,
                              rhs, output);
}

extern "C" void RvvMulFloatScalarKernel(int size, float activation_min,
                                         float activation_max,
                                         float lhs_scalar,
                                         const float* rhs, float* output) {
  MulFloatScalarKernelImpl(size, activation_min, activation_max, lhs_scalar,
                           rhs, output);
}

extern "C" void RvvSubFloatElementwise(int size, float activation_min,
                                        float activation_max, const float* lhs,
                                        const float* rhs, float* output) {
  SubFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void RvvSubInt16Elementwise(int size, const int16_t* lhs,
                                       const int16_t* rhs, int16_t* output) {
  SubInt16ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void RvvDivFloatElementwise(int size, float activation_min,
                                        float activation_max, const float* lhs,
                                        const float* rhs, float* output) {
  DivFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void RvvFullyConnectedFloatOperator(
    int n_batch, int input_size, int num_units, const float* input,
    const float* weights, const float* bias, float* output) {
  FullyConnectedFloatOperatorImpl(n_batch, input_size, num_units, input,
                                  weights, bias, output);
}

extern "C" void RvvLstmGateFloatOperator(
    int n_batch, int n_input, int n_output, int n_cell, const float* input,
    const float* input_to_gate_weights, const float* output_state,
    const float* recurrent_to_gate_weights, const float* gate_bias,
    float* gate, float* scratch) {
  LstmGateFloatOperatorImpl(n_batch, n_input, n_output, n_cell, input,
                            input_to_gate_weights, output_state,
                            recurrent_to_gate_weights, gate_bias, gate,
                            scratch);
}

extern "C" void RvvLstmOutputFloatOperator(
    int n_batch, int n_cell, int n_output, const float* cell_state,
    const float* output_gate, const float* projection_weights,
    const float* projection_bias, float* output_state, float* scratch,
    float* projection_bias_scratch) {
  LstmOutputFloatOperatorImpl(n_batch, n_cell, n_output, cell_state,
                              output_gate, projection_weights, projection_bias,
                              output_state, scratch, projection_bias_scratch);
}

extern "C" void RvvAveragePoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output) {
  AveragePoolUint8OperatorImpl(batches, input_height, input_width, depth,
                               filter_height, filter_width, stride_height,
                               stride_width, input, output);
}

extern "C" void RvvMaxPoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output) {
  MaxPoolUint8OperatorImpl(batches, input_height, input_width, depth,
                           filter_height, filter_width, stride_height,
                           stride_width, input, output);
}

extern "C" void RvvAveragePoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output) {
  AveragePoolInt8OperatorImpl(batches, input_height, input_width, depth,
                              filter_height, filter_width, stride_height,
                              stride_width, input, output);
}

extern "C" void RvvMaxPoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output) {
  MaxPoolInt8OperatorImpl(batches, input_height, input_width, depth,
                          filter_height, filter_width, stride_height,
                          stride_width, input, output);
}

extern "C" void RvvDepthwiseConvInt8Operator(
    int batches, int input_height, int input_width, int input_depth,
    int filter_height, int filter_width, int depth_multiplier,
    int stride_height, int stride_width, const int8_t* input,
    const int8_t* filter, const int32_t* bias,
    const int32_t* output_multiplier, const int32_t* output_shift,
    int8_t* output) {
  DepthwiseConvInt8OperatorImpl(
      batches, input_height, input_width, input_depth, filter_height,
      filter_width, depth_multiplier, stride_height, stride_width, input,
      filter, bias, output_multiplier, output_shift, output);
}

extern "C" void RvvHardSwishUint8Operator(int size, const uint8_t* input,
                                          uint8_t* output) {
  HardSwishQuantizedOperatorImpl(size, input, output);
}

extern "C" void RvvHardSwishInt8Operator(int size, const int8_t* input,
                                         int8_t* output) {
  HardSwishQuantizedOperatorImpl(size, input, output);
}

extern "C" void RvvAffineQuantizeInt8Operator(int size, int32_t zero_point,
                                              float scale, const float* input,
                                              int8_t* output) {
  AffineQuantizeOperatorImpl(size, zero_point, scale, input, output);
}

extern "C" void RvvAffineQuantizeUint8Operator(int size, int32_t zero_point,
                                               float scale, const float* input,
                                               uint8_t* output) {
  AffineQuantizeOperatorImpl(size, zero_point, scale, input, output);
}

extern "C" void RvvAffineQuantizeInt16Operator(int size, int32_t zero_point,
                                               float scale, const float* input,
                                               int16_t* output) {
  AffineQuantizeOperatorImpl(size, zero_point, scale, input, output);
}

extern "C" void RvvLeakyReluInt16Operator(int size, const int16_t* input,
                                          int16_t* output) {
  LeakyReluInt16OperatorImpl(size, input, output);
}

extern "C" void RvvLookupTableUint8Operator(int size, const uint8_t* input,
                                            const uint8_t* lut,
                                            uint8_t* output) {
  LookupTableUint8OperatorImpl(size, input, lut, output);
}

extern "C" void RvvLookupTableInt8Operator(int size, const int8_t* input,
                                           const int8_t* lut,
                                           int8_t* output) {
  LookupTableInt8OperatorImpl(size, input, lut, output);
}

extern "C" void RvvMeanInt8Operator(int batches, int input_height,
                                    int input_width, int depth,
                                    const int8_t* input, int8_t* output) {
  MeanInt8OperatorImpl(batches, input_height, input_width, depth, input,
                       output);
}

extern "C" int RvvArgMinFloatOperator(const float* input, int size) {
  return ArgMinFloatOperatorImpl(input, size);
}

extern "C" int RvvArgMaxFloatOperator(const float* input, int size) {
  return ArgMaxFloatOperatorImpl(input, size);
}

extern "C" int RvvArgMaxInt8Operator(const int8_t* input, int size) {
  return ArgMaxInt8OperatorImpl(input, size);
}

extern "C" int RvvArgMaxUint8Operator(const uint8_t* input, int size) {
  return ArgMaxUint8OperatorImpl(input, size);
}

#else

extern "C" void ScalarAddFloatElementwise(int size, float activation_min,
                                           float activation_max,
                                           const float* lhs, const float* rhs,
                                           float* output) {
  AddFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void ScalarAddInt8Elementwise(int size, const int8_t* lhs,
                                         const int8_t* rhs, int8_t* output) {
  AddInt8ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void ScalarAddInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                             const int8_t* rhs,
                                             int8_t* output) {
  AddInt8ScalarBroadcastImpl(size, lhs_scalar, rhs, output);
}

extern "C" void ScalarAddInt8ScalarKernel(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output) {
  AddInt8ScalarKernelImpl(size, lhs_scalar, rhs, output);
}

extern "C" void ScalarAddInt16Elementwise(int size, const int16_t* lhs,
                                          const int16_t* rhs,
                                          int16_t* output) {
  AddInt16ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void ScalarAddFloatScalarBroadcast(int size, float activation_min,
                                               float activation_max,
                                               float lhs_scalar,
                                               const float* rhs,
                                               float* output) {
  AddFloatScalarBroadcastImpl(size, activation_min, activation_max, lhs_scalar,
                              rhs, output);
}

extern "C" void ScalarAddFloatScalarKernel(int size, float activation_min,
                                            float activation_max,
                                            float lhs_scalar,
                                            const float* rhs,
                                            float* output) {
  AddFloatScalarKernelImpl(size, activation_min, activation_max, lhs_scalar,
                           rhs, output);
}

extern "C" void ScalarMulFloatElementwise(int size, float activation_min,
                                           float activation_max,
                                           const float* lhs, const float* rhs,
                                           float* output) {
  MulFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void ScalarMulInt8Elementwise(int size, const int8_t* lhs,
                                         const int8_t* rhs, int8_t* output) {
  MulInt8ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void ScalarMulInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                             const int8_t* rhs,
                                             int8_t* output) {
  MulInt8ScalarBroadcastImpl(size, lhs_scalar, rhs, output);
}

extern "C" void ScalarMulInt8ScalarKernel(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output) {
  MulInt8ScalarKernelImpl(size, lhs_scalar, rhs, output);
}

extern "C" void ScalarMulFloatScalarBroadcast(int size, float activation_min,
                                               float activation_max,
                                               float lhs_scalar,
                                               const float* rhs,
                                               float* output) {
  MulFloatScalarBroadcastImpl(size, activation_min, activation_max, lhs_scalar,
                              rhs, output);
}

extern "C" void ScalarMulFloatScalarKernel(int size, float activation_min,
                                            float activation_max,
                                            float lhs_scalar,
                                            const float* rhs,
                                            float* output) {
  MulFloatScalarKernelImpl(size, activation_min, activation_max, lhs_scalar,
                           rhs, output);
}

extern "C" void ScalarSubFloatElementwise(int size, float activation_min,
                                           float activation_max,
                                           const float* lhs, const float* rhs,
                                           float* output) {
  SubFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void ScalarSubInt16Elementwise(int size, const int16_t* lhs,
                                          const int16_t* rhs,
                                          int16_t* output) {
  SubInt16ElementwiseImpl(size, lhs, rhs, output);
}

extern "C" void ScalarDivFloatElementwise(int size, float activation_min,
                                           float activation_max,
                                           const float* lhs, const float* rhs,
                                           float* output) {
  DivFloatElementwiseImpl(size, activation_min, activation_max, lhs, rhs,
                          output);
}

extern "C" void ScalarFullyConnectedFloatOperator(
    int n_batch, int input_size, int num_units, const float* input,
    const float* weights, const float* bias, float* output) {
  FullyConnectedFloatOperatorImpl(n_batch, input_size, num_units, input,
                                  weights, bias, output);
}

extern "C" void ScalarLstmGateFloatOperator(
    int n_batch, int n_input, int n_output, int n_cell, const float* input,
    const float* input_to_gate_weights, const float* output_state,
    const float* recurrent_to_gate_weights, const float* gate_bias,
    float* gate, float* scratch) {
  LstmGateFloatOperatorImpl(n_batch, n_input, n_output, n_cell, input,
                            input_to_gate_weights, output_state,
                            recurrent_to_gate_weights, gate_bias, gate,
                            scratch);
}

extern "C" void ScalarLstmOutputFloatOperator(
    int n_batch, int n_cell, int n_output, const float* cell_state,
    const float* output_gate, const float* projection_weights,
    const float* projection_bias, float* output_state, float* scratch,
    float* projection_bias_scratch) {
  LstmOutputFloatOperatorImpl(n_batch, n_cell, n_output, cell_state,
                              output_gate, projection_weights, projection_bias,
                              output_state, scratch, projection_bias_scratch);
}

extern "C" void ScalarAveragePoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output) {
  AveragePoolUint8OperatorImpl(batches, input_height, input_width, depth,
                               filter_height, filter_width, stride_height,
                               stride_width, input, output);
}

extern "C" void ScalarMaxPoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output) {
  MaxPoolUint8OperatorImpl(batches, input_height, input_width, depth,
                           filter_height, filter_width, stride_height,
                           stride_width, input, output);
}

extern "C" void ScalarAveragePoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output) {
  AveragePoolInt8OperatorImpl(batches, input_height, input_width, depth,
                              filter_height, filter_width, stride_height,
                              stride_width, input, output);
}

extern "C" void ScalarMaxPoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output) {
  MaxPoolInt8OperatorImpl(batches, input_height, input_width, depth,
                          filter_height, filter_width, stride_height,
                          stride_width, input, output);
}

extern "C" void ScalarDepthwiseConvInt8Operator(
    int batches, int input_height, int input_width, int input_depth,
    int filter_height, int filter_width, int depth_multiplier,
    int stride_height, int stride_width, const int8_t* input,
    const int8_t* filter, const int32_t* bias,
    const int32_t* output_multiplier, const int32_t* output_shift,
    int8_t* output) {
  DepthwiseConvInt8OperatorImpl(
      batches, input_height, input_width, input_depth, filter_height,
      filter_width, depth_multiplier, stride_height, stride_width, input,
      filter, bias, output_multiplier, output_shift, output);
}

extern "C" void ScalarHardSwishUint8Operator(int size, const uint8_t* input,
                                             uint8_t* output) {
  HardSwishQuantizedOperatorImpl(size, input, output);
}

extern "C" void ScalarHardSwishInt8Operator(int size, const int8_t* input,
                                            int8_t* output) {
  HardSwishQuantizedOperatorImpl(size, input, output);
}

extern "C" void ScalarAffineQuantizeInt8Operator(
    int size, int32_t zero_point, float scale, const float* input,
    int8_t* output) {
  AffineQuantizeOperatorImpl(size, zero_point, scale, input, output);
}

extern "C" void ScalarAffineQuantizeUint8Operator(
    int size, int32_t zero_point, float scale, const float* input,
    uint8_t* output) {
  AffineQuantizeOperatorImpl(size, zero_point, scale, input, output);
}

extern "C" void ScalarAffineQuantizeInt16Operator(
    int size, int32_t zero_point, float scale, const float* input,
    int16_t* output) {
  AffineQuantizeOperatorImpl(size, zero_point, scale, input, output);
}

extern "C" void ScalarLeakyReluInt16Operator(int size, const int16_t* input,
                                             int16_t* output) {
  LeakyReluInt16OperatorImpl(size, input, output);
}

extern "C" void ScalarLookupTableUint8Operator(int size, const uint8_t* input,
                                               const uint8_t* lut,
                                               uint8_t* output) {
  LookupTableUint8OperatorImpl(size, input, lut, output);
}

extern "C" void ScalarLookupTableInt8Operator(int size, const int8_t* input,
                                              const int8_t* lut,
                                              int8_t* output) {
  LookupTableInt8OperatorImpl(size, input, lut, output);
}

extern "C" void ScalarMeanInt8Operator(int batches, int input_height,
                                       int input_width, int depth,
                                       const int8_t* input, int8_t* output) {
  MeanInt8OperatorImpl(batches, input_height, input_width, depth, input,
                       output);
}

extern "C" int ScalarArgMinFloatOperator(const float* input, int size) {
  return ArgMinFloatOperatorImpl(input, size);
}

extern "C" int ScalarArgMaxFloatOperator(const float* input, int size) {
  return ArgMaxFloatOperatorImpl(input, size);
}

extern "C" int ScalarArgMaxInt8Operator(const int8_t* input, int size) {
  return ArgMaxInt8OperatorImpl(input, size);
}

extern "C" int ScalarArgMaxUint8Operator(const uint8_t* input, int size) {
  return ArgMaxUint8OperatorImpl(input, size);
}

#endif
