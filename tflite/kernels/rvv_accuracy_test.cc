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
// Accuracy tests for RVV-optimized kernels vs scalar reference.
// Each test calls both the optimized path and the reference path with
// identical inputs, then compares outputs element-by-element.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/sub.h"
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/reference/add.h"
#include "tflite/kernels/internal/reference/integer_ops/add.h"
#include "tflite/kernels/internal/reference/integer_ops/mul.h"
#include "tflite/kernels/internal/reference/sub.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace {

constexpr int kTestSeed = 42;

std::vector<int8_t> RandomInt8(int size, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist(-128, 127);
  std::vector<int8_t> v(size);
  for (auto& x : v) x = static_cast<int8_t>(dist(rng));
  return v;
}

std::vector<int16_t> RandomInt16(int size, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist(-32768, 32767);
  std::vector<int16_t> v(size);
  for (auto& x : v) x = static_cast<int16_t>(dist(rng));
  return v;
}

std::vector<float> RandomFloat(int size, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> v(size);
  for (auto& x : v) x = dist(rng);
  return v;
}

ArithmeticParams MakeQuantizedAddParams() {
  ArithmeticParams p = {};
  p.input1_offset = 3;
  p.input2_offset = -5;
  p.output_offset = 2;
  p.left_shift = 5;
  p.input1_multiplier = 1073741824;
  p.input1_shift = -1;
  p.input2_multiplier = 1073741824;
  p.input2_shift = -2;
  p.output_multiplier = 1073741824;
  p.output_shift = -3;
  p.quantized_activation_min = -128;
  p.quantized_activation_max = 127;
  return p;
}

ArithmeticParams MakeQuantizedMulParams() {
  ArithmeticParams p = {};
  p.input1_offset = 2;
  p.input2_offset = -3;
  p.output_offset = 1;
  p.output_multiplier = 1073741824;
  p.output_shift = -4;
  p.quantized_activation_min = -128;
  p.quantized_activation_max = 127;
  return p;
}

ArithmeticParams MakeFloatAddParams() {
  ArithmeticParams p = {};
  p.float_activation_min = -100.0f;
  p.float_activation_max = 100.0f;
  return p;
}

ArithmeticParams MakeFloatMulParams() {
  ArithmeticParams p = {};
  p.float_activation_min = -1000.0f;
  p.float_activation_max = 1000.0f;
  return p;
}

// --- Float Add ---

class FloatAddTest : public ::testing::TestWithParam<int> {};

TEST_P(FloatAddTest, MatchesReference) {
  const int size = GetParam();
  std::mt19937 rng(kTestSeed);
  auto input1 = RandomFloat(size, rng);
  auto input2 = RandomFloat(size, rng);
  std::vector<float> ref_out(size), opt_out(size);

  auto params = MakeFloatAddParams();

  for (int i = 0; i < size; i++) {
    float x = input1[i] + input2[i];
    ref_out[i] = std::min(params.float_activation_max,
                          std::max(params.float_activation_min, x));
  }

  optimized_ops::AddElementwise(size, params, input1.data(), input2.data(),
                                opt_out.data());

  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(ref_out[i], opt_out[i], 1e-5f)
        << "Mismatch at index " << i << " (size=" << size << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, FloatAddTest,
                         ::testing::Values(1, 3, 7, 15, 16, 17, 31, 32, 33, 63,
                                           64, 100, 256, 1000, 1023, 1024));

// --- Float Mul ---

class FloatMulTest : public ::testing::TestWithParam<int> {};

TEST_P(FloatMulTest, MatchesReference) {
  const int size = GetParam();
  std::mt19937 rng(kTestSeed + 1);
  auto input1 = RandomFloat(size, rng);
  auto input2 = RandomFloat(size, rng);
  std::vector<float> ref_out(size), opt_out(size);

  auto params = MakeFloatMulParams();

  for (int i = 0; i < size; i++) {
    float x = input1[i] * input2[i];
    ref_out[i] = std::min(params.float_activation_max,
                          std::max(params.float_activation_min, x));
  }

  optimized_ops::MulElementwise(size, params, input1.data(), input2.data(),
                                opt_out.data());

  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(ref_out[i], opt_out[i], 1e-5f)
        << "Mismatch at index " << i << " (size=" << size << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, FloatMulTest,
                         ::testing::Values(1, 7, 16, 33, 64, 256, 1024));

// --- Int8 Add ---

class Int8AddTest : public ::testing::TestWithParam<int> {};

TEST_P(Int8AddTest, MatchesScalar) {
  const int size = GetParam();
  std::mt19937 rng(kTestSeed + 2);
  auto input1 = RandomInt8(size, rng);
  auto input2 = RandomInt8(size, rng);
  std::vector<int8_t> opt_out(size);

  auto params = MakeQuantizedAddParams();

  // Compute reference using scalar loop directly.
  std::vector<int8_t> ref_out(size);
  for (int i = 0; i < size; i++) {
    const int32_t input1_val = params.input1_offset + input1[i];
    const int32_t input2_val = params.input2_offset + input2[i];
    const int32_t shifted1 = input1_val * (1 << params.left_shift);
    const int32_t shifted2 = input2_val * (1 << params.left_shift);
    const int32_t scaled1 = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        shifted1, params.input1_multiplier, params.input1_shift);
    const int32_t scaled2 = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        shifted2, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled1 + scaled2;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    ref_out[i] = static_cast<int8_t>(
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output)));
  }

  optimized_integer_ops::AddElementwiseInt8(size, params, input1.data(),
                                            input2.data(), opt_out.data());

  for (int i = 0; i < size; i++) {
    EXPECT_LE(std::abs(static_cast<int>(ref_out[i]) -
                       static_cast<int>(opt_out[i])),
              1)
        << "Mismatch at index " << i << ": ref=" << static_cast<int>(ref_out[i])
        << " opt=" << static_cast<int>(opt_out[i]) << " (size=" << size << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, Int8AddTest,
                         ::testing::Values(1, 3, 7, 15, 16, 17, 31, 32, 33, 63,
                                           64, 100, 128, 255, 256, 1000, 1024));

// --- Int8 Mul ---

class Int8MulTest : public ::testing::TestWithParam<int> {};

TEST_P(Int8MulTest, MatchesScalar) {
  const int size = GetParam();
  std::mt19937 rng(kTestSeed + 3);
  auto input1 = RandomInt8(size, rng);
  auto input2 = RandomInt8(size, rng);
  std::vector<int8_t> opt_out(size);

  auto params = MakeQuantizedMulParams();

  std::vector<int8_t> ref_out(size);
  for (int i = 0; i < size; i++) {
    const int32_t input1_val = params.input1_offset + input1[i];
    const int32_t input2_val = params.input2_offset + input2[i];
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    ref_out[i] = static_cast<int8_t>(
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result)));
  }

  optimized_integer_ops::MulElementwise(size, params, input1.data(),
                                        input2.data(), opt_out.data());

  for (int i = 0; i < size; i++) {
    EXPECT_LE(std::abs(static_cast<int>(ref_out[i]) -
                       static_cast<int>(opt_out[i])),
              1)
        << "Mismatch at index " << i << " (size=" << size << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, Int8MulTest,
                         ::testing::Values(1, 7, 16, 33, 64, 128, 256, 1024));

// --- Int16 Sub ---

class Int16SubTest : public ::testing::TestWithParam<int> {};

TEST_P(Int16SubTest, MatchesScalar) {
  const int size = GetParam();
  std::mt19937 rng(kTestSeed + 4);
  auto input1 = RandomInt16(size, rng);
  auto input2 = RandomInt16(size, rng);
  std::vector<int16_t> opt_out(size);

  ArithmeticParams params = {};
  params.input1_offset = 0;
  params.input2_offset = 0;
  params.output_offset = 0;
  params.left_shift = 3;
  params.input1_multiplier = 1073741824;
  params.input1_shift = -1;
  params.input2_multiplier = 1073741824;
  params.input2_shift = -2;
  params.output_multiplier = 1073741824;
  params.output_shift = -3;
  params.quantized_activation_min = -32768;
  params.quantized_activation_max = 32767;

  std::vector<int16_t> ref_out(size);
  for (int i = 0; i < size; i++) {
    const int32_t v1 = params.input1_offset + input1[i];
    const int32_t v2 = params.input2_offset + input2[i];
    const int32_t s1 = v1 * (1 << params.left_shift);
    const int32_t s2 = v2 * (1 << params.left_shift);
    const int32_t sc1 = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        s1, params.input1_multiplier, params.input1_shift);
    const int32_t sc2 = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        s2, params.input2_multiplier, params.input2_shift);
    const int32_t raw = sc1 - sc2;
    const int32_t out = MultiplyByQuantizedMultiplierSmallerThanOneExp(
                            raw, params.output_multiplier, params.output_shift) +
                        params.output_offset;
    ref_out[i] = static_cast<int16_t>(
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, out)));
  }

  optimized_integer_ops::SubElementwiseInt16(size, params, input1.data(),
                                             input2.data(), opt_out.data());

  for (int i = 0; i < size; i++) {
    EXPECT_LE(std::abs(static_cast<int>(ref_out[i]) -
                       static_cast<int>(opt_out[i])),
              1)
        << "Mismatch at index " << i << " (size=" << size << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, Int16SubTest,
                         ::testing::Values(1, 7, 16, 33, 64, 128, 256, 1024));

// --- Float HardSwish ---

class FloatHardSwishTest : public ::testing::TestWithParam<int> {};

TEST_P(FloatHardSwishTest, MatchesReference) {
  const int size = GetParam();
  std::mt19937 rng(kTestSeed + 5);
  auto input = RandomFloat(size, rng);
  std::vector<float> ref_out(size), opt_out(size);

  for (int i = 0; i < size; i++) {
    float in = input[i];
    ref_out[i] = in * std::min(6.0f, std::max(0.0f, in + 3.0f)) * (1.0f / 6.0f);
  }

  RuntimeShape shape({1, 1, 1, size});
  optimized_ops::HardSwish(shape, input.data(), shape, opt_out.data());

  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(ref_out[i], opt_out[i], 1e-5f)
        << "Mismatch at index " << i << " (size=" << size << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, FloatHardSwishTest,
                         ::testing::Values(1, 7, 16, 33, 64, 256, 1024));

// --- Int8 MaxPool ---

TEST(Int8MaxPoolTest, MatchesReference) {
  const int batch = 1, input_h = 4, input_w = 4, depth = 32;
  const int filter_h = 2, filter_w = 2, stride = 2;
  const int output_h = 2, output_w = 2;

  std::mt19937 rng(kTestSeed + 6);
  auto input = RandomInt8(batch * input_h * input_w * depth, rng);

  PoolParams params;
  params.stride_height = stride;
  params.stride_width = stride;
  params.filter_height = filter_h;
  params.filter_width = filter_w;
  params.padding_values.height = 0;
  params.padding_values.width = 0;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  RuntimeShape input_shape({batch, input_h, input_w, depth});
  RuntimeShape output_shape({batch, output_h, output_w, depth});

  std::vector<int8_t> ref_out(batch * output_h * output_w * depth, -128);
  std::vector<int8_t> opt_out(batch * output_h * output_w * depth, -128);

  reference_ops::MaxPool(params, input_shape, input.data(), output_shape,
                         ref_out.data());
  optimized_integer_ops::MaxPool(params, input_shape, input.data(),
                                 output_shape, opt_out.data());

  for (int i = 0; i < static_cast<int>(ref_out.size()); i++) {
    EXPECT_EQ(ref_out[i], opt_out[i]) << "Mismatch at index " << i;
  }
}

// --- Int8 AveragePool ---

TEST(Int8AveragePoolTest, MatchesReference) {
  const int batch = 1, input_h = 4, input_w = 4, depth = 16;
  const int filter_h = 2, filter_w = 2, stride = 2;
  const int output_h = 2, output_w = 2;

  std::mt19937 rng(kTestSeed + 7);
  auto input = RandomInt8(batch * input_h * input_w * depth, rng);

  PoolParams params;
  params.stride_height = stride;
  params.stride_width = stride;
  params.filter_height = filter_h;
  params.filter_width = filter_w;
  params.padding_values.height = 0;
  params.padding_values.width = 0;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  RuntimeShape input_shape({batch, input_h, input_w, depth});
  RuntimeShape output_shape({batch, output_h, output_w, depth});

  std::vector<int8_t> ref_out(batch * output_h * output_w * depth, 0);
  std::vector<int8_t> opt_out(batch * output_h * output_w * depth, 0);

  reference_ops::AveragePool(params, input_shape, input.data(), output_shape,
                             ref_out.data());
  optimized_integer_ops::AveragePool(params, input_shape, input.data(),
                                     output_shape, opt_out.data());

  for (int i = 0; i < static_cast<int>(ref_out.size()); i++) {
    EXPECT_LE(std::abs(static_cast<int>(ref_out[i]) -
                       static_cast<int>(opt_out[i])),
              1)
        << "Mismatch at index " << i;
  }
}

// --- Int32 Mul ---

class Int32MulTest : public ::testing::TestWithParam<int> {};

TEST_P(Int32MulTest, MatchesScalar) {
  const int size = GetParam();
  std::mt19937 rng(kTestSeed + 8);
  std::uniform_int_distribution<int32_t> dist(-100, 100);
  std::vector<int32_t> input1(size), input2(size);
  for (auto& x : input1) x = dist(rng);
  for (auto& x : input2) x = dist(rng);
  std::vector<int32_t> ref_out(size), opt_out(size);

  ArithmeticParams params = {};
  params.quantized_activation_min = -100000;
  params.quantized_activation_max = 100000;

  for (int i = 0; i < size; i++) {
    int32_t product = input1[i] * input2[i];
    ref_out[i] = std::min(params.quantized_activation_max,
                          std::max(params.quantized_activation_min, product));
  }

  optimized_ops::MulElementwise(static_cast<int32_t>(size), params,
                                input1.data(), input2.data(), opt_out.data());

  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ref_out[i], opt_out[i]) << "Mismatch at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, Int32MulTest,
                         ::testing::Values(1, 7, 16, 33, 64, 128, 256, 1024));

}  // namespace
}  // namespace tflite
