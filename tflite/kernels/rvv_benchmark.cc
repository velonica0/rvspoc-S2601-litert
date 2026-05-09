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
// Benchmarks for RVV-optimized kernels.
// On RVV hardware, the optimized path uses RVV intrinsics.
// On non-RVV hardware, it falls back to scalar (same as reference).

#include <cstdint>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace {

constexpr int kBenchSeed = 123;

// --- Float Add ---

static void BM_FloatAdd(benchmark::State& state) {
  const int size = state.range(0);
  std::mt19937 rng(kBenchSeed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> in1(size), in2(size), out(size);
  for (auto& x : in1) x = dist(rng);
  for (auto& x : in2) x = dist(rng);

  ArithmeticParams params = {};
  params.float_activation_min = -100.0f;
  params.float_activation_max = 100.0f;

  for (auto _ : state) {
    optimized_ops::AddElementwise(size, params, in1.data(), in2.data(),
                                  out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size *
                          sizeof(float) * 3);
}
BENCHMARK(BM_FloatAdd)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

// --- Float Mul ---

static void BM_FloatMul(benchmark::State& state) {
  const int size = state.range(0);
  std::mt19937 rng(kBenchSeed + 1);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> in1(size), in2(size), out(size);
  for (auto& x : in1) x = dist(rng);
  for (auto& x : in2) x = dist(rng);

  ArithmeticParams params = {};
  params.float_activation_min = -1000.0f;
  params.float_activation_max = 1000.0f;

  for (auto _ : state) {
    optimized_ops::MulElementwise(size, params, in1.data(), in2.data(),
                                  out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size);
}
BENCHMARK(BM_FloatMul)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

// --- Float HardSwish ---

static void BM_FloatHardSwish(benchmark::State& state) {
  const int size = state.range(0);
  std::mt19937 rng(kBenchSeed + 2);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> in(size), out(size);
  for (auto& x : in) x = dist(rng);

  RuntimeShape shape({1, 1, 1, size});

  for (auto _ : state) {
    optimized_ops::HardSwish(shape, in.data(), shape, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size);
}
BENCHMARK(BM_FloatHardSwish)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// --- Int8 Add ---

static void BM_Int8Add(benchmark::State& state) {
  const int size = state.range(0);
  std::mt19937 rng(kBenchSeed + 3);
  std::uniform_int_distribution<int> dist(-128, 127);
  std::vector<int8_t> in1(size), in2(size), out(size);
  for (auto& x : in1) x = static_cast<int8_t>(dist(rng));
  for (auto& x : in2) x = static_cast<int8_t>(dist(rng));

  ArithmeticParams params = {};
  params.input1_offset = 3;
  params.input2_offset = -5;
  params.output_offset = 2;
  params.left_shift = 5;
  params.input1_multiplier = 1073741824;
  params.input1_shift = -1;
  params.input2_multiplier = 1073741824;
  params.input2_shift = -2;
  params.output_multiplier = 1073741824;
  params.output_shift = -3;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  for (auto _ : state) {
    optimized_integer_ops::AddElementwiseInt8(size, params, in1.data(),
                                              in2.data(), out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size);
}
BENCHMARK(BM_Int8Add)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

// --- Int8 Mul ---

static void BM_Int8Mul(benchmark::State& state) {
  const int size = state.range(0);
  std::mt19937 rng(kBenchSeed + 4);
  std::uniform_int_distribution<int> dist(-128, 127);
  std::vector<int8_t> in1(size), in2(size), out(size);
  for (auto& x : in1) x = static_cast<int8_t>(dist(rng));
  for (auto& x : in2) x = static_cast<int8_t>(dist(rng));

  ArithmeticParams params = {};
  params.input1_offset = 2;
  params.input2_offset = -3;
  params.output_offset = 1;
  params.output_multiplier = 1073741824;
  params.output_shift = -4;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  for (auto _ : state) {
    optimized_integer_ops::MulElementwise(size, params, in1.data(), in2.data(),
                                          out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size);
}
BENCHMARK(BM_Int8Mul)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

// --- Int32 Mul ---

static void BM_Int32Mul(benchmark::State& state) {
  const int size = state.range(0);
  std::mt19937 rng(kBenchSeed + 5);
  std::uniform_int_distribution<int32_t> dist(-100, 100);
  std::vector<int32_t> in1(size), in2(size), out(size);
  for (auto& x : in1) x = dist(rng);
  for (auto& x : in2) x = dist(rng);

  ArithmeticParams params = {};
  params.quantized_activation_min = -100000;
  params.quantized_activation_max = 100000;

  for (auto _ : state) {
    optimized_ops::MulElementwise(static_cast<int32_t>(size), params,
                                  in1.data(), in2.data(), out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size);
}
BENCHMARK(BM_Int32Mul)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// --- Int8 MaxPool ---

static void BM_Int8MaxPool(benchmark::State& state) {
  const int depth = state.range(0);
  const int input_h = 8, input_w = 8;
  const int filter_h = 2, filter_w = 2, stride = 2;
  const int output_h = 4, output_w = 4;

  std::mt19937 rng(kBenchSeed + 6);
  std::uniform_int_distribution<int> dist(-128, 127);
  std::vector<int8_t> input(input_h * input_w * depth);
  for (auto& x : input) x = static_cast<int8_t>(dist(rng));
  std::vector<int8_t> output(output_h * output_w * depth);

  PoolParams params;
  params.stride_height = stride;
  params.stride_width = stride;
  params.filter_height = filter_h;
  params.filter_width = filter_w;
  params.padding_values.height = 0;
  params.padding_values.width = 0;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  RuntimeShape input_shape({1, input_h, input_w, depth});
  RuntimeShape output_shape({1, output_h, output_w, depth});

  for (auto _ : state) {
    optimized_integer_ops::MaxPool(params, input_shape, input.data(),
                                   output_shape, output.data());
    benchmark::DoNotOptimize(output.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          output_h * output_w * depth);
}
BENCHMARK(BM_Int8MaxPool)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256);

// --- Int8 AveragePool ---

static void BM_Int8AvgPool(benchmark::State& state) {
  const int depth = state.range(0);
  const int input_h = 8, input_w = 8;
  const int filter_h = 2, filter_w = 2, stride = 2;
  const int output_h = 4, output_w = 4;

  std::mt19937 rng(kBenchSeed + 7);
  std::uniform_int_distribution<int> dist(-128, 127);
  std::vector<int8_t> input(input_h * input_w * depth);
  for (auto& x : input) x = static_cast<int8_t>(dist(rng));
  std::vector<int8_t> output(output_h * output_w * depth);

  PoolParams params;
  params.stride_height = stride;
  params.stride_width = stride;
  params.filter_height = filter_h;
  params.filter_width = filter_w;
  params.padding_values.height = 0;
  params.padding_values.width = 0;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  RuntimeShape input_shape({1, input_h, input_w, depth});
  RuntimeShape output_shape({1, output_h, output_w, depth});

  for (auto _ : state) {
    optimized_integer_ops::AveragePool(params, input_shape, input.data(),
                                       output_shape, output.data());
    benchmark::DoNotOptimize(output.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          output_h * output_w * depth);
}
BENCHMARK(BM_Int8AvgPool)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256);

}  // namespace
}  // namespace tflite

BENCHMARK_MAIN();
