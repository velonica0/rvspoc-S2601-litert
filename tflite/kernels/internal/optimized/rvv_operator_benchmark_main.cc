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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

extern "C" void ScalarAddFloatElementwise(int size, float activation_min,
                                          float activation_max,
                                          const float* lhs, const float* rhs,
                                          float* output);
extern "C" void ScalarAddInt8Elementwise(int size, const int8_t* lhs,
                                         const int8_t* rhs, int8_t* output);
extern "C" void ScalarAddInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                             const int8_t* rhs,
                                             int8_t* output);
extern "C" void ScalarAddInt8ScalarKernel(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output);
extern "C" void ScalarAddInt16Elementwise(int size, const int16_t* lhs,
                                          const int16_t* rhs,
                                          int16_t* output);
extern "C" void ScalarAddFloatScalarBroadcast(int size, float activation_min,
                                              float activation_max,
                                              float lhs_scalar,
                                              const float* rhs, float* output);
extern "C" void ScalarAddFloatScalarKernel(int size, float activation_min,
                                           float activation_max,
                                           float lhs_scalar,
                                           const float* rhs, float* output);
extern "C" void ScalarMulFloatElementwise(int size, float activation_min,
                                          float activation_max,
                                          const float* lhs, const float* rhs,
                                          float* output);
extern "C" void ScalarMulInt8Elementwise(int size, const int8_t* lhs,
                                         const int8_t* rhs, int8_t* output);
extern "C" void ScalarMulInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                             const int8_t* rhs,
                                             int8_t* output);
extern "C" void ScalarMulInt8ScalarKernel(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output);
extern "C" void ScalarMulFloatScalarBroadcast(int size, float activation_min,
                                              float activation_max,
                                              float lhs_scalar,
                                              const float* rhs, float* output);
extern "C" void ScalarMulFloatScalarKernel(int size, float activation_min,
                                           float activation_max,
                                           float lhs_scalar,
                                           const float* rhs, float* output);
extern "C" void ScalarSubFloatElementwise(int size, float activation_min,
                                          float activation_max,
                                          const float* lhs, const float* rhs,
                                          float* output);
extern "C" void ScalarSubInt16Elementwise(int size, const int16_t* lhs,
                                          const int16_t* rhs,
                                          int16_t* output);
extern "C" void ScalarDivFloatElementwise(int size, float activation_min,
                                          float activation_max,
                                          const float* lhs, const float* rhs,
                                          float* output);
extern "C" void ScalarFullyConnectedFloatOperator(
    int n_batch, int input_size, int num_units, const float* input,
    const float* weights, const float* bias, float* output);
extern "C" void ScalarLstmGateFloatOperator(
    int n_batch, int n_input, int n_output, int n_cell, const float* input,
    const float* input_to_gate_weights, const float* output_state,
    const float* recurrent_to_gate_weights, const float* gate_bias,
    float* gate, float* scratch);
extern "C" void ScalarLstmOutputFloatOperator(
    int n_batch, int n_cell, int n_output, const float* cell_state,
    const float* output_gate, const float* projection_weights,
    const float* projection_bias, float* output_state, float* scratch,
    float* projection_bias_scratch);
extern "C" void ScalarAveragePoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output);
extern "C" void ScalarMaxPoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output);
extern "C" void ScalarAveragePoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output);
extern "C" void ScalarMaxPoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output);
extern "C" void ScalarDepthwiseConvInt8Operator(
    int batches, int input_height, int input_width, int input_depth,
    int filter_height, int filter_width, int depth_multiplier,
    int stride_height, int stride_width, const int8_t* input,
    const int8_t* filter, const int32_t* bias,
    const int32_t* output_multiplier, const int32_t* output_shift,
    int8_t* output);
extern "C" void ScalarAffineQuantizeInt8Operator(int size, int32_t zero_point,
                                                 float scale,
                                                 const float* input,
                                                 int8_t* output);
extern "C" void ScalarAffineQuantizeUint8Operator(int size, int32_t zero_point,
                                                  float scale,
                                                  const float* input,
                                                  uint8_t* output);
extern "C" void ScalarAffineQuantizeInt16Operator(int size, int32_t zero_point,
                                                  float scale,
                                                  const float* input,
                                                  int16_t* output);
extern "C" void ScalarHardSwishUint8Operator(int size, const uint8_t* input,
                                             uint8_t* output);
extern "C" void ScalarHardSwishInt8Operator(int size, const int8_t* input,
                                            int8_t* output);
extern "C" void ScalarLeakyReluInt16Operator(int size, const int16_t* input,
                                             int16_t* output);
extern "C" void ScalarLookupTableUint8Operator(int size, const uint8_t* input,
                                               const uint8_t* lut,
                                               uint8_t* output);
extern "C" void ScalarLookupTableInt8Operator(int size, const int8_t* input,
                                              const int8_t* lut,
                                              int8_t* output);
extern "C" void ScalarMeanInt8Operator(int batches, int input_height,
                                       int input_width, int depth,
                                       const int8_t* input, int8_t* output);
extern "C" void ScalarMeanUint8Operator(int batches, int input_height,
                                        int input_width, int depth,
                                        const uint8_t* input,
                                        uint8_t* output);
extern "C" void ScalarMeanFloatLastDimOperator(int rows, int cols,
                                               const float* input,
                                               float* output);
extern "C" void ScalarResizeBilinearFloatOperator(
    int batches, int input_height, int input_width, int depth,
    int output_height, int output_width, bool align_corners,
    bool half_pixel_centers, const float* input, float* output);
extern "C" void ScalarResizeBilinearUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int output_height, int output_width, bool align_corners,
    bool half_pixel_centers, const uint8_t* input, uint8_t* output);
extern "C" int ScalarArgMinFloatOperator(const float* input, int size);
extern "C" int ScalarArgMaxFloatOperator(const float* input, int size);
extern "C" int ScalarArgMaxInt8Operator(const int8_t* input, int size);
extern "C" int ScalarArgMaxUint8Operator(const uint8_t* input, int size);

extern "C" void RvvAddFloatElementwise(int size, float activation_min,
                                       float activation_max, const float* lhs,
                                       const float* rhs, float* output);
extern "C" void RvvAddInt8Elementwise(int size, const int8_t* lhs,
                                      const int8_t* rhs, int8_t* output);
extern "C" void RvvAddInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output);
extern "C" void RvvAddInt8ScalarKernel(int size, int8_t lhs_scalar,
                                       const int8_t* rhs, int8_t* output);
extern "C" void RvvAddInt16Elementwise(int size, const int16_t* lhs,
                                       const int16_t* rhs, int16_t* output);
extern "C" void RvvAddFloatScalarBroadcast(int size, float activation_min,
                                           float activation_max,
                                           float lhs_scalar,
                                           const float* rhs, float* output);
extern "C" void RvvAddFloatScalarKernel(int size, float activation_min,
                                        float activation_max,
                                        float lhs_scalar, const float* rhs,
                                        float* output);
extern "C" void RvvMulFloatElementwise(int size, float activation_min,
                                       float activation_max, const float* lhs,
                                       const float* rhs, float* output);
extern "C" void RvvMulInt8Elementwise(int size, const int8_t* lhs,
                                      const int8_t* rhs, int8_t* output);
extern "C" void RvvMulInt8ScalarBroadcast(int size, int8_t lhs_scalar,
                                          const int8_t* rhs, int8_t* output);
extern "C" void RvvMulInt8ScalarKernel(int size, int8_t lhs_scalar,
                                       const int8_t* rhs, int8_t* output);
extern "C" void RvvMulFloatScalarBroadcast(int size, float activation_min,
                                           float activation_max,
                                           float lhs_scalar,
                                           const float* rhs, float* output);
extern "C" void RvvMulFloatScalarKernel(int size, float activation_min,
                                        float activation_max,
                                        float lhs_scalar, const float* rhs,
                                        float* output);
extern "C" void RvvSubFloatElementwise(int size, float activation_min,
                                       float activation_max, const float* lhs,
                                       const float* rhs, float* output);
extern "C" void RvvSubInt16Elementwise(int size, const int16_t* lhs,
                                       const int16_t* rhs, int16_t* output);
extern "C" void RvvDivFloatElementwise(int size, float activation_min,
                                       float activation_max, const float* lhs,
                                       const float* rhs, float* output);
extern "C" void RvvFullyConnectedFloatOperator(
    int n_batch, int input_size, int num_units, const float* input,
    const float* weights, const float* bias, float* output);
extern "C" void RvvLstmGateFloatOperator(
    int n_batch, int n_input, int n_output, int n_cell, const float* input,
    const float* input_to_gate_weights, const float* output_state,
    const float* recurrent_to_gate_weights, const float* gate_bias,
    float* gate, float* scratch);
extern "C" void RvvLstmOutputFloatOperator(
    int n_batch, int n_cell, int n_output, const float* cell_state,
    const float* output_gate, const float* projection_weights,
    const float* projection_bias, float* output_state, float* scratch,
    float* projection_bias_scratch);
extern "C" void RvvAveragePoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output);
extern "C" void RvvMaxPoolUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const uint8_t* input, uint8_t* output);
extern "C" void RvvAveragePoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output);
extern "C" void RvvMaxPoolInt8Operator(
    int batches, int input_height, int input_width, int depth,
    int filter_height, int filter_width, int stride_height, int stride_width,
    const int8_t* input, int8_t* output);
extern "C" void RvvDepthwiseConvInt8Operator(
    int batches, int input_height, int input_width, int input_depth,
    int filter_height, int filter_width, int depth_multiplier,
    int stride_height, int stride_width, const int8_t* input,
    const int8_t* filter, const int32_t* bias,
    const int32_t* output_multiplier, const int32_t* output_shift,
    int8_t* output);
extern "C" void RvvAffineQuantizeInt8Operator(int size, int32_t zero_point,
                                              float scale, const float* input,
                                              int8_t* output);
extern "C" void RvvAffineQuantizeUint8Operator(int size, int32_t zero_point,
                                               float scale,
                                               const float* input,
                                               uint8_t* output);
extern "C" void RvvAffineQuantizeInt16Operator(int size, int32_t zero_point,
                                               float scale,
                                               const float* input,
                                               int16_t* output);
extern "C" void RvvHardSwishUint8Operator(int size, const uint8_t* input,
                                          uint8_t* output);
extern "C" void RvvHardSwishInt8Operator(int size, const int8_t* input,
                                         int8_t* output);
extern "C" void RvvLeakyReluInt16Operator(int size, const int16_t* input,
                                          int16_t* output);
extern "C" void RvvLookupTableUint8Operator(int size, const uint8_t* input,
                                            const uint8_t* lut,
                                            uint8_t* output);
extern "C" void RvvLookupTableInt8Operator(int size, const int8_t* input,
                                           const int8_t* lut,
                                           int8_t* output);
extern "C" void RvvMeanInt8Operator(int batches, int input_height,
                                    int input_width, int depth,
                                    const int8_t* input, int8_t* output);
extern "C" void RvvMeanUint8Operator(int batches, int input_height,
                                     int input_width, int depth,
                                     const uint8_t* input, uint8_t* output);
extern "C" void RvvMeanFloatLastDimOperator(int rows, int cols,
                                            const float* input,
                                            float* output);
extern "C" void RvvResizeBilinearFloatOperator(
    int batches, int input_height, int input_width, int depth,
    int output_height, int output_width, bool align_corners,
    bool half_pixel_centers, const float* input, float* output);
extern "C" void RvvResizeBilinearUint8Operator(
    int batches, int input_height, int input_width, int depth,
    int output_height, int output_width, bool align_corners,
    bool half_pixel_centers, const uint8_t* input, uint8_t* output);
extern "C" int RvvArgMinFloatOperator(const float* input, int size);
extern "C" int RvvArgMaxFloatOperator(const float* input, int size);
extern "C" int RvvArgMaxInt8Operator(const int8_t* input, int size);
extern "C" int RvvArgMaxUint8Operator(const uint8_t* input, int size);

namespace {

struct BenchmarkStats {
  double total_ms;
  double mean_us;
  double gops;
  double sink;
};

struct FloatAccuracy {
  float max_abs_diff;
  float max_rel_diff;
  double mean_abs_diff;
};

struct IntegerAccuracy {
  int max_abs_diff;
  int mismatches;
};

struct VectorCase {
  const char* name;
  int size;
};

struct FullyConnectedCase {
  const char* name;
  int n_batch;
  int input_size;
  int num_units;
};

struct LstmGateCase {
  const char* name;
  int n_batch;
  int n_input;
  int n_output;
  int n_cell;
};

struct LstmOutputCase {
  const char* name;
  int n_batch;
  int n_cell;
  int n_output;
};

struct PoolCase {
  const char* name;
  int n_batch;
  int input_height;
  int input_width;
  int depth;
  int filter_height;
  int filter_width;
  int stride_height;
  int stride_width;
};

struct DepthwiseCase {
  const char* name;
  int n_batch;
  int input_height;
  int input_width;
  int input_depth;
  int filter_height;
  int filter_width;
  int stride_height;
  int stride_width;
  int depth_multiplier;
};

struct MeanCase {
  const char* name;
  int n_batch;
  int input_height;
  int input_width;
  int depth;
};

std::vector<float> MakeRandomFloatVector(int size, std::mt19937* rng,
                                         float min_value = -1.0f,
                                         float max_value = 1.0f) {
  std::uniform_real_distribution<float> dist(min_value, max_value);
  std::vector<float> values(size);
  for (float& value : values) {
    value = dist(*rng);
  }
  return values;
}

template <typename T>
std::vector<T> MakeRandomIntegralVector(int size, std::mt19937* rng,
                                        int min_value, int max_value) {
  std::uniform_int_distribution<int> dist(min_value, max_value);
  std::vector<T> values(size);
  for (T& value : values) {
    value = static_cast<T>(dist(*rng));
  }
  return values;
}

int ChooseIterations(int64_t work_per_iter) {
  constexpr int64_t kTargetWork = 128000000;
  const int64_t iterations =
      kTargetWork / std::max<int64_t>(1, work_per_iter);
  return std::max<int64_t>(20, std::min<int64_t>(4000, iterations));
}

template <typename Runner>
BenchmarkStats RunBenchmark(int iterations, double ops_per_iter, Runner runner) {
  volatile double sink = 0.0;

  for (int warmup = 0; warmup < 5; ++warmup) {
    sink += runner();
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    sink += runner();
  }
  const auto end = std::chrono::steady_clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  const double mean_us = total_ms * 1000.0 / iterations;
  const double gops = ops_per_iter / (mean_us * 1000.0);
  return {total_ms, mean_us, gops, sink};
}

FloatAccuracy CompareFloatVectors(const std::vector<float>& lhs,
                                  const std::vector<float>& rhs) {
  FloatAccuracy accuracy = {0.0f, 0.0f, 0.0};
  for (size_t i = 0; i < lhs.size(); ++i) {
    const float abs_diff = std::abs(lhs[i] - rhs[i]);
    const float denom = std::max(std::abs(lhs[i]), 1.0e-6f);
    accuracy.max_abs_diff = std::max(accuracy.max_abs_diff, abs_diff);
    accuracy.max_rel_diff =
        std::max(accuracy.max_rel_diff, abs_diff / denom);
    accuracy.mean_abs_diff += abs_diff;
  }
  accuracy.mean_abs_diff /= std::max<size_t>(1, lhs.size());
  return accuracy;
}

template <typename T>
IntegerAccuracy CompareIntegerVectors(const std::vector<T>& lhs,
                                      const std::vector<T>& rhs) {
  IntegerAccuracy accuracy = {0, 0};
  for (size_t i = 0; i < lhs.size(); ++i) {
    const int lhs_value = static_cast<int>(lhs[i]);
    const int rhs_value = static_cast<int>(rhs[i]);
    const int abs_diff = std::abs(lhs_value - rhs_value);
    accuracy.max_abs_diff = std::max(accuracy.max_abs_diff, abs_diff);
    if (lhs_value != rhs_value) {
      ++accuracy.mismatches;
    }
  }
  return accuracy;
}

IntegerAccuracy CompareIndexValues(int lhs, int rhs) {
  return {std::abs(lhs - rhs), lhs == rhs ? 0 : 1};
}

bool FloatAccuracyWithinTolerance(const FloatAccuracy& accuracy) {
  return accuracy.max_abs_diff <= 1.0e-6f ||
         (accuracy.max_abs_diff <= 4.0e-6f &&
          accuracy.mean_abs_diff <= 1.0e-7);
}

bool FloatAccuracyWithinLstmTolerance(const FloatAccuracy& accuracy) {
  return accuracy.max_abs_diff <= 2.0e-4f ||
         (accuracy.max_abs_diff <= 1.0e-3f &&
          accuracy.mean_abs_diff <= 5.0e-5);
}

bool FloatAccuracyWithinMatVecTolerance(const FloatAccuracy& accuracy) {
  return accuracy.max_abs_diff <= 1.0e-4f ||
         (accuracy.max_abs_diff <= 2.0e-4f &&
          accuracy.mean_abs_diff <= 2.0e-5);
}

bool FloatAccuracyWithinCodeXOperatorTolerance(const FloatAccuracy& accuracy) {
  return accuracy.max_abs_diff <= 1.0e-5f ||
         (accuracy.max_rel_diff <= 1.0e-5f &&
          accuracy.mean_abs_diff <= 1.0e-6);
}

bool IntegerAccuracyWithinTolerance(const IntegerAccuracy& accuracy,
                                    int max_abs_diff) {
  return accuracy.max_abs_diff <= max_abs_diff;
}

std::string MakeFullyConnectedShape(const FullyConnectedCase& bench) {
  return "B" + std::to_string(bench.n_batch) + " I" +
         std::to_string(bench.input_size) + " O" +
         std::to_string(bench.num_units);
}

std::string MakeLstmGateShape(const LstmGateCase& bench) {
  return "B" + std::to_string(bench.n_batch) + " I" +
         std::to_string(bench.n_input) + " O" +
         std::to_string(bench.n_output) + " C" +
         std::to_string(bench.n_cell);
}

std::string MakeLstmOutputShape(const LstmOutputCase& bench) {
  return "B" + std::to_string(bench.n_batch) + " C" +
         std::to_string(bench.n_cell) + " O" +
         std::to_string(bench.n_output);
}

int ComputeValidOutputSize(int input_size, int filter_size, int stride) {
  return (input_size - filter_size) / stride + 1;
}

std::string MakePoolShape(const PoolCase& bench) {
  return "B" + std::to_string(bench.n_batch) + " I" +
         std::to_string(bench.input_height) + "x" +
         std::to_string(bench.input_width) + "x" +
         std::to_string(bench.depth) + " F" +
         std::to_string(bench.filter_height) + "x" +
         std::to_string(bench.filter_width) + " S" +
         std::to_string(bench.stride_height) + "x" +
         std::to_string(bench.stride_width);
}

std::string MakeDepthwiseShape(const DepthwiseCase& bench) {
  return "B" + std::to_string(bench.n_batch) + " I" +
         std::to_string(bench.input_height) + "x" +
         std::to_string(bench.input_width) + "x" +
         std::to_string(bench.input_depth) + " F" +
         std::to_string(bench.filter_height) + "x" +
         std::to_string(bench.filter_width) + " S" +
         std::to_string(bench.stride_height) + "x" +
         std::to_string(bench.stride_width) + " DM" +
         std::to_string(bench.depth_multiplier);
}

std::string MakeMeanShape(const MeanCase& bench) {
  return "B" + std::to_string(bench.n_batch) + " I" +
         std::to_string(bench.input_height) + "x" +
         std::to_string(bench.input_width) + "x" +
         std::to_string(bench.depth) + " A(1,2)";
}

void PrintSourceFileSection(const char* path, const char* summary) {
  std::cout << "## `" << path << "`\n\n";
  std::cout << "- Summary: " << summary << "\n\n";
}

void PrintDispatchOnlySection(const char* path, const char* note) {
  std::cout << "## `" << path << "`\n\n";
  std::cout << "- Note: " << note << "\n\n";
}

void PrintBenchmarkTitle(const char* title, const char* affects) {
  std::cout << "### " << title << "\n\n";
  std::cout << "- Affects: " << affects << "\n\n";
}

void PrintFloatHeader(const std::string& shape_column_name) {
  std::cout << "| Case | " << shape_column_name
            << " | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | "
               "RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |\n";
  std::cout << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
               "---: | ---: |\n";
}

void PrintIntegerHeader(const std::string& shape_column_name) {
  std::cout << "| Case | " << shape_column_name
            << " | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | "
               "RVV GOPS | Max abs diff | Mismatches |\n";
  std::cout << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
               "---: |\n";
}

template <typename ScalarRunner, typename RvvRunner>
void BenchmarkFloatBinaryOp(const char* title, const char* affects,
                            const VectorCase* cases, int num_cases,
                            ScalarRunner scalar_runner,
                            RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintFloatHeader("Vector size");

  for (int idx = 0; idx < num_cases; ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0x1234 + bench.size * 17 + idx);
    const std::vector<float> lhs = MakeRandomFloatVector(bench.size, &rng);
    std::vector<float> rhs = MakeRandomFloatVector(bench.size, &rng);
    std::vector<float> scalar_output(bench.size, 0.0f);
    std::vector<float> rvv_output(bench.size, 0.0f);

    scalar_runner(lhs, rhs, &scalar_output);
    rvv_runner(lhs, rhs, &rvv_output);
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       scalar_runner(lhs, rhs, &scalar_output);
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       rvv_runner(lhs, rhs, &rvv_output);
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";

    if (!FloatAccuracyWithinTolerance(accuracy)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename T, typename ScalarRunner, typename RvvRunner>
void BenchmarkBinaryIntegerVectorOp(const char* title, const char* affects,
                                    const VectorCase* cases, int num_cases,
                                    int input_min_value, int input_max_value,
                                    int allowed_max_abs_diff,
                                    ScalarRunner scalar_runner,
                                    RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Vector size");

  for (int idx = 0; idx < num_cases; ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0x99AA + bench.size * 19 + idx);
    const std::vector<T> lhs =
        MakeRandomIntegralVector<T>(bench.size, &rng, input_min_value,
                                    input_max_value);
    const std::vector<T> rhs =
        MakeRandomIntegralVector<T>(bench.size, &rng, input_min_value,
                                    input_max_value);
    std::vector<T> scalar_output(bench.size, 0);
    std::vector<T> rvv_output(bench.size, 0);

    scalar_runner(lhs.data(), rhs.data(), bench.size, scalar_output.data());
    rvv_runner(lhs.data(), rhs.data(), bench.size, rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       scalar_runner(lhs.data(), rhs.data(), bench.size,
                                     scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       rvv_runner(lhs.data(), rhs.data(), bench.size,
                                  rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, allowed_max_abs_diff)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename T, typename ScalarRunner, typename RvvRunner>
void BenchmarkScalarBroadcastIntegerVectorOp(
    const char* title, const char* affects, const VectorCase* cases,
    int num_cases, int input_min_value, int input_max_value,
    int allowed_max_abs_diff, ScalarRunner scalar_runner,
    RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Vector size");

  for (int idx = 0; idx < num_cases; ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0xA5A5 + bench.size * 23 + idx);
    const std::vector<T> lhs =
        MakeRandomIntegralVector<T>(bench.size, &rng, input_min_value,
                                    input_max_value);
    const std::vector<T> rhs =
        MakeRandomIntegralVector<T>(bench.size, &rng, input_min_value,
                                    input_max_value);
    const T lhs_scalar = lhs[0];
    std::vector<T> scalar_output(bench.size, 0);
    std::vector<T> rvv_output(bench.size, 0);

    scalar_runner(lhs_scalar, rhs.data(), bench.size, scalar_output.data());
    rvv_runner(lhs_scalar, rhs.data(), bench.size, rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       scalar_runner(lhs_scalar, rhs.data(), bench.size,
                                     scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       rvv_runner(lhs_scalar, rhs.data(), bench.size,
                                  rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, allowed_max_abs_diff)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename T>
std::vector<T> MakeLookupTable() {
  std::vector<T> lut(256);
  for (int i = 0; i < 256; ++i) {
    lut[i] = static_cast<T>((i * 37 + 13) & 0xff);
  }
  return lut;
}

template <typename T, typename ScalarRunner, typename RvvRunner>
void BenchmarkLookupTableOp(const char* title, const char* affects,
                            const VectorCase* cases, int num_cases,
                            int input_min_value, int input_max_value,
                            int allowed_max_abs_diff,
                            ScalarRunner scalar_runner,
                            RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Vector size");

  for (int idx = 0; idx < num_cases; ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0xB6B6 + bench.size * 29 + idx);
    const std::vector<T> input =
        MakeRandomIntegralVector<T>(bench.size, &rng, input_min_value,
                                    input_max_value);
    const std::vector<T> lut = MakeLookupTable<T>();
    std::vector<T> scalar_output(bench.size, 0);
    std::vector<T> rvv_output(bench.size, 0);

    scalar_runner(input.data(), bench.size, lut.data(), scalar_output.data());
    rvv_runner(input.data(), bench.size, lut.data(), rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       scalar_runner(input.data(), bench.size, lut.data(),
                                     scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       rvv_runner(input.data(), bench.size, lut.data(),
                                  rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, allowed_max_abs_diff)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename OutputT, typename ScalarRunner, typename RvvRunner>
void BenchmarkAffineQuantizeOp(const char* title, const char* affects,
                               const VectorCase* cases, int num_cases,
                               int32_t zero_point, float scale,
                               float input_min, float input_max,
                               int allowed_max_abs_diff,
                               ScalarRunner scalar_runner,
                               RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Vector size");

  for (int idx = 0; idx < num_cases; ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0x5678 + bench.size * 31 + idx);
    const std::vector<float> input =
        MakeRandomFloatVector(bench.size, &rng, input_min, input_max);
    std::vector<OutputT> scalar_output(bench.size);
    std::vector<OutputT> rvv_output(bench.size);

    scalar_runner(input, &scalar_output, zero_point, scale);
    rvv_runner(input, &rvv_output, zero_point, scale);
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       scalar_runner(input, &scalar_output, zero_point, scale);
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       rvv_runner(input, &rvv_output, zero_point, scale);
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, allowed_max_abs_diff)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename T, typename ScalarRunner, typename RvvRunner>
void BenchmarkPoolOp(const char* title, const char* affects,
                     const PoolCase* cases, int num_cases,
                     int input_min_value, int input_max_value,
                     ScalarRunner scalar_runner, RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Shape");

  for (int idx = 0; idx < num_cases; ++idx) {
    const PoolCase& bench = cases[idx];
    const int output_height = ComputeValidOutputSize(
        bench.input_height, bench.filter_height, bench.stride_height);
    const int output_width = ComputeValidOutputSize(
        bench.input_width, bench.filter_width, bench.stride_width);
    const int input_size =
        bench.n_batch * bench.input_height * bench.input_width * bench.depth;
    const int output_size =
        bench.n_batch * output_height * output_width * bench.depth;
    const int64_t ops_per_iter =
        static_cast<int64_t>(output_size) * bench.filter_height *
        bench.filter_width;
    const int iterations = ChooseIterations(ops_per_iter);
    std::mt19937 rng(0x6789 + idx * 43 + bench.depth);
    const std::vector<T> input =
        MakeRandomIntegralVector<T>(input_size, &rng, input_min_value,
                                    input_max_value);
    std::vector<T> scalar_output(output_size, 0);
    std::vector<T> rvv_output(output_size, 0);

    scalar_runner(bench, input.data(), scalar_output.data());
    rvv_runner(bench, input.data(), rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       scalar_runner(bench, input.data(), scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       rvv_runner(bench, input.data(), rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << MakePoolShape(bench) << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, 0)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename T, typename ScalarRunner, typename RvvRunner>
void BenchmarkUnaryIntegerVectorOp(const char* title, const char* affects,
                                   const VectorCase* cases, int num_cases,
                                   int input_min_value, int input_max_value,
                                   int allowed_max_abs_diff,
                                   ScalarRunner scalar_runner,
                                   RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Vector size");

  for (int idx = 0; idx < num_cases; ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0x88AA + bench.size * 17 + idx);
    const std::vector<T> input =
        MakeRandomIntegralVector<T>(bench.size, &rng, input_min_value,
                                    input_max_value);
    std::vector<T> scalar_output(bench.size, 0);
    std::vector<T> rvv_output(bench.size, 0);

    scalar_runner(input.data(), bench.size, scalar_output.data());
    rvv_runner(input.data(), bench.size, rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       scalar_runner(input.data(), bench.size,
                                     scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       rvv_runner(input.data(), bench.size, rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, allowed_max_abs_diff)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename ScalarRunner, typename RvvRunner>
void BenchmarkDepthwiseOp(const char* title, const char* affects,
                          const DepthwiseCase* cases, int num_cases,
                          ScalarRunner scalar_runner, RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Shape");

  for (int idx = 0; idx < num_cases; ++idx) {
    const DepthwiseCase& bench = cases[idx];
    const int output_height = ComputeValidOutputSize(
        bench.input_height, bench.filter_height, bench.stride_height);
    const int output_width = ComputeValidOutputSize(
        bench.input_width, bench.filter_width, bench.stride_width);
    const int output_depth = bench.input_depth * bench.depth_multiplier;
    const int input_size = bench.n_batch * bench.input_height *
                           bench.input_width * bench.input_depth;
    const int filter_size = bench.filter_height * bench.filter_width *
                            output_depth;
    const int output_size =
        bench.n_batch * output_height * output_width * output_depth;
    const int64_t ops_per_iter =
        static_cast<int64_t>(2) * output_size * bench.filter_height *
        bench.filter_width;
    const int iterations = ChooseIterations(ops_per_iter);

    std::mt19937 rng(0xD00D + idx * 97 + output_depth * 13);
    const std::vector<int8_t> input =
        MakeRandomIntegralVector<int8_t>(input_size, &rng, -8, 8);
    const std::vector<int8_t> filter =
        MakeRandomIntegralVector<int8_t>(filter_size, &rng, -4, 4);
    const std::vector<int32_t> bias =
        MakeRandomIntegralVector<int32_t>(output_depth, &rng, -64, 64);
    const std::vector<int32_t> output_multiplier(output_depth, 1 << 30);
    const std::vector<int32_t> output_shift(output_depth, -4);
    std::vector<int8_t> scalar_output(output_size, 0);
    std::vector<int8_t> rvv_output(output_size, 0);

    scalar_runner(bench, input.data(), filter.data(), bias.data(),
                  output_multiplier.data(), output_shift.data(),
                  scalar_output.data());
    rvv_runner(bench, input.data(), filter.data(), bias.data(),
               output_multiplier.data(), output_shift.data(),
               rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       scalar_runner(bench, input.data(), filter.data(),
                                     bias.data(), output_multiplier.data(),
                                     output_shift.data(), scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       rvv_runner(bench, input.data(), filter.data(),
                                  bias.data(), output_multiplier.data(),
                                  output_shift.data(), rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << MakeDepthwiseShape(bench)
              << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, 0)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename ScalarRunner, typename RvvRunner>
void BenchmarkMeanOp(const char* title, const char* affects,
                     const MeanCase* cases, int num_cases,
                     ScalarRunner scalar_runner, RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Shape");

  for (int idx = 0; idx < num_cases; ++idx) {
    const MeanCase& bench = cases[idx];
    const int input_size =
        bench.n_batch * bench.input_height * bench.input_width * bench.depth;
    const int output_size = bench.n_batch * bench.depth;
    const int64_t ops_per_iter = static_cast<int64_t>(output_size) *
                                 bench.input_height * bench.input_width;
    const int iterations = ChooseIterations(ops_per_iter);
    std::mt19937 rng(0xC7C7 + idx * 37 + bench.depth);
    const std::vector<int8_t> input = MakeRandomIntegralVector<int8_t>(
        input_size, &rng, std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max());
    std::vector<int8_t> scalar_output(output_size, 0);
    std::vector<int8_t> rvv_output(output_size, 0);

    scalar_runner(bench, input.data(), scalar_output.data());
    rvv_runner(bench, input.data(), rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       scalar_runner(bench, input.data(), scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       rvv_runner(bench, input.data(), rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << MakeMeanShape(bench) << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, 0)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

template <typename InputT, typename ScalarRunner, typename RvvRunner,
          typename Filler>
void BenchmarkIndexOp(const char* title, const char* affects,
                      const VectorCase* cases, int num_cases, Filler filler,
                      ScalarRunner scalar_runner, RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Vector size");

  for (int idx = 0; idx < num_cases; ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0x789A + bench.size * 13 + idx);
    std::vector<InputT> input = filler(bench.size, &rng);
    const int scalar_index = scalar_runner(input.data(), bench.size);
    const int rvv_index = rvv_runner(input.data(), bench.size);
    const IntegerAccuracy accuracy =
        CompareIndexValues(scalar_index, rvv_index);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       return scalar_runner(input.data(), bench.size);
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       return rvv_runner(input.data(), bench.size);
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, 0)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

}  // namespace

#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_optimized_ops.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_reduce.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_resize_bilinear.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_fully_connected.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_lstm_eval.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_add.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_mul.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_sub.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_pooling.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_depthwise.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_leaky_relu.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_lut.inc"
#include "tflite/kernels/internal/optimized/rvv_operator_benchmark_integer_mean.inc"

int main() {
  std::cout << "# RVV vs Scalar: Operator-level RVV coverage\n\n";
  std::cout << "- Scalar kernels: same wrapper compiled with "
               "`-fno-tree-vectorize -fno-tree-slp-vectorize`\n";
  std::cout << "- RVV kernels: same wrapper compiled with "
               "`-march=rv64gcv_zvl128b -mabi=lp64d`\n";
  std::cout << "- Organization: benchmark sections are split by the source "
               "file that owns the RVV path.\n";
  std::cout << "- Scope: includes the earlier recent-five-commit operator "
               "set plus follow-up P1 gap closures for `reduce.h` and "
               "`resize_bilinear.h`.\n\n";

  RunOptimizedOpsBenchmarks();
  RunReduceBenchmarks();
  RunResizeBilinearBenchmarks();
  RunFullyConnectedBenchmarks();
  RunLstmEvalBenchmarks();
  RunIntegerAddBenchmarks();
  RunIntegerMulBenchmarks();
  RunIntegerSubBenchmarks();
  RunIntegerPoolingBenchmarks();
  RunIntegerDepthwiseBenchmarks();
  RunIntegerLeakyReluBenchmarks();
  RunIntegerLutBenchmarks();
  RunIntegerMeanBenchmarks();
  return 0;
}
