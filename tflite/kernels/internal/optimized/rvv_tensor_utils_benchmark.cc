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
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "tflite/kernels/internal/optimized/rvv_tensor_utils_impl.h"
#include "tflite/kernels/internal/reference/portable_tensor_utils_impl.h"

#ifndef USE_RVV
#error "This benchmark must be compiled with RVV enabled."
#endif

namespace tflite {
namespace tensor_utils {
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

struct IntAccuracy {
  int32_t max_abs_diff;
  int mismatches;
};

struct VectorCase {
  const char* name;
  int size;
};

struct ReductionCase {
  const char* name;
  int output_size;
  int reduction_size;
};

struct MatrixCase {
  const char* name;
  int rows;
  int cols;
};

struct MatVecCase {
  const char* name;
  int rows;
  int cols;
  int batch;
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

std::vector<int8_t> MakeRandomInt8Vector(int size, std::mt19937* rng,
                                         int min_value = -127,
                                         int max_value = 127) {
  std::uniform_int_distribution<int> dist(min_value, max_value);
  std::vector<int8_t> values(size);
  for (int8_t& value : values) {
    value = static_cast<int8_t>(dist(*rng));
  }
  return values;
}

std::vector<int32_t> MakeRandomInt32Vector(int size, std::mt19937* rng,
                                           int min_value = -1024,
                                           int max_value = 1024) {
  std::uniform_int_distribution<int32_t> dist(min_value, max_value);
  std::vector<int32_t> values(size);
  for (int32_t& value : values) {
    value = dist(*rng);
  }
  return values;
}

int ChooseIterations(int64_t work_per_iter) {
  constexpr int64_t kTargetWork = 160000000;
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

  return {
      total_ms,
      mean_us,
      gops,
      sink,
  };
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

IntAccuracy CompareIntVectors(const std::vector<int32_t>& lhs,
                              const std::vector<int32_t>& rhs) {
  IntAccuracy accuracy = {0, 0};
  for (size_t i = 0; i < lhs.size(); ++i) {
    const int32_t abs_diff = std::abs(lhs[i] - rhs[i]);
    accuracy.max_abs_diff = std::max(accuracy.max_abs_diff, abs_diff);
    accuracy.mismatches += lhs[i] != rhs[i];
  }
  return accuracy;
}

bool FloatAccuracyWithinTolerance(const FloatAccuracy& accuracy) {
  return accuracy.max_abs_diff <= 2.0e-4f ||
         (accuracy.max_abs_diff <= 3.0e-4f &&
          accuracy.mean_abs_diff <= 2.0e-5);
}

void PrintFloatHeader(const std::string& shape_column_name) {
  std::cout << "| Case | " << shape_column_name
            << " | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | "
               "RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |\n";
  std::cout << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
               "---: | ---: |\n";
}

void PrintIntHeader(const std::string& shape_column_name) {
  std::cout << "| Case | " << shape_column_name
            << " | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | "
               "RVV GOPS | Max abs diff | Mismatches |\n";
  std::cout << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
               "---: |\n";
}

void BenchmarkFloatDotProduct(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"dot_64", 64},
      {"dot_256", 256},
      {"dot_1024", 1024},
      {"dot_4096", 4096},
  };

  std::cout << "## VectorVectorDotProduct<float>\n\n";
  std::cout << "- Affects: `fully_connected`, recurrent math, shared float "
               "tensor-utils call sites.\n\n";
  PrintFloatHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<float> lhs = MakeRandomFloatVector(bench.size, rng);
    const std::vector<float> rhs = MakeRandomFloatVector(bench.size, rng);

    const float scalar_value =
        PortableVectorVectorDotProduct(lhs.data(), rhs.data(), bench.size);
    const float rvv_value =
        RvvVectorVectorDotProduct(lhs.data(), rhs.data(), bench.size);
    const std::vector<float> scalar_output = {scalar_value};
    const std::vector<float> rvv_output = {rvv_value};
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * bench.size, [&]() -> double {
          return PortableVectorVectorDotProduct(lhs.data(), rhs.data(),
                                                bench.size);
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * bench.size, [&]() -> double {
          return RvvVectorVectorDotProduct(lhs.data(), rhs.data(), bench.size);
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
  }

  std::cout << "\n";
}

void BenchmarkFloatReduction(std::mt19937* rng) {
  const ReductionCase cases[] = {
      {"svdf_small", 64, 8},
      {"svdf_mid", 128, 16},
      {"svdf_large", 256, 64},
      {"reduce_wide", 256, 256},
  };

  std::cout << "## ReductionSumVector<float>\n\n";
  std::cout << "- Affects: `svdf` and other float reduction-style helper "
               "paths.\n\n";
  PrintFloatHeader("Output x reduction");

  for (const ReductionCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.output_size) * bench.reduction_size;
    const int iterations = ChooseIterations(work);
    const std::vector<float> input =
        MakeRandomFloatVector(bench.output_size * bench.reduction_size, rng);
    std::vector<float> scalar_output(bench.output_size, 0.0f);
    std::vector<float> rvv_output(bench.output_size, 0.0f);

    PortableReductionSumVector(input.data(), scalar_output.data(),
                               bench.output_size, bench.reduction_size);
    RvvReductionSumVector(input.data(), rvv_output.data(), bench.output_size,
                          bench.reduction_size);
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          PortableReductionSumVector(input.data(), scalar_output.data(),
                                     bench.output_size, bench.reduction_size);
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          RvvReductionSumVector(input.data(), rvv_output.data(), bench.output_size,
                                bench.reduction_size);
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.output_size << " x "
              << bench.reduction_size << " | " << iterations << " | "
              << std::fixed << std::setprecision(2) << scalar_stats.mean_us
              << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";
    if (!FloatAccuracyWithinTolerance(accuracy)) {
      std::cerr << "Float reduction accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkInt8Reduction(std::mt19937* rng) {
  const ReductionCase cases[] = {
      {"row_sum_small", 64, 32},
      {"row_sum_mid", 128, 128},
      {"row_sum_large", 256, 256},
      {"row_sum_conv_like", 512, 512},
  };

  std::cout << "## ReductionSumVector<int8 -> int32>\n\n";
  std::cout << "- Affects: `conv`, `batch_matmul`, `kernel_utils`, cached row "
               "sum preparation.\n\n";
  PrintIntHeader("Output x reduction");

  for (const ReductionCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.output_size) * bench.reduction_size;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> input =
        MakeRandomInt8Vector(bench.output_size * bench.reduction_size, rng);
    std::vector<int32_t> scalar_output(bench.output_size, 0);
    std::vector<int32_t> rvv_output(bench.output_size, 0);

    PortableReductionSumVector(input.data(), scalar_output.data(),
                               bench.output_size, bench.reduction_size);
    RvvReductionSumVector(input.data(), rvv_output.data(), bench.output_size,
                          bench.reduction_size);
    const IntAccuracy accuracy = CompareIntVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          PortableReductionSumVector(input.data(), scalar_output.data(),
                                     bench.output_size, bench.reduction_size);
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          RvvReductionSumVector(input.data(), rvv_output.data(), bench.output_size,
                                bench.reduction_size);
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.output_size << " x "
              << bench.reduction_size << " | " << iterations << " | "
              << std::fixed << std::setprecision(2) << scalar_stats.mean_us
              << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "Int8 reduction accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkMatrixScalarMultiply(std::mt19937* rng) {
  const MatrixCase cases[] = {
      {"fc_rows_64", 64, 64},
      {"fc_rows_256", 256, 128},
      {"conv_rows_512", 512, 256},
      {"conv_rows_1024", 1024, 512},
  };

  std::cout << "## MatrixScalarMultiplyAccumulate<int8>\n\n";
  std::cout << "- Affects: quantized recurrent helpers and row-wise reduction "
               "paths.\n\n";
  PrintIntHeader("Rows x cols");

  for (const MatrixCase& bench : cases) {
    const int64_t work = static_cast<int64_t>(bench.rows) * bench.cols;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> matrix =
        MakeRandomInt8Vector(bench.rows * bench.cols, rng);
    const int32_t scalar = MakeRandomInt32Vector(1, rng, -7, 7)[0];
    std::vector<int32_t> scalar_output(bench.rows, 0);
    std::vector<int32_t> rvv_output(bench.rows, 0);

    PortableMatrixScalarMultiplyAccumulate(matrix.data(), scalar, bench.rows,
                                           bench.cols, scalar_output.data());
    RvvMatrixScalarMultiplyAccumulate(matrix.data(), scalar, bench.rows,
                                      bench.cols, rvv_output.data());
    const IntAccuracy accuracy = CompareIntVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          std::fill(scalar_output.begin(), scalar_output.end(), 0);
          PortableMatrixScalarMultiplyAccumulate(matrix.data(), scalar,
                                                 bench.rows, bench.cols,
                                                 scalar_output.data());
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          std::fill(rvv_output.begin(), rvv_output.end(), 0);
          RvvMatrixScalarMultiplyAccumulate(matrix.data(), scalar, bench.rows,
                                            bench.cols, rvv_output.data());
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "MatrixScalar accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkFloatMatVec(std::mt19937* rng) {
  const MatVecCase cases[] = {
      {"small_fc", 64, 64, 4},
      {"mid_fc", 128, 128, 4},
      {"lstm_like", 640, 2048, 4},
      {"sqrnn_like", 1024, 1024, 8},
  };

  std::cout << "## MatrixBatchVectorMultiplyAccumulate<float>\n\n";
  std::cout << "- Affects: float `fully_connected` and recurrent kernels.\n\n";
  PrintFloatHeader("Rows x cols x batch");

  for (const MatVecCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.rows) * bench.cols * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<float> matrix =
        MakeRandomFloatVector(bench.rows * bench.cols, rng);
    const std::vector<float> vector =
        MakeRandomFloatVector(bench.cols * bench.batch, rng);
    std::vector<float> scalar_output(bench.rows * bench.batch, 0.0f);
    std::vector<float> rvv_output(bench.rows * bench.batch, 0.0f);

    PortableMatrixBatchVectorMultiplyAccumulate(
        matrix.data(), bench.rows, bench.cols, vector.data(), bench.batch,
        scalar_output.data());
    RvvMatrixBatchVectorMultiplyAccumulate(matrix.data(), bench.rows, bench.cols,
                                           vector.data(), bench.batch,
                                           rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(scalar_output.begin(), scalar_output.end(), 0.0f);
          PortableMatrixBatchVectorMultiplyAccumulate(
              matrix.data(), bench.rows, bench.cols, vector.data(), bench.batch,
              scalar_output.data());
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(rvv_output.begin(), rvv_output.end(), 0.0f);
          RvvMatrixBatchVectorMultiplyAccumulate(
              matrix.data(), bench.rows, bench.cols, vector.data(), bench.batch,
              rvv_output.data());
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";
    if (!FloatAccuracyWithinTolerance(accuracy)) {
      std::cerr << "Float matvec accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkInt8MatVec(std::mt19937* rng) {
  const MatVecCase cases[] = {
      {"q_fc_small", 64, 64, 4},
      {"q_fc_mid", 128, 128, 4},
      {"q_lstm_like", 640, 1024, 4},
      {"q_conv_like", 1024, 512, 8},
  };

  std::cout << "## MatrixBatchVectorMultiplyAccumulate<int8>\n\n";
  std::cout << "- Affects: quantized `fully_connected`, hybrid recurrent "
               "helpers, shared int8 GEMV-style call sites.\n\n";
  PrintFloatHeader("Rows x cols x batch");

  for (const MatVecCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.rows) * bench.cols * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> matrix =
        MakeRandomInt8Vector(bench.rows * bench.cols, rng);
    const std::vector<int8_t> vector =
        MakeRandomInt8Vector(bench.cols * bench.batch, rng);
    const std::vector<float> scaling_factors =
        MakeRandomFloatVector(bench.batch, rng, 0.0005f, 0.02f);
    std::vector<float> scalar_output(bench.rows * bench.batch, 0.0f);
    std::vector<float> rvv_output(bench.rows * bench.batch, 0.0f);

    PortableMatrixBatchVectorMultiplyAccumulate(
        matrix.data(), bench.rows, bench.cols, vector.data(),
        scaling_factors.data(), bench.batch, scalar_output.data());
    RvvMatrixBatchVectorMultiplyAccumulate(
        matrix.data(), bench.rows, bench.cols, vector.data(),
        scaling_factors.data(), bench.batch, rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(scalar_output.begin(), scalar_output.end(), 0.0f);
          PortableMatrixBatchVectorMultiplyAccumulate(
              matrix.data(), bench.rows, bench.cols, vector.data(),
              scaling_factors.data(), bench.batch, scalar_output.data());
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(rvv_output.begin(), rvv_output.end(), 0.0f);
          RvvMatrixBatchVectorMultiplyAccumulate(
              matrix.data(), bench.rows, bench.cols, vector.data(),
              scaling_factors.data(), bench.batch, rvv_output.data());
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";
    if (!FloatAccuracyWithinTolerance(accuracy)) {
      std::cerr << "Int8 matvec accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkInt8MatVecWithOffsets(std::mt19937* rng) {
  const MatVecCase cases[] = {
      {"q_row_sum_small", 64, 64, 4},
      {"q_row_sum_mid", 128, 128, 4},
      {"q_row_sum_conv", 256, 576, 8},
      {"q_batch_matmul_like", 512, 512, 8},
  };

  std::cout << "## MatrixBatchVectorMultiplyAccumulate<int8, per-channel + "
               "input_offset>\n\n";
  std::cout << "- Affects: quantized `conv`, `batch_matmul`, and any path that "
               "uses cached row sums plus per-channel scale.\n\n";
  PrintFloatHeader("Rows x cols x batch");

  for (const MatVecCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.rows) * bench.cols * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> matrix =
        MakeRandomInt8Vector(bench.rows * bench.cols, rng);
    const std::vector<int8_t> vector =
        MakeRandomInt8Vector(bench.cols * bench.batch, rng);
    const std::vector<float> scaling_factors =
        MakeRandomFloatVector(bench.batch, rng, 0.0005f, 0.02f);
    const std::vector<float> per_channel_scale =
        MakeRandomFloatVector(bench.rows, rng, 0.25f, 1.25f);
    const std::vector<int32_t> input_offset =
        MakeRandomInt32Vector(bench.batch, rng, -16, 16);
    std::vector<float> scalar_output(bench.rows * bench.batch, 0.0f);
    std::vector<float> rvv_output(bench.rows * bench.batch, 0.0f);
    std::vector<int32_t> scalar_scratch(bench.rows * bench.batch, 0);
    std::vector<int32_t> rvv_scratch(bench.rows * bench.batch, 0);
    std::vector<int32_t> scalar_row_sums(bench.rows, 0);
    std::vector<int32_t> rvv_row_sums(bench.rows, 0);

    bool scalar_compute_row_sums = true;
    PortableMatrixBatchVectorMultiplyAccumulate(
        matrix.data(), bench.rows, bench.cols, vector.data(),
        scaling_factors.data(), bench.batch, scalar_output.data(),
        per_channel_scale.data(), input_offset.data(), scalar_scratch.data(),
        scalar_row_sums.data(), &scalar_compute_row_sums, nullptr);

    bool rvv_compute_row_sums = true;
    RvvMatrixBatchVectorMultiplyAccumulate(
        matrix.data(), bench.rows, bench.cols, vector.data(),
        scaling_factors.data(), bench.batch, rvv_output.data(),
        per_channel_scale.data(), input_offset.data(), rvv_scratch.data(),
        rvv_row_sums.data(), &rvv_compute_row_sums, nullptr);

    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(scalar_output.begin(), scalar_output.end(), 0.0f);
          bool compute_row_sums = true;
          PortableMatrixBatchVectorMultiplyAccumulate(
              matrix.data(), bench.rows, bench.cols, vector.data(),
              scaling_factors.data(), bench.batch, scalar_output.data(),
              per_channel_scale.data(), input_offset.data(),
              scalar_scratch.data(), scalar_row_sums.data(), &compute_row_sums,
              nullptr);
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(rvv_output.begin(), rvv_output.end(), 0.0f);
          bool compute_row_sums = true;
          RvvMatrixBatchVectorMultiplyAccumulate(
              matrix.data(), bench.rows, bench.cols, vector.data(),
              scaling_factors.data(), bench.batch, rvv_output.data(),
              per_channel_scale.data(), input_offset.data(), rvv_scratch.data(),
              rvv_row_sums.data(), &compute_row_sums, nullptr);
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";
    if (!FloatAccuracyWithinTolerance(accuracy)) {
      std::cerr << "Int8 offset matvec accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

}  // namespace

int Main() {
  std::cout << "# RVV vs Scalar: Shared tensor_utils coverage\n\n";
  std::cout << "- Runtime VLEN bits: " << (__riscv_vlenb() * 8) << "\n";
  std::cout << "- Scalar kernels: `Portable*` compiled with "
               "`-fno-tree-vectorize -fno-tree-slp-vectorize`\n";
  std::cout << "- RVV kernels: `Rvv*` compiled with "
               "`-march=rv64gcv_zvl128b -mabi=lp64d`\n";
  std::cout << "- Covered helper families: float dot/reduction, int8 "
               "row-sum/reduction, int8/float matvec.\n\n";

  std::mt19937 rng(20260506);
  BenchmarkFloatDotProduct(&rng);
  BenchmarkFloatReduction(&rng);
  BenchmarkInt8Reduction(&rng);
  BenchmarkMatrixScalarMultiply(&rng);
  BenchmarkFloatMatVec(&rng);
  BenchmarkInt8MatVec(&rng);
  BenchmarkInt8MatVecWithOffsets(&rng);
  return 0;
}

}  // namespace tensor_utils
}  // namespace tflite

int main() { return tflite::tensor_utils::Main(); }
