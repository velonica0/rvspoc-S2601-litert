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

struct SymmetricQuantAccuracy {
  IntAccuracy output_accuracy;
  float min_abs_diff;
  float max_abs_diff;
  float scale_abs_diff;
};

struct AsymmetricQuantAccuracy {
  IntAccuracy output_accuracy;
  float scale_abs_diff;
  int32_t offset_abs_diff;
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

struct BatchVectorCase {
  const char* name;
  int batch;
  int size;
};

struct SparseCase {
  const char* name;
  int rows;
  int cols;
  int batch;
  float density;
};

template <typename T>
struct SegmentedSparseData {
  std::vector<T> matrix;
  std::vector<int32_t> segments;
  std::vector<int32_t> indices;
  int64_t nonzero_values;
};

template <typename T>
struct LedgerSparseData {
  std::vector<T> matrix;
  std::vector<uint8_t> ledger;
  int64_t nonzero_values;
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

std::vector<int16_t> MakeRandomInt16Vector(int size, std::mt19937* rng,
                                           int min_value = -32768,
                                           int max_value = 32767) {
  std::uniform_int_distribution<int> dist(min_value, max_value);
  std::vector<int16_t> values(size);
  for (int16_t& value : values) {
    value = static_cast<int16_t>(dist(*rng));
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

SegmentedSparseData<float> MakeRandomSegmentedSparseFloat(
    int rows, int cols, int block_size, float density, std::mt19937* rng) {
  const int blocks_per_row = cols / block_size;
  std::bernoulli_distribution select_block(density);
  std::uniform_int_distribution<int> block_dist(0, blocks_per_row - 1);
  std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);

  SegmentedSparseData<float> data;
  data.segments.reserve(rows + 1);
  data.segments.push_back(0);
  data.nonzero_values = 0;

  for (int row = 0; row < rows; ++row) {
    std::vector<int32_t> row_indices;
    row_indices.reserve(blocks_per_row);
    for (int block = 0; block < blocks_per_row; ++block) {
      if (select_block(*rng)) {
        row_indices.push_back(block);
      }
    }
    if (row_indices.empty()) {
      row_indices.push_back(block_dist(*rng));
    }

    data.indices.insert(data.indices.end(), row_indices.begin(),
                        row_indices.end());
    data.segments.push_back(static_cast<int32_t>(data.indices.size()));
    data.nonzero_values += static_cast<int64_t>(row_indices.size()) * block_size;

    for (size_t block = 0; block < row_indices.size(); ++block) {
      for (int c = 0; c < block_size; ++c) {
        data.matrix.push_back(value_dist(*rng));
      }
    }
  }

  return data;
}

SegmentedSparseData<int8_t> MakeRandomSegmentedSparseInt8(
    int rows, int cols, int block_size, float density, std::mt19937* rng) {
  const int blocks_per_row = cols / block_size;
  std::bernoulli_distribution select_block(density);
  std::uniform_int_distribution<int> block_dist(0, blocks_per_row - 1);
  std::uniform_int_distribution<int> value_dist(-127, 127);

  SegmentedSparseData<int8_t> data;
  data.segments.reserve(rows + 1);
  data.segments.push_back(0);
  data.nonzero_values = 0;

  for (int row = 0; row < rows; ++row) {
    std::vector<int32_t> row_indices;
    row_indices.reserve(blocks_per_row);
    for (int block = 0; block < blocks_per_row; ++block) {
      if (select_block(*rng)) {
        row_indices.push_back(block);
      }
    }
    if (row_indices.empty()) {
      row_indices.push_back(block_dist(*rng));
    }

    data.indices.insert(data.indices.end(), row_indices.begin(),
                        row_indices.end());
    data.segments.push_back(static_cast<int32_t>(data.indices.size()));
    data.nonzero_values += static_cast<int64_t>(row_indices.size()) * block_size;

    for (size_t block = 0; block < row_indices.size(); ++block) {
      for (int c = 0; c < block_size; ++c) {
        data.matrix.push_back(static_cast<int8_t>(value_dist(*rng)));
      }
    }
  }

  return data;
}

LedgerSparseData<float> MakeRandomLedgerSparseFloat(int rows, int cols,
                                                    int block_size,
                                                    float density,
                                                    std::mt19937* rng) {
  const int blocks_per_row = cols / block_size;
  std::bernoulli_distribution select_block(density);
  std::uniform_int_distribution<int> block_dist(0, blocks_per_row - 1);
  std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);

  LedgerSparseData<float> data;
  data.nonzero_values = 0;

  for (int row = 0; row < rows; ++row) {
    std::vector<uint8_t> row_indices;
    row_indices.reserve(blocks_per_row);
    for (int block = 0; block < blocks_per_row; ++block) {
      if (select_block(*rng)) {
        row_indices.push_back(static_cast<uint8_t>(block));
      }
    }
    if (row_indices.empty()) {
      row_indices.push_back(static_cast<uint8_t>(block_dist(*rng)));
    }

    data.ledger.push_back(static_cast<uint8_t>(row_indices.size()));
    data.ledger.insert(data.ledger.end(), row_indices.begin(), row_indices.end());
    data.nonzero_values += static_cast<int64_t>(row_indices.size()) * block_size;

    for (size_t block = 0; block < row_indices.size(); ++block) {
      for (int c = 0; c < block_size; ++c) {
        data.matrix.push_back(value_dist(*rng));
      }
    }
  }

  return data;
}

LedgerSparseData<int8_t> MakeRandomLedgerSparseInt8(int rows, int cols,
                                                    int block_size,
                                                    float density,
                                                    std::mt19937* rng) {
  const int blocks_per_row = cols / block_size;
  std::bernoulli_distribution select_block(density);
  std::uniform_int_distribution<int> block_dist(0, blocks_per_row - 1);
  std::uniform_int_distribution<int> value_dist(-127, 127);

  LedgerSparseData<int8_t> data;
  data.nonzero_values = 0;

  for (int row = 0; row < rows; ++row) {
    std::vector<uint8_t> row_indices;
    row_indices.reserve(blocks_per_row);
    for (int block = 0; block < blocks_per_row; ++block) {
      if (select_block(*rng)) {
        row_indices.push_back(static_cast<uint8_t>(block));
      }
    }
    if (row_indices.empty()) {
      row_indices.push_back(static_cast<uint8_t>(block_dist(*rng)));
    }

    data.ledger.push_back(static_cast<uint8_t>(row_indices.size()));
    data.ledger.insert(data.ledger.end(), row_indices.begin(), row_indices.end());
    data.nonzero_values += static_cast<int64_t>(row_indices.size()) * block_size;

    for (size_t block = 0; block < row_indices.size(); ++block) {
      for (int c = 0; c < block_size; ++c) {
        data.matrix.push_back(static_cast<int8_t>(value_dist(*rng)));
      }
    }
  }

  return data;
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

template <typename T>
IntAccuracy CompareIntegerVectors(const std::vector<T>& lhs,
                                  const std::vector<T>& rhs) {
  IntAccuracy accuracy = {0, 0};
  for (size_t i = 0; i < lhs.size(); ++i) {
    const int32_t abs_diff = std::abs(static_cast<int32_t>(lhs[i]) -
                                      static_cast<int32_t>(rhs[i]));
    accuracy.max_abs_diff = std::max(accuracy.max_abs_diff, abs_diff);
    accuracy.mismatches += lhs[i] != rhs[i];
  }
  return accuracy;
}

SymmetricQuantAccuracy CompareSymmetricQuantResults(
    const std::vector<int8_t>& scalar_output,
    const std::vector<int8_t>& rvv_output, float scalar_min, float rvv_min,
    float scalar_max, float rvv_max, float scalar_scale, float rvv_scale) {
  return {
      CompareIntegerVectors(scalar_output, rvv_output),
      std::abs(scalar_min - rvv_min),
      std::abs(scalar_max - rvv_max),
      std::abs(scalar_scale - rvv_scale),
  };
}

AsymmetricQuantAccuracy CompareAsymmetricQuantResults(
    const std::vector<int8_t>& scalar_output,
    const std::vector<int8_t>& rvv_output, float scalar_scale, float rvv_scale,
    int32_t scalar_offset, int32_t rvv_offset) {
  return {
      CompareIntegerVectors(scalar_output, rvv_output),
      std::abs(scalar_scale - rvv_scale),
      std::abs(scalar_offset - rvv_offset),
  };
}

bool FloatAccuracyWithinTolerance(const FloatAccuracy& accuracy) {
  return accuracy.max_abs_diff <= 2.0e-4f ||
         (accuracy.max_abs_diff <= 3.0e-4f &&
          accuracy.mean_abs_diff <= 2.0e-5);
}

bool IntAccuracyWithinTolerance(const IntAccuracy& accuracy,
                                int32_t tolerance) {
  return accuracy.max_abs_diff <= tolerance;
}

bool SymmetricQuantAccuracyWithinTolerance(
    const SymmetricQuantAccuracy& accuracy) {
  return IntAccuracyWithinTolerance(accuracy.output_accuracy, 0) &&
         accuracy.min_abs_diff <= 1.0e-6f &&
         accuracy.max_abs_diff <= 1.0e-6f &&
         accuracy.scale_abs_diff <= 1.0e-6f;
}

bool AsymmetricQuantAccuracyWithinTolerance(
    const AsymmetricQuantAccuracy& accuracy) {
  return IntAccuracyWithinTolerance(accuracy.output_accuracy, 0) &&
         accuracy.scale_abs_diff <= 1.0e-6f && accuracy.offset_abs_diff == 0;
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

void PrintSymmetricQuantHeader(const std::string& shape_column_name) {
  std::cout << "| Case | " << shape_column_name
            << " | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | "
               "RVV GOPS | Max abs diff | Mismatches | Min diff | Max diff | "
               "Scale diff |\n";
  std::cout << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
               "---: | ---: | ---: | ---: |\n";
}

void PrintAsymmetricQuantHeader(const std::string& shape_column_name) {
  std::cout << "| Case | " << shape_column_name
            << " | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | "
               "RVV GOPS | Max abs diff | Mismatches | Scale diff | Offset "
               "diff |\n";
  std::cout << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
               "---: | ---: | ---: |\n";
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

void BenchmarkBatchVectorDotProductInt16(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"svdf_dot_q15_small", 4, 64},
      {"svdf_dot_q15_mid", 8, 256},
      {"svdf_dot_q15_large", 16, 1024},
  };

  std::cout << "## BatchVectorBatchVectorDotProduct<int16>\n\n";
  std::cout << "- Affects: `svdf` and batched recurrent dot-product helper "
               "paths.\n\n";
  PrintIntHeader("Batch x size");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(static_cast<int64_t>(total) * 2);
    const std::vector<int16_t> lhs =
        MakeRandomInt16Vector(total, rng, -1024, 1024);
    const std::vector<int16_t> rhs =
        MakeRandomInt16Vector(total, rng, -1024, 1024);
    std::vector<int32_t> scalar_output(bench.batch, 0);
    std::vector<int32_t> rvv_output(bench.batch, 0);

    PortableBatchVectorBatchVectorDotProduct(lhs.data(), rhs.data(), bench.size,
                                             bench.batch, scalar_output.data());
    RvvBatchVectorBatchVectorDotProduct(lhs.data(), rhs.data(), bench.size,
                                        bench.batch, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * total, [&]() -> double {
          PortableBatchVectorBatchVectorDotProduct(
              lhs.data(), rhs.data(), bench.size, bench.batch,
              scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * total, [&]() -> double {
          RvvBatchVectorBatchVectorDotProduct(lhs.data(), rhs.data(), bench.size,
                                              bench.batch, rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "BatchVectorBatchVectorDotProduct accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
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
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

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

void BenchmarkInt32Reduction(std::mt19937* rng) {
  const ReductionCase cases[] = {
      {"reduce_i32_small", 64, 32},
      {"reduce_i32_mid", 128, 128},
      {"reduce_i32_large", 256, 256},
  };

  std::cout << "## ReductionSumVector<int32>\n\n";
  std::cout << "- Affects: scalar accumulation helper paths such as reference "
               "SVDF and utility reductions.\n\n";
  PrintIntHeader("Output x reduction");

  for (const ReductionCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.output_size) * bench.reduction_size;
    const int iterations = ChooseIterations(work);
    const std::vector<int32_t> input =
        MakeRandomInt32Vector(bench.output_size * bench.reduction_size, rng,
                              -2048, 2048);
    std::vector<int32_t> scalar_output(bench.output_size, 0);
    std::vector<int32_t> rvv_output(bench.output_size, 0);

    PortableReductionSumVector(input.data(), scalar_output.data(),
                               bench.output_size, bench.reduction_size);
    RvvReductionSumVector(input.data(), rvv_output.data(), bench.output_size,
                          bench.reduction_size);
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          PortableReductionSumVector(input.data(), scalar_output.data(),
                                     bench.output_size, bench.reduction_size);
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(work), [&]() -> double {
          RvvReductionSumVector(input.data(), rvv_output.data(),
                                bench.output_size, bench.reduction_size);
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
      std::cerr << "Int32 reduction accuracy check failed for " << bench.name
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
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

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

void BenchmarkGateMatVecInt16(std::mt19937* rng) {
  const MatVecCase cases[] = {
      {"gate_q15_small", 128, 128, 4},
      {"gate_q15_mid", 256, 256, 4},
      {"gate_q15_large", 512, 512, 8},
  };
  constexpr int32_t kMultiplier = 1073741824;
  constexpr int32_t kShift = -8;
  constexpr int32_t kOutputZp = 3;

  std::cout << "## MatrixBatchVectorMultiplyAccumulate<int8 -> int16>\n\n";
  std::cout << "- Affects: quantized recurrent gate accumulate helper paths.\n\n";
  PrintIntHeader("Rows x cols x batch");

  for (const MatVecCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.rows) * bench.cols * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> input =
        MakeRandomInt8Vector(bench.batch * bench.cols, rng);
    const std::vector<int8_t> weights =
        MakeRandomInt8Vector(bench.rows * bench.cols, rng);
    const std::vector<int32_t> bias =
        MakeRandomInt32Vector(bench.rows, rng, -4096, 4096);
    const std::vector<int16_t> base_output =
        MakeRandomInt16Vector(bench.batch * bench.rows, rng, -4096, 4096);
    std::vector<int16_t> scalar_output = base_output;
    std::vector<int16_t> rvv_output = base_output;
    std::vector<int32_t> scalar_scratch(bench.batch * bench.rows, 0);
    std::vector<int32_t> rvv_scratch(bench.batch * bench.rows, 0);

    PortableMatrixBatchVectorMultiplyAccumulate(
        input.data(), bias.data(), weights.data(), kMultiplier, kShift,
        bench.batch, bench.cols, bench.rows, kOutputZp, scalar_scratch.data(),
        scalar_output.data(), nullptr);
    RvvMatrixBatchVectorMultiplyAccumulate(
        input.data(), bias.data(), weights.data(), kMultiplier, kShift,
        bench.batch, bench.cols, bench.rows, kOutputZp, rvv_scratch.data(),
        rvv_output.data(), nullptr);
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          scalar_output = base_output;
          PortableMatrixBatchVectorMultiplyAccumulate(
              input.data(), bias.data(), weights.data(), kMultiplier, kShift,
              bench.batch, bench.cols, bench.rows, kOutputZp,
              scalar_scratch.data(), scalar_output.data(), nullptr);
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          rvv_output = base_output;
          RvvMatrixBatchVectorMultiplyAccumulate(
              input.data(), bias.data(), weights.data(), kMultiplier, kShift,
              bench.batch, bench.cols, bench.rows, kOutputZp,
              rvv_scratch.data(), rvv_output.data(), nullptr);
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "Gate int16 matvec accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkGateMatVecInt8(std::mt19937* rng) {
  const MatVecCase cases[] = {
      {"gate_q8_small", 128, 128, 4},
      {"gate_q8_mid", 256, 256, 4},
      {"gate_q8_large", 512, 512, 8},
  };
  constexpr int32_t kMultiplier = 1073741824;
  constexpr int32_t kShift = -8;
  constexpr int32_t kOutputZp = -5;

  std::cout << "## MatrixBatchVectorMultiplyAccumulate<int8 -> int8>\n\n";
  std::cout << "- Affects: quantized projection and low-precision recurrent "
               "accumulate helper paths.\n\n";
  PrintIntHeader("Rows x cols x batch");

  for (const MatVecCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.rows) * bench.cols * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> input =
        MakeRandomInt8Vector(bench.batch * bench.cols, rng);
    const std::vector<int8_t> weights =
        MakeRandomInt8Vector(bench.rows * bench.cols, rng);
    const std::vector<int32_t> bias =
        MakeRandomInt32Vector(bench.rows, rng, -4096, 4096);
    const std::vector<int8_t> base_output =
        MakeRandomInt8Vector(bench.batch * bench.rows, rng, -64, 64);
    std::vector<int8_t> scalar_output = base_output;
    std::vector<int8_t> rvv_output = base_output;
    std::vector<int32_t> scalar_scratch(bench.batch * bench.rows, 0);
    std::vector<int32_t> rvv_scratch(bench.batch * bench.rows, 0);

    PortableMatrixBatchVectorMultiplyAccumulate(
        input.data(), bias.data(), weights.data(), kMultiplier, kShift,
        bench.batch, bench.cols, bench.rows, kOutputZp, scalar_scratch.data(),
        scalar_output.data(), nullptr);
    RvvMatrixBatchVectorMultiplyAccumulate(
        input.data(), bias.data(), weights.data(), kMultiplier, kShift,
        bench.batch, bench.cols, bench.rows, kOutputZp, rvv_scratch.data(),
        rvv_output.data(), nullptr);
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          scalar_output = base_output;
          PortableMatrixBatchVectorMultiplyAccumulate(
              input.data(), bias.data(), weights.data(), kMultiplier, kShift,
              bench.batch, bench.cols, bench.rows, kOutputZp,
              scalar_scratch.data(), scalar_output.data(), nullptr);
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          rvv_output = base_output;
          RvvMatrixBatchVectorMultiplyAccumulate(
              input.data(), bias.data(), weights.data(), kMultiplier, kShift,
              bench.batch, bench.cols, bench.rows, kOutputZp,
              rvv_scratch.data(), rvv_output.data(), nullptr);
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "Gate int8 matvec accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkGateMatVecNoAccumulateInt8(std::mt19937* rng) {
  const MatVecCase cases[] = {
      {"gate_noacc_q8_small", 128, 128, 4},
      {"gate_noacc_q8_mid", 256, 256, 4},
      {"gate_noacc_q8_large", 512, 512, 8},
  };
  constexpr int32_t kInputZp = -7;
  constexpr int32_t kMultiplier = 1347771520;
  constexpr int32_t kShift = -8;
  constexpr int32_t kOutputZp = -11;

  std::cout << "## MatrixBatchVectorMultiply<int8 -> int8>\n\n";
  std::cout << "- Affects: quantized LSTM gate matmul paths before "
               "saturating-add and activation.\n\n";
  PrintIntHeader("Rows x cols x batch");

  for (const MatVecCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.rows) * bench.cols * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> input =
        MakeRandomInt8Vector(bench.batch * bench.cols, rng);
    const std::vector<int8_t> weights =
        MakeRandomInt8Vector(bench.rows * bench.cols, rng);
    std::vector<int8_t> scalar_output(bench.batch * bench.rows, 0);
    std::vector<int8_t> rvv_output(bench.batch * bench.rows, 0);

    PortableMatrixBatchVectorMultiply(
        input.data(), kInputZp, weights.data(), kMultiplier, kShift,
        bench.batch, bench.cols, bench.rows, scalar_output.data(), kOutputZp);
    RvvMatrixBatchVectorMultiply(
        input.data(), kInputZp, weights.data(), kMultiplier, kShift,
        bench.batch, bench.cols, bench.rows, rvv_output.data(), kOutputZp);
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          PortableMatrixBatchVectorMultiply(
              input.data(), kInputZp, weights.data(), kMultiplier, kShift,
              bench.batch, bench.cols, bench.rows, scalar_output.data(),
              kOutputZp);
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          RvvMatrixBatchVectorMultiply(
              input.data(), kInputZp, weights.data(), kMultiplier, kShift,
              bench.batch, bench.cols, bench.rows, rvv_output.data(),
              kOutputZp);
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "MatrixBatchVectorMultiply<int8> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkProjectionMatVecInt8(std::mt19937* rng) {
  const MatVecCase cases[] = {
      {"proj_q8_small", 128, 128, 4},
      {"proj_q8_mid", 256, 256, 4},
      {"proj_q8_large", 512, 512, 8},
  };
  constexpr int32_t kMultiplier = 1347771520;
  constexpr int32_t kShift = -8;
  constexpr int32_t kOutputZp = -11;

  std::cout << "## MatrixBatchVectorMultiply<int16 x int8 -> int8>\n\n";
  std::cout << "- Affects: quantized projection/output matmul helper paths.\n\n";
  PrintIntHeader("Rows x cols x batch");

  for (const MatVecCase& bench : cases) {
    const int64_t work =
        static_cast<int64_t>(bench.rows) * bench.cols * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int16_t> hidden =
        MakeRandomInt16Vector(bench.batch * bench.cols, rng, -4096, 4096);
    const std::vector<int8_t> weights =
        MakeRandomInt8Vector(bench.rows * bench.cols, rng);
    const std::vector<int32_t> bias =
        MakeRandomInt32Vector(bench.rows, rng, -2048, 2048);
    std::vector<int8_t> scalar_output(bench.batch * bench.rows, 0);
    std::vector<int8_t> rvv_output(bench.batch * bench.rows, 0);

    PortableMatrixBatchVectorMultiply(hidden.data(), weights.data(),
                                      kMultiplier, kShift, bias.data(),
                                      bench.batch, bench.cols, bench.rows,
                                      kOutputZp, scalar_output.data());
    RvvMatrixBatchVectorMultiply(hidden.data(), weights.data(), kMultiplier,
                                 kShift, bias.data(), bench.batch, bench.cols,
                                 bench.rows, kOutputZp, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          PortableMatrixBatchVectorMultiply(
              hidden.data(), weights.data(), kMultiplier, kShift, bias.data(),
              bench.batch, bench.cols, bench.rows, kOutputZp,
              scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          RvvMatrixBatchVectorMultiply(
              hidden.data(), weights.data(), kMultiplier, kShift, bias.data(),
              bench.batch, bench.cols, bench.rows, kOutputZp,
              rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "Projection matvec accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkSparseFloat1x4(std::mt19937* rng) {
  const SparseCase cases[] = {
      {"sparse_1x4_small", 64, 256, 4, 0.25f},
      {"sparse_1x4_mid", 128, 512, 4, 0.25f},
      {"sparse_1x4_large", 256, 1024, 8, 0.25f},
  };

  std::cout << "## SparseMatrixBatchVectorMultiplyAccumulate1x4<float>\n\n";
  std::cout << "- Affects: sparse float `fully_connected` and sparse recurrent "
               "helper paths.\n\n";
  PrintFloatHeader("Rows x cols x batch");

  for (const SparseCase& bench : cases) {
    const auto sparse = MakeRandomSegmentedSparseFloat(
        bench.rows, bench.cols, 4, bench.density, rng);
    const int64_t work = sparse.nonzero_values * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<float> vector =
        MakeRandomFloatVector(bench.cols * bench.batch, rng, -1.0f, 1.0f);
    std::vector<float> scalar_output(bench.rows * bench.batch, 0.0f);
    std::vector<float> rvv_output(bench.rows * bench.batch, 0.0f);

    PortableSparseMatrixBatchVectorMultiplyAccumulate1x4(
        sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
        bench.rows, bench.cols, vector.data(), bench.batch,
        scalar_output.data());
    RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
        sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
        bench.rows, bench.cols, vector.data(), bench.batch, rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(scalar_output.begin(), scalar_output.end(), 0.0f);
          PortableSparseMatrixBatchVectorMultiplyAccumulate1x4(
              sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
              bench.rows, bench.cols, vector.data(), bench.batch,
              scalar_output.data());
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(rvv_output.begin(), rvv_output.end(), 0.0f);
          RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
              sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
              bench.rows, bench.cols, vector.data(), bench.batch,
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
      std::cerr << "Sparse 1x4 float accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkSparseFloatLedger(std::mt19937* rng) {
  const SparseCase cases[] = {
      {"sparse_ledger_small", 64, 256, 4, 0.25f},
      {"sparse_ledger_mid", 128, 512, 4, 0.25f},
      {"sparse_ledger_large", 256, 1024, 8, 0.25f},
  };

  std::cout << "## SparseMatrixBatchVectorMultiplyAccumulate<float ledger>\n\n";
  std::cout << "- Affects: sparse float matvec helper paths with ledger "
               "format.\n\n";
  PrintFloatHeader("Rows x cols x batch");

  for (const SparseCase& bench : cases) {
    const auto sparse =
        MakeRandomLedgerSparseFloat(bench.rows, bench.cols, 16, bench.density,
                                    rng);
    const int64_t work = sparse.nonzero_values * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<float> vector =
        MakeRandomFloatVector(bench.cols * bench.batch, rng, -1.0f, 1.0f);
    std::vector<float> scalar_output(bench.rows * bench.batch, 0.0f);
    std::vector<float> rvv_output(bench.rows * bench.batch, 0.0f);

    PortableSparseMatrixBatchVectorMultiplyAccumulate(
        sparse.matrix.data(), sparse.ledger.data(), bench.rows, bench.cols,
        vector.data(), bench.batch, scalar_output.data());
    RvvSparseMatrixBatchVectorMultiplyAccumulate(
        sparse.matrix.data(), sparse.ledger.data(), bench.rows, bench.cols,
        vector.data(), bench.batch, rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(scalar_output.begin(), scalar_output.end(), 0.0f);
          PortableSparseMatrixBatchVectorMultiplyAccumulate(
              sparse.matrix.data(), sparse.ledger.data(), bench.rows,
              bench.cols, vector.data(), bench.batch, scalar_output.data());
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(rvv_output.begin(), rvv_output.end(), 0.0f);
          RvvSparseMatrixBatchVectorMultiplyAccumulate(
              sparse.matrix.data(), sparse.ledger.data(), bench.rows,
              bench.cols, vector.data(), bench.batch, rvv_output.data());
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
      std::cerr << "Sparse ledger float accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkSparseInt8Ledger(std::mt19937* rng) {
  const SparseCase cases[] = {
      {"sparse_qfloat_small", 64, 256, 4, 0.25f},
      {"sparse_qfloat_mid", 128, 512, 4, 0.25f},
      {"sparse_qfloat_large", 256, 1024, 8, 0.25f},
  };

  std::cout << "## SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>\n\n";
  std::cout << "- Affects: sparse quantized matvec helper paths that dequantize "
               "to float.\n\n";
  PrintFloatHeader("Rows x cols x batch");

  for (const SparseCase& bench : cases) {
    const auto sparse =
        MakeRandomLedgerSparseInt8(bench.rows, bench.cols, 16, bench.density,
                                   rng);
    const int64_t work = sparse.nonzero_values * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> vectors =
        MakeRandomInt8Vector(bench.cols * bench.batch, rng);
    const std::vector<float> scaling_factors =
        MakeRandomFloatVector(bench.batch, rng, 0.001f, 0.02f);
    const std::vector<float> per_channel_scale =
        MakeRandomFloatVector(bench.rows, rng, 0.25f, 1.25f);
    std::vector<float> scalar_output(bench.rows * bench.batch, 0.0f);
    std::vector<float> rvv_output(bench.rows * bench.batch, 0.0f);

    PortableSparseMatrixBatchVectorMultiplyAccumulate(
        sparse.matrix.data(), sparse.ledger.data(), bench.rows, bench.cols,
        vectors.data(), scaling_factors.data(), bench.batch,
        scalar_output.data(), per_channel_scale.data());
    RvvSparseMatrixBatchVectorMultiplyAccumulate(
        sparse.matrix.data(), sparse.ledger.data(), bench.rows, bench.cols,
        vectors.data(), scaling_factors.data(), bench.batch, rvv_output.data(),
        per_channel_scale.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(scalar_output.begin(), scalar_output.end(), 0.0f);
          PortableSparseMatrixBatchVectorMultiplyAccumulate(
              sparse.matrix.data(), sparse.ledger.data(), bench.rows,
              bench.cols, vectors.data(), scaling_factors.data(), bench.batch,
              scalar_output.data(), per_channel_scale.data());
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          std::fill(rvv_output.begin(), rvv_output.end(), 0.0f);
          RvvSparseMatrixBatchVectorMultiplyAccumulate(
              sparse.matrix.data(), sparse.ledger.data(), bench.rows,
              bench.cols, vectors.data(), scaling_factors.data(), bench.batch,
              rvv_output.data(), per_channel_scale.data());
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
      std::cerr << "Sparse int8->float accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkSparseInt8Output(std::mt19937* rng) {
  const SparseCase cases[] = {
      {"sparse_q8_small", 64, 256, 4, 0.25f},
      {"sparse_q8_mid", 128, 512, 4, 0.25f},
      {"sparse_q8_large", 256, 1024, 8, 0.25f},
  };
  constexpr int32_t kOutputMultiplier = 1073741824;
  constexpr int32_t kOutputShift = -8;
  constexpr int32_t kOutputOffset = 7;
  constexpr int32_t kOutputActivationMin = -128;
  constexpr int32_t kOutputActivationMax = 127;

  std::cout << "## SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>\n\n";
  std::cout << "- Affects: sparse quantized fully-connected and recurrent "
               "output helper paths.\n\n";
  PrintIntHeader("Rows x cols x batch");

  for (const SparseCase& bench : cases) {
    const auto sparse = MakeRandomSegmentedSparseInt8(
        bench.rows, bench.cols, 16, bench.density, rng);
    const int64_t work = sparse.nonzero_values * bench.batch;
    const int iterations = ChooseIterations(work);
    const std::vector<int8_t> vector =
        MakeRandomInt8Vector(bench.cols * bench.batch, rng);
    const std::vector<int32_t> bias =
        MakeRandomInt32Vector(bench.rows, rng, -2048, 2048);
    const std::vector<int32_t> per_channel_scale =
        MakeRandomInt32Vector(bench.rows, rng, 1 << 29, 1 << 30);
    const std::vector<int32_t> per_channel_shift =
        MakeRandomInt32Vector(bench.rows, rng, -8, -4);
    const int32_t input_offset = MakeRandomInt32Vector(1, rng, -8, 8)[0];
    std::vector<int8_t> scalar_output(bench.rows * bench.batch, 0);
    std::vector<int8_t> rvv_output(bench.rows * bench.batch, 0);

    PortableSparseMatrixBatchVectorMultiplyAccumulate1x16(
        sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
        bench.rows, bench.cols, vector.data(), bias.data(), bench.batch,
        input_offset, kOutputMultiplier, kOutputShift, per_channel_scale.data(),
        per_channel_shift.data(), kOutputOffset, kOutputActivationMin,
        kOutputActivationMax, scalar_output.data());
    RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
        sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
        bench.rows, bench.cols, vector.data(), bias.data(), bench.batch,
        input_offset, kOutputMultiplier, kOutputShift, per_channel_scale.data(),
        per_channel_shift.data(), kOutputOffset, kOutputActivationMin,
        kOutputActivationMax, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          PortableSparseMatrixBatchVectorMultiplyAccumulate1x16(
              sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
              bench.rows, bench.cols, vector.data(), bias.data(), bench.batch,
              input_offset, kOutputMultiplier, kOutputShift,
              per_channel_scale.data(), per_channel_shift.data(),
              kOutputOffset, kOutputActivationMin, kOutputActivationMax,
              scalar_output.data());
          return scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * work, [&]() -> double {
          RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
              sparse.matrix.data(), sparse.segments.data(), sparse.indices.data(),
              bench.rows, bench.cols, vector.data(), bias.data(), bench.batch,
              input_offset, kOutputMultiplier, kOutputShift,
              per_channel_scale.data(), per_channel_shift.data(),
              kOutputOffset, kOutputActivationMin, kOutputActivationMax,
              rvv_output.data());
          return rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.rows << " x "
              << bench.cols << " x " << bench.batch << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "Sparse int8 output accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkIsZeroVectorFloat(std::mt19937* rng) {
  (void)rng;
  const VectorCase cases[] = {
      {"zero_f32_64", 64},
      {"zero_f32_1024", 1024},
      {"zero_f32_4096", 4096},
  };

  std::cout << "## IsZeroVector<float>\n\n";
  std::cout << "- Affects: zero-skip precheck paths in hybrid and recurrent "
               "helpers.\n\n";
  PrintIntHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<float> input(bench.size, 0.0f);
    const int scalar_value = PortableIsZeroVector(input.data(), bench.size);
    const int rvv_value = RvvIsZeroVector(input.data(), bench.size);
    const std::vector<int> scalar_output = {scalar_value};
    const std::vector<int> rvv_output = {rvv_value};
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       return PortableIsZeroVector(input.data(), bench.size);
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       return RvvIsZeroVector(input.data(), bench.size);
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "IsZeroVector<float> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkIsZeroVectorInt8(std::mt19937* rng) {
  (void)rng;
  const VectorCase cases[] = {
      {"zero_i8_64", 64},
      {"zero_i8_1024", 1024},
      {"zero_i8_4096", 4096},
  };

  std::cout << "## IsZeroVector<int8>\n\n";
  std::cout << "- Affects: zero-skip checks for quantized helper paths.\n\n";
  PrintIntHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<int8_t> input(bench.size, 0);
    const int scalar_value = PortableIsZeroVector(input.data(), bench.size);
    const int rvv_value = RvvIsZeroVector(input.data(), bench.size);
    const std::vector<int> scalar_output = {scalar_value};
    const std::vector<int> rvv_output = {rvv_value};
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       return PortableIsZeroVector(input.data(), bench.size);
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       return RvvIsZeroVector(input.data(), bench.size);
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "IsZeroVector<int8> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkSymmetricQuantizeFloats(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"sym_quant_small", 64},
      {"sym_quant_mid", 1024},
      {"sym_quant_large", 4096},
  };

  std::cout << "## SymmetricQuantizeFloats\n\n";
  std::cout << "- Affects: hybrid `fully_connected`, `batch_matmul`, weight "
               "and activation pre-quant helper paths.\n\n";
  PrintSymmetricQuantHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<float> input =
        MakeRandomFloatVector(bench.size, rng, -1000.0f, 1000.0f);
    std::vector<int8_t> scalar_output(bench.size);
    std::vector<int8_t> rvv_output(bench.size);
    float scalar_min = 0.0f;
    float scalar_max = 0.0f;
    float scalar_scale = 0.0f;
    float rvv_min = 0.0f;
    float rvv_max = 0.0f;
    float rvv_scale = 0.0f;

    PortableSymmetricQuantizeFloats(input.data(), bench.size,
                                    scalar_output.data(), &scalar_min,
                                    &scalar_max, &scalar_scale);
    RvvSymmetricQuantizeFloats(input.data(), bench.size, rvv_output.data(),
                               &rvv_min, &rvv_max, &rvv_scale);
    const SymmetricQuantAccuracy accuracy = CompareSymmetricQuantResults(
        scalar_output, rvv_output, scalar_min, rvv_min, scalar_max, rvv_max,
        scalar_scale, rvv_scale);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 3.0 * bench.size, [&]() -> double {
          PortableSymmetricQuantizeFloats(input.data(), bench.size,
                                          scalar_output.data(), &scalar_min,
                                          &scalar_max, &scalar_scale);
          return scalar_output.empty() ? scalar_scale
                                       : scalar_output[0] + scalar_scale;
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 3.0 * bench.size, [&]() -> double {
          RvvSymmetricQuantizeFloats(input.data(), bench.size,
                                     rvv_output.data(), &rvv_min, &rvv_max,
                                     &rvv_scale);
          return rvv_output.empty() ? rvv_scale : rvv_output[0] + rvv_scale;
        });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.output_accuracy.max_abs_diff
              << " | " << accuracy.output_accuracy.mismatches << " | "
              << std::setprecision(8) << accuracy.min_abs_diff << " | "
              << accuracy.max_abs_diff << " | " << accuracy.scale_abs_diff
              << " |\n";
    if (!SymmetricQuantAccuracyWithinTolerance(accuracy)) {
      std::cerr << "SymmetricQuantizeFloats accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkAsymmetricQuantizeFloats(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"asym_quant_small", 64},
      {"asym_quant_mid", 1024},
      {"asym_quant_large", 4096},
  };

  std::cout << "## AsymmetricQuantizeFloats\n\n";
  std::cout << "- Affects: hybrid `conv`, `depthwise_conv`, "
               "`transpose_conv`, and int8 input staging paths.\n\n";
  PrintAsymmetricQuantHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<float> input =
        MakeRandomFloatVector(bench.size, rng, -320.0f, 960.0f);
    std::vector<int8_t> scalar_output(bench.size);
    std::vector<int8_t> rvv_output(bench.size);
    float scalar_scale = 0.0f;
    float rvv_scale = 0.0f;
    int32_t scalar_offset = 0;
    int32_t rvv_offset = 0;

    PortableAsymmetricQuantizeFloats(input.data(), bench.size,
                                     scalar_output.data(), &scalar_scale,
                                     &scalar_offset);
    RvvAsymmetricQuantizeFloats(input.data(), bench.size, rvv_output.data(),
                                &rvv_scale, &rvv_offset);
    const AsymmetricQuantAccuracy accuracy = CompareAsymmetricQuantResults(
        scalar_output, rvv_output, scalar_scale, rvv_scale, scalar_offset,
        rvv_offset);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 3.0 * bench.size, [&]() -> double {
          PortableAsymmetricQuantizeFloats(input.data(), bench.size,
                                           scalar_output.data(), &scalar_scale,
                                           &scalar_offset);
          return scalar_output.empty()
                     ? scalar_scale + scalar_offset
                     : scalar_output[0] + scalar_scale + scalar_offset;
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 3.0 * bench.size, [&]() -> double {
          RvvAsymmetricQuantizeFloats(input.data(), bench.size,
                                      rvv_output.data(), &rvv_scale,
                                      &rvv_offset);
          return rvv_output.empty() ? rvv_scale + rvv_offset
                                    : rvv_output[0] + rvv_scale + rvv_offset;
        });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.output_accuracy.max_abs_diff
              << " | " << accuracy.output_accuracy.mismatches << " | "
              << std::setprecision(8) << accuracy.scale_abs_diff << " | "
              << accuracy.offset_abs_diff << " |\n";
    if (!AsymmetricQuantAccuracyWithinTolerance(accuracy)) {
      std::cerr << "AsymmetricQuantizeFloats accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkApplyLayerNorm(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"ln_small", 4, 64},
      {"ln_mid", 4, 256},
      {"ln_gate_like", 8, 1024},
  };
  constexpr int32_t kMultiplier = 1895840000;
  constexpr int32_t kShift = -13;
  constexpr int32_t kVarianceLimit = 1;

  std::cout << "## ApplyLayerNorm<int16>\n\n";
  std::cout << "- Affects: quantized recurrent gate normalization helper "
               "paths.\n\n";
  PrintIntHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(static_cast<int64_t>(total) * 8);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(total, rng, -1500, 1500);
    const std::vector<int16_t> weights =
        MakeRandomInt16Vector(bench.size, rng, 12000, 28000);
    const std::vector<int32_t> bias =
        MakeRandomInt32Vector(bench.size, rng, -16000000, -12000000);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableApplyLayerNorm(input.data(), weights.data(), bias.data(),
                           kMultiplier, kShift, kVarianceLimit, bench.batch,
                           bench.size, scalar_output.data());
    RvvApplyLayerNorm(input.data(), weights.data(), bias.data(), kMultiplier,
                      kShift, kVarianceLimit, bench.batch, bench.size,
                      rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 8.0 * total, [&]() -> double {
          PortableApplyLayerNorm(input.data(), weights.data(), bias.data(),
                                 kMultiplier, kShift, kVarianceLimit,
                                 bench.batch, bench.size, scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 8.0 * total, [&]() -> double {
          RvvApplyLayerNorm(input.data(), weights.data(), bias.data(),
                            kMultiplier, kShift, kVarianceLimit, bench.batch,
                            bench.size, rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (!IntAccuracyWithinTolerance(accuracy, 2)) {
      std::cerr << "ApplyLayerNorm accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkApplySigmoid(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"sigmoid_small", 4, 64},
      {"sigmoid_mid", 8, 256},
      {"sigmoid_large", 4, 1024},
  };

  std::cout << "## ApplySigmoid<int16>\n\n";
  std::cout << "- Affects: quantized recurrent gate activation helper paths.\n\n";
  PrintIntHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total * 8LL);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableApplySigmoid(input.data(), bench.batch, bench.size,
                         scalar_output.data());
    RvvApplySigmoid(input.data(), bench.batch, bench.size, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          PortableApplySigmoid(input.data(), bench.batch, bench.size,
                               scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          RvvApplySigmoid(input.data(), bench.batch, bench.size,
                          rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "ApplySigmoid accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

struct TanhCase {
  const char* name;
  int integer_bits;
  int batch;
  int size;
};

void BenchmarkApplyTanh(std::mt19937* rng) {
  const TanhCase cases[] = {
      {"tanh_q0_small", 0, 4, 64},
      {"tanh_q3_mid", 3, 8, 256},
      {"tanh_q4_large", 4, 4, 1024},
  };

  std::cout << "## ApplyTanh<int16>\n\n";
  std::cout << "- Affects: quantized recurrent state activation helper paths.\n\n";
  PrintIntHeader("Bits x batch x input");

  for (const TanhCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total * 8LL);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableApplyTanh(bench.integer_bits, input.data(), bench.batch, bench.size,
                      scalar_output.data());
    RvvApplyTanh(bench.integer_bits, input.data(), bench.batch, bench.size,
                 rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          PortableApplyTanh(bench.integer_bits, input.data(), bench.batch,
                            bench.size, scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          RvvApplyTanh(bench.integer_bits, input.data(), bench.batch,
                       bench.size, rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.integer_bits << " x "
              << bench.batch << " x " << bench.size << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "ApplyTanh accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkApplyLayerNormFloat(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"lnf_small", 4, 64},
      {"lnf_mid", 4, 256},
      {"lnf_gate_like", 8, 1024},
  };
  constexpr int32_t kMultiplier = 1895840000;
  constexpr int32_t kShift = -13;

  std::cout << "## ApplyLayerNormFloat<int16>\n\n";
  std::cout << "- Affects: float-reference layer-norm helper paths used by "
               "quantized LSTM eval.\n\n";
  PrintIntHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(static_cast<int64_t>(total) * 8);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(total, rng, -1500, 1500);
    const std::vector<int16_t> weights =
        MakeRandomInt16Vector(bench.size, rng, 12000, 28000);
    const std::vector<int32_t> bias =
        MakeRandomInt32Vector(bench.size, rng, -16000000, -12000000);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableApplyLayerNormFloat(input.data(), weights.data(), kMultiplier,
                                kShift, bias.data(), bench.batch, bench.size,
                                scalar_output.data());
    RvvApplyLayerNormFloat(input.data(), weights.data(), kMultiplier, kShift,
                           bias.data(), bench.batch, bench.size,
                           rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 8.0 * total, [&]() -> double {
          PortableApplyLayerNormFloat(
              input.data(), weights.data(), kMultiplier, kShift, bias.data(),
              bench.batch, bench.size, scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 8.0 * total, [&]() -> double {
          RvvApplyLayerNormFloat(input.data(), weights.data(), kMultiplier,
                                 kShift, bias.data(), bench.batch, bench.size,
                                 rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "ApplyLayerNormFloat accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkApplySigmoidFloat(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"sigmoidf_small", 4, 64},
      {"sigmoidf_mid", 8, 256},
      {"sigmoidf_large", 4, 1024},
  };

  std::cout << "## ApplySigmoidFloat<int16>\n\n";
  std::cout << "- Affects: float-reference gate activation helper paths.\n\n";
  PrintFloatHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total * 8LL);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableApplySigmoidFloat(input.data(), bench.batch, bench.size,
                              scalar_output.data());
    RvvApplySigmoidFloat(input.data(), bench.batch, bench.size,
                         rvv_output.data());
    std::vector<float> scalar_output_f(total, 0.0f);
    std::vector<float> rvv_output_f(total, 0.0f);
    std::transform(scalar_output.begin(), scalar_output.end(),
                   scalar_output_f.begin(),
                   [](int16_t value) { return static_cast<float>(value); });
    std::transform(rvv_output.begin(), rvv_output.end(), rvv_output_f.begin(),
                   [](int16_t value) { return static_cast<float>(value); });
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output_f, rvv_output_f);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          PortableApplySigmoidFloat(input.data(), bench.batch, bench.size,
                                    scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          RvvApplySigmoidFloat(input.data(), bench.batch, bench.size,
                               rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";
    if (accuracy.max_abs_diff > 8.0f || accuracy.mean_abs_diff > 3.0f) {
      std::cerr << "ApplySigmoidFloat accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

struct FloatTanhCase {
  const char* name;
  int integer_bits;
  int batch;
  int size;
};

void BenchmarkApplyTanhFloat(std::mt19937* rng) {
  const FloatTanhCase cases[] = {
      {"tanhf_qm12_small", -12, 4, 64},
      {"tanhf_qm12_mid", -12, 8, 256},
      {"tanhf_qm15_large", -15, 4, 1024},
  };

  std::cout << "## ApplyTanhFloat<int16>\n\n";
  std::cout << "- Affects: float-reference state activation helper paths.\n\n";
  PrintIntHeader("Bits x batch x input");

  for (const FloatTanhCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total * 8LL);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableApplyTanhFloat(input.data(), bench.batch, bench.size,
                           bench.integer_bits, scalar_output.data());
    RvvApplyTanhFloat(input.data(), bench.batch, bench.size,
                      bench.integer_bits, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          PortableApplyTanhFloat(input.data(), bench.batch, bench.size,
                                 bench.integer_bits, scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          RvvApplyTanhFloat(input.data(), bench.batch, bench.size,
                            bench.integer_bits, rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.integer_bits << " x "
              << bench.batch << " x " << bench.size << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "ApplyTanhFloat accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkCwiseMulInt16(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"mul_q15_small", 4, 64},
      {"mul_q15_mid", 8, 256},
      {"mul_q15_large", 4, 1024},
  };
  constexpr int kShift = 15;

  std::cout << "## CwiseMul<int16 -> int16>\n\n";
  std::cout << "- Affects: quantized recurrent elementwise gate math.\n\n";
  PrintIntHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total);
    const std::vector<int16_t> lhs =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    const std::vector<int16_t> rhs =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableCwiseMul(lhs.data(), rhs.data(), bench.batch, bench.size, kShift,
                     scalar_output.data());
    RvvCwiseMul(lhs.data(), rhs.data(), bench.batch, bench.size, kShift,
                rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          PortableCwiseMul(lhs.data(), rhs.data(), bench.batch, bench.size,
                           kShift, scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          RvvCwiseMul(lhs.data(), rhs.data(), bench.batch, bench.size, kShift,
                      rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "CwiseMul<int16> accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkCwiseMulInt8(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"mul_q8_small", 4, 64},
      {"mul_q8_mid", 8, 256},
      {"mul_q8_large", 4, 1024},
  };
  constexpr int32_t kMultiplier = 1970324837;
  constexpr int32_t kShift = -15;
  constexpr int32_t kOutputZp = 3;

  std::cout << "## CwiseMul<int16 -> int8>\n\n";
  std::cout << "- Affects: quantized projection and low-precision elementwise "
               "paths.\n\n";
  PrintIntHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total);
    const std::vector<int16_t> lhs =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    const std::vector<int16_t> rhs =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    std::vector<int8_t> scalar_output(total, 0);
    std::vector<int8_t> rvv_output(total, 0);

    PortableCwiseMul(lhs.data(), rhs.data(), kMultiplier, kShift, bench.batch,
                     bench.size, kOutputZp, scalar_output.data());
    RvvCwiseMul(lhs.data(), rhs.data(), kMultiplier, kShift, bench.batch,
                bench.size, kOutputZp, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          PortableCwiseMul(lhs.data(), rhs.data(), kMultiplier, kShift,
                           bench.batch, bench.size, kOutputZp,
                           scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          RvvCwiseMul(lhs.data(), rhs.data(), kMultiplier, kShift, bench.batch,
                      bench.size, kOutputZp, rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "CwiseMul<int8> accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkCwiseAdd(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"add_small", 4, 64},
      {"add_mid", 8, 256},
      {"add_large", 4, 1024},
  };

  std::cout << "## CwiseAdd<int16>\n\n";
  std::cout << "- Affects: recurrent residual/additive helper paths.\n\n";
  PrintIntHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total);
    const std::vector<int16_t> lhs =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    const std::vector<int16_t> rhs =
        MakeRandomInt16Vector(total, rng, -32768, 32767);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableCwiseAdd(lhs.data(), rhs.data(), bench.batch, bench.size,
                     scalar_output.data());
    RvvCwiseAdd(lhs.data(), rhs.data(), bench.batch, bench.size,
                rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          PortableCwiseAdd(lhs.data(), rhs.data(), bench.batch, bench.size,
                           scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(total), [&]() -> double {
          RvvCwiseAdd(lhs.data(), rhs.data(), bench.batch, bench.size,
                      rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "CwiseAdd accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkTwoGateSaturatingAdd(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"two_gate_small", 4, 64},
      {"two_gate_mid", 8, 256},
      {"two_gate_large", 4, 1024},
  };
  constexpr int8_t kInputZp = 10;
  constexpr int8_t kRecurrentZp = -5;
  constexpr int32_t kInputMultiplier = 1347771520;
  constexpr int32_t kInputShift = -7;
  constexpr int32_t kRecurrentMultiplier = 1047577121;
  constexpr int32_t kRecurrentShift = -6;

  std::cout << "## TwoGateSaturatingAdd<int8 -> int16>\n\n";
  std::cout << "- Affects: quantized LSTM gate merge helper paths.\n\n";
  PrintIntHeader("Batch x cell");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(static_cast<int64_t>(total) * 4);
    const std::vector<int8_t> input =
        MakeRandomInt8Vector(total, rng, -120, 120);
    const std::vector<int8_t> recurrent =
        MakeRandomInt8Vector(total, rng, -120, 120);
    std::vector<int16_t> scalar_output(total, 0);
    std::vector<int16_t> rvv_output(total, 0);

    PortableTwoGateSaturatingAdd(
        input.data(), kInputZp, recurrent.data(), kRecurrentZp,
        kInputMultiplier, kInputShift, kRecurrentMultiplier, kRecurrentShift,
        bench.batch, bench.size, scalar_output.data());
    RvvTwoGateSaturatingAdd(
        input.data(), kInputZp, recurrent.data(), kRecurrentZp,
        kInputMultiplier, kInputShift, kRecurrentMultiplier, kRecurrentShift,
        bench.batch, bench.size, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 4.0 * total, [&]() -> double {
          PortableTwoGateSaturatingAdd(
              input.data(), kInputZp, recurrent.data(), kRecurrentZp,
              kInputMultiplier, kInputShift, kRecurrentMultiplier,
              kRecurrentShift, bench.batch, bench.size, scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 4.0 * total, [&]() -> double {
          RvvTwoGateSaturatingAdd(
              input.data(), kInputZp, recurrent.data(), kRecurrentZp,
              kInputMultiplier, kInputShift, kRecurrentMultiplier,
              kRecurrentShift, bench.batch, bench.size, rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "TwoGateSaturatingAdd accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkCwiseClippingFloat(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"clip_f32_64", 64},
      {"clip_f32_1024", 1024},
      {"clip_f32_4096", 4096},
  };
  constexpr float kClip = 2.0f;

  std::cout << "## CwiseClipping<float>\n\n";
  std::cout << "- Affects: activation clamp paths in float recurrent/helper "
               "code.\n\n";
  PrintFloatHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<float> input =
        MakeRandomFloatVector(bench.size, rng, -6.0f, 6.0f);
    std::vector<float> scalar_output = input;
    std::vector<float> rvv_output = input;

    PortableCwiseClipping(scalar_output.data(), bench.size, kClip);
    RvvCwiseClipping(rvv_output.data(), bench.size, kClip);
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       PortableCwiseClipping(scalar_output.data(), bench.size,
                                             kClip);
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       RvvCwiseClipping(rvv_output.data(), bench.size, kClip);
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
    if (accuracy.max_abs_diff > 0.0f) {
      std::cerr << "CwiseClipping<float> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkCwiseClippingInt16(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"clip_i16_64", 64},
      {"clip_i16_1024", 1024},
      {"clip_i16_4096", 4096},
  };
  constexpr int16_t kClip = 300;

  std::cout << "## CwiseClipping<int16>\n\n";
  std::cout << "- Affects: quantized activation clamp helper paths.\n\n";
  PrintIntHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(bench.size, rng, -1200, 1200);
    std::vector<int16_t> scalar_output = input;
    std::vector<int16_t> rvv_output = input;

    PortableCwiseClipping(scalar_output.data(), bench.size, kClip);
    RvvCwiseClipping(rvv_output.data(), bench.size, kClip);
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       PortableCwiseClipping(scalar_output.data(), bench.size,
                                             kClip);
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       RvvCwiseClipping(rvv_output.data(), bench.size, kClip);
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "CwiseClipping<int16> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkCwiseClippingInt8(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"clip_i8_64", 64},
      {"clip_i8_1024", 1024},
      {"clip_i8_4096", 4096},
  };
  constexpr int8_t kClip = 32;

  std::cout << "## CwiseClipping<int8>\n\n";
  std::cout << "- Affects: low-precision activation clamp helper paths.\n\n";
  PrintIntHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<int8_t> input =
        MakeRandomInt8Vector(bench.size, rng, -100, 100);
    std::vector<int8_t> scalar_output = input;
    std::vector<int8_t> rvv_output = input;

    PortableCwiseClipping(scalar_output.data(), bench.size, kClip);
    RvvCwiseClipping(rvv_output.data(), bench.size, kClip);
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       PortableCwiseClipping(scalar_output.data(), bench.size,
                                             kClip);
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       RvvCwiseClipping(rvv_output.data(), bench.size, kClip);
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "CwiseClipping<int8> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkVectorBatchVectorCwiseProductAccumulate(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"vbv_cwise_small", 4, 64},
      {"vbv_cwise_mid", 8, 256},
      {"vbv_cwise_large", 4, 1024},
  };
  constexpr int32_t kMultiplier = 1073741824;
  constexpr int32_t kShift = -1;

  std::cout << "## VectorBatchVectorCwiseProductAccumulate<int16>\n\n";
  std::cout << "- Affects: quantized recurrent gate/state accumulation helper "
               "paths.\n\n";
  PrintIntHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(total * 2LL);
    const std::vector<int16_t> vector =
        MakeRandomInt16Vector(bench.size, rng, -256, 256);
    const std::vector<int16_t> batch_vector =
        MakeRandomInt16Vector(total, rng, -256, 256);
    const std::vector<int16_t> base_result =
        MakeRandomInt16Vector(total, rng, -256, 256);
    std::vector<int16_t> scalar_output = base_result;
    std::vector<int16_t> rvv_output = base_result;

    PortableVectorBatchVectorCwiseProductAccumulate(
        vector.data(), bench.size, batch_vector.data(), bench.batch,
        kMultiplier, kShift, scalar_output.data());
    RvvVectorBatchVectorCwiseProductAccumulate(
        vector.data(), bench.size, batch_vector.data(), bench.batch,
        kMultiplier, kShift, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * total, [&]() -> double {
          std::copy(base_result.begin(), base_result.end(),
                    scalar_output.begin());
          PortableVectorBatchVectorCwiseProductAccumulate(
              vector.data(), bench.size, batch_vector.data(), bench.batch,
              kMultiplier, kShift, scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * total, [&]() -> double {
          std::copy(base_result.begin(), base_result.end(), rvv_output.begin());
          RvvVectorBatchVectorCwiseProductAccumulate(
              vector.data(), bench.size, batch_vector.data(), bench.batch,
              kMultiplier, kShift, rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";
    if (accuracy.mismatches != 0) {
      std::cerr << "VectorBatchVectorCwiseProductAccumulate accuracy check "
                   "failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkSub1VectorFloat(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"sub1_f32_64", 64},
      {"sub1_f32_1024", 1024},
      {"sub1_f32_4096", 4096},
  };

  std::cout << "## Sub1Vector<float>\n\n";
  std::cout << "- Affects: float recurrent helper post-processing paths.\n\n";
  PrintFloatHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<float> input =
        MakeRandomFloatVector(bench.size, rng, -2.0f, 2.0f);
    std::vector<float> scalar_output(bench.size, 0.0f);
    std::vector<float> rvv_output(bench.size, 0.0f);

    PortableSub1Vector(input.data(), bench.size, scalar_output.data());
    RvvSub1Vector(input.data(), bench.size, rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       PortableSub1Vector(input.data(), bench.size,
                                          scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       RvvSub1Vector(input.data(), bench.size,
                                     rvv_output.data());
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
    if (accuracy.max_abs_diff > 1.0e-6f) {
      std::cerr << "Sub1Vector<float> accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkSub1VectorInt16(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"sub1_i16_64", 64},
      {"sub1_i16_1024", 1024},
      {"sub1_i16_4096", 4096},
  };

  std::cout << "## Sub1Vector<int16>\n\n";
  std::cout << "- Affects: quantized recurrent helper post-processing paths.\n\n";
  PrintIntHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<int16_t> input =
        MakeRandomInt16Vector(bench.size, rng, -32768, 32767);
    std::vector<int16_t> scalar_output(bench.size, 0);
    std::vector<int16_t> rvv_output(bench.size, 0);

    PortableSub1Vector(input.data(), bench.size, scalar_output.data());
    RvvSub1Vector(input.data(), bench.size, rvv_output.data());
    const IntAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       PortableSub1Vector(input.data(), bench.size,
                                          scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       RvvSub1Vector(input.data(), bench.size,
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
    if (accuracy.mismatches != 0) {
      std::cerr << "Sub1Vector<int16> accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkVectorScalarMultiply(std::mt19937* rng) {
  const VectorCase cases[] = {
      {"scale_i8_64", 64},
      {"scale_i8_1024", 1024},
      {"scale_i8_4096", 4096},
  };
  constexpr float kScale = 0.0625f;

  std::cout << "## VectorScalarMultiply<int8 -> float>\n\n";
  std::cout << "- Affects: hybrid dequant-style helper paths.\n\n";
  PrintFloatHeader("Vector size");

  for (const VectorCase& bench : cases) {
    const int iterations = ChooseIterations(bench.size);
    const std::vector<int8_t> input =
        MakeRandomInt8Vector(bench.size, rng, -120, 120);
    std::vector<float> scalar_output(bench.size, 0.0f);
    std::vector<float> rvv_output(bench.size, 0.0f);

    PortableVectorScalarMultiply(input.data(), bench.size, kScale,
                                 scalar_output.data());
    RvvVectorScalarMultiply(input.data(), bench.size, kScale,
                            rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 2.0 * bench.size, [&]() -> double {
          PortableVectorScalarMultiply(input.data(), bench.size, kScale,
                                       scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 2.0 * bench.size, [&]() -> double {
          RvvVectorScalarMultiply(input.data(), bench.size, kScale,
                                  rvv_output.data());
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
    if (accuracy.max_abs_diff > 1.0e-6f) {
      std::cerr << "VectorScalarMultiply accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkMeanStddevNormalization(std::mt19937* rng) {
  const BatchVectorCase cases[] = {
      {"norm_1x64", 1, 64},
      {"norm_4x256", 4, 256},
      {"norm_8x1024", 8, 1024},
  };

  std::cout << "## MeanStddevNormalization<float>\n\n";
  std::cout << "- Affects: float recurrent normalization helper paths.\n\n";
  PrintFloatHeader("Batch x input");

  for (const BatchVectorCase& bench : cases) {
    const int total = bench.batch * bench.size;
    const int iterations = ChooseIterations(static_cast<int64_t>(total) * 4);
    const std::vector<float> input =
        MakeRandomFloatVector(total, rng, -200.0f, 200.0f);
    std::vector<float> scalar_output(total, 0.0f);
    std::vector<float> rvv_output(total, 0.0f);

    PortableMeanStddevNormalization(input.data(), scalar_output.data(),
                                    bench.size, bench.batch);
    RvvMeanStddevNormalization(input.data(), rvv_output.data(), bench.size,
                               bench.batch);
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, 4.0 * total, [&]() -> double {
          PortableMeanStddevNormalization(input.data(), scalar_output.data(),
                                          bench.size, bench.batch);
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, 4.0 * total, [&]() -> double {
          RvvMeanStddevNormalization(input.data(), rvv_output.data(),
                                     bench.size, bench.batch);
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << bench.batch << " x "
              << bench.size << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";
    if (!FloatAccuracyWithinTolerance(accuracy)) {
      std::cerr << "MeanStddevNormalization accuracy check failed for "
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
  std::cout << "- Covered helper families: zero-check, dot/reduction, dense "
               "matvec, sparse matvec, hybrid quantization, recurrent "
               "accumulate, layer-norm, cwise math, clipping, sub1, "
               "normalization, dequant-scale.\n\n";

  std::mt19937 rng(20260506);
  BenchmarkIsZeroVectorFloat(&rng);
  BenchmarkIsZeroVectorInt8(&rng);
  BenchmarkFloatDotProduct(&rng);
  BenchmarkBatchVectorDotProductInt16(&rng);
  BenchmarkFloatReduction(&rng);
  BenchmarkInt8Reduction(&rng);
  BenchmarkInt32Reduction(&rng);
  BenchmarkMatrixScalarMultiply(&rng);
  BenchmarkFloatMatVec(&rng);
  BenchmarkInt8MatVec(&rng);
  BenchmarkInt8MatVecWithOffsets(&rng);
  BenchmarkGateMatVecInt16(&rng);
  BenchmarkGateMatVecInt8(&rng);
  BenchmarkGateMatVecNoAccumulateInt8(&rng);
  BenchmarkProjectionMatVecInt8(&rng);
  BenchmarkSparseFloat1x4(&rng);
  BenchmarkSparseFloatLedger(&rng);
  BenchmarkSparseInt8Ledger(&rng);
  BenchmarkSparseInt8Output(&rng);
  BenchmarkSymmetricQuantizeFloats(&rng);
  BenchmarkAsymmetricQuantizeFloats(&rng);
  BenchmarkApplyLayerNorm(&rng);
  BenchmarkApplySigmoid(&rng);
  BenchmarkApplyTanh(&rng);
  BenchmarkApplyLayerNormFloat(&rng);
  BenchmarkApplySigmoidFloat(&rng);
  BenchmarkApplyTanhFloat(&rng);
  BenchmarkCwiseMulInt16(&rng);
  BenchmarkCwiseMulInt8(&rng);
  BenchmarkCwiseAdd(&rng);
  BenchmarkTwoGateSaturatingAdd(&rng);
  BenchmarkCwiseClippingFloat(&rng);
  BenchmarkCwiseClippingInt16(&rng);
  BenchmarkCwiseClippingInt8(&rng);
  BenchmarkVectorBatchVectorCwiseProductAccumulate(&rng);
  BenchmarkSub1VectorFloat(&rng);
  BenchmarkSub1VectorInt16(&rng);
  BenchmarkVectorScalarMultiply(&rng);
  BenchmarkMeanStddevNormalization(&rng);
  return 0;
}

}  // namespace tensor_utils
}  // namespace tflite

int main() { return tflite::tensor_utils::Main(); }
