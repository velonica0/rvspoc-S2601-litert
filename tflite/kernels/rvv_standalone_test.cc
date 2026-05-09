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
// Standalone accuracy test + benchmark for RVV kernels.
// Can be compiled directly with the tensorflow-lite library without the
// full test framework:
//   g++ -std=c++17 -O2 -march=rv64gcv -I<repo_root> \
//       rvv_standalone_test.cc -L<build_dir> -ltensorflow-lite -lpthread -ldl \
//       -o rvv_standalone_test
//
// Or via CMake after building the library:
//   add_executable(rvv_standalone_test rvv_standalone_test.cc)
//   target_link_libraries(rvv_standalone_test tensorflow-lite)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/sub.h"
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/types.h"

using namespace tflite;

static int g_pass = 0, g_fail = 0;

#define CHECK_TRUE(cond, msg)                                      \
  do {                                                             \
    if (!(cond)) {                                                 \
      fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
      g_fail++;                                                    \
    } else {                                                       \
      g_pass++;                                                    \
    }                                                              \
  } while (0)

constexpr int kSeed = 42;

std::vector<int8_t> RandI8(int n, std::mt19937& rng) {
  std::uniform_int_distribution<int> d(-128, 127);
  std::vector<int8_t> v(n);
  for (auto& x : v) x = static_cast<int8_t>(d(rng));
  return v;
}

std::vector<float> RandF32(int n, std::mt19937& rng) {
  std::uniform_real_distribution<float> d(-10.0f, 10.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// ---- Float Add ----
void TestFloatAdd(int size) {
  std::mt19937 rng(kSeed);
  auto in1 = RandF32(size, rng);
  auto in2 = RandF32(size, rng);
  std::vector<float> ref(size), opt(size);

  ArithmeticParams p = {};
  p.float_activation_min = -100.0f;
  p.float_activation_max = 100.0f;

  for (int i = 0; i < size; i++) {
    float x = in1[i] + in2[i];
    ref[i] = std::min(p.float_activation_max,
                      std::max(p.float_activation_min, x));
  }
  optimized_ops::AddElementwise(size, p, in1.data(), in2.data(), opt.data());

  float max_err = 0;
  for (int i = 0; i < size; i++)
    max_err = std::max(max_err, std::fabs(ref[i] - opt[i]));

  char msg[128];
  snprintf(msg, sizeof(msg), "FloatAdd(size=%d) max_err=%.2e <= 1e-5", size,
           max_err);
  CHECK_TRUE(max_err <= 1e-5f, msg);
}

// ---- Float Mul ----
void TestFloatMul(int size) {
  std::mt19937 rng(kSeed + 1);
  auto in1 = RandF32(size, rng);
  auto in2 = RandF32(size, rng);
  std::vector<float> ref(size), opt(size);

  ArithmeticParams p = {};
  p.float_activation_min = -1000.0f;
  p.float_activation_max = 1000.0f;

  for (int i = 0; i < size; i++) {
    float x = in1[i] * in2[i];
    ref[i] = std::min(p.float_activation_max,
                      std::max(p.float_activation_min, x));
  }
  optimized_ops::MulElementwise(size, p, in1.data(), in2.data(), opt.data());

  float max_err = 0;
  for (int i = 0; i < size; i++)
    max_err = std::max(max_err, std::fabs(ref[i] - opt[i]));

  char msg[128];
  snprintf(msg, sizeof(msg), "FloatMul(size=%d) max_err=%.2e <= 1e-5", size,
           max_err);
  CHECK_TRUE(max_err <= 1e-5f, msg);
}

// ---- Float HardSwish ----
void TestFloatHardSwish(int size) {
  std::mt19937 rng(kSeed + 2);
  auto in = RandF32(size, rng);
  std::vector<float> ref(size), opt(size);

  for (int i = 0; i < size; i++)
    ref[i] = in[i] * std::min(6.0f, std::max(0.0f, in[i] + 3.0f)) / 6.0f;

  RuntimeShape shape({1, 1, 1, size});
  optimized_ops::HardSwish(shape, in.data(), shape, opt.data());

  float max_err = 0;
  for (int i = 0; i < size; i++)
    max_err = std::max(max_err, std::fabs(ref[i] - opt[i]));

  char msg[128];
  snprintf(msg, sizeof(msg), "FloatHardSwish(size=%d) max_err=%.2e <= 1e-5",
           size, max_err);
  CHECK_TRUE(max_err <= 1e-5f, msg);
}

// ---- Int8 Add ----
void TestInt8Add(int size) {
  std::mt19937 rng(kSeed + 3);
  auto in1 = RandI8(size, rng);
  auto in2 = RandI8(size, rng);
  std::vector<int8_t> opt(size);

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

  std::vector<int8_t> ref(size);
  for (int i = 0; i < size; i++) {
    int32_t v1 = p.input1_offset + in1[i];
    int32_t v2 = p.input2_offset + in2[i];
    int32_t s1 = v1 * (1 << p.left_shift);
    int32_t s2 = v2 * (1 << p.left_shift);
    int32_t sc1 = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        s1, p.input1_multiplier, p.input1_shift);
    int32_t sc2 = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        s2, p.input2_multiplier, p.input2_shift);
    int32_t raw = MultiplyByQuantizedMultiplierSmallerThanOneExp(
                      sc1 + sc2, p.output_multiplier, p.output_shift) +
                  p.output_offset;
    ref[i] = static_cast<int8_t>(
        std::min(p.quantized_activation_max,
                 std::max(p.quantized_activation_min, raw)));
  }

  optimized_integer_ops::AddElementwiseInt8(size, p, in1.data(), in2.data(),
                                            opt.data());

  int max_diff = 0;
  for (int i = 0; i < size; i++)
    max_diff =
        std::max(max_diff, std::abs(static_cast<int>(ref[i]) -
                                    static_cast<int>(opt[i])));

  char msg[128];
  snprintf(msg, sizeof(msg), "Int8Add(size=%d) max_diff=%d <= 1", size,
           max_diff);
  CHECK_TRUE(max_diff <= 1, msg);
}

// ---- Int8 Mul ----
void TestInt8Mul(int size) {
  std::mt19937 rng(kSeed + 4);
  auto in1 = RandI8(size, rng);
  auto in2 = RandI8(size, rng);
  std::vector<int8_t> opt(size);

  ArithmeticParams p = {};
  p.input1_offset = 2;
  p.input2_offset = -3;
  p.output_offset = 1;
  p.output_multiplier = 1073741824;
  p.output_shift = -4;
  p.quantized_activation_min = -128;
  p.quantized_activation_max = 127;

  std::vector<int8_t> ref(size);
  for (int i = 0; i < size; i++) {
    int32_t v1 = p.input1_offset + in1[i];
    int32_t v2 = p.input2_offset + in2[i];
    int32_t raw = p.output_offset +
                  MultiplyByQuantizedMultiplier(v1 * v2, p.output_multiplier,
                                                p.output_shift);
    ref[i] = static_cast<int8_t>(
        std::min(p.quantized_activation_max,
                 std::max(p.quantized_activation_min, raw)));
  }

  optimized_integer_ops::MulElementwise(size, p, in1.data(), in2.data(),
                                        opt.data());

  int max_diff = 0;
  for (int i = 0; i < size; i++)
    max_diff =
        std::max(max_diff, std::abs(static_cast<int>(ref[i]) -
                                    static_cast<int>(opt[i])));

  char msg[128];
  snprintf(msg, sizeof(msg), "Int8Mul(size=%d) max_diff=%d <= 1", size,
           max_diff);
  CHECK_TRUE(max_diff <= 1, msg);
}

// ---- Int8 MaxPool ----
void TestInt8MaxPool() {
  const int B = 1, H = 4, W = 4, D = 32;
  std::mt19937 rng(kSeed + 5);
  auto input = RandI8(B * H * W * D, rng);

  PoolParams pp;
  pp.stride_height = 2;
  pp.stride_width = 2;
  pp.filter_height = 2;
  pp.filter_width = 2;
  pp.padding_values.height = 0;
  pp.padding_values.width = 0;
  pp.quantized_activation_min = -128;
  pp.quantized_activation_max = 127;

  RuntimeShape in_shape({B, H, W, D});
  RuntimeShape out_shape({B, 2, 2, D});
  int out_size = B * 2 * 2 * D;

  std::vector<int8_t> ref(out_size, -128), opt(out_size, -128);
  // Compute reference with scalar inner loop: same logic as optimized but
  // guaranteed no SIMD.
  for (int oy = 0; oy < 2; oy++) {
    for (int ox = 0; ox < 2; ox++) {
      for (int d = 0; d < D; d++) {
        int8_t mx = -128;
        for (int fy = 0; fy < 2; fy++) {
          for (int fx = 0; fx < 2; fx++) {
            int iy = oy * 2 + fy, ix = ox * 2 + fx;
            mx = std::max(mx, input[(iy * W + ix) * D + d]);
          }
        }
        ref[(oy * 2 + ox) * D + d] = mx;
      }
    }
  }
  optimized_integer_ops::MaxPool(pp, in_shape, input.data(), out_shape,
                                 opt.data());

  int max_diff = 0;
  for (int i = 0; i < out_size; i++)
    max_diff =
        std::max(max_diff, std::abs(static_cast<int>(ref[i]) -
                                    static_cast<int>(opt[i])));

  char msg[128];
  snprintf(msg, sizeof(msg), "Int8MaxPool(D=%d) max_diff=%d == 0", D,
           max_diff);
  CHECK_TRUE(max_diff == 0, msg);
}

// ---- Benchmark helper ----
template <typename Fn>
double BenchmarkMs(Fn fn, int iters) {
  // Warmup
  for (int i = 0; i < 3; i++) fn();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) fn();
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

void RunBenchmarks() {
  printf("\n=== Benchmarks ===\n");

  const int N = 16384;
  const int iters = 1000;
  std::mt19937 rng(kSeed);

  // Float Add
  {
    auto in1 = RandF32(N, rng);
    auto in2 = RandF32(N, rng);
    std::vector<float> out(N);
    ArithmeticParams p = {};
    p.float_activation_min = -100.0f;
    p.float_activation_max = 100.0f;
    double ms = BenchmarkMs(
        [&]() {
          optimized_ops::AddElementwise(N, p, in1.data(), in2.data(),
                                        out.data());
        },
        iters);
    printf("FloatAdd(%d):   %.3f ms/iter  (%.1f Mops/s)\n", N, ms,
           N / ms / 1000.0);
  }

  // Float Mul
  {
    auto in1 = RandF32(N, rng);
    auto in2 = RandF32(N, rng);
    std::vector<float> out(N);
    ArithmeticParams p = {};
    p.float_activation_min = -1000.0f;
    p.float_activation_max = 1000.0f;
    double ms = BenchmarkMs(
        [&]() {
          optimized_ops::MulElementwise(N, p, in1.data(), in2.data(),
                                        out.data());
        },
        iters);
    printf("FloatMul(%d):   %.3f ms/iter  (%.1f Mops/s)\n", N, ms,
           N / ms / 1000.0);
  }

  // Float HardSwish
  {
    auto in = RandF32(N, rng);
    std::vector<float> out(N);
    RuntimeShape shape({1, 1, 1, N});
    double ms = BenchmarkMs(
        [&]() {
          optimized_ops::HardSwish(shape, in.data(), shape, out.data());
        },
        iters);
    printf("FloatHardSwish(%d): %.3f ms/iter  (%.1f Mops/s)\n", N, ms,
           N / ms / 1000.0);
  }

  // Int8 Add
  {
    auto in1 = RandI8(N, rng);
    auto in2 = RandI8(N, rng);
    std::vector<int8_t> out(N);
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
    double ms = BenchmarkMs(
        [&]() {
          optimized_integer_ops::AddElementwiseInt8(N, p, in1.data(),
                                                    in2.data(), out.data());
        },
        iters);
    printf("Int8Add(%d):    %.3f ms/iter  (%.1f Mops/s)\n", N, ms,
           N / ms / 1000.0);
  }

  // Int8 Mul
  {
    auto in1 = RandI8(N, rng);
    auto in2 = RandI8(N, rng);
    std::vector<int8_t> out(N);
    ArithmeticParams p = {};
    p.input1_offset = 2;
    p.input2_offset = -3;
    p.output_offset = 1;
    p.output_multiplier = 1073741824;
    p.output_shift = -4;
    p.quantized_activation_min = -128;
    p.quantized_activation_max = 127;
    double ms = BenchmarkMs(
        [&]() {
          optimized_integer_ops::MulElementwise(N, p, in1.data(), in2.data(),
                                                out.data());
        },
        iters);
    printf("Int8Mul(%d):    %.3f ms/iter  (%.1f Mops/s)\n", N, ms,
           N / ms / 1000.0);
  }

  // Int8 MaxPool
  {
    const int D = 128, IH = 8, IW = 8;
    auto input = RandI8(IH * IW * D, rng);
    std::vector<int8_t> output(4 * 4 * D);
    PoolParams pp;
    pp.stride_height = 2;
    pp.stride_width = 2;
    pp.filter_height = 2;
    pp.filter_width = 2;
    pp.padding_values.height = 0;
    pp.padding_values.width = 0;
    pp.quantized_activation_min = -128;
    pp.quantized_activation_max = 127;
    RuntimeShape in_shape({1, IH, IW, D});
    RuntimeShape out_shape({1, 4, 4, D});
    double ms = BenchmarkMs(
        [&]() {
          optimized_integer_ops::MaxPool(pp, in_shape, input.data(), out_shape,
                                         output.data());
        },
        iters);
    printf("Int8MaxPool(%dx%dx%d): %.3f ms/iter\n", IH, IW, D, ms);
  }
}

int main() {
  printf("=== RVV Accuracy Tests ===\n");
#ifdef USE_RVV
  printf("USE_RVV is defined — RVV code paths active\n");
#else
  printf("USE_RVV not defined — scalar fallback paths active\n");
#endif

  int sizes[] = {1, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100,
                 128, 255, 256, 512, 1000, 1023, 1024};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int i = 0; i < nsizes; i++) TestFloatAdd(sizes[i]);
  for (int i = 0; i < nsizes; i++) TestFloatMul(sizes[i]);
  for (int i = 0; i < nsizes; i++) TestFloatHardSwish(sizes[i]);
  for (int i = 0; i < nsizes; i++) TestInt8Add(sizes[i]);
  for (int i = 0; i < nsizes; i++) TestInt8Mul(sizes[i]);
  TestInt8MaxPool();

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

  RunBenchmarks();

  return g_fail > 0 ? 1 : 0;
}
