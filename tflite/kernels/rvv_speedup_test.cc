/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"). */
// RVV vs Scalar speedup comparison.
// Measures both the RVV-optimized path and a pure scalar reference loop.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#ifdef USE_RVV
#include "tflite/kernels/internal/optimized/rvv_tensor_utils.h"
#else
#include "tflite/kernels/internal/optimized/neon_tensor_utils.h"
#endif
#include "tflite/kernels/internal/reference/portable_tensor_utils_impl.h"
#include "tflite/kernels/internal/types.h"

using namespace tflite;

static constexpr int kSeed = 42;

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

template <typename Fn>
double BenchMs(Fn fn, int iters) {
  for (int i = 0; i < 5; i++) fn();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) fn();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// ============================================================
// Scalar reference implementations (no SIMD, matches portable)
// ============================================================

void ScalarFloatAdd(int size, const float* in1, const float* in2, float* out,
                    float act_min, float act_max) {
  for (int i = 0; i < size; i++) {
    float x = in1[i] + in2[i];
    out[i] = std::min(act_max, std::max(act_min, x));
  }
}

void ScalarFloatMul(int size, const float* in1, const float* in2, float* out,
                    float act_min, float act_max) {
  for (int i = 0; i < size; i++) {
    float x = in1[i] * in2[i];
    out[i] = std::min(act_max, std::max(act_min, x));
  }
}

void ScalarFloatHardSwish(int size, const float* in, float* out) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i] * std::min(6.0f, std::max(0.0f, in[i] + 3.0f)) / 6.0f;
  }
}

void ScalarInt8Add(int size, const ArithmeticParams& p, const int8_t* in1,
                   const int8_t* in2, int8_t* out) {
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
    out[i] = static_cast<int8_t>(std::min(p.quantized_activation_max,
                                          std::max(p.quantized_activation_min, raw)));
  }
}

void ScalarInt8Mul(int size, const ArithmeticParams& p, const int8_t* in1,
                   const int8_t* in2, int8_t* out) {
  for (int i = 0; i < size; i++) {
    int32_t v1 = p.input1_offset + in1[i];
    int32_t v2 = p.input2_offset + in2[i];
    int32_t raw = p.output_offset +
                  MultiplyByQuantizedMultiplier(v1 * v2, p.output_multiplier,
                                                p.output_shift);
    out[i] = static_cast<int8_t>(std::min(p.quantized_activation_max,
                                          std::max(p.quantized_activation_min, raw)));
  }
}

void ScalarInt8MaxPool(const int8_t* input, int8_t* output, int IH, int IW,
                       int D, int stride) {
  int OH = IH / stride, OW = IW / stride;
  for (int oy = 0; oy < OH; oy++) {
    for (int ox = 0; ox < OW; ox++) {
      for (int d = 0; d < D; d++) {
        int8_t mx = -128;
        for (int fy = 0; fy < stride; fy++)
          for (int fx = 0; fx < stride; fx++) {
            int iy = oy * stride + fy, ix = ox * stride + fx;
            mx = std::max(mx, input[(iy * IW + ix) * D + d]);
          }
        output[(oy * OW + ox) * D + d] = mx;
      }
    }
  }
}

float ScalarDotProduct(const float* v1, const float* v2, int size) {
  float sum = 0;
  for (int i = 0; i < size; i++) sum += v1[i] * v2[i];
  return sum;
}

void ScalarSub1Vector(const float* in, int size, float* out) {
  for (int i = 0; i < size; i++) out[i] = 1.0f - in[i];
}

void ScalarReductionSum(const float* in, float* out, int out_size,
                        int red_size) {
  for (int o = 0; o < out_size; o++) {
    float s = 0;
    for (int r = 0; r < red_size; r++) s += in[r];
    out[o] = s;
    in += red_size;
  }
}

void ScalarMeanStddevNorm(const float* in, float* out, int v_size,
                          int n_batch) {
  for (int b = 0; b < n_batch; b++) {
    float sum = 0;
    for (int i = 0; i < v_size; i++) sum += in[i];
    float mean = sum / v_size;
    float ssd = 0;
    for (int i = 0; i < v_size; i++) {
      float d = in[i] - mean;
      ssd += d * d;
    }
    float inv = 1.0f / std::sqrt(ssd / v_size + 1e-8f);
    for (int i = 0; i < v_size; i++) out[i] = (in[i] - mean) * inv;
    in += v_size;
    out += v_size;
  }
}

void ScalarCwiseClipF32(float* v, int size, float clip) {
  for (int i = 0; i < size; i++)
    v[i] = std::min(clip, std::max(-clip, v[i]));
}

bool ScalarIsZeroF32(const float* v, int size) {
  for (int i = 0; i < size; i++)
    if (v[i] != 0.0f) return false;
  return true;
}

void ScalarVectorScalarMul(const int8_t* v, int size, float scale, float* out) {
  for (int i = 0; i < size; i++) out[i] = v[i] * scale;
}

void ScalarMatVecF32(const float* mat, int rows, int cols, const float* vec,
                     float* result) {
  for (int r = 0; r < rows; r++) {
    float sum = 0;
    for (int c = 0; c < cols; c++) sum += mat[r * cols + c] * vec[c];
    result[r] += sum;
  }
}

// ============================================================

void PrintRow(const char* name, double scalar_ms, double rvv_ms) {
  double speedup = scalar_ms / rvv_ms;
  printf("  %-32s  scalar=%7.3f ms  rvv=%7.3f ms  speedup=%.2fx\n", name,
         scalar_ms, rvv_ms, speedup);
}

int main() {
  printf("=== RVV vs Scalar Speedup Comparison ===\n");
#ifdef USE_RVV
  printf("USE_RVV is defined — RVV code paths active\n\n");
#else
  printf("USE_RVV not defined — both paths are scalar\n\n");
#endif

  const int N = 16384;
  const int iters = 2000;
  std::mt19937 rng(kSeed);

  // Float Add
  {
    auto in1 = RandF32(N, rng), in2 = RandF32(N, rng);
    std::vector<float> out(N);
    ArithmeticParams p = {};
    p.float_activation_min = -100.0f;
    p.float_activation_max = 100.0f;
    double s = BenchMs([&]() { ScalarFloatAdd(N, in1.data(), in2.data(), out.data(), -100, 100); }, iters);
    double r = BenchMs([&]() { optimized_ops::AddElementwise(N, p, in1.data(), in2.data(), out.data()); }, iters);
    PrintRow("FloatAdd(16384)", s, r);
  }

  // Float Mul
  {
    auto in1 = RandF32(N, rng), in2 = RandF32(N, rng);
    std::vector<float> out(N);
    ArithmeticParams p = {};
    p.float_activation_min = -1000.0f;
    p.float_activation_max = 1000.0f;
    double s = BenchMs([&]() { ScalarFloatMul(N, in1.data(), in2.data(), out.data(), -1000, 1000); }, iters);
    double r = BenchMs([&]() { optimized_ops::MulElementwise(N, p, in1.data(), in2.data(), out.data()); }, iters);
    PrintRow("FloatMul(16384)", s, r);
  }

  // Float HardSwish
  {
    auto in = RandF32(N, rng);
    std::vector<float> out(N);
    RuntimeShape shape({1, 1, 1, N});
    double s = BenchMs([&]() { ScalarFloatHardSwish(N, in.data(), out.data()); }, iters);
    double r = BenchMs([&]() { optimized_ops::HardSwish(shape, in.data(), shape, out.data()); }, iters);
    PrintRow("FloatHardSwish(16384)", s, r);
  }

  // Int8 Add
  {
    auto in1 = RandI8(N, rng), in2 = RandI8(N, rng);
    std::vector<int8_t> out(N);
    ArithmeticParams p = {};
    p.input1_offset = 3; p.input2_offset = -5; p.output_offset = 2;
    p.left_shift = 5;
    p.input1_multiplier = 1073741824; p.input1_shift = -1;
    p.input2_multiplier = 1073741824; p.input2_shift = -2;
    p.output_multiplier = 1073741824; p.output_shift = -3;
    p.quantized_activation_min = -128; p.quantized_activation_max = 127;
    double s = BenchMs([&]() { ScalarInt8Add(N, p, in1.data(), in2.data(), out.data()); }, iters);
    double r = BenchMs([&]() { optimized_integer_ops::AddElementwiseInt8(N, p, in1.data(), in2.data(), out.data()); }, iters);
    PrintRow("Int8Add(16384)", s, r);
  }

  // Int8 Mul
  {
    auto in1 = RandI8(N, rng), in2 = RandI8(N, rng);
    std::vector<int8_t> out(N);
    ArithmeticParams p = {};
    p.input1_offset = 2; p.input2_offset = -3; p.output_offset = 1;
    p.output_multiplier = 1073741824; p.output_shift = -4;
    p.quantized_activation_min = -128; p.quantized_activation_max = 127;
    double s = BenchMs([&]() { ScalarInt8Mul(N, p, in1.data(), in2.data(), out.data()); }, iters);
    double r = BenchMs([&]() { optimized_integer_ops::MulElementwise(N, p, in1.data(), in2.data(), out.data()); }, iters);
    PrintRow("Int8Mul(16384)", s, r);
  }

  // Int8 MaxPool
  {
    const int D = 128, IH = 8, IW = 8;
    auto input = RandI8(IH * IW * D, rng);
    std::vector<int8_t> out(4 * 4 * D);
    PoolParams pp; pp.stride_height = 2; pp.stride_width = 2;
    pp.filter_height = 2; pp.filter_width = 2;
    pp.padding_values.height = 0; pp.padding_values.width = 0;
    pp.quantized_activation_min = -128; pp.quantized_activation_max = 127;
    RuntimeShape is({1, IH, IW, D}), os({1, 4, 4, D});
    double s = BenchMs([&]() { ScalarInt8MaxPool(input.data(), out.data(), IH, IW, D, 2); }, iters);
    double r = BenchMs([&]() { optimized_integer_ops::MaxPool(pp, is, input.data(), os, out.data()); }, iters);
    PrintRow("Int8MaxPool(8x8x128)", s, r);
  }

  // Float MatVec (tensor_utils)
  {
    const int rows = 256, cols = 256;
    auto mat = RandF32(rows * cols, rng);
    auto vec = RandF32(cols, rng);
    std::vector<float> out_s(rows, 0), out_r(rows, 0);
    double s = BenchMs([&]() {
      std::fill(out_s.begin(), out_s.end(), 0.0f);
      ScalarMatVecF32(mat.data(), rows, cols, vec.data(), out_s.data());
    }, iters);
    double r = BenchMs([&]() {
      std::fill(out_r.begin(), out_r.end(), 0.0f);
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          mat.data(), rows, cols, vec.data(), 1, out_r.data());
    }, iters);
    PrintRow("FloatMatVec(256x256)", s, r);
  }

  // Float DotProduct (tensor_utils)
  {
    auto v1 = RandF32(N, rng), v2 = RandF32(N, rng);
    volatile float sink;
    double s = BenchMs([&]() { sink = ScalarDotProduct(v1.data(), v2.data(), N); }, iters);
    double r = BenchMs([&]() { sink = tensor_utils::VectorVectorDotProduct(v1.data(), v2.data(), N); }, iters);
    PrintRow("FloatDotProduct(16384)", s, r);
  }

  // Sub1Vector float
  {
    auto in = RandF32(N, rng);
    std::vector<float> out(N);
    double s = BenchMs([&]() { ScalarSub1Vector(in.data(), N, out.data()); }, iters);
    double r = BenchMs([&]() { tensor_utils::Sub1Vector(in.data(), N, out.data()); }, iters);
    PrintRow("FloatSub1Vec(16384)", s, r);
  }

  // ReductionSum float
  {
    auto in = RandF32(256 * 64, rng);
    std::vector<float> out(256);
    double s = BenchMs([&]() { ScalarReductionSum(in.data(), out.data(), 256, 64); }, iters);
    double r = BenchMs([&]() { tensor_utils::ReductionSumVector(in.data(), out.data(), 256, 64); }, iters);
    PrintRow("FloatRedSum(256x64)", s, r);
  }

  // MeanStddevNormalization
  {
    const int V = 1024, B = 4;
    auto in = RandF32(V * B, rng);
    std::vector<float> out_s(V * B), out_r(V * B);
    double s = BenchMs([&]() { ScalarMeanStddevNorm(in.data(), out_s.data(), V, B); }, iters);
    double r = BenchMs([&]() { tensor_utils::MeanStddevNormalization(in.data(), out_r.data(), V, B); }, iters);
    PrintRow("MeanStddevNorm(1024x4)", s, r);
  }

  // CwiseClipping float
  {
    auto v = RandF32(N, rng);
    std::vector<float> v_s(v), v_r(v);
    double s = BenchMs([&]() { std::copy(v.begin(), v.end(), v_s.begin()); ScalarCwiseClipF32(v_s.data(), N, 5.0f); }, iters);
    double r = BenchMs([&]() { std::copy(v.begin(), v.end(), v_r.begin()); tensor_utils::CwiseClipping(v_r.data(), N, 5.0f); }, iters);
    PrintRow("FloatCwiseClip(16384)", s, r);
  }

  // IsZeroVector float
  {
    std::vector<float> v(N, 0.0f);
    double s = BenchMs([&]() { volatile bool x = ScalarIsZeroF32(v.data(), N); (void)x; }, iters);
    double r = BenchMs([&]() { volatile bool x = tensor_utils::IsZeroVector(v.data(), N); (void)x; }, iters);
    PrintRow("FloatIsZero(16384)", s, r);
  }

  // VectorScalarMultiply
  {
    auto vi = RandI8(N, rng);
    std::vector<float> out(N);
    double s = BenchMs([&]() { ScalarVectorScalarMul(vi.data(), N, 0.5f, out.data()); }, iters);
    double r = BenchMs([&]() { tensor_utils::VectorScalarMultiply(vi.data(), N, 0.5f, out.data()); }, iters);
    PrintRow("VecScalarMul(16384)", s, r);
  }

  printf("\n");
  return 0;
}
