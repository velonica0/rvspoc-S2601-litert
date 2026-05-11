/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"). */
// Unified RVV operator accuracy + speedup test.
// Combines accuracy verification and scalar-vs-RVV timing in one binary.
// No external dependencies (no gtest, no google benchmark).
//
// Usage: ./rvv_bench

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/optimized/integer_ops/add.h"
#include "tflite/kernels/internal/optimized/integer_ops/mul.h"
#include "tflite/kernels/internal/optimized/integer_ops/sub.h"
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

static int g_pass = 0, g_fail = 0;
static constexpr int kSeed = 42;

#define CHECK(cond, msg) do { \
  if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
  else { g_pass++; } \
} while(0)

// --- Random data generators ---

std::vector<int8_t> RandI8(int n, std::mt19937& rng) {
  std::uniform_int_distribution<int> d(-128, 127);
  std::vector<int8_t> v(n);
  for (auto& x : v) x = static_cast<int8_t>(d(rng));
  return v;
}
std::vector<int16_t> RandI16(int n, std::mt19937& rng) {
  std::uniform_int_distribution<int> d(-32768, 32767);
  std::vector<int16_t> v(n);
  for (auto& x : v) x = static_cast<int16_t>(d(rng));
  return v;
}
std::vector<float> RandF32(int n, std::mt19937& rng) {
  std::uniform_real_distribution<float> d(-10.0f, 10.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// --- Benchmark helper ---

template <typename Fn>
double BenchMs(Fn fn, int iters) {
  for (int i = 0; i < 5; i++) fn();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) fn();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// --- Scalar reference implementations ---

void ScalarFloatAdd(int n, const float* a, const float* b, float* o, float mn, float mx) {
  for (int i = 0; i < n; i++) o[i] = std::min(mx, std::max(mn, a[i]+b[i]));
}
void ScalarFloatMul(int n, const float* a, const float* b, float* o, float mn, float mx) {
  for (int i = 0; i < n; i++) o[i] = std::min(mx, std::max(mn, a[i]*b[i]));
}
void ScalarFloatHardSwish(int n, const float* in, float* out) {
  for (int i = 0; i < n; i++) out[i] = in[i]*std::min(6.f,std::max(0.f,in[i]+3.f))/6.f;
}
void ScalarInt8Add(int n, const ArithmeticParams& p, const int8_t* a, const int8_t* b, int8_t* o) {
  for (int i = 0; i < n; i++) {
    int32_t v1 = p.input1_offset + a[i], v2 = p.input2_offset + b[i];
    int32_t s1 = MultiplyByQuantizedMultiplierSmallerThanOneExp(v1*(1<<p.left_shift), p.input1_multiplier, p.input1_shift);
    int32_t s2 = MultiplyByQuantizedMultiplierSmallerThanOneExp(v2*(1<<p.left_shift), p.input2_multiplier, p.input2_shift);
    int32_t r = MultiplyByQuantizedMultiplierSmallerThanOneExp(s1+s2, p.output_multiplier, p.output_shift) + p.output_offset;
    o[i] = static_cast<int8_t>(std::min(p.quantized_activation_max, std::max(p.quantized_activation_min, r)));
  }
}
void ScalarInt8Mul(int n, const ArithmeticParams& p, const int8_t* a, const int8_t* b, int8_t* o) {
  for (int i = 0; i < n; i++) {
    int32_t v1 = p.input1_offset+a[i], v2 = p.input2_offset+b[i];
    int32_t r = p.output_offset + MultiplyByQuantizedMultiplier(v1*v2, p.output_multiplier, p.output_shift);
    o[i] = static_cast<int8_t>(std::min(p.quantized_activation_max, std::max(p.quantized_activation_min, r)));
  }
}
float ScalarDot(const float* a, const float* b, int n) { float s=0; for(int i=0;i<n;i++) s+=a[i]*b[i]; return s; }
void ScalarSub1(const float* in, int n, float* out) { for(int i=0;i<n;i++) out[i]=1.f-in[i]; }
void ScalarRedSum(const float* in, float* out, int os, int rs) {
  for(int o=0;o<os;o++){float s=0;for(int r=0;r<rs;r++)s+=in[r];out[o]=s;in+=rs;}
}
void ScalarMeanStddev(const float* in, float* out, int v, int nb) {
  for(int b=0;b<nb;b++){float s=0;for(int i=0;i<v;i++)s+=in[i];float m=s/v;
  float ss=0;for(int i=0;i<v;i++){float d=in[i]-m;ss+=d*d;}
  float inv=1.f/std::sqrt(ss/v+1e-8f);for(int i=0;i<v;i++)out[i]=(in[i]-m)*inv;in+=v;out+=v;}
}
void ScalarClip(float* v, int n, float c) { for(int i=0;i<n;i++) v[i]=std::min(c,std::max(-c,v[i])); }
bool ScalarIsZero(const float* v, int n) { for(int i=0;i<n;i++) if(v[i]!=0.f) return false; return true; }
void ScalarVecScalarMul(const int8_t* v, int n, float s, float* o) { for(int i=0;i<n;i++) o[i]=v[i]*s; }
void ScalarMatVec(const float* m, int r, int c, const float* v, float* o) {
  for(int i=0;i<r;i++){float s=0;for(int j=0;j<c;j++)s+=m[i*c+j]*v[j];o[i]+=s;}
}

// --- Accuracy tests ---

ArithmeticParams MakeAddParams() {
  ArithmeticParams p={};
  p.input1_offset=3;p.input2_offset=-5;p.output_offset=2;p.left_shift=5;
  p.input1_multiplier=1073741824;p.input1_shift=-1;p.input2_multiplier=1073741824;p.input2_shift=-2;
  p.output_multiplier=1073741824;p.output_shift=-3;p.quantized_activation_min=-128;p.quantized_activation_max=127;
  return p;
}
ArithmeticParams MakeMulParams() {
  ArithmeticParams p={};p.input1_offset=2;p.input2_offset=-3;p.output_offset=1;
  p.output_multiplier=1073741824;p.output_shift=-4;p.quantized_activation_min=-128;p.quantized_activation_max=127;
  return p;
}

void RunAccuracyTests() {
  printf("=== Accuracy Tests ===\n");
  int sizes[] = {1,3,7,15,16,17,31,32,33,63,64,100,128,255,256,512,1000,1023,1024};
  int ns = sizeof(sizes)/sizeof(sizes[0]);
  std::mt19937 rng(kSeed);

  // FloatAdd
  for (int si=0;si<ns;si++){int n=sizes[si];auto a=RandF32(n,rng),b=RandF32(n,rng);
    std::vector<float> ref(n),opt(n);ArithmeticParams p={};p.float_activation_min=-100;p.float_activation_max=100;
    for(int i=0;i<n;i++)ref[i]=std::min(100.f,std::max(-100.f,a[i]+b[i]));
    optimized_ops::AddElementwise(n,p,a.data(),b.data(),opt.data());
    float mx=0;for(int i=0;i<n;i++)mx=std::max(mx,std::fabs(ref[i]-opt[i]));
    char msg[64];snprintf(msg,sizeof(msg),"FloatAdd(%d) err=%.1e",n,mx);CHECK(mx<=1e-5f,msg);}

  // FloatMul
  for (int si=0;si<ns;si++){int n=sizes[si];auto a=RandF32(n,rng),b=RandF32(n,rng);
    std::vector<float> ref(n),opt(n);ArithmeticParams p={};p.float_activation_min=-1000;p.float_activation_max=1000;
    for(int i=0;i<n;i++)ref[i]=std::min(1000.f,std::max(-1000.f,a[i]*b[i]));
    optimized_ops::MulElementwise(n,p,a.data(),b.data(),opt.data());
    float mx=0;for(int i=0;i<n;i++)mx=std::max(mx,std::fabs(ref[i]-opt[i]));
    char msg[64];snprintf(msg,sizeof(msg),"FloatMul(%d) err=%.1e",n,mx);CHECK(mx<=1e-5f,msg);}

  // FloatHardSwish
  for (int si=0;si<ns;si++){int n=sizes[si];auto in=RandF32(n,rng);
    std::vector<float> ref(n),opt(n);
    for(int i=0;i<n;i++)ref[i]=in[i]*std::min(6.f,std::max(0.f,in[i]+3.f))/6.f;
    RuntimeShape s({1,1,1,n});optimized_ops::HardSwish(s,in.data(),s,opt.data());
    float mx=0;for(int i=0;i<n;i++)mx=std::max(mx,std::fabs(ref[i]-opt[i]));
    char msg[64];snprintf(msg,sizeof(msg),"FloatHardSwish(%d) err=%.1e",n,mx);CHECK(mx<=1e-5f,msg);}

  // Int8Add
  auto pa=MakeAddParams();
  for (int si=0;si<ns;si++){int n=sizes[si];auto a=RandI8(n,rng),b=RandI8(n,rng);
    std::vector<int8_t> ref(n),opt(n);ScalarInt8Add(n,pa,a.data(),b.data(),ref.data());
    optimized_integer_ops::AddElementwiseInt8(n,pa,a.data(),b.data(),opt.data());
    int mx=0;for(int i=0;i<n;i++)mx=std::max(mx,std::abs((int)ref[i]-(int)opt[i]));
    char msg[64];snprintf(msg,sizeof(msg),"Int8Add(%d) diff=%d",n,mx);CHECK(mx<=1,msg);}

  // Int8Mul
  auto pm=MakeMulParams();
  for (int si=0;si<ns;si++){int n=sizes[si];auto a=RandI8(n,rng),b=RandI8(n,rng);
    std::vector<int8_t> ref(n),opt(n);ScalarInt8Mul(n,pm,a.data(),b.data(),ref.data());
    optimized_integer_ops::MulElementwise(n,pm,a.data(),b.data(),opt.data());
    int mx=0;for(int i=0;i<n;i++)mx=std::max(mx,std::abs((int)ref[i]-(int)opt[i]));
    char msg[64];snprintf(msg,sizeof(msg),"Int8Mul(%d) diff=%d",n,mx);CHECK(mx<=1,msg);}

  printf("Accuracy: %d passed, %d failed\n\n", g_pass, g_fail);
}

// --- Speedup benchmarks ---

void PrintRow(const char* name, double scalar_ms, double rvv_ms) {
  printf("  %-32s  scalar=%7.3f ms  rvv=%7.3f ms  speedup=%.2fx\n",
         name, scalar_ms, rvv_ms, scalar_ms/rvv_ms);
}

void RunSpeedupTests() {
  printf("=== Operator Speedup (scalar vs optimized) ===\n");
  const int N=16384, iters=2000;
  std::mt19937 rng(kSeed);
  ArithmeticParams pa=MakeAddParams(), pm=MakeMulParams();

  {auto a=RandF32(N,rng),b=RandF32(N,rng);std::vector<float> o(N);
   ArithmeticParams p={};p.float_activation_min=-100;p.float_activation_max=100;
   double s=BenchMs([&](){ScalarFloatAdd(N,a.data(),b.data(),o.data(),-100,100);},iters);
   double r=BenchMs([&](){optimized_ops::AddElementwise(N,p,a.data(),b.data(),o.data());},iters);
   PrintRow("FloatAdd(16384)",s,r);}

  {auto a=RandF32(N,rng),b=RandF32(N,rng);std::vector<float> o(N);
   ArithmeticParams p={};p.float_activation_min=-1000;p.float_activation_max=1000;
   double s=BenchMs([&](){ScalarFloatMul(N,a.data(),b.data(),o.data(),-1000,1000);},iters);
   double r=BenchMs([&](){optimized_ops::MulElementwise(N,p,a.data(),b.data(),o.data());},iters);
   PrintRow("FloatMul(16384)",s,r);}

  {auto in=RandF32(N,rng);std::vector<float> o(N);RuntimeShape sh({1,1,1,N});
   double s=BenchMs([&](){ScalarFloatHardSwish(N,in.data(),o.data());},iters);
   double r=BenchMs([&](){optimized_ops::HardSwish(sh,in.data(),sh,o.data());},iters);
   PrintRow("FloatHardSwish(16384)",s,r);}

  {auto a=RandI8(N,rng),b=RandI8(N,rng);std::vector<int8_t> o(N);
   double s=BenchMs([&](){ScalarInt8Add(N,pa,a.data(),b.data(),o.data());},iters);
   double r=BenchMs([&](){optimized_integer_ops::AddElementwiseInt8(N,pa,a.data(),b.data(),o.data());},iters);
   PrintRow("Int8Add(16384)",s,r);}

  {auto a=RandI8(N,rng),b=RandI8(N,rng);std::vector<int8_t> o(N);
   double s=BenchMs([&](){ScalarInt8Mul(N,pm,a.data(),b.data(),o.data());},iters);
   double r=BenchMs([&](){optimized_integer_ops::MulElementwise(N,pm,a.data(),b.data(),o.data());},iters);
   PrintRow("Int8Mul(16384)",s,r);}

  printf("\n=== Tensor Utility Speedup ===\n");

  {const int R=256,C=256;auto m=RandF32(R*C,rng),v=RandF32(C,rng);
   std::vector<float> os(R,0),or2(R,0);
   double s=BenchMs([&](){std::fill(os.begin(),os.end(),0.f);ScalarMatVec(m.data(),R,C,v.data(),os.data());},iters);
   double r=BenchMs([&](){std::fill(or2.begin(),or2.end(),0.f);tensor_utils::MatrixBatchVectorMultiplyAccumulate(m.data(),R,C,v.data(),1,or2.data());},iters);
   PrintRow("FloatMatVec(256x256)",s,r);}

  {auto a=RandF32(N,rng),b=RandF32(N,rng);volatile float sink;
   double s=BenchMs([&](){sink=ScalarDot(a.data(),b.data(),N);},iters);
   double r=BenchMs([&](){sink=tensor_utils::VectorVectorDotProduct(a.data(),b.data(),N);},iters);
   PrintRow("FloatDotProduct(16384)",s,r);}

  {auto in=RandF32(N,rng);std::vector<float> o(N);
   double s=BenchMs([&](){ScalarSub1(in.data(),N,o.data());},iters);
   double r=BenchMs([&](){tensor_utils::Sub1Vector(in.data(),N,o.data());},iters);
   PrintRow("FloatSub1Vec(16384)",s,r);}

  {auto in=RandF32(256*64,rng);std::vector<float> o(256);
   double s=BenchMs([&](){ScalarRedSum(in.data(),o.data(),256,64);},iters);
   double r=BenchMs([&](){tensor_utils::ReductionSumVector(in.data(),o.data(),256,64);},iters);
   PrintRow("FloatRedSum(256x64)",s,r);}

  {const int V=1024,B=4;auto in=RandF32(V*B,rng);std::vector<float> os(V*B),or2(V*B);
   double s=BenchMs([&](){ScalarMeanStddev(in.data(),os.data(),V,B);},iters);
   double r=BenchMs([&](){tensor_utils::MeanStddevNormalization(in.data(),or2.data(),V,B);},iters);
   PrintRow("MeanStddevNorm(1024x4)",s,r);}

  {auto v=RandF32(N,rng);std::vector<float> vs(v),vr(v);
   double s=BenchMs([&](){std::copy(v.begin(),v.end(),vs.begin());ScalarClip(vs.data(),N,5.f);},iters);
   double r=BenchMs([&](){std::copy(v.begin(),v.end(),vr.begin());tensor_utils::CwiseClipping(vr.data(),N,5.f);},iters);
   PrintRow("FloatCwiseClip(16384)",s,r);}

  {std::vector<float> v(N,0.0f);
   double s=BenchMs([&](){volatile bool x=ScalarIsZero(v.data(),N);(void)x;},iters);
   double r=BenchMs([&](){volatile bool x=tensor_utils::IsZeroVector(v.data(),N);(void)x;},iters);
   PrintRow("FloatIsZero(16384)",s,r);}

  {auto vi=RandI8(N,rng);std::vector<float> o(N);
   double s=BenchMs([&](){ScalarVecScalarMul(vi.data(),N,0.5f,o.data());},iters);
   double r=BenchMs([&](){tensor_utils::VectorScalarMultiply(vi.data(),N,0.5f,o.data());},iters);
   PrintRow("VecScalarMul(16384)",s,r);}

  printf("\n");
}

int main() {
#ifdef USE_RVV
  printf("USE_RVV: active\n\n");
#else
  printf("USE_RVV: not active (scalar fallback)\n\n");
#endif
  RunAccuracyTests();
  RunSpeedupTests();
  return g_fail > 0 ? 1 : 0;
}
