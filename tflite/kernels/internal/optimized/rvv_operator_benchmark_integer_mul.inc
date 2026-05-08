void BenchmarkMulInt8Elementwise() {
  const VectorCase cases[] = {
      {"mul_i8_1k", 1024},
      {"mul_i8_4k", 4096},
      {"mul_i8_16k", 16384},
      {"mul_i8_64k", 65536},
  };
  BenchmarkBinaryIntegerVectorOp<int8_t>(
      "Mul<int8>",
      "`integer_ops/mul.h` int8 elementwise quantized mul path.",
      cases, sizeof(cases) / sizeof(cases[0]), -63, 63, 0,
      [](const int8_t* lhs, const int8_t* rhs, int size, int8_t* output) {
        ScalarMulInt8Elementwise(size, lhs, rhs, output);
      },
      [](const int8_t* lhs, const int8_t* rhs, int size, int8_t* output) {
        RvvMulInt8Elementwise(size, lhs, rhs, output);
      });
}

void BenchmarkMulInt8ScalarKernel() {
  const VectorCase cases[] = {
      {"mul_scalar_kernel_i8_1k", 1024},
      {"mul_scalar_kernel_i8_4k", 4096},
      {"mul_scalar_kernel_i8_16k", 16384},
      {"mul_scalar_kernel_i8_64k", 65536},
  };
  BenchmarkScalarBroadcastIntegerVectorOp<int8_t>(
      "MulSimpleBroadcast<int8> direct kernel",
      "Direct int8 scalar-broadcast mul kernel used by the fivefold broadcast "
      "fast path in `integer_ops/mul.h`.",
      cases, sizeof(cases) / sizeof(cases[0]), -31, 31, 0,
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        ScalarMulInt8ScalarKernel(size, lhs_scalar, rhs, output);
      },
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        RvvMulInt8ScalarKernel(size, lhs_scalar, rhs, output);
      });
}

void BenchmarkMulInt8ScalarBroadcast() {
  const VectorCase cases[] = {
      {"mul_bcast_i8_1k", 1024},
      {"mul_bcast_i8_4k", 4096},
      {"mul_bcast_i8_16k", 16384},
      {"mul_bcast_i8_64k", 65536},
  };
  BenchmarkScalarBroadcastIntegerVectorOp<int8_t>(
      "BroadcastMulDispatch<int8> scalar lhs",
      "`integer_ops/mul.h` int8 scalar-broadcast fast path via "
      "`BroadcastMulDispatch`.",
      cases, sizeof(cases) / sizeof(cases[0]), -31, 31, 0,
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        ScalarMulInt8ScalarBroadcast(size, lhs_scalar, rhs, output);
      },
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        RvvMulInt8ScalarBroadcast(size, lhs_scalar, rhs, output);
      });
}

void RunIntegerMulBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/mul.h",
      "Recent RVV work added both int8 elementwise mul coverage and the "
      "scalar-broadcast kernels used by the quantized broadcast fast path.");
  BenchmarkMulInt8Elementwise();
  BenchmarkMulInt8ScalarKernel();
  BenchmarkMulInt8ScalarBroadcast();
}
