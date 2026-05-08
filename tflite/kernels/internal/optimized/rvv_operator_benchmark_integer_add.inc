void BenchmarkAddInt8Elementwise() {
  const VectorCase cases[] = {
      {"add_i8_1k", 1024},
      {"add_i8_4k", 4096},
      {"add_i8_16k", 16384},
      {"add_i8_64k", 65536},
  };
  BenchmarkBinaryIntegerVectorOp<int8_t>(
      "Add<int8>",
      "`integer_ops/add.h` int8 elementwise quantized add path.",
      cases, sizeof(cases) / sizeof(cases[0]), -48, 48, 0,
      [](const int8_t* lhs, const int8_t* rhs, int size, int8_t* output) {
        ScalarAddInt8Elementwise(size, lhs, rhs, output);
      },
      [](const int8_t* lhs, const int8_t* rhs, int size, int8_t* output) {
        RvvAddInt8Elementwise(size, lhs, rhs, output);
      });
}

void BenchmarkAddInt8ScalarKernel() {
  const VectorCase cases[] = {
      {"add_scalar_kernel_i8_1k", 1024},
      {"add_scalar_kernel_i8_4k", 4096},
      {"add_scalar_kernel_i8_16k", 16384},
      {"add_scalar_kernel_i8_64k", 65536},
  };
  BenchmarkScalarBroadcastIntegerVectorOp<int8_t>(
      "AddScalarBroadcast<int8> direct kernel",
      "Direct int8 scalar-broadcast add kernel used by the fivefold broadcast "
      "fast path in `integer_ops/add.h`.",
      cases, sizeof(cases) / sizeof(cases[0]), -48, 48, 0,
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        ScalarAddInt8ScalarKernel(size, lhs_scalar, rhs, output);
      },
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        RvvAddInt8ScalarKernel(size, lhs_scalar, rhs, output);
      });
}

void BenchmarkAddInt8ScalarBroadcast() {
  const VectorCase cases[] = {
      {"add_bcast_i8_1k", 1024},
      {"add_bcast_i8_4k", 4096},
      {"add_bcast_i8_16k", 16384},
      {"add_bcast_i8_64k", 65536},
  };
  BenchmarkScalarBroadcastIntegerVectorOp<int8_t>(
      "BroadcastAddDispatch<int8> scalar lhs",
      "`integer_ops/add.h` int8 scalar-broadcast fast path via "
      "`BroadcastAddDispatch`.",
      cases, sizeof(cases) / sizeof(cases[0]), -48, 48, 0,
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        ScalarAddInt8ScalarBroadcast(size, lhs_scalar, rhs, output);
      },
      [](int8_t lhs_scalar, const int8_t* rhs, int size, int8_t* output) {
        RvvAddInt8ScalarBroadcast(size, lhs_scalar, rhs, output);
      });
}

void BenchmarkAddInt16Elementwise() {
  const VectorCase cases[] = {
      {"add_i16_1k", 1024},
      {"add_i16_4k", 4096},
      {"add_i16_16k", 16384},
      {"add_i16_64k", 65536},
  };
  BenchmarkBinaryIntegerVectorOp<int16_t>(
      "Add<int16>",
      "`integer_ops/add.h` int16 elementwise quantized add path.",
      cases, sizeof(cases) / sizeof(cases[0]), -4096, 4096, 0,
      [](const int16_t* lhs, const int16_t* rhs, int size, int16_t* output) {
        ScalarAddInt16Elementwise(size, lhs, rhs, output);
      },
      [](const int16_t* lhs, const int16_t* rhs, int size, int16_t* output) {
        RvvAddInt16Elementwise(size, lhs, rhs, output);
      });
}

void RunIntegerAddBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/add.h",
      "Recent RVV work added both int8/int16 elementwise add coverage and the "
      "scalar-broadcast kernels used by the fivefold broadcast fast path.");
  BenchmarkAddInt8Elementwise();
  BenchmarkAddInt8ScalarKernel();
  BenchmarkAddInt8ScalarBroadcast();
  BenchmarkAddInt16Elementwise();
}
