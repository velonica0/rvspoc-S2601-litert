void BenchmarkSubInt16Elementwise() {
  const VectorCase cases[] = {
      {"sub_i16_1k", 1024},
      {"sub_i16_4k", 4096},
      {"sub_i16_16k", 16384},
      {"sub_i16_64k", 65536},
  };
  BenchmarkBinaryIntegerVectorOp<int16_t>(
      "Sub<int16>",
      "`integer_ops/sub.h` int16 elementwise quantized sub path.",
      cases, sizeof(cases) / sizeof(cases[0]), -4096, 4096, 0,
      [](const int16_t* lhs, const int16_t* rhs, int size, int16_t* output) {
        ScalarSubInt16Elementwise(size, lhs, rhs, output);
      },
      [](const int16_t* lhs, const int16_t* rhs, int size, int16_t* output) {
        RvvSubInt16Elementwise(size, lhs, rhs, output);
      });
}

void RunIntegerSubBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/sub.h",
      "This section isolates the int16 quantized subtract path that gained a "
      "dedicated RVV elementwise implementation.");
  BenchmarkSubInt16Elementwise();
}
