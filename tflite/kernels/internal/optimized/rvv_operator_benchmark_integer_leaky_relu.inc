void BenchmarkLeakyReluInt16() {
  const VectorCase cases[] = {
      {"leaky_relu_i16_1k", 1024},
      {"leaky_relu_i16_4k", 4096},
      {"leaky_relu_i16_16k", 16384},
      {"leaky_relu_i16_64k", 65536},
  };
  BenchmarkUnaryIntegerVectorOp<int16_t>(
      "LeakyRelu<int16>",
      "`integer_ops/leaky_relu.h` int16 quantized leaky-relu path.",
      cases, sizeof(cases) / sizeof(cases[0]), -8192, 8191, 0,
      [](const int16_t* input, int size, int16_t* output) {
        ScalarLeakyReluInt16Operator(size, input, output);
      },
      [](const int16_t* input, int size, int16_t* output) {
        RvvLeakyReluInt16Operator(size, input, output);
      });
}

void RunIntegerLeakyReluBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/leaky_relu.h",
      "This section isolates the int16 quantized LeakyReLU path that received "
      "a dedicated RVV implementation.");
  BenchmarkLeakyReluInt16();
}
