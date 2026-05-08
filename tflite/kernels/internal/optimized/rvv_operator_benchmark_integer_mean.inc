void BenchmarkMeanInt8() {
  const MeanCase cases[] = {
      {"mean_i8_8x8x64", 1, 8, 8, 64},
      {"mean_i8_16x16x128", 1, 16, 16, 128},
      {"mean_i8_32x32x256", 1, 32, 32, 256},
  };
  BenchmarkMeanOp(
      "Mean<int8>",
      "`integer_ops/mean.h` int8 height-width reduction path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [](const MeanCase& bench, const int8_t* input, int8_t* output) {
        ScalarMeanInt8Operator(bench.n_batch, bench.input_height,
                               bench.input_width, bench.depth, input, output);
      },
      [](const MeanCase& bench, const int8_t* input, int8_t* output) {
        RvvMeanInt8Operator(bench.n_batch, bench.input_height,
                            bench.input_width, bench.depth, input, output);
      });
}

void RunIntegerMeanBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/mean.h",
      "This section tracks the int8 height-width reduction path that gained "
      "RVV coverage in the latest integer-ops commit.");
  BenchmarkMeanInt8();
}
