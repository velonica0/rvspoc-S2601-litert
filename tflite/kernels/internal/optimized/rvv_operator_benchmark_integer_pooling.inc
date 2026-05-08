void BenchmarkAveragePoolInt8() {
  const PoolCase cases[] = {
      {"avgpool_i8_32x32x64", 1, 32, 32, 64, 2, 2, 2, 2},
      {"avgpool_i8_56x56x128", 1, 56, 56, 128, 3, 3, 2, 2},
      {"avgpool_i8_14x14x256", 1, 14, 14, 256, 3, 3, 1, 1},
  };
  BenchmarkPoolOp<int8_t>(
      "AveragePool<int8>",
      "`average_pool` int8 depth inner loop, signed round-away-from-zero, "
      "and clamp-store path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
      [&](const PoolCase& bench, const int8_t* input, int8_t* output) {
        ScalarAveragePoolInt8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      },
      [&](const PoolCase& bench, const int8_t* input, int8_t* output) {
        RvvAveragePoolInt8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      });
}

void BenchmarkMaxPoolInt8() {
  const PoolCase cases[] = {
      {"maxpool_i8_32x32x64", 1, 32, 32, 64, 2, 2, 2, 2},
      {"maxpool_i8_56x56x128", 1, 56, 56, 128, 3, 3, 2, 2},
      {"maxpool_i8_14x14x256", 1, 14, 14, 256, 3, 3, 1, 1},
  };
  BenchmarkPoolOp<int8_t>(
      "MaxPool<int8>",
      "`max_pool` int8 depth inner loop and clamp-store path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
      [&](const PoolCase& bench, const int8_t* input, int8_t* output) {
        ScalarMaxPoolInt8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      },
      [&](const PoolCase& bench, const int8_t* input, int8_t* output) {
        RvvMaxPoolInt8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      });
}

void RunIntegerPoolingBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/pooling.h",
      "This section keeps the signed int8 pooling kernels separate from the "
      "uint8 `optimized_ops.h` pooling paths, because their RVV logic and "
      "rounding semantics are different.");
  BenchmarkAveragePoolInt8();
  BenchmarkMaxPoolInt8();
}
