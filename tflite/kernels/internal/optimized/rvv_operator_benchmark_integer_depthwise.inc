void BenchmarkDepthwiseConvInt8() {
  const DepthwiseCase cases[] = {
      {"dwconv_i8_dm1_d4_s1", 1, 32, 32, 4, 3, 3, 1, 1, 1},
      {"dwconv_i8_dm1_d8_s1", 1, 32, 32, 8, 3, 3, 1, 1, 1},
      {"dwconv_i8_dm1_d12_s1", 1, 32, 32, 12, 3, 3, 1, 1, 1},
      {"dwconv_i8_dm1_d16_s2", 1, 32, 32, 16, 3, 3, 2, 2, 1},
      {"dwconv_i8_dm2_d4_s1", 1, 32, 32, 4, 3, 3, 1, 1, 2},
      {"dwconv_i8_dm2_d8_s2", 1, 32, 32, 8, 3, 3, 2, 2, 2},
      {"dwconv_i8_dm4_d1_s1", 1, 32, 32, 1, 3, 3, 1, 1, 4},
      {"dwconv_i8_dm4_d4_s1", 1, 32, 32, 4, 3, 3, 1, 1, 4},
      {"dwconv_i8_dm8_d2_s1", 1, 32, 32, 2, 3, 3, 1, 1, 8},
      {"dwconv_i8_dm8_d1_s2", 1, 32, 32, 1, 3, 3, 2, 2, 8},
      {"dwconv_i8_dm16_d1_s2", 1, 32, 32, 1, 3, 3, 2, 2, 16},
      {"dwconv_i8_dm20_d1_s2", 1, 32, 32, 1, 3, 3, 2, 2, 20},
      {"dwconv_i8_dm32_d1_s2", 1, 32, 32, 1, 3, 3, 2, 2, 32},
  };
  BenchmarkDepthwiseOp(
      "DepthwiseConv<int8>",
      "`depthwise_conv` int8 general path, including fixed input-depth and "
      "depth-multiplier RVV kernels aligned with NEON dispatch.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const DepthwiseCase& bench, const int8_t* input, const int8_t* filter,
          const int32_t* bias, const int32_t* output_multiplier,
          const int32_t* output_shift, int8_t* output) {
        ScalarDepthwiseConvInt8Operator(
            bench.n_batch, bench.input_height, bench.input_width,
            bench.input_depth, bench.filter_height, bench.filter_width,
            bench.depth_multiplier, bench.stride_height, bench.stride_width,
            input, filter, bias, output_multiplier, output_shift, output);
      },
      [&](const DepthwiseCase& bench, const int8_t* input, const int8_t* filter,
          const int32_t* bias, const int32_t* output_multiplier,
          const int32_t* output_shift, int8_t* output) {
        RvvDepthwiseConvInt8Operator(
            bench.n_batch, bench.input_height, bench.input_width,
            bench.input_depth, bench.filter_height, bench.filter_width,
            bench.depth_multiplier, bench.stride_height, bench.stride_width,
            input, filter, bias, output_multiplier, output_shift, output);
      });
}

void RunIntegerDepthwiseBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h",
      "This section tracks the RVV general depthwise int8 kernels, including "
      "the fixed-depth and fixed-depth-multiplier specializations added to "
      "mirror the NEON dispatch table.");
  BenchmarkDepthwiseConvInt8();
  PrintDispatchOnlySection(
      "tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h",
      "The recent change in this file only reroutes hybrid dispatch to reuse "
      "the RVV depthwise kernels above, so it intentionally shares the same "
      "numeric benchmark coverage.");
}
