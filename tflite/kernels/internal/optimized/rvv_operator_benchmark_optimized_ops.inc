void BenchmarkAddFloatElementwise() {
  const VectorCase cases[] = {
      {"add_f32_1k", 1024},
      {"add_f32_4k", 4096},
      {"add_f32_16k", 16384},
      {"add_f32_64k", 65536},
  };
  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();
  BenchmarkFloatBinaryOp(
      "Add<float>", "`add` float optimized path, non-broadcast dense tensors.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        ScalarAddFloatElementwise(lhs.size(), kActMin, kActMax, lhs.data(),
                                  rhs.data(), output->data());
      },
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        RvvAddFloatElementwise(lhs.size(), kActMin, kActMax, lhs.data(),
                               rhs.data(), output->data());
      });
}

void BenchmarkAddFloatScalarBroadcast() {
  const VectorCase cases[] = {
      {"add_bcast_f32_1k", 1024},
      {"add_bcast_f32_4k", 4096},
      {"add_bcast_f32_16k", 16384},
      {"add_bcast_f32_64k", 65536},
  };
  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();
  BenchmarkFloatBinaryOp(
      "BroadcastAddDispatch<float> scalar lhs",
      "`add` float scalar-broadcast fast path via `BroadcastAddDispatch`.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        ScalarAddFloatScalarBroadcast(rhs.size(), kActMin, kActMax, lhs_scalar,
                                      rhs.data(), output->data());
      },
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        RvvAddFloatScalarBroadcast(rhs.size(), kActMin, kActMax, lhs_scalar,
                                   rhs.data(), output->data());
      });
}

void BenchmarkAddFloatScalarKernel() {
  const VectorCase cases[] = {
      {"add_scalar_kernel_f32_1k", 1024},
      {"add_scalar_kernel_f32_4k", 4096},
      {"add_scalar_kernel_f32_16k", 16384},
      {"add_scalar_kernel_f32_64k", 65536},
  };
  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();
  BenchmarkFloatBinaryOp(
      "AddScalarBroadcast<float> direct kernel",
      "Direct scalar-broadcast float add kernel used by the fivefold "
      "broadcast fast path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        ScalarAddFloatScalarKernel(rhs.size(), kActMin, kActMax, lhs_scalar,
                                   rhs.data(), output->data());
      },
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        RvvAddFloatScalarKernel(rhs.size(), kActMin, kActMax, lhs_scalar,
                                rhs.data(), output->data());
      });
}

void BenchmarkMulFloatElementwise() {
  const VectorCase cases[] = {
      {"mul_f32_1k", 1024},
      {"mul_f32_4k", 4096},
      {"mul_f32_16k", 16384},
      {"mul_f32_64k", 65536},
  };
  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();
  BenchmarkFloatBinaryOp(
      "Mul<float>", "`mul` float optimized path, non-broadcast dense tensors.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        ScalarMulFloatElementwise(lhs.size(), kActMin, kActMax, lhs.data(),
                                  rhs.data(), output->data());
      },
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        RvvMulFloatElementwise(lhs.size(), kActMin, kActMax, lhs.data(),
                               rhs.data(), output->data());
      });
}

void BenchmarkMulFloatScalarBroadcast() {
  const VectorCase cases[] = {
      {"mul_bcast_f32_1k", 1024},
      {"mul_bcast_f32_4k", 4096},
      {"mul_bcast_f32_16k", 16384},
      {"mul_bcast_f32_64k", 65536},
  };
  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();
  BenchmarkFloatBinaryOp(
      "BroadcastMulDispatch<float> scalar lhs",
      "`mul` float scalar-broadcast fast path via `BroadcastMulDispatch`.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        ScalarMulFloatScalarBroadcast(rhs.size(), kActMin, kActMax, lhs_scalar,
                                      rhs.data(), output->data());
      },
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        RvvMulFloatScalarBroadcast(rhs.size(), kActMin, kActMax, lhs_scalar,
                                   rhs.data(), output->data());
      });
}

void BenchmarkMulFloatScalarKernel() {
  const VectorCase cases[] = {
      {"mul_scalar_kernel_f32_1k", 1024},
      {"mul_scalar_kernel_f32_4k", 4096},
      {"mul_scalar_kernel_f32_16k", 16384},
      {"mul_scalar_kernel_f32_64k", 65536},
  };
  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();
  BenchmarkFloatBinaryOp(
      "MulSimpleBroadcast<float> direct kernel",
      "Direct scalar-broadcast float mul kernel used by the fivefold "
      "broadcast fast path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        ScalarMulFloatScalarKernel(rhs.size(), kActMin, kActMax, lhs_scalar,
                                   rhs.data(), output->data());
      },
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        const float lhs_scalar = lhs[0];
        RvvMulFloatScalarKernel(rhs.size(), kActMin, kActMax, lhs_scalar,
                                rhs.data(), output->data());
      });
}

void BenchmarkSubFloatElementwise() {
  const VectorCase cases[] = {
      {"sub_f32_1k", 1024},
      {"sub_f32_4k", 4096},
      {"sub_f32_16k", 16384},
      {"sub_f32_64k", 65536},
  };
  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();
  BenchmarkFloatBinaryOp(
      "SubWithActivation<float>",
      "`sub` float non-broadcast optimized path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        ScalarSubFloatElementwise(lhs.size(), kActMin, kActMax, lhs.data(),
                                  rhs.data(), output->data());
      },
      [&](const std::vector<float>& lhs, const std::vector<float>& rhs,
          std::vector<float>* output) {
        RvvSubFloatElementwise(lhs.size(), kActMin, kActMax, lhs.data(),
                               rhs.data(), output->data());
      });
}

void BenchmarkDivFloatElementwise() {
  const VectorCase cases[] = {
      {"div_f32_1k", 1024},
      {"div_f32_4k", 4096},
      {"div_f32_16k", 16384},
      {"div_f32_64k", 65536},
  };
  PrintBenchmarkTitle("Div<float>", "`div` float non-broadcast optimized path.");
  PrintFloatHeader("Vector size");

  constexpr float kActMin = std::numeric_limits<float>::lowest();
  constexpr float kActMax = std::numeric_limits<float>::max();

  for (int idx = 0; idx < static_cast<int>(sizeof(cases) / sizeof(cases[0]));
       ++idx) {
    const VectorCase& bench = cases[idx];
    const int iterations = ChooseIterations(bench.size);
    std::mt19937 rng(0x2345 + bench.size * 29 + idx);
    const std::vector<float> lhs =
        MakeRandomFloatVector(bench.size, &rng, -8.0f, 8.0f);
    std::vector<float> rhs =
        MakeRandomFloatVector(bench.size, &rng, 0.25f, 8.0f);
    std::vector<float> scalar_output(bench.size, 0.0f);
    std::vector<float> rvv_output(bench.size, 0.0f);

    ScalarDivFloatElementwise(bench.size, kActMin, kActMax, lhs.data(),
                              rhs.data(), scalar_output.data());
    RvvDivFloatElementwise(bench.size, kActMin, kActMax, lhs.data(),
                           rhs.data(), rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       ScalarDivFloatElementwise(
                           bench.size, kActMin, kActMax, lhs.data(), rhs.data(),
                           scalar_output.data());
                       return scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(bench.size),
                     [&]() -> double {
                       RvvDivFloatElementwise(bench.size, kActMin, kActMax,
                                              lhs.data(), rhs.data(),
                                              rvv_output.data());
                       return rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << bench.size << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";

    if (!FloatAccuracyWithinTolerance(accuracy)) {
      std::cerr << "Div<float> accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkAffineQuantizeInt8() {
  const VectorCase cases[] = {
      {"affine_q_i8_1k", 1024},
      {"affine_q_i8_4k", 4096},
      {"affine_q_i8_16k", 16384},
      {"affine_q_i8_64k", 65536},
  };
  BenchmarkAffineQuantizeOp<int8_t>(
      "AffineQuantize<int8>",
      "`quantize` float-to-int8 path used by int8 activations and weights.",
      cases, sizeof(cases) / sizeof(cases[0]), -3, 1.0f / 32.0f, -8.0f, 8.0f,
      1,
      [&](const std::vector<float>& input, std::vector<int8_t>* output,
          int32_t zero_point, float scale) {
        ScalarAffineQuantizeInt8Operator(input.size(), zero_point, scale,
                                         input.data(), output->data());
      },
      [&](const std::vector<float>& input, std::vector<int8_t>* output,
          int32_t zero_point, float scale) {
        RvvAffineQuantizeInt8Operator(input.size(), zero_point, scale,
                                      input.data(), output->data());
      });
}

void BenchmarkAffineQuantizeUint8() {
  const VectorCase cases[] = {
      {"affine_q_u8_1k", 1024},
      {"affine_q_u8_4k", 4096},
      {"affine_q_u8_16k", 16384},
      {"affine_q_u8_64k", 65536},
  };
  BenchmarkAffineQuantizeOp<uint8_t>(
      "AffineQuantize<uint8>",
      "`quantize` float-to-uint8 path used by uint8 activations.",
      cases, sizeof(cases) / sizeof(cases[0]), 127, 1.0f / 32.0f, -4.0f, 4.0f,
      1,
      [&](const std::vector<float>& input, std::vector<uint8_t>* output,
          int32_t zero_point, float scale) {
        ScalarAffineQuantizeUint8Operator(input.size(), zero_point, scale,
                                          input.data(), output->data());
      },
      [&](const std::vector<float>& input, std::vector<uint8_t>* output,
          int32_t zero_point, float scale) {
        RvvAffineQuantizeUint8Operator(input.size(), zero_point, scale,
                                       input.data(), output->data());
      });
}

void BenchmarkAffineQuantizeInt16() {
  const VectorCase cases[] = {
      {"affine_q_i16_1k", 1024},
      {"affine_q_i16_4k", 4096},
      {"affine_q_i16_16k", 16384},
      {"affine_q_i16_64k", 65536},
  };
  BenchmarkAffineQuantizeOp<int16_t>(
      "AffineQuantize<int16>",
      "`quantize` float-to-int16 path used by q15-style operator staging.",
      cases, sizeof(cases) / sizeof(cases[0]), 0, 1.0f / 256.0f, -32.0f, 32.0f,
      1,
      [&](const std::vector<float>& input, std::vector<int16_t>* output,
          int32_t zero_point, float scale) {
        ScalarAffineQuantizeInt16Operator(input.size(), zero_point, scale,
                                          input.data(), output->data());
      },
      [&](const std::vector<float>& input, std::vector<int16_t>* output,
          int32_t zero_point, float scale) {
        RvvAffineQuantizeInt16Operator(input.size(), zero_point, scale,
                                       input.data(), output->data());
      });
}

void BenchmarkAveragePoolUint8() {
  const PoolCase cases[] = {
      {"avgpool_u8_32x32x64", 1, 32, 32, 64, 2, 2, 2, 2},
      {"avgpool_u8_56x56x128", 1, 56, 56, 128, 3, 3, 2, 2},
      {"avgpool_u8_14x14x256", 1, 14, 14, 256, 3, 3, 1, 1},
  };
  BenchmarkPoolOp<uint8_t>(
      "AveragePool<uint8>",
      "`average_pool` quantized depth inner loop and clamp-store path.",
      cases, sizeof(cases) / sizeof(cases[0]), 0, 255,
      [&](const PoolCase& bench, const uint8_t* input, uint8_t* output) {
        ScalarAveragePoolUint8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      },
      [&](const PoolCase& bench, const uint8_t* input, uint8_t* output) {
        RvvAveragePoolUint8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      });
}

void BenchmarkMaxPoolUint8() {
  const PoolCase cases[] = {
      {"maxpool_u8_32x32x64", 1, 32, 32, 64, 2, 2, 2, 2},
      {"maxpool_u8_56x56x128", 1, 56, 56, 128, 3, 3, 2, 2},
      {"maxpool_u8_14x14x256", 1, 14, 14, 256, 3, 3, 1, 1},
  };
  BenchmarkPoolOp<uint8_t>(
      "MaxPool<uint8>",
      "`max_pool` quantized depth inner loop and clamp-store path.",
      cases, sizeof(cases) / sizeof(cases[0]), 0, 255,
      [&](const PoolCase& bench, const uint8_t* input, uint8_t* output) {
        ScalarMaxPoolUint8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      },
      [&](const PoolCase& bench, const uint8_t* input, uint8_t* output) {
        RvvMaxPoolUint8Operator(
            bench.n_batch, bench.input_height, bench.input_width, bench.depth,
            bench.filter_height, bench.filter_width, bench.stride_height,
            bench.stride_width, input, output);
      });
}

void BenchmarkHardSwishUint8() {
  const VectorCase cases[] = {
      {"hardswish_u8_1k", 1024},
      {"hardswish_u8_4k", 4096},
      {"hardswish_u8_16k", 16384},
      {"hardswish_u8_64k", 65536},
  };
  BenchmarkUnaryIntegerVectorOp<uint8_t>(
      "HardSwish<uint8>",
      "`hard_swish` quantized uint8 fixed-point multiplier path.",
      cases, sizeof(cases) / sizeof(cases[0]), 0, 255, 0,
      [](const uint8_t* input, int size, uint8_t* output) {
        ScalarHardSwishUint8Operator(size, input, output);
      },
      [](const uint8_t* input, int size, uint8_t* output) {
        RvvHardSwishUint8Operator(size, input, output);
      });
}

void BenchmarkHardSwishInt8() {
  const VectorCase cases[] = {
      {"hardswish_i8_1k", 1024},
      {"hardswish_i8_4k", 4096},
      {"hardswish_i8_16k", 16384},
      {"hardswish_i8_64k", 65536},
  };
  BenchmarkUnaryIntegerVectorOp<int8_t>(
      "HardSwish<int8>",
      "`hard_swish` quantized int8 fixed-point multiplier path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
      0,
      [](const int8_t* input, int size, int8_t* output) {
        ScalarHardSwishInt8Operator(size, input, output);
      },
      [](const int8_t* input, int size, int8_t* output) {
        RvvHardSwishInt8Operator(size, input, output);
      });
}

void BenchmarkArgMinFloat() {
  const VectorCase cases[] = {
      {"argmin_f32_1k", 1024},
      {"argmin_f32_16k", 16384},
      {"argmin_f32_64k", 65536},
  };
  BenchmarkIndexOp<float>(
      "ArgMin<float>",
      "`arg_min` float last-axis reduction, including tie-keep-first "
      "semantics.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [](int size, std::mt19937* rng) {
        std::vector<float> values =
            MakeRandomFloatVector(size, rng, -8.0f, 8.0f);
        if (size > 32) {
          values[17] = -99.0f;
          values[size / 2] = -99.0f;
          values[size - 3] = -99.0f;
        }
        return values;
      },
      [](const float* input, int size) {
        return ScalarArgMinFloatOperator(input, size);
      },
      [](const float* input, int size) { return RvvArgMinFloatOperator(input, size); });
}

void BenchmarkArgMaxFloat() {
  const VectorCase cases[] = {
      {"argmax_f32_1k", 1024},
      {"argmax_f32_16k", 16384},
      {"argmax_f32_64k", 65536},
  };
  BenchmarkIndexOp<float>(
      "ArgMax<float>",
      "`arg_max` float last-axis reduction, including tie-keep-first "
      "semantics.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [](int size, std::mt19937* rng) {
        std::vector<float> values =
            MakeRandomFloatVector(size, rng, -8.0f, 8.0f);
        if (size > 32) {
          values[17] = 99.0f;
          values[size / 2] = 99.0f;
          values[size - 3] = 99.0f;
        }
        return values;
      },
      [](const float* input, int size) {
        return ScalarArgMaxFloatOperator(input, size);
      },
      [](const float* input, int size) { return RvvArgMaxFloatOperator(input, size); });
}

void BenchmarkArgMaxInt8() {
  const VectorCase cases[] = {
      {"argmax_i8_1k", 1024},
      {"argmax_i8_16k", 16384},
      {"argmax_i8_64k", 65536},
  };
  BenchmarkIndexOp<int8_t>(
      "ArgMax<int8>",
      "`arg_max` int8 last-axis reduction, including tie-keep-first "
      "semantics.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [](int size, std::mt19937* rng) {
        std::vector<int8_t> values =
            MakeRandomIntegralVector<int8_t>(size, rng, -127, 127);
        if (size > 32) {
          values[17] = 127;
          values[size / 2] = 127;
          values[size - 3] = 127;
        }
        return values;
      },
      [](const int8_t* input, int size) {
        return ScalarArgMaxInt8Operator(input, size);
      },
      [](const int8_t* input, int size) { return RvvArgMaxInt8Operator(input, size); });
}

void BenchmarkArgMaxUint8() {
  const VectorCase cases[] = {
      {"argmax_u8_1k", 1024},
      {"argmax_u8_16k", 16384},
      {"argmax_u8_64k", 65536},
  };
  BenchmarkIndexOp<uint8_t>(
      "ArgMax<uint8>",
      "`arg_max` uint8 last-axis reduction, including tie-keep-first "
      "semantics.",
      cases, sizeof(cases) / sizeof(cases[0]),
      [](int size, std::mt19937* rng) {
        std::vector<uint8_t> values =
            MakeRandomIntegralVector<uint8_t>(size, rng, 0, 255);
        if (size > 32) {
          values[17] = 255;
          values[size / 2] = 255;
          values[size - 3] = 255;
        }
        return values;
      },
      [](const uint8_t* input, int size) {
        return ScalarArgMaxUint8Operator(input, size);
      },
      [](const uint8_t* input, int size) { return RvvArgMaxUint8Operator(input, size); });
}

void RunOptimizedOpsBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/optimized_ops.h",
      "Float arithmetic fast paths, quantize/dequant-style helpers, uint8 "
      "pooling, quantized HardSwish, and argmin/argmax cases that gained RVV "
      "coverage in the latest operator-level commits.");
  BenchmarkAddFloatElementwise();
  BenchmarkAddFloatScalarKernel();
  BenchmarkAddFloatScalarBroadcast();
  BenchmarkMulFloatElementwise();
  BenchmarkMulFloatScalarKernel();
  BenchmarkMulFloatScalarBroadcast();
  BenchmarkSubFloatElementwise();
  BenchmarkDivFloatElementwise();
  BenchmarkAffineQuantizeInt8();
  BenchmarkAffineQuantizeUint8();
  BenchmarkAffineQuantizeInt16();
  BenchmarkAveragePoolUint8();
  BenchmarkMaxPoolUint8();
  BenchmarkHardSwishUint8();
  BenchmarkHardSwishInt8();
  BenchmarkArgMinFloat();
  BenchmarkArgMaxFloat();
  BenchmarkArgMaxInt8();
  BenchmarkArgMaxUint8();
}
