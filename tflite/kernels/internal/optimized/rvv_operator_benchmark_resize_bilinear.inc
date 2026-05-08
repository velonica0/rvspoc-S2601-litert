struct ResizeBilinearCase {
  const char* name;
  int n_batch;
  int input_height;
  int input_width;
  int output_height;
  int output_width;
  int depth;
  bool align_corners;
  bool half_pixel_centers;
};

std::string MakeResizeBilinearShape(const ResizeBilinearCase& bench) {
  return "B" + std::to_string(bench.n_batch) + " I" +
         std::to_string(bench.input_height) + "x" +
         std::to_string(bench.input_width) + "x" +
         std::to_string(bench.depth) + " O" +
         std::to_string(bench.output_height) + "x" +
         std::to_string(bench.output_width) + " AC" +
         std::to_string(bench.align_corners ? 1 : 0) + " HPC" +
         std::to_string(bench.half_pixel_centers ? 1 : 0);
}

void BenchmarkResizeBilinearFloat() {
  const ResizeBilinearCase cases[] = {
      {"resize_f32_17x19x48_to_29x31", 1, 17, 19, 29, 31, 48, false, false},
      {"resize_f32_23x27x96_to_37x41", 1, 23, 27, 37, 41, 96, false, true},
      {"resize_f32_15x21x160_to_28x35", 1, 15, 21, 28, 35, 160, true, false},
  };
  PrintBenchmarkTitle(
      "ResizeBilinear<float>",
      "`resize_bilinear.h` generic float bilinear accumulation kernel.");
  PrintFloatHeader("Shape");

  for (int idx = 0; idx < static_cast<int>(sizeof(cases) / sizeof(cases[0]));
       ++idx) {
    const ResizeBilinearCase& bench = cases[idx];
    const int input_size = bench.n_batch * bench.input_height *
                           bench.input_width * bench.depth;
    const int output_size = bench.n_batch * bench.output_height *
                            bench.output_width * bench.depth;
    const int64_t ops_per_iter = static_cast<int64_t>(output_size) * 7;
    const int iterations = ChooseIterations(ops_per_iter);
    std::mt19937 rng(0x7171 + idx * 53 + bench.depth);
    const std::vector<float> input =
        MakeRandomFloatVector(input_size, &rng, -3.0f, 3.0f);
    std::vector<float> scalar_output(output_size, 0.0f);
    std::vector<float> rvv_output(output_size, 0.0f);

    ScalarResizeBilinearFloatOperator(
        bench.n_batch, bench.input_height, bench.input_width, bench.depth,
        bench.output_height, bench.output_width, bench.align_corners,
        bench.half_pixel_centers, input.data(), scalar_output.data());
    RvvResizeBilinearFloatOperator(
        bench.n_batch, bench.input_height, bench.input_width, bench.depth,
        bench.output_height, bench.output_width, bench.align_corners,
        bench.half_pixel_centers, input.data(), rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       ScalarResizeBilinearFloatOperator(
                           bench.n_batch, bench.input_height, bench.input_width,
                           bench.depth, bench.output_height, bench.output_width,
                           bench.align_corners, bench.half_pixel_centers,
                           input.data(), scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       RvvResizeBilinearFloatOperator(
                           bench.n_batch, bench.input_height, bench.input_width,
                           bench.depth, bench.output_height, bench.output_width,
                           bench.align_corners, bench.half_pixel_centers,
                           input.data(), rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | "
              << MakeResizeBilinearShape(bench) << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";

    if (!FloatAccuracyWithinCodeXOperatorTolerance(accuracy)) {
      std::cerr << "ResizeBilinear<float> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkResizeBilinearUint8() {
  const ResizeBilinearCase cases[] = {
      {"resize_u8_17x19x48_to_29x31", 1, 17, 19, 29, 31, 48, false, false},
      {"resize_u8_23x27x96_to_37x41", 1, 23, 27, 37, 41, 96, false, true},
      {"resize_u8_15x21x160_to_28x35", 1, 15, 21, 28, 35, 160, true, false},
  };
  PrintBenchmarkTitle(
      "ResizeBilinear<uint8>",
      "`resize_bilinear.h` uint8 generic-small-channel interpolation path.");
  PrintIntegerHeader("Shape");

  for (int idx = 0; idx < static_cast<int>(sizeof(cases) / sizeof(cases[0]));
       ++idx) {
    const ResizeBilinearCase& bench = cases[idx];
    const int input_size = bench.n_batch * bench.input_height *
                           bench.input_width * bench.depth;
    const int output_size = bench.n_batch * bench.output_height *
                            bench.output_width * bench.depth;
    const int64_t ops_per_iter = static_cast<int64_t>(output_size) * 7;
    const int iterations = ChooseIterations(ops_per_iter);
    std::mt19937 rng(0x8181 + idx * 59 + bench.depth);
    const std::vector<uint8_t> input =
        MakeRandomIntegralVector<uint8_t>(input_size, &rng, 0, 255);
    std::vector<uint8_t> scalar_output(output_size, 0);
    std::vector<uint8_t> rvv_output(output_size, 0);

    ScalarResizeBilinearUint8Operator(
        bench.n_batch, bench.input_height, bench.input_width, bench.depth,
        bench.output_height, bench.output_width, bench.align_corners,
        bench.half_pixel_centers, input.data(), scalar_output.data());
    RvvResizeBilinearUint8Operator(
        bench.n_batch, bench.input_height, bench.input_width, bench.depth,
        bench.output_height, bench.output_width, bench.align_corners,
        bench.half_pixel_centers, input.data(), rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       ScalarResizeBilinearUint8Operator(
                           bench.n_batch, bench.input_height, bench.input_width,
                           bench.depth, bench.output_height, bench.output_width,
                           bench.align_corners, bench.half_pixel_centers,
                           input.data(), scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       RvvResizeBilinearUint8Operator(
                           bench.n_batch, bench.input_height, bench.input_width,
                           bench.depth, bench.output_height, bench.output_width,
                           bench.align_corners, bench.half_pixel_centers,
                           input.data(), rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | "
              << MakeResizeBilinearShape(bench) << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, 1)) {
      std::cerr << "ResizeBilinear<uint8> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void RunResizeBilinearBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/resize_bilinear.h",
      "Follow-up P1 closure for the generic float bilinear kernel and the "
      "uint8 generic-small-channel interpolation path.");
  BenchmarkResizeBilinearFloat();
  BenchmarkResizeBilinearUint8();
}
