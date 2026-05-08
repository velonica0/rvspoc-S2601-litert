struct ReduceFloatMeanCase {
  const char* name;
  int rows;
  int cols;
};

std::string MakeReduceFloatMeanShape(const ReduceFloatMeanCase& bench) {
  return "R" + std::to_string(bench.rows) + " C" +
         std::to_string(bench.cols) + " axis(1)";
}

template <typename T, typename ScalarRunner, typename RvvRunner>
void BenchmarkReduceMeanQuantizedOp(const char* title, const char* affects,
                                    const MeanCase* cases, int num_cases,
                                    int allowed_max_abs_diff,
                                    ScalarRunner scalar_runner,
                                    RvvRunner rvv_runner) {
  PrintBenchmarkTitle(title, affects);
  PrintIntegerHeader("Shape");

  for (int idx = 0; idx < num_cases; ++idx) {
    const MeanCase& bench = cases[idx];
    const int input_size =
        bench.n_batch * bench.input_height * bench.input_width * bench.depth;
    const int output_size = bench.n_batch * bench.depth;
    const int64_t ops_per_iter = static_cast<int64_t>(output_size) *
                                 bench.input_height * bench.input_width;
    const int iterations = ChooseIterations(ops_per_iter);
    std::mt19937 rng(0xDADA + idx * 41 + bench.depth * 3);
    const std::vector<T> input = MakeRandomIntegralVector<T>(
        input_size, &rng, std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max());
    std::vector<T> scalar_output(output_size, 0);
    std::vector<T> rvv_output(output_size, 0);

    scalar_runner(bench, input.data(), scalar_output.data());
    rvv_runner(bench, input.data(), rvv_output.data());
    const IntegerAccuracy accuracy =
        CompareIntegerVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       scalar_runner(bench, input.data(), scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       rvv_runner(bench, input.data(), rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | " << MakeMeanShape(bench) << " | "
              << iterations << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << accuracy.max_abs_diff << " | "
              << accuracy.mismatches << " |\n";

    if (!IntegerAccuracyWithinTolerance(accuracy, allowed_max_abs_diff)) {
      std::cerr << title << " accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkMeanUint8Reduce() {
  const MeanCase cases[] = {
      {"mean_u8_8x8x48", 1, 8, 8, 48},
      {"mean_u8_16x16x96", 1, 16, 16, 96},
      {"mean_u8_32x8x160", 1, 32, 8, 160},
  };
  BenchmarkReduceMeanQuantizedOp<uint8_t>(
      "MeanImpl<uint8>",
      "`reduce.h` 4D keep-dims height-width reduction path for uint8.",
      cases, sizeof(cases) / sizeof(cases[0]),
      /*allowed_max_abs_diff=*/1,
      [](const MeanCase& bench, const uint8_t* input, uint8_t* output) {
        ScalarMeanUint8Operator(bench.n_batch, bench.input_height,
                                bench.input_width, bench.depth, input, output);
      },
      [](const MeanCase& bench, const uint8_t* input, uint8_t* output) {
        RvvMeanUint8Operator(bench.n_batch, bench.input_height,
                             bench.input_width, bench.depth, input, output);
      });
}

void BenchmarkMeanFloatLastDimReduce() {
  const ReduceFloatMeanCase cases[] = {
      {"mean_f32_lastdim_256x48", 256, 48},
      {"mean_f32_lastdim_256x192", 256, 192},
      {"mean_f32_lastdim_128x640", 128, 640},
  };
  PrintBenchmarkTitle(
      "Mean<float> last-dim",
      "`reduce.h` float specialization that reduces only the last dimension.");
  PrintFloatHeader("Shape");

  for (int idx = 0; idx < static_cast<int>(sizeof(cases) / sizeof(cases[0]));
       ++idx) {
    const ReduceFloatMeanCase& bench = cases[idx];
    const int input_size = bench.rows * bench.cols;
    const int output_size = bench.rows;
    const int64_t ops_per_iter = static_cast<int64_t>(input_size);
    const int iterations = ChooseIterations(ops_per_iter);
    std::mt19937 rng(0xEAEA + idx * 47 + bench.cols);
    const std::vector<float> input =
        MakeRandomFloatVector(input_size, &rng, -4.0f, 4.0f);
    std::vector<float> scalar_output(output_size, 0.0f);
    std::vector<float> rvv_output(output_size, 0.0f);

    ScalarMeanFloatLastDimOperator(bench.rows, bench.cols, input.data(),
                                   scalar_output.data());
    RvvMeanFloatLastDimOperator(bench.rows, bench.cols, input.data(),
                                rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       ScalarMeanFloatLastDimOperator(
                           bench.rows, bench.cols, input.data(),
                           scalar_output.data());
                       return scalar_output.empty() ? 0.0 : scalar_output[0];
                     });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, static_cast<double>(ops_per_iter),
                     [&]() -> double {
                       RvvMeanFloatLastDimOperator(bench.rows, bench.cols,
                                                   input.data(),
                                                   rvv_output.data());
                       return rvv_output.empty() ? 0.0 : rvv_output[0];
                     });

    std::cout << "| " << bench.name << " | "
              << MakeReduceFloatMeanShape(bench) << " | " << iterations
              << " | " << std::fixed << std::setprecision(2)
              << scalar_stats.mean_us << " | " << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";

    if (!FloatAccuracyWithinCodeXOperatorTolerance(accuracy)) {
      std::cerr << "Mean<float> last-dim accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void RunReduceBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/reduce.h",
      "Follow-up P1 closure for the uint8 keep-dims height-width reduction "
      "path and the float last-dimension mean specialization.");
  BenchmarkMeanUint8Reduce();
  BenchmarkMeanFloatLastDimReduce();
}
