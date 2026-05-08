void BenchmarkFullyConnectedFloat() {
  const FullyConnectedCase cases[] = {
      {"fc_dense_small_b4", 4, 64, 64},
      {"fc_dense_mid_b4", 4, 128, 128},
      {"fc_dense_large_b4", 4, 2048, 640},
  };

  PrintBenchmarkTitle(
      "FullyConnected<float> dense EvalPie route",
      "`fully_connected.cc` dense float RVV dispatch that now routes through "
      "`EvalPie`; this benchmark mirrors that matrix-batch-vector "
      "accumulation path inside the split operator harness.");
  PrintFloatHeader("Shape");

  for (int idx = 0; idx < static_cast<int>(sizeof(cases) / sizeof(cases[0]));
       ++idx) {
    const FullyConnectedCase& bench = cases[idx];
    const int input_count = bench.n_batch * bench.input_size;
    const int weight_count = bench.input_size * bench.num_units;
    const int output_count = bench.n_batch * bench.num_units;
    const double ops_per_iter = 2.0 * bench.n_batch * bench.input_size *
                                bench.num_units;
    const int iterations =
        ChooseIterations(static_cast<int64_t>(std::max(1.0, ops_per_iter)));

    std::mt19937 rng(0xF00D + idx * 173);
    const std::vector<float> input =
        MakeRandomFloatVector(input_count, &rng, -1.0f, 1.0f);
    const std::vector<float> weights =
        MakeRandomFloatVector(weight_count, &rng, -0.25f, 0.25f);
    const std::vector<float> bias =
        MakeRandomFloatVector(bench.num_units, &rng, -0.1f, 0.1f);
    std::vector<float> scalar_output(output_count, 0.0f);
    std::vector<float> rvv_output(output_count, 0.0f);

    ScalarFullyConnectedFloatOperator(
        bench.n_batch, bench.input_size, bench.num_units, input.data(),
        weights.data(), bias.data(), scalar_output.data());
    RvvFullyConnectedFloatOperator(
        bench.n_batch, bench.input_size, bench.num_units, input.data(),
        weights.data(), bias.data(), rvv_output.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, ops_per_iter, [&]() -> double {
          ScalarFullyConnectedFloatOperator(
              bench.n_batch, bench.input_size, bench.num_units, input.data(),
              weights.data(), bias.data(), scalar_output.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, ops_per_iter, [&]() -> double {
          RvvFullyConnectedFloatOperator(
              bench.n_batch, bench.input_size, bench.num_units, input.data(),
              weights.data(), bias.data(), rvv_output.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << MakeFullyConnectedShape(bench)
              << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";

    if (!FloatAccuracyWithinMatVecTolerance(accuracy)) {
      std::cerr << "FullyConnected<float> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void RunFullyConnectedBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/fully_connected.cc",
      "Dense float FullyConnected on RVV now takes the `EvalPie` route, so "
      "this split benchmark keeps the measurement alongside the source-file "
      "section and exercises the same matvec-style accumulation shape.");
  BenchmarkFullyConnectedFloat();
}
