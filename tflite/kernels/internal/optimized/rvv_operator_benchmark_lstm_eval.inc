void BenchmarkLstmGateFloatOperator() {
  const LstmGateCase cases[] = {
      {"lstm_gate_small_b1", 1, 128, 128, 512},
      {"lstm_gate_small_b4", 4, 128, 128, 512},
      {"lstm_gate_large_b1", 1, 256, 256, 1024},
  };

  PrintBenchmarkTitle(
      "LstmGate<float>",
      "`lstm_eval` float gate path "
      "(`input_to_gate` + `recurrent_to_gate` + sigmoid).");
  PrintFloatHeader("Shape");

  for (int idx = 0; idx < static_cast<int>(sizeof(cases) / sizeof(cases[0]));
       ++idx) {
    const LstmGateCase& bench = cases[idx];
    const double ops_per_iter =
        2.0 * static_cast<double>(bench.n_batch) *
        static_cast<double>(bench.n_cell) *
        static_cast<double>(bench.n_input + bench.n_output);
    const int iterations =
        ChooseIterations(static_cast<int64_t>(std::max(1.0, ops_per_iter)));
    std::mt19937 rng(0x3456 + idx * 131);
    const std::vector<float> input =
        MakeRandomFloatVector(bench.n_batch * bench.n_input, &rng);
    const std::vector<float> input_to_gate_weights =
        MakeRandomFloatVector(bench.n_cell * bench.n_input, &rng, -0.25f,
                              0.25f);
    const std::vector<float> output_state =
        MakeRandomFloatVector(bench.n_batch * bench.n_output, &rng);
    const std::vector<float> recurrent_to_gate_weights =
        MakeRandomFloatVector(bench.n_cell * bench.n_output, &rng, -0.25f,
                              0.25f);
    const std::vector<float> gate_bias =
        MakeRandomFloatVector(bench.n_cell, &rng, -0.1f, 0.1f);
    std::vector<float> scalar_gate(bench.n_batch * bench.n_cell, 0.0f);
    std::vector<float> rvv_gate(bench.n_batch * bench.n_cell, 0.0f);
    std::vector<float> scalar_scratch(bench.n_batch * bench.n_cell, 0.0f);
    std::vector<float> rvv_scratch(bench.n_batch * bench.n_cell, 0.0f);

    ScalarLstmGateFloatOperator(
        bench.n_batch, bench.n_input, bench.n_output, bench.n_cell,
        input.data(), input_to_gate_weights.data(), output_state.data(),
        recurrent_to_gate_weights.data(), gate_bias.data(), scalar_gate.data(),
        scalar_scratch.data());
    RvvLstmGateFloatOperator(
        bench.n_batch, bench.n_input, bench.n_output, bench.n_cell,
        input.data(), input_to_gate_weights.data(), output_state.data(),
        recurrent_to_gate_weights.data(), gate_bias.data(), rvv_gate.data(),
        rvv_scratch.data());
    const FloatAccuracy accuracy = CompareFloatVectors(scalar_gate, rvv_gate);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, ops_per_iter, [&]() -> double {
          ScalarLstmGateFloatOperator(
              bench.n_batch, bench.n_input, bench.n_output, bench.n_cell,
              input.data(), input_to_gate_weights.data(), output_state.data(),
              recurrent_to_gate_weights.data(), gate_bias.data(),
              scalar_gate.data(), scalar_scratch.data());
          return scalar_gate.empty() ? 0.0 : scalar_gate[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, ops_per_iter, [&]() -> double {
          RvvLstmGateFloatOperator(
              bench.n_batch, bench.n_input, bench.n_output, bench.n_cell,
              input.data(), input_to_gate_weights.data(), output_state.data(),
              recurrent_to_gate_weights.data(), gate_bias.data(),
              rvv_gate.data(), rvv_scratch.data());
          return rvv_gate.empty() ? 0.0 : rvv_gate[0];
        });

    std::cout << "| " << bench.name << " | " << MakeLstmGateShape(bench)
              << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";

    if (!FloatAccuracyWithinLstmTolerance(accuracy)) {
      std::cerr << "LstmGate<float> accuracy check failed for " << bench.name
                << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void BenchmarkLstmOutputFloatOperator() {
  const LstmOutputCase cases[] = {
      {"lstm_output_small_b1", 1, 512, 128},
      {"lstm_output_small_b4", 4, 512, 128},
      {"lstm_output_large_b1", 1, 1024, 256},
  };

  PrintBenchmarkTitle(
      "LstmOutput<float>",
      "`lstm_eval` float output/projection path "
      "(`tanh(cell)` + output gate + projection).");
  PrintFloatHeader("Shape");

  for (int idx = 0; idx < static_cast<int>(sizeof(cases) / sizeof(cases[0]));
       ++idx) {
    const LstmOutputCase& bench = cases[idx];
    const double ops_per_iter =
        2.0 * static_cast<double>(bench.n_batch) *
        static_cast<double>(bench.n_cell) *
        static_cast<double>(bench.n_output) +
        static_cast<double>(bench.n_batch) * bench.n_cell;
    const int iterations =
        ChooseIterations(static_cast<int64_t>(std::max(1.0, ops_per_iter)));
    std::mt19937 rng(0x4567 + idx * 149);
    const std::vector<float> cell_state =
        MakeRandomFloatVector(bench.n_batch * bench.n_cell, &rng, -2.0f, 2.0f);
    const std::vector<float> output_gate =
        MakeRandomFloatVector(bench.n_batch * bench.n_cell, &rng);
    const std::vector<float> projection_weights =
        MakeRandomFloatVector(bench.n_output * bench.n_cell, &rng, -0.25f,
                              0.25f);
    const std::vector<float> projection_bias =
        MakeRandomFloatVector(bench.n_output, &rng, -0.1f, 0.1f);
    std::vector<float> scalar_output(bench.n_batch * bench.n_output, 0.0f);
    std::vector<float> rvv_output(bench.n_batch * bench.n_output, 0.0f);
    std::vector<float> scalar_scratch(bench.n_batch * bench.n_cell, 0.0f);
    std::vector<float> rvv_scratch(bench.n_batch * bench.n_cell, 0.0f);
    std::vector<float> scalar_bias_scratch(bench.n_batch * bench.n_output,
                                           0.0f);
    std::vector<float> rvv_bias_scratch(bench.n_batch * bench.n_output, 0.0f);

    ScalarLstmOutputFloatOperator(
        bench.n_batch, bench.n_cell, bench.n_output, cell_state.data(),
        output_gate.data(), projection_weights.data(), projection_bias.data(),
        scalar_output.data(), scalar_scratch.data(),
        scalar_bias_scratch.data());
    RvvLstmOutputFloatOperator(
        bench.n_batch, bench.n_cell, bench.n_output, cell_state.data(),
        output_gate.data(), projection_weights.data(), projection_bias.data(),
        rvv_output.data(), rvv_scratch.data(), rvv_bias_scratch.data());
    const FloatAccuracy accuracy =
        CompareFloatVectors(scalar_output, rvv_output);

    const BenchmarkStats scalar_stats =
        RunBenchmark(iterations, ops_per_iter, [&]() -> double {
          ScalarLstmOutputFloatOperator(
              bench.n_batch, bench.n_cell, bench.n_output, cell_state.data(),
              output_gate.data(), projection_weights.data(),
              projection_bias.data(), scalar_output.data(),
              scalar_scratch.data(), scalar_bias_scratch.data());
          return scalar_output.empty() ? 0.0 : scalar_output[0];
        });
    const BenchmarkStats rvv_stats =
        RunBenchmark(iterations, ops_per_iter, [&]() -> double {
          RvvLstmOutputFloatOperator(
              bench.n_batch, bench.n_cell, bench.n_output, cell_state.data(),
              output_gate.data(), projection_weights.data(),
              projection_bias.data(), rvv_output.data(), rvv_scratch.data(),
              rvv_bias_scratch.data());
          return rvv_output.empty() ? 0.0 : rvv_output[0];
        });

    std::cout << "| " << bench.name << " | " << MakeLstmOutputShape(bench)
              << " | " << iterations << " | " << std::fixed
              << std::setprecision(2) << scalar_stats.mean_us << " | "
              << rvv_stats.mean_us << " | "
              << (scalar_stats.mean_us / rvv_stats.mean_us) << " | "
              << std::setprecision(3) << scalar_stats.gops << " | "
              << rvv_stats.gops << " | " << std::setprecision(8)
              << accuracy.max_abs_diff << " | " << accuracy.max_rel_diff
              << " | " << std::setprecision(10) << accuracy.mean_abs_diff
              << " |\n";

    if (!FloatAccuracyWithinLstmTolerance(accuracy)) {
      std::cerr << "LstmOutput<float> accuracy check failed for "
                << bench.name << "\n";
      std::exit(1);
    }
  }

  std::cout << "\n";
}

void RunLstmEvalBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/lstm_eval.cc",
      "Float gate and output/projection paths now reuse the RVV-enabled "
      "`tensor_utils` matvec helpers, so this section keeps the operator-level "
      "numbers separate from generic helper benchmarks.");
  BenchmarkLstmGateFloatOperator();
  BenchmarkLstmOutputFloatOperator();
}
