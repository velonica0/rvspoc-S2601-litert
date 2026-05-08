void BenchmarkLookupTableUint8() {
  const VectorCase cases[] = {
      {"lut_u8_1k", 1024},
      {"lut_u8_4k", 4096},
      {"lut_u8_16k", 16384},
      {"lut_u8_64k", 65536},
  };
  BenchmarkLookupTableOp<uint8_t>(
      "LookupTable<uint8>",
      "`integer_ops/lut.h` uint8 lookup-table path.",
      cases, sizeof(cases) / sizeof(cases[0]), 0, 255, 0,
      [](const uint8_t* input, int size, const uint8_t* lut, uint8_t* output) {
        ScalarLookupTableUint8Operator(size, input, lut, output);
      },
      [](const uint8_t* input, int size, const uint8_t* lut, uint8_t* output) {
        RvvLookupTableUint8Operator(size, input, lut, output);
      });
}

void BenchmarkLookupTableInt8() {
  const VectorCase cases[] = {
      {"lut_i8_1k", 1024},
      {"lut_i8_4k", 4096},
      {"lut_i8_16k", 16384},
      {"lut_i8_64k", 65536},
  };
  BenchmarkLookupTableOp<int8_t>(
      "LookupTable<int8>",
      "`integer_ops/lut.h` int8 lookup-table path.",
      cases, sizeof(cases) / sizeof(cases[0]),
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
      0,
      [](const int8_t* input, int size, const int8_t* lut, int8_t* output) {
        ScalarLookupTableInt8Operator(size, input, lut, output);
      },
      [](const int8_t* input, int size, const int8_t* lut, int8_t* output) {
        RvvLookupTableInt8Operator(size, input, lut, output);
      });
}

void RunIntegerLutBenchmarks() {
  PrintSourceFileSection(
      "tflite/kernels/internal/optimized/integer_ops/lut.h",
      "This section keeps the uint8 and int8 lookup-table kernels separate "
      "because they were vectorized in the same commit but have different "
      "input domains.");
  BenchmarkLookupTableUint8();
  BenchmarkLookupTableInt8();
}
