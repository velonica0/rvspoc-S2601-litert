# RVV Scalar Benchmark Results

- Date: 2026-05-08 12:00:18 CST
- Host: Linux k3 6.18.3-generic #1.0.0~rc4.4 SMP PREEMPT_DYNAMIC Wed Apr 29 10:51:17 CST 2026 riscv64 GNU/Linux
- Compiler: g++ (Bianbu 15.2.0-16ubuntu1bb2) 15.2.0
- Scope: recent five commits that introduced or wired RVV coverage

## Recent Five Commits

| Commit | Date | Summary |
| --- | --- | --- |
| `8edc8e0d` | 2026-05-08 | integer_ops |
| `36a4fa2f` | 2026-05-08 | optimized_ops.h |
| `d447fe69` | 2026-05-07 | Portable 2 RVV |
| `84e86354` | 2026-05-07 | all rvv_tensor_utils ops |
| `82d3365e` | 2026-05-06 | rvv_tensor_utils_impl |

## Coverage Map

| Source file | Recent commit(s) | Benchmark coverage |
| --- | --- | --- |
| `tflite/kernels/internal/optimized/rvv_tensor_utils.cc` | `82d3365e`, `84e86354`, `d447fe69` | Shared helper-level scalar vs RVV coverage |
| `tflite/kernels/internal/optimized/optimized_ops.h` | `36a4fa2f` | Float arithmetic, affine quantize, uint8 pooling, HardSwish, ArgMin/ArgMax |
| `tflite/kernels/fully_connected.cc` | `36a4fa2f` | Dense float `EvalPie` route benchmark |
| `tflite/kernels/lstm_eval.cc` | `36a4fa2f` | Float gate and output/projection operator benchmarks |
| `tflite/kernels/internal/optimized/integer_ops/add.h` | `8edc8e0d` | int8/int16 add plus scalar-broadcast kernels |
| `tflite/kernels/internal/optimized/integer_ops/mul.h` | `8edc8e0d` | int8 mul plus scalar-broadcast kernels |
| `tflite/kernels/internal/optimized/integer_ops/sub.h` | `8edc8e0d` | int16 sub |
| `tflite/kernels/internal/optimized/integer_ops/pooling.h` | `8edc8e0d` | int8 average/max pooling |
| `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h` | `8edc8e0d` | int8 depthwise conv general and specialized kernels |
| `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h` | `8edc8e0d` | Dispatch-only; shares numeric coverage with `depthwise_conv.h` |
| `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h` | `8edc8e0d` | int16 LeakyReLU |
| `tflite/kernels/internal/optimized/integer_ops/lut.h` | `8edc8e0d` | uint8/int8 lookup table |
| `tflite/kernels/internal/optimized/integer_ops/mean.h` | `8edc8e0d` | int8 mean reduction |

## `tflite/kernels/internal/optimized/rvv_tensor_utils.cc`

- Summary: shared tensor_utils helper coverage accumulated across the first three of the recent five RVV commits.

- Runtime VLEN bits: 256
- Scalar kernels: `Portable*` compiled with `-fno-tree-vectorize -fno-tree-slp-vectorize`
- RVV kernels: `Rvv*` compiled with `-march=rv64gcv_zvl128b -mabi=lp64d`
- Covered helper families: zero-check, dot/reduction, dense matvec, sparse matvec, hybrid quantization, recurrent accumulate, layer-norm, cwise math, clipping, sub1, normalization, dequant-scale.

### IsZeroVector<float>

- Affects: zero-skip precheck paths in hybrid and recurrent helpers.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_f32_64 | 64 | 4000 | 0.11 | 0.06 | 1.96 | 0.569 | 1.114 | 0 | 0 |
| zero_f32_1024 | 1024 | 4000 | 1.76 | 0.91 | 1.93 | 0.582 | 1.121 | 0 | 0 |
| zero_f32_4096 | 4096 | 4000 | 7.01 | 3.62 | 1.94 | 0.584 | 1.132 | 0 | 0 |

### IsZeroVector<int8>

- Affects: zero-skip checks for quantized helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_i8_64 | 64 | 4000 | 0.03 | 0.02 | 2.00 | 2.007 | 4.019 | 0 | 0 |
| zero_i8_1024 | 1024 | 4000 | 0.48 | 0.22 | 2.17 | 2.142 | 4.645 | 0 | 0 |
| zero_i8_4096 | 4096 | 4000 | 1.89 | 0.88 | 2.14 | 2.171 | 4.639 | 0 | 0 |

### VectorVectorDotProduct<float>

- Affects: `fully_connected`, recurrent math, shared float tensor-utils call sites.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dot_64 | 64 | 4000 | 0.09 | 0.03 | 2.95 | 1.401 | 4.139 | 0.00000000 | 0.00000000 | 0.0000000000 |
| dot_256 | 256 | 4000 | 0.36 | 0.08 | 4.46 | 1.435 | 6.399 | 0.00000119 | 0.00000046 | 0.0000011921 |
| dot_1024 | 1024 | 4000 | 1.41 | 0.28 | 5.09 | 1.456 | 7.410 | 0.00000191 | 0.00000016 | 0.0000019073 |
| dot_4096 | 4096 | 4000 | 5.61 | 1.07 | 5.26 | 1.459 | 7.682 | 0.00002384 | 0.00000264 | 0.0000238419 |

### BatchVectorBatchVectorDotProduct<int16>

- Affects: `svdf` and batched recurrent dot-product helper paths.

| Case | Batch x size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_dot_q15_small | 4 x 64 | 4000 | 0.29 | 0.09 | 3.11 | 1.788 | 5.562 | 0 | 0 |
| svdf_dot_q15_mid | 8 x 256 | 4000 | 2.08 | 0.49 | 4.23 | 1.967 | 8.324 | 0 | 0 |
| svdf_dot_q15_large | 16 x 1024 | 4000 | 16.86 | 3.66 | 4.60 | 1.943 | 8.942 | 0 | 0 |

### ReductionSumVector<float>

- Affects: `svdf` and other float reduction-style helper paths.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_small | 64 x 8 | 4000 | 1.00 | 0.77 | 1.29 | 0.514 | 0.665 | 0.00000048 | 0.00000080 | 0.0000000559 |
| svdf_mid | 128 x 16 | 4000 | 4.25 | 1.76 | 2.41 | 0.482 | 1.161 | 0.00000095 | 0.00001488 | 0.0000001378 |
| svdf_large | 256 x 64 | 4000 | 35.06 | 6.31 | 5.55 | 0.467 | 2.595 | 0.00000572 | 0.00001453 | 0.0000005609 |
| reduce_wide | 256 x 256 | 2441 | 141.46 | 17.52 | 8.07 | 0.463 | 3.741 | 0.00001144 | 0.00016069 | 0.0000021656 |

### ReductionSumVector<int8 -> int32>

- Affects: `conv`, `batch_matmul`, `kernel_utils`, cached row sum preparation.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| row_sum_small | 64 x 32 | 4000 | 1.09 | 1.07 | 1.02 | 1.885 | 1.922 | 0 | 0 |
| row_sum_mid | 128 x 128 | 4000 | 5.66 | 4.73 | 1.20 | 2.895 | 3.466 | 0 | 0 |
| row_sum_large | 256 x 256 | 2441 | 20.64 | 16.66 | 1.24 | 3.175 | 3.933 | 0 | 0 |
| row_sum_conv_like | 512 x 512 | 610 | 79.58 | 61.14 | 1.30 | 3.294 | 4.288 | 0 | 0 |

### ReductionSumVector<int32>

- Affects: scalar accumulation helper paths such as reference SVDF and utility reductions.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reduce_i32_small | 64 x 32 | 4000 | 0.93 | 0.77 | 1.21 | 2.198 | 2.665 | 0 | 0 |
| reduce_i32_mid | 128 x 128 | 4000 | 4.65 | 3.02 | 1.54 | 3.521 | 5.427 | 0 | 0 |
| reduce_i32_large | 256 x 256 | 2441 | 16.73 | 10.69 | 1.57 | 3.916 | 6.130 | 0 | 0 |

### MatrixScalarMultiplyAccumulate<int8>

- Affects: quantized recurrent helpers and row-wise reduction paths.

| Case | Rows x cols | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_rows_64 | 64 x 64 | 4000 | 4.11 | 1.58 | 2.60 | 0.998 | 2.591 | 0 | 0 |
| fc_rows_256 | 256 x 128 | 4000 | 30.50 | 9.78 | 3.12 | 1.074 | 3.352 | 0 | 0 |
| conv_rows_512 | 512 x 256 | 1220 | 122.36 | 33.51 | 3.65 | 1.071 | 3.912 | 0 | 0 |
| conv_rows_1024 | 1024 x 512 | 305 | 483.77 | 122.96 | 3.93 | 1.084 | 4.264 | 0 | 0 |

### MatrixBatchVectorMultiplyAccumulate<float>

- Affects: float `fully_connected` and recurrent kernels.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_fc | 64 x 64 x 4 | 4000 | 24.27 | 7.23 | 3.36 | 1.350 | 4.531 | 0.00000286 | 0.00001540 | 0.0000003163 |
| mid_fc | 128 x 128 x 4 | 2441 | 87.61 | 24.10 | 3.63 | 1.496 | 5.438 | 0.00000525 | 0.00006299 | 0.0000006228 |
| lstm_like | 640 x 2048 x 4 | 30 | 7213.57 | 2400.59 | 3.00 | 1.454 | 4.368 | 0.00008774 | 0.00183174 | 0.0000094324 |
| sqrnn_like | 1024 x 1024 x 8 | 20 | 11577.09 | 3635.46 | 3.18 | 1.449 | 4.615 | 0.00005531 | 0.00156548 | 0.0000045446 |

### MatrixBatchVectorMultiplyAccumulate<int8>

- Affects: quantized `fully_connected`, hybrid recurrent helpers, shared int8 GEMV-style call sites.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_fc_small | 64 x 64 x 4 | 4000 | 16.53 | 6.98 | 2.37 | 1.983 | 4.698 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_fc_mid | 128 x 128 x 4 | 2441 | 70.31 | 20.93 | 3.36 | 1.864 | 6.263 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_lstm_like | 640 x 1024 x 4 | 61 | 2792.04 | 594.29 | 4.70 | 1.878 | 8.822 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_conv_like | 1024 x 512 x 8 | 38 | 4461.00 | 1007.25 | 4.43 | 1.880 | 8.328 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MatrixBatchVectorMultiplyAccumulate<int8, per-channel + input_offset>

- Affects: quantized `conv`, `batch_matmul`, and any path that uses cached row sums plus per-channel scale.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_row_sum_small | 64 x 64 x 4 | 4000 | 26.07 | 8.63 | 3.02 | 1.257 | 3.799 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_mid | 128 x 128 x 4 | 2441 | 100.01 | 25.91 | 3.86 | 1.311 | 5.059 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_conv | 256 x 576 x 8 | 135 | 1721.20 | 314.55 | 5.47 | 1.371 | 7.501 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_batch_matmul_like | 512 x 512 x 8 | 76 | 3062.19 | 566.84 | 5.40 | 1.370 | 7.399 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MatrixBatchVectorMultiplyAccumulate<int8 -> int16>

- Affects: quantized recurrent gate accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q15_small | 128 x 128 x 4 | 2441 | 76.41 | 21.45 | 3.56 | 1.715 | 6.112 | 0 | 0 |
| gate_q15_mid | 256 x 256 x 4 | 610 | 287.61 | 70.99 | 4.05 | 1.823 | 7.386 | 0 | 0 |
| gate_q15_large | 512 x 512 x 8 | 76 | 2298.52 | 507.39 | 4.53 | 1.825 | 8.266 | 0 | 0 |

### MatrixBatchVectorMultiplyAccumulate<int8 -> int8>

- Affects: quantized projection and low-precision recurrent accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q8_small | 128 x 128 x 4 | 2441 | 77.65 | 21.63 | 3.59 | 1.688 | 6.058 | 0 | 0 |
| gate_q8_mid | 256 x 256 x 4 | 610 | 292.22 | 71.40 | 4.09 | 1.794 | 7.343 | 0 | 0 |
| gate_q8_large | 512 x 512 x 8 | 76 | 2250.69 | 509.54 | 4.42 | 1.864 | 8.232 | 0 | 0 |

### MatrixBatchVectorMultiply<int8 -> int8>

- Affects: quantized LSTM gate matmul paths before saturating-add and activation.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_noacc_q8_small | 128 x 128 x 4 | 2441 | 81.96 | 30.55 | 2.68 | 1.599 | 4.291 | 0 | 0 |
| gate_noacc_q8_mid | 256 x 256 x 4 | 610 | 291.25 | 100.75 | 2.89 | 1.800 | 5.204 | 0 | 0 |
| gate_noacc_q8_large | 512 x 512 x 8 | 76 | 2225.43 | 701.78 | 3.17 | 1.885 | 5.977 | 0 | 0 |

### MatrixBatchVectorMultiply<int16 x int8 -> int8>

- Affects: quantized projection/output matmul helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| proj_q8_small | 128 x 128 x 4 | 2441 | 420.84 | 32.28 | 13.04 | 0.311 | 4.060 | 0 | 0 |
| proj_q8_mid | 256 x 256 x 4 | 610 | 1663.75 | 117.47 | 14.16 | 0.315 | 4.463 | 0 | 0 |
| proj_q8_large | 512 x 512 x 8 | 76 | 13291.97 | 894.91 | 14.85 | 0.316 | 4.687 | 0 | 0 |

### SparseMatrixBatchVectorMultiplyAccumulate1x4<float>

- Affects: sparse float `fully_connected` and sparse recurrent helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_1x4_small | 64 x 256 x 4 | 4000 | 22.56 | 11.90 | 1.90 | 1.438 | 2.727 | 0.00000143 | 0.00012937 | 0.0000003011 |
| sparse_1x4_mid | 128 x 512 x 4 | 2448 | 97.87 | 50.33 | 1.94 | 1.335 | 2.597 | 0.00000477 | 0.00038041 | 0.0000006075 |
| sparse_1x4_large | 256 x 1024 x 8 | 304 | 754.54 | 373.93 | 2.02 | 1.392 | 2.809 | 0.00000954 | 0.00038029 | 0.0000011599 |

### SparseMatrixBatchVectorMultiplyAccumulate<float ledger>

- Affects: sparse float matvec helper paths with ledger format.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_ledger_small | 64 x 256 x 4 | 4000 | 24.97 | 11.68 | 2.14 | 1.292 | 2.761 | 0.00000238 | 0.00000709 | 0.0000003208 |
| sparse_ledger_mid | 128 x 512 x 4 | 2362 | 103.26 | 41.66 | 2.48 | 1.311 | 3.251 | 0.00000572 | 0.00038695 | 0.0000006152 |
| sparse_ledger_large | 256 x 1024 x 8 | 303 | 809.38 | 294.76 | 2.75 | 1.305 | 3.583 | 0.00001240 | 0.00644636 | 0.0000011960 |

### SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>

- Affects: sparse quantized matvec helper paths that dequantize to float.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_qfloat_small | 64 x 256 x 4 | 4000 | 15.54 | 10.38 | 1.50 | 2.084 | 3.121 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_mid | 128 x 512 x 4 | 2465 | 58.86 | 32.97 | 1.79 | 2.205 | 3.937 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_large | 256 x 1024 x 8 | 299 | 483.67 | 234.01 | 2.07 | 2.209 | 4.566 | 0.00000000 | 0.00000000 | 0.0000000000 |

### SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>

- Affects: sparse quantized fully-connected and recurrent output helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_q8_small | 64 x 256 x 4 | 4000 | 33.83 | 15.73 | 2.15 | 1.010 | 2.172 | 0 | 0 |
| sparse_q8_mid | 128 x 512 x 4 | 2394 | 125.89 | 46.89 | 2.68 | 1.061 | 2.850 | 0 | 0 |
| sparse_q8_large | 256 x 1024 x 8 | 302 | 944.17 | 286.58 | 3.29 | 1.121 | 3.692 | 0 | 0 |

### SymmetricQuantizeFloats

- Affects: hybrid `fully_connected`, `batch_matmul`, weight and activation pre-quant helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Min diff | Max diff | Scale diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sym_quant_small | 64 | 4000 | 0.46 | 0.16 | 2.81 | 0.421 | 1.186 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_mid | 1024 | 4000 | 7.26 | 2.68 | 2.71 | 0.423 | 1.148 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_large | 4096 | 4000 | 32.40 | 15.30 | 2.12 | 0.379 | 0.803 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |

### AsymmetricQuantizeFloats

- Affects: hybrid `conv`, `depthwise_conv`, `transpose_conv`, and int8 input staging paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Scale diff | Offset diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asym_quant_small | 64 | 4000 | 0.57 | 0.25 | 2.31 | 0.335 | 0.774 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_mid | 1024 | 4000 | 8.72 | 3.27 | 2.66 | 0.352 | 0.938 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_large | 4096 | 4000 | 37.45 | 16.36 | 2.29 | 0.328 | 0.751 | 0 | 0 | 0.00000000 | 0 |

### ApplyLayerNorm<int16>

- Affects: quantized recurrent gate normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ln_small | 4 x 64 | 4000 | 5.89 | 2.38 | 2.47 | 0.347 | 0.860 | 0 | 0 |
| ln_mid | 4 x 256 | 4000 | 24.10 | 8.65 | 2.79 | 0.340 | 0.947 | 0 | 0 |
| ln_gate_like | 8 x 1024 | 2441 | 192.88 | 67.46 | 2.86 | 0.340 | 0.972 | 0 | 0 |

### ApplySigmoid<int16>

- Affects: quantized recurrent gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoid_small | 4 x 64 | 4000 | 22.28 | 8.62 | 2.58 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_mid | 8 x 256 | 4000 | 192.30 | 68.86 | 2.79 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_large | 4 x 1024 | 4000 | 386.37 | 137.76 | 2.80 | 0.011 | 0.030 | 0 | 0 |

### ApplyTanh<int16>

- Affects: quantized recurrent state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanh_q0_small | 0 x 4 x 64 | 4000 | 21.23 | 8.25 | 2.57 | 0.012 | 0.031 | 0 | 0 |
| tanh_q3_mid | 3 x 8 x 256 | 4000 | 200.69 | 71.81 | 2.79 | 0.010 | 0.029 | 0 | 0 |
| tanh_q4_large | 4 x 4 x 1024 | 4000 | 405.26 | 150.33 | 2.70 | 0.010 | 0.027 | 0 | 0 |

### ApplyLayerNormFloat<int16>

- Affects: float-reference layer-norm helper paths used by quantized LSTM eval.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lnf_small | 4 x 64 | 4000 | 3.69 | 1.47 | 2.51 | 0.555 | 1.392 | 0 | 0 |
| lnf_mid | 4 x 256 | 4000 | 13.90 | 5.28 | 2.63 | 0.589 | 1.551 | 0 | 0 |
| lnf_gate_like | 8 x 1024 | 2441 | 109.33 | 40.93 | 2.67 | 0.599 | 1.601 | 0 | 0 |

### ApplySigmoidFloat<int16>

- Affects: float-reference gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoidf_small | 4 x 64 | 4000 | 9.38 | 8.62 | 1.09 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 2.0703125000 |
| sigmoidf_mid | 8 x 256 | 4000 | 74.77 | 68.86 | 1.09 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 1.9697265625 |
| sigmoidf_large | 4 x 1024 | 4000 | 149.29 | 137.72 | 1.08 | 0.027 | 0.030 | 6.00000000 | 0.21052632 | 1.9594726562 |

### ApplyTanhFloat<int16>

- Affects: float-reference state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanhf_qm12_small | -12 x 4 x 64 | 4000 | 24.79 | 8.90 | 2.79 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm12_mid | -12 x 8 x 256 | 4000 | 197.79 | 70.15 | 2.82 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm15_large | -15 x 4 x 1024 | 4000 | 395.42 | 140.14 | 2.82 | 0.010 | 0.029 | 0 | 0 |

### CwiseMul<int16 -> int16>

- Affects: quantized recurrent elementwise gate math.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q15_small | 4 x 64 | 4000 | 0.57 | 0.44 | 1.30 | 0.449 | 0.585 | 0 | 0 |
| mul_q15_mid | 8 x 256 | 4000 | 4.35 | 3.47 | 1.26 | 0.470 | 0.591 | 0 | 0 |
| mul_q15_large | 4 x 1024 | 4000 | 8.66 | 6.93 | 1.25 | 0.473 | 0.591 | 0 | 0 |

### CwiseMul<int16 -> int8>

- Affects: quantized projection and low-precision elementwise paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q8_small | 4 x 64 | 4000 | 2.37 | 0.77 | 3.07 | 0.108 | 0.332 | 0 | 0 |
| mul_q8_mid | 8 x 256 | 4000 | 23.55 | 6.03 | 3.90 | 0.087 | 0.340 | 0 | 0 |
| mul_q8_large | 4 x 1024 | 4000 | 49.07 | 12.06 | 4.07 | 0.083 | 0.340 | 0 | 0 |

### CwiseAdd<int16>

- Affects: recurrent residual/additive helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_small | 4 x 64 | 4000 | 0.73 | 0.33 | 2.19 | 0.350 | 0.767 | 0 | 0 |
| add_mid | 8 x 256 | 4000 | 6.33 | 2.61 | 2.42 | 0.324 | 0.784 | 0 | 0 |
| add_large | 4 x 1024 | 4000 | 15.12 | 5.25 | 2.88 | 0.271 | 0.780 | 0 | 0 |

### TwoGateSaturatingAdd<int8 -> int16>

- Affects: quantized LSTM gate merge helper paths.

| Case | Batch x cell | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| two_gate_small | 4 x 64 | 4000 | 4.16 | 1.30 | 3.20 | 0.246 | 0.787 | 0 | 0 |
| two_gate_mid | 8 x 256 | 4000 | 42.13 | 10.24 | 4.11 | 0.194 | 0.800 | 0 | 0 |
| two_gate_large | 4 x 1024 | 4000 | 84.09 | 20.46 | 4.11 | 0.195 | 0.801 | 0 | 0 |

### CwiseClipping<float>

- Affects: activation clamp paths in float recurrent/helper code.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_f32_64 | 64 | 4000 | 0.07 | 0.05 | 1.54 | 0.880 | 1.353 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_1024 | 1024 | 4000 | 1.17 | 0.76 | 1.54 | 0.879 | 1.352 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_4096 | 4096 | 4000 | 4.67 | 3.04 | 1.54 | 0.877 | 1.347 | 0.00000000 | 0.00000000 | 0.0000000000 |

### CwiseClipping<int16>

- Affects: quantized activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i16_64 | 64 | 4000 | 0.02 | 0.02 | 1.00 | 3.195 | 3.195 | 0 | 0 |
| clip_i16_1024 | 1024 | 4000 | 0.32 | 0.32 | 1.00 | 3.200 | 3.200 | 0 | 0 |
| clip_i16_4096 | 4096 | 4000 | 1.29 | 1.29 | 1.00 | 3.170 | 3.167 | 0 | 0 |

### CwiseClipping<int8>

- Affects: low-precision activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i8_64 | 64 | 4000 | 0.01 | 0.02 | 0.93 | 4.462 | 4.160 | 0 | 0 |
| clip_i8_1024 | 1024 | 4000 | 0.16 | 0.16 | 1.00 | 6.399 | 6.400 | 0 | 0 |
| clip_i8_4096 | 4096 | 4000 | 0.64 | 0.64 | 1.00 | 6.373 | 6.389 | 0 | 0 |

### VectorBatchVectorCwiseProductAccumulate<int16>

- Affects: quantized recurrent gate/state accumulation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vbv_cwise_small | 4 x 64 | 4000 | 2.30 | 0.81 | 2.85 | 0.223 | 0.634 | 0 | 0 |
| vbv_cwise_mid | 8 x 256 | 4000 | 23.04 | 6.30 | 3.66 | 0.178 | 0.650 | 0 | 0 |
| vbv_cwise_large | 4 x 1024 | 4000 | 47.12 | 12.60 | 3.74 | 0.174 | 0.650 | 0 | 0 |

### Sub1Vector<float>

- Affects: float recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_f32_64 | 64 | 4000 | 0.06 | 0.04 | 1.70 | 1.035 | 1.759 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_1024 | 1024 | 4000 | 0.94 | 0.58 | 1.62 | 1.087 | 1.760 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_4096 | 4096 | 4000 | 3.73 | 2.35 | 1.59 | 1.097 | 1.745 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Sub1Vector<int16>

- Affects: quantized recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_i16_64 | 64 | 4000 | 0.06 | 0.02 | 3.37 | 1.042 | 3.515 | 0 | 0 |
| sub1_i16_1024 | 1024 | 4000 | 0.94 | 0.29 | 3.23 | 1.090 | 3.520 | 0 | 0 |
| sub1_i16_4096 | 4096 | 4000 | 3.74 | 1.18 | 3.17 | 1.096 | 3.472 | 0 | 0 |

### VectorScalarMultiply<int8 -> float>

- Affects: hybrid dequant-style helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale_i8_64 | 64 | 4000 | 0.09 | 0.02 | 4.30 | 1.422 | 6.114 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_1024 | 1024 | 4000 | 1.41 | 0.32 | 4.34 | 1.456 | 6.323 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_4096 | 4096 | 4000 | 5.60 | 1.28 | 4.36 | 1.462 | 6.379 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MeanStddevNormalization<float>

- Affects: float recurrent normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| norm_1x64 | 1 x 64 | 4000 | 0.27 | 0.21 | 1.28 | 0.943 | 1.211 | 0.00000024 | 0.00000023 | 0.0000000843 |
| norm_4x256 | 4 x 256 | 4000 | 3.94 | 2.89 | 1.36 | 1.039 | 1.419 | 0.00000048 | 0.00002272 | 0.0000001109 |
| norm_8x1024 | 8 x 1024 | 4000 | 30.23 | 21.88 | 1.38 | 1.084 | 1.497 | 0.00000083 | 0.00009705 | 0.0000001304 |


## `tflite/kernels/internal/optimized/optimized_ops.h`

- Summary: Float arithmetic fast paths, quantize/dequant-style helpers, uint8 pooling, quantized HardSwish, and argmin/argmax cases that gained RVV coverage in the latest operator-level commits.

### Add<float>

- Affects: `add` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_f32_1k | 1024 | 4000 | 1.89 | 0.33 | 5.69 | 0.542 | 3.085 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_4k | 4096 | 4000 | 7.51 | 1.27 | 5.90 | 0.545 | 3.218 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_16k | 16384 | 4000 | 29.91 | 7.35 | 4.07 | 0.548 | 2.229 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_64k | 65536 | 1953 | 135.79 | 28.00 | 4.85 | 0.483 | 2.341 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AddScalarBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float add kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_f32_1k | 1024 | 4000 | 1.87 | 0.26 | 7.13 | 0.546 | 3.898 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_4k | 4096 | 4000 | 7.49 | 1.05 | 7.12 | 0.547 | 3.892 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_16k | 16384 | 4000 | 31.02 | 4.26 | 7.28 | 0.528 | 3.846 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_64k | 65536 | 1953 | 124.26 | 16.85 | 7.37 | 0.527 | 3.888 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastAddDispatch<float> scalar lhs

- Affects: `add` float scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_f32_1k | 1024 | 4000 | 2.40 | 0.33 | 7.21 | 0.426 | 3.074 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_4k | 4096 | 4000 | 9.40 | 1.12 | 8.36 | 0.436 | 3.641 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_16k | 16384 | 4000 | 37.72 | 4.44 | 8.49 | 0.434 | 3.689 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_64k | 65536 | 1953 | 149.43 | 16.95 | 8.82 | 0.439 | 3.866 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Mul<float>

- Affects: `mul` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_f32_1k | 1024 | 4000 | 2.36 | 0.36 | 6.52 | 0.433 | 2.827 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_4k | 4096 | 4000 | 9.36 | 1.41 | 6.66 | 0.437 | 2.913 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_16k | 16384 | 4000 | 37.37 | 7.46 | 5.01 | 0.438 | 2.196 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_64k | 65536 | 1953 | 149.26 | 26.98 | 5.53 | 0.439 | 2.429 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MulSimpleBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float mul kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_f32_1k | 1024 | 4000 | 2.35 | 0.29 | 8.06 | 0.437 | 3.520 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_4k | 4096 | 4000 | 9.34 | 1.16 | 8.02 | 0.439 | 3.517 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_16k | 16384 | 4000 | 37.42 | 4.69 | 7.98 | 0.438 | 3.493 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_64k | 65536 | 1953 | 149.43 | 18.71 | 7.99 | 0.439 | 3.503 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastMulDispatch<float> scalar lhs

- Affects: `mul` float scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_f32_1k | 1024 | 4000 | 2.49 | 0.35 | 7.11 | 0.411 | 2.922 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_4k | 4096 | 4000 | 9.71 | 1.24 | 7.86 | 0.422 | 3.314 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_16k | 16384 | 4000 | 38.92 | 4.93 | 7.90 | 0.421 | 3.325 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_64k | 65536 | 1953 | 154.01 | 18.85 | 8.17 | 0.426 | 3.477 | 0.00000000 | 0.00000000 | 0.0000000000 |

### SubWithActivation<float>

- Affects: `sub` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_f32_1k | 1024 | 4000 | 2.36 | 0.33 | 7.08 | 0.434 | 3.073 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_4k | 4096 | 4000 | 9.35 | 1.28 | 7.32 | 0.438 | 3.208 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_16k | 16384 | 4000 | 37.37 | 7.35 | 5.09 | 0.438 | 2.229 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_64k | 65536 | 1953 | 149.46 | 24.85 | 6.01 | 0.438 | 2.637 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Div<float>

- Affects: `div` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| div_f32_1k | 1024 | 4000 | 4.69 | 2.35 | 2.00 | 0.218 | 0.437 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_4k | 4096 | 4000 | 18.68 | 9.34 | 2.00 | 0.219 | 0.438 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_16k | 16384 | 4000 | 74.72 | 37.37 | 2.00 | 0.219 | 0.438 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_64k | 65536 | 1953 | 298.88 | 149.11 | 2.00 | 0.219 | 0.440 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AffineQuantize<int8>

- Affects: `quantize` float-to-int8 path used by int8 activations and weights.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i8_1k | 1024 | 4000 | 15.56 | 1.11 | 14.07 | 0.066 | 0.926 | 0 | 0 |
| affine_q_i8_4k | 4096 | 4000 | 78.13 | 4.47 | 17.47 | 0.052 | 0.916 | 0 | 0 |
| affine_q_i8_16k | 16384 | 4000 | 332.26 | 17.77 | 18.70 | 0.049 | 0.922 | 0 | 0 |
| affine_q_i8_64k | 65536 | 1953 | 1324.71 | 71.00 | 18.66 | 0.049 | 0.923 | 0 | 0 |

### AffineQuantize<uint8>

- Affects: `quantize` float-to-uint8 path used by uint8 activations.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_u8_1k | 1024 | 4000 | 8.66 | 1.40 | 6.20 | 0.118 | 0.733 | 0 | 0 |
| affine_q_u8_4k | 4096 | 4000 | 35.64 | 5.66 | 6.30 | 0.115 | 0.724 | 0 | 0 |
| affine_q_u8_16k | 16384 | 4000 | 142.96 | 22.44 | 6.37 | 0.115 | 0.730 | 0 | 0 |
| affine_q_u8_64k | 65536 | 1953 | 570.28 | 89.63 | 6.36 | 0.115 | 0.731 | 0 | 0 |

### AffineQuantize<int16>

- Affects: `quantize` float-to-int16 path used by q15-style operator staging.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i16_1k | 1024 | 4000 | 8.92 | 1.00 | 8.90 | 0.115 | 1.022 | 0 | 0 |
| affine_q_i16_4k | 4096 | 4000 | 35.45 | 3.98 | 8.90 | 0.116 | 1.028 | 0 | 0 |
| affine_q_i16_16k | 16384 | 4000 | 142.15 | 15.91 | 8.93 | 0.115 | 1.030 | 0 | 0 |
| affine_q_i16_64k | 65536 | 1953 | 567.63 | 63.52 | 8.94 | 0.115 | 1.032 | 0 | 0 |

### AveragePool<uint8>

- Affects: `average_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 122.36 | 77.74 | 1.57 | 0.536 | 0.843 | 0 | 0 |
| avgpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1115.30 | 856.09 | 1.30 | 0.753 | 0.981 | 0 | 0 |
| avgpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 440.51 | 338.55 | 1.30 | 0.753 | 0.980 | 0 | 0 |

### MaxPool<uint8>

- Affects: `max_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 206.16 | 24.05 | 8.57 | 0.318 | 2.725 | 0 | 0 |
| maxpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2067.87 | 234.22 | 8.83 | 0.406 | 3.586 | 0 | 0 |
| maxpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 793.76 | 92.36 | 8.59 | 0.418 | 3.592 | 0 | 0 |

### HardSwish<uint8>

- Affects: `hard_swish` quantized uint8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_u8_1k | 1024 | 4000 | 11.31 | 7.77 | 1.46 | 0.091 | 0.132 | 0 | 0 |
| hardswish_u8_4k | 4096 | 4000 | 50.99 | 30.94 | 1.65 | 0.080 | 0.132 | 0 | 0 |
| hardswish_u8_16k | 16384 | 4000 | 209.54 | 123.64 | 1.69 | 0.078 | 0.133 | 0 | 0 |
| hardswish_u8_64k | 65536 | 1953 | 840.17 | 494.63 | 1.70 | 0.078 | 0.132 | 0 | 0 |

### HardSwish<int8>

- Affects: `hard_swish` quantized int8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_i8_1k | 1024 | 4000 | 10.54 | 7.03 | 1.50 | 0.097 | 0.146 | 0 | 0 |
| hardswish_i8_4k | 4096 | 4000 | 47.23 | 28.04 | 1.68 | 0.087 | 0.146 | 0 | 0 |
| hardswish_i8_16k | 16384 | 4000 | 194.55 | 111.96 | 1.74 | 0.084 | 0.146 | 0 | 0 |
| hardswish_i8_64k | 65536 | 1953 | 781.21 | 447.88 | 1.74 | 0.084 | 0.146 | 0 | 0 |

### ArgMin<float>

- Affects: `arg_min` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmin_f32_1k | 1024 | 4000 | 1.88 | 0.90 | 2.08 | 0.544 | 1.132 | 0 | 0 |
| argmin_f32_16k | 16384 | 4000 | 29.87 | 13.11 | 2.28 | 0.549 | 1.249 | 0 | 0 |
| argmin_f32_64k | 65536 | 1953 | 119.42 | 51.11 | 2.34 | 0.549 | 1.282 | 0 | 0 |

### ArgMax<float>

- Affects: `arg_max` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_f32_1k | 1024 | 4000 | 1.88 | 0.92 | 2.05 | 0.545 | 1.116 | 0 | 0 |
| argmax_f32_16k | 16384 | 4000 | 29.88 | 13.09 | 2.28 | 0.548 | 1.251 | 0 | 0 |
| argmax_f32_64k | 65536 | 1953 | 119.47 | 50.99 | 2.34 | 0.549 | 1.285 | 0 | 0 |

### ArgMax<int8>

- Affects: `arg_max` int8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_i8_1k | 1024 | 4000 | 1.42 | 0.18 | 7.81 | 0.721 | 5.634 | 0 | 0 |
| argmax_i8_16k | 16384 | 4000 | 22.38 | 1.55 | 14.44 | 0.732 | 10.567 | 0 | 0 |
| argmax_i8_64k | 65536 | 1953 | 89.57 | 6.29 | 14.25 | 0.732 | 10.425 | 0 | 0 |

### ArgMax<uint8>

- Affects: `arg_max` uint8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_u8_1k | 1024 | 4000 | 1.41 | 0.18 | 7.67 | 0.728 | 5.583 | 0 | 0 |
| argmax_u8_16k | 16384 | 4000 | 22.41 | 1.54 | 14.57 | 0.731 | 10.655 | 0 | 0 |
| argmax_u8_64k | 65536 | 1953 | 89.65 | 6.25 | 14.36 | 0.731 | 10.494 | 0 | 0 |

## `tflite/kernels/fully_connected.cc`

- Summary: Dense float FullyConnected on RVV now takes the `EvalPie` route, so this split benchmark keeps the measurement alongside the source-file section and exercises the same matvec-style accumulation shape.

### FullyConnected<float> dense EvalPie route

- Affects: `fully_connected.cc` dense float RVV dispatch that now routes through `EvalPie`; this benchmark mirrors that matrix-batch-vector accumulation path inside the split operator harness.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_dense_small_b4 | B4 I64 O64 | 3906 | 24.35 | 7.31 | 3.33 | 1.346 | 4.485 | 0.00000072 | 0.00005428 | 0.0000000883 |
| fc_dense_mid_b4 | B4 I128 O128 | 976 | 86.97 | 24.79 | 3.51 | 1.507 | 5.287 | 0.00000095 | 0.00078133 | 0.0000001498 |
| fc_dense_large_b4 | B4 I2048 O640 | 20 | 7216.35 | 2359.18 | 3.06 | 1.453 | 4.445 | 0.00002098 | 0.00051320 | 0.0000023717 |

## `tflite/kernels/lstm_eval.cc`

- Summary: Float gate and output/projection paths now reuse the RVV-enabled `tensor_utils` matvec helpers, so this section keeps the operator-level numbers separate from generic helper benchmarks.

### LstmGate<float>

- Affects: `lstm_eval` float gate path (`input_to_gate` + `recurrent_to_gate` + sigmoid).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_gate_small_b1 | B1 I128 O128 C512 | 488 | 198.19 | 63.91 | 3.10 | 1.323 | 4.102 | 0.00000024 | 0.00000094 | 0.0000000416 |
| lstm_gate_small_b4 | B4 I128 O128 C512 | 122 | 795.26 | 254.35 | 3.13 | 1.319 | 4.123 | 0.00000024 | 0.00000113 | 0.0000000392 |
| lstm_gate_large_b1 | B1 I256 O256 C1024 | 122 | 764.15 | 197.85 | 3.86 | 1.372 | 5.300 | 0.00000060 | 0.00000266 | 0.0000000650 |

### LstmOutput<float>

- Affects: `lstm_eval` float output/projection path (`tanh(cell)` + output gate + projection).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_output_small_b1 | B1 C512 O128 | 972 | 107.91 | 36.06 | 2.99 | 1.219 | 3.649 | 0.00000095 | 0.00001072 | 0.0000003226 |
| lstm_output_small_b4 | B4 C512 O128 | 243 | 430.88 | 144.74 | 2.98 | 1.222 | 3.636 | 0.00000286 | 0.00063764 | 0.0000003949 |
| lstm_output_large_b1 | B1 C1024 O256 | 243 | 394.53 | 107.43 | 3.67 | 1.331 | 4.890 | 0.00000620 | 0.00026430 | 0.0000007587 |

## `tflite/kernels/internal/optimized/integer_ops/add.h`

- Summary: Recent RVV work added both int8/int16 elementwise add coverage and the scalar-broadcast kernels used by the fivefold broadcast fast path.

### Add<int8>

- Affects: `integer_ops/add.h` int8 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i8_1k | 1024 | 4000 | 10.52 | 5.89 | 1.78 | 0.097 | 0.174 | 0 | 0 |
| add_i8_4k | 4096 | 4000 | 52.13 | 23.58 | 2.21 | 0.079 | 0.174 | 0 | 0 |
| add_i8_16k | 16384 | 4000 | 212.00 | 94.27 | 2.25 | 0.077 | 0.174 | 0 | 0 |
| add_i8_64k | 65536 | 1953 | 847.96 | 377.16 | 2.25 | 0.077 | 0.174 | 0 | 0 |

### AddScalarBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast add kernel used by the fivefold broadcast fast path in `integer_ops/add.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_i8_1k | 1024 | 4000 | 6.48 | 4.18 | 1.55 | 0.158 | 0.245 | 0 | 0 |
| add_scalar_kernel_i8_4k | 4096 | 4000 | 34.28 | 16.71 | 2.05 | 0.119 | 0.245 | 0 | 0 |
| add_scalar_kernel_i8_16k | 16384 | 4000 | 124.72 | 66.71 | 1.87 | 0.131 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_64k | 65536 | 1953 | 506.08 | 266.80 | 1.90 | 0.129 | 0.246 | 0 | 0 |

### BroadcastAddDispatch<int8> scalar lhs

- Affects: `integer_ops/add.h` int8 scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_i8_1k | 1024 | 4000 | 14.56 | 14.50 | 1.00 | 0.070 | 0.071 | 0 | 0 |
| add_bcast_i8_4k | 4096 | 4000 | 68.64 | 68.66 | 1.00 | 0.060 | 0.060 | 0 | 0 |
| add_bcast_i8_16k | 16384 | 4000 | 276.31 | 276.23 | 1.00 | 0.059 | 0.059 | 0 | 0 |
| add_bcast_i8_64k | 65536 | 1953 | 1065.18 | 1064.23 | 1.00 | 0.062 | 0.062 | 0 | 0 |

### Add<int16>

- Affects: `integer_ops/add.h` int16 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i16_1k | 1024 | 4000 | 10.52 | 5.79 | 1.82 | 0.097 | 0.177 | 0 | 0 |
| add_i16_4k | 4096 | 4000 | 50.66 | 23.12 | 2.19 | 0.081 | 0.177 | 0 | 0 |
| add_i16_16k | 16384 | 4000 | 209.63 | 92.44 | 2.27 | 0.078 | 0.177 | 0 | 0 |
| add_i16_64k | 65536 | 1953 | 842.63 | 369.53 | 2.28 | 0.078 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mul.h`

- Summary: Recent RVV work added both int8 elementwise mul coverage and the scalar-broadcast kernels used by the quantized broadcast fast path.

### Mul<int8>

- Affects: `integer_ops/mul.h` int8 elementwise quantized mul path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_i8_1k | 1024 | 4000 | 10.72 | 3.22 | 3.33 | 0.096 | 0.318 | 0 | 0 |
| mul_i8_4k | 4096 | 4000 | 47.97 | 12.86 | 3.73 | 0.085 | 0.319 | 0 | 0 |
| mul_i8_16k | 16384 | 4000 | 195.79 | 51.33 | 3.81 | 0.084 | 0.319 | 0 | 0 |
| mul_i8_64k | 65536 | 1953 | 786.91 | 205.29 | 3.83 | 0.083 | 0.319 | 0 | 0 |

### MulSimpleBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast mul kernel used by the fivefold broadcast fast path in `integer_ops/mul.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_i8_1k | 1024 | 4000 | 9.85 | 3.30 | 2.99 | 0.104 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_4k | 4096 | 4000 | 44.97 | 13.20 | 3.41 | 0.091 | 0.310 | 0 | 0 |
| mul_scalar_kernel_i8_16k | 16384 | 4000 | 184.03 | 52.70 | 3.49 | 0.089 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_64k | 65536 | 1953 | 739.49 | 210.91 | 3.51 | 0.089 | 0.311 | 0 | 0 |

### BroadcastMulDispatch<int8> scalar lhs

- Affects: `integer_ops/mul.h` int8 scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_i8_1k | 1024 | 4000 | 10.40 | 10.23 | 1.02 | 0.098 | 0.100 | 0 | 0 |
| mul_bcast_i8_4k | 4096 | 4000 | 47.45 | 47.47 | 1.00 | 0.086 | 0.086 | 0 | 0 |
| mul_bcast_i8_16k | 16384 | 4000 | 193.96 | 193.73 | 1.00 | 0.084 | 0.085 | 0 | 0 |
| mul_bcast_i8_64k | 65536 | 1953 | 779.64 | 781.31 | 1.00 | 0.084 | 0.084 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/sub.h`

- Summary: This section isolates the int16 quantized subtract path that gained a dedicated RVV elementwise implementation.

### Sub<int16>

- Affects: `integer_ops/sub.h` int16 elementwise quantized sub path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_i16_1k | 1024 | 4000 | 10.04 | 5.79 | 1.73 | 0.102 | 0.177 | 0 | 0 |
| sub_i16_4k | 4096 | 4000 | 50.51 | 23.13 | 2.18 | 0.081 | 0.177 | 0 | 0 |
| sub_i16_16k | 16384 | 4000 | 205.93 | 92.46 | 2.23 | 0.080 | 0.177 | 0 | 0 |
| sub_i16_64k | 65536 | 1953 | 827.37 | 369.58 | 2.24 | 0.079 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/pooling.h`

- Summary: This section keeps the signed int8 pooling kernels separate from the uint8 `optimized_ops.h` pooling paths, because their RVV logic and rounding semantics are different.

### AveragePool<int8>

- Affects: `average_pool` int8 depth inner loop, signed round-away-from-zero, and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 175.91 | 64.76 | 2.72 | 0.373 | 1.012 | 0 | 0 |
| avgpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1406.82 | 684.10 | 2.06 | 0.597 | 1.228 | 0 | 0 |
| avgpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 552.05 | 271.64 | 2.03 | 0.601 | 1.221 | 0 | 0 |

### MaxPool<int8>

- Affects: `max_pool` int8 depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 223.07 | 24.44 | 9.13 | 0.294 | 2.681 | 0 | 0 |
| maxpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2191.65 | 236.90 | 9.25 | 0.383 | 3.545 | 0 | 0 |
| maxpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 834.48 | 95.21 | 8.76 | 0.398 | 3.485 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h`

- Summary: This section tracks the RVV general depthwise int8 kernels, including the fixed-depth and fixed-depth-multiplier specializations added to mirror the NEON dispatch table.

### DepthwiseConv<int8>

- Affects: `depthwise_conv` int8 general path, including fixed input-depth and depth-multiplier RVV kernels aligned with NEON dispatch.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dwconv_i8_dm1_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM1 | 1975 | 186.06 | 115.74 | 1.61 | 0.348 | 0.560 | 0 | 0 |
| dwconv_i8_dm1_d8_s1 | B1 I32x32x8 F3x3 S1x1 DM1 | 987 | 345.39 | 154.12 | 2.24 | 0.375 | 0.841 | 0 | 0 |
| dwconv_i8_dm1_d12_s1 | B1 I32x32x12 F3x3 S1x1 DM1 | 658 | 510.43 | 194.15 | 2.63 | 0.381 | 1.001 | 0 | 0 |
| dwconv_i8_dm1_d16_s2 | B1 I32x32x16 F3x3 S2x2 DM1 | 1975 | 171.09 | 71.30 | 2.40 | 0.379 | 0.909 | 0 | 0 |
| dwconv_i8_dm2_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM2 | 987 | 328.30 | 195.83 | 1.68 | 0.395 | 0.662 | 0 | 0 |
| dwconv_i8_dm2_d8_s2 | B1 I32x32x8 F3x3 S2x2 DM2 | 1975 | 159.37 | 96.14 | 1.66 | 0.407 | 0.674 | 0 | 0 |
| dwconv_i8_dm4_d1_s1 | B1 I32x32x1 F3x3 S1x1 DM4 | 1975 | 145.93 | 108.12 | 1.35 | 0.444 | 0.599 | 0 | 0 |
| dwconv_i8_dm4_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM4 | 493 | 510.01 | 380.11 | 1.34 | 0.508 | 0.682 | 0 | 0 |
| dwconv_i8_dm8_d2_s1 | B1 I32x32x2 F3x3 S1x1 DM8 | 493 | 449.66 | 278.87 | 1.61 | 0.576 | 0.929 | 0 | 0 |
| dwconv_i8_dm8_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM8 | 3950 | 58.93 | 43.64 | 1.35 | 0.550 | 0.742 | 0 | 0 |
| dwconv_i8_dm16_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM16 | 1975 | 104.07 | 63.89 | 1.63 | 0.623 | 1.014 | 0 | 0 |
| dwconv_i8_dm20_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM20 | 1580 | 127.69 | 97.42 | 1.31 | 0.634 | 0.831 | 0 | 0 |
| dwconv_i8_dm32_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM32 | 987 | 193.59 | 129.09 | 1.50 | 0.669 | 1.004 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h`

- Note: The recent change in this file only reroutes hybrid dispatch to reuse the RVV depthwise kernels above, so it intentionally shares the same numeric benchmark coverage.

## `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h`

- Summary: This section isolates the int16 quantized LeakyReLU path that received a dedicated RVV implementation.

### LeakyRelu<int16>

- Affects: `integer_ops/leaky_relu.h` int16 quantized leaky-relu path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leaky_relu_i16_1k | 1024 | 4000 | 12.58 | 4.09 | 3.08 | 0.081 | 0.250 | 0 | 0 |
| leaky_relu_i16_4k | 4096 | 4000 | 50.83 | 16.35 | 3.11 | 0.081 | 0.251 | 0 | 0 |
| leaky_relu_i16_16k | 16384 | 4000 | 201.79 | 65.30 | 3.09 | 0.081 | 0.251 | 0 | 0 |
| leaky_relu_i16_64k | 65536 | 1953 | 810.43 | 261.27 | 3.10 | 0.081 | 0.251 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/lut.h`

- Summary: This section keeps the uint8 and int8 lookup-table kernels separate because they were vectorized in the same commit but have different input domains.

### LookupTable<uint8>

- Affects: `integer_ops/lut.h` uint8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_u8_1k | 1024 | 4000 | 1.41 | 0.86 | 1.65 | 0.725 | 1.194 | 0 | 0 |
| lut_u8_4k | 4096 | 4000 | 5.60 | 3.39 | 1.65 | 0.731 | 1.208 | 0 | 0 |
| lut_u8_16k | 16384 | 4000 | 22.40 | 13.59 | 1.65 | 0.732 | 1.206 | 0 | 0 |
| lut_u8_64k | 65536 | 1953 | 89.63 | 54.20 | 1.65 | 0.731 | 1.209 | 0 | 0 |

### LookupTable<int8>

- Affects: `integer_ops/lut.h` int8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_i8_1k | 1024 | 4000 | 1.41 | 0.85 | 1.66 | 0.727 | 1.207 | 0 | 0 |
| lut_i8_4k | 4096 | 4000 | 5.60 | 3.39 | 1.65 | 0.732 | 1.208 | 0 | 0 |
| lut_i8_16k | 16384 | 4000 | 22.39 | 13.57 | 1.65 | 0.732 | 1.208 | 0 | 0 |
| lut_i8_64k | 65536 | 1953 | 89.62 | 54.18 | 1.65 | 0.731 | 1.210 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mean.h`

- Summary: This section tracks the int8 height-width reduction path that gained RVV coverage in the latest integer-ops commit.

### Mean<int8>

- Affects: `integer_ops/mean.h` int8 height-width reduction path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_i8_8x8x64 | B1 I8x8x64 A(1,2) | 4000 | 5.83 | 1.63 | 3.59 | 0.702 | 2.520 | 0 | 0 |
| mean_i8_16x16x128 | B1 I16x16x128 A(1,2) | 3906 | 36.56 | 11.60 | 3.15 | 0.896 | 2.824 | 0 | 0 |
| mean_i8_32x32x256 | B1 I32x32x256 A(1,2) | 488 | 1450.92 | 91.88 | 15.79 | 0.181 | 2.853 | 0 | 0 |
