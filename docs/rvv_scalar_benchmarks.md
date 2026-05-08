# RVV Scalar Benchmark Results

- Date: 2026-05-08 13:27:17 CST
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
| zero_f32_64 | 64 | 4000 | 0.12 | 0.06 | 2.07 | 0.538 | 1.114 | 0 | 0 |
| zero_f32_1024 | 1024 | 4000 | 1.76 | 0.91 | 1.93 | 0.583 | 1.124 | 0 | 0 |
| zero_f32_4096 | 4096 | 4000 | 7.00 | 3.62 | 1.93 | 0.585 | 1.132 | 0 | 0 |

### IsZeroVector<int8>

- Affects: zero-skip checks for quantized helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_i8_64 | 64 | 4000 | 0.03 | 0.02 | 2.00 | 2.007 | 4.019 | 0 | 0 |
| zero_i8_1024 | 1024 | 4000 | 0.47 | 0.22 | 2.12 | 2.173 | 4.613 | 0 | 0 |
| zero_i8_4096 | 4096 | 4000 | 1.89 | 0.88 | 2.14 | 2.166 | 4.629 | 0 | 0 |

### VectorVectorDotProduct<float>

- Affects: `fully_connected`, recurrent math, shared float tensor-utils call sites.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dot_64 | 64 | 4000 | 0.09 | 0.03 | 2.77 | 1.395 | 3.866 | 0.00000000 | 0.00000000 | 0.0000000000 |
| dot_256 | 256 | 4000 | 0.36 | 0.08 | 4.46 | 1.435 | 6.400 | 0.00000119 | 0.00000046 | 0.0000011921 |
| dot_1024 | 1024 | 4000 | 1.41 | 0.30 | 4.68 | 1.451 | 6.794 | 0.00000191 | 0.00000016 | 0.0000019073 |
| dot_4096 | 4096 | 4000 | 5.60 | 1.07 | 5.24 | 1.463 | 7.670 | 0.00002384 | 0.00000264 | 0.0000238419 |

### BatchVectorBatchVectorDotProduct<int16>

- Affects: `svdf` and batched recurrent dot-product helper paths.

| Case | Batch x size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_dot_q15_small | 4 x 64 | 4000 | 0.29 | 0.09 | 3.14 | 1.774 | 5.562 | 0 | 0 |
| svdf_dot_q15_mid | 8 x 256 | 4000 | 2.09 | 0.49 | 4.28 | 1.959 | 8.381 | 0 | 0 |
| svdf_dot_q15_large | 16 x 1024 | 4000 | 16.72 | 3.64 | 4.60 | 1.959 | 9.007 | 0 | 0 |

### ReductionSumVector<float>

- Affects: `svdf` and other float reduction-style helper paths.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_small | 64 x 8 | 4000 | 1.00 | 0.77 | 1.30 | 0.513 | 0.665 | 0.00000048 | 0.00000080 | 0.0000000559 |
| svdf_mid | 128 x 16 | 4000 | 4.22 | 1.77 | 2.38 | 0.485 | 1.156 | 0.00000095 | 0.00001488 | 0.0000001378 |
| svdf_large | 256 x 64 | 4000 | 35.03 | 6.32 | 5.55 | 0.468 | 2.594 | 0.00000572 | 0.00001453 | 0.0000005609 |
| reduce_wide | 256 x 256 | 2441 | 141.53 | 17.52 | 8.08 | 0.463 | 3.741 | 0.00001144 | 0.00016069 | 0.0000021656 |

### ReductionSumVector<int8 -> int32>

- Affects: `conv`, `batch_matmul`, `kernel_utils`, cached row sum preparation.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| row_sum_small | 64 x 32 | 4000 | 1.09 | 1.07 | 1.02 | 1.882 | 1.922 | 0 | 0 |
| row_sum_mid | 128 x 128 | 4000 | 5.66 | 4.73 | 1.20 | 2.894 | 3.466 | 0 | 0 |
| row_sum_large | 256 x 256 | 2441 | 20.66 | 16.56 | 1.25 | 3.173 | 3.957 | 0 | 0 |
| row_sum_conv_like | 512 x 512 | 610 | 79.49 | 61.13 | 1.30 | 3.298 | 4.288 | 0 | 0 |

### ReductionSumVector<int32>

- Affects: scalar accumulation helper paths such as reference SVDF and utility reductions.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reduce_i32_small | 64 x 32 | 4000 | 0.91 | 0.77 | 1.19 | 2.246 | 2.671 | 0 | 0 |
| reduce_i32_mid | 128 x 128 | 4000 | 4.65 | 3.01 | 1.54 | 3.523 | 5.437 | 0 | 0 |
| reduce_i32_large | 256 x 256 | 2441 | 16.75 | 10.65 | 1.57 | 3.913 | 6.155 | 0 | 0 |

### MatrixScalarMultiplyAccumulate<int8>

- Affects: quantized recurrent helpers and row-wise reduction paths.

| Case | Rows x cols | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_rows_64 | 64 x 64 | 4000 | 4.11 | 1.58 | 2.60 | 0.998 | 2.590 | 0 | 0 |
| fc_rows_256 | 256 x 128 | 4000 | 30.48 | 9.77 | 3.12 | 1.075 | 3.355 | 0 | 0 |
| conv_rows_512 | 512 x 256 | 1220 | 122.28 | 33.49 | 3.65 | 1.072 | 3.914 | 0 | 0 |
| conv_rows_1024 | 1024 x 512 | 305 | 484.61 | 122.98 | 3.94 | 1.082 | 4.263 | 0 | 0 |

### MatrixBatchVectorMultiplyAccumulate<float>

- Affects: float `fully_connected` and recurrent kernels.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_fc | 64 x 64 x 4 | 4000 | 24.26 | 7.21 | 3.36 | 1.351 | 4.542 | 0.00000286 | 0.00001540 | 0.0000003163 |
| mid_fc | 128 x 128 x 4 | 2441 | 87.47 | 23.86 | 3.67 | 1.498 | 5.493 | 0.00000525 | 0.00006299 | 0.0000006228 |
| lstm_like | 640 x 2048 x 4 | 30 | 7214.01 | 2348.29 | 3.07 | 1.454 | 4.465 | 0.00008774 | 0.00183174 | 0.0000094324 |
| sqrnn_like | 1024 x 1024 x 8 | 20 | 11576.84 | 3732.15 | 3.10 | 1.449 | 4.495 | 0.00005531 | 0.00156548 | 0.0000045446 |

### MatrixBatchVectorMultiplyAccumulate<int8>

- Affects: quantized `fully_connected`, hybrid recurrent helpers, shared int8 GEMV-style call sites.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_fc_small | 64 x 64 x 4 | 4000 | 16.48 | 6.97 | 2.36 | 1.989 | 4.702 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_fc_mid | 128 x 128 x 4 | 2441 | 70.17 | 20.93 | 3.35 | 1.868 | 6.263 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_lstm_like | 640 x 1024 x 4 | 61 | 2787.93 | 593.70 | 4.70 | 1.881 | 8.831 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_conv_like | 1024 x 512 x 8 | 38 | 4463.01 | 1010.75 | 4.42 | 1.880 | 8.299 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MatrixBatchVectorMultiplyAccumulate<int8, per-channel + input_offset>

- Affects: quantized `conv`, `batch_matmul`, and any path that uses cached row sums plus per-channel scale.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_row_sum_small | 64 x 64 x 4 | 4000 | 26.08 | 8.62 | 3.02 | 1.257 | 3.799 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_mid | 128 x 128 x 4 | 2441 | 100.09 | 25.92 | 3.86 | 1.310 | 5.057 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_conv | 256 x 576 x 8 | 135 | 1723.00 | 314.29 | 5.48 | 1.369 | 7.507 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_batch_matmul_like | 512 x 512 x 8 | 76 | 3061.61 | 566.92 | 5.40 | 1.370 | 7.398 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MatrixBatchVectorMultiplyAccumulate<int8 -> int16>

- Affects: quantized recurrent gate accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q15_small | 128 x 128 x 4 | 2441 | 76.43 | 21.48 | 3.56 | 1.715 | 6.103 | 0 | 0 |
| gate_q15_mid | 256 x 256 x 4 | 610 | 287.80 | 71.00 | 4.05 | 1.822 | 7.385 | 0 | 0 |
| gate_q15_large | 512 x 512 x 8 | 76 | 2298.78 | 507.27 | 4.53 | 1.825 | 8.268 | 0 | 0 |

### MatrixBatchVectorMultiplyAccumulate<int8 -> int8>

- Affects: quantized projection and low-precision recurrent accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q8_small | 128 x 128 x 4 | 2441 | 77.48 | 21.63 | 3.58 | 1.692 | 6.059 | 0 | 0 |
| gate_q8_mid | 256 x 256 x 4 | 610 | 292.53 | 71.47 | 4.09 | 1.792 | 7.336 | 0 | 0 |
| gate_q8_large | 512 x 512 x 8 | 76 | 2249.81 | 508.56 | 4.42 | 1.864 | 8.247 | 0 | 0 |

### MatrixBatchVectorMultiply<int8 -> int8>

- Affects: quantized LSTM gate matmul paths before saturating-add and activation.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_noacc_q8_small | 128 x 128 x 4 | 2441 | 82.24 | 30.48 | 2.70 | 1.594 | 4.300 | 0 | 0 |
| gate_noacc_q8_mid | 256 x 256 x 4 | 610 | 299.15 | 100.71 | 2.97 | 1.753 | 5.206 | 0 | 0 |
| gate_noacc_q8_large | 512 x 512 x 8 | 76 | 2270.79 | 703.06 | 3.23 | 1.847 | 5.966 | 0 | 0 |

### MatrixBatchVectorMultiply<int16 x int8 -> int8>

- Affects: quantized projection/output matmul helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| proj_q8_small | 128 x 128 x 4 | 2441 | 424.99 | 32.31 | 13.15 | 0.308 | 4.056 | 0 | 0 |
| proj_q8_mid | 256 x 256 x 4 | 610 | 1676.29 | 117.44 | 14.27 | 0.313 | 4.464 | 0 | 0 |
| proj_q8_large | 512 x 512 x 8 | 76 | 13284.95 | 894.83 | 14.85 | 0.316 | 4.687 | 0 | 0 |

### SparseMatrixBatchVectorMultiplyAccumulate1x4<float>

- Affects: sparse float `fully_connected` and sparse recurrent helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_1x4_small | 64 x 256 x 4 | 4000 | 22.08 | 12.15 | 1.82 | 1.469 | 2.670 | 0.00000143 | 0.00012937 | 0.0000003011 |
| sparse_1x4_mid | 128 x 512 x 4 | 2448 | 98.05 | 50.19 | 1.95 | 1.333 | 2.604 | 0.00000477 | 0.00038041 | 0.0000006075 |
| sparse_1x4_large | 256 x 1024 x 8 | 304 | 755.52 | 374.28 | 2.02 | 1.390 | 2.806 | 0.00000954 | 0.00038029 | 0.0000011599 |

### SparseMatrixBatchVectorMultiplyAccumulate<float ledger>

- Affects: sparse float matvec helper paths with ledger format.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_ledger_small | 64 x 256 x 4 | 4000 | 24.97 | 11.68 | 2.14 | 1.292 | 2.761 | 0.00000238 | 0.00000709 | 0.0000003208 |
| sparse_ledger_mid | 128 x 512 x 4 | 2362 | 103.79 | 41.69 | 2.49 | 1.305 | 3.248 | 0.00000572 | 0.00038695 | 0.0000006152 |
| sparse_ledger_large | 256 x 1024 x 8 | 303 | 810.01 | 294.33 | 2.75 | 1.304 | 3.588 | 0.00001240 | 0.00644636 | 0.0000011960 |

### SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>

- Affects: sparse quantized matvec helper paths that dequantize to float.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_qfloat_small | 64 x 256 x 4 | 4000 | 15.54 | 10.39 | 1.50 | 2.084 | 3.118 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_mid | 128 x 512 x 4 | 2465 | 58.56 | 32.98 | 1.78 | 2.217 | 3.935 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_large | 256 x 1024 x 8 | 299 | 485.36 | 234.16 | 2.07 | 2.202 | 4.563 | 0.00000000 | 0.00000000 | 0.0000000000 |

### SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>

- Affects: sparse quantized fully-connected and recurrent output helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_q8_small | 64 x 256 x 4 | 4000 | 33.86 | 15.74 | 2.15 | 1.009 | 2.171 | 0 | 0 |
| sparse_q8_mid | 128 x 512 x 4 | 2394 | 124.72 | 46.98 | 2.65 | 1.071 | 2.845 | 0 | 0 |
| sparse_q8_large | 256 x 1024 x 8 | 302 | 943.92 | 286.54 | 3.29 | 1.121 | 3.692 | 0 | 0 |

### SymmetricQuantizeFloats

- Affects: hybrid `fully_connected`, `batch_matmul`, weight and activation pre-quant helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Min diff | Max diff | Scale diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sym_quant_small | 64 | 4000 | 0.46 | 0.15 | 2.98 | 0.420 | 1.252 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_mid | 1024 | 4000 | 7.26 | 2.65 | 2.74 | 0.423 | 1.160 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_large | 4096 | 4000 | 32.30 | 15.01 | 2.15 | 0.380 | 0.819 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |

### AsymmetricQuantizeFloats

- Affects: hybrid `conv`, `depthwise_conv`, `transpose_conv`, and int8 input staging paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Scale diff | Offset diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asym_quant_small | 64 | 4000 | 0.57 | 0.25 | 2.32 | 0.336 | 0.780 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_mid | 1024 | 4000 | 8.55 | 3.25 | 2.63 | 0.359 | 0.945 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_large | 4096 | 4000 | 37.18 | 16.29 | 2.28 | 0.331 | 0.754 | 0 | 0 | 0.00000000 | 0 |

### ApplyLayerNorm<int16>

- Affects: quantized recurrent gate normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ln_small | 4 x 64 | 4000 | 5.90 | 2.38 | 2.48 | 0.347 | 0.861 | 0 | 0 |
| ln_mid | 4 x 256 | 4000 | 24.08 | 8.66 | 2.78 | 0.340 | 0.946 | 0 | 0 |
| ln_gate_like | 8 x 1024 | 2441 | 193.45 | 67.50 | 2.87 | 0.339 | 0.971 | 0 | 0 |

### ApplySigmoid<int16>

- Affects: quantized recurrent gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoid_small | 4 x 64 | 4000 | 22.29 | 8.63 | 2.58 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_mid | 8 x 256 | 4000 | 192.32 | 68.89 | 2.79 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_large | 4 x 1024 | 4000 | 386.15 | 137.83 | 2.80 | 0.011 | 0.030 | 0 | 0 |

### ApplyTanh<int16>

- Affects: quantized recurrent state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanh_q0_small | 0 x 4 x 64 | 4000 | 21.19 | 8.25 | 2.57 | 0.012 | 0.031 | 0 | 0 |
| tanh_q3_mid | 3 x 8 x 256 | 4000 | 200.61 | 71.79 | 2.79 | 0.010 | 0.029 | 0 | 0 |
| tanh_q4_large | 4 x 4 x 1024 | 4000 | 405.19 | 150.38 | 2.69 | 0.010 | 0.027 | 0 | 0 |

### ApplyLayerNormFloat<int16>

- Affects: float-reference layer-norm helper paths used by quantized LSTM eval.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lnf_small | 4 x 64 | 4000 | 3.69 | 1.48 | 2.50 | 0.554 | 1.388 | 0 | 0 |
| lnf_mid | 4 x 256 | 4000 | 13.92 | 5.28 | 2.64 | 0.589 | 1.551 | 0 | 0 |
| lnf_gate_like | 8 x 1024 | 2441 | 109.37 | 40.91 | 2.67 | 0.599 | 1.602 | 0 | 0 |

### ApplySigmoidFloat<int16>

- Affects: float-reference gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoidf_small | 4 x 64 | 4000 | 9.38 | 8.63 | 1.09 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 2.0703125000 |
| sigmoidf_mid | 8 x 256 | 4000 | 74.78 | 68.85 | 1.09 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 1.9697265625 |
| sigmoidf_large | 4 x 1024 | 4000 | 149.27 | 137.73 | 1.08 | 0.027 | 0.030 | 6.00000000 | 0.21052632 | 1.9594726562 |

### ApplyTanhFloat<int16>

- Affects: float-reference state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanhf_qm12_small | -12 x 4 x 64 | 4000 | 24.80 | 8.88 | 2.79 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm12_mid | -12 x 8 x 256 | 4000 | 197.72 | 70.20 | 2.82 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm15_large | -15 x 4 x 1024 | 4000 | 395.32 | 140.11 | 2.82 | 0.010 | 0.029 | 0 | 0 |

### CwiseMul<int16 -> int16>

- Affects: quantized recurrent elementwise gate math.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q15_small | 4 x 64 | 4000 | 0.57 | 0.44 | 1.30 | 0.450 | 0.583 | 0 | 0 |
| mul_q15_mid | 8 x 256 | 4000 | 4.32 | 3.46 | 1.25 | 0.475 | 0.591 | 0 | 0 |
| mul_q15_large | 4 x 1024 | 4000 | 8.63 | 6.95 | 1.24 | 0.474 | 0.590 | 0 | 0 |

### CwiseMul<int16 -> int8>

- Affects: quantized projection and low-precision elementwise paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q8_small | 4 x 64 | 4000 | 2.44 | 0.77 | 3.18 | 0.105 | 0.333 | 0 | 0 |
| mul_q8_mid | 8 x 256 | 4000 | 23.72 | 6.02 | 3.94 | 0.086 | 0.340 | 0 | 0 |
| mul_q8_large | 4 x 1024 | 4000 | 49.57 | 12.05 | 4.11 | 0.083 | 0.340 | 0 | 0 |

### CwiseAdd<int16>

- Affects: recurrent residual/additive helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_small | 4 x 64 | 4000 | 0.71 | 0.33 | 2.12 | 0.362 | 0.767 | 0 | 0 |
| add_mid | 8 x 256 | 4000 | 6.29 | 2.62 | 2.40 | 0.325 | 0.780 | 0 | 0 |
| add_large | 4 x 1024 | 4000 | 15.11 | 5.26 | 2.87 | 0.271 | 0.778 | 0 | 0 |

### TwoGateSaturatingAdd<int8 -> int16>

- Affects: quantized LSTM gate merge helper paths.

| Case | Batch x cell | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| two_gate_small | 4 x 64 | 4000 | 4.15 | 1.30 | 3.19 | 0.247 | 0.788 | 0 | 0 |
| two_gate_mid | 8 x 256 | 4000 | 42.26 | 10.24 | 4.13 | 0.194 | 0.800 | 0 | 0 |
| two_gate_large | 4 x 1024 | 4000 | 84.15 | 20.46 | 4.11 | 0.195 | 0.801 | 0 | 0 |

### CwiseClipping<float>

- Affects: activation clamp paths in float recurrent/helper code.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_f32_64 | 64 | 4000 | 0.07 | 0.05 | 1.54 | 0.880 | 1.353 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_1024 | 1024 | 4000 | 1.17 | 0.76 | 1.55 | 0.874 | 1.354 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_4096 | 4096 | 4000 | 4.67 | 3.04 | 1.53 | 0.877 | 1.345 | 0.00000000 | 0.00000000 | 0.0000000000 |

### CwiseClipping<int16>

- Affects: quantized activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i16_64 | 64 | 4000 | 0.02 | 0.02 | 1.00 | 3.194 | 3.195 | 0 | 0 |
| clip_i16_1024 | 1024 | 4000 | 0.32 | 0.32 | 1.01 | 3.175 | 3.200 | 0 | 0 |
| clip_i16_4096 | 4096 | 4000 | 1.29 | 1.29 | 1.00 | 3.170 | 3.166 | 0 | 0 |

### CwiseClipping<int8>

- Affects: low-precision activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i8_64 | 64 | 4000 | 0.01 | 0.02 | 0.91 | 4.462 | 4.048 | 0 | 0 |
| clip_i8_1024 | 1024 | 4000 | 0.17 | 0.16 | 1.03 | 6.184 | 6.399 | 0 | 0 |
| clip_i8_4096 | 4096 | 4000 | 0.64 | 0.64 | 1.00 | 6.400 | 6.389 | 0 | 0 |

### VectorBatchVectorCwiseProductAccumulate<int16>

- Affects: quantized recurrent gate/state accumulation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vbv_cwise_small | 4 x 64 | 4000 | 2.29 | 0.81 | 2.83 | 0.223 | 0.633 | 0 | 0 |
| vbv_cwise_mid | 8 x 256 | 4000 | 23.26 | 6.29 | 3.70 | 0.176 | 0.651 | 0 | 0 |
| vbv_cwise_large | 4 x 1024 | 4000 | 47.17 | 12.59 | 3.75 | 0.174 | 0.651 | 0 | 0 |

### Sub1Vector<float>

- Affects: float recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_f32_64 | 64 | 4000 | 0.06 | 0.04 | 1.70 | 1.035 | 1.759 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_1024 | 1024 | 4000 | 0.94 | 0.58 | 1.61 | 1.089 | 1.755 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_4096 | 4096 | 4000 | 3.74 | 2.34 | 1.60 | 1.096 | 1.750 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Sub1Vector<int16>

- Affects: quantized recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_i16_64 | 64 | 4000 | 0.07 | 0.02 | 3.66 | 0.959 | 3.515 | 0 | 0 |
| sub1_i16_1024 | 1024 | 4000 | 0.94 | 0.29 | 3.21 | 1.092 | 3.506 | 0 | 0 |
| sub1_i16_4096 | 4096 | 4000 | 3.74 | 1.18 | 3.17 | 1.095 | 3.474 | 0 | 0 |

### VectorScalarMultiply<int8 -> float>

- Affects: hybrid dequant-style helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale_i8_64 | 64 | 4000 | 0.09 | 0.02 | 4.30 | 1.422 | 6.114 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_1024 | 1024 | 4000 | 1.41 | 0.32 | 4.38 | 1.455 | 6.376 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_4096 | 4096 | 4000 | 5.60 | 1.28 | 4.37 | 1.462 | 6.384 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MeanStddevNormalization<float>

- Affects: float recurrent normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| norm_1x64 | 1 x 64 | 4000 | 0.27 | 0.21 | 1.29 | 0.939 | 1.211 | 0.00000024 | 0.00000023 | 0.0000000843 |
| norm_4x256 | 4 x 256 | 4000 | 3.94 | 2.88 | 1.37 | 1.039 | 1.425 | 0.00000048 | 0.00002272 | 0.0000001109 |
| norm_8x1024 | 8 x 1024 | 4000 | 30.24 | 21.89 | 1.38 | 1.084 | 1.497 | 0.00000083 | 0.00009705 | 0.0000001304 |


## `tflite/kernels/internal/optimized/optimized_ops.h`

- Summary: Float arithmetic fast paths, quantize/dequant-style helpers, uint8 pooling, quantized HardSwish, and argmin/argmax cases that gained RVV coverage in the latest operator-level commits.

### Add<float>

- Affects: `add` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_f32_1k | 1024 | 4000 | 1.89 | 0.34 | 5.61 | 0.541 | 3.038 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_4k | 4096 | 4000 | 7.50 | 1.29 | 5.80 | 0.546 | 3.171 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_16k | 16384 | 4000 | 29.92 | 7.35 | 4.07 | 0.548 | 2.229 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_64k | 65536 | 1953 | 130.33 | 27.93 | 4.67 | 0.503 | 2.347 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AddScalarBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float add kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_f32_1k | 1024 | 4000 | 1.95 | 0.26 | 7.44 | 0.524 | 3.897 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_4k | 4096 | 4000 | 7.84 | 1.05 | 7.46 | 0.522 | 3.893 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_16k | 16384 | 4000 | 31.28 | 4.27 | 7.32 | 0.524 | 3.837 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_64k | 65536 | 1953 | 125.93 | 16.89 | 7.46 | 0.520 | 3.880 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastAddDispatch<float> scalar lhs

- Affects: `add` float scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_f32_1k | 1024 | 4000 | 2.41 | 0.34 | 7.11 | 0.425 | 3.023 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_4k | 4096 | 4000 | 9.41 | 1.12 | 8.39 | 0.435 | 3.651 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_16k | 16384 | 4000 | 37.84 | 4.49 | 8.43 | 0.433 | 3.648 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_64k | 65536 | 1953 | 149.37 | 16.96 | 8.81 | 0.439 | 3.864 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Mul<float>

- Affects: `mul` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_f32_1k | 1024 | 4000 | 2.36 | 0.37 | 6.38 | 0.434 | 2.771 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_4k | 4096 | 4000 | 9.35 | 1.39 | 6.71 | 0.438 | 2.937 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_16k | 16384 | 4000 | 37.39 | 7.20 | 5.19 | 0.438 | 2.274 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_64k | 65536 | 1953 | 149.39 | 24.93 | 5.99 | 0.439 | 2.629 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MulSimpleBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float mul kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_f32_1k | 1024 | 4000 | 2.35 | 0.29 | 8.08 | 0.435 | 3.516 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_4k | 4096 | 4000 | 9.35 | 1.17 | 8.02 | 0.438 | 3.513 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_16k | 16384 | 4000 | 37.39 | 4.72 | 7.93 | 0.438 | 3.475 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_64k | 65536 | 1953 | 149.37 | 18.77 | 7.96 | 0.439 | 3.491 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastMulDispatch<float> scalar lhs

- Affects: `mul` float scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_f32_1k | 1024 | 4000 | 2.48 | 0.35 | 7.04 | 0.413 | 2.907 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_4k | 4096 | 4000 | 9.71 | 1.23 | 7.92 | 0.422 | 3.341 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_16k | 16384 | 4000 | 39.28 | 4.93 | 7.96 | 0.417 | 3.322 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_64k | 65536 | 1953 | 154.05 | 18.97 | 8.12 | 0.425 | 3.455 | 0.00000000 | 0.00000000 | 0.0000000000 |

### SubWithActivation<float>

- Affects: `sub` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_f32_1k | 1024 | 4000 | 2.37 | 0.33 | 7.07 | 0.432 | 3.058 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_4k | 4096 | 4000 | 9.35 | 1.29 | 7.27 | 0.438 | 3.186 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_16k | 16384 | 4000 | 37.37 | 7.09 | 5.27 | 0.438 | 2.312 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_64k | 65536 | 1953 | 149.30 | 24.63 | 6.06 | 0.439 | 2.660 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Div<float>

- Affects: `div` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| div_f32_1k | 1024 | 4000 | 4.69 | 2.34 | 2.00 | 0.218 | 0.437 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_4k | 4096 | 4000 | 18.67 | 9.32 | 2.00 | 0.219 | 0.439 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_16k | 16384 | 4000 | 74.72 | 37.34 | 2.00 | 0.219 | 0.439 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_64k | 65536 | 1953 | 298.71 | 149.09 | 2.00 | 0.219 | 0.440 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AffineQuantize<int8>

- Affects: `quantize` float-to-int8 path used by int8 activations and weights.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i8_1k | 1024 | 4000 | 15.02 | 1.12 | 13.38 | 0.068 | 0.912 | 0 | 0 |
| affine_q_i8_4k | 4096 | 4000 | 77.94 | 4.45 | 17.52 | 0.053 | 0.921 | 0 | 0 |
| affine_q_i8_16k | 16384 | 4000 | 332.03 | 17.78 | 18.67 | 0.049 | 0.921 | 0 | 0 |
| affine_q_i8_64k | 65536 | 1953 | 1323.60 | 70.93 | 18.66 | 0.050 | 0.924 | 0 | 0 |

### AffineQuantize<uint8>

- Affects: `quantize` float-to-uint8 path used by uint8 activations.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_u8_1k | 1024 | 4000 | 8.90 | 1.40 | 6.37 | 0.115 | 0.733 | 0 | 0 |
| affine_q_u8_4k | 4096 | 4000 | 35.62 | 5.62 | 6.34 | 0.115 | 0.729 | 0 | 0 |
| affine_q_u8_16k | 16384 | 4000 | 142.97 | 22.49 | 6.36 | 0.115 | 0.729 | 0 | 0 |
| affine_q_u8_64k | 65536 | 1953 | 570.03 | 89.55 | 6.37 | 0.115 | 0.732 | 0 | 0 |

### AffineQuantize<int16>

- Affects: `quantize` float-to-int16 path used by q15-style operator staging.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i16_1k | 1024 | 4000 | 8.91 | 1.00 | 8.88 | 0.115 | 1.020 | 0 | 0 |
| affine_q_i16_4k | 4096 | 4000 | 35.47 | 3.99 | 8.89 | 0.115 | 1.027 | 0 | 0 |
| affine_q_i16_16k | 16384 | 4000 | 142.19 | 15.90 | 8.94 | 0.115 | 1.030 | 0 | 0 |
| affine_q_i16_64k | 65536 | 1953 | 567.29 | 63.48 | 8.94 | 0.116 | 1.032 | 0 | 0 |

### AveragePool<uint8>

- Affects: `average_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 121.77 | 77.37 | 1.57 | 0.538 | 0.847 | 0 | 0 |
| avgpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1112.88 | 857.27 | 1.30 | 0.755 | 0.980 | 0 | 0 |
| avgpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 440.10 | 337.89 | 1.30 | 0.754 | 0.982 | 0 | 0 |

### MaxPool<uint8>

- Affects: `max_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 205.40 | 23.91 | 8.59 | 0.319 | 2.741 | 0 | 0 |
| maxpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2069.21 | 241.14 | 8.58 | 0.406 | 3.483 | 0 | 0 |
| maxpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 792.10 | 95.25 | 8.32 | 0.419 | 3.483 | 0 | 0 |

### HardSwish<uint8>

- Affects: `hard_swish` quantized uint8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_u8_1k | 1024 | 4000 | 11.31 | 7.77 | 1.46 | 0.091 | 0.132 | 0 | 0 |
| hardswish_u8_4k | 4096 | 4000 | 50.98 | 30.94 | 1.65 | 0.080 | 0.132 | 0 | 0 |
| hardswish_u8_16k | 16384 | 4000 | 209.66 | 123.59 | 1.70 | 0.078 | 0.133 | 0 | 0 |
| hardswish_u8_64k | 65536 | 1953 | 840.02 | 494.55 | 1.70 | 0.078 | 0.133 | 0 | 0 |

### HardSwish<int8>

- Affects: `hard_swish` quantized int8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_i8_1k | 1024 | 4000 | 10.48 | 7.03 | 1.49 | 0.098 | 0.146 | 0 | 0 |
| hardswish_i8_4k | 4096 | 4000 | 47.25 | 28.03 | 1.69 | 0.087 | 0.146 | 0 | 0 |
| hardswish_i8_16k | 16384 | 4000 | 194.54 | 111.95 | 1.74 | 0.084 | 0.146 | 0 | 0 |
| hardswish_i8_64k | 65536 | 1953 | 781.15 | 448.08 | 1.74 | 0.084 | 0.146 | 0 | 0 |

### ArgMin<float>

- Affects: `arg_min` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmin_f32_1k | 1024 | 4000 | 1.89 | 0.91 | 2.08 | 0.542 | 1.129 | 0 | 0 |
| argmin_f32_16k | 16384 | 4000 | 29.87 | 13.01 | 2.30 | 0.548 | 1.260 | 0 | 0 |
| argmin_f32_64k | 65536 | 1953 | 119.41 | 51.27 | 2.33 | 0.549 | 1.278 | 0 | 0 |

### ArgMax<float>

- Affects: `arg_max` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_f32_1k | 1024 | 4000 | 1.87 | 0.91 | 2.07 | 0.547 | 1.131 | 0 | 0 |
| argmax_f32_16k | 16384 | 4000 | 29.85 | 12.97 | 2.30 | 0.549 | 1.263 | 0 | 0 |
| argmax_f32_64k | 65536 | 1953 | 119.42 | 51.39 | 2.32 | 0.549 | 1.275 | 0 | 0 |

### ArgMax<int8>

- Affects: `arg_max` int8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_i8_1k | 1024 | 4000 | 1.41 | 0.18 | 7.71 | 0.726 | 5.592 | 0 | 0 |
| argmax_i8_16k | 16384 | 4000 | 22.39 | 1.61 | 13.91 | 0.732 | 10.179 | 0 | 0 |
| argmax_i8_64k | 65536 | 1953 | 89.55 | 6.47 | 13.85 | 0.732 | 10.133 | 0 | 0 |

### ArgMax<uint8>

- Affects: `arg_max` uint8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_u8_1k | 1024 | 4000 | 1.41 | 0.18 | 7.77 | 0.725 | 5.631 | 0 | 0 |
| argmax_u8_16k | 16384 | 4000 | 22.40 | 1.61 | 13.92 | 0.731 | 10.184 | 0 | 0 |
| argmax_u8_64k | 65536 | 1953 | 89.66 | 6.37 | 14.08 | 0.731 | 10.296 | 0 | 0 |

## `tflite/kernels/fully_connected.cc`

- Summary: Dense float FullyConnected on RVV now takes the `EvalPie` route, so this split benchmark keeps the measurement alongside the source-file section and exercises the same matvec-style accumulation shape.

### FullyConnected<float> dense EvalPie route

- Affects: `fully_connected.cc` dense float RVV dispatch that now routes through `EvalPie`; this benchmark mirrors that matrix-batch-vector accumulation path inside the split operator harness.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_dense_small_b4 | B4 I64 O64 | 3906 | 24.32 | 21.20 | 1.15 | 1.347 | 1.545 | 0.00000000 | 0.00000000 | 0.0000000000 |
| fc_dense_mid_b4 | B4 I128 O128 | 976 | 86.84 | 87.36 | 0.99 | 1.509 | 1.500 | 0.00000000 | 0.00000000 | 0.0000000000 |
| fc_dense_large_b4 | B4 I2048 O640 | 20 | 7224.08 | 7211.14 | 1.00 | 1.452 | 1.454 | 0.00000000 | 0.00000000 | 0.0000000000 |

## `tflite/kernels/lstm_eval.cc`

- Summary: Float gate and output/projection paths now reuse the RVV-enabled `tensor_utils` matvec helpers, so this section keeps the operator-level numbers separate from generic helper benchmarks.

### LstmGate<float>

- Affects: `lstm_eval` float gate path (`input_to_gate` + `recurrent_to_gate` + sigmoid).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_gate_small_b1 | B1 I128 O128 C512 | 488 | 198.10 | 188.66 | 1.05 | 1.323 | 1.390 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_gate_small_b4 | B4 I128 O128 C512 | 122 | 791.63 | 752.59 | 1.05 | 1.325 | 1.393 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_gate_large_b1 | B1 I256 O256 C1024 | 122 | 763.77 | 763.91 | 1.00 | 1.373 | 1.373 | 0.00000000 | 0.00000000 | 0.0000000000 |

### LstmOutput<float>

- Affects: `lstm_eval` float output/projection path (`tanh(cell)` + output gate + projection).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_output_small_b1 | B1 C512 O128 | 972 | 108.11 | 107.64 | 1.00 | 1.217 | 1.222 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_output_small_b4 | B4 C512 O128 | 243 | 430.67 | 430.20 | 1.00 | 1.222 | 1.223 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_output_large_b1 | B1 C1024 O256 | 243 | 394.57 | 394.78 | 1.00 | 1.331 | 1.331 | 0.00000000 | 0.00000000 | 0.0000000000 |

## `tflite/kernels/internal/optimized/integer_ops/add.h`

- Summary: Recent RVV work added both int8/int16 elementwise add coverage and the scalar-broadcast kernels used by the fivefold broadcast fast path.

### Add<int8>

- Affects: `integer_ops/add.h` int8 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i8_1k | 1024 | 4000 | 10.51 | 5.89 | 1.78 | 0.097 | 0.174 | 0 | 0 |
| add_i8_4k | 4096 | 4000 | 52.39 | 23.58 | 2.22 | 0.078 | 0.174 | 0 | 0 |
| add_i8_16k | 16384 | 4000 | 211.09 | 94.24 | 2.24 | 0.078 | 0.174 | 0 | 0 |
| add_i8_64k | 65536 | 1953 | 861.14 | 377.15 | 2.28 | 0.076 | 0.174 | 0 | 0 |

### AddScalarBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast add kernel used by the fivefold broadcast fast path in `integer_ops/add.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_i8_1k | 1024 | 4000 | 6.46 | 4.17 | 1.55 | 0.159 | 0.245 | 0 | 0 |
| add_scalar_kernel_i8_4k | 4096 | 4000 | 34.29 | 16.67 | 2.06 | 0.119 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_16k | 16384 | 4000 | 122.91 | 66.67 | 1.84 | 0.133 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_64k | 65536 | 1953 | 489.38 | 266.89 | 1.83 | 0.134 | 0.246 | 0 | 0 |

### BroadcastAddDispatch<int8> scalar lhs

- Affects: `integer_ops/add.h` int8 scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_i8_1k | 1024 | 4000 | 14.51 | 14.54 | 1.00 | 0.071 | 0.070 | 0 | 0 |
| add_bcast_i8_4k | 4096 | 4000 | 68.73 | 68.59 | 1.00 | 0.060 | 0.060 | 0 | 0 |
| add_bcast_i8_16k | 16384 | 4000 | 276.18 | 276.12 | 1.00 | 0.059 | 0.059 | 0 | 0 |
| add_bcast_i8_64k | 65536 | 1953 | 1063.54 | 1061.61 | 1.00 | 0.062 | 0.062 | 0 | 0 |

### Add<int16>

- Affects: `integer_ops/add.h` int16 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i16_1k | 1024 | 4000 | 10.62 | 5.80 | 1.83 | 0.096 | 0.177 | 0 | 0 |
| add_i16_4k | 4096 | 4000 | 50.83 | 23.12 | 2.20 | 0.081 | 0.177 | 0 | 0 |
| add_i16_16k | 16384 | 4000 | 209.63 | 92.44 | 2.27 | 0.078 | 0.177 | 0 | 0 |
| add_i16_64k | 65536 | 1953 | 842.77 | 369.45 | 2.28 | 0.078 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mul.h`

- Summary: Recent RVV work added both int8 elementwise mul coverage and the scalar-broadcast kernels used by the quantized broadcast fast path.

### Mul<int8>

- Affects: `integer_ops/mul.h` int8 elementwise quantized mul path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_i8_1k | 1024 | 4000 | 10.73 | 3.22 | 3.33 | 0.095 | 0.318 | 0 | 0 |
| mul_i8_4k | 4096 | 4000 | 47.89 | 12.87 | 3.72 | 0.086 | 0.318 | 0 | 0 |
| mul_i8_16k | 16384 | 4000 | 195.75 | 51.41 | 3.81 | 0.084 | 0.319 | 0 | 0 |
| mul_i8_64k | 65536 | 1953 | 786.93 | 205.46 | 3.83 | 0.083 | 0.319 | 0 | 0 |

### MulSimpleBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast mul kernel used by the fivefold broadcast fast path in `integer_ops/mul.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_i8_1k | 1024 | 4000 | 9.82 | 3.29 | 2.99 | 0.104 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_4k | 4096 | 4000 | 44.99 | 13.18 | 3.41 | 0.091 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_16k | 16384 | 4000 | 183.96 | 52.71 | 3.49 | 0.089 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_64k | 65536 | 1953 | 739.63 | 210.98 | 3.51 | 0.089 | 0.311 | 0 | 0 |

### BroadcastMulDispatch<int8> scalar lhs

- Affects: `integer_ops/mul.h` int8 scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_i8_1k | 1024 | 4000 | 10.32 | 10.21 | 1.01 | 0.099 | 0.100 | 0 | 0 |
| mul_bcast_i8_4k | 4096 | 4000 | 47.47 | 47.46 | 1.00 | 0.086 | 0.086 | 0 | 0 |
| mul_bcast_i8_16k | 16384 | 4000 | 194.21 | 193.95 | 1.00 | 0.084 | 0.084 | 0 | 0 |
| mul_bcast_i8_64k | 65536 | 1953 | 779.90 | 779.55 | 1.00 | 0.084 | 0.084 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/sub.h`

- Summary: This section isolates the int16 quantized subtract path that gained a dedicated RVV elementwise implementation.

### Sub<int16>

- Affects: `integer_ops/sub.h` int16 elementwise quantized sub path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_i16_1k | 1024 | 4000 | 9.97 | 5.79 | 1.72 | 0.103 | 0.177 | 0 | 0 |
| sub_i16_4k | 4096 | 4000 | 50.35 | 23.10 | 2.18 | 0.081 | 0.177 | 0 | 0 |
| sub_i16_16k | 16384 | 4000 | 205.70 | 92.49 | 2.22 | 0.080 | 0.177 | 0 | 0 |
| sub_i16_64k | 65536 | 1953 | 827.32 | 369.52 | 2.24 | 0.079 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/pooling.h`

- Summary: This section keeps the signed int8 pooling kernels separate from the uint8 `optimized_ops.h` pooling paths, because their RVV logic and rounding semantics are different.

### AveragePool<int8>

- Affects: `average_pool` int8 depth inner loop, signed round-away-from-zero, and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 176.12 | 64.69 | 2.72 | 0.372 | 1.013 | 0 | 0 |
| avgpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1401.95 | 684.16 | 2.05 | 0.599 | 1.228 | 0 | 0 |
| avgpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 551.48 | 271.64 | 2.03 | 0.602 | 1.221 | 0 | 0 |

### MaxPool<int8>

- Affects: `max_pool` int8 depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 222.45 | 24.15 | 9.21 | 0.295 | 2.714 | 0 | 0 |
| maxpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2191.75 | 239.42 | 9.15 | 0.383 | 3.508 | 0 | 0 |
| maxpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 834.54 | 95.04 | 8.78 | 0.398 | 3.491 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h`

- Summary: This section tracks the RVV general depthwise int8 kernels, including the fixed-depth and fixed-depth-multiplier specializations added to mirror the NEON dispatch table.

### DepthwiseConv<int8>

- Affects: `depthwise_conv` int8 general path, including fixed input-depth and depth-multiplier RVV kernels aligned with NEON dispatch.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dwconv_i8_dm1_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM1 | 1975 | 185.47 | 116.47 | 1.59 | 0.349 | 0.556 | 0 | 0 |
| dwconv_i8_dm1_d8_s1 | B1 I32x32x8 F3x3 S1x1 DM1 | 987 | 342.97 | 153.20 | 2.24 | 0.378 | 0.846 | 0 | 0 |
| dwconv_i8_dm1_d12_s1 | B1 I32x32x12 F3x3 S1x1 DM1 | 658 | 509.75 | 196.75 | 2.59 | 0.381 | 0.988 | 0 | 0 |
| dwconv_i8_dm1_d16_s2 | B1 I32x32x16 F3x3 S2x2 DM1 | 1975 | 170.66 | 72.82 | 2.34 | 0.380 | 0.890 | 0 | 0 |
| dwconv_i8_dm2_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM2 | 987 | 326.97 | 194.34 | 1.68 | 0.396 | 0.667 | 0 | 0 |
| dwconv_i8_dm2_d8_s2 | B1 I32x32x8 F3x3 S2x2 DM2 | 1975 | 158.94 | 97.61 | 1.63 | 0.408 | 0.664 | 0 | 0 |
| dwconv_i8_dm4_d1_s1 | B1 I32x32x1 F3x3 S1x1 DM4 | 1975 | 146.07 | 109.22 | 1.34 | 0.444 | 0.593 | 0 | 0 |
| dwconv_i8_dm4_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM4 | 493 | 508.94 | 389.50 | 1.31 | 0.509 | 0.665 | 0 | 0 |
| dwconv_i8_dm8_d2_s1 | B1 I32x32x2 F3x3 S1x1 DM8 | 493 | 448.59 | 284.59 | 1.58 | 0.578 | 0.911 | 0 | 0 |
| dwconv_i8_dm8_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM8 | 3950 | 59.10 | 44.14 | 1.34 | 0.548 | 0.734 | 0 | 0 |
| dwconv_i8_dm16_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM16 | 1975 | 103.84 | 65.81 | 1.58 | 0.624 | 0.985 | 0 | 0 |
| dwconv_i8_dm20_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM20 | 1580 | 127.26 | 97.24 | 1.31 | 0.636 | 0.833 | 0 | 0 |
| dwconv_i8_dm32_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM32 | 987 | 193.47 | 130.65 | 1.48 | 0.670 | 0.992 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h`

- Note: The recent change in this file only reroutes hybrid dispatch to reuse the RVV depthwise kernels above, so it intentionally shares the same numeric benchmark coverage.

## `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h`

- Summary: This section isolates the int16 quantized LeakyReLU path that received a dedicated RVV implementation.

### LeakyRelu<int16>

- Affects: `integer_ops/leaky_relu.h` int16 quantized leaky-relu path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leaky_relu_i16_1k | 1024 | 4000 | 12.58 | 4.08 | 3.08 | 0.081 | 0.251 | 0 | 0 |
| leaky_relu_i16_4k | 4096 | 4000 | 50.70 | 16.34 | 3.10 | 0.081 | 0.251 | 0 | 0 |
| leaky_relu_i16_16k | 16384 | 4000 | 201.43 | 65.28 | 3.09 | 0.081 | 0.251 | 0 | 0 |
| leaky_relu_i16_64k | 65536 | 1953 | 809.74 | 261.39 | 3.10 | 0.081 | 0.251 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/lut.h`

- Summary: This section keeps the uint8 and int8 lookup-table kernels separate because they were vectorized in the same commit but have different input domains.

### LookupTable<uint8>

- Affects: `integer_ops/lut.h` uint8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_u8_1k | 1024 | 4000 | 1.41 | 0.85 | 1.66 | 0.728 | 1.208 | 0 | 0 |
| lut_u8_4k | 4096 | 4000 | 5.61 | 3.40 | 1.65 | 0.730 | 1.206 | 0 | 0 |
| lut_u8_16k | 16384 | 4000 | 22.41 | 13.57 | 1.65 | 0.731 | 1.208 | 0 | 0 |
| lut_u8_64k | 65536 | 1953 | 89.57 | 54.30 | 1.65 | 0.732 | 1.207 | 0 | 0 |

### LookupTable<int8>

- Affects: `integer_ops/lut.h` int8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_i8_1k | 1024 | 4000 | 1.41 | 0.85 | 1.66 | 0.729 | 1.209 | 0 | 0 |
| lut_i8_4k | 4096 | 4000 | 5.60 | 3.39 | 1.65 | 0.732 | 1.208 | 0 | 0 |
| lut_i8_16k | 16384 | 4000 | 22.39 | 13.57 | 1.65 | 0.732 | 1.208 | 0 | 0 |
| lut_i8_64k | 65536 | 1953 | 89.57 | 54.18 | 1.65 | 0.732 | 1.210 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mean.h`

- Summary: This section tracks the int8 height-width reduction path that gained RVV coverage in the latest integer-ops commit.

### Mean<int8>

- Affects: `integer_ops/mean.h` int8 height-width reduction path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_i8_8x8x64 | B1 I8x8x64 A(1,2) | 4000 | 5.84 | 1.63 | 3.58 | 0.702 | 2.515 | 0 | 0 |
| mean_i8_16x16x128 | B1 I16x16x128 A(1,2) | 3906 | 36.56 | 11.61 | 3.15 | 0.896 | 2.823 | 0 | 0 |
| mean_i8_32x32x256 | B1 I32x32x256 A(1,2) | 488 | 1446.95 | 96.43 | 15.01 | 0.181 | 2.719 | 0 | 0 |

