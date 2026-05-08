# RVV Scalar Benchmark Results

- Date: 2026-05-08 15:01:54 CST
- Host: Linux k3 6.18.3-generic #1.0.0~rc4.4 SMP PREEMPT_DYNAMIC Wed Apr 29 10:51:17 CST 2026 riscv64 GNU/Linux
- Compiler: g++ (Bianbu 15.2.0-16ubuntu1bb2) 15.2.0
- Scope: recent five commits that introduced or wired RVV coverage, plus follow-up P1 gap closures for `reduce.h` and `resize_bilinear.h`

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
| `tflite/kernels/internal/optimized/reduce.h` | Follow-up P1 closure (2026-05-08) | uint8 keep-dims height-width mean and float last-dim mean |
| `tflite/kernels/internal/optimized/resize_bilinear.h` | Follow-up P1 closure (2026-05-08) | float and uint8 generic bilinear resize kernels |

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
| zero_f32_1024 | 1024 | 4000 | 1.76 | 0.91 | 1.93 | 0.583 | 1.124 | 0 | 0 |
| zero_f32_4096 | 4096 | 4000 | 7.02 | 3.63 | 1.94 | 0.584 | 1.130 | 0 | 0 |

### IsZeroVector<int8>

- Affects: zero-skip checks for quantized helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_i8_64 | 64 | 4000 | 0.03 | 0.02 | 2.00 | 2.007 | 4.019 | 0 | 0 |
| zero_i8_1024 | 1024 | 4000 | 0.47 | 0.22 | 2.14 | 2.167 | 4.645 | 0 | 0 |
| zero_i8_4096 | 4096 | 4000 | 1.89 | 0.88 | 2.14 | 2.170 | 4.640 | 0 | 0 |

### VectorVectorDotProduct<float>

- Affects: `fully_connected`, recurrent math, shared float tensor-utils call sites.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dot_64 | 64 | 4000 | 0.09 | 0.03 | 2.95 | 1.401 | 4.138 | 0.00000000 | 0.00000000 | 0.0000000000 |
| dot_256 | 256 | 4000 | 0.36 | 0.08 | 4.46 | 1.433 | 6.399 | 0.00000119 | 0.00000046 | 0.0000011921 |
| dot_1024 | 1024 | 4000 | 1.41 | 0.29 | 4.92 | 1.457 | 7.171 | 0.00000191 | 0.00000016 | 0.0000019073 |
| dot_4096 | 4096 | 4000 | 5.62 | 1.07 | 5.25 | 1.458 | 7.649 | 0.00002384 | 0.00000264 | 0.0000238419 |

### BatchVectorBatchVectorDotProduct<int16>

- Affects: `svdf` and batched recurrent dot-product helper paths.

| Case | Batch x size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_dot_q15_small | 4 x 64 | 4000 | 0.29 | 0.09 | 3.11 | 1.791 | 5.562 | 0 | 0 |
| svdf_dot_q15_mid | 8 x 256 | 4000 | 2.13 | 0.49 | 4.36 | 1.921 | 8.369 | 0 | 0 |
| svdf_dot_q15_large | 16 x 1024 | 4000 | 16.81 | 3.66 | 4.59 | 1.950 | 8.942 | 0 | 0 |

### ReductionSumVector<float>

- Affects: `svdf` and other float reduction-style helper paths.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_small | 64 x 8 | 4000 | 1.00 | 0.77 | 1.30 | 0.514 | 0.666 | 0.00000048 | 0.00000080 | 0.0000000559 |
| svdf_mid | 128 x 16 | 4000 | 4.22 | 1.76 | 2.39 | 0.485 | 1.161 | 0.00000095 | 0.00001488 | 0.0000001378 |
| svdf_large | 256 x 64 | 4000 | 35.04 | 6.34 | 5.53 | 0.468 | 2.584 | 0.00000572 | 0.00001453 | 0.0000005609 |
| reduce_wide | 256 x 256 | 2441 | 141.37 | 17.50 | 8.08 | 0.464 | 3.744 | 0.00001144 | 0.00016069 | 0.0000021656 |

### ReductionSumVector<int8 -> int32>

- Affects: `conv`, `batch_matmul`, `kernel_utils`, cached row sum preparation.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| row_sum_small | 64 x 32 | 4000 | 1.09 | 1.07 | 1.02 | 1.885 | 1.923 | 0 | 0 |
| row_sum_mid | 128 x 128 | 4000 | 5.66 | 4.75 | 1.19 | 2.896 | 3.453 | 0 | 0 |
| row_sum_large | 256 x 256 | 2441 | 20.64 | 16.59 | 1.24 | 3.175 | 3.951 | 0 | 0 |
| row_sum_conv_like | 512 x 512 | 610 | 79.55 | 61.12 | 1.30 | 3.296 | 4.289 | 0 | 0 |

### ReductionSumVector<int32>

- Affects: scalar accumulation helper paths such as reference SVDF and utility reductions.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reduce_i32_small | 64 x 32 | 4000 | 0.91 | 0.77 | 1.19 | 2.247 | 2.668 | 0 | 0 |
| reduce_i32_mid | 128 x 128 | 4000 | 4.64 | 2.99 | 1.55 | 3.529 | 5.473 | 0 | 0 |
| reduce_i32_large | 256 x 256 | 2441 | 16.74 | 10.68 | 1.57 | 3.914 | 6.134 | 0 | 0 |

### MatrixScalarMultiplyAccumulate<int8>

- Affects: quantized recurrent helpers and row-wise reduction paths.

| Case | Rows x cols | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_rows_64 | 64 x 64 | 4000 | 4.11 | 1.58 | 2.60 | 0.996 | 2.588 | 0 | 0 |
| fc_rows_256 | 256 x 128 | 4000 | 30.48 | 9.76 | 3.12 | 1.075 | 3.357 | 0 | 0 |
| conv_rows_512 | 512 x 256 | 1220 | 122.56 | 33.49 | 3.66 | 1.069 | 3.913 | 0 | 0 |
| conv_rows_1024 | 1024 x 512 | 305 | 483.59 | 122.88 | 3.94 | 1.084 | 4.267 | 0 | 0 |

### MatrixBatchVectorMultiplyAccumulate<float>

- Affects: float `fully_connected` and recurrent kernels.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_fc | 64 x 64 x 4 | 4000 | 24.25 | 7.25 | 3.34 | 1.351 | 4.518 | 0.00000286 | 0.00001540 | 0.0000003163 |
| mid_fc | 128 x 128 x 4 | 2441 | 87.74 | 24.13 | 3.64 | 1.494 | 5.433 | 0.00000525 | 0.00006299 | 0.0000006228 |
| lstm_like | 640 x 2048 x 4 | 30 | 7207.66 | 2353.35 | 3.06 | 1.455 | 4.456 | 0.00008774 | 0.00183174 | 0.0000094324 |
| sqrnn_like | 1024 x 1024 x 8 | 20 | 11561.51 | 3659.95 | 3.16 | 1.451 | 4.584 | 0.00005531 | 0.00156548 | 0.0000045446 |

### MatrixBatchVectorMultiplyAccumulate<int8>

- Affects: quantized `fully_connected`, hybrid recurrent helpers, shared int8 GEMV-style call sites.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_fc_small | 64 x 64 x 4 | 4000 | 16.59 | 6.97 | 2.38 | 1.975 | 4.698 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_fc_mid | 128 x 128 x 4 | 2441 | 70.44 | 20.89 | 3.37 | 1.861 | 6.273 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_lstm_like | 640 x 1024 x 4 | 61 | 2786.08 | 597.10 | 4.67 | 1.882 | 8.781 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_conv_like | 1024 x 512 x 8 | 38 | 4466.62 | 1004.56 | 4.45 | 1.878 | 8.351 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MatrixBatchVectorMultiplyAccumulate<int8, per-channel + input_offset>

- Affects: quantized `conv`, `batch_matmul`, and any path that uses cached row sums plus per-channel scale.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_row_sum_small | 64 x 64 x 4 | 4000 | 26.08 | 8.62 | 3.02 | 1.257 | 3.799 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_mid | 128 x 128 x 4 | 2441 | 100.11 | 25.89 | 3.87 | 1.309 | 5.062 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_conv | 256 x 576 x 8 | 135 | 1721.95 | 314.58 | 5.47 | 1.370 | 7.500 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_batch_matmul_like | 512 x 512 x 8 | 76 | 3064.62 | 565.69 | 5.42 | 1.369 | 7.414 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MatrixBatchVectorMultiplyAccumulate<int8 -> int16>

- Affects: quantized recurrent gate accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q15_small | 128 x 128 x 4 | 2441 | 76.44 | 21.46 | 3.56 | 1.715 | 6.108 | 0 | 0 |
| gate_q15_mid | 256 x 256 x 4 | 610 | 287.73 | 71.03 | 4.05 | 1.822 | 7.381 | 0 | 0 |
| gate_q15_large | 512 x 512 x 8 | 76 | 2298.89 | 507.54 | 4.53 | 1.824 | 8.264 | 0 | 0 |

### MatrixBatchVectorMultiplyAccumulate<int8 -> int8>

- Affects: quantized projection and low-precision recurrent accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q8_small | 128 x 128 x 4 | 2441 | 77.68 | 21.61 | 3.59 | 1.687 | 6.064 | 0 | 0 |
| gate_q8_mid | 256 x 256 x 4 | 610 | 292.00 | 71.41 | 4.09 | 1.796 | 7.341 | 0 | 0 |
| gate_q8_large | 512 x 512 x 8 | 76 | 2251.57 | 508.75 | 4.43 | 1.863 | 8.244 | 0 | 0 |

### MatrixBatchVectorMultiply<int8 -> int8>

- Affects: quantized LSTM gate matmul paths before saturating-add and activation.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_noacc_q8_small | 128 x 128 x 4 | 2441 | 81.44 | 30.53 | 2.67 | 1.609 | 4.294 | 0 | 0 |
| gate_noacc_q8_mid | 256 x 256 x 4 | 610 | 303.18 | 100.57 | 3.01 | 1.729 | 5.213 | 0 | 0 |
| gate_noacc_q8_large | 512 x 512 x 8 | 76 | 2231.40 | 702.75 | 3.18 | 1.880 | 5.968 | 0 | 0 |

### MatrixBatchVectorMultiply<int16 x int8 -> int8>

- Affects: quantized projection/output matmul helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| proj_q8_small | 128 x 128 x 4 | 2441 | 422.75 | 32.30 | 13.09 | 0.310 | 4.058 | 0 | 0 |
| proj_q8_mid | 256 x 256 x 4 | 610 | 1671.52 | 117.46 | 14.23 | 0.314 | 4.464 | 0 | 0 |
| proj_q8_large | 512 x 512 x 8 | 76 | 13283.31 | 896.47 | 14.82 | 0.316 | 4.679 | 0 | 0 |

### SparseMatrixBatchVectorMultiplyAccumulate1x4<float>

- Affects: sparse float `fully_connected` and sparse recurrent helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_1x4_small | 64 x 256 x 4 | 4000 | 22.09 | 11.93 | 1.85 | 1.469 | 2.721 | 0.00000143 | 0.00012937 | 0.0000003011 |
| sparse_1x4_mid | 128 x 512 x 4 | 2448 | 97.59 | 50.03 | 1.95 | 1.339 | 2.612 | 0.00000477 | 0.00038041 | 0.0000006075 |
| sparse_1x4_large | 256 x 1024 x 8 | 304 | 753.89 | 373.76 | 2.02 | 1.393 | 2.810 | 0.00000954 | 0.00038029 | 0.0000011599 |

### SparseMatrixBatchVectorMultiplyAccumulate<float ledger>

- Affects: sparse float matvec helper paths with ledger format.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_ledger_small | 64 x 256 x 4 | 4000 | 24.99 | 11.69 | 2.14 | 1.291 | 2.758 | 0.00000238 | 0.00000709 | 0.0000003208 |
| sparse_ledger_mid | 128 x 512 x 4 | 2362 | 103.82 | 41.53 | 2.50 | 1.304 | 3.261 | 0.00000572 | 0.00038695 | 0.0000006152 |
| sparse_ledger_large | 256 x 1024 x 8 | 303 | 809.81 | 294.31 | 2.75 | 1.304 | 3.588 | 0.00001240 | 0.00644636 | 0.0000011960 |

### SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>

- Affects: sparse quantized matvec helper paths that dequantize to float.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_qfloat_small | 64 x 256 x 4 | 4000 | 15.46 | 10.38 | 1.49 | 2.094 | 3.120 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_mid | 128 x 512 x 4 | 2465 | 58.55 | 32.98 | 1.78 | 2.217 | 3.936 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_large | 256 x 1024 x 8 | 299 | 482.52 | 234.15 | 2.06 | 2.214 | 4.563 | 0.00000000 | 0.00000000 | 0.0000000000 |

### SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>

- Affects: sparse quantized fully-connected and recurrent output helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_q8_small | 64 x 256 x 4 | 4000 | 33.82 | 15.78 | 2.14 | 1.011 | 2.166 | 0 | 0 |
| sparse_q8_mid | 128 x 512 x 4 | 2394 | 125.03 | 46.99 | 2.66 | 1.069 | 2.844 | 0 | 0 |
| sparse_q8_large | 256 x 1024 x 8 | 302 | 944.38 | 286.55 | 3.30 | 1.120 | 3.692 | 0 | 0 |

### SymmetricQuantizeFloats

- Affects: hybrid `fully_connected`, `batch_matmul`, weight and activation pre-quant helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Min diff | Max diff | Scale diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sym_quant_small | 64 | 4000 | 0.46 | 0.15 | 2.99 | 0.419 | 1.252 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_mid | 1024 | 4000 | 7.28 | 2.68 | 2.72 | 0.422 | 1.147 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_large | 4096 | 4000 | 32.50 | 14.99 | 2.17 | 0.378 | 0.820 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |

### AsymmetricQuantizeFloats

- Affects: hybrid `conv`, `depthwise_conv`, `transpose_conv`, and int8 input staging paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Scale diff | Offset diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asym_quant_small | 64 | 4000 | 0.58 | 0.26 | 2.19 | 0.332 | 0.727 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_mid | 1024 | 4000 | 8.71 | 3.30 | 2.64 | 0.353 | 0.930 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_large | 4096 | 4000 | 37.30 | 16.31 | 2.29 | 0.329 | 0.753 | 0 | 0 | 0.00000000 | 0 |

### ApplyLayerNorm<int16>

- Affects: quantized recurrent gate normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ln_small | 4 x 64 | 4000 | 5.91 | 2.38 | 2.48 | 0.347 | 0.861 | 0 | 0 |
| ln_mid | 4 x 256 | 4000 | 24.14 | 8.66 | 2.79 | 0.339 | 0.946 | 0 | 0 |
| ln_gate_like | 8 x 1024 | 2441 | 192.82 | 67.50 | 2.86 | 0.340 | 0.971 | 0 | 0 |

### ApplySigmoid<int16>

- Affects: quantized recurrent gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoid_small | 4 x 64 | 4000 | 22.39 | 8.62 | 2.60 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_mid | 8 x 256 | 4000 | 192.34 | 68.89 | 2.79 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_large | 4 x 1024 | 4000 | 386.19 | 137.79 | 2.80 | 0.011 | 0.030 | 0 | 0 |

### ApplyTanh<int16>

- Affects: quantized recurrent state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanh_q0_small | 0 x 4 x 64 | 4000 | 21.23 | 8.25 | 2.57 | 0.012 | 0.031 | 0 | 0 |
| tanh_q3_mid | 3 x 8 x 256 | 4000 | 200.83 | 71.78 | 2.80 | 0.010 | 0.029 | 0 | 0 |
| tanh_q4_large | 4 x 4 x 1024 | 4000 | 405.20 | 150.34 | 2.70 | 0.010 | 0.027 | 0 | 0 |

### ApplyLayerNormFloat<int16>

- Affects: float-reference layer-norm helper paths used by quantized LSTM eval.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lnf_small | 4 x 64 | 4000 | 3.69 | 1.48 | 2.50 | 0.554 | 1.385 | 0 | 0 |
| lnf_mid | 4 x 256 | 4000 | 13.90 | 5.34 | 2.60 | 0.589 | 1.533 | 0 | 0 |
| lnf_gate_like | 8 x 1024 | 2441 | 109.26 | 40.94 | 2.67 | 0.600 | 1.601 | 0 | 0 |

### ApplySigmoidFloat<int16>

- Affects: float-reference gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoidf_small | 4 x 64 | 4000 | 9.39 | 8.62 | 1.09 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 2.0703125000 |
| sigmoidf_mid | 8 x 256 | 4000 | 74.72 | 68.92 | 1.08 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 1.9697265625 |
| sigmoidf_large | 4 x 1024 | 4000 | 149.44 | 137.76 | 1.08 | 0.027 | 0.030 | 6.00000000 | 0.21052632 | 1.9594726562 |

### ApplyTanhFloat<int16>

- Affects: float-reference state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanhf_qm12_small | -12 x 4 x 64 | 4000 | 24.74 | 8.88 | 2.78 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm12_mid | -12 x 8 x 256 | 4000 | 197.84 | 70.14 | 2.82 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm15_large | -15 x 4 x 1024 | 4000 | 395.59 | 140.16 | 2.82 | 0.010 | 0.029 | 0 | 0 |

### CwiseMul<int16 -> int16>

- Affects: quantized recurrent elementwise gate math.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q15_small | 4 x 64 | 4000 | 0.57 | 0.44 | 1.29 | 0.450 | 0.582 | 0 | 0 |
| mul_q15_mid | 8 x 256 | 4000 | 4.37 | 3.46 | 1.26 | 0.469 | 0.592 | 0 | 0 |
| mul_q15_large | 4 x 1024 | 4000 | 8.62 | 6.92 | 1.24 | 0.475 | 0.592 | 0 | 0 |

### CwiseMul<int16 -> int8>

- Affects: quantized projection and low-precision elementwise paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q8_small | 4 x 64 | 4000 | 2.48 | 0.77 | 3.23 | 0.103 | 0.333 | 0 | 0 |
| mul_q8_mid | 8 x 256 | 4000 | 23.89 | 6.02 | 3.97 | 0.086 | 0.340 | 0 | 0 |
| mul_q8_large | 4 x 1024 | 4000 | 49.42 | 12.05 | 4.10 | 0.083 | 0.340 | 0 | 0 |

### CwiseAdd<int16>

- Affects: recurrent residual/additive helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_small | 4 x 64 | 4000 | 0.70 | 0.33 | 2.10 | 0.364 | 0.766 | 0 | 0 |
| add_mid | 8 x 256 | 4000 | 6.33 | 2.69 | 2.36 | 0.323 | 0.762 | 0 | 0 |
| add_large | 4 x 1024 | 4000 | 15.08 | 5.23 | 2.88 | 0.272 | 0.783 | 0 | 0 |

### TwoGateSaturatingAdd<int8 -> int16>

- Affects: quantized LSTM gate merge helper paths.

| Case | Batch x cell | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| two_gate_small | 4 x 64 | 4000 | 4.17 | 1.30 | 3.20 | 0.246 | 0.786 | 0 | 0 |
| two_gate_mid | 8 x 256 | 4000 | 41.95 | 10.25 | 4.09 | 0.195 | 0.800 | 0 | 0 |
| two_gate_large | 4 x 1024 | 4000 | 84.24 | 20.44 | 4.12 | 0.195 | 0.802 | 0 | 0 |

### CwiseClipping<float>

- Affects: activation clamp paths in float recurrent/helper code.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_f32_64 | 64 | 4000 | 0.07 | 0.05 | 1.54 | 0.880 | 1.353 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_1024 | 1024 | 4000 | 1.17 | 0.76 | 1.54 | 0.879 | 1.350 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_4096 | 4096 | 4000 | 4.67 | 3.05 | 1.53 | 0.877 | 1.343 | 0.00000000 | 0.00000000 | 0.0000000000 |

### CwiseClipping<int16>

- Affects: quantized activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i16_64 | 64 | 4000 | 0.02 | 0.02 | 1.00 | 3.195 | 3.195 | 0 | 0 |
| clip_i16_1024 | 1024 | 4000 | 0.32 | 0.32 | 1.00 | 3.200 | 3.184 | 0 | 0 |
| clip_i16_4096 | 4096 | 4000 | 1.29 | 1.29 | 1.00 | 3.173 | 3.164 | 0 | 0 |

### CwiseClipping<int8>

- Affects: low-precision activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i8_64 | 64 | 4000 | 0.01 | 0.02 | 0.91 | 4.462 | 4.045 | 0 | 0 |
| clip_i8_1024 | 1024 | 4000 | 0.16 | 0.16 | 0.99 | 6.399 | 6.350 | 0 | 0 |
| clip_i8_4096 | 4096 | 4000 | 0.64 | 0.64 | 1.00 | 6.400 | 6.377 | 0 | 0 |

### VectorBatchVectorCwiseProductAccumulate<int16>

- Affects: quantized recurrent gate/state accumulation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vbv_cwise_small | 4 x 64 | 4000 | 2.33 | 0.81 | 2.87 | 0.220 | 0.632 | 0 | 0 |
| vbv_cwise_mid | 8 x 256 | 4000 | 23.15 | 6.30 | 3.68 | 0.177 | 0.650 | 0 | 0 |
| vbv_cwise_large | 4 x 1024 | 4000 | 47.36 | 12.61 | 3.76 | 0.173 | 0.650 | 0 | 0 |

### Sub1Vector<float>

- Affects: float recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_f32_64 | 64 | 4000 | 0.06 | 0.04 | 1.70 | 1.035 | 1.759 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_1024 | 1024 | 4000 | 0.94 | 0.58 | 1.62 | 1.088 | 1.760 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_4096 | 4096 | 4000 | 3.75 | 2.35 | 1.60 | 1.094 | 1.747 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Sub1Vector<int16>

- Affects: quantized recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_i16_64 | 64 | 4000 | 0.06 | 0.02 | 3.37 | 1.042 | 3.517 | 0 | 0 |
| sub1_i16_1024 | 1024 | 4000 | 0.94 | 0.29 | 3.22 | 1.088 | 3.504 | 0 | 0 |
| sub1_i16_4096 | 4096 | 4000 | 3.74 | 1.18 | 3.17 | 1.097 | 3.480 | 0 | 0 |

### VectorScalarMultiply<int8 -> float>

- Affects: hybrid dequant-style helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale_i8_64 | 64 | 4000 | 0.09 | 0.02 | 4.30 | 1.422 | 6.114 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_1024 | 1024 | 4000 | 1.41 | 0.32 | 4.39 | 1.456 | 6.391 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_4096 | 4096 | 4000 | 5.61 | 1.28 | 4.38 | 1.461 | 6.393 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MeanStddevNormalization<float>

- Affects: float recurrent normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| norm_1x64 | 1 x 64 | 4000 | 0.27 | 0.21 | 1.29 | 0.939 | 1.211 | 0.00000024 | 0.00000023 | 0.0000000843 |
| norm_4x256 | 4 x 256 | 4000 | 3.94 | 2.87 | 1.37 | 1.039 | 1.427 | 0.00000048 | 0.00002272 | 0.0000001109 |
| norm_8x1024 | 8 x 1024 | 4000 | 30.25 | 21.91 | 1.38 | 1.083 | 1.496 | 0.00000083 | 0.00009705 | 0.0000001304 |


## `tflite/kernels/internal/optimized/optimized_ops.h`

- Summary: Float arithmetic fast paths, quantize/dequant-style helpers, uint8 pooling, quantized HardSwish, and argmin/argmax cases that gained RVV coverage in the latest operator-level commits.

### Add<float>

- Affects: `add` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_f32_1k | 1024 | 4000 | 1.89 | 0.34 | 5.60 | 0.542 | 3.035 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_4k | 4096 | 4000 | 7.49 | 1.28 | 5.83 | 0.547 | 3.189 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_16k | 16384 | 4000 | 29.91 | 7.36 | 4.06 | 0.548 | 2.225 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_64k | 65536 | 1953 | 130.78 | 27.93 | 4.68 | 0.501 | 2.346 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AddScalarBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float add kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_f32_1k | 1024 | 4000 | 1.95 | 0.26 | 7.43 | 0.525 | 3.897 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_4k | 4096 | 4000 | 7.80 | 1.05 | 7.44 | 0.525 | 3.908 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_16k | 16384 | 4000 | 31.29 | 4.28 | 7.31 | 0.524 | 3.828 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_64k | 65536 | 1953 | 125.85 | 16.87 | 7.46 | 0.521 | 3.885 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastAddDispatch<float> scalar lhs

- Affects: `add` float scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_f32_1k | 1024 | 4000 | 2.40 | 0.34 | 7.12 | 0.426 | 3.032 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_4k | 4096 | 4000 | 9.39 | 1.12 | 8.38 | 0.436 | 3.653 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_16k | 16384 | 4000 | 38.49 | 4.43 | 8.69 | 0.426 | 3.697 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_64k | 65536 | 1953 | 149.44 | 16.98 | 8.80 | 0.439 | 3.861 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Mul<float>

- Affects: `mul` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_f32_1k | 1024 | 4000 | 2.82 | 0.36 | 7.76 | 0.363 | 2.817 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_4k | 4096 | 4000 | 11.22 | 1.40 | 8.01 | 0.365 | 2.925 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_16k | 16384 | 4000 | 44.81 | 7.46 | 6.00 | 0.366 | 2.196 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_64k | 65536 | 1953 | 179.22 | 25.27 | 7.09 | 0.366 | 2.594 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MulSimpleBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float mul kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_f32_1k | 1024 | 4000 | 2.34 | 0.29 | 8.05 | 0.437 | 3.520 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_4k | 4096 | 4000 | 9.34 | 1.16 | 8.02 | 0.438 | 3.518 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_16k | 16384 | 4000 | 37.40 | 4.70 | 7.96 | 0.438 | 3.487 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_64k | 65536 | 1953 | 149.42 | 18.73 | 7.98 | 0.439 | 3.499 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastMulDispatch<float> scalar lhs

- Affects: `mul` float scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_f32_1k | 1024 | 4000 | 2.40 | 0.37 | 6.50 | 0.426 | 2.768 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_4k | 4096 | 4000 | 9.46 | 1.24 | 7.66 | 0.433 | 3.316 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_16k | 16384 | 4000 | 38.16 | 4.92 | 7.75 | 0.429 | 3.328 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_64k | 65536 | 1953 | 150.79 | 18.87 | 7.99 | 0.435 | 3.473 | 0.00000000 | 0.00000000 | 0.0000000000 |

### SubWithActivation<float>

- Affects: `sub` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_f32_1k | 1024 | 4000 | 1.89 | 0.33 | 5.68 | 0.543 | 3.080 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_4k | 4096 | 4000 | 7.49 | 1.28 | 5.87 | 0.547 | 3.209 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_16k | 16384 | 4000 | 29.94 | 7.36 | 4.07 | 0.547 | 2.227 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_64k | 65536 | 1953 | 119.45 | 27.11 | 4.41 | 0.549 | 2.417 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Div<float>

- Affects: `div` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| div_f32_1k | 1024 | 4000 | 4.69 | 2.35 | 2.00 | 0.218 | 0.437 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_4k | 4096 | 4000 | 18.70 | 9.31 | 2.01 | 0.219 | 0.440 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_16k | 16384 | 4000 | 74.70 | 37.36 | 2.00 | 0.219 | 0.438 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_64k | 65536 | 1953 | 298.79 | 149.12 | 2.00 | 0.219 | 0.439 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AffineQuantize<int8>

- Affects: `quantize` float-to-int8 path used by int8 activations and weights.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i8_1k | 1024 | 4000 | 15.67 | 1.12 | 13.96 | 0.065 | 0.912 | 0 | 0 |
| affine_q_i8_4k | 4096 | 4000 | 79.42 | 4.45 | 17.84 | 0.052 | 0.920 | 0 | 0 |
| affine_q_i8_16k | 16384 | 4000 | 332.36 | 17.78 | 18.69 | 0.049 | 0.922 | 0 | 0 |
| affine_q_i8_64k | 65536 | 1953 | 1324.05 | 70.94 | 18.67 | 0.049 | 0.924 | 0 | 0 |

### AffineQuantize<uint8>

- Affects: `quantize` float-to-uint8 path used by uint8 activations.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_u8_1k | 1024 | 4000 | 8.91 | 1.40 | 6.38 | 0.115 | 0.733 | 0 | 0 |
| affine_q_u8_4k | 4096 | 4000 | 35.66 | 5.63 | 6.34 | 0.115 | 0.728 | 0 | 0 |
| affine_q_u8_16k | 16384 | 4000 | 142.87 | 22.45 | 6.36 | 0.115 | 0.730 | 0 | 0 |
| affine_q_u8_64k | 65536 | 1953 | 569.69 | 89.58 | 6.36 | 0.115 | 0.732 | 0 | 0 |

### AffineQuantize<int16>

- Affects: `quantize` float-to-int16 path used by q15-style operator staging.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i16_1k | 1024 | 4000 | 8.89 | 1.01 | 8.78 | 0.115 | 1.011 | 0 | 0 |
| affine_q_i16_4k | 4096 | 4000 | 35.49 | 3.99 | 8.89 | 0.115 | 1.027 | 0 | 0 |
| affine_q_i16_16k | 16384 | 4000 | 142.13 | 15.90 | 8.94 | 0.115 | 1.030 | 0 | 0 |
| affine_q_i16_64k | 65536 | 1953 | 567.27 | 63.49 | 8.93 | 0.116 | 1.032 | 0 | 0 |

### AveragePool<uint8>

- Affects: `average_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 121.44 | 77.71 | 1.56 | 0.540 | 0.843 | 0 | 0 |
| avgpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1111.66 | 856.33 | 1.30 | 0.755 | 0.981 | 0 | 0 |
| avgpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 435.39 | 338.15 | 1.29 | 0.762 | 0.981 | 0 | 0 |

### MaxPool<uint8>

- Affects: `max_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 214.42 | 23.68 | 9.06 | 0.306 | 2.768 | 0 | 0 |
| maxpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2126.27 | 236.15 | 9.00 | 0.395 | 3.556 | 0 | 0 |
| maxpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 816.97 | 93.30 | 8.76 | 0.406 | 3.556 | 0 | 0 |

### HardSwish<uint8>

- Affects: `hard_swish` quantized uint8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_u8_1k | 1024 | 4000 | 11.35 | 7.76 | 1.46 | 0.090 | 0.132 | 0 | 0 |
| hardswish_u8_4k | 4096 | 4000 | 51.17 | 30.95 | 1.65 | 0.080 | 0.132 | 0 | 0 |
| hardswish_u8_16k | 16384 | 4000 | 211.57 | 123.59 | 1.71 | 0.077 | 0.133 | 0 | 0 |
| hardswish_u8_64k | 65536 | 1953 | 848.50 | 494.66 | 1.72 | 0.077 | 0.132 | 0 | 0 |

### HardSwish<int8>

- Affects: `hard_swish` quantized int8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_i8_1k | 1024 | 4000 | 10.71 | 7.04 | 1.52 | 0.096 | 0.146 | 0 | 0 |
| hardswish_i8_4k | 4096 | 4000 | 49.00 | 28.03 | 1.75 | 0.084 | 0.146 | 0 | 0 |
| hardswish_i8_16k | 16384 | 4000 | 202.63 | 111.96 | 1.81 | 0.081 | 0.146 | 0 | 0 |
| hardswish_i8_64k | 65536 | 1953 | 814.11 | 447.95 | 1.82 | 0.081 | 0.146 | 0 | 0 |

### ArgMin<float>

- Affects: `arg_min` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmin_f32_1k | 1024 | 4000 | 0.95 | 0.90 | 1.05 | 1.077 | 1.132 | 0 | 0 |
| argmin_f32_16k | 16384 | 4000 | 15.05 | 12.97 | 1.16 | 1.088 | 1.263 | 0 | 0 |
| argmin_f32_64k | 65536 | 1953 | 59.73 | 51.09 | 1.17 | 1.097 | 1.283 | 0 | 0 |

### ArgMax<float>

- Affects: `arg_max` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_f32_1k | 1024 | 4000 | 0.94 | 0.91 | 1.04 | 1.086 | 1.128 | 0 | 0 |
| argmax_f32_16k | 16384 | 4000 | 15.01 | 13.00 | 1.16 | 1.091 | 1.261 | 0 | 0 |
| argmax_f32_64k | 65536 | 1953 | 59.73 | 51.12 | 1.17 | 1.097 | 1.282 | 0 | 0 |

### ArgMax<int8>

- Affects: `arg_max` int8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_i8_1k | 1024 | 4000 | 1.41 | 0.18 | 7.76 | 0.726 | 5.632 | 0 | 0 |
| argmax_i8_16k | 16384 | 4000 | 22.39 | 1.61 | 13.91 | 0.732 | 10.177 | 0 | 0 |
| argmax_i8_64k | 65536 | 1953 | 89.62 | 6.54 | 13.71 | 0.731 | 10.022 | 0 | 0 |

### ArgMax<uint8>

- Affects: `arg_max` uint8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_u8_1k | 1024 | 4000 | 1.41 | 0.18 | 7.76 | 0.725 | 5.621 | 0 | 0 |
| argmax_u8_16k | 16384 | 4000 | 22.40 | 1.61 | 13.95 | 0.731 | 10.206 | 0 | 0 |
| argmax_u8_64k | 65536 | 1953 | 89.58 | 6.48 | 13.83 | 0.732 | 10.120 | 0 | 0 |

## `tflite/kernels/internal/optimized/reduce.h`

- Summary: Follow-up P1 closure for the uint8 keep-dims height-width reduction path and the float last-dimension mean specialization.

### MeanImpl<uint8>

- Affects: `reduce.h` 4D keep-dims height-width reduction path for uint8.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_u8_8x8x48 | B1 I8x8x48 A(1,2) | 4000 | 3.45 | 2.26 | 1.53 | 0.890 | 1.362 | 0 | 0 |
| mean_u8_16x16x96 | B1 I16x16x96 A(1,2) | 4000 | 18.84 | 14.92 | 1.26 | 1.304 | 1.647 | 0 | 0 |
| mean_u8_32x8x160 | B1 I32x8x160 A(1,2) | 3125 | 40.14 | 24.80 | 1.62 | 1.020 | 1.652 | 0 | 0 |

### Mean<float> last-dim

- Affects: `reduce.h` float specialization that reduces only the last dimension.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_f32_lastdim_256x48 | R256 C48 axis(1) | 4000 | 13.71 | 13.65 | 1.00 | 0.896 | 0.900 | 0.00000024 | 0.00003184 | 0.0000000346 |
| mean_f32_lastdim_256x192 | R256 C192 axis(1) | 2604 | 69.04 | 38.01 | 1.82 | 0.712 | 1.293 | 0.00000021 | 0.00029705 | 0.0000000323 |
| mean_f32_lastdim_128x640 | R128 C640 axis(1) | 1562 | 113.08 | 61.19 | 1.85 | 0.724 | 1.339 | 0.00000018 | 0.00002661 | 0.0000000364 |

## `tflite/kernels/internal/optimized/resize_bilinear.h`

- Summary: Follow-up P1 closure for the generic float bilinear kernel and the uint8 generic-small-channel interpolation path.

### ResizeBilinear<float>

- Affects: `resize_bilinear.h` generic float bilinear accumulation kernel.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resize_f32_17x19x48_to_29x31 | B1 I17x19x48 O29x31 AC0 HPC0 | 423 | 241.26 | 241.37 | 1.00 | 1.252 | 1.251 | 0.00000000 | 0.00000000 | 0.0000000000 |
| resize_f32_23x27x96_to_37x41 | B1 I23x27x96 O37x41 AC0 HPC1 | 125 | 788.80 | 787.86 | 1.00 | 1.292 | 1.294 | 0.00000000 | 0.00000000 | 0.0000000000 |
| resize_f32_15x21x160_to_28x35 | B1 I15x21x160 O28x35 AC1 HPC0 | 116 | 850.56 | 850.68 | 1.00 | 1.290 | 1.290 | 0.00000000 | 0.00000000 | 0.0000000000 |

### ResizeBilinear<uint8>

- Affects: `resize_bilinear.h` uint8 generic-small-channel interpolation path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resize_u8_17x19x48_to_29x31 | B1 I17x19x48 O29x31 AC0 HPC0 | 423 | 287.81 | 140.92 | 2.04 | 1.050 | 2.143 | 0 | 0 |
| resize_u8_23x27x96_to_37x41 | B1 I23x27x96 O37x41 AC0 HPC1 | 125 | 946.95 | 440.50 | 2.15 | 1.077 | 2.314 | 0 | 0 |
| resize_u8_15x21x160_to_28x35 | B1 I15x21x160 O28x35 AC1 HPC0 | 116 | 1010.36 | 459.32 | 2.20 | 1.086 | 2.390 | 0 | 0 |

## `tflite/kernels/fully_connected.cc`

- Summary: Dense float FullyConnected on RVV now takes the `EvalPie` route, so this split benchmark keeps the measurement alongside the source-file section and exercises the same matvec-style accumulation shape.

### FullyConnected<float> dense EvalPie route

- Affects: `fully_connected.cc` dense float RVV dispatch that now routes through `EvalPie`; this benchmark mirrors that matrix-batch-vector accumulation path inside the split operator harness.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_dense_small_b4 | B4 I64 O64 | 3906 | 24.33 | 21.23 | 1.15 | 1.347 | 1.544 | 0.00000000 | 0.00000000 | 0.0000000000 |
| fc_dense_mid_b4 | B4 I128 O128 | 976 | 87.07 | 87.16 | 1.00 | 1.505 | 1.504 | 0.00000000 | 0.00000000 | 0.0000000000 |
| fc_dense_large_b4 | B4 I2048 O640 | 20 | 7216.75 | 7236.22 | 1.00 | 1.453 | 1.449 | 0.00000000 | 0.00000000 | 0.0000000000 |

## `tflite/kernels/lstm_eval.cc`

- Summary: Float gate and output/projection paths now reuse the RVV-enabled `tensor_utils` matvec helpers, so this section keeps the operator-level numbers separate from generic helper benchmarks.

### LstmGate<float>

- Affects: `lstm_eval` float gate path (`input_to_gate` + `recurrent_to_gate` + sigmoid).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_gate_small_b1 | B1 I128 O128 C512 | 488 | 189.70 | 189.24 | 1.00 | 1.382 | 1.385 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_gate_small_b4 | B4 I128 O128 C512 | 122 | 757.14 | 752.45 | 1.01 | 1.385 | 1.394 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_gate_large_b1 | B1 I256 O256 C1024 | 122 | 763.98 | 763.19 | 1.00 | 1.373 | 1.374 | 0.00000000 | 0.00000000 | 0.0000000000 |

### LstmOutput<float>

- Affects: `lstm_eval` float output/projection path (`tanh(cell)` + output gate + projection).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_output_small_b1 | B1 C512 O128 | 972 | 107.91 | 107.64 | 1.00 | 1.219 | 1.222 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_output_small_b4 | B4 C512 O128 | 243 | 430.89 | 430.67 | 1.00 | 1.221 | 1.222 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_output_large_b1 | B1 C1024 O256 | 243 | 394.48 | 394.26 | 1.00 | 1.332 | 1.332 | 0.00000000 | 0.00000000 | 0.0000000000 |

## `tflite/kernels/internal/optimized/integer_ops/add.h`

- Summary: Recent RVV work added both int8/int16 elementwise add coverage and the scalar-broadcast kernels used by the fivefold broadcast fast path.

### Add<int8>

- Affects: `integer_ops/add.h` int8 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i8_1k | 1024 | 4000 | 10.23 | 5.90 | 1.73 | 0.100 | 0.174 | 0 | 0 |
| add_i8_4k | 4096 | 4000 | 50.84 | 23.59 | 2.16 | 0.081 | 0.174 | 0 | 0 |
| add_i8_16k | 16384 | 4000 | 209.70 | 94.24 | 2.23 | 0.078 | 0.174 | 0 | 0 |
| add_i8_64k | 65536 | 1953 | 840.36 | 377.08 | 2.23 | 0.078 | 0.174 | 0 | 0 |

### AddScalarBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast add kernel used by the fivefold broadcast fast path in `integer_ops/add.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_i8_1k | 1024 | 4000 | 7.08 | 4.16 | 1.70 | 0.145 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_4k | 4096 | 4000 | 33.63 | 16.68 | 2.02 | 0.122 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_16k | 16384 | 4000 | 129.84 | 66.71 | 1.95 | 0.126 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_64k | 65536 | 1953 | 516.77 | 266.85 | 1.94 | 0.127 | 0.246 | 0 | 0 |

### BroadcastAddDispatch<int8> scalar lhs

- Affects: `integer_ops/add.h` int8 scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_i8_1k | 1024 | 4000 | 14.43 | 14.43 | 1.00 | 0.071 | 0.071 | 0 | 0 |
| add_bcast_i8_4k | 4096 | 4000 | 67.50 | 67.44 | 1.00 | 0.061 | 0.061 | 0 | 0 |
| add_bcast_i8_16k | 16384 | 4000 | 271.33 | 271.45 | 1.00 | 0.060 | 0.060 | 0 | 0 |
| add_bcast_i8_64k | 65536 | 1953 | 1055.61 | 1055.43 | 1.00 | 0.062 | 0.062 | 0 | 0 |

### Add<int16>

- Affects: `integer_ops/add.h` int16 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i16_1k | 1024 | 4000 | 10.42 | 5.79 | 1.80 | 0.098 | 0.177 | 0 | 0 |
| add_i16_4k | 4096 | 4000 | 50.14 | 23.10 | 2.17 | 0.082 | 0.177 | 0 | 0 |
| add_i16_16k | 16384 | 4000 | 207.67 | 92.47 | 2.25 | 0.079 | 0.177 | 0 | 0 |
| add_i16_64k | 65536 | 1953 | 833.16 | 369.62 | 2.25 | 0.079 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mul.h`

- Summary: Recent RVV work added both int8 elementwise mul coverage and the scalar-broadcast kernels used by the quantized broadcast fast path.

### Mul<int8>

- Affects: `integer_ops/mul.h` int8 elementwise quantized mul path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_i8_1k | 1024 | 4000 | 10.71 | 3.23 | 3.32 | 0.096 | 0.317 | 0 | 0 |
| mul_i8_4k | 4096 | 4000 | 47.86 | 12.86 | 3.72 | 0.086 | 0.318 | 0 | 0 |
| mul_i8_16k | 16384 | 4000 | 198.44 | 51.31 | 3.87 | 0.083 | 0.319 | 0 | 0 |
| mul_i8_64k | 65536 | 1953 | 797.88 | 205.34 | 3.89 | 0.082 | 0.319 | 0 | 0 |

### MulSimpleBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast mul kernel used by the fivefold broadcast fast path in `integer_ops/mul.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_i8_1k | 1024 | 4000 | 9.97 | 3.29 | 3.03 | 0.103 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_4k | 4096 | 4000 | 45.70 | 13.18 | 3.47 | 0.090 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_16k | 16384 | 4000 | 187.43 | 52.69 | 3.56 | 0.087 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_64k | 65536 | 1953 | 752.82 | 210.82 | 3.57 | 0.087 | 0.311 | 0 | 0 |

### BroadcastMulDispatch<int8> scalar lhs

- Affects: `integer_ops/mul.h` int8 scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_i8_1k | 1024 | 4000 | 10.59 | 10.56 | 1.00 | 0.097 | 0.097 | 0 | 0 |
| mul_bcast_i8_4k | 4096 | 4000 | 48.04 | 48.10 | 1.00 | 0.085 | 0.085 | 0 | 0 |
| mul_bcast_i8_16k | 16384 | 4000 | 196.19 | 196.09 | 1.00 | 0.084 | 0.084 | 0 | 0 |
| mul_bcast_i8_64k | 65536 | 1953 | 787.98 | 788.10 | 1.00 | 0.083 | 0.083 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/sub.h`

- Summary: This section isolates the int16 quantized subtract path that gained a dedicated RVV elementwise implementation.

### Sub<int16>

- Affects: `integer_ops/sub.h` int16 elementwise quantized sub path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_i16_1k | 1024 | 4000 | 10.00 | 5.80 | 1.72 | 0.102 | 0.177 | 0 | 0 |
| sub_i16_4k | 4096 | 4000 | 50.59 | 23.14 | 2.19 | 0.081 | 0.177 | 0 | 0 |
| sub_i16_16k | 16384 | 4000 | 206.82 | 92.43 | 2.24 | 0.079 | 0.177 | 0 | 0 |
| sub_i16_64k | 65536 | 1953 | 833.22 | 369.47 | 2.26 | 0.079 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/pooling.h`

- Summary: This section keeps the signed int8 pooling kernels separate from the uint8 `optimized_ops.h` pooling paths, because their RVV logic and rounding semantics are different.

### AveragePool<int8>

- Affects: `average_pool` int8 depth inner loop, signed round-away-from-zero, and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 185.19 | 64.57 | 2.87 | 0.354 | 1.015 | 0 | 0 |
| avgpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1436.59 | 684.76 | 2.10 | 0.585 | 1.226 | 0 | 0 |
| avgpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 566.24 | 271.89 | 2.08 | 0.586 | 1.220 | 0 | 0 |

### MaxPool<int8>

- Affects: `max_pool` int8 depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 227.63 | 24.03 | 9.47 | 0.288 | 2.728 | 0 | 0 |
| maxpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2219.76 | 239.76 | 9.26 | 0.378 | 3.503 | 0 | 0 |
| maxpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 857.41 | 94.93 | 9.03 | 0.387 | 3.495 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h`

- Summary: This section tracks the RVV general depthwise int8 kernels, including the fixed-depth and fixed-depth-multiplier specializations added to mirror the NEON dispatch table.

### DepthwiseConv<int8>

- Affects: `depthwise_conv` int8 general path, including fixed input-depth and depth-multiplier RVV kernels aligned with NEON dispatch.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dwconv_i8_dm1_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM1 | 1975 | 185.13 | 115.15 | 1.61 | 0.350 | 0.563 | 0 | 0 |
| dwconv_i8_dm1_d8_s1 | B1 I32x32x8 F3x3 S1x1 DM1 | 987 | 342.55 | 151.96 | 2.25 | 0.378 | 0.853 | 0 | 0 |
| dwconv_i8_dm1_d12_s1 | B1 I32x32x12 F3x3 S1x1 DM1 | 658 | 509.15 | 195.44 | 2.61 | 0.382 | 0.995 | 0 | 0 |
| dwconv_i8_dm1_d16_s2 | B1 I32x32x16 F3x3 S2x2 DM1 | 1975 | 170.81 | 71.88 | 2.38 | 0.379 | 0.901 | 0 | 0 |
| dwconv_i8_dm2_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM2 | 987 | 307.71 | 193.58 | 1.59 | 0.421 | 0.669 | 0 | 0 |
| dwconv_i8_dm2_d8_s2 | B1 I32x32x8 F3x3 S2x2 DM2 | 1975 | 150.79 | 97.04 | 1.55 | 0.430 | 0.668 | 0 | 0 |
| dwconv_i8_dm4_d1_s1 | B1 I32x32x1 F3x3 S1x1 DM4 | 1975 | 131.93 | 108.18 | 1.22 | 0.491 | 0.599 | 0 | 0 |
| dwconv_i8_dm4_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM4 | 493 | 454.18 | 385.41 | 1.18 | 0.571 | 0.673 | 0 | 0 |
| dwconv_i8_dm8_d2_s1 | B1 I32x32x2 F3x3 S1x1 DM8 | 493 | 394.67 | 282.36 | 1.40 | 0.657 | 0.918 | 0 | 0 |
| dwconv_i8_dm8_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM8 | 3950 | 53.89 | 44.36 | 1.21 | 0.601 | 0.730 | 0 | 0 |
| dwconv_i8_dm16_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM16 | 1975 | 94.91 | 64.76 | 1.47 | 0.683 | 1.001 | 0 | 0 |
| dwconv_i8_dm20_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM20 | 1580 | 116.17 | 97.06 | 1.20 | 0.697 | 0.835 | 0 | 0 |
| dwconv_i8_dm32_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM32 | 987 | 176.81 | 129.87 | 1.36 | 0.733 | 0.998 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h`

- Note: The recent change in this file only reroutes hybrid dispatch to reuse the RVV depthwise kernels above, so it intentionally shares the same numeric benchmark coverage.

## `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h`

- Summary: This section isolates the int16 quantized LeakyReLU path that received a dedicated RVV implementation.

### LeakyRelu<int16>

- Affects: `integer_ops/leaky_relu.h` int16 quantized leaky-relu path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leaky_relu_i16_1k | 1024 | 4000 | 13.42 | 4.08 | 3.29 | 0.076 | 0.251 | 0 | 0 |
| leaky_relu_i16_4k | 4096 | 4000 | 54.44 | 16.36 | 3.33 | 0.075 | 0.250 | 0 | 0 |
| leaky_relu_i16_16k | 16384 | 4000 | 216.67 | 65.33 | 3.32 | 0.076 | 0.251 | 0 | 0 |
| leaky_relu_i16_64k | 65536 | 1953 | 866.61 | 261.28 | 3.32 | 0.076 | 0.251 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/lut.h`

- Summary: This section keeps the uint8 and int8 lookup-table kernels separate because they were vectorized in the same commit but have different input domains.

### LookupTable<uint8>

- Affects: `integer_ops/lut.h` uint8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_u8_1k | 1024 | 4000 | 1.41 | 0.85 | 1.66 | 0.728 | 1.207 | 0 | 0 |
| lut_u8_4k | 4096 | 4000 | 5.60 | 3.39 | 1.65 | 0.732 | 1.208 | 0 | 0 |
| lut_u8_16k | 16384 | 4000 | 22.41 | 13.55 | 1.65 | 0.731 | 1.209 | 0 | 0 |
| lut_u8_64k | 65536 | 1953 | 89.61 | 54.31 | 1.65 | 0.731 | 1.207 | 0 | 0 |

### LookupTable<int8>

- Affects: `integer_ops/lut.h` int8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_i8_1k | 1024 | 4000 | 0.94 | 0.85 | 1.11 | 1.088 | 1.207 | 0 | 0 |
| lut_i8_4k | 4096 | 4000 | 3.74 | 3.40 | 1.10 | 1.096 | 1.206 | 0 | 0 |
| lut_i8_16k | 16384 | 4000 | 14.92 | 13.56 | 1.10 | 1.098 | 1.208 | 0 | 0 |
| lut_i8_64k | 65536 | 1953 | 59.99 | 54.24 | 1.11 | 1.093 | 1.208 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mean.h`

- Summary: This section tracks the int8 height-width reduction path that gained RVV coverage in the latest integer-ops commit.

### Mean<int8>

- Affects: `integer_ops/mean.h` int8 height-width reduction path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_i8_8x8x64 | B1 I8x8x64 A(1,2) | 4000 | 6.31 | 1.63 | 3.88 | 0.649 | 2.517 | 0 | 0 |
| mean_i8_16x16x128 | B1 I16x16x128 A(1,2) | 3906 | 37.49 | 11.61 | 3.23 | 0.874 | 2.823 | 0 | 0 |
| mean_i8_32x32x256 | B1 I32x32x256 A(1,2) | 488 | 1446.34 | 92.21 | 15.69 | 0.181 | 2.843 | 0 | 0 |

